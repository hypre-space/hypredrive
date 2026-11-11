/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/lsseq.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _MSC_VER
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/utils.h"

/* Owns the stream and all metadata. Payload sources borrow this reader after
 * LSSeqPrepareRead has selected a valid system and rank-local part range. */
typedef struct LSSeqData_struct
{
   FILE                *fp;
   LSSeqHeader          header;
   LSSeqInfoHeader      info_header;
   char                *info_payload;
   size_t               info_payload_size;
   LSSeqPartMeta       *parts;
   LSSeqPatternMeta    *patterns;
   LSSeqSystemPartMeta *sys_parts;
   uint64_t            *part_blob_table; /* 6*num_parts entries */
   LSSeqTimestepEntry  *timesteps;
   uint32_t            *part_order;
   uint32_t             first_part;
   uint32_t             num_local_parts;
   int                  ls_id;
} LSSeqData;

enum
{
   LSSEQ_INFO_PAYLOAD_MAX_BYTES = 16u * 1024u * 1024u,
   LSSEQ_MAX_META_BYTES         = 512u * 1024u * 1024u,
   LSSEQ_MAX_BLOB_BYTES         = 512u * 1024u * 1024u,
   LSSEQ_MAX_PARTS              = 1024u * 1024u,
   LSSEQ_MAX_SYSTEMS            = 1024u * 1024u,
   LSSEQ_MAX_PATTERNS           = 1024u * 1024u,
};

static int
LSSeqCheckedMulSize(size_t a, size_t b, size_t *result, const char *what)
{
   /* GCOVR_EXCL_START */
   if (a != 0 && b > SIZE_MAX / a)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq size overflow while computing %s (%zu * %zu)", what, a,
                           b);
      return 0;
   }

   *result = a * b;
   return 1;
   /* GCOVR_EXCL_STOP */
}

static int
LSSeqCheckedAddU64(uint64_t a, uint64_t b, uint64_t *result, const char *what)
{
   /* GCOVR_EXCL_START */
   if (UINT64_MAX - a < b)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq offset overflow while computing %s", what);
      return 0;
   }

   *result = a + b;
   return 1;
   /* GCOVR_EXCL_STOP */
}

static int
LSSeqValidateByteLimit(size_t nbytes, size_t max_nbytes, const char *what)
{
   /* GCOVR_EXCL_START */
   if (nbytes > max_nbytes)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq %s exceeds limit (%zu > %zu bytes)", what, nbytes,
                           max_nbytes);
      return 0;
   }

   return 1;
   /* GCOVR_EXCL_STOP */
}

static int
LSSeqReadAt(FILE *fp, uint64_t offset, void *buffer, size_t nbytes, const char *what)
{
   /* GCOVR_EXCL_BR_START */
#ifdef _MSC_VER
   if (offset > (uint64_t)INT64_MAX || _fseeki64(fp, (int64_t)offset, SEEK_SET) != 0)
#else
   off_t seek_offset = (off_t)offset;
   if (seek_offset < 0 || (uint64_t)seek_offset != offset ||
       fseeko(fp, seek_offset, SEEK_SET) != 0)
#endif
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not seek to offset %llu while reading %s",
                           (unsigned long long)offset, what);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (nbytes > 0 && fread(buffer, 1, nbytes, fp) != nbytes) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read %s (%zu bytes)", what, nbytes);
      return 0;
   }

   return 1;
}

static void
LSSeqDataDestroy(LSSeqData *seq)
{
   if (seq->fp)
   {
      fclose(seq->fp);
   }
   free(seq->part_order);
   free(seq->info_payload);
   free(seq->parts);
   free(seq->patterns);
   free(seq->sys_parts);
   free(seq->part_blob_table);
   free(seq->timesteps);
   memset(seq, 0, sizeof(*seq));
}

/* Reads and validates the mandatory info block. The reader owns the stream
 * and payload, including on failure. */
static int
LSSeqLoadInfoHeader(LSSeqData *seq, const char *filename)
{
   const uint64_t info_offset        = (uint64_t)sizeof(LSSeqHeader);
   uint64_t expected_min_part_offset = info_offset + (uint64_t)sizeof(LSSeqInfoHeader);
   uint64_t expected_payload_end     = 0;

   if (!(seq->header.flags & LSSEQ_FLAG_HAS_INFO))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Missing mandatory LSSeq info header in '%s'", filename);
      return 0;
   }

   if (seq->header.offset_part_meta < expected_min_part_offset)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq info offsets in '%s' (offset_part_meta=%llu)",
                           filename, (unsigned long long)seq->header.offset_part_meta);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(seq->fp, info_offset, &seq->info_header, sizeof(seq->info_header),
                    "info header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.magic != LSSEQ_INFO_MAGIC ||
       seq->info_header.version != LSSEQ_INFO_VERSION)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq info header in '%s' (magic=%llu version=%u)",
                           filename, (unsigned long long)seq->info_header.magic,
                           seq->info_header.version);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.endian_tag != UINT32_C(0x01020304)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Unsupported LSSeq info endianness tag in '%s' "
                           "(tag=0x%08x)",
                           filename, (unsigned int)seq->info_header.endian_tag);
      return 0;
   }

   /* Bound payload size to avoid accidental huge allocations. */
   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.payload_size > (uint64_t)LSSEQ_INFO_PAYLOAD_MAX_BYTES ||
       seq->info_header.payload_size > (uint64_t)SIZE_MAX - 1u)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq info payload too large in '%s' (%llu bytes)", filename,
                           (unsigned long long)seq->info_header.payload_size);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqCheckedAddU64(expected_min_part_offset,
                           (uint64_t)seq->info_header.payload_size, &expected_payload_end,
                           "info payload end"))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   if (seq->header.offset_part_meta < expected_payload_end)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq info payload overlaps part metadata in '%s' "
                           "(payload_end=%llu part_off=%llu)",
                           filename, (unsigned long long)expected_payload_end,
                           (unsigned long long)seq->header.offset_part_meta);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.payload_size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      uint64_t hash          = UINT64_C(1469598103934665603);
      seq->info_payload_size = (size_t)seq->info_header.payload_size;
      seq->info_payload =
         (char *)hypredrv_Malloc(seq->info_payload_size + 1u, "LSSeq info payload");
      /* GCOVR_EXCL_BR_START */
      if (!seq->info_payload) /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqReadAt(seq->fp, expected_min_part_offset, seq->info_payload,
                       seq->info_payload_size, "info payload"))
      /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }
      seq->info_payload[seq->info_payload_size] = '\0';

      for (size_t i = 0; i < seq->info_payload_size; i++)
      {
         hash ^= (unsigned char)seq->info_payload[i];
         hash *= UINT64_C(1099511628211);
      }
      /* GCOVR_EXCL_BR_START */
      if (hash != seq->info_header.payload_hash_fnv1a64) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'", filename);
         return 0;
      }
   }

   return 1;
}

/* Metadata enters through this bounded read, before any reader uses its
 * fields. Partial buffers are released here; successful buffers belong to seq. */
static void *
LSSeqReadMetadata(FILE *fp, uint64_t offset, size_t count, size_t width, const char *what)
{
   size_t nbytes = 0;
   if (!LSSeqCheckedMulSize(count, width, &nbytes, what) ||
       !LSSeqValidateByteLimit(nbytes, LSSEQ_MAX_META_BYTES, what))
   {
      return NULL;
   }
   void *data = hypredrv_Malloc(nbytes, what);
   if (!data)
   {
      return NULL;
   }
   if (!LSSeqReadAt(fp, offset, data, nbytes, what))
   {
      free(data);
      return NULL;
   }
   return data;
}

/* The reader owns the file from its first header read through payload import.
 * Closing and reopening the path could mix metadata and payloads from different
 * files if the path is replaced in between. */
static int
LSSeqOpenHeader(const char *filename, LSSeqData *seq)
{
   if (!filename)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing sequence filename");
      return 0;
   }
   seq->fp = fopen(filename, "rb");
   if (!seq->fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }
   if (!LSSeqReadAt(seq->fp, 0, &seq->header, sizeof(seq->header), "lsseq header"))
   {
      return 0;
   }
   const LSSeqHeader *header = &seq->header;
   /* GCOVR_EXCL_BR_START */
   if (header->magic != LSSEQ_MAGIC) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq magic for file '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->version != LSSEQ_VERSION) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Unsupported LSSeq version %u in '%s'", header->version,
                           filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->offset_part_blob_table == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "LSSeq format requires part blob table (offset_part_blob_table) in '%s'",
         filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->num_systems == 0 || header->num_parts == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq shape in '%s': num_systems=%u, num_parts=%u",
                           filename, header->num_systems, header->num_parts);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->num_parts > LSSEQ_MAX_PARTS || header->num_systems > LSSEQ_MAX_SYSTEMS ||
       header->num_patterns > LSSEQ_MAX_PATTERNS ||
       header->num_timesteps > LSSEQ_MAX_TIMESTEPS)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq dimensions exceed limits in '%s' (systems=%u parts=%u "
                           "patterns=%u timesteps=%u)",
                           filename, header->num_systems, header->num_parts,
                           header->num_patterns, header->num_timesteps);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->codec > (uint32_t)COMP_BLOSC) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq compression codec id %u in '%s'", header->codec,
                           filename);
      return 0;
   }

   return 1;
}

static int
LSSeqDataLoad(const char *filename, LSSeqData *seq)
{
   size_t n_sys_parts = 0;
   memset(seq, 0, sizeof(*seq));
   if (!LSSeqOpenHeader(filename, seq) || !LSSeqLoadInfoHeader(seq, filename))
   {
      goto fail;
   }

   seq->parts =
      LSSeqReadMetadata(seq->fp, seq->header.offset_part_meta, seq->header.num_parts,
                        sizeof(*seq->parts), "part metadata");
   if (!seq->parts)
   {
      goto fail;
   }
   /* Readers size buffers from both the row range and nrows. */
   for (uint32_t p = 0; p < seq->header.num_parts; p++)
   {
      const LSSeqPartMeta *part = &seq->parts[p];
      if (part->row_upper < part->row_lower ||
          part->nrows != part->row_upper - part->row_lower + 1u)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd(
            "Invalid LSSeq part %u row range [%llu, %llu] for %llu rows", p,
            (unsigned long long)part->row_lower, (unsigned long long)part->row_upper,
            (unsigned long long)part->nrows);
         goto fail;
      }
   }
   if (seq->header.num_patterns > 0)
   {
      seq->patterns = LSSeqReadMetadata(seq->fp, seq->header.offset_pattern_meta,
                                        seq->header.num_patterns, sizeof(*seq->patterns),
                                        "pattern metadata");
      if (!seq->patterns)
      {
         goto fail;
      }
   }
   if (!LSSeqCheckedMulSize(seq->header.num_systems, seq->header.num_parts, &n_sys_parts,
                            "system-part count"))
   {
      goto fail;
   }
   seq->sys_parts =
      LSSeqReadMetadata(seq->fp, seq->header.offset_sys_part_meta, n_sys_parts,
                        sizeof(*seq->sys_parts), "system-part metadata");
   if (!seq->sys_parts)
   {
      goto fail;
   }
   seq->part_blob_table = LSSeqReadMetadata(
      seq->fp, seq->header.offset_part_blob_table, seq->header.num_parts,
      LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t), "part blob table");
   if (!seq->part_blob_table)
   {
      goto fail;
   }
   if ((seq->header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq->header.num_timesteps > 0)
   {
      seq->timesteps = LSSeqReadMetadata(seq->fp, seq->header.offset_timestep_meta,
                                         seq->header.num_timesteps,
                                         sizeof(*seq->timesteps), "timestep metadata");
      if (!seq->timesteps)
      {
         goto fail;
      }
   }
   return 1;

fail:
   LSSeqDataDestroy(seq);
   return 0;
}

int
hypredrv_LSSeqReadInfo(const char *filename, char **payload_ptr, size_t *payload_size)
{
   LSSeqData seq = {0};
   int       ok  = 0;
   if (!payload_ptr || !payload_size)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid LSSeqReadInfo outputs");
      return 0;
   }
   *payload_ptr  = NULL;
   *payload_size = 0;

   /* Same header and info checks as the full loader, without the metadata. */
   if (!LSSeqOpenHeader(filename, &seq) || !LSSeqLoadInfoHeader(&seq, filename))
   {
      goto cleanup;
   }
   /* An empty payload is still returned as a terminated string. */
   if (!seq.info_payload)
   {
      seq.info_payload = hypredrv_Calloc(1, 1, "seq.info_payload");
      if (!seq.info_payload)
      {
         goto cleanup;
      }
   }
   *payload_ptr     = seq.info_payload;
   *payload_size    = seq.info_payload_size;
   seq.info_payload = NULL;
   ok               = 1;

cleanup:
   LSSeqDataDestroy(&seq);
   return ok;
}

/* Reads the blob at `offset` and decodes it; it must decode to exactly
 * `expected_size` bytes (0 for an empty pattern). */
static int
LSSeqReadBlob(const LSSeqData *seq, uint64_t offset, uint64_t blob_size,
              size_t expected_size, void **output)
{
   void  *blob_data    = NULL;
   void  *decoded      = NULL;
   size_t decoded_size = 0;

   *output = NULL;

   if (blob_size > (uint64_t)SIZE_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Blob size too large to decode (%llu bytes)",
                           (unsigned long long)blob_size);
      return 0;
   }
   if (blob_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Blob size exceeds limit (%llu bytes)",
                           (unsigned long long)blob_size);
      return 0;
   }
   if (expected_size > LSSEQ_MAX_BLOB_BYTES) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Expected decoded blob size exceeds limit (%zu bytes)",
                           expected_size);
      return 0;
   }

   if (blob_size == 0)
   {
      /* GCOVR_EXCL_BR_START */
      if (expected_size != 0) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd(
            "Encountered empty blob for non-empty expected payload (%zu bytes)",
            expected_size);
         return 0;
      }
      return 1;
   }

   blob_data = hypredrv_Malloc((size_t)blob_size, "blob_data");
   if (!blob_data) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(seq->fp, offset, blob_data, (size_t)blob_size, "blob payload"))
   /* GCOVR_EXCL_BR_STOP */
   {
      free(blob_data);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->header.codec == COMP_NONE) /* GCOVR_EXCL_BR_STOP */
   {
      /* Raw payload: hand the read buffer over instead of copying it. */
      decoded      = blob_data;
      decoded_size = (size_t)blob_size;
      blob_data    = NULL;
   }
   else
   {
      /* hypredrv_decompress allocates the size the blob's header declares: check
       * it against the expected size first, so a corrupt header cannot request a
       * buffer far larger than the payload the metadata describes. */
      uint64_t declared = 0;
      if (blob_size >= sizeof(declared))
      {
         memcpy(&declared, blob_data, sizeof(declared));
      }
      if (declared != (uint64_t)expected_size)
      {
         free(blob_data);
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Blob declares %llu decoded bytes, expected %zu",
                              (unsigned long long)declared, expected_size);
         return 0;
      }
      hypredrv_decompress((comp_alg_t)seq->header.codec, (size_t)blob_size, blob_data,
                          &decoded_size, &decoded);
      if (hypredrv_ErrorCodeActive() || !decoded) /* GCOVR_EXCL_BR_LINE */
      {
         free(blob_data);
         return 0;
      }
   }

   free(blob_data);
   blob_data = NULL;

   if (decoded_size != expected_size)
   {
      free(decoded);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Decoded blob size mismatch: expected=%zu got=%zu",
                           expected_size, decoded_size);
      return 0;
   }

   *output = decoded;
   return 1;
}

/* Resumable decoders for recently read part blobs. Systems are usually read
 * in order, and each part blob batches every system's payload, so resuming
 * the previous decode keeps sequential reads from re-inflating (and
 * re-reading) all earlier systems. Entries are keyed by file identity (device,
 * inode, size, modification time), blob offset, size and codec, plus the bytes
 * at both ends of the compressed blob, so a modified file misses the cache. Each entry
 * holds one compressed blob and decoder state, never decoded payload. POSIX only;
 * elsewhere every read decodes from the start of the blob. */
enum
{
   LSSEQ_STREAM_CACHE_SIZE = 8
};

typedef struct
{
   unsigned long long dev, ino, size, mtime_sec, mtime_nsec;
   uint64_t           blob_offset, blob_size;
   int                codec;
   unsigned char      head[128], tail[128]; /* content fingerprint of the blob */
} LSSeqStreamKey;

typedef struct
{
   hypredrv_SliceStream *stream;
   LSSeqStreamKey        key;
   unsigned long         used;
} LSSeqStreamEntry;

static LSSeqStreamEntry g_lsseq_streams[LSSEQ_STREAM_CACHE_SIZE];
static unsigned long    g_lsseq_stream_clock;

/* Fills `key` for the blob at `offset`; returns 0 when the file identity is
 * unavailable (no caching then). */
static int
LSSeqStreamKeyMake(FILE *fp, comp_alg_t codec, uint64_t offset, uint64_t size,
                   LSSeqStreamKey *key)
{
#if defined(_WIN32)
   (void)fp;
   (void)codec;
   (void)offset;
   (void)size;
   (void)key;
   return 0;
#else
   struct stat st;

   if (fstat(fileno(fp), &st) != 0) /* GCOVR_EXCL_BR_LINE */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   memset(key, 0, sizeof(*key));
   key->dev       = (unsigned long long)st.st_dev;
   key->ino       = (unsigned long long)st.st_ino;
   key->size      = (unsigned long long)st.st_size;
   key->mtime_sec = (unsigned long long)st.st_mtime;
#if defined(__linux__)
   key->mtime_nsec = (unsigned long long)st.st_mtim.tv_nsec;
#endif
   key->blob_offset = offset;
   key->blob_size   = size;
   key->codec       = (int)codec;

   /* Fingerprint both ends of the compressed blob so a rewritten file that
    * reuses the inode within one timestamp tick still misses. */
   size_t n = (size < sizeof(key->head)) ? (size_t)size : sizeof(key->head);
   /* GCOVR_EXCL_BR_START */
   return LSSeqReadAt(fp, offset, key->head, n, "blob fingerprint") &&
          LSSeqReadAt(fp, offset + size - n, key->tail, n, "blob fingerprint");
   /* GCOVR_EXCL_BR_STOP */
#endif
}

static int
LSSeqStreamKeyEqual(const LSSeqStreamKey *a, const LSSeqStreamKey *b)
{
   return a->dev == b->dev && a->ino == b->ino && a->size == b->size &&
          a->mtime_sec == b->mtime_sec && a->mtime_nsec == b->mtime_nsec &&
          a->blob_offset == b->blob_offset && a->blob_size == b->blob_size &&
          a->codec == b->codec && !memcmp(a->head, b->head, sizeof(a->head)) &&
          !memcmp(a->tail, b->tail, sizeof(a->tail));
}

static hypredrv_SliceStream *
LSSeqStreamCacheFind(const LSSeqStreamKey *key)
{
   for (int i = 0; i < LSSEQ_STREAM_CACHE_SIZE; i++)
   {
      if (g_lsseq_streams[i].stream && LSSeqStreamKeyEqual(&g_lsseq_streams[i].key, key))
      {
         g_lsseq_streams[i].used = ++g_lsseq_stream_clock;
         return g_lsseq_streams[i].stream;
      }
   }
   return NULL;
}

/* Creates a stream over the blob behind `src` and caches it, evicting the least
 * recently used entry. Returns NULL for codecs without resumable decoding. */
static hypredrv_SliceStream *
LSSeqStreamCacheInsert(const LSSeqStreamKey *key, comp_alg_t codec, size_t blob_size,
                       const hypredrv_SliceSource *src)
{
   hypredrv_SliceStream *stream = hypredrv_SliceStreamCreate(codec, blob_size, src);
   int                   victim = 0;

   if (!stream)
   {
      return NULL;
   }
   for (int i = 1; i < LSSEQ_STREAM_CACHE_SIZE; i++)
   {
      if (g_lsseq_streams[i].used < g_lsseq_streams[victim].used)
      {
         victim = i;
      }
   }
   hypredrv_SliceStreamDestroy(&g_lsseq_streams[victim].stream);
   g_lsseq_streams[victim].stream = stream;
   g_lsseq_streams[victim].key    = *key;
   g_lsseq_streams[victim].used   = ++g_lsseq_stream_clock;
   return stream;
}

void
hypredrv_LSSeqReleaseCaches(void)
{
   for (int i = 0; i < LSSEQ_STREAM_CACHE_SIZE; i++)
   {
      hypredrv_SliceStreamDestroy(&g_lsseq_streams[i].stream);
      g_lsseq_streams[i].used = 0;
   }
   g_lsseq_stream_clock = 0;
}

/* Slice-stream source reading a compressed blob that starts at `base` in `fp`. */
typedef struct
{
   FILE    *fp;
   uint64_t base;
} LSSeqFileBlob;

static int
LSSeqFileBlobRead(void *ctx, uint64_t offset, void *buffer, size_t size)
{
   const LSSeqFileBlob *blob = (const LSSeqFileBlob *)ctx;
   return LSSeqReadAt(blob->fp, blob->base + offset, buffer, size, "blob payload");
}

/* Uncompressed blobs: read the slice straight from the file. */
static int
LSSeqReadRawBlobSlice(FILE *fp, uint64_t blob_offset, uint64_t c_size,
                      uint64_t decomp_offset, uint64_t decomp_size, void **output)
{
   /* GCOVR_EXCL_BR_START */
   if (decomp_offset > c_size || decomp_size > c_size - decomp_offset)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Raw blob slice exceeds its part blob");
      return 0;
   }
   if (decomp_size == 0)
   {
      return 1;
   }
   *output = malloc((size_t)decomp_size);
   /* GCOVR_EXCL_BR_START */
   if (!*output || !LSSeqReadAt(fp, blob_offset + decomp_offset, *output,
                                (size_t)decomp_size, "blob payload"))
   /* GCOVR_EXCL_BR_STOP */
   {
      free(*output);
      *output = NULL;
      return 0;
   }
   return 1;
}

/* No file identity (no caching) or no resumable decoder for this codec:
 * read the whole blob and decode just the slice from memory. */
static int
LSSeqDecodeWholeBlobSlice(FILE *fp, comp_alg_t codec, uint64_t blob_offset,
                          uint64_t c_size, uint64_t decomp_offset, uint64_t decomp_size,
                          void **output)
{
   uint64_t declared_size = 0;
   void    *blob          = NULL;
   int      ok            = 0;

   /* The fallback allocates the entire decoded blob, not just the slice.
    * Validate its header before allowing the decoder to allocate it. */
   if (c_size < sizeof(declared_size) ||
       !LSSeqReadAt(fp, blob_offset, &declared_size, sizeof(declared_size),
                    "decoded blob size") ||
       declared_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid decoded size for non-streamed part blob "
                           "(%llu bytes; limit %u)",
                           (unsigned long long)declared_size, LSSEQ_MAX_BLOB_BYTES);
      return 0;
   }
   blob = hypredrv_Malloc((size_t)c_size, "blob");
   if (!blob) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   ok = LSSeqReadAt(fp, blob_offset, blob, (size_t)c_size, "blob payload") &&
        hypredrv_decompress_slice(codec, (size_t)c_size, blob, (size_t)decomp_offset,
                                  (size_t)decomp_size, output);
   /* GCOVR_EXCL_BR_STOP */
   free(blob);
   return ok;
}

/* Read one system's slice from a part's batched blob. Slot is selected by
 * the reader: 0 = values, 1 = RHS, 2 = dofmap. */
static int
LSSeqReadPartBlobSlice(const LSSeqData *seq, uint32_t part_id, int slot,
                       uint64_t decomp_offset, uint64_t decomp_size, void **output)
{
   const uint64_t *part_blob =
      seq->part_blob_table + ((size_t)part_id * LSSEQ_PART_BLOB_ENTRIES);
   uint64_t   c_off       = part_blob[(size_t)slot * 2];
   uint64_t   c_size      = part_blob[((size_t)slot * 2) + 1];
   uint64_t   blob_offset = 0, blob_end = 0;
   FILE      *fp    = seq->fp;
   comp_alg_t codec = (comp_alg_t)seq->header.codec;
   *output          = NULL;
   if (decomp_offset > (uint64_t)SIZE_MAX || decomp_size > (uint64_t)SIZE_MAX ||
       decomp_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Requested decoded slice exceeds limits (offset=%llu size=%llu)",
         (unsigned long long)decomp_offset, (unsigned long long)decomp_size);
      return 0;
   }
   if (!c_size)
   {
      return decomp_size == 0;
   }
   if (c_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Blob size exceeds limit (%llu bytes)",
                           (unsigned long long)c_size);
      return 0;
   }
   if (!LSSeqCheckedAddU64(seq->header.offset_blob_data, c_off, &blob_offset,
                           "part blob offset") ||
       !LSSeqCheckedAddU64(blob_offset, c_size, &blob_end, "part blob end"))
   {
      return 0;
   }
   if (codec == COMP_NONE)
   {
      return LSSeqReadRawBlobSlice(fp, blob_offset, c_size, decomp_offset, decomp_size,
                                   output);
   }

   /* Streaming codecs decode through a cached decoder for this exact blob that
    * reads its compressed input from the file on demand: reading a part's
    * slices in order inflates the blob once and never holds a copy of it. */
   LSSeqFileBlob         file_blob = {fp, blob_offset};
   hypredrv_SliceSource  src       = {LSSeqFileBlobRead, &file_blob};
   LSSeqStreamKey        key;
   int                   keyed = LSSeqStreamKeyMake(fp, codec, blob_offset, c_size, &key);
   hypredrv_SliceStream *stream = keyed ? LSSeqStreamCacheFind(&key) : NULL;
   if (!stream && keyed)
   {
      stream = LSSeqStreamCacheInsert(&key, codec, (size_t)c_size, &src);
   }
   if (stream)
   {
      return hypredrv_SliceStreamRead(stream, &src, (size_t)decomp_offset,
                                      (size_t)decomp_size, output);
   }
   return LSSeqDecodeWholeBlobSlice(fp, codec, blob_offset, c_size, decomp_offset,
                                    decomp_size, output);
}

static int
LSSeqSynchronizeMPIStatus(MPI_Comm comm, int local_ok, hypredrv_error_t fallback_code,
                          const char *fallback_msg)
{
   int global_ok = 0;

   if (!local_ok && !hypredrv_ErrorCodeActive())
   {
      hypredrv_ErrorCodeSet(fallback_code);
      hypredrv_ErrorMsgAdd("%s", fallback_msg ? fallback_msg : "LSSeq MPI read failed");
   }

   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, comm);
   if (!global_ok)
   {
      (void)hypredrv_DistributedErrorStateSync(comm);
      return 0;
   }

   return 1;
}

int
hypredrv_LSSeqReadSummary(const char *filename, int *num_systems, int *num_patterns,
                          int *has_dofmap, int *has_timesteps)
{
   LSSeqData   seq    = {0};
   int         ok     = LSSeqOpenHeader(filename, &seq);
   LSSeqHeader header = seq.header;
   LSSeqDataDestroy(&seq);
   if (!ok)
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      *num_systems = (int)header.num_systems;
   }
   if (num_patterns)
   {
      *num_patterns = (int)header.num_patterns;
   }
   if (has_dofmap)
   {
      *has_dofmap = ((header.flags & LSSEQ_FLAG_HAS_DOFMAP) != 0);
   }
   if (has_timesteps)
   {
      *has_timesteps = ((header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) != 0);
   }

   return 1;
}

/* Checks that a decoded per-system slice holds exactly `count` entries of
 * `width` bytes, so consumers never read past a short (corrupt) slice. */
static int
LSSeqSliceSizeMatches(uint64_t got, uint64_t count, uint64_t width, int ls_id,
                      uint32_t part_id, const char *what)
{
   size_t expected = 0;

   if (!LSSeqCheckedMulSize((size_t)count, (size_t)width, &expected, what))
   {
      return 0;
   }
   if (got != (uint64_t)expected)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq %s for system %d part %u has %llu bytes, expected %zu",
                           what, ls_id, part_id, (unsigned long long)got, expected);
      return 0;
   }
   return 1;
}

static int
LSSeqPartSourceLoad(void *ctx, uint32_t p, int want, hypredrv_IJMatrixMemPart *out)
{
   const LSSeqData           *seq     = (const LSSeqData *)ctx;
   uint32_t                   part_id = seq->part_order[seq->first_part + p];
   const LSSeqPartMeta       *part    = &seq->parts[part_id];
   const LSSeqSystemPartMeta *sys =
      &seq->sys_parts[((size_t)seq->ls_id * (size_t)seq->header.num_parts) +
                      (size_t)part_id];
   const LSSeqPatternMeta *pattern       = NULL;
   size_t                  expected_size = 0;

   if (sys->pattern_id >= seq->header.num_patterns)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid pattern id %u for system %d part %u", sys->pattern_id,
                           seq->ls_id, part_id);
      return 0;
   }
   pattern = &seq->patterns[sys->pattern_id];
   if (pattern->part_id != part_id)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Pattern-part mismatch for system %d part %u (pattern part=%u)", seq->ls_id,
         part_id, pattern->part_id);
      return 0;
   }

   out->nrows      = part->row_upper - part->row_lower + 1;
   out->nnz        = pattern->nnz;
   out->index_size = part->row_index_size;
   out->value_size = part->value_size;
   out->label      = "LSSeq matrix part";

   /* GCOVR_EXCL_BR_START */
   if ((want & HYPREDRV_PART_INDICES) &&
       !(LSSeqCheckedMulSize((size_t)pattern->nnz, (size_t)part->row_index_size,
                             &expected_size, "matrix index blob size") &&
         LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES,
                                "matrix index blob") &&
         LSSeqReadBlob(seq, pattern->rows_blob_offset, pattern->rows_blob_size,
                       expected_size, &out->rows) &&
         LSSeqReadBlob(seq, pattern->cols_blob_offset, pattern->cols_blob_size,
                       expected_size, &out->cols)))
   {
      return 0;
   }
   /* Check the stored slice size against the pattern before reading, so a
    * corrupt size cannot drive the allocation. */
   if ((want & HYPREDRV_PART_VALUES) &&
       !(LSSeqSliceSizeMatches(sys->values_blob_size, out->nnz, out->value_size,
                               seq->ls_id, part_id, "matrix values") &&
         LSSeqReadPartBlobSlice(seq, part_id, 0, sys->values_blob_offset,
                                sys->values_blob_size, &out->vals)))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   return 1;
}

/* Selects one system and this rank's contiguous range of row-ordered parts.
 * Local failures are synchronized by the caller before entering any builder. */
static int
LSSeqPrepareRead(MPI_Comm comm, LSSeqData *seq, int ls_id)
{
   if (ls_id < 0 || ls_id >= (int)seq->header.num_systems)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq->header.num_systems);
      return 0;
   }
   int nprocs = 0, myid = 0;
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   if (seq->header.num_parts < (uint32_t)nprocs)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Invalid number of sequence parts (%u) for communicator size %d",
         seq->header.num_parts, nprocs);
      return 0;
   }
   uint64_t first_part = 0, local_nparts = 0;
   hypredrv_MultipartRange(seq->header.num_parts, nprocs, myid, &first_part,
                           &local_nparts);
   seq->first_part      = (uint32_t)first_part;
   seq->num_local_parts = (uint32_t)local_nparts;
   seq->ls_id           = ls_id;
   uint32_t  n          = seq->header.num_parts;
   uint32_t *order      = hypredrv_Malloc((size_t)n * sizeof(*order), "part order");
   if (!order) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   for (uint32_t i = 0; i < n; i++)
   {
      order[i] = i;
   }

   /* Stable insertion sort by row_lower then row_upper. */
   for (uint32_t i = 1; i < n; i++)
   {
      uint32_t key    = order[i];
      uint64_t key_lo = seq->parts[key].row_lower;
      uint64_t key_hi = seq->parts[key].row_upper;
      int      j      = (int)i - 1;
      while (j >= 0)
      {
         uint32_t cur    = order[(size_t)j];
         uint64_t cur_lo = seq->parts[cur].row_lower;
         uint64_t cur_hi = seq->parts[cur].row_upper;
         /* GCOVR_EXCL_BR_START */
         if (cur_lo < key_lo || (cur_lo == key_lo && cur_hi <= key_hi))
         /* GCOVR_EXCL_BR_STOP */
         {
            break;
         }
         order[(size_t)j + 1u] = cur;
         j--;
      }
      order[(size_t)j + 1u] = key;
   }

   seq->part_order = order;
   return 1;
}

/* hypredrv_IJVectorPartSource over this rank's RHS parts of one LSSeq system;
 * values are decoded only when requested. */
static int
LSSeqRHSSourceLoad(void *ctx, uint32_t p, int want_values, hypredrv_IJVectorMemPart *out)
{
   const LSSeqData           *seq     = (const LSSeqData *)ctx;
   uint32_t                   part_id = seq->part_order[seq->first_part + p];
   const LSSeqSystemPartMeta *sys =
      &seq->sys_parts[((size_t)seq->ls_id * (size_t)seq->header.num_parts) +
                      (size_t)part_id];

   out->nrows      = seq->parts[part_id].nrows;
   out->value_size = seq->parts[part_id].value_size;
   out->label      = "LSSeq RHS part";

   return !want_values ||
          (LSSeqSliceSizeMatches(sys->rhs_blob_size, out->nrows, out->value_size,
                                 seq->ls_id, part_id, "RHS values") &&
           LSSeqReadPartBlobSlice(seq, part_id, 1, sys->rhs_blob_offset,
                                  sys->rhs_blob_size, &out->vals));
}

int
hypredrv_LSSeqReadMatrix(MPI_Comm comm, const char *filename, int ls_id,
                         HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *matrix_ptr)
{
   if (!matrix_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null matrix pointer for LSSeqReadMatrix");
      return 0;
   }
   *matrix_ptr = NULL;

   LSSeqData seq = {0};
   int local_ok  = LSSeqDataLoad(filename, &seq) && LSSeqPrepareRead(comm, &seq, ls_id);
   int ok        = 0;
   if (LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                 "LSSeq matrix local decode failed"))
   {
      hypredrv_IJMatrixPartSource source = {&seq, seq.num_local_parts,
                                            LSSeqPartSourceLoad};
      hypredrv_IJMatrixBuildFromSource(comm, &source, memory_location, matrix_ptr);
      local_ok = *matrix_ptr != NULL && !hypredrv_ErrorCodeActive();
      ok       = LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_UNKNOWN,
                                           "LSSeq matrix collective import failed");
   }
   LSSeqDataDestroy(&seq);
   if (!ok && *matrix_ptr)
   {
      HYPRE_IJMatrixDestroy(*matrix_ptr);
      *matrix_ptr = NULL;
   }
   return ok;
}

int
hypredrv_LSSeqReadRHS(MPI_Comm comm, const char *filename, int ls_id,
                      HYPRE_MemoryLocation memory_location, HYPRE_IJVector *rhs_ptr)
{
   if (!rhs_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null RHS pointer for LSSeqReadRHS");
      return 0;
   }
   *rhs_ptr = NULL;

   LSSeqData seq = {0};
   int local_ok  = LSSeqDataLoad(filename, &seq) && LSSeqPrepareRead(comm, &seq, ls_id);
   int ok        = 0;
   if (LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                 "LSSeq RHS local decode failed"))
   {
      hypredrv_IJVectorPartSource source = {&seq, seq.num_local_parts,
                                            LSSeqRHSSourceLoad};
      hypredrv_IJVectorBuildFromSource(comm, &source, memory_location, rhs_ptr);
      local_ok = *rhs_ptr != NULL && !hypredrv_ErrorCodeActive();
      ok       = LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_UNKNOWN,
                                           "LSSeq RHS collective import failed");
   }
   LSSeqDataDestroy(&seq);
   if (!ok && *rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }
   return ok;
}

/* The dofmap payload is int32; IntArray stores it as int. */
_Static_assert(sizeof(int) == sizeof(int32_t), "LSSeq dofmap requires 32-bit int");

/* This rank's parts are the same contiguous range IntArrayParRead would
 * assign, so concatenate their slices and build the distributed array.
 * Rank-local only. On failure, *local_ptr keeps what was gathered so far for
 * the caller to free. */
static int
LSSeqGatherLocalDofmap(const LSSeqData *seq, int **local_ptr, size_t *count_ptr)
{
   for (uint32_t i = 0; i < seq->num_local_parts; i++)
   {
      uint32_t                   part_id = seq->part_order[seq->first_part + i];
      const LSSeqSystemPartMeta *sys =
         &seq->sys_parts[((size_t)seq->ls_id * seq->header.num_parts) + part_id];
      if (sys->dof_num_entries > LSSEQ_MAX_BLOB_BYTES / sizeof(int32_t))
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("LSSeq dofmap part %u exceeds the payload limit", part_id);
         return 0;
      }
      size_t entries       = (size_t)sys->dof_num_entries;
      size_t expected_size = entries * sizeof(int32_t);
      void  *dof_data      = NULL;
      if (!entries)
      {
         continue;
      }
      size_t max_entries = SIZE_MAX / sizeof(int);
      if (max_entries > (size_t)INT_MAX)
      {
         max_entries = (size_t)INT_MAX;
      }
      if (entries > max_entries - *count_ptr)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("LSSeq local dofmap exceeds the supported entry count");
         return 0;
      }
      if (!LSSeqReadPartBlobSlice(seq, part_id, 2, sys->dof_blob_offset, expected_size,
                                  &dof_data) ||
          !dof_data)
      {
         free(dof_data);
         return 0;
      }
      int *grown = realloc(*local_ptr, (*count_ptr + entries) * sizeof(int));
      if (!grown)
      {
         free(dof_data);
         hypredrv_AllocationFailure("Failed to allocate LSSeq dofmap buffer");
         return 0;
      }
      memcpy(grown + *count_ptr, dof_data, entries * sizeof(int));
      *local_ptr = grown;
      *count_ptr += entries;
      free(dof_data);
   }
   return 1;
}

int
hypredrv_LSSeqReadDofmap(MPI_Comm comm, const char *filename, int ls_id,
                         IntArray **dofmap_ptr)
{
   LSSeqData seq      = {0};
   int      *local    = NULL;
   size_t    count    = 0;
   int       local_ok = 1;
   int       ok       = 0;

   if (!dofmap_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null dofmap pointer for LSSeqReadDofmap");
      return 0;
   }

   if (*dofmap_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(dofmap_ptr);
   }
   *dofmap_ptr = NULL;

   /* GCOVR_EXCL_BR_START */
   local_ok = LSSeqDataLoad(filename, &seq);
   if (local_ok && (seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP)) /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = LSSeqPrepareRead(comm, &seq, ls_id);
   }

   /* Agree on the local status before the no-dofmap early return and the
    * collective build, so no rank is left waiting in a collective that a
    * failed rank skipped. */
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq dofmap local decode failed"))
   {
      goto cleanup;
   }

   if (!(seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP))
   {
      *dofmap_ptr = hypredrv_IntArrayCreate(0);
      LSSeqDataDestroy(&seq);
      return (*dofmap_ptr != NULL);
   }

   local_ok = LSSeqGatherLocalDofmap(&seq, &local, &count);

   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq dofmap local decode failed"))
   {
      goto cleanup;
   }

   hypredrv_IntArrayBuild(comm, (int)count, local, dofmap_ptr);
   local_ok = (!hypredrv_ErrorCodeActive() && *dofmap_ptr != NULL);
   /* GCOVR_EXCL_BR_START */
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_UNKNOWN,
                                  "LSSeq dofmap collective import failed"))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (!ok && *dofmap_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(dofmap_ptr);
   }
   free(local);
   LSSeqDataDestroy(&seq);
   return ok;
}

int
hypredrv_LSSeqReadTimestepsWithIds(const char *filename, IntArray **timestep_ids,
                                   IntArray **timestep_starts)
{
   LSSeqData seq;
   IntArray *ids    = NULL;
   IntArray *starts = NULL;

   if (!timestep_starts)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid output pointer for LSSeqReadTimesteps");
      return 0;
   }

   if (timestep_ids && *timestep_ids)
   {
      hypredrv_IntArrayDestroy(timestep_ids);
   }

   if (*timestep_starts)
   {
      hypredrv_IntArrayDestroy(timestep_starts);
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!(seq.header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) || seq.header.num_timesteps == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }

   if (timestep_ids)
   {
      ids = hypredrv_IntArrayCreate((size_t)seq.header.num_timesteps);
      if (!ids) /* GCOVR_EXCL_BR_LINE */
      {
         LSSeqDataDestroy(&seq);
         hypredrv_AllocationFailure("Failed to allocate LSSeq timestep ids array");
         return 0;
      }
   }

   starts = hypredrv_IntArrayCreate((size_t)seq.header.num_timesteps);
   if (!starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(&ids);
      LSSeqDataDestroy(&seq);
      hypredrv_AllocationFailure("Failed to allocate LSSeq timestep starts array");
      return 0;
   }

   for (uint32_t i = 0; i < seq.header.num_timesteps; i++)
   {
      if (ids)
      {
         ids->data[i] = seq.timesteps[i].timestep;
      }
      starts->data[i] = seq.timesteps[i].ls_start;
   }

   if (timestep_ids)
   {
      *timestep_ids = ids;
   }
   *timestep_starts = starts;
   LSSeqDataDestroy(&seq);
   return 1;
}
