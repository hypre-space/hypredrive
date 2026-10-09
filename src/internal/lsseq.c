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

typedef struct LSSeqData_struct
{
   LSSeqHeader          header;
   LSSeqInfoHeader      info_header;
   char                *info_payload;
   size_t               info_payload_size;
   LSSeqPartMeta       *parts;
   LSSeqPatternMeta    *patterns;
   LSSeqSystemPartMeta *sys_parts;
   uint64_t            *part_blob_table; /* 6*num_parts entries */
   LSSeqTimestepEntry  *timesteps;
} LSSeqData;

enum
{
   LSSEQ_INFO_PAYLOAD_MAX_BYTES = 16u * 1024u * 1024u,
   LSSEQ_MAX_META_BYTES         = 512u * 1024u * 1024u,
   LSSEQ_MAX_BLOB_BYTES         = 512u * 1024u * 1024u,
   LSSEQ_MAX_PARTS              = 1024u * 1024u,
   LSSEQ_MAX_SYSTEMS            = 1024u * 1024u,
   LSSEQ_MAX_PATTERNS           = 1024u * 1024u,
   LSSEQ_MAX_TIMESTEPS          = 1024u * 1024u,
};

static int
LSSeqCheckedMulSize(size_t a, size_t b, size_t *result, const char *what)
{
   /* GCOVR_EXCL_START */
   if (!result)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null output while checking LSSeq size multiplication");
      return 0;
   }

   if (a != 0 && b > SIZE_MAX / a)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq size overflow while computing %s (%zu * %zu)",
                           what ? what : "allocation size", a, b);
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
   if (!result)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null output while checking LSSeq offset addition");
      return 0;
   }

   if (UINT64_MAX - a < b)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq offset overflow while computing %s",
                           what ? what : "offset");
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
      hypredrv_ErrorMsgAdd("LSSeq %s exceeds limit (%zu > %zu bytes)",
                           what ? what : "allocation", nbytes, max_nbytes);
      return 0;
   }

   return 1;
   /* GCOVR_EXCL_STOP */
}

static int
LSSeqBuildPartOrder(const LSSeqData *seq, uint32_t **order_ptr)
{
   uint32_t *order = NULL;
   uint32_t  n     = 0;

   if (!seq || !order_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments for LSSeqBuildPartOrder");
      return 0;
   }
   *order_ptr = NULL;
   n          = seq->header.num_parts;

   order = (uint32_t *)malloc((size_t)n * sizeof(*order));
   if (!order) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate part order (%u entries)", n);
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

   *order_ptr = order;
   return 1;
}

static uint64_t
LSSeqFNV1a64(const void *data, size_t nbytes, uint64_t hash)
{
   const unsigned char *bytes = (const unsigned char *)data;
   if (!bytes) /* GCOVR_EXCL_BR_LINE */
   {
      return hash;
   }

   for (size_t i = 0; i < nbytes; i++)
   {
      hash ^= (uint64_t)bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static int
LSSeqReadAt(FILE *fp, uint64_t offset, void *buffer, size_t nbytes, const char *what)
{
   /* GCOVR_EXCL_BR_START */
   if (!fp || !buffer) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments while reading %s",
                           what ? what : "lsseq data");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
#ifdef _MSC_VER
   if (_fseeki64(fp, (int64_t)offset, SEEK_SET) != 0) /* GCOVR_EXCL_BR_STOP */
#else
   if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) /* GCOVR_EXCL_BR_STOP */
#endif
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not seek to offset %llu while reading %s",
                           (unsigned long long)offset, what ? what : "lsseq data");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (nbytes > 0 && fread(buffer, 1, nbytes, fp) != nbytes) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read %s (%zu bytes)", what ? what : "lsseq data",
                           nbytes);
      return 0;
   }

   return 1;
}

static int
LSSeqValidateHeader(const LSSeqHeader *header, const char *filename)
{
   /* GCOVR_EXCL_BR_START */
   if (!header) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null lsseq header");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->magic != LSSEQ_MAGIC) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq magic for file '%s'", filename ? filename : "");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->version != LSSEQ_VERSION) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Unsupported LSSeq version %u in '%s'", header->version,
                           filename ? filename : "");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->offset_part_blob_table == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "LSSeq format requires part blob table (offset_part_blob_table) in '%s'",
         filename ? filename : "");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->num_systems == 0 || header->num_parts == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq shape in '%s': num_systems=%u, num_parts=%u",
                           filename ? filename : "", header->num_systems,
                           header->num_parts);
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
                           filename ? filename : "", header->num_systems,
                           header->num_parts, header->num_patterns,
                           header->num_timesteps);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (header->codec > (uint32_t)COMP_BLOSC) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq compression codec id %u in '%s'", header->codec,
                           filename ? filename : "");
      return 0;
   }

   return 1;
}

static void
LSSeqDataDestroy(LSSeqData *seq)
{
   /* GCOVR_EXCL_BR_START */
   if (!seq) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }
   free(seq->info_payload);
   free(seq->parts);
   free(seq->patterns);
   free(seq->sys_parts);
   free(seq->part_blob_table);
   free(seq->timesteps);
   memset(seq, 0, sizeof(*seq));
}

/* Reads and validates the mandatory LSSeq info header and its payload. The
 * stream is closed on every failure path, matching the caller's contract. */
static int
LSSeqLoadInfoHeader(FILE *fp, LSSeqData *seq, const char *filename, uint64_t info_offset)
{
   uint64_t expected_min_part_offset = info_offset + (uint64_t)sizeof(LSSeqInfoHeader);
   uint64_t expected_payload_end     = 0;

   if (!(seq->header.flags & LSSEQ_FLAG_HAS_INFO))
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Missing mandatory LSSeq info header in '%s'", filename);
      return 0;
   }

   if (seq->header.offset_part_meta < expected_min_part_offset)
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq info offsets in '%s' (offset_part_meta=%llu)",
                           filename, (unsigned long long)seq->header.offset_part_meta);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, info_offset, &seq->info_header, sizeof(seq->info_header),
                    "info header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.magic != LSSEQ_INFO_MAGIC ||
       seq->info_header.version != LSSEQ_INFO_VERSION)
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq info header in '%s' (magic=%llu version=%u)",
                           filename, (unsigned long long)seq->info_header.magic,
                           seq->info_header.version);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->info_header.endian_tag != UINT32_C(0x01020304)) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
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
      fclose(fp);
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
      fclose(fp);
      return 0;
   }
   if (seq->header.offset_part_meta < expected_payload_end)
   {
      fclose(fp);
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
      seq->info_payload      = (char *)malloc(seq->info_payload_size + 1u);
      /* GCOVR_EXCL_BR_START */
      if (!seq->info_payload) /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate LSSeq info payload (%zu bytes)",
                              seq->info_payload_size);
         return 0;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqReadAt(fp, expected_min_part_offset, seq->info_payload,
                       seq->info_payload_size, "info payload"))
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
      seq->info_payload[seq->info_payload_size] = '\0';

      hash = LSSeqFNV1a64(seq->info_payload, seq->info_payload_size, hash);
      /* GCOVR_EXCL_BR_START */
      if (hash != seq->info_header.payload_hash_fnv1a64) /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'", filename);
         LSSeqDataDestroy(seq);
         return 0;
      }
   }

   return 1;
}

/* Allocates the part, pattern, system-part and timestep metadata arrays sized by
 * the validated header. Closes the stream and reports on any failure. */
static int
LSSeqAllocMetadata(FILE *fp, LSSeqData *seq, size_t *n_sys_parts_out)
{
   size_t n_sys_parts = 0;

   {
      size_t part_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_parts, sizeof(LSSeqPartMeta),
                               &part_meta_bytes, "part metadata bytes") ||
          !LSSeqValidateByteLimit(part_meta_bytes, LSSEQ_MAX_META_BYTES, "part metadata"))
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         return 0;
      }
   }
   seq->parts =
      (LSSeqPartMeta *)calloc((size_t)seq->header.num_parts, sizeof(LSSeqPartMeta));
   /* GCOVR_EXCL_BR_START */
   if (!seq->parts) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq part metadata");
      return 0;
   }

   if (seq->header.num_patterns > 0)
   {
      size_t pattern_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_patterns, sizeof(LSSeqPatternMeta),
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */
                               &pattern_meta_bytes, "pattern metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(pattern_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "pattern metadata"))
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
      seq->patterns =
         (LSSeqPatternMeta *)calloc(seq->header.num_patterns, sizeof(LSSeqPatternMeta));
      if (!seq->patterns) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate LSSeq pattern metadata");
         LSSeqDataDestroy(seq);
         return 0;
      }
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqCheckedMulSize((size_t)seq->header.num_systems,
                            /* GCOVR_EXCL_BR_STOP */
                            (size_t)seq->header.num_parts, &n_sys_parts,
                            "system-part count"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }
   {
      size_t sys_part_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize(n_sys_parts, sizeof(LSSeqSystemPartMeta),
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */
                               &sys_part_meta_bytes, "system-part metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(sys_part_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "system-part metadata"))
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
   }
   seq->sys_parts =
      (LSSeqSystemPartMeta *)calloc(n_sys_parts, sizeof(LSSeqSystemPartMeta));
   if (!seq->sys_parts) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq system-part metadata");
      LSSeqDataDestroy(seq);
      return 0;
   }

   if ((seq->header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq->header.num_timesteps > 0)
   {
      size_t timestep_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_timesteps,
                               /* GCOVR_EXCL_BR_STOP */
                               sizeof(LSSeqTimestepEntry), &timestep_meta_bytes,
                               /* GCOVR_EXCL_BR_START */
                               "timestep metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(timestep_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "timestep metadata"))
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
      seq->timesteps = (LSSeqTimestepEntry *)calloc(seq->header.num_timesteps,
                                                    sizeof(LSSeqTimestepEntry));
      if (!seq->timesteps) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate LSSeq timesteps metadata");
         LSSeqDataDestroy(seq);
         return 0;
      }
   }

   *n_sys_parts_out = n_sys_parts;

   return 1;
}

static int
LSSeqDataLoad(const char *filename, LSSeqData *seq)
{
   FILE          *fp          = NULL;
   size_t         n_sys_parts = 0;
   const uint64_t info_offset = (uint64_t)sizeof(LSSeqHeader);

   /* GCOVR_EXCL_BR_START */
   if (!filename || !seq) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments to LSSeqDataLoad");
      return 0;
   }

   memset(seq, 0, sizeof(*seq));
   fp = fopen(filename, "rb");
   /* GCOVR_EXCL_BR_START */
   if (!fp) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, 0, &seq->header, sizeof(seq->header), "lsseq header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      return 0;
   }

   if (!LSSeqValidateHeader(&seq->header, filename))
   {
      fclose(fp);
      return 0;
   }

   if (!LSSeqLoadInfoHeader(fp, seq, filename, info_offset))
   {
      return 0;
   }

   if (!LSSeqAllocMetadata(fp, seq, &n_sys_parts))
   {
      return 0;
   }

   if (!LSSeqReadAt(fp, seq->header.offset_part_meta, seq->parts,
                    (size_t)seq->header.num_parts * sizeof(LSSeqPartMeta),
                    "part metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->header.num_patterns > 0 &&
       /* GCOVR_EXCL_BR_STOP */
       !LSSeqReadAt(fp, seq->header.offset_pattern_meta, seq->patterns,
                    (size_t)seq->header.num_patterns * sizeof(LSSeqPatternMeta),
                    "pattern metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, seq->header.offset_sys_part_meta, seq->sys_parts,
                    /* GCOVR_EXCL_BR_STOP */
                    n_sys_parts * sizeof(LSSeqSystemPartMeta), "system-part metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   {
      size_t pt_entries = 0;
      size_t pt_size    = 0;
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize((size_t)LSSEQ_PART_BLOB_ENTRIES,
                               /* GCOVR_EXCL_BR_STOP */
                               (size_t)seq->header.num_parts, &pt_entries,
                               /* GCOVR_EXCL_BR_START */
                               "part blob table entries") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqCheckedMulSize(pt_entries, sizeof(uint64_t), &pt_size,
                               /* GCOVR_EXCL_BR_START */
                               "part blob table bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(pt_size, LSSEQ_MAX_META_BYTES, "part blob table"))
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
      seq->part_blob_table = (uint64_t *)malloc(pt_size);
      /* GCOVR_EXCL_BR_START */
      if (!seq->part_blob_table ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqReadAt(fp, seq->header.offset_part_blob_table, seq->part_blob_table,
                       pt_size, "part blob table"))
      {
         fclose(fp);
         LSSeqDataDestroy(seq);
         return 0;
      }
   }

   /* GCOVR_EXCL_BR_START */
   if (seq->timesteps &&
       /* GCOVR_EXCL_BR_STOP */
       !LSSeqReadAt(fp, seq->header.offset_timestep_meta, seq->timesteps,
                    (size_t)seq->header.num_timesteps * sizeof(LSSeqTimestepEntry),
                    "timestep metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   fclose(fp);
   return 1;
}

int
hypredrv_LSSeqReadInfo(const char *filename, char **payload_ptr, size_t *payload_size)
{
   FILE           *fp = NULL;
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   char           *payload     = NULL;
   size_t          nbytes      = 0;
   uint64_t        hash        = UINT64_C(1469598103934665603);
   const uint64_t  info_offset = (uint64_t)sizeof(LSSeqHeader);

   if (!payload_ptr || !payload_size)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid LSSeqReadInfo outputs");
      return 0;
   }
   *payload_ptr  = NULL;
   *payload_size = 0;

   if (!filename)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing sequence filename");
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, 0, &header, sizeof(header), "lsseq header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      return 0;
   }
   if (!LSSeqValidateHeader(&header, filename)) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      return 0;
   }

   if (!(header.flags & LSSEQ_FLAG_HAS_INFO))
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Missing mandatory LSSeq info header in '%s'", filename);
      return 0;
   }

   if (!LSSeqReadAt(fp, info_offset, &info, sizeof(info), "info header"))
   {
      fclose(fp);
      return 0;
   }

   if (info.magic != LSSEQ_INFO_MAGIC || info.version != LSSEQ_INFO_VERSION ||
       info.endian_tag != UINT32_C(0x01020304))
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq info header in '%s'", filename);
      return 0;
   }

   if (info.payload_size > (uint64_t)LSSEQ_INFO_PAYLOAD_MAX_BYTES ||
       /* GCOVR_EXCL_BR_START */
       info.payload_size > (uint64_t)SIZE_MAX - 1u)
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq info payload too large in '%s' (%llu bytes)", filename,
                           (unsigned long long)info.payload_size);
      return 0;
   }

   nbytes  = (size_t)info.payload_size;
   payload = (char *)malloc(nbytes + 1u);
   if (!payload) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq info payload (%zu bytes)", nbytes);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (nbytes > 0 && !LSSeqReadAt(fp, info_offset + (uint64_t)sizeof(info), payload,
                                  /* GCOVR_EXCL_BR_STOP */
                                  nbytes, "info payload"))
   {
      free(payload);
      fclose(fp);
      return 0;
   }
   payload[nbytes] = '\0';

   hash = LSSeqFNV1a64(payload, nbytes, hash);
   if (hash != info.payload_hash_fnv1a64)
   {
      free(payload);
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'", filename);
      return 0;
   }

   fclose(fp);
   *payload_ptr  = payload;
   *payload_size = nbytes;
   return 1;
}

static int
LSSeqLocalPartIDs(MPI_Comm comm, uint32_t g_nparts, int **partids_ptr, int *nparts_ptr)
{
   int      nprocs       = 0;
   int      myid         = 0;
   int      nparts       = 0;
   int      offset       = 0;
   int     *partids      = NULL;
   uint64_t local_nparts = 0, first_part = 0;

   if (!partids_ptr || !nparts_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid LSSeqLocalPartIDs outputs");
      return 0;
   }

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   hypredrv_MultipartRange((uint64_t)g_nparts, nprocs, myid, &first_part, &local_nparts);
   nparts = (int)local_nparts;
   /* GCOVR_EXCL_BR_START */
   if (g_nparts < (uint32_t)nprocs) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Invalid number of sequence parts (%u) for communicator size %d", g_nparts,
         nprocs);
      return 0;
   }

   partids = (int *)calloc((size_t)nparts, sizeof(int));
   if (!partids) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate partids array");
      return 0;
   }

   /* Keep the same mapping as multipart readers in matrix/vector containers. */
   offset = (int)first_part;
   for (int i = 0; i < nparts; i++)
   {
      partids[i] = offset + i;
   }

   *partids_ptr = partids;
   *nparts_ptr  = nparts;
   return 1;
}

static int
LSSeqReadBlob(FILE *fp, comp_alg_t codec, uint64_t offset, uint64_t blob_size,
              size_t expected_size, void **output, size_t *output_size)
{
   void  *blob_data    = NULL;
   void  *decoded      = NULL;
   size_t decoded_size = 0;

   if (!fp || !output || !output_size) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid LSSeq blob read arguments");
      return 0;
   }

   *output      = NULL;
   *output_size = 0;

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

   blob_data = malloc((size_t)blob_size);
   if (!blob_data) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate %llu bytes for blob read",
                           (unsigned long long)blob_size);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, offset, blob_data, (size_t)blob_size, "blob payload"))
   /* GCOVR_EXCL_BR_STOP */
   {
      free(blob_data);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (codec == COMP_NONE) /* GCOVR_EXCL_BR_STOP */
   {
      /* Raw payload: hand the read buffer over instead of copying it. */
      decoded      = blob_data;
      decoded_size = (size_t)blob_size;
      blob_data    = NULL;
   }
   else
   {
      hypredrv_decompress(codec, (size_t)blob_size, blob_data, &decoded_size, &decoded);
      if (hypredrv_ErrorCodeActive() || !decoded) /* GCOVR_EXCL_BR_LINE */
      {
         free(blob_data);
         return 0;
      }
   }

   free(blob_data);
   blob_data = NULL;

   if (expected_size != 0 && decoded_size != expected_size)
   {
      free(decoded);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Decoded blob size mismatch: expected=%zu got=%zu",
                           expected_size, decoded_size);
      return 0;
   }

   *output      = decoded;
   *output_size = decoded_size;
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

/* Creates a stream for `blob` and caches it, evicting the least recently used
 * entry. Returns NULL for codecs without resumable decoding. */
static hypredrv_SliceStream *
LSSeqStreamCacheInsert(const LSSeqStreamKey *key, comp_alg_t codec, size_t blob_size,
                       const void *blob)
{
   hypredrv_SliceStream *stream = hypredrv_SliceStreamCreate(codec, blob_size, blob);
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

/* v2 only: read a slice from a part's batched blob (slot: 0=values, 1=rhs, 2=dof) */
static int
LSSeqReadPartBlobSlice(FILE *fp, comp_alg_t codec, uint64_t blob_base,
                       const uint64_t *part_blob_table, uint32_t part_id, int slot,
                       uint64_t decomp_offset, uint64_t decomp_size, void **output,
                       size_t *output_size)
{
   uint64_t c_off, c_size;
   void    *blob = NULL;
   int      ok   = 0;

   if (!fp || !part_blob_table || !output || !output_size || slot < 0 || slot > 2)
   /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }
   *output      = NULL;
   *output_size = 0;
   c_off =
      part_blob_table[((size_t)part_id * LSSEQ_PART_BLOB_ENTRIES) + (size_t)(slot * 2)];
   c_size = part_blob_table[((size_t)part_id * LSSEQ_PART_BLOB_ENTRIES) +
                            (size_t)(slot * 2) + 1];
   if (decomp_size > (uint64_t)SIZE_MAX || decomp_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES)
   /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Requested decoded slice exceeds limit (%llu bytes)",
                           (unsigned long long)decomp_size);
      return 0;
   }
   if (c_size == 0)
   {
      /* GCOVR_EXCL_BR_START */
      if (decomp_size != 0) /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }
      return 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (c_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Blob size exceeds limit (%llu bytes)",
                           (unsigned long long)c_size);
      return 0;
   }
   /* Raw part blobs: read just this system's slice from the file. */
   if (codec == COMP_NONE)
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
      if (!*output ||
          !LSSeqReadAt(fp, blob_base + c_off + decomp_offset, *output,
                       (size_t)decomp_size, "blob payload")) /* GCOVR_EXCL_BR_STOP */
      {
         free(*output);
         *output = NULL;
         return 0;
      }
      *output_size = (size_t)decomp_size;
      return 1;
   }

   /* A cached decoder for this exact blob skips re-reading it. */
   LSSeqStreamKey key;
   int            keyed = LSSeqStreamKeyMake(fp, codec, blob_base + c_off, c_size, &key);
   hypredrv_SliceStream *stream = keyed ? LSSeqStreamCacheFind(&key) : NULL;
   if (stream)
   {
      if (!hypredrv_SliceStreamRead(stream, (size_t)decomp_offset, (size_t)decomp_size,
                                    output))
      {
         return 0;
      }
      *output_size = (size_t)decomp_size;
      return 1;
   }

   blob = malloc((size_t)c_size);
   if (!blob) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate %llu bytes for blob read",
                           (unsigned long long)c_size);
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   if (!LSSeqReadAt(fp, blob_base + c_off, blob, (size_t)c_size, "blob payload"))
   /* GCOVR_EXCL_BR_STOP */
   {
      free(blob);
      return 0;
   }

   /* The part blob batches every system's payload; decode only this system's
    * slice, resuming a cached decoder when the previous read left off before
    * it (streaming codecs stop at the slice end instead of inflating the rest). */
   stream = keyed ? LSSeqStreamCacheInsert(&key, codec, (size_t)c_size, blob) : NULL;
   ok     = stream
               ? hypredrv_SliceStreamRead(stream, (size_t)decomp_offset, (size_t)decomp_size,
                                          output)
               : hypredrv_decompress_slice(codec, (size_t)c_size, blob, (size_t)decomp_offset,
                                           (size_t)decomp_size, output);
   free(blob);
   if (!ok)
   {
      return 0;
   }
   *output_size = (size_t)decomp_size;
   return 1;
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
   LSSeqHeader header;
   FILE       *fp = NULL;

   if (!filename)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing sequence filename");
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }
   if (fread(&header, sizeof(header), 1, fp) != 1)
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read sequence header from '%s'", filename);
      return 0;
   }
   fclose(fp);

   if (!LSSeqValidateHeader(&header, filename))
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

/* hypredrv_IJMatrixPartSource over this rank's parts of one LSSeq system:
 * metadata comes from the sequence headers; indices and values are decoded
 * only when requested. */
typedef struct
{
   FILE            *fp;
   const LSSeqData *seq;
   int              ls_id;
   const int       *partids;
   const uint32_t  *part_order;
} LSSeqMatrixSource;

static int
LSSeqMatrixSourceLoad(void *ctx, uint32_t p, int want, hypredrv_IJMatrixMemPart *out)
{
   const LSSeqMatrixSource   *src     = (const LSSeqMatrixSource *)ctx;
   const LSSeqData           *seq     = src->seq;
   uint32_t                   part_id = src->part_order[src->partids[p]];
   const LSSeqPartMeta       *part    = &seq->parts[part_id];
   const LSSeqSystemPartMeta *sys =
      &seq->sys_parts[((size_t)src->ls_id * (size_t)seq->header.num_parts) +
                      (size_t)part_id];
   const LSSeqPatternMeta *pattern       = NULL;
   size_t                  expected_size = 0, rows_size = 0, cols_size = 0, vals_size = 0;

   if (sys->pattern_id >= seq->header.num_patterns)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid pattern id %u for system %d part %u", sys->pattern_id,
                           src->ls_id, part_id);
      return 0;
   }
   pattern = &seq->patterns[sys->pattern_id];
   if (pattern->part_id != part_id)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Pattern-part mismatch for system %d part %u (pattern part=%u)", src->ls_id,
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
         LSSeqReadBlob(src->fp, (comp_alg_t)seq->header.codec, pattern->rows_blob_offset,
                       pattern->rows_blob_size, expected_size, &out->rows, &rows_size) &&
         LSSeqReadBlob(src->fp, (comp_alg_t)seq->header.codec, pattern->cols_blob_offset,
                       pattern->cols_blob_size, expected_size, &out->cols, &cols_size)))
   {
      return 0;
   }
   if ((want & HYPREDRV_PART_VALUES) &&
       !LSSeqReadPartBlobSlice(src->fp, (comp_alg_t)seq->header.codec,
                               seq->header.offset_blob_data, seq->part_blob_table,
                               part_id, 0, sys->values_blob_offset, sys->values_blob_size,
                               &out->vals, &vals_size))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   return 1;
}

/* Prepares the rank-local state for reading system ls_id: validates the
 * id, collects this rank's part ids and the stored-to-runtime part order, and
 * opens the sequence file. Purely local (no collectives); returns 0 on any
 * local failure. */
static int
LSSeqPrepareRead(MPI_Comm comm, const LSSeqData *seq, int ls_id, const char *filename,
                 int **partids, int *nparts, uint32_t **part_order, FILE **fp)
{
   /* GCOVR_EXCL_BR_START */
   if (ls_id < 0 || ls_id >= (int)seq->header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq->header.num_systems);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqLocalPartIDs(comm, seq->header.num_parts, partids, nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   if (!LSSeqBuildPartOrder(seq, part_order)) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   *fp = fopen(filename, "rb");
   if (!*fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   return 1;
}

/* Decodes one part's RHS slice for system ls_id into `out` (vals owned by the
 * caller). Returns zero on failure. */
static int
LSSeqDecodeRHSPart(FILE *fp, const LSSeqData *seq, int ls_id, uint32_t part_id,
                   hypredrv_IJVectorMemPart *out)
{
   const LSSeqSystemPartMeta *sys =
      &seq->sys_parts[((size_t)ls_id * (size_t)seq->header.num_parts) + (size_t)part_id];
   size_t vals_size = 0;

   memset(out, 0, sizeof(*out));
   out->nrows      = seq->parts[part_id].nrows;
   out->value_size = seq->parts[part_id].value_size;
   out->label      = "LSSeq RHS part";

   return LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq->header.codec,
                                 seq->header.offset_blob_data, seq->part_blob_table,
                                 part_id, 1, sys->rhs_blob_offset, sys->rhs_blob_size,
                                 &out->vals, &vals_size);
}

int
hypredrv_LSSeqReadMatrix(MPI_Comm comm, const char *filename, int ls_id,
                         HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *matrix_ptr)
{
   LSSeqData seq        = {0};
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   int       local_ok   = 1;
   int       ok         = 0;

   if (!matrix_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null matrix pointer for LSSeqReadMatrix");
      return 0;
   }
   *matrix_ptr = NULL;

   /* GCOVR_EXCL_BR_START */
   local_ok =
      LSSeqDataLoad(filename, &seq) &&
      LSSeqPrepareRead(comm, &seq, ls_id, filename, &partids, &nparts, &part_order, &fp);
   /* GCOVR_EXCL_BR_STOP */
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq matrix local decode failed"))
   {
      goto cleanup;
   }

   {
      LSSeqMatrixSource           ctx = {fp, &seq, ls_id, partids, part_order};
      hypredrv_IJMatrixPartSource src = {&ctx, (uint32_t)nparts, LSSeqMatrixSourceLoad};
      hypredrv_IJMatrixBuildFromSource(comm, &src, memory_location, matrix_ptr);
   }
   local_ok = (!hypredrv_ErrorCodeActive() && *matrix_ptr != NULL);

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_UNKNOWN,
                                  "LSSeq matrix collective import failed"))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (!ok && *matrix_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      HYPRE_IJMatrixDestroy(*matrix_ptr);
      *matrix_ptr = NULL;
   }
   /* GCOVR_EXCL_BR_START */
   if (fp) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
   }
   free(part_order);
   free(partids);
   LSSeqDataDestroy(&seq);
   return ok;
}

int
hypredrv_LSSeqReadRHS(MPI_Comm comm, const char *filename, int ls_id,
                      HYPRE_MemoryLocation memory_location, HYPRE_IJVector *rhs_ptr)
{
   LSSeqData                 seq        = {0};
   FILE                     *fp         = NULL;
   int                      *partids    = NULL;
   uint32_t                 *part_order = NULL;
   int                       nparts     = 0;
   int                       local_ok   = 1;
   int                       ok         = 0;
   hypredrv_IJVectorMemPart *mem_parts  = NULL;

   if (!rhs_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null RHS pointer for LSSeqReadRHS");
      return 0;
   }
   *rhs_ptr = NULL;

   /* GCOVR_EXCL_BR_START */
   local_ok =
      LSSeqDataLoad(filename, &seq) &&
      LSSeqPrepareRead(comm, &seq, ls_id, filename, &partids, &nparts, &part_order, &fp);
   if (local_ok)
   {
      mem_parts = (hypredrv_IJVectorMemPart *)calloc(nparts ? (size_t)nparts : 1u,
                                                     sizeof(*mem_parts));
      local_ok = (mem_parts != NULL);
   }
   /* GCOVR_EXCL_BR_STOP */
   for (int i = 0; i < nparts && local_ok; i++)
   {
      local_ok =
         LSSeqDecodeRHSPart(fp, &seq, ls_id, part_order[partids[i]], &mem_parts[i]);
   }

   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq RHS local decode failed"))
   {
      goto cleanup;
   }

   hypredrv_IJVectorBuildFromParts(comm, mem_parts, (uint32_t)nparts, memory_location,
                                   rhs_ptr);
   local_ok = (!hypredrv_ErrorCodeActive() && *rhs_ptr != NULL);

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_UNKNOWN,
                                  "LSSeq RHS collective import failed"))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (!ok && *rhs_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }
   /* GCOVR_EXCL_BR_START */
   if (fp) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
   }
   for (int i = 0; mem_parts && i < nparts; i++)
   {
      free(mem_parts[i].vals);
   }
   free(mem_parts);
   free(part_order);
   free(partids);
   LSSeqDataDestroy(&seq);
   return ok;
}

/* The dofmap payload is int32; IntArray stores it as int. */
_Static_assert(sizeof(int) == sizeof(int32_t), "LSSeq dofmap requires 32-bit int");

/* Appends one part's dofmap slice for system ls_id to *local (growing it and
 * *count). Returns zero on any local failure. */
static int
LSSeqAppendDofmapPart(FILE *fp, const LSSeqData *seq, int ls_id, uint32_t part_id,
                      int **local, size_t *count)
{
   const LSSeqSystemPartMeta *sys =
      &seq->sys_parts[((size_t)ls_id * (size_t)seq->header.num_parts) + (size_t)part_id];
   size_t entries       = (size_t)sys->dof_num_entries;
   size_t expected_size = 0;
   void  *dof_data      = NULL;
   size_t dof_size      = 0;

   if (entries == 0)
   {
      return 1;
   }
   /* GCOVR_EXCL_BR_START */
   if (!LSSeqCheckedMulSize(entries, sizeof(int32_t), &expected_size,
                            "dof payload size") ||
       !LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES, "dof payload") ||
       !LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq->header.codec,
                               seq->header.offset_blob_data, seq->part_blob_table,
                               part_id, 2, sys->dof_blob_offset, (uint64_t)expected_size,
                               &dof_data, &dof_size) ||
       !dof_data || *count > (size_t)INT_MAX - entries) /* GCOVR_EXCL_BR_STOP */
   {
      free(dof_data);
      return 0;
   }

   int *grown = (int *)realloc(*local, (*count + entries) * sizeof(int));
   if (!grown) /* GCOVR_EXCL_BR_LINE */
   {
      free(dof_data);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq dofmap buffer");
      return 0;
   }
   memcpy(grown + *count, dof_data, entries * sizeof(int));
   *local = grown;
   *count += entries;
   free(dof_data);

   return 1;
}

int
hypredrv_LSSeqReadDofmap(MPI_Comm comm, const char *filename, int ls_id,
                         IntArray **dofmap_ptr)
{
   LSSeqData seq        = {0};
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   int      *local      = NULL;
   size_t    count      = 0;
   int       local_ok   = 1;
   int       ok         = 0;

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
      local_ok = LSSeqPrepareRead(comm, &seq, ls_id, filename, &partids, &nparts,
                                  &part_order, &fp);
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

   /* This rank's parts are the same contiguous range IntArrayParRead would
    * assign, so concatenate their slices and build the distributed array. */
   for (int i = 0; i < nparts && local_ok; i++)
   {
      local_ok =
         LSSeqAppendDofmapPart(fp, &seq, ls_id, part_order[partids[i]], &local, &count);
   }

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
   /* GCOVR_EXCL_BR_START */
   if (fp) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
   }
   free(local);
   free(part_order);
   free(partids);
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
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate LSSeq timestep ids array");
         return 0;
      }
   }

   starts = hypredrv_IntArrayCreate((size_t)seq.header.num_timesteps);
   if (!starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(&ids);
      LSSeqDataDestroy(&seq);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq timestep starts array");
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

int
hypredrv_LSSeqReadTimesteps(const char *filename, IntArray **timestep_starts)
{
   return hypredrv_LSSeqReadTimestepsWithIds(filename, NULL, timestep_starts);
}
