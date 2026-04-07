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
#include <sys/types.h>
#include <unistd.h>
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
LSSeqFormatPartFilename(char *filename, size_t filename_size, const char *prefix,
                        uint32_t part_id, const char *suffix)
{
   char   id_buf[16];
   int    id_len     = 0;
   size_t prefix_len = 0;
   size_t suffix_len = 0;
   size_t total_len  = 0;

   /* GCOVR_EXCL_BR_START */
   if (!filename || filename_size == 0 || !prefix) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments for LSSeqFormatPartFilename");
      return 0;
   }
   if (!suffix)
   {
      suffix = "";
   }
   if (!hypredrv_BinaryPathPrefixIsSafe(prefix))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid sequence staging path prefix");
      return 0;
   }

   id_len = snprintf(id_buf, sizeof(id_buf), "%05u", part_id);
   /* GCOVR_EXCL_BR_START */
   if (id_len < 0 || (size_t)id_len >= sizeof(id_buf)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Could not format part id %u", part_id);
      return 0;
   }

   prefix_len = strlen(prefix);
   suffix_len = strlen(suffix);
   total_len  = prefix_len + 1u + (size_t)id_len + suffix_len;
   /* GCOVR_EXCL_BR_START */
   if (total_len + 1u > filename_size) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LSSeq temporary part filename is too long for buffer (%zu "
                           "bytes)",
                           filename_size);
      return 0;
   }

   memcpy(filename, prefix, prefix_len);
   filename[prefix_len] = '.';
   memcpy(filename + prefix_len + 1u, id_buf, (size_t)id_len);
   memcpy(filename + prefix_len + 1u + (size_t)id_len, suffix, suffix_len);
   filename[total_len] = '\0';

   if (!hypredrv_BinaryPathPrefixIsSafe(filename))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid sequence part filename");
      return 0;
   }

   return 1;
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
   if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) /* GCOVR_EXCL_BR_STOP */
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
         hypredrv_ErrorMsgAdd(
            "Invalid LSSeq info offsets in '%s' (offset_part_meta=%llu)", filename,
            (unsigned long long)seq->header.offset_part_meta);
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
         hypredrv_ErrorMsgAdd("LSSeq info payload too large in '%s' (%llu bytes)",
                              filename,
                              (unsigned long long)seq->info_header.payload_size);
         return 0;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedAddU64(expected_min_part_offset,
                              (uint64_t)seq->info_header.payload_size,
                              &expected_payload_end, "info payload end"))
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
   }

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
   int  nprocs  = 0;
   int  myid    = 0;
   int  nparts  = 0;
   int  offset  = 0;
   int  rem     = 0;
   int *partids = NULL;

   if (!partids_ptr || !nparts_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid LSSeqLocalPartIDs outputs");
      return 0;
   }

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = (int)(g_nparts / (uint32_t)nprocs);
   rem    = (int)(g_nparts % (uint32_t)nprocs);
   nparts += (myid < rem) ? 1 : 0;
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
   offset = myid * nparts;
   offset += (myid < rem) ? myid : rem;
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
      decoded = malloc((size_t)blob_size);
      if (!decoded) /* GCOVR_EXCL_BR_LINE */
      {
         free(blob_data);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate %llu bytes for blob decode",
                              (unsigned long long)blob_size);
         return 0;
      }
      memcpy(decoded, blob_data, (size_t)blob_size);
      decoded_size = (size_t)blob_size;
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

/* v2 only: read a slice from a part's batched blob (slot: 0=values, 1=rhs, 2=dof) */
static int
LSSeqReadPartBlobSlice(FILE *fp, comp_alg_t codec, uint64_t blob_base,
                       const uint64_t *part_blob_table, uint32_t part_id, int slot,
                       uint64_t decomp_offset, uint64_t decomp_size, void **output,
                       size_t *output_size)
{
   uint64_t c_off, c_size;
   void    *decoded      = NULL;
   size_t   decoded_size = 0;
   void    *slice        = NULL;

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
   if (!LSSeqReadBlob(fp, codec, blob_base + c_off, c_size, 0, &decoded, &decoded_size))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   if (decomp_offset > UINT64_MAX - decomp_size ||
       /* GCOVR_EXCL_BR_STOP */
       decomp_offset + decomp_size > (uint64_t)decoded_size)
   {
      free(decoded);
      return 0;
   }
   slice = malloc((size_t)decomp_size);
   if (!slice) /* GCOVR_EXCL_BR_LINE */
   {
      free(decoded);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate slice buffer (%llu bytes)",
                           (unsigned long long)decomp_size);
      return 0;
   }
   memcpy(slice, (const char *)decoded + (size_t)decomp_offset, (size_t)decomp_size);
   free(decoded);
   *output      = slice;
   *output_size = (size_t)decomp_size;
   return 1;
}

/* Copy TMPDIR into a local buffer after validation; do not thread raw getenv into paths.
 */
static void
LSSeqSanitizedTmpRoot(char *out, size_t out_len)
{
   const char *raw = getenv("TMPDIR");

   if (!out || out_len == 0)
   {
      return;
   }
   if (!raw || raw[0] == '\0' || strstr(raw, "..") != NULL || strlen(raw) >= out_len ||
       !hypredrv_BinaryPathPrefixIsSafe(raw))
   {
      (void)snprintf(out, out_len, "/tmp");
   }
   else
   {
      (void)snprintf(out, out_len, "%s", raw);
   }
}

static int
LSSeqTempPrefixBuild(MPI_Comm comm, int ls_id, const char *tag, char *prefix,
                     size_t prefix_size)
{
   char        tmp_root_buf[MAX_FILENAME_LENGTH];
   const char *tmp_root = tmp_root_buf;
   char        tmpdir_template[MAX_FILENAME_LENGTH];
   int         written = 0;
   int         myid    = 0;

   if (!prefix || prefix_size == 0) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid temporary prefix output");
      return 0;
   }

   LSSeqSanitizedTmpRoot(tmp_root_buf, sizeof(tmp_root_buf));

   MPI_Comm_rank(comm, &myid);
   /* GCOVR_EXCL_BR_START */
   written = snprintf(tmpdir_template, sizeof(tmpdir_template),
                      /* GCOVR_EXCL_BR_STOP */
                      "%s/hypredrv_lsseq_%s_%d_%d_%d_XXXXXX", tmp_root, tag ? tag : "tmp",
                      (int)getpid(), myid, ls_id);
   if (written < 0 || (size_t)written >= sizeof(tmpdir_template)) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Failed to format LSSeq temporary directory template");
      return 0;
   }

   if (!mkdtemp(tmpdir_template)) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not create LSSeq temporary directory under '%s'",
                           tmp_root);
      return 0;
   }

   written = snprintf(prefix, prefix_size, "%s/part", tmpdir_template);
   if (written < 0 || (size_t)written >= prefix_size) /* GCOVR_EXCL_BR_LINE */
   {
      (void)rmdir(tmpdir_template);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LSSeq temporary prefix exceeds buffer size (%zu bytes)",
                           prefix_size);
      return 0;
   }

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

static int
LSSeqSharedTempPrefixBuild(MPI_Comm comm, int ls_id, const char *tag, char *prefix,
                           size_t prefix_size)
{
   char        tmp_root_buf[MAX_FILENAME_LENGTH];
   const char *tmp_root = tmp_root_buf;
   char        tmpdir_path[MAX_FILENAME_LENGTH];
   int         myid    = 0;
   int         success = 1;
   int         written = 0;

   if (!prefix || prefix_size == 0) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid shared temporary prefix output");
      return 0;
   }

   LSSeqSanitizedTmpRoot(tmp_root_buf, sizeof(tmp_root_buf));

   memset(tmpdir_path, 0, sizeof(tmpdir_path));
   MPI_Comm_rank(comm, &myid);

   if (!myid)
   {
      char tmpdir_template[MAX_FILENAME_LENGTH];

      /* GCOVR_EXCL_BR_START */
      written = snprintf(tmpdir_template, sizeof(tmpdir_template),
                         /* GCOVR_EXCL_BR_STOP */
                         "%s/hypredrv_lsseq_%s_%d_%d_XXXXXX", tmp_root, tag ? tag : "tmp",
                         (int)getpid(), ls_id);
      /* GCOVR_EXCL_BR_START */
      if (written < 0 || (size_t)written >= sizeof(tmpdir_template))
      /* GCOVR_EXCL_BR_STOP */
      {
         success = 0;
      }
      else if (!mkdtemp(tmpdir_template)) /* GCOVR_EXCL_BR_LINE */
      {
         success = 0;
      }
      else
      {
         strncpy(tmpdir_path, tmpdir_template, sizeof(tmpdir_path) - 1);
      }
   }

   MPI_Bcast(&success, 1, MPI_INT, 0, comm);
   if (!success) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not create shared LSSeq temporary directory");
      return 0;
   }

   MPI_Bcast(tmpdir_path, (int)sizeof(tmpdir_path), MPI_CHAR, 0, comm);
   written = snprintf(prefix, prefix_size, "%s/part", tmpdir_path);
   if (written < 0 || (size_t)written >= prefix_size) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Shared LSSeq temporary prefix exceeds buffer size (%zu "
                           "bytes)",
                           prefix_size);
      return 0;
   }

   return 1;
}

static void
LSSeqCleanupPartFiles(const char *prefix, const int *partids, int nparts,
                      const char *suffix)
{
   /* Buffer sized for prefix + ".%05d" + suffix; avoids -Wformat-truncation */
   char filename[MAX_FILENAME_LENGTH + 32];
   if (!prefix || prefix[0] == '\0') /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   for (int i = 0; i < nparts; i++)
   {
      /* GCOVR_EXCL_BR_START */
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[i],
               /* GCOVR_EXCL_BR_STOP */
               suffix ? suffix : "");
      remove(filename);
   }

   {
      char        dirname[MAX_FILENAME_LENGTH];
      const char *slash = strrchr(prefix, '/');
      if (!slash || slash == prefix) /* GCOVR_EXCL_BR_LINE */
      {
         return;
      }
      if ((size_t)(slash - prefix) >= sizeof(dirname)) /* GCOVR_EXCL_BR_LINE */
      {
         return;
      }
      memcpy(dirname, prefix, (size_t)(slash - prefix));
      dirname[(size_t)(slash - prefix)] = '\0';
      (void)rmdir(dirname);
   }
}

static int
LSSeqWriteMatrixPartFile(const char *filename, const LSSeqPartMeta *part,
                         const LSSeqPatternMeta *pattern, const void *rows,
                         const void *cols, const void *vals)
{
   FILE    *fp         = NULL;
   uint64_t header[11] = {0};

   if (!filename || !part || !pattern) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid matrix part-file write arguments");
      return 0;
   }
   if (!hypredrv_BinaryPathPrefixIsSafe(filename))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid matrix temporary part path");
      return 0;
   }

   fp = hypredrv_FopenCreateRestricted(filename, 0, 1);
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not create matrix temporary part '%s'", filename);
      return 0;
   }

   header[1] = part->row_index_size;
   header[2] = part->value_size;
   header[5] = part->row_upper - part->row_lower + 1;
   header[6] = pattern->nnz;
   header[7] = part->row_lower;
   header[8] = part->row_upper;

   if (fwrite(header, sizeof(uint64_t), 11, fp) != 11) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not write matrix header to '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (pattern->nnz > 0) /* GCOVR_EXCL_BR_STOP */
   {
      size_t nnz = (size_t)pattern->nnz;
      /* GCOVR_EXCL_BR_START */
      if ((rows && fwrite(rows, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */
          (cols && fwrite(cols, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */
          (vals && fwrite(vals, (size_t)part->value_size, nnz, fp) != nnz))
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not write matrix data to '%s'", filename);
         return 0;
      }
   }

   fclose(fp);
   return 1;
}

static int
LSSeqWriteRHSPartFile(const char *filename, const LSSeqPartMeta *part, const void *vals)
{
   FILE    *fp        = NULL;
   uint64_t header[8] = {0};

   if (!filename || !part) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid RHS part-file write arguments");
      return 0;
   }
   if (!hypredrv_BinaryPathPrefixIsSafe(filename))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid RHS temporary part path");
      return 0;
   }

   fp = hypredrv_FopenCreateRestricted(filename, 0, 1);
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not create RHS temporary part '%s'", filename);
      return 0;
   }

   header[1] = part->value_size;
   header[5] = part->nrows;

   if (fwrite(header, sizeof(uint64_t), 8, fp) != 8) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not write RHS header to '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (part->nrows > 0) /* GCOVR_EXCL_BR_STOP */
   {
      size_t nrows = (size_t)part->nrows;
      /* GCOVR_EXCL_BR_START */
      if (vals && fwrite(vals, (size_t)part->value_size, nrows, fp) != nrows)
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not write RHS values to '%s'", filename);
         return 0;
      }
   }

   fclose(fp);
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

int
hypredrv_LSSeqReadMatrix(MPI_Comm comm, const char *filename, int ls_id,
                         HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *matrix_ptr)
{
   LSSeqData seq                         = {0};
   FILE     *fp                          = NULL;
   int      *partids                     = NULL;
   uint32_t *part_order                  = NULL;
   int       nparts                      = 0;
   char      prefix[MAX_FILENAME_LENGTH] = {0};
   char      part_filename[MAX_FILENAME_LENGTH];
   int       local_ok = 1;
   int       ok       = 0;

   if (!matrix_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null matrix pointer for LSSeqReadMatrix");
      return 0;
   }
   *matrix_ptr = NULL;

   if (!LSSeqDataLoad(filename, &seq))
   {
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_LINE */
   {
      local_ok = 0;
      goto stage_sync;
   }

   fp = fopen(filename, "rb");
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqTempPrefixBuild(comm, ls_id, "A", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }
   for (int i = 0; i < nparts && local_ok; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqPartMeta       *part        = &seq.parts[part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      void                   *rows = NULL, *cols = NULL, *vals = NULL;
      size_t                  rows_size = 0, cols_size = 0, vals_size = 0;
      const LSSeqPatternMeta *pattern       = NULL;
      size_t                  expected_size = 0;

      if (sys->pattern_id >= seq.header.num_patterns)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid pattern id %u for system %d part %u",
                              sys->pattern_id, ls_id, part_id);
         local_ok = 0;
         break;
      }
      pattern = &seq.patterns[sys->pattern_id];
      if (pattern->part_id != part_id)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd(
            "Pattern-part mismatch for system %d part %u (pattern part=%u)", ls_id,
            part_id, pattern->part_id);
         local_ok = 0;
         break;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqCheckedMulSize((size_t)pattern->nnz, (size_t)part->row_index_size,
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */
                               &expected_size, "matrix index blob size") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES,
                                  "matrix index blob"))
      {
         local_ok = 0;
         break;
      }
      if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->rows_blob_offset,
                         /* GCOVR_EXCL_BR_START */
                         pattern->rows_blob_size, expected_size, &rows, &rows_size) ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->cols_blob_offset,
                         pattern->cols_blob_size, expected_size, &cols, &cols_size))
      {
         free(rows);
         free(cols);
         local_ok = 0;
         break;
      }

      if (!LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq.header.codec,
                                  seq.header.offset_blob_data, seq.part_blob_table,
                                  part_id, 0, sys->values_blob_offset,
                                  sys->values_blob_size, &vals, &vals_size))
      {
         free(rows);
         free(cols);
         free(vals);
         local_ok = 0;
         break;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, ".bin"))
      {
         free(rows);
         free(cols);
         free(vals);
         local_ok = 0;
         break;
      }
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqWriteMatrixPartFile(part_filename, part, pattern, rows, cols, vals))
      /* GCOVR_EXCL_BR_STOP */
      {
         free(rows);
         free(cols);
         free(vals);
         local_ok = 0;
         break;
      }

      free(rows);
      free(cols);
      free(vals);
   }

stage_sync:
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq matrix local staging failed"))
   {
      goto cleanup;
   }

   hypredrv_IJMatrixReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                                        memory_location, matrix_ptr);
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
   LSSeqCleanupPartFiles(prefix, partids, nparts, ".bin");
   free(part_order);
   free(partids);
   LSSeqDataDestroy(&seq);
   return ok;
}

int
hypredrv_LSSeqReadRHS(MPI_Comm comm, const char *filename, int ls_id,
                      HYPRE_MemoryLocation memory_location, HYPRE_IJVector *rhs_ptr)
{
   LSSeqData seq                         = {0};
   FILE     *fp                          = NULL;
   int      *partids                     = NULL;
   uint32_t *part_order                  = NULL;
   int       nparts                      = 0;
   char      prefix[MAX_FILENAME_LENGTH] = {0};
   char      part_filename[MAX_FILENAME_LENGTH];
   int       local_ok = 1;
   int       ok       = 0;

   if (!rhs_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null RHS pointer for LSSeqReadRHS");
      return 0;
   }
   *rhs_ptr = NULL;

   if (!LSSeqDataLoad(filename, &seq))
   {
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_LINE */
   {
      local_ok = 0;
      goto stage_sync;
   }

   fp = fopen(filename, "rb");
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      local_ok = 0;
      goto stage_sync;
   }

   /* GCOVR_EXCL_BR_START */
   if (!LSSeqTempPrefixBuild(comm, ls_id, "b", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }
   for (int i = 0; i < nparts && local_ok; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqPartMeta       *part        = &seq.parts[part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      void  *vals      = NULL;
      size_t vals_size = 0;

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq.header.codec,
                                  /* GCOVR_EXCL_BR_STOP */
                                  seq.header.offset_blob_data, seq.part_blob_table,
                                  part_id, 1, sys->rhs_blob_offset, sys->rhs_blob_size,
                                  &vals, &vals_size))
      {
         free(vals);
         local_ok = 0;
         break;
      }

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, ".bin"))
      {
         free(vals);
         local_ok = 0;
         break;
      }
      /* GCOVR_EXCL_BR_START */
      if (!LSSeqWriteRHSPartFile(part_filename, part, vals)) /* GCOVR_EXCL_BR_STOP */
      {
         free(vals);
         local_ok = 0;
         break;
      }
      free(vals);
   }

stage_sync:
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq RHS local staging failed"))
   {
      goto cleanup;
   }

   hypredrv_IJVectorReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                                        memory_location, rhs_ptr);
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
   LSSeqCleanupPartFiles(prefix, partids, nparts, ".bin");
   free(part_order);
   free(partids);
   LSSeqDataDestroy(&seq);
   return ok;
}

int
hypredrv_LSSeqReadDofmap(MPI_Comm comm, const char *filename, int ls_id,
                         IntArray **dofmap_ptr)
{
   LSSeqData seq                         = {0};
   FILE     *fp                          = NULL;
   int      *partids                     = NULL;
   uint32_t *part_order                  = NULL;
   int       nparts                      = 0;
   char      prefix[MAX_FILENAME_LENGTH] = {0};
   char      part_filename[MAX_FILENAME_LENGTH];
   int       myid     = 0;
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
   if (!LSSeqDataLoad(filename, &seq)) /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }

   if (!(seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP))
   {
      *dofmap_ptr = hypredrv_IntArrayCreate(0);
      LSSeqDataDestroy(&seq);
      return (*dofmap_ptr != NULL);
   }

   /* GCOVR_EXCL_BR_START */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      local_ok = 0;
      goto stage_sync;
   }

   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_LINE */
   {
      local_ok = 0;
      goto stage_sync;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_LINE */
   {
      local_ok = 0;
      goto stage_sync;
   }

   fp = fopen(filename, "rb");
   if (!fp) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      local_ok = 0;
      goto stage_sync;
   }

   prefix[0] = '\0';
   MPI_Comm_rank(comm, &myid);
   /* GCOVR_EXCL_BR_START */
   if (!LSSeqSharedTempPrefixBuild(comm, ls_id, "dof", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      local_ok = 0;
      goto stage_sync;
   }

   for (int i = 0; i < nparts && local_ok; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      int32_t *dof_data = NULL;
      size_t   dof_size = 0;
      FILE    *out      = NULL;

      /* GCOVR_EXCL_BR_START */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, NULL))
      {
         local_ok = 0;
         break;
      }
      out = hypredrv_FopenCreateRestricted(part_filename, 0, 0);
      if (!out) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Could not create dofmap temporary part '%s'",
                              part_filename);
         local_ok = 0;
         break;
      }

      /* GCOVR_EXCL_BR_START */
      if (sys->dof_num_entries > 0) /* GCOVR_EXCL_BR_STOP */
      {
         size_t expected_size = 0;
         /* GCOVR_EXCL_BR_START */
         if (!LSSeqCheckedMulSize((size_t)sys->dof_num_entries, sizeof(int32_t),
                                  /* GCOVR_EXCL_BR_STOP */
                                  /* GCOVR_EXCL_BR_START */
                                  &expected_size, "dof payload size") ||
             /* GCOVR_EXCL_BR_STOP */
             !LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES, "dof payload"))
         {
            fclose(out);
            local_ok = 0;
            break;
         }
         /* GCOVR_EXCL_BR_START */
         if (!LSSeqReadPartBlobSlice(
                /* GCOVR_EXCL_BR_STOP */
                fp, (comp_alg_t)seq.header.codec, seq.header.offset_blob_data,
                seq.part_blob_table, part_id, 2, sys->dof_blob_offset,
                (uint64_t)expected_size, (void **)&dof_data, &dof_size))
         {
            fclose(out);
            free(dof_data);
            local_ok = 0;
            break;
         }
      }

      fprintf(out, "%llu\n", (unsigned long long)sys->dof_num_entries);
      for (uint64_t j = 0; j < sys->dof_num_entries; j++)
      {
         /* GCOVR_EXCL_BR_START */
         int value = dof_data ? (int)dof_data[j] : 0;
         /* GCOVR_EXCL_BR_STOP */
         fprintf(out, "%d\n", value);
      }
      fclose(out);
      free(dof_data);
   }

stage_sync:
   if (!LSSeqSynchronizeMPIStatus(comm, local_ok, ERROR_FILE_UNEXPECTED_ENTRY,
                                  "LSSeq dofmap local staging failed"))
   {
      goto cleanup;
   }

   /* Ensure all rank-local dof files are visible before parallel read. */
   MPI_Barrier(comm);
   hypredrv_IntArrayParRead(comm, prefix, dofmap_ptr);
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
   if (ok) /* GCOVR_EXCL_BR_STOP */
   {
      MPI_Barrier(comm);
   }
   /* GCOVR_EXCL_BR_START */
   if (fp) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
   }
   /* GCOVR_EXCL_BR_START */
   if (prefix[0] != '\0') /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqCleanupPartFiles(prefix, partids, nparts, "");
   }
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
