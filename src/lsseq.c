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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!result)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Null output while checking LSSeq size multiplication"); /* GCOVR_EXCL_LINE */
      return 0;                                                   /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */       /* low-signal branch under CI */
   if (a != 0 && b > SIZE_MAX / a) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("LSSeq size overflow while computing %s (%zu * %zu)",
                           /* GCOVR_EXCL_BR_STOP */
                           what ? what : "allocation size", a, b);
      return 0; /* GCOVR_EXCL_LINE */
   }

   *result = a * b;
   return 1;
}

static int
LSSeqCheckedAddU64(uint64_t a, uint64_t b, uint64_t *result, const char *what)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!result)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Null output while checking LSSeq offset addition"); /* GCOVR_EXCL_LINE */
      return 0;                                               /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (UINT64_MAX - a < b)   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("LSSeq offset overflow while computing %s",
                           /* GCOVR_EXCL_BR_STOP */
                           what ? what : "offset");
      return 0; /* GCOVR_EXCL_LINE */
   }

   *result = a + b;
   return 1;
}

static int
LSSeqValidateByteLimit(size_t nbytes, size_t max_nbytes, const char *what)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (nbytes > max_nbytes)  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("LSSeq %s exceeds limit (%zu > %zu bytes)",
                           /* GCOVR_EXCL_BR_STOP */
                           what ? what : "allocation", nbytes, max_nbytes);
      return 0; /* GCOVR_EXCL_LINE */
   }

   return 1;
}

static int
LSSeqBuildPartOrder(const LSSeqData *seq, uint32_t **order_ptr)
{
   uint32_t *order = NULL;
   uint32_t  n     = 0;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!seq || !order_ptr)   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Invalid arguments for LSSeqBuildPartOrder"); /* GCOVR_EXCL_LINE */
      return 0;                                        /* GCOVR_EXCL_LINE */
   }
   *order_ptr = NULL;
   n          = seq->header.num_parts;
   order      = (uint32_t *)malloc((size_t)n * sizeof(*order));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!order)               /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq part order (%u entries)",
                           n); /* GCOVR_EXCL_LINE */
      return 0;                /* GCOVR_EXCL_LINE */
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
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!bytes)               /* GCOVR_EXCL_BR_STOP */
   {
      return hash; /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (!filename || filename_size == 0 || !prefix) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Invalid arguments for LSSeqFormatPartFilename"); /* GCOVR_EXCL_LINE */
      return 0;                                            /* GCOVR_EXCL_LINE */
   }
   if (!suffix)
   {
      suffix = "";
   }

   id_len = snprintf(id_buf, sizeof(id_buf), "%05u", part_id);
   /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
   if (id_len < 0 || (size_t)id_len >= sizeof(id_buf)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not format LSSeq part id %u",
                           part_id); /* GCOVR_EXCL_LINE */
      return 0;                      /* GCOVR_EXCL_LINE */
   }

   prefix_len = strlen(prefix);
   suffix_len = strlen(suffix);
   total_len  = prefix_len + 1u + (size_t)id_len + suffix_len;
   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   if (total_len + 1u > filename_size) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(                     /* GCOVR_EXCL_LINE */
                           "LSSeq temporary part filename is too long for buffer (%zu "
                           "bytes)",
                           filename_size);
      return 0; /* GCOVR_EXCL_LINE */
   }

   memcpy(filename, prefix, prefix_len);
   filename[prefix_len] = '.';
   memcpy(filename + prefix_len + 1u, id_buf, (size_t)id_len);
   memcpy(filename + prefix_len + 1u + (size_t)id_len, suffix, suffix_len);
   filename[total_len] = '\0';

   return 1;
}

static int
LSSeqReadAt(FILE *fp, uint64_t offset, void *buffer, size_t nbytes, const char *what)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp || !buffer)       /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid arguments while reading %s", /* GCOVR_EXCL_LINE */
                           what ? what : "lsseq data");
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */                     /* low-signal branch under CI */
   if (fseeko(fp, (off_t)offset, SEEK_SET) != 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("Could not seek to offset %llu while reading %s",
                           /* GCOVR_EXCL_BR_STOP */
                           (unsigned long long)offset, what ? what : "lsseq data");
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!header)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Null lsseq header"); /* GCOVR_EXCL_LINE */
      return 0;                                  /* GCOVR_EXCL_LINE */
   }

   if (header->magic != LSSEQ_MAGIC)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("Invalid LSSeq magic for file '%s'", filename ? filename : "");
      /* GCOVR_EXCL_BR_STOP */
      return 0;
   }
   if (header->version != LSSEQ_VERSION)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("Unsupported LSSeq version %u in '%s'", header->version,
                           /* GCOVR_EXCL_BR_STOP */
                           filename ? filename : "");
      return 0;
   }
   if (header->offset_part_blob_table == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd(
         /* GCOVR_EXCL_BR_STOP */
         "LSSeq format requires part blob table (offset_part_blob_table) in '%s'",
         filename ? filename : "");
      return 0;
   }
   if (header->num_systems == 0 || header->num_parts == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid LSSeq shape in '%s': num_systems=%u, num_parts=%u",
                           filename ? filename : "", header->num_systems,
                           /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                           header->num_parts);
      /* GCOVR_EXCL_BR_STOP */
      return 0;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (header->num_parts > LSSEQ_MAX_PARTS || header->num_systems > LSSEQ_MAX_SYSTEMS ||
       /* GCOVR_EXCL_BR_STOP */
       header->num_patterns > LSSEQ_MAX_PATTERNS ||
       header->num_timesteps > LSSEQ_MAX_TIMESTEPS)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("LSSeq dimensions exceed limits in '%s' (systems=%u parts=%u "
                           "patterns=%u timesteps=%u)",
                           filename ? filename : "", header->num_systems,
                           header->num_parts, header->num_patterns,
                           /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                           header->num_timesteps);
      /* GCOVR_EXCL_BR_STOP */
      return 0;
   }
   if (header->codec > (uint32_t)COMP_BLOSC)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      hypredrv_ErrorMsgAdd("Invalid LSSeq compression codec id %u in '%s'", header->codec,
                           /* GCOVR_EXCL_BR_STOP */
                           filename ? filename : "");
      return 0;
   }

   return 1;
}

static void
LSSeqDataDestroy(LSSeqData *seq)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!seq)                 /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!filename || !seq)    /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                   /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid arguments to LSSeqDataLoad"); /* GCOVR_EXCL_LINE */
      return 0;                                                   /* GCOVR_EXCL_LINE */
   }

   memset(seq, 0, sizeof(*seq));
   fp = fopen(filename, "rb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqReadAt(fp, 0, &seq->header, sizeof(seq->header), "lsseq header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp); /* GCOVR_EXCL_LINE */
      return 0;   /* GCOVR_EXCL_LINE */
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

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqReadAt(fp, info_offset, &seq->info_header, sizeof(seq->info_header),
                       /* GCOVR_EXCL_BR_STOP */
                       "info header"))
      {
         fclose(fp); /* GCOVR_EXCL_LINE */
         return 0;   /* GCOVR_EXCL_LINE */
      }

      if (seq->info_header.magic != LSSEQ_INFO_MAGIC ||
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (seq->info_header.endian_tag != UINT32_C(0x01020304)) /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                                         /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(                               /* GCOVR_EXCL_LINE */
                              "Unsupported LSSeq info endianness tag in '%s' "
                              "(tag=0x%08x)",
                              filename,
                              (unsigned int)
                                 seq->info_header.endian_tag); /* GCOVR_EXCL_LINE */
         return 0;                                             /* GCOVR_EXCL_LINE */
      }

      /* Bound payload size to avoid accidental huge allocations. */
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (seq->info_header.payload_size > (uint64_t)LSSEQ_INFO_PAYLOAD_MAX_BYTES ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
             seq->info_header.payload_size > (uint64_t)SIZE_MAX - 1u)
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                                         /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "LSSeq info payload too large in '%s' (%llu bytes)", /* GCOVR_EXCL_LINE */
            filename,
            (unsigned long long)seq->info_header.payload_size); /* GCOVR_EXCL_LINE */
         return 0;                                              /* GCOVR_EXCL_LINE */
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedAddU64(expected_min_part_offset,
                              /* GCOVR_EXCL_BR_STOP */
                              (uint64_t)seq->info_header.payload_size,
                              &expected_payload_end, "info payload end"))
      {
         fclose(fp); /* GCOVR_EXCL_LINE */
         return 0;   /* GCOVR_EXCL_LINE */
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

      /* GCOVR_EXCL_BR_START */              /* low-signal branch under CI */
      if (seq->info_header.payload_size > 0) /* GCOVR_EXCL_BR_STOP */
      {
         uint64_t hash          = UINT64_C(1469598103934665603);
         seq->info_payload_size = (size_t)seq->info_header.payload_size;
         seq->info_payload      = (char *)malloc(seq->info_payload_size + 1u);
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!seq->info_payload)   /* GCOVR_EXCL_BR_STOP */
         {
            fclose(fp);                              /* GCOVR_EXCL_LINE */
            hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
            hypredrv_ErrorMsgAdd(
               "Failed to allocate LSSeq info payload (%zu bytes)", /* GCOVR_EXCL_LINE */
               seq->info_payload_size);
            return 0; /* GCOVR_EXCL_LINE */
         }

         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!LSSeqReadAt(fp, expected_min_part_offset, seq->info_payload,
                          /* GCOVR_EXCL_BR_STOP */
                          seq->info_payload_size, "info payload"))
         {
            fclose(fp);            /* GCOVR_EXCL_LINE */
            LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
            return 0;              /* GCOVR_EXCL_LINE */
         }
         seq->info_payload[seq->info_payload_size] = '\0';

         hash = LSSeqFNV1a64(seq->info_payload, seq->info_payload_size, hash);
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (hash != seq->info_header.payload_hash_fnv1a64) /* GCOVR_EXCL_BR_STOP */
         {
            fclose(fp);                                         /* GCOVR_EXCL_LINE */
            hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
            hypredrv_ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'",
                                 filename); /* GCOVR_EXCL_LINE */
            LSSeqDataDestroy(seq);          /* GCOVR_EXCL_LINE */
            return 0;                       /* GCOVR_EXCL_LINE */
         }
      }
   }

   {
      size_t part_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_parts, sizeof(LSSeqPartMeta),
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               &part_meta_bytes, "part metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(part_meta_bytes, LSSEQ_MAX_META_BYTES, "part metadata"))
      {
         fclose(fp); /* GCOVR_EXCL_LINE */
         return 0;   /* GCOVR_EXCL_LINE */
      }
   }
   seq->parts =
      (LSSeqPartMeta *)calloc((size_t)seq->header.num_parts, sizeof(LSSeqPartMeta));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!seq->parts)          /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);                              /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate LSSeq part metadata"); /* GCOVR_EXCL_LINE */
      return 0;                                     /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */         /* low-signal branch under CI */
   if (seq->header.num_patterns > 0) /* GCOVR_EXCL_BR_STOP */
   {
      size_t pattern_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_patterns, sizeof(LSSeqPatternMeta),
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               &pattern_meta_bytes, "pattern metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(pattern_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "pattern metadata"))
      {
         fclose(fp);            /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
         return 0;              /* GCOVR_EXCL_LINE */
      }
      seq->patterns =
         (LSSeqPatternMeta *)calloc(seq->header.num_patterns, sizeof(LSSeqPatternMeta));
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!seq->patterns)       /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                              /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate LSSeq pattern metadata"); /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq);                           /* GCOVR_EXCL_LINE */
         return 0;                                        /* GCOVR_EXCL_LINE */
      }
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqCheckedMulSize((size_t)seq->header.num_systems,
                            /* GCOVR_EXCL_BR_STOP */
                            (size_t)seq->header.num_parts, &n_sys_parts,
                            "system-part count"))
   {
      fclose(fp);            /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
      return 0;              /* GCOVR_EXCL_LINE */
   }
   {
      size_t sys_part_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize(n_sys_parts, sizeof(LSSeqSystemPartMeta),
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               &sys_part_meta_bytes, "system-part metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(sys_part_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "system-part metadata"))
      {
         fclose(fp);            /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
         return 0;              /* GCOVR_EXCL_LINE */
      }
   }
   seq->sys_parts =
      (LSSeqSystemPartMeta *)calloc(n_sys_parts, sizeof(LSSeqSystemPartMeta));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!seq->sys_parts)      /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);                              /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate LSSeq system-part metadata"); /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(seq);                               /* GCOVR_EXCL_LINE */
      return 0;                                            /* GCOVR_EXCL_LINE */
   }

   if ((seq->header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq->header.num_timesteps > 0)
   {
      size_t timestep_meta_bytes = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize((size_t)seq->header.num_timesteps,
                               /* GCOVR_EXCL_BR_STOP */
                               sizeof(LSSeqTimestepEntry), &timestep_meta_bytes,
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               "timestep metadata bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(timestep_meta_bytes, LSSEQ_MAX_META_BYTES,
                                  "timestep metadata"))
      {
         fclose(fp);            /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
         return 0;              /* GCOVR_EXCL_LINE */
      }
      seq->timesteps = (LSSeqTimestepEntry *)calloc(seq->header.num_timesteps,
                                                    sizeof(LSSeqTimestepEntry));
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!seq->timesteps)      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                              /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate LSSeq timesteps metadata"); /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq);                             /* GCOVR_EXCL_LINE */
         return 0;                                          /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqReadAt(fp, seq->header.offset_sys_part_meta, seq->sys_parts,
                    /* GCOVR_EXCL_BR_STOP */
                    n_sys_parts * sizeof(LSSeqSystemPartMeta), "system-part metadata"))
   {
      fclose(fp);            /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
      return 0;              /* GCOVR_EXCL_LINE */
   }

   {
      size_t pt_entries = 0;
      size_t pt_size    = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize((size_t)LSSEQ_PART_BLOB_ENTRIES,
                               /* GCOVR_EXCL_BR_STOP */
                               (size_t)seq->header.num_parts, &pt_entries,
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               "part blob table entries") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqCheckedMulSize(pt_entries, sizeof(uint64_t), &pt_size,
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               "part blob table bytes") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(pt_size, LSSEQ_MAX_META_BYTES, "part blob table"))
      {
         fclose(fp);            /* GCOVR_EXCL_LINE */
         LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
         return 0;
      }
      seq->part_blob_table = (uint64_t *)malloc(pt_size);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (seq->timesteps &&
       /* GCOVR_EXCL_BR_STOP */
       !LSSeqReadAt(fp, seq->header.offset_timestep_meta, seq->timesteps,
                    (size_t)seq->header.num_timesteps * sizeof(LSSeqTimestepEntry),
                    "timestep metadata"))
   {
      fclose(fp);            /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(seq); /* GCOVR_EXCL_LINE */
      return 0;              /* GCOVR_EXCL_LINE */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqReadAt(fp, 0, &header, sizeof(header), "lsseq header"))
   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp); /* GCOVR_EXCL_LINE */
      return 0;   /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (!LSSeqValidateHeader(&header, filename)) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp); /* GCOVR_EXCL_LINE */
      return 0;   /* GCOVR_EXCL_LINE */
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
       /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!payload)             /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);                              /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate LSSeq info payload (%zu bytes)",
                           nbytes); /* GCOVR_EXCL_LINE */
      return 0;                     /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (nbytes > 0 && !LSSeqReadAt(fp, info_offset + (uint64_t)sizeof(info), payload,
                                  /* GCOVR_EXCL_BR_STOP */
                                  nbytes, "info payload"))
   {
      free(payload); /* GCOVR_EXCL_LINE */
      fclose(fp);    /* GCOVR_EXCL_LINE */
      return 0;      /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (!partids_ptr || !nparts_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid LSSeqLocalPartIDs outputs"); /* GCOVR_EXCL_LINE */
      return 0;                                                  /* GCOVR_EXCL_LINE */
   }

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = (int)(g_nparts / (uint32_t)nprocs);
   rem    = (int)(g_nparts % (uint32_t)nprocs);
   nparts += (myid < rem) ? 1 : 0;
   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (g_nparts < (uint32_t)nprocs) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Invalid number of sequence parts (%u) for communicator size %d", g_nparts,
         nprocs);
      return 0;
   }

   partids = (int *)calloc((size_t)nparts, sizeof(int));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!partids)             /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate partids array"); /* GCOVR_EXCL_LINE */
      return 0;                                                 /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   if (!fp || !output || !output_size) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid LSSeq blob read arguments"); /* GCOVR_EXCL_LINE */
      return 0;                                                  /* GCOVR_EXCL_LINE */
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
   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (blob_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);          /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Blob size exceeds limit (%llu bytes)", /* GCOVR_EXCL_LINE */
                           (unsigned long long)blob_size);
      return 0; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                 /* low-signal branch under CI */
   if (expected_size > LSSEQ_MAX_BLOB_BYTES) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Expected decoded blob size exceeds limit (%zu bytes)", /* GCOVR_EXCL_LINE */
         expected_size);
      return 0; /* GCOVR_EXCL_LINE */
   }

   if (blob_size == 0)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (expected_size != 0)   /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!blob_data)           /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate %llu bytes for blob read", /* GCOVR_EXCL_LINE */
         (unsigned long long)blob_size);
      return 0; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqReadAt(fp, offset, blob_data, (size_t)blob_size, "blob payload"))
   /* GCOVR_EXCL_BR_STOP */
   {
      free(blob_data); /* GCOVR_EXCL_LINE */
      return 0;        /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (codec == COMP_NONE)   /* GCOVR_EXCL_BR_STOP */
   {
      decoded = malloc((size_t)blob_size);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!decoded)             /* GCOVR_EXCL_BR_STOP */
      {
         free(blob_data);                         /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate %llu bytes for blob decode", /* GCOVR_EXCL_LINE */
            (unsigned long long)blob_size);
         return 0; /* GCOVR_EXCL_LINE */
      }
      memcpy(decoded, blob_data, (size_t)blob_size);
      decoded_size = (size_t)blob_size;
   }
   else
   {
      hypredrv_decompress(codec, (size_t)blob_size, blob_data, &decoded_size, &decoded);
      /* GCOVR_EXCL_BR_START */                   /* low-signal branch under CI */
      if (hypredrv_ErrorCodeActive() || !decoded) /* GCOVR_EXCL_BR_STOP */
      {
         free(blob_data); /* GCOVR_EXCL_LINE */
         return 0;        /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp || !part_blob_table || !output || !output_size || slot < 0 || slot > 2)
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   *output      = NULL;
   *output_size = 0;
   c_off =
      part_blob_table[((size_t)part_id * LSSEQ_PART_BLOB_ENTRIES) + (size_t)(slot * 2)];
   c_size = part_blob_table[((size_t)part_id * LSSEQ_PART_BLOB_ENTRIES) +
                            (size_t)(slot * 2) + 1];
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (decomp_size > (uint64_t)SIZE_MAX || decomp_size > (uint64_t)LSSEQ_MAX_BLOB_BYTES)
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Requested decoded slice exceeds limit (%llu bytes)", /* GCOVR_EXCL_LINE */
         (unsigned long long)decomp_size);
      return 0; /* GCOVR_EXCL_LINE */
   }
   if (c_size == 0)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (decomp_size != 0)     /* GCOVR_EXCL_BR_STOP */
      {
         return 0;
      }
      return 1;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqReadBlob(fp, codec, blob_base + c_off, c_size, 0, &decoded, &decoded_size))
   /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (decomp_offset > UINT64_MAX - decomp_size ||
       /* GCOVR_EXCL_BR_STOP */
       decomp_offset + decomp_size > (uint64_t)decoded_size)
   {
      free(decoded);
      return 0;
   }
   slice = malloc((size_t)decomp_size);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!slice)               /* GCOVR_EXCL_BR_STOP */
   {
      free(decoded);                           /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate slice buffer (%llu bytes)", /* GCOVR_EXCL_LINE */
         (unsigned long long)decomp_size);
      return 0; /* GCOVR_EXCL_LINE */
   }
   memcpy(slice, (const char *)decoded + (size_t)decomp_offset, (size_t)decomp_size);
   free(decoded);
   *output      = slice;
   *output_size = (size_t)decomp_size;
   return 1;
}

static int
LSSeqTempPrefixBuild(MPI_Comm comm, int ls_id, const char *tag, char *prefix,
                     size_t prefix_size)
{
   const char *tmp_root = getenv("TMPDIR");
   char        tmpdir_template[MAX_FILENAME_LENGTH];
   int         written = 0;
   int         myid    = 0;

   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (!prefix || prefix_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid temporary prefix output"); /* GCOVR_EXCL_LINE */
      return 0;                                                /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */             /* low-signal branch under CI */
   if (!tmp_root || tmp_root[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      tmp_root = "/tmp";
   }

   MPI_Comm_rank(comm, &myid);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   written = snprintf(tmpdir_template, sizeof(tmpdir_template),
                      /* GCOVR_EXCL_BR_STOP */
                      "%s/hypredrv_lsseq_%s_%d_%d_%d_XXXXXX", tmp_root, tag ? tag : "tmp",
                      (int)getpid(), myid, ls_id);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (written < 0 || (size_t)written >= sizeof(tmpdir_template)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to format LSSeq temporary directory template"); /* GCOVR_EXCL_LINE */
      return 0;                                                  /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
   if (!mkdtemp(tmpdir_template)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Could not create LSSeq temporary directory under '%s'", /* GCOVR_EXCL_LINE */
         tmp_root);
      return 0; /* GCOVR_EXCL_LINE */
   }

   written = snprintf(prefix, prefix_size, "%s/part", tmpdir_template);
   /* GCOVR_EXCL_BR_START */                          /* low-signal branch under CI */
   if (written < 0 || (size_t)written >= prefix_size) /* GCOVR_EXCL_BR_STOP */
   {
      (void)rmdir(tmpdir_template);             /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "LSSeq temporary prefix exceeds buffer size (%zu bytes)", /* GCOVR_EXCL_LINE */
         prefix_size);
      return 0; /* GCOVR_EXCL_LINE */
   }

   return 1;
}

static int
LSSeqSharedTempPrefixBuild(MPI_Comm comm, int ls_id, const char *tag, char *prefix,
                           size_t prefix_size)
{
   const char *tmp_root = getenv("TMPDIR");
   char        tmpdir_path[MAX_FILENAME_LENGTH];
   int         myid    = 0;
   int         success = 1;
   int         written = 0;

   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (!prefix || prefix_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Invalid shared temporary prefix output"); /* GCOVR_EXCL_LINE */
      return 0;                                     /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */             /* low-signal branch under CI */
   if (!tmp_root || tmp_root[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      tmp_root = "/tmp";
   }

   memset(tmpdir_path, 0, sizeof(tmpdir_path));
   MPI_Comm_rank(comm, &myid);

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!myid)                /* GCOVR_EXCL_BR_STOP */
   {
      char tmpdir_template[MAX_FILENAME_LENGTH];

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      written = snprintf(tmpdir_template, sizeof(tmpdir_template),
                         /* GCOVR_EXCL_BR_STOP */
                         "%s/hypredrv_lsseq_%s_%d_%d_XXXXXX", tmp_root, tag ? tag : "tmp",
                         (int)getpid(), ls_id);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (written < 0 || (size_t)written >= sizeof(tmpdir_template))
      /* GCOVR_EXCL_BR_STOP */
      {
         success = 0; /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
      else if (!mkdtemp(tmpdir_template)) /* GCOVR_EXCL_BR_STOP */
      {
         success = 0; /* GCOVR_EXCL_LINE */
      }
      else
      {
         strncpy(tmpdir_path, tmpdir_template, sizeof(tmpdir_path) - 1);
      }
   }

   MPI_Bcast(&success, 1, MPI_INT, 0, comm);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!success)             /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Could not create shared LSSeq temporary directory"); /* GCOVR_EXCL_LINE */
      return 0;                                                /* GCOVR_EXCL_LINE */
   }

   MPI_Bcast(tmpdir_path, (int)sizeof(tmpdir_path), MPI_CHAR, 0, comm);
   written = snprintf(prefix, prefix_size, "%s/part", tmpdir_path);
   /* GCOVR_EXCL_BR_START */                          /* low-signal branch under CI */
   if (written < 0 || (size_t)written >= prefix_size) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(                     /* GCOVR_EXCL_LINE */
                           "Shared LSSeq temporary prefix exceeds buffer size (%zu "
                           "bytes)",
                           prefix_size);
      return 0; /* GCOVR_EXCL_LINE */
   }

   return 1;
}

static void
LSSeqCleanupPartFiles(const char *prefix, const int *partids, int nparts,
                      const char *suffix)
{
   /* Buffer sized for prefix + ".%05d" + suffix; avoids -Wformat-truncation */
   char filename[MAX_FILENAME_LENGTH + 32];
   /* GCOVR_EXCL_BR_START */         /* low-signal branch under CI */
   if (!prefix || prefix[0] == '\0') /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   for (int i = 0; i < nparts; i++)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[i],
               /* GCOVR_EXCL_BR_STOP */
               suffix ? suffix : "");
      remove(filename);
   }

   {
      char        dirname[MAX_FILENAME_LENGTH];
      const char *slash = strrchr(prefix, '/');
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      if (!slash || slash == prefix) /* GCOVR_EXCL_BR_STOP */
      {
         return; /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */                        /* low-signal branch under CI */
      if ((size_t)(slash - prefix) >= sizeof(dirname)) /* GCOVR_EXCL_BR_STOP */
      {
         return; /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   if (!filename || !part || !pattern) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Invalid matrix part-file write arguments"); /* GCOVR_EXCL_LINE */
      return 0;                                       /* GCOVR_EXCL_LINE */
   }

   fp = fopen(filename, "wb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not create matrix temporary part '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   header[1] = part->row_index_size;
   header[2] = part->value_size;
   header[5] = part->row_upper - part->row_lower + 1;
   header[6] = pattern->nnz;
   header[7] = part->row_lower;
   header[8] = part->row_upper;

   /* GCOVR_EXCL_BR_START */                           /* low-signal branch under CI */
   if (fwrite(header, sizeof(uint64_t), 11, fp) != 11) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);                                         /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not write matrix header to '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (pattern->nnz > 0)     /* GCOVR_EXCL_BR_STOP */
   {
      size_t nnz = (size_t)pattern->nnz;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if ((rows && fwrite(rows, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
          (cols && fwrite(cols, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
          (vals && fwrite(vals, (size_t)part->value_size, nnz, fp) != nnz))
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                                         /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Could not write matrix data to '%s'",
                              filename); /* GCOVR_EXCL_LINE */
         return 0;                       /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!filename || !part)   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid RHS part-file write arguments"); /* GCOVR_EXCL_LINE */
      return 0;                                                      /* GCOVR_EXCL_LINE */
   }

   fp = fopen(filename, "wb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not create RHS temporary part '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   header[1] = part->value_size;
   header[5] = part->nrows;

   /* GCOVR_EXCL_BR_START */                         /* low-signal branch under CI */
   if (fwrite(header, sizeof(uint64_t), 8, fp) != 8) /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);                                         /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not write RHS header to '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (part->nrows > 0)      /* GCOVR_EXCL_BR_STOP */
   {
      size_t nrows = (size_t)part->nrows;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (vals && fwrite(vals, (size_t)part->value_size, nrows, fp) != nrows)
      /* GCOVR_EXCL_BR_STOP */
      {
         fclose(fp);                                         /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Could not write RHS values to '%s'",
                              filename); /* GCOVR_EXCL_LINE */
         return 0;                       /* GCOVR_EXCL_LINE */
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

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (num_systems)          /* GCOVR_EXCL_BR_STOP */
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
   LSSeqData seq;
   FILE     *fp                          = NULL;
   int      *partids                     = NULL;
   uint32_t *part_order                  = NULL;
   int       nparts                      = 0;
   char      prefix[MAX_FILENAME_LENGTH] = {0};
   char      part_filename[MAX_FILENAME_LENGTH];
   int       ok = 0;

   if (!matrix_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null matrix pointer for LSSeqReadMatrix");
      return 0;
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */                              /* low-signal branch under CI */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);          /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq); /* GCOVR_EXCL_LINE */
      return 0;               /* GCOVR_EXCL_LINE */
   }

   fp = fopen(filename, "rb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);                               /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqTempPrefixBuild(comm, ls_id, "A", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }
   for (int i = 0; i < nparts; i++)
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
         goto cleanup;
      }
      pattern = &seq.patterns[sys->pattern_id];
      if (pattern->part_id != part_id)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd(
            "Pattern-part mismatch for system %d part %u (pattern part=%u)", ls_id,
            part_id, pattern->part_id);
         goto cleanup;
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqCheckedMulSize((size_t)pattern->nnz, (size_t)part->row_index_size,
                               /* GCOVR_EXCL_BR_STOP */
                               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                               &expected_size, "matrix index blob size") ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES,
                                  "matrix index blob"))
      {
         goto cleanup; /* GCOVR_EXCL_LINE */
      }
      if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->rows_blob_offset,
                         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                         pattern->rows_blob_size,
                         expected_size, &rows, &rows_size) ||
          /* GCOVR_EXCL_BR_STOP */
          !LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->cols_blob_offset,
                         pattern->cols_blob_size, expected_size, &cols, &cols_size))
      {
         free(rows);
         free(cols);
         goto cleanup;
      }

      if (!LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq.header.codec,
                                  seq.header.offset_blob_data, seq.part_blob_table,
                                  part_id, 0, sys->values_blob_offset,
                                  sys->values_blob_size, &vals, &vals_size))
      {
         free(rows);
         free(cols);
         free(vals);
         goto cleanup;
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, ".bin"))
      {
         free(rows);   /* GCOVR_EXCL_LINE */
         free(cols);   /* GCOVR_EXCL_LINE */
         free(vals);   /* GCOVR_EXCL_LINE */
         goto cleanup; /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqWriteMatrixPartFile(part_filename, part, pattern, rows, cols, vals))
      /* GCOVR_EXCL_BR_STOP */
      {
         free(rows);   /* GCOVR_EXCL_LINE */
         free(cols);   /* GCOVR_EXCL_LINE */
         free(vals);   /* GCOVR_EXCL_LINE */
         goto cleanup; /* GCOVR_EXCL_LINE */
      }

      free(rows);
      free(cols);
      free(vals);
   }

   hypredrv_IJMatrixReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                                        memory_location, matrix_ptr);
   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (hypredrv_ErrorCodeActive() || !*matrix_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }

   ok = 1;

cleanup:
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (fp)                   /* GCOVR_EXCL_BR_STOP */
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
   LSSeqData seq;
   FILE     *fp                          = NULL;
   int      *partids                     = NULL;
   uint32_t *part_order                  = NULL;
   int       nparts                      = 0;
   char      prefix[MAX_FILENAME_LENGTH] = {0};
   char      part_filename[MAX_FILENAME_LENGTH];
   int       ok = 0;

   if (!rhs_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null RHS pointer for LSSeqReadRHS");
      return 0;
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */                              /* low-signal branch under CI */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq); /* GCOVR_EXCL_LINE */
      return 0;               /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);          /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq); /* GCOVR_EXCL_LINE */
      return 0;               /* GCOVR_EXCL_LINE */
   }

   fp = fopen(filename, "rb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);                               /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqTempPrefixBuild(comm, ls_id, "b", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }
   for (int i = 0; i < nparts; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqPartMeta       *part        = &seq.parts[part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      void  *vals      = NULL;
      size_t vals_size = 0;

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqReadPartBlobSlice(fp, (comp_alg_t)seq.header.codec,
                                  /* GCOVR_EXCL_BR_STOP */
                                  seq.header.offset_blob_data, seq.part_blob_table,
                                  part_id, 1, sys->rhs_blob_offset, sys->rhs_blob_size,
                                  &vals, &vals_size))
      {
         free(vals);   /* GCOVR_EXCL_LINE */
         goto cleanup; /* GCOVR_EXCL_LINE */
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, ".bin"))
      {
         free(vals);   /* GCOVR_EXCL_LINE */
         goto cleanup; /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqWriteRHSPartFile(part_filename, part, vals)) /* GCOVR_EXCL_BR_STOP */
      {
         free(vals);   /* GCOVR_EXCL_LINE */
         goto cleanup; /* GCOVR_EXCL_LINE */
      }
      free(vals);
   }

   hypredrv_IJVectorReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                                        memory_location, rhs_ptr);
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (hypredrv_ErrorCodeActive() || !*rhs_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }

   ok = 1;

cleanup:
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (fp)                   /* GCOVR_EXCL_BR_STOP */
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
   LSSeqData seq;
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   char      prefix[MAX_FILENAME_LENGTH];
   char      part_filename[MAX_FILENAME_LENGTH];
   int       myid = 0;
   int       ok   = 0;

   if (!dofmap_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null dofmap pointer for LSSeqReadDofmap");
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (*dofmap_ptr)          /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayDestroy(dofmap_ptr); /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   if (!LSSeqDataLoad(filename, &seq)) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   if (!(seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP))
   {
      *dofmap_ptr = hypredrv_IntArrayCreate(0);
      LSSeqDataDestroy(&seq);
      return (*dofmap_ptr != NULL);
   }

   /* GCOVR_EXCL_BR_START */                              /* low-signal branch under CI */
   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems) /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                           seq.header.num_systems);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq); /* GCOVR_EXCL_LINE */
      return 0;               /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (!LSSeqBuildPartOrder(&seq, &part_order)) /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);          /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq); /* GCOVR_EXCL_LINE */
      return 0;               /* GCOVR_EXCL_LINE */
   }

   fp = fopen(filename, "rb");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      free(partids);                               /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Could not open sequence file '%s'",
                           filename); /* GCOVR_EXCL_LINE */
      return 0;                       /* GCOVR_EXCL_LINE */
   }

   prefix[0] = '\0';
   MPI_Comm_rank(comm, &myid);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!LSSeqSharedTempPrefixBuild(comm, ls_id, "dof", prefix, sizeof(prefix)))
   /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }

   for (int i = 0; i < nparts; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      int32_t *dof_data = NULL;
      size_t   dof_size = 0;
      FILE    *out      = NULL;

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!LSSeqFormatPartFilename(part_filename, sizeof(part_filename), prefix,
                                   /* GCOVR_EXCL_BR_STOP */
                                   tmp_part_id, NULL))
      {
         goto cleanup; /* GCOVR_EXCL_LINE */
      }
      out = fopen(part_filename, "w");
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!out)                 /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Could not create dofmap temporary part '%s'", /* GCOVR_EXCL_LINE */
            part_filename);
         goto cleanup; /* GCOVR_EXCL_LINE */
      }

      /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
      if (sys->dof_num_entries > 0) /* GCOVR_EXCL_BR_STOP */
      {
         size_t expected_size = 0;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!LSSeqCheckedMulSize(
                (size_t)sys->dof_num_entries, sizeof(int32_t),
                /* GCOVR_EXCL_BR_STOP */
                /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                &expected_size, "dof payload size") ||
             /* GCOVR_EXCL_BR_STOP */
             !LSSeqValidateByteLimit(expected_size, LSSEQ_MAX_BLOB_BYTES, "dof payload"))
         {
            fclose(out);  /* GCOVR_EXCL_LINE */
            goto cleanup; /* GCOVR_EXCL_LINE */
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!LSSeqReadPartBlobSlice(
                /* GCOVR_EXCL_BR_STOP */
                fp, (comp_alg_t)seq.header.codec, seq.header.offset_blob_data,
                seq.part_blob_table, part_id, 2, sys->dof_blob_offset,
                (uint64_t)expected_size, (void **)&dof_data, &dof_size))
         {
            fclose(out);    /* GCOVR_EXCL_LINE */
            free(dof_data); /* GCOVR_EXCL_LINE */
            goto cleanup;   /* GCOVR_EXCL_LINE */
         }
      }

      fprintf(out, "%llu\n", (unsigned long long)sys->dof_num_entries);
      for (uint64_t j = 0; j < sys->dof_num_entries; j++)
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         int value = dof_data ? (int)dof_data[j] : 0;
         /* GCOVR_EXCL_BR_STOP */
         fprintf(out, "%d\n", value);
      }
      fclose(out);
      free(dof_data);
   }

   /* Ensure all rank-local dof files are visible before parallel read. */
   MPI_Barrier(comm);
   hypredrv_IntArrayParRead(comm, prefix, dofmap_ptr);
   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (hypredrv_ErrorCodeActive() || !*dofmap_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      goto cleanup; /* GCOVR_EXCL_LINE */
   }

   ok = 1;

cleanup:
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (ok)                   /* GCOVR_EXCL_BR_STOP */
   {
      MPI_Barrier(comm);
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (fp)                   /* GCOVR_EXCL_BR_STOP */
   {
      fclose(fp);
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (prefix[0] != '\0')    /* GCOVR_EXCL_BR_STOP */
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

   /* GCOVR_EXCL_BR_START */          /* low-signal branch under CI */
   if (timestep_ids && *timestep_ids) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayDestroy(timestep_ids);
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (*timestep_starts)     /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayDestroy(timestep_starts);
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!(seq.header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) || seq.header.num_timesteps == 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }

   if (timestep_ids)
   {
      ids = hypredrv_IntArrayCreate((size_t)seq.header.num_timesteps);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!ids)                 /* GCOVR_EXCL_BR_STOP */
      {
         LSSeqDataDestroy(&seq);                  /* GCOVR_EXCL_LINE */
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate LSSeq timestep ids array"); /* GCOVR_EXCL_LINE */
         return 0;                                          /* GCOVR_EXCL_LINE */
      }
   }

   starts = hypredrv_IntArrayCreate((size_t)seq.header.num_timesteps);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!starts)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_IntArrayDestroy(&ids);          /* GCOVR_EXCL_LINE */
      LSSeqDataDestroy(&seq);                  /* GCOVR_EXCL_LINE */
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate LSSeq timestep starts array"); /* GCOVR_EXCL_LINE */
      return 0;                                             /* GCOVR_EXCL_LINE */
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
