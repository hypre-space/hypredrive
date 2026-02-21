/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "lsseq.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "error.h"
#include "linsys.h"

typedef struct LSSeqData_struct
{
   LSSeqHeader          header;
   LSSeqInfoHeader      info_header;
   char                *info_payload;
   size_t               info_payload_size;
   LSSeqPartMeta       *parts;
   LSSeqPatternMeta    *patterns;
   LSSeqSystemPartMeta *sys_parts;
   LSSeqTimestepEntry  *timesteps;
} LSSeqData;

static int
LSSeqBuildPartOrder(const LSSeqData *seq, uint32_t **order_ptr)
{
   uint32_t *order = NULL;
   uint32_t  n     = 0;

   if (!seq || !order_ptr)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments for LSSeqBuildPartOrder");
      return 0;
   }
   *order_ptr = NULL;
   n          = seq->header.num_parts;
   order      = (uint32_t *)malloc((size_t)n * sizeof(*order));
   if (!order)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate LSSeq part order (%u entries)", n);
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
         if (cur_lo < key_lo || (cur_lo == key_lo && cur_hi <= key_hi))
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
   if (!bytes)
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
   if (!fp || !buffer)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments while reading %s", what ? what : "lsseq data");
      return 0;
   }

   if (fseeko(fp, (off_t)offset, SEEK_SET) != 0)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Could not seek to offset %llu while reading %s",
                  (unsigned long long)offset, what ? what : "lsseq data");
      return 0;
   }

   if (nbytes > 0 && fread(buffer, 1, nbytes, fp) != nbytes)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Could not read %s (%zu bytes)", what ? what : "lsseq data", nbytes);
      return 0;
   }

   return 1;
}

static int
LSSeqValidateHeader(const LSSeqHeader *header, const char *filename)
{
   if (!header)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Null lsseq header");
      return 0;
   }

   if (header->magic != LSSEQ_MAGIC)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid LSSeq magic for file '%s'", filename ? filename : "");
      return 0;
   }
   if (header->version != LSSEQ_VERSION)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Unsupported LSSeq version %u in '%s'", header->version,
                  filename ? filename : "");
      return 0;
   }
   if (header->num_systems == 0 || header->num_parts == 0)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid LSSeq shape in '%s': num_systems=%u, num_parts=%u",
                  filename ? filename : "", header->num_systems, header->num_parts);
      return 0;
   }
   if (header->codec > (uint32_t)COMP_BLOSC)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid LSSeq compression codec id %u in '%s'", header->codec,
                  filename ? filename : "");
      return 0;
   }

   return 1;
}

static void
LSSeqDataDestroy(LSSeqData *seq)
{
   if (!seq)
   {
      return;
   }
   free(seq->info_payload);
   free(seq->parts);
   free(seq->patterns);
   free(seq->sys_parts);
   free(seq->timesteps);
   memset(seq, 0, sizeof(*seq));
}

static int
LSSeqDataLoad(const char *filename, LSSeqData *seq)
{
   FILE          *fp          = NULL;
   size_t         n_sys_parts = 0;
   const uint64_t info_offset = (uint64_t)sizeof(LSSeqHeader);

   if (!filename || !seq)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments to LSSeqDataLoad");
      return 0;
   }

   memset(seq, 0, sizeof(*seq));
   fp = fopen(filename, "rb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   if (!LSSeqReadAt(fp, 0, &seq->header, sizeof(seq->header), "lsseq header"))
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
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Missing mandatory LSSeq info header in '%s'", filename);
         return 0;
      }

      if (seq->header.offset_part_meta < expected_min_part_offset)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid LSSeq info offsets in '%s' (offset_part_meta=%llu)",
                     filename, (unsigned long long)seq->header.offset_part_meta);
         return 0;
      }

      if (!LSSeqReadAt(fp, info_offset, &seq->info_header, sizeof(seq->info_header),
                       "info header"))
      {
         fclose(fp);
         return 0;
      }

      if (seq->info_header.magic != LSSEQ_INFO_MAGIC ||
          seq->info_header.version != LSSEQ_INFO_VERSION)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid LSSeq info header in '%s' (magic=%llu version=%u)",
                     filename, (unsigned long long)seq->info_header.magic,
                     seq->info_header.version);
         return 0;
      }

      if (seq->info_header.endian_tag != UINT32_C(0x01020304))
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Unsupported LSSeq info endianness tag in '%s' (tag=0x%08x)",
                     filename, (unsigned int)seq->info_header.endian_tag);
         return 0;
      }

      /* Bound payload size to avoid accidental huge allocations. */
      if (seq->info_header.payload_size > (uint64_t)(16u * 1024u * 1024u) ||
          seq->info_header.payload_size > (uint64_t)SIZE_MAX - 1u)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("LSSeq info payload too large in '%s' (%llu bytes)", filename,
                     (unsigned long long)seq->info_header.payload_size);
         return 0;
      }

      expected_payload_end =
         expected_min_part_offset + (uint64_t)seq->info_header.payload_size;
      if (seq->header.offset_part_meta < expected_payload_end)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("LSSeq info payload overlaps part metadata in '%s' "
                     "(payload_end=%llu part_off=%llu)",
                     filename, (unsigned long long)expected_payload_end,
                     (unsigned long long)seq->header.offset_part_meta);
         return 0;
      }

      if (seq->info_header.payload_size > 0)
      {
         uint64_t hash          = UINT64_C(1469598103934665603);
         seq->info_payload_size = (size_t)seq->info_header.payload_size;
         seq->info_payload      = (char *)malloc(seq->info_payload_size + 1u);
         if (!seq->info_payload)
         {
            fclose(fp);
            ErrorCodeSet(ERROR_ALLOCATION);
            ErrorMsgAdd("Failed to allocate LSSeq info payload (%zu bytes)",
                        seq->info_payload_size);
            return 0;
         }

         if (!LSSeqReadAt(fp, expected_min_part_offset, seq->info_payload,
                          seq->info_payload_size, "info payload"))
         {
            fclose(fp);
            LSSeqDataDestroy(seq);
            return 0;
         }
         seq->info_payload[seq->info_payload_size] = '\0';

         hash = LSSeqFNV1a64(seq->info_payload, seq->info_payload_size, hash);
         if (hash != seq->info_header.payload_hash_fnv1a64)
         {
            fclose(fp);
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'", filename);
            LSSeqDataDestroy(seq);
            return 0;
         }
      }
   }

   seq->parts = (LSSeqPartMeta *)calloc(seq->header.num_parts, sizeof(LSSeqPartMeta));
   if (!seq->parts)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate LSSeq part metadata");
      return 0;
   }

   if (seq->header.num_patterns > 0)
   {
      seq->patterns =
         (LSSeqPatternMeta *)calloc(seq->header.num_patterns, sizeof(LSSeqPatternMeta));
      if (!seq->patterns)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Failed to allocate LSSeq pattern metadata");
         LSSeqDataDestroy(seq);
         return 0;
      }
   }

   n_sys_parts = (size_t)seq->header.num_systems * (size_t)seq->header.num_parts;
   seq->sys_parts =
      (LSSeqSystemPartMeta *)calloc(n_sys_parts, sizeof(LSSeqSystemPartMeta));
   if (!seq->sys_parts)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate LSSeq system-part metadata");
      LSSeqDataDestroy(seq);
      return 0;
   }

   if ((seq->header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq->header.num_timesteps > 0)
   {
      seq->timesteps = (LSSeqTimestepEntry *)calloc(seq->header.num_timesteps,
                                                    sizeof(LSSeqTimestepEntry));
      if (!seq->timesteps)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Failed to allocate LSSeq timesteps metadata");
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

   if (seq->header.num_patterns > 0 &&
       !LSSeqReadAt(fp, seq->header.offset_pattern_meta, seq->patterns,
                    (size_t)seq->header.num_patterns * sizeof(LSSeqPatternMeta),
                    "pattern metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   if (!LSSeqReadAt(fp, seq->header.offset_sys_part_meta, seq->sys_parts,
                    n_sys_parts * sizeof(LSSeqSystemPartMeta), "system-part metadata"))
   {
      fclose(fp);
      LSSeqDataDestroy(seq);
      return 0;
   }

   if (seq->timesteps &&
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
LSSeqReadInfo(const char *filename, char **payload_ptr, size_t *payload_size)
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
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid LSSeqReadInfo outputs");
      return 0;
   }
   *payload_ptr  = NULL;
   *payload_size = 0;

   if (!filename)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Missing sequence filename");
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   if (!LSSeqReadAt(fp, 0, &header, sizeof(header), "lsseq header"))
   {
      fclose(fp);
      return 0;
   }
   if (!LSSeqValidateHeader(&header, filename))
   {
      fclose(fp);
      return 0;
   }

   if (!(header.flags & LSSEQ_FLAG_HAS_INFO))
   {
      fclose(fp);
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Missing mandatory LSSeq info header in '%s'", filename);
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
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid LSSeq info header in '%s'", filename);
      return 0;
   }

   if (info.payload_size > (uint64_t)SIZE_MAX - 1u)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("LSSeq info payload too large in '%s' (%llu bytes)", filename,
                  (unsigned long long)info.payload_size);
      return 0;
   }

   nbytes  = (size_t)info.payload_size;
   payload = (char *)malloc(nbytes + 1u);
   if (!payload)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate LSSeq info payload (%zu bytes)", nbytes);
      return 0;
   }

   if (nbytes > 0 && !LSSeqReadAt(fp, info_offset + (uint64_t)sizeof(info), payload,
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
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("LSSeq info payload hash mismatch in '%s'", filename);
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

   if (!partids_ptr || !nparts_ptr)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid LSSeqLocalPartIDs outputs");
      return 0;
   }

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = (int)(g_nparts / (uint32_t)nprocs);
   rem    = (int)(g_nparts % (uint32_t)nprocs);
   nparts += (myid < rem) ? 1 : 0;
   if (g_nparts < (uint32_t)nprocs)
   {
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid number of sequence parts (%u) for communicator size %d",
                  g_nparts, nprocs);
      return 0;
   }

   partids = (int *)calloc((size_t)nparts, sizeof(int));
   if (!partids)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate partids array");
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

   if (!fp || !output || !output_size)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid LSSeq blob read arguments");
      return 0;
   }

   *output      = NULL;
   *output_size = 0;

   if (blob_size > (uint64_t)SIZE_MAX)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Blob size too large to decode (%llu bytes)",
                  (unsigned long long)blob_size);
      return 0;
   }

   if (blob_size == 0)
   {
      if (expected_size != 0)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Encountered empty blob for non-empty expected payload (%zu bytes)",
                     expected_size);
         return 0;
      }
      return 1;
   }

   blob_data = malloc((size_t)blob_size);
   if (!blob_data)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate %llu bytes for blob read",
                  (unsigned long long)blob_size);
      return 0;
   }

   if (!LSSeqReadAt(fp, offset, blob_data, (size_t)blob_size, "blob payload"))
   {
      free(blob_data);
      return 0;
   }

   if (codec == COMP_NONE)
   {
      decoded = malloc((size_t)blob_size);
      if (!decoded)
      {
         free(blob_data);
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Failed to allocate %llu bytes for blob decode",
                     (unsigned long long)blob_size);
         return 0;
      }
      memcpy(decoded, blob_data, (size_t)blob_size);
      decoded_size = (size_t)blob_size;
   }
   else
   {
      hypredrv_decompress(codec, (size_t)blob_size, blob_data, &decoded_size, &decoded);
      if (ErrorCodeActive() || !decoded)
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
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Decoded blob size mismatch: expected=%zu got=%zu", expected_size,
                  decoded_size);
      return 0;
   }

   *output      = decoded;
   *output_size = decoded_size;
   return 1;
}

static void
LSSeqTempPrefixBuild(MPI_Comm comm, int ls_id, const char *tag, char *prefix,
                     size_t prefix_size)
{
   int myid = 0;
   MPI_Comm_rank(comm, &myid);
   snprintf(prefix, prefix_size, "/tmp/hypredrv_lsseq_%s_%d_%d_%d", tag ? tag : "tmp",
            (int)getpid(), myid, ls_id);
}

static void
LSSeqCleanupPartFiles(const char *prefix, const int *partids, int nparts,
                      const char *suffix)
{
   char filename[MAX_FILENAME_LENGTH];
   for (int i = 0; i < nparts; i++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, partids[i],
               suffix ? suffix : "");
      remove(filename);
   }
}

static int
LSSeqWriteMatrixPartFile(const char *filename, const LSSeqPartMeta *part,
                         const LSSeqPatternMeta *pattern, const void *rows,
                         const void *cols, const void *vals)
{
   FILE    *fp         = NULL;
   uint64_t header[11] = {0};

   if (!filename || !part || !pattern)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid matrix part-file write arguments");
      return 0;
   }

   fp = fopen(filename, "wb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not create matrix temporary part '%s'", filename);
      return 0;
   }

   header[1] = part->row_index_size;
   header[2] = part->value_size;
   header[5] = part->row_upper - part->row_lower + 1;
   header[6] = pattern->nnz;
   header[7] = part->row_lower;
   header[8] = part->row_upper;

   if (fwrite(header, sizeof(uint64_t), 11, fp) != 11)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Could not write matrix header to '%s'", filename);
      return 0;
   }

   if (pattern->nnz > 0)
   {
      size_t nnz = (size_t)pattern->nnz;
      if ((rows && fwrite(rows, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          (cols && fwrite(cols, (size_t)part->row_index_size, nnz, fp) != nnz) ||
          (vals && fwrite(vals, (size_t)part->value_size, nnz, fp) != nnz))
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not write matrix data to '%s'", filename);
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

   if (!filename || !part)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid RHS part-file write arguments");
      return 0;
   }

   fp = fopen(filename, "wb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not create RHS temporary part '%s'", filename);
      return 0;
   }

   header[1] = part->value_size;
   header[5] = part->nrows;

   if (fwrite(header, sizeof(uint64_t), 8, fp) != 8)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Could not write RHS header to '%s'", filename);
      return 0;
   }

   if (part->nrows > 0)
   {
      size_t nrows = (size_t)part->nrows;
      if (vals && fwrite(vals, (size_t)part->value_size, nrows, fp) != nrows)
      {
         fclose(fp);
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not write RHS values to '%s'", filename);
         return 0;
      }
   }

   fclose(fp);
   return 1;
}

int
LSSeqReadSummary(const char *filename, int *num_systems, int *num_patterns,
                 int *has_dofmap, int *has_timesteps)
{
   LSSeqHeader header;
   FILE       *fp = NULL;

   if (!filename)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Missing sequence filename");
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }
   if (fread(&header, sizeof(header), 1, fp) != 1)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Could not read sequence header from '%s'", filename);
      return 0;
   }
   fclose(fp);

   if (!LSSeqValidateHeader(&header, filename))
   {
      return 0;
   }

   if (num_systems)
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
LSSeqReadMatrix(MPI_Comm comm, const char *filename, int ls_id,
                HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *matrix_ptr)
{
   LSSeqData seq;
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   char      prefix[MAX_FILENAME_LENGTH];
   char      part_filename[MAX_FILENAME_LENGTH];
   int       ok = 0;

   if (!matrix_ptr)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Null matrix pointer for LSSeqReadMatrix");
      return 0;
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems)
   {
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                  seq.header.num_systems);
      return 0;
   }

   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order))
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   LSSeqTempPrefixBuild(comm, ls_id, "A", prefix, sizeof(prefix));
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
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid pattern id %u for system %d part %u", sys->pattern_id,
                     ls_id, part_id);
         goto cleanup;
      }
      pattern = &seq.patterns[sys->pattern_id];
      if (pattern->part_id != part_id)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Pattern-part mismatch for system %d part %u (pattern part=%u)",
                     ls_id, part_id, pattern->part_id);
         goto cleanup;
      }

      expected_size = (size_t)pattern->nnz * (size_t)part->row_index_size;
      if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->rows_blob_offset,
                         pattern->rows_blob_size, expected_size, &rows, &rows_size) ||
          !LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, pattern->cols_blob_offset,
                         pattern->cols_blob_size, expected_size, &cols, &cols_size))
      {
         free(rows);
         free(cols);
         goto cleanup;
      }

      expected_size = (size_t)sys->nnz * (size_t)part->value_size;
      if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, sys->values_blob_offset,
                         sys->values_blob_size, expected_size, &vals, &vals_size))
      {
         free(rows);
         free(cols);
         free(vals);
         goto cleanup;
      }

      snprintf(part_filename, sizeof(part_filename), "%s.%05u.bin", prefix, tmp_part_id);
      if (!LSSeqWriteMatrixPartFile(part_filename, part, pattern, rows, cols, vals))
      {
         free(rows);
         free(cols);
         free(vals);
         goto cleanup;
      }

      free(rows);
      free(cols);
      free(vals);
   }

   IJMatrixReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                               memory_location, matrix_ptr);
   if (ErrorCodeActive() || !*matrix_ptr)
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (fp)
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
LSSeqReadRHS(MPI_Comm comm, const char *filename, int ls_id,
             HYPRE_MemoryLocation memory_location, HYPRE_IJVector *rhs_ptr)
{
   LSSeqData seq;
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   char      prefix[MAX_FILENAME_LENGTH];
   char      part_filename[MAX_FILENAME_LENGTH];
   int       ok = 0;

   if (!rhs_ptr)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Null RHS pointer for LSSeqReadRHS");
      return 0;
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems)
   {
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                  seq.header.num_systems);
      return 0;
   }

   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order))
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   LSSeqTempPrefixBuild(comm, ls_id, "b", prefix, sizeof(prefix));
   for (int i = 0; i < nparts; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqPartMeta       *part        = &seq.parts[part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      void  *vals          = NULL;
      size_t vals_size     = 0;
      size_t expected_size = (size_t)part->nrows * (size_t)part->value_size;

      if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, sys->rhs_blob_offset,
                         sys->rhs_blob_size, expected_size, &vals, &vals_size))
      {
         free(vals);
         goto cleanup;
      }

      snprintf(part_filename, sizeof(part_filename), "%s.%05u.bin", prefix, tmp_part_id);
      if (!LSSeqWriteRHSPartFile(part_filename, part, vals))
      {
         free(vals);
         goto cleanup;
      }
      free(vals);
   }

   IJVectorReadMultipartBinary(prefix, comm, (uint64_t)seq.header.num_parts,
                               memory_location, rhs_ptr);
   if (ErrorCodeActive() || !*rhs_ptr)
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (fp)
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
LSSeqReadDofmap(MPI_Comm comm, const char *filename, int ls_id, IntArray **dofmap_ptr)
{
   LSSeqData seq;
   FILE     *fp         = NULL;
   int      *partids    = NULL;
   uint32_t *part_order = NULL;
   int       nparts     = 0;
   char      prefix[MAX_FILENAME_LENGTH];
   char      part_filename[MAX_FILENAME_LENGTH];
   int       myid = 0, root_pid = 0;
   int       ok = 0;

   if (!dofmap_ptr)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Null dofmap pointer for LSSeqReadDofmap");
      return 0;
   }

   if (*dofmap_ptr)
   {
      IntArrayDestroy(dofmap_ptr);
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   if (!(seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP))
   {
      *dofmap_ptr = IntArrayCreate(0);
      LSSeqDataDestroy(&seq);
      return (*dofmap_ptr != NULL);
   }

   if (ls_id < 0 || ls_id >= (int)seq.header.num_systems)
   {
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid sequence linear-system id %d (max: %u)", ls_id,
                  seq.header.num_systems);
      return 0;
   }

   if (!LSSeqLocalPartIDs(comm, seq.header.num_parts, &partids, &nparts))
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }
   if (!LSSeqBuildPartOrder(&seq, &part_order))
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      free(partids);
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open sequence file '%s'", filename);
      return 0;
   }

   prefix[0] = '\0';
   MPI_Comm_rank(comm, &myid);
   if (!myid)
   {
      root_pid = (int)getpid();
   }
   MPI_Bcast(&root_pid, 1, MPI_INT, 0, comm);
   snprintf(prefix, sizeof(prefix), "/tmp/hypredrv_lsseq_dof_%d_%d", root_pid, ls_id);

   for (int i = 0; i < nparts; i++)
   {
      uint32_t                   tmp_part_id = (uint32_t)partids[i];
      uint32_t                   part_id     = part_order[tmp_part_id];
      const LSSeqSystemPartMeta *sys =
         &seq.sys_parts[((size_t)ls_id * (size_t)seq.header.num_parts) + (size_t)part_id];
      int32_t *dof_data = NULL;
      size_t   dof_size = 0;
      FILE    *out      = NULL;

      snprintf(part_filename, sizeof(part_filename), "%s.%05u", prefix, tmp_part_id);
      out = fopen(part_filename, "w");
      if (!out)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAdd("Could not create dofmap temporary part '%s'", part_filename);
         goto cleanup;
      }

      if (sys->dof_blob_size > 0 && sys->dof_num_entries > 0)
      {
         size_t expected_size = (size_t)sys->dof_num_entries * sizeof(int32_t);
         if (!LSSeqReadBlob(fp, (comp_alg_t)seq.header.codec, sys->dof_blob_offset,
                            sys->dof_blob_size, expected_size, (void **)&dof_data,
                            &dof_size))
         {
            fclose(out);
            free(dof_data);
            goto cleanup;
         }
      }

      fprintf(out, "%llu\n", (unsigned long long)sys->dof_num_entries);
      for (uint64_t j = 0; j < sys->dof_num_entries; j++)
      {
         int value = dof_data ? (int)dof_data[j] : 0;
         fprintf(out, "%d\n", value);
      }
      fclose(out);
      free(dof_data);
   }

   /* Ensure all rank-local dof files are visible before parallel read. */
   MPI_Barrier(comm);
   IntArrayParRead(comm, prefix, dofmap_ptr);
   if (ErrorCodeActive() || !*dofmap_ptr)
   {
      goto cleanup;
   }

   ok = 1;

cleanup:
   if (ok)
   {
      MPI_Barrier(comm);
   }
   if (fp)
   {
      fclose(fp);
   }
   if (prefix[0] != '\0')
   {
      LSSeqCleanupPartFiles(prefix, partids, nparts, "");
   }
   free(part_order);
   free(partids);
   LSSeqDataDestroy(&seq);
   return ok;
}

int
LSSeqReadTimesteps(const char *filename, IntArray **timestep_starts)
{
   LSSeqData seq;
   IntArray *starts = NULL;

   if (!timestep_starts)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid output pointer for LSSeqReadTimesteps");
      return 0;
   }

   if (*timestep_starts)
   {
      IntArrayDestroy(timestep_starts);
   }

   if (!LSSeqDataLoad(filename, &seq))
   {
      return 0;
   }

   if (!(seq.header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) || seq.header.num_timesteps == 0)
   {
      LSSeqDataDestroy(&seq);
      return 0;
   }

   starts = IntArrayCreate((size_t)seq.header.num_timesteps);
   if (!starts)
   {
      LSSeqDataDestroy(&seq);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate LSSeq timestep starts array");
      return 0;
   }

   for (uint32_t i = 0; i < seq.header.num_timesteps; i++)
   {
      starts->data[i] = seq.timesteps[i].ls_start;
   }

   *timestep_starts = starts;
   LSSeqDataDestroy(&seq);
   return 1;
}
