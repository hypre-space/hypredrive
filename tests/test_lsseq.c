#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/comp.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "test_helpers.h"

static uint64_t
fnv1a64(const void *data, size_t nbytes)
{
   const unsigned char *bytes = (const unsigned char *)data;
   uint64_t             hash  = UINT64_C(1469598103934665603);
   for (size_t i = 0; i < nbytes; i++)
   {
      hash ^= (uint64_t)bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static void
write_test_container(const char *filename)
{
   LSSeqHeader         header;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[2];
   LSSeqSystemPartMeta sys_meta[2];
   LSSeqTimestepEntry  timesteps[2];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   HYPRE_BigInt        rows1[3] = {0, 0, 1};
   HYPRE_BigInt        cols1[3] = {0, 1, 1};
   double              vals0[2] = {10.0, 20.0};
   double              vals1[3] = {5.0, 1.0, 3.0};
   double              rhs0[2]  = {1.0, 2.0};
   double              rhs1[2]  = {3.0, 4.0};
   int32_t             dof0[2]  = {0, 1};
   int32_t             dof1[2]  = {1, 1};
   uint64_t            blob_offset;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 2;
   header.num_parts     = 1;
   header.num_patterns  = 2;
   header.num_timesteps = 2;

   header.offset_part_meta     = sizeof(LSSeqHeader);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_timestep_meta = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_blob_data     = header.offset_timestep_meta + sizeof(timesteps);

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id         = 0;
   pattern_meta[0].nnz             = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pattern_meta[1].part_id         = 0;
   pattern_meta[1].nnz             = 3;
   pattern_meta[1].rows_blob_offset = blob_offset;
   pattern_meta[1].rows_blob_size   = sizeof(rows1);
   blob_offset += sizeof(rows1);
   pattern_meta[1].cols_blob_offset = blob_offset;
   pattern_meta[1].cols_blob_size   = sizeof(cols1);
   blob_offset += sizeof(cols1);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = blob_offset;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   blob_offset += sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = blob_offset;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   blob_offset += sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = blob_offset;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;
   blob_offset += sizeof(dof0);

   sys_meta[1].pattern_id         = 1;
   sys_meta[1].nnz                = 3;
   sys_meta[1].values_blob_offset = blob_offset;
   sys_meta[1].values_blob_size   = sizeof(vals1);
   blob_offset += sizeof(vals1);
   sys_meta[1].rhs_blob_offset    = blob_offset;
   sys_meta[1].rhs_blob_size      = sizeof(rhs1);
   blob_offset += sizeof(rhs1);
   sys_meta[1].dof_blob_offset    = blob_offset;
   sys_meta[1].dof_blob_size      = sizeof(dof1);
   sys_meta[1].dof_num_entries    = 2;
   blob_offset += sizeof(dof1);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;
   timesteps[1].timestep = 1;
   timesteps[1].ls_start = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof1, sizeof(dof1), 1, fp), 1);

   fclose(fp);
}

static void
write_test_container_with_info(const char *filename)
{
   const char          payload[] = "foo=bar\nhello=world\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[2];
   LSSeqSystemPartMeta sys_meta[2];
   LSSeqTimestepEntry  timesteps[2];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   HYPRE_BigInt        rows1[3] = {0, 0, 1};
   HYPRE_BigInt        cols1[3] = {0, 1, 1};
   double              vals0[2] = {10.0, 20.0};
   double              vals1[3] = {5.0, 1.0, 3.0};
   double              rhs0[2]  = {1.0, 2.0};
   double              rhs1[2]  = {3.0, 4.0};
   int32_t             dof0[2]  = {0, 1};
   int32_t             dof1[2]  = {1, 1};
   uint64_t            blob_offset;
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 2;
   header.num_parts     = 1;
   header.num_patterns  = 2;
   header.num_timesteps = 2;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta     = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta    = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta    = header.offset_part_blob_table +
                                 (uint64_t)(1 * LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic               = LSSEQ_INFO_MAGIC;
   info.version             = LSSEQ_INFO_VERSION;
   info.flags               = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag          = UINT32_C(0x01020304);
   info.payload_size        = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64   = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pattern_meta[1].part_id          = 0;
   pattern_meta[1].nnz              = 3;
   pattern_meta[1].rows_blob_offset = blob_offset;
   pattern_meta[1].rows_blob_size   = sizeof(rows1);
   blob_offset += sizeof(rows1);
   pattern_meta[1].cols_blob_offset = blob_offset;
   pattern_meta[1].cols_blob_size   = sizeof(cols1);
   blob_offset += sizeof(cols1);

   {
      uint64_t pattern_size = blob_offset - header.offset_blob_data;
      part_blob_table[0] = pattern_size;
      part_blob_table[1] = sizeof(vals0) + sizeof(vals1);
      part_blob_table[2] = pattern_size + part_blob_table[1];
      part_blob_table[3] = sizeof(rhs0) + sizeof(rhs1);
      part_blob_table[4] = part_blob_table[2] + part_blob_table[3];
      part_blob_table[5] = sizeof(dof0) + sizeof(dof1);

      sys_meta[0].pattern_id         = 0;
      sys_meta[0].nnz                = 2;
      sys_meta[0].values_blob_offset = 0;
      sys_meta[0].values_blob_size   = sizeof(vals0);
      sys_meta[0].rhs_blob_offset    = 0;
      sys_meta[0].rhs_blob_size     = sizeof(rhs0);
      sys_meta[0].dof_blob_offset   = 0;
      sys_meta[0].dof_blob_size     = sizeof(dof0);
      sys_meta[0].dof_num_entries   = 2;

      sys_meta[1].pattern_id         = 1;
      sys_meta[1].nnz                = 3;
      sys_meta[1].values_blob_offset = sizeof(vals0);
      sys_meta[1].values_blob_size  = sizeof(vals1);
      sys_meta[1].rhs_blob_offset   = sizeof(rhs0);
      sys_meta[1].rhs_blob_size     = sizeof(rhs1);
      sys_meta[1].dof_blob_offset   = sizeof(dof0);
      sys_meta[1].dof_blob_size     = sizeof(dof1);
      sys_meta[1].dof_num_entries   = 2;

      blob_offset = header.offset_blob_data + pattern_size + part_blob_table[1] +
                    part_blob_table[3] + part_blob_table[5];
   }

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;
   timesteps[1].timestep = 1;
   timesteps[1].ls_start = 1;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   /* Blobs: pattern then batched part 0 (vals, rhs, dof) */
   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof1, sizeof(dof1), 1, fp), 1);

   fclose(fp);
}

/* nnz>0 but rows_blob_size==0: LSSeqReadBlob sees empty blob with non-zero expected size. */
static void
write_lsseq_empty_row_blob_mismatch(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = 0;
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id       = 0;
   sys_meta[0].nnz              = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

/* One linear system, two parts: part metadata order has high row range before low so
 * LSSeqBuildPartOrder must reorder (exercises insertion sort in lsseq.c). */
static void
write_two_part_container_with_info(const char *filename)
{
   const char          payload[] = "foo=bar\nhello=world\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[2];
   LSSeqPatternMeta    pattern_meta[2];
   LSSeqSystemPartMeta sys_meta[2];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES * 2];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   HYPRE_BigInt        rows1[3] = {0, 0, 1};
   HYPRE_BigInt        cols1[3] = {0, 1, 1};
   double              vals0[2] = {10.0, 20.0};
   double              vals1[3] = {5.0, 1.0, 3.0};
   double              rhs0[2]  = {1.0, 2.0};
   double              rhs1[2]  = {3.0, 4.0};
   int32_t             dof0[2]  = {0, 1};
   int32_t             dof1[2]  = {1, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   uint64_t            p0_chunk;
   uint64_t            part1_start;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 2;
   header.num_patterns  = 2;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta =
      header.offset_part_blob_table +
      (uint64_t)(2 * LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   /* Part 0 in file: high global rows first (sort should place part 1 before part 0). */
   part_meta[0].row_lower      = 100;
   part_meta[0].row_upper      = 101;
   part_meta[0].nrows            = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   part_meta[1].row_lower      = 0;
   part_meta[1].row_upper      = 1;
   part_meta[1].nrows          = 2;
   part_meta[1].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[1].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pattern_meta[1].part_id          = 1;
   pattern_meta[1].nnz              = 3;
   pattern_meta[1].rows_blob_offset = blob_offset;
   pattern_meta[1].rows_blob_size   = sizeof(rows1);
   blob_offset += sizeof(rows1);
   pattern_meta[1].cols_blob_offset = blob_offset;
   pattern_meta[1].cols_blob_size   = sizeof(cols1);
   blob_offset += sizeof(cols1);

   pat_end   = blob_offset - header.offset_blob_data;
   p0_chunk  = sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);
   part1_start = pat_end + p0_chunk;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   part_blob_table[6]  = part1_start;
   part_blob_table[7]  = sizeof(vals1);
   part_blob_table[8]  = part1_start + sizeof(vals1);
   part_blob_table[9]  = sizeof(rhs1);
   part_blob_table[10] = part1_start + sizeof(vals1) + sizeof(rhs1);
   part_blob_table[11] = sizeof(dof1);

   sys_meta[0].pattern_id       = 0;
   sys_meta[0].nnz              = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   sys_meta[1].pattern_id       = 1;
   sys_meta[1].nnz              = 3;
   sys_meta[1].values_blob_offset = 0;
   sys_meta[1].values_blob_size   = sizeof(vals1);
   sys_meta[1].rhs_blob_offset    = 0;
   sys_meta[1].rhs_blob_size      = sizeof(rhs1);
   sys_meta[1].dof_blob_offset    = 0;
   sys_meta[1].dof_blob_size      = sizeof(dof1);
   sys_meta[1].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + p0_chunk + sizeof(vals1) +
                 sizeof(rhs1) + sizeof(dof1);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof1, sizeof(dof1), 1, fp), 1);

   fclose(fp);
}

static void
write_lsseq_with_oversized_info_payload(const char *filename)
{
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   FILE           *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));

   header.magic                = LSSEQ_MAGIC;
   header.version              = LSSEQ_VERSION;
   header.flags                = LSSEQ_FLAG_HAS_INFO;
   header.codec                = (uint32_t)COMP_NONE;
   header.num_systems          = 1;
   header.num_parts            = 1;
   header.offset_part_meta     = sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader);
   header.offset_part_blob_table = 1;

   info.magic      = LSSEQ_INFO_MAGIC;
   info.version    = LSSEQ_INFO_VERSION;
   info.endian_tag = UINT32_C(0x01020304);
   info.payload_size = (uint64_t)(16u * 1024u * 1024u) + 1u;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_with_excessive_counts(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));

   header.magic                 = LSSEQ_MAGIC;
   header.version               = LSSEQ_VERSION;
   header.flags                 = LSSEQ_FLAG_HAS_INFO;
   header.codec                 = (uint32_t)COMP_NONE;
   header.num_systems           = 1;
   header.num_parts             = 2000000u;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_bad_magic(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = UINT64_C(0xdeadbeefcafebabe);
   header.version                = LSSEQ_VERSION;
   header.offset_part_blob_table = 1;
   header.num_systems            = 1;
   header.num_parts              = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_bad_version(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = 999u;
   header.offset_part_blob_table = 1;
   header.num_systems            = 1;
   header.num_parts              = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_truncated_header(const char *filename)
{
   char        buf[8] = {0};
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   ASSERT_EQ_SIZE(fwrite(buf, sizeof(buf), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_invalid_codec(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = 99999u;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_zero_systems(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 0;
   header.num_parts              = 1;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_zero_blob_table_offset(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 0;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_missing_info_flag(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   /* v1 containers require HAS_INFO; omit flag to hit LSSeqDataLoad error path */
   header.flags                  = LSSEQ_FLAG_HAS_DOFMAP;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_header_only(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 400;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_readinfo_no_has_info_flag(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_DOFMAP;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_readinfo_bad_info_magic(const char *filename)
{
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   FILE           *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;

   info.magic         = UINT64_C(0xAAAAAAAAAAAAAAAA);
   info.version       = LSSEQ_INFO_VERSION;
   info.endian_tag    = UINT32_C(0x01020304);
   info.payload_size  = 0;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_readinfo_bad_info_endian(const char *filename)
{
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   FILE           *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;

   info.magic        = LSSEQ_INFO_MAGIC;
   info.version      = LSSEQ_INFO_VERSION;
   info.endian_tag   = UINT32_C(0xFFFFFFFF);
   info.payload_size = 0;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_readinfo_hash_mismatch(const char *filename)
{
   const char          payload[] = "abc";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = UINT64_C(1);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_num_parts_zero(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 0;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_excessive_patterns(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.num_patterns           = 2000000u;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_excessive_timesteps(const char *filename)
{
   LSSeqHeader header;
   FILE       *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.num_patterns           = 1;
   header.num_timesteps          = 2000000u;
   header.offset_part_blob_table = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_info_bad_version(const char *filename)
{
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   FILE           *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;

   info.magic    = LSSEQ_INFO_MAGIC;
   info.version  = 999u;
   info.endian_tag = UINT32_C(0x01020304);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_payload_overlaps_part_meta(const char *filename)
{
   const char          payload[] = "overlap";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   uint64_t            expected_payload_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);

   expected_payload_end =
      (uint64_t)sizeof(LSSeqHeader) + (uint64_t)sizeof(LSSeqInfoHeader) + info.payload_size;
   /* Part metadata would start inside the KV payload region. */
   header.offset_part_meta = expected_payload_end - 1u;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_offset_part_meta_too_small(const char *filename)
{
   LSSeqHeader     header;
   LSSeqInfoHeader info;
   FILE           *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic                  = LSSEQ_MAGIC;
   header.version                = LSSEQ_VERSION;
   header.flags                  = LSSEQ_FLAG_HAS_INFO;
   header.codec                  = (uint32_t)COMP_NONE;
   header.num_systems            = 1;
   header.num_parts              = 1;
   header.offset_part_blob_table = 500;
   /* Immediately after header: invalid (must be >= header + sizeof(info)). */
   header.offset_part_meta       = (uint64_t)sizeof(LSSeqHeader) + 4u;

   info.magic    = LSSEQ_INFO_MAGIC;
   info.version  = LSSEQ_INFO_VERSION;
   info.endian_tag = UINT32_C(0x01020304);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_truncated_part_metadata(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   char                part_stub[4] = {0};
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 0;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta     = header.offset_part_meta + sizeof(LSSeqPartMeta);
   header.offset_sys_part_meta    = header.offset_pattern_meta + sizeof(LSSeqPatternMeta);
   header.offset_part_blob_table  = header.offset_sys_part_meta + sizeof(LSSeqSystemPartMeta);
   header.offset_timestep_meta    = header.offset_part_blob_table +
                                   (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data        = header.offset_timestep_meta;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_stub, sizeof(part_stub), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_truncated_pattern_metadata(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   char                pat_stub[4] = {0};
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 0;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta     = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta    = header.offset_pattern_meta + sizeof(LSSeqPatternMeta);
   header.offset_part_blob_table  = header.offset_sys_part_meta + sizeof(LSSeqSystemPartMeta);
   header.offset_timestep_meta    = header.offset_part_blob_table +
                                   (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data        = header.offset_timestep_meta;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pat_stub, sizeof(pat_stub), 1, fp), 1);
   fclose(fp);
}

static void
write_lsseq_blob_table_offset_eof(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 0;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta     = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta    = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table  = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta    = header.offset_part_blob_table +
                                   (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data        = header.offset_timestep_meta;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   fclose(fp);
}

/* Valid v2 layout except system references pattern id out of range. */
static void
write_lsseq_pattern_id_oob(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id         = 5;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

static void
write_lsseq_pattern_part_mismatch(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 1;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

/* Batched values slice has c_size==0 but system requests non-zero decoded bytes. */
static void
write_lsseq_values_slice_zero_csize_nonzero_decomp(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   /* Values slot: zero-sized compressed chunk, but sys requests sizeof(vals0) decoded bytes. */
   part_blob_table[0] = pat_end;
   part_blob_table[1] = 0;
   part_blob_table[2] = pat_end;
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   /* No values payload: rhs/dof follow pattern blobs immediately. */
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

/* Decoded batched blob is shorter than requested slice (bounds check in LSSeqReadPartBlobSlice). */
static void
write_lsseq_values_slice_out_of_bounds(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   /* Only one double in values batch, but sys requests two (slice out of decoded buffer). */
   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(double);
   part_blob_table[2] = pat_end + sizeof(double);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(double) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = 2u * sizeof(double);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(double) + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   {
      double one = 10.0;
      ASSERT_EQ_SIZE(fwrite(&one, sizeof(one), 1, fp), 1);
   }
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

/* Pattern rows blob decodes to wrong size vs nnz*row_index_size (COMP_NONE path). */
static void
write_lsseq_rows_blob_decoded_size_mismatch(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[1] = {0};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = sizeof(dof0);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);

   fclose(fp);
}

#if defined(HYPREDRV_USING_ZLIB)
static void
write_lsseq_zlib_pattern_rows_cols(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   int32_t             dof0[2]  = {0, 1};
   void               *zrows = NULL, *zcols = NULL;
   void               *zvals = NULL, *zrhs = NULL, *zdof = NULL;
   size_t              zrows_sz = 0, zcols_sz = 0;
   size_t              zvals_sz = 0, zrhs_sz = 0, zdof_sz = 0;
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   hypredrv_ErrorCodeResetAll();
   hypredrv_compress(COMP_ZLIB, sizeof(rows0), rows0, &zrows_sz, &zrows, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(zrows);
   hypredrv_compress(COMP_ZLIB, sizeof(cols0), cols0, &zcols_sz, &zcols, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(zcols);
   hypredrv_compress(COMP_ZLIB, sizeof(vals0), vals0, &zvals_sz, &zvals, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(zvals);
   hypredrv_compress(COMP_ZLIB, sizeof(rhs0), rhs0, &zrhs_sz, &zrhs, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(zrhs);
   hypredrv_compress(COMP_ZLIB, sizeof(dof0), dof0, &zdof_sz, &zdof, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(zdof);

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_ZLIB;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = (uint64_t)zrows_sz;
   blob_offset += (uint64_t)zrows_sz;
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = (uint64_t)zcols_sz;
   blob_offset += (uint64_t)zcols_sz;

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = (uint64_t)zvals_sz;
   part_blob_table[2] = pat_end + (uint64_t)zvals_sz;
   part_blob_table[3] = (uint64_t)zrhs_sz;
   part_blob_table[4] = pat_end + (uint64_t)zvals_sz + (uint64_t)zrhs_sz;
   part_blob_table[5] = (uint64_t)zdof_sz;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + (uint64_t)zvals_sz + (uint64_t)zrhs_sz +
                 (uint64_t)zdof_sz;

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(zrows, zrows_sz, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(zcols, zcols_sz, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(zvals, zvals_sz, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(zrhs, zrhs_sz, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(zdof, zdof_sz, 1, fp), 1);

   free(zrows);
   free(zcols);
   free(zvals);
   free(zrhs);
   free(zdof);
   fclose(fp);
}
#endif /* HYPREDRV_USING_ZLIB */

static void
write_lsseq_info_no_dofmap_timesteps(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = 0;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = 0;
   sys_meta[0].dof_num_entries    = 0;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);

   fclose(fp);
}

static void
write_lsseq_timesteps_flag_zero_count(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 0;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta;

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = 0;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = 0;
   sys_meta[0].dof_num_entries    = 0;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0);

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);

   fclose(fp);
}

/* HAS_DOFMAP with dof_num_entries==0 (exercises dof_data==NULL branch in ReadDofmap). */
static void
write_lsseq_has_dofmap_zero_dof_entries(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   double              rhs0[2]  = {1.0, 2.0};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = sizeof(rhs0);
   part_blob_table[4] = pat_end + sizeof(vals0) + sizeof(rhs0);
   part_blob_table[5] = 0;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = 0;
   sys_meta[0].dof_num_entries    = 0;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0) + sizeof(rhs0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);

   fclose(fp);
}

/* pattern->nnz==0 and empty matrix blobs (COMP_NONE raw path for rows/cols/vals). */
static void
write_lsseq_zero_pattern_nnz(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 0;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = 0;
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = 0;

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = 0;
   part_blob_table[2] = pat_end;
   part_blob_table[3] = 0;
   part_blob_table[4] = pat_end;
   part_blob_table[5] = 0;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 0;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = 0;
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = 0;
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = 0;
   sys_meta[0].dof_num_entries    = 0;

   blob_offset = header.offset_blob_data + pat_end;

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   fclose(fp);
}

/* part->nrows==0 and empty RHS blob (LSSeqWriteRHSPartFile skips value write). */
static void
write_lsseq_zero_nrows_rhs(const char *filename)
{
   const char          payload[] = "k=v\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[1];
   LSSeqSystemPartMeta sys_meta[1];
   LSSeqTimestepEntry  timesteps[1];
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   double              vals0[2] = {10.0, 20.0};
   uint64_t            blob_offset;
   uint64_t            pat_end;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);
   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));
   memset(part_blob_table, 0, sizeof(part_blob_table));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 1;
   header.num_parts     = 1;
   header.num_patterns  = 1;
   header.num_timesteps = 1;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta = header.offset_part_blob_table +
                                 (uint64_t)(LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t));
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = fnv1a64(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 0;
   part_meta[0].nrows          = 0;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pat_end = blob_offset - header.offset_blob_data;

   part_blob_table[0] = pat_end;
   part_blob_table[1] = sizeof(vals0);
   part_blob_table[2] = pat_end + sizeof(vals0);
   part_blob_table[3] = 0;
   part_blob_table[4] = pat_end + sizeof(vals0);
   part_blob_table[5] = 0;

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = 0;
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = 0;
   sys_meta[0].dof_num_entries    = 0;

   blob_offset = header.offset_blob_data + pat_end + sizeof(vals0);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);

   fclose(fp);
}

static void
test_lsseq_summary_and_timesteps(void)
{
   const char *filename = "test_lsseq_summary.bin";
   int         num_systems = 0, num_patterns = 0, has_dofmap = 0, has_timesteps = 0;
   IntArray   *ids = NULL;
   IntArray   *starts = NULL;

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(
      hypredrv_LSSeqReadSummary(filename, &num_systems, &num_patterns, &has_dofmap, &has_timesteps));
   ASSERT_EQ(num_systems, 2);
   ASSERT_EQ(num_patterns, 2);
   ASSERT_TRUE(has_dofmap);
   ASSERT_TRUE(has_timesteps);

   {
      int ns_only = 0;
      hypredrv_ErrorCodeResetAll();
      ASSERT_TRUE(hypredrv_LSSeqReadSummary(filename, &ns_only, NULL, NULL, NULL));
      ASSERT_EQ(ns_only, 2);
   }

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadTimesteps(filename, &starts));
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)starts->size, 2);
   ASSERT_EQ(starts->data[0], 0);
   ASSERT_EQ(starts->data[1], 1);
   hypredrv_IntArrayDestroy(&starts);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadTimestepsWithIds(filename, &ids, &starts));
   ASSERT_NOT_NULL(ids);
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)ids->size, 2);
   ASSERT_EQ((int)starts->size, 2);
   ASSERT_EQ(ids->data[0], 0);
   ASSERT_EQ(ids->data[1], 1);
   ASSERT_EQ(starts->data[0], 0);
   ASSERT_EQ(starts->data[1], 1);
   hypredrv_IntArrayDestroy(&ids);
   hypredrv_IntArrayDestroy(&starts);
}

static void
test_lsseq_info_block(void)
{
   const char *filename = "test_lsseq_info.bin";
   char       *payload  = NULL;
   size_t      nbytes   = 0;
   HYPRE_IJMatrix mat0  = NULL;

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadInfo(filename, &payload, &nbytes));
   ASSERT_NOT_NULL(payload);
   ASSERT_TRUE(nbytes > 0);
   ASSERT_TRUE(strstr(payload, "foo=bar") != NULL);
   free(payload);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat0));
   ASSERT_NOT_NULL(mat0);
   HYPRE_IJMatrixDestroy(mat0);
}

static void
test_lsseq_matrix_rhs_dofmap(void)
{
   const char     *filename = "test_lsseq_data.bin";
   HYPRE_IJMatrix  mat0 = NULL;
   HYPRE_IJMatrix  mat1 = NULL;
   HYPRE_IJVector  rhs = NULL;
   IntArray       *dofmap = NULL;

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat0));
   ASSERT_NOT_NULL(mat0);
   ASSERT_EQ((int)hypredrv_LinearSystemMatrixGetNumNonzeros(mat0), 2);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadRHS(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &rhs));
   ASSERT_NOT_NULL(rhs);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 2);
   ASSERT_EQ(dofmap->data[0], 0);
   ASSERT_EQ(dofmap->data[1], 1);

   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat0);
   rhs = NULL;
   mat0 = NULL;

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 1, HYPRE_MEMORY_HOST, &mat1));
   ASSERT_NOT_NULL(mat1);
   ASSERT_EQ((int)hypredrv_LinearSystemMatrixGetNumNonzeros(mat1), 3);
   HYPRE_IJMatrixDestroy(mat1);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 1, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 2);
   ASSERT_EQ(dofmap->data[0], 1);
   ASSERT_EQ(dofmap->data[1], 1);
   hypredrv_IntArrayDestroy(&dofmap);
}

static void
test_lsseq_two_parts_row_order_sort(void)
{
   const char     *filename = "test_lsseq_twopart_sort.bin";
   HYPRE_IJMatrix  mat     = NULL;
   HYPRE_IJVector  rhs     = NULL;
   IntArray       *dofmap   = NULL;

   write_two_part_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NOT_NULL(mat);
   ASSERT_EQ((int)hypredrv_LinearSystemMatrixGetNumNonzeros(mat), 5);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadRHS(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &rhs));
   ASSERT_NOT_NULL(rhs);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 4);

   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
}

static void
test_lsseq_matrix_read_fails_empty_row_blob_expected_nonempty(void)
{
   const char    *filename = "test_lsseq_empty_row_blob.bin";
   HYPRE_IJMatrix mat       = NULL;

   write_lsseq_empty_row_blob_mismatch(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   ASSERT_NULL(mat);
}

static void
test_lsseq_requires_info_header(void)
{
   const char *filename = "test_lsseq_missing_info.bin";
   HYPRE_IJMatrix mat   = NULL;
   IntArray      *starts = NULL;

   write_test_container(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat));

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadTimesteps(filename, &starts));
}

static void
test_lsseq_rejects_oversized_info_payload(void)
{
   const char *filename = "test_lsseq_oversized_info.bin";
   char       *payload  = NULL;
   size_t      nbytes   = 0;

   write_lsseq_with_oversized_info_payload(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadInfo(filename, &payload, &nbytes));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_lsseq_rejects_excessive_part_count(void)
{
   const char *filename = "test_lsseq_excessive_counts.bin";
   int         num_systems = 0;

   write_lsseq_with_excessive_counts(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(filename, &num_systems, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_lsseq_rejects_bad_magic_and_version(void)
{
   const char *bad_magic = "test_lsseq_bad_magic.bin";
   const char *bad_ver   = "test_lsseq_bad_version.bin";
   int         ns        = 0;

   write_lsseq_bad_magic(bad_magic);
   add_temp_file(bad_magic);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(bad_magic, &ns, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_bad_version(bad_ver);
   add_temp_file(bad_ver);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(bad_ver, &ns, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_lsseq_truncated_file_and_missing_path(void)
{
   const char *trunc = "test_lsseq_trunc.bin";

   write_lsseq_truncated_header(trunc);
   add_temp_file(trunc);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(trunc, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(
      hypredrv_LSSeqReadSummary("/tmp/hypredrive_lsseq_nonexistent_path_12345.bin", NULL, NULL,
                                NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_lsseq_validate_header_branches(void)
{
   const char *bad_codec = "test_lsseq_bad_codec.bin";
   const char *zero_sys  = "test_lsseq_zero_sys.bin";
   const char *zero_blob = "test_lsseq_zero_blob.bin";

   write_lsseq_invalid_codec(bad_codec);
   add_temp_file(bad_codec);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(bad_codec, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_zero_systems(zero_sys);
   add_temp_file(zero_sys);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(zero_sys, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_zero_blob_table_offset(zero_blob);
   add_temp_file(zero_blob);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(zero_blob, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_lsseq_rejects_missing_info_flag_on_load(void)
{
   const char *filename = "test_lsseq_no_info_flag.bin";
   HYPRE_IJMatrix mat = NULL;

   write_lsseq_missing_info_flag(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   ASSERT_NULL(mat);
}

static void
test_lsseq_read_info_error_paths(void)
{
   char   *payload = NULL;
   size_t  nbytes  = 0;

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadInfo(NULL, &payload, &nbytes));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadInfo("unused.bin", NULL, &nbytes));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadInfo("unused.bin", &payload, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   {
      const char *fn = "test_lsseq_hdr_only_readinfo.bin";
      write_lsseq_header_only(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
      ASSERT_NULL(payload);
   }

   {
      const char *fn = "test_lsseq_readinfo_flag_missing.bin";
      write_lsseq_readinfo_no_has_info_flag(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }

   {
      const char *fn = "test_lsseq_readinfo_bad_magic.bin";
      write_lsseq_readinfo_bad_info_magic(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }

   {
      const char *fn = "test_lsseq_readinfo_bad_endian.bin";
      write_lsseq_readinfo_bad_info_endian(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }

   {
      const char *fn = "test_lsseq_readinfo_bad_hash.bin";
      write_lsseq_readinfo_hash_mismatch(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }
}

static void
test_lsseq_public_api_null_pointers_and_bad_ls_id(void)
{
   const char *filename = "test_lsseq_api_guards.bin";

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadRHS(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 0, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadTimestepsWithIds(filename, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   {
      HYPRE_IJMatrix mat = NULL;
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(
         hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 99, HYPRE_MEMORY_HOST, &mat));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(mat);
   }

   {
      HYPRE_IJVector rhs = NULL;
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(
         hypredrv_LSSeqReadRHS(MPI_COMM_SELF, filename, 99, HYPRE_MEMORY_HOST, &rhs));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(rhs);
   }

   {
      IntArray *dofmap = NULL;
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 99, &dofmap));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(dofmap);
   }
}

static void
test_lsseq_summary_missing_filename(void)
{
   int ns = 0;

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(NULL, &ns, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
}

static void
test_lsseq_header_shape_and_dimension_limits(void)
{
   const char *zp = "test_lsseq_np0.bin";
   const char *pat = "test_lsseq_patlim.bin";
   const char *ts = "test_lsseq_tslim.bin";

   write_lsseq_num_parts_zero(zp);
   add_temp_file(zp);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(zp, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_excessive_patterns(pat);
   add_temp_file(pat);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(pat, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_excessive_timesteps(ts);
   add_temp_file(ts);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadSummary(ts, NULL, NULL, NULL, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_lsseq_data_load_info_layout_and_truncation(void)
{
   const char    *badver = "test_lsseq_load_bad_info_ver.bin";
   const char    *overlap = "test_lsseq_overlap.bin";
   const char    *early = "test_lsseq_early_pm.bin";
   const char    *tpart = "test_lsseq_trunc_part.bin";
   const char    *tpat = "test_lsseq_trunc_pat.bin";
   const char    *tblob = "test_lsseq_trunc_blobtbl.bin";
   HYPRE_IJMatrix mat    = NULL;

   write_lsseq_info_bad_version(badver);
   add_temp_file(badver);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, badver, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_payload_overlaps_part_meta(overlap);
   add_temp_file(overlap);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, overlap, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_offset_part_meta_too_small(early);
   add_temp_file(early);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, early, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_truncated_part_metadata(tpart);
   add_temp_file(tpart);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, tpart, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_truncated_pattern_metadata(tpat);
   add_temp_file(tpat);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, tpat, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   write_lsseq_blob_table_offset_eof(tblob);
   add_temp_file(tblob);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, tblob, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   {
      char       *payload = NULL;
      size_t      nbytes  = 0;
      const char *fn      = "test_lsseq_readinfo_badver.bin";
      write_lsseq_info_bad_version(fn);
      add_temp_file(fn);
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(hypredrv_LSSeqReadInfo(fn, &payload, &nbytes));
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
      ASSERT_NULL(payload);
   }
}

static void
test_lsseq_matrix_pattern_and_blob_errors(void)
{
   const char *oob = "test_lsseq_pat_oob.bin";
   const char *mismatch = "test_lsseq_pat_part.bin";
   const char *z0 = "test_lsseq_vals_z0.bin";
   const char *bounds = "test_lsseq_vals_bounds.bin";
   const char *rowsz = "test_lsseq_rowsz.bin";
   HYPRE_IJMatrix mat = NULL;

   write_lsseq_pattern_id_oob(oob);
   add_temp_file(oob);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, oob, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);

   write_lsseq_pattern_part_mismatch(mismatch);
   add_temp_file(mismatch);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, mismatch, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);

   write_lsseq_values_slice_zero_csize_nonzero_decomp(z0);
   add_temp_file(z0);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, z0, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);

   write_lsseq_values_slice_out_of_bounds(bounds);
   add_temp_file(bounds);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, bounds, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);

   write_lsseq_rows_blob_decoded_size_mismatch(rowsz);
   add_temp_file(rowsz);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, rowsz, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
}

#if defined(HYPREDRV_USING_ZLIB)
static void
test_lsseq_zlib_matrix_roundtrip(void)
{
   const char    *filename = "test_lsseq_zlib_matrix.bin";
   HYPRE_IJMatrix mat       = NULL;

   write_lsseq_zlib_pattern_rows_cols(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NOT_NULL(mat);
   ASSERT_EQ((int)hypredrv_LinearSystemMatrixGetNumNonzeros(mat), 2);
   HYPRE_IJMatrixDestroy(mat);
}
#endif

static void
test_lsseq_dofmap_no_flag_and_timestep_modes(void)
{
   const char *fn_nd = "test_lsseq_no_dofmap.bin";
   const char *fn_zt = "test_lsseq_zero_ts.bin";
   const char *fn_nt = "test_lsseq_no_tsflag.bin";
   IntArray      *starts = NULL;
   IntArray      *ids    = NULL;
   IntArray      *dofmap = NULL;
   HYPRE_IJMatrix mat    = NULL;

   write_lsseq_info_no_dofmap_timesteps(fn_nd);
   add_temp_file(fn_nd);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, fn_nd, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 0);
   hypredrv_IntArrayDestroy(&dofmap);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, fn_nd, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NOT_NULL(mat);
   HYPRE_IJMatrixDestroy(mat);
   mat = NULL;

   write_lsseq_timesteps_flag_zero_count(fn_zt);
   add_temp_file(fn_zt);
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadTimestepsWithIds(fn_zt, &ids, &starts));
   ASSERT_NULL(ids);
   ASSERT_NULL(starts);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadTimesteps(fn_zt, &starts));
   ASSERT_NULL(starts);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadTimestepsWithIds(fn_nd, NULL, &starts));
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)starts->size, 1);
   ASSERT_EQ(starts->data[0], 0);
   hypredrv_IntArrayDestroy(&starts);

   write_test_container_with_info(fn_nt);
   add_temp_file(fn_nt);
   /* Strip timestep flag: rewrite header in-place (same layout). */
   {
      FILE       *fp = fopen(fn_nt, "r+b");
      LSSeqHeader header;
      ASSERT_NOT_NULL(fp);
      ASSERT_EQ_SIZE(fread(&header, sizeof(header), 1, fp), 1);
      header.flags &= ~(unsigned int)LSSEQ_FLAG_HAS_TIMESTEPS;
      ASSERT_EQ(fseek(fp, 0, SEEK_SET), 0);
      ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
      fclose(fp);
   }
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadTimestepsWithIds(fn_nt, NULL, &starts));
   ASSERT_NULL(starts);
}

static void
test_lsseq_timesteps_read_optional_ids_null(void)
{
   const char *filename = "test_lsseq_ts_null_ids.bin";
   IntArray      *starts = NULL;

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadTimestepsWithIds(filename, NULL, &starts));
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)starts->size, 2);
   hypredrv_IntArrayDestroy(&starts);
}

static void
test_lsseq_read_missing_file_paths(void)
{
   const char    *gone = "/tmp/hypredrive_lsseq_missing_read_paths_998877.bin";
   char            *payload = NULL;
   size_t           nbytes  = 0;
   HYPRE_IJMatrix   mat     = NULL;
   HYPRE_IJVector   rhs     = NULL;
   IntArray        *dofmap  = NULL;

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadInfo(gone, &payload, &nbytes));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, gone, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadRHS(MPI_COMM_SELF, gone, 0, HYPRE_MEMORY_HOST, &rhs));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, gone, 0, &dofmap));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_lsseq_timesteps_replace_existing_arrays(void)
{
   const char *filename = "test_lsseq_ts_replace.bin";
   IntArray     *ids     = hypredrv_IntArrayCreate(3);
   IntArray     *starts  = hypredrv_IntArrayCreate(1);

   ASSERT_NOT_NULL(ids);
   ASSERT_NOT_NULL(starts);
   ids->data[0]    = -1;
   starts->data[0] = -1;

   write_test_container_with_info(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadTimestepsWithIds(filename, &ids, &starts));
   ASSERT_NOT_NULL(ids);
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)ids->size, 2);
   ASSERT_EQ((int)starts->size, 2);
   ASSERT_EQ(ids->data[0], 0);
   ASSERT_EQ(ids->data[1], 1);
   hypredrv_IntArrayDestroy(&ids);
   hypredrv_IntArrayDestroy(&starts);
}

static void
test_lsseq_dofmap_zero_entries_with_flag(void)
{
   const char    *filename = "test_lsseq_dof_zero_entries.bin";
   IntArray        *dofmap = NULL;

   write_lsseq_has_dofmap_zero_dof_entries(filename);
   add_temp_file(filename);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, filename, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 0);
   hypredrv_IntArrayDestroy(&dofmap);
}

static void
test_lsseq_read_matrix_zero_nnz_and_rhs_zero_nrows(void)
{
   const char    *fn_z = "test_lsseq_zero_nnz.bin";
   const char    *fn_r = "test_lsseq_zero_nrows_rhs.bin";
   HYPRE_IJMatrix mat  = NULL;
   HYPRE_IJVector rhs  = NULL;

   write_lsseq_zero_pattern_nnz(fn_z);
   add_temp_file(fn_z);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, fn_z, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NOT_NULL(mat);
   ASSERT_EQ((int)hypredrv_LinearSystemMatrixGetNumNonzeros(mat), 0);
   HYPRE_IJMatrixDestroy(mat);

   write_lsseq_zero_nrows_rhs(fn_r);
   add_temp_file(fn_r);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadRHS(MPI_COMM_SELF, fn_r, 0, HYPRE_MEMORY_HOST, &rhs));
   ASSERT_NOT_NULL(rhs);
   HYPRE_IJVectorDestroy(rhs);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   RUN_TEST(test_lsseq_summary_and_timesteps);
   RUN_TEST(test_lsseq_info_block);
   RUN_TEST(test_lsseq_matrix_rhs_dofmap);
   RUN_TEST(test_lsseq_matrix_read_fails_empty_row_blob_expected_nonempty);
   RUN_TEST(test_lsseq_two_parts_row_order_sort);
   RUN_TEST(test_lsseq_requires_info_header);
   RUN_TEST(test_lsseq_rejects_oversized_info_payload);
   RUN_TEST(test_lsseq_rejects_excessive_part_count);
   RUN_TEST(test_lsseq_rejects_bad_magic_and_version);
   RUN_TEST(test_lsseq_truncated_file_and_missing_path);
   RUN_TEST(test_lsseq_validate_header_branches);
   RUN_TEST(test_lsseq_rejects_missing_info_flag_on_load);
   RUN_TEST(test_lsseq_read_info_error_paths);
   RUN_TEST(test_lsseq_public_api_null_pointers_and_bad_ls_id);
   RUN_TEST(test_lsseq_summary_missing_filename);
   RUN_TEST(test_lsseq_header_shape_and_dimension_limits);
   RUN_TEST(test_lsseq_data_load_info_layout_and_truncation);
   RUN_TEST(test_lsseq_matrix_pattern_and_blob_errors);
#if defined(HYPREDRV_USING_ZLIB)
   RUN_TEST(test_lsseq_zlib_matrix_roundtrip);
#endif
   RUN_TEST(test_lsseq_dofmap_no_flag_and_timestep_modes);
   RUN_TEST(test_lsseq_timesteps_read_optional_ids_null);
   RUN_TEST(test_lsseq_read_missing_file_paths);
   RUN_TEST(test_lsseq_timesteps_replace_existing_arrays);
   RUN_TEST(test_lsseq_dofmap_zero_entries_with_flag);
   RUN_TEST(test_lsseq_read_matrix_zero_nnz_and_rhs_zero_nrows);

   cleanup_temp_files();

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
