#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "linsys.h"
#include "lsseq.h"
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

   ASSERT_EQ(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ(fwrite(dof1, sizeof(dof1), 1, fp), 1);

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
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_timestep_meta = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_blob_data     = header.offset_timestep_meta + sizeof(timesteps);

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

   info.blob_bytes = blob_offset - header.offset_blob_data;

   ASSERT_EQ(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ(fwrite(dof1, sizeof(dof1), 1, fp), 1);

   fclose(fp);
}

static void
test_lsseq_summary_and_timesteps(void)
{
   const char *filename = "test_lsseq_summary.bin";
   int         num_systems = 0, num_patterns = 0, has_dofmap = 0, has_timesteps = 0;
   IntArray   *starts = NULL;

   write_test_container(filename);
   add_temp_file(filename);

   ErrorCodeResetAll();
   ASSERT_TRUE(
      LSSeqReadSummary(filename, &num_systems, &num_patterns, &has_dofmap, &has_timesteps));
   ASSERT_EQ(num_systems, 2);
   ASSERT_EQ(num_patterns, 2);
   ASSERT_TRUE(has_dofmap);
   ASSERT_TRUE(has_timesteps);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadTimesteps(filename, &starts));
   ASSERT_NOT_NULL(starts);
   ASSERT_EQ((int)starts->size, 2);
   ASSERT_EQ(starts->data[0], 0);
   ASSERT_EQ(starts->data[1], 1);
   IntArrayDestroy(&starts);
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

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadInfo(filename, &payload, &nbytes));
   ASSERT_NOT_NULL(payload);
   ASSERT_TRUE(nbytes > 0);
   ASSERT_TRUE(strstr(payload, "foo=bar") != NULL);
   free(payload);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat0));
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

   write_test_container(filename);
   add_temp_file(filename);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadMatrix(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &mat0));
   ASSERT_NOT_NULL(mat0);
   ASSERT_EQ((int)LinearSystemMatrixGetNumNonzeros(mat0), 2);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadRHS(MPI_COMM_SELF, filename, 0, HYPRE_MEMORY_HOST, &rhs));
   ASSERT_NOT_NULL(rhs);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadDofmap(MPI_COMM_SELF, filename, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 2);
   ASSERT_EQ(dofmap->data[0], 0);
   ASSERT_EQ(dofmap->data[1], 1);

   IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat0);
   rhs = NULL;
   mat0 = NULL;

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadMatrix(MPI_COMM_SELF, filename, 1, HYPRE_MEMORY_HOST, &mat1));
   ASSERT_NOT_NULL(mat1);
   ASSERT_EQ((int)LinearSystemMatrixGetNumNonzeros(mat1), 3);
   HYPRE_IJMatrixDestroy(mat1);

   ErrorCodeResetAll();
   ASSERT_TRUE(LSSeqReadDofmap(MPI_COMM_SELF, filename, 1, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_EQ((int)dofmap->size, 2);
   ASSERT_EQ(dofmap->data[0], 1);
   ASSERT_EQ(dofmap->data[1], 1);
   IntArrayDestroy(&dofmap);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   RUN_TEST(test_lsseq_summary_and_timesteps);
   RUN_TEST(test_lsseq_info_block);
   RUN_TEST(test_lsseq_matrix_rhs_dofmap);

   cleanup_temp_files();

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
