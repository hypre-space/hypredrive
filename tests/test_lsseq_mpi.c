/******************************************************************************
 * MPI coverage for hypredrv_LSSeq* (LSSeqLocalPartIDs, shared temp dirs, barriers).
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

#include "internal/comp.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "test_helpers.h"

/* Absolute paths under the repo root: some MPI launchers (notably on macOS) do not
 * guarantee every rank shares the same current working directory as rank 0. */
static void
lsseq_mpi_testpath(char *buf, size_t buflen, const char *basename)
{
   int n = snprintf(buf, buflen, "%s/%s", HYPREDRIVE_SOURCE_DIR, basename);
   ASSERT_TRUE(n > 0 && (size_t)n < buflen);
}

static uint64_t
fnv1a64_mpi(const void *data, size_t nbytes)
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

/* Minimal valid 1-part sequence (same shape as write_lsseq_zero_pattern_nnz in test_lsseq.c). */
static void
lsseq_mpi_write_one_part_zero_nnz(const char *filename)
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
   info.payload_hash_fnv1a64 = fnv1a64_mpi(payload, sizeof(payload) - 1u);
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

static void
lsseq_mpi_write_two_part_with_info(const char *filename)
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
   info.payload_hash_fnv1a64 = fnv1a64_mpi(payload, sizeof(payload) - 1u);
   info.blob_hash_fnv1a64    = 0;

   part_meta[0].row_lower      = 100;
   part_meta[0].row_upper      = 101;
   part_meta[0].nrows          = 2;
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

   pat_end     = blob_offset - header.offset_blob_data;
   p0_chunk    = sizeof(vals0) + sizeof(rhs0) + sizeof(dof0);
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

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = 0;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = 0;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = 0;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;

   sys_meta[1].pattern_id         = 1;
   sys_meta[1].nnz                = 3;
   sys_meta[1].values_blob_offset = 0;
   sys_meta[1].values_blob_size   = sizeof(vals1);
   sys_meta[1].rhs_blob_offset    = 0;
   sys_meta[1].rhs_blob_size      = sizeof(rhs1);
   sys_meta[1].dof_blob_offset    = 0;
   sys_meta[1].dof_blob_size      = sizeof(dof1);
   sys_meta[1].dof_num_entries    = 2;

   blob_offset = header.offset_blob_data + pat_end + p0_chunk + sizeof(vals1) + sizeof(rhs1) +
                 sizeof(dof1);

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
test_lsseq_mpi_one_part_with_two_ranks_fails(void)
{
   int           myid = 0;
   char          path[4096];
   HYPRE_IJMatrix mat = NULL;
   uint32_t      local_code = 0;
   uint32_t      global_code = 0;

   lsseq_mpi_testpath(path, sizeof(path), "test_lsseq_mpi_1part.bin");

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (myid == 0)
   {
      lsseq_mpi_write_one_part_zero_nnz(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_LSSeqReadMatrix(MPI_COMM_WORLD, path, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   local_code = hypredrv_ErrorCodeGet();
   MPI_Allreduce(&local_code, &global_code, 1, MPI_UINT32_T, MPI_BOR, MPI_COMM_WORLD);
   ASSERT_TRUE(global_code & ERROR_FILE_UNEXPECTED_ENTRY);

   if (myid == 0)
   {
      (void)remove(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

static void
test_lsseq_mpi_two_part_dofmap_and_matrix(void)
{
   int         myid = 0;
   char        path[4096];
   HYPRE_IJMatrix mat = NULL;
   IntArray    *dofmap = NULL;

   lsseq_mpi_testpath(path, sizeof(path), "test_lsseq_mpi_2part.bin");

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (myid == 0)
   {
      lsseq_mpi_write_two_part_with_info(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadMatrix(MPI_COMM_WORLD, path, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   HYPRE_IJMatrixDestroy(mat);
   mat = NULL;

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_WORLD, path, 0, &dofmap));
   ASSERT_NOT_NULL(dofmap);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_IntArrayDestroy(&dofmap);

   if (myid == 0)
   {
      (void)remove(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   int nprocs = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs < 2)
   {
      fprintf(stderr, "SKIP: test_lsseq_mpi requires at least 2 MPI ranks\n");
      TEST_HYPRE_FINALIZE();
      MPI_Finalize();
      return 77;
   }

   RUN_TEST(test_lsseq_mpi_one_part_with_two_ranks_fails);
   RUN_TEST(test_lsseq_mpi_two_part_dofmap_and_matrix);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
