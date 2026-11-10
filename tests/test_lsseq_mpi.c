/******************************************************************************
 * MPI coverage for LSSeq part distribution and collective read failures.
 ******************************************************************************/

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

/* Share rank 0's scratch path so ranks need not share a working directory. */
static void
lsseq_mpi_testpath(char *buf, size_t buflen, const char *basename)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (!rank)
   {
      memset(buf, 0, buflen);
      test_scratch_path(buf, buflen, basename);
   }
   MPI_Bcast(buf, (int)buflen, MPI_CHAR, 0, MPI_COMM_WORLD);
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

/* Minimal valid 1-part sequence (same shape as write_lsseq_zero_pattern_nnz in
 * test_lsseq.c). */
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
   header.offset_pattern_meta    = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta   = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta   = header.offset_part_blob_table +
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

/* Store one diagonal entry per part, with parts in reverse row order. */
static void
lsseq_mpi_write_parts_with_info(const char *filename, uint32_t nparts)
{
   const char           payload[] = "test=mpi\n";
   LSSeqHeader          header    = {0};
   LSSeqInfoHeader      info      = {0};
   LSSeqPartMeta       *parts     = calloc(nparts, sizeof(*parts));
   LSSeqPatternMeta    *patterns  = calloc(nparts, sizeof(*patterns));
   LSSeqSystemPartMeta *systems   = calloc(nparts, sizeof(*systems));
   uint64_t            *blobs = calloc(nparts * LSSEQ_PART_BLOB_ENTRIES, sizeof(*blobs));
   ASSERT_NOT_NULL(parts);
   ASSERT_NOT_NULL(patterns);
   ASSERT_NOT_NULL(systems);
   ASSERT_NOT_NULL(blobs);

   header.magic                = LSSEQ_MAGIC;
   header.version              = LSSEQ_VERSION;
   header.flags                = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP;
   header.codec                = COMP_NONE;
   header.num_systems          = 1;
   header.num_parts            = nparts;
   header.num_patterns         = nparts;
   header.offset_part_meta     = sizeof(header) + sizeof(info) + sizeof(payload) - 1;
   header.offset_pattern_meta  = header.offset_part_meta + nparts * sizeof(*parts);
   header.offset_sys_part_meta = header.offset_pattern_meta + nparts * sizeof(*patterns);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + nparts * sizeof(*systems);
   header.offset_blob_data =
      header.offset_part_blob_table + nparts * LSSEQ_PART_BLOB_ENTRIES * sizeof(*blobs);
   uint64_t offset = header.offset_blob_data;
   for (uint32_t p = 0; p < nparts; p++)
   {
      uint64_t row = nparts - p - 1;
      parts[p]     = (LSSeqPartMeta){row, row, 1, sizeof(HYPRE_BigInt), sizeof(double)};
      patterns[p].part_id          = p;
      patterns[p].nnz              = 1;
      patterns[p].rows_blob_offset = offset;
      patterns[p].rows_blob_size   = sizeof(HYPRE_BigInt);
      offset += sizeof(HYPRE_BigInt);
      patterns[p].cols_blob_offset = offset;
      patterns[p].cols_blob_size   = sizeof(HYPRE_BigInt);
      offset += sizeof(HYPRE_BigInt);
      uint64_t *part_blobs = blobs + p * LSSEQ_PART_BLOB_ENTRIES;
      for (int slot = 0; slot < 3; slot++)
      {
         part_blobs[slot * 2]     = offset - header.offset_blob_data;
         part_blobs[slot * 2 + 1] = slot == 2 ? sizeof(int32_t) : sizeof(double);
         offset += part_blobs[slot * 2 + 1];
      }
      systems[p].pattern_id       = p;
      systems[p].nnz              = 1;
      systems[p].values_blob_size = sizeof(double);
      systems[p].rhs_blob_size    = sizeof(double);
      systems[p].dof_blob_size    = sizeof(int32_t);
      systems[p].dof_num_entries  = 1;
   }
   info.magic                = LSSEQ_INFO_MAGIC;
   info.version              = LSSEQ_INFO_VERSION;
   info.flags                = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag           = UINT32_C(0x01020304);
   info.payload_size         = sizeof(payload) - 1;
   info.payload_hash_fnv1a64 = fnv1a64_mpi(payload, sizeof(payload) - 1);
   info.blob_bytes           = offset - header.offset_blob_data;

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(parts, sizeof(*parts), nparts, fp), nparts);
   ASSERT_EQ_SIZE(fwrite(patterns, sizeof(*patterns), nparts, fp), nparts);
   ASSERT_EQ_SIZE(fwrite(systems, sizeof(*systems), nparts, fp), nparts);
   ASSERT_EQ_SIZE(fwrite(blobs, sizeof(*blobs), nparts * LSSEQ_PART_BLOB_ENTRIES, fp),
                  nparts * LSSEQ_PART_BLOB_ENTRIES);
   for (uint32_t p = 0; p < nparts; p++)
   {
      HYPRE_BigInt row   = (HYPRE_BigInt)(nparts - p - 1);
      double       value = 10.0 + row, rhs = 20.0 + row;
      int32_t      dof = (int32_t)row;
      ASSERT_EQ_SIZE(fwrite(&row, sizeof(row), 1, fp), 1);
      ASSERT_EQ_SIZE(fwrite(&row, sizeof(row), 1, fp), 1);
      ASSERT_EQ_SIZE(fwrite(&value, sizeof(value), 1, fp), 1);
      ASSERT_EQ_SIZE(fwrite(&rhs, sizeof(rhs), 1, fp), 1);
      ASSERT_EQ_SIZE(fwrite(&dof, sizeof(dof), 1, fp), 1);
   }
   ASSERT_EQ(fclose(fp), 0);
   free(blobs);
   free(systems);
   free(patterns);
   free(parts);
}

static void
test_lsseq_mpi_one_part_with_two_ranks_fails(void)
{
   int            myid = 0;
   char           path[4096];
   HYPRE_IJMatrix mat = NULL;

   lsseq_mpi_testpath(path, sizeof(path), "test_lsseq_mpi_1part.bin");

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (myid == 0)
   {
      lsseq_mpi_write_one_part_zero_nnz(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(
      hypredrv_LSSeqReadMatrix(MPI_COMM_WORLD, path, 0, HYPRE_MEMORY_HOST, &mat));
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   if (myid == 0)
   {
      (void)remove(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

static void
test_lsseq_mpi_part_distribution(void)
{
   int rank = 0, nprocs = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   char path[4096];
   lsseq_mpi_testpath(path, sizeof(path), "test_lsseq_mpi_parts.bin");
   for (int extra = 0; extra < 2; extra++)
   {
      if (!rank)
      {
         lsseq_mpi_write_parts_with_info(path, (uint32_t)(nprocs + extra));
      }
      MPI_Barrier(MPI_COMM_WORLD);
      HYPRE_IJMatrix matrix = NULL;
      HYPRE_IJVector rhs    = NULL;
      IntArray      *dofmap = NULL;
      hypredrv_ErrorCodeResetAll();
      ASSERT_TRUE(
         hypredrv_LSSeqReadMatrix(MPI_COMM_WORLD, path, 0, HYPRE_MEMORY_HOST, &matrix));
      ASSERT_TRUE(
         hypredrv_LSSeqReadRHS(MPI_COMM_WORLD, path, 0, HYPRE_MEMORY_HOST, &rhs));
      ASSERT_TRUE(hypredrv_LSSeqReadDofmap(MPI_COMM_WORLD, path, 0, &dofmap));
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_BigInt lower = 0, upper = 0, clower = 0, cupper = 0;
      ASSERT_EQ(HYPRE_IJMatrixGetLocalRange(matrix, &lower, &upper, &clower, &cupper), 0);
      int expected_rows = !rank && extra ? 2 : 1;
      ASSERT_EQ(lower, rank ? rank + extra : 0);
      ASSERT_EQ(upper - lower + 1, expected_rows);
      ASSERT_EQ(dofmap->size, expected_rows);
      for (int i = 0; i < expected_rows; i++)
      {
         HYPRE_BigInt  row   = lower + i;
         HYPRE_Int     ncols = 1;
         HYPRE_Complex value = 0;
         ASSERT_EQ(HYPRE_IJMatrixGetValues(matrix, 1, &ncols, &row, &row, &value), 0);
         ASSERT_EQ_DOUBLE((double)value, 10.0 + row, 1.0e-12);
         ASSERT_EQ(HYPRE_IJVectorGetValues(rhs, 1, &row, &value), 0);
         ASSERT_EQ_DOUBLE((double)value, 20.0 + row, 1.0e-12);
         ASSERT_EQ(dofmap->data[i], row);
      }
      HYPRE_IJMatrixDestroy(matrix);
      HYPRE_IJVectorDestroy(rhs);
      hypredrv_IntArrayDestroy(&dofmap);
      MPI_Barrier(MPI_COMM_WORLD);
   }
}

/* Failure before a builder and failure during payload import must reach every
 * rank, with all partial objects released. */
static void
test_lsseq_mpi_rank_local_failures(void)
{
   int rank = 0, nprocs = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   char path[4096], missing[4096];
   lsseq_mpi_testpath(path, sizeof(path), "test_lsseq_mpi_failure.bin");
   lsseq_mpi_testpath(missing, sizeof(missing), "missing.bin");
   for (int phase = 0; phase < 3; phase++)
   {
      if (!rank)
      {
         lsseq_mpi_write_parts_with_info(path, (uint32_t)nprocs);
         if (phase > 0)
         {
            FILE       *fp = fopen(path, "r+b");
            LSSeqHeader header;
            ASSERT_NOT_NULL(fp);
            ASSERT_EQ_SIZE(fread(&header, sizeof(header), 1, fp), 1);
            /* Stored part 0 belongs to the last rank after row ordering. */
            if (phase == 1)
            {
               LSSeqPartMeta part;
               ASSERT_EQ(fseek(fp, (long)header.offset_part_meta, SEEK_SET), 0);
               ASSERT_EQ_SIZE(fread(&part, sizeof(part), 1, fp), 1);
               part.value_size = 3; /* invalid matrix/RHS metadata */
               ASSERT_EQ(fseek(fp, (long)header.offset_part_meta, SEEK_SET), 0);
               ASSERT_EQ_SIZE(fwrite(&part, sizeof(part), 1, fp), 1);
            }
            else
            {
               uint64_t blobs[LSSEQ_PART_BLOB_ENTRIES] = {0};
               /* Valid metadata with missing nonempty values/RHS/dof blobs. */
               ASSERT_EQ(fseek(fp, (long)header.offset_part_blob_table, SEEK_SET), 0);
               ASSERT_EQ_SIZE(fwrite(blobs, sizeof(blobs), 1, fp), 1);
            }
            ASSERT_EQ(fclose(fp), 0);
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const char    *input  = phase == 0 && rank == nprocs - 1 ? missing : path;
      HYPRE_IJMatrix matrix = NULL;
      HYPRE_IJVector rhs    = NULL;
      IntArray      *dofmap = NULL;
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(
         hypredrv_LSSeqReadMatrix(MPI_COMM_WORLD, input, 0, HYPRE_MEMORY_HOST, &matrix));
      ASSERT_NULL(matrix);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_ErrorCodeResetAll();
      ASSERT_FALSE(
         hypredrv_LSSeqReadRHS(MPI_COMM_WORLD, input, 0, HYPRE_MEMORY_HOST, &rhs));
      ASSERT_NULL(rhs);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      if (phase != 1) /* Dofmap decoding does not use the coefficient dtype. */
      {
         hypredrv_ErrorCodeResetAll();
         ASSERT_FALSE(hypredrv_LSSeqReadDofmap(MPI_COMM_WORLD, input, 0, &dofmap));
         ASSERT_NULL(dofmap);
         ASSERT_TRUE(hypredrv_ErrorCodeActive());
      }
      hypredrv_ErrorCodeResetAll();
      HYPRE_ClearAllErrors();
      MPI_Barrier(MPI_COMM_WORLD);
   }
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
   RUN_TEST(test_lsseq_mpi_part_distribution);
   RUN_TEST(test_lsseq_mpi_rank_local_failures);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
