/******************************************************************************
 * MPI coverage for hypredrv_IJMatrixReadMultipartBinary (rank/part branches).
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "test_helpers.h"

static void
write_matrix_part_file(const char *prefix, int part_id, HYPRE_BigInt row_lower,
                       HYPRE_BigInt row_upper, size_t nnz, const HYPRE_BigInt *rows,
                       const HYPRE_BigInt *cols, const double *vals)
{
   char filename[256];
   snprintf(filename, sizeof(filename), "%s.%05d.bin", prefix, part_id);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[5]           = (uint64_t)(row_upper - row_lower + 1);
   header[6]           = (uint64_t)nnz;
   header[7]           = (uint64_t)row_lower;
   header[8]           = (uint64_t)row_upper;

   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   if (nnz > 0)
   {
      ASSERT_EQ(fwrite(rows, sizeof(HYPRE_BigInt), nnz, fp), nnz);
      ASSERT_EQ(fwrite(cols, sizeof(HYPRE_BigInt), nnz, fp), nnz);
      ASSERT_EQ(fwrite(vals, sizeof(double), nnz, fp), nnz);
   }
   fclose(fp);
   add_temp_file(filename);
}

static void
test_gnparts_lt_nprocs(void)
{
   int nprocs;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs < 2)
   {
      return;
   }

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary("test_matrix_mpi_noprefix", MPI_COMM_WORLD, 1,
                                        HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_two_ranks_offdiagonal_and_multipart_distribution(void)
{
   int nprocs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (nprocs < 2)
   {
      return;
   }

   /* g_nparts=2: rank0 -> part 0, rank1 -> part 1; entry (0,1) uses offd on rank0 */
   const char *prefix_offd = "test_matrix_mpi_offd";
   if (myid == 0)
   {
      HYPRE_BigInt r0[1] = {0};
      HYPRE_BigInt c0[1] = {1};
      double       v0[1] = {2.0};
      write_matrix_part_file(prefix_offd, 0, 0, 0, 1, r0, c0, v0);

      HYPRE_BigInt r1[1] = {1};
      HYPRE_BigInt c1[1] = {1};
      double       v1[1] = {3.0};
      write_matrix_part_file(prefix_offd, 1, 1, 1, 1, r1, c1, v1);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   {
      HYPRE_IJMatrix mat = NULL;
      void          *obj = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJMatrixReadMultipartBinary(prefix_offd, MPI_COMM_WORLD, 2,
                                           HYPRE_MEMORY_HOST, &mat);
      ASSERT_NOT_NULL(mat);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_IJMatrixGetObject(mat, &obj);
      ASSERT_NOT_NULL(obj);
      HYPRE_IJMatrixDestroy(mat);
   }
   if (myid == 0)
   {
      cleanup_temp_files();
   }
   MPI_Barrier(MPI_COMM_WORLD);

   /* g_nparts=3 on 2 ranks: rank0 parts {0,1}, rank1 part {2} */
   const char *prefix_three = "test_matrix_mpi_gp3";
   if (myid == 0)
   {
      write_matrix_part_file(prefix_three, 0, 0, 0, 0, NULL, NULL, NULL);
      write_matrix_part_file(prefix_three, 1, 1, 1, 0, NULL, NULL, NULL);
      write_matrix_part_file(prefix_three, 2, 2, 2, 0, NULL, NULL, NULL);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   {
      HYPRE_IJMatrix mat = NULL;
      void          *obj = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJMatrixReadMultipartBinary(prefix_three, MPI_COMM_WORLD, 3,
                                           HYPRE_MEMORY_HOST, &mat);
      ASSERT_NOT_NULL(mat);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_IJMatrixGetObject(mat, &obj);
      ASSERT_NOT_NULL(obj);
      HYPRE_IJMatrixDestroy(mat);
   }
   if (myid == 0)
   {
      cleanup_temp_files();
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   int nprocs;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs < 2)
   {
      fprintf(stderr, "SKIP: test_matrix_mpi requires at least 2 MPI ranks\n");
      TEST_HYPRE_FINALIZE();
      MPI_Finalize();
      return 77;
   }

   RUN_TEST(test_gnparts_lt_nprocs);
   RUN_TEST(test_two_ranks_offdiagonal_and_multipart_distribution);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
