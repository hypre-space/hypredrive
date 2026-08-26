/******************************************************************************
 * MPI coverage for hypredrv_IJVectorReadMultipartBinary (rank/part branches).
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
write_vector_part_file(const char *prefix, int part_id, uint64_t nrows,
                       const double *values)
{
   char filename[256];
   snprintf(filename, sizeof(filename), "%s.%05d.bin", prefix, part_id);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[8] = {0};
   header[1]          = sizeof(double);
   header[5]          = nrows;

   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 8, fp), 8u);
   if (nrows > 0)
   {
      ASSERT_EQ_SIZE(fwrite(values, sizeof(double), (size_t)nrows, fp), (size_t)nrows);
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

   HYPRE_IJVector vec = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJVectorReadMultipartBinary("test_vector_mpi_noprefix", MPI_COMM_WORLD, 1,
                                        HYPRE_MEMORY_HOST, &vec);
   ASSERT_NULL(vec);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_two_ranks_multipart(void)
{
   int nprocs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (nprocs < 2)
   {
      return;
   }

   const char *prefix = "test_vector_mpi_gp2";
   if (myid == 0)
   {
      double v0[1] = {1.0};
      double v1[1] = {2.0};
      write_vector_part_file(prefix, 0, 1, v0);
      write_vector_part_file(prefix, 1, 1, v1);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   {
      HYPRE_IJVector vec = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJVectorReadMultipartBinary(prefix, MPI_COMM_WORLD, 2, HYPRE_MEMORY_HOST,
                                           &vec);
      ASSERT_NOT_NULL(vec);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_IJVectorDestroy(vec);
   }
   if (myid == 0)
   {
      cleanup_temp_files();
   }
   MPI_Barrier(MPI_COMM_WORLD);

   /* g_nparts=3 on 2 ranks: rank0 must concatenate parts {0,1}, while rank1
    * reads part {2}.  This catches multipart SetValues calls that restart at
    * the rank's first local row and overwrite the preceding part. */
   const char *prefix_three = "test_vector_mpi_gp3";
   if (myid == 0)
   {
      double v0[1] = {10.0};
      double v1[2] = {20.0, 30.0};
      double v2[1] = {40.0};
      write_vector_part_file(prefix_three, 0, 1, v0);
      write_vector_part_file(prefix_three, 1, 2, v1);
      write_vector_part_file(prefix_three, 2, 1, v2);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   {
      HYPRE_IJVector vec    = NULL;
      HYPRE_BigInt   ilower = 0, iupper = -1;
      HYPRE_BigInt   indices[3] = {0, 0, 0};
      HYPRE_Complex  values[3]  = {0.0, 0.0, 0.0};

      hypredrv_ErrorCodeResetAll();
      hypredrv_IJVectorReadMultipartBinary(prefix_three, MPI_COMM_WORLD, 3,
                                           HYPRE_MEMORY_HOST, &vec);
      ASSERT_NOT_NULL(vec);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(HYPRE_IJVectorGetLocalRange(vec, &ilower, &iupper), 0);

      HYPRE_Int nlocal = (HYPRE_Int)(iupper - ilower + 1);
      ASSERT_EQ(nlocal, myid == 0 ? 3 : 1);
      for (HYPRE_Int i = 0; i < nlocal; i++)
      {
         indices[i] = ilower + i;
      }
      ASSERT_EQ(HYPRE_IJVectorGetValues(vec, nlocal, indices, values), 0);
      for (HYPRE_Int i = 0; i < nlocal; i++)
      {
         ASSERT_EQ_DOUBLE((double)values[i], 10.0 * (double)(indices[i] + 1), 1.0e-12);
      }
      HYPRE_IJVectorDestroy(vec);
   }
   if (myid == 0)
   {
      cleanup_temp_files();
   }
   MPI_Barrier(MPI_COMM_WORLD);
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
      fprintf(stderr, "SKIP: test_vector_mpi requires at least 2 MPI ranks\n");
      TEST_HYPRE_FINALIZE();
      MPI_Finalize();
      return 77;
   }

   RUN_TEST(test_gnparts_lt_nprocs);
   RUN_TEST(test_two_ranks_multipart);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
