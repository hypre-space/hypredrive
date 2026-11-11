/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "internal/error.h"
#include "object.h"
#include "test_helpers.h"

static void
test_distributed_eigenspectrum(void)
{
   int rank = 0, size = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   /* Exercise both an empty output rank and empty ranks after the matrix. */
   for (int empty_root = 0; empty_root < 2; empty_root++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
      char  yaml[] = "general:\n  exec_policy: host\n  statistics: off\n"
                     "linear_system:\n  eigspec:\n    enable: on\n"
                     "solver: pcg\npreconditioner: amg\n";
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      HYPRE_BigInt first = rank < 2 ? rank : 2;
      HYPRE_BigInt last  = rank < 2 ? rank : 1;
      if (empty_root)
      {
         first = rank < 2 ? 0 : 2;
         last  = rank == 0 ? -1 : 1;
      }
      HYPRE_IJMatrix matrix = NULL;
      ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, first, last, first, last, &matrix),
                0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST), 0);
      if (last >= first)
      {
         HYPRE_Int     counts[] = {1, 1};
         HYPRE_BigInt  rows[]   = {first, first + 1};
         HYPRE_Complex values[] = {first ? 3.0 : 2.0, 3.0};
         ASSERT_EQ(HYPRE_IJMatrixSetValues(matrix, (HYPRE_Int)(last - first + 1), counts,
                                           rows, rows, values),
                   0);
      }
      ASSERT_EQ(HYPRE_IJMatrixAssemble(matrix), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
      EigSpec_args *args = &obj->iargs->ls.eigspec;
      if (!rank)
      {
         test_scratch_path(args->output_prefix, MAX_FILENAME_LENGTH,
                           empty_root ? "empty-root" : "distributed");
      }
      MPI_Bcast(args->output_prefix, MAX_FILENAME_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
      for (int preconditioned = 0; preconditioned < 2; preconditioned++)
      {
         args->preconditioned = preconditioned;
         for (int symmetric = 0; symmetric < 2; symmetric++)
         {
            args->hermitian = symmetric;
            ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
            if (!rank)
            {
               char path[MAX_FILENAME_LENGTH + 32];
               snprintf(path, sizeof(path), "%s.values.txt", args->output_prefix);
               FILE *fp = fopen(path, "r");
               ASSERT_NOT_NULL(fp);
               double eigenvalues[2], imaginary = 0;
               for (int ev = 0; ev < 2; ev++)
               {
                  ASSERT_EQ(fscanf(fp, symmetric ? "%lf" : "%lf %lf", &eigenvalues[ev],
                                   &imaginary),
                            symmetric ? 1 : 2);
                  ASSERT_EQ_DOUBLE(imaginary, 0.0, 1e-12);
               }
               ASSERT_EQ(fclose(fp), 0);
               ASSERT_EQ_DOUBLE(eigenvalues[0] + eigenvalues[1],
                                preconditioned ? 2.0 : 5.0, 1e-12);
               ASSERT_EQ_DOUBLE(eigenvalues[0] * eigenvalues[1],
                                preconditioned ? 1.0 : 6.0, 1e-12);
            }
         }
      }

      /* A root-only output error must reach every rank and its diagnostic state. */
      args->preconditioned = 0;
      if (!rank)
      {
         test_scratch_path(args->output_prefix, MAX_FILENAME_LENGTH,
                           "missing-directory/eig");
      }
      MPI_Bcast(args->output_prefix, MAX_FILENAME_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
      ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_FILE_NOT_FOUND);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_FILE_NOT_FOUND);

      /* Rank-local invalid input must stop all ranks before the collective gather. */
      HYPRE_IJMatrix saved = obj->mat_A.handle;
      if (rank == size - 1) obj->mat_A.handle = NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_FILE_NOT_FOUND);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_FILE_NOT_FOUND);
      obj->mat_A.handle = saved;
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_distributed_eigenspectrum);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
