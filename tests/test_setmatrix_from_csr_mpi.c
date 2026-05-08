/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*-----------------------------------------------------------------------------
 * MPI coverage for HYPREDRV_LinearSystemSetMatrixFromCSR /
 * HYPREDRV_LinearSystemSetRHSFromArray with a distributed CSR layout.
 *
 * Each rank owns a contiguous block of rows of a 1D Laplacian; the test
 * exercises the cross-rank ParCSR diag/offd split (the boundary rows have
 * one column index that lives on the neighboring rank).
 *-----------------------------------------------------------------------------*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "test_helpers.h"

static const char *kBaseYAML =
   "general:\n"
   "  statistics: off\n"
   "  exec_policy: host\n"
   "linear_system:\n"
   "  init_guess_mode: zeros\n"
   "solver:\n"
   "  pcg:\n"
   "    max_iter: 100\n"
   "    relative_tol: 1.0e-8\n"
   "    print_level: 0\n"
   "preconditioner:\n"
   "  amg:\n"
   "    print_level: 0\n";

static HYPREDRV_t
create_lib_obj_world(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char *argv[] = {(char *)kBaseYAML};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   return obj;
}

/* Compute this rank's row range for a 1D problem of size N split as evenly
 * as possible across nprocs ranks. Returns row_start and row_end (inclusive). */
static void
partition_rows(int N, int nprocs, int rank, HYPRE_BigInt *row_start,
               HYPRE_BigInt *row_end)
{
   int base    = N / nprocs;
   int rem     = N % nprocs;
   int local_n = base + (rank < rem ? 1 : 0);
   int start   = rank * base + (rank < rem ? rank : rem);
   *row_start  = (HYPRE_BigInt)start;
   *row_end    = (HYPRE_BigInt)(start + local_n - 1);
}

/* Build the local CSR slab of a 1D Laplacian for rows [row_start, row_end]. */
static void
build_local_laplacian_csr(HYPRE_BigInt N, HYPRE_BigInt row_start, HYPRE_BigInt row_end,
                          HYPRE_BigInt **indptr_out, HYPRE_BigInt **cols_out,
                          HYPRE_Real **data_out, HYPRE_Real **rhs_out)
{
   HYPRE_BigInt nrows = row_end - row_start + 1;
   /* Worst case 3 entries per row. */
   HYPRE_BigInt  max_nnz = 3 * nrows;
   HYPRE_BigInt *indptr =
      (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)(nrows + 1));
   HYPRE_BigInt *cols = (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)max_nnz);
   HYPRE_Real   *data = (HYPRE_Real *)malloc(sizeof(HYPRE_Real) * (size_t)max_nnz);
   HYPRE_Real   *rhs  = (HYPRE_Real *)malloc(sizeof(HYPRE_Real) * (size_t)nrows);

   ASSERT_NOT_NULL(indptr);
   ASSERT_NOT_NULL(cols);
   ASSERT_NOT_NULL(data);
   ASSERT_NOT_NULL(rhs);

   HYPRE_BigInt k = 0;
   indptr[0]      = 0;
   for (HYPRE_BigInt i = 0; i < nrows; i++)
   {
      HYPRE_BigInt grow = row_start + i;
      if (grow > 0)
      {
         cols[k] = grow - 1;
         data[k] = -1.0;
         k++;
      }
      cols[k] = grow;
      data[k] = 2.0;
      k++;
      if (grow < N - 1)
      {
         cols[k] = grow + 1;
         data[k] = -1.0;
         k++;
      }
      indptr[i + 1] = k;
      rhs[i]        = 1.0;
   }

   *indptr_out = indptr;
   *cols_out   = cols;
   *data_out   = data;
   *rhs_out    = rhs;
}

static void
free_csr(HYPRE_BigInt *indptr, HYPRE_BigInt *cols, HYPRE_Real *data, HYPRE_Real *rhs)
{
   free(indptr);
   free(cols);
   free(data);
   free(rhs);
}

/*-----------------------------------------------------------------------------
 * test_distributed_laplacian_solve
 *
 * Each rank assembles its slab of a 1D Laplacian on N=64 rows, then we run
 * a real PCG+AMG solve. The test fails fast if any rank's solve returns an
 * error or if the resulting solution norm is non-positive.
 *-----------------------------------------------------------------------------*/

static void
test_distributed_laplacian_solve(void)
{
   int nprocs = 1, myid = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (nprocs < 2)
   {
      return;
   }

   const HYPRE_BigInt N         = 64;
   HYPRE_BigInt       row_start = 0, row_end = 0;
   partition_rows((int)N, nprocs, myid, &row_start, &row_end);

   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_local_laplacian_csr(N, row_start, row_end, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj_world();

   ASSERT_EQ(
      HYPREDRV_LinearSystemSetMatrixFromCSR(obj, row_start, row_end, indptr, cols, data),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, row_start, row_end, rhs),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   /* Cross-rank correctness: the global solution norm must be positive. */
   double sol_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "l2", &sol_norm), ERROR_NONE);
   ASSERT_GT(sol_norm, 0.0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_distributed_laplacian_solve);

   MPI_Finalize();
   return 0;
}
