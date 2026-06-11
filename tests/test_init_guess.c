/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*-----------------------------------------------------------------------------
 * Regression tests for linear_system.init_guess_mode in library mode
 * (serial, MPI_COMM_SELF).
 *
 * init_guess_mode "previous" historically degenerated to "zeros": the
 * working solution was destroyed and recreated inside SetInitialGuess
 * before the mode-4 copy could read it. Both bugs produced correct (just
 * slower) solves, so these tests assert on iteration counts rather than
 * on solutions alone. GMRES is used because it accepts an
 * already-converged initial guess at iteration 0, whereas hypre's PCG
 * only checks convergence inside the first iteration.
 *-----------------------------------------------------------------------------*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "test_helpers.h"

static const char *kPreviousYAML =
   "general:\n"
   "  statistics: off\n"
   "  exec_policy: host\n"
   "linear_system:\n"
   "  init_guess_mode: previous\n"
   "solver:\n"
   "  gmres:\n"
   "    max_iter: 100\n"
   "    relative_tol: 1.0e-8\n"
   "    print_level: 0\n"
   "preconditioner:\n"
   "  amg:\n"
   "    print_level: 0\n";

static const char *kOnesYAML =
   "general:\n"
   "  statistics: off\n"
   "  exec_policy: host\n"
   "linear_system:\n"
   "  init_guess_mode: ones\n"
   "solver:\n"
   "  gmres:\n"
   "    max_iter: 100\n"
   "    relative_tol: 1.0e-8\n"
   "    print_level: 0\n"
   "preconditioner:\n"
   "  amg:\n"
   "    print_level: 0\n";

static HYPREDRV_t
create_lib_obj(const char *yaml)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char *argv[] = {(char *)yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   return obj;
}

/* Build a 1D Laplacian (-1, 2, -1) of size n on a single rank, in CSR form.
 * rhs_mode 0: all-ones RHS; rhs_mode 1: rhs = A @ ones (exact solution ones). */
static void
build_laplacian_1d_csr(int n, int rhs_mode, HYPRE_BigInt **indptr_out,
                       HYPRE_BigInt **cols_out, HYPRE_Real **data_out,
                       HYPRE_Real **rhs_out)
{
   HYPRE_Int     nnz = 3 * n - 2;
   HYPRE_BigInt *indptr =
      (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)(n + 1));
   HYPRE_BigInt *cols = (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)nnz);
   HYPRE_Real   *data = (HYPRE_Real *)malloc(sizeof(HYPRE_Real) * (size_t)nnz);
   HYPRE_Real   *rhs  = (HYPRE_Real *)malloc(sizeof(HYPRE_Real) * (size_t)n);

   ASSERT_NOT_NULL(indptr);
   ASSERT_NOT_NULL(cols);
   ASSERT_NOT_NULL(data);
   ASSERT_NOT_NULL(rhs);

   HYPRE_Int k = 0;
   indptr[0]   = 0;
   for (int i = 0; i < n; i++)
   {
      if (i > 0)
      {
         cols[k] = (HYPRE_BigInt)(i - 1);
         data[k] = -1.0;
         k++;
      }
      cols[k] = (HYPRE_BigInt)i;
      data[k] = 2.0;
      k++;
      if (i < n - 1)
      {
         cols[k] = (HYPRE_BigInt)(i + 1);
         data[k] = -1.0;
         k++;
      }
      indptr[i + 1] = (HYPRE_BigInt)k;
      rhs[i]        = (rhs_mode == 0) ? 1.0 : 0.0;
   }
   ASSERT_EQ(k, nnz);
   if (rhs_mode == 1)
   {
      rhs[0]     = 1.0; /* A @ ones */
      rhs[n - 1] = 1.0;
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

/* One full solve cycle (initial guess + solver lifecycle); returns the
 * iteration count. Mirrors the CLI/examples flow, including the
 * ResetInitialGuess call that copies x0 into the working solution. */
static int
solve_cycle(HYPREDRV_t obj)
{
   int iters = -1;

   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &iters), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   return iters;
}

/*-----------------------------------------------------------------------------
 * test_init_guess_previous_reuses_solution
 *
 * Re-solving an identical system with init_guess_mode "previous" must take
 * zero iterations: the previous solution already satisfies the stopping
 * criterion. Anything > 0 means the previous solution never reached the
 * solver (e.g. the working solution was zeroed before x0 captured it).
 *-----------------------------------------------------------------------------*/

static void
test_init_guess_previous_reuses_solution(void)
{
   const int     n      = 32;
   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_laplacian_1d_csr(n, 0, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj(kPreviousYAML);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   int first_iters = solve_cycle(obj);
   ASSERT_GT(first_iters, 0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   int second_iters = solve_cycle(obj);
   ASSERT_EQ(second_iters, 0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

/*-----------------------------------------------------------------------------
 * test_init_guess_previous_first_solve_zeros_fallback
 *
 * With no previous solution available, "previous" must fall back to zeros
 * and solve normally rather than fail.
 *-----------------------------------------------------------------------------*/

static void
test_init_guess_previous_first_solve_zeros_fallback(void)
{
   const int     n      = 32;
   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_laplacian_1d_csr(n, 0, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj(kPreviousYAML);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   int iters = solve_cycle(obj);
   ASSERT_GT(iters, 0);

   double sol_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "l2", &sol_norm), ERROR_NONE);
   ASSERT_GT(sol_norm, 0.0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

/*-----------------------------------------------------------------------------
 * test_init_guess_ones_reaches_solver
 *
 * The 1D Laplacian with rhs = A @ ones has the all-ones exact solution, so
 * an honored "ones" initial guess converges in zero iterations. Guards the
 * x0-to-working-solution copy independently of mode "previous".
 *-----------------------------------------------------------------------------*/

static void
test_init_guess_ones_reaches_solver(void)
{
   const int     n      = 32;
   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_laplacian_1d_csr(n, 1, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj(kOnesYAML);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   int iters = solve_cycle(obj);
   ASSERT_EQ(iters, 0);

   /* ||ones||_2 = sqrt(n) */
   double sol_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "l2", &sol_norm), ERROR_NONE);
   ASSERT_EQ_DOUBLE(sol_norm, sqrt((double)n), 1e-6);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_init_guess_previous_reuses_solution);
   RUN_TEST(test_init_guess_previous_first_solve_zeros_fallback);
   RUN_TEST(test_init_guess_ones_reaches_solver);

   MPI_Finalize();
   return 0;
}
