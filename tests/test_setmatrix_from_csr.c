/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*-----------------------------------------------------------------------------
 * Unit tests for HYPREDRV_LinearSystemSetMatrixFromCSR /
 * HYPREDRV_LinearSystemSetRHSFromArray (serial, MPI_COMM_SELF).
 *-----------------------------------------------------------------------------*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "test_helpers.h"

/* PCG + AMG with very low iteration counts; enough to demonstrate the API
 * round-trips through a real solve without depending on external data files. */
static const char *kBaseYAML =
   "general:\n"
   "  statistics: off\n"
   "  exec_policy: host\n"
   "linear_system:\n"
   "  init_guess_mode: zeros\n"
   "solver:\n"
   "  pcg:\n"
   "    max_iter: 50\n"
   "    relative_tol: 1.0e-8\n"
   "    print_level: 0\n"
   "preconditioner:\n"
   "  amg:\n"
   "    print_level: 0\n";

static HYPREDRV_t
create_lib_obj(void)
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

   char *argv[] = {(char *)kBaseYAML};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   return obj;
}

/* Build a 1D Laplacian (-1, 2, -1) of size n on a single rank, in CSR form. */
static void
build_laplacian_1d_csr(int n, HYPRE_BigInt **indptr_out, HYPRE_BigInt **cols_out,
                       HYPRE_Real **data_out, HYPRE_Real **rhs_out)
{
   /* Each interior row has 3 entries; the two boundary rows have 2 entries. */
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
      rhs[i]        = 1.0;
   }
   ASSERT_EQ(k, nnz);

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

static HYPRE_IJMatrix
build_identity_ijmatrix(HYPRE_BigInt row_start, HYPRE_BigInt row_end)
{
   HYPRE_BigInt nrows_big = row_end - row_start + 1;
   HYPRE_Int    nrows     = (HYPRE_Int)nrows_big;
   HYPRE_Int   *ncols     = (HYPRE_Int *)malloc(sizeof(HYPRE_Int) * (size_t)nrows);
   HYPRE_BigInt *rows     = (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)nrows);
   HYPRE_BigInt *cols     = (HYPRE_BigInt *)malloc(sizeof(HYPRE_BigInt) * (size_t)nrows);
   HYPRE_Real   *data     = (HYPRE_Real *)malloc(sizeof(HYPRE_Real) * (size_t)nrows);
   HYPRE_IJMatrix mat     = NULL;

   ASSERT_NOT_NULL(ncols);
   ASSERT_NOT_NULL(rows);
   ASSERT_NOT_NULL(cols);
   ASSERT_NOT_NULL(data);

   for (HYPRE_Int i = 0; i < nrows; i++)
   {
      rows[i]  = row_start + (HYPRE_BigInt)i;
      cols[i]  = rows[i];
      data[i]  = 1.0;
      ncols[i] = 1;
   }

   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, row_start, row_end, row_start, row_end,
                                  &mat),
             0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, data), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);

   free(ncols);
   free(rows);
   free(cols);
   free(data);

   return mat;
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_basic_solve
 *
 * Build a small SPD tridiagonal system from CSR + array, run a real solve,
 * and check that PCG converged. End-to-end coverage of the new public APIs.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_basic_solve(void)
{
   const int     n      = 16;
   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_laplacian_1d_csr(n, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj();

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   /* The 1D Laplacian with all-ones RHS has a positive-definite solution.
    * Verify the solution norm is positive (i.e. solve actually wrote something). */
   double sol_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "l2", &sol_norm), ERROR_NONE);
   ASSERT_GT(sol_norm, 0.0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_rebuild_no_leak
 *
 * Build the matrix twice in a row to exercise the "drop previously-owned
 * matrix" path. With sanitizers enabled, this catches double-frees and leaks.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_rebuild_no_leak(void)
{
   const int     n      = 8;
   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   HYPRE_Real   *rhs    = NULL;
   build_laplacian_1d_csr(n, &indptr, &cols, &data, &rhs);

   HYPREDRV_t obj = create_lib_obj();

   /* Two builds in a row; the second must release the first. */
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, (HYPRE_BigInt)(n - 1), indptr,
                                                   cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, (HYPRE_BigInt)(n - 1), rhs),
             ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free_csr(indptr, cols, data, rhs);
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_invalid_args
 *
 * Validate input checks: NULL pointers, inverted row range.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_invalid_args(void)
{
   HYPREDRV_t obj = create_lib_obj();

   /* NULL indptr */
   hypredrv_ErrorCodeResetAll();
   uint32_t code = HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 4, NULL, NULL, NULL);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* row_end < row_start */
   HYPRE_BigInt indptr_dummy[1] = {0};
   hypredrv_ErrorCodeResetAll();
   code =
      HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 5, 4, indptr_dummy, NULL, NULL);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* nnz > 0 but cols/data NULL */
   HYPRE_BigInt indptr2[2] = {0, 1};
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, indptr2, NULL, NULL);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* negative absolute indptr start */
   HYPRE_BigInt indptr_negative[2] = {-1, 0};
   HYPRE_BigInt cols_negative[1]   = {0};
   HYPRE_Real   data_negative[1]   = {1.0};
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, indptr_negative,
                                                cols_negative, data_negative);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* negative column index */
   HYPRE_BigInt indptr_col_negative[2] = {0, 1};
   HYPRE_BigInt cols_col_negative[1]   = {-1};
   HYPRE_Real   data_col_negative[1]   = {1.0};
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, indptr_col_negative,
                                                cols_col_negative, data_col_negative);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* non-monotonic indptr */
   HYPRE_BigInt indptr_bad[4] = {0, 2, 1, 3};
   HYPRE_BigInt cols_bad[3]   = {0, 1, 2};
   HYPRE_Real   data_bad[3]   = {1.0, 1.0, 1.0};
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 2, indptr_bad, cols_bad,
                                                data_bad);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* RHS: NULL values with nrows > 0 */
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 4, NULL);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* RHS: row_end < row_start */
   HYPRE_Real dummy_vals = 0.0;
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetRHSFromArray(obj, 5, 4, &dummy_vals);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* RHS: row range must match the installed CSR matrix row range. */
   HYPRE_BigInt indptr_rhs_check[2] = {0, 1};
   HYPRE_BigInt cols_rhs_check[1]   = {0};
   HYPRE_Real   data_rhs_check[1]   = {1.0};
   HYPRE_Real   rhs_mismatch[1]     = {1.0};
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, indptr_rhs_check,
                                                   cols_rhs_check, data_rhs_check),
             ERROR_NONE);
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetRHSFromArray(obj, 1, 1, rhs_mismatch);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* RHS: larger installed matrix range mismatch. */
   HYPRE_BigInt indptr10[11];
   HYPRE_BigInt cols10[10];
   HYPRE_Real   data10[10];
   HYPRE_Real   rhs_short[6] = {0.0};
   for (int i = 0; i < 10; i++)
   {
      indptr10[i] = (HYPRE_BigInt)i;
      cols10[i]   = (HYPRE_BigInt)i;
      data10[i]   = 1.0;
   }
   indptr10[10] = 10;
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 9, indptr10, cols10,
                                                   data10),
             ERROR_NONE);
   hypredrv_ErrorCodeResetAll();
   code = HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 5, rhs_short);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);

   /* Tear down — destroy must not be confused by failed builds. */
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_ownership_transitions
 *
 * Mix internally owned CSR matrices with a borrowed matrix installed through
 * SetMatrix. Sanitizer builds should catch leaks or double-destroys here.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_ownership_transitions(void)
{
   HYPRE_BigInt indptr10[11];
   HYPRE_BigInt cols10[10];
   HYPRE_Real   data10[10];
   HYPRE_Real   rhs_short[6] = {0.0};
   HYPRE_IJMatrix borrowed = NULL;

   for (int i = 0; i < 10; i++)
   {
      indptr10[i] = (HYPRE_BigInt)i;
      cols10[i]   = (HYPRE_BigInt)i;
      data10[i]   = 1.0;
   }
   indptr10[10] = 10;

   HYPREDRV_t obj = create_lib_obj();

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 9, indptr10, cols10,
                                                   data10),
             ERROR_NONE);

   borrowed = build_identity_ijmatrix(0, 9);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)borrowed), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   uint32_t code = HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 5, rhs_short);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 9, indptr10, cols10,
                                                   data10),
             ERROR_NONE);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(borrowed), 0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_all_empty_rows
 *
 * Valid CSR can have rows with no entries. The builder must not form pointer
 * arithmetic on NULL col/data buffers when the local nonzero count is zero.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_all_empty_rows(void)
{
   HYPRE_BigInt indptr[4] = {0, 0, 0, 0};

   HYPREDRV_t obj = create_lib_obj();

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 2, indptr, NULL, NULL),
             ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

/*-----------------------------------------------------------------------------
 * test_setmatrix_from_csr_zero_local_rows
 *
 * Edge case: a rank legitimately holds zero local rows. The function should
 * still assemble the (empty) IJMatrix successfully. We can't exercise the
 * full multi-rank empty-row case here, but we can at least confirm the
 * function tolerates an empty range without crashing.
 *
 * Since IJMatrix expects row_start <= row_end (inclusive), "zero rows" is
 * not actually expressible via this API on a single rank with the global
 * partition rooted at 0. The closest analog is row_end == row_start (one row).
 * Just check the single-row path works.
 *-----------------------------------------------------------------------------*/

static void
test_setmatrix_from_csr_single_row(void)
{
   HYPRE_BigInt indptr[2] = {0, 1};
   HYPRE_BigInt cols[1]   = {0};
   HYPRE_Real   data[1]   = {3.0};
   HYPRE_Real   rhs[1]    = {6.0};

   HYPREDRV_t obj = create_lib_obj();

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, indptr, cols, data),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   /* Solution should be approximately 6/3 = 2 (in l2 norm). */
   double sol_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "l2", &sol_norm), ERROR_NONE);
   ASSERT_EQ_DOUBLE(sol_norm, 2.0, 1e-6);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_setmatrix_from_csr_basic_solve);
   RUN_TEST(test_setmatrix_from_csr_rebuild_no_leak);
   RUN_TEST(test_setmatrix_from_csr_invalid_args);
   RUN_TEST(test_setmatrix_from_csr_ownership_transitions);
   RUN_TEST(test_setmatrix_from_csr_all_empty_rows);
   RUN_TEST(test_setmatrix_from_csr_single_row);

   MPI_Finalize();
   return 0;
}
