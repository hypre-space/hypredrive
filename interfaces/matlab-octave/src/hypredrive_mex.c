/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/**
 * @file hypredrive_mex.c
 * @brief Serial MATLAB/Octave MEX entry point for solving sparse systems.
 *
 * This interface is intentionally one-shot in v1: MATLAB/Octave owns a sparse
 * matrix and dense RHS, the MEX function converts CSC storage to the CSR slab
 * expected by hypredrive, solves on MPI_COMM_SELF, copies the solution into a
 * double vector, and releases all hypredrive objects before returning.
 */

#include "mex.h"

#include <stdint.h>
#include <string.h>

#include "HYPREDRV.h"
#include "internal/compatibility.h"

static const char *HYPREDRV_MATLAB_DEFAULT_YAML = "solver:\n"
                                                  "  pcg:\n"
                                                  "    max_iter: 100\n"
                                                  "    relative_tol: 1.0e-8\n"
                                                  "preconditioner:\n"
                                                  "  amg:\n"
                                                  "    max_iter: 1\n"
                                                  "    tolerance: 0.0\n"
                                                  "    print_level: 0\n"
                                                  "general:\n"
                                                  "  statistics: 0\n";

static int HYPREDRV_matlab_initialized     = 0;
static int HYPREDRV_matlab_initialized_mpi = 0;
static int HYPREDRV_matlab_atexit          = 0;

typedef char HYPREDRV_matlab_requires_double_hypre_real
   [(sizeof(HYPRE_Real) == sizeof(double)) ? 1 : -1];
typedef char HYPREDRV_matlab_rejects_complex_hypre
   [(sizeof(HYPRE_Complex) == sizeof(HYPRE_Real)) ? 1 : -1];

static void
HYPREDRV_MatlabAtExit(void)
{
   if (HYPREDRV_matlab_initialized)
   {
      uint32_t code = HYPREDRV_Finalize();
      if (code != HYPREDRV_SUCCESS)
      {
         mexWarnMsgIdAndTxt("hypredrive:FinalizeFailed",
                            "HYPREDRV_Finalize failed with code 0x%x",
                            (unsigned int)code);
      }
      HYPREDRV_matlab_initialized = 0;
   }
   if (HYPREDRV_matlab_initialized_mpi)
   {
      int mpi_finalized = 0;
      MPI_Finalized(&mpi_finalized);
      if (!mpi_finalized)
      {
         int mpi_code = MPI_Finalize();
         if (mpi_code != MPI_SUCCESS)
         {
            mexWarnMsgIdAndTxt("hypredrive:MPIFinalizeFailed",
                               "MPI_Finalize failed with code %d", mpi_code);
         }
      }
      HYPREDRV_matlab_initialized_mpi = 0;
   }
}

static void
HYPREDRV_MatlabFail(const char *id, const char *message)
{
   mexErrMsgIdAndTxt(id, "%s", message);
}

static void
HYPREDRV_MatlabFailCode(uint32_t code, const char *what)
{
   HYPREDRV_ErrorCodeDescribe(code);
   mexErrMsgIdAndTxt("hypredrive:HypreDriveError", "%s failed with code 0x%x", what,
                     (unsigned int)code);
}

static void
HYPREDRV_MatlabEnsureInitialized(void)
{
   if (!HYPREDRV_matlab_initialized)
   {
      int mpi_initialized = 0;
      MPI_Initialized(&mpi_initialized);
      if (!mpi_initialized)
      {
         int mpi_finalized = 0;
         MPI_Finalized(&mpi_finalized);
         if (mpi_finalized)
         {
            HYPREDRV_MatlabFail("hypredrive:MPIAlreadyFinalized",
                                "MPI was already finalized before hypredrive was called");
         }

         int    provided = MPI_THREAD_SINGLE;
         int    argc     = 0;
         char **argv     = NULL;
         if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) !=
             MPI_SUCCESS)
         {
            HYPREDRV_MatlabFail("hypredrive:MPIInitFailed", "MPI_Init_thread failed");
         }
         HYPREDRV_matlab_initialized_mpi = 1;
         if (!HYPREDRV_matlab_atexit)
         {
            mexAtExit(HYPREDRV_MatlabAtExit);
            HYPREDRV_matlab_atexit = 1;
         }
         if (provided < MPI_THREAD_SERIALIZED)
         {
            HYPREDRV_MatlabAtExit();
            HYPREDRV_MatlabFail("hypredrive:MPIThreadLevel",
                                "MPI did not provide the MPI_THREAD_SERIALIZED level "
                                "required by hypredrive");
         }
      }

      uint32_t code = HYPREDRV_Initialize();
      if (code != HYPREDRV_SUCCESS)
      {
         HYPREDRV_MatlabAtExit();
         HYPREDRV_MatlabFailCode(code, "HYPREDRV_Initialize");
      }
      HYPREDRV_matlab_initialized = 1;
      mexLock();
   }
}

static HYPRE_BigInt
HYPREDRV_MatlabBigIntFromMwSize(mwSize value, const char *name)
{
   HYPRE_BigInt converted = 0;
   if (!hypredrv_BigIntFromU64((uint64_t)value, &converted) || (mwSize)converted != value)
   {
      mexErrMsgIdAndTxt("hypredrive:IndexOverflow",
                        "%s is outside the active HYPRE_BigInt range", name);
   }
   return converted;
}

static HYPRE_BigInt
HYPREDRV_MatlabBigIntFromMwIndex(mwIndex value, const char *name)
{
   HYPRE_BigInt converted = 0;
   if (!hypredrv_BigIntFromU64((uint64_t)value, &converted) ||
       (mwIndex)converted != value)
   {
      mexErrMsgIdAndTxt("hypredrive:IndexOverflow",
                        "%s is outside the active HYPRE_BigInt range", name);
   }
   return converted;
}

static char *
HYPREDRV_MatlabCopyDefaultYaml(void)
{
   size_t len  = strlen(HYPREDRV_MATLAB_DEFAULT_YAML);
   char  *yaml = (char *)mxMalloc(len + 1);
   memcpy(yaml, HYPREDRV_MATLAB_DEFAULT_YAML, len + 1);
   return yaml;
}

static char *
HYPREDRV_MatlabGetYaml(const mxArray *arg)
{
   if (arg == NULL)
   {
      return HYPREDRV_MatlabCopyDefaultYaml();
   }
   if (!mxIsChar(arg))
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidOptions",
                          "options must be a YAML character vector");
   }
   char *yaml = mxArrayToString(arg);
   if (!yaml)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidOptions",
                          "failed to convert options to a C string");
   }
   if (yaml[0] == '\0')
   {
      mxFree(yaml);
      return HYPREDRV_MatlabCopyDefaultYaml();
   }
   return yaml;
}

static void
HYPREDRV_MatlabValidateInputs(int nrhs, const mxArray *prhs[])
{
   if (nrhs < 2 || nrhs > 3)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidNumInputs",
                          "usage: x = hypredrive_solve(A, b) or [x, info] = "
                          "hypredrive_solve(A, b, options)");
   }
   if (!mxIsSparse(prhs[0]) || !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidMatrix",
                          "A must be a real double sparse matrix");
   }
   if (mxGetM(prhs[0]) != mxGetN(prhs[0]))
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidMatrix", "A must be square");
   }
   if (mxGetM(prhs[0]) == 0)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidMatrix", "A must be non-empty");
   }
   if (mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidRHS",
                          "b must be a real dense double vector");
   }
   if (mxGetNumberOfElements(prhs[1]) != mxGetM(prhs[0]))
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidRHS", "numel(b) must match size(A,1)");
   }
}

static void
HYPREDRV_MatlabConvertCscToCsr(const mxArray *A, HYPRE_BigInt **indptr_out,
                               HYPRE_BigInt **cols_out, HYPRE_Real **data_out,
                               mwSize *nnz_out)
{
   mwSize         n      = mxGetM(A);
   const mwIndex *jc     = mxGetJc(A);
   const mwIndex *ir     = mxGetIr(A);
   const double  *values = mxGetPr(A);
   mwSize         nnz    = jc[n];

   /* Validate that the dimensions and non-zero count fit in HYPRE_BigInt before any
    * counting/indexing: with a 32-bit HYPRE_BigInt build and nnz > INT32_MAX the
    * running index computations below would overflow and write out of bounds. */
   (void)HYPREDRV_MatlabBigIntFromMwSize(n, "row count");
   (void)HYPREDRV_MatlabBigIntFromMwSize(nnz, "nnz");

   HYPRE_BigInt *indptr = (HYPRE_BigInt *)mxCalloc((size_t)n + 1, sizeof(HYPRE_BigInt));
   HYPRE_BigInt *cols =
      (HYPRE_BigInt *)mxCalloc((size_t)(nnz > 0 ? nnz : 1), sizeof(HYPRE_BigInt));
   HYPRE_Real *data =
      (HYPRE_Real *)mxCalloc((size_t)(nnz > 0 ? nnz : 1), sizeof(HYPRE_Real));
   HYPRE_BigInt *next =
      (HYPRE_BigInt *)mxCalloc((size_t)(n > 0 ? n : 1), sizeof(HYPRE_BigInt));

   for (mwIndex col = 0; col < n; col++)
   {
      for (mwIndex p = jc[col]; p < jc[col + 1]; p++)
      {
         mwIndex row = ir[p];
         if (row >= n)
         {
            HYPREDRV_MatlabFail("hypredrive:InvalidMatrix",
                                "A contains an out-of-range row index");
         }
         indptr[row + 1]++;
      }
   }

   for (mwIndex row = 0; row < n; row++)
   {
      indptr[row + 1] += indptr[row];
      next[row] = indptr[row];
   }

   for (mwIndex col = 0; col < n; col++)
   {
      HYPRE_BigInt hypre_col = HYPREDRV_MatlabBigIntFromMwIndex(col, "column index");
      for (mwIndex p = jc[col]; p < jc[col + 1]; p++)
      {
         mwIndex      row  = ir[p];
         HYPRE_BigInt dest = next[row]++;
         cols[dest]        = hypre_col;
         data[dest]        = (HYPRE_Real)values[p];
      }
   }

   mxFree(next);
   *indptr_out = indptr;
   *cols_out   = cols;
   *data_out   = data;
   *nnz_out    = nnz;
}

static void
HYPREDRV_MatlabDestroyDriver(HYPREDRV_t *drv)
{
   if (drv && *drv)
   {
      uint32_t code = HYPREDRV_Destroy(drv);
      if (code != HYPREDRV_SUCCESS)
      {
         mexWarnMsgIdAndTxt("hypredrive:DestroyFailed",
                            "HYPREDRV_Destroy failed with code 0x%x", (unsigned int)code);
      }
   }
}

static void
HYPREDRV_MatlabCleanup(HYPREDRV_t *drv, HYPRE_BigInt *indptr, HYPRE_BigInt *cols,
                       HYPRE_Real *data, char *yaml)
{
   HYPREDRV_MatlabDestroyDriver(drv);
   if (indptr)
   {
      mxFree(indptr);
   }
   if (cols)
   {
      mxFree(cols);
   }
   if (data)
   {
      mxFree(data);
   }
   if (yaml)
   {
      mxFree(yaml);
   }
}

static void
HYPREDRV_MatlabFailCodeWithCleanup(uint32_t code, const char *what, HYPREDRV_t *drv,
                                   HYPRE_BigInt *indptr, HYPRE_BigInt *cols,
                                   HYPRE_Real *data, char *yaml)
{
   HYPREDRV_MatlabCleanup(drv, indptr, cols, data, yaml);
   HYPREDRV_MatlabFailCode(code, what);
}

static mxArray *
HYPREDRV_MatlabCreateInfoStruct(int iterations, int converged, double final_res_norm,
                                double setup_time, double solve_time,
                                double solution_norm)
{
   const char *fields[] = {"iterations", "converged",  "final_res_norm",
                           "setup_time", "solve_time", "solution_norm"};
   mxArray    *info     = mxCreateStructMatrix(1, 1, 6, fields);
   mxSetField(info, 0, "iterations", mxCreateDoubleScalar((double)iterations));
   mxSetField(info, 0, "converged", mxCreateLogicalScalar(converged != 0));
   mxSetField(info, 0, "final_res_norm", mxCreateDoubleScalar(final_res_norm));
   mxSetField(info, 0, "setup_time", mxCreateDoubleScalar(setup_time));
   mxSetField(info, 0, "solve_time", mxCreateDoubleScalar(solve_time));
   mxSetField(info, 0, "solution_norm", mxCreateDoubleScalar(solution_norm));
   return info;
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   if (nlhs < 1 || nlhs > 2)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidNumOutputs",
                          "hypredrive_solve returns x and optional info");
   }
   HYPREDRV_MatlabValidateInputs(nrhs, prhs);
   HYPREDRV_MatlabEnsureInitialized();

   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;
   HYPRE_Real   *data   = NULL;
   char         *yaml   = HYPREDRV_MatlabGetYaml(nrhs == 3 ? prhs[2] : NULL);
   mwSize        nnz    = 0;
   mwSize        n      = mxGetM(prhs[0]);
   HYPREDRV_t    drv    = NULL;

   HYPREDRV_MatlabConvertCscToCsr(prhs[0], &indptr, &cols, &data, &nnz);
   HYPRE_BigInt row_start = 0;
   HYPRE_BigInt row_end   = HYPREDRV_MatlabBigIntFromMwSize(n - 1, "row_end");
   (void)HYPREDRV_MatlabBigIntFromMwSize(nnz, "nnz");

   uint32_t code = HYPREDRV_Create(MPI_COMM_SELF, &drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_Create", &drv, indptr, cols,
                                         data, yaml);
   }
   code = HYPREDRV_SetLibraryMode(drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_SetLibraryMode", &drv, indptr,
                                         cols, data, yaml);
   }

   char *argv[2] = {yaml, NULL};
   code          = HYPREDRV_InputArgsParse(1, argv, drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_InputArgsParse", &drv, indptr,
                                         cols, data, yaml);
   }

   code =
      HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols, data);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemSetMatrixFromCSR",
                                         &drv, indptr, cols, data, yaml);
   }

   const double *rhs = mxGetPr(prhs[1]);
   code              = HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end,
                                                            (const HYPRE_Real *)rhs);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemSetRHSFromArray",
                                         &drv, indptr, cols, data, yaml);
   }

   code = HYPREDRV_LinearSystemSetInitialGuess(drv, NULL);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemSetInitialGuess",
                                         &drv, indptr, cols, data, yaml);
   }

   code = HYPREDRV_LinearSolverCreate(drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverCreate", &drv,
                                         indptr, cols, data, yaml);
   }

   code = HYPREDRV_LinearSolverSetup(drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverSetup", &drv, indptr,
                                         cols, data, yaml);
   }
   code = HYPREDRV_LinearSolverApply(drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverApply", &drv, indptr,
                                         cols, data, yaml);
   }

   HYPRE_BigInt   solution_length = 0;
   HYPRE_Complex *solution        = NULL;
   int            iterations      = 0;
   int            converged       = 0;
   double         final_res_norm  = 0.0;
   double         setup_time      = 0.0;
   double         solve_time      = 0.0;
   double         solution_norm   = 0.0;

   code = HYPREDRV_LinearSystemGetSolutionLength(drv, &solution_length);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemGetSolutionLength",
                                         &drv, indptr, cols, data, yaml);
   }
   if (solution_length != (HYPRE_BigInt)n)
   {
      HYPREDRV_MatlabCleanup(&drv, indptr, cols, data, yaml);
      mexErrMsgIdAndTxt("hypredrive:SolutionSizeMismatch",
                        "solution length does not match size(A,1)");
   }
   code = HYPREDRV_LinearSystemGetSolutionValues(drv, &solution);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemGetSolutionValues",
                                         &drv, indptr, cols, data, yaml);
   }

   plhs[0]     = mxCreateDoubleMatrix(n, 1, mxREAL);
   double *out = mxGetPr(plhs[0]);
   memcpy(out, solution, (size_t)n * sizeof(double));

   code = HYPREDRV_LinearSolverGetNumIter(drv, &iterations);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverGetNumIter", &drv,
                                         indptr, cols, data, yaml);
   }
   code = HYPREDRV_LinearSolverGetConverged(drv, &converged);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverGetConverged", &drv,
                                         indptr, cols, data, yaml);
   }
   code = HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(drv, &final_res_norm);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(
         code, "HYPREDRV_LinearSolverGetFinalRelativeResidualNorm", &drv, indptr, cols,
         data, yaml);
   }
   code = HYPREDRV_LinearSolverGetSetupTime(drv, &setup_time);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverGetSetupTime", &drv,
                                         indptr, cols, data, yaml);
   }
   code = HYPREDRV_LinearSolverGetSolveTime(drv, &solve_time);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSolverGetSolveTime", &drv,
                                         indptr, cols, data, yaml);
   }
   code = HYPREDRV_LinearSystemGetSolutionNorm(drv, "l2", &solution_norm);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemGetSolutionNorm",
                                         &drv, indptr, cols, data, yaml);
   }

   if (nlhs == 2)
   {
      plhs[1] = HYPREDRV_MatlabCreateInfoStruct(iterations, converged, final_res_norm,
                                                setup_time, solve_time, solution_norm);
   }

   HYPREDRV_MatlabCleanup(&drv, indptr, cols, data, yaml);
}
