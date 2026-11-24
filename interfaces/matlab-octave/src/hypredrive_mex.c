/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/**
 * @file hypredrive_mex.c
 * @brief Serial MATLAB/Octave MEX entry point for hypredrive.
 *
 * Two call styles share this one MEX function:
 *
 *  1. One-shot solve (unchanged): [x, info] = hypredrive_mex(A, b, yaml).
 *     MATLAB/Octave owns a sparse matrix and dense RHS, the MEX function
 *     converts CSC storage to the CSR slab expected by hypredrive, solves on
 *     MPI_COMM_SELF, copies the solution into a double vector, and releases all
 *     hypredrive objects before returning.
 *
 *  2. Command dispatcher: out = hypredrive_mex('command', handle, args...).
 *     Exposes every public HYPREDRV_ API. Drivers and HYPRE IJ matrices/vectors
 *     are referred to by opaque uint64 handles that are validated on each use.
 *     See the README for the command list. All live handles are released by
 *     hypredrive_mex('destroy_all') or automatically at mexAtExit.
 */

#include "mex.h"

#include <stdint.h>
#include <stdlib.h>
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

static void HYPREDRV_MatlabDestroyAllHandles(void);

static void
HYPREDRV_MatlabAtExit(void)
{
   if (HYPREDRV_matlab_initialized)
   {
      HYPREDRV_MatlabDestroyAllHandles();
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
   HYPREDRV_ErrorCodeClear();
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
HYPREDRV_MatlabConvertCscToCsrRect(const mxArray *A, HYPRE_BigInt **indptr_out,
                                   HYPRE_BigInt **cols_out, HYPRE_Real **data_out,
                                   mwSize *nnz_out)
{
   mwSize         n      = mxGetM(A); /* number of rows */
   mwSize         ncol   = mxGetN(A);
   const mwIndex *jc     = mxGetJc(A);
   const mwIndex *ir     = mxGetIr(A);
   const double  *values = mxGetPr(A);
   mwSize         nnz    = jc[ncol];

   /* Validate that the dimensions and non-zero count fit in HYPRE_BigInt before any
    * counting/indexing: with a 32-bit HYPRE_BigInt build and nnz > INT32_MAX the
    * running index computations below would overflow and write out of bounds. */
   (void)HYPREDRV_MatlabBigIntFromMwSize(n, "row count");
   (void)HYPREDRV_MatlabBigIntFromMwSize(ncol, "column count");
   (void)HYPREDRV_MatlabBigIntFromMwSize(nnz, "nnz");

   HYPRE_BigInt *indptr = (HYPRE_BigInt *)mxCalloc((size_t)n + 1, sizeof(HYPRE_BigInt));
   HYPRE_BigInt *cols =
      (HYPRE_BigInt *)mxCalloc((size_t)(nnz > 0 ? nnz : 1), sizeof(HYPRE_BigInt));
   HYPRE_Real *data =
      (HYPRE_Real *)mxCalloc((size_t)(nnz > 0 ? nnz : 1), sizeof(HYPRE_Real));
   HYPRE_BigInt *next =
      (HYPRE_BigInt *)mxCalloc((size_t)(n > 0 ? n : 1), sizeof(HYPRE_BigInt));

   for (mwIndex col = 0; col < ncol; col++)
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

   for (mwIndex col = 0; col < ncol; col++)
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
HYPREDRV_MatlabConvertCscToCsr(const mxArray *A, HYPRE_BigInt **indptr_out,
                               HYPRE_BigInt **cols_out, HYPRE_Real **data_out,
                               mwSize *nnz_out)
{
   HYPREDRV_MatlabConvertCscToCsrRect(A, indptr_out, cols_out, data_out, nnz_out);
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

static void
HYPREDRV_MatlabOneShot(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
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

   code = HYPREDRV_LinearSystemResetInitialGuess(drv);
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemResetInitialGuess",
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
   plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
   code = HYPREDRV_LinearSystemCopySolutionValues(drv, solution_length, mxGetPr(plhs[0]));
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCodeWithCleanup(code, "HYPREDRV_LinearSystemCopySolutionValues",
                                         &drv, indptr, cols, data, yaml);
   }

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

/*==========================================================================
 * Handle-based command API
 *
 * hypredrive_mex('command', args...) dispatches to one of the commands in
 * HYPREDRV_matlab_commands below. Every public HYPREDRV_ function is reachable
 * through exactly one command. Objects are exposed as opaque scalar uint64
 * handles, never as raw pointers: handles are issued from a monotonically
 * increasing counter (never reused) and looked up in a table on every call, so a
 * stale, forged, wrong-kind, or already-destroyed handle raises
 * 'hypredrive:InvalidHandle' instead of dereferencing memory.
 *
 * Handle kinds:
 *   driver  - a HYPREDRV_t created on MPI_COMM_SELF in library mode.
 *   matrix  - a HYPRE_IJMatrix (owned: built from a MATLAB sparse matrix;
 *             borrowed: returned by 'get_matrix', valid while its driver lives).
 *   vector  - a HYPRE_IJVector (owned or borrowed likewise).
 *
 * Drivers run in library mode, so matrices/vectors attached to a driver stay
 * owned by this table. 'destroy' on a driver also invalidates borrowed handles
 * issued from it. 'destroy' on an owned matrix/vector that is still attached to a
 * live driver is refused (hypredrive:ObjectInUse); destroy the driver first.
 * 'destroy_all' and mexAtExit release everything (drivers first).
 *==========================================================================*/

typedef enum
{
   HYPREDRV_MATLAB_KIND_DRIVER = 1,
   HYPREDRV_MATLAB_KIND_MATRIX = 2,
   HYPREDRV_MATLAB_KIND_VECTOR = 3
} HYPREDRV_MatlabKind;

typedef struct
{
   uint64_t id;
   int      kind;
   void    *ptr;
   uint64_t owner;    /* driver id for borrowed objects, else 0 */
   int      owned;    /* 1 if this table must destroy ptr */
   uint64_t attached; /* last driver this matrix/vector was attached to */
} HYPREDRV_MatlabEntry;

static HYPREDRV_MatlabEntry *HYPREDRV_matlab_table     = NULL;
static size_t                HYPREDRV_matlab_count     = 0;
static size_t                HYPREDRV_matlab_capacity  = 0;
static uint64_t              HYPREDRV_matlab_next_id   = 1;

typedef void (*HYPREDRV_MatlabCommandFn)(int nlhs, mxArray *plhs[], int na,
                                         const mxArray *a[]);

typedef struct
{
   const char              *name;
   HYPREDRV_MatlabCommandFn fn;
   int                      min_args;
   int                      max_args;
} HYPREDRV_MatlabCommand;

#define HD_CHECK(call) HYPREDRV_MatlabCheck((call), #call)

static void
HYPREDRV_MatlabCheck(uint32_t code, const char *what)
{
   if (code != HYPREDRV_SUCCESS)
   {
      HYPREDRV_MatlabFailCode(code, what);
   }
}

static HYPREDRV_MatlabEntry *
HYPREDRV_MatlabFindEntry(uint64_t id)
{
   for (size_t i = 0; i < HYPREDRV_matlab_count; i++)
   {
      if (HYPREDRV_matlab_table[i].id == id)
      {
         return &HYPREDRV_matlab_table[i];
      }
   }
   return NULL;
}

static uint64_t
HYPREDRV_MatlabRegister(int kind, void *ptr, uint64_t owner, int owned)
{
   if (HYPREDRV_matlab_count == HYPREDRV_matlab_capacity)
   {
      size_t                new_cap = HYPREDRV_matlab_capacity ? 2 * HYPREDRV_matlab_capacity : 16;
      HYPREDRV_MatlabEntry *grown   = (HYPREDRV_MatlabEntry *)realloc(
         HYPREDRV_matlab_table, new_cap * sizeof(HYPREDRV_MatlabEntry));
      if (!grown)
      {
         HYPREDRV_MatlabFail("hypredrive:OutOfMemory", "failed to grow the handle table");
      }
      HYPREDRV_matlab_table    = grown;
      HYPREDRV_matlab_capacity = new_cap;
   }
   HYPREDRV_MatlabEntry *e = &HYPREDRV_matlab_table[HYPREDRV_matlab_count++];
   e->id                   = HYPREDRV_matlab_next_id++;
   e->kind                 = kind;
   e->ptr                  = ptr;
   e->owner                = owner;
   e->owned                = owned;
   e->attached             = 0;
   return e->id;
}

static void
HYPREDRV_MatlabRemoveEntry(uint64_t id)
{
   for (size_t i = 0; i < HYPREDRV_matlab_count; i++)
   {
      if (HYPREDRV_matlab_table[i].id == id)
      {
         HYPREDRV_matlab_table[i] = HYPREDRV_matlab_table[--HYPREDRV_matlab_count];
         return;
      }
   }
}

static void
HYPREDRV_MatlabReleaseObject(HYPREDRV_MatlabEntry *e)
{
   if (e->kind == HYPREDRV_MATLAB_KIND_MATRIX && e->owned)
   {
      HYPRE_IJMatrixDestroy((HYPRE_IJMatrix)e->ptr);
   }
   else if (e->kind == HYPREDRV_MATLAB_KIND_VECTOR && e->owned)
   {
      HYPRE_IJVectorDestroy((HYPRE_IJVector)e->ptr);
   }
}

/* Destroys a driver entry and invalidates every borrowed handle issued from it.
 * Returns the HYPREDRV_Destroy status. */
static uint32_t
HYPREDRV_MatlabDestroyDriverEntry(uint64_t id)
{
   HYPREDRV_MatlabEntry *e = HYPREDRV_MatlabFindEntry(id);
   if (!e)
   {
      return HYPREDRV_SUCCESS;
   }
   HYPREDRV_t drv = (HYPREDRV_t)e->ptr;
   HYPREDRV_MatlabRemoveEntry(id);
   for (size_t i = 0; i < HYPREDRV_matlab_count;)
   {
      if (HYPREDRV_matlab_table[i].owner == id)
      {
         HYPREDRV_matlab_table[i] = HYPREDRV_matlab_table[--HYPREDRV_matlab_count];
      }
      else
      {
         i++;
      }
   }
   return HYPREDRV_Destroy(&drv);
}

static void
HYPREDRV_MatlabDestroyAllHandles(void)
{
   /* Drivers first: they may reference the matrices and vectors below. */
   for (size_t i = 0; i < HYPREDRV_matlab_count;)
   {
      if (HYPREDRV_matlab_table[i].kind == HYPREDRV_MATLAB_KIND_DRIVER)
      {
         uint32_t code = HYPREDRV_MatlabDestroyDriverEntry(HYPREDRV_matlab_table[i].id);
         if (code != HYPREDRV_SUCCESS)
         {
            HYPREDRV_ErrorCodeClear();
         }
         i = 0; /* table was compacted; rescan */
      }
      else
      {
         i++;
      }
   }
   for (size_t i = 0; i < HYPREDRV_matlab_count; i++)
   {
      HYPREDRV_MatlabReleaseObject(&HYPREDRV_matlab_table[i]);
   }
   HYPREDRV_matlab_count = 0;
   free(HYPREDRV_matlab_table);
   HYPREDRV_matlab_table    = NULL;
   HYPREDRV_matlab_capacity = 0;
}

/*--- argument helpers ----------------------------------------------------*/

static mxArray *
HYPREDRV_MatlabMakeHandle(uint64_t id)
{
   mxArray *h = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
   *(uint64_t *)mxGetData(h) = id;
   return h;
}

static uint64_t
HYPREDRV_MatlabHandleId(const mxArray *arg)
{
   if (!arg || mxGetClassID(arg) != mxUINT64_CLASS || mxIsComplex(arg) ||
       mxGetNumberOfElements(arg) != 1)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                          "handle must be a scalar uint64 returned by hypredrive_mex");
   }
   return *(const uint64_t *)mxGetData(arg);
}

static HYPREDRV_MatlabEntry *
HYPREDRV_MatlabArgEntry(const mxArray *arg, int kind)
{
   HYPREDRV_MatlabEntry *e = HYPREDRV_MatlabFindEntry(HYPREDRV_MatlabHandleId(arg));
   if (!e || e->kind != kind)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                          "handle is stale, destroyed, or of the wrong kind");
   }
   return e;
}

static HYPREDRV_t
HYPREDRV_MatlabArgDriver(const mxArray *arg)
{
   return (HYPREDRV_t)HYPREDRV_MatlabArgEntry(arg, HYPREDRV_MATLAB_KIND_DRIVER)->ptr;
}

/* Matrix/vector argument; marks it as attached to driver `drv_arg`. An empty
 * array means NULL when allow_null is set. Returns NULL for the empty case. */
static void *
HYPREDRV_MatlabArgObject(const mxArray *arg, int kind, const mxArray *drv_arg,
                         int allow_null)
{
   if (allow_null && arg && mxIsEmpty(arg) && !mxIsUint64(arg))
   {
      return NULL;
   }
   HYPREDRV_MatlabEntry *e = HYPREDRV_MatlabArgEntry(arg, kind);
   if (drv_arg)
   {
      e->attached = HYPREDRV_MatlabHandleId(drv_arg);
   }
   return e->ptr;
}

static char *
HYPREDRV_MatlabArgString(const mxArray *arg, const char *name)
{
   if (!arg || !mxIsChar(arg))
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidArgument", "%s must be a character vector",
                        name);
   }
   char *str = mxArrayToString(arg); /* freed automatically when the MEX call returns */
   if (!str)
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidArgument", "failed to read %s", name);
   }
   return str;
}

static double
HYPREDRV_MatlabArgDouble(const mxArray *arg, const char *name)
{
   if (!arg || !mxIsNumeric(arg) || mxIsComplex(arg) || mxIsSparse(arg) ||
       mxGetNumberOfElements(arg) != 1)
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidArgument", "%s must be a real scalar", name);
   }
   return mxGetScalar(arg);
}

static int
HYPREDRV_MatlabArgInt(const mxArray *arg, const char *name)
{
   if (arg && mxIsLogical(arg) && mxGetNumberOfElements(arg) == 1)
   {
      return mxIsLogicalScalarTrue(arg) ? 1 : 0;
   }
   double v = HYPREDRV_MatlabArgDouble(arg, name);
   if (v != (double)(int)v || v > 2147483647.0 || v < -2147483648.0)
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidArgument", "%s must be an integer", name);
   }
   return (int)v;
}

static const double *
HYPREDRV_MatlabArgDenseReal(const mxArray *arg, const char *name, mwSize *n)
{
   if (!arg || mxIsSparse(arg) || !mxIsDouble(arg) || mxIsComplex(arg))
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidArgument",
                        "%s must be a real dense double array", name);
   }
   *n = mxGetNumberOfElements(arg);
   return mxGetPr(arg);
}

static void
HYPREDRV_MatlabArgSparse(const mxArray *arg, const char *name)
{
   if (!arg || !mxIsSparse(arg) || !mxIsDouble(arg) || mxIsComplex(arg))
   {
      mexErrMsgIdAndTxt("hypredrive:InvalidMatrix", "%s must be a real double sparse matrix",
                        name);
   }
}

static int *
HYPREDRV_MatlabArgIntArray(const mxArray *arg, const char *name, int *n_out)
{
   mwSize        n = 0;
   const double *v = HYPREDRV_MatlabArgDenseReal(arg, name, &n);
   int          *out = (int *)mxCalloc(n > 0 ? (size_t)n : 1, sizeof(int));
   if (n > 2147483647)
   {
      mexErrMsgIdAndTxt("hypredrive:IndexOverflow", "%s is too long", name);
   }
   for (mwSize i = 0; i < n; i++)
   {
      if (v[i] != (double)(int)v[i])
      {
         mexErrMsgIdAndTxt("hypredrive:InvalidArgument", "%s must hold integers", name);
      }
      out[i] = (int)v[i];
   }
   *n_out = (int)n;
   return out;
}

static HYPRE_BigInt
HYPREDRV_MatlabSolutionLength(HYPREDRV_t drv)
{
   HYPRE_BigInt len = 0;
   HD_CHECK(HYPREDRV_LinearSystemGetSolutionLength(drv, &len));
   return len;
}

static mxArray *
HYPREDRV_MatlabColumnFromHost(const HYPRE_Complex *data, HYPRE_BigInt n)
{
   mxArray *out = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
   if (n > 0)
   {
      memcpy(mxGetPr(out), data, (size_t)n * sizeof(double));
   }
   return out;
}

/* Returns a handle to a borrowed IJ object, reusing an existing registration. */
static mxArray *
HYPREDRV_MatlabBorrowed(int kind, void *ptr, const mxArray *drv_arg)
{
   uint64_t owner = HYPREDRV_MatlabHandleId(drv_arg);
   if (!ptr)
   {
      return mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
   }
   for (size_t i = 0; i < HYPREDRV_matlab_count; i++)
   {
      HYPREDRV_MatlabEntry *e = &HYPREDRV_matlab_table[i];
      if (e->kind == kind && e->ptr == ptr)
      {
         return HYPREDRV_MatlabMakeHandle(e->id);
      }
   }
   return HYPREDRV_MatlabMakeHandle(HYPREDRV_MatlabRegister(kind, ptr, owner, 0));
}

/*--- command macros --------------------------------------------------------*/

#define HD_UNUSED \
   (void)nlhs;    \
   (void)plhs;    \
   (void)na;      \
   (void)a

/* driver -> void */
#define HD_DEF_DRV(cname, api)                                             \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[]) \
   {                                                                       \
      HD_UNUSED;                                                           \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                     \
      HD_CHECK(api(drv));                                                  \
   }

/* driver, string -> void */
#define HD_DEF_DRV_STR(cname, api)                                           \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])  \
   {                                                                         \
      HD_UNUSED;                                                             \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                       \
      HD_CHECK(api(drv, HYPREDRV_MatlabArgString(a[1], "argument")));        \
   }

/* driver -> int */
#define HD_DEF_DRV_GETINT(cname, api)                                        \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])  \
   {                                                                         \
      (void)nlhs;                                                            \
      (void)na;                                                              \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                       \
      int        v   = 0;                                                    \
      HD_CHECK(api(drv, &v));                                                \
      plhs[0] = mxCreateDoubleScalar((double)v);                             \
   }

/* driver -> double */
#define HD_DEF_DRV_GETDBL(cname, api)                                        \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])  \
   {                                                                         \
      (void)nlhs;                                                            \
      (void)na;                                                              \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                       \
      double     v   = 0.0;                                                  \
      HD_CHECK(api(drv, &v));                                                \
      plhs[0] = mxCreateDoubleScalar(v);                                     \
   }

/* driver, matrix-or-[] -> void */
#define HD_DEF_DRV_MATRIX(cname, api)                                             \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])       \
   {                                                                              \
      HD_UNUSED;                                                                  \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                            \
      void      *m   = HYPREDRV_MatlabArgObject(a[1], HYPREDRV_MATLAB_KIND_MATRIX, \
                                                a[0], 1);                         \
      HD_CHECK(api(drv, (HYPRE_Matrix)m));                                        \
   }

/* driver, vector-or-[] -> void */
#define HD_DEF_DRV_VECTOR(cname, api)                                             \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])       \
   {                                                                              \
      HD_UNUSED;                                                                  \
      HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);                            \
      void      *v   = HYPREDRV_MatlabArgObject(a[1], HYPREDRV_MATLAB_KIND_VECTOR, \
                                                a[0], 1);                         \
      HD_CHECK(api(drv, (HYPRE_Vector)v));                                        \
   }

/* driver -> borrowed matrix/vector handle */
#define HD_DEF_DRV_GETMATRIX(cname, api)                                          \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])       \
   {                                                                              \
      (void)nlhs;                                                                 \
      (void)na;                                                                   \
      HYPREDRV_t   drv = HYPREDRV_MatlabArgDriver(a[0]);                          \
      HYPRE_Matrix m   = NULL;                                                    \
      HD_CHECK(api(drv, &m));                                                     \
      plhs[0] = HYPREDRV_MatlabBorrowed(HYPREDRV_MATLAB_KIND_MATRIX, (void *)m, a[0]); \
   }

#define HD_DEF_DRV_GETVECTOR(cname, api)                                          \
   static void cname(int nlhs, mxArray *plhs[], int na, const mxArray *a[])       \
   {                                                                              \
      (void)nlhs;                                                                 \
      (void)na;                                                                   \
      HYPREDRV_t   drv = HYPREDRV_MatlabArgDriver(a[0]);                          \
      HYPRE_Vector v   = NULL;                                                    \
      HD_CHECK(api(drv, &v));                                                     \
      plhs[0] = HYPREDRV_MatlabBorrowed(HYPREDRV_MATLAB_KIND_VECTOR, (void *)v, a[0]); \
   }

/*--- global / error / print commands ---------------------------------------*/

static void
HYPREDRV_MatlabCmdErrorCodeClear(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_ErrorCodeClear();
}

static void
HYPREDRV_MatlabCmdErrorInvalidValue(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   /* Raises the error state, reports the code, then clears it again. */
   uint32_t code = HYPREDRV_ErrorInvalidValue(HYPREDRV_MatlabArgString(a[0], "message"));
   HYPREDRV_ErrorCodeClear();
   plhs[0] = mxCreateDoubleScalar((double)code);
}

static void
HYPREDRV_MatlabCmdPrintLibInfo(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   int print_datetime = (na > 0) ? HYPREDRV_MatlabArgInt(a[0], "print_datetime") : 0;
   HD_CHECK(HYPREDRV_PrintLibInfo(MPI_COMM_SELF, print_datetime));
}

static void
HYPREDRV_MatlabCmdPrintSystemInfo(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HD_CHECK(HYPREDRV_PrintSystemInfo(MPI_COMM_SELF));
}

static void
HYPREDRV_MatlabCmdPrintExitInfo(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   const char *argv0 = (na > 0) ? HYPREDRV_MatlabArgString(a[0], "argv0") : "hypredrive";
   HD_CHECK(HYPREDRV_PrintExitInfo(MPI_COMM_SELF, argv0));
}

static void
HYPREDRV_MatlabCmdSolverPresetRegister(int nlhs, mxArray *plhs[], int na,
                                       const mxArray *a[])
{
   HD_UNUSED;
   const char *help = (na > 2) ? HYPREDRV_MatlabArgString(a[2], "help") : NULL;
   HD_CHECK(HYPREDRV_SolverPresetRegister(HYPREDRV_MatlabArgString(a[0], "name"),
                                          HYPREDRV_MatlabArgString(a[1], "yaml"), help));
}

static void
HYPREDRV_MatlabCmdPreconPresetRegister(int nlhs, mxArray *plhs[], int na,
                                       const mxArray *a[])
{
   HD_UNUSED;
   const char *help = (na > 2) ? HYPREDRV_MatlabArgString(a[2], "help") : NULL;
   HD_CHECK(HYPREDRV_PreconPresetRegister(HYPREDRV_MatlabArgString(a[0], "name"),
                                          HYPREDRV_MatlabArgString(a[1], "yaml"), help));
}

/*--- driver lifetime -------------------------------------------------------*/

static void
HYPREDRV_MatlabCmdCreate(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   HYPREDRV_t drv = NULL;
   HD_CHECK(HYPREDRV_Create(MPI_COMM_SELF, &drv));
   uint32_t code = HYPREDRV_SetLibraryMode(drv);
   if (code == HYPREDRV_SUCCESS && na > 0)
   {
      /* Empty text selects the interface's quiet default options. */
      char *yaml = HYPREDRV_MatlabGetYaml(a[0]);
      char *argv[2] = {yaml, NULL};
      code          = HYPREDRV_InputArgsParse(1, argv, drv);
      mxFree(yaml);
   }
   if (code != HYPREDRV_SUCCESS)
   {
      (void)HYPREDRV_Destroy(&drv);
      HYPREDRV_MatlabFailCode(code, "HYPREDRV_Create/SetLibraryMode/InputArgsParse");
   }
   plhs[0] = HYPREDRV_MatlabMakeHandle(
      HYPREDRV_MatlabRegister(HYPREDRV_MATLAB_KIND_DRIVER, (void *)drv, 0, 1));
}

static void
HYPREDRV_MatlabCmdParse(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv  = HYPREDRV_MatlabArgDriver(a[0]);
   char      *yaml = HYPREDRV_MatlabGetYaml(a[1]);
   char      *argv[2] = {yaml, NULL};
   uint32_t   code    = HYPREDRV_InputArgsParse(1, argv, drv);
   mxFree(yaml);
   HYPREDRV_MatlabCheck(code, "HYPREDRV_InputArgsParse");
}

static void
HYPREDRV_MatlabCmdDestroy(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_MatlabEntry *e = HYPREDRV_MatlabFindEntry(HYPREDRV_MatlabHandleId(a[0]));
   if (!e)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                          "handle is stale or already destroyed");
   }
   if (e->kind == HYPREDRV_MATLAB_KIND_DRIVER)
   {
      HYPREDRV_MatlabCheck(HYPREDRV_MatlabDestroyDriverEntry(e->id),
                           "HYPREDRV_Destroy");
      return;
   }
   if (!e->owned)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                          "borrowed handles are released with their driver");
   }
   if (e->attached && HYPREDRV_MatlabFindEntry(e->attached))
   {
      HYPREDRV_MatlabFail("hypredrive:ObjectInUse",
                          "object is attached to a live driver; destroy the driver first");
   }
   HYPREDRV_MatlabReleaseObject(e);
   HYPREDRV_MatlabRemoveEntry(e->id);
}

static void
HYPREDRV_MatlabCmdDestroyAll(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_MatlabDestroyAllHandles();
}

static void
HYPREDRV_MatlabCmdIsValid(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   int ok = 0;
   if (mxGetClassID(a[0]) == mxUINT64_CLASS && !mxIsComplex(a[0]) &&
       mxGetNumberOfElements(a[0]) == 1)
   {
      ok = HYPREDRV_MatlabFindEntry(HYPREDRV_MatlabHandleId(a[0])) != NULL;
   }
   plhs[0] = mxCreateLogicalScalar(ok);
}

static void
HYPREDRV_MatlabCmdNumHandles(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   (void)a;
   plhs[0] = mxCreateDoubleScalar((double)HYPREDRV_matlab_count);
}

/*--- matrix / vector objects -----------------------------------------------*/

static void
HYPREDRV_MatlabCmdMatrixFromSparse(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_MatlabArgSparse(a[0], "A");
   if (mxGetM(a[0]) == 0 || mxGetN(a[0]) == 0)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidMatrix", "A must be non-empty");
   }
   HYPRE_BigInt *indptr = NULL, *cols = NULL;
   HYPRE_Real   *data = NULL;
   mwSize        nnz  = 0;
   HYPREDRV_MatlabConvertCscToCsrRect(a[0], &indptr, &cols, &data, &nnz);
   mwSize nrows = mxGetM(a[0]);
   mwSize ncols = mxGetN(a[0]);

   HYPRE_BigInt row_hi = HYPREDRV_MatlabBigIntFromMwSize(nrows - 1, "row_end");
   HYPRE_BigInt col_hi = HYPREDRV_MatlabBigIntFromMwSize(ncols - 1, "col_end");
   HYPRE_Int   *rowlen = (HYPRE_Int *)mxCalloc((size_t)nrows, sizeof(HYPRE_Int));
   HYPRE_BigInt *rows  = (HYPRE_BigInt *)mxCalloc((size_t)nrows, sizeof(HYPRE_BigInt));
   for (mwSize i = 0; i < nrows; i++)
   {
      rowlen[i] = (HYPRE_Int)(indptr[i + 1] - indptr[i]);
      rows[i]   = (HYPRE_BigInt)i;
   }

   HYPRE_IJMatrix mat = NULL;
   int            err = HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, row_hi, 0, col_hi, &mat);
   if (!err) err = HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   if (!err) err = HYPRE_IJMatrixInitialize(mat);
   if (!err && nnz > 0)
   {
      err = HYPRE_IJMatrixSetValues(mat, (HYPRE_Int)nrows, rowlen, rows, cols, data);
   }
   if (!err) err = HYPRE_IJMatrixAssemble(mat);
   if (err)
   {
      if (mat)
      {
         HYPRE_IJMatrixDestroy(mat);
      }
      HYPREDRV_MatlabFail("hypredrive:HypreError", "failed to build the HYPRE IJ matrix");
   }
   plhs[0] = HYPREDRV_MatlabMakeHandle(
      HYPREDRV_MatlabRegister(HYPREDRV_MATLAB_KIND_MATRIX, (void *)mat, 0, 1));
}

static void
HYPREDRV_MatlabCmdVectorFromArray(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   mwSize        n = 0;
   const double *v = HYPREDRV_MatlabArgDenseReal(a[0], "v", &n);
   if (n == 0)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidArgument", "v must be non-empty");
   }
   HYPRE_BigInt  hi  = HYPREDRV_MatlabBigIntFromMwSize(n - 1, "row_end");
   HYPRE_BigInt *idx = (HYPRE_BigInt *)mxCalloc((size_t)n, sizeof(HYPRE_BigInt));
   for (mwSize i = 0; i < n; i++)
   {
      idx[i] = (HYPRE_BigInt)i;
   }
   HYPRE_IJVector vec = NULL;
   int            err = HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, hi, &vec);
   if (!err) err = HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
   if (!err) err = HYPRE_IJVectorInitialize(vec);
   if (!err) err = HYPRE_IJVectorSetValues(vec, (HYPRE_Int)n, idx, (const HYPRE_Complex *)v);
   if (!err) err = HYPRE_IJVectorAssemble(vec);
   if (err)
   {
      if (vec)
      {
         HYPRE_IJVectorDestroy(vec);
      }
      HYPREDRV_MatlabFail("hypredrive:HypreError", "failed to build the HYPRE IJ vector");
   }
   plhs[0] = HYPREDRV_MatlabMakeHandle(
      HYPREDRV_MatlabRegister(HYPREDRV_MATLAB_KIND_VECTOR, (void *)vec, 0, 1));
}

static void
HYPREDRV_MatlabCmdVectorGetValues(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPRE_IJVector vec =
      (HYPRE_IJVector)HYPREDRV_MatlabArgObject(a[0], HYPREDRV_MATLAB_KIND_VECTOR, NULL, 0);
   HYPRE_BigInt lo = 0, hi = 0;
   if (HYPRE_IJVectorGetLocalRange(vec, &lo, &hi))
   {
      HYPREDRV_MatlabFail("hypredrive:HypreError", "failed to query the vector range");
   }
   mwSize        n   = (mwSize)(hi - lo + 1);
   HYPRE_BigInt *idx = (HYPRE_BigInt *)mxCalloc(n > 0 ? (size_t)n : 1, sizeof(HYPRE_BigInt));
   for (mwSize i = 0; i < n; i++)
   {
      idx[i] = lo + (HYPRE_BigInt)i;
   }
   plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
   if (n > 0 && HYPRE_IJVectorGetValues(vec, (HYPRE_Int)n, idx, mxGetPr(plhs[0])))
   {
      HYPREDRV_MatlabFail("hypredrive:HypreError", "failed to read vector values");
   }
}

static void
HYPREDRV_MatlabCmdVectorSetValues(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPRE_IJVector vec =
      (HYPRE_IJVector)HYPREDRV_MatlabArgObject(a[0], HYPREDRV_MATLAB_KIND_VECTOR, NULL, 0);
   mwSize        n = 0;
   const double *v = HYPREDRV_MatlabArgDenseReal(a[1], "v", &n);
   HYPRE_BigInt  lo = 0, hi = 0;
   if (HYPRE_IJVectorGetLocalRange(vec, &lo, &hi) || (mwSize)(hi - lo + 1) != n)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidArgument",
                          "v must match the vector length");
   }
   HYPRE_BigInt *idx = (HYPRE_BigInt *)mxCalloc(n > 0 ? (size_t)n : 1, sizeof(HYPRE_BigInt));
   for (mwSize i = 0; i < n; i++)
   {
      idx[i] = lo + (HYPRE_BigInt)i;
   }
   if (n > 0 && HYPRE_IJVectorSetValues(vec, (HYPRE_Int)n, idx, (const HYPRE_Complex *)v))
   {
      HYPREDRV_MatlabFail("hypredrive:HypreError", "failed to set vector values");
   }
}

/*--- input arguments / presets ---------------------------------------------*/

HD_DEF_DRV_STR(HYPREDRV_MatlabCmdObjectSetName, HYPREDRV_ObjectSetName)
HD_DEF_DRV_GETINT(HYPREDRV_MatlabCmdInputArgsGetWarmup, HYPREDRV_InputArgsGetWarmup)
HD_DEF_DRV_GETINT(HYPREDRV_MatlabCmdInputArgsGetNumRepetitions,
                  HYPREDRV_InputArgsGetNumRepetitions)
HD_DEF_DRV_GETINT(HYPREDRV_MatlabCmdInputArgsGetNumLinearSystems,
                  HYPREDRV_InputArgsGetNumLinearSystems)
HD_DEF_DRV_GETINT(HYPREDRV_MatlabCmdInputArgsGetNumPreconVariants,
                  HYPREDRV_InputArgsGetNumPreconVariants)
HD_DEF_DRV_STR(HYPREDRV_MatlabCmdInputArgsSetPreconPreset,
               HYPREDRV_InputArgsSetPreconPreset)
HD_DEF_DRV_STR(HYPREDRV_MatlabCmdInputArgsSetSolverPreset,
               HYPREDRV_InputArgsSetSolverPreset)

static void
HYPREDRV_MatlabCmdInputArgsSetPreconVariant(int nlhs, mxArray *plhs[], int na,
                                            const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_InputArgsSetPreconVariant(drv, HYPREDRV_MatlabArgInt(a[1], "variant")));
}

/*--- linear system -----------------------------------------------------------*/

HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemBuild, HYPREDRV_LinearSystemBuild)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemReadMatrix, HYPREDRV_LinearSystemReadMatrix)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemResetInitialGuess,
           HYPREDRV_LinearSystemResetInitialGuess)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemReadDofmap, HYPREDRV_LinearSystemReadDofmap)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemPrint, HYPREDRV_LinearSystemPrint)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSystemComputeEigenspectrum,
           HYPREDRV_LinearSystemComputeEigenspectrum)
HD_DEF_DRV_STR(HYPREDRV_MatlabCmdLinearSystemPrintDofmap,
               HYPREDRV_LinearSystemPrintDofmap)

HD_DEF_DRV_MATRIX(HYPREDRV_MatlabCmdLinearSystemSetMatrix, HYPREDRV_LinearSystemSetMatrix)
HD_DEF_DRV_MATRIX(HYPREDRV_MatlabCmdLinearSystemSetPrecMatrix,
                  HYPREDRV_LinearSystemSetPrecMatrix)
HD_DEF_DRV_MATRIX(HYPREDRV_MatlabCmdLinearSystemSetDiscreteGradient,
                  HYPREDRV_LinearSystemSetDiscreteGradient)
HD_DEF_DRV_MATRIX(HYPREDRV_MatlabCmdLinearSystemSetDiscreteCurl,
                  HYPREDRV_LinearSystemSetDiscreteCurl)
HD_DEF_DRV_VECTOR(HYPREDRV_MatlabCmdLinearSystemSetRHS, HYPREDRV_LinearSystemSetRHS)
HD_DEF_DRV_VECTOR(HYPREDRV_MatlabCmdLinearSystemSetInitialGuess,
                  HYPREDRV_LinearSystemSetInitialGuess)
HD_DEF_DRV_VECTOR(HYPREDRV_MatlabCmdLinearSystemSetSolution,
                  HYPREDRV_LinearSystemSetSolution)
HD_DEF_DRV_VECTOR(HYPREDRV_MatlabCmdLinearSystemSetReferenceSolution,
                  HYPREDRV_LinearSystemSetReferenceSolution)

HD_DEF_DRV_GETMATRIX(HYPREDRV_MatlabCmdLinearSystemGetMatrix, HYPREDRV_LinearSystemGetMatrix)
HD_DEF_DRV_GETVECTOR(HYPREDRV_MatlabCmdLinearSystemGetRHS, HYPREDRV_LinearSystemGetRHS)
HD_DEF_DRV_GETVECTOR(HYPREDRV_MatlabCmdLinearSystemGetSolution,
                     HYPREDRV_LinearSystemGetSolution)

static void
HYPREDRV_MatlabCmdSetMatrixFromCsr(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HYPREDRV_MatlabArgSparse(a[1], "A");
   if (mxGetM(a[1]) != mxGetN(a[1]) || mxGetM(a[1]) == 0)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidMatrix", "A must be square and non-empty");
   }
   HYPRE_BigInt *indptr = NULL, *cols = NULL;
   HYPRE_Real   *data = NULL;
   mwSize        nnz  = 0;
   HYPREDRV_MatlabConvertCscToCsr(a[1], &indptr, &cols, &data, &nnz);
   HYPRE_BigInt row_end = HYPREDRV_MatlabBigIntFromMwSize(mxGetM(a[1]) - 1, "row_end");
   HD_CHECK(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, 0, row_end, indptr, cols, data));
}

static void
HYPREDRV_MatlabCmdSetRhsFromArray(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t    drv = HYPREDRV_MatlabArgDriver(a[0]);
   mwSize        n   = 0;
   const double *b   = HYPREDRV_MatlabArgDenseReal(a[1], "b", &n);
   if (n == 0)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidRHS", "b must be non-empty");
   }
   HYPRE_BigInt row_end = HYPREDRV_MatlabBigIntFromMwSize(n - 1, "row_end");
   HD_CHECK(HYPREDRV_LinearSystemSetRHSFromArray(drv, 0, row_end, (const HYPRE_Real *)b));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetCoordinates(int nlhs, mxArray *plhs[], int na,
                                             const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   void      *x   = HYPREDRV_MatlabArgObject(a[1], HYPREDRV_MATLAB_KIND_VECTOR, a[0], 0);
   void      *y   = (na > 2) ? HYPREDRV_MatlabArgObject(a[2], HYPREDRV_MATLAB_KIND_VECTOR,
                                                        a[0], 1)
                             : NULL;
   void      *z   = (na > 3) ? HYPREDRV_MatlabArgObject(a[3], HYPREDRV_MATLAB_KIND_VECTOR,
                                                        a[0], 1)
                             : NULL;
   HD_CHECK(HYPREDRV_LinearSystemSetCoordinates(drv, (HYPRE_Vector)x, (HYPRE_Vector)y,
                                                (HYPRE_Vector)z));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetCoarseSchur(int nlhs, mxArray *plhs[], int na,
                                             const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   int        lvl = HYPREDRV_MatlabArgInt(a[1], "level");
   void      *m   = HYPREDRV_MatlabArgObject(a[2], HYPREDRV_MATLAB_KIND_MATRIX, a[0], 1);
   HD_CHECK(HYPREDRV_LinearSystemSetCoarseSchur(drv, lvl, (HYPRE_Matrix)m));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetDofmap(int nlhs, mxArray *plhs[], int na,
                                        const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   int        n   = 0;
   int       *map = HYPREDRV_MatlabArgIntArray(a[1], "dofmap", &n);
   HD_CHECK(HYPREDRV_LinearSystemSetDofmap(drv, n, map));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetInterleavedDofmap(int nlhs, mxArray *plhs[], int na,
                                                   const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_LinearSystemSetInterleavedDofmap(
      drv, HYPREDRV_MatlabArgInt(a[1], "num_local_blocks"),
      HYPREDRV_MatlabArgInt(a[2], "num_dof_types")));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetContiguousDofmap(int nlhs, mxArray *plhs[], int na,
                                                  const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_LinearSystemSetContiguousDofmap(
      drv, HYPREDRV_MatlabArgInt(a[1], "num_local_blocks"),
      HYPREDRV_MatlabArgInt(a[2], "num_dof_types")));
}

/* values is num_entries-by-num_components (column-major == component-major). */
static const double *
HYPREDRV_MatlabArgModes(const mxArray *arg, int *num_entries, int *num_components)
{
   mwSize n = 0;
   const double *v = HYPREDRV_MatlabArgDenseReal(arg, "values", &n);
   if (mxGetNumberOfDimensions(arg) != 2 || mxGetM(arg) > 2147483647 ||
       mxGetN(arg) > 2147483647)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidArgument",
                          "values must be a num_entries-by-num_components matrix");
   }
   *num_entries    = (int)mxGetM(arg);
   *num_components = (int)mxGetN(arg);
   return v;
}

static void
HYPREDRV_MatlabCmdLinearSystemSetNearNullSpace(int nlhs, mxArray *plhs[], int na,
                                               const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   int        ne = 0, nc = 0;
   const double *v = HYPREDRV_MatlabArgModes(a[1], &ne, &nc);
   HD_CHECK(HYPREDRV_LinearSystemSetNearNullSpace(drv, ne, nc, (const HYPRE_Complex *)v));
}

static void
HYPREDRV_MatlabCmdLinearSystemSetNullSpace(int nlhs, mxArray *plhs[], int na,
                                           const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   int        ne = 0, nc = 0;
   const double *v = HYPREDRV_MatlabArgModes(a[1], &ne, &nc);
   HD_CHECK(HYPREDRV_LinearSystemSetNullSpace(drv, ne, nc, (const HYPRE_Complex *)v));
}

static void
HYPREDRV_MatlabCmdGetSolutionValues(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_t           drv  = HYPREDRV_MatlabArgDriver(a[0]);
   HYPRE_Complex       *data = NULL;
   HYPRE_MemoryLocation loc  = HYPRE_MEMORY_HOST;
   HYPRE_BigInt         n    = HYPREDRV_MatlabSolutionLength(drv);
   HD_CHECK(HYPREDRV_LinearSystemGetSolutionValues(drv, &data, &loc));
   if (loc != HYPRE_MEMORY_HOST)
   {
      HYPREDRV_MatlabFail("hypredrive:UnsupportedMemory",
                          "the solution lives in device memory; host HYPRE builds only");
   }
   plhs[0] = HYPREDRV_MatlabColumnFromHost(data, n);
}

static void
HYPREDRV_MatlabCmdGetRhsValues(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_t     drv  = HYPREDRV_MatlabArgDriver(a[0]);
   HYPRE_Complex *data = NULL;
   /* The RHS has the same local length as the solution. */
   HYPRE_BigInt n = HYPREDRV_MatlabSolutionLength(drv);
   HD_CHECK(HYPREDRV_LinearSystemGetRHSValues(drv, &data));
   plhs[0] = HYPREDRV_MatlabColumnFromHost(data, n);
}

static void
HYPREDRV_MatlabCmdGetSolutionNorm(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   HYPREDRV_t drv  = HYPREDRV_MatlabArgDriver(a[0]);
   const char *type = (na > 1) ? HYPREDRV_MatlabArgString(a[1], "norm_type") : "l2";
   double      norm = 0.0;
   HD_CHECK(HYPREDRV_LinearSystemGetSolutionNorm(drv, type, &norm));
   plhs[0] = mxCreateDoubleScalar(norm);
}

static void
HYPREDRV_MatlabCmdGetSolutionLength(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   plhs[0] = mxCreateDoubleScalar(
      (double)HYPREDRV_MatlabSolutionLength(HYPREDRV_MatlabArgDriver(a[0])));
}

static void
HYPREDRV_MatlabCmdCopySolutionValues(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_t   drv = HYPREDRV_MatlabArgDriver(a[0]);
   HYPRE_BigInt n   = HYPREDRV_MatlabSolutionLength(drv);
   plhs[0]          = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
   HD_CHECK(HYPREDRV_LinearSystemCopySolutionValues(drv, n, mxGetPr(plhs[0])));
}

/*--- state vectors -------------------------------------------------------------*/

static void
HYPREDRV_MatlabCmdStateVectorSet(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   if (mxGetClassID(a[1]) != mxUINT64_CLASS || mxIsComplex(a[1]) ||
       mxGetNumberOfElements(a[1]) == 0 || mxGetNumberOfElements(a[1]) > 1024)
   {
      HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                          "state vectors must be a uint64 array of vector handles");
   }
   int             n  = (int)mxGetNumberOfElements(a[1]);
   const uint64_t *ids = (const uint64_t *)mxGetData(a[1]);
   HYPRE_IJVector *vecs = (HYPRE_IJVector *)mxCalloc((size_t)n, sizeof(HYPRE_IJVector));
   uint64_t        did  = HYPREDRV_MatlabHandleId(a[0]);
   for (int i = 0; i < n; i++)
   {
      HYPREDRV_MatlabEntry *e = HYPREDRV_MatlabFindEntry(ids[i]);
      if (!e || e->kind != HYPREDRV_MATLAB_KIND_VECTOR)
      {
         HYPREDRV_MatlabFail("hypredrive:InvalidHandle",
                             "state vector handle is stale or not a vector");
      }
      vecs[i]     = (HYPRE_IJVector)e->ptr;
      e->attached = did;
   }
   HD_CHECK(HYPREDRV_StateVectorSet(drv, n, vecs));
}

static void
HYPREDRV_MatlabCmdStateVectorGetValues(int nlhs, mxArray *plhs[], int na,
                                       const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_t     drv  = HYPREDRV_MatlabArgDriver(a[0]);
   HYPRE_Complex *data = NULL;
   HYPRE_BigInt   n    = HYPREDRV_MatlabSolutionLength(drv);
   HD_CHECK(HYPREDRV_StateVectorGetValues(drv, HYPREDRV_MatlabArgInt(a[1], "index"),
                                          &data));
   plhs[0] = HYPREDRV_MatlabColumnFromHost(data, n);
}

static void
HYPREDRV_MatlabCmdStateVectorCopy(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_StateVectorCopy(drv, HYPREDRV_MatlabArgInt(a[1], "index_in"),
                                     HYPREDRV_MatlabArgInt(a[2], "index_out")));
}

HD_DEF_DRV(HYPREDRV_MatlabCmdStateVectorUpdateAll, HYPREDRV_StateVectorUpdateAll)

static void
HYPREDRV_MatlabCmdStateVectorApplyCorrection(int nlhs, mxArray *plhs[], int na,
                                             const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_StateVectorApplyCorrection(drv,
                                                HYPREDRV_MatlabArgInt(a[1], "state_idx")));
}

/*--- preconditioner and solver ---------------------------------------------*/

HD_DEF_DRV(HYPREDRV_MatlabCmdPreconCreate, HYPREDRV_PreconCreate)
HD_DEF_DRV(HYPREDRV_MatlabCmdPreconSetup, HYPREDRV_PreconSetup)
HD_DEF_DRV(HYPREDRV_MatlabCmdPreconDestroy, HYPREDRV_PreconDestroy)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSolverCreate, HYPREDRV_LinearSolverCreate)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSolverSetup, HYPREDRV_LinearSolverSetup)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSolverApply, HYPREDRV_LinearSolverApply)
HD_DEF_DRV(HYPREDRV_MatlabCmdLinearSolverDestroy, HYPREDRV_LinearSolverDestroy)
HD_DEF_DRV_GETINT(HYPREDRV_MatlabCmdLinearSolverGetNumIter, HYPREDRV_LinearSolverGetNumIter)
HD_DEF_DRV_GETDBL(HYPREDRV_MatlabCmdLinearSolverGetFinalRelativeResidualNorm,
                  HYPREDRV_LinearSolverGetFinalRelativeResidualNorm)
HD_DEF_DRV_GETDBL(HYPREDRV_MatlabCmdLinearSolverGetSetupTime,
                  HYPREDRV_LinearSolverGetSetupTime)
HD_DEF_DRV_GETDBL(HYPREDRV_MatlabCmdLinearSolverGetSolveTime,
                  HYPREDRV_LinearSolverGetSolveTime)

static void
HYPREDRV_MatlabCmdLinearSolverGetConverged(int nlhs, mxArray *plhs[], int na,
                                           const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   int converged = 0;
   HD_CHECK(HYPREDRV_LinearSolverGetConverged(HYPREDRV_MatlabArgDriver(a[0]), &converged));
   plhs[0] = mxCreateLogicalScalar(converged != 0);
}

static void
HYPREDRV_MatlabCmdLinearSolverSetRelativeTol(int nlhs, mxArray *plhs[], int na,
                                             const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   HD_CHECK(HYPREDRV_LinearSolverSetRelativeTol(drv, HYPREDRV_MatlabArgDouble(a[1], "tol")));
}

static void
HYPREDRV_MatlabCmdPreconApply(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   void *b = HYPREDRV_MatlabArgObject(a[1], HYPREDRV_MATLAB_KIND_VECTOR, a[0], 0);
   void *x = HYPREDRV_MatlabArgObject(a[2], HYPREDRV_MATLAB_KIND_VECTOR, a[0], 0);
   HD_CHECK(HYPREDRV_PreconApply(drv, (HYPRE_Vector)b, (HYPRE_Vector)x));
}

static void
HYPREDRV_MatlabCmdSolverInfo(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   HYPREDRV_t drv = HYPREDRV_MatlabArgDriver(a[0]);
   int        iterations = 0, converged = 0;
   double     res = 0.0, setup = 0.0, solve = 0.0, norm = 0.0;
   HD_CHECK(HYPREDRV_LinearSolverGetNumIter(drv, &iterations));
   HD_CHECK(HYPREDRV_LinearSolverGetConverged(drv, &converged));
   HD_CHECK(HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(drv, &res));
   HD_CHECK(HYPREDRV_LinearSolverGetSetupTime(drv, &setup));
   HD_CHECK(HYPREDRV_LinearSolverGetSolveTime(drv, &solve));
   HD_CHECK(HYPREDRV_LinearSystemGetSolutionNorm(drv, "l2", &norm));
   plhs[0] = HYPREDRV_MatlabCreateInfoStruct(iterations, converged, res, setup, solve, norm);
}

/*--- statistics and annotations ---------------------------------------------*/

HD_DEF_DRV(HYPREDRV_MatlabCmdStatsPrint, HYPREDRV_StatsPrint)

static void
HYPREDRV_MatlabCmdStatsLevelPrint(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HD_CHECK(HYPREDRV_StatsLevelPrint(HYPREDRV_MatlabArgDriver(a[0]),
                                     HYPREDRV_MatlabArgInt(a[1], "level")));
}

static void
HYPREDRV_MatlabCmdStatsLevelGetCount(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   int count = 0;
   HD_CHECK(HYPREDRV_StatsLevelGetCount(HYPREDRV_MatlabArgDriver(a[0]),
                                        HYPREDRV_MatlabArgInt(a[1], "level"), &count));
   plhs[0] = mxCreateDoubleScalar((double)count);
}

static void
HYPREDRV_MatlabCmdStatsLevelGetEntry(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   (void)nlhs;
   (void)na;
   int    id = 0, solves = 0, iters = 0;
   double setup = 0.0, solve = 0.0;
   HD_CHECK(HYPREDRV_StatsLevelGetEntry(
      HYPREDRV_MatlabArgDriver(a[0]), HYPREDRV_MatlabArgInt(a[1], "level"),
      HYPREDRV_MatlabArgInt(a[2], "index"), &id, &solves, &iters, &setup, &solve));
   const char *fields[] = {"entry_id", "num_solves", "linear_iters", "setup_time",
                           "solve_time"};
   mxArray    *s        = mxCreateStructMatrix(1, 1, 5, fields);
   mxSetField(s, 0, "entry_id", mxCreateDoubleScalar((double)id));
   mxSetField(s, 0, "num_solves", mxCreateDoubleScalar((double)solves));
   mxSetField(s, 0, "linear_iters", mxCreateDoubleScalar((double)iters));
   mxSetField(s, 0, "setup_time", mxCreateDoubleScalar(setup));
   mxSetField(s, 0, "solve_time", mxCreateDoubleScalar(solve));
   plhs[0] = s;
}

static void
HYPREDRV_MatlabCmdAnnotateBegin(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   int id = (na > 2) ? HYPREDRV_MatlabArgInt(a[2], "id") : -1;
   HD_CHECK(HYPREDRV_AnnotateBegin(HYPREDRV_MatlabArgDriver(a[0]),
                                   HYPREDRV_MatlabArgString(a[1], "name"), id));
}

static void
HYPREDRV_MatlabCmdAnnotateEnd(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   int id = (na > 2) ? HYPREDRV_MatlabArgInt(a[2], "id") : -1;
   HD_CHECK(HYPREDRV_AnnotateEnd(HYPREDRV_MatlabArgDriver(a[0]),
                                 HYPREDRV_MatlabArgString(a[1], "name"), id));
}

static void
HYPREDRV_MatlabCmdAnnotateLevelBegin(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HD_CHECK(HYPREDRV_AnnotateLevelBegin(
      HYPREDRV_MatlabArgDriver(a[0]), HYPREDRV_MatlabArgInt(a[1], "level"),
      HYPREDRV_MatlabArgString(a[2], "name"), HYPREDRV_MatlabArgInt(a[3], "id")));
}

static void
HYPREDRV_MatlabCmdAnnotateLevelEnd(int nlhs, mxArray *plhs[], int na, const mxArray *a[])
{
   HD_UNUSED;
   HD_CHECK(HYPREDRV_AnnotateLevelEnd(
      HYPREDRV_MatlabArgDriver(a[0]), HYPREDRV_MatlabArgInt(a[1], "level"),
      HYPREDRV_MatlabArgString(a[2], "name"), HYPREDRV_MatlabArgInt(a[3], "id")));
}

/*--- dispatcher ----------------------------------------------------------------*/

#define HD_CMD(name, fn, lo, hi) {name, fn, lo, hi}

static const HYPREDRV_MatlabCommand HYPREDRV_matlab_commands[] = {
   /* lifetime, validation */
   HD_CMD("create", HYPREDRV_MatlabCmdCreate, 0, 1),
   HD_CMD("parse", HYPREDRV_MatlabCmdParse, 2, 2),
   HD_CMD("destroy", HYPREDRV_MatlabCmdDestroy, 1, 1),
   HD_CMD("destroy_all", HYPREDRV_MatlabCmdDestroyAll, 0, 0),
   HD_CMD("is_valid", HYPREDRV_MatlabCmdIsValid, 1, 1),
   HD_CMD("num_handles", HYPREDRV_MatlabCmdNumHandles, 0, 0),
   /* errors, banners */
   HD_CMD("error_clear", HYPREDRV_MatlabCmdErrorCodeClear, 0, 0),
   HD_CMD("error_invalid_value", HYPREDRV_MatlabCmdErrorInvalidValue, 1, 1),
   HD_CMD("print_lib_info", HYPREDRV_MatlabCmdPrintLibInfo, 0, 1),
   HD_CMD("print_system_info", HYPREDRV_MatlabCmdPrintSystemInfo, 0, 0),
   HD_CMD("print_exit_info", HYPREDRV_MatlabCmdPrintExitInfo, 0, 1),
   /* IJ matrices and vectors */
   HD_CMD("matrix_from_sparse", HYPREDRV_MatlabCmdMatrixFromSparse, 1, 1),
   HD_CMD("vector_from_array", HYPREDRV_MatlabCmdVectorFromArray, 1, 1),
   HD_CMD("vector_get_values", HYPREDRV_MatlabCmdVectorGetValues, 1, 1),
   HD_CMD("vector_set_values", HYPREDRV_MatlabCmdVectorSetValues, 2, 2),
   /* input args, presets */
   HD_CMD("set_name", HYPREDRV_MatlabCmdObjectSetName, 2, 2),
   HD_CMD("get_warmup", HYPREDRV_MatlabCmdInputArgsGetWarmup, 1, 1),
   HD_CMD("get_num_repetitions", HYPREDRV_MatlabCmdInputArgsGetNumRepetitions, 1, 1),
   HD_CMD("get_num_linear_systems", HYPREDRV_MatlabCmdInputArgsGetNumLinearSystems, 1, 1),
   HD_CMD("get_num_precon_variants", HYPREDRV_MatlabCmdInputArgsGetNumPreconVariants, 1, 1),
   HD_CMD("set_precon_variant", HYPREDRV_MatlabCmdInputArgsSetPreconVariant, 2, 2),
   HD_CMD("set_precon_preset", HYPREDRV_MatlabCmdInputArgsSetPreconPreset, 2, 2),
   HD_CMD("set_solver_preset", HYPREDRV_MatlabCmdInputArgsSetSolverPreset, 2, 2),
   HD_CMD("register_solver_preset", HYPREDRV_MatlabCmdSolverPresetRegister, 2, 3),
   HD_CMD("register_precon_preset", HYPREDRV_MatlabCmdPreconPresetRegister, 2, 3),
   /* linear system */
   HD_CMD("build", HYPREDRV_MatlabCmdLinearSystemBuild, 1, 1),
   HD_CMD("read_matrix", HYPREDRV_MatlabCmdLinearSystemReadMatrix, 1, 1),
   HD_CMD("set_matrix", HYPREDRV_MatlabCmdLinearSystemSetMatrix, 2, 2),
   HD_CMD("set_matrix_csr", HYPREDRV_MatlabCmdSetMatrixFromCsr, 2, 2),
   HD_CMD("set_prec_matrix", HYPREDRV_MatlabCmdLinearSystemSetPrecMatrix, 2, 2),
   HD_CMD("set_discrete_gradient", HYPREDRV_MatlabCmdLinearSystemSetDiscreteGradient, 2, 2),
   HD_CMD("set_discrete_curl", HYPREDRV_MatlabCmdLinearSystemSetDiscreteCurl, 2, 2),
   HD_CMD("set_coordinates", HYPREDRV_MatlabCmdLinearSystemSetCoordinates, 2, 4),
   HD_CMD("set_coarse_schur", HYPREDRV_MatlabCmdLinearSystemSetCoarseSchur, 3, 3),
   HD_CMD("set_rhs", HYPREDRV_MatlabCmdLinearSystemSetRHS, 2, 2),
   HD_CMD("set_rhs_array", HYPREDRV_MatlabCmdSetRhsFromArray, 2, 2),
   HD_CMD("set_initial_guess", HYPREDRV_MatlabCmdLinearSystemSetInitialGuess, 2, 2),
   HD_CMD("reset_initial_guess", HYPREDRV_MatlabCmdLinearSystemResetInitialGuess, 1, 1),
   HD_CMD("set_solution", HYPREDRV_MatlabCmdLinearSystemSetSolution, 2, 2),
   HD_CMD("set_reference_solution", HYPREDRV_MatlabCmdLinearSystemSetReferenceSolution, 2, 2),
   HD_CMD("set_dofmap", HYPREDRV_MatlabCmdLinearSystemSetDofmap, 2, 2),
   HD_CMD("set_interleaved_dofmap", HYPREDRV_MatlabCmdLinearSystemSetInterleavedDofmap, 3, 3),
   HD_CMD("set_contiguous_dofmap", HYPREDRV_MatlabCmdLinearSystemSetContiguousDofmap, 3, 3),
   HD_CMD("read_dofmap", HYPREDRV_MatlabCmdLinearSystemReadDofmap, 1, 1),
   HD_CMD("print_dofmap", HYPREDRV_MatlabCmdLinearSystemPrintDofmap, 2, 2),
   HD_CMD("print", HYPREDRV_MatlabCmdLinearSystemPrint, 1, 1),
   HD_CMD("set_near_null_space", HYPREDRV_MatlabCmdLinearSystemSetNearNullSpace, 2, 2),
   HD_CMD("set_null_space", HYPREDRV_MatlabCmdLinearSystemSetNullSpace, 2, 2),
   HD_CMD("compute_eigenspectrum", HYPREDRV_MatlabCmdLinearSystemComputeEigenspectrum, 1, 1),
   HD_CMD("get_matrix", HYPREDRV_MatlabCmdLinearSystemGetMatrix, 1, 1),
   HD_CMD("get_rhs", HYPREDRV_MatlabCmdLinearSystemGetRHS, 1, 1),
   HD_CMD("get_solution", HYPREDRV_MatlabCmdLinearSystemGetSolution, 1, 1),
   HD_CMD("get_solution_values", HYPREDRV_MatlabCmdGetSolutionValues, 1, 1),
   HD_CMD("copy_solution_values", HYPREDRV_MatlabCmdCopySolutionValues, 1, 1),
   HD_CMD("get_solution_length", HYPREDRV_MatlabCmdGetSolutionLength, 1, 1),
   HD_CMD("get_solution_norm", HYPREDRV_MatlabCmdGetSolutionNorm, 1, 2),
   HD_CMD("get_rhs_values", HYPREDRV_MatlabCmdGetRhsValues, 1, 1),
   /* state vectors */
   HD_CMD("state_set", HYPREDRV_MatlabCmdStateVectorSet, 2, 2),
   HD_CMD("state_get_values", HYPREDRV_MatlabCmdStateVectorGetValues, 2, 2),
   HD_CMD("state_copy", HYPREDRV_MatlabCmdStateVectorCopy, 3, 3),
   HD_CMD("state_update_all", HYPREDRV_MatlabCmdStateVectorUpdateAll, 1, 1),
   HD_CMD("state_apply_correction", HYPREDRV_MatlabCmdStateVectorApplyCorrection, 2, 2),
   /* preconditioner and solver */
   HD_CMD("precon_create", HYPREDRV_MatlabCmdPreconCreate, 1, 1),
   HD_CMD("precon_setup", HYPREDRV_MatlabCmdPreconSetup, 1, 1),
   HD_CMD("precon_apply", HYPREDRV_MatlabCmdPreconApply, 3, 3),
   HD_CMD("precon_destroy", HYPREDRV_MatlabCmdPreconDestroy, 1, 1),
   HD_CMD("solver_create", HYPREDRV_MatlabCmdLinearSolverCreate, 1, 1),
   HD_CMD("solver_setup", HYPREDRV_MatlabCmdLinearSolverSetup, 1, 1),
   HD_CMD("solver_apply", HYPREDRV_MatlabCmdLinearSolverApply, 1, 1),
   HD_CMD("solver_destroy", HYPREDRV_MatlabCmdLinearSolverDestroy, 1, 1),
   HD_CMD("solver_set_relative_tol", HYPREDRV_MatlabCmdLinearSolverSetRelativeTol, 2, 2),
   HD_CMD("solver_num_iter", HYPREDRV_MatlabCmdLinearSolverGetNumIter, 1, 1),
   HD_CMD("solver_converged", HYPREDRV_MatlabCmdLinearSolverGetConverged, 1, 1),
   HD_CMD("solver_final_res_norm",
          HYPREDRV_MatlabCmdLinearSolverGetFinalRelativeResidualNorm, 1, 1),
   HD_CMD("solver_setup_time", HYPREDRV_MatlabCmdLinearSolverGetSetupTime, 1, 1),
   HD_CMD("solver_solve_time", HYPREDRV_MatlabCmdLinearSolverGetSolveTime, 1, 1),
   HD_CMD("solver_info", HYPREDRV_MatlabCmdSolverInfo, 1, 1),
   /* statistics, annotations */
   HD_CMD("stats_print", HYPREDRV_MatlabCmdStatsPrint, 1, 1),
   HD_CMD("stats_level_count", HYPREDRV_MatlabCmdStatsLevelGetCount, 2, 2),
   HD_CMD("stats_level_entry", HYPREDRV_MatlabCmdStatsLevelGetEntry, 3, 3),
   HD_CMD("stats_level_print", HYPREDRV_MatlabCmdStatsLevelPrint, 2, 2),
   HD_CMD("annotate_begin", HYPREDRV_MatlabCmdAnnotateBegin, 2, 3),
   HD_CMD("annotate_end", HYPREDRV_MatlabCmdAnnotateEnd, 2, 3),
   HD_CMD("annotate_level_begin", HYPREDRV_MatlabCmdAnnotateLevelBegin, 4, 4),
   HD_CMD("annotate_level_end", HYPREDRV_MatlabCmdAnnotateLevelEnd, 4, 4),
   /* one-shot solve is selected by a numeric first argument, not a command */
};

#define HYPREDRV_MATLAB_NUM_COMMANDS \
   (sizeof(HYPREDRV_matlab_commands) / sizeof(HYPREDRV_matlab_commands[0]))

static void
HYPREDRV_MatlabCommands(mxArray *plhs[])
{
   mxArray *names = mxCreateCellMatrix(1, (mwSize)HYPREDRV_MATLAB_NUM_COMMANDS);
   for (size_t i = 0; i < HYPREDRV_MATLAB_NUM_COMMANDS; i++)
   {
      mxSetCell(names, (mwIndex)i, mxCreateString(HYPREDRV_matlab_commands[i].name));
   }
   plhs[0] = names;
}

static void
HYPREDRV_MatlabDispatch(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   char *name = HYPREDRV_MatlabArgString(prhs[0], "command");
   if (strcmp(name, "commands") == 0)
   {
      HYPREDRV_MatlabCommands(plhs);
      return;
   }
   for (size_t i = 0; i < HYPREDRV_MATLAB_NUM_COMMANDS; i++)
   {
      const HYPREDRV_MatlabCommand *c = &HYPREDRV_matlab_commands[i];
      if (strcmp(name, c->name) != 0)
      {
         continue;
      }
      int na = nrhs - 1;
      if (na < c->min_args || na > c->max_args)
      {
         mexErrMsgIdAndTxt("hypredrive:InvalidNumInputs",
                           "command '%s' takes %d to %d arguments, got %d", c->name,
                           c->min_args, c->max_args, na);
      }
      HYPREDRV_MatlabEnsureInitialized();
      c->fn(nlhs, plhs, na, prhs + 1);
      return;
   }
   mexErrMsgIdAndTxt("hypredrive:UnknownCommand", "unknown hypredrive_mex command '%s'",
                     name);
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   if (nrhs >= 1 && mxIsChar(prhs[0]))
   {
      HYPREDRV_MatlabDispatch(nlhs, plhs, nrhs, prhs);
      return;
   }
   HYPREDRV_MatlabOneShot(nlhs, plhs, nrhs, prhs);
}
