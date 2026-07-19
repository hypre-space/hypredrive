/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"

#include <mpi.h>

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Process-global MPI ownership matches the current Python bridge policy. It is
 * suitable for one Julia runtime per process; do not mix independent MPI owners
 * inside the same process.
 */
static int hypredrv_julia_mpi_owned = 0;

_Static_assert(sizeof(HYPRE_BigInt) == 4u || sizeof(HYPRE_BigInt) == 8u,
               "Julia bridge supports only 32-bit or 64-bit HYPRE_BigInt");
_Static_assert(sizeof(HYPRE_Real) == sizeof(double),
               "Julia bridge requires double-precision HYPRE_Real");
_Static_assert(sizeof(HYPRE_Complex) == sizeof(double),
               "Julia bridge does not support complex HYPRE builds");

static uint32_t
HYPREDRV_JuliaBigIntFromI64(const char *name, int64_t value, HYPRE_BigInt *out)
{
   HYPRE_BigInt converted = (HYPRE_BigInt)value;
   if ((int64_t)converted != value)
   {
      char msg[256];
      int  nwritten = snprintf(
         msg, sizeof(msg), "%s=%" PRId64 " is outside HYPRE_BigInt range", name, value);
      if (nwritten < 0 || (size_t)nwritten >= sizeof(msg))
      {
         return HYPREDRV_ErrorInvalidValue("integer value is outside HYPRE_BigInt range");
      }
      return HYPREDRV_ErrorInvalidValue(msg);
   }

   *out = converted;
   return HYPREDRV_SUCCESS;
}

/** @brief Initialize the HYPREDRV runtime for Julia callers. */
uint32_t
HYPREDRV_JuliaInitialize(void)
{
   return HYPREDRV_Initialize();
}

/** @brief Finalize the HYPREDRV runtime for Julia callers. */
uint32_t
HYPREDRV_JuliaFinalize(void)
{
   return HYPREDRV_Finalize();
}

/** @brief Initialize MPI if Julia/HYPREDRV owns MPI startup. */
uint32_t
HYPREDRV_JuliaMPIInitialize(void)
{
   int finalized = 0;
   int ierr      = MPI_Finalized(&finalized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Finalized failed");
   }
   if (finalized)
   {
      return HYPREDRV_ErrorInvalidValue("MPI has already been finalized");
   }

   int initialized = 0;
   ierr            = MPI_Initialized(&initialized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Initialized failed");
   }
   if (!initialized)
   {
      int provided = 0;
      ierr         = MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
      if (ierr != MPI_SUCCESS)
      {
         return HYPREDRV_ErrorInvalidValue("MPI_Init_thread failed");
      }
      hypredrv_julia_mpi_owned = 1;
   }
   else
   {
      hypredrv_julia_mpi_owned = 0;
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Finalize MPI only if this Julia bridge initialized MPI. */
uint32_t
HYPREDRV_JuliaMPIFinalize(void)
{
   int finalized = 0;
   int ierr      = MPI_Finalized(&finalized);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Finalized failed");
   }
   if (hypredrv_julia_mpi_owned && !finalized)
   {
      ierr = MPI_Finalize();
      if (ierr != MPI_SUCCESS)
      {
         return HYPREDRV_ErrorInvalidValue("MPI_Finalize failed");
      }
      hypredrv_julia_mpi_owned = 0;
   }
   if (finalized)
   {
      hypredrv_julia_mpi_owned = 0;
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Return the byte width of HYPRE_BigInt in this HYPRE build. */
size_t
HYPREDRV_JuliaBigIntSize(void)
{
   return sizeof(HYPRE_BigInt);
}

/** @brief Return the byte width of HYPRE_Real in this HYPRE build. */
size_t
HYPREDRV_JuliaRealSize(void)
{
   return sizeof(HYPRE_Real);
}

/** @brief Return the byte width of HYPRE_Complex solution entries. */
size_t
HYPREDRV_JuliaSolutionEntrySize(void)
{
   return sizeof(HYPRE_Complex);
}

/** @brief Create a HYPREDRV handle on MPI_COMM_SELF. */
uint32_t
HYPREDRV_JuliaCreateWithSelf(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV output pointer is NULL");
   }
   return HYPREDRV_Create(MPI_COMM_SELF, hypredrv_ptr);
}

/** @brief Create a HYPREDRV handle on MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaCreateWithWorld(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV output pointer is NULL");
   }
   return HYPREDRV_Create(MPI_COMM_WORLD, hypredrv_ptr);
}

/** @brief Return the MPI_COMM_WORLD rank. */
uint32_t
HYPREDRV_JuliaWorldCommRank(int *rank)
{
   if (rank == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI rank output pointer is NULL");
   }
   int ierr = MPI_Comm_rank(MPI_COMM_WORLD, rank);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Comm_rank failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Return the MPI_COMM_WORLD size. */
uint32_t
HYPREDRV_JuliaWorldCommSize(int *size)
{
   if (size == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI size output pointer is NULL");
   }
   int ierr = MPI_Comm_size(MPI_COMM_WORLD, size);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Comm_size failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Allreduce a double value with SUM over MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaWorldAllreduceDoubleSum(double value, double *sum)
{
   if (sum == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI allreduce output pointer is NULL");
   }
   int ierr = MPI_Allreduce(&value, sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Allreduce failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Gather variable-length double arrays on MPI_COMM_WORLD. */
uint32_t
HYPREDRV_JuliaWorldAllgathervDouble(const double *send, int send_count, double *recv,
                                    const int *counts, const int *displs)
{
   if (send_count < 0)
   {
      return HYPREDRV_ErrorInvalidValue("send_count is negative");
   }
   if ((send_count > 0 && send == NULL) || recv == NULL || counts == NULL ||
       displs == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("MPI allgatherv pointer is NULL");
   }

   int ierr = MPI_Allgatherv(send, send_count, MPI_DOUBLE, recv, counts, displs,
                             MPI_DOUBLE, MPI_COMM_WORLD);
   if (ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue("MPI_Allgatherv failed");
   }
   return HYPREDRV_SUCCESS;
}

/** @brief Destroy a HYPREDRV handle. */
uint32_t
HYPREDRV_JuliaDestroy(HYPREDRV_t *hypredrv_ptr)
{
   if (hypredrv_ptr == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV pointer is NULL");
   }
   if (*hypredrv_ptr == NULL)
   {
      return HYPREDRV_SUCCESS;
   }
   return HYPREDRV_Destroy(hypredrv_ptr);
}

/** @brief Put a HYPREDRV handle in library mode. */
uint32_t
HYPREDRV_JuliaSetLibraryMode(HYPREDRV_t hypredrv)
{
   return HYPREDRV_SetLibraryMode(hypredrv);
}

/** @brief Parse in-memory YAML text into the HYPREDRV input-argument state. */
uint32_t
HYPREDRV_JuliaInputArgsParseYaml(HYPREDRV_t hypredrv, const char *yaml_text)
{
   if (yaml_text == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("YAML text pointer is NULL");
   }

   char *argv[1] = {(char *)yaml_text};
   return HYPREDRV_InputArgsParse(1, argv, hypredrv);
}

/** @brief Parse a full HYPREDRV argv vector for Julia callers. */
uint32_t
HYPREDRV_JuliaInputArgsParseArgv(HYPREDRV_t hypredrv, int argc, const char **argv)
{
   if (argc <= 0)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV argv is empty");
   }
   if (argv == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV argv pointer is NULL");
   }

   /* Heap-allocate rather than using a caller-sized VLA: VLAs are optional in C11
    * (unsupported by MSVC) and a large argc would overflow the stack. */
   char **parse_argv = (char **)malloc((size_t)argc * sizeof(char *));
   if (!parse_argv)
   {
      return HYPREDRV_ErrorInvalidValue("HYPREDRV failed to allocate argv buffer");
   }
   for (int i = 0; i < argc; i++)
   {
      if (argv[i] == NULL)
      {
         free(parse_argv);
         return HYPREDRV_ErrorInvalidValue("HYPREDRV argv entry is NULL");
      }
      parse_argv[i] = (char *)argv[i];
   }

   uint32_t code = HYPREDRV_InputArgsParse(argc, parse_argv, hypredrv);
   free(parse_argv);
   return code;
}

/** @brief Install a CSR matrix slab from int64 row bounds and HYPRE_BigInt arrays. */
uint32_t
HYPREDRV_JuliaSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start_i64,
                               int64_t row_end_i64, const void *indptr,
                               const void *col_indices, const void *data)
{
   if (indptr == NULL || col_indices == NULL || data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("CSR input pointer is NULL");
   }

   HYPRE_BigInt row_start = 0;
   HYPRE_BigInt row_end   = 0;
   uint32_t code = HYPREDRV_JuliaBigIntFromI64("row_start", row_start_i64, &row_start);
   if (code != 0u)
   {
      return code;
   }
   code = HYPREDRV_JuliaBigIntFromI64("row_end", row_end_i64, &row_end);
   if (code != 0u)
   {
      return code;
   }

   /* Contract: callers must pass indptr/col_indices buffers whose element type
    * has exactly sizeof(HYPRE_BigInt), and data whose element type has exactly
    * sizeof(HYPRE_Real). The public Julia API enforces this before reaching the
    * bridge; these opaque casts keep the C ABI stable across HYPRE integer
    * widths.
    */
   return HYPREDRV_LinearSystemSetMatrixFromCSR(
      hypredrv, row_start, row_end, (const HYPRE_BigInt *)indptr,
      (const HYPRE_BigInt *)col_indices, (const HYPRE_Real *)data);
}

/** @brief Install an RHS slab from int64 row bounds and HYPRE_Real values. */
uint32_t
HYPREDRV_JuliaSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start_i64,
                              int64_t row_end_i64, const void *data)
{
   if (data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("RHS data pointer is NULL");
   }

   HYPRE_BigInt row_start = 0;
   HYPRE_BigInt row_end   = 0;
   uint32_t code = HYPREDRV_JuliaBigIntFromI64("row_start", row_start_i64, &row_start);
   if (code != 0u)
   {
      return code;
   }
   code = HYPREDRV_JuliaBigIntFromI64("row_end", row_end_i64, &row_end);
   if (code != 0u)
   {
      return code;
   }

   /* Contract: callers must pass a data buffer whose element type has exactly
    * sizeof(HYPRE_Real). The public Julia API enforces double precision before
    * reaching the bridge.
    */
   return HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end,
                                               (const HYPRE_Real *)data);
}

/** @brief Install a local integer dofmap. */
uint32_t
HYPREDRV_JuliaSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   if (size < 0)
   {
      return HYPREDRV_ErrorInvalidValue("dofmap size is negative");
   }
   if (size > 0 && dofmap == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("dofmap pointer is NULL");
   }
   return HYPREDRV_LinearSystemSetDofmap(hypredrv, size, dofmap);
}

/** @brief Request a zero initial guess through the HYPREDRV default path. */
uint32_t
HYPREDRV_JuliaSetInitialGuessZero(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL);
}

/** @brief Reset the existing initial guess to zero values. */
uint32_t
HYPREDRV_JuliaResetInitialGuess(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemResetInitialGuess(hypredrv);
}

/** @brief Create the configured linear solver. */
uint32_t
HYPREDRV_JuliaLinearSolverCreate(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSolverCreate(hypredrv);
}

/** @brief Set up the configured linear solver. */
uint32_t
HYPREDRV_JuliaLinearSolverSetup(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSolverSetup(hypredrv);
}

/** @brief Apply the configured linear solver. */
uint32_t
HYPREDRV_JuliaLinearSolverApply(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSolverApply(hypredrv);
}

/** @brief Destroy the configured linear solver. */
uint32_t
HYPREDRV_JuliaLinearSolverDestroy(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSolverDestroy(hypredrv);
}

/** @brief Return iteration count from the last solve. */
uint32_t
HYPREDRV_JuliaLinearSolverGetNumIter(HYPREDRV_t hypredrv, int *iters)
{
   return HYPREDRV_LinearSolverGetNumIter(hypredrv, iters);
}

/** @brief Return the convergence flag from the last solve. */
uint32_t
HYPREDRV_JuliaLinearSolverGetConverged(HYPREDRV_t hypredrv, int *converged)
{
   return HYPREDRV_LinearSolverGetConverged(hypredrv, converged);
}

/** @brief Return the final relative residual norm from the last solve. */
uint32_t
HYPREDRV_JuliaLinearSolverGetFinalRelativeResidualNorm(HYPREDRV_t hypredrv, double *norm)
{
   return HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(hypredrv, norm);
}

/** @brief Return setup time from the last solve. */
uint32_t
HYPREDRV_JuliaLinearSolverGetSetupTime(HYPREDRV_t hypredrv, double *seconds)
{
   return HYPREDRV_LinearSolverGetSetupTime(hypredrv, seconds);
}

/** @brief Return solve time from the last solve. */
uint32_t
HYPREDRV_JuliaLinearSolverGetSolveTime(HYPREDRV_t hypredrv, double *seconds)
{
   return HYPREDRV_LinearSolverGetSolveTime(hypredrv, seconds);
}

/** @brief Return a named norm of the current solution vector. */
uint32_t
HYPREDRV_JuliaGetSolutionNorm(HYPREDRV_t hypredrv, const char *norm_type, double *norm)
{
   return HYPREDRV_LinearSystemGetSolutionNorm(hypredrv, norm_type, norm);
}

/** @brief Return a borrowed solution pointer valid until the next solve/reset or destroy.
 */
uint32_t
HYPREDRV_JuliaGetSolutionValues(HYPREDRV_t hypredrv, const void **sol_data,
                                int64_t *sol_length)
{
   if (sol_data == NULL || sol_length == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution output pointer is NULL");
   }

   HYPRE_BigInt length = 0;
   uint32_t     code   = HYPREDRV_LinearSystemGetSolutionLength(hypredrv, &length);
   if (code != 0u)
   {
      return code;
   }
   if (length < 0)
   {
      return HYPREDRV_ErrorInvalidValue("solution length is negative");
   }

   HYPRE_Complex *values = NULL;
   code                  = HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &values);
   if (code != 0u)
   {
      return code;
   }

   *sol_data   = (const void *)values;
   *sol_length = (int64_t)length;
   return HYPREDRV_SUCCESS;
}

/** @brief Print the HYPREDRV error chain for a nonzero error code. */
void
HYPREDRV_JuliaErrorCodeDescribe(uint32_t error_code)
{
   HYPREDRV_ErrorCodeDescribe(error_code);
}
