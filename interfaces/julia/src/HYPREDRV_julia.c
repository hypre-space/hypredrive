/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "internal/bridge_row_bounds.h"

#include <mpi.h>

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
HYPREDRV_JuliaBigIntFromI64(int64_t value, const char *name, HYPRE_BigInt *out)
{
   if (!hypredrv_BigIntFromI64(value, out))
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
   uint32_t code = hypredrv_BridgeRowBoundsFromI64(row_start_i64, row_end_i64, &row_start,
                                                   &row_end, HYPREDRV_JuliaBigIntFromI64);
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
   code = HYPREDRV_AnnotateBegin(hypredrv, "matrix", -1);
   if (code != 0u)
   {
      return code;
   }
   code = HYPREDRV_LinearSystemSetMatrixFromCSR(
      hypredrv, row_start, row_end, (const HYPRE_BigInt *)indptr,
      (const HYPRE_BigInt *)col_indices, (const HYPRE_Real *)data);
   uint32_t end_code = HYPREDRV_AnnotateEnd(hypredrv, "matrix", -1);
   return code != 0u ? code : end_code;
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
   uint32_t code = hypredrv_BridgeRowBoundsFromI64(row_start_i64, row_end_i64, &row_start,
                                                   &row_end, HYPREDRV_JuliaBigIntFromI64);
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

/** @brief Set the Krylov relative tolerance for the next solve. */
uint32_t
HYPREDRV_JuliaLinearSolverSetRelativeTol(HYPREDRV_t hypredrv, double tol)
{
   return HYPREDRV_LinearSolverSetRelativeTol(hypredrv, tol);
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

/** @brief Copy local solution values into a caller-owned host array. */
uint32_t
HYPREDRV_JuliaCopySolutionValues(HYPREDRV_t hypredrv, void *sol_data, int64_t sol_length)
{
   HYPRE_BigInt count;
   uint32_t     code = HYPREDRV_JuliaBigIntFromI64(sol_length, "solution length", &count);
   if (code) return code;
   return HYPREDRV_LinearSystemCopySolutionValues(hypredrv, count,
                                                  (HYPRE_Complex *)sol_data);
}

/** @brief Print the HYPREDRV error chain for a nonzero error code. */
void
HYPREDRV_JuliaErrorCodeDescribe(uint32_t error_code)
{
   HYPREDRV_ErrorCodeDescribe(error_code);
}

/* ---------------------------------------------------------------------------
 * Additional shims covering the remaining public HYPREDRV API.
 *
 * Opaque HYPRE objects (HYPRE_Matrix, HYPRE_Vector, HYPRE_IJVector) cross the
 * ABI as void pointers. MPI_Comm arguments are selected with use_world:
 * nonzero maps to MPI_COMM_WORLD, zero to MPI_COMM_SELF.
 * ------------------------------------------------------------------------- */

static MPI_Comm
HYPREDRV_JuliaComm(int use_world)
{
   return use_world ? MPI_COMM_WORLD : MPI_COMM_SELF;
}

/** @brief Clear the process-global HYPREDRV error state. */
void
HYPREDRV_JuliaErrorCodeClear(void)
{
   HYPREDRV_ErrorCodeClear();
}

/** @brief Print library information. */
uint32_t
HYPREDRV_JuliaPrintLibInfo(int use_world, int print_datetime)
{
   return HYPREDRV_PrintLibInfo(HYPREDRV_JuliaComm(use_world), print_datetime);
}

/** @brief Print system information. */
uint32_t
HYPREDRV_JuliaPrintSystemInfo(int use_world)
{
   return HYPREDRV_PrintSystemInfo(HYPREDRV_JuliaComm(use_world));
}

/** @brief Print exit information. */
uint32_t
HYPREDRV_JuliaPrintExitInfo(int use_world, const char *argv0)
{
   return HYPREDRV_PrintExitInfo(HYPREDRV_JuliaComm(use_world), argv0);
}

/** @brief Set (or clear with NULL) the display name of a HYPREDRV object. */
uint32_t
HYPREDRV_JuliaObjectSetName(HYPREDRV_t hypredrv, const char *name)
{
   return HYPREDRV_ObjectSetName(hypredrv, name);
}

/** @brief Return the warmup count from the parsed input. */
uint32_t
HYPREDRV_JuliaInputArgsGetWarmup(HYPREDRV_t hypredrv, int *warmup)
{
   return HYPREDRV_InputArgsGetWarmup(hypredrv, warmup);
}

/** @brief Return the number of repetitions from the parsed input. */
uint32_t
HYPREDRV_JuliaInputArgsGetNumRepetitions(HYPREDRV_t hypredrv, int *num_reps)
{
   return HYPREDRV_InputArgsGetNumRepetitions(hypredrv, num_reps);
}

/** @brief Return the number of linear systems from the parsed input. */
uint32_t
HYPREDRV_JuliaInputArgsGetNumLinearSystems(HYPREDRV_t hypredrv, int *num_ls)
{
   return HYPREDRV_InputArgsGetNumLinearSystems(hypredrv, num_ls);
}

/** @brief Return the number of preconditioner variants from the parsed input. */
uint32_t
HYPREDRV_JuliaInputArgsGetNumPreconVariants(HYPREDRV_t hypredrv, int *num_variants)
{
   return HYPREDRV_InputArgsGetNumPreconVariants(hypredrv, num_variants);
}

/** @brief Select the active preconditioner variant. */
uint32_t
HYPREDRV_JuliaInputArgsSetPreconVariant(HYPREDRV_t hypredrv, int variant_idx)
{
   return HYPREDRV_InputArgsSetPreconVariant(hypredrv, variant_idx);
}

/** @brief Select a registered preconditioner preset. */
uint32_t
HYPREDRV_JuliaInputArgsSetPreconPreset(HYPREDRV_t hypredrv, const char *preset)
{
   return HYPREDRV_InputArgsSetPreconPreset(hypredrv, preset);
}

/** @brief Select a registered solver preset. */
uint32_t
HYPREDRV_JuliaInputArgsSetSolverPreset(HYPREDRV_t hypredrv, const char *preset)
{
   return HYPREDRV_InputArgsSetSolverPreset(hypredrv, preset);
}

/** @brief Register a named solver preset. */
uint32_t
HYPREDRV_JuliaSolverPresetRegister(const char *name, const char *yaml_text,
                                   const char *help)
{
   return HYPREDRV_SolverPresetRegister(name, yaml_text, help);
}

/** @brief Register a named preconditioner preset. */
uint32_t
HYPREDRV_JuliaPreconPresetRegister(const char *name, const char *yaml_text,
                                   const char *help)
{
   return HYPREDRV_PreconPresetRegister(name, yaml_text, help);
}

/** @brief Build the linear system from the parsed input. */
uint32_t
HYPREDRV_JuliaLinearSystemBuild(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemBuild(hypredrv);
}

/** @brief Read the matrix from the parsed input. */
uint32_t
HYPREDRV_JuliaLinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemReadMatrix(hypredrv);
}

/** @brief Attach an opaque HYPRE_Matrix handle. */
uint32_t
HYPREDRV_JuliaLinearSystemSetMatrix(HYPREDRV_t hypredrv, void *mat)
{
   return HYPREDRV_LinearSystemSetMatrix(hypredrv, (HYPRE_Matrix)mat);
}

/** @brief Attach an opaque HYPRE_Matrix as the preconditioner matrix. */
uint32_t
HYPREDRV_JuliaLinearSystemSetPrecMatrix(HYPREDRV_t hypredrv, void *mat)
{
   return HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, (HYPRE_Matrix)mat);
}

/** @brief Attach an opaque discrete gradient matrix. */
uint32_t
HYPREDRV_JuliaLinearSystemSetDiscreteGradient(HYPREDRV_t hypredrv, void *mat)
{
   return HYPREDRV_LinearSystemSetDiscreteGradient(hypredrv, (HYPRE_Matrix)mat);
}

/** @brief Attach an opaque discrete curl matrix. */
uint32_t
HYPREDRV_JuliaLinearSystemSetDiscreteCurl(HYPREDRV_t hypredrv, void *mat)
{
   return HYPREDRV_LinearSystemSetDiscreteCurl(hypredrv, (HYPRE_Matrix)mat);
}

/** @brief Attach opaque coordinate vectors (y and z may be NULL). */
uint32_t
HYPREDRV_JuliaLinearSystemSetCoordinates(HYPREDRV_t hypredrv, void *x, void *y, void *z)
{
   return HYPREDRV_LinearSystemSetCoordinates(hypredrv, (HYPRE_Vector)x, (HYPRE_Vector)y,
                                              (HYPRE_Vector)z);
}

/** @brief Attach an opaque coarse Schur complement matrix at a level. */
uint32_t
HYPREDRV_JuliaLinearSystemSetCoarseSchur(HYPREDRV_t hypredrv, int level, void *mat)
{
   return HYPREDRV_LinearSystemSetCoarseSchur(hypredrv, level, (HYPRE_Matrix)mat);
}

/** @brief Attach an opaque HYPRE_Vector as the right-hand side. */
uint32_t
HYPREDRV_JuliaLinearSystemSetRHS(HYPREDRV_t hypredrv, void *vec)
{
   return HYPREDRV_LinearSystemSetRHS(hypredrv, (HYPRE_Vector)vec);
}

/** @brief Attach an opaque HYPRE_Vector as the solution vector. */
uint32_t
HYPREDRV_JuliaLinearSystemSetSolution(HYPREDRV_t hypredrv, void *vec)
{
   return HYPREDRV_LinearSystemSetSolution(hypredrv, (HYPRE_Vector)vec);
}

/** @brief Attach an opaque HYPRE_Vector as the reference solution. */
uint32_t
HYPREDRV_JuliaLinearSystemSetReferenceSolution(HYPREDRV_t hypredrv, void *vec)
{
   return HYPREDRV_LinearSystemSetReferenceSolution(hypredrv, (HYPRE_Vector)vec);
}

/** @brief Set an interleaved dofmap. */
uint32_t
HYPREDRV_JuliaLinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                               int num_dof_types)
{
   return HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv, num_local_blocks,
                                                    num_dof_types);
}

/** @brief Set a contiguous dofmap. */
uint32_t
HYPREDRV_JuliaLinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                              int num_dof_types)
{
   return HYPREDRV_LinearSystemSetContiguousDofmap(hypredrv, num_local_blocks,
                                                   num_dof_types);
}

/** @brief Read the dofmap from the parsed input. */
uint32_t
HYPREDRV_JuliaLinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemReadDofmap(hypredrv);
}

/** @brief Print the dofmap to a file. */
uint32_t
HYPREDRV_JuliaLinearSystemPrintDofmap(HYPREDRV_t hypredrv, const char *filename)
{
   return HYPREDRV_LinearSystemPrintDofmap(hypredrv, filename);
}

/** @brief Print the linear system. */
uint32_t
HYPREDRV_JuliaLinearSystemPrint(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemPrint(hypredrv);
}

/** @brief Compute the eigenspectrum of the linear system. */
uint32_t
HYPREDRV_JuliaLinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)
{
   return HYPREDRV_LinearSystemComputeEigenspectrum(hypredrv);
}

/** @brief Set near-nullspace modes (component-major, copied by HYPREDRV). */
uint32_t
HYPREDRV_JuliaLinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                           int num_components, const double *values)
{
   return HYPREDRV_LinearSystemSetNearNullSpace(hypredrv, num_entries, num_components,
                                                (const HYPRE_Complex *)values);
}

/** @brief Set nullspace modes (component-major, copied by HYPREDRV). */
uint32_t
HYPREDRV_JuliaLinearSystemSetNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                       int num_components, const double *values)
{
   return HYPREDRV_LinearSystemSetNullSpace(hypredrv, num_entries, num_components,
                                            (const HYPRE_Complex *)values);
}

/** @brief Borrow the native local solution pointer; is_host reports its location. */
uint32_t
HYPREDRV_JuliaLinearSystemGetSolutionValues(HYPREDRV_t hypredrv, double **data,
                                            int *is_host)
{
   if (data == NULL || is_host == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution values output pointer is NULL");
   }
   HYPRE_Complex       *ptr = NULL;
   HYPRE_MemoryLocation loc = HYPRE_MEMORY_HOST;
   uint32_t code = HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &ptr, &loc);
   if (code != 0u)
   {
      return code;
   }
   *data    = (double *)ptr;
   *is_host = (loc == HYPRE_MEMORY_HOST);
   return HYPREDRV_SUCCESS;
}

/** @brief Return the local solution length. */
uint32_t
HYPREDRV_JuliaLinearSystemGetSolutionLength(HYPREDRV_t hypredrv, int64_t *length)
{
   if (length == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution length output pointer is NULL");
   }
   HYPRE_BigInt n    = 0;
   uint32_t     code = HYPREDRV_LinearSystemGetSolutionLength(hypredrv, &n);
   if (code != 0u)
   {
      return code;
   }
   *length = (int64_t)n;
   return HYPREDRV_SUCCESS;
}

/** @brief Borrow the opaque solution HYPRE_Vector. */
uint32_t
HYPREDRV_JuliaLinearSystemGetSolution(HYPREDRV_t hypredrv, void **vec)
{
   if (vec == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("solution vector output pointer is NULL");
   }
   HYPRE_Vector v    = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetSolution(hypredrv, &v);
   if (code != 0u)
   {
      return code;
   }
   *vec = (void *)v;
   return HYPREDRV_SUCCESS;
}

/** @brief Borrow the native local RHS pointer (host memory). */
uint32_t
HYPREDRV_JuliaLinearSystemGetRHSValues(HYPREDRV_t hypredrv, double **data)
{
   if (data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("RHS values output pointer is NULL");
   }
   HYPRE_Complex *ptr  = NULL;
   uint32_t       code = HYPREDRV_LinearSystemGetRHSValues(hypredrv, &ptr);
   if (code != 0u)
   {
      return code;
   }
   *data = (double *)ptr;
   return HYPREDRV_SUCCESS;
}

/** @brief Borrow the opaque RHS HYPRE_Vector. */
uint32_t
HYPREDRV_JuliaLinearSystemGetRHS(HYPREDRV_t hypredrv, void **vec)
{
   if (vec == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("RHS vector output pointer is NULL");
   }
   HYPRE_Vector v    = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetRHS(hypredrv, &v);
   if (code != 0u)
   {
      return code;
   }
   *vec = (void *)v;
   return HYPREDRV_SUCCESS;
}

/** @brief Borrow the opaque HYPRE_Matrix. */
uint32_t
HYPREDRV_JuliaLinearSystemGetMatrix(HYPREDRV_t hypredrv, void **mat)
{
   if (mat == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("matrix output pointer is NULL");
   }
   HYPRE_Matrix m    = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetMatrix(hypredrv, &m);
   if (code != 0u)
   {
      return code;
   }
   *mat = (void *)m;
   return HYPREDRV_SUCCESS;
}

/** @brief Register nstates opaque HYPRE_IJVector state vectors. */
uint32_t
HYPREDRV_JuliaStateVectorSet(HYPREDRV_t hypredrv, int nstates, void **vecs)
{
   if (nstates < 0 || (nstates > 0 && vecs == NULL))
   {
      return HYPREDRV_ErrorInvalidValue("invalid state vector array");
   }
   return HYPREDRV_StateVectorSet(hypredrv, nstates, (HYPRE_IJVector *)vecs);
}

/** @brief Borrow the local host values of a state vector. */
uint32_t
HYPREDRV_JuliaStateVectorGetValues(HYPREDRV_t hypredrv, int index, double **data)
{
   if (data == NULL)
   {
      return HYPREDRV_ErrorInvalidValue("state vector values output pointer is NULL");
   }
   HYPRE_Complex *ptr  = NULL;
   uint32_t       code = HYPREDRV_StateVectorGetValues(hypredrv, index, &ptr);
   if (code != 0u)
   {
      return code;
   }
   *data = (double *)ptr;
   return HYPREDRV_SUCCESS;
}

/** @brief Copy one state vector into another. */
uint32_t
HYPREDRV_JuliaStateVectorCopy(HYPREDRV_t hypredrv, int index_in, int index_out)
{
   return HYPREDRV_StateVectorCopy(hypredrv, index_in, index_out);
}

/** @brief Cycle all state vectors. */
uint32_t
HYPREDRV_JuliaStateVectorUpdateAll(HYPREDRV_t hypredrv)
{
   return HYPREDRV_StateVectorUpdateAll(hypredrv);
}

/** @brief Apply the solution correction to a state vector. */
uint32_t
HYPREDRV_JuliaStateVectorApplyCorrection(HYPREDRV_t hypredrv, int state_idx)
{
   return HYPREDRV_StateVectorApplyCorrection(hypredrv, state_idx);
}

/** @brief Create the preconditioner. */
uint32_t
HYPREDRV_JuliaPreconCreate(HYPREDRV_t hypredrv)
{
   return HYPREDRV_PreconCreate(hypredrv);
}

/** @brief Set up the preconditioner. */
uint32_t
HYPREDRV_JuliaPreconSetup(HYPREDRV_t hypredrv)
{
   return HYPREDRV_PreconSetup(hypredrv);
}

/** @brief Apply the preconditioner to opaque HYPRE_Vector handles. */
uint32_t
HYPREDRV_JuliaPreconApply(HYPREDRV_t hypredrv, void *vec_b, void *vec_x)
{
   return HYPREDRV_PreconApply(hypredrv, (HYPRE_Vector)vec_b, (HYPRE_Vector)vec_x);
}

/** @brief Destroy the preconditioner. */
uint32_t
HYPREDRV_JuliaPreconDestroy(HYPREDRV_t hypredrv)
{
   return HYPREDRV_PreconDestroy(hypredrv);
}

/** @brief Print the statistics summary. */
uint32_t
HYPREDRV_JuliaStatsPrint(HYPREDRV_t hypredrv)
{
   return HYPREDRV_StatsPrint(hypredrv);
}

/** @brief Begin a hierarchical annotation level. */
uint32_t
HYPREDRV_JuliaAnnotateLevelBegin(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   return HYPREDRV_AnnotateLevelBegin(hypredrv, level, name, id);
}

/** @brief End a hierarchical annotation level. */
uint32_t
HYPREDRV_JuliaAnnotateLevelEnd(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   return HYPREDRV_AnnotateLevelEnd(hypredrv, level, name, id);
}

/** @brief Return the number of stats entries at a level. */
uint32_t
HYPREDRV_JuliaStatsLevelGetCount(HYPREDRV_t hypredrv, int level, int *count)
{
   return HYPREDRV_StatsLevelGetCount(hypredrv, level, count);
}

/** @brief Return one stats entry at a level. */
uint32_t
HYPREDRV_JuliaStatsLevelGetEntry(HYPREDRV_t hypredrv, int level, int index, int *entry_id,
                                 int *num_solves, int *linear_iters, double *setup_time,
                                 double *solve_time)
{
   return HYPREDRV_StatsLevelGetEntry(hypredrv, level, index, entry_id, num_solves,
                                      linear_iters, setup_time, solve_time);
}

/** @brief Print the stats summary for a level. */
uint32_t
HYPREDRV_JuliaStatsLevelPrint(HYPREDRV_t hypredrv, int level)
{
   return HYPREDRV_StatsLevelPrint(hypredrv, level);
}
