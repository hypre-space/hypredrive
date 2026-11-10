/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/info.h"
#include "internal/linsys.h"
#include "internal/presets.h"
#include "internal/scaling.h"
#include "internal/stats.h"
#include "internal/utils.h"
#ifdef HYPREDRV_ENABLE_CALIPER
#include <caliper/cali.h>
#endif

/* Undefine autotools package macros from hypre */
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#include "HYPREDRV.h"
#include "HYPREDRV_utils.h"
#include "logging.h"
#include "object.h"

/*=============================================================================
 * File-scope helpers
 *=============================================================================*/

/* Every public operation starts a fresh diagnostic scope. Internal helpers
 * must not use these entry macros: they preserve their caller's failures. */
#define HYPREDRV_CHECK_INIT()            \
   do                                    \
   {                                     \
      hypredrv_ErrorStateReset();        \
      if (hypredrv_ObjectCheckInit())    \
      {                                  \
         return hypredrv_ErrorCodeGet(); \
      }                                  \
   } while (0)

#define HYPREDRV_API_ENTER()                  \
   do                                         \
   {                                          \
      hypredrv_ErrorStateReset();             \
      if (hypredrv_ObjectCheckLive(hypredrv)) \
      {                                       \
         return hypredrv_ErrorCodeGet();      \
      }                                       \
   } while (0)

/* Input arguments are checked at public entry points before private workflows. */
#define HYPREDRV_CHECK_ARGS()                                      \
   if (!hypredrv->iargs)                                           \
   {                                                               \
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                    \
      hypredrv_ErrorMsgAdd("%s: input arguments not parsed (call " \
                           "HYPREDRV_InputArgsParse first)",       \
                           __func__);                              \
      return hypredrv_ErrorCodeGet();                              \
   }

/*=============================================================================
 * Public API
 *=============================================================================*/

/*-----------------------------------------------------------------------------
 * Initialize the HYPREDRV library runtime
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Initialize(void)
{
   hypredrv_ErrorStateReset();
   hypredrv_LogInitializeFromEnv();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Initialize begin");
   hypredrv_ObjectInitialize();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Initialize end (code=0x0)");
   return ERROR_NONE;
}

/*-----------------------------------------------------------------------------
 * Finalize the HYPREDRV library, destroying any objects still alive
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Finalize(void)
{
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Finalize begin");
   hypredrv_ObjectFinalize();

#ifdef HYPREDRV_ENABLE_CALIPER
   /* Flush Caliper data before MPI_Finalize to avoid mpireport warning */
   cali_flush(0);
#endif

   uint32_t code = hypredrv_ErrorCodeGet();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Finalize end (code=0x%x)",
                      code);
   return code;
}

/*-----------------------------------------------------------------------------
 * Describe an error code, printing queued error messages and backtrace
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_ErrorCodeDescribe(uint32_t error_code)
{
   if (!error_code)
   {
      return;
   }

   hypredrv_ErrorCodeDescribe(error_code);
   hypredrv_ErrorMsgPrint();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorBacktracePrint();
}

/*-----------------------------------------------------------------------------
 * Clear the process-global sticky error state
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_ErrorCodeClear(void)
{
   hypredrv_ErrorStateReset();
}

/*-----------------------------------------------------------------------------
 * Public wrapper backing the HYPREDRV_SAFE_CALL* macros
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_SafeCallHandleError(uint32_t error_code, MPI_Comm comm, const char *file,
                             int line, const char *func)
{
   hypredrv_SafeCallHandleError(error_code, comm, file, line, func);
}

/*-----------------------------------------------------------------------------
 * Record and return an invalid-value error with optional context
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_ErrorInvalidValue(const char *message)
{
   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   if (message)
   {
      hypredrv_ErrorMsgAdd("%s", message);
   }
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Create a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *hypredrv_ptr)
{
   int mypid = hypredrv_LogEnabled(1) ? hypredrv_LogRankFromComm(comm) : -1;
   HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create begin");
   HYPREDRV_CHECK_INIT();

   if (!hypredrv_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_Create requires a non-NULL output pointer");
      HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create failed: output pointer is NULL");
      return hypredrv_ErrorCodeGet();
   }

   return hypredrv_ObjectCreate(comm, hypredrv_ptr);
}

/*-----------------------------------------------------------------------------
 * Destroy a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();
   if (!hypredrv_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_Destroy requires a non-NULL HYPREDRV_t pointer");
      return hypredrv_ErrorCodeGet();
   }

   return hypredrv_ObjectDestroy(hypredrv_ptr);
}

/*-----------------------------------------------------------------------------
 * Print library information at entrance
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintLibInfo(MPI_Comm comm, int print_datetime)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintLibInfo(comm, print_datetime);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Print system (hardware/software environment) information
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintSystemInfo(MPI_Comm comm)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintSystemInfo(comm);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Print library information at exit
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintExitInfo(comm, argv0);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Parse YAML input arguments and load auxiliary schedules for an object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_InputArgsParseObject(hypredrv, argc, argv);
}

/*-----------------------------------------------------------------------------
 * Enable library (borrowed-ownership) mode for a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();

   /* Set* calls fix ownership when they run, so the mode must be chosen first. */
   bool attached =
      (bool)(hypredrv->mat_A.handle || hypredrv->mat_M.handle || hypredrv->vec_b.handle ||
             hypredrv->vec_x.handle || hypredrv->vec_x0.handle ||
             hypredrv->vec_xref.handle || hypredrv->mat_G.handle ||
             hypredrv->mat_C.handle || hypredrv->vec_coord[0].handle ||
             hypredrv->vec_coord[1].handle || hypredrv->vec_coord[2].handle);
   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      attached = (bool)(attached || hypredrv->mat_coarse_schur[i].handle);
   }
   if (attached && !hypredrv->lib_mode)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_SetLibraryMode must be called before any matrix or "
                           "vector is attached to the object");
      return hypredrv_ErrorCodeGet();
   }

   hypredrv->lib_mode = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Set or clear an optional display name for a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_ObjectSetName(HYPREDRV_t hypredrv, const char *name)
{
   HYPREDRV_API_ENTER();
   hypredrv_StatsSetObjectName(hypredrv->stats, name);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the warmup setting from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv, int *warmup)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   if (!warmup)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }
   *warmup = hypredrv->iargs->general.warmup;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the number of repetitions from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv, int *num_reps)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   if (!num_reps)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }
   *num_reps = hypredrv->iargs->general.num_repetitions;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the number of linear systems from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv, int *num_ls)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   if (!num_ls)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }
   *num_ls = hypredrv->iargs->ls.num_systems;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the number of preconditioner variants from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumPreconVariants(HYPREDRV_t hypredrv, int *num_variants)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   if (!num_variants)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }
   *num_variants = hypredrv->iargs->num_precon_variants;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Set the active preconditioner variant index
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv, int variant_idx)
{
   HYPREDRV_API_ENTER();
   return hypredrv_InputArgsSetPreconVariant(hypredrv, variant_idx);
}

/*-----------------------------------------------------------------------------
 * Configure the active preconditioner variant using a predefined preset
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_API_ENTER();
   return hypredrv_InputArgsSetPreset(hypredrv, HYPREDRV_PRESET_PRECON, preset);
}

/*-----------------------------------------------------------------------------
 * Configure the solver using a predefined or registered preset name
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetSolverPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_API_ENTER();
   return hypredrv_InputArgsSetPreset(hypredrv, HYPREDRV_PRESET_SOLVER, preset);
}

/*-----------------------------------------------------------------------------
 * Register a custom solver preset
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SolverPresetRegister(const char *name, const char *yaml_text, const char *help)
{
   hypredrv_ErrorStateReset();
   return hypredrv_PresetRegisterTyped(name, yaml_text, help, HYPREDRV_PRESET_SOLVER);
}

/*-----------------------------------------------------------------------------
 * Register a custom preconditioner preset
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconPresetRegister(const char *name, const char *yaml_text, const char *help)
{
   hypredrv_ErrorStateReset();
   return hypredrv_PresetRegisterTyped(name, yaml_text, help, HYPREDRV_PRESET_PRECON);
}

/*-----------------------------------------------------------------------------
 * Set state vectors for time-stepping or multi-state applications
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates, HYPRE_IJVector *vecs)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectSetStateVectors(hypredrv, nstates, vecs);
}

/*-----------------------------------------------------------------------------
 * Retrieve a pointer to the data array of a state vector
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorGetValues(HYPREDRV_t hypredrv, int index, HYPRE_Complex **data_ptr)
{
   HYPREDRV_API_ENTER();
   if (index < 0 || index >= hypredrv->nstates || !data_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_StateVectorGetValues: invalid index %d or output "
                           "pointer (nstates=%d)",
                           index, hypredrv->nstates);
      return hypredrv_ErrorCodeGet();
   }
   *data_ptr = hypredrv_IJVectorData(hypredrv->vec_s[index], HYPRE_MEMORY_HOST);
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Copy one state vector to another
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv, int index_in, int index_out)
{
   HYPREDRV_API_ENTER();
   if (index_in < 0 || index_in >= hypredrv->nstates || index_out < 0 ||
       index_out >= hypredrv->nstates)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_StateVectorCopy: index out of range "
                           "(index_in=%d index_out=%d nstates=%d)",
                           index_in, index_out, hypredrv->nstates);
      return hypredrv_ErrorCodeGet();
   }
   return hypredrv_ObjectCopyStateVectors(hypredrv, index_in, index_out);
}

/*-----------------------------------------------------------------------------
 * Cycle through state vector indices (advance state mapping)
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   if (hypredrv->nstates > 1)
   {
      HYPRE_IJVector first = hypredrv->vec_s[0];
      memmove(hypredrv->vec_s, hypredrv->vec_s + 1,
              (size_t)(hypredrv->nstates - 1) * sizeof(*hypredrv->vec_s));
      hypredrv->vec_s[hypredrv->nstates - 1] = first;
   }
   HYPREDRV_LOG_OBJECTF(3, hypredrv, "rotated %d state vectors", hypredrv->nstates);
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Apply the linear solver correction to a state vector
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv, int state_idx)
{
   HYPREDRV_API_ENTER();
   if (state_idx < 0 || state_idx >= hypredrv->nstates || !hypredrv->vec_x.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_StateVectorApplyCorrection: invalid state index %d "
                           "or missing solution vector (nstates=%d)",
                           state_idx, hypredrv->nstates);
      return hypredrv_ErrorCodeGet();
   }
   return hypredrv_ObjectApplyStateCorrection(hypredrv, state_idx);
}

/*-----------------------------------------------------------------------------
 * Build the linear system (matrix, RHS, LHS) according to the YAML input
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectLinearSystemBuild(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Read the linear system matrix from file
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemReadMatrix(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Set the linear system matrix from a user-supplied handle
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetMatrix(hypredrv, (HYPRE_IJMatrix)mat_A);
}

/*-----------------------------------------------------------------------------
 * Set the discrete gradient operator (G) used by AMS/ADS preconditioners
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDiscreteGradient(HYPREDRV_t hypredrv, HYPRE_Matrix G)
{
   HYPREDRV_API_ENTER();
   return hypredrv_PreconSetDiscreteOperator(hypredrv, PRECON_OPERATOR_GRADIENT,
                                             (HYPRE_IJMatrix)G);
}

/*-----------------------------------------------------------------------------
 * Set the discrete curl operator (C) used by the ADS preconditioner
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDiscreteCurl(HYPREDRV_t hypredrv, HYPRE_Matrix C)
{
   HYPREDRV_API_ENTER();
   return hypredrv_PreconSetDiscreteOperator(hypredrv, PRECON_OPERATOR_CURL,
                                             (HYPRE_IJMatrix)C);
}

/*-----------------------------------------------------------------------------
 * Set the vertex coordinate vectors used by AMS/ADS preconditioners
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetCoordinates(HYPREDRV_t hypredrv, HYPRE_Vector x, HYPRE_Vector y,
                                    HYPRE_Vector z)
{
   HYPREDRV_API_ENTER();
   return hypredrv_PreconSetCoordinates(hypredrv, (HYPRE_IJVector)x, (HYPRE_IJVector)y,
                                        (HYPRE_IJVector)z);
}

/*-----------------------------------------------------------------------------
 * Set the linear system right-hand side vector from a user-supplied handle
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetRHS(hypredrv, (HYPRE_IJVector)vec);
}

/*-----------------------------------------------------------------------------
 * Build the linear-system matrix from per-rank CSR arrays
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrixFromCSR(HYPREDRV_t hypredrv, HYPRE_BigInt row_start,
                                      HYPRE_BigInt row_end, const HYPRE_BigInt *indptr,
                                      const HYPRE_BigInt *col_indices,
                                      const HYPRE_Real   *data)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetMatrixFromCSR(hypredrv, row_start, row_end, indptr,
                                                col_indices, data);
}

/*-----------------------------------------------------------------------------
 * Build the linear-system right-hand side from a per-rank values array
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHSFromArray(HYPREDRV_t hypredrv, HYPRE_BigInt row_start,
                                     HYPRE_BigInt row_end, const HYPRE_Real *values)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end, values);
}

/*-----------------------------------------------------------------------------
 * Set near-nullspace modes for the current linear system (e.g., elastic RBMs)
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                      int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   return hypredrv_LinearSystemSetNullSpaceModes(hypredrv, false, num_entries,
                                                 num_components, values);
}

/*-----------------------------------------------------------------------------
 * Provide an application-assembled coarse (Schur-complement) operator for MGR.
 * Used when the MGR coarse_level_type is "user"; the matrix must be sized to the
 * coarse (C-point) DOF set. Driver mode takes ownership; library mode borrows it.
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetCoarseSchur(HYPREDRV_t hypredrv, int level, HYPRE_Matrix mat_S)
{
   HYPREDRV_API_ENTER();
   return hypredrv_PreconSetCoarseSchur(hypredrv, level, (HYPRE_IJMatrix)mat_S);
}

/*-----------------------------------------------------------------------------
 * Set the exact null space modes of the linear system from a user-supplied array
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                  int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();
   return hypredrv_LinearSystemSetNullSpaceModes(hypredrv, true, num_entries,
                                                 num_components, values);
}

/*-----------------------------------------------------------------------------
 * Set the initial guess for the solution vector of the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetInitialGuess(hypredrv, GUESS_REPLACE,
                                               (HYPRE_IJVector)vec);
}

/*-----------------------------------------------------------------------------
 * Set the solution vector of the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetSolution(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetSolution(hypredrv, (HYPRE_IJVector)vec);
}

/*-----------------------------------------------------------------------------
 * Set or refresh the reference solution used by GMRES error reporting
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetReferenceSolution(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetReferenceSolution(hypredrv, (HYPRE_IJVector)vec);
}

/*-----------------------------------------------------------------------------
 * Reset the initial guess of the solution vector to its original state
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemResetInitialGuess(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Retrieve the solution values from the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data)
{
   HYPREDRV_API_ENTER();

   if (!sol_data || !hypredrv->vec_x.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   *sol_data = hypredrv_IJVectorData(hypredrv->vec_x.handle, HYPRE_MEMORY_HOST);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Return the local solution-vector length
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionLength(HYPREDRV_t hypredrv, HYPRE_BigInt *length)
{
   HYPREDRV_API_ENTER();

   if (!length)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemGetSolutionLength: length must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }
   if (!hypredrv->vec_x.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemGetSolutionLength: no solution available; run "
         "solve() first");
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_BigInt ilower = 0;
   HYPRE_BigInt iupper = -1;
   HYPRE_Int    hypre_ierr =
      HYPRE_IJVectorGetLocalRange(hypredrv->vec_x.handle, &ilower, &iupper);
   if (hypre_ierr != 0)
   {
      char hypre_err_msg[HYPRE_MAX_MSG_LEN];
      HYPRE_DescribeError(hypre_ierr, hypre_err_msg);
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemGetSolutionLength: failed to query solution local "
         "range: %s",
         hypre_err_msg);
      return hypredrv_ErrorCodeGet();
   }

   *length = iupper - ilower + 1;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Compute a norm of the solution vector from the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv, const char *norm_type,
                                     double *norm)
{
   HYPREDRV_API_ENTER();

   if (!norm_type || !norm)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x.handle, norm_type, norm);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the solution vector handle from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolution(HYPREDRV_t hypredrv, HYPRE_Vector *vec)
{
   HYPREDRV_API_ENTER();

   if (!vec || !hypredrv->vec_x.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   *vec = (HYPRE_Vector)hypredrv->vec_x.handle;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the right-hand side values from the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data)
{
   HYPREDRV_API_ENTER();

   if (!rhs_data || !hypredrv->vec_b.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   *rhs_data = hypredrv_IJVectorData(hypredrv->vec_b.handle, HYPRE_MEMORY_UNDEFINED);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the right-hand side vector handle from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHS(HYPREDRV_t hypredrv, HYPRE_Vector *vec)
{
   HYPREDRV_API_ENTER();

   if (!vec || !hypredrv->vec_b.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   *vec = (HYPRE_Vector)hypredrv->vec_b.handle;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Retrieve the system matrix handle from a HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix *mat)
{
   HYPREDRV_API_ENTER();

   if (!mat || !hypredrv->mat_A.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      return hypredrv_ErrorCodeGet();
   }

   *mat = (HYPRE_Matrix)hypredrv->mat_A.handle;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Set the matrix used to compute the preconditioner
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetPrecMatrix(hypredrv, (HYPRE_IJMatrix)mat);
}

/*-----------------------------------------------------------------------------
 * Set the degree-of-freedom map for the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetDofmap(hypredrv, DOFMAP_FROM_ARRAY, size, 0, dofmap);
}

/*-----------------------------------------------------------------------------
 * Set an interleaved degree-of-freedom map for the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                          int num_dof_types)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetDofmap(hypredrv, DOFMAP_INTERLEAVED, num_local_blocks,
                                         num_dof_types, NULL);
}

/*-----------------------------------------------------------------------------
 * Set a contiguous degree-of-freedom map for the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                         int num_dof_types)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetDofmap(hypredrv, DOFMAP_CONTIGUOUS, num_local_blocks,
                                         num_dof_types, NULL);
}

/*-----------------------------------------------------------------------------
 * Read the degree-of-freedom map for the linear system from file
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_LinearSystemSetDofmap(hypredrv, DOFMAP_FROM_INPUT, 0, 0, NULL);
}

/*-----------------------------------------------------------------------------
 * Print the current degree-of-freedom map of the linear system to file
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t hypredrv, const char *filename)
{
   HYPREDRV_API_ENTER();

   if (!filename)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Filename cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv->dofmap || !hypredrv->dofmap->data)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      hypredrv_ErrorMsgAdd("DOF map not set.");
   }
   else
   {
      hypredrv_IntArrayWriteAsciiByRank(hypredrv->comm, hypredrv->dofmap, filename);
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Print the linear system (matrix and RHS) and the DOF map (if present)
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_CHECK_ARGS();

   /* Delegate printing to linsys */
   hypredrv_LinearSystemPrintData(hypredrv->comm, &hypredrv->iargs->ls,
                                  hypredrv->mat_A.handle, hypredrv->vec_b.handle,
                                  hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Create a preconditioner for the object based on the configured method
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectPreconCreate(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Create a linear solver for the object based on the configured method
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectLinearSolverCreate(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Set up the preconditioner for the HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectPreconSetup(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Set up the linear solver, rebuilding or reusing the preconditioner as decided
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectLinearSolverSetup(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Apply the linear solver to solve the linear system
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectLinearSolverApply(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Apply the preconditioner to an input vector
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectPreconApply(hypredrv, (HYPRE_IJVector)vec_b,
                                     (HYPRE_IJVector)vec_x);
}

/*-----------------------------------------------------------------------------
 * Destroy the preconditioner associated with the HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectPreconDestroy(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Destroy the linear solver associated with the HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   return hypredrv_ObjectLinearSolverDestroy(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Print the statistics associated with the HYPREDRV object
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StatsPrint");
   hypredrv_ObjectPrintStats(hypredrv, hypredrv->iargs
                                          ? hypredrv->iargs->general.statistics
                                          : 0); /* GCOVR_EXCL_BR_LINE */
   hypredrv->stats_printed = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Begin annotation of a code region for timing and Caliper instrumentation
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, name,
                                       id);
}

/*-----------------------------------------------------------------------------
 * End annotation of a code region for timing and Caliper instrumentation
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_END, name, id);
}

/*-----------------------------------------------------------------------------
 * Begin hierarchical annotation of a code region with a specified level
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsAnnotateLevelWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN,
                                            level, name, id);
}

/*-----------------------------------------------------------------------------
 * End hierarchical annotation of a code region with a specified level
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsAnnotateLevelWithId(hypredrv->stats, HYPREDRV_ANNOTATE_END, level,
                                            name, id);
}

/*-----------------------------------------------------------------------------
 * Compute the full eigenspectrum of the (preconditioned) system matrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)
{
   HYPREDRV_API_ENTER();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum begin");
#ifdef HYPREDRV_ENABLE_EIGSPEC
   HYPREDRV_CHECK_ARGS();
#endif
   return hypredrv_EigSpecCompute(hypredrv);
}

/*-----------------------------------------------------------------------------
 * Get the iteration count from the last linear solve
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetNumIter(HYPREDRV_t hypredrv, int *iters)
{
   HYPREDRV_API_ENTER();

   if (!iters)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("iters pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *iters = hypredrv_StatsGetLastIter(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Get the convergence flag from the last linear solve
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetConverged(HYPREDRV_t hypredrv, int *converged)
{
   HYPREDRV_API_ENTER();

   if (!converged)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("converged pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("No linear solver is available; call"
                           " HYPREDRV_LinearSolverApply first");
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Int flag = 0;
   hypredrv_SolverGetConverged(hypredrv->solver, &flag);
   *converged = (int)flag;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Get the final relative residual norm from the last linear solve
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(HYPREDRV_t hypredrv, double *norm)
{
   HYPREDRV_API_ENTER();

   if (!norm)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("norm pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("No linear solver is available; call"
                           " HYPREDRV_LinearSolverApply first");
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Real res = 0.0;
   hypredrv_SolverGetFinalRelativeResidualNorm(hypredrv->solver, &res);
   *norm = (double)res;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Get the preconditioner setup time from the last linear solve
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetSetupTime(HYPREDRV_t hypredrv, double *seconds)
{
   HYPREDRV_API_ENTER();

   if (!seconds)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("seconds pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *seconds = hypredrv_StatsGetLastSetupTime(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Get the solver apply time from the last linear solve
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetSolveTime(HYPREDRV_t hypredrv, double *seconds)
{
   HYPREDRV_API_ENTER();

   if (!seconds)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("seconds pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *seconds = hypredrv_StatsGetLastSolveTime(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Get the number of statistics entries recorded at a specific level
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelGetCount(HYPREDRV_t hypredrv, int level, int *count)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsLevelGetCountChecked(hypredrv->stats, level, count,
                                             "HYPREDRV_StatsLevelGetCount");
}

/*-----------------------------------------------------------------------------
 * Get a statistics entry by level and index
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelGetEntry(HYPREDRV_t hypredrv, int level, int index, int *entry_id,
                            int *num_solves, int *linear_iters, double *setup_time,
                            double *solve_time)
{
   HYPREDRV_API_ENTER();

   return hypredrv_StatsLevelGetEntrySummary(hypredrv->stats, level, index, entry_id,
                                             num_solves, linear_iters, setup_time,
                                             solve_time);
}

/*-----------------------------------------------------------------------------
 * Print the statistics summary for a specific level
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelPrint(HYPREDRV_t hypredrv, int level)
{
   HYPREDRV_API_ENTER();

   hypredrv_StatsLevelPrint(hypredrv->stats, level);
   return hypredrv_ErrorCodeGet();
}
