/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include "_hypre_parcsr_mv.h" /* For hypre_VectorData, hypre_ParVectorLocalVector */
#include "args.h"
#include "containers.h"
#include "info.h"
#include "linsys.h"
#include "scaling.h"
#include "stats.h"
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

// Flag to check if HYPREDRV is initialized
static bool hypredrv_is_initialized = false;
/* Default stats object used by no-object helper APIs. */
static Stats *hypredrv_default_stats = NULL;

// Macro to check if HYPREDRV is initialized
#define HYPREDRV_CHECK_INIT()                       \
   if (!hypredrv_is_initialized)                    \
   {                                                \
      ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED); \
      return ErrorCodeGet();                        \
   }

// Macro to check if HYPREDRV object is valid
#define HYPREDRV_CHECK_OBJ()                    \
   if (!hypredrv)                               \
   {                                            \
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ); \
      return ErrorCodeGet();                    \
   }

/*-----------------------------------------------------------------------------
 * hypredrv_t data type
 *-----------------------------------------------------------------------------*/

typedef struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   int      nstates;
   int     *states;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix  mat_A;
   HYPRE_IJMatrix  mat_M;
   HYPRE_IJVector  vec_b;
   HYPRE_IJVector  vec_x;
   HYPRE_IJVector  vec_x0;
   HYPRE_IJVector  vec_xref;
   HYPRE_IJVector  vec_nn;
   HYPRE_IJVector *vec_s;

   HYPRE_Precon precon;
   HYPRE_Solver solver;

   Scaling_context *scaling_ctx;
   IntArray        *precon_reuse_timestep_starts;

   Stats *stats;
} hypredrv_t;

/*-----------------------------------------------------------------------------
 * HYPREDRV_Initialize
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Initialize()
{
   if (!hypredrv_is_initialized)
   {
      /* A fresh runtime initialization owns a fresh error-state view. */
      ErrorStateReset();

      /* Initialize hypre */
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Initialize();
#if HYPRE_CHECK_MIN_VERSION(23100, 0)
      HYPRE_DeviceInitialize();
#endif
#endif

#if HYPRE_CHECK_MIN_VERSION(23100, 16)
      /* Check for environment variables */
      const char *env_log_level = getenv("HYPRE_LOG_LEVEL");
      HYPRE_Int   log_level =
         (env_log_level) ? (HYPRE_Int)strtol(env_log_level, NULL, 10) : 0;

      HYPRE_SetLogLevel(log_level);
#endif

      /* Set library state to initialized */
      hypredrv_is_initialized = true;
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Finalize
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Finalize()
{
   if (hypredrv_is_initialized)
   {
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Finalize();
#endif
      hypredrv_is_initialized = false;
   }

   /* Do not leak message buffers across independent initialize/finalize cycles. */
   ErrorStateReset();

#ifdef HYPREDRV_ENABLE_CALIPER
   /* Flush Caliper data before MPI_Finalize to avoid mpireport warning */
   cali_flush(0);
#endif

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_ErrorCodeDescribe
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_ErrorCodeDescribe(uint32_t error_code)
{
   if (!error_code)
   {
      return;
   }

   ErrorCodeDescribe(error_code);
   ErrorMsgPrint();
   ErrorMsgClear();
   ErrorBacktracePrint();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Create
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t hypredrv = (HYPREDRV_t)malloc(sizeof(hypredrv_t));

   MPI_Comm_rank(comm, &hypredrv->mypid);
   MPI_Comm_size(comm, &hypredrv->nprocs);

   hypredrv->comm     = comm;
   hypredrv->nstates  = 0;
   hypredrv->states   = NULL;
   hypredrv->iargs    = NULL;
   hypredrv->mat_A    = NULL;
   hypredrv->mat_M    = NULL;
   hypredrv->vec_b    = NULL;
   hypredrv->vec_x    = NULL;
   hypredrv->vec_x0   = NULL;
   hypredrv->vec_xref = NULL;
   hypredrv->vec_nn   = NULL;
   hypredrv->vec_s    = NULL;
   hypredrv->dofmap   = NULL;

   hypredrv->precon      = NULL;
   hypredrv->solver      = NULL;
   hypredrv->scaling_ctx = NULL;
   hypredrv->precon_reuse_timestep_starts = NULL;
   hypredrv->stats       = NULL;

   /* Disable library mode by default */
   hypredrv->lib_mode = false;

   /* Create statistics object and set as active context */
   hypredrv->stats = StatsCreate();
   if (!hypredrv_default_stats)
   {
      hypredrv_default_stats = hypredrv->stats;
   }

   /* Set output pointer */
   *hypredrv_ptr = hypredrv;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Destroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t hypredrv = *hypredrv_ptr;

   if (!hypredrv)
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return ErrorCodeGet();
   }

   /* Destroy solver/preconditioner objects before tearing down dependent state. */
   if (hypredrv->iargs)
   {
      if (hypredrv->solver)
      {
         SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
      if (hypredrv->precon)
      {
         PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                       &hypredrv->precon);
      }
   }

   /* Destroy scaling context */
   if (hypredrv->scaling_ctx)
   {
      ScalingContextDestroy(&hypredrv->scaling_ctx);
   }

   if (hypredrv->mat_A != hypredrv->mat_M)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_M);
   }
   if (!hypredrv->lib_mode)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_A);
      HYPRE_IJVectorDestroy(hypredrv->vec_b);
      for (int i = 0; i < hypredrv->nstates; i++)
      {
         HYPRE_IJVectorDestroy(hypredrv->vec_s[i]);
      }
   }

   /* Always destroy these vectors since they are created by HYPREDRV. */
   if (hypredrv->vec_x)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x);
   }
   if (hypredrv->vec_x0)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x0);
   }
   if (hypredrv->vec_nn)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_nn);
   }
   if (hypredrv->vec_xref && hypredrv->vec_xref != hypredrv->vec_b)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_xref);
   }

   IntArrayDestroy(&hypredrv->dofmap);
   PreconReuseTimestepsClear(&hypredrv->precon_reuse_timestep_starts);
   InputArgsDestroy(&hypredrv->iargs);

   /* Destroy statistics object */
   if (hypredrv_default_stats == hypredrv->stats)
   {
      hypredrv_default_stats = NULL;
   }
   StatsDestroy(&hypredrv->stats);

   if ((*hypredrv_ptr)->states) free((*hypredrv_ptr)->states);
   if ((*hypredrv_ptr)->vec_s) free((void *)(*hypredrv_ptr)->vec_s);
   free(*hypredrv_ptr);
   *hypredrv_ptr = NULL;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintLibInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintLibInfo(MPI_Comm comm, int print_datetime)
{
   HYPREDRV_CHECK_INIT();

   PrintLibInfo(comm, print_datetime);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintSystemInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintSystemInfo(MPI_Comm comm)
{
   HYPREDRV_CHECK_INIT();

   PrintSystemInfo(comm);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintExitInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   HYPREDRV_CHECK_INIT();

   PrintExitInfo(comm, argv0);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsParse
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* If preset/defaults were configured before parsing, clear old args first. */
   InputArgsDestroy(&hypredrv->iargs);

   InputArgsParse(hypredrv->comm, hypredrv->lib_mode, argc, argv, &hypredrv->iargs);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetLibraryMode
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv->lib_mode = true;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetGlobalOptions
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetGlobalOptions(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Initialize Stats from input args (works for both YAML and preset-based config) */
   StatsSetNumReps(hypredrv->stats, hypredrv->iargs->general.num_repetitions);
   StatsSetNumLinearSystems(hypredrv->stats, hypredrv->iargs->ls.num_systems);
   if (hypredrv->iargs->general.use_millisec)
   {
      StatsTimerSetMilliseconds(hypredrv->stats);
   }
   else
   {
      StatsTimerSetSeconds(hypredrv->stats);
   }

   /* Set HYPRE execution policy */
   if (hypredrv->iargs->general.exec_policy)
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      HYPRE_SetSpGemmUseVendor(0); // TODO: Control this via input option
      HYPRE_SetSpMVUseVendor(0);   // TODO: Control this via input option
#endif
#endif

#ifdef HYPRE_USING_UMPIRE
      /* Setup Umpire pools */
      HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE");
      HYPRE_SetUmpireUMPoolName("HYPRE_UM");
      HYPRE_SetUmpireHostPoolName("HYPRE_HOST");
      HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED");

      HYPRE_SetUmpireDevicePoolSize(hypredrv->iargs->general.dev_pool_size);
      HYPRE_SetUmpireUMPoolSize(hypredrv->iargs->general.uvm_pool_size);
      HYPRE_SetUmpireHostPoolSize(hypredrv->iargs->general.host_pool_size);
      HYPRE_SetUmpirePinnedPoolSize(hypredrv->iargs->general.pinned_pool_size);
#endif
   }
   else
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif
   }

   PreconReuseTimestepsLoad(&hypredrv->iargs->precon_reuse,
                            hypredrv->iargs->ls.timestep_filename,
                            &hypredrv->precon_reuse_timestep_starts);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetWarmup
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->general.warmup : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumRepetitions
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->general.num_repetitions : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumLinearSystems
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->ls.num_systems : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumPreconVariants
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumPreconVariants(HYPREDRV_t hypredrv)
{
   return (hypredrv && hypredrv->iargs) ? hypredrv->iargs->num_precon_variants : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsSetPreconVariant
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv, int variant_idx)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->iargs)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Input arguments not parsed");
      return ErrorCodeGet();
   }

   if (variant_idx < 0 || variant_idx >= hypredrv->iargs->num_precon_variants)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid preconditioner variant index: %d (valid range: 0-%d)",
                  variant_idx, hypredrv->iargs->num_precon_variants - 1);
      return ErrorCodeGet();
   }

   int current_variant = hypredrv->iargs->active_precon_variant;
   int variant_changed = (variant_idx != current_variant);

   /* Only rebuild solver/preconditioner when switching variants. */
   if (variant_changed)
   {
      if (hypredrv->solver)
      {
         SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
      if (hypredrv->precon)
      {
         PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                       &hypredrv->precon);
      }
   }

   /* Set active variant */
   hypredrv->iargs->active_precon_variant = variant_idx;
   hypredrv->iargs->precon_method         = hypredrv->iargs->precon_methods[variant_idx];
   hypredrv->iargs->precon                = hypredrv->iargs->precon_variants[variant_idx];

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsSetPreconPreset
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!preset)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Preconditioner preset name cannot be NULL");
      return ErrorCodeGet();
   }

   if (!hypredrv->iargs)
   {
      InputArgsCreate(hypredrv->lib_mode, &hypredrv->iargs);
      PreconArgsSetDefaultsForMethod(hypredrv->iargs->precon_method,
                                     &hypredrv->iargs->precon);
   }

   int variant_idx = hypredrv->iargs->active_precon_variant;
   if (variant_idx < 0)
   {
      variant_idx                            = 0;
      hypredrv->iargs->active_precon_variant = 0;
   }

   /* Destroy existing preconditioner object if any */
   if (hypredrv->precon)
   {
      PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                    &hypredrv->precon);
   }

   if (hypredrv->iargs->num_precon_variants <= 0)
   {
      hypredrv->iargs->num_precon_variants   = 1;
      hypredrv->iargs->active_precon_variant = 0;
      variant_idx                            = 0;
   }

   if (variant_idx >= hypredrv->iargs->num_precon_variants)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Active preconditioner variant index %d is out of range (0-%d)",
                  variant_idx, hypredrv->iargs->num_precon_variants - 1);
      return ErrorCodeGet();
   }

   InputArgsApplyPreconPreset(hypredrv->iargs, preset, variant_idx);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsSetSolverPreset
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetSolverPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!preset)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Solver preset name cannot be NULL");
      return ErrorCodeGet();
   }

   if (!hypredrv->iargs)
   {
      InputArgsCreate(hypredrv->lib_mode, &hypredrv->iargs);
   }

   /* Validate solver name */
   if (!StrIntMapArrayDomainEntryExists(SolverGetValidTypeIntMap(), preset))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd(
         "Unknown solver preset: '%s'. Valid options: pcg, gmres, fgmres, bicgstab",
         preset);
      return ErrorCodeGet();
   }

   /* Destroy existing solver if any */
   if (hypredrv->solver)
   {
      SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   }

   /* Set solver method and defaults */
   hypredrv->iargs->solver_method =
      (solver_t)StrIntMapArrayGetImage(SolverGetValidTypeIntMap(), preset);
   SolverArgsSetDefaultsForMethod(hypredrv->iargs->solver_method,
                                  &hypredrv->iargs->solver);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StateVectorSet
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates, HYPRE_IJVector *vecs)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv->nstates = nstates;
   hypredrv->states  = (int *)malloc(sizeof(int) * (size_t)nstates);
   hypredrv->vec_s   = (HYPRE_IJVector *)malloc(sizeof(HYPRE_IJVector) * (size_t)nstates);
   for (int i = 0; i < nstates; i++)
   {
      hypredrv->states[i] = i;
      if (vecs && vecs[i])
      {
         hypredrv->vec_s[i] = vecs[i];
      }
      else
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         return ErrorCodeGet();
      }
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorGetValues(HYPREDRV_t hypredrv, int index, HYPRE_Complex **data_ptr)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int             state   = hypredrv->states[index];
   HYPRE_ParVector par_vec = NULL;
   hypre_Vector   *seq_vec = NULL;
   void           *obj     = NULL;

   if (hypredrv->vec_s[state])
   {
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state], &obj);
      par_vec   = (HYPRE_ParVector)obj;
      seq_vec   = hypre_ParVectorLocalVector(par_vec);
      *data_ptr = hypre_VectorData(seq_vec);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Cycle through state vectors
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv, int index_in, int index_out)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int   state_in  = hypredrv->states[index_in];
   int   state_out = hypredrv->states[index_out];
   void *obj_in    = NULL;
   void *obj_out   = NULL;

   if (hypredrv->vec_s[state_in] && hypredrv->vec_s[state_out])
   {
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state_in], &obj_in);
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state_out], &obj_out);

      HYPRE_ParVectorCopy((HYPRE_ParVector)obj_in, (HYPRE_ParVector)obj_out);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Cycle through state vectors
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   for (int i = 0; i < hypredrv->nstates; i++)
   {
      hypredrv->states[i] = (hypredrv->states[i] + 1) % hypredrv->nstates;
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StateVectorApplyCorrection
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   void *obj_s = NULL, *obj_delta = NULL;
   int   current = hypredrv->states[0];

   HYPRE_IJVectorGetObject(hypredrv->vec_x, &obj_delta);
   HYPRE_IJVectorGetObject(hypredrv->vec_s[current], &obj_s);

   /* U = U + Δx */
   HYPRE_ParVectorAxpy((HYPRE_Complex)1.0, (HYPRE_ParVector)obj_delta,
                       (HYPRE_ParVector)obj_s);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemBuild
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, NULL));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetReferenceSolution(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetVectorTags(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */

   /* Reset scaling state for new system */
   if (hypredrv->scaling_ctx)
   {
      hypredrv->scaling_ctx->is_applied = 0;
   }

   long long int num_rows     = LinearSystemMatrixGetNumRows(hypredrv->mat_A);
   long long int num_nonzeros = LinearSystemMatrixGetNumNonzeros(hypredrv->mat_A);
   if (!hypredrv->mypid)
   {
      PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH);
      printf("Solving linear system #%d ", StatsGetLinearSystemID(hypredrv->stats) + 1);
      printf("with %lld rows and %lld nonzeros...\n", num_rows, num_nonzeros);
   }
   HYPRE_ClearAllErrors();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemReadMatrix(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->mat_A,
                          hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Don't annotate "matrix" here - users annotate with "system" in their code */
   /* This was causing build times and solve times to be recorded in separate entries
    */
   hypredrv->mat_A = (HYPRE_IJMatrix)mat_A;
   hypredrv->mat_M = (HYPRE_IJMatrix)mat_A;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      LinearSystemSetRHS(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                         &hypredrv->vec_xref, &hypredrv->vec_b, hypredrv->stats);
   }
   else
   {
      hypredrv->vec_b = (HYPRE_IJVector)vec;
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetNearNullSpace
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                      int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetNearNullSpace(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                num_entries, num_components, values, &hypredrv->vec_nn);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetInitialGuess
 *
 * TODO: add vector as input parameter
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetInitialGuess(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                               hypredrv->vec_b, &hypredrv->vec_x0, &hypredrv->vec_x,
                               hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetReferenceSolution
 *
 * TODO: add vector as input parameter
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetReferenceSolution(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetReferenceSolution(hypredrv->comm, &hypredrv->iargs->ls,
                                    &hypredrv->vec_xref, hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->vec_x0 || !hypredrv->vec_x)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return ErrorCodeGet();
   }

   LinearSystemResetInitialGuess(hypredrv->vec_x0, hypredrv->vec_x, hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetVectorTags
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetVectorTags(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->dofmap || !hypredrv->dofmap->data || hypredrv->dofmap->size == 0)
   {
      return ErrorCodeGet();
   }

   LinearSystemSetVectorTags(hypredrv->vec_b, hypredrv->dofmap);
   LinearSystemSetVectorTags(hypredrv->vec_x, hypredrv->dofmap);
   LinearSystemSetVectorTags(hypredrv->vec_x0, hypredrv->dofmap);
   LinearSystemSetVectorTags(hypredrv->vec_xref, hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!sol_data || !hypredrv->vec_x)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return ErrorCodeGet();
   }

   LinearSystemGetSolutionValues(hypredrv->vec_x, sol_data);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionNorm
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv, const char *norm_type,
                                     double *norm)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!norm_type || !norm)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return ErrorCodeGet();
   }

   LinearSystemComputeVectorNorm(hypredrv->vec_x, norm_type, norm);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetRHSValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!rhs_data || !hypredrv->vec_b)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return ErrorCodeGet();
   }

   LinearSystemGetRHSValues(hypredrv->vec_b, rhs_data);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetPrecMatrix(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                             &hypredrv->mat_M, hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Keep ownership clean when this is called multiple times */
   IntArrayDestroy(&hypredrv->dofmap);
   IntArrayBuild(hypredrv->comm, size, dofmap, &hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetInterleavedDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                          int num_dof_types)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   IntArrayDestroy(&hypredrv->dofmap);
   IntArrayBuildInterleaved(hypredrv->comm, num_local_blocks, num_dof_types,
                            &hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetContiguousDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                         int num_dof_types)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   IntArrayDestroy(&hypredrv->dofmap);
   IntArrayBuildContiguous(hypredrv->comm, num_local_blocks, num_dof_types,
                           &hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemReadDofmap(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->dofmap,
                          hypredrv->stats);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemPrintDofmap
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t hypredrv, const char *filename)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!filename)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Filename cannot be NULL");
      return ErrorCodeGet();
   }

   if (!hypredrv->dofmap || !hypredrv->dofmap->data)
   {
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ErrorMsgAdd("DOF map not set.");
   }
   else
   {
      IntArrayWriteAsciiByRank(hypredrv->comm, hypredrv->dofmap, filename);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemPrint
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Delegate printing to linsys */
   LinearSystemPrintData(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                         hypredrv->vec_b, hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int  next_ls_id   = StatsGetLinearSystemID(hypredrv->stats) + 1;
   bool should_create =
      ((hypredrv->precon == NULL) ||
       PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                  hypredrv->precon_reuse_timestep_starts,
                                  next_ls_id)) != 0;

   if (should_create)
   {
      /* If we're recreating, destroy the existing preconditioner first to avoid leaks. */
      if (hypredrv->precon)
      {
         PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                       &hypredrv->precon);
      }
      PreconCreate(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                   hypredrv->dofmap, hypredrv->vec_nn, &hypredrv->precon);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* First, create the preconditioner if we need */
   if (!hypredrv->precon)
   {
      if (HYPREDRV_PreconCreate(hypredrv))
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         return ErrorCodeGet();
      }
   }

   /* Always recreate solver per linear system.
    * Reuse policy applies to preconditioner lifecycle; solver internals should
    * be rebuilt against the current system vectors/matrix each cycle. */
   if (hypredrv->solver)
   {
      SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   }
   SolverCreate(hypredrv->comm, hypredrv->iargs->solver_method,
                &hypredrv->iargs->solver, &hypredrv->solver);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   PreconSetup(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   return ErrorCodeGet();
}

static void
HYPREDRV_GMRESSetRefSolution(HYPREDRV_t hypredrv)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (!hypredrv || hypredrv->iargs->solver_method != SOLVER_GMRES)
   {
      return;
   }

   void           *obj_ref = NULL;
   HYPRE_ParVector par_ref = NULL;

   if (hypredrv->vec_xref)
   {
      HYPRE_IJVectorGetObject(hypredrv->vec_xref, &obj_ref);
      par_ref = (HYPRE_ParVector)obj_ref;
   }

   HYPRE_ParCSRGMRESSetRefSolution(hypredrv->solver, par_ref);
#else
   (void)hypredrv;
#endif
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int next_ls_id = StatsGetLinearSystemID(hypredrv->stats) + 1;
   int recompute  = PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                               hypredrv->precon_reuse_timestep_starts,
                                               next_ls_id);

   /* Create scaling context if needed and not already created */
   if (hypredrv->iargs->scaling.enabled && !hypredrv->scaling_ctx)
   {
      ScalingContextCreate(&hypredrv->scaling_ctx);
   }

   /* Compute scaling if enabled (recompute for each system if needed) */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      ScalingCompute(hypredrv->comm, &hypredrv->iargs->scaling, hypredrv->scaling_ctx,
                     hypredrv->mat_A, hypredrv->vec_b, hypredrv->dofmap);
      if (ErrorCodeGet())
      {
         return ErrorCodeGet();
      }
   }

   /* Apply scaling before setup if enabled */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A, hypredrv->mat_M,
                           hypredrv->vec_b, hypredrv->vec_x);
      if (ErrorCodeGet())
      {
         return ErrorCodeGet();
      }
   }

   HYPREDRV_GMRESSetRefSolution(hypredrv);

   SolverSetupWithReuse(hypredrv->iargs->precon_method, hypredrv->iargs->solver_method,
                        hypredrv->precon, hypredrv->solver, hypredrv->mat_M,
                        hypredrv->vec_b, hypredrv->vec_x, hypredrv->stats,
                        recompute ? 0 : 1);

   HYPRE_ClearAllErrors();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->solver)
   {
      ErrorCodeSet(ERROR_INVALID_SOLVER);
      return ErrorCodeGet();
   }

   double e_norm = 0.0, x_norm = 0.0, xref_norm = 0.0;
   double b_norm = 0.0, r_norm = 0.0, r0_norm = 0.0;

   /* Ensure GMRES always sees the current reference solution, including on reused
    * preconditioner cycles where SolverSetup may be skipped. */
   HYPREDRV_GMRESSetRefSolution(hypredrv);

   /* Apply scaling if enabled but not yet applied (e.g., when preconditioner is reused)
    */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       !hypredrv->scaling_ctx->is_applied)
   {
      ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A, hypredrv->mat_M,
                           hypredrv->vec_b, hypredrv->vec_x);
      if (ErrorCodeGet())
      {
         return ErrorCodeGet();
      }
   }

   /* Perform solve */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       hypredrv->scaling_ctx->is_applied)
   {
      int xref_scaled = 0;

      /* Compute initial residual norm before solve (on current system state) */
      LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x,
                                      "L2", &r0_norm);

      if (hypredrv->vec_xref)
      {
         ScalingApplyToVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                              SCALING_VECTOR_UNKNOWN);
         if (ErrorCodeGet())
         {
            return ErrorCodeGet();
         }
         xref_scaled = 1;
      }

      StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
      StatsInitialResNormSet(hypredrv->stats, r0_norm);

      /* Solve on scaled system */
      HYPRE_Int iters =
         SolverSolveOnly(hypredrv->iargs->solver_method, hypredrv->solver,
                         hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x);
      if (iters < 0)
      {
         if (xref_scaled)
         {
            ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                SCALING_VECTOR_UNKNOWN);
         }
         StatsIterSet(hypredrv->stats, 0);
         StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");
         return ErrorCodeGet();
      }

      StatsIterSet(hypredrv->stats, (int)iters);
      StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");

      /* Undo scaling on solution and restore original system */
      ScalingUndoOnSystem(hypredrv->scaling_ctx, hypredrv->mat_A, hypredrv->mat_M,
                          hypredrv->vec_b, hypredrv->vec_x);
      if (ErrorCodeGet())
      {
         if (xref_scaled)
         {
            ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                SCALING_VECTOR_UNKNOWN);
         }
         return ErrorCodeGet();
      }

      if (xref_scaled)
      {
         ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                             SCALING_VECTOR_UNKNOWN);
         if (ErrorCodeGet())
         {
            return ErrorCodeGet();
         }
      }

      /* Compute residual norms on original (now restored) system */
      LinearSystemComputeVectorNorm(hypredrv->vec_b, "L2", &b_norm);
      LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x,
                                      "L2", &r_norm);
      b_norm = (b_norm > 0.0) ? b_norm : 1.0;
      StatsRelativeResNormSet(hypredrv->stats, r_norm / b_norm);
   }
   else
   {
      /* No scaling - use standard SolverApply which handles everything including stats */
      SolverApply(hypredrv->iargs->solver_method, hypredrv->solver, hypredrv->mat_A,
                  hypredrv->vec_b, hypredrv->vec_x, hypredrv->stats);
      /* SolverApply already computed and set all stats */
   }

   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   if (hypredrv->vec_xref)
   {
      LinearSystemComputeVectorNorm(hypredrv->vec_xref, "L2", &xref_norm);
      LinearSystemComputeVectorNorm(hypredrv->vec_x, "L2", &x_norm);
      LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, "L2", &e_norm);
      if (!hypredrv->mypid)
      {
         printf("L2 norm of error: %e\n", (double)e_norm);
         printf("L2 norm of solution: %e\n", (double)x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
      }
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   PreconApply(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A,
               (HYPRE_IJVector)vec_b, (HYPRE_IJVector)vec_x);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int  next_ls_id     = StatsGetLinearSystemID(hypredrv->stats) + 1;
   bool should_destroy =
      PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                 hypredrv->precon_reuse_timestep_starts,
                                 next_ls_id) != 0;

   if (should_destroy)
   {
      PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                    &hypredrv->precon);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* First, destroy the preconditioner if we need */
   if (hypredrv->precon)
   {
      if (HYPREDRV_PreconDestroy(hypredrv))
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         return ErrorCodeGet();
      }
   }

   SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsPrint
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   StatsPrint(hypredrv->stats, hypredrv->iargs->general.statistics);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   if (id >= 0)
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   }
   else
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s", name);
   }
   StatsAnnotate(hypredrv_default_stats, HYPREDRV_ANNOTATE_BEGIN, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   if (id >= 0)
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   }
   else
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s", name);
   }
   StatsAnnotate(hypredrv_default_stats, HYPREDRV_ANNOTATE_END, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   StatsAnnotateLevelBegin(hypredrv_default_stats, level, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   StatsAnnotateLevelEnd(hypredrv_default_stats, level, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

#ifdef HYPREDRV_ENABLE_EIGSPEC
static void
hypredrv_PreconApplyWrapper(void *ctx, void *b, void *x)
{
   HYPREDRV_PreconApply((HYPREDRV_t)ctx, (HYPRE_Vector)b, (HYPRE_Vector)x);
}
#endif

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemComputeEigenspectrum
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

#ifdef HYPREDRV_ENABLE_EIGSPEC
   /* Exit early if not computing eigenspectrum */
   if (!hypredrv->iargs->ls.eigspec.enable)
   {
      return ErrorCodeGet();
   }

   if (!hypredrv->mypid)
   {
      printf("[EigenSpectrum] | mode=%s | vectors=%s | prefix='%s'\n",
             hypredrv->iargs->ls.eigspec.hermitian ? "Hermitian" : "General",
             hypredrv->iargs->ls.eigspec.vectors ? "on" : "off",
             hypredrv->iargs->ls.eigspec.output_prefix[0]
                ? hypredrv->iargs->ls.eigspec.output_prefix
                : "eig");
      fflush(stdout);
   }

   /* pass preconditioner apply callback directly */
   if (hypredrv->iargs->ls.eigspec.preconditioned)
   {
      HYPREDRV_PreconCreate(hypredrv);
      HYPREDRV_PreconSetup(hypredrv);

      return hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec,
                                     (void *)hypredrv->mat_A, (void *)hypredrv,
                                     hypredrv_PreconApplyWrapper, hypredrv->stats);
   }
   else
   {
      return hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec,
                                     (void *)hypredrv->mat_A, NULL, NULL,
                                     hypredrv->stats);
   }
#else
   (void)hypredrv;
   ErrorCodeSet(ERROR_UNKNOWN);
   ErrorMsgAdd("Eigenspectrum feature disabled at build time. Reconfigure with "
               "-DHYPREDRV_ENABLE_EIGSPEC=ON");
#endif

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_GetLastStat
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_GetLastStat(HYPREDRV_t hypredrv, const char *name, void *value)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!name || !value)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Stat name and value cannot be NULL");
      return ErrorCodeGet();
   }

   if (!strcmp(name, "iter"))
   {
      *(int *)value = StatsGetLastIter(hypredrv->stats);
   }
   else if (!strcmp(name, "setup"))
   {
      *(double *)value = StatsGetLastSetupTime(hypredrv->stats);
   }
   else if (!strcmp(name, "solve"))
   {
      *(double *)value = StatsGetLastSolveTime(hypredrv->stats);
   }
   else
   {
      // Unknown stat name
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Unknown stat name: '%s'", name);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelGetCount
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_StatsLevelGetCount(int level)
{
   return StatsLevelGetCount(hypredrv_default_stats, level);
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelGetEntry
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_StatsLevelGetEntry(int level, int index, int *entry_id, int *num_solves,
                            int *linear_iters, double *setup_time, double *solve_time)
{
   LevelEntry entry;
   int        ret = StatsLevelGetEntry(hypredrv_default_stats, level, index, &entry);

   if (ret == 0)
   {
      if (entry_id) *entry_id = entry.id;

      /* Compute aggregates from solve index range */
      int    n_solves = entry.solve_end - entry.solve_start;
      int    l_iters  = 0;
      double s_time   = 0.0;
      double p_time   = 0.0;

      Stats *stats = hypredrv_default_stats;
      if (stats)
      {
         for (int i = entry.solve_start; i < entry.solve_end; i++)
         {
            l_iters += stats->iters[i];
            p_time += stats->prec[i];
            s_time += stats->solve[i];
         }
      }

      if (num_solves) *num_solves = n_solves;
      if (linear_iters) *linear_iters = l_iters;
      if (setup_time) *setup_time = p_time;
      if (solve_time) *solve_time = s_time;
   }

   return ret;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelPrint
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelPrint(int level)
{
   StatsLevelPrint(hypredrv_default_stats, level);
   return ErrorCodeGet();
}
