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
#include "lsseq.h"
#include "presets.h"
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
#include "HYPREDRV_utils.h"
#include "logging.h"
#include "object.h"
#include "runtime.h"

/* Forward declarations for file-local helpers */
static uint32_t    LinearSystemSetVectorTagsInternal(HYPREDRV_t hypredrv);
static uint32_t    DestroyObjectInternal(HYPREDRV_t hypredrv);
static const char *ResolveLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                                        size_t default_object_name_size);

// Macro to check if HYPREDRV is initialized
#define HYPREDRV_CHECK_INIT()                                \
   if (!hypredrv_RuntimeIsInitialized())                     \
   {                                                         \
      hypredrv_ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED); \
      return hypredrv_ErrorCodeGet();                        \
   }

// Macro to check if HYPREDRV object is valid
#define HYPREDRV_CHECK_OBJ()                                 \
   if (!hypredrv || !hypredrv_RuntimeObjectIsLive(hypredrv)) \
   {                                                         \
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);     \
      return hypredrv_ErrorCodeGet();                        \
   }

static void
DestroyActiveSolver(HYPREDRV_t hypredrv)
{
   if (hypredrv && hypredrv->iargs && hypredrv->solver)
   {
      hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   }
}

static const char *
ResolveLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                     size_t default_object_name_size)
{
   const char *object_name = NULL;

   if (hypredrv && hypredrv->stats)
   {
      object_name = hypredrv->stats->object_name;
   }

   if ((!object_name || object_name[0] == '\0') && hypredrv &&
       hypredrv->runtime_object_id > 0 && default_object_name &&
       default_object_name_size > 0)
   {
      snprintf(default_object_name, default_object_name_size, "obj-%d",
               hypredrv->runtime_object_id);
      object_name = default_object_name;
   }

   return object_name;
}

static void
SetPendingSolvePathContext(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv->stats)
   {
      return;
   }

   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, -1);

   if (!hypredrv->precon_reuse_timesteps.starts ||
       !hypredrv->precon_reuse_timesteps.starts->data ||
       hypredrv->precon_reuse_timesteps.starts->size == 0)
   {
      return;
   }

   int next_ls_id  = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int timestep_id = -1;

   if (!hypredrv_PreconReuseResolveTimestep(&hypredrv->precon_reuse_timesteps,
                                            hypredrv->stats, next_ls_id, &timestep_id,
                                            NULL, NULL))
   {
      return;
   }

   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, timestep_id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
LinearSystemDropOwnedPrecMatrix(HYPREDRV_t hypredrv)
{
   if (hypredrv->mat_M && hypredrv->mat_M != hypredrv->mat_A && hypredrv->owns_mat_M)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_M);
   }

   hypredrv->mat_M      = NULL;
   hypredrv->owns_mat_M = false;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
LinearSystemDropOwnedInitialGuess(HYPREDRV_t hypredrv)
{
   if (hypredrv->vec_x0 && hypredrv->owns_vec_x0)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x0);
   }

   hypredrv->vec_x0      = NULL;
   hypredrv->owns_vec_x0 = false;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
LinearSystemDropOwnedReferenceSolution(HYPREDRV_t hypredrv)
{
   if (hypredrv->vec_xref && hypredrv->vec_xref != hypredrv->vec_b &&
       hypredrv->owns_vec_xref)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_xref);
   }

   hypredrv->vec_xref      = NULL;
   hypredrv->owns_vec_xref = false;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static uint32_t
ApplyGlobalRuntimeSettings(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv->iargs || hypredrv->lib_mode)
   {
      return ERROR_NONE;
   }

   if (hypredrv->iargs->general.exec_policy)
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      HYPRE_SetSpGemmUseVendor(hypredrv->iargs->general.use_vendor_spgemm);
      HYPRE_SetSpMVUseVendor(hypredrv->iargs->general.use_vendor_spmv);
#endif
#endif

#ifdef HYPRE_USING_UMPIRE
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

   return ERROR_NONE;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Initialize()
{
   hypredrv_LogInitializeFromEnv();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Initialize begin");
   uint32_t code = hypredrv_RuntimeInitialize();
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Initialize end (code=0x%x)",
                      code);
   return code;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Finalize()
{
   HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "HYPREDRV_Finalize begin");
   if (hypredrv_RuntimeIsInitialized())
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                         "HYPREDRV_Finalize sees %d active object(s)",
                         hypredrv_RuntimeGetActiveCount());
      hypredrv_RuntimeDestroyAllLiveObjects(DestroyObjectInternal);
   }

   hypredrv_RuntimeFinalizeState();

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
 *-----------------------------------------------------------------------------*/

static uint32_t
DestroyObjectInternal(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv_RuntimeObjectIsLive(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "DestroyObjectInternal begin");

   /* Remove the handle from the live registry before tearing down owned state
    * so finalize-time sweeps never revisit an object that is already mid-destroy. */
   hypredrv_RuntimeUnregisterObject(hypredrv);
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "live object removed from registry (active=%d)",
                        hypredrv_RuntimeGetActiveCount());

   /* Embedded/library-mode consumers like GEOS expect general.statistics to
    * flush with the object lifetime rather than from a separate driver hook. */
   int print_statistics = 0;
   if (hypredrv->lib_mode && hypredrv->iargs && hypredrv->mypid == 0)
   {
      print_statistics = hypredrv->iargs->general.statistics;
   }

   /* Destroy solver/preconditioner objects before tearing down dependent state. */
   if (hypredrv->iargs)
   {
      if (hypredrv->solver)
      {
         hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
      if (hypredrv->precon)
      {
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon);
         hypredrv->precon_is_setup = false;
      }
   }

   if (hypredrv->scaling_ctx)
   {
      hypredrv_ScalingContextDestroy(hypredrv->comm, &hypredrv->scaling_ctx);
   }

   if (hypredrv->mat_A != hypredrv->mat_M && hypredrv->owns_mat_M)
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
   if (hypredrv->vec_x && hypredrv->owns_vec_x)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x);
   }
   if (hypredrv->vec_x0 && hypredrv->owns_vec_x0)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x0);
   }
   if (hypredrv->vec_nn)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_nn);
   }
   if (hypredrv->vec_xref && hypredrv->vec_xref != hypredrv->vec_b &&
       hypredrv->owns_vec_xref)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_xref);
   }

   hypredrv_IntArrayDestroy(&hypredrv->dofmap);
   hypredrv_PreconReuseTimestepsClear(&hypredrv->precon_reuse_timesteps);
   hypredrv_InputArgsDestroy(&hypredrv->iargs);

   if (print_statistics > 0 && !hypredrv->stats_printed)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "printing statistics on destroy (level=%d)",
                           print_statistics);
      hypredrv_StatsPrint(hypredrv->stats, print_statistics);
   }
   else if (hypredrv->stats_printed)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "skipping statistics on destroy (already printed)");
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "DestroyObjectInternal end");
   hypredrv_StatsDestroy(&hypredrv->stats);

   free(hypredrv->states);
   free((void *)hypredrv->vec_s);
   free(hypredrv);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *hypredrv_ptr)
{
   int mypid = hypredrv_LogRankFromComm(comm);
   HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create begin");
   HYPREDRV_CHECK_INIT();
   hypredrv_ErrorCodeResetAll();

   if (!hypredrv_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("HYPREDRV_Create requires a non-NULL output pointer");
      HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create failed: output pointer is NULL");
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_t hypredrv = (HYPREDRV_t)calloc(1, sizeof(hypredrv_t));
   if (!hypredrv)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate HYPREDRV object");
      HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create failed: allocation error");
      return hypredrv_ErrorCodeGet();
   }

   MPI_Comm_rank(comm, &hypredrv->mypid);
   MPI_Comm_size(comm, &hypredrv->nprocs);

   hypredrv->comm          = comm;
   hypredrv->nstates       = 0;
   hypredrv->states        = NULL;
   hypredrv->iargs         = NULL;
   hypredrv->mat_A         = NULL;
   hypredrv->mat_M         = NULL;
   hypredrv->vec_b         = NULL;
   hypredrv->vec_x         = NULL;
   hypredrv->vec_x0        = NULL;
   hypredrv->vec_xref      = NULL;
   hypredrv->vec_nn        = NULL;
   hypredrv->vec_s         = NULL;
   hypredrv->dofmap        = NULL;
   hypredrv->owns_mat_M    = false;
   hypredrv->owns_vec_x    = false;
   hypredrv->owns_vec_x0   = false;
   hypredrv->owns_vec_xref = false;

   hypredrv->precon                        = NULL;
   hypredrv->solver                        = NULL;
   hypredrv->precon_is_setup               = false;
   hypredrv->scaling_ctx                   = NULL;
   hypredrv->precon_reuse_timesteps.ids    = NULL;
   hypredrv->precon_reuse_timesteps.starts = NULL;
   hypredrv->stats                         = NULL;
   hypredrv->runtime_object_id             = 0;
   hypredrv->next_live                     = NULL;

   /* Disable library mode by default */
   hypredrv->lib_mode = false;

   /* Each object owns its own stats context. */
   hypredrv->stats = hypredrv_StatsCreate();
   if (!hypredrv->stats)
   {
      free(hypredrv);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate HYPREDRV statistics context");
      HYPREDRV_LOGF(1, mypid, NULL, 0,
                    "HYPREDRV_Create failed: statistics allocation error");
      return hypredrv_ErrorCodeGet();
   }

   if (hypredrv_RuntimeRegisterObject(hypredrv))
   {
      HYPREDRV_LOGF(1, mypid, NULL, 0,
                    "HYPREDRV_Create failed: runtime registration error");
      hypredrv_StatsDestroy(&hypredrv->stats);
      free(hypredrv);
      return hypredrv_ErrorCodeGet();
   }

   /* Propagate runtime ID to stats so solver/linsys log helpers can
      fall back to "obj-N" without access to the full hypredrv object. */
   hypredrv->stats->runtime_object_id = hypredrv->runtime_object_id;

   /* Set output pointer */
   *hypredrv_ptr = hypredrv;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_Create end (active=%d)",
                        hypredrv_RuntimeGetActiveCount());

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();
   hypredrv_ErrorCodeResetAll();

   if (!hypredrv_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("HYPREDRV_Destroy requires a non-NULL HYPREDRV_t pointer");
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_t hypredrv = *hypredrv_ptr;

   if (!hypredrv || !hypredrv_RuntimeObjectIsLive(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return hypredrv_ErrorCodeGet();
   }

   int mypid = hypredrv->mypid;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_Destroy begin");
   DestroyObjectInternal(hypredrv);
   *hypredrv_ptr = NULL;
   HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Destroy end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintLibInfo(MPI_Comm comm, int print_datetime)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintLibInfo(comm, print_datetime);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintSystemInfo(MPI_Comm comm)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintSystemInfo(comm);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   HYPREDRV_CHECK_INIT();

   hypredrv_PrintExitInfo(comm, argv0);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t hypredrv)
{
   char log_object_name[32];

   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse begin (argc=%d)", argc);

   /* If preset/defaults were configured before parsing, clear old args first. */
   hypredrv_InputArgsDestroy(&hypredrv->iargs);

   log_object_name[0] = '\0';
   hypredrv_InputArgsParseWithObjectName(
      hypredrv->comm, hypredrv->lib_mode, argc, argv, &hypredrv->iargs,
      ResolveLogObjectName(hypredrv, log_object_name, sizeof(log_object_name)));

   if (hypredrv_ErrorCodeGet())
   {
      HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse failed (code=0x%x)",
                           hypredrv_ErrorCodeGet());
      return hypredrv_ErrorCodeGet();
   }

   /* Apply parsed configuration ------------------------------------------- */

   /* Initialize stats from input args */
   hypredrv_StatsSetNumReps(hypredrv->stats, hypredrv->iargs->general.num_repetitions);
   hypredrv_StatsSetNumLinearSystems(hypredrv->stats, hypredrv->iargs->ls.num_systems);
   if (hypredrv->iargs->general.use_millisec)
   {
      hypredrv_StatsTimerSetMilliseconds(hypredrv->stats);
   }
   else
   {
      hypredrv_StatsTimerSetSeconds(hypredrv->stats);
   }
   hypredrv_StatsSetObjectName(hypredrv->stats, hypredrv->iargs->general.name);

   /* Load timestep schedule for preconditioner reuse */
   hypredrv_PreconReuseTimestepsClear(&hypredrv->precon_reuse_timesteps);
   if (hypredrv->iargs->ls.timestep_filename[0] != '\0')
   {
      hypredrv_PreconReuseTimestepsLoad(&hypredrv->iargs->precon_reuse,
                                        hypredrv->iargs->ls.timestep_filename,
                                        &hypredrv->precon_reuse_timesteps);
   }
   else if (hypredrv->iargs->ls.sequence_filename[0] != '\0' &&
            hypredrv->iargs->precon_reuse.enabled &&
            hypredrv->iargs->precon_reuse.per_timestep)
   {
      hypredrv_LSSeqReadTimestepsWithIds(hypredrv->iargs->ls.sequence_filename,
                                         &hypredrv->precon_reuse_timesteps.ids,
                                         &hypredrv->precon_reuse_timesteps.starts);
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv->lib_mode = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_ObjectSetName(HYPREDRV_t hypredrv, const char *name)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv_StatsSetObjectName(hypredrv->stats, name);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv, int *warmup)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   if (!warmup)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }
   *warmup = hypredrv->iargs->general.warmup;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv, int *num_reps)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   if (!num_reps)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }
   *num_reps = hypredrv->iargs->general.num_repetitions;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv, int *num_ls)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   if (!num_ls)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }
   *num_ls = hypredrv->iargs->ls.num_systems;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetNumPreconVariants(HYPREDRV_t hypredrv, int *num_variants)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   if (!num_variants)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }
   *num_variants = hypredrv->iargs->num_precon_variants;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv, int variant_idx)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsSetPreconVariant begin (idx=%d)",
                        variant_idx);

   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Input arguments not parsed");
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "preconditioner variant switch failed: input args missing");
      return hypredrv_ErrorCodeGet();
   }

   if (variant_idx < 0 || variant_idx >= hypredrv->iargs->num_precon_variants)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid preconditioner variant index: %d (valid range: 0-%d)",
                           variant_idx, hypredrv->iargs->num_precon_variants - 1);
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv, "preconditioner variant switch failed: invalid idx=%d (max=%d)",
         variant_idx, hypredrv->iargs->num_precon_variants - 1);
      return hypredrv_ErrorCodeGet();
   }

   int current_variant = hypredrv->iargs->active_precon_variant;
   int variant_changed = (variant_idx != current_variant);
   HYPREDRV_LOG_OBJECTF(
      2, hypredrv, "preconditioner variant selection: current=%d requested=%d changed=%d",
      current_variant, variant_idx, variant_changed);

   /* Only rebuild solver/preconditioner when switching variants. */
   if (variant_changed)
   {
      if (hypredrv->solver)
      {
         HYPREDRV_LOG_OBJECTF(
            2, hypredrv, "switching preconditioner variant: destroying active solver");
         hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
      if (hypredrv->precon)
      {
         HYPREDRV_LOG_OBJECTF(
            2, hypredrv,
            "switching preconditioner variant: destroying active preconditioner");
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon);
         hypredrv->precon_is_setup = false;
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "preconditioner variant unchanged; reusing setup");
   }

   /* Set active variant */
   hypredrv->iargs->active_precon_variant = variant_idx;
   hypredrv->iargs->precon_method         = hypredrv->iargs->precon_methods[variant_idx];
   hypredrv->iargs->precon                = hypredrv->iargs->precon_variants[variant_idx];
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner variant selected: idx=%d method=%d",
                        variant_idx, (int)hypredrv->iargs->precon_method);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsSetPreconVariant end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!preset)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner preset name cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv->iargs)
   {
      hypredrv_InputArgsCreate(hypredrv->lib_mode, &hypredrv->iargs);
      hypredrv_PreconArgsSetDefaultsForMethod(hypredrv->iargs->precon_method,
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
      hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
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
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Active preconditioner variant index %d is out of range (0-%d)", variant_idx,
         hypredrv->iargs->num_precon_variants - 1);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_InputArgsApplyPreconPreset(hypredrv->iargs, preset, variant_idx);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetSolverPreset(HYPREDRV_t hypredrv, const char *preset)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!preset)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Solver preset name cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv->iargs)
   {
      hypredrv_InputArgsCreate(hypredrv->lib_mode, &hypredrv->iargs);
   }

   /* Validate solver name */
   if (!hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_SolverGetValidTypeIntMap(),
                                                 preset))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Unknown solver preset: '%s'. Valid options: pcg, gmres, fgmres, bicgstab",
         preset);
      return hypredrv_ErrorCodeGet();
   }

   /* Destroy existing solver if any */
   DestroyActiveSolver(hypredrv);

   /* Set solver method and defaults */
   hypredrv->iargs->solver_method = (solver_t)hypredrv_StrIntMapArrayGetImage(
      hypredrv_SolverGetValidTypeIntMap(), preset);
   hypredrv_SolverArgsSetDefaultsForMethod(hypredrv->iargs->solver_method,
                                           &hypredrv->iargs->solver);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconPresetRegister(const char *name, const char *yaml_text, const char *help)
{
   if (!name || !yaml_text)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_PreconPresetRegister: name and yaml_text must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_PresetRegister(name, yaml_text, help);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates, HYPRE_IJVector *vecs)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorSet begin (nstates=%d)",
                        nstates);

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
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         HYPREDRV_LOG_OBJECTF(
            2, hypredrv, "HYPREDRV_StateVectorSet failed: missing vector at index=%d", i);
         return hypredrv_ErrorCodeGet();
      }
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorSet end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorGetValues(HYPREDRV_t hypredrv, int index, HYPRE_Complex **data_ptr)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorGetValues begin (index=%d)",
                        index);

   int             state   = hypredrv->states[index];
   HYPRE_ParVector par_vec = NULL;
   hypre_Vector   *seq_vec = NULL;
   void           *obj     = NULL;

   if (hypredrv->vec_s[state])
   {
#if defined(HYPRE_USING_GPU)
      /* Ensure data is accessible on host before returning the pointer */
      HYPRE_IJVectorMigrate(hypredrv->vec_s[state], HYPRE_MEMORY_HOST);
#endif
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state], &obj);
      par_vec   = (HYPRE_ParVector)obj;
      seq_vec   = hypre_ParVectorLocalVector(par_vec);
      *data_ptr = hypre_VectorData(seq_vec);
      HYPREDRV_LOG_OBJECTF(3, hypredrv, "HYPREDRV_StateVectorGetValues resolved state=%d",
                           state);
   }
   else
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "HYPREDRV_StateVectorGetValues failed: state vector is NULL");
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorGetValues end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv, int index_in, int index_out)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv,
                        "HYPREDRV_StateVectorCopy begin (index_in=%d index_out=%d)",
                        index_in, index_out);

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
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "HYPREDRV_StateVectorCopy failed: source or destination vector is NULL");
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorCopy end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Cycle through state vectors
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorUpdateAll begin");

   for (int i = 0; i < hypredrv->nstates; i++)
   {
      hypredrv->states[i] = (hypredrv->states[i] + 1) % hypredrv->nstates;
   }

   HYPREDRV_LOG_OBJECTF(3, hypredrv,
                        "HYPREDRV_StateVectorUpdateAll rotated %d state slots",
                        hypredrv->nstates);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorUpdateAll end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv, int state_idx)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPREDRV_StateVectorApplyCorrection begin (state_idx=%d)", state_idx);

   if (state_idx < 0 || state_idx >= hypredrv->nstates)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("state_idx %d out of range [0, %d)", state_idx,
                           hypredrv->nstates);
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "HYPREDRV_StateVectorApplyCorrection failed: state index out of range");
      return hypredrv_ErrorCodeGet();
   }

   void *obj_s = NULL, *obj_delta = NULL;
   int   current = hypredrv->states[state_idx];

   HYPRE_IJVectorGetObject(hypredrv->vec_x, &obj_delta);
#if defined(HYPRE_USING_GPU)
   /* vec_x (delta) lives on device after the solve; bring vec_s to device to match */
   HYPRE_IJVectorMigrate(hypredrv->vec_s[current], HYPRE_MEMORY_DEVICE);
#endif
   HYPRE_IJVectorGetObject(hypredrv->vec_s[current], &obj_s);

   /* U = U + Δx */
   HYPRE_ParVectorAxpy((HYPRE_Complex)1.0, (HYPRE_ParVector)obj_delta,
                       (HYPRE_ParVector)obj_s);

   HYPREDRV_LOG_OBJECTF(3, hypredrv,
                        "HYPREDRV_StateVectorApplyCorrection applied to state slot=%d",
                        current);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorApplyCorrection end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild begin");

   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, NULL));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetReferenceSolution(hypredrv, NULL));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   HYPREDRV_SAFE_CALL(LinearSystemSetVectorTagsInternal(hypredrv));
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
   /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */

   /* Reset scaling state for new system */
   if (hypredrv->scaling_ctx)
   {
      hypredrv->scaling_ctx->is_applied = 0;
   }

   long long int num_rows = hypredrv_LinearSystemMatrixGetNumRows(hypredrv->mat_A);
   long long int num_nonzeros =
      hypredrv_LinearSystemMatrixGetNumNonzeros(hypredrv->mat_A);
   if (!hypredrv->mypid)
   {
      PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH);
      printf("Solving linear system #%d ",
             hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
      printf("with %lld rows and %lld nonzeros...\n", num_rows, num_nonzeros);
   }
   HYPRE_ClearAllErrors();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   hypredrv_LinearSystemReadMatrix(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->mat_A,
                                   hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemDropOwnedPrecMatrix(hypredrv);

   /* Don't annotate "matrix" here - users annotate with "system" in their code */
   /* This was causing build times and solve times to be recorded in separate entries
    */
   hypredrv->mat_A      = (HYPRE_IJMatrix)mat_A;
   hypredrv->mat_M      = (HYPRE_IJMatrix)mat_A;
   hypredrv->owns_mat_M = false;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      if (hypredrv->vec_xref && !hypredrv->owns_vec_xref)
      {
         hypredrv->vec_xref = NULL;
      }
      hypredrv_LinearSystemSetRHS(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                  &hypredrv->vec_xref, &hypredrv->vec_b, hypredrv->stats);
      hypredrv->owns_vec_xref = (hypredrv->vec_xref != NULL);
   }
   else
   {
      hypredrv->vec_b = (HYPRE_IJVector)vec;
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                      int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   hypredrv_LinearSystemSetNearNullSpace(hypredrv->comm, &hypredrv->iargs->ls,
                                         hypredrv->mat_A, num_entries, num_components,
                                         values, &hypredrv->vec_nn);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      if (hypredrv->vec_x0 && !hypredrv->owns_vec_x0)
      {
         hypredrv->vec_x0 = NULL;
      }
      hypredrv_LinearSystemSetInitialGuess(
         hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A, hypredrv->vec_b,
         &hypredrv->vec_x0, &hypredrv->vec_x, hypredrv->stats);
      hypredrv->owns_vec_x0 = (hypredrv->vec_x0 != NULL);
      hypredrv->owns_vec_x  = (hypredrv->vec_x != NULL);
   }
   else
   {
      LinearSystemDropOwnedInitialGuess(hypredrv);
      hypredrv->vec_x0 = (HYPRE_IJVector)vec;
      hypredrv->owns_vec_x0 =
         (bool)(!hypredrv->lib_mode && hypredrv->vec_x0 != hypredrv->vec_x &&
                hypredrv->vec_x0 != hypredrv->vec_b);
      if (hypredrv->vec_x && !hypredrv->owns_vec_x)
      {
         hypredrv->vec_x = NULL;
      }
      hypredrv_LinearSystemCreateWorkingSolution(hypredrv->comm, &hypredrv->iargs->ls,
                                                 hypredrv->vec_b, &hypredrv->vec_x);
      hypredrv->owns_vec_x = (hypredrv->vec_x != NULL);
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetSolution(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      /* Discard any borrowed reference before recreating. */
      if (!hypredrv->owns_vec_x)
      {
         hypredrv->vec_x = NULL;
      }
      if (!hypredrv->vec_b)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("SetSolution(NULL): RHS vector must be set first");
         return hypredrv_ErrorCodeGet();
      }
      hypredrv_LinearSystemCreateWorkingSolution(hypredrv->comm, &hypredrv->iargs->ls,
                                                 hypredrv->vec_b, &hypredrv->vec_x);
      hypredrv->owns_vec_x = (hypredrv->vec_x != NULL);
   }
   else
   {
      /* Destroy existing owned solution before replacing. */
      if (hypredrv->vec_x && hypredrv->owns_vec_x)
      {
         HYPRE_IJVectorDestroy(hypredrv->vec_x);
      }
      hypredrv->vec_x      = (HYPRE_IJVector)vec;
      hypredrv->owns_vec_x = false; /* always borrow: caller manages lifetime */
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetReferenceSolution(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      const bool uses_xref_file = (bool)(hypredrv->iargs->ls.xref_filename[0] != '\0' ||
                                         hypredrv->iargs->ls.xref_basename[0] != '\0');
      if (uses_xref_file && hypredrv->vec_xref && !hypredrv->owns_vec_xref)
      {
         hypredrv->vec_xref = NULL;
      }
      hypredrv_LinearSystemSetReferenceSolution(hypredrv->comm, &hypredrv->iargs->ls,
                                                &hypredrv->vec_xref, hypredrv->stats);
      if (uses_xref_file)
      {
         hypredrv->owns_vec_xref = (hypredrv->vec_xref != NULL);
      }
   }
   else
   {
      LinearSystemDropOwnedReferenceSolution(hypredrv);
      hypredrv->vec_xref = (HYPRE_IJVector)vec;
      hypredrv->owns_vec_xref =
         (bool)(!hypredrv->lib_mode && hypredrv->vec_xref != hypredrv->vec_b);
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->vec_x0 || !hypredrv->vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemResetInitialGuess(hypredrv->vec_x0, hypredrv->vec_x,
                                          hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static uint32_t
LinearSystemSetVectorTagsInternal(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!hypredrv->dofmap || !hypredrv->dofmap->data || hypredrv->dofmap->size == 0)
   {
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemSetVectorTags(hypredrv->vec_b, hypredrv->dofmap);
   hypredrv_LinearSystemSetVectorTags(hypredrv->vec_x, hypredrv->dofmap);
   hypredrv_LinearSystemSetVectorTags(hypredrv->vec_x0, hypredrv->dofmap);
   hypredrv_LinearSystemSetVectorTags(hypredrv->vec_xref, hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!sol_data || !hypredrv->vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemGetSolutionValues(hypredrv->vec_x, sol_data);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv, const char *norm_type,
                                     double *norm)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!norm_type || !norm)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x, norm_type, norm);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolution(HYPREDRV_t hypredrv, HYPRE_Vector *vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec || !hypredrv->vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   *vec = (HYPRE_Vector)hypredrv->vec_x;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!rhs_data || !hypredrv->vec_b)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LinearSystemGetRHSValues(hypredrv->vec_b, rhs_data);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHS(HYPREDRV_t hypredrv, HYPRE_Vector *vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec || !hypredrv->vec_b)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   *vec = (HYPRE_Vector)hypredrv->vec_b;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix *mat)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!mat || !hypredrv->mat_A)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   *mat = (HYPRE_Matrix)hypredrv->mat_A;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!mat)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      LinearSystemDropOwnedPrecMatrix(hypredrv);
      hypredrv_LinearSystemSetPrecMatrix(hypredrv->comm, &hypredrv->iargs->ls,
                                         hypredrv->mat_A, &hypredrv->mat_M,
                                         hypredrv->stats);
      hypredrv->owns_mat_M =
         (bool)(hypredrv->mat_M != NULL && hypredrv->mat_M != hypredrv->mat_A);
   }
   else
   {
      LinearSystemDropOwnedPrecMatrix(hypredrv);
      hypredrv->mat_M = (HYPRE_IJMatrix)mat;
      hypredrv->owns_mat_M =
         (bool)(!hypredrv->lib_mode && hypredrv->mat_M != hypredrv->mat_A);
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Keep ownership clean when this is called multiple times */
   hypredrv_IntArrayDestroy(&hypredrv->dofmap);
   hypredrv_IntArrayBuild(hypredrv->comm, size, dofmap, &hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                          int num_dof_types)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv_IntArrayDestroy(&hypredrv->dofmap);
   hypredrv_IntArrayBuildInterleaved(hypredrv->comm, num_local_blocks, num_dof_types,
                                     &hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                         int num_dof_types)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv_IntArrayDestroy(&hypredrv->dofmap);
   hypredrv_IntArrayBuildContiguous(hypredrv->comm, num_local_blocks, num_dof_types,
                                    &hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv_LinearSystemReadDofmap(hypredrv->comm, &hypredrv->iargs->ls,
                                   &hypredrv->dofmap, hypredrv->stats);

   return hypredrv_ErrorCodeGet();
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
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
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
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Delegate printing to linsys */
   hypredrv_LinearSystemPrintData(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                  hypredrv->vec_b, hypredrv->dofmap);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconCreate begin");
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   int  next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   bool should_create =
      ((hypredrv->precon == NULL) ||
       hypredrv_PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                           &hypredrv->precon_reuse_timesteps,
                                           hypredrv->stats, next_ls_id)) != 0;
   HYPREDRV_LOG_OBJECTF(2, hypredrv,
                        "preconditioner reuse decision: should_create=%d next_ls_id=%d",
                        (int)should_create, next_ls_id);

   if (should_create)
   {
      /* If we're recreating, destroy the existing preconditioner first to avoid leaks. */
      if (hypredrv->precon)
      {
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon);
         hypredrv->precon_is_setup = false;
      }
      hypredrv_PreconCreate(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                            hypredrv->dofmap, hypredrv->vec_nn, &hypredrv->precon);
      hypredrv->precon_is_setup = false;
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "reusing existing preconditioner without recreation");
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconCreate end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverCreate begin");

   /* Delegate preconditioner lifecycle to PreconCreate.
    *
    * Skip the call only when a preconditioner already exists but has NOT been
    * set up yet: that means it was just created by an explicit
    * HYPREDRV_PreconCreate() call from the caller (e.g. main.c) for this
    * same linear system, so re-evaluating the reuse policy would be
    * redundant (and would destroy an uninitialized object, which crashes).
    *
    * If the precon exists AND is already set up, the reuse policy must still
    * be checked: a new timestep may require re-creating the preconditioner. */
   if (hypredrv->precon == NULL || hypredrv->precon_is_setup)
   {
      if (HYPREDRV_PreconCreate(hypredrv))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         HYPREDRV_LOG_OBJECTF(
            1, hypredrv, "HYPREDRV_LinearSolverCreate failed: preconditioner create");
         return hypredrv_ErrorCodeGet();
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "skipping preconditioner creation (precon already created, pending setup)");
   }

   /* Always recreate solver per linear system.
    * Reuse policy applies to preconditioner lifecycle; solver internals should
    * be rebuilt against the current system vectors/matrix each cycle. */
   if (hypredrv->solver)
   {
      hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   }
   hypredrv_SolverCreate(hypredrv->comm, hypredrv->iargs->solver_method,
                         &hypredrv->iargs->solver, &hypredrv->solver);
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "solver created (method=%d)",
                        (int)hypredrv->iargs->solver_method);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverCreate end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup begin");

   hypredrv_PreconSetup(hypredrv->iargs->precon_method, hypredrv->precon,
                        hypredrv->mat_A);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */
   if (hypredrv->precon)
   {
      hypredrv->precon_is_setup = true;
   }
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
GMRESSetRefSolution(HYPREDRV_t hypredrv)
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
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup begin");

   int next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int recompute  = hypredrv_PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                                        &hypredrv->precon_reuse_timesteps,
                                                        hypredrv->stats, next_ls_id);
   int skip_precon_setup =
      (hypredrv->precon != NULL) && hypredrv->precon_is_setup && !recompute;
   HYPREDRV_LOG_OBJECTF(
      2, hypredrv, "solver setup decisions: recompute_precon=%d skip_precon_setup=%d",
      recompute, skip_precon_setup);

   /* Propagate dofmap to vectors (no-op if no dofmap is set) */
   HYPREDRV_SAFE_CALL(LinearSystemSetVectorTagsInternal(hypredrv));

   /* Create scaling context if needed and not already created */
   if (hypredrv->iargs->scaling.enabled && !hypredrv->scaling_ctx)
   {
      hypredrv_ScalingContextCreate(hypredrv->comm, &hypredrv->scaling_ctx);
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "created scaling context (type=%d)",
                           (int)hypredrv->iargs->scaling.type);
   }

   /* Compute scaling if enabled (recompute for each system if needed) */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      hypredrv_ScalingCompute(hypredrv->comm, &hypredrv->iargs->scaling,
                              hypredrv->scaling_ctx, hypredrv->mat_A, hypredrv->vec_b,
                              hypredrv->dofmap);
      if (hypredrv_ErrorCodeGet())
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Apply scaling before setup if enabled */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                    hypredrv->mat_M, hypredrv->vec_b, hypredrv->vec_x);
      if (hypredrv_ErrorCodeGet())
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   GMRESSetRefSolution(hypredrv);

   hypredrv_SolverSetupWithReuse(hypredrv->iargs->precon_method,
                                 hypredrv->iargs->solver_method, hypredrv->precon,
                                 hypredrv->solver, hypredrv->mat_M, hypredrv->vec_b,
                                 hypredrv->vec_x, hypredrv->stats, skip_precon_setup);

   HYPRE_ClearAllErrors();
   if (hypredrv->precon && !skip_precon_setup)
   {
      hypredrv->precon_is_setup = true;
   }
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply begin");

   if (!hypredrv->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      HYPREDRV_LOG_OBJECTF(1, hypredrv,
                           "HYPREDRV_LinearSolverApply failed: solver is NULL");
      return hypredrv_ErrorCodeGet();
   }

   double e_norm = 0.0, x_norm = 0.0, xref_norm = 0.0;
   double b_norm = 0.0, r_norm = 0.0, r0_norm = 0.0;

   /* Ensure GMRES always sees the current reference solution, including on reused
    * preconditioner cycles where SolverSetup may be skipped. */
   GMRESSetRefSolution(hypredrv);

   /* Apply scaling if enabled but not yet applied (e.g., when preconditioner is reused)
    */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       !hypredrv->scaling_ctx->is_applied)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "applying deferred system scaling");
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                    hypredrv->mat_M, hypredrv->vec_b, hypredrv->vec_x);
      if (hypredrv_ErrorCodeGet())
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Perform solve */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       hypredrv->scaling_ctx->is_applied)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving scaled system");
      int xref_scaled = 0;

      /* Compute initial residual norm before solve (on current system state) */
      hypredrv_LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b,
                                               hypredrv->vec_x, "L2", &r0_norm);

      if (hypredrv->vec_xref)
      {
         hypredrv_ScalingApplyToVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                       SCALING_VECTOR_UNKNOWN);
         if (hypredrv_ErrorCodeGet())
         {
            return hypredrv_ErrorCodeGet();
         }
         xref_scaled = 1;
      }

      SetPendingSolvePathContext(hypredrv);
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
      hypredrv_StatsInitialResNormSet(hypredrv->stats, r0_norm);

      /* Solve on scaled system */
      HYPRE_Int iters =
         hypredrv_SolverSolveOnly(hypredrv->iargs->solver_method, hypredrv->solver,
                                  hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x);
      if (iters < 0)
      {
         if (xref_scaled)
         {
            hypredrv_ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                         SCALING_VECTOR_UNKNOWN);
         }
         hypredrv_StatsIterSet(hypredrv->stats, 0);
         hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");
         return hypredrv_ErrorCodeGet();
      }

      hypredrv_StatsIterSet(hypredrv->stats, (int)iters);
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");

      /* Undo scaling on solution and restore original system */
      hypredrv_ScalingUndoOnSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                   hypredrv->mat_M, hypredrv->vec_b, hypredrv->vec_x);
      if (hypredrv_ErrorCodeGet())
      {
         if (xref_scaled)
         {
            hypredrv_ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                         SCALING_VECTOR_UNKNOWN);
         }
         return hypredrv_ErrorCodeGet();
      }

      if (xref_scaled)
      {
         hypredrv_ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                      SCALING_VECTOR_UNKNOWN);
         if (hypredrv_ErrorCodeGet())
         {
            return hypredrv_ErrorCodeGet();
         }
      }

      /* Compute residual norms on original (now restored) system */
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_b, "L2", &b_norm);
      hypredrv_LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b,
                                               hypredrv->vec_x, "L2", &r_norm);
      b_norm = (b_norm > 0.0) ? b_norm : 1.0;
      hypredrv_StatsRelativeResNormSet(hypredrv->stats, r_norm / b_norm);
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving unscaled system");
      SetPendingSolvePathContext(hypredrv);
      /* No scaling - use standard hypredrv_SolverApply which handles everything including
       * stats */
      hypredrv_SolverApply(hypredrv->iargs->solver_method, hypredrv->solver,
                           hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x,
                           hypredrv->stats);
      /* hypredrv_SolverApply already computed and set all stats */
   }

   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   if (hypredrv->vec_xref)
   {
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_xref, "L2", &xref_norm);
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x, "L2", &x_norm);
      hypredrv_LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, "L2",
                                            &e_norm);
      if (!hypredrv->mypid)
      {
         printf("L2 norm of error: %e\n", (double)e_norm);
         printf("L2 norm of solution: %e\n", (double)x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
      }
   }
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "solve finished (iters=%d)",
                        hypredrv_StatsGetLastIter(hypredrv->stats));
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply begin");

   if (!hypredrv->precon)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("HYPREDRV_PreconApply: preconditioner is NULL");
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "HYPREDRV_PreconApply failed: preconditioner is NULL");
      return hypredrv_ErrorCodeGet();
   }
   if (!hypredrv->mat_A || !vec_b || !vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_PreconApply requires matrix, rhs, and solution vectors");
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "HYPREDRV_PreconApply failed: matrix or vector is NULL");
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner apply: method=%d has_precon=%d",
                        (int)hypredrv->iargs->precon_method, (hypredrv->precon != NULL));

   hypredrv_PreconApply(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A,
                        (HYPRE_IJVector)vec_b, (HYPRE_IJVector)vec_x);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconDestroy begin");

   int  next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   bool should_destroy =
      hypredrv_PreconReuseShouldRecompute(&hypredrv->iargs->precon_reuse,
                                          &hypredrv->precon_reuse_timesteps,
                                          hypredrv->stats, next_ls_id) != 0;
   HYPREDRV_LOG_OBJECTF(
      2, hypredrv, "preconditioner destroy decision: should_destroy=%d next_ls_id=%d",
      (int)should_destroy, next_ls_id);

   if (should_destroy)
   {
      if (hypredrv->precon)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv, "destroying preconditioner object");
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon);
         hypredrv->precon_is_setup = false;
      }
      else
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "preconditioner destroy requested but object is NULL");
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner kept for reuse");
   }

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconDestroy end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverDestroy begin");

   /* First, destroy the preconditioner if we need */
   if (hypredrv->precon)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "linear solver destroy: evaluating preconditioner teardown");
      if (HYPREDRV_PreconDestroy(hypredrv))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         HYPREDRV_LOG_OBJECTF(
            2, hypredrv, "linear solver destroy failed: preconditioner teardown error");
         return hypredrv_ErrorCodeGet();
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "linear solver destroy: no preconditioner present");
   }

   hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "linear solver destroy: solver object released");

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverDestroy end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StatsPrint");
   hypredrv_StatsPrint(hypredrv->stats,
                       hypredrv->iargs ? hypredrv->iargs->general.statistics : 0);
   hypredrv->stats_printed = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, name,
                                       id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_END, name, id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsAnnotateLevelWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN,
                                            level, name, id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsAnnotateLevelWithId(hypredrv->stats, HYPREDRV_ANNOTATE_END, level,
                                            name, id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

#ifdef HYPREDRV_ENABLE_EIGSPEC
static void
PreconApplyWrapper(void *ctx, void *b, void *x)
{
   HYPREDRV_PreconApply((HYPREDRV_t)ctx, (HYPRE_Vector)b, (HYPRE_Vector)x);
}
#endif

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum begin");

#ifdef HYPREDRV_ENABLE_EIGSPEC
   /* Exit early if not computing eigenspectrum */
   if (!hypredrv->iargs->ls.eigspec.enable)
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "eigenspectrum computation skipped: linear_system.eigspec.enable=0");
      HYPREDRV_LOG_OBJECTF(1, hypredrv,
                           "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)",
                           hypredrv_ErrorCodeGet());
      return hypredrv_ErrorCodeGet();
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
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: preconditioned operator");
      HYPREDRV_PreconCreate(hypredrv);
      HYPREDRV_PreconSetup(hypredrv);

      uint32_t code =
         hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec, (void *)hypredrv->mat_A,
                                 (void *)hypredrv, PreconApplyWrapper, hypredrv->stats);
      HYPREDRV_LOG_OBJECTF(
         1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)", code);
      return code;
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: unpreconditioned operator");
      uint32_t code =
         hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec, (void *)hypredrv->mat_A,
                                 NULL, NULL, hypredrv->stats);
      HYPREDRV_LOG_OBJECTF(
         1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)", code);
      return code;
   }
#else
   static bool warned_eigspec_disabled = false;

   if (!warned_eigspec_disabled && !hypredrv->mypid)
   {
      fprintf(stderr,
              "[HYPREDRV] Warning: HYPREDRV_LinearSystemComputeEigenspectrum called "
              "but eigenspectrum support is disabled. "
              "Reconfigure with -DHYPREDRV_ENABLE_EIGSPEC=ON to enable it.\n");
      fflush(stderr);
      warned_eigspec_disabled = true;
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum support disabled in build; emitted warning");
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv, "eigenspectrum support disabled in build; warning already emitted");
   }
#endif

   HYPREDRV_LOG_OBJECTF(1, hypredrv,
                        "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)",
                        hypredrv_ErrorCodeGet());
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetNumIter(HYPREDRV_t hypredrv, int *iters)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!iters)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("iters pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *iters = hypredrv_StatsGetLastIter(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetSetupTime(HYPREDRV_t hypredrv, double *seconds)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!seconds)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("seconds pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *seconds = hypredrv_StatsGetLastSetupTime(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverGetSolveTime(HYPREDRV_t hypredrv, double *seconds)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!seconds)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("seconds pointer cannot be NULL");
      return hypredrv_ErrorCodeGet();
   }

   *seconds = hypredrv_StatsGetLastSolveTime(hypredrv->stats);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelGetCount(HYPREDRV_t hypredrv, int level, int *count)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsLevelGetCountChecked(hypredrv->stats, level, count,
                                             "HYPREDRV_StatsLevelGetCount");
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelGetEntry(HYPREDRV_t hypredrv, int level, int index, int *entry_id,
                            int *num_solves, int *linear_iters, double *setup_time,
                            double *solve_time)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   return hypredrv_StatsLevelGetEntrySummary(hypredrv->stats, level, index, entry_id,
                                             num_solves, linear_iters, setup_time,
                                             solve_time);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelPrint(HYPREDRV_t hypredrv, int level)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv_StatsLevelPrint(hypredrv->stats, level);
   return hypredrv_ErrorCodeGet();
}
