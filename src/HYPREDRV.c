/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "_hypre_parcsr_mv.h" /* For hypre_VectorData, hypre_ParVectorLocalVector */
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/info.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
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
#include "runtime.h"

/* Forward declarations for file-local helpers */
static uint32_t    LinearSystemSetVectorTagsInternal(HYPREDRV_t hypredrv);
static uint32_t    DestroyObjectInternal(HYPREDRV_t hypredrv);
static const char *ResolveLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                                        size_t default_object_name_size);
static uint32_t    LoadTimestepScheduleFromFile(const char *filename,
                                                IntArray  **timestep_ids,
                                                IntArray  **timestep_starts);
static int         PrintSystemNeedsTimestepSchedule(const PrintSystem_args *cfg);
static int  ResolveTimestepIndex(const IntArray *timestep_starts, const Stats *stats,
                                 int system_index);
static void AdvanceLibraryManagedSystemIndex(HYPREDRV_t hypredrv);
static void BuildPrintSystemContext(HYPREDRV_t hypredrv, int stage,
                                    PrintSystemContext *ctx);
static void MaybeDumpLinearSystem(HYPREDRV_t hypredrv, int stage);
static bool PushDefaultLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                                     size_t default_object_name_size);
static void PopDefaultLogObjectName(HYPREDRV_t hypredrv, const char *default_object_name,
                                    bool pushed_default_name);
static int  PreconReuseShouldRebuildCollective(HYPREDRV_t hypredrv, int next_ls_id,
                                               PreconReuseDecision *decision);
static uint32_t ApplyGlobalRuntimeSettings(HYPREDRV_t hypredrv);
static void PrepareExplicitObjectForConfiguredExecution(HYPREDRV_t hypredrv, void *obj,
                                                        int is_matrix);
/* Check if hypredrive is initialized */
static inline uint32_t
hypredrv_CheckInit(void)
{
   /* GCOVR_EXCL_BR_LINE */
   if (!hypredrv_RuntimeIsInitialized())
   {
      hypredrv_ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED);
      return 1;
   }
   return 0;
   /* GCOVR_EXCL_BR_LINE */
}

/* Check if hypredrive is initialized and if HYPREDRV object is valid */
static inline uint32_t
hypredrv_CheckInitAndObj(HYPREDRV_t hypredrv)
{
   /* GCOVR_EXCL_BR_LINE */
   if (!hypredrv_RuntimeIsInitialized())
   {
      hypredrv_ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED);
      return 1;
   }

   if (!hypredrv || !hypredrv_RuntimeObjectIsLive(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return 1;
   }

   return 0;
   /* GCOVR_EXCL_BR_LINE */
}

// Macro to check if hypredrive is initialized
#define HYPREDRV_CHECK_INIT()         \
   if (hypredrv_CheckInit())          \
   {                                  \
      return hypredrv_ErrorCodeGet(); \
   }

// Macro to check if hypredrive is initialized and if HYPREDRV object is valid
#define HYPREDRV_CHECK_INIT_AND_OBJ()      \
   if (hypredrv_CheckInitAndObj(hypredrv)) \
   {                                       \
      return hypredrv_ErrorCodeGet();      \
   }

static void
DestroyActiveSolver(HYPREDRV_t hypredrv)
{
   if (hypredrv && hypredrv->iargs && hypredrv->solver) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
   }
}

static const char *
ResolveLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                     size_t default_object_name_size)
{
   const char *object_name = NULL;
   if (hypredrv && hypredrv->stats) /* GCOVR_EXCL_BR_LINE */
   {
      object_name = hypredrv->stats->object_name;
   }
   if ((!object_name || object_name[0] == '\0') && hypredrv &&   /* GCOVR_EXCL_BR_LINE */
       hypredrv->runtime_object_id > 0 && default_object_name && /* GCOVR_EXCL_BR_LINE */
       default_object_name_size > 0)
   {
      snprintf(default_object_name, default_object_name_size, "obj-%d",
               hypredrv->runtime_object_id);
      object_name = default_object_name;
   }

   return object_name;
}

static bool
PushDefaultLogObjectName(HYPREDRV_t hypredrv, char *default_object_name,
                         size_t default_object_name_size)
{
   if (!hypredrv || !hypredrv->stats ||
       hypredrv->stats->object_name[0] != '\0' ||             /* GCOVR_EXCL_BR_LINE */
       !default_object_name || default_object_name_size == 0) /* GCOVR_EXCL_BR_LINE */
   {
      return false;
   }

   default_object_name[0]    = '\0';
   const char *resolved_name = ResolveLogObjectName(
      hypredrv, default_object_name, default_object_name_size); /* GCOVR_EXCL_BR_LINE */
   if (!resolved_name || resolved_name[0] == '\0')              /* GCOVR_EXCL_BR_LINE */
   {
      return false;
   }

   hypredrv_StatsSetObjectName(hypredrv->stats, resolved_name);
   return true;
}

static void
PopDefaultLogObjectName(HYPREDRV_t hypredrv, const char *default_object_name,
                        bool pushed_default_name)
{
   if (!pushed_default_name || !hypredrv || !hypredrv->stats ||
       !default_object_name ||         /* GCOVR_EXCL_BR_LINE */
       default_object_name[0] == '\0') /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   if (!strcmp(hypredrv->stats->object_name,
               default_object_name)) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_StatsSetObjectName(hypredrv->stats, "");
   }
}

static int
PreconReuseShouldRebuildCollective(HYPREDRV_t hypredrv, int next_ls_id,
                                   PreconReuseDecision *decision)
{
   /* static helper: hypredrv always live */
   if (!hypredrv)   /* GCOVR_EXCL_BR_LINE */
   {                /* GCOVR_EXCL_BR_LINE */
      if (decision) /* GCOVR_EXCL_BR_LINE */
      {
         memset(decision, 0, sizeof(*decision));
         decision->should_rebuild = 1;
         snprintf(decision->summary, sizeof(decision->summary), "%s",
                  "collective reuse decision unavailable");
      }
      return 1;
   }

   PreconReuseDecision local_decision;
   if (!decision) /* GCOVR_EXCL_BR_LINE */
   {
      memset(&local_decision, 0, sizeof(local_decision));
      decision = &local_decision;
   }

   int local_should_rebuild = hypredrv_PreconReuseShouldRebuild(
      &hypredrv->iargs->precon_reuse, hypredrv->precon_reuse_timesteps.starts,
      hypredrv->stats, &hypredrv->precon_reuse_state, next_ls_id, decision);
   int global_should_rebuild = local_should_rebuild;
   MPI_Allreduce(&local_should_rebuild, &global_should_rebuild, 1, MPI_INT, MPI_MAX,
                 hypredrv->comm);

   /* MPI rank mismatch */
   if (global_should_rebuild != local_should_rebuild) /* GCOVR_EXCL_BR_LINE */
   {
      size_t used = strlen(decision->summary);
      snprintf(decision->summary + used,
               sizeof(decision->summary) - used, /* GCOVR_EXCL_BR_LINE */
               "%scollective_rebuild=%d local_rebuild=%d", used ? " " : "",
               global_should_rebuild, local_should_rebuild);
   }

   decision->should_rebuild = global_should_rebuild;
   return global_should_rebuild;
}

static void
PrepareExplicitObjectForConfiguredExecution(HYPREDRV_t hypredrv, void *obj, int is_matrix)
{
#if !defined(HYPRE_USING_GPU) || !HYPRE_CHECK_MIN_VERSION(22000, 0)
   (void)hypredrv;
   (void)obj;
   (void)is_matrix;
#else
   HYPRE_MemoryLocation target_memory = HYPRE_MEMORY_HOST;

   if (!hypredrv || !hypredrv->iargs || !obj)
   {
      return;
   }

   (void)ApplyGlobalRuntimeSettings(hypredrv);

   if (hypredrv->iargs->ls.exec_policy)
   {
      target_memory = HYPRE_MEMORY_DEVICE;
   }

   if (is_matrix)
   {
      HYPRE_IJMatrixMigrate((HYPRE_IJMatrix)obj, target_memory);
   }
   else
   {
      HYPRE_IJVectorMigrate((HYPRE_IJVector)obj, target_memory);
   }
#endif
}

static void
SetPendingSolvePathContext(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv->stats) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, -1);

   int next_ls_id   = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int timestep_idx = ResolveTimestepIndex(hypredrv->precon_reuse_timesteps.starts,
                                           hypredrv->stats, next_ls_id);
   if (timestep_idx < 0)
   {
      return;
   }
   /* GCOVR_EXCL_BR_LINE */
   if (hypredrv->precon_reuse_timesteps.starts &&
       hypredrv->precon_reuse_timesteps.starts->data) /* GCOVR_EXCL_BR_LINE */
   {
      if ((size_t)timestep_idx >= hypredrv->precon_reuse_timesteps.starts->size ||
          hypredrv->precon_reuse_timesteps.starts->data[timestep_idx] >
             next_ls_id) /* GCOVR_EXCL_BR_LINE */
      {
         return;
      }
   }
   else if (!(hypredrv->stats->level_active & (1 << 0)) ||      /* GCOVR_EXCL_BR_LINE */
            hypredrv->stats->level_solve_start[0] < 0 ||        /* GCOVR_EXCL_BR_LINE */
            hypredrv->stats->level_solve_start[0] > next_ls_id) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   int timestep_id = timestep_idx + 1;
   if (hypredrv->precon_reuse_timesteps.ids &&
       hypredrv->precon_reuse_timesteps.ids->data &&
       (size_t)timestep_idx <
          hypredrv->precon_reuse_timesteps.ids->size) /* GCOVR_EXCL_BR_LINE */
   {
      timestep_id = hypredrv->precon_reuse_timesteps.ids->data[timestep_idx];
   }

   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, timestep_id);
}

static void
PrintStatsWithConfiguredDestination(HYPREDRV_t hypredrv, int print_level)
{
   if (!hypredrv || !hypredrv->stats || print_level < 1) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   const char *filename = NULL;
   if (hypredrv->iargs)
   {
      filename = hypredrv->iargs->general.statistics_filename;
   }

   if (!filename || filename[0] == '\0')
   {
      hypredrv_StatsPrint(hypredrv->stats, print_level);
      return;
   }

   FILE *stream = hypredrv_FopenCreateRestricted(filename, 1, 0);
   if (!stream)
   {
      int saved_errno = errno;
      fprintf(stderr,
              "[HYPREDRV] warning: failed to open general.statistics_filename '%s' "
              "for append (%s). Falling back to stdout.\n",
              filename, strerror(saved_errno));
      hypredrv_StatsPrint(hypredrv->stats, print_level);
      return;
   }

   hypredrv_StatsPrintToStream(hypredrv->stats, print_level, stream);
   fclose(stream);
}

/* Load a timestep schedule from a file.
 * timestep_ids is optional: pass NULL to skip loading per-timestep IDs (caller
 * then derives IDs from array position).  timestep_starts is required. */
static uint32_t
LoadTimestepScheduleFromFile(const char *filename, IntArray **timestep_ids,
                             IntArray **timestep_starts)
{
   if (!timestep_starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid output pointer for timestep schedule");
      return hypredrv_ErrorCodeGet();
   }

   if (timestep_ids && *timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(timestep_ids);
   }

   if (*timestep_starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(timestep_starts);
   }

   if (!filename || filename[0] == '\0') /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing timestep schedule filename");
      return hypredrv_ErrorCodeGet();
   }

   FILE *fp = fopen(filename, "r");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open timestep file: '%s'", filename);
      return hypredrv_ErrorCodeGet();
   }

   int total = 0;
   if (fscanf(fp, "%d", &total) != 1 || total <= 0)
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid timestep file header in '%s'", filename);
      return hypredrv_ErrorCodeGet();
   }

   IntArray *ids = NULL;
   if (timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      ids = hypredrv_IntArrayCreate((size_t)total);
      if (!ids) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate timestep ids array");
         return hypredrv_ErrorCodeGet();
      }
   }

   IntArray *starts = hypredrv_IntArrayCreate((size_t)total);
   if (!starts) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_IntArrayDestroy(&ids);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate timestep starts array");
      return hypredrv_ErrorCodeGet();
   }

   for (int i = 0; i < total; i++)
   {
      int timestep = 0;
      int ls_start = 0;
      if (fscanf(fp, "%d %d", &timestep, &ls_start) != 2 ||
          ls_start < 0) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_IntArrayDestroy(&ids);
         hypredrv_IntArrayDestroy(&starts);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid timestep entry in '%s' at line %d", filename,
                              i + 2);
         return hypredrv_ErrorCodeGet();
      }

      if (ids) /* GCOVR_EXCL_BR_LINE */
      {
         ids->data[i] = timestep;
      }
      starts->data[i] = ls_start;
   }

   fclose(fp);
   if (timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      *timestep_ids = ids;
   }
   *timestep_starts = starts;
   return hypredrv_ErrorCodeGet();
}

static int
PrintSystemNeedsTimestepSchedule(const PrintSystem_args *cfg)
{
   if (!cfg || !cfg->enabled)
   {
      return 0;
   }

   if (cfg->type == PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS)
   {
      return 1;
   }

   if (cfg->type != PRINT_SYSTEM_TYPE_SELECTORS ||
       !cfg->selectors) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   for (size_t i = 0; i < cfg->num_selectors; i++)
   {
      if (cfg->selectors[i].basis == PRINT_SYSTEM_BASIS_TIMESTEP)
      {
         return 1;
      }
   }

   return 0;
}

static int
ResolveTimestepIndex(const IntArray *timestep_starts, const Stats *stats,
                     int system_index)
{
   if (system_index < 0) /* GCOVR_EXCL_BR_LINE */
   {
      return -1;
   }

   if (timestep_starts && timestep_starts->data)
   {
      int found = -1;
      for (size_t i = 0; i < timestep_starts->size; i++)
      {
         if (timestep_starts->data[i] > system_index)
         {
            break;
         }
         found = (int)i;
      }
      if (found >= 0)
      {
         return found;
      }
   }

   if (stats && (stats->level_active & (1 << 0)) && stats->level_current_id[0] > 0)
   {
      return stats->level_current_id[0] - 1;
   }

   return -1;
}

static void
AdvanceLibraryManagedSystemIndex(HYPREDRV_t hypredrv)
{
   int stats_ls_id = -1;

   if (!hypredrv || !hypredrv->lib_mode)
   {
      return;
   }

   if (hypredrv->stats)
   {
      stats_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
   }

   if (hypredrv->current_system_index <= stats_ls_id)
   {
      hypredrv->current_system_index = stats_ls_id + 1;
   }
}

static void
BuildPrintSystemContext(HYPREDRV_t hypredrv, int stage, PrintSystemContext *ctx)
{
   if (!ctx) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   memset(ctx, 0, sizeof(*ctx));
   ctx->stage            = stage;
   ctx->system_index     = 0;
   ctx->timestep_index   = -1;
   ctx->last_iter        = -1;
   ctx->variant_index    = 0;
   ctx->repetition_index = 0;
   ctx->stats_ls_id      = -1;
   ctx->last_setup_time  = -1.0;
   ctx->last_solve_time  = -1.0;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      ctx->level_ids[level] = -1;
   }

   if (!hypredrv) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   if (hypredrv->current_system_index >= 0) /* GCOVR_EXCL_BR_LINE */
   {
      ctx->system_index = hypredrv->current_system_index;
   }

   if (hypredrv->stats)
   {
      ctx->stats_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
      if (hypredrv->current_system_index < 0 && ctx->stats_ls_id >= 0)
      {
         ctx->system_index = ctx->stats_ls_id;
      }

      if (hypredrv->stats->reps > 0)
      {
         ctx->repetition_index = hypredrv->stats->reps - 1;
      }

      for (int level = 0; level < STATS_MAX_LEVELS; level++)
      {
         if (hypredrv->stats->level_current_id[level] > 0)
         {
            ctx->level_ids[level] = hypredrv->stats->level_current_id[level] - 1;
         }
      }

      if (stage == PRINT_SYSTEM_STAGE_SETUP || stage == PRINT_SYSTEM_STAGE_APPLY)
      {
         ctx->last_setup_time = hypredrv_StatsGetLastSetupTime(hypredrv->stats);
      }
      if (stage == PRINT_SYSTEM_STAGE_APPLY)
      {
         ctx->last_iter       = hypredrv_StatsGetLastIter(hypredrv->stats);
         ctx->last_solve_time = hypredrv_StatsGetLastSolveTime(hypredrv->stats);
      }
   }

   if (hypredrv->iargs)
   {
      ctx->variant_index = hypredrv->iargs->active_precon_variant;
   }

   ctx->timestep_index = ResolveTimestepIndex(hypredrv->precon_reuse_timesteps.starts,
                                              hypredrv->stats, ctx->system_index);
}

static void
MaybeDumpLinearSystem(HYPREDRV_t hypredrv, int stage)
{
   if (!hypredrv || !hypredrv->iargs) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   PrintSystemContext ctx;
   BuildPrintSystemContext(hypredrv, stage, &ctx);

   char object_name_buffer[32];
   object_name_buffer[0] = '\0';
   const char *object_name =
      ResolveLogObjectName(hypredrv, object_name_buffer, sizeof(object_name_buffer));
   HYPREDRV_LOG_OBJECTF(
      3, hypredrv,
      "print_system context: stage=%d system_index=%d stats_ls_id=%d "
      "timestep_index=%d last_iter=%d last_setup=%.17g last_solve=%.17g "
      "variant=%d repetition=%d level0=%d level1=%d",
      ctx.stage, ctx.system_index, ctx.stats_ls_id, ctx.timestep_index, ctx.last_iter,
      ctx.last_setup_time, ctx.last_solve_time, ctx.variant_index, ctx.repetition_index,
      ctx.level_ids[0], ctx.level_ids[1]);
   hypredrv_LinearSystemDumpScheduled(
      hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A, hypredrv->mat_M,
      hypredrv->vec_b, hypredrv->vec_x0, hypredrv->vec_xref, hypredrv->vec_x,
      hypredrv->dofmap, &ctx, object_name);
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
   if (hypredrv->vec_x0 && hypredrv->owns_vec_x0) /* GCOVR_EXCL_BR_LINE */
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
   if (!hypredrv || !hypredrv->iargs || hypredrv->lib_mode) /* GCOVR_EXCL_BR_LINE */
   {
      return ERROR_NONE;
   }

   if (hypredrv->iargs->general.exec_policy) /* GCOVR_EXCL_BR_LINE */
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

static uint32_t
DestroyObjectInternal(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv_RuntimeObjectIsLive(hypredrv)) /* GCOVR_EXCL_BR_LINE */
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
                                &hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
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
   if (hypredrv->mat_A && hypredrv->owns_mat_A)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_A);
   }
   if (hypredrv->vec_b && hypredrv->owns_vec_b)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_b);
   }
   if (!hypredrv->lib_mode)
   {
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
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.ids);
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.starts);
   hypredrv_PreconReuseStateDestroy(&hypredrv->precon_reuse_state);
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
   if (hypredrv->iargs)
   {
      hypredrv_PreconArgsDestroyRuntimeState(hypredrv->iargs->precon_method,
                                             &hypredrv->iargs->precon);
   }
#endif
   hypredrv_InputArgsDestroy(&hypredrv->iargs);

   if (print_statistics > 0 && !hypredrv->stats_printed)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "printing statistics on destroy (level=%d)",
                           print_statistics);
      PrintStatsWithConfiguredDestination(hypredrv, print_statistics);
   }
   else if (hypredrv->stats_printed) /* GCOVR_EXCL_BR_LINE */
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

   if (!hypredrv_ptr) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("HYPREDRV_Create requires a non-NULL output pointer");
      HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Create failed: output pointer is NULL");
      return hypredrv_ErrorCodeGet();
   }

   HYPREDRV_t hypredrv = (HYPREDRV_t)calloc(1, sizeof(hypredrv_t));
   /* allocation failure */
   if (!hypredrv) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate HYPREDRV object"); /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOGF(1, mypid, NULL, 0,
                    "HYPREDRV_Create failed: allocation error"); /* GCOVR_EXCL_BR_LINE */
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
   hypredrv->owns_mat_A    = false;
   hypredrv->owns_mat_M    = false;
   hypredrv->owns_vec_b    = false;
   hypredrv->owns_vec_x    = false;
   hypredrv->owns_vec_x0   = false;
   hypredrv->owns_vec_xref = false;

   hypredrv->precon                        = NULL;
   hypredrv->solver                        = NULL;
   hypredrv->precon_is_setup               = false;
   hypredrv->scaling_ctx                   = NULL;
   hypredrv->precon_reuse_timesteps.ids    = NULL;
   hypredrv->precon_reuse_timesteps.starts = NULL;
   hypredrv_PreconReuseStateInit(&hypredrv->precon_reuse_state);
   hypredrv->stats                 = NULL;
   hypredrv->stats_printed         = false;
   hypredrv->runtime_object_id     = 0;
   hypredrv->current_system_index  = -1;
   hypredrv->preferred_exec_policy = 0;
   hypredrv->next_live             = NULL;

   /* Disable library mode by default */
   hypredrv->lib_mode = false;

   /* Each object owns its own stats context. */
   hypredrv->stats = hypredrv_StatsCreate();
   /* allocation failure */
   if (!hypredrv->stats) /* GCOVR_EXCL_BR_LINE */
   {
      free(hypredrv);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate HYPREDRV statistics context");
      HYPREDRV_LOGF(1, mypid, NULL, 0,
                    "HYPREDRV_Create failed: statistics allocation error");
      return hypredrv_ErrorCodeGet();
   }

   /* registry failure */
   if (hypredrv_RuntimeRegisterObject(hypredrv)) /* GCOVR_EXCL_BR_LINE */
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
   if (!hypredrv_ptr) /* GCOVR_EXCL_BR_LINE */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse begin (argc=%d)", argc);

   /* If preset/defaults were configured before parsing, clear old args first.
    * Tear down active solver/preconditioner objects so HYPRE handles are not
    * orphaned when iargs (and any variant storage) is freed. */
   if (hypredrv->iargs)
   {
      if (hypredrv->solver)
      {
         hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
      if (hypredrv->precon)
      {
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
         hypredrv->precon_is_setup = false;
      }
   }
   hypredrv_InputArgsDestroy(&hypredrv->iargs);

   log_object_name[0] = '\0';
   hypredrv_InputArgsParseWithObjectName(
      hypredrv->comm, hypredrv->lib_mode, argc, argv, &hypredrv->iargs,
      ResolveLogObjectName(hypredrv, log_object_name, sizeof(log_object_name)));
   if (hypredrv_ErrorCodeGet()) /* GCOVR_EXCL_BR_LINE */
   {
      HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse failed (code=0x%x)",
                           hypredrv_ErrorCodeGet());
      return hypredrv_ErrorCodeGet();
   }

   /* Apply parsed configuration ------------------------------------------- */

   /* Initialize stats from input args */
   hypredrv_PreconReuseStateReset(&hypredrv->precon_reuse_state);
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

   /* Load timestep schedule for any feature that needs timestep-aware context:
    * preconditioner reuse, stats path labeling, or print-system dumping. */
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.ids);
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.starts);
   if (hypredrv->iargs->ls.timestep_filename[0] != '\0' &&
       ((hypredrv->iargs->general.statistics > 0) ||
        (hypredrv->iargs->precon_reuse.enabled &&
         hypredrv->iargs->precon_reuse.per_timestep) ||
        PrintSystemNeedsTimestepSchedule(&hypredrv->iargs->ls.print_system)))
   {
      LoadTimestepScheduleFromFile(hypredrv->iargs->ls.timestep_filename,
                                   &hypredrv->precon_reuse_timesteps.ids,
                                   &hypredrv->precon_reuse_timesteps.starts);
   } /* GCOVR_EXCL_BR_LINE */
   else if (hypredrv->iargs->ls.sequence_filename[0] != '\0' &&
            ((hypredrv->iargs->general.statistics > 0) ||
             (hypredrv->iargs->precon_reuse.enabled &&
              hypredrv->iargs->precon_reuse.per_timestep) ||
             PrintSystemNeedsTimestepSchedule(
                &hypredrv->iargs->ls.print_system))) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_LSSeqReadTimestepsWithIds(hypredrv->iargs->ls.sequence_filename,
                                         &hypredrv->precon_reuse_timesteps.ids,
                                         &hypredrv->precon_reuse_timesteps.starts);
   }

#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   hypredrv->preferred_exec_policy = hypredrv->iargs->general.exec_policy;
   if (hypredrv->preferred_exec_policy &&
       hypredrv->iargs->precon_method == PRECON_BOOMERAMG)
   {
      int interp_type = hypredrv->iargs->precon.amg.interpolation.prolongation_type;
      if (interp_type == 8 || interp_type == 9)
      {
         hypredrv->iargs->general.exec_policy = 0;
         hypredrv->iargs->ls.exec_policy      = 0;
         HYPREDRV_LOG_OBJECTF(
            1, hypredrv,
            "forcing host execution for compatibility in InputArgsParse: "
            "BoomerAMG standard interpolation");
         HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      }
   }
#endif

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   hypredrv->lib_mode = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_ObjectSetName(HYPREDRV_t hypredrv, const char *name)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_StatsSetObjectName(hypredrv->stats, name);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv, int *warmup)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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

#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
static void
LogMGRCachedHandles(HYPREDRV_t hypredrv, const MGR_args *mgr, const char *msg)
{
   int num_frelax = 0;
   int num_grelax = 0;
   int num_coarse = 0;
   hypredrv_MGRCountCachedSolvers(mgr, &num_frelax, &num_grelax, &num_coarse);
   if (num_frelax || num_grelax || num_coarse) /* GCOVR_EXCL_BR_LINE */
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "%s: coarse=%d frelax=%d grelax=%d", msg,
                           num_coarse, num_frelax, num_grelax);
   }
}
#endif

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv, int variant_idx)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsSetPreconVariant begin (idx=%d)",
                        variant_idx);

   if (!hypredrv->iargs) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Input arguments not parsed");
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "preconditioner variant switch failed: input args missing");
      return hypredrv_ErrorCodeGet();
   }

   /* GCOVR_EXCL_BR_START */
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
   /* GCOVR_EXCL_BR_STOP */

   int      current_variant = hypredrv->iargs->active_precon_variant;
   int      variant_changed = (variant_idx != current_variant);
   precon_t current_method  = hypredrv->iargs->precon_method;
   HYPREDRV_LOG_OBJECTF(
      2, hypredrv, "preconditioner variant selection: current=%d requested=%d changed=%d",
      current_variant, variant_idx, variant_changed);

   /* Only rebuild solver/preconditioner when switching variants. */
   if (variant_changed)
   {
      bool had_precon = (hypredrv->precon != NULL);

      /* GCOVR_EXCL_BR_LINE */
      if (hypredrv->solver)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "switching preconditioner variant: destroying active "
                              "solver");
         hypredrv_SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }

      /* GCOVR_EXCL_BR_LINE */
      if (hypredrv->precon)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "switching preconditioner variant: destroying active "
                              "preconditioner");
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
         hypredrv->precon_is_setup = false;
      }

#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
      if (!had_precon && current_method == PRECON_MGR)
      {
         LogMGRCachedHandles(
            hypredrv, &hypredrv->iargs->precon.mgr,
            "discarding cached MGR handles before switching preconditioner variant");
      }
#endif
      if (!had_precon)
      {
         hypredrv_PreconArgsDestroyRuntimeState(current_method, &hypredrv->iargs->precon);
      }
      hypredrv->precon_is_setup = false;
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "preconditioner variant unchanged; reusing setup");
   }

   /* Preserve runtime-managed state (for example cached MGR component handles)
    * when the active variant is unchanged. */
   hypredrv->iargs->active_precon_variant = variant_idx;
   if (variant_changed)
   {
      hypredrv->iargs->precon_method = hypredrv->iargs->precon_methods[variant_idx];
      hypredrv->iargs->precon        = hypredrv->iargs->precon_variants[variant_idx];
   }

   /* GPU-only exec-policy coupling */
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   int desired_exec_policy = hypredrv->preferred_exec_policy;
   if (hypredrv->iargs->precon_method == PRECON_BOOMERAMG)
   {
      int interp_type = hypredrv->iargs->precon.amg.interpolation.prolongation_type;
      if (interp_type == 8 || interp_type == 9)
      {
         desired_exec_policy = 0;
      }
   }
   if (hypredrv->iargs->general.exec_policy != desired_exec_policy)
   {
      hypredrv->iargs->general.exec_policy = desired_exec_policy;
      hypredrv->iargs->ls.exec_policy      = desired_exec_policy;
      if (desired_exec_policy)
      {
         HYPREDRV_LOG_OBJECTF(1, hypredrv,
                              "restoring device execution in InputArgsSetPreconVariant: "
                              "active variant is GPU-compatible");
      }
      else
      {
         HYPREDRV_LOG_OBJECTF(
            1, hypredrv,
            "forcing host execution for compatibility in InputArgsSetPreconVariant: "
            "BoomerAMG standard interpolation");
      }
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->mat_A, 1);
      if (hypredrv->mat_M && hypredrv->mat_M != hypredrv->mat_A)
      {
         PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->mat_M, 1);
      }
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->vec_b, 0);
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->vec_x, 0);
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->vec_x0, 0);
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->vec_xref, 0);
      PrepareExplicitObjectForConfiguredExecution(hypredrv, hypredrv->vec_nn, 0);
   } /* GCOVR_EXCL_BR_LINE */
#endif
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   if (!preset) /* GCOVR_EXCL_BR_LINE */
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

   int      variant_idx    = hypredrv->iargs->active_precon_variant;
   precon_t current_method = hypredrv->iargs->precon_method;
   if (variant_idx < 0) /* GCOVR_EXCL_BR_LINE */
   {
      variant_idx                            = 0;
      hypredrv->iargs->active_precon_variant = 0;
   }

   /* Destroy existing preconditioner object if any */
   if (hypredrv->precon) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                             &hypredrv->precon, hypredrv->stats,
                             hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
      hypredrv->precon_is_setup = false;
   }
   else
   {
#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
      if (current_method == PRECON_MGR)
      {
         LogMGRCachedHandles(
            hypredrv, &hypredrv->iargs->precon.mgr,
            "discarding cached MGR handles before applying preconditioner preset");
      }
#endif
      hypredrv_PreconArgsDestroyRuntimeState(current_method, &hypredrv->iargs->precon);
   }
   if (hypredrv->iargs->num_precon_variants <= 0) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv->iargs->num_precon_variants   = 1;
      hypredrv->iargs->active_precon_variant = 0;
      variant_idx                            = 0;
   }
   if (variant_idx >= hypredrv->iargs->num_precon_variants) /* GCOVR_EXCL_BR_LINE */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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

   if (hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_SolverGetValidTypeIntMap(),
                                                preset))
   {
      DestroyActiveSolver(hypredrv);
      hypredrv->iargs->solver_method = (solver_t)hypredrv_StrIntMapArrayGetImage(
         hypredrv_SolverGetValidTypeIntMap(), preset);
      hypredrv_SolverArgsSetDefaultsForMethod(hypredrv->iargs->solver_method,
                                              &hypredrv->iargs->solver);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_InputArgsApplySolverPreset(hypredrv->iargs, preset);
   if (hypredrv_ErrorCodeGet())
   {
      return hypredrv_ErrorCodeGet();
   }

   DestroyActiveSolver(hypredrv);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SolverPresetRegister(const char *name, const char *yaml_text, const char *help)
{
   if (!name || !yaml_text)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_SolverPresetRegister: name and yaml_text must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_SolverGetValidTypeIntMap(),
                                                name))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_SolverPresetRegister: preset '%s' conflicts with built-in solver",
         name);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_PresetRegisterTyped(name, yaml_text, help, HYPREDRV_PRESET_SOLVER);

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

   hypredrv_PresetRegisterTyped(name, yaml_text, help, HYPREDRV_PRESET_PRECON);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates, HYPRE_IJVector *vecs)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
         /* GCOVR_EXCL_BR_START */
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "HYPREDRV_StateVectorSet failed: missing vector at "
                              "index=%d",
                              i);
         return hypredrv_ErrorCodeGet();
         /* GCOVR_EXCL_BR_STOP */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "HYPREDRV_StateVectorCopy failed: source or destination "
                           "vector is NULL");
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPREDRV_StateVectorApplyCorrection begin (state_idx=%d)", state_idx);

   if (state_idx < 0 || state_idx >= hypredrv->nstates)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("state_idx %d out of range [0, %d)", state_idx,
                           hypredrv->nstates);
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "HYPREDRV_StateVectorApplyCorrection failed: state index out "
                           "of range");
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild begin");

   hypredrv->current_system_index++;
   /* GCOVR_EXCL_BR_START */
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetReferenceSolution(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, NULL));
   HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));
   HYPREDRV_SAFE_CALL(LinearSystemSetVectorTagsInternal(hypredrv));
   /* GCOVR_EXCL_BR_STOP */

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
   MaybeDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_BUILD);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   hypredrv_LinearSystemReadMatrix(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->mat_A,
                                   hypredrv->stats);
   hypredrv->owns_mat_A = (hypredrv->mat_A != NULL);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJMatrix)mat_A, 1);

   LinearSystemDropOwnedPrecMatrix(hypredrv);

   if (hypredrv->mat_A && hypredrv->owns_mat_A &&
       hypredrv->mat_A != (HYPRE_IJMatrix)mat_A)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_A);
   }

   hypredrv->mat_A      = (HYPRE_IJMatrix)mat_A;
   hypredrv->mat_M      = (HYPRE_IJMatrix)mat_A;
   hypredrv->owns_mat_A = (bool)(!hypredrv->lib_mode && mat_A != NULL);
   hypredrv->owns_mat_M = false;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   if (!vec)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));
      if (hypredrv->vec_xref && !hypredrv->owns_vec_xref) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv->vec_xref = NULL;
      }
      hypredrv_LinearSystemSetRHS(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                  &hypredrv->vec_xref, &hypredrv->vec_b, hypredrv->stats);
      hypredrv->owns_vec_b    = (hypredrv->vec_b != NULL);
      hypredrv->owns_vec_xref = (hypredrv->vec_xref != NULL);
   }
   else
   {
      PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJVector)vec, 0);
      if (hypredrv->vec_b && hypredrv->owns_vec_b &&
          hypredrv->vec_b != (HYPRE_IJVector)vec)
      {
         HYPRE_IJVectorDestroy(hypredrv->vec_b);
      }
      hypredrv->vec_b      = (HYPRE_IJVector)vec;
      hypredrv->owns_vec_b = (bool)(!hypredrv->lib_mode);
   }

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrixFromCSR(HYPREDRV_t hypredrv, HYPRE_BigInt row_start,
                                      HYPRE_BigInt row_end, const HYPRE_BigInt *indptr,
                                      const HYPRE_BigInt *col_indices,
                                      const HYPRE_Real   *data)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   /* Replacing mat_A invalidates any preconditioner matrix derived from it. */
   LinearSystemDropOwnedPrecMatrix(hypredrv);

   if (hypredrv->mat_A && hypredrv->owns_mat_A)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_A);
   }
   hypredrv->mat_A      = NULL;
   hypredrv->owns_mat_A = false;

   /* Caller-provided CSR buffers are host pointers, so the builder runs on host
    * regardless of the configured execution policy. */
   (void)hypredrv_LinearSystemBuildMatrixFromCSR(hypredrv->comm, HYPRE_MEMORY_HOST,
                                                 row_start, row_end, indptr, col_indices,
                                                 data, &hypredrv->mat_A);

   hypredrv->owns_mat_A = (hypredrv->mat_A != NULL);
   hypredrv->mat_M      = hypredrv->mat_A;
   hypredrv->owns_mat_M = false;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHSFromArray(HYPREDRV_t hypredrv, HYPRE_BigInt row_start,
                                     HYPRE_BigInt row_end, const HYPRE_Real *values)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   if (hypredrv->mat_A)
   {
      HYPRE_BigInt ilower = 0, iupper = -1, jlower = 0, jupper = -1;
      if (HYPRE_IJMatrixGetLocalRange(hypredrv->mat_A, &ilower, &iupper, &jlower,
                                      &jupper) == 0 &&
          (row_start != ilower || row_end != iupper))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "HYPREDRV_LinearSystemSetRHSFromArray: RHS row range [%lld, %lld] "
            "does not match matrix row range [%lld, %lld]",
            (long long)row_start, (long long)row_end, (long long)ilower,
            (long long)iupper);
         return hypredrv_ErrorCodeGet();
      }
   }

   if (hypredrv->vec_b && hypredrv->owns_vec_b)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_b);
   }
   hypredrv->vec_b      = NULL;
   hypredrv->owns_vec_b = false;

   /* Caller-provided RHS buffers are host pointers, so the builder runs on host
    * regardless of the configured execution policy. */
   (void)hypredrv_LinearSystemBuildRHSFromArray(
      hypredrv->comm, HYPRE_MEMORY_HOST, row_start, row_end, values, &hypredrv->vec_b);

   hypredrv->owns_vec_b = (hypredrv->vec_b != NULL);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                      int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
      PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJVector)vec, 0);
      LinearSystemDropOwnedInitialGuess(hypredrv);
      hypredrv->vec_x0 = (HYPRE_IJVector)vec;
      hypredrv->owns_vec_x0 =
         (bool)(!hypredrv->lib_mode && hypredrv->vec_x0 != hypredrv->vec_x &&
                hypredrv->vec_x0 != hypredrv->vec_b); /* GCOVR_EXCL_BR_LINE */
      if (hypredrv->vec_x && !hypredrv->owns_vec_x)   /* GCOVR_EXCL_BR_LINE */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

   if (!vec)
   {
      /* Discard any borrowed reference before recreating. */
      if (!hypredrv->owns_vec_x) /* GCOVR_EXCL_BR_LINE */
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
      PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJVector)vec, 0);
      /* Destroy existing owned solution before replacing. */
      if (hypredrv->vec_x && hypredrv->owns_vec_x) /* GCOVR_EXCL_BR_LINE */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
      if (uses_xref_file) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv->owns_vec_xref = (hypredrv->vec_xref != NULL);
      }
   }
   else
   {
      PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJVector)vec, 0);
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

   if (!hypredrv->vec_x0 || !hypredrv->vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return hypredrv_ErrorCodeGet();
   }

   char default_object_name[32];
   bool pushed_default_name = PushDefaultLogObjectName(hypredrv, default_object_name,
                                                       sizeof(default_object_name));
   hypredrv_LinearSystemResetInitialGuess(hypredrv->vec_x0, hypredrv->vec_x,
                                          hypredrv->stats);
   PopDefaultLogObjectName(hypredrv, default_object_name, pushed_default_name);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static uint32_t
LinearSystemSetVectorTagsInternal(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

   if (!mat)
   {
      HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv)); /* GCOVR_EXCL_BR_LINE */
      LinearSystemDropOwnedPrecMatrix(hypredrv);
      hypredrv_LinearSystemSetPrecMatrix(hypredrv->comm, &hypredrv->iargs->ls,
                                         hypredrv->mat_A, &hypredrv->mat_M,
                                         hypredrv->stats); /* GCOVR_EXCL_BR_LINE */
      hypredrv->owns_mat_M =
         (bool)(hypredrv->mat_M != NULL &&
                hypredrv->mat_M != hypredrv->mat_A); /* GCOVR_EXCL_BR_LINE */
   }
   else
   {
      PrepareExplicitObjectForConfiguredExecution(hypredrv, (HYPRE_IJMatrix)mat, 1);
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconCreate begin");
   HYPREDRV_SAFE_CALL(ApplyGlobalRuntimeSettings(hypredrv));

   int                 next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   PreconReuseDecision decision;
   bool                should_create = (hypredrv->precon == NULL);
   if (!should_create)
   {
      should_create =
         PreconReuseShouldRebuildCollective(hypredrv, next_ls_id, &decision) != 0;
   }
   else
   {
      memset(&decision, 0, sizeof(decision));
      decision.should_rebuild = 1;
      decision.used_adaptive =
         hypredrv->iargs->precon_reuse.policy == PRECON_REUSE_POLICY_ADAPTIVE;
      snprintf(decision.summary, sizeof(decision.summary), "%s",
               "create because preconditioner is NULL");
   }
   hypredrv_PreconReuseLogDecision(hypredrv, next_ls_id, &decision, "PreconCreate");

   if (should_create)
   {
      if (hypredrv->iargs->precon_reuse.policy == PRECON_REUSE_POLICY_ADAPTIVE)
      {
         hypredrv_PreconReuseMarkRebuild(hypredrv, &hypredrv->precon_reuse_state);
      }
      /* If we're recreating, destroy the existing preconditioner first to avoid leaks. */
      if (hypredrv->precon)
      {
         if (hypredrv->iargs->precon_method == PRECON_MGR)
         {
            hypredrv_MGRSelectCachedSolversToKeep(&hypredrv->iargs->precon.mgr,
                                                  hypredrv->precon_reuse_timesteps.starts,
                                                  hypredrv->stats, next_ls_id);
            int num_frelax = 0;
            int num_grelax = 0;
            int num_coarse = 0;
            hypredrv_MGRCountKeepFlags(&hypredrv->iargs->precon.mgr, &num_frelax,
                                       &num_grelax, &num_coarse);
            if (num_frelax || num_grelax || num_coarse) /* GCOVR_EXCL_BR_LINE */
            {
               HYPREDRV_LOG_OBJECTF(2, hypredrv,
                                    "preserving cached MGR handles across recreate: "
                                    "coarse=%d frelax=%d "
                                    "grelax=%d",
                                    num_coarse, num_frelax, num_grelax);
            }
         }
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
         hypredrv->precon_is_setup = false;
      }
      else if (hypredrv->iargs->precon_method == PRECON_MGR)
      {
         int num_frelax = 0;
         int num_grelax = 0;
         int num_coarse = 0;
         hypredrv_MGRCountCachedSolvers(&hypredrv->iargs->precon.mgr, &num_frelax,
                                        &num_grelax, &num_coarse);
         if (num_frelax || num_grelax || num_coarse) /* GCOVR_EXCL_BR_LINE */
         {
            HYPREDRV_LOG_OBJECTF(2, hypredrv,
                                 "cached MGR handles before create: coarse=%d frelax=%d "
                                 "grelax=%d",
                                 num_coarse, num_frelax, num_grelax);
         }
      }
      hypredrv_PreconCreate(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                            hypredrv->dofmap, hypredrv->vec_nn, &hypredrv->precon,
                            hypredrv->stats, next_ls_id);
      hypredrv->precon_is_setup = false;
      if (hypredrv_ErrorCodeActive() && hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_LINE */
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner create failed (code=0x%x)",
                              hypredrv_ErrorCodeGet());
         hypredrv_ErrorCodeDescribe(hypredrv_ErrorCodeGet());
         hypredrv_ErrorMsgPrint();
      }
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
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
      /* Precon create failure */
      if (HYPREDRV_PreconCreate(hypredrv))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);

         if (hypredrv_LogEnabled(2))
         {
            HYPREDRV_LOG_OBJECTF(2, hypredrv,
                                 "linear solver create returning with code=0x%x",
                                 hypredrv_ErrorCodeGet());
         }
         HYPREDRV_LOG_OBJECTF(1, hypredrv,
                              "HYPREDRV_LinearSolverCreate failed: preconditioner "
                              "create");
         return hypredrv_ErrorCodeGet();
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "skipping preconditioner creation (precon already created, "
                           "pending setup)");
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup begin");

   int next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   if (hypredrv->precon && hypredrv->precon_is_setup &&
       hypredrv->iargs->precon_method == PRECON_MGR &&
       hypredrv_MGRComponentReuseSetupMode(&hypredrv->iargs->precon.mgr, hypredrv->stats,
                                           next_ls_id))
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "rerunning MGR setup with preserved component handles");
   }

   hypredrv_PreconSetup(hypredrv->iargs->precon_method, hypredrv->precon,
                        hypredrv->mat_A);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */
   if (hypredrv->precon)   /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv->precon_is_setup = true;
   }
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup end"); /* GCOVR_EXCL_BR_LINE */

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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup begin");

   AdvanceLibraryManagedSystemIndex(hypredrv);

   int                 next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   PreconReuseDecision decision;
   int                 should_rebuild =
      PreconReuseShouldRebuildCollective(hypredrv, next_ls_id, &decision);
   hypredrv_PreconReuseLogDecision(hypredrv, next_ls_id, &decision, "LinearSolverSetup");
   int rerun_mgr_component_setup =
      (hypredrv->precon != NULL) && hypredrv->precon_is_setup &&
      hypredrv->iargs->precon_method == PRECON_MGR &&
      hypredrv_MGRComponentReuseSetupMode(&hypredrv->iargs->precon.mgr, hypredrv->stats,
                                          next_ls_id);
   int skip_precon_setup = (hypredrv->precon != NULL) && hypredrv->precon_is_setup &&
                           !should_rebuild && !rerun_mgr_component_setup;
   HYPREDRV_LOG_OBJECTF(2, hypredrv,
                        "solver setup decisions: rebuild_precon=%d rerun_mgr_setup=%d "
                        "skip_precon_setup=%d",
                        should_rebuild, rerun_mgr_component_setup, skip_precon_setup);

   /* Propagate dofmap to vectors (no-op if no dofmap is set) */
   /* GCOVR_EXCL_BR_LINE */ /* SAFE_CALL fail-fast wrapper */
   HYPREDRV_SAFE_CALL(
      LinearSystemSetVectorTagsInternal(hypredrv)); /* GCOVR_EXCL_BR_LINE */

   /* Create scaling context if needed and not already created */
   if (hypredrv->iargs->scaling.enabled && !hypredrv->scaling_ctx)
   {
      hypredrv_ScalingContextCreate(hypredrv->comm,
                                    &hypredrv->scaling_ctx); /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "created scaling context (type=%d)",
                           (int)hypredrv->iargs->scaling.type); /* GCOVR_EXCL_BR_LINE */
   }

   /* Compute scaling if enabled (recompute for each system if needed) */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      hypredrv_ScalingCompute(hypredrv->comm, &hypredrv->iargs->scaling,
                              hypredrv->scaling_ctx, hypredrv->mat_A, hypredrv->vec_b,
                              hypredrv->dofmap); /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_ErrorCodeGet())               /* GCOVR_EXCL_BR_LINE */
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Apply scaling before setup if enabled */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                    hypredrv->mat_M, hypredrv->vec_b,
                                    hypredrv->vec_x); /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_ErrorCodeGet())                    /* GCOVR_EXCL_BR_LINE */
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   GMRESSetRefSolution(hypredrv);

   char default_object_name[32];
   bool pushed_default_name = PushDefaultLogObjectName(hypredrv, default_object_name,
                                                       sizeof(default_object_name));
   hypredrv_SolverSetupWithReuse(hypredrv->iargs->precon_method,
                                 hypredrv->iargs->solver_method, hypredrv->precon,
                                 hypredrv->solver, hypredrv->mat_M, hypredrv->vec_b,
                                 hypredrv->vec_x, hypredrv->stats, skip_precon_setup);
   PopDefaultLogObjectName(hypredrv, default_object_name, pushed_default_name);

   HYPRE_ClearAllErrors();
   if (hypredrv->precon && !skip_precon_setup)
   {
      hypredrv->precon_is_setup = true;
   }
   MaybeDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_SETUP);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply begin");

   if (!hypredrv->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER); /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(
         1, hypredrv,
         "HYPREDRV_LinearSolverApply failed: solver is NULL"); /* GCOVR_EXCL_BR_LINE */
      return hypredrv_ErrorCodeGet();
   }

   double e_norm = 0.0, x_norm = 0.0, xref_norm = 0.0;
   double b_norm = 0.0, r_norm = 0.0, r0_norm = 0.0;
   int    solve_succeeded = 1;

   /* Ensure GMRES always sees the current reference solution, including on reused
    * preconditioner cycles where SolverSetup may be skipped. */
   GMRESSetRefSolution(hypredrv);

   /* Apply scaling if enabled but not yet applied (e.g., when preconditioner is reused)
    */    /* GCOVR_EXCL_BR_LINE */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       !hypredrv->scaling_ctx->is_applied) /* GCOVR_EXCL_BR_LINE */
   {                                       /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "applying deferred system scaling"); /* GCOVR_EXCL_BR_LINE */
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                    hypredrv->mat_M, hypredrv->vec_b,
                                    hypredrv->vec_x); /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_ErrorCodeGet())                    /* GCOVR_EXCL_BR_LINE */
      {
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Perform solve */ /* GCOVR_EXCL_BR_LINE */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       hypredrv->scaling_ctx->is_applied)                         /* GCOVR_EXCL_BR_LINE */
   {                                                              /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving scaled system"); /* GCOVR_EXCL_BR_LINE */
      int xref_scaled = 0;

      /* Compute initial residual norm before solve (on current system state) */
      hypredrv_LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b,
                                               hypredrv->vec_x, "L2", &r0_norm);

      if (hypredrv->vec_xref)
      {
         hypredrv_ScalingApplyToVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                       SCALING_VECTOR_UNKNOWN); /* GCOVR_EXCL_BR_LINE */
         if (hypredrv_ErrorCodeGet())                           /* GCOVR_EXCL_BR_LINE */
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
      /* GCOVR_EXCL_BR_LINE */ /* solver failure requires hypre injection */
      if (iters < 0)           /* GCOVR_EXCL_BR_LINE */
      {
         solve_succeeded = 0; /* GCOVR_EXCL_BR_LINE */ /* paired with iters<0 path */
         if (xref_scaled)                              /* GCOVR_EXCL_BR_LINE */
         {
            hypredrv_ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref,
                                         SCALING_VECTOR_UNKNOWN);
         }
         hypredrv_StatsIterSet(hypredrv->stats, 0);
         hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END,
                                "solve"); /* GCOVR_EXCL_BR_LINE */
         if (hypredrv->iargs &&
             hypredrv->iargs->precon_reuse.enabled && /* GCOVR_EXCL_BR_LINE */
             hypredrv->iargs->precon_reuse.policy ==
                PRECON_REUSE_POLICY_ADAPTIVE) /* GCOVR_EXCL_BR_LINE */
         {
            PreconReuseObservation obs;
            hypredrv_PreconReuseBuildObservation(
               hypredrv, hypredrv->precon_reuse_timesteps.starts, &obs);
            obs.solve_succeeded = solve_succeeded;
            hypredrv_PreconReuseStateRecordObservation(&hypredrv->precon_reuse_state,
                                                       &obs);
         }
         return hypredrv_ErrorCodeGet();
      }

      hypredrv_StatsIterSet(hypredrv->stats, (int)iters);
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");

      /* Undo scaling on solution and restore original system */
      hypredrv_ScalingUndoOnSystem(hypredrv->scaling_ctx, hypredrv->mat_A,
                                   hypredrv->mat_M, hypredrv->vec_b, hypredrv->vec_x);
      /* GCOVR_EXCL_BR_LINE */     /* undo failure needs injection */
      if (hypredrv_ErrorCodeGet()) /* GCOVR_EXCL_BR_LINE */
      {                            /* GCOVR_EXCL_BR_LINE */
         if (xref_scaled)          /* GCOVR_EXCL_BR_LINE */
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
         /* GCOVR_EXCL_BR_LINE */     /* xref undo failure */
         if (hypredrv_ErrorCodeGet()) /* GCOVR_EXCL_BR_LINE */
         {
            return hypredrv_ErrorCodeGet();
         }
      }

      /* Compute residual norms on original (now restored) system */
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_b, "L2", &b_norm);
      hypredrv_LinearSystemComputeResidualNorm(hypredrv->mat_A, hypredrv->vec_b,
                                               hypredrv->vec_x, "L2",
                                               &r_norm); /* GCOVR_EXCL_BR_LINE */
      b_norm = (b_norm > 0.0) ? b_norm : 1.0;            /* GCOVR_EXCL_BR_LINE */
      hypredrv_StatsRelativeResNormSet(hypredrv->stats, r_norm / b_norm);
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving unscaled system");
      SetPendingSolvePathContext(hypredrv);
      /* No scaling - use standard hypredrv_SolverApply which handles everything including
       * stats */
      uint32_t error_before_solve = hypredrv_ErrorCodeGet();
      char     default_object_name[32];
      bool pushed_default_name = PushDefaultLogObjectName(hypredrv, default_object_name,
                                                          sizeof(default_object_name));
      hypredrv_SolverApply(hypredrv->iargs->solver_method, hypredrv->solver,
                           hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x,
                           hypredrv->stats);
      PopDefaultLogObjectName(hypredrv, default_object_name, pushed_default_name);
      solve_succeeded = (hypredrv_ErrorCodeGet() == error_before_solve);
      /* hypredrv_SolverApply already computed and set all stats */
   }

   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   if (hypredrv->vec_xref)
   {
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_xref, "L2", &xref_norm);
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x, "L2", &x_norm);
      hypredrv_LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, "L2",
                                            &e_norm); /* GCOVR_EXCL_BR_LINE */
      if (!hypredrv->mypid)                           /* GCOVR_EXCL_BR_LINE */
      {
         printf("L2 norm of error: %e\n", (double)e_norm);
         printf("L2 norm of solution: %e\n", (double)x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
      }
   }
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "solve finished (iters=%d)",
                        hypredrv_StatsGetLastIter(hypredrv->stats));
   if (hypredrv->iargs && hypredrv->iargs->precon_reuse.enabled &&
       hypredrv->iargs->precon_reuse.policy == PRECON_REUSE_POLICY_ADAPTIVE)
   {
      PreconReuseObservation obs;
      hypredrv_PreconReuseBuildObservation(hypredrv,
                                           hypredrv->precon_reuse_timesteps.starts, &obs);
      obs.solve_succeeded = solve_succeeded;
      hypredrv_PreconReuseStateRecordObservation(&hypredrv->precon_reuse_state, &obs);
   }
   MaybeDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_APPLY);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply end");

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   hypredrv_ErrorStateReset();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply begin");

   if (!hypredrv->precon)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_PreconApply: preconditioner is NULL"); /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "HYPREDRV_PreconApply failed: preconditioner is NULL"); /* GCOVR_EXCL_BR_LINE */
      return hypredrv_ErrorCodeGet();
   }
   if (!hypredrv->mat_A || !vec_b || !vec_x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("HYPREDRV_PreconApply requires matrix, rhs, and solution "
                           "vectors"); /* GCOVR_EXCL_BR_LINE
                                        */
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "HYPREDRV_PreconApply failed: matrix or vector is NULL"); /* GCOVR_EXCL_BR_LINE
                                                                    */
      return hypredrv_ErrorCodeGet();
   }
   /* GCOVR_EXCL_BR_LINE */
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner apply: method=%d has_precon=%d",
                        (int)hypredrv->iargs->precon_method,
                        (hypredrv->precon != NULL)); /* GCOVR_EXCL_BR_LINE */

   hypredrv_PreconApply(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A,
                        (HYPRE_IJVector)vec_b, (HYPRE_IJVector)vec_x);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */  /* GCOVR_EXCL_BR_LINE */
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply end"); /* GCOVR_EXCL_BR_LINE */

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconDestroy begin");

   int                 next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   PreconReuseDecision decision;
   bool                should_destroy =
      PreconReuseShouldRebuildCollective(hypredrv, next_ls_id, &decision) != 0;
   hypredrv_PreconReuseLogDecision(hypredrv, next_ls_id, &decision, "PreconDestroy");

   if (should_destroy)
   {
      if (hypredrv->precon)
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
         if (hypredrv->iargs->precon_method == PRECON_MGR)
         {
            hypredrv_MGRSelectCachedSolversToKeep(&hypredrv->iargs->precon.mgr,
                                                  hypredrv->precon_reuse_timesteps.starts,
                                                  hypredrv->stats, next_ls_id);
            int num_frelax = 0;
            int num_grelax = 0;
            int num_coarse = 0;
            hypredrv_MGRCountKeepFlags(&hypredrv->iargs->precon.mgr, &num_frelax,
                                       &num_grelax, &num_coarse);
            if (num_frelax || num_grelax || num_coarse)
            {
               HYPREDRV_LOG_OBJECTF(
                  2, hypredrv,
                  "preserving cached MGR handles across destroy: coarse=%d frelax=%d "
                  "grelax=%d",
                  num_coarse, num_frelax, num_grelax);
            }
         }
#endif
         HYPREDRV_LOG_OBJECTF(2, hypredrv, "destroying preconditioner object");
         hypredrv_PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                                &hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
         hypredrv->precon_is_setup = false;
      }
      else
      { /* GCOVR_EXCL_BR_LINE */
         HYPREDRV_LOG_OBJECTF(
            2, hypredrv,
            "preconditioner destroy requested but object is NULL"); /* GCOVR_EXCL_BR_LINE
                                                                     */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverDestroy begin");

   /* First, destroy the preconditioner if we need */
   if (hypredrv->precon)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "linear solver destroy: evaluating preconditioner teardown");
      /* GCOVR_EXCL_BR_LINE */              /* rare teardown failure */
      if (HYPREDRV_PreconDestroy(hypredrv)) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON); /* GCOVR_EXCL_BR_LINE */
         HYPREDRV_LOG_OBJECTF(                        /* GCOVR_EXCL_BR_LINE */
                              2, hypredrv,
                              "linear solver destroy failed: preconditioner teardown "
                              "error"); /* GCOVR_EXCL_BR_LINE */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();
   /* GCOVR_EXCL_BR_LINE */
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StatsPrint");
   /* GCOVR_EXCL_BR_LINE */ /* GCOVR_EXCL_BR_LINE */
   PrintStatsWithConfiguredDestination(hypredrv, hypredrv->iargs
                                                    ? hypredrv->iargs->general.statistics
                                                    : 0); /* GCOVR_EXCL_BR_LINE */
   hypredrv->stats_printed = true;

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, name,
                                       id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(HYPREDRV_t hypredrv, const char *name, int id)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   return hypredrv_StatsAnnotateWithId(hypredrv->stats, HYPREDRV_ANNOTATE_END, name, id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   return hypredrv_StatsAnnotateLevelWithId(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN,
                                            level, name, id);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(HYPREDRV_t hypredrv, int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();
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
      warned_eigspec_disabled = true; /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "eigenspectrum support disabled in build; emitted warning"); /* GCOVR_EXCL_BR_LINE
                                                                       */
   }
   else
   { /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum support disabled in build; warning already "
                           "emitted"); /* GCOVR_EXCL_BR_LINE
                                        */
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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

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
   HYPREDRV_CHECK_INIT_AND_OBJ();

   return hypredrv_StatsLevelGetEntrySummary(hypredrv->stats, level, index, entry_id,
                                             num_solves, linear_iters, setup_time,
                                             solve_time);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelPrint(HYPREDRV_t hypredrv, int level)
{
   HYPREDRV_CHECK_INIT_AND_OBJ();

   hypredrv_StatsLevelPrint(hypredrv->stats, level);
   return hypredrv_ErrorCodeGet();
}
