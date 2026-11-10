/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "object.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "_hypre_parcsr_mv.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "internal/presets.h"
#include "internal/utils.h"
#include "logging.h"

typedef struct
{
   bool        initialized;
   int         active_count;
   int         next_object_id;
   hypredrv_t *live_head;
} RuntimeState;

static RuntimeState g_runtime_state = {.next_object_id = 1};

/* Return the registry link as well as validating the opaque handle. Destruction
 * uses the same link to unlink it without another lookup or invalid dereference. */
static hypredrv_t **
FindLiveObjectLink(hypredrv_t *hypredrv)
{
   for (hypredrv_t **link = &g_runtime_state.live_head; *link; link = &(*link)->next_live)
   {
      if (*link == hypredrv)
      {
         return link;
      }
   }
   return NULL;
}

/* Slot order fixes ownership priority, independently of destruction order. */
void
hypredrv_ObjectInstallVectors(hypredrv_t *hypredrv, int count,
                              const hypredrv_VectorUpdate *updates)
{
   hypredrv_VectorSlot *slots[] = {&hypredrv->vec_b,        &hypredrv->vec_x,
                                   &hypredrv->vec_x0,       &hypredrv->vec_xref,
                                   &hypredrv->vec_coord[0], &hypredrv->vec_coord[1],
                                   &hypredrv->vec_coord[2]};
   enum
   {
      nslots = sizeof(slots) / sizeof(slots[0])
   };
   hypredrv_VectorSlot before[nslots], after[nslots];
   bool                clear_tags = false;
   for (int i = 0; i < nslots; i++)
   {
      before[i] = after[i] = *slots[i];
      for (int j = 0; j < count; j++)
      {
         if (updates[j].slot == slots[i])
         {
            after[i] = (hypredrv_VectorSlot){updates[j].handle, updates[j].owned};
            clear_tags |= i < 4;
         }
      }
   }
   if (clear_tags) hypredrv_VectorTagsClear(hypredrv);
   for (int i = 0; i < nslots; i++)
   {
      bool existing = false, owned = false, requested = false, earlier = false;
      for (int j = 0; j < nslots; j++)
      {
         existing |= before[j].handle == after[i].handle;
         owned |= before[j].handle == after[i].handle && before[j].owned;
         requested |= after[j].handle == after[i].handle && after[j].owned;
         earlier |= j < i && after[j].handle == after[i].handle;
      }
      after[i].owned = after[i].handle && (owned || (!existing && requested)) && !earlier;
   }
   for (int i = 0; i < nslots; i++)
   {
      if (!before[i].handle || !before[i].owned) continue;
      bool retained = false, earlier = false;
      for (int j = 0; j < nslots; j++)
      {
         retained |= after[j].handle == before[i].handle;
         earlier |= j < i && before[j].handle == before[i].handle && before[j].owned;
      }
      if (!retained && !earlier) HYPRE_IJVectorDestroy(before[i].handle);
   }
   for (int i = 0; i < nslots; i++) *slots[i] = after[i];
}

void
hypredrv_ObjectReplaceMatrix(hypredrv_t *hypredrv, hypredrv_MatrixSlot *slot,
                             HYPRE_IJMatrix handle, bool owned)
{
   enum
   {
      nslots = MAX_MGR_LEVELS + 3
   };
   hypredrv_MatrixSlot *slots[nslots] = {&hypredrv->mat_A, &hypredrv->mat_M,
                                         &hypredrv->mat_G, &hypredrv->mat_C};
   hypredrv_MatrixSlot  before[nslots], after[nslots];
   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
      slots[4 + i] = &hypredrv->mat_coarse_schur[i];
   for (int i = 0; i < nslots; i++)
   {
      before[i] = after[i] = *slots[i];
      if (slot == slots[i]) after[i] = (hypredrv_MatrixSlot){handle, owned};
   }
   for (int i = 0; i < nslots; i++)
   {
      bool existing = false, has_owner = false, requested = false, earlier = false;
      for (int j = 0; j < nslots; j++)
      {
         existing |= before[j].handle == after[i].handle;
         has_owner |= before[j].handle == after[i].handle && before[j].owned;
         requested |= after[j].handle == after[i].handle && after[j].owned;
         earlier |= j < i && after[j].handle == after[i].handle;
      }
      after[i].owned =
         after[i].handle && (has_owner || (!existing && requested)) && !earlier;
   }
   for (int i = 0; i < nslots; i++)
   {
      if (!before[i].handle || !before[i].owned) continue;
      bool retained = false, earlier = false;
      for (int j = 0; j < nslots; j++)
      {
         retained |= after[j].handle == before[i].handle;
         earlier |= j < i && before[j].handle == before[i].handle && before[j].owned;
      }
      if (!retained && !earlier) HYPRE_IJMatrixDestroy(before[i].handle);
   }
   for (int i = 0; i < nslots; i++) *slots[i] = after[i];
}

uint32_t
hypredrv_ObjectCheckInit(void)
{
   if (!g_runtime_state.initialized)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED);
      return hypredrv_ErrorCodeGet();
   }
   return ERROR_NONE;
}

uint32_t
hypredrv_ObjectCheckLive(hypredrv_t *hypredrv)
{
   if (hypredrv_ObjectCheckInit())
   {
      return hypredrv_ErrorCodeGet();
   }
   if (!FindLiveObjectLink(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return hypredrv_ErrorCodeGet();
   }
   return ERROR_NONE;
}

void
hypredrv_ObjectApplyRuntimeSettings(hypredrv_t *hypredrv)
{
   if (!hypredrv->iargs) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   if (hypredrv->iargs->general.exec_policy) /* GCOVR_EXCL_BR_LINE */
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#endif

#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      HYPRE_SetSpGemmUseVendor(hypredrv->iargs->general.use_vendor_spgemm);
      HYPRE_SetSpMVUseVendor(hypredrv->iargs->general.use_vendor_spmv);
      HYPRE_SetSpTransUseVendor(hypredrv->iargs->general.use_vendor_sptrans);
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
}

void
hypredrv_ObjectPrepareExecution(hypredrv_t *hypredrv, void *obj, int is_matrix)
{
#if !defined(HYPRE_USING_GPU)
   (void)hypredrv;
   (void)obj;
   (void)is_matrix;
#else
   if (!hypredrv->iargs || !obj)
   {
      return;
   }

   hypredrv_ObjectApplyRuntimeSettings(hypredrv);

   HYPRE_MemoryLocation target_memory =
      hypredrv->iargs->ls.exec_policy ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
   if (is_matrix)
   {
      hypredrv_IJMatrixMigrate((HYPRE_IJMatrix)obj, target_memory);
   }
   else
   {
      hypredrv_IJVectorMigrate((HYPRE_IJVector)obj, target_memory);
   }
#endif
}

static void
SetPendingSolvePathContext(hypredrv_t *hypredrv)
{
   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, -1);

   int next_ls_id   = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int timestep_idx = hypredrv_PreconReuseResolveTimestepIndex(
      hypredrv->precon_reuse_timesteps.starts, hypredrv->stats, next_ls_id);
   if (timestep_idx < 0)
   {
      return;
   }
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

void
hypredrv_ObjectPrintStats(hypredrv_t *hypredrv, int print_level)
{
   if (print_level < 1) /* GCOVR_EXCL_BR_LINE */
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

void
hypredrv_ObjectDumpLinearSystem(hypredrv_t *hypredrv, int stage)
{
   PrintSystemContext ctx = {0};
   ctx.stage              = stage;
   ctx.system_index =
      hypredrv->current_system_index >= 0 ? hypredrv->current_system_index : 0;
   ctx.stats_ls_id      = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
   ctx.last_iter        = -1;
   ctx.last_setup_time  = -1.0;
   ctx.last_solve_time  = -1.0;
   ctx.variant_index    = hypredrv->iargs->active_precon_variant;
   ctx.repetition_index = hypredrv->stats->reps > 0 ? hypredrv->stats->reps - 1 : 0;
   if (hypredrv->current_system_index < 0 && ctx.stats_ls_id >= 0)
   {
      ctx.system_index = ctx.stats_ls_id;
   }
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      int id               = hypredrv->stats->level_current_id[level];
      ctx.level_ids[level] = id > 0 ? id - 1 : -1;
   }
   if (stage == PRINT_SYSTEM_STAGE_SETUP || stage == PRINT_SYSTEM_STAGE_APPLY)
   {
      ctx.last_setup_time = hypredrv_StatsGetLastSetupTime(hypredrv->stats);
   }
   if (stage == PRINT_SYSTEM_STAGE_APPLY)
   {
      ctx.last_iter       = hypredrv_StatsGetLastIter(hypredrv->stats);
      ctx.last_solve_time = hypredrv_StatsGetLastSolveTime(hypredrv->stats);
   }
   ctx.timestep_index = hypredrv_PreconReuseResolveTimestepIndex(
      hypredrv->precon_reuse_timesteps.starts, hypredrv->stats, ctx.system_index);

   char        object_name_buffer[32] = {0};
   const char *object_name = hypredrv_ResolveLogObjectName(hypredrv, object_name_buffer,
                                                           sizeof(object_name_buffer));
   HYPREDRV_LOG_OBJECTF(
      3, hypredrv,
      "print_system context: stage=%d system_index=%d stats_ls_id=%d "
      "timestep_index=%d last_iter=%d last_setup=%.17g last_solve=%.17g "
      "variant=%d repetition=%d level0=%d level1=%d",
      ctx.stage, ctx.system_index, ctx.stats_ls_id, ctx.timestep_index, ctx.last_iter,
      ctx.last_setup_time, ctx.last_solve_time, ctx.variant_index, ctx.repetition_index,
      ctx.level_ids[0], ctx.level_ids[1]);
   hypredrv_LinearSystemDumpScheduled(hypredrv->comm, &hypredrv->iargs->ls,
                                      hypredrv->mat_A.handle, hypredrv->mat_M.handle,
                                      hypredrv->vec_b.handle, hypredrv->vec_x0.handle,
                                      hypredrv->vec_xref.handle, hypredrv->vec_x.handle,
                                      hypredrv->dofmap, &ctx, object_name);
}

void
hypredrv_ObjectDestroySolvers(hypredrv_t *hypredrv)
{
   hypredrv_SolverDestroy(&hypredrv->solver);
   if (hypredrv->precon)
   {
      hypredrv_PreconDestroy(&hypredrv->precon, hypredrv->stats,
                             hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
      hypredrv->precon_is_setup = false;
   }
}

static void
DestroyLiveObject(hypredrv_t **live_link)
{
   hypredrv_t *hypredrv = *live_link;

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "DestroyObjectInternal begin");

   /* Remove the handle from the live registry before tearing down owned state
    * so finalize-time sweeps never revisit an object that is already mid-destroy. */
   *live_link          = hypredrv->next_live;
   hypredrv->next_live = NULL;
   g_runtime_state.active_count--;
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "unregistered object (active=%d)",
                        g_runtime_state.active_count);
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "live object removed from registry (active=%d)",
                        g_runtime_state.active_count);

   /* Embedded/library-mode consumers like GEOS expect general.statistics to
    * flush with the object lifetime rather than from a separate driver hook. */
   int print_statistics = 0;
   if (hypredrv->lib_mode && hypredrv->iargs && hypredrv->mypid == 0)
   {
      print_statistics = hypredrv->iargs->general.statistics;
   }

   /* Destroy solver/preconditioner objects before tearing down dependent state. */
   hypredrv_ObjectDestroySolvers(hypredrv);

   hypredrv_VectorTagsClear(hypredrv);
   if (hypredrv->scaling_ctx)
   {
      hypredrv_ScalingContextDestroy(&hypredrv->scaling_ctx);
   }
   hypredrv_MatrixSlotRelease(&hypredrv->mat_M);
   hypredrv->precmat_sequence_cache_args = NULL;
   hypredrv_MatrixSlotRelease(&hypredrv->mat_A);
   /* Attachment installation assigns each shared handle a single owner. */
   hypredrv_VectorSlotRelease(&hypredrv->vec_xref);
   hypredrv_VectorSlotRelease(&hypredrv->vec_x0);
   hypredrv_VectorSlotRelease(&hypredrv->vec_x);
   hypredrv_VectorSlotRelease(&hypredrv->vec_b);
   if (hypredrv->vec_nn)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_nn);
   }
   if (hypredrv->vec_ns)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_ns);
   }
   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      hypredrv_MatrixSlotRelease(&hypredrv->mat_coarse_schur[i]);
   }
   hypredrv_MatrixSlotRelease(&hypredrv->mat_G);
   hypredrv_MatrixSlotRelease(&hypredrv->mat_C);
   for (int i = 0; i < 3; i++)
   {
      hypredrv_VectorSlotRelease(&hypredrv->vec_coord[i]);
   }

   hypredrv_IntArrayDestroy(&hypredrv->dofmap);
   hypredrv_PreconReuseStateDestroy(&hypredrv->precon_reuse_state);
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.ids);
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.starts);
   if (hypredrv->iargs)
   {
      hypredrv_PreconArgsDestroyRuntimeState(hypredrv->iargs->precon_method,
                                             &hypredrv->iargs->precon);
   }

   if (print_statistics > 0 && !hypredrv->stats_printed)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "printing statistics on destroy (level=%d)",
                           print_statistics);
      hypredrv_ObjectPrintStats(hypredrv, print_statistics);
   }
   else if (hypredrv->stats_printed) /* GCOVR_EXCL_BR_LINE */
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "skipping statistics on destroy (already printed)");
   }

   hypredrv_InputArgsDestroy(&hypredrv->iargs);

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "DestroyObjectInternal end");
   hypredrv_StatsDestroy(&hypredrv->stats);

   free(hypredrv->vec_s);
   free(hypredrv);
}

void
hypredrv_ObjectInitialize(void)
{
   if (!g_runtime_state.initialized)
   {
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "runtime initialize begin");

      /* Initialize hypre */
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Initialize();
      /* Preserve HYPRE's source-level diagnostics until HypreDrive consumes
       * the corresponding error code at an API boundary.  At trace verbosity,
       * print at the point of failure as well: nested solver setup can return a
       * nonzero callback status after the originating HYPRE error state has
       * already been changed by its caller. */
      HYPRE_SetPrintErrorMode(hypredrv_LogEnabled(3) ? 0 : 1);
#if HYPRE_CHECK_MIN_VERSION(23100, 0)
      static const char *const env_names[] = {
         "MV2_USE_CUDA",
         "MV2_USE_HIP",
         "MPIR_CVAR_ENABLE_GPU",
         "MPICH_GPU_SUPPORT_ENABLED",
      };
      const char *gpu_aware_mpi_env = NULL;
      for (size_t i = 0; i < sizeof(env_names) / sizeof(env_names[0]); i++)
      {
         const char *value = getenv(env_names[i]);
         if (value && strcmp(value, "1") == 0)
         {
            gpu_aware_mpi_env = env_names[i];
            break;
         }
      }
      HYPRE_SetGpuAwareMPI(gpu_aware_mpi_env != NULL);
      if (gpu_aware_mpi_env)
      {
         HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "GPU-aware MPI enabled by %s=1",
                            gpu_aware_mpi_env);
      }
      else
      {
         HYPREDRV_LOG_COMMF(
            1, MPI_COMM_WORLD, NULL, 0,
            "GPU-aware MPI disabled (no recognized environment variable is set to 1)");
      }

      /*
       * HYPRE_DeviceInitialize() is optional.  Leave device initialization
       * lazy so that HYPREDRV_Initialize() does not eagerly create HIP and
       * vendor-library state before the input configuration has selected the
       * execution policy.  HYPRE initializes the device state on the first
       * device operation.
       */
#endif
#endif

#if HYPRE_CHECK_MIN_VERSION(23100, 16)
      const char *env_log_level = getenv("HYPRE_LOG_LEVEL");
      HYPRE_Int   log_level     = 0;
      if (env_log_level)
      {
         char *end = NULL;
         long  v   = strtol(env_log_level, &end, 10);
         /* Validate and clamp: ignore garbage and out-of-range values rather than
          * forwarding them verbatim to hypre. */
         if (end != env_log_level && *end == '\0' && v >= 0 && v <= 4)
         {
            log_level = (HYPRE_Int)v;
         }
      }

      HYPRE_SetLogLevel(log_level);
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                         "forwarded HYPRE_LOG_LEVEL=%d to hypre", (int)log_level);
#endif

      g_runtime_state.initialized = true;
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "runtime initialize end");
   }
}

void
hypredrv_ObjectFinalize(void)
{
   if (g_runtime_state.initialized)
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                         "HYPREDRV_Finalize sees %d active object(s)",
                         g_runtime_state.active_count);
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "finalize sweep begin (active=%d)",
                         g_runtime_state.active_count);
      while (g_runtime_state.live_head)
      {
         HYPREDRV_LOG_OBJECTF(2, g_runtime_state.live_head,
                              "auto-destroying live object during finalize");
         DestroyLiveObject(&g_runtime_state.live_head);
      }
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "finalize sweep end (active=%d)",
                         g_runtime_state.active_count);
   }

   hypredrv_LSSeqReleaseCaches();
   if (g_runtime_state.initialized)
   {
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "runtime finalize begin");
      hypredrv_PresetFreeUserPresets();
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Finalize();
#endif
      g_runtime_state.initialized = false;
      HYPREDRV_LOG_COMMF(1, MPI_COMM_WORLD, NULL, 0, "runtime finalize end");
   }
   g_runtime_state = (RuntimeState){.next_object_id = 1};

   /* Release message buffers and logging state between independent lifetimes. */
   hypredrv_ErrorStateReset();
   hypredrv_LogReset();
}

uint32_t
hypredrv_ObjectCreate(MPI_Comm comm, hypredrv_t **hypredrv_ptr)
{
   hypredrv_t *hypredrv = (hypredrv_t *)calloc(1, sizeof(hypredrv_t));
   /* allocation failure */
   if (!hypredrv) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_AllocationFailure("Failed to allocate HYPREDRV object");
      HYPREDRV_LOG_COMMF(
         1, comm, NULL, 0,
         "HYPREDRV_Create failed: allocation error"); /* GCOVR_EXCL_BR_LINE */
      return hypredrv_ErrorCodeGet();
   }

   MPI_Comm_rank(comm, &hypredrv->mypid);
   MPI_Comm_size(comm, &hypredrv->nprocs);

   /* calloc zeroed every pointer, counter and ownership flag (library mode off). */
   hypredrv->comm                 = comm;
   hypredrv->current_system_index = -1;

   /* Each object owns its own stats context. */
   hypredrv->stats = hypredrv_StatsCreate();
   /* allocation failure */
   if (!hypredrv->stats) /* GCOVR_EXCL_BR_LINE */
   {
      free(hypredrv);
      hypredrv_AllocationFailure("Failed to allocate HYPREDRV statistics context");
      HYPREDRV_LOG_COMMF(1, comm, NULL, 0,
                         "HYPREDRV_Create failed: statistics allocation error");
      return hypredrv_ErrorCodeGet();
   }

   hypredrv->runtime_object_id = g_runtime_state.next_object_id++;
   hypredrv->next_live         = g_runtime_state.live_head;
   g_runtime_state.live_head   = hypredrv;
   g_runtime_state.active_count++;
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "registered object (active=%d)",
                        g_runtime_state.active_count);

   /* Propagate runtime ID to stats so solver/linsys log helpers can
      fall back to "obj-N" without access to the full hypredrv object. */
   hypredrv->stats->runtime_object_id = hypredrv->runtime_object_id;

   /* Set output pointer */
   *hypredrv_ptr = hypredrv;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_Create end (active=%d)",
                        g_runtime_state.active_count);

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectDestroy(hypredrv_t **hypredrv_ptr)
{
   hypredrv_t *hypredrv = *hypredrv_ptr;

   hypredrv_t **live_link = FindLiveObjectLink(hypredrv);
   if (!live_link)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return hypredrv_ErrorCodeGet();
   }

   int mypid = hypredrv->mypid;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_Destroy begin");
   DestroyLiveObject(live_link);
   *hypredrv_ptr = NULL;
   HYPREDRV_LOGF(1, mypid, NULL, 0, "HYPREDRV_Destroy end");

   return hypredrv_ErrorCodeGet();
}

/* Solver workflows receive live objects and own collective input validation,
 * execution preparation and scaling/reuse cleanup. Diagnostics are preserved. */

static void
RequireSolverArguments(hypredrv_t *hypredrv)
{
   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Solver workflows require parsed input arguments; call "
                           "HYPREDRV_InputArgsParse or select a preset first");
   }
}

typedef enum
{
   PREPARE_PRECON_SETUP,
   PREPARE_PRECON_APPLY,
   PREPARE_SOLVER_SETUP,
   PREPARE_SOLVER_APPLY,
} SolverPreparation;

/* Reading host values can migrate a registered vector after attachment. Native
 * setup must create its work vectors in the configured memory space, and apply
 * must restore that space before reusing the resulting solver/preconditioner. */
static bool
PrepareSolverExecution(hypredrv_t *hypredrv, SolverPreparation operation,
                       HYPRE_IJVector rhs, HYPRE_IJVector solution)
{
   RequireSolverArguments(hypredrv);
   bool solver = operation == PREPARE_SOLVER_SETUP || operation == PREPARE_SOLVER_APPLY;
   bool apply  = operation == PREPARE_SOLVER_APPLY || operation == PREPARE_PRECON_APPLY;
   HYPRE_IJMatrix matrix =
      operation == PREPARE_SOLVER_SETUP ? hypredrv->mat_M.handle : hypredrv->mat_A.handle;
   if (solver && !hypredrv->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("Linear solver must be created before setup or apply");
   }
   bool needs_precon =
      !solver || (hypredrv->iargs && hypredrv->iargs->precon_method != PRECON_NONE);
   if (needs_precon &&
       (!hypredrv->precon ||
        (apply && hypredrv->precon->method != PRECON_NONE && !hypredrv->precon_is_setup)))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(apply ? "Apply requires a successfully set up preconditioner; "
                                   "check the preceding setup error"
                                 : "Preconditioner must be created before setup");
   }
   bool needs_matrix = operation != PREPARE_PRECON_SETUP ||
                       (hypredrv->iargs && hypredrv->iargs->precon_method != PRECON_NONE);
   if ((needs_matrix && !matrix) ||
       (operation != PREPARE_PRECON_SETUP && (!rhs || (apply && !solution))) ||
       (solver && !hypredrv->mat_A.handle))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Solver setup/apply requires its configured matrix and vectors");
   }
   hypredrv_HypreConsumeErrors(hypredrv->comm);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return true;
#ifdef HYPRE_USING_GPU
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   HYPRE_MemoryLocation location =
      hypredrv->iargs->general.exec_policy ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
   if (matrix) hypredrv_IJMatrixMigrate(matrix, location);
   if (rhs) hypredrv_IJVectorMigrate(rhs, location);
   if (solution) hypredrv_IJVectorMigrate(solution, location);
   hypredrv_HypreConsumeErrors(hypredrv->comm);
   return hypredrv_DistributedErrorStateSync(hypredrv->comm);
#else
   return false;
#endif
}

/* Restore every caller-owned object after a scaled operation, even when an
 * earlier error is still active. ScalingUndoOnSystem preserves that error
 * around its inverse transforms; the final synchronization also propagates a
 * rank-local cleanup failure without creating a second divergent return. */
static bool
RestoreScaledSystemState(hypredrv_t *hypredrv, int reference_is_scaled)
{
   if (hypredrv->scaling_ctx && hypredrv_ScalingIsApplied(hypredrv->scaling_ctx))
   {
      hypredrv_ScalingUndoOnSystem(hypredrv->scaling_ctx);
   }
   if (reference_is_scaled && hypredrv->vec_xref.handle)
   {
      hypredrv_ScalingUndoOnVector(hypredrv->scaling_ctx, hypredrv->vec_xref.handle,
                                   SCALING_VECTOR_UNKNOWN);
   }

   return hypredrv_DistributedErrorStateSync(hypredrv->comm);
}

/* State arithmetic is local. Report native failures without introducing a
 * communicator-wide diagnostic collective into GetValues/Copy/correction. */
static bool
StateVectorCallSucceeded(HYPRE_Int code, const char *operation)
{
   if (!code) return true;
   char message[HYPRE_MAX_MSG_LEN];
   HYPRE_DescribeError(code, message);
   hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
   hypredrv_ErrorMsgAdd("%s: HYPRE call failed: %s", operation, message);
   return false;
}

/* Validate a newly supplied IJ handle once before using its native storage.
 * Empty ranks legitimately have no coefficient allocation. */
static hypre_ParVector *
StateVectorReadInput(HYPRE_IJVector vector, const char *operation)
{
   HYPRE_Int type   = -1;
   void     *object = NULL;
   if (!StateVectorCallSucceeded(HYPRE_IJVectorGetObjectType(vector, &type), operation))
   {
      return NULL;
   }
   if (type != HYPRE_PARCSR)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: state/solution must be a ParCSR vector", operation);
      return NULL;
   }
   if (!StateVectorCallSucceeded(HYPRE_IJVectorGetObject(vector, &object), operation))
   {
      return NULL;
   }
   hypre_ParVector *par   = (hypre_ParVector *)object;
   hypre_Vector    *local = par ? hypre_ParVectorLocalVector(par) : NULL;
   if (!local || hypre_VectorSize(local) < 0 || hypre_VectorNumVectors(local) < 1 ||
       (hypre_VectorSize(local) > 0 && !hypre_VectorData(local)))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: state/solution must be initialized", operation);
      return NULL;
   }
   return par;
}

static bool
StateVectorLayoutsMatch(HYPRE_IJVector source_ij, HYPRE_IJVector target_ij,
                        hypre_ParVector *source, hypre_ParVector *target,
                        const char *operation)
{
   HYPRE_BigInt source_first = 0, source_last = -1, target_first = 0, target_last = -1;
   if (!StateVectorCallSucceeded(
          HYPRE_IJVectorGetLocalRange(source_ij, &source_first, &source_last),
          operation) ||
       !StateVectorCallSucceeded(
          HYPRE_IJVectorGetLocalRange(target_ij, &target_first, &target_last), operation))
   {
      return false;
   }
   /* ParVector indices are relative to the IJ vector's global first row. */
   if (source_first == target_first && source_last == target_last &&
       hypre_ParVectorGlobalSize(source) == hypre_ParVectorGlobalSize(target) &&
       hypre_ParVectorFirstIndex(source) == hypre_ParVectorFirstIndex(target) &&
       hypre_ParVectorLastIndex(source) == hypre_ParVectorLastIndex(target) &&
       hypre_VectorSize(hypre_ParVectorLocalVector(source)) ==
          hypre_VectorSize(hypre_ParVectorLocalVector(target)) &&
       hypre_ParVectorNumVectors(source) == hypre_ParVectorNumVectors(target))
   {
      return true;
   }
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd("%s: vector row ranges, global sizes, and component counts "
                        "must match",
                        operation);
   return false;
}

uint32_t
hypredrv_ObjectSetStateVectors(hypredrv_t *hypredrv, int count,
                               const HYPRE_IJVector *vectors)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorSet begin (nstates=%d)", count);
   if (count <= 0 || !vectors)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("HYPREDRV_StateVectorSet: nstates must be > 0 and vecs "
                           "must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }
   HYPRE_IJVector *replacement =
      hypredrv_Calloc((size_t)count, sizeof(*replacement), "state vector handles");
   if (!replacement) return hypredrv_ErrorCodeGet();
   for (int i = 0; i < count; i++)
   {
      if (!vectors[i])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("HYPREDRV_StateVectorSet: NULL state vector at index %d",
                              i);
         free(replacement);
         return hypredrv_ErrorCodeGet();
      }
      if (!StateVectorReadInput(vectors[i], "HYPREDRV_StateVectorSet"))
      {
         free(replacement);
         return hypredrv_ErrorCodeGet();
      }
      replacement[i] = vectors[i];
   }
   free(hypredrv->vec_s);
   hypredrv->vec_s   = replacement;
   hypredrv->nstates = count;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_StateVectorSet end");
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectCopyStateVectors(hypredrv_t *hypredrv, int source, int target)
{
   if (hypredrv->vec_s[source] == hypredrv->vec_s[target])
   {
      return hypredrv_ErrorCodeGet();
   }
   void *source_object = NULL, *target_object = NULL;
   if (!StateVectorCallSucceeded(
          HYPRE_IJVectorGetObject(hypredrv->vec_s[source], &source_object),
          "StateVectorCopy") ||
       !StateVectorCallSucceeded(
          HYPRE_IJVectorGetObject(hypredrv->vec_s[target], &target_object),
          "StateVectorCopy"))
   {
      return hypredrv_ErrorCodeGet();
   }
   hypre_ParVector *from = (hypre_ParVector *)source_object;
   hypre_ParVector *to   = (hypre_ParVector *)target_object;
   if (StateVectorLayoutsMatch(hypredrv->vec_s[source], hypredrv->vec_s[target], from, to,
                               "StateVectorCopy"))
   {
      /* Native Copy supports distinct source and destination memory locations. */
      StateVectorCallSucceeded(
         HYPRE_ParVectorCopy((HYPRE_ParVector)from, (HYPRE_ParVector)to),
         "StateVectorCopy");
   }
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectApplyStateCorrection(hypredrv_t *hypredrv, int index)
{
   hypre_ParVector *correction =
      StateVectorReadInput(hypredrv->vec_x.handle, "StateVectorApplyCorrection");
   void *state_object = NULL;
   if (!correction || !StateVectorCallSucceeded(
                         HYPRE_IJVectorGetObject(hypredrv->vec_s[index], &state_object),
                         "StateVectorApplyCorrection"))
   {
      return hypredrv_ErrorCodeGet();
   }
   hypre_ParVector *state = (hypre_ParVector *)state_object;
   if (!StateVectorLayoutsMatch(hypredrv->vec_x.handle, hypredrv->vec_s[index],
                                correction, state, "StateVectorApplyCorrection"))
   {
      return hypredrv_ErrorCodeGet();
   }
#ifdef HYPRE_USING_GPU
   /* GetSolutionValues may have exposed the correction on the host. Match the
    * state to its actual storage, leaving the correction and its pointers alone. */
   if (!StateVectorCallSucceeded(
          hypredrv_IJVectorMigrate(hypredrv->vec_s[index],
                                   hypre_ParVectorMemoryLocation(correction)),
          "StateVectorApplyCorrection"))
   {
      return hypredrv_ErrorCodeGet();
   }
#endif
   StateVectorCallSucceeded(HYPRE_ParVectorAxpy((HYPRE_Complex)1.0,
                                                (HYPRE_ParVector)correction,
                                                (HYPRE_ParVector)state),
                            "StateVectorApplyCorrection");
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectLinearSystemBuild(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild begin");
   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Linear-system construction requires parsed arguments");
   }
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return hypredrv_ErrorCodeGet();
   if (RestoreScaledSystemState(hypredrv, 0)) return hypredrv_ErrorCodeGet();
   hypredrv->current_system_index++;
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LinearSystemReadMatrix(hypredrv)) return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetRHS(hypredrv, NULL)) return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetInitialGuess(hypredrv, GUESS_REPLACE, NULL))
      return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetReferenceSolution(hypredrv, NULL))
      return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetPrecMatrix(hypredrv, NULL)) return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetDofmap(hypredrv, DOFMAP_FROM_INPUT, 0, 0, NULL))
      return hypredrv_ErrorCodeGet();
   hypredrv_VectorTagsSet(hypredrv);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return hypredrv_ErrorCodeGet();
   /* GCOVR_EXCL_BR_STOP */

   /* The new system prepares its own scaling coefficients. */
   hypredrv_ScalingContextDestroy(&hypredrv->scaling_ctx);

   long long int num_rows = hypredrv_LinearSystemMatrixGetNumRows(hypredrv->mat_A.handle);
   long long int num_nonzeros =
      hypredrv_LinearSystemMatrixGetNumNonzeros(hypredrv->mat_A.handle);
   if (!hypredrv->mypid)
   {
      PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH);
      printf("Solving linear system #%d ",
             hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
      printf("with %lld rows and %lld nonzeros...\n", num_rows, num_nonzeros);
   }
   hypredrv_HypreConsumeErrors(hypredrv->comm);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return hypredrv_ErrorCodeGet();
   hypredrv_ObjectDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_BUILD);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSystemBuild end");

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectPreconCreate(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconCreate begin");
   RequireSolverArguments(hypredrv);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) goto cleanup;
   if (hypredrv->iargs->precon_method == PRECON_MGR)
      hypredrv_MGRValidateOuterSolver(&hypredrv->iargs->precon.mgr,
                                      (int)hypredrv->iargs->solver_method, "MGR");
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
   {
      /* Drop a preconditioner built for an earlier system: this configuration is
       * invalid for the current outer solver, and leaving the previous object
       * installed lets a caller that only logs the error keep applying it. */
      if (hypredrv->precon)
      {
         hypredrv_PreconDestroy(&hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
      }
      hypredrv->precon_is_setup = false;
      goto cleanup;
   }
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);

   int  next_ls_id    = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   bool should_create = hypredrv_PreconReuseDecide(hypredrv, PRECON_REUSE_CREATE);
   if (hypredrv_ErrorCodeActive()) goto cleanup;
   if (should_create)
   {
      /* If we're recreating, destroy the existing preconditioner first to avoid leaks. */
      if (hypredrv->precon)
      {
         hypredrv_PreconDestroy(&hypredrv->precon, hypredrv->stats,
                                hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
         hypredrv->precon_is_setup = false;
      }
      PreconOperators precon_ops = {hypredrv->mat_G.handle,
                                    hypredrv->mat_C.handle,
                                    {hypredrv->vec_coord[0].handle,
                                     hypredrv->vec_coord[1].handle,
                                     hypredrv->vec_coord[2].handle},
                                    hypredrv->mat_coarse_schur};
      hypredrv_PreconCreate(hypredrv->comm, hypredrv->iargs->precon_method,
                            &hypredrv->iargs->precon, hypredrv->dofmap, hypredrv->vec_nn,
                            &hypredrv->precon, hypredrv->stats, next_ls_id, &precon_ops);
      hypredrv->precon_is_setup = false;
      if (!hypredrv_ErrorCodeActive() && hypredrv->precon &&
          hypredrv->iargs->precon_method == PRECON_BOOMERAMG)
      {
         /* Label the unknowns for systems AMG (no-op without a dofmap or with
            num_functions <= 1). The preconditioner matrix is already bound to
            the object at this point in both driver and library flows. */
         hypredrv_AMGSetDofFunc(
            &hypredrv->iargs->precon.amg, hypredrv->dofmap, hypredrv->precon->main,
            hypredrv->mat_M.handle ? hypredrv->mat_M.handle : hypredrv->mat_A.handle);
      }
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

cleanup:
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconCreate end");

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectLinearSolverCreate(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverCreate begin");

   /* The reuse module collectively handles missing and pending native handles;
    * no rank may skip its decision based on local lifecycle state. */
   if (hypredrv_ObjectPreconCreate(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);

      if (hypredrv_LogEnabled(2))
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "linear solver create returning with code=0x%x",
                              hypredrv_ErrorCodeGet());
      }
      HYPREDRV_LOG_OBJECTF(1, hypredrv,
                           "HYPREDRV_LinearSolverCreate failed: preconditioner create");
      return hypredrv_ErrorCodeGet();
   }

   /* Always recreate solver per linear system.
    * Reuse policy applies to preconditioner lifecycle; solver internals should
    * be rebuilt against the current system vectors/matrix each cycle. */
   hypredrv_Solver *created = NULL;
   hypredrv_SolverCreate(hypredrv->comm, hypredrv->iargs->solver_method,
                         &hypredrv->iargs->solver, &created);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
   {
      hypredrv_SolverDestroy(&created);
      return hypredrv_ErrorCodeGet();
   }
   hypredrv_SolverDestroy(&hypredrv->solver);
   hypredrv->solver = created;
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "solver created (method=%d)",
                        (int)hypredrv->iargs->solver_method);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverCreate end");

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectPreconSetup(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup begin");
   if (PrepareSolverExecution(hypredrv, PREPARE_PRECON_SETUP, NULL, NULL))
      return hypredrv_ErrorCodeGet();
   if (RestoreScaledSystemState(hypredrv, 0)) return hypredrv_ErrorCodeGet();
   /* Standalone setup builds the hierarchy in the original coordinates. */
   hypredrv_ScalingContextDestroy(&hypredrv->scaling_ctx);

   int next_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   if (hypredrv->precon && hypredrv->precon_is_setup &&
       hypredrv->iargs->precon_method == PRECON_MGR &&
       hypredrv_MGRComponentReuseSetupMode(&hypredrv->iargs->precon.mgr, hypredrv->stats,
                                           next_ls_id))
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "rerunning MGR setup with preserved component handles");
   }

   hypredrv_PreconSetup(hypredrv->precon, hypredrv->mat_A.handle);
   hypredrv_HypreConsumeErrors(hypredrv->comm);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
   {
      hypredrv->precon_is_setup = false;
      HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup failed");
      return hypredrv_ErrorCodeGet();
   }
   if (hypredrv->precon) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv->precon_is_setup = true;
   }
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconSetup end"); /* GCOVR_EXCL_BR_LINE */

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectLinearSolverSetup(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup begin");

   if (PrepareSolverExecution(hypredrv, PREPARE_SOLVER_SETUP, hypredrv->vec_b.handle,
                              hypredrv->vec_x.handle))
   {
      RestoreScaledSystemState(hypredrv, 0);
      return hypredrv_ErrorCodeGet();
   }
   if (RestoreScaledSystemState(hypredrv, 0)) return hypredrv_ErrorCodeGet();
   if (hypredrv_LinearSystemSetInitialGuess(hypredrv, GUESS_ENSURE_SOLUTION, NULL))
   {
      RestoreScaledSystemState(hypredrv, 0);
      return hypredrv_ErrorCodeGet();
   }
   uint32_t code = hypredrv_VectorTagsSet(hypredrv);
   if (code != ERROR_NONE)
   {
      return code;
   }

   if (hypredrv->lib_mode)
   {
      int stats_ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats);
      if (hypredrv->current_system_index <= stats_ls_id)
      {
         hypredrv->current_system_index = stats_ls_id + 1;
      }
   }

   int next_ls_id     = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int should_rebuild = hypredrv_PreconReuseDecide(hypredrv, PRECON_REUSE_SETUP);
   if (hypredrv_ErrorCodeActive())
   {
      RestoreScaledSystemState(hypredrv, 0);
      return hypredrv_ErrorCodeGet();
   }
   int rerun_mgr_component_setup =
      (hypredrv->precon != NULL) && hypredrv->iargs->precon_method == PRECON_MGR &&
      hypredrv_MGRComponentReuseSetupMode(&hypredrv->iargs->precon.mgr, hypredrv->stats,
                                          next_ls_id);
   int skip_precon_setup = (hypredrv->precon != NULL) && hypredrv->precon_is_setup &&
                           !should_rebuild && !rerun_mgr_component_setup;
   HYPREDRV_LOG_OBJECTF(2, hypredrv,
                        "solver setup decisions: rebuild_precon=%d rerun_mgr_setup=%d "
                        "skip_precon_setup=%d",
                        should_rebuild, rerun_mgr_component_setup, skip_precon_setup);

   char diagnostic_object_name[32];
   diagnostic_object_name[0]     = '\0';
   const char *setup_object_name = hypredrv_ResolveLogObjectName(
      hypredrv, diagnostic_object_name, sizeof(diagnostic_object_name));
   hypredrv_LinearSystemLogBlockFrobenius(
      hypredrv->comm, hypredrv->mat_A.handle, hypredrv->dofmap,
      hypredrv->iargs->ls.dof_labels, setup_object_name, next_ls_id);

   /* Compute and apply the scaling before setup. */
   if (hypredrv->iargs->scaling.enabled)
   {
      if (!hypredrv_ScalingPrepare(hypredrv->comm, &hypredrv->iargs->scaling,
                                   &hypredrv->scaling_ctx, hypredrv->mat_A.handle,
                                   hypredrv->vec_b.handle, hypredrv->dofmap))
         return hypredrv_ErrorCodeGet();

      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A.handle,
                                    hypredrv->mat_M.handle, hypredrv->vec_b.handle,
                                    hypredrv->vec_x.handle);  /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) /* GCOVR_EXCL_BR_LINE */
      {
         RestoreScaledSystemState(hypredrv, 0);
         return hypredrv_ErrorCodeGet();
      }
   }

   if (rerun_mgr_component_setup)
   {
      hypredrv_MGRRefreshComponentsForSetup(
         &hypredrv->iargs->precon.mgr, hypredrv->precon->main,
         hypredrv->precon_reuse_timesteps.starts, hypredrv->stats, next_ls_id);
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
      {
         if (hypredrv->precon)
         {
            hypredrv->precon_is_setup = false;
         }
         RestoreScaledSystemState(hypredrv, 0);
         return hypredrv_ErrorCodeGet();
      }
   }

   hypredrv_SolverSetupWithReuse(hypredrv->solver, hypredrv->precon,
                                 hypredrv->mat_M.handle, hypredrv->vec_b.handle,
                                 hypredrv->vec_x.handle, hypredrv->vec_xref.handle,
                                 hypredrv->stats, skip_precon_setup);

   hypredrv_HypreConsumeErrors(hypredrv->comm);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
   {
      if (hypredrv->precon && !skip_precon_setup)
      {
         hypredrv->precon_is_setup = false;
      }
      RestoreScaledSystemState(hypredrv, 0);
      HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup failed");
      return hypredrv_ErrorCodeGet();
   }
   if (hypredrv->precon && !skip_precon_setup)
   {
      hypredrv->precon_is_setup = true;
   }
   hypredrv_ObjectDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_SETUP);
   RestoreScaledSystemState(hypredrv, 0);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverSetup end");

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectPreconDestroy(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconDestroy begin");
   RequireSolverArguments(hypredrv);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return hypredrv_ErrorCodeGet();

   bool should_destroy = hypredrv_PreconReuseDecide(hypredrv, PRECON_REUSE_DESTROY);
   if (hypredrv_ErrorCodeActive()) return hypredrv_ErrorCodeGet();
   if (should_destroy)
   {
      if (hypredrv->precon)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv, "destroying preconditioner object");
         hypredrv_PreconDestroy(&hypredrv->precon, hypredrv->stats,
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

uint32_t
hypredrv_ObjectLinearSolverDestroy(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverDestroy begin");

   HYPREDRV_LOG_OBJECTF(2, hypredrv,
                        "linear solver destroy: evaluating preconditioner teardown");
   if (hypredrv_ObjectPreconDestroy(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "linear solver destroy failed: preconditioner teardown error");
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_SolverDestroy(&hypredrv->solver);
   hypredrv_VectorTagsClear(hypredrv);
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "linear solver destroy: solver object released");

   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverDestroy end");
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectLinearSolverApply(hypredrv_t *hypredrv)
{
   HYPRE_IJVector vec_r           = NULL;
   int            solve_succeeded = 1;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply begin");
   if (PrepareSolverExecution(hypredrv, PREPARE_SOLVER_APPLY, hypredrv->vec_b.handle,
                              hypredrv->vec_x.handle))
   {
      RestoreScaledSystemState(hypredrv, 0);
      return hypredrv_ErrorCodeGet();
   }

   /* Apply scaling if enabled but not yet applied (e.g., when preconditioner is reused)
    */    /* GCOVR_EXCL_BR_LINE */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       !hypredrv_ScalingIsApplied(hypredrv->scaling_ctx)) /* GCOVR_EXCL_BR_LINE */
   {                                                      /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "applying deferred system scaling"); /* GCOVR_EXCL_BR_LINE */
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A.handle,
                                    hypredrv->mat_M.handle, hypredrv->vec_b.handle,
                                    hypredrv->vec_x.handle);  /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) /* GCOVR_EXCL_BR_LINE */
      {
         RestoreScaledSystemState(hypredrv, 0);
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Perform solve */ /* GCOVR_EXCL_BR_LINE */
   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled &&
       hypredrv_ScalingIsApplied(hypredrv->scaling_ctx)) /* GCOVR_EXCL_BR_LINE */
   {
      double b_norm = 0.0, r_norm = 0.0, r0_norm = 0.0;

      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving scaled system"); /* GCOVR_EXCL_BR_LINE */
      int xref_scaled = 0;

      /* Compute initial residual norm before solve (on current system state) */
      hypredrv_LinearSystemComputeResidualNormWork(
         hypredrv->mat_A.handle, hypredrv->vec_b.handle, hypredrv->vec_x.handle, "L2",
         &vec_r, &r0_norm);
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
      {
         RestoreScaledSystemState(hypredrv, 0);
         goto solve_cleanup;
      }

      if (hypredrv->vec_xref.handle)
      {
         hypredrv_ScalingApplyToVector(hypredrv->scaling_ctx, hypredrv->vec_xref.handle,
                                       SCALING_VECTOR_UNKNOWN); /* GCOVR_EXCL_BR_LINE */
         xref_scaled = !hypredrv_ErrorCodeActive();
         if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) /* GCOVR_EXCL_BR_LINE */
         {
            RestoreScaledSystemState(hypredrv, xref_scaled);
            goto solve_cleanup;
         }
         xref_scaled = 1;
      }

      SetPendingSolvePathContext(hypredrv);
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
      hypredrv_StatsInitialResNormSet(hypredrv->stats, r0_norm);

      /* Solve on scaled system */
      HYPRE_Int iters = hypredrv_SolverSolveOnly(
         hypredrv->solver, hypredrv->mat_A.handle, hypredrv->vec_b.handle,
         hypredrv->vec_x.handle, hypredrv->vec_xref.handle);
      if (iters < 0 && !hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
         hypredrv_ErrorMsgAdd("Scaled solver returned a negative iteration count");
      }
      if (hypredrv_DistributedErrorStateSync(
             hypredrv->comm)) /* GCOVR_EXCL_BR_LINE */ /* solver failure injection */
      {
         solve_succeeded = 0; /* GCOVR_EXCL_BR_LINE */
         hypredrv_StatsIterSet(hypredrv->stats, 0);
         hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END,
                                "solve"); /* GCOVR_EXCL_BR_LINE */
         RestoreScaledSystemState(hypredrv, xref_scaled);
         hypredrv_PreconReuseRecordSolve(hypredrv, solve_succeeded);
         goto solve_cleanup;
      }

      hypredrv_StatsIterSet(hypredrv->stats, (int)iters);
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");

      /* Undo scaling on the solution/reference and restore the caller's system. */
      if (RestoreScaledSystemState(hypredrv, xref_scaled)) /* GCOVR_EXCL_BR_LINE */
      {
         goto solve_cleanup;
      }

      /* Compute residual norms on original (now restored) system */
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_b.handle, "L2", &b_norm);
      hypredrv_LinearSystemComputeResidualNormWork(
         hypredrv->mat_A.handle, hypredrv->vec_b.handle, hypredrv->vec_x.handle, "L2",
         &vec_r, &r_norm);                    /* GCOVR_EXCL_BR_LINE */
      b_norm = (b_norm > 0.0) ? b_norm : 1.0; /* GCOVR_EXCL_BR_LINE */
      hypredrv_StatsRelativeResNormSet(hypredrv->stats, r_norm / b_norm);

      HYPRE_IJVectorDestroy(vec_r);
      vec_r = NULL;
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv, "solving unscaled system");
      SetPendingSolvePathContext(hypredrv);
      /* SolverApply records the residual norms and statistics for this path. */
      uint32_t error_before_solve = hypredrv_ErrorCodeGet();
      hypredrv_SolverApply(hypredrv->solver, hypredrv->mat_A.handle,
                           hypredrv->vec_b.handle, hypredrv->vec_x.handle,
                           hypredrv->vec_xref.handle, hypredrv->stats);
      solve_succeeded = (hypredrv_ErrorCodeGet() == error_before_solve);
   }

   hypredrv_HypreConsumeErrors(hypredrv->comm);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) /* GCOVR_EXCL_BR_LINE */
   {
      solve_succeeded = 0;
      hypredrv_PreconReuseRecordSolve(hypredrv, solve_succeeded);
      return hypredrv_ErrorCodeGet();
   }

   /* Fix the solution gauge when exact null space modes are set */
   if (hypredrv->vec_ns && hypredrv->num_ns > 0)
   {
      hypredrv_LinearSystemProjectOutNullSpace(hypredrv->vec_ns, hypredrv->num_ns,
                                               hypredrv->vec_x.handle);
   }

   char        residual_object_name[32] = {0};
   const char *solve_object_name        = hypredrv_ResolveLogObjectName(
      hypredrv, residual_object_name, sizeof(residual_object_name));
   hypredrv_LinearSystemLogBlockResidualNorms(
      hypredrv->comm, hypredrv->mat_A.handle, hypredrv->vec_b.handle,
      hypredrv->vec_x.handle, hypredrv->dofmap, hypredrv->iargs->ls.dof_labels,
      solve_object_name, hypredrv_StatsGetLinearSystemID(hypredrv->stats));

   if (hypredrv->vec_xref.handle)
   {
      double e_norm = 0.0, x_norm = 0.0, xref_norm = 0.0;
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_xref.handle, "L2", &xref_norm);
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x.handle, "L2", &x_norm);
      hypredrv_LinearSystemComputeErrorNorm(hypredrv->vec_xref.handle,
                                            hypredrv->vec_x.handle, "L2", &e_norm);
      if (!hypredrv->mypid) /* GCOVR_EXCL_BR_LINE */
      {
         printf("L2 norm of error: %e\n", (double)e_norm);
         printf("L2 norm of solution: %e\n", (double)x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
      }
   }

   HYPREDRV_LOG_OBJECTF(2, hypredrv, "solve finished (iters=%d)",
                        hypredrv_StatsGetLastIter(hypredrv->stats));
   hypredrv_PreconReuseRecordSolve(hypredrv, solve_succeeded);
   hypredrv_ObjectDumpLinearSystem(hypredrv, PRINT_SYSTEM_STAGE_APPLY);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_LinearSolverApply end");

solve_cleanup:
   if (vec_r)
   {
      HYPRE_IJVectorDestroy(vec_r);
   }
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_ObjectPreconApply(hypredrv_t *hypredrv, HYPRE_IJVector vec_b,
                           HYPRE_IJVector vec_x)
{
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply begin");
   if (PrepareSolverExecution(hypredrv, PREPARE_PRECON_APPLY, vec_b, vec_x))
   {
      RestoreScaledSystemState(hypredrv, 0);
      return hypredrv_ErrorCodeGet();
   }
   HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner apply: method=%d has_precon=%d",
                        (int)hypredrv->iargs->precon_method,
                        (hypredrv->precon != NULL)); /* GCOVR_EXCL_BR_LINE */

   if (hypredrv->scaling_ctx && hypredrv->iargs->scaling.enabled)
   {
      hypredrv_ScalingApplyToSystem(hypredrv->scaling_ctx, hypredrv->mat_A.handle,
                                    hypredrv->mat_M.handle, vec_b, vec_x);
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
      {
         RestoreScaledSystemState(hypredrv, 0);
         return hypredrv_ErrorCodeGet();
      }
   }
   hypredrv_PreconApply(hypredrv->precon, hypredrv->mat_A.handle, vec_b, vec_x);
   hypredrv_HypreConsumeErrors(hypredrv->comm); /* GCOVR_EXCL_BR_LINE */
   if (RestoreScaledSystemState(hypredrv, 0)) return hypredrv_ErrorCodeGet();
   /* Fix the solution gauge when exact null space modes are set */
   if (hypredrv->vec_ns && hypredrv->num_ns > 0)
   {
      hypredrv_LinearSystemProjectOutNullSpace(hypredrv->vec_ns, hypredrv->num_ns, vec_x);
   }
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_PreconApply end"); /* GCOVR_EXCL_BR_LINE */

   return hypredrv_ErrorCodeGet();
}
