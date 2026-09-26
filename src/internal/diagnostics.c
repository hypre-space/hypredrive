/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "diagnostics.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "internal/linsys.h"
#include "internal/utils.h"
#include "logging.h"
#include "object.h"

/*-----------------------------------------------------------------------------
 * Record the timestep context of the upcoming solve in the stats object
 *-----------------------------------------------------------------------------*/

void
hypredrv_SetPendingSolvePathContext(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv->stats) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   hypredrv_StatsSetPendingTimestepContext(hypredrv->stats, -1);

   int next_ls_id   = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int timestep_idx = hypredrv_PreconReuseResolveTimestepIndex(
      hypredrv->precon_reuse_timesteps.starts, hypredrv->stats, next_ls_id);
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

/*-----------------------------------------------------------------------------
 * Print statistics to general.statistics_filename, falling back to stdout
 *-----------------------------------------------------------------------------*/

void
hypredrv_PrintStatsWithConfiguredDestination(HYPREDRV_t hypredrv, int print_level)
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

/*-----------------------------------------------------------------------------
 * Gather solve-state metadata used to decide and label print-system dumps
 *-----------------------------------------------------------------------------*/

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

   ctx->timestep_index = hypredrv_PreconReuseResolveTimestepIndex(
      hypredrv->precon_reuse_timesteps.starts, hypredrv->stats, ctx->system_index);
}

/*-----------------------------------------------------------------------------
 * Dump the linear system at the given stage when the print config requests it
 *-----------------------------------------------------------------------------*/

void
hypredrv_MaybeDumpLinearSystem(HYPREDRV_t hypredrv, int stage)
{
   if (!hypredrv || !hypredrv->iargs) /* GCOVR_EXCL_BR_LINE */
   {
      return;
   }

   PrintSystemContext ctx;
   BuildPrintSystemContext(hypredrv, stage, &ctx);

   char object_name_buffer[32];
   object_name_buffer[0]   = '\0';
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
   hypredrv_LinearSystemDumpScheduled(
      hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A, hypredrv->mat_M,
      hypredrv->vec_b, hypredrv->vec_x0, hypredrv->vec_xref, hypredrv->vec_x,
      hypredrv->dofmap, &ctx, object_name);
}

/*-----------------------------------------------------------------------------
 * Log counts of cached MGR component solvers (experimental diagnostics)
 *-----------------------------------------------------------------------------*/

#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
void
hypredrv_LogMGRCachedHandles(HYPREDRV_t hypredrv, const MGR_args *mgr, const char *msg)
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

/* Post-solve diagnostics: per-block residual norms, and the error/solution
 * norms against a reference solution when one was supplied. */
void
hypredrv_ReportSolveDiagnostics(HYPREDRV_t hypredrv, double *x_norm, double *xref_norm,
                                double *e_norm)
{
   char residual_object_name[32];
   residual_object_name[0]       = '\0';
   const char *solve_object_name = hypredrv_ResolveLogObjectName(
      hypredrv, residual_object_name, sizeof(residual_object_name));
   hypredrv_LinearSystemLogBlockResidualNorms(
      hypredrv->comm, hypredrv->mat_A, hypredrv->vec_b, hypredrv->vec_x, hypredrv->dofmap,
      hypredrv->iargs->ls.dof_labels, solve_object_name,
      hypredrv_StatsGetLinearSystemID(hypredrv->stats));

   if (hypredrv->vec_xref)
   {
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_xref, "L2", xref_norm);
      hypredrv_LinearSystemComputeVectorNorm(hypredrv->vec_x, "L2", x_norm);
      hypredrv_LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, "L2",
                                            e_norm); /* GCOVR_EXCL_BR_LINE */
      if (!hypredrv->mypid)                          /* GCOVR_EXCL_BR_LINE */
      {
         printf("L2 norm of error: %e\n", (double)*e_norm);
         printf("L2 norm of solution: %e\n", (double)*x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)*xref_norm);
      }
   }
}
