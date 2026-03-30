/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/precon_reuse.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include "internal/precon.h"
#include "logging.h"
#include "object.h"

typedef struct PreconReuseSample_struct
{
   int    num_solves;
   int    iterations;
   double setup_time;
   double solve_time;
} PreconReuseSample;

enum
{
   /* Number of solves used to establish the initial baseline before adaptive
    * scoring begins.  Three observations give a stable mean while keeping the
    * warm-up cost low for typical time-stepping workloads. */
   PRECON_REUSE_BOOTSTRAP_SOLVES = 3,
};

static void
PreconReuseMeanSetDefaults(PreconReuseMean_args *mean)
{
   if (!mean)
   {
      return;
   }

   mean->kind  = PRECON_REUSE_MEAN_ARITHMETIC;
   mean->power = 1.0;
}

static void
PreconReuseTransformSetDefaults(PreconReuseTransform_args *transform)
{
   if (!transform)
   {
      return;
   }

   transform->kind                = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   transform->baseline            = PRECON_REUSE_BASELINE_REBUILD;
   transform->amortization_window = 10;
}

static void
PreconReuseHistorySetDefaults(PreconReuseHistory_args *history)
{
   if (!history)
   {
      return;
   }

   history->source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   history->level      = -1;
   history->max_points = 5;
   history->reduction  = PRECON_REUSE_REDUCTION_NONE;
}

static void
PreconReuseScoreComponentSetDefaults(PreconReuseScoreComponent_args *component)
{
   if (!component)
   {
      return;
   }

   memset(component, 0, sizeof(*component));
   snprintf(component->name, sizeof(component->name), "%s", "component");
   component->enabled   = 1;
   component->metric    = PRECON_REUSE_METRIC_ITERATIONS;
   component->weight    = 1.0;
   component->direction = PRECON_REUSE_DIRECTION_ABOVE;
   component->target    = 1.25;
   component->scale     = 0.25;
   PreconReuseMeanSetDefaults(&component->mean);
   PreconReuseTransformSetDefaults(&component->transform);
   PreconReuseHistorySetDefaults(&component->history);
}

void
hypredrv_PreconReuseSetDefaultArgs(PreconReuse_args *args)
{
   if (!args)
   {
      return;
   }

   memset(args, 0, sizeof(*args));
   args->enabled                          = 0;
   args->frequency                        = 0;
   args->linear_system_ids                = NULL;
   args->per_timestep                     = 0;
   args->policy                           = PRECON_REUSE_POLICY_STATIC;
   args->guards.min_reuse_solves          = 0;
   args->guards.max_reuse_solves          = -1;
   args->guards.min_history_points        = 1;
   args->guards.bad_decisions_to_rebuild  = 1;
   args->guards.max_iteration_ratio       = -1.0;
   args->guards.max_solve_time_ratio      = -1.0;
   args->guards.rebuild_on_new_timestep   = 0;
   args->guards.rebuild_on_solver_failure = 1;
   args->guards.rebuild_on_new_level      = NULL;
   args->adaptive.rebuild_threshold       = 1.0;
   args->adaptive.positive_floor          = 1.0e-12;
   args->adaptive.components              = NULL;
   args->adaptive.num_components          = 0;
}

void
hypredrv_PreconReuseDestroyArgs(PreconReuse_args *args)
{
   if (!args)
   {
      return;
   }

   if (args->linear_system_ids)
   {
      hypredrv_IntArrayDestroy(&args->linear_system_ids);
   }

   if (args->guards.rebuild_on_new_level)
   {
      hypredrv_IntArrayDestroy(&args->guards.rebuild_on_new_level);
   }

   if (args->adaptive.components)
   {
      free(args->adaptive.components);
      args->adaptive.components = NULL;
   }

   args->adaptive.num_components = 0;
   args->frequency               = 0;
   args->per_timestep            = 0;
   args->enabled                 = 0;
}

static int
PreconReuseParseOnOff(const char *value, int *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "on") || !strcasecmp(value, "yes") ||
       !strcasecmp(value, "true") || !strcmp(value, "1"))
   {
      *out = 1;
      return 1;
   }
   if (!strcasecmp(value, "off") || !strcasecmp(value, "no") ||
       !strcasecmp(value, "false") || !strcmp(value, "0"))
   {
      *out = 0;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseInt(const char *value, int *out)
{
   if (!value || !out)
   {
      return 0;
   }

   return sscanf(value, "%d", out) == 1;
}

static int
PreconReuseParseDouble(const char *value, double *out)
{
   if (!value || !out)
   {
      return 0;
   }

   return sscanf(value, "%lf", out) == 1;
}

static int
PreconReuseParsePolicy(const char *value, PreconReusePolicy *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "static"))
   {
      *out = PRECON_REUSE_POLICY_STATIC;
      return 1;
   }
   if (!strcasecmp(value, "adaptive"))
   {
      *out = PRECON_REUSE_POLICY_ADAPTIVE;
      return 1;
   }

   return 0;
}

static int
PreconReuseInstallDefaultAdaptiveComponents(PreconReuseAdaptive_args *adaptive)
{
   if (!adaptive)
   {
      return 0;
   }

   PreconReuseScoreComponent_args *components =
      (PreconReuseScoreComponent_args *)calloc(2, sizeof(*components));
   if (!components)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate default adaptive reuse components");
      return 0;
   }

   /* Default adaptive model:
    *   1. "efficiency" watches whether reused solves are repaying setup cost.
    *   2. "stability" watches for growing iteration counts under reuse.
    *
    * Keep these defaults centralized so YAML shorthand like "reuse: adaptive"
    * remains predictable and future formula updates have one implementation point.
    */
   PreconReuseScoreComponentSetDefaults(&components[0]);
   snprintf(components[0].name, sizeof(components[0].name), "%s", "efficiency");
   components[0].metric         = PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP;
   components[0].target         = 1.0;
   components[0].scale          = 1.0;
   components[0].mean.kind      = PRECON_REUSE_MEAN_ARITHMETIC;
   components[0].transform.kind = PRECON_REUSE_TRANSFORM_RAW;
   components[0].transform.amortization_window = 10;
   components[0].history.source                = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   components[0].history.max_points            = 5;

   PreconReuseScoreComponentSetDefaults(&components[1]);
   snprintf(components[1].name, sizeof(components[1].name), "%s", "stability");
   components[1].metric             = PRECON_REUSE_METRIC_ITERATIONS;
   components[1].target             = 1.5;
   components[1].scale              = 0.5;
   components[1].mean.kind          = PRECON_REUSE_MEAN_RMS;
   components[1].transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   components[1].transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   components[1].history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   components[1].history.max_points = 5;

   free(adaptive->components);
   adaptive->components     = components;
   adaptive->num_components = 2;

   return 1;
}

static void
PreconReuseApplyAdaptiveShorthandDefaults(PreconReuse_args *args)
{
   if (!args)
   {
      return;
   }

   args->guards.min_history_points       = PRECON_REUSE_BOOTSTRAP_SOLVES;
   args->guards.bad_decisions_to_rebuild = 2;
}

static int
PreconReuseParseDirection(const char *value, PreconReuseDirection *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "above"))
   {
      *out = PRECON_REUSE_DIRECTION_ABOVE;
      return 1;
   }
   if (!strcasecmp(value, "below"))
   {
      *out = PRECON_REUSE_DIRECTION_BELOW;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseMeanKind(const char *value, PreconReuseMeanKind *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "arithmetic"))
   {
      *out = PRECON_REUSE_MEAN_ARITHMETIC;
      return 1;
   }
   if (!strcasecmp(value, "geometric"))
   {
      *out = PRECON_REUSE_MEAN_GEOMETRIC;
      return 1;
   }
   if (!strcasecmp(value, "harmonic"))
   {
      *out = PRECON_REUSE_MEAN_HARMONIC;
      return 1;
   }
   if (!strcasecmp(value, "rms"))
   {
      *out = PRECON_REUSE_MEAN_RMS;
      return 1;
   }
   if (!strcasecmp(value, "min"))
   {
      *out = PRECON_REUSE_MEAN_MIN;
      return 1;
   }
   if (!strcasecmp(value, "max"))
   {
      *out = PRECON_REUSE_MEAN_MAX;
      return 1;
   }
   if (!strcasecmp(value, "power"))
   {
      *out = PRECON_REUSE_MEAN_POWER;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseTransformKind(const char *value, PreconReuseTransformKind *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "raw"))
   {
      *out = PRECON_REUSE_TRANSFORM_RAW;
      return 1;
   }
   if (!strcasecmp(value, "delta_from_baseline"))
   {
      *out = PRECON_REUSE_TRANSFORM_DELTA_FROM_BASELINE;
      return 1;
   }
   if (!strcasecmp(value, "ratio_to_baseline"))
   {
      *out = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      return 1;
   }
   if (!strcasecmp(value, "relative_increase"))
   {
      *out = PRECON_REUSE_TRANSFORM_RELATIVE_INCREASE;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseBaselineKind(const char *value, PreconReuseBaselineKind *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "rebuild"))
   {
      *out = PRECON_REUSE_BASELINE_REBUILD;
      return 1;
   }
   if (!strcasecmp(value, "window_mean"))
   {
      *out = PRECON_REUSE_BASELINE_WINDOW_MEAN;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseMetric(const char *value, PreconReuseMetric *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "iterations"))
   {
      *out = PRECON_REUSE_METRIC_ITERATIONS;
      return 1;
   }
   if (!strcasecmp(value, "solve_time"))
   {
      *out = PRECON_REUSE_METRIC_SOLVE_TIME;
      return 1;
   }
   if (!strcasecmp(value, "setup_time"))
   {
      *out = PRECON_REUSE_METRIC_SETUP_TIME;
      return 1;
   }
   if (!strcasecmp(value, "total_time"))
   {
      *out = PRECON_REUSE_METRIC_TOTAL_TIME;
      return 1;
   }
   if (!strcasecmp(value, "solve_overhead_vs_setup"))
   {
      *out = PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseHistorySource(const char *value, PreconReuseHistorySource *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "linear_solves"))
   {
      *out = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      return 1;
   }
   if (!strcasecmp(value, "active_level"))
   {
      *out = PRECON_REUSE_HISTORY_ACTIVE_LEVEL;
      return 1;
   }
   if (!strcasecmp(value, "completed_level"))
   {
      *out = PRECON_REUSE_HISTORY_COMPLETED_LEVEL;
      return 1;
   }

   return 0;
}

static int
PreconReuseParseReduction(const char *value, PreconReuseReduction *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "none"))
   {
      *out = PRECON_REUSE_REDUCTION_NONE;
      return 1;
   }
   if (!strcasecmp(value, "mean"))
   {
      *out = PRECON_REUSE_REDUCTION_MEAN;
      return 1;
   }
   if (!strcasecmp(value, "sum"))
   {
      *out = PRECON_REUSE_REDUCTION_SUM;
      return 1;
   }

   return 0;
}

static int
PreconReuseIntArrayContains(const IntArray *arr, int value)
{
   if (!arr || !arr->data)
   {
      return 0;
   }

   for (size_t i = 0; i < arr->size; i++)
   {
      if (arr->data[i] == value)
      {
         return 1;
      }
   }

   return 0;
}

static int
PreconReuseFindTimestepIndex(const IntArray *starts, int ls_id)
{
   if (!starts || !starts->data)
   {
      return -1;
   }

   int found_idx = -1;
   for (size_t i = 0; i < starts->size; i++)
   {
      if (starts->data[i] > ls_id)
      {
         break;
      }

      found_idx = (int)i;
   }

   return found_idx;
}

static int
PreconReuseGetEmbeddedTimestepStart(const Stats *stats, int ls_id)
{
   if (!stats || !(stats->level_active & (1 << 0)))
   {
      return -1;
   }

   int timestep_start = stats->level_solve_start[0];
   if (timestep_start < 0 || timestep_start > ls_id)
   {
      return -1;
   }

   return timestep_start;
}

static int
PreconReuseGetEmbeddedTimestepIndex(const Stats *stats)
{
   if (!stats || !(stats->level_active & (1 << 0)))
   {
      return -1;
   }

   if (stats->level_current_id[0] <= 0)
   {
      return -1;
   }

   return stats->level_current_id[0] - 1;
}

static int
PreconReuseResolveTimestepContext(const IntArray *starts, const Stats *stats, int ls_id,
                                  int *timestep_idx, int *timestep_start)
{
   if (!timestep_idx || !timestep_start)
   {
      return 0;
   }

   *timestep_idx   = -1;
   *timestep_start = -1;

   int const starts_idx = PreconReuseFindTimestepIndex(starts, ls_id);
   if (starts_idx >= 0)
   {
      *timestep_idx   = starts_idx;
      *timestep_start = starts->data[starts_idx];
   }
   else
   {
      *timestep_idx   = PreconReuseGetEmbeddedTimestepIndex(stats);
      *timestep_start = PreconReuseGetEmbeddedTimestepStart(stats, ls_id);
   }

   if (*timestep_start < 0 || ls_id < *timestep_start)
   {
      return 0;
   }

   return 1;
}

void
hypredrv_PreconReuseBuildObservation(HYPREDRV_t hypredrv, const IntArray *timestep_starts,
                                     PreconReuseObservation *obs)
{
   if (!obs)
   {
      return;
   }

   memset(obs, 0, sizeof(*obs));
   obs->system_index    = -1;
   obs->timestep_index  = -1;
   obs->iters           = -1;
   obs->solve_succeeded = 1;
   obs->setup_time      = -1.0;
   obs->solve_time      = -1.0;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      obs->level_ids[level] = -1;
   }

   if (!hypredrv || !hypredrv->stats)
   {
      return;
   }

   obs->system_index = hypredrv->current_system_index >= 0
                          ? hypredrv->current_system_index
                          : hypredrv_StatsGetLinearSystemID(hypredrv->stats);

   int timestep_start = -1;
   PreconReuseResolveTimestepContext(timestep_starts, hypredrv->stats, obs->system_index,
                                     &obs->timestep_index, &timestep_start);

   obs->iters      = hypredrv_StatsGetLastIter(hypredrv->stats);
   obs->setup_time = hypredrv_StatsGetLastSetupTime(hypredrv->stats);
   obs->solve_time = hypredrv_StatsGetLastSolveTime(hypredrv->stats);

   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      if (hypredrv->stats->level_current_id[level] > 0)
      {
         obs->level_ids[level] = hypredrv->stats->level_current_id[level] - 1;
      }
   }
}

static void
PreconReuseResolveBootstrapBaseline(PreconReuseState *state)
{
   if (!state || state->count < PRECON_REUSE_BOOTSTRAP_SOLVES)
   {
      return;
   }

   state->baseline = state->observations[0];

   double iters_sum = 0.0;
   double setup_sum = 0.0;
   double solve_sum = 0.0;
   for (int i = 0; i < PRECON_REUSE_BOOTSTRAP_SOLVES; i++)
   {
      iters_sum += (double)state->observations[i].iters;
      setup_sum += state->observations[i].setup_time;
      solve_sum += state->observations[i].solve_time;
   }

   state->baseline_iters      = iters_sum / (double)PRECON_REUSE_BOOTSTRAP_SOLVES;
   state->baseline_setup_time = setup_sum / (double)PRECON_REUSE_BOOTSTRAP_SOLVES;
   state->baseline_solve_time = solve_sum / (double)PRECON_REUSE_BOOTSTRAP_SOLVES;
   state->baseline.iters      = (int)lrint(state->baseline_iters);
   state->baseline.setup_time = state->baseline_setup_time;
   state->baseline.solve_time = state->baseline_solve_time;
   state->baseline_valid      = 1;
}

static int
PreconReuseReuseAgeGet(const PreconReuseState *state)
{
   if (!state)
   {
      return 0;
   }

   return (int)state->count;
}

/* Track consecutive "bad" decisions (score >= threshold but below the streak limit).
 * A rebuild is deferred until the streak reaches bad_decisions_to_rebuild, which
 * filters out transient spikes without rebuilding on every bad observation.
 * The last_decision_ls_id guard prevents double-counting when multiple call sites
 * evaluate the same linear-system ID. */
static int
PreconReuseUpdateBadDecisionStreak(PreconReuseState *state, int next_ls_id, int is_bad)
{
   if (!state)
   {
      return is_bad ? 1 : 0;
   }

   if (state->last_decision_ls_id == next_ls_id)
   {
      return state->bad_decision_streak;
   }

   state->last_decision_ls_id = next_ls_id;
   state->bad_decision_streak = is_bad ? (state->bad_decision_streak + 1) : 0;
   return state->bad_decision_streak;
}

void
hypredrv_PreconReuseMarkRebuild(HYPREDRV_t hypredrv, PreconReuseState *state)
{
   if (!hypredrv || !state)
   {
      return;
   }

   hypredrv_PreconReuseStateReset(state);
   if (!hypredrv->stats)
   {
      return;
   }

   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      state->last_rebuild_level_ids[level] =
         (hypredrv->stats->level_current_id[level] > 0)
            ? (hypredrv->stats->level_current_id[level] - 1)
            : -1;
   }
}

void
hypredrv_PreconReuseLogDecision(HYPREDRV_t hypredrv, int next_ls_id,
                                const PreconReuseDecision *decision, const char *caller)
{
   if (!hypredrv || !decision)
   {
      return;
   }

   if (decision->used_adaptive)
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "%s adaptive reuse decision: next_ls_id=%d rebuild=%d score=%.6g age=%d "
         "history_points=%d %s",
         caller ? caller : "reuse", next_ls_id, decision->should_rebuild, decision->score,
         decision->age, decision->history_points, decision->summary);
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv, "%s preconditioner reuse decision: next_ls_id=%d rebuild=%d %s",
         caller ? caller : "reuse", next_ls_id, decision->should_rebuild,
         decision->summary);
   }
}

static void
hypredrv_PreconReuseDecisionInit(PreconReuseDecision *decision)
{
   if (!decision)
   {
      return;
   }

   memset(decision, 0, sizeof(*decision));
   snprintf(decision->summary, sizeof(decision->summary), "%s", "static");
}

void
hypredrv_PreconReuseStateInit(PreconReuseState *state)
{
   if (!state)
   {
      return;
   }

   memset(state, 0, sizeof(*state));
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      state->last_rebuild_level_ids[i] = -1;
   }
   state->last_solve_succeeded = 1;
   state->last_decision_ls_id  = -1;
}

void
hypredrv_PreconReuseStateDestroy(PreconReuseState *state)
{
   if (!state)
   {
      return;
   }

   free(state->observations);
   hypredrv_PreconReuseStateInit(state);
}

void
hypredrv_PreconReuseStateReset(PreconReuseState *state)
{
   if (!state)
   {
      return;
   }

   free(state->observations);
   state->observations         = NULL;
   state->count                = 0;
   state->capacity             = 0;
   state->bootstrap_count      = 0;
   state->baseline_valid       = 0;
   state->bad_decision_streak  = 0;
   state->last_solve_succeeded = 1;
   state->last_decision_ls_id  = -1;
   state->baseline_iters       = 0.0;
   state->baseline_setup_time  = 0.0;
   state->baseline_solve_time  = 0.0;
   memset(&state->baseline, 0, sizeof(state->baseline));
   memset(&state->last_observation, 0, sizeof(state->last_observation));
   state->baseline.system_index            = -1;
   state->baseline.timestep_index          = -1;
   state->baseline.iters                   = -1;
   state->baseline.solve_succeeded         = 1;
   state->baseline.setup_time              = -1.0;
   state->baseline.solve_time              = -1.0;
   state->last_observation.system_index    = -1;
   state->last_observation.timestep_index  = -1;
   state->last_observation.iters           = -1;
   state->last_observation.solve_succeeded = 1;
   state->last_observation.setup_time      = -1.0;
   state->last_observation.solve_time      = -1.0;
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      state->baseline.level_ids[i]         = -1;
      state->last_observation.level_ids[i] = -1;
   }
}

void
hypredrv_PreconReuseStateRecordObservation(PreconReuseState             *state,
                                           const PreconReuseObservation *obs)
{
   if (!state || !obs)
   {
      return;
   }

   if (state->count == state->capacity)
   {
      size_t                  new_capacity = state->capacity ? state->capacity * 2u : 16u;
      PreconReuseObservation *new_observations = (PreconReuseObservation *)realloc(
         state->observations, new_capacity * sizeof(*new_observations));
      if (!new_observations)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to grow adaptive preconditioner history");
         return;
      }
      state->observations = new_observations;
      state->capacity     = new_capacity;
   }

   state->observations[state->count++] = *obs;
   state->last_observation             = *obs;
   state->last_solve_succeeded         = obs->solve_succeeded;

   if (!state->baseline_valid)
   {
      size_t bootstrap_count = state->count;
      if (bootstrap_count > PRECON_REUSE_BOOTSTRAP_SOLVES)
      {
         bootstrap_count = PRECON_REUSE_BOOTSTRAP_SOLVES;
      }
      state->bootstrap_count = (int)bootstrap_count;
      if (state->bootstrap_count >= PRECON_REUSE_BOOTSTRAP_SOLVES)
      {
         PreconReuseResolveBootstrapBaseline(state);
      }
   }
   state->last_decision_ls_id = -1;
}

static int
PreconReuseShouldRebuildStatic(const PreconReuse_args *args,
                               const IntArray *timestep_starts, const Stats *stats,
                               int next_ls_id)
{
   if (!args)
   {
      return 1;
   }

   if (next_ls_id < 0)
   {
      next_ls_id = 0;
   }

   int freq = args->frequency;
   if (!args->enabled || freq < 0)
   {
      freq = 0;
   }

   if (args->enabled && args->linear_system_ids && args->linear_system_ids->size > 0)
   {
      return PreconReuseIntArrayContains(args->linear_system_ids, next_ls_id);
   }

   if (args->enabled && args->per_timestep)
   {
      int timestep_idx   = -1;
      int timestep_start = -1;
      if (!PreconReuseResolveTimestepContext(timestep_starts, stats, next_ls_id,
                                             &timestep_idx, &timestep_start))
      {
         return 0;
      }

      if (next_ls_id != timestep_start)
      {
         return 0;
      }

      int timestep_period = (freq > 0) ? freq : 1;
      if (timestep_idx < 0)
      {
         return 1;
      }

      return (timestep_idx % timestep_period) == 0;
   }

   return (next_ls_id % (freq + 1)) == 0;
}

static double
PreconReuseSampleMetricGet(const PreconReuseSample *sample, PreconReuseMetric metric,
                           PreconReuseReduction reduction)
{
   if (!sample)
   {
      return -1.0;
   }

   double value = -1.0;
   switch (metric)
   {
      case PRECON_REUSE_METRIC_ITERATIONS:
         value = (double)sample->iterations;
         break;
      case PRECON_REUSE_METRIC_SOLVE_TIME:
         value = sample->solve_time;
         break;
      case PRECON_REUSE_METRIC_SETUP_TIME:
         value = sample->setup_time;
         break;
      case PRECON_REUSE_METRIC_TOTAL_TIME:
         value = sample->setup_time + sample->solve_time;
         break;
      case PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP:
         value = sample->solve_time;
         break;
      default:
         break;
   }

   if (reduction == PRECON_REUSE_REDUCTION_MEAN && sample->num_solves > 0 &&
       metric != PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP)
   {
      value /= (double)sample->num_solves;
   }

   return value;
}

static int
PreconReuseCurrentLevelID(const Stats *stats, int level)
{
   if (!stats || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return -1;
   }

   if (!(stats->level_active & (1 << level)) || stats->level_current_id[level] <= 0)
   {
      return -1;
   }

   return stats->level_current_id[level] - 1;
}

static int
PreconReuseCollectSamples(const PreconReuseScoreComponent_args *component,
                          const PreconReuseState *state, const Stats *stats,
                          PreconReuseSample *samples, int max_samples)
{
   if (!component || !state || !samples || max_samples <= 0)
   {
      return 0;
   }

   int count               = 0;
   int rebuild_start_ls_id = (state->baseline_valid && state->baseline.system_index >= 0)
                                ? state->baseline.system_index
                                : 0;

   if (component->history.source == PRECON_REUSE_HISTORY_LINEAR_SOLVES)
   {
      for (size_t i = state->count; i > 0 && count < max_samples; i--)
      {
         const PreconReuseObservation *obs = &state->observations[i - 1];
         samples[count].num_solves         = 1;
         samples[count].iterations         = obs->iters;
         samples[count].setup_time         = obs->setup_time;
         samples[count].solve_time         = obs->solve_time;
         count++;
      }
   }
   else if (component->history.source == PRECON_REUSE_HISTORY_ACTIVE_LEVEL)
   {
      int const active_level_id =
         PreconReuseCurrentLevelID(stats, component->history.level);
      if (active_level_id < 0)
      {
         return 0;
      }

      for (size_t i = state->count; i > 0 && count < max_samples; i--)
      {
         const PreconReuseObservation *obs = &state->observations[i - 1];
         if (obs->level_ids[component->history.level] != active_level_id)
         {
            continue;
         }
         samples[count].num_solves = 1;
         samples[count].iterations = obs->iters;
         samples[count].setup_time = obs->setup_time;
         samples[count].solve_time = obs->solve_time;
         count++;
      }
   }
   else if (component->history.source == PRECON_REUSE_HISTORY_COMPLETED_LEVEL)
   {
      int const level_count =
         hypredrv_StatsLevelGetCount(stats, component->history.level);
      for (int idx = level_count - 1; idx >= 0 && count < max_samples; idx--)
      {
         LevelEntry entry;
         int        entry_id     = 0;
         int        num_solves   = 0;
         int        linear_iters = 0;
         double     setup_time   = 0.0;
         double     solve_time   = 0.0;
         if (hypredrv_StatsLevelGetEntry(stats, component->history.level, idx, &entry) !=
             0)
         {
            /* Skip entries that can't be read rather than aborting collection; the
             * adaptive decision will simply use fewer data points. */
            hypredrv_ErrorCodeResetAll();
            continue;
         }

         /* Completed-level summaries are global. Ignore entries that started before
          * the current rebuild baseline so adaptive decisions only use the active
          * preconditioner lifetime. */
         if (entry.solve_start < rebuild_start_ls_id)
         {
            continue;
         }

         if (hypredrv_StatsLevelGetEntrySummary(stats, component->history.level, idx,
                                                &entry_id, &num_solves, &linear_iters,
                                                &setup_time, &solve_time) != 0)
         {
            /* Same: skip unreadable summary entries rather than failing. */
            hypredrv_ErrorCodeResetAll();
            continue;
         }
         (void)entry_id;
         samples[count].num_solves = num_solves;
         samples[count].iterations = linear_iters;
         samples[count].setup_time = setup_time;
         samples[count].solve_time = solve_time;
         count++;
      }
   }

   for (int i = 0; i < count / 2; i++)
   {
      PreconReuseSample tmp  = samples[i];
      samples[i]             = samples[count - i - 1];
      samples[count - i - 1] = tmp;
   }

   return count;
}

static double
PreconReuseArithmeticMean(const double *values, int count)
{
   double sum = 0.0;
   for (int i = 0; i < count; i++)
   {
      sum += values[i];
   }
   return count > 0 ? sum / (double)count : 0.0;
}

static double
PreconReuseGeneralizedMean(const double *values, int count,
                           const PreconReuseMean_args *mean, double positive_floor)
{
   if (!values || count <= 0 || !mean)
   {
      return -1.0;
   }

   switch (mean->kind)
   {
      case PRECON_REUSE_MEAN_MIN:
      {
         double min_value = values[0];
         for (int i = 1; i < count; i++)
         {
            if (values[i] < min_value)
            {
               min_value = values[i];
            }
         }
         return min_value;
      }
      case PRECON_REUSE_MEAN_MAX:
      {
         double max_value = values[0];
         for (int i = 1; i < count; i++)
         {
            if (values[i] > max_value)
            {
               max_value = values[i];
            }
         }
         return max_value;
      }
      case PRECON_REUSE_MEAN_GEOMETRIC:
      {
         double sum_logs = 0.0;
         for (int i = 0; i < count; i++)
         {
            double value = values[i] > positive_floor ? values[i] : positive_floor;
            sum_logs += log(value);
         }
         return exp(sum_logs / (double)count);
      }
      case PRECON_REUSE_MEAN_HARMONIC:
      {
         double denom = 0.0;
         for (int i = 0; i < count; i++)
         {
            double value = values[i] > positive_floor ? values[i] : positive_floor;
            denom += 1.0 / value;
         }
         return denom > 0.0 ? (double)count / denom : 0.0;
      }
      case PRECON_REUSE_MEAN_RMS:
      {
         double ssq = 0.0;
         for (int i = 0; i < count; i++)
         {
            ssq += values[i] * values[i];
         }
         return sqrt(ssq / (double)count);
      }
      case PRECON_REUSE_MEAN_POWER:
      {
         double p   = mean->power;
         double acc = 0.0;
         if (fabs(p) < 1.0e-12)
         {
            PreconReuseMean_args geometric = *mean;
            geometric.kind                 = PRECON_REUSE_MEAN_GEOMETRIC;
            return PreconReuseGeneralizedMean(values, count, &geometric, positive_floor);
         }
         for (int i = 0; i < count; i++)
         {
            double value = values[i];
            if (p <= 0.0 && value < positive_floor)
            {
               value = positive_floor;
            }
            acc += pow(value, p);
         }
         return pow(acc / (double)count, 1.0 / p);
      }
      case PRECON_REUSE_MEAN_ARITHMETIC:
      default:
         return PreconReuseArithmeticMean(values, count);
   }
}

static double
PreconReuseBaselineValue(const PreconReuseScoreComponent_args *component,
                         const PreconReuseState *state, const PreconReuseSample *samples,
                         int count, double positive_floor)
{
   if (!component || !state)
   {
      return -1.0;
   }

   if (component->metric == PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP)
   {
      return 1.0;
   }

   if (component->transform.baseline == PRECON_REUSE_BASELINE_WINDOW_MEAN && samples &&
       count > 0)
   {
      enum
      {
         WINDOW_MEAN_CAPACITY = 128
      };
      double values[WINDOW_MEAN_CAPACITY];
      int    usable = 0;
      if (count > WINDOW_MEAN_CAPACITY)
      {
         fprintf(
            stderr,
            "[HYPREDRV] warning: adaptive reuse baseline window_mean capped at %d "
            "samples (max_points=%d); only the first %d are used for the baseline.\n",
            WINDOW_MEAN_CAPACITY, count, WINDOW_MEAN_CAPACITY);
      }
      for (int i = 0; i < count && i < WINDOW_MEAN_CAPACITY; i++)
      {
         values[usable++] = PreconReuseSampleMetricGet(samples + i, component->metric,
                                                       component->history.reduction);
      }
      return PreconReuseArithmeticMean(values, usable);
   }

   if (!state->baseline_valid)
   {
      return positive_floor;
   }

   PreconReuseSample baseline_sample;
   baseline_sample.num_solves = 1;
   baseline_sample.iterations = (int)lrint(state->baseline_iters);
   baseline_sample.setup_time = state->baseline_setup_time;
   baseline_sample.solve_time = state->baseline_solve_time;
   return PreconReuseSampleMetricGet(&baseline_sample, component->metric,
                                     PRECON_REUSE_REDUCTION_NONE);
}

static double
PreconReuseTransformSample(const PreconReuseScoreComponent_args *component,
                           const PreconReuseState *state, const PreconReuseSample *sample,
                           double baseline_value, double positive_floor)
{
   if (!component || !state || !sample)
   {
      return -1.0;
   }

   if (component->metric == PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP)
   {
      double baseline_setup = state->baseline_valid ? state->baseline_setup_time : 0.0;
      double baseline_solve = state->baseline_valid ? state->baseline_solve_time : 0.0;
      double budget = baseline_setup / (double)component->transform.amortization_window;
      double sample_solve = PreconReuseSampleMetricGet(
         sample, PRECON_REUSE_METRIC_SOLVE_TIME, component->history.reduction);
      if (budget < positive_floor)
      {
         budget = positive_floor;
      }
      return fmax(sample_solve - baseline_solve, 0.0) / budget;
   }

   double raw_value =
      PreconReuseSampleMetricGet(sample, component->metric, component->history.reduction);
   double baseline = baseline_value;

   if (baseline < positive_floor)
   {
      baseline = positive_floor;
   }

   switch (component->transform.kind)
   {
      case PRECON_REUSE_TRANSFORM_RAW:
         return raw_value;
      case PRECON_REUSE_TRANSFORM_DELTA_FROM_BASELINE:
         return fmax(raw_value - baseline, 0.0);
      case PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE:
         return raw_value / baseline;
      case PRECON_REUSE_TRANSFORM_RELATIVE_INCREASE:
         return fmax(raw_value - baseline, 0.0) / baseline;
      default:
         return raw_value;
   }
}

static void
PreconReuseAppendSummary(char *summary, size_t summary_size, const char *text)
{
   if (!summary || summary_size == 0 || !text)
   {
      return;
   }

   size_t used = strlen(summary);
   if (used >= summary_size - 1)
   {
      return;
   }

   const char *sep      = used ? "; " : "";
   size_t      sep_len  = used ? 2u : 0u;
   size_t      text_len = strlen(text);

   if (used + sep_len + text_len >= summary_size - 1)
   {
      /* Mark truncation if it fits and hasn't already been marked. */
      if (used + 4 <= summary_size - 1 &&
          (used < 3 || strcmp(summary + used - 3, "...") != 0))
      {
         snprintf(summary + used, summary_size - used, "%s...", sep);
      }
      return;
   }

   /* Space verified above — use memcpy to avoid GCC format-truncation analysis. */
   memcpy(summary + used, sep, sep_len);
   memcpy(summary + used + sep_len, text, text_len + 1); /* +1 copies the null */
}

static double
PreconReuseNormalizedDistance(PreconReuseDirection direction, double aggregate,
                              double target, double scale)
{
   double normalized = 0.0;

   if (direction == PRECON_REUSE_DIRECTION_ABOVE)
   {
      normalized = (aggregate - target) / scale;
   }
   else
   {
      normalized = (target - aggregate) / scale;
   }

   return normalized > 0.0 ? normalized : 0.0;
}

static int
PreconReuseShouldRebuildAdaptive(const PreconReuse_args *args,
                                 const IntArray *timestep_starts, const Stats *stats,
                                 PreconReuseState *state, int next_ls_id,
                                 PreconReuseDecision *decision)
{
   if (!args || !state)
   {
      return 1;
   }

   decision->used_adaptive  = 1;
   decision->age            = PreconReuseReuseAgeGet(state);
   decision->history_points = (int)state->count;

   if (next_ls_id <= 0 || (state->count == 0 && !state->baseline_valid))
   {
      decision->should_rebuild = 1;
      snprintf(decision->summary, sizeof(decision->summary), "%s",
               "adaptive initial build");
      return 1;
   }

   if (args->guards.max_reuse_solves >= 0 &&
       decision->age >= args->guards.max_reuse_solves)
   {
      decision->should_rebuild = 1;
      snprintf(decision->summary, sizeof(decision->summary),
               "guard=max_reuse_solves age=%d limit=%d", decision->age,
               args->guards.max_reuse_solves);
      return 1;
   }

   if (args->guards.rebuild_on_solver_failure && state->count > 0 &&
       !state->last_solve_succeeded)
   {
      decision->should_rebuild = 1;
      snprintf(decision->summary, sizeof(decision->summary),
               "guard=solver_failure last_solve_succeeded=0");
      return 1;
   }

   if (args->guards.rebuild_on_new_timestep && state->count > 0)
   {
      int current_timestep_idx   = -1;
      int current_timestep_start = -1;
      if (PreconReuseResolveTimestepContext(timestep_starts, stats, next_ls_id,
                                            &current_timestep_idx,
                                            &current_timestep_start) &&
          current_timestep_idx >= 0 && state->last_observation.timestep_index >= 0 &&
          current_timestep_idx != state->last_observation.timestep_index)
      {
         decision->should_rebuild = 1;
         snprintf(decision->summary, sizeof(decision->summary),
                  "guard=new_timestep previous=%d current=%d start=%d",
                  state->last_observation.timestep_index, current_timestep_idx,
                  current_timestep_start);
         return 1;
      }
   }

   if (args->guards.rebuild_on_new_level)
   {
      for (size_t i = 0; i < args->guards.rebuild_on_new_level->size; i++)
      {
         int level            = args->guards.rebuild_on_new_level->data[i];
         int current_level_id = PreconReuseCurrentLevelID(stats, level);
         if (current_level_id >= 0 &&
             current_level_id != state->last_rebuild_level_ids[level])
         {
            decision->should_rebuild = 1;
            snprintf(decision->summary, sizeof(decision->summary),
                     "guard=new_level level=%d previous=%d current=%d", level,
                     state->last_rebuild_level_ids[level], current_level_id);
            return 1;
         }
      }
   }

   if (state->baseline_valid && state->count > 0 && state->last_solve_succeeded)
   {
      if (args->guards.max_iteration_ratio > 0.0 && state->baseline_iters > 0.0)
      {
         double ratio = (double)state->last_observation.iters / state->baseline_iters;
         if (ratio >= args->guards.max_iteration_ratio)
         {
            decision->should_rebuild = 1;
            snprintf(decision->summary, sizeof(decision->summary),
                     "guard=max_iteration_ratio ratio=%.6g limit=%.6g baseline=%.6g "
                     "iters=%d",
                     ratio, args->guards.max_iteration_ratio, state->baseline_iters,
                     state->last_observation.iters);
            return 1;
         }
      }

      if (args->guards.max_solve_time_ratio > 0.0 && state->baseline_solve_time > 0.0)
      {
         double ratio = state->last_observation.solve_time / state->baseline_solve_time;
         if (ratio >= args->guards.max_solve_time_ratio)
         {
            decision->should_rebuild = 1;
            snprintf(decision->summary, sizeof(decision->summary),
                     "guard=max_solve_time_ratio ratio=%.6g limit=%.6g baseline=%.6g "
                     "solve_time=%.6g",
                     ratio, args->guards.max_solve_time_ratio, state->baseline_solve_time,
                     state->last_observation.solve_time);
            return 1;
         }
      }
   }

   if (!state->baseline_valid)
   {
      PreconReuseUpdateBadDecisionStreak(state, next_ls_id, 0);
      decision->should_rebuild = 0;
      snprintf(decision->summary, sizeof(decision->summary),
               "mode=bootstrap bootstrap=%d/%d baseline=pending streak=%d",
               state->bootstrap_count, PRECON_REUSE_BOOTSTRAP_SOLVES,
               state->bad_decision_streak);
      return 0;
   }

   if (args->guards.min_reuse_solves > 0 && decision->age < args->guards.min_reuse_solves)
   {
      PreconReuseUpdateBadDecisionStreak(state, next_ls_id, 0);
      decision->should_rebuild = 0;
      snprintf(decision->summary, sizeof(decision->summary),
               "guard=min_reuse_solves age=%d limit=%d baseline_iters=%.6g "
               "baseline_setup=%.6g baseline_solve=%.6g streak=%d",
               decision->age, args->guards.min_reuse_solves, state->baseline_iters,
               state->baseline_setup_time, state->baseline_solve_time,
               state->bad_decision_streak);
      return 0;
   }

   int components_used  = 0;
   int max_history      = 0;
   decision->score      = 0.0;
   decision->summary[0] = '\0';

   /* Find the largest history window across all enabled components so we can
    * allocate the working buffers once rather than once per component. */
   int buf_size = 1;
   for (size_t i = 0; i < args->adaptive.num_components; i++)
   {
      const PreconReuseScoreComponent_args *c = &args->adaptive.components[i];
      if (c->enabled && c->history.max_points > buf_size)
      {
         buf_size = c->history.max_points;
      }
   }

   PreconReuseSample *samples =
      (PreconReuseSample *)malloc((size_t)buf_size * sizeof(*samples));
   double *values = (double *)malloc((size_t)buf_size * sizeof(*values));
   if (!samples || !values)
   {
      free(samples);
      free(values);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate adaptive reuse working buffers");
      return 1;
   }

   /* Adaptive score formula:
    *   score = sum_i weight_i * max(0, distance_i)
    *   distance_i =
    *      (aggregate_i - target_i) / scale_i   when direction = above
    *      (target_i - aggregate_i) / scale_i   when direction = below
    *   aggregate_i = generalized_mean_p(transformed history samples)
    *
    * The named means are specialized cases of the power mean:
    * arithmetic (p = 1), geometric (p = 0), harmonic (p = -1), and rms (p = 2).
    * Keep this formula localized here so future policy changes have one place
    * to update both the code and the documentation.
    */
   for (size_t i = 0; i < args->adaptive.num_components; i++)
   {
      const PreconReuseScoreComponent_args *component = &args->adaptive.components[i];
      if (!component->enabled)
      {
         continue;
      }

      int max_points = component->history.max_points;
      if (max_points <= 0)
      {
         max_points = 1;
      }

      int sample_count =
         PreconReuseCollectSamples(component, state, stats, samples, max_points);
      if (sample_count > max_history)
      {
         max_history = sample_count;
      }

      if (sample_count < args->guards.min_history_points)
      {
         continue;
      }

      double baseline = PreconReuseBaselineValue(component, state, samples, sample_count,
                                                 args->adaptive.positive_floor);
      for (int j = 0; j < sample_count; j++)
      {
         values[j] = PreconReuseTransformSample(component, state, samples + j, baseline,
                                                args->adaptive.positive_floor);
      }

      double aggregate = PreconReuseGeneralizedMean(
         values, sample_count, &component->mean, args->adaptive.positive_floor);
      double normalized = PreconReuseNormalizedDistance(
         component->direction, aggregate, component->target, component->scale);

      decision->score += component->weight * normalized;
      components_used++;

      char entry[256];
      snprintf(entry, sizeof(entry), "%s agg=%.6g contrib=%.6g pts=%d", component->name,
               aggregate, component->weight * normalized, sample_count);
      PreconReuseAppendSummary(decision->summary, sizeof(decision->summary), entry);
   }

   free(samples);
   free(values);

   decision->history_points = max_history;

   if (components_used == 0)
   {
      PreconReuseUpdateBadDecisionStreak(state, next_ls_id, 0);
      decision->should_rebuild = 0;
      snprintf(decision->summary, sizeof(decision->summary),
               "mode=adaptive history=insufficient points=%d required=%d "
               "baseline_iters=%.6g baseline_setup=%.6g baseline_solve=%.6g streak=%d",
               max_history, args->guards.min_history_points, state->baseline_iters,
               state->baseline_setup_time, state->baseline_solve_time,
               state->bad_decision_streak);
      return 0;
   }

   int is_bad = decision->score >= args->adaptive.rebuild_threshold;
   int streak = PreconReuseUpdateBadDecisionStreak(state, next_ls_id, is_bad);
   decision->should_rebuild = is_bad && streak >= args->guards.bad_decisions_to_rebuild;

   const char *status_label = "good";
   if (decision->should_rebuild)
   {
      status_label = "rebuild";
   }
   else if (is_bad)
   {
      status_label = "hold";
   }

   char status[256];
   snprintf(status, sizeof(status),
            "mode=adaptive bootstrap=%d/%d baseline_iters=%.6g baseline_setup=%.6g "
            "baseline_solve=%.6g streak=%d/%d threshold=%.6g status=%s",
            state->bootstrap_count, PRECON_REUSE_BOOTSTRAP_SOLVES, state->baseline_iters,
            state->baseline_setup_time, state->baseline_solve_time, streak,
            args->guards.bad_decisions_to_rebuild, args->adaptive.rebuild_threshold,
            status_label);
   PreconReuseAppendSummary(decision->summary, sizeof(decision->summary), status);

   return decision->should_rebuild;
}

int
hypredrv_PreconReuseShouldRebuild(const PreconReuse_args *args,
                                  const IntArray *timestep_starts, const Stats *stats,
                                  PreconReuseState *state, int next_ls_id,
                                  PreconReuseDecision *decision)
{
   hypredrv_PreconReuseDecisionInit(decision);

   if (!args)
   {
      if (decision)
      {
         decision->should_rebuild = 1;
      }
      return 1;
   }

   if (args->policy == PRECON_REUSE_POLICY_ADAPTIVE && args->enabled)
   {
      int should_rebuild = PreconReuseShouldRebuildAdaptive(args, timestep_starts, stats,
                                                            state, next_ls_id, decision);
      return should_rebuild;
   }

   int should_rebuild =
      PreconReuseShouldRebuildStatic(args, timestep_starts, stats, next_ls_id);
   if (decision)
   {
      decision->should_rebuild = should_rebuild;
      decision->used_adaptive  = 0;
      decision->age            = state ? (int)state->count : 0;
      snprintf(decision->summary, sizeof(decision->summary),
               "static enabled=%d freq=%d per_timestep=%d", args->enabled,
               args->frequency, args->per_timestep);
   }
   return should_rebuild;
}

static int
PreconReuseParseMeanNode(YAMLnode *node, PreconReuseMean_args *mean)
{
   if (!node || !mean)
   {
      return 0;
   }

   if (!node->children)
   {
      const char *value = node->mapped_val ? node->mapped_val : node->val;
      if (!PreconReuseParseMeanKind(value, &mean->kind))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse.adaptive.score mean: '%s'",
                              value ? value : "");
         return 0;
      }
      YAML_NODE_SET_VALID(node);
      return 1;
   }

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "kind"))
      {
         if (!PreconReuseParseMeanKind(value, &mean->kind))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse.adaptive.score mean: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "power"))
      {
         if (!PreconReuseParseDouble(value, &mean->power))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.adaptive.score power: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd(
            "Unknown key under preconditioner.reuse.adaptive.score.mean: '%s'",
            child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

static int
PreconReuseParseTransformNode(YAMLnode *node, PreconReuseTransform_args *transform)
{
   if (!node || !transform)
   {
      return 0;
   }

   if (!node->children)
   {
      const char *value = node->mapped_val ? node->mapped_val : node->val;
      if (!PreconReuseParseTransformKind(value, &transform->kind))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse transform: '%s'",
                              value ? value : "");
         return 0;
      }
      YAML_NODE_SET_VALID(node);
      return 1;
   }

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "kind"))
      {
         if (!PreconReuseParseTransformKind(value, &transform->kind))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse transform: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "baseline"))
      {
         if (!PreconReuseParseBaselineKind(value, &transform->baseline))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse baseline: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "amortization_window"))
      {
         if (!PreconReuseParseInt(value, &transform->amortization_window) ||
             transform->amortization_window <= 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse amortization_window: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd(
            "Unknown key under preconditioner.reuse.adaptive.score.transform: '%s'",
            child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

static int
PreconReuseParseHistoryNode(YAMLnode *node, PreconReuseHistory_args *history)
{
   if (!node || !history)
   {
      return 0;
   }

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "source"))
      {
         if (!PreconReuseParseHistorySource(value, &history->source))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse history source: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "level"))
      {
         if (!PreconReuseParseInt(value, &history->level))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse history level: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "max_points"))
      {
         if (!PreconReuseParseInt(value, &history->max_points) ||
             history->max_points <= 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse history max_points: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "reduction"))
      {
         if (!PreconReuseParseReduction(value, &history->reduction))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse history reduction: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd(
            "Unknown key under preconditioner.reuse.adaptive.score.history: '%s'",
            child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   if (history->source != PRECON_REUSE_HISTORY_LINEAR_SOLVES &&
       (history->level < 0 || history->level >= STATS_MAX_LEVELS))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("preconditioner.reuse history source '%d' requires level",
                           (int)history->source);
      return 0;
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

static int
PreconReuseParseComponentNode(YAMLnode *node, PreconReuseScoreComponent_args *component,
                              int component_idx)
{
   if (!node || !component)
   {
      return 0;
   }

   PreconReuseScoreComponentSetDefaults(component);
   snprintf(component->name, sizeof(component->name), "component_%d", component_idx);

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "name"))
      {
         snprintf(component->name, sizeof(component->name), "%s", value ? value : "");
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "enabled"))
      {
         if (!PreconReuseParseOnOff(value, &component->enabled))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component enabled: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "metric"))
      {
         if (!PreconReuseParseMetric(value, &component->metric))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component metric: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "weight"))
      {
         if (!PreconReuseParseDouble(value, &component->weight) ||
             component->weight < 0.0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component weight: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "direction"))
      {
         if (!PreconReuseParseDirection(value, &component->direction))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component direction: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "target"))
      {
         if (!PreconReuseParseDouble(value, &component->target))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component target: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "scale"))
      {
         if (!PreconReuseParseDouble(value, &component->scale) || component->scale <= 0.0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse component scale: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "mean"))
      {
         if (!PreconReuseParseMeanNode(child, &component->mean))
         {
            return 0;
         }
      }
      else if (!strcmp(child->key, "transform"))
      {
         if (!PreconReuseParseTransformNode(child, &component->transform))
         {
            return 0;
         }
      }
      else if (!strcmp(child->key, "history"))
      {
         if (!PreconReuseParseHistoryNode(child, &component->history))
         {
            return 0;
         }
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd(
            "Unknown key under preconditioner.reuse.adaptive.score.components: '%s'",
            child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   if (component->metric == PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP &&
       component->transform.amortization_window <= 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "preconditioner.reuse solve_overhead_vs_setup requires amortization_window");
      return 0;
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

static int
PreconReuseParseGuardsNode(YAMLnode *node, PreconReuseGuards_args *guards)
{
   if (!node || !guards)
   {
      return 0;
   }

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "min_reuse_solves"))
      {
         if (!PreconReuseParseInt(value, &guards->min_reuse_solves) ||
             guards->min_reuse_solves < 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.min_reuse_solves: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "max_reuse_solves"))
      {
         if (!PreconReuseParseInt(value, &guards->max_reuse_solves))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.max_reuse_solves: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "min_history_points"))
      {
         if (!PreconReuseParseInt(value, &guards->min_history_points) ||
             guards->min_history_points <= 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.min_history_points: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "bad_decisions_to_rebuild"))
      {
         if (!PreconReuseParseInt(value, &guards->bad_decisions_to_rebuild) ||
             guards->bad_decisions_to_rebuild <= 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.bad_decisions_to_rebuild: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "max_iteration_ratio"))
      {
         if (!PreconReuseParseDouble(value, &guards->max_iteration_ratio) ||
             guards->max_iteration_ratio <= 0.0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.max_iteration_ratio: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "max_solve_time_ratio"))
      {
         if (!PreconReuseParseDouble(value, &guards->max_solve_time_ratio) ||
             guards->max_solve_time_ratio <= 0.0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.max_solve_time_ratio: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "rebuild_on_new_timestep"))
      {
         if (!PreconReuseParseOnOff(value, &guards->rebuild_on_new_timestep))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.rebuild_on_new_timestep: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "rebuild_on_solver_failure"))
      {
         if (!PreconReuseParseOnOff(value, &guards->rebuild_on_solver_failure))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.rebuild_on_solver_failure: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "rebuild_on_new_level"))
      {
         IntArray *levels = NULL;
         if (child->children)
         {
            size_t count = 0;
            for (YAMLnode *item = child->children; item != NULL; item = item->next)
            {
               if (!strcmp(item->key, "-"))
               {
                  count++;
               }
            }
            levels = hypredrv_IntArrayCreate(count);
            if (!levels)
            {
               hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
               hypredrv_ErrorMsgAdd("Failed to allocate guard levels");
               return 0;
            }
            size_t idx = 0;
            for (YAMLnode *item = child->children; item != NULL; item = item->next)
            {
               if (!strcmp(item->key, "-"))
               {
                  const char *item_value =
                     item->mapped_val ? item->mapped_val : item->val;
                  YAMLnode *level_node = item->children;
                  while (level_node && strcmp(level_node->key, "level") != 0)
                  {
                     level_node = level_node->next;
                  }
                  if (level_node)
                  {
                     item_value =
                        level_node->mapped_val ? level_node->mapped_val : level_node->val;
                  }
                  if (!PreconReuseParseInt(item_value, &levels->data[idx]))
                  {
                     hypredrv_IntArrayDestroy(&levels);
                     hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
                     hypredrv_ErrorMsgAdd(
                        "Invalid preconditioner.reuse.guards.rebuild_on_new_level");
                     return 0;
                  }
                  idx++;
                  YAML_NODE_SET_VALID(item);
               }
            }
         }
         else
         {
            hypredrv_StrToIntArray(value, &levels);
         }
         if (!levels)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.guards.rebuild_on_new_level");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         hypredrv_IntArrayDestroy(&guards->rebuild_on_new_level);
         guards->rebuild_on_new_level = levels;
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Unknown key under preconditioner.reuse.guards: '%s'",
                              child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

static int
PreconReuseParseAdaptiveNode(YAMLnode *node, PreconReuseAdaptive_args *adaptive)
{
   if (!node || !adaptive)
   {
      return 0;
   }

   YAML_NODE_ITERATE(node, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "rebuild_threshold"))
      {
         if (!PreconReuseParseDouble(value, &adaptive->rebuild_threshold))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.adaptive.rebuild_threshold: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "positive_floor"))
      {
         if (!PreconReuseParseDouble(value, &adaptive->positive_floor) ||
             adaptive->positive_floor <= 0.0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid preconditioner.reuse.adaptive.positive_floor: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return 0;
         }
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "components"))
      {
         size_t count = 0;
         for (YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            if (!strcmp(item->key, "-"))
            {
               count++;
            }
         }
         if (count == 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "preconditioner.reuse.adaptive.components cannot be empty");
            return 0;
         }
         PreconReuseScoreComponent_args *components =
            (PreconReuseScoreComponent_args *)calloc(count, sizeof(*components));
         if (!components)
         {
            hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
            hypredrv_ErrorMsgAdd("Failed to allocate adaptive components");
            return 0;
         }
         size_t idx = 0;
         for (YAMLnode *item = child->children; item != NULL; item = item->next)
         {
            if (strcmp(item->key, "-") != 0)
            {
               continue;
            }
            if (!PreconReuseParseComponentNode(item, &components[idx], (int)idx))
            {
               free(components);
               return 0;
            }
            idx++;
         }
         free(adaptive->components);
         adaptive->components     = components;
         adaptive->num_components = count;
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Unknown key under preconditioner.reuse.adaptive: '%s'",
                              child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return 0;
      }
   }

   YAML_NODE_SET_VALID(node);
   return 1;
}

void
hypredrv_PreconReuseSetArgsFromYAML(PreconReuse_args *args, YAMLnode *parent)
{
   if (!args || !parent)
   {
      return;
   }

   if (!parent->children && parent->val && strcmp(parent->val, "") != 0)
   {
      if (PreconReuseParseInt(parent->val, &args->frequency))
      {
         args->enabled = 1;
         args->policy  = PRECON_REUSE_POLICY_STATIC;
         YAML_NODE_SET_VALID(parent);
         return;
      }

      if (!PreconReuseParsePolicy(parent->val, &args->policy))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid preconditioner.reuse value: '%s'", parent->val);
         YAML_NODE_SET_INVALID_VAL(parent);
         return;
      }

      args->enabled = 1;
      if (args->policy == PRECON_REUSE_POLICY_ADAPTIVE &&
          !PreconReuseInstallDefaultAdaptiveComponents(&args->adaptive))
      {
         YAML_NODE_SET_INVALID_VAL(parent);
         return;
      }
      if (args->policy == PRECON_REUSE_POLICY_ADAPTIVE)
      {
         PreconReuseApplyAdaptiveShorthandDefaults(args);
      }
      YAML_NODE_SET_VALID(parent);
      return;
   }

   int seen_enabled           = 0;
   int seen_frequency         = 0;
   int seen_linear_system_ids = 0;
   int seen_per_timestep      = 0;
   int seen_policy            = 0;
   int seen_adaptive          = 0;

   YAML_NODE_ITERATE(parent, child)
   {
      const char *value = child->mapped_val ? child->mapped_val : child->val;
      if (!strcmp(child->key, "enabled"))
      {
         if (!PreconReuseParseOnOff(value, &args->enabled))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid value for preconditioner.reuse.enabled: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_enabled = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "frequency"))
      {
         if (!PreconReuseParseInt(value, &args->frequency) || args->frequency < 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid value for preconditioner.reuse.frequency: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_frequency = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "linear_system_ids") ||
               !strcmp(child->key, "linear_solver_ids"))
      {
         IntArray *ids = NULL;
         hypredrv_StrToIntArray(value, &ids);
         if (!ids)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Failed to parse preconditioner.reuse.linear_system_ids");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         hypredrv_IntArrayDestroy(&args->linear_system_ids);
         args->linear_system_ids = ids;
         seen_linear_system_ids  = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "per_timestep"))
      {
         if (!PreconReuseParseOnOff(value, &args->per_timestep))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid value for preconditioner.reuse.per_timestep: '%s'",
               value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_per_timestep = args->per_timestep ? 1 : 0;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "type") || !strcmp(child->key, "policy"))
      {
         if (!PreconReuseParsePolicy(value, &args->policy))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid value for preconditioner.reuse.type: '%s'",
                                 value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_policy = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "guards"))
      {
         if (!PreconReuseParseGuardsNode(child, &args->guards))
         {
            return;
         }
      }
      else if (!strcmp(child->key, "adaptive"))
      {
         if (!PreconReuseParseAdaptiveNode(child, &args->adaptive))
         {
            return;
         }
         seen_adaptive = 1;
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Unknown key under preconditioner.reuse: '%s'", child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return;
      }
   }

   if (!seen_enabled)
   {
      args->enabled = 1;
   }

   if (seen_adaptive && !seen_policy)
   {
      args->policy = PRECON_REUSE_POLICY_ADAPTIVE;
   }

   if (args->policy == PRECON_REUSE_POLICY_STATIC &&
       (seen_adaptive || args->adaptive.num_components > 0))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("preconditioner.reuse adaptive block requires type: adaptive");
      YAML_NODE_SET_INVALID_VAL(parent);
      return;
   }

   if (args->policy == PRECON_REUSE_POLICY_ADAPTIVE &&
       (seen_frequency || seen_linear_system_ids || seen_per_timestep))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("preconditioner.reuse type: adaptive cannot be used with "
                           "frequency, linear_system_ids, or per_timestep; "
                           "these keys produce a parse-time error when combined");
      YAML_NODE_SET_INVALID_VAL(parent);
      return;
   }

   if (args->policy == PRECON_REUSE_POLICY_STATIC && seen_linear_system_ids &&
       (seen_frequency || seen_per_timestep))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "preconditioner.reuse.linear_system_ids cannot be combined with "
         "frequency or per_timestep");
      YAML_NODE_SET_INVALID_VAL(parent);
      return;
   }

   if (args->policy == PRECON_REUSE_POLICY_ADAPTIVE &&
       args->adaptive.num_components == 0 &&
       !PreconReuseInstallDefaultAdaptiveComponents(&args->adaptive))
   {
      YAML_NODE_SET_INVALID_VAL(parent);
      return;
   }

   if (!args->enabled)
   {
      hypredrv_IntArrayDestroy(&args->linear_system_ids);
      args->frequency    = 0;
      args->per_timestep = 0;
   }
}
