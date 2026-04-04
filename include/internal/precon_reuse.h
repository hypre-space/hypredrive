/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PRECON_REUSE_HEADER
#define PRECON_REUSE_HEADER

#include "HYPREDRV.h"
#include "internal/containers.h"
#include "internal/stats.h"
#include "internal/yaml.h"

typedef enum PreconReusePolicy_enum
{
   PRECON_REUSE_POLICY_STATIC = 0,
   PRECON_REUSE_POLICY_ADAPTIVE,
} PreconReusePolicy;

typedef enum PreconReuseDirection_enum
{
   PRECON_REUSE_DIRECTION_ABOVE = 0,
   PRECON_REUSE_DIRECTION_BELOW,
} PreconReuseDirection;

typedef enum PreconReuseMeanKind_enum
{
   PRECON_REUSE_MEAN_ARITHMETIC = 0,
   PRECON_REUSE_MEAN_GEOMETRIC,
   PRECON_REUSE_MEAN_HARMONIC,
   PRECON_REUSE_MEAN_RMS,
   PRECON_REUSE_MEAN_MIN,
   PRECON_REUSE_MEAN_MAX,
   PRECON_REUSE_MEAN_POWER,
} PreconReuseMeanKind;

typedef enum PreconReuseTransformKind_enum
{
   PRECON_REUSE_TRANSFORM_RAW = 0,
   PRECON_REUSE_TRANSFORM_DELTA_FROM_BASELINE,
   PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE,
   PRECON_REUSE_TRANSFORM_RELATIVE_INCREASE,
} PreconReuseTransformKind;

typedef enum PreconReuseBaselineKind_enum
{
   PRECON_REUSE_BASELINE_REBUILD = 0,
   PRECON_REUSE_BASELINE_WINDOW_MEAN,
} PreconReuseBaselineKind;

typedef enum PreconReuseMetric_enum
{
   PRECON_REUSE_METRIC_ITERATIONS = 0,
   PRECON_REUSE_METRIC_SOLVE_TIME,
   PRECON_REUSE_METRIC_SETUP_TIME,
   PRECON_REUSE_METRIC_TOTAL_TIME,
   PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP,
} PreconReuseMetric;

typedef enum PreconReuseHistorySource_enum
{
   PRECON_REUSE_HISTORY_LINEAR_SOLVES = 0,
   PRECON_REUSE_HISTORY_ACTIVE_LEVEL,
   PRECON_REUSE_HISTORY_COMPLETED_LEVEL,
} PreconReuseHistorySource;

typedef enum PreconReuseReduction_enum
{
   PRECON_REUSE_REDUCTION_NONE = 0,
   PRECON_REUSE_REDUCTION_MEAN,
   PRECON_REUSE_REDUCTION_SUM,
} PreconReuseReduction;

typedef struct PreconReuseMean_args_struct
{
   PreconReuseMeanKind kind;
   double              power;
} PreconReuseMean_args;

typedef struct PreconReuseTransform_args_struct
{
   PreconReuseTransformKind kind;
   PreconReuseBaselineKind  baseline;
   int                      amortization_window;
} PreconReuseTransform_args;

typedef struct PreconReuseHistory_args_struct
{
   PreconReuseHistorySource source;
   int                      level;
   int                      max_points;
   PreconReuseReduction     reduction;
} PreconReuseHistory_args;

typedef struct PreconReuseScoreComponent_args_struct
{
   char                      name[64];
   int                       enabled;
   PreconReuseMetric         metric;
   double                    weight;
   PreconReuseDirection      direction;
   double                    target;
   double                    scale;
   PreconReuseMean_args      mean;
   PreconReuseTransform_args transform;
   PreconReuseHistory_args   history;
} PreconReuseScoreComponent_args;

typedef struct PreconReuseGuards_args_struct
{
   int       min_reuse_solves;
   int       max_reuse_solves;
   int       min_history_points;
   int       bad_decisions_to_rebuild;
   double    max_iteration_ratio;
   double    max_solve_time_ratio;
   int       rebuild_on_new_timestep;
   int       rebuild_on_solver_failure;
   IntArray *rebuild_on_new_level;
} PreconReuseGuards_args;

typedef struct PreconReuseAdaptive_args_struct
{
   double                          rebuild_threshold;
   double                          positive_floor;
   PreconReuseScoreComponent_args *components;
   size_t                          num_components;
} PreconReuseAdaptive_args;

typedef struct PreconReuse_args_struct
{
   int                      enabled;
   int                      frequency;
   IntArray                *linear_system_ids;
   int                      per_timestep;
   PreconReusePolicy        policy;
   PreconReuseGuards_args   guards;
   PreconReuseAdaptive_args adaptive;
} PreconReuse_args;

typedef struct PreconReuseObservation_struct
{
   int    system_index;
   int    timestep_index;
   int    level_ids[STATS_MAX_LEVELS];
   int    iters;
   int    solve_succeeded;
   double setup_time;
   double solve_time;
} PreconReuseObservation;

typedef struct PreconReuseState_struct
{
   PreconReuseObservation *observations;
   size_t                  count;
   size_t                  capacity;
   int                     last_rebuild_level_ids[STATS_MAX_LEVELS];
   int                     bootstrap_count;
   int                     baseline_valid;
   int                     bad_decision_streak;
   int                     last_solve_succeeded;
   int                     last_decision_ls_id;
   double                  baseline_iters;
   double                  baseline_setup_time;
   double                  baseline_solve_time;
   PreconReuseObservation  baseline;
   PreconReuseObservation  last_observation;
} PreconReuseState;

typedef struct PreconReuseDecision_struct
{
   int    should_rebuild;
   int    used_adaptive;
   double score;
   int    age;
   int    history_points;
   char   summary[1024];
} PreconReuseDecision;

typedef struct PreconReuseTimesteps_struct
{
   IntArray *ids;
   IntArray *starts;
} PreconReuseTimesteps;

void hypredrv_PreconReuseSetDefaultArgs(PreconReuse_args *);
void hypredrv_PreconReuseDestroyArgs(PreconReuse_args *);
void hypredrv_PreconReuseSetArgsFromYAML(PreconReuse_args *, YAMLnode *);
void hypredrv_PreconReuseStateInit(PreconReuseState *);
void hypredrv_PreconReuseStateDestroy(PreconReuseState *);
void hypredrv_PreconReuseStateReset(PreconReuseState *);
void hypredrv_PreconReuseStateRecordObservation(PreconReuseState *,
                                                const PreconReuseObservation *);
void hypredrv_PreconReuseBuildObservation(HYPREDRV_t, const IntArray *,
                                          PreconReuseObservation *);
void hypredrv_PreconReuseMarkRebuild(HYPREDRV_t, PreconReuseState *);
void hypredrv_PreconReuseLogDecision(HYPREDRV_t, int, const PreconReuseDecision *,
                                     const char *);
int  hypredrv_PreconReuseShouldRebuildStatic(const PreconReuse_args *, const IntArray *,
                                             const struct Stats_struct *, int);
int  hypredrv_PreconReuseShouldRebuild(const PreconReuse_args *, const IntArray *,
                                       const struct Stats_struct *, PreconReuseState *,
                                       int, PreconReuseDecision *);

#endif /* PRECON_REUSE_HEADER */
