#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "internal/amg.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/fsai.h"
#include "internal/ilu.h"
#include "internal/mgr.h"
#include "internal/krylov.h"
#include "internal/precon.h"
#include "test_helpers.h"
#include "internal/yaml.h"
#include "logging.h"

/* Forward declarations for internal AMG functions */
void           hypredrv_AMGSetFieldByName(void *, const YAMLnode *);
void           hypredrv_AMGintSetFieldByName(void *, const YAMLnode *);
void           hypredrv_AMGintSetDefaultArgs(AMGint_args *);
StrIntMapArray hypredrv_AMGintGetValidValues(const char *);
void           hypredrv_AMGcsnSetFieldByName(void *, const YAMLnode *);
void           hypredrv_AMGcsnSetDefaultArgs(AMGcsn_args *);
StrIntMapArray hypredrv_AMGcsnGetValidValues(const char *);
StrIntMapArray hypredrv_AMGaggGetValidValues(const char *);
StrIntMapArray hypredrv_AMGrlxGetValidValues(const char *);
StrIntMapArray hypredrv_AMGsmtGetValidValues(const char *);

void           hypredrv_ILUSetFieldByName(void *, const YAMLnode *);
void           hypredrv_ILUSetDefaultArgs(ILU_args *);
StrArray       hypredrv_ILUGetValidKeys(void);
StrIntMapArray hypredrv_ILUGetValidValues(const char *);
void           hypredrv_FSAISetFieldByName(void *, const YAMLnode *);
void           hypredrv_FSAISetDefaultArgs(FSAI_args *);
StrArray       hypredrv_FSAIGetValidKeys(void);
StrIntMapArray hypredrv_FSAIGetValidValues(const char *);

void hypredrv_MGRSetDefaultArgs(MGR_args *);

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = hypredrv_YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static PreconReuseObservation
make_reuse_observation(int system_index, int timestep_index, int iters, double setup_time,
                       double solve_time)
{
   PreconReuseObservation obs;
   memset(&obs, 0, sizeof(obs));
   obs.system_index    = system_index;
   obs.timestep_index  = timestep_index;
   obs.iters           = iters;
   obs.solve_succeeded = 1;
   obs.setup_time      = setup_time;
   obs.solve_time      = solve_time;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      obs.level_ids[level] = -1;
   }
   return obs;
}

static void
seed_post_bootstrap_state(PreconReuseState *state, int baseline_iters, double setup_time,
                          double solve_time)
{
   ASSERT_NOT_NULL(state);

   for (int i = 0; i < 3; i++)
   {
      PreconReuseObservation obs =
         make_reuse_observation(i, 0, baseline_iters, setup_time, solve_time);
      hypredrv_PreconReuseStateRecordObservation(state, &obs);
   }

   ASSERT_TRUE(state->baseline_valid);
   ASSERT_EQ(state->bootstrap_count, 3);
}

/* Invalid preconditioner enum for exercising default: branches in precon.c */
static const precon_t PRECON_INVALID = (precon_t)9999;

static HYPRE_IJMatrix
precon_test_ij_matrix_1x1(double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows     = 1;
   HYPRE_Int    ncols[1]  = {1};
   HYPRE_BigInt rows[1]   = {0};
   HYPRE_BigInt cols[1]  = {0};
   double       values[1] = {diag};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static HYPRE_IJVector
precon_test_ij_vector_1x1(double value)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[1] = {0};
   double       val[1] = {value};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 1, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
}

static void
precon_test_set_static_mgr_component_reuse(MGRComponentReuse_args *reuse, int frequency)
{
   ASSERT_NOT_NULL(reuse);
   memset(reuse, 0, sizeof(*reuse));
   hypredrv_PreconReuseSetDefaultArgs(&reuse->args);
   reuse->present      = 1;
   reuse->args.enabled = 1;
   reuse->args.policy  = PRECON_REUSE_POLICY_STATIC;
   reuse->args.frequency = frequency;
}

static void
test_PreconGetValidKeys_contains_expected(void)
{
   StrArray keys = hypredrv_PreconGetValidKeys();

   ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, "amg"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, "mgr"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, "ilu"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, "fsai"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, "reuse"));
}

static void
test_PreconGetValidTypeIntMap_contains_known_types(void)
{
   StrIntMapArray map = hypredrv_PreconGetValidTypeIntMap();

   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "amg"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "amg"), PRECON_BOOMERAMG);

   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "mgr"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "mgr"), PRECON_MGR);
}

static void
test_PreconSetDefaultArgs_resets_reuse(void)
{
   precon_args args;
   args.reuse = 42;

   hypredrv_PreconSetDefaultArgs(&args);

   ASSERT_EQ(args.reuse, 0);
}

static void
test_PreconSetArgsFromYAML_sets_fields(void)
{
   precon_args args;

   hypredrv_PreconSetDefaultArgs(&args);
   args.amg.max_iter = 1; /* default */

   YAMLnode *parent     = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *reuse_node = add_child(parent, "reuse", "1", 1);

   YAMLnode *amg_node = add_child(parent, "amg", "", 1);
   add_child(amg_node, "max_iter", "5", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(reuse_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(args.reuse, 1);
   ASSERT_EQ(args.amg.max_iter, 5);

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_null_or_empty_parent_noop(void)
{
   precon_args args;

   hypredrv_PreconSetDefaultArgs(&args);
   args.reuse = 7;

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, NULL);
   ASSERT_EQ(args.reuse, 7);

   {
      YAMLnode *empty = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
      hypredrv_PreconSetArgsFromYAML(&args, empty);
      hypredrv_YAMLnodeDestroy(empty);
   }
   ASSERT_EQ(args.reuse, 7);
}

static void
test_PreconSetArgsFromYAML_mgr_coarsest_level_spdirect_flat(void)
{
   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   YAMLnode *parent   = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   add_child(mgr_node, "coarsest_level", "spdirect", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, parent);

   /* Flat value must map to type=29 (spdirect) */
   ASSERT_EQ(args.mgr.coarsest_level.type, 29);

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_flat_sets_type(void)
{
   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   YAMLnode *parent   = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   add_child(mgr_node, "coarsest_level", "ilu", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, parent);

   /* Flat value must map to MGR coarsest ILU type=32 */
   ASSERT_EQ(args.mgr.coarsest_level.type, 32);

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_nested_sets_type_and_args(void)
{
   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   YAMLnode *parent   = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   YAMLnode *cls_node = add_child(mgr_node, "coarsest_level", "", 2);
   YAMLnode *ilu_node = add_child(cls_node, "ilu", "", 3);
   add_child(ilu_node, "type", "bj-ilut", 4);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.mgr.coarsest_level.type, 32);
   /* ILU type bj-ilut should map to 1 */
   ASSERT_EQ(args.mgr.coarsest_level.ilu.type, 1);

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_ignores_unknown_key(void)
{
   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);

   YAMLnode *parent       = hypredrv_YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *unknown_node = add_child(parent, "unknown", "value", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(unknown_node->valid, YAML_NODE_INVALID_KEY);
   ASSERT_EQ(args.reuse, 0); /* remains default */

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuse_null_args_noop(void)
{
   hypredrv_PreconReuseSetDefaultArgs(NULL);
   hypredrv_PreconReuseDestroyArgs(NULL);
   hypredrv_PreconReuseStateInit(NULL);
   hypredrv_PreconReuseStateDestroy(NULL);
   hypredrv_PreconReuseStateReset(NULL);
   hypredrv_PreconReuseStateRecordObservation(NULL, NULL);
}

static void
test_mgr_nested_frelax_wrapper_api_null_safe(void)
{
   ASSERT_EQ(hypredrv_MGRNestedFRelaxWrapperIsLive((HYPRE_Solver)(uintptr_t)1), 0);
   ASSERT_NULL(hypredrv_MGRNestedFRelaxWrapperGetInner(NULL));
   ASSERT_NULL(hypredrv_MGRNestedFRelaxWrapperDetachInner(NULL));
   hypredrv_MGRNestedFRelaxWrapperFree(NULL);
   {
      HYPRE_Solver s = NULL;
      hypredrv_MGRNestedFRelaxWrapperFree(&s);
   }
   hypredrv_MGRSetDofLabels(NULL);
}

static void
test_hypredrv_PreconArgsSetDefaultsForMethod_and_GetValidValues(void)
{
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG, NULL);

   StrIntMapArray void_map = hypredrv_PreconGetValidValues("type");
   ASSERT_EQ(void_map.size, 0);

   StrIntMapArray types = hypredrv_PreconGetValidTypeIntMap();
   ASSERT_TRUE(types.size > 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(types, "mgr"), (int)PRECON_MGR);

   precon_args args;
   memset(&args, 0, sizeof(args));
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG, &args);
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &args);
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_ILU, &args);
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_FSAI, &args);
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_NONE, &args);
   hypredrv_PreconArgsSetDefaultsForMethod((precon_t)999, &args);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_type_parses_components(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *guards = add_child(parent, "guards", "", 1);
   add_child(guards, "min_reuse_solves", "1", 2);
   add_child(guards, "max_reuse_solves", "7", 2);
   add_child(guards, "min_history_points", "2", 2);
   add_child(guards, "bad_decisions_to_rebuild", "3", 2);
   add_child(guards, "max_iteration_ratio", "2.25", 2);
   add_child(guards, "max_solve_time_ratio", "1.75", 2);
   add_child(guards, "rebuild_on_new_timestep", "yes", 2);
   add_child(guards, "rebuild_on_solver_failure", "no", 2);

   YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
   add_child(adaptive, "rebuild_threshold", "0.75", 2);
   YAMLnode *components = add_child(adaptive, "components", "", 2);
   YAMLnode *item       = add_child(components, "-", "", 3);
   add_child(item, "name", "iters-rms", 4);
   add_child(item, "metric", "iterations", 4);
   add_child(item, "weight", "2.0", 4);
   add_child(item, "target", "1.5", 4);
   add_child(item, "scale", "0.25", 4);
   YAMLnode *mean = add_child(item, "mean", "", 4);
   add_child(mean, "kind", "rms", 5);
   YAMLnode *transform = add_child(item, "transform", "", 4);
   add_child(transform, "kind", "ratio_to_baseline", 5);
   add_child(transform, "baseline", "rebuild", 5);
   YAMLnode *history = add_child(item, "history", "", 4);
   add_child(history, "source", "active_level", 5);
   add_child(history, "level", "0", 5);
   add_child(history, "max_points", "3", 5);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ(args.guards.min_reuse_solves, 1);
   ASSERT_EQ(args.guards.max_reuse_solves, 7);
   ASSERT_EQ(args.guards.min_history_points, 2);
   ASSERT_EQ(args.guards.bad_decisions_to_rebuild, 3);
   ASSERT_EQ_DOUBLE(args.guards.max_iteration_ratio, 2.25, 1.0e-12);
   ASSERT_EQ_DOUBLE(args.guards.max_solve_time_ratio, 1.75, 1.0e-12);
   ASSERT_EQ(args.guards.rebuild_on_new_timestep, 1);
   ASSERT_EQ(args.guards.rebuild_on_solver_failure, 0);
   ASSERT_EQ_DOUBLE(args.adaptive.rebuild_threshold, 0.75, 1.0e-12);
   ASSERT_EQ_SIZE(args.adaptive.num_components, 1);
   ASSERT_STREQ(args.adaptive.components[0].name, "iters-rms");
   ASSERT_EQ(args.adaptive.components[0].metric, PRECON_REUSE_METRIC_ITERATIONS);
   ASSERT_EQ_DOUBLE(args.adaptive.components[0].weight, 2.0, 1.0e-12);
   ASSERT_EQ(args.adaptive.components[0].mean.kind, PRECON_REUSE_MEAN_RMS);
   ASSERT_EQ(args.adaptive.components[0].history.source,
             PRECON_REUSE_HISTORY_ACTIVE_LEVEL);
   ASSERT_EQ(args.adaptive.components[0].history.level, 0);
   ASSERT_EQ(args.adaptive.components[0].history.max_points, 3);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_scalar_installs_default_components(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "adaptive", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ(args.guards.min_history_points, 3);
   ASSERT_EQ(args.guards.bad_decisions_to_rebuild, 2);
   ASSERT_EQ_DOUBLE(args.guards.max_iteration_ratio, 3.0, 1.0e-12);
   ASSERT_EQ_SIZE(args.adaptive.num_components, 2);

   ASSERT_STREQ(args.adaptive.components[0].name, "efficiency");
   ASSERT_EQ(args.adaptive.components[0].metric,
             PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP);
   ASSERT_EQ(args.adaptive.components[0].transform.kind, PRECON_REUSE_TRANSFORM_RAW);
   ASSERT_EQ(args.adaptive.components[0].history.source,
             PRECON_REUSE_HISTORY_LINEAR_SOLVES);

   ASSERT_STREQ(args.adaptive.components[1].name, "stability");
   ASSERT_EQ(args.adaptive.components[1].metric, PRECON_REUSE_METRIC_ITERATIONS);
   ASSERT_EQ(args.adaptive.components[1].mean.kind, PRECON_REUSE_MEAN_RMS);
   ASSERT_EQ(args.adaptive.components[1].transform.kind,
             PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_allows_disabled_hard_guards(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *guards = add_child(parent, "guards", "", 1);
   add_child(guards, "max_iteration_ratio", "-1", 2);
   add_child(guards, "max_solve_time_ratio", "-1", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ_DOUBLE(args.guards.max_iteration_ratio, -1.0, 1.0e-12);
   ASSERT_EQ_DOUBLE(args.guards.max_solve_time_ratio, -1.0, 1.0e-12);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_rebuild_on_new_level_block_sequence(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *guards = add_child(parent, "guards", "", 1);
   YAMLnode *levels = add_child(guards, "rebuild_on_new_level", "", 2);
   add_child(levels, "-", "0", 3);
   add_child(levels, "-", "2", 3);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_NOT_NULL(args.guards.rebuild_on_new_level);
   ASSERT_EQ_SIZE(args.guards.rebuild_on_new_level->size, 2);
   ASSERT_EQ(args.guards.rebuild_on_new_level->data[0], 0);
   ASSERT_EQ(args.guards.rebuild_on_new_level->data[1], 2);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_linear_solver_ids_alias(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "linear_solver_ids", "[0, 2]", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_NOT_NULL(args.linear_system_ids);
   ASSERT_EQ_SIZE(args.linear_system_ids->size, 2);
   ASSERT_EQ(args.linear_system_ids->data[0], 0);
   ASSERT_EQ(args.linear_system_ids->data[1], 2);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_rebuild_on_new_level_rejects_out_of_range(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *guards = add_child(parent, "guards", "", 1);
   add_child(guards, "rebuild_on_new_level", "[99]", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_NE(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_NULL(args.guards.rebuild_on_new_level);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_type_without_components_uses_defaults(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
   add_child(adaptive, "rebuild_threshold", "2.5", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ(args.guards.min_history_points, 3);
   ASSERT_EQ(args.guards.bad_decisions_to_rebuild, 2);
   ASSERT_EQ_DOUBLE(args.guards.max_iteration_ratio, 3.0, 1.0e-12);
   ASSERT_EQ_DOUBLE(args.adaptive.rebuild_threshold, 2.5, 1.0e-12);
   ASSERT_EQ_SIZE(args.adaptive.num_components, 2);
   ASSERT_STREQ(args.adaptive.components[0].name, "efficiency");
   ASSERT_STREQ(args.adaptive.components[1].name, "stability");

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_scalar_and_explicit_type_share_defaults(void)
{
   PreconReuse_args scalar_args, explicit_args;
   hypredrv_PreconReuseSetDefaultArgs(&scalar_args);
   hypredrv_PreconReuseSetDefaultArgs(&explicit_args);

   YAMLnode *scalar_parent = hypredrv_YAMLnodeCreate("reuse", "adaptive", 0);

   YAMLnode *explicit_parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(explicit_parent, "enabled", "on", 1);
   add_child(explicit_parent, "type", "adaptive", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&scalar_args, scalar_parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&explicit_args, explicit_parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_EQ(scalar_args.enabled, explicit_args.enabled);
   ASSERT_EQ(scalar_args.policy, explicit_args.policy);
   ASSERT_EQ(scalar_args.guards.min_reuse_solves, explicit_args.guards.min_reuse_solves);
   ASSERT_EQ(scalar_args.guards.max_reuse_solves, explicit_args.guards.max_reuse_solves);
   ASSERT_EQ(scalar_args.guards.min_history_points,
             explicit_args.guards.min_history_points);
   ASSERT_EQ(scalar_args.guards.bad_decisions_to_rebuild,
             explicit_args.guards.bad_decisions_to_rebuild);
   ASSERT_EQ_DOUBLE(scalar_args.guards.max_iteration_ratio,
                    explicit_args.guards.max_iteration_ratio, 1.0e-12);
   ASSERT_EQ_DOUBLE(scalar_args.adaptive.rebuild_threshold,
                    explicit_args.adaptive.rebuild_threshold, 1.0e-12);
   ASSERT_EQ_DOUBLE(scalar_args.adaptive.positive_floor,
                    explicit_args.adaptive.positive_floor, 1.0e-12);
   ASSERT_EQ_SIZE(scalar_args.adaptive.num_components, explicit_args.adaptive.num_components);

   for (size_t i = 0; i < scalar_args.adaptive.num_components; i++)
   {
      const PreconReuseScoreComponent_args *lhs = &scalar_args.adaptive.components[i];
      const PreconReuseScoreComponent_args *rhs = &explicit_args.adaptive.components[i];
      ASSERT_STREQ(lhs->name, rhs->name);
      ASSERT_EQ(lhs->enabled, rhs->enabled);
      ASSERT_EQ(lhs->metric, rhs->metric);
      ASSERT_EQ_DOUBLE(lhs->weight, rhs->weight, 1.0e-12);
      ASSERT_EQ(lhs->direction, rhs->direction);
      ASSERT_EQ_DOUBLE(lhs->target, rhs->target, 1.0e-12);
      ASSERT_EQ_DOUBLE(lhs->scale, rhs->scale, 1.0e-12);
      ASSERT_EQ(lhs->mean.kind, rhs->mean.kind);
      ASSERT_EQ_DOUBLE(lhs->mean.power, rhs->mean.power, 1.0e-12);
      ASSERT_EQ(lhs->transform.kind, rhs->transform.kind);
      ASSERT_EQ(lhs->transform.baseline, rhs->transform.baseline);
      ASSERT_EQ(lhs->transform.amortization_window, rhs->transform.amortization_window);
      ASSERT_EQ(lhs->history.source, rhs->history.source);
      ASSERT_EQ(lhs->history.level, rhs->history.level);
      ASSERT_EQ(lhs->history.max_points, rhs->history.max_points);
      ASSERT_EQ(lhs->history.reduction, rhs->history.reduction);
   }

   hypredrv_PreconReuseDestroyArgs(&scalar_args);
   hypredrv_PreconReuseDestroyArgs(&explicit_args);
   hypredrv_YAMLnodeDestroy(scalar_parent);
   hypredrv_YAMLnodeDestroy(explicit_parent);
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_rejects_negative_component_weight(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "adaptive", 1);

   YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
   YAMLnode *components = add_child(adaptive, "components", "", 2);
   YAMLnode *item       = add_child(components, "-", "", 3);
   YAMLnode *weight     = add_child(item, "weight", "-1.0", 4);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_EQ(weight->valid, YAML_NODE_INVALID_VAL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseShouldRebuild_adaptive_iterations_rms(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                   = 1;
   args.policy                    = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points = 2;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components   = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "iters-rms");
   component->enabled                = 1;
   component->metric                 = PRECON_REUSE_METRIC_ITERATIONS;
   component->weight                 = 1.0;
   component->direction              = PRECON_REUSE_DIRECTION_ABOVE;
   component->target                 = 1.5;
   component->scale                  = 0.25;
   component->mean.kind              = PRECON_REUSE_MEAN_RMS;
   component->transform.kind         = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   component->transform.baseline     = PRECON_REUSE_BASELINE_REBUILD;
   component->transform.amortization_window = 10;
   component->history.source         = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   component->history.level          = -1;
   component->history.max_points     = 2;
   component->history.reduction      = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 25, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(decision.score >= args.adaptive.rebuild_threshold);
   ASSERT_TRUE(strstr(decision.summary, "iters-rms") != NULL);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_max_reuse_solves_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.max_reuse_solves    = 4;
   args.adaptive.rebuild_threshold = 100.0; /* high threshold so only the guard fires */

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 5, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 5, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_EQ(decision.age, 4);
   ASSERT_EQ(decision.should_rebuild, 1);
   ASSERT_TRUE(strstr(decision.summary, "max_reuse_solves") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_rebuild_on_new_level_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   Stats              *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.adaptive.rebuild_threshold = 100.0; /* high threshold so only the guard fires */

   const int level_list[1]       = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, level_list, &args.guards.rebuild_on_new_level);
   ASSERT_NOT_NULL(args.guards.rebuild_on_new_level);

   /* level_current_id is 1-based; PreconReuseCurrentLevelID returns it minus 1 */
   stats->level_active        |= (1 << 0);
   stats->level_current_id[0]  = 5; /* current = 4 */

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 5, 1.0, 1.0);
   state.last_rebuild_level_ids[0] = 3; /* differs from current 4 */

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 1,
                                                    &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_EQ(decision.should_rebuild, 1);
   ASSERT_TRUE(strstr(decision.summary, "guard=new_level") != NULL);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_active_level_scope(void)
{
   PreconReuse_args  args;
   PreconReuseState  state;
   PreconReuseDecision decision;
   Stats            *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 2;
   args.adaptive.rebuild_threshold = 1.0;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "active-level");
   component->enabled                = 1;
   component->metric                 = PRECON_REUSE_METRIC_ITERATIONS;
   component->weight                 = 1.0;
   component->direction              = PRECON_REUSE_DIRECTION_ABOVE;
   component->target                 = 2.0;
   component->scale                  = 1.0;
   component->mean.kind              = PRECON_REUSE_MEAN_ARITHMETIC;
   component->transform.kind         = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   component->transform.baseline     = PRECON_REUSE_BASELINE_REBUILD;
   component->history.source         = PRECON_REUSE_HISTORY_ACTIVE_LEVEL;
   component->history.level          = 0;
   component->history.max_points     = 2;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   obs.level_ids[0] = 1;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   obs.iters        = 40;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   stats->level_active |= (1 << 0);
   stats->level_current_id[0] = 2;

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 5,
                                                   &decision));
   ASSERT_TRUE(strstr(decision.summary, "active-level") != NULL);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_completed_level_scope(void)
{
   PreconReuse_args  args;
   PreconReuseState  state;
   PreconReuseDecision decision;
   Stats            *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 2;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "completed-level");
   component->enabled                = 1;
   component->metric                 = PRECON_REUSE_METRIC_SOLVE_TIME;
   component->weight                 = 1.0;
   component->direction              = PRECON_REUSE_DIRECTION_ABOVE;
   component->target                 = 1.5;
   component->scale                  = 0.5;
   component->mean.kind              = PRECON_REUSE_MEAN_ARITHMETIC;
   component->transform.kind         = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   component->transform.baseline     = PRECON_REUSE_BASELINE_REBUILD;
   component->history.source         = PRECON_REUSE_HISTORY_COMPLETED_LEVEL;
   component->history.level          = 0;
   component->history.max_points     = 2;
   component->history.reduction      = PRECON_REUSE_REDUCTION_MEAN;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);
   state.baseline.system_index = 0;

   stats->counter             = 1;
   stats->ls_counter          = 2;
   stats->solve[0]            = 3.0;
   stats->solve[1]            = 4.0;
   stats->prec[0]             = 1.0;
   stats->prec[1]             = 1.0;
   stats->iters[0]            = 10;
   stats->iters[1]            = 12;
   stats->level_count[0]      = 2;
   stats->level_entries[0][0] = (LevelEntry){.id = 1, .solve_start = 0, .solve_end = 1};
   stats->level_entries[0][1] = (LevelEntry){.id = 2, .solve_start = 1, .solve_end = 2};

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 4,
                                                   &decision));
   ASSERT_TRUE(strstr(decision.summary, "completed-level") != NULL);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_completed_level_scope_excludes_pre_rebuild_history(
   void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   Stats              *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "completed-level-window");
   component->enabled            = 1;
   component->metric             = PRECON_REUSE_METRIC_SOLVE_TIME;
   component->weight             = 1.0;
   component->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   component->target             = 1.5;
   component->scale              = 0.5;
   component->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   component->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   component->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   component->history.source     = PRECON_REUSE_HISTORY_COMPLETED_LEVEL;
   component->history.level      = 0;
   component->history.max_points = 2;
   component->history.reduction  = PRECON_REUSE_REDUCTION_MEAN;

   hypredrv_PreconReuseStateInit(&state);
   state.bootstrap_count      = 3;
   state.baseline_valid       = 1;
   state.baseline.system_index = 2;
   state.baseline.solve_time   = 1.0;
   state.baseline.setup_time   = 1.0;
   state.baseline.iters        = 10;
   state.baseline_iters        = 10.0;
   state.baseline_setup_time   = 1.0;
   state.baseline_solve_time   = 1.0;

   stats->counter             = 3;
   stats->ls_counter          = 4;
   stats->solve[0]            = 100.0;
   stats->solve[1]            = 100.0;
   stats->solve[2]            = 1.0;
   stats->solve[3]            = 1.0;
   stats->prec[0]             = 1.0;
   stats->prec[1]             = 1.0;
   stats->prec[2]             = 1.0;
   stats->prec[3]             = 1.0;
   stats->iters[0]            = 10;
   stats->iters[1]            = 10;
   stats->iters[2]            = 10;
   stats->iters[3]            = 10;
   stats->level_count[0]      = 2;
   stats->level_entries[0][0] = (LevelEntry){.id = 1, .solve_start = 0, .solve_end = 2};
   stats->level_entries[0][1] = (LevelEntry){.id = 2, .solve_start = 2, .solve_end = 4};

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 4,
                                                    &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_EQ(decision.should_rebuild, 0);
   ASSERT_TRUE(strstr(decision.summary, "completed-level-window") != NULL);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_completed_level_solve_overhead_honors_mean_reduction(
   void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   Stats              *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "completed-overhead-mean");
   component->enabled                       = 1;
   component->metric                        = PRECON_REUSE_METRIC_SOLVE_OVERHEAD_VS_SETUP;
   component->weight                        = 1.0;
   component->direction                     = PRECON_REUSE_DIRECTION_ABOVE;
   component->target                        = 1.0;
   component->scale                         = 0.25;
   component->mean.kind                     = PRECON_REUSE_MEAN_ARITHMETIC;
   component->transform.kind                = PRECON_REUSE_TRANSFORM_RAW;
   component->transform.baseline            = PRECON_REUSE_BASELINE_REBUILD;
   component->transform.amortization_window = 2;
   component->history.source                = PRECON_REUSE_HISTORY_COMPLETED_LEVEL;
   component->history.level                 = 0;
   component->history.max_points            = 1;
   component->history.reduction             = PRECON_REUSE_REDUCTION_MEAN;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 4.0, 1.0);
   state.baseline.system_index = 0;

   stats->counter             = 3;
   stats->ls_counter          = 4;
   stats->solve[0]            = 2.0;
   stats->solve[1]            = 2.0;
   stats->solve[2]            = 2.0;
   stats->solve[3]            = 2.0;
   stats->prec[0]             = 0.0;
   stats->prec[1]             = 0.0;
   stats->prec[2]             = 0.0;
   stats->prec[3]             = 0.0;
   stats->iters[0]            = 10;
   stats->iters[1]            = 10;
   stats->iters[2]            = 10;
   stats->iters[3]            = 10;
   stats->level_count[0]      = 1;
   stats->level_entries[0][0] = (LevelEntry){.id = 1, .solve_start = 0, .solve_end = 4};

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 4,
                                                    &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_EQ(decision.should_rebuild, 0);
   ASSERT_TRUE(strstr(decision.summary, "completed-overhead-mean") != NULL);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_bootstrap_defers_scoring(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.adaptive.rebuild_threshold = 0.0;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   snprintf(args.adaptive.components[0].name, sizeof(args.adaptive.components[0].name),
            "%s", "bootstrap");
   args.adaptive.components[0].enabled            = 1;
   args.adaptive.components[0].metric             = PRECON_REUSE_METRIC_ITERATIONS;
   args.adaptive.components[0].weight             = 1.0;
   args.adaptive.components[0].direction          = PRECON_REUSE_DIRECTION_ABOVE;
   args.adaptive.components[0].target             = 1.5;
   args.adaptive.components[0].scale              = 0.25;
   args.adaptive.components[0].mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   args.adaptive.components[0].transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   args.adaptive.components[0].transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   args.adaptive.components[0].history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   args.adaptive.components[0].history.max_points = 2;

   hypredrv_PreconReuseStateInit(&state);

   PreconReuseObservation obs = make_reuse_observation(0, 0, 10, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 1;
   obs.iters        = 50;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 2,
                                                    &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_EQ(decision.should_rebuild, 0);
   ASSERT_TRUE(strstr(decision.summary, "mode=bootstrap") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_bad_decision_streak(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                         = 1;
   args.policy                          = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points       = 1;
   args.guards.bad_decisions_to_rebuild = 2;
   args.adaptive.rebuild_threshold      = 0.5;
   args.adaptive.num_components         = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);

   PreconReuseScoreComponent_args *component = &args.adaptive.components[0];
   snprintf(component->name, sizeof(component->name), "%s", "iters-streak");
   component->enabled            = 1;
   component->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   component->weight             = 1.0;
   component->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   component->target             = 1.5;
   component->scale              = 0.25;
   component->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   component->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   component->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   component->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   component->history.max_points = 1;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                    &decision));
   ASSERT_EQ(state.bad_decision_streak, 1);
   ASSERT_TRUE(strstr(decision.summary, "status=hold") != NULL);

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                    &decision));
   ASSERT_EQ(state.bad_decision_streak, 1);

   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(state.bad_decision_streak, 2);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_default_max_iteration_ratio_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "adaptive", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_DOUBLE(args.guards.max_iteration_ratio, 3.0, 1.0e-12);

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 31, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                 &decision));
   ASSERT_TRUE(strstr(decision.summary, "max_iteration_ratio") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseShouldRebuild_adaptive_max_iteration_ratio_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.max_iteration_ratio = 2.0;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 25, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                   &decision));
   ASSERT_TRUE(strstr(decision.summary, "max_iteration_ratio") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_max_solve_time_ratio_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.max_solve_time_ratio = 2.0;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 1.0, 2.5);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                   &decision));
   ASSERT_TRUE(strstr(decision.summary, "max_solve_time_ratio") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_rebuild_on_new_timestep_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   IntArray           *timestep_starts = NULL;
   int                 starts[2] = {0, 3};

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                        = 1;
   args.policy                         = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.rebuild_on_new_timestep = 1;

   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, starts, &timestep_starts);
   ASSERT_NOT_NULL(timestep_starts);

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, timestep_starts, NULL, &state,
                                                   3, &decision));
   ASSERT_TRUE(strstr(decision.summary, "new_timestep") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_IntArrayDestroy(&timestep_starts);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_solver_failure_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                         = 1;
   args.policy                          = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.rebuild_on_solver_failure = 1;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 0, 1.0, 1.0);
   obs.solve_succeeded = 0;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                   &decision));
   ASSERT_TRUE(strstr(decision.summary, "solver_failure") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_setup_time_metric(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "setup-time");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_SETUP_TIME;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 1.5;
   comp->scale              = 0.5;
   comp->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   /* setup_time=3.0 → ratio=3.0, arith mean=3.0, dist=(3-1.5)/0.5=3.0 > threshold=0.5 */
   PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 3.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(decision.score >= args.adaptive.rebuild_threshold);
   ASSERT_TRUE(strstr(decision.summary, "setup-time") != NULL);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_geometric_mean(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "iters-geometric");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 1.5;
   comp->scale              = 0.5;
   comp->mean.kind          = PRECON_REUSE_MEAN_GEOMETRIC;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   /* iters=30 → ratio=3.0; geom_mean([3,3])=3.0, dist=(3-1.5)/0.5=3.0 > threshold=0.5 */
   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(decision.score >= args.adaptive.rebuild_threshold);
   ASSERT_TRUE(strstr(decision.summary, "iters-geometric") != NULL);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_harmonic_mean(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "solve-time-harmonic");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_SOLVE_TIME;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 1.5;
   comp->scale              = 0.5;
   comp->mean.kind          = PRECON_REUSE_MEAN_HARMONIC;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   /* solve_time=4.0 → ratio=4.0; harm_mean([4,4])=4.0, dist=(4-1.5)/0.5=5.0 > threshold=0.5 */
   PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 1.0, 4.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(decision.score >= args.adaptive.rebuild_threshold);
   ASSERT_TRUE(strstr(decision.summary, "solve-time-harmonic") != NULL);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_null_args_early_exit(void)
{
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(NULL, NULL, NULL, &state, 0, &decision), 1);
   ASSERT_EQ(decision.should_rebuild, 1);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(NULL, NULL, NULL, &state, 0, NULL), 1);

   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_adaptive_min_max_mean_iterations(void)
{
   /* MIN over per-solve iteration ratios (exercises min loop in PreconReuseGeneralizedMean) */
   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
      snprintf(comp->name, sizeof(comp->name), "%s", "iters-min");
      comp->enabled            = 1;
      comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      comp->weight             = 1.0;
      comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      comp->target             = 1.0;
      comp->scale              = 0.5;
      comp->mean.kind          = PRECON_REUSE_MEAN_MIN;
      comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      comp->history.level      = -1;
      comp->history.max_points = 2;
      comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      obs.iters          = 15;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                    &decision));
      ASSERT_EQ(decision.used_adaptive, 1);
      ASSERT_TRUE(strstr(decision.summary, "iters-min") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   /* MAX over per-solve iteration ratios */
   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
      snprintf(comp->name, sizeof(comp->name), "%s", "iters-max");
      comp->enabled            = 1;
      comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      comp->weight             = 1.0;
      comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      comp->target             = 1.0;
      comp->scale              = 0.5;
      comp->mean.kind          = PRECON_REUSE_MEAN_MAX;
      comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      comp->history.level      = -1;
      comp->history.max_points = 2;
      comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 15, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      obs.iters          = 30;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                    &decision));
      ASSERT_EQ(decision.used_adaptive, 1);
      ASSERT_TRUE(strstr(decision.summary, "iters-max") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }
}

static void
test_PreconReuseShouldRebuild_adaptive_power_mean_near_zero(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "iters-power");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 1.5;
   comp->scale              = 0.5;
   comp->mean.kind          = PRECON_REUSE_MEAN_POWER;
   comp->mean.power         = 0.0;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5, &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(strstr(decision.summary, "iters-power") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_delta_from_baseline_transform(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "iters-delta");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 5.0;
   comp->scale              = 1.0;
   comp->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_DELTA_FROM_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   /* iters=20 → delta=fmax(20-10,0)=10; arith_mean([10,10])=10, dist=(10-5)/1=5 > threshold=0.5 */
   PreconReuseObservation obs = make_reuse_observation(3, 0, 20, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   obs.system_index = 4;
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                   &decision));
   ASSERT_EQ(decision.used_adaptive, 1);
   ASSERT_TRUE(decision.score >= args.adaptive.rebuild_threshold);
   ASSERT_TRUE(strstr(decision.summary, "iters-delta") != NULL);
   ASSERT_TRUE(strstr(decision.summary, "status=rebuild") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_min_reuse_solves_guard(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_reuse_solves    = 4;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *comp = &args.adaptive.components[0];
   snprintf(comp->name, sizeof(comp->name), "%s", "iters-arith");
   comp->enabled            = 1;
   comp->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   comp->weight             = 1.0;
   comp->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   comp->target             = 1.5;
   comp->scale              = 0.5;
   comp->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   comp->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   comp->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   comp->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   comp->history.level      = -1;
   comp->history.max_points = 2;
   comp->history.reduction  = PRECON_REUSE_REDUCTION_NONE;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 3,
                                                    &decision));
   ASSERT_EQ(decision.age, 3);
   ASSERT_EQ(decision.should_rebuild, 0);
   ASSERT_TRUE(strstr(decision.summary, "min_reuse_solves") != NULL);

   /* Each observation has iters=30 (ratio=3.0, score >> threshold) so the scoring
    * path would always fire once the age guard is satisfied. */
   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                   &decision));
   ASSERT_EQ(decision.age, 4);
   ASSERT_EQ(decision.should_rebuild, 1);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuse_api_null_observation_and_log_noop(void)
{
   PreconReuseObservation obs;
   PreconReuseDecision    decision;

   memset(&decision, 0, sizeof(decision));
   hypredrv_PreconReuseLogDecision(NULL, 0, &decision, "unit");
   hypredrv_PreconReuseLogDecision(NULL, 0, NULL, "unit");

   hypredrv_PreconReuseBuildObservation(NULL, NULL, NULL);
   hypredrv_PreconReuseBuildObservation(NULL, NULL, &obs);
   ASSERT_EQ(obs.system_index, -1);

   hypredrv_PreconReuseMarkRebuild(NULL, NULL);
}

static void
test_PreconReuseSetArgsFromYAML_scalar_static_frequency_shorthand(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "2", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_STATIC);
   ASSERT_EQ(args.frequency, 2);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_scalar_always_shorthand(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "always", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_STATIC);
   ASSERT_EQ(args.frequency, 0);
   ASSERT_NOT_NULL(args.linear_system_ids);
   ASSERT_EQ_SIZE(args.linear_system_ids->size, 1);
   ASSERT_EQ(args.linear_system_ids->data[0], 0);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_type_always_alias(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "type", "always", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_STATIC);
   ASSERT_EQ(args.frequency, 0);
   ASSERT_NOT_NULL(args.linear_system_ids);
   ASSERT_EQ_SIZE(args.linear_system_ids->size, 1);
   ASSERT_EQ(args.linear_system_ids->data[0], 0);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_always_key_rejected(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent      = hypredrv_YAMLnodeCreate("reuse", "", 0);
   YAMLnode *always_node = add_child(parent, "always", "yes", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
   ASSERT_EQ(always_node->valid, YAML_NODE_INVALID_KEY);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_scalar_invalid_value(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "not-a-number-or-policy", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_EQ(parent->valid, YAML_NODE_INVALID_VAL);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_enabled_off_clears_static_fields(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "off", 1);
   add_child(parent, "frequency", "5", 1);
   add_child(parent, "per_timestep", "on", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.enabled, 0);
   ASSERT_EQ(args.frequency, 0);
   ASSERT_EQ(args.per_timestep, 0);
   ASSERT_NULL(args.linear_system_ids);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_static_conflict_requires_explicit_adaptive(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "static", 1);
   YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
   add_child(adaptive, "rebuild_threshold", "1.0", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_EQ(parent->valid, YAML_NODE_INVALID_VAL);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_adaptive_conflict_with_frequency(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "frequency", "2", 1);
   YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
   add_child(adaptive, "rebuild_threshold", "1.0", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_EQ(parent->valid, YAML_NODE_INVALID_VAL);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_static_conflict_linear_system_ids_with_frequency(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   add_child(parent, "enabled", "yes", 1);
   add_child(parent, "type", "static", 1);
   add_child(parent, "linear_system_ids", "[0]", 1);
   add_child(parent, "frequency", "1", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_EQ(parent->valid, YAML_NODE_INVALID_VAL);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_infer_adaptive_policy_from_adaptive_block(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
   add_child(adaptive, "rebuild_threshold", "1.25", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.policy, PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ(args.enabled, 1);
   ASSERT_EQ_DOUBLE(args.adaptive.rebuild_threshold, 1.25, 1.0e-12);
   ASSERT_EQ_SIZE(args.adaptive.num_components, 2u);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_PreconReuseSetArgsFromYAML_unknown_root_key(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
   YAMLnode *bad     = add_child(parent, "not_a_valid_reuse_key", "1", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconReuseSetArgsFromYAML(&args, parent);

   ASSERT_EQ(bad->valid, YAML_NODE_INVALID_KEY);

   hypredrv_PreconReuseDestroyArgs(&args);
   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorCodeResetAll();
}

static void
test_PreconReuseSetArgsFromYAML_parse_mean_transform_history_errors(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "mean", "not-a-mean-kind", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *mean       = add_child(item, "mean", "", 4);
      add_child(mean, "kind", "power", 5);
      add_child(mean, "power", "not-a-double", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *mean       = add_child(item, "mean", "", 4);
      add_child(mean, "kind", "arithmetic", 5);
      add_child(mean, "bogus_child", "1", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "transform", "not-a-transform", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *transform  = add_child(item, "transform", "", 4);
      add_child(transform, "kind", "raw", 5);
      add_child(transform, "baseline", "not-a-baseline", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *transform  = add_child(item, "transform", "", 4);
      add_child(transform, "kind", "raw", 5);
      add_child(transform, "amortization_window", "0", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *transform  = add_child(item, "transform", "", 4);
      add_child(transform, "kind", "raw", 5);
      add_child(transform, "extra_key", "1", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *history    = add_child(item, "history", "", 4);
      add_child(history, "source", "not-a-source", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *history    = add_child(item, "history", "", 4);
      add_child(history, "source", "active_level", 5);
      add_child(history, "level", "-2", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *history    = add_child(item, "history", "", 4);
      add_child(history, "source", "linear_solves", 5);
      add_child(history, "max_points", "0", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *history    = add_child(item, "history", "", 4);
      add_child(history, "source", "linear_solves", 5);
      add_child(history, "reduction", "not-a-reduction", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      YAMLnode *history    = add_child(item, "history", "", 4);
      add_child(history, "source", "linear_solves", 5);
      add_child(history, "typo_key", "1", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }
}

static void
test_PreconReuseSetArgsFromYAML_parse_component_guards_adaptive_errors(void)
{
   PreconReuse_args args;
   hypredrv_PreconReuseSetDefaultArgs(&args);

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "metric", "not-a-metric", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "enabled", "maybe", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "scale", "0", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "metric", "solve_overhead_vs_setup", 4);
      YAMLnode *transform  = add_child(item, "transform", "", 4);
      add_child(transform, "kind", "raw", 5);
      add_child(transform, "amortization_window", "0", 5);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);
      YAMLnode *item       = add_child(components, "-", "", 3);
      add_child(item, "not_a_component_key", "1", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      add_child(guards, "min_reuse_solves", "-1", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      add_child(guards, "bad_decisions_to_rebuild", "0", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      add_child(guards, "rebuild_on_new_timestep", "maybe", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      add_child(guards, "bogus_guard", "1", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive   = add_child(parent, "adaptive", "", 1);
      YAMLnode *components = add_child(adaptive, "components", "", 2);

      (void)components;

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *adaptive = add_child(parent, "adaptive", "", 1);
      add_child(adaptive, "unknown_adaptive_key", "1", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      YAMLnode *levels = add_child(guards, "rebuild_on_new_level", "", 2);
      YAMLnode *item   = add_child(levels, "-", "", 3);
      add_child(item, "level", "not-int", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
      hypredrv_ErrorCodeResetAll();
   }

   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("reuse", "", 0);
      add_child(parent, "enabled", "yes", 1);
      add_child(parent, "type", "adaptive", 1);
      YAMLnode *guards = add_child(parent, "guards", "", 1);
      add_child(guards, "rebuild_on_new_level", "[0, 1]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_PreconReuseSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(args.guards.rebuild_on_new_level);
      ASSERT_EQ_SIZE(args.guards.rebuild_on_new_level->size, 2u);

      hypredrv_PreconReuseDestroyArgs(&args);
      hypredrv_YAMLnodeDestroy(parent);
   }
}

static void
test_PreconReuseShouldRebuild_static_frequency_and_linear_system_ids(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled  = 1;
   args.policy   = PRECON_REUSE_POLICY_STATIC;
   args.frequency = 2;
   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5, &decision), 0);
   ASSERT_EQ(decision.should_rebuild, 0);
   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 6, &decision), 1);
   ASSERT_EQ(decision.should_rebuild, 1);

   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, (int[]){ 1, 3 }, &args.linear_system_ids);
   ASSERT_NOT_NULL(args.linear_system_ids);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 2, &decision), 0);
   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 1, &decision), 1);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_static_per_timestep_with_starts(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   IntArray           *starts = NULL;
   int                 data[2] = { 0, 10 };

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled      = 1;
   args.policy       = PRECON_REUSE_POLICY_STATIC;
   args.per_timestep = 1;
   args.frequency    = 2;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, data, &starts);
   ASSERT_NOT_NULL(starts);

   hypredrv_PreconReuseStateInit(&state);

   /* Timestep index 1 with period 2 -> no rebuild on first solve of that timestep. */
   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, starts, NULL, &state, 10, &decision),
             0);
   /* Timestep index 0 matches (0 % 2) == 0. */
   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, starts, NULL, &state, 0, &decision),
             1);

   hypredrv_IntArrayDestroy(&starts);
   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_static_per_timestep_missing_context_returns_hold(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled      = 1;
   args.policy       = PRECON_REUSE_POLICY_STATIC;
   args.per_timestep = 1;
   args.frequency    = 1;
   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5, &decision),
             0);
   ASSERT_EQ(decision.should_rebuild, 0);

   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_static_disabled_always_rebuilds(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled   = 0;
   args.policy    = PRECON_REUSE_POLICY_STATIC;
   args.frequency = 5;
   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 7, &decision), 1);
   ASSERT_EQ(decision.should_rebuild, 1);

   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_adaptive_null_state_requests_rebuild(void)
{
   PreconReuse_args    args;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled = 1;
   args.policy  = PRECON_REUSE_POLICY_ADAPTIVE;

   /* Adaptive early exit when state is NULL returns 1 without writing decision fields. */
   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, NULL, 1, &decision), 1);

   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_adaptive_initial_build_next_ls_id_nonpositive(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled = 1;
   args.policy  = PRECON_REUSE_POLICY_ADAPTIVE;

   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 0, &decision), 1);
   ASSERT_TRUE(strstr(decision.summary, "initial build") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_adaptive_status_good_and_insufficient_history(void)
{
   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 1.0e9;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "low-score");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 1.0e9;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                     &decision));
      ASSERT_TRUE(strstr(decision.summary, "status=good") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 50;
      args.adaptive.rebuild_threshold = 0.0;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "need-more-points");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 0.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                     &decision));
      ASSERT_TRUE(strstr(decision.summary, "history=insufficient") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }
}

static void
test_PreconReuseShouldRebuild_adaptive_relative_increase_total_time_below_direction(void)
{
   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "rel-inc");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 1.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RELATIVE_INCREASE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                    &decision));
      ASSERT_TRUE(strstr(decision.summary, "rel-inc") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "total-time");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_TOTAL_TIME;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 1.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 3.0, 4.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                    &decision));
      ASSERT_TRUE(strstr(decision.summary, "total-time") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "below-dir");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_BELOW;
      /* Keep aggregate near baseline ratio 1.0 so (target-aggregate) <= 0 for direction=below. */
      c->target             = 1.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 10, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                     &decision));
      ASSERT_TRUE(strstr(decision.summary, "below-dir") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;
      Stats              *stats = hypredrv_StatsCreate();

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "active-miss");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 0.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_ACTIVE_LEVEL;
      c->history.level      = 0;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      /* No active level in stats -> sample_count == 0 -> insufficient history. */
      ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 5,
                                                     &decision));
      ASSERT_TRUE(strstr(decision.summary, "history=insufficient") != NULL);

      hypredrv_StatsDestroy(&stats);
      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }
}

static void
test_PreconReuseShouldRebuild_adaptive_power_mean_negative_p_and_summary_truncation(void)
{
   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.adaptive.rebuild_threshold = 0.5;
      args.adaptive.num_components    = 1;
      args.adaptive.components =
         (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);
      PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
      snprintf(c->name, sizeof(c->name), "%s", "power-neg");
      c->enabled            = 1;
      c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
      c->weight             = 1.0;
      c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
      c->target             = 1.0;
      c->scale              = 1.0;
      c->mean.kind          = PRECON_REUSE_MEAN_POWER;
      c->mean.power         = -1.0;
      c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
      c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
      c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
      c->history.max_points = 2;

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
      obs.system_index = 4;
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 5,
                                                    &decision));
      ASSERT_TRUE(strstr(decision.summary, "power-neg") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }

   {
      PreconReuse_args    args;
      PreconReuseState    state;
      PreconReuseDecision decision;

      hypredrv_PreconReuseSetDefaultArgs(&args);
      args.enabled                    = 1;
      args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
      args.guards.min_history_points  = 1;
      args.guards.bad_decisions_to_rebuild = 1;
      args.adaptive.rebuild_threshold = 0.1;
      args.adaptive.num_components    = 24;
      args.adaptive.components = (PreconReuseScoreComponent_args *)calloc(
         args.adaptive.num_components, sizeof(PreconReuseScoreComponent_args));
      ASSERT_NOT_NULL(args.adaptive.components);

      for (size_t i = 0; i < args.adaptive.num_components; i++)
      {
         PreconReuseScoreComponent_args *c = &args.adaptive.components[i];
         snprintf(c->name, sizeof(c->name), "long_named_component_%04zu_extra", i);
         c->enabled            = 1;
         c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
         c->weight             = 0.01;
         c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
         c->target             = 1.0;
         c->scale              = 1.0;
         c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
         c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
         c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
         c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
         c->history.max_points = 1;
      }

      hypredrv_PreconReuseStateInit(&state);
      seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

      PreconReuseObservation obs = make_reuse_observation(3, 0, 50, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);

      ASSERT_TRUE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4,
                                                    &decision));
      ASSERT_TRUE(strstr(decision.summary, "...") != NULL);

      hypredrv_PreconReuseStateDestroy(&state);
      hypredrv_PreconReuseDestroyArgs(&args);
   }
}

static void
test_PreconReuseStateRecordObservation_realloc_and_cap_drop_oldest(void)
{
   PreconReuseState state;
   hypredrv_PreconReuseStateInit(&state);

   for (int i = 0; i < 256; i++)
   {
      PreconReuseObservation obs = make_reuse_observation(i, 0, 1, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &obs);
   }
   ASSERT_EQ_SIZE(state.count, 256u);
   ASSERT_TRUE(state.capacity >= 256u);

   {
      PreconReuseObservation extra = make_reuse_observation(999, 0, 1, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &extra);
   }

   ASSERT_EQ_SIZE(state.count, 256u);
   ASSERT_EQ(state.observations[0].system_index, 1);

   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconReuseShouldRebuild_adaptive_window_mean_baseline_warns_and_scores(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                    = 1;
   args.policy                     = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points  = 1;
   args.adaptive.rebuild_threshold = 0.5;
   args.adaptive.num_components    = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);
   PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
   snprintf(c->name, sizeof(c->name), "%s", "window-mean");
   c->enabled            = 1;
   c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   c->weight             = 1.0;
   c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   c->target             = 1.0;
   c->scale              = 10.0;
   c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   c->transform.baseline = PRECON_REUSE_BASELINE_WINDOW_MEAN;
   c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   c->history.max_points = 200;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   /* Oldest window (first 128 samples for baseline) stays low; recent history is high so
    * ratio_to_baseline triggers a rebuild while exercising window_mean > 128. */
   for (int i = 0; i < 128; i++)
   {
      PreconReuseObservation o = make_reuse_observation(3 + i, 0, 10, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &o);
   }
   for (int i = 0; i < 23; i++)
   {
      PreconReuseObservation o = make_reuse_observation(200 + i, 0, 100, 1.0, 1.0);
      hypredrv_PreconReuseStateRecordObservation(&state, &o);
   }

   PreconReuseObservation obs = make_reuse_observation(500, 0, 100, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   (void)hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 501, &decision);
   ASSERT_TRUE(strstr(decision.summary, "window-mean") != NULL);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_duplicate_ls_id_skips_bad_streak_increment(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled                         = 1;
   args.policy                          = PRECON_REUSE_POLICY_ADAPTIVE;
   args.guards.min_history_points       = 1;
   args.guards.bad_decisions_to_rebuild = 2;
   args.adaptive.rebuild_threshold      = 0.5;
   args.adaptive.num_components         = 1;
   args.adaptive.components =
      (PreconReuseScoreComponent_args *)calloc(1, sizeof(PreconReuseScoreComponent_args));
   ASSERT_NOT_NULL(args.adaptive.components);

   PreconReuseScoreComponent_args *c = &args.adaptive.components[0];
   snprintf(c->name, sizeof(c->name), "%s", "streak-dup");
   c->enabled            = 1;
   c->metric             = PRECON_REUSE_METRIC_ITERATIONS;
   c->weight             = 1.0;
   c->direction          = PRECON_REUSE_DIRECTION_ABOVE;
   c->target             = 1.0;
   c->scale              = 0.25;
   c->mean.kind          = PRECON_REUSE_MEAN_ARITHMETIC;
   c->transform.kind     = PRECON_REUSE_TRANSFORM_RATIO_TO_BASELINE;
   c->transform.baseline = PRECON_REUSE_BASELINE_REBUILD;
   c->history.source     = PRECON_REUSE_HISTORY_LINEAR_SOLVES;
   c->history.max_points = 1;

   hypredrv_PreconReuseStateInit(&state);
   seed_post_bootstrap_state(&state, 10, 1.0, 1.0);

   PreconReuseObservation obs = make_reuse_observation(3, 0, 30, 1.0, 1.0);
   hypredrv_PreconReuseStateRecordObservation(&state, &obs);

   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4, &decision));
   ASSERT_EQ(state.bad_decision_streak, 1);
   ASSERT_FALSE(hypredrv_PreconReuseShouldRebuild(&args, NULL, NULL, &state, 4, &decision));
   ASSERT_EQ(state.bad_decision_streak, 1);

   hypredrv_PreconReuseStateDestroy(&state);
   hypredrv_PreconReuseDestroyArgs(&args);
}

static void
test_PreconReuseShouldRebuild_static_per_timestep_embedded_stats(void)
{
   PreconReuse_args    args;
   PreconReuseState    state;
   PreconReuseDecision decision;
   Stats              *stats = hypredrv_StatsCreate();

   hypredrv_PreconReuseSetDefaultArgs(&args);
   args.enabled      = 1;
   args.policy       = PRECON_REUSE_POLICY_STATIC;
   args.per_timestep = 1;
   args.frequency    = 1;

   stats->level_active           |= (1u << 0);
   stats->level_solve_start[0]    = 5;
   stats->level_current_id[0]     = 2;

   hypredrv_PreconReuseStateInit(&state);

   ASSERT_EQ(hypredrv_PreconReuseShouldRebuild(&args, NULL, stats, &state, 5, &decision),
             1);

   hypredrv_StatsDestroy(&stats);
   hypredrv_PreconReuseStateDestroy(&state);
}

static void
test_PreconDestroy_null_precon(void)
{
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   hypredrv_LogInitializeFromEnv();

   HYPRE_Precon precon = NULL;
   precon_args  args;
   hypredrv_PreconSetDefaultArgs(&args);

   /* hypredrv_PreconDestroy with NULL should not crash */
   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();
}

static void
test_PreconDestroy_null_main(void)
{
   HYPRE_Precon precon = malloc(sizeof(struct hypre_Precon_struct));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);

   /* hypredrv_PreconDestroy with null main should not crash */
   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
}

static void
test_PreconSetup_default_case(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat = precon_test_ij_matrix_1x1(1.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_INVALID, precon, mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_PRECON);

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconApply_default_case(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(1.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_BOOMERAMG, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_INVALID, precon, mat, vec_b, vec_x);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_PRECON);

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconApply_precon_none(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_NONE, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(1.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_NONE, precon, mat, vec_b, vec_x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_PreconDestroy(PRECON_NONE, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
test_PreconApply_mgr_minimal(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);
   args.mgr.num_levels = 1;

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(1.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_MGR, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_MGR, precon, mat, vec_b, vec_x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

static void
test_PreconDestroy_amg_log_dispatch_no_rbms(void)
{
   TEST_HYPRE_INIT();

   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   hypredrv_LogInitializeFromEnv();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);
   args.amg.max_iter = 1;
   args.amg.num_rbms = 0;

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconCreate_invalid_method(void)
{
   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_INVALID, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NULL(precon);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_PRECON);
}

static void
test_PreconCreate_precon_none(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_NONE, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   ASSERT_NULL(precon->main);

   hypredrv_PreconDestroy(PRECON_NONE, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   TEST_HYPRE_FINALIZE();
}

static void
test_PreconSetup_null_precon(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix mat = precon_test_ij_matrix_1x1(1.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_BOOMERAMG, NULL, mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_PRECON);

   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconSetup_null_A(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_BOOMERAMG, precon, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   TEST_HYPRE_FINALIZE();
}

static void
test_Precon_lifecycle_boomeramg_1x1(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);
   args.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(4.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_BOOMERAMG, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_BOOMERAMG, precon, mat, vec_b, vec_x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
test_Precon_lifecycle_ilu_1x1(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_ILUSetDefaultArgs(&args.ilu);
   args.ilu.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_ILU, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(4.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_ILU, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_ILU, precon, mat, vec_b, vec_x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_PreconDestroy(PRECON_ILU, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}
#endif

#if HYPRE_CHECK_MIN_VERSION(22500, 0)
static void
test_Precon_lifecycle_fsai_1x1(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_FSAISetDefaultArgs(&args.fsai);

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_FSAI, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat    = precon_test_ij_matrix_1x1(4.0);
   HYPRE_IJVector vec_b = precon_test_ij_vector_1x1(1.0);
   HYPRE_IJVector vec_x = precon_test_ij_vector_1x1(0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_FSAI, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconApply(PRECON_FSAI, precon, mat, vec_b, vec_x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_PreconDestroy(PRECON_FSAI, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}
#endif

static void
test_PreconDestroy_amg_rbms_loop(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_AMGSetDefaultArgs(&args.amg);
   args.amg.max_iter = 1;

   HYPRE_BigInt    part[2] = {0, 1};
   HYPRE_ParVector pv      = NULL;
   ASSERT_EQ(HYPRE_ParVectorCreate(MPI_COMM_SELF, 1, part, &pv), 0);
   ASSERT_EQ(HYPRE_ParVectorInitialize(pv), 0);

   args.amg.num_rbms = 1;
   args.amg.rbms[0]  = pv;

   HYPRE_Precon precon = NULL;
   hypredrv_PreconCreate(PRECON_BOOMERAMG, &args, NULL, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_PreconDestroy(PRECON_BOOMERAMG, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
   ASSERT_NULL(args.amg.rbms[0]);

   TEST_HYPRE_FINALIZE();
}

static void
test_PreconDestroy_none_with_main_logs(void)
{
   TEST_HYPRE_INIT();

   static char precon_none_dummy_main_storage;
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   hypredrv_LogInitializeFromEnv();

   HYPRE_Precon precon = malloc(sizeof(struct hypre_Precon_struct));
   ASSERT_NOT_NULL(precon);
   precon->main  = (HYPRE_Solver)&precon_none_dummy_main_storage;
   precon->stats = NULL;

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);

   hypredrv_PreconDestroy(PRECON_NONE, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();

   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_coarsest_level_branches(void)
{
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1; /* minimal valid MGR setup (num_levels-1 == 0) */

   /* Minimal dofmap with global unique size set */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   fprintf(stderr, "SKIP: MGR tests require hypre >= 2.21.0\n");
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
   return;
#endif

   /* 1) Explicit ILU coarsest solver type */
   mgr.coarsest_level.type = 32;
   hypredrv_ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   /* Clean up coarsest solver (explicit ILU) */
   if (mgr.csolver)
   {
 #if HYPRE_CHECK_MIN_VERSION(21900, 0)
      HYPRE_ILUDestroy(mgr.csolver);
 #endif
      mgr.csolver = NULL;
      mgr.csolver_type = -1;
   }

   /* 2) Infer AMG when type == -1 */
   mgr.coarsest_level.type = -1;
   hypredrv_ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   precon                      = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   /* Clean up coarsest solver (inferred as AMG) */
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
      mgr.csolver_type = -1;
   }

   /* 3) Explicit AMG coarsest solver type */
   mgr.coarsest_level.type = 0;
   hypredrv_AMGSetDefaultArgs(&mgr.coarsest_level.amg);
   precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   /* Clean up coarsest solver (explicit AMG) */
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
      mgr.csolver_type = -1;
   }

   /* 4) Explicit ILU coarsest solver type (repeat to cover reuse) */
   mgr.coarsest_level.type = 32;
   hypredrv_ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
 #if HYPRE_CHECK_MIN_VERSION(21900, 0)
      HYPRE_ILUDestroy(mgr.csolver);
 #endif
      mgr.csolver = NULL;
      mgr.csolver_type = -1;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconCreate_mgr_coarsest_level_krylov_nested(void)
{
#if !HYPRE_CHECK_MIN_VERSION(30100, 2)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   args.mgr.num_levels = 1; /* minimal valid MGR setup (num_levels-1 == 0) */

   /* Minimal dofmap required by MGR */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.coarsest_level.use_krylov = 1;
   args.mgr.coarsest_level.krylov =
      (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
   ASSERT_NOT_NULL(args.mgr.coarsest_level.krylov);
   hypredrv_NestedKrylovSetDefaultArgs(args.mgr.coarsest_level.krylov);
   args.mgr.coarsest_level.krylov->is_set = 1;
   args.mgr.coarsest_level.krylov->solver_method = SOLVER_GMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_GMRES, &args.mgr.coarsest_level.krylov->solver);
   args.mgr.coarsest_level.krylov->solver.gmres.max_iter = 2;
   args.mgr.coarsest_level.krylov->has_precon = 1;
   args.mgr.coarsest_level.krylov->precon_method = PRECON_BOOMERAMG;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG,
                                  &args.mgr.coarsest_level.krylov->precon);
   args.mgr.coarsest_level.krylov->precon.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
test_PreconDestroy_mgr_coarsest_use_krylov_without_krylov_ptr(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);
   args.mgr.num_levels = 1;

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   /* use_krylov true, krylov NULL: short-circuit branch of && in PreconDestroyMGRSolver */
   args.mgr.coarsest_level.use_krylov = 1;
   args.mgr.coarsest_level.krylov     = NULL;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

static void
test_PreconSetup_mgr_frelax_nested_mgr_dof_labels(void)
{
#if !HYPRE_CHECK_MIN_VERSION(30100, 5)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   args.mgr.num_levels = 2; /* one MGR level + coarsest */
   args.mgr.level[0].f_dofs.size = 2;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_dofs.data[1] = 2; /* non-contiguous labels to force projection */
   args.mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
   args.mgr.level[0].f_relaxation.mgr = (MGR_args *)malloc(sizeof(MGR_args));
   ASSERT_NOT_NULL(args.mgr.level[0].f_relaxation.mgr);
   hypredrv_MGRSetDefaultArgs(args.mgr.level[0].f_relaxation.mgr);

   MGR_args *inner = args.mgr.level[0].f_relaxation.mgr;
   inner->num_levels = 2; /* one inner level + coarsest */
   inner->level[0].f_dofs.size = 1;
   inner->level[0].f_dofs.data[0] = 2; /* preserved parent label (no relabeling) */
   inner->level[0].g_relaxation.type = -1;
   inner->level[0].f_relaxation.type = 7;

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 2, 0, 2, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   for (int row = 0; row < 3; row++)
   {
      HYPRE_Int    ncols = 1;
      HYPRE_BigInt irow  = row;
      HYPRE_BigInt col   = row;
      HYPRE_Real   val   = 1.0;
      HYPRE_IJMatrixSetValues(mat, 1, &ncols, &irow, &col, &val);
   }
   HYPRE_IJMatrixAssemble(mat);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_MGR, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJMatrixDestroy(mat);
   hypredrv_IntArrayDestroy(&dofmap);
   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconSetup_mgr_frelax_nested_mgr_body_split_labels(void)
{
#if !HYPRE_CHECK_MIN_VERSION(23100, 9)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   args.mgr.num_levels = 2; /* one MGR level + coarsest */
   args.mgr.level[0].f_dofs.size    = 6;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_dofs.data[1] = 1;
   args.mgr.level[0].f_dofs.data[2] = 2;
   args.mgr.level[0].f_dofs.data[3] = 3;
   args.mgr.level[0].f_dofs.data[4] = 4;
   args.mgr.level[0].f_dofs.data[5] = 5;
   args.mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
   args.mgr.level[0].f_relaxation.mgr = (MGR_args *)malloc(sizeof(MGR_args));
   ASSERT_NOT_NULL(args.mgr.level[0].f_relaxation.mgr);
   hypredrv_MGRSetDefaultArgs(args.mgr.level[0].f_relaxation.mgr);

   MGR_args *inner = args.mgr.level[0].f_relaxation.mgr;
   inner->num_levels = 2; /* one inner level + coarsest */
   inner->level[0].f_dofs.size    = 3;
   inner->level[0].f_dofs.data[0] = 0;
   inner->level[0].f_dofs.data[1] = 1;
   inner->level[0].f_dofs.data[2] = 2;
   inner->level[0].g_relaxation.type = -1;
   inner->level[0].f_relaxation.type = 7;

   IntArray *dofmap = NULL;
   const int map[8] = {0, 1, 2, 3, 4, 5, 6, 7};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 8, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 7, 0, 7, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   for (int row = 0; row < 8; row++)
   {
      HYPRE_Int    ncols = 1;
      HYPRE_BigInt irow  = row;
      HYPRE_BigInt col   = row;
      HYPRE_Real   val   = 1.0;
      HYPRE_IJMatrixSetValues(mat, 1, &ncols, &irow, &col, &val);
   }
   HYPRE_IJMatrixAssemble(mat);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_MGR, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJMatrixDestroy(mat);
   hypredrv_IntArrayDestroy(&dofmap);
   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconCreate_mgr_nested_krylov_inner_mgr_recreate_without_reuse(void)
{
#if !HYPRE_CHECK_MIN_VERSION(23100, 9)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 6;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_dofs.data[1] = 1;
   args.mgr.level[0].f_dofs.data[2] = 2;
   args.mgr.level[0].f_dofs.data[3] = 3;
   args.mgr.level[0].f_dofs.data[4] = 4;
   args.mgr.level[0].f_dofs.data[5] = 5;
   args.mgr.level[0].f_relaxation.use_krylov = 1;
   args.mgr.level[0].f_relaxation.krylov =
      (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
   ASSERT_NOT_NULL(args.mgr.level[0].f_relaxation.krylov);
   hypredrv_NestedKrylovSetDefaultArgs(args.mgr.level[0].f_relaxation.krylov);

   NestedKrylov_args *nested = args.mgr.level[0].f_relaxation.krylov;
   nested->is_set            = 1;
   nested->solver_method     = SOLVER_FGMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_FGMRES, &nested->solver);
   nested->solver.fgmres.max_iter   = 1;
   nested->solver.fgmres.krylov_dim = 1;
   nested->has_precon               = 1;
   nested->precon_method            = PRECON_MGR;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &nested->precon);

   MGR_args *inner = &nested->precon.mgr;
   inner->num_levels = 2;
   inner->level[0].f_dofs.size    = 3;
   inner->level[0].f_dofs.data[0] = 0;
   inner->level[0].f_dofs.data[1] = 1;
   inner->level[0].f_dofs.data[2] = 2;
   inner->level[0].f_relaxation.type = 2;
   inner->level[0].f_relaxation.amg.max_iter = 1;
   inner->level[0].g_relaxation.type = -1;
   inner->coarsest_level.type = 0;
   inner->coarsest_level.amg.max_iter = 1;

   args.mgr.coarsest_level.type = 0;
   args.mgr.coarsest_level.amg.max_iter = 1;

   IntArray *dofmap = NULL;
   const int map[8] = {0, 1, 2, 3, 4, 5, 6, 7};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 8, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   ASSERT_NOT_NULL(inner->frelax[0]);
   ASSERT_NOT_NULL(inner->csolver);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconDestroy_mgr_csolver_destroy_branches(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   /* Minimal dofmap required by MGR */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 1;

   HYPRE_Precon precon = NULL;

   /* Coarsest explicit AMG (csolver_type == 0) */
   args.mgr.coarsest_level.type = 0;
   hypredrv_AMGSetDefaultArgs(&args.mgr.coarsest_level.amg);
   args.mgr.coarsest_level.amg.max_iter = 1;

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   /* Coarsest level ILU -> expect hypredrv_PreconDestroy to hit HYPRE_ILUDestroy(args.mgr.csolver) */
   args.mgr.coarsest_level.type = 32;
   hypredrv_ILUSetDefaultArgs(&args.mgr.coarsest_level.ilu);
   args.mgr.coarsest_level.ilu.max_iter = 1;

   precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

#if defined(HYPRE_USING_DSUPERLU)
   /* Coarsest level direct solver -> cover HYPRE_MGRDirectSolverDestroy branch when available */
   args.mgr.coarsest_level.type = 29;
   precon                      = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
#endif

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRComponentReuseSetupMode_policy_shape_and_selector_paths(void)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   return;
#else
   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels                  = 2;
   mgr.num_active_levels           = 1;
   mgr.active_level_map[0]         = 0;
   mgr.level[0].f_relaxation.type  = 2;
   mgr.level[0].g_relaxation.type  = -1;
   mgr.coarsest_level.type         = 0;

   precon_test_set_static_mgr_component_reuse(&mgr.level[0].f_relaxation.reuse, 1);
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 1);

   mgr.level[0].f_relaxation.reuse.args.policy = PRECON_REUSE_POLICY_ADAPTIVE;
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 0);
   ASSERT_EQ(mgr.level[0].f_relaxation.reuse.warned_policy_unsupported, 1);

   mgr.level[0].f_relaxation.reuse.args.policy = PRECON_REUSE_POLICY_STATIC;
   mgr.level[0].f_relaxation.type              = 7;
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 0);
   ASSERT_EQ(mgr.level[0].f_relaxation.reuse.warned_type_unsupported, 1);

   mgr.level[0].f_relaxation.reuse.present = 0;
   mgr.level[0].g_relaxation.type          = 16;
   precon_test_set_static_mgr_component_reuse(&mgr.level[0].g_relaxation.reuse, 1);
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 1);

   mgr.level[0].g_relaxation.reuse.present = 0;
   mgr.coarsest_level.type                 = 7;
   precon_test_set_static_mgr_component_reuse(&mgr.coarsest_level.reuse, 1);
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 0);
   ASSERT_EQ(mgr.coarsest_level.reuse.warned_type_unsupported, 1);

   mgr.coarsest_level.type = 32;
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 1);

   mgr.coarsest_level.reuse.present = 0;
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 0);
#endif
}

static void
test_MGRComponentReuseSetupMode_nested_shape_unsupported(void)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   return;
#else
   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels                  = 2;
   mgr.num_active_levels           = 1;
   mgr.active_level_map[0]         = 0;
   mgr.level[0].f_relaxation.type  = MGR_FRLX_TYPE_NESTED_MGR;

   precon_test_set_static_mgr_component_reuse(&mgr.level[0].f_relaxation.reuse, 1);
   ASSERT_EQ(hypredrv_MGRComponentReuseSetupMode(&mgr, NULL, 1), 0);
   ASSERT_EQ(mgr.level[0].f_relaxation.reuse.warned_type_unsupported, 1);

   mgr.keep_frelax[0] = 1;
   mgr.keep_grelax[0] = 1;
   mgr.keep_csolver   = 1;
   hypredrv_MGRSelectCachedSolversToKeep(&mgr, NULL, NULL, 1);
   ASSERT_EQ(mgr.keep_frelax[0], 0);
   ASSERT_EQ(mgr.keep_grelax[0], 0);
   ASSERT_EQ(mgr.keep_csolver, 0);
#endif
}

static void
test_MGRComponentReuseShouldKeepOuter_and_SelectKeepFlags(void)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   return;
#else
   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels                 = 2;
   mgr.num_active_levels          = 1;
   mgr.active_level_map[0]        = 0;
   mgr.level[0].f_relaxation.type = 2;
   mgr.level[0].g_relaxation.type = 16;
   mgr.coarsest_level.type        = 32;

   precon_test_set_static_mgr_component_reuse(&mgr.level[0].f_relaxation.reuse, 1);
   precon_test_set_static_mgr_component_reuse(&mgr.level[0].g_relaxation.reuse, 1);
   precon_test_set_static_mgr_component_reuse(&mgr.coarsest_level.reuse, 1);

   ASSERT_EQ(hypredrv_MGRComponentReuseShouldKeepOuter(&mgr, NULL, NULL, 1), 1);
   ASSERT_EQ(hypredrv_MGRComponentReuseShouldKeepOuter(&mgr, NULL, NULL, 2), 0);

   hypredrv_MGRSelectCachedSolversToKeep(&mgr, NULL, NULL, 1);
   ASSERT_EQ(mgr.keep_frelax[0], 1);
   ASSERT_EQ(mgr.keep_grelax[0], 1);
   ASSERT_EQ(mgr.keep_csolver, 1);

   int keep_f = -1, keep_g = -1, keep_c = -1;
   hypredrv_MGRCountKeepFlags(&mgr, &keep_f, &keep_g, &keep_c);
   ASSERT_EQ(keep_f, 1);
   ASSERT_EQ(keep_g, 1);
   ASSERT_EQ(keep_c, 1);

   hypredrv_MGRSelectCachedSolversToKeep(&mgr, NULL, NULL, 2);
   ASSERT_EQ(mgr.keep_frelax[0], 0);
   ASSERT_EQ(mgr.keep_grelax[0], 0);
   ASSERT_EQ(mgr.keep_csolver, 0);
#endif
}

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
static void
test_MGRRefreshComponentsForSetup_rebuilds_fsai_handles(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
   return;
#else
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels                   = 2;
   args.mgr.level[0].f_dofs.size         = 1;
   args.mgr.level[0].f_dofs.data[0]      = 0;
   args.mgr.level[0].f_relaxation.type   = 33;
   args.mgr.level[0].g_relaxation.type   = 33;
   args.mgr.coarsest_level.type          = 32;
   hypredrv_FSAISetDefaultArgs(&args.mgr.level[0].f_relaxation.fsai);
   hypredrv_FSAISetDefaultArgs(&args.mgr.level[0].g_relaxation.fsai);
   hypredrv_ILUSetDefaultArgs(&args.mgr.coarsest_level.ilu);
   args.mgr.level[0].f_relaxation.fsai.max_iter = 1;
   args.mgr.level[0].g_relaxation.fsai.max_iter = 1;
   args.mgr.coarsest_level.ilu.max_iter         = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   ASSERT_NOT_NULL(args.mgr.frelax[0]);
   ASSERT_NOT_NULL(args.mgr.grelax[0]);
   ASSERT_NOT_NULL(args.mgr.csolver);

   precon_test_set_static_mgr_component_reuse(&args.mgr.level[0].f_relaxation.reuse, 0);
   precon_test_set_static_mgr_component_reuse(&args.mgr.level[0].g_relaxation.reuse, 0);
   precon_test_set_static_mgr_component_reuse(&args.mgr.coarsest_level.reuse, 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRRefreshComponentsForSetup(&args.mgr, precon->main, NULL, NULL, 1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args.mgr.frelax[0]);
   ASSERT_NOT_NULL(args.mgr.grelax[0]);
   ASSERT_NOT_NULL(args.mgr.csolver);

   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
#endif
}
#endif

#if HYPRE_CHECK_MIN_VERSION(22600, 0)
static void
test_PreconDestroy_mgr_frelax_amg_type2(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 2;
   hypredrv_AMGSetDefaultArgs(&args.mgr.level[0].f_relaxation.amg);
   args.mgr.level[0].f_relaxation.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
static void
test_PreconDestroy_mgr_grelax_krylov_nested(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 7;

   args.mgr.level[0].g_relaxation.use_krylov = 1;
   args.mgr.level[0].g_relaxation.krylov =
      (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
   ASSERT_NOT_NULL(args.mgr.level[0].g_relaxation.krylov);
   hypredrv_NestedKrylovSetDefaultArgs(args.mgr.level[0].g_relaxation.krylov);
   args.mgr.level[0].g_relaxation.krylov->is_set = 1;
   args.mgr.level[0].g_relaxation.krylov->solver_method = SOLVER_GMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_GMRES,
                                           &args.mgr.level[0].g_relaxation.krylov->solver);
   args.mgr.level[0].g_relaxation.krylov->solver.gmres.max_iter = 2;
   args.mgr.level[0].g_relaxation.krylov->has_precon = 1;
   args.mgr.level[0].g_relaxation.krylov->precon_method = PRECON_BOOMERAMG;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG,
                                           &args.mgr.level[0].g_relaxation.krylov->precon);
   args.mgr.level[0].g_relaxation.krylov->precon.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
/* IntArray with only data/size (no unique / g_unique): exercises mgr.c plain-dofmap branch */
static IntArray *
precon_test_intarray_plain_data_only(int n, const int *vals)
{
   IntArray *a = hypredrv_IntArrayCreate((size_t)n);
   memcpy(a->data, vals, (size_t)n * sizeof(int));
   return a;
}

static void
test_MGRCreate_missing_dofmap(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;
   mgr.dofmap     = NULL;

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_plain_dofmap_data_only_branch(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;

   const int      vals[3] = {0, 1, 2};
   IntArray *dofmap = precon_test_intarray_plain_data_only(3, vals);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_unique_data_dofmap_branch(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;

   IntArray *dofmap = hypredrv_IntArrayCreate(3);
   dofmap->data[0] = 0;
   dofmap->data[1] = 1;
   dofmap->data[2] = 2;
   dofmap->unique_size = 3;
   dofmap->unique_data   = (int *)malloc(3U * sizeof(int));
   dofmap->unique_data[0] = 0;
   dofmap->unique_data[1] = 1;
   dofmap->unique_data[2] = 2;

   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_g_unique_unsorted_dense_fallback(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;

   IntArray *dofmap = hypredrv_IntArrayCreate(3);
   dofmap->data[0]  = 0;
   dofmap->data[1]  = 1;
   dofmap->data[2]  = 2;
   dofmap->g_unique_size = 3;
   dofmap->g_unique_data = (int *)malloc(3U * sizeof(int));
   ASSERT_NOT_NULL(dofmap->g_unique_data);
   /* Not strictly increasing -> first branch clears labels -> dense g_unique_size path */
   dofmap->g_unique_data[0] = 0;
   dofmap->g_unique_data[1] = 2;
   dofmap->g_unique_data[2] = 1;

   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_prefers_local_dof_labels_over_dense_global_fallback(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size      = 1;
   mgr.level[0].f_dofs.data[0]   = 3;
   mgr.level[0].f_relaxation.type = 7;

   IntArray *dofmap = hypredrv_IntArrayCreate(4);
   ASSERT_NOT_NULL(dofmap);
   dofmap->data[0] = 0;
   dofmap->data[1] = 3;
   dofmap->data[2] = 3;
   dofmap->data[3] = 0;
   /* Simulate distributed metadata that only reports a coarse global count.
    * MGRCreate should prefer the explicit local labels over this dense fallback. */
   dofmap->g_unique_size = 2;

   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_compact_global_labels_ignore_missing_f_dofs(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size      = 1;
   mgr.level[0].f_dofs.data[0]   = 3;
   mgr.level[0].f_relaxation.type = 7;

   IntArray *dofmap = hypredrv_IntArrayCreate(0);
   ASSERT_NOT_NULL(dofmap);
   dofmap->g_unique_size = 3;

   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_global_unique_metadata_ignore_missing_f_dofs(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size       = 1;
   mgr.level[0].f_dofs.data[0]    = 3;
   mgr.level[0].f_relaxation.type = 7;

   IntArray *dofmap = hypredrv_IntArrayCreate(6);
   ASSERT_NOT_NULL(dofmap);
   dofmap->data[0]      = 0;
   dofmap->data[1]      = 1;
   dofmap->data[2]      = 2;
   dofmap->data[3]      = 0;
   dofmap->data[4]      = 1;
   dofmap->data[5]      = 2;
   dofmap->g_unique_size = 3;

   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_dense_remap_sparse_labels(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size      = 1;
   mgr.level[0].f_dofs.data[0]   = 0;
   mgr.level[0].f_relaxation.type = 7;

   const int      vals[4] = {0, 3, 3, 0};
   IntArray *dofmap = precon_test_intarray_plain_data_only(4, vals);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_f_dofs_out_of_range(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size    = 1;
   mgr.level[0].f_dofs.data[0] = 9;
   mgr.level[0].f_relaxation.type = 7;

   const int      vals[3] = {0, 1, 2};
   IntArray *dofmap = precon_test_intarray_plain_data_only(3, vals);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_f_dofs_label_not_in_dofmap_is_ignored(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size    = 1;
   mgr.level[0].f_dofs.data[0] = 1;
   mgr.level[0].f_relaxation.type = 7;

   const int      vals[4] = {0, 2, 2, 0};
   IntArray *dofmap = precon_test_intarray_plain_data_only(4, vals);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.point_marker_data)
   {
      free(mgr.point_marker_data);
      mgr.point_marker_data = NULL;
   }
   if (mgr.csolver)
   {
      HYPRE_BoomerAMGDestroy(mgr.csolver);
      mgr.csolver = NULL;
   }

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_f_dofs_duplicate_label(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 2;
   mgr.level[0].f_dofs.size    = 2;
   mgr.level[0].f_dofs.data[0] = 0;
   mgr.level[0].f_dofs.data[1] = 0;
   mgr.level[0].f_relaxation.type = 7;

   const int      vals[3] = {0, 1, 2};
   IntArray *dofmap = precon_test_intarray_plain_data_only(3, vals);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_empty_dofmap(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;

   IntArray *dofmap = hypredrv_IntArrayCreate(0);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_dofmap_negative_label(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   MGR_args mgr;
   hypredrv_MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1;

   const int      v      = -1;
   IntArray *dofmap = precon_test_intarray_plain_data_only(1, &v);
   hypredrv_MGRSetDofmap(&mgr, dofmap);

   HYPRE_Solver precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

#if HYPRE_CHECK_MIN_VERSION(30100, 5)
static void
test_MGRCreate_nested_mgr_validation_errors(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   /* Nested MGR F-relaxation only supported at outer level 0 */
   {
      MGR_args mgr;
      hypredrv_MGRSetDefaultArgs(&mgr);
      mgr.num_levels = 3;

      IntArray *dofmap = NULL;
      const int map[3] = {0, 1, 2};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
      hypredrv_MGRSetDofmap(&mgr, dofmap);

      mgr.level[0].f_dofs.size          = 1;
      mgr.level[0].f_dofs.data[0]       = 0;
      mgr.level[0].f_relaxation.type    = 7;

      mgr.level[1].f_dofs.size       = 1;
      mgr.level[1].f_dofs.data[0]    = 1;
      mgr.level[1].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
      MGR_args *inner                = (MGR_args *)malloc(sizeof(MGR_args));
      ASSERT_NOT_NULL(inner);
      hypredrv_MGRSetDefaultArgs(inner);
      inner->num_levels = 1;
      mgr.level[1].f_relaxation.mgr = inner;

      HYPRE_Solver precon = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_NULL(precon);

      free(inner);
      hypredrv_IntArrayDestroy(&dofmap);
   }

   /* Missing nested mgr: block */
   {
      MGR_args mgr;
      hypredrv_MGRSetDefaultArgs(&mgr);
      mgr.num_levels = 2;

      IntArray *dofmap = NULL;
      const int map[3] = {0, 1, 2};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
      hypredrv_MGRSetDofmap(&mgr, dofmap);

      mgr.level[0].f_dofs.size       = 1;
      mgr.level[0].f_dofs.data[0]    = 0;
      mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
      mgr.level[0].f_relaxation.mgr = NULL;

      HYPRE_Solver precon = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_NULL(precon);

      hypredrv_IntArrayDestroy(&dofmap);
   }

   /* Empty parent f_dofs leaves the outer level inactive, so nested projection
    * validation is not reached. Keep this case as a no-op regression check. */
   {
      MGR_args mgr;
      hypredrv_MGRSetDefaultArgs(&mgr);
      mgr.num_levels = 2;

      IntArray *dofmap = NULL;
      const int map[3] = {0, 1, 2};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
      hypredrv_MGRSetDofmap(&mgr, dofmap);

      mgr.level[0].f_dofs.size       = 0;
      mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
      MGR_args *inner                = (MGR_args *)malloc(sizeof(MGR_args));
      ASSERT_NOT_NULL(inner);
      hypredrv_MGRSetDefaultArgs(inner);
      inner->num_levels = 1;
      mgr.level[0].f_relaxation.mgr = inner;

      HYPRE_Solver precon = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      if (precon)
      {
         HYPRE_MGRDestroy(precon);
      }
      if (mgr.point_marker_data)
      {
         free(mgr.point_marker_data);
         mgr.point_marker_data = NULL;
      }
      if (mgr.csolver)
      {
         HYPRE_BoomerAMGDestroy(mgr.csolver);
         mgr.csolver = NULL;
         mgr.csolver_type = -1;
      }

      free(inner);
      hypredrv_IntArrayDestroy(&dofmap);
   }

   /* A parent f_dofs label that is absent from the dofmap leaves the level
    * inactive before nested projection validation runs. Keep this as a no-op
    * regression check. */
   {
      MGR_args mgr;
      hypredrv_MGRSetDefaultArgs(&mgr);
      mgr.num_levels = 2;

      const int      vals[4] = {0, 2, 2, 0};
      IntArray *dofmap = precon_test_intarray_plain_data_only(4, vals);
      hypredrv_MGRSetDofmap(&mgr, dofmap);

      mgr.level[0].f_dofs.size       = 1;
      mgr.level[0].f_dofs.data[0]    = 1;
      mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
      MGR_args *inner                = (MGR_args *)malloc(sizeof(MGR_args));
      ASSERT_NOT_NULL(inner);
      hypredrv_MGRSetDefaultArgs(inner);
      inner->num_levels = 1;
      mgr.level[0].f_relaxation.mgr = inner;

      HYPRE_Solver precon = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      if (precon)
      {
         HYPRE_MGRDestroy(precon);
      }
      if (mgr.point_marker_data)
      {
         free(mgr.point_marker_data);
         mgr.point_marker_data = NULL;
      }
      if (mgr.csolver)
      {
         HYPRE_BoomerAMGDestroy(mgr.csolver);
         mgr.csolver = NULL;
         mgr.csolver_type = -1;
      }

      free(inner);
      hypredrv_IntArrayDestroy(&dofmap);
   }

   /* Nested projection: duplicate parent f_dofs labels */
   {
      MGR_args mgr;
      hypredrv_MGRSetDefaultArgs(&mgr);
      mgr.num_levels = 2;

      IntArray *dofmap = NULL;
      const int map[3] = {0, 1, 2};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
      hypredrv_MGRSetDofmap(&mgr, dofmap);

      mgr.level[0].f_dofs.size       = 2;
      mgr.level[0].f_dofs.data[0]    = 0;
      mgr.level[0].f_dofs.data[1]    = 0;
      mgr.level[0].f_relaxation.type = MGR_FRLX_TYPE_NESTED_MGR;
      MGR_args *inner                = (MGR_args *)malloc(sizeof(MGR_args));
      ASSERT_NOT_NULL(inner);
      hypredrv_MGRSetDefaultArgs(inner);
      inner->num_levels = 1;
      mgr.level[0].f_relaxation.mgr = inner;

      HYPRE_Solver precon = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_MGRCreate(&mgr, &precon, NULL, 0);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_NULL(precon);

      free(inner);
      hypredrv_IntArrayDestroy(&dofmap);
   }

   TEST_HYPRE_FINALIZE();
}
#endif /* HYPRE_CHECK_MIN_VERSION(30100, 5) */

static void
test_PreconCreate_mgr_with_near_null_vector(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);
   args.mgr.num_levels = 1;

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   HYPRE_IJVector vec_nn = precon_test_ij_vector_1x1(1.0);

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, vec_nn, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   HYPRE_IJVectorDestroy(vec_nn);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif /* HYPRE_CHECK_MIN_VERSION(21900, 0) */

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
static void
test_PreconDestroy_mgr_frelax_krylov_nested(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 7;

   args.mgr.level[0].f_relaxation.use_krylov = 1;
   args.mgr.level[0].f_relaxation.krylov =
      (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
   ASSERT_NOT_NULL(args.mgr.level[0].f_relaxation.krylov);
   hypredrv_NestedKrylovSetDefaultArgs(args.mgr.level[0].f_relaxation.krylov);
   args.mgr.level[0].f_relaxation.krylov->is_set = 1;
   args.mgr.level[0].f_relaxation.krylov->solver_method = SOLVER_GMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_GMRES,
                                           &args.mgr.level[0].f_relaxation.krylov->solver);
   args.mgr.level[0].f_relaxation.krylov->solver.gmres.max_iter = 2;
   args.mgr.level[0].f_relaxation.krylov->has_precon            = 1;
   args.mgr.level[0].f_relaxation.krylov->precon_method         = PRECON_BOOMERAMG;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG,
                                           &args.mgr.level[0].f_relaxation.krylov->precon);
   args.mgr.level[0].f_relaxation.krylov->precon.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_MGRDestroyNestedSolverArgs(&args.mgr);
   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

#if HYPRE_CHECK_MIN_VERSION(23200, 14)
static void
test_PreconDestroy_mgr_frelax_ilu_type32(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 32;
   hypredrv_ILUSetDefaultArgs(&args.mgr.level[0].f_relaxation.ilu);
   args.mgr.level[0].f_relaxation.ilu.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

#if defined(HYPRE_USING_DSUPERLU) && HYPRE_CHECK_MIN_VERSION(23100, 9)
static void
test_PreconDestroy_mgr_frelax_spdirect_type29(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 29;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}
#endif

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
static void
test_PreconDestroy_mgr_grelax_amg_type20(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 7;
   args.mgr.level[0].g_relaxation.type = 20;
   hypredrv_AMGSetDefaultArgs(&args.mgr.level[0].g_relaxation.amg);
   args.mgr.level[0].g_relaxation.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat = precon_test_ij_matrix_1x1(1.0);
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_MGR, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
   HYPRE_IJMatrixDestroy(mat);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconDestroy_mgr_grelax_ilu_type16(void)
{
#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#endif
   TEST_HYPRE_INIT();

   precon_args args;
   hypredrv_PreconSetDefaultArgs(&args);
   hypredrv_MGRSetDefaultArgs(&args.mgr);

   IntArray *dofmap = NULL;
   const int map[3] = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 3, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 2;
   args.mgr.level[0].f_dofs.size    = 1;
   args.mgr.level[0].f_dofs.data[0] = 0;
   args.mgr.level[0].f_relaxation.type = 7;
   args.mgr.level[0].g_relaxation.type = 16;
   hypredrv_ILUSetDefaultArgs(&args.mgr.level[0].g_relaxation.ilu);
   args.mgr.level[0].g_relaxation.ilu.max_iter = 1;

   HYPRE_Precon precon = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon, NULL, 0);
   ASSERT_NOT_NULL(precon);

   HYPRE_IJMatrix mat = precon_test_ij_matrix_1x1(1.0);
   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconSetup(PRECON_MGR, precon, mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_PreconDestroy(PRECON_MGR, &args, &precon, NULL, 0);
   ASSERT_NULL(precon);
   HYPRE_IJMatrixDestroy(mat);

   hypredrv_IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_ILUSetFieldByName_all_fields(void)
{
   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "max_iter", .value = "3"},
      {.key = "print_level", .value = "2"},
      {.key = "type", .value = "1"},
      {.key = "fill_level", .value = "2"},
      {.key = "reordering", .value = "1"},
      {.key = "tri_solve", .value = "0"},
      {.key = "lower_jac_iters", .value = "3"},
      {.key = "upper_jac_iters", .value = "4"},
      {.key = "max_row_nnz", .value = "300"},
      {.key = "schur_max_iter", .value = "5"},
      {.key = "droptol", .value = "1.0e-3"},
      {.key = "nsh_droptol", .value = "1.0e-4"},
      {.key = "tolerance", .value = "1.0e-5"},
   };

   ILU_args args;
   hypredrv_ILUSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_ILUSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 3);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ(args.type, 1);
   ASSERT_EQ(args.fill_level, 2);
   ASSERT_EQ(args.reordering, 1);
   ASSERT_EQ(args.tri_solve, 0);
   ASSERT_EQ(args.lower_jac_iters, 3);
   ASSERT_EQ(args.upper_jac_iters, 4);
   ASSERT_EQ(args.max_row_nnz, 300);
   ASSERT_EQ(args.schur_max_iter, 5);
   ASSERT_EQ_DOUBLE(args.droptol, 1.0e-3, 1e-12);
   ASSERT_EQ_DOUBLE(args.nsh_droptol, 1.0e-4, 1e-12);
   ASSERT_EQ_DOUBLE(args.tolerance, 1.0e-5, 1e-12);
}

static void
test_hypredrv_ILUSetFieldByName_unknown_key(void)
{
   ILU_args args;
   hypredrv_ILUSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_ILUSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_ILUGetValidValues_type(void)
{
   StrIntMapArray map = hypredrv_ILUGetValidValues("type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "bj-iluk"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "bj-ilut"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "gmres-iluk"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "rap-mod-ilu0"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "bj-iluk"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "bj-ilut"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "gmres-iluk"), 10);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "rap-mod-ilu0"), 50);
}

static void
test_hypredrv_ILUGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_ILUGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
test_hypredrv_ILUCreate_hits_schur_and_nsh_branches(void)
{
   ILU_args     args;
   HYPRE_Solver precon = NULL;

   hypredrv_ILUSetDefaultArgs(&args);
   /* nsh-ilut (21): Schur coarse solve + NSH drop threshold paths in hypredrv_ILUCreate */
   args.type = 21;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ILUCreate(&args, &precon);
   ASSERT_NOT_NULL(precon);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   HYPRE_ILUDestroy(precon);
}
#endif

static void
test_hypredrv_FSAISetFieldByName_all_fields(void)
{
   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "max_iter", .value = "2"},
      {.key = "print_level", .value = "1"},
      {.key = "algo_type", .value = "2"},
      {.key = "ls_type", .value = "2"},
      {.key = "max_steps", .value = "6"},
      {.key = "max_step_size", .value = "4"},
      {.key = "max_nnz_row", .value = "20"},
      {.key = "num_levels", .value = "2"},
      {.key = "eig_max_iters", .value = "6"},
      {.key = "threshold", .value = "1.0e-4"},
      {.key = "kap_tolerance", .value = "1.0e-4"},
      {.key = "tolerance", .value = "1.0e-6"},
   };

   FSAI_args args;
   hypredrv_FSAISetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_FSAISetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 2);
   ASSERT_EQ(args.print_level, 1);
   ASSERT_EQ(args.algo_type, 2);
   ASSERT_EQ(args.ls_type, 2);
   ASSERT_EQ(args.max_steps, 6);
   ASSERT_EQ(args.max_step_size, 4);
   ASSERT_EQ(args.max_nnz_row, 20);
   ASSERT_EQ(args.num_levels, 2);
   ASSERT_EQ(args.eig_max_iters, 6);
   ASSERT_EQ_DOUBLE(args.threshold, 1.0e-4, 1e-12);
   ASSERT_EQ_DOUBLE(args.kap_tolerance, 1.0e-4, 1e-12);
   ASSERT_EQ_DOUBLE(args.tolerance, 1.0e-6, 1e-12);
}

static void
test_hypredrv_FSAISetFieldByName_unknown_key(void)
{
   FSAI_args args;
   hypredrv_FSAISetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_FSAISetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_FSAIGetValidValues_algo_type(void)
{
   StrIntMapArray map = hypredrv_FSAIGetValidValues("algo_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "bj-afsai"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "bj-afsai-omp"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "bj-sfsai"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "bj-afsai"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "bj-afsai-omp"), 2);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "bj-sfsai"), 3);
}

static void
test_hypredrv_FSAIGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_FSAIGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

/* AMG argument tests (merged from test_amg_args.c) */
static void
test_hypredrv_AMGSetFieldByName_all_fields(void)
{
   AMG_args args;
   hypredrv_AMGSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "max_iter", .value = "5"},
      {.key = "print_level", .value = "2"},
      {.key = "tolerance", .value = "1.0e-6"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_AMGSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 5);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ_DOUBLE(args.tolerance, 1.0e-6, 1e-12);
}

static void
test_hypredrv_AMGSetFieldByName_unknown_key(void)
{
   AMG_args args;
   hypredrv_AMGSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_AMGSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - it just doesn't match any field */
   /* Verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_AMGintGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = hypredrv_AMGintGetValidValues("prolongation_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "mod_classical"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "extended+i"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "one_point"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "mod_classical"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "extended+i"), 6);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "one_point"), 100);
}

static void
test_hypredrv_AMGintGetValidValues_restriction_type(void)
{
   StrIntMapArray map = hypredrv_AMGintGetValidValues("restriction_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "p_transpose"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "air_1"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "air_1.5"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "p_transpose"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "air_1"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "air_1.5"), 15);
}

static void
test_hypredrv_AMGintGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_AMGintGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_AMGcsnGetValidValues_type(void)
{
   StrIntMapArray map = hypredrv_AMGcsnGetValidValues("type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "cljp"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "falgout"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "hmis"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "cljp"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "falgout"), 6);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "hmis"), 10);
}

static void
test_hypredrv_AMGcsnGetValidValues_on_off_keys(void)
{
   StrIntMapArray map1 = hypredrv_AMGcsnGetValidValues("filter_functions");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map1, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map1, "off"));

   StrIntMapArray map2 = hypredrv_AMGcsnGetValidValues("nodal");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map2, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map2, "off"));

   StrIntMapArray map3 = hypredrv_AMGcsnGetValidValues("rap2");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map3, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map3, "off"));
}

static void
test_hypredrv_AMGcsnGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_AMGcsnGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_AMGaggGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = hypredrv_AMGaggGetValidValues("prolongation_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "2_stage_extended+i"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "multipass"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "mm_extended+e"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "2_stage_extended+i"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "multipass"), 4);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "mm_extended+e"), 7);
}

static void
test_hypredrv_AMGaggGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_AMGaggGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_AMGrlxGetValidValues_down_type(void)
{
   StrIntMapArray map = hypredrv_AMGrlxGetValidValues("down_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "jacobi_non_mv"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "chebyshev"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "l1sym-hgs"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "jacobi_non_mv"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "chebyshev"), 16);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "l1sym-hgs"), 89);
}

static void
test_hypredrv_AMGrlxGetValidValues_up_type(void)
{
   StrIntMapArray map = hypredrv_AMGrlxGetValidValues("up_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "backward-hgs"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "cg"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "backward-hgs"), 4);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "cg"), 15);
}

static void
test_hypredrv_AMGrlxGetValidValues_coarse_type(void)
{
   StrIntMapArray map = hypredrv_AMGrlxGetValidValues("coarse_type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "lu_piv"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "lu_inv"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "lu_piv"), 99);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "lu_inv"), 199);
}

static void
test_hypredrv_AMGrlxGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_AMGrlxGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_AMGsmtGetValidValues_type(void)
{
   StrIntMapArray map = hypredrv_AMGsmtGetValidValues("type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "fsai"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "ilu"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "euclid"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "fsai"), 4);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "ilu"), 5);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "euclid"), 9);
}

static void
test_hypredrv_AMGsmtGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_AMGsmtGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_AMGintSetFieldByName_all_fields(void)
{
   AMGint_args args;
   hypredrv_AMGintSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "prolongation_type", .value = "8"},
      {.key = "restriction_type", .value = "1"},
      {.key = "max_nnz_row", .value = "6"},
      {.key = "trunc_factor", .value = "0.5"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_AMGintSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.prolongation_type, 8);
   ASSERT_EQ(args.restriction_type, 1);
   ASSERT_EQ(args.max_nnz_row, 6);
   ASSERT_EQ_DOUBLE(args.trunc_factor, 0.5, 1e-12);
}

static void
test_hypredrv_AMGcsnSetFieldByName_all_fields(void)
{
   AMGcsn_args args;
   hypredrv_AMGcsnSetDefaultArgs(&args);

   static const struct
   {
      const char *key;
      const char *value;
   } updates[] = {
      {.key = "type", .value = "8"},
      {.key = "rap2", .value = "1"},
      {.key = "mod_rap2", .value = "0"},
      {.key = "keep_transpose", .value = "1"},
      {.key = "num_functions", .value = "2"},
      {.key = "filter_functions", .value = "1"},
      {.key = "nodal", .value = "0"},
      {.key = "seq_amg_th", .value = "1"},
      {.key = "min_coarse_size", .value = "10"},
      {.key = "max_coarse_size", .value = "100"},
      {.key = "max_levels", .value = "15"},
      {.key = "max_row_sum", .value = "0.8"},
      {.key = "strong_th", .value = "0.3"},
   };

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_AMGcsnSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.type, 8);
   ASSERT_EQ(args.rap2, 1);
   ASSERT_EQ(args.mod_rap2, 0);
   ASSERT_EQ(args.keep_transpose, 1);
   ASSERT_EQ(args.num_functions, 2);
   ASSERT_EQ(args.filter_functions, 1);
   ASSERT_EQ(args.nodal, 0);
   ASSERT_EQ(args.seq_amg_th, 1);
   ASSERT_EQ(args.min_coarse_size, 10);
   ASSERT_EQ(args.max_coarse_size, 100);
   ASSERT_EQ(args.max_levels, 15);
   ASSERT_EQ_DOUBLE(args.max_row_sum, 0.8, 1e-12);
   ASSERT_EQ_DOUBLE(args.strong_th, 0.3, 1e-12);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

#if !HYPRE_CHECK_MIN_VERSION(22100, 0)
   fprintf(stderr, "SKIP: preconditioner tests require hypre >= 2.21.0\n");
   MPI_Finalize();
   return 0;
#endif

   RUN_TEST(test_PreconGetValidKeys_contains_expected);
   RUN_TEST(test_PreconGetValidTypeIntMap_contains_known_types);
   RUN_TEST(test_PreconSetDefaultArgs_resets_reuse);
   RUN_TEST(test_PreconSetArgsFromYAML_sets_fields);
   RUN_TEST(test_PreconSetArgsFromYAML_null_or_empty_parent_noop);
   RUN_TEST(test_PreconSetArgsFromYAML_ignores_unknown_key);
   RUN_TEST(test_PreconReuse_null_args_noop);
   RUN_TEST(test_PreconReuseShouldRebuild_null_args_early_exit);
   RUN_TEST(test_mgr_nested_frelax_wrapper_api_null_safe);
   RUN_TEST(test_hypredrv_PreconArgsSetDefaultsForMethod_and_GetValidValues);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_type_parses_components);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_scalar_installs_default_components);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_allows_disabled_hard_guards);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_rebuild_on_new_level_block_sequence);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_linear_solver_ids_alias);
   RUN_TEST(
      test_PreconReuseSetArgsFromYAML_adaptive_rebuild_on_new_level_rejects_out_of_range);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_type_without_components_uses_defaults);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_scalar_and_explicit_type_share_defaults);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_rejects_negative_component_weight);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_bootstrap_defers_scoring);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_iterations_rms);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_bad_decision_streak);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_max_reuse_solves_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_default_max_iteration_ratio_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_max_iteration_ratio_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_max_solve_time_ratio_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_rebuild_on_new_timestep_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_solver_failure_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_rebuild_on_new_level_guard);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_active_level_scope);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_completed_level_scope);
   RUN_TEST(
      test_PreconReuseShouldRebuild_adaptive_completed_level_scope_excludes_pre_rebuild_history);
   RUN_TEST(
      test_PreconReuseShouldRebuild_adaptive_completed_level_solve_overhead_honors_mean_reduction);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_setup_time_metric);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_geometric_mean);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_harmonic_mean);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_min_max_mean_iterations);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_power_mean_near_zero);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_delta_from_baseline_transform);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_min_reuse_solves_guard);
   RUN_TEST(test_PreconReuse_api_null_observation_and_log_noop);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_scalar_static_frequency_shorthand);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_scalar_always_shorthand);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_type_always_alias);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_always_key_rejected);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_scalar_invalid_value);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_enabled_off_clears_static_fields);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_static_conflict_requires_explicit_adaptive);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_adaptive_conflict_with_frequency);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_static_conflict_linear_system_ids_with_frequency);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_infer_adaptive_policy_from_adaptive_block);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_unknown_root_key);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_parse_mean_transform_history_errors);
   RUN_TEST(test_PreconReuseSetArgsFromYAML_parse_component_guards_adaptive_errors);
   RUN_TEST(test_PreconReuseShouldRebuild_static_frequency_and_linear_system_ids);
   RUN_TEST(test_PreconReuseShouldRebuild_static_per_timestep_with_starts);
   RUN_TEST(test_PreconReuseShouldRebuild_static_per_timestep_missing_context_returns_hold);
   RUN_TEST(test_PreconReuseShouldRebuild_static_disabled_always_rebuilds);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_null_state_requests_rebuild);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_initial_build_next_ls_id_nonpositive);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_status_good_and_insufficient_history);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_relative_increase_total_time_below_direction);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_power_mean_negative_p_and_summary_truncation);
   RUN_TEST(test_PreconReuseStateRecordObservation_realloc_and_cap_drop_oldest);
   RUN_TEST(test_PreconReuseShouldRebuild_adaptive_window_mean_baseline_warns_and_scores);
   RUN_TEST(test_PreconReuseShouldRebuild_duplicate_ls_id_skips_bad_streak_increment);
   RUN_TEST(test_PreconReuseShouldRebuild_static_per_timestep_embedded_stats);
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_spdirect_flat);
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_flat_sets_type);
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_nested_sets_type_and_args);
   RUN_TEST(test_PreconDestroy_null_precon);
   RUN_TEST(test_PreconDestroy_null_main);
   RUN_TEST(test_PreconCreate_invalid_method);
   RUN_TEST(test_PreconCreate_precon_none);
   RUN_TEST(test_PreconSetup_null_precon);
   RUN_TEST(test_PreconSetup_null_A);
   RUN_TEST(test_Precon_lifecycle_boomeramg_1x1);
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_Precon_lifecycle_ilu_1x1);
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   RUN_TEST(test_Precon_lifecycle_fsai_1x1);
#endif
   RUN_TEST(test_PreconDestroy_amg_rbms_loop);
   RUN_TEST(test_PreconDestroy_none_with_main_logs);
   RUN_TEST(test_PreconSetup_default_case);
   RUN_TEST(test_PreconApply_default_case);
   RUN_TEST(test_PreconApply_precon_none);
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_PreconApply_mgr_minimal);
#endif
   RUN_TEST(test_PreconDestroy_amg_log_dispatch_no_rbms);
   RUN_TEST(test_MGRCreate_coarsest_level_branches);
   RUN_TEST(test_PreconCreate_mgr_coarsest_level_krylov_nested);
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_PreconDestroy_mgr_coarsest_use_krylov_without_krylov_ptr);
#endif
   RUN_TEST(test_PreconSetup_mgr_frelax_nested_mgr_dof_labels);
   RUN_TEST(test_PreconSetup_mgr_frelax_nested_mgr_body_split_labels);
   RUN_TEST(test_PreconCreate_mgr_nested_krylov_inner_mgr_recreate_without_reuse);
   RUN_TEST(test_PreconDestroy_mgr_csolver_destroy_branches);
   RUN_TEST(test_MGRComponentReuseSetupMode_policy_shape_and_selector_paths);
   RUN_TEST(test_MGRComponentReuseSetupMode_nested_shape_unsupported);
   RUN_TEST(test_MGRComponentReuseShouldKeepOuter_and_SelectKeepFlags);
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   RUN_TEST(test_MGRRefreshComponentsForSetup_rebuilds_fsai_handles);
#endif
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_PreconDestroy_mgr_frelax_amg_type2);
#endif
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   RUN_TEST(test_PreconDestroy_mgr_grelax_krylov_nested);
#endif
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_MGRCreate_missing_dofmap);
   RUN_TEST(test_MGRCreate_plain_dofmap_data_only_branch);
   RUN_TEST(test_MGRCreate_unique_data_dofmap_branch);
   RUN_TEST(test_MGRCreate_g_unique_unsorted_dense_fallback);
   RUN_TEST(test_MGRCreate_prefers_local_dof_labels_over_dense_global_fallback);
   RUN_TEST(test_MGRCreate_compact_global_labels_ignore_missing_f_dofs);
   RUN_TEST(test_MGRCreate_global_unique_metadata_ignore_missing_f_dofs);
   RUN_TEST(test_MGRCreate_dense_remap_sparse_labels);
   RUN_TEST(test_MGRCreate_f_dofs_out_of_range);
   RUN_TEST(test_MGRCreate_f_dofs_label_not_in_dofmap_is_ignored);
   RUN_TEST(test_MGRCreate_f_dofs_duplicate_label);
   RUN_TEST(test_MGRCreate_empty_dofmap);
   RUN_TEST(test_MGRCreate_dofmap_negative_label);
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   RUN_TEST(test_MGRCreate_nested_mgr_validation_errors);
#endif
   RUN_TEST(test_PreconCreate_mgr_with_near_null_vector);
#endif
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   RUN_TEST(test_PreconDestroy_mgr_frelax_krylov_nested);
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
   RUN_TEST(test_PreconDestroy_mgr_frelax_ilu_type32);
#endif
#if defined(HYPRE_USING_DSUPERLU) && HYPRE_CHECK_MIN_VERSION(23100, 9)
   RUN_TEST(test_PreconDestroy_mgr_frelax_spdirect_type29);
#endif
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   RUN_TEST(test_PreconDestroy_mgr_grelax_amg_type20);
   RUN_TEST(test_PreconDestroy_mgr_grelax_ilu_type16);
   RUN_TEST(test_PreconDestroy_mgr_grelax_ilu_later_level_type16);
#endif
   RUN_TEST(test_hypredrv_ILUSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_ILUSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_ILUGetValidValues_type);
   RUN_TEST(test_hypredrv_ILUGetValidValues_unknown_key);
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_hypredrv_ILUCreate_hits_schur_and_nsh_branches);
#endif
   RUN_TEST(test_hypredrv_FSAISetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_FSAISetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_FSAIGetValidValues_algo_type);
   RUN_TEST(test_hypredrv_FSAIGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_AMGSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_AMGintGetValidValues_prolongation_type);
   RUN_TEST(test_hypredrv_AMGintGetValidValues_restriction_type);
   RUN_TEST(test_hypredrv_AMGintGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGcsnGetValidValues_type);
   RUN_TEST(test_hypredrv_AMGcsnGetValidValues_on_off_keys);
   RUN_TEST(test_hypredrv_AMGcsnGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGaggGetValidValues_prolongation_type);
   RUN_TEST(test_hypredrv_AMGaggGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGrlxGetValidValues_down_type);
   RUN_TEST(test_hypredrv_AMGrlxGetValidValues_up_type);
   RUN_TEST(test_hypredrv_AMGrlxGetValidValues_coarse_type);
   RUN_TEST(test_hypredrv_AMGrlxGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGsmtGetValidValues_type);
   RUN_TEST(test_hypredrv_AMGsmtGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_AMGintSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_AMGcsnSetFieldByName_all_fields);

   MPI_Finalize();
   return 0;
}
