#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "internal/args.h"
#include "internal/presets.h"
#include "internal/precon_reuse.h"
#include "internal/scaling.h"
#include "internal/error.h"
#include "internal/stats.h"
#include "test_helpers.h"
#include "internal/utils.h"
#include "internal/yaml.h"

static YAMLtree *
build_tree(const char *text)
{
   char     *buffer = strdup(text);
   YAMLtree *tree   = NULL;
   hypredrv_YAMLtreeBuild(2, buffer, &tree);
   free(buffer);
   return tree;
}

static input_args *
parse_config(const char *yaml_text)
{
   input_args *args    = NULL;
   char       *argv0   = strdup(yaml_text);
   char       *argv[1] = {argv0};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);
   free(argv0);
   return args;
}

static input_args *
parse_config_with_overrides(const char *yaml_text, int override_argc,
                            char **override_argv)
{
   input_args *args  = NULL;
   char       *argv0 = strdup(yaml_text);

   char **argv = (char **)calloc((size_t)override_argc + 1, sizeof(char *));
   argv[0]     = argv0;
   for (int i = 0; i < override_argc; i++)
   {
      argv[i + 1] = override_argv[i];
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, override_argc + 1, argv, &args);

   free(argv);
   free(argv0);
   return args;
}

static void
test_InputArgsCreate_general_vendor_defaults(void)
{
   input_args *args = NULL;

   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);

#ifdef HYPRE_USING_GPU
   ASSERT_EQ(args->general.use_vendor_spgemm, 1);
   ASSERT_EQ(args->general.use_vendor_spmv, 1);
#else
   ASSERT_EQ(args->general.use_vendor_spgemm, 0);
   ASSERT_EQ(args->general.use_vendor_spmv, 0);
#endif
   ASSERT_STREQ(args->general.name, "");
   ASSERT_STREQ(args->general.statistics_filename, "");

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParseGeneral_flags(void)
{
   const char yaml_text[] = "general:\n"
                            "  name: flow-solver\n"
                            "  statistics_filename: stats.out\n"
                            "  warmup: yes\n"
                            "  statistics: off\n"
                            "  use_millisec: yes\n"
                            "  print_config_params: no\n"
                            "  use_vendor_spgemm: yes\n"
                            "  use_vendor_spmv: yes\n"
                            "  num_repetitions: 3\n"
                            "  dev_pool_size: 2\n"
                            "  uvm_pool_size: 3\n"
                            "  host_pool_size: 4\n"
                            "  pinned_pool_size: 0.25\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 50\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_STREQ(args->general.name, "flow-solver");
   ASSERT_STREQ(args->general.statistics_filename, "stats.out");
   ASSERT_EQ(args->general.warmup, 1);
   ASSERT_EQ(args->general.statistics, 0);
   ASSERT_EQ(args->general.print_config_params, 0);
   ASSERT_EQ(args->general.use_vendor_spgemm, 1);
   ASSERT_EQ(args->general.use_vendor_spmv, 1);
   ASSERT_EQ(args->general.num_repetitions, 3);
   ASSERT_EQ((int)(args->general.dev_pool_size / GB_TO_BYTES), 2);
   ASSERT_EQ((int)(args->general.uvm_pool_size / GB_TO_BYTES), 3);
   ASSERT_EQ((int)(args->general.host_pool_size / GB_TO_BYTES), 4);
   ASSERT_TRUE(args->general.pinned_pool_size > 0);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParseGeneral_use_millisec_sets_timer(void)
{
   const char yaml_text[] = "general:\n"
                            "  use_millisec: yes\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   /* Verify parsed value; stats initialization happens inside HYPREDRV_InputArgsParse */
   ASSERT_TRUE(args->general.use_millisec);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParseSolver_value_only(void)
{
   const char yaml_text[] = "solver: bicgstab\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_BICGSTAB);
   ASSERT_EQ(args->solver.bicgstab.print_level, 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_value_only(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 30\n"
                            "preconditioner: fsai\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_FSAI);
   ASSERT_EQ(args->precon.fsai.print_level, 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_preset_value_only(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 30\n"
                            "preconditioner: poisson\n";

   input_args *args = parse_config(yaml_text);
   /* Value-only presets are not supported. */
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_NULL(args);
}

static void
test_InputArgsParsePrecon_preset_explicit_key(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  preset: elasticity-2D\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon.amg.coarsening.num_functions, 2);
   ASSERT_TRUE(args->precon.amg.coarsening.strong_th > 0.79 &&
               args->precon.amg.coarsening.strong_th < 0.81);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_missing(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 40\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_KEY);
   ASSERT_NULL(args);
}

static void
test_InputArgsParsePrecon_variants(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: HMIS\n"
                            "        strong_th: 0.25\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: PMIS\n"
                            "        strong_th: 0.5\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 2);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);

   /* Check first variant */
   ASSERT_EQ(args->precon_variants[0].amg.print_level, 1);
   ASSERT_EQ(args->precon_variants[0].amg.coarsening.type, 10); /* HMIS */
   ASSERT_TRUE(args->precon_variants[0].amg.coarsening.strong_th > 0.24 &&
               args->precon_variants[0].amg.coarsening.strong_th < 0.26);

   /* Check second variant */
   ASSERT_EQ(args->precon_variants[1].amg.print_level, 1);
   ASSERT_EQ(args->precon_variants[1].amg.coarsening.type, 8); /* PMIS */
   ASSERT_TRUE(args->precon_variants[1].amg.coarsening.strong_th > 0.49 &&
               args->precon_variants[1].amg.coarsening.strong_th < 0.51);

   /* Check active variant is set to first */
   ASSERT_EQ(args->active_precon_variant, 0);
   ASSERT_EQ(args->precon.amg.print_level, 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_mgr_flat_g_relaxation_amg_defaults(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  mgr:\n"
                            "    level:\n"
                            "      0:\n"
                            "        f_dofs: [2]\n"
                            "        g_relaxation: amg\n"
                            "    coarsest_level: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_MGR);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.type, 20);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.amg.max_iter, 1);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.amg.interpolation.prolongation_type,
             6);
   ASSERT_TRUE(args->precon.mgr.level[0].g_relaxation.amg.coarsening.type == 8 ||
               args->precon.mgr.level[0].g_relaxation.amg.coarsening.type == 10);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_mgr_flat_g_relaxation_ilu_defaults(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  mgr:\n"
                            "    level:\n"
                            "      0:\n"
                            "        f_dofs: [2]\n"
                            "        g_relaxation: ilu\n"
                            "    coarsest_level: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_MGR);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.type, 16);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.ilu.max_iter, 1);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.ilu.type, 0);
   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.ilu.fill_level, 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_mgr_component_reuse_blocks(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  mgr:\n"
                            "    level:\n"
                            "      0:\n"
                            "        f_dofs: [0]\n"
                            "        f_relaxation:\n"
                            "          amg:\n"
                            "            max_iter: 1\n"
                            "          reuse:\n"
                            "            frequency: 2\n"
                            "        g_relaxation:\n"
                            "          ilu:\n"
                            "            max_iter: 1\n"
                            "          reuse: adaptive\n"
                            "    coarsest_level:\n"
                            "      amg:\n"
                            "        max_iter: 1\n"
                            "      reuse:\n"
                            "        enabled: off\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_MGR);

   ASSERT_EQ(args->precon.mgr.level[0].f_relaxation.reuse.present, 1);
   ASSERT_EQ(args->precon.mgr.level[0].f_relaxation.reuse.args.frequency, 2);
   ASSERT_EQ((int)args->precon.mgr.level[0].f_relaxation.reuse.args.policy,
             (int)PRECON_REUSE_POLICY_STATIC);

   ASSERT_EQ(args->precon.mgr.level[0].g_relaxation.reuse.present, 1);
   ASSERT_EQ((int)args->precon.mgr.level[0].g_relaxation.reuse.args.policy,
             (int)PRECON_REUSE_POLICY_ADAPTIVE);

   ASSERT_EQ(args->precon.mgr.coarsest_level.reuse.present, 1);
   ASSERT_EQ(args->precon.mgr.coarsest_level.reuse.args.enabled, 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_mgr_component_reuse_unknown_nested_key(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  mgr:\n"
                            "    level:\n"
                            "      0:\n"
                            "        f_dofs: [0]\n"
                            "        f_relaxation:\n"
                            "          amg:\n"
                            "            max_iter: 1\n"
                            "          reuse:\n"
                            "            frequency: 1\n"
                            "            not_a_reuse_key: 0\n"
                            "    coarsest_level: amg\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_mgr_component_reuse_always_alias(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  mgr:\n"
                            "    level:\n"
                            "      0:\n"
                            "        f_dofs: [0]\n"
                            "        f_relaxation:\n"
                            "          amg:\n"
                            "            max_iter: 1\n"
                            "          reuse: always\n"
                            "    coarsest_level: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_MGR);

   ASSERT_EQ(args->precon.mgr.level[0].f_relaxation.reuse.present, 1);
   ASSERT_EQ(args->precon.mgr.level[0].f_relaxation.reuse.args.enabled, 1);
   ASSERT_EQ((int)args->precon.mgr.level[0].f_relaxation.reuse.args.policy,
             (int)PRECON_REUSE_POLICY_STATIC);
   ASSERT_NOT_NULL(args->precon.mgr.level[0].f_relaxation.reuse.args.linear_system_ids);
   ASSERT_EQ_SIZE(args->precon.mgr.level[0].f_relaxation.reuse.args.linear_system_ids->size,
                  1);
   ASSERT_EQ(args->precon.mgr.level[0].f_relaxation.reuse.args.linear_system_ids->data[0],
             0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_root_sequence_variants(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  - amg:\n"
                            "      print_level: 2\n"
                            "  - fsai:\n"
                            "      print_level: 3\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 2);
   ASSERT_EQ(args->active_precon_variant, 0);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon_methods[0], PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon_methods[1], PRECON_FSAI);
   ASSERT_EQ(args->precon_variants[0].amg.print_level, 2);
   ASSERT_EQ(args->precon_variants[1].fsai.print_level, 3);
   ASSERT_EQ(args->precon.amg.print_level, 2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_root_sequence_with_preset(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  - preset: elasticity-2D\n"
                            "  - fsai:\n"
                            "      print_level: 4\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 2);
   ASSERT_EQ(args->active_precon_variant, 0);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon_methods[0], PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon_variants[0].amg.coarsening.num_functions, 2);
   ASSERT_EQ(args->precon_methods[1], PRECON_FSAI);
   ASSERT_EQ(args->precon_variants[1].fsai.print_level, 4);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_frequency(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    frequency: 2\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ(args->precon_reuse.frequency, 2);
   ASSERT_EQ(args->precon_reuse.per_timestep, 0);
   ASSERT_NULL(args->precon_reuse.linear_system_ids);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_linear_system_ids(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    linear_system_ids: [0, 3, 5]\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_NOT_NULL(args->precon_reuse.linear_system_ids);
   ASSERT_EQ((int)args->precon_reuse.linear_system_ids->size, 3);
   ASSERT_EQ(args->precon_reuse.linear_system_ids->data[0], 0);
   ASSERT_EQ(args->precon_reuse.linear_system_ids->data[1], 3);
   ASSERT_EQ(args->precon_reuse.linear_system_ids->data[2], 5);
   ASSERT_EQ(args->precon_reuse.per_timestep, 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_per_timestep(void)
{
   const char yaml_text[] = "linear_system:\n"
                            "  timestep_filename: timesteps.txt\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    per_timestep: on\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ(args->precon_reuse.per_timestep, 1);
   ASSERT_STREQ(args->ls.timestep_filename, "timesteps.txt");

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_per_timestep_with_frequency(void)
{
   const char yaml_text[] = "linear_system:\n"
                            "  timestep_filename: timesteps.txt\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    per_timestep: on\n"
                            "    frequency: 2\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ(args->precon_reuse.per_timestep, 1);
   ASSERT_EQ(args->precon_reuse.frequency, 2);
   ASSERT_STREQ(args->ls.timestep_filename, "timesteps.txt");

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_scalar_frequency_shorthand(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse: 7\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ(args->precon_reuse.frequency, 7);
   ASSERT_EQ((int)args->precon_reuse.policy, (int)PRECON_REUSE_POLICY_STATIC);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_scalar_adaptive_shorthand(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse: adaptive\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ((int)args->precon_reuse.policy, (int)PRECON_REUSE_POLICY_ADAPTIVE);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_invalid_scalar_value(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse: not_a_valid_policy_or_int\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_reuse_unknown_nested_key(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    frequency: 1\n"
                            "    not_a_reuse_key: 0\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_reuse_linear_solver_ids_alias(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    linear_solver_ids: [1, 2]\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_NOT_NULL(args->precon_reuse.linear_system_ids);
   ASSERT_EQ((int)args->precon_reuse.linear_system_ids->size, 2);
   ASSERT_EQ(args->precon_reuse.linear_system_ids->data[0], 1);
   ASSERT_EQ(args->precon_reuse.linear_system_ids->data[1], 2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_adaptive_with_guards(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  reuse:\n"
                            "    type: adaptive\n"
                            "    guards:\n"
                            "      min_history_points: 2\n"
                            "      bad_decisions_to_rebuild: 2\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ((int)args->precon_reuse.policy, (int)PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_EQ(args->precon_reuse.guards.min_history_points, 2);
   ASSERT_EQ(args->precon_reuse.guards.bad_decisions_to_rebuild, 2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_unknown_preset_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  preset: __hypredrv_no_such_preset__\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_preset_missing_name_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  preset:\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParse_linear_system_matrix_rhs_file_keys(void)
{
   const char yaml_text[] = "linear_system:\n"
                            "  matrix_filename: m.bin\n"
                            "  rhs_filename: r.bin\n"
                            "  rhs_mode: file\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_STREQ(args->ls.matrix_filename, "m.bin");
   ASSERT_STREQ(args->ls.rhs_filename, "r.bin");
   ASSERT_EQ((int)args->ls.rhs_mode, 2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_reuse_adaptive_with_component_sequence(void)
{
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  reuse:\n"
      "    enabled: yes\n"
      "    type: adaptive\n"
      "    guards:\n"
      "      min_history_points: 2\n"
      "      min_reuse_solves: 1\n"
      "    adaptive:\n"
      "      rebuild_threshold: 0.5\n"
      "      components:\n"
      "        - name: solve-time-floor\n"
      "          metric: solve_time\n"
      "          target: 1.0e-12\n"
      "          scale: 1.0e-12\n"
      "          mean:\n"
      "            kind: arithmetic\n"
      "          transform:\n"
      "            kind: raw\n"
      "          history:\n"
      "            source: linear_solves\n"
      "            max_points: 2\n"
      "  amg:\n"
      "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ((int)args->precon_reuse.policy, (int)PRECON_REUSE_POLICY_ADAPTIVE);
   ASSERT_TRUE(args->precon_reuse.adaptive.num_components >= 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_scaling_enabled_rhs_l2(void)
{
   /* scaling is a child of the root "solver" section (see InputArgsParseSolver). */
   const char yaml_text[] = "solver:\n"
                            "  scaling:\n"
                            "    enabled: 1\n"
                            "    type: rhs_l2\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->scaling.enabled, 1);
   ASSERT_EQ((int)args->scaling.type, (int)SCALING_RHS_L2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_unknown_type_value_only_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner: totally_unknown_precon_method\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_extra_sibling_key_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n"
                            "  orphan_key: 1\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_variants_two_top_level_keys_under_item_rejected(void)
{
   /* Each sequence item under preconditioner.variants must contain exactly one
    * preconditioner type key (see InputArgsParsePreconRootSequence). */
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  variants:\n"
                            "    - amg:\n"
                            "        print_level: 0\n"
                            "      fsai:\n"
                            "        print_level: 1\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_YAMLtreeBuild_inconsistent_indent(void)
{
   const char yaml_text[] = "root:\n"
                            "   child: value\n";

   YAMLtree *tree = build_tree(yaml_text);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeValidate(tree);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_YAML_INVALID_INDENT);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtextRead_missing_file(void)
{
   int    base_indent = -1;
   size_t length      = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead("nonexistent_dir", "missing.yml", 0, &base_indent, &length, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_YAMLtreeUpdate_overrides_solver_and_precon(void)
{
   const char yaml_text[] = "solver: gmres\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   char *overrides[] = {
      "--solver:pcg:max_iter", "50",  "--preconditioner:amg:print_level", "2",
      "--general:statistics",  "off", "--general:use_vendor_spgemm",      "on",
      "--general:use_vendor_spmv", "on", "--general:statistics_filename",
      "stats_cli.out",
   };

   input_args *args = parse_config_with_overrides(yaml_text, 12, overrides);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   ASSERT_EQ(args->solver.pcg.max_iter, 50);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon.amg.print_level, 2);
   ASSERT_EQ(args->general.statistics, 0);
   ASSERT_EQ(args->general.use_vendor_spgemm, 1);
   ASSERT_EQ(args->general.use_vendor_spmv, 1);
   ASSERT_STREQ(args->general.statistics_filename, "stats_cli.out");

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_driver_mode_with_config_file(void)
{
   /* Test the branch where config file is found anywhere in argv (driver mode) */
   input_args *args        = NULL;
   char        yaml_file[] = "/tmp/test_config.yml";
   FILE       *fp          = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: pcg\npreconditioner: amg\n");
   fclose(fp);

   char *argv[] = {"hypredrive", "-q", yaml_file};
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 3, argv, &args);

   if (!hypredrv_ErrorCodeActive())
   {
      ASSERT_NOT_NULL(args);
      ASSERT_EQ(args->solver_method, SOLVER_PCG);
      hypredrv_InputArgsDestroy(&args);
   }

   unlink(yaml_file);
}

static void
test_InputArgsParse_null_argv0_error(void)
{
   /* Test the branch where argv[0] is NULL (error path) */
   input_args *args   = NULL;
   char       *argv[] = {NULL};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_UNKNOWN);
}

static void
test_InputArgsParse_file_not_found_error(void)
{
   /* Test the branch where file doesn't exist */
   input_args *args   = NULL;
   char       *argv[] = {"nonexistent.yml"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_InputArgsParse_legacy_mode_with_overrides(void)
{
   /* Test legacy mode: argv[0] is YAML filename, argv[1..] are override pairs */
   input_args *args        = NULL;
   char        yaml_file[] = "/tmp/test_config2.yml";
   FILE       *fp          = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: gmres\npreconditioner: amg\n");
   fclose(fp);

   char *argv[] = {yaml_file, "--solver:pcg:max_iter", "100"};
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 3, argv, &args);

   if (!hypredrv_ErrorCodeActive())
   {
      ASSERT_NOT_NULL(args);
      ASSERT_EQ(args->solver_method, SOLVER_PCG);
      ASSERT_EQ(args->solver.pcg.max_iter, 100);
      hypredrv_InputArgsDestroy(&args);
   }

   unlink(yaml_file);
}

static void
test_InputArgsParse_driver_mode_with_nodash_overrides(void)
{
   input_args *args        = NULL;
   char        yaml_file[] = "/tmp/test_config3.yml";
   FILE       *fp          = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: gmres\n"
               "preconditioner:\n"
               "  mgr:\n"
               "    print_level: 0\n"
               "    level:\n"
               "      0:\n"
               "        f_dofs: [2]\n"
               "        f_relaxation: single\n"
               "    coarsest_level: amg\n");
   fclose(fp);

   char *argv2[] = {"hypredrive-cli", "-q", yaml_file, "--args",
                    "preconditioner:mgr:print_level", "1"};
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 6, argv2, &args);

   if (!hypredrv_ErrorCodeActive())
   {
      ASSERT_NOT_NULL(args);
      ASSERT_EQ(args->precon_method, PRECON_MGR);
      ASSERT_EQ(args->precon.mgr.print_level, 1);
      hypredrv_InputArgsDestroy(&args);
   }

   unlink(yaml_file);
}

static void
test_InputArgsParse_root_solver_not_shadowed_by_nested_solver(void)
{
   const char yaml_text[] = "wrapper:\n"
                            "  solver: bicgstab\n"
                            "solver: pcg\n"
                            "preconditioner: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_root_precon_not_shadowed_by_nested_precon(void)
{
   const char yaml_text[] = "wrapper:\n"
                            "  preconditioner: fsai\n"
                            "solver: pcg\n"
                            "preconditioner: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_duplicate_root_solver_rejected(void)
{
   const char yaml_text[] = "solver: pcg\n"
                            "solver: gmres\n"
                            "preconditioner: amg\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParse_duplicate_root_preconditioner_rejected(void)
{
   const char yaml_text[] = "solver: pcg\n"
                            "preconditioner: amg\n"
                            "preconditioner: fsai\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParse_duplicate_root_general_rejected(void)
{
   const char yaml_text[] = "general:\n"
                            "  name: a\n"
                            "general:\n"
                            "  name: b\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParse_linear_system_unexpected_scalar_value(void)
{
   /* Scalar + nested entries on linear_system -> YAML_NODE_UNEXPECTED_VAL */
   const char yaml_text[] = "linear_system: not_allowed\n"
                            "  matrix_filename: m.bin\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNEXPECTED_VAL) != 0);
}

static void
test_InputArgsParse_solver_section_missing_keeps_defaults(void)
{
   /* No top-level solver: early-return path in hypredrv_InputArgsParseSolver() */
   const char yaml_text[] = "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_solver_empty_block_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_SOLVER) != 0);
}

static void
test_InputArgsParse_solver_two_types_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "  gmres:\n"
                            "    max_iter: 20\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParsePrecon_root_sequence_item_empty_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  -\n"
                            "  - amg:\n"
                            "      print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_typed_block_empty_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_PRECON) != 0);
}

static void
test_InputArgsParsePrecon_typed_block_two_children_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n"
                            "  fsai:\n"
                            "    print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParsePrecon_variants_not_sequence_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  variants: 1\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParsePrecon_reuse_after_amg_detach_path(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n"
                            "  reuse:\n"
                            "    frequency: 2\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon_reuse.enabled, 1);
   ASSERT_EQ(args->precon_reuse.frequency, 2);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_general_statistics_level_2(void)
{
   const char yaml_text[] = "general:\n"
                            "  statistics: 2\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->general.statistics, 2);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsCreate_lib_mode_print_config_params(void)
{
   input_args *lib = NULL;
   input_args *drv = NULL;

   hypredrv_InputArgsCreate(true, &lib);
   hypredrv_InputArgsCreate(false, &drv);
   ASSERT_NOT_NULL(lib);
   ASSERT_NOT_NULL(drv);
   ASSERT_EQ(lib->general.print_config_params, 0);
   ASSERT_EQ(drv->general.print_config_params, 1);

   hypredrv_InputArgsDestroy(&lib);
   hypredrv_InputArgsDestroy(&drv);
}

static void
test_hypredrv_InputArgsApplyPreconPreset_null_args(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplyPreconPreset(NULL, "elasticity-2D", 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
}

static void
test_hypredrv_InputArgsApplyPreconPreset_null_preset(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplyPreconPreset(args, NULL, 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplyPreconPreset_bad_variant_idx(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  - amg:\n"
                            "      print_level: 0\n"
                            "  - fsai:\n"
                            "      print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplyPreconPreset(args, "elasticity-2D", 2);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplyPreconPreset_lazy_alloc_and_apply(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 1);
   ASSERT_NULL(args->precon_methods);
   ASSERT_NULL(args->precon_variants);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplyPreconPreset(args, "elasticity-2D", 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args->precon_methods);
   ASSERT_NOT_NULL(args->precon_variants);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon.amg.coarsening.num_functions, 2);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplyPreconPreset_unknown_preset(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplyPreconPreset(args, "__no_such_preset__", 0);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplySolverPreset_null_args(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplySolverPreset(NULL, "my_solver");
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
}

static void
test_hypredrv_InputArgsApplySolverPreset_null_preset(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplySolverPreset(args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplySolverPreset_apply(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);

   ASSERT_EQ(hypredrv_PresetRegisterTyped(
                "my_solver_preset",
                "fgmres:\n"
                "  max_iter: 17\n"
                "  krylov_dim: 23\n"
                "  print_level: 0",
                "custom solver",
                HYPREDRV_PRESET_SOLVER),
             0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplySolverPreset(args, "my_solver_preset");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args->solver_method, SOLVER_FGMRES);
   ASSERT_EQ(args->solver.fgmres.max_iter, 17);
   ASSERT_EQ(args->solver.fgmres.krylov_dim, 23);
   ASSERT_EQ(args->solver.fgmres.print_level, 0);

   hypredrv_PresetFreeUserPresets();
   hypredrv_InputArgsDestroy(&args);
}

static void
test_hypredrv_InputArgsApplySolverPreset_unknown_preset(void)
{
   input_args *args = NULL;
   hypredrv_InputArgsCreate(false, &args);
   ASSERT_NOT_NULL(args);

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsApplySolverPreset(args, "__no_such_solver_preset__");
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParse_invalid_yaml_file_on_disk(void)
{
   char path[256];
   (void)snprintf(path, sizeof(path), "/tmp/hypred_args_bad_%d.yml", (int)getpid());

   FILE *fp = fopen(path, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "linear_system: x\n");
   fprintf(fp, "  matrix_filename: m.bin\n");
   fclose(fp);

   char *argv[] = {path};
   input_args *args = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   unlink(path);
}

static void
test_InputArgsParse_include_from_subdirectory(void)
{
   char dir[256];
   char outer[288];
   char inner[288];
   (void)snprintf(dir, sizeof(dir), "/tmp/hypred_args_inc_%d", (int)getpid());
   (void)snprintf(outer, sizeof(outer), "%s/outer.yml", dir);
   (void)snprintf(inner, sizeof(inner), "%s/inner.yml", dir);

   ASSERT_EQ(mkdir(dir, 0700), 0);

   FILE *fo = fopen(outer, "w");
   ASSERT_NOT_NULL(fo);
   fprintf(fo, "include: inner.yml\n");
   fclose(fo);

   FILE *fi = fopen(inner, "w");
   ASSERT_NOT_NULL(fi);
   fprintf(fi, "solver: pcg\n");
   fprintf(fi, "preconditioner:\n");
   fprintf(fi, "  amg:\n");
   fprintf(fi, "    print_level: 0\n");
   fclose(fi);

   char *argv[] = {outer};
   input_args *args = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);

   hypredrv_InputArgsDestroy(&args);
   unlink(outer);
   unlink(inner);
   rmdir(dir);
}

static void
test_InputArgsParse_driver_mode_config_not_argv0_with_overrides(void)
{
   char yaml_file[256];
   (void)snprintf(yaml_file, sizeof(yaml_file), "/tmp/hypred_args_drv_%d.yml",
                  (int)getpid());

   FILE *fp = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: gmres\n");
   fprintf(fp, "preconditioner:\n");
   fprintf(fp, "  amg:\n");
   fprintf(fp, "    print_level: 0\n");
   fclose(fp);

   /* Config is argv[2], not argv[0]; driver mode requires --args before override pairs. */
   char *argv[] = {"hypredrive", "-q", yaml_file, "--args",
                   "--preconditioner:amg:print_level", "2"};
   input_args *args = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 6, argv, &args);

   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_GMRES);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon.amg.print_level, 2);

   hypredrv_InputArgsDestroy(&args);
   unlink(yaml_file);
}

static void
test_InputArgsParseWithObjectName_basic(void)
{
   const char yaml_text[] = "solver: pcg\n"
                              "preconditioner:\n"
                              "  amg:\n"
                              "    print_level: 0\n";

   input_args *args    = NULL;
   char       *argv0   = strdup(yaml_text);
   char       *argv[1] = {argv0};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_SELF, true, 1, argv, &args,
                                         "test_args_object");
   free(argv0);

   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParseWithObjectName_invalid_yaml_tree(void)
{
   /* Triggers rank-0 error before section parsing; exercises HYPREDRV_LOGF paths with a
    * non-NULL log object name. */
   const char yaml_text[] = "solver: [\n";
   input_args *args       = NULL;
   char       *argv0      = strdup(yaml_text);
   char       *argv[1]    = {argv0};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_SELF, true, 1, argv, &args,
                                         "args_parse_obj");
   free(argv0);

   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParseWithObjectName_duplicate_solver_post_parse(void)
{
   /* ERROR_EXTRA_KEY is raised during section parsing; post-parse validation still runs
    * and returns through the failure path with logging (non-NULL log object name). */
   const char yaml_text[] = "solver: pcg\n"
                            "solver: gmres\n"
                            "preconditioner: amg\n";

   input_args *args    = NULL;
   char       *argv0   = strdup(yaml_text);
   char       *argv[1] = {argv0};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_SELF, false, 1, argv, &args,
                                         "args_parse_obj");
   free(argv0);

   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_EXTRA_KEY) != 0);
}

static void
test_InputArgsParse_general_exec_policy_mirrors_to_ls(void)
{
   const char yaml_text[] = "general:\n"
                            "  exec_policy: device\n"
                            "solver: pcg\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->general.exec_policy, 1);
   ASSERT_EQ(args->ls.exec_policy, 1);

   hypredrv_InputArgsDestroy(&args);

   const char yaml_host[] = "general:\n"
                            "  exec_policy: host\n"
                            "solver: pcg\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   args = parse_config(yaml_host);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->general.exec_policy, 0);
   ASSERT_EQ(args->ls.exec_policy, 0);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsRead_null_filename(void)
{
   int    base_indent = -1;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsRead(MPI_COMM_SELF, NULL, &base_indent, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND) != 0);
}

static void
test_InputArgsParse_solver_value_only_fgmres_and_bicgstab(void)
{
   const char yaml_f[] = "solver: fgmres\n"
                         "preconditioner:\n"
                         "  amg:\n"
                         "    print_level: 0\n";
   const char yaml_b[] = "solver: bicgstab\n"
                         "preconditioner:\n"
                         "  amg:\n"
                         "    print_level: 0\n";

   input_args *af = parse_config(yaml_f);
   ASSERT_NOT_NULL(af);
   ASSERT_EQ(af->solver_method, SOLVER_FGMRES);
   ASSERT_EQ(af->solver.fgmres.print_level, 0);
   hypredrv_InputArgsDestroy(&af);

   input_args *ab = parse_config(yaml_b);
   ASSERT_NOT_NULL(ab);
   ASSERT_EQ(ab->solver_method, SOLVER_BICGSTAB);
   ASSERT_EQ(ab->solver.bicgstab.print_level, 0);
   hypredrv_InputArgsDestroy(&ab);
}

static void
test_InputArgsParsePrecon_amg_typed_sequence_variants(void)
{
   /* Sequence items directly under typed preconditioner.amg (InputArgsParsePreconTypedBlock). */
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    - print_level: 1\n"
                            "    - print_level: 2\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->num_precon_variants, 2);
   ASSERT_EQ(args->precon_variants[0].amg.print_level, 1);
   ASSERT_EQ(args->precon_variants[1].amg.print_level, 2);
   hypredrv_InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_root_sequence_unknown_type_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  - not_a_real_precon_type:\n"
                            "      print_level: 0\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParse_driver_mode_missing_config_file(void)
{
   input_args *args   = NULL;
   char       *argv[] = {"hypredrive", "-q", "/tmp/hypred_args_missing_config_xyz.yml"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 3, argv, &args);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND) != 0);
}

static void
test_InputArgsParsePrecon_root_sequence_preset_invalid_rejected(void)
{
   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 10\n"
                            "preconditioner:\n"
                            "  - preset: __hypredrv_no_such_preset__\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_InputArgsParse_file_include_target_missing(void)
{
   /* hypredrv_YAMLtextRead fails during include expansion -> InputArgsRead error path. */
   char path[256];
   (void)snprintf(path, sizeof(path), "/tmp/hypred_args_inc_miss_%d.yml", (int)getpid());

   FILE *fp = fopen(path, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "include: this_include_file_does_not_exist.yml\n");
   fclose(fp);

   char *argv[] = {path};
   input_args *args = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   unlink(path);
}

static void
test_PreconPreset_user_registered_two_types_rejected(void)
{
   /* Drives PreconPresetBuildArgs() multi-child preset failure (args.c). */
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_PresetRegister(
                "hypredrv_test_two_precon_types",
                "amg:\n  print_level: 0\nfsai:\n  print_level: 0\n", "unit"),
             0);

   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 1\n"
                            "preconditioner:\n"
                            "  preset: hypredrv_test_two_precon_types\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_PresetFreeUserPresets();
}

static void
test_PreconPreset_user_registered_unknown_type_rejected(void)
{
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_PresetRegister(
                "hypredrv_test_bad_precon_type",
                "not_a_real_precon_keyword:\n  print_level: 0\n", "unit"),
             0);

   const char yaml_text[] = "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 1\n"
                            "preconditioner:\n"
                            "  preset: hypredrv_test_bad_precon_type\n";

   hypredrv_ErrorCodeResetAll();
   input_args *args = parse_config(yaml_text);
   ASSERT_NULL(args);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_PresetFreeUserPresets();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_InputArgsCreate_general_vendor_defaults);
   RUN_TEST(test_InputArgsParseGeneral_flags);
   RUN_TEST(test_InputArgsParseGeneral_use_millisec_sets_timer);
   RUN_TEST(test_InputArgsParseSolver_value_only);
   RUN_TEST(test_InputArgsParsePrecon_value_only);
   RUN_TEST(test_InputArgsParsePrecon_preset_value_only);
   RUN_TEST(test_InputArgsParsePrecon_preset_explicit_key);
   RUN_TEST(test_InputArgsParsePrecon_missing);
   RUN_TEST(test_InputArgsParsePrecon_variants);
   RUN_TEST(test_InputArgsParsePrecon_mgr_flat_g_relaxation_amg_defaults);
   RUN_TEST(test_InputArgsParsePrecon_mgr_flat_g_relaxation_ilu_defaults);
   RUN_TEST(test_InputArgsParsePrecon_mgr_component_reuse_blocks);
   RUN_TEST(test_InputArgsParsePrecon_mgr_component_reuse_unknown_nested_key);
   RUN_TEST(test_InputArgsParsePrecon_mgr_component_reuse_always_alias);
   RUN_TEST(test_InputArgsParsePrecon_root_sequence_variants);
   RUN_TEST(test_InputArgsParsePrecon_root_sequence_with_preset);
   RUN_TEST(test_InputArgsParse_linear_system_matrix_rhs_file_keys);
   RUN_TEST(test_InputArgsParse_scaling_enabled_rhs_l2);
   RUN_TEST(test_InputArgsParsePrecon_reuse_adaptive_with_component_sequence);
   RUN_TEST(test_InputArgsParsePrecon_reuse_frequency);
   RUN_TEST(test_InputArgsParsePrecon_reuse_linear_system_ids);
   RUN_TEST(test_InputArgsParsePrecon_reuse_per_timestep);
   RUN_TEST(test_InputArgsParsePrecon_reuse_per_timestep_with_frequency);
   RUN_TEST(test_InputArgsParsePrecon_reuse_scalar_frequency_shorthand);
   RUN_TEST(test_InputArgsParsePrecon_reuse_scalar_adaptive_shorthand);
   RUN_TEST(test_InputArgsParsePrecon_reuse_invalid_scalar_value);
   RUN_TEST(test_InputArgsParsePrecon_reuse_unknown_nested_key);
   RUN_TEST(test_InputArgsParsePrecon_reuse_linear_solver_ids_alias);
   RUN_TEST(test_InputArgsParsePrecon_reuse_adaptive_with_guards);
   RUN_TEST(test_InputArgsParsePrecon_unknown_preset_rejected);
   RUN_TEST(test_InputArgsParsePrecon_preset_missing_name_rejected);
   RUN_TEST(test_InputArgsParsePrecon_variants_two_top_level_keys_under_item_rejected);
   RUN_TEST(test_InputArgsParsePrecon_unknown_type_value_only_rejected);
   RUN_TEST(test_InputArgsParsePrecon_extra_sibling_key_rejected);
   RUN_TEST(test_YAMLtreeBuild_inconsistent_indent);
   RUN_TEST(test_YAMLtextRead_missing_file);
   RUN_TEST(test_YAMLtreeUpdate_overrides_solver_and_precon);
   RUN_TEST(test_InputArgsParse_driver_mode_with_config_file);
   RUN_TEST(test_InputArgsParse_null_argv0_error);
   RUN_TEST(test_InputArgsParse_file_not_found_error);
   RUN_TEST(test_InputArgsParse_legacy_mode_with_overrides);
   RUN_TEST(test_InputArgsParse_driver_mode_with_nodash_overrides);
   RUN_TEST(test_InputArgsParse_root_solver_not_shadowed_by_nested_solver);
   RUN_TEST(test_InputArgsParse_root_precon_not_shadowed_by_nested_precon);
   RUN_TEST(test_InputArgsParse_duplicate_root_solver_rejected);
   RUN_TEST(test_InputArgsParse_duplicate_root_preconditioner_rejected);
   RUN_TEST(test_InputArgsParse_duplicate_root_general_rejected);
   RUN_TEST(test_InputArgsParse_linear_system_unexpected_scalar_value);
   RUN_TEST(test_InputArgsParse_solver_section_missing_keeps_defaults);
   RUN_TEST(test_InputArgsParse_solver_empty_block_rejected);
   RUN_TEST(test_InputArgsParse_solver_two_types_rejected);
   RUN_TEST(test_InputArgsParsePrecon_root_sequence_item_empty_rejected);
   RUN_TEST(test_InputArgsParsePrecon_typed_block_empty_rejected);
   RUN_TEST(test_InputArgsParsePrecon_typed_block_two_children_rejected);
   RUN_TEST(test_InputArgsParsePrecon_variants_not_sequence_rejected);
   RUN_TEST(test_InputArgsParsePrecon_reuse_after_amg_detach_path);
   RUN_TEST(test_InputArgsParse_general_statistics_level_2);
   RUN_TEST(test_InputArgsCreate_lib_mode_print_config_params);
   RUN_TEST(test_hypredrv_InputArgsApplyPreconPreset_null_args);
   RUN_TEST(test_hypredrv_InputArgsApplyPreconPreset_null_preset);
   RUN_TEST(test_hypredrv_InputArgsApplyPreconPreset_bad_variant_idx);
   RUN_TEST(test_hypredrv_InputArgsApplyPreconPreset_lazy_alloc_and_apply);
   RUN_TEST(test_hypredrv_InputArgsApplyPreconPreset_unknown_preset);
   RUN_TEST(test_hypredrv_InputArgsApplySolverPreset_null_args);
   RUN_TEST(test_hypredrv_InputArgsApplySolverPreset_null_preset);
   RUN_TEST(test_hypredrv_InputArgsApplySolverPreset_apply);
   RUN_TEST(test_hypredrv_InputArgsApplySolverPreset_unknown_preset);
   RUN_TEST(test_InputArgsParse_invalid_yaml_file_on_disk);
   RUN_TEST(test_InputArgsParse_include_from_subdirectory);
   RUN_TEST(test_InputArgsParse_driver_mode_config_not_argv0_with_overrides);
   RUN_TEST(test_InputArgsParseWithObjectName_basic);
   RUN_TEST(test_InputArgsParseWithObjectName_invalid_yaml_tree);
   RUN_TEST(test_InputArgsParseWithObjectName_duplicate_solver_post_parse);
   RUN_TEST(test_InputArgsParse_general_exec_policy_mirrors_to_ls);
   RUN_TEST(test_InputArgsRead_null_filename);
   RUN_TEST(test_InputArgsParse_solver_value_only_fgmres_and_bicgstab);
   RUN_TEST(test_InputArgsParsePrecon_amg_typed_sequence_variants);
   RUN_TEST(test_InputArgsParsePrecon_root_sequence_unknown_type_rejected);
   RUN_TEST(test_InputArgsParse_driver_mode_missing_config_file);
   RUN_TEST(test_InputArgsParsePrecon_root_sequence_preset_invalid_rejected);
   RUN_TEST(test_InputArgsParse_file_include_target_missing);
   RUN_TEST(test_PreconPreset_user_registered_two_types_rejected);
   RUN_TEST(test_PreconPreset_user_registered_unknown_type_rejected);

   MPI_Finalize();
   return 0;
}
