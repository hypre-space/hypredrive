#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "args.h"
#include "error.h"
#include "stats.h"
#include "test_helpers.h"
#include "utils.h"
#include "yaml.h"

static YAMLtree *
build_tree(const char *text)
{
   char     *buffer = strdup(text);
   YAMLtree *tree   = NULL;
   YAMLtreeBuild(2, buffer, &tree);
   free(buffer);
   return tree;
}

static input_args *
parse_config(const char *yaml_text)
{
   input_args *args    = NULL;
   char       *argv0   = strdup(yaml_text);
   char       *argv[1] = {argv0};

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);
   free(argv0);
   return args;
}

static input_args *
parse_config_with_overrides(const char *yaml_text, int override_argc, char **override_argv)
{
   input_args *args = NULL;
   char       *argv0 = strdup(yaml_text);

   char **argv = (char **)calloc((size_t)override_argc + 1, sizeof(char *));
   argv[0]     = argv0;
   for (int i = 0; i < override_argc; i++)
   {
      argv[i + 1] = override_argv[i];
   }

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, override_argc + 1, argv, &args);

   free(argv);
   free(argv0);
   return args;
}

static void
test_InputArgsParseGeneral_flags(void)
{
   const char yaml_text[] = "general:\n"
                            "  warmup: yes\n"
                            "  statistics: off\n"
                            "  use_millisec: yes\n"
                            "  print_config_params: no\n"
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
   ASSERT_EQ(args->general.warmup, 1);
   ASSERT_EQ(args->general.statistics, 0);
   ASSERT_EQ(args->general.print_config_params, 0);
   ASSERT_EQ(args->general.num_repetitions, 3);
   ASSERT_EQ((int)(args->general.dev_pool_size / GB_TO_BYTES), 2);
   ASSERT_EQ((int)(args->general.uvm_pool_size / GB_TO_BYTES), 3);
   ASSERT_EQ((int)(args->general.host_pool_size / GB_TO_BYTES), 4);
   ASSERT_TRUE(args->general.pinned_pool_size > 0);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);

   InputArgsDestroy(&args);
}

static void
test_InputArgsParseGeneral_use_millisec_sets_timer(void)
{
   Stats *stats = StatsCreate();
   StatsSetContext(stats);

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
   ASSERT_NOT_NULL(StatsGetContext());
   ASSERT_TRUE(StatsGetContext()->use_millisec);

   InputArgsDestroy(&args);
   StatsDestroy(&stats);
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

   InputArgsDestroy(&args);
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

   InputArgsDestroy(&args);
}

static void
test_InputArgsParsePrecon_missing(void)
{
   const char yaml_text[] = "solver:\n"
                            "  gmres:\n"
                            "    max_iter: 40\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_MISSING_KEY);
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

   InputArgsDestroy(&args);
}

static void
test_YAMLtreeBuild_inconsistent_indent(void)
{
   const char yaml_text[] = "root:\n"
                            "   child: value\n";

   YAMLtree *tree = build_tree(yaml_text);

   ErrorCodeResetAll();
   YAMLtreeValidate(tree);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_YAML_INVALID_INDENT);

   YAMLtreeDestroy(&tree);
}

static void
test_YAMLtextRead_missing_file(void)
{
   int    base_indent = -1;
   size_t length      = 0;
   char  *text        = NULL;

   ErrorCodeResetAll();
   YAMLtextRead("nonexistent_dir", "missing.yml", 0, &base_indent, &length, &text);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_YAMLtreeUpdate_overrides_solver_and_precon(void)
{
   const char yaml_text[] = "solver: gmres\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n";

   char *overrides[] = {
      "--solver:pcg:max_iter",
      "50",
      "--preconditioner:amg:print_level",
      "2",
      "--general:statistics",
      "off",
   };

   input_args *args = parse_config_with_overrides(yaml_text, 6, overrides);
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);
   ASSERT_EQ(args->solver.pcg.max_iter, 50);
   ASSERT_EQ(args->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(args->precon.amg.print_level, 2);
   ASSERT_EQ(args->general.statistics, 0);

   InputArgsDestroy(&args);
}

static void
test_InputArgsParse_driver_mode_with_config_file(void)
{
   /* Test the branch where config file is found anywhere in argv (driver mode) */
   input_args *args = NULL;
   char        yaml_file[] = "/tmp/test_config.yml";
   FILE       *fp          = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: pcg\npreconditioner: amg\n");
   fclose(fp);

   char *argv[] = {"hypredrive", "-q", yaml_file};
   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, 3, argv, &args);

   if (!ErrorCodeActive())
   {
      ASSERT_NOT_NULL(args);
      ASSERT_EQ(args->solver_method, SOLVER_PCG);
      InputArgsDestroy(&args);
   }

   unlink(yaml_file);
}

static void
test_InputArgsParse_null_argv0_error(void)
{
   /* Test the branch where argv[0] is NULL (error path) */
   input_args *args = NULL;
   char       *argv[] = {NULL};

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_NULL(args);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_UNKNOWN);
}

static void
test_InputArgsParse_file_not_found_error(void)
{
   /* Test the branch where file doesn't exist */
   input_args *args = NULL;
   char       *argv[] = {"nonexistent.yml"};

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);

   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_NULL(args);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_InputArgsParse_legacy_mode_with_overrides(void)
{
   /* Test legacy mode: argv[0] is YAML filename, argv[1..] are override pairs */
   input_args *args = NULL;
   char        yaml_file[] = "/tmp/test_config2.yml";
   FILE       *fp          = fopen(yaml_file, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "solver: gmres\npreconditioner: amg\n");
   fclose(fp);

   char *argv[] = {yaml_file, "--solver:pcg:max_iter", "100"};
   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, false, 3, argv, &args);

   if (!ErrorCodeActive())
   {
      ASSERT_NOT_NULL(args);
      ASSERT_EQ(args->solver_method, SOLVER_PCG);
      ASSERT_EQ(args->solver.pcg.max_iter, 100);
      InputArgsDestroy(&args);
   }

   unlink(yaml_file);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_InputArgsParseGeneral_flags);
   RUN_TEST(test_InputArgsParseGeneral_use_millisec_sets_timer);
   RUN_TEST(test_InputArgsParseSolver_value_only);
   RUN_TEST(test_InputArgsParsePrecon_value_only);
   RUN_TEST(test_InputArgsParsePrecon_missing);
   RUN_TEST(test_InputArgsParsePrecon_variants);
   RUN_TEST(test_YAMLtreeBuild_inconsistent_indent);
   RUN_TEST(test_YAMLtextRead_missing_file);
   RUN_TEST(test_YAMLtreeUpdate_overrides_solver_and_precon);
   RUN_TEST(test_InputArgsParse_driver_mode_with_config_file);
   RUN_TEST(test_InputArgsParse_null_argv0_error);
   RUN_TEST(test_InputArgsParse_file_not_found_error);
   RUN_TEST(test_InputArgsParse_legacy_mode_with_overrides);

   MPI_Finalize();
   return 0;
}
