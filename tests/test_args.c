#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
#include "error.h"
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
   ASSERT_EQ(args->warmup, 1);
   ASSERT_EQ(args->statistics, 0);
   ASSERT_EQ(args->print_config_params, 0);
   ASSERT_EQ(args->num_repetitions, 3);
   ASSERT_EQ((int)(args->dev_pool_size / GB_TO_BYTES), 2);
   ASSERT_EQ((int)(args->uvm_pool_size / GB_TO_BYTES), 3);
   ASSERT_EQ((int)(args->host_pool_size / GB_TO_BYTES), 4);
   ASSERT_TRUE(args->pinned_pool_size > 0);
   ASSERT_EQ(args->solver_method, SOLVER_PCG);

   InputArgsDestroy(&args);
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

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_InputArgsParseGeneral_flags);
   RUN_TEST(test_InputArgsParseSolver_value_only);
   RUN_TEST(test_InputArgsParsePrecon_value_only);
   RUN_TEST(test_InputArgsParsePrecon_missing);
   RUN_TEST(test_YAMLtreeBuild_inconsistent_indent);
   RUN_TEST(test_YAMLtextRead_missing_file);

   MPI_Finalize();
   return 0;
}
