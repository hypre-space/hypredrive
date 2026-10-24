#include <string.h>

#include "internal/error.h"
#include "internal/eigspec.h"
#include "internal/yaml.h"
#include "test_helpers.h"
#include "test_ij_helpers.h"

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static void
test_EigSpecSetDefaultArgs(void)
{
   EigSpec_args args;
   memset(&args, 0, sizeof(args));
   hypredrv_EigSpecSetDefaultArgs(&args);
   ASSERT_EQ(args.enable, 0);
   ASSERT_EQ(args.vectors, 0);
   ASSERT_EQ(args.hermitian, 0);
   ASSERT_EQ(args.preconditioned, 0);
   ASSERT_STREQ(args.output_prefix, "eig");
}

static void
test_EigSpecSetArgsFromYAML_sets_fields(void)
{
   EigSpec_args args;
   hypredrv_EigSpecSetDefaultArgs(&args);

   YAMLnode *parent        = hypredrv_YAMLnodeCreate("eigspectrum", "", 0);
   YAMLnode *enable_node   = add_child(parent, "enable", "on", 1);
   YAMLnode *vectors_node  = add_child(parent, "vectors", "on", 1);
   YAMLnode *herm_node     = add_child(parent, "hermitian", "on", 1);
   YAMLnode *precon_node   = add_child(parent, "preconditioned", "on", 1);
   YAMLnode *prefix_node   = add_child(parent, "output_prefix", "diag", 1);

   hypredrv_ErrorStateReset();
   hypredrv_EigSpecSetArgs(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(enable_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(vectors_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(herm_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(precon_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(prefix_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(args.enable, 1);
   ASSERT_EQ(args.vectors, 1);
   ASSERT_EQ(args.hermitian, 1);
   ASSERT_EQ(args.preconditioned, 1);
   ASSERT_STREQ(args.output_prefix, "diag");

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_EigSpecSetArgsFromYAML_invalid_toggle_sets_error(void)
{
   EigSpec_args args;
   hypredrv_EigSpecSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("eigspectrum", "", 0);
   YAMLnode *enable_node = add_child(parent, "enable", "maybe", 1);

   hypredrv_ErrorStateReset();
   hypredrv_EigSpecSetArgs(&args, parent);

   ASSERT_EQ(enable_node->valid, YAML_NODE_INVALID_VAL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.enable, 0);

   hypredrv_YAMLnodeDestroy(parent);
   hypredrv_ErrorStateReset();
}

#ifdef HYPREDRV_ENABLE_EIGSPEC
static uint32_t
fail_preconditioner(void *ctx, void *b, void *x)
{
   (void)ctx; (void)b; (void)x;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   return ERROR_INVALID_PRECON;
}

static void
test_EigSpecCompute_own_status(void)
{
   HYPRE_IJMatrix matrix = create_test_ijmatrix_2x2(2.0, 0.0, 0.0, 3.0);
   EigSpec_args args;
   hypredrv_EigSpecSetDefaultArgs(&args);
   args.enable = args.vectors = 1;
   test_scratch_path(args.output_prefix, sizeof(args.output_prefix), "eig-status");
   for (int symmetric = 0; symmetric < 2; symmetric++)
   {
      args.hermitian = symmetric;
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ASSERT_EQ(hypredrv_EigSpecCompute(&args, matrix, NULL, NULL, NULL), ERROR_NONE);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
      char path[2 * MAX_FILENAME_LENGTH];
      snprintf(path, sizeof(path), "%s.values.txt", args.output_prefix);
      FILE *fp = fopen(path, "r");
      ASSERT_NOT_NULL(fp);
      double v1 = 0, v2 = 0, imaginary = 0;
      ASSERT_EQ(fscanf(fp, symmetric ? "%lf" : "%lf %lf", &v1, &imaginary), symmetric ? 1 : 2);
      ASSERT_EQ(fscanf(fp, symmetric ? "%lf" : "%lf %lf", &v2, &imaginary), symmetric ? 1 : 2);
      ASSERT_EQ_DOUBLE(v1 + v2, 5.0, 1e-12);
      fclose(fp);
   }
   args.preconditioned = 1;
   ASSERT_EQ(hypredrv_EigSpecCompute(&args, matrix, NULL, fail_preconditioner, NULL), ERROR_INVALID_PRECON);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP | ERROR_INVALID_PRECON);
   HYPRE_IJMatrixDestroy(matrix);
   hypredrv_ErrorStateReset();
}
#endif

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();
   RUN_TEST(test_EigSpecSetDefaultArgs);
   RUN_TEST(test_EigSpecSetArgsFromYAML_sets_fields);
   RUN_TEST(test_EigSpecSetArgsFromYAML_invalid_toggle_sets_error);
#ifdef HYPREDRV_ENABLE_EIGSPEC
   RUN_TEST(test_EigSpecCompute_own_status);
#endif
   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
