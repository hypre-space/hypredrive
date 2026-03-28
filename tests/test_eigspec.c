#include <string.h>

#include "internal/error.h"
#include "internal/eigspec.h"
#include "internal/yaml.h"
#include "test_helpers.h"

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

int
main(void)
{
   RUN_TEST(test_EigSpecSetDefaultArgs);
   RUN_TEST(test_EigSpecSetArgsFromYAML_sets_fields);
   RUN_TEST(test_EigSpecSetArgsFromYAML_invalid_toggle_sets_error);
   return 0;
}
