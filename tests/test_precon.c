#include <mpi.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "containers.h"
#include "error.h"
#include "precon.h"
#include "test_helpers.h"
#include "yaml.h"

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = YAMLnodeCreate(key, val, level);
   YAMLnodeAddChild(parent, child);
   return child;
}

static void
test_PreconGetValidKeys_contains_expected(void)
{
   StrArray keys = PreconGetValidKeys();

   ASSERT_TRUE(StrArrayEntryExists(keys, "amg"));
   ASSERT_TRUE(StrArrayEntryExists(keys, "mgr"));
   ASSERT_TRUE(StrArrayEntryExists(keys, "ilu"));
   ASSERT_TRUE(StrArrayEntryExists(keys, "fsai"));
   ASSERT_TRUE(StrArrayEntryExists(keys, "reuse"));
}

static void
test_PreconGetValidTypeIntMap_contains_known_types(void)
{
   StrIntMapArray map = PreconGetValidTypeIntMap();

   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "amg"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "amg"), PRECON_BOOMERAMG);

   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mgr"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mgr"), PRECON_MGR);
}

static void
test_PreconSetDefaultArgs_resets_reuse(void)
{
   precon_args args;
   args.reuse = 42;

   PreconSetDefaultArgs(&args);

   ASSERT_EQ(args.reuse, 0);
}

static void
test_PreconSetArgsFromYAML_sets_fields(void)
{
   precon_args args;

   PreconSetDefaultArgs(&args);
   args.amg.max_iter = 1; /* default */

   YAMLnode *parent     = YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *reuse_node = add_child(parent, "reuse", "1", 1);

   YAMLnode *amg_node = add_child(parent, "amg", "", 1);
   add_child(amg_node, "max_iter", "5", 2);

   ErrorCodeResetAll();
   PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(reuse_node->valid, YAML_NODE_VALID);
   ASSERT_EQ(args.reuse, 1);
   ASSERT_EQ(args.amg.max_iter, 5);

   YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_ignores_unknown_key(void)
{
   precon_args args;
   PreconSetDefaultArgs(&args);

   YAMLnode *parent       = YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *unknown_node = add_child(parent, "unknown", "value", 1);

   ErrorCodeResetAll();
   PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(unknown_node->valid, YAML_NODE_INVALID_KEY);
   ASSERT_EQ(args.reuse, 0); /* remains default */

   YAMLnodeDestroy(parent);
}

static void
test_PreconDestroy_null_precon(void)
{
   HYPRE_Precon precon = NULL;
   precon_args  args;
   PreconSetDefaultArgs(&args);

   /* PreconDestroy with NULL should not crash */
   PreconDestroy(PRECON_BOOMERAMG, &args, &precon);
   ASSERT_NULL(precon);
}

static void
test_PreconDestroy_null_main(void)
{
   HYPRE_Precon precon = malloc(sizeof(struct hypre_Precon_struct));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   precon_args args;
   PreconSetDefaultArgs(&args);

   /* PreconDestroy with null main should not crash */
   PreconDestroy(PRECON_BOOMERAMG, &args, &precon);
   ASSERT_NULL(precon);
}

static void
test_PreconSetup_default_case(void)
{
   HYPRE_Initialize();

   HYPRE_Precon precon = malloc(sizeof(struct hypre_Precon_struct));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test default case in switch (invalid precon_method) */
   PreconSetup(PRECON_NONE, precon, mat);

   free(precon);
   HYPRE_IJMatrixDestroy(mat);
   HYPRE_Finalize();
}

static void
test_PreconApply_default_case(void)
{
   HYPRE_Initialize();

   HYPRE_Precon precon = malloc(sizeof(struct hypre_Precon_struct));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   HYPRE_IJVector vec_b = NULL, vec_x = NULL;
   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec_b);
   HYPRE_IJVectorSetObjectType(vec_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(vec_b);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec_x);
   HYPRE_IJVectorSetObjectType(vec_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(vec_x);

   /* Test default case in switch (invalid precon_method) */
   PreconApply(PRECON_NONE, precon, mat, vec_b, vec_x);

   free(precon);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJVectorDestroy(vec_x);
   HYPRE_IJMatrixDestroy(mat);
   HYPRE_Finalize();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_PreconGetValidKeys_contains_expected);
   RUN_TEST(test_PreconGetValidTypeIntMap_contains_known_types);
   RUN_TEST(test_PreconSetDefaultArgs_resets_reuse);
   RUN_TEST(test_PreconSetArgsFromYAML_sets_fields);
   RUN_TEST(test_PreconSetArgsFromYAML_ignores_unknown_key);
   RUN_TEST(test_PreconDestroy_null_precon);
   RUN_TEST(test_PreconDestroy_null_main);
   RUN_TEST(test_PreconSetup_default_case);
   RUN_TEST(test_PreconApply_default_case);

   MPI_Finalize();
   return 0;
}
