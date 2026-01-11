#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "containers.h"
#include "error.h"
#include "fsai.h"
#include "ilu.h"
#include "precon.h"
#include "test_helpers.h"
#include "yaml.h"

void           ILUSetFieldByName(ILU_args *, const YAMLnode *);
void           ILUSetDefaultArgs(ILU_args *);
StrArray       ILUGetValidKeys(void);
StrIntMapArray ILUGetValidValues(const char *);
void           FSAISetFieldByName(FSAI_args *, const YAMLnode *);
void           FSAISetDefaultArgs(FSAI_args *);
StrArray       FSAIGetValidKeys(void);
StrIntMapArray FSAIGetValidValues(const char *);

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = YAMLnodeCreate(key, val, level);
   YAMLnodeAddChild(parent, child);
   return child;
}

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
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
test_PreconSetArgsFromYAML_mgr_coarsest_level_spdirect_flat(void)
{
   precon_args args;
   PreconSetDefaultArgs(&args);

   YAMLnode *parent   = YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   add_child(mgr_node, "coarsest_level", "spdirect", 2);

   ErrorCodeResetAll();
   PreconSetArgsFromYAML(&args, parent);

   /* Flat value must map to type=29 (spdirect) */
   ASSERT_EQ(args.mgr.coarsest_level.type, 29);

   YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_flat_sets_type(void)
{
   precon_args args;
   PreconSetDefaultArgs(&args);

   YAMLnode *parent   = YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   add_child(mgr_node, "coarsest_level", "ilu", 2);

   ErrorCodeResetAll();
   PreconSetArgsFromYAML(&args, parent);

   /* Flat value must map to MGR coarsest ILU type=32 */
   ASSERT_EQ(args.mgr.coarsest_level.type, 32);

   YAMLnodeDestroy(parent);
}

static void
test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_nested_sets_type_and_args(void)
{
   precon_args args;
   PreconSetDefaultArgs(&args);

   YAMLnode *parent   = YAMLnodeCreate("preconditioner", "", 0);
   YAMLnode *mgr_node = add_child(parent, "mgr", "", 1);
   YAMLnode *cls_node = add_child(mgr_node, "coarsest_level", "", 2);
   YAMLnode *ilu_node = add_child(cls_node, "ilu", "", 3);
   add_child(ilu_node, "type", "bj-ilut", 4);

   ErrorCodeResetAll();
   PreconSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.mgr.coarsest_level.type, 32);
   /* ILU type bj-ilut should map to 1 */
   ASSERT_EQ(args.mgr.coarsest_level.ilu.type, 1);

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

static void
test_ILUSetFieldByName_all_fields(void)
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
   ILUSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      ILUSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
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
test_ILUSetFieldByName_unknown_key(void)
{
   ILU_args args;
   ILUSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   ILUSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_ILUGetValidValues_type(void)
{
   StrIntMapArray map = ILUGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "bj-iluk"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "bj-ilut"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "gmres-iluk"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "rap-mod-ilu0"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "bj-iluk"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "bj-ilut"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "gmres-iluk"), 10);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "rap-mod-ilu0"), 50);
}

static void
test_ILUGetValidValues_unknown_key(void)
{
   StrIntMapArray map = ILUGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_FSAISetFieldByName_all_fields(void)
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
   FSAISetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      FSAISetFieldByName(&args, node);
      YAMLnodeDestroy(node);
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
test_FSAISetFieldByName_unknown_key(void)
{
   FSAI_args args;
   FSAISetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   FSAISetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_FSAIGetValidValues_algo_type(void)
{
   StrIntMapArray map = FSAIGetValidValues("algo_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "bj-afsai"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "bj-afsai-omp"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "bj-sfsai"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "bj-afsai"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "bj-afsai-omp"), 2);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "bj-sfsai"), 3);
}

static void
test_FSAIGetValidValues_unknown_key(void)
{
   StrIntMapArray map = FSAIGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
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
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_spdirect_flat);
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_flat_sets_type);
   RUN_TEST(test_PreconSetArgsFromYAML_mgr_coarsest_level_ilu_nested_sets_type_and_args);
   RUN_TEST(test_PreconDestroy_null_precon);
   RUN_TEST(test_PreconDestroy_null_main);
   RUN_TEST(test_PreconSetup_default_case);
   RUN_TEST(test_PreconApply_default_case);
   RUN_TEST(test_ILUSetFieldByName_all_fields);
   RUN_TEST(test_ILUSetFieldByName_unknown_key);
   RUN_TEST(test_ILUGetValidValues_type);
   RUN_TEST(test_ILUGetValidValues_unknown_key);
   RUN_TEST(test_FSAISetFieldByName_all_fields);
   RUN_TEST(test_FSAISetFieldByName_unknown_key);
   RUN_TEST(test_FSAIGetValidValues_algo_type);
   RUN_TEST(test_FSAIGetValidValues_unknown_key);

   MPI_Finalize();
   return 0;
}
