#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "amg.h"
#include "containers.h"
#include "error.h"
#include "fsai.h"
#include "ilu.h"
#include "mgr.h"
#include "nested_krylov.h"
#include "precon.h"
#include "test_helpers.h"
#include "yaml.h"

/* Forward declarations for internal AMG functions */
void           AMGSetFieldByName(void *, const YAMLnode *);
void           AMGintSetFieldByName(void *, const YAMLnode *);
void           AMGintSetDefaultArgs(AMGint_args *);
StrIntMapArray AMGintGetValidValues(const char *);
void           AMGcsnSetFieldByName(void *, const YAMLnode *);
void           AMGcsnSetDefaultArgs(AMGcsn_args *);
StrIntMapArray AMGcsnGetValidValues(const char *);
StrIntMapArray AMGaggGetValidValues(const char *);
StrIntMapArray AMGrlxGetValidValues(const char *);
StrIntMapArray AMGsmtGetValidValues(const char *);

void           ILUSetFieldByName(void *, const YAMLnode *);
void           ILUSetDefaultArgs(ILU_args *);
StrArray       ILUGetValidKeys(void);
StrIntMapArray ILUGetValidValues(const char *);
void           FSAISetFieldByName(void *, const YAMLnode *);
void           FSAISetDefaultArgs(FSAI_args *);
StrArray       FSAIGetValidKeys(void);
StrIntMapArray FSAIGetValidValues(const char *);

void MGRSetDefaultArgs(MGR_args *);

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
   TEST_HYPRE_INIT();

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
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconApply_default_case(void)
{
   TEST_HYPRE_INIT();

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
   TEST_HYPRE_FINALIZE();
}

static void
test_MGRCreate_coarsest_level_branches(void)
{
   TEST_HYPRE_INIT();

   MGR_args mgr;
   MGRSetDefaultArgs(&mgr);
   mgr.num_levels = 1; /* minimal valid MGR setup (num_levels-1 == 0) */

   /* Minimal dofmap with global unique size set */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);
   MGRSetDofmap(&mgr, dofmap);

   /* 1) Explicit ILU coarsest solver type */
   mgr.coarsest_level.type = 32;
   ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   HYPRE_Solver precon = NULL;
   ErrorCodeResetAll();
   MGRCreate(&mgr, &precon);
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
   ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   precon                      = NULL;
   ErrorCodeResetAll();
   MGRCreate(&mgr, &precon);
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
   AMGSetDefaultArgs(&mgr.coarsest_level.amg);
   precon = NULL;
   ErrorCodeResetAll();
   MGRCreate(&mgr, &precon);
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
   ILUSetDefaultArgs(&mgr.coarsest_level.ilu);
   precon = NULL;
   ErrorCodeResetAll();
   MGRCreate(&mgr, &precon);
   ASSERT_NOT_NULL(precon);
   HYPRE_MGRDestroy(precon);
   if (mgr.csolver)
   {
 #if HYPRE_CHECK_MIN_VERSION(21900, 0)
      HYPRE_ILUDestroy(mgr.csolver);
 #endif
      mgr.csolver = NULL;
      mgr.csolver_type = -1;
   }

   IntArrayDestroy(&dofmap);
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
   PreconSetDefaultArgs(&args);
   MGRSetDefaultArgs(&args.mgr);

   args.mgr.num_levels = 1; /* minimal valid MGR setup (num_levels-1 == 0) */

   /* Minimal dofmap required by MGR */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.coarsest_level.use_krylov = 1;
   args.mgr.coarsest_level.krylov =
      (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
   ASSERT_NOT_NULL(args.mgr.coarsest_level.krylov);
   NestedKrylovSetDefaultArgs(args.mgr.coarsest_level.krylov);
   args.mgr.coarsest_level.krylov->is_set = 1;
   args.mgr.coarsest_level.krylov->solver_method = SOLVER_GMRES;
   SolverArgsSetDefaultsForMethod(SOLVER_GMRES, &args.mgr.coarsest_level.krylov->solver);
   args.mgr.coarsest_level.krylov->solver.gmres.max_iter = 2;
   args.mgr.coarsest_level.krylov->has_precon = 1;
   args.mgr.coarsest_level.krylov->precon_method = PRECON_BOOMERAMG;
   PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG,
                                  &args.mgr.coarsest_level.krylov->precon);
   args.mgr.coarsest_level.krylov->precon.amg.max_iter = 1;

   HYPRE_Precon precon = NULL;
   ErrorCodeResetAll();
   PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon);
   ASSERT_NOT_NULL(precon);

   ErrorCodeResetAll();
   PreconDestroy(PRECON_MGR, &args, &precon);
   ASSERT_NULL(precon);

   MGRDestroyNestedKrylovArgs(&args.mgr);
   IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
}

static void
test_PreconDestroy_mgr_csolver_destroy_branches(void)
{
   TEST_HYPRE_INIT();

   precon_args args;
   PreconSetDefaultArgs(&args);

   /* Minimal dofmap required by MGR */
   IntArray *dofmap = NULL;
   const int map[1] = {0};
   IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   args.mgr.num_levels = 1;

   /* Coarsest level ILU -> expect PreconDestroy to hit HYPRE_ILUDestroy(args.mgr.csolver) */
   args.mgr.coarsest_level.type = 32;
   ILUSetDefaultArgs(&args.mgr.coarsest_level.ilu);
   args.mgr.coarsest_level.ilu.max_iter = 1;

   HYPRE_Precon precon = NULL;
   ErrorCodeResetAll();
   PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon);
   ASSERT_NOT_NULL(precon);

   ErrorCodeResetAll();
   PreconDestroy(PRECON_MGR, &args, &precon);
   ASSERT_NULL(precon);

#if defined(HYPRE_USING_DSUPERLU)
   /* Coarsest level direct solver -> cover HYPRE_MGRDirectSolverDestroy branch when available */
   args.mgr.coarsest_level.type = 29;
   precon                      = NULL;
   ErrorCodeResetAll();
   PreconCreate(PRECON_MGR, &args, dofmap, NULL, &precon);
   ASSERT_NOT_NULL(precon);
   ErrorCodeResetAll();
   PreconDestroy(PRECON_MGR, &args, &precon);
   ASSERT_NULL(precon);
#endif

   IntArrayDestroy(&dofmap);
   TEST_HYPRE_FINALIZE();
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

/* AMG argument tests (merged from test_amg_args.c) */
static void
test_AMGSetFieldByName_all_fields(void)
{
   AMG_args args;
   AMGSetDefaultArgs(&args);

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
      AMGSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 5);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ_DOUBLE(args.tolerance, 1.0e-6, 1e-12);
}

static void
test_AMGSetFieldByName_unknown_key(void)
{
   AMG_args args;
   AMGSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   AMGSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - it just doesn't match any field */
   /* Verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_AMGintGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = AMGintGetValidValues("prolongation_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mod_classical"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "extended+i"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "one_point"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mod_classical"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "extended+i"), 6);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "one_point"), 100);
}

static void
test_AMGintGetValidValues_restriction_type(void)
{
   StrIntMapArray map = AMGintGetValidValues("restriction_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "p_transpose"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "air_1"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "air_1.5"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "p_transpose"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "air_1"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "air_1.5"), 15);
}

static void
test_AMGintGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGintGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGcsnGetValidValues_type(void)
{
   StrIntMapArray map = AMGcsnGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "cljp"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "falgout"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "hmis"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "cljp"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "falgout"), 6);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "hmis"), 10);
}

static void
test_AMGcsnGetValidValues_on_off_keys(void)
{
   StrIntMapArray map1 = AMGcsnGetValidValues("filter_functions");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map1, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map1, "off"));

   StrIntMapArray map2 = AMGcsnGetValidValues("nodal");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map2, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map2, "off"));

   StrIntMapArray map3 = AMGcsnGetValidValues("rap2");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map3, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map3, "off"));
}

static void
test_AMGcsnGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGcsnGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGaggGetValidValues_prolongation_type(void)
{
   StrIntMapArray map = AMGaggGetValidValues("prolongation_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "2_stage_extended+i"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "multipass"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mm_extended+e"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "2_stage_extended+i"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "multipass"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mm_extended+e"), 7);
}

static void
test_AMGaggGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGaggGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGrlxGetValidValues_down_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("down_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "jacobi_non_mv"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "chebyshev"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "l1sym-hgs"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "jacobi_non_mv"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "chebyshev"), 16);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "l1sym-hgs"), 89);
}

static void
test_AMGrlxGetValidValues_up_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("up_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "backward-hgs"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "cg"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "backward-hgs"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "cg"), 15);
}

static void
test_AMGrlxGetValidValues_coarse_type(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("coarse_type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "lu_piv"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "lu_inv"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "lu_piv"), 99);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "lu_inv"), 199);
}

static void
test_AMGrlxGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGrlxGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGsmtGetValidValues_type(void)
{
   StrIntMapArray map = AMGsmtGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "fsai"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "ilu"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "euclid"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "fsai"), 4);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "ilu"), 5);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "euclid"), 9);
}

static void
test_AMGsmtGetValidValues_unknown_key(void)
{
   StrIntMapArray map = AMGsmtGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_AMGintSetFieldByName_all_fields(void)
{
   AMGint_args args;
   AMGintSetDefaultArgs(&args);

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
      AMGintSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.prolongation_type, 8);
   ASSERT_EQ(args.restriction_type, 1);
   ASSERT_EQ(args.max_nnz_row, 6);
   ASSERT_EQ_DOUBLE(args.trunc_factor, 0.5, 1e-12);
}

static void
test_AMGcsnSetFieldByName_all_fields(void)
{
   AMGcsn_args args;
   AMGcsnSetDefaultArgs(&args);

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
      AMGcsnSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
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
   RUN_TEST(test_MGRCreate_coarsest_level_branches);
   RUN_TEST(test_PreconCreate_mgr_coarsest_level_krylov_nested);
   RUN_TEST(test_PreconDestroy_mgr_csolver_destroy_branches);
   RUN_TEST(test_ILUSetFieldByName_all_fields);
   RUN_TEST(test_ILUSetFieldByName_unknown_key);
   RUN_TEST(test_ILUGetValidValues_type);
   RUN_TEST(test_ILUGetValidValues_unknown_key);
   RUN_TEST(test_FSAISetFieldByName_all_fields);
   RUN_TEST(test_FSAISetFieldByName_unknown_key);
   RUN_TEST(test_FSAIGetValidValues_algo_type);
   RUN_TEST(test_FSAIGetValidValues_unknown_key);
   RUN_TEST(test_AMGSetFieldByName_all_fields);
   RUN_TEST(test_AMGSetFieldByName_unknown_key);
   RUN_TEST(test_AMGintGetValidValues_prolongation_type);
   RUN_TEST(test_AMGintGetValidValues_restriction_type);
   RUN_TEST(test_AMGintGetValidValues_unknown_key);
   RUN_TEST(test_AMGcsnGetValidValues_type);
   RUN_TEST(test_AMGcsnGetValidValues_on_off_keys);
   RUN_TEST(test_AMGcsnGetValidValues_unknown_key);
   RUN_TEST(test_AMGaggGetValidValues_prolongation_type);
   RUN_TEST(test_AMGaggGetValidValues_unknown_key);
   RUN_TEST(test_AMGrlxGetValidValues_down_type);
   RUN_TEST(test_AMGrlxGetValidValues_up_type);
   RUN_TEST(test_AMGrlxGetValidValues_coarse_type);
   RUN_TEST(test_AMGrlxGetValidValues_unknown_key);
   RUN_TEST(test_AMGsmtGetValidValues_type);
   RUN_TEST(test_AMGsmtGetValidValues_unknown_key);
   RUN_TEST(test_AMGintSetFieldByName_all_fields);
   RUN_TEST(test_AMGcsnSetFieldByName_all_fields);

   MPI_Finalize();
   return 0;
}
