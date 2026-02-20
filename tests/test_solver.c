#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "bicgstab.h"
#include "cheby.h"
#include "containers.h"
#include "error.h"
#include "fgmres.h"
#include "gmres.h"
#include "pcg.h"
#include "precon.h"
#include "solver.h"
#include "test_helpers.h"
#include "yaml.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

/* Forward declarations */
void           GMRESSetFieldByName(void *, const YAMLnode *);
void           GMRESSetDefaultArgs(GMRES_args *);
StrArray       GMRESGetValidKeys(void);
StrIntMapArray GMRESGetValidValues(const char *);
void           PCGSetFieldByName(void *, const YAMLnode *);
void           PCGSetDefaultArgs(PCG_args *);
StrIntMapArray PCGGetValidValues(const char *);
void           BiCGSTABSetFieldByName(void *, const YAMLnode *);
void           BiCGSTABSetDefaultArgs(BiCGSTAB_args *);
StrIntMapArray BiCGSTABGetValidValues(const char *);
void           FGMRESSetFieldByName(void *, const YAMLnode *);
void           FGMRESSetDefaultArgs(FGMRES_args *);
StrArray       FGMRESGetValidKeys(void);
StrIntMapArray FGMRESGetValidValues(const char *);
void           ChebySetFieldByName(void *, const YAMLnode *);
void           ChebySetDefaultArgs(Cheby_args *);
StrArray       ChebyGetValidKeys(void);
StrIntMapArray ChebyGetValidValues(const char *);

typedef struct
{
   const char *key;
   const char *value;
} keyval_pair;

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static void
test_GMRESSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "min_iter", .value = "2"},
      {.key = "max_iter", .value = "75"},
      {.key = "stop_crit", .value = "1"},
      {.key = "skip_real_res_check", .value = "1"},
      {.key = "krylov_dim", .value = "40"},
      {.key = "rel_change", .value = "1"},
      {.key = "logging", .value = "0"},
      {.key = "print_level", .value = "3"},
      {.key = "relative_tol", .value = "1.0e-7"},
      {.key = "absolute_tol", .value = "0.5"},
      {.key = "conv_fac_tol", .value = "0.25"},
   };

   GMRES_args args;
   GMRESSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      GMRESSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 2);
   ASSERT_EQ(args.max_iter, 75);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.skip_real_res_check, 1);
   ASSERT_EQ(args.krylov_dim, 40);
   ASSERT_EQ(args.rel_change, 1);
   ASSERT_EQ(args.logging, 0);
   ASSERT_EQ(args.print_level, 3);
   ASSERT_EQ_DOUBLE(args.relative_tol, 1.0e-7, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.5, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.25, 1e-12);

   StrArray keys = GMRESGetValidKeys();
   ASSERT_EQ(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(StrArrayEntryExists(keys, updates[i].key));
   }

   StrIntMapArray bool_map = GMRESGetValidValues("skip_real_res_check");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "off"));

   StrIntMapArray rel_change_map = GMRESGetValidValues("rel_change");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rel_change_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rel_change_map, "off"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = GMRESGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   /* Test else branch with another non-matching key */
   StrIntMapArray void_map2 = GMRESGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);

   YAMLnode *parent = YAMLnodeCreate("gmres", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      YAMLnodeAddChild(parent, child);
   }

   GMRES_args args_from_yaml;
   GMRESSetArgs(&args_from_yaml, parent);
   YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.max_iter, 75);
   ASSERT_EQ_DOUBLE(args_from_yaml.conv_fac_tol, 0.25, 1e-12);
}

static void
test_PCGSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "max_iter", .value = "150"},
      {.key = "two_norm", .value = "0"},
      {.key = "stop_crit", .value = "1"},
      {.key = "rel_change", .value = "1"},
      {.key = "print_level", .value = "2"},
      {.key = "recompute_res", .value = "1"},
      {.key = "relative_tol", .value = "9.0e-8"},
      {.key = "absolute_tol", .value = "0.75"},
      {.key = "residual_tol", .value = "0.33"},
      {.key = "conv_fac_tol", .value = "0.12"},
   };

   PCG_args args;
   PCGSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      PCGSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.max_iter, 150);
   ASSERT_EQ(args.two_norm, 0);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.rel_change, 1);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ(args.recompute_res, 1);
   ASSERT_EQ_DOUBLE(args.relative_tol, 9.0e-8, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.75, 1e-12);
   ASSERT_EQ_DOUBLE(args.residual_tol, 0.33, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.12, 1e-12);

   StrIntMapArray bool_map = PCGGetValidValues("two_norm");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(bool_map, "off"));

   StrIntMapArray stop_crit_map = PCGGetValidValues("stop_crit");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(stop_crit_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(stop_crit_map, "off"));

   StrIntMapArray rel_change_map = PCGGetValidValues("rel_change");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rel_change_map, "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rel_change_map, "off"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = PCGGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   /* Test else branch with another non-matching key */
   StrIntMapArray void_map2 = PCGGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);

   StrIntMapArray void_map3 = PCGGetValidValues("relative_tol");
   ASSERT_EQ(void_map3.size, 0);
}

static void
test_BiCGSTABSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "min_iter", .value = "3"},
      {.key = "max_iter", .value = "120"},
      {.key = "stop_crit", .value = "1"},
      {.key = "logging", .value = "0"},
      {.key = "print_level", .value = "4"},
      {.key = "relative_tol", .value = "5.0e-7"},
      {.key = "absolute_tol", .value = "0.21"},
      {.key = "conv_fac_tol", .value = "0.09"},
   };

   BiCGSTAB_args args;
   BiCGSTABSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      BiCGSTABSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 3);
   ASSERT_EQ(args.max_iter, 120);
   ASSERT_EQ(args.stop_crit, 1);
   ASSERT_EQ(args.logging, 0);
   ASSERT_EQ(args.print_level, 4);
   ASSERT_EQ_DOUBLE(args.relative_tol, 5.0e-7, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.21, 1e-12);
   ASSERT_EQ_DOUBLE(args.conv_fac_tol, 0.09, 1e-12);
}

static void
test_BiCGSTABGetValidValues_void_branch(void)
{
   /* BiCGSTAB uses DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = BiCGSTABGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = BiCGSTABGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_FGMRESGetValidValues_void_branch(void)
{
   /* FGMRES uses DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = FGMRESGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = FGMRESGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_ChebyGetValidValues_void_branch(void)
{
   /* Cheby uses DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = ChebyGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = ChebyGetValidValues("order");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_FGMRESSetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "min_iter", .value = "1"},
      {.key = "max_iter", .value = "200"},
      {.key = "krylov_dim", .value = "25"},
      {.key = "logging", .value = "1"},
      {.key = "print_level", .value = "2"},
      {.key = "relative_tol", .value = "1.0e-8"},
      {.key = "absolute_tol", .value = "0.1"},
   };

   FGMRES_args args;
   FGMRESSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      FGMRESSetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 1);
   ASSERT_EQ(args.max_iter, 200);
   ASSERT_EQ(args.krylov_dim, 25);
   ASSERT_EQ(args.logging, 1);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ_DOUBLE(args.relative_tol, 1.0e-8, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.1, 1e-12);

   StrArray keys = FGMRESGetValidKeys();
   ASSERT_EQ(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(StrArrayEntryExists(keys, updates[i].key));
   }

   YAMLnode *parent = YAMLnodeCreate("fgmres", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      YAMLnodeAddChild(parent, child);
   }

   FGMRES_args args_from_yaml;
   FGMRESSetDefaultArgs(&args_from_yaml);
   FGMRESSetArgs(&args_from_yaml, parent);
   YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.max_iter, 200);
   ASSERT_EQ_DOUBLE(args_from_yaml.absolute_tol, 0.1, 1e-12);
}

static void
test_ChebySetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "order", .value = "3"},
      {.key = "eig_est", .value = "15"},
      {.key = "variant", .value = "1"},
      {.key = "scale", .value = "2"},
      {.key = "fraction", .value = "0.4"},
   };

   Cheby_args args;
   ChebySetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      ChebySetFieldByName(&args, node);
      YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.order, 3);
   ASSERT_EQ(args.eig_est, 15);
   ASSERT_EQ(args.variant, 1);
   ASSERT_EQ(args.scale, 2);
   ASSERT_EQ_DOUBLE(args.fraction, 0.4, 1e-12);

   StrArray keys = ChebyGetValidKeys();
   ASSERT_EQ(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(StrArrayEntryExists(keys, updates[i].key));
   }

   YAMLnode *parent = YAMLnodeCreate("cheby", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      YAMLnodeAddChild(parent, child);
   }

   Cheby_args args_from_yaml;
   ChebySetDefaultArgs(&args_from_yaml);
   ChebySetArgs(&args_from_yaml, parent);
   YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.order, 3);
   ASSERT_EQ_DOUBLE(args_from_yaml.fraction, 0.4, 1e-12);
}

static void
test_GMRESSetFieldByName_unknown_key(void)
{
   GMRES_args args;
   GMRESSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   GMRESSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_PCGSetFieldByName_unknown_key(void)
{
   PCG_args args;
   PCGSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   PCGSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_BiCGSTABSetFieldByName_unknown_key(void)
{
   BiCGSTAB_args args;
   BiCGSTABSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   BiCGSTABSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_FGMRESSetFieldByName_unknown_key(void)
{
   FGMRES_args args;
   FGMRESSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   FGMRESSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   YAMLnodeDestroy(unknown_node);
}

static void
test_ChebySetFieldByName_unknown_key(void)
{
   Cheby_args args;
   ChebySetDefaultArgs(&args);
   int original_order = args.order;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   ErrorCodeResetAll();
   ChebySetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.order, original_order);
   YAMLnodeDestroy(unknown_node);
}

/*-----------------------------------------------------------------------------
 * Solver dispatch tests (from test_solver_dispatch.c)
 *-----------------------------------------------------------------------------*/

static void
test_SolverCreate_all_cases(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG */
   PCGSetDefaultArgs(&args.pcg);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES */
   GMRESSetDefaultArgs(&args.gmres);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES */
   FGMRESSetDefaultArgs(&args.fgmres);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB */
   BiCGSTABSetDefaultArgs(&args.bicgstab);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverCreate_default_case(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test default case with invalid enum value */
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, &solver);
   ASSERT_NULL(solver);
   /* Default case should not set error, just return NULL */

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverDestroy_all_cases(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG destroy */
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES destroy */
   GMRESSetDefaultArgs(&args.gmres);
   SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES destroy */
   FGMRESSetDefaultArgs(&args.fgmres);
   SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB destroy */
   BiCGSTABSetDefaultArgs(&args.bicgstab);
   SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverDestroy_default_case(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Create a solver first */
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);

   /* Test default case with invalid enum - should return early */
   SolverDestroy((solver_t)999, &solver);
   /* Solver should still exist since default case returns early */
   ASSERT_NOT_NULL(solver);

   /* Clean up properly */
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverDestroy_null_solver(void)
{
   HYPRE_Solver solver = NULL;

   /* Destroy with NULL should not crash */
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);
}

static void
test_SolverSetup_default_case(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix M = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Create minimal matrix and vectors */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &M);
   HYPRE_IJMatrixSetObjectType(M, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(M);
   HYPRE_IJMatrixAssemble(M);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);

   /* Create a dummy precon */
   HYPRE_Precon precon = malloc(sizeof(hypre_Precon));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   HYPRE_Solver solver = NULL;
   solver_args  args;
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, (solver_t)999, precon, solver, M, b, x, NULL);
   /* Should return early without error */

   SolverDestroy(SOLVER_PCG, &solver);
   free(precon);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(M);
   TEST_HYPRE_FINALIZE();
}

static void
test_SolverApply_default_case(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Create minimal matrix and vectors */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);
   HYPRE_IJMatrixAssemble(A);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);

   HYPRE_Solver solver = NULL;
   solver_args  args;
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   ErrorCodeResetAll();
   SolverApply((solver_t)999, solver, A, b, x, NULL);
   /* Should return early without error */

   SolverDestroy(SOLVER_PCG, &solver);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);
   TEST_HYPRE_FINALIZE();
}

static void
test_SolverCreate_default_case_comprehensive(void)
{
   TEST_HYPRE_INIT();

   solver_args args;

   /* Test multiple invalid solver types to exercise default case */
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, NULL);
   ASSERT_TRUE(ErrorCodeActive()); /* Should set error for invalid solver */

   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)-1, &args, NULL);
   ASSERT_TRUE(ErrorCodeActive()); /* Should set error for invalid solver */

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverApply_error_cases(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, NULL, A, b, x, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL matrix */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, NULL, b, x, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL vectors */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, NULL, x, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, b, NULL, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   TEST_HYPRE_FINALIZE();
}

static void
test_SolverSetup_error_cases(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, NULL, A, b, x, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL matrix */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, (HYPRE_Solver)1, NULL, b, x, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   TEST_HYPRE_FINALIZE();
}

/*-----------------------------------------------------------------------------
 * Solver-precon integration tests (from test_solver_precon_integration.c)
 *-----------------------------------------------------------------------------*/

static void
test_solver_precon_combination(const char *solver_name, const char *precon_name)
{
   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   snprintf(matrix_path, sizeof(matrix_path), "%s/data/ps3d10pt7/np1/IJ.out.A",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, sizeof(rhs_path), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);
   TEST_REQUIRE_FILE(matrix_path);
   TEST_REQUIRE_FILE(rhs_path);

   HYPREDRV_Initialize();

   HYPREDRV_t obj = NULL;
   HYPREDRV_Create(MPI_COMM_SELF, &obj);

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver: %s\n"
            "preconditioner: %s\n",
            matrix_path, rhs_path, solver_name, precon_name);

   char *argv[] = {yaml_config};
   ErrorCodeResetAll();
   HYPREDRV_InputArgsParse(1, argv, obj);

   if (ErrorCodeActive())
   {
      HYPREDRV_Destroy(&obj);
      HYPREDRV_Finalize();
      return; /* Skip invalid combinations */
   }

   HYPREDRV_SetGlobalOptions(obj);
   HYPREDRV_LinearSystemBuild(obj);

   /* Test create/setup/apply/destroy cycle */
   HYPREDRV_PreconCreate(obj);
   HYPREDRV_LinearSolverCreate(obj);
   HYPREDRV_LinearSolverSetup(obj);
   HYPREDRV_LinearSolverApply(obj);
   HYPREDRV_PreconDestroy(obj);
   HYPREDRV_LinearSolverDestroy(obj);

   HYPREDRV_Destroy(&obj);
   HYPREDRV_Finalize();
}

static void
test_all_solver_precon_combinations(void)
{
   const char *solvers[] = {"pcg", "gmres", "fgmres", "bicgstab"};
   const char *precons[] = {"amg", "ilu", "fsai"};

   for (size_t i = 0; i < sizeof(solvers) / sizeof(solvers[0]); i++)
   {
      for (size_t j = 0; j < sizeof(precons) / sizeof(precons[0]); j++)
      {
         test_solver_precon_combination(solvers[i], precons[j]);
      }
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   /* Solver argument parsing tests */
   RUN_TEST(test_GMRESSetFieldByName_all_fields);
   RUN_TEST(test_PCGSetFieldByName_all_fields);
   RUN_TEST(test_BiCGSTABSetFieldByName_all_fields);
   RUN_TEST(test_FGMRESSetFieldByName_all_fields);
   RUN_TEST(test_ChebySetFieldByName_all_fields);
   RUN_TEST(test_GMRESSetFieldByName_unknown_key);
   RUN_TEST(test_PCGSetFieldByName_unknown_key);
   RUN_TEST(test_BiCGSTABSetFieldByName_unknown_key);
   RUN_TEST(test_FGMRESSetFieldByName_unknown_key);
   RUN_TEST(test_ChebySetFieldByName_unknown_key);
   RUN_TEST(test_BiCGSTABGetValidValues_void_branch);
   RUN_TEST(test_FGMRESGetValidValues_void_branch);
   RUN_TEST(test_ChebyGetValidValues_void_branch);

   /* Solver dispatch tests */
   RUN_TEST(test_SolverCreate_all_cases);
   RUN_TEST(test_SolverCreate_default_case);
   RUN_TEST(test_SolverDestroy_all_cases);
   RUN_TEST(test_SolverDestroy_default_case);
   RUN_TEST(test_SolverDestroy_null_solver);
   RUN_TEST(test_SolverSetup_default_case);
   RUN_TEST(test_SolverApply_default_case);
   RUN_TEST(test_SolverCreate_default_case_comprehensive);
   RUN_TEST(test_SolverApply_error_cases);
   RUN_TEST(test_SolverSetup_error_cases);

   /* Solver-precon integration tests */
   RUN_TEST(test_all_solver_precon_combinations);

   MPI_Finalize();
   return 0;
}
