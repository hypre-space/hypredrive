#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "internal/bicgstab.h"
#include "internal/cheby.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/fgmres.h"
#include "internal/gmres.h"
#include "internal/pcg.h"
#include "internal/precon.h"
#include "internal/solver.h"
#include "logging.h"
#include "test_helpers.h"
#include "internal/yaml.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

/* Forward declarations */
void           hypredrv_GMRESSetFieldByName(void *, const YAMLnode *);
void           hypredrv_GMRESSetDefaultArgs(GMRES_args *);
StrArray       hypredrv_GMRESGetValidKeys(void);
StrIntMapArray hypredrv_GMRESGetValidValues(const char *);
void           hypredrv_PCGSetFieldByName(void *, const YAMLnode *);
void           hypredrv_PCGSetDefaultArgs(PCG_args *);
StrIntMapArray hypredrv_PCGGetValidValues(const char *);
void           hypredrv_BiCGSTABSetFieldByName(void *, const YAMLnode *);
void           hypredrv_BiCGSTABSetDefaultArgs(BiCGSTAB_args *);
StrIntMapArray hypredrv_BiCGSTABGetValidValues(const char *);
void           hypredrv_FGMRESSetFieldByName(void *, const YAMLnode *);
void           hypredrv_FGMRESSetDefaultArgs(FGMRES_args *);
StrArray       hypredrv_FGMRESGetValidKeys(void);
StrIntMapArray hypredrv_FGMRESGetValidValues(const char *);
void           hypredrv_ChebySetFieldByName(void *, const YAMLnode *);
void           hypredrv_ChebySetDefaultArgs(Cheby_args *);
StrArray       hypredrv_ChebyGetValidKeys(void);
StrIntMapArray hypredrv_ChebyGetValidValues(const char *);

typedef struct
{
   const char *key;
   const char *value;
} keyval_pair;

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = hypredrv_YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

typedef void (*CapturedStreamFn)(void *);

static void
capture_stderr_output(CapturedStreamFn fn, void *context, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();
   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   fn(context);
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

static void
test_hypredrv_GMRESSetFieldByName_all_fields(void)
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
   hypredrv_GMRESSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_GMRESSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
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

   StrArray keys = hypredrv_GMRESGetValidKeys();
   ASSERT_EQ_SIZE(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, updates[i].key));
   }

   StrIntMapArray bool_map = hypredrv_GMRESGetValidValues("skip_real_res_check");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(bool_map, "off"));

   StrIntMapArray rel_change_map = hypredrv_GMRESGetValidValues("rel_change");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rel_change_map, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rel_change_map, "off"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = hypredrv_GMRESGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   /* Test else branch with another non-matching key */
   StrIntMapArray void_map2 = hypredrv_GMRESGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("gmres", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_YAMLnodeAddChild(parent, child);
   }

   GMRES_args args_from_yaml;
   hypredrv_GMRESSetArgs(&args_from_yaml, parent);
   hypredrv_YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.max_iter, 75);
   ASSERT_EQ_DOUBLE(args_from_yaml.conv_fac_tol, 0.25, 1e-12);
}

static void
test_hypredrv_PCGSetFieldByName_all_fields(void)
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
   hypredrv_PCGSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_PCGSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
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

   StrIntMapArray bool_map = hypredrv_PCGGetValidValues("two_norm");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(bool_map, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(bool_map, "off"));

   StrIntMapArray stop_crit_map = hypredrv_PCGGetValidValues("stop_crit");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(stop_crit_map, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(stop_crit_map, "off"));

   StrIntMapArray rel_change_map = hypredrv_PCGGetValidValues("rel_change");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rel_change_map, "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rel_change_map, "off"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = hypredrv_PCGGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   /* Test else branch with another non-matching key */
   StrIntMapArray void_map2 = hypredrv_PCGGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);

   StrIntMapArray void_map3 = hypredrv_PCGGetValidValues("relative_tol");
   ASSERT_EQ(void_map3.size, 0);
}

static void
test_hypredrv_BiCGSTABSetFieldByName_all_fields(void)
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
   hypredrv_BiCGSTABSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_BiCGSTABSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
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
   /* BiCGSTAB uses hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = hypredrv_BiCGSTABGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = hypredrv_BiCGSTABGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_FGMRESGetValidValues_void_branch(void)
{
   /* FGMRES uses hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = hypredrv_FGMRESGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = hypredrv_FGMRESGetValidValues("max_iter");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_hypredrv_ChebyGetValidValues_void_branch(void)
{
   /* Cheby uses hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC, so all keys return void */
   StrIntMapArray void_map1 = hypredrv_ChebyGetValidValues("unknown_key");
   ASSERT_EQ(void_map1.size, 0);

   StrIntMapArray void_map2 = hypredrv_ChebyGetValidValues("order");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_hypredrv_FGMRESSetFieldByName_all_fields(void)
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
   hypredrv_FGMRESSetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_FGMRESSetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.min_iter, 1);
   ASSERT_EQ(args.max_iter, 200);
   ASSERT_EQ(args.krylov_dim, 25);
   ASSERT_EQ(args.logging, 1);
   ASSERT_EQ(args.print_level, 2);
   ASSERT_EQ_DOUBLE(args.relative_tol, 1.0e-8, 1e-18);
   ASSERT_EQ_DOUBLE(args.absolute_tol, 0.1, 1e-12);

   StrArray keys = hypredrv_FGMRESGetValidKeys();
   ASSERT_EQ_SIZE(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, updates[i].key));
   }

   YAMLnode *parent = hypredrv_YAMLnodeCreate("fgmres", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_YAMLnodeAddChild(parent, child);
   }

   FGMRES_args args_from_yaml;
   hypredrv_FGMRESSetDefaultArgs(&args_from_yaml);
   hypredrv_FGMRESSetArgs(&args_from_yaml, parent);
   hypredrv_YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.max_iter, 200);
   ASSERT_EQ_DOUBLE(args_from_yaml.absolute_tol, 0.1, 1e-12);
}

static void
test_hypredrv_ChebySetFieldByName_all_fields(void)
{
   static const keyval_pair updates[] = {
      {.key = "order", .value = "3"},
      {.key = "eig_est", .value = "15"},
      {.key = "variant", .value = "1"},
      {.key = "scale", .value = "2"},
      {.key = "fraction", .value = "0.4"},
   };

   Cheby_args args;
   hypredrv_ChebySetDefaultArgs(&args);

   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *node = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_ChebySetFieldByName(&args, node);
      hypredrv_YAMLnodeDestroy(node);
   }

   ASSERT_EQ(args.order, 3);
   ASSERT_EQ(args.eig_est, 15);
   ASSERT_EQ(args.variant, 1);
   ASSERT_EQ(args.scale, 2);
   ASSERT_EQ_DOUBLE(args.fraction, 0.4, 1e-12);

   StrArray keys = hypredrv_ChebyGetValidKeys();
   ASSERT_EQ_SIZE(keys.size, sizeof(updates) / sizeof(updates[0]));
   for (size_t i = 0; i < keys.size; i++)
   {
      ASSERT_TRUE(hypredrv_StrArrayEntryExists(keys, updates[i].key));
   }

   YAMLnode *parent = hypredrv_YAMLnodeCreate("cheby", "", 0);
   for (size_t i = 0; i < sizeof(updates) / sizeof(updates[0]); i++)
   {
      YAMLnode *child = make_scalar_node(updates[i].key, updates[i].value);
      hypredrv_YAMLnodeAddChild(parent, child);
   }

   Cheby_args args_from_yaml;
   hypredrv_ChebySetDefaultArgs(&args_from_yaml);
   hypredrv_ChebySetArgs(&args_from_yaml, parent);
   hypredrv_YAMLnodeDestroy(parent);

   ASSERT_EQ(args_from_yaml.order, 3);
   ASSERT_EQ_DOUBLE(args_from_yaml.fraction, 0.4, 1e-12);
}

static void
test_hypredrv_GMRESSetFieldByName_unknown_key(void)
{
   GMRES_args args;
   hypredrv_GMRESSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_GMRESSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_PCGSetFieldByName_unknown_key(void)
{
   PCG_args args;
   hypredrv_PCGSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_PCGSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_BiCGSTABSetFieldByName_unknown_key(void)
{
   BiCGSTAB_args args;
   hypredrv_BiCGSTABSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_BiCGSTABSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_FGMRESSetFieldByName_unknown_key(void)
{
   FGMRES_args args;
   hypredrv_FGMRESSetDefaultArgs(&args);
   int original_max_iter = args.max_iter;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_FGMRESSetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.max_iter, original_max_iter);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

static void
test_hypredrv_ChebySetFieldByName_unknown_key(void)
{
   Cheby_args args;
   hypredrv_ChebySetDefaultArgs(&args);
   int original_order = args.order;

   YAMLnode *unknown_node = make_scalar_node("unknown_key", "value");
   hypredrv_ErrorCodeResetAll();
   hypredrv_ChebySetFieldByName(&args, unknown_node);

   /* SetFieldByName doesn't validate - verify args weren't modified */
   ASSERT_EQ(args.order, original_order);
   hypredrv_YAMLnodeDestroy(unknown_node);
}

/*-----------------------------------------------------------------------------
 * Solver dispatch tests (from test_solver_dispatch.c)
 *-----------------------------------------------------------------------------*/

static void
test_hypredrv_SolverCreate_all_cases(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG */
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES */
   hypredrv_GMRESSetDefaultArgs(&args.gmres);
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES */
   hypredrv_FGMRESSetDefaultArgs(&args.fgmres);
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB */
   hypredrv_BiCGSTABSetDefaultArgs(&args.bicgstab);
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverCreate_default_case(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test default case with invalid enum value */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, &solver);
   ASSERT_NULL(solver);
   /* Default case should not set error, just return NULL */

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverDestroy_all_cases(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG destroy */
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES destroy */
   hypredrv_GMRESSetDefaultArgs(&args.gmres);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   hypredrv_SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES destroy */
   hypredrv_FGMRESSetDefaultArgs(&args.fgmres);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   hypredrv_SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB destroy */
   hypredrv_BiCGSTABSetDefaultArgs(&args.bicgstab);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   hypredrv_SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverDestroy_default_case(void)
{
   TEST_HYPRE_INIT();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Create a solver first */
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);

   /* Test default case with invalid enum - should return early */
   hypredrv_SolverDestroy((solver_t)999, &solver);
   /* Solver should still exist since default case returns early */
   ASSERT_NOT_NULL(solver);

   /* Clean up properly */
   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverDestroy_null_solver(void)
{
   HYPRE_Solver solver = NULL;

   /* Destroy with NULL should not crash */
   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);
}

static void
test_hypredrv_SolverSetup_default_case(void)
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
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverSetup(PRECON_NONE, (solver_t)999, precon, solver, M, b, x, NULL);
   /* Should return early without error */

   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   free(precon);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(M);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverApply_default_case(void)
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
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverApply((solver_t)999, solver, A, b, x, NULL);
   /* Should return early without error */

   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverCreate_default_case_comprehensive(void)
{
   TEST_HYPRE_INIT();

   solver_args args;

   /* Test multiple invalid solver types to exercise default case */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive()); /* Should set error for invalid solver */

   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, (solver_t)-1, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive()); /* Should set error for invalid solver */

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverApply_error_cases(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverApply(SOLVER_PCG, NULL, A, b, x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   /* Test with NULL matrix */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverApply(SOLVER_PCG, (HYPRE_Solver)1, NULL, b, x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   /* Test with NULL vectors */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, NULL, x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, b, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_SolverSetup_error_cases(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, NULL, A, b, x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   /* Test with NULL matrix */
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, (HYPRE_Solver)1, NULL, b, x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   TEST_HYPRE_FINALIZE();
}

struct SolverLogContext
{
   HYPRE_Solver   solver;
   HYPRE_IJMatrix A;
   HYPRE_IJVector b;
   HYPRE_IJVector x;
};

static void
run_solver_failure_logging_capture(void *context)
{
   struct SolverLogContext *log_context = (struct SolverLogContext *)context;

   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverSetup(PRECON_NONE, (solver_t)999, NULL, log_context->solver,
                        log_context->A, log_context->b, log_context->x, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_SolverSolveOnly((solver_t)999, log_context->solver, log_context->A,
                                      log_context->b, log_context->x),
             -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   solver_args args;
   hypredrv_ErrorCodeResetAll();
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_hypredrv_solver_failure_paths_emit_logs(void)
{
   TEST_HYPRE_INIT();

   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   hypredrv_LogInitializeFromEnv();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;
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
   hypredrv_PCGSetDefaultArgs(&args.pcg);
   hypredrv_SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);

   struct SolverLogContext context = {solver, A, b, x};
   char                    output[16384];
   capture_stderr_output(run_solver_failure_logging_capture, &context, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "solver setup failed: invalid solver method=999"));
   ASSERT_NOT_NULL(strstr(output, "solver solve failed: invalid solver method=999"));
   ASSERT_NOT_NULL(strstr(output, "solver create failed: solver_ptr is NULL"));

   hypredrv_SolverDestroy(SOLVER_PCG, &solver);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
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
   hypredrv_ErrorCodeResetAll();
   HYPREDRV_InputArgsParse(1, argv, obj);

   if (hypredrv_ErrorCodeActive())
   {
      HYPREDRV_Destroy(&obj);
      HYPREDRV_Finalize();
      return; /* Skip invalid combinations */
   }

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
   RUN_TEST(test_hypredrv_GMRESSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_PCGSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_BiCGSTABSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_FGMRESSetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_ChebySetFieldByName_all_fields);
   RUN_TEST(test_hypredrv_GMRESSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_PCGSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_BiCGSTABSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_FGMRESSetFieldByName_unknown_key);
   RUN_TEST(test_hypredrv_ChebySetFieldByName_unknown_key);
   RUN_TEST(test_BiCGSTABGetValidValues_void_branch);
   RUN_TEST(test_FGMRESGetValidValues_void_branch);
   RUN_TEST(test_hypredrv_ChebyGetValidValues_void_branch);

   /* Solver dispatch tests */
   RUN_TEST(test_hypredrv_SolverCreate_all_cases);
   RUN_TEST(test_hypredrv_SolverCreate_default_case);
   RUN_TEST(test_hypredrv_SolverDestroy_all_cases);
   RUN_TEST(test_hypredrv_SolverDestroy_default_case);
   RUN_TEST(test_hypredrv_SolverDestroy_null_solver);
   RUN_TEST(test_hypredrv_SolverSetup_default_case);
   RUN_TEST(test_hypredrv_SolverApply_default_case);
   RUN_TEST(test_hypredrv_SolverCreate_default_case_comprehensive);
   RUN_TEST(test_hypredrv_SolverApply_error_cases);
   RUN_TEST(test_hypredrv_SolverSetup_error_cases);
   RUN_TEST(test_hypredrv_solver_failure_paths_emit_logs);

   /* Solver-precon integration tests */
   RUN_TEST(test_all_solver_precon_combinations);

   MPI_Finalize();
   return 0;
}
