/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/precon.h"
#include "internal/solver.h"
#include "internal/yaml.h"
#include "test_helpers.h"
#include "test_ij_helpers.h"

static YAMLnode *
add_yaml_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static void
configure_nested_solver(NestedKrylov_args *nk, solver_t method)
{
   hypredrv_NestedKrylovSetDefaultArgs(nk);
   nk->is_set        = 1;
   nk->solver_method = method;
   hypredrv_SolverArgsSetDefaultsForMethod(method, &nk->solver);
   switch (method)
   {
      case SOLVER_PCG:
         nk->solver.pcg.max_iter = 20;
         break;
      case SOLVER_GMRES:
         nk->solver.gmres.max_iter = 20;
         break;
      case SOLVER_FGMRES:
         nk->solver.fgmres.max_iter = 20;
         break;
      case SOLVER_BICGSTAB:
         nk->solver.bicgstab.max_iter = 20;
         break;
      default:
         break;
   }
}

static HYPREDRV_MAYBE_UNUSED void
configure_nested_matched_schur(NestedKrylov_args *nk, solver_t method)
{
   configure_nested_solver(nk, method);
   nk->has_precon    = 1;
   nk->precon_method = PRECON_MGR;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &nk->precon);

   MGR_args *mgr = &nk->precon.mgr;
   mgr->num_levels = 2;
   mgr->level[0].f_dofs.size = 1;
   mgr->level[0].f_dofs.data[0] = 0;
   mgr->level[0].prolongation_type = 2;
   mgr->level[0].restriction_type = 0;
   mgr->level[0].coarse_level_type = 0;
   mgr->level[0].matched_f_backsolve = 2;
   mgr->level[0].f_relaxation.type = 2;
   mgr->level[0].f_relaxation.num_sweeps = 1;
   mgr->level[0].f_relaxation.symmetric_diagonal_scaling = 1;
   mgr->level[0].f_relaxation.amg.max_iter = 1;
   mgr->level[0].f_relaxation.amg.tolerance = 0.0;
   mgr->level[0].g_relaxation.type = -1;
   mgr->coarsest_level.type = 0;
   mgr->coarsest_level.amg.max_iter = 1;
   mgr->coarsest_level.amg.tolerance = 0.0;
}

static void
run_nested_lifecycle(solver_t method, int use_precon_amg)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, method);

   if (use_precon_amg)
   {
      nk.has_precon    = 1;
      nk.precon_method = PRECON_BOOMERAMG;
      hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG, &nk.precon);
      nk.precon.amg.max_iter = 1;
   }

   IntArray *dofmap = NULL;

   if (use_precon_amg)
   {
      const int map[1] = {0};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
   }

   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL));
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);
   ASSERT_FALSE(nk.base.is_setup);
   HYPRE_PtrToSolverFcn setup = hypredrv_NestedKrylovSetup;
   HYPRE_PtrToSolverFcn solve = hypredrv_NestedKrylovSolve;
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
   hypre_Solver *native = (hypre_Solver *)&nk;
   setup                = hypre_SolverSetup(native);
   solve                = hypre_SolverSolve(native);
#endif

   HYPRE_IJMatrix ij_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_test_ijvector_1x1(0.0);

   void *par_A = NULL, *par_b = NULL, *par_x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &par_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &par_x), 0);

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(solve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A, (HYPRE_Vector)par_b,
                   (HYPRE_Vector)par_x),
             0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_FALSE(nk.base.is_setup);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(setup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A, (HYPRE_Vector)par_b,
                   (HYPRE_Vector)par_x),
             0);
   ASSERT_TRUE(nk.base.is_setup);

   const void *original_runtime = nk.runtime;
   ASSERT_TRUE(hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL));
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_PTR_EQ(nk.runtime, original_runtime);
   ASSERT_TRUE(nk.base.is_setup);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(solve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A, (HYPRE_Vector)par_b,
                   (HYPRE_Vector)par_x),
             0);

   /* Native execution and destruction use the operations bound at creation,
    * even if pending configuration no longer names that solver method. */
   nk.solver_method = (solver_t)999;
   ASSERT_TRUE(hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL));
   ASSERT_PTR_EQ(nk.runtime, original_runtime);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)par_x, 0.0), 0);
   ASSERT_EQ(solve((HYPRE_Solver)&nk, NULL, (HYPRE_Vector)par_b, (HYPRE_Vector)par_x), 0);
   HYPRE_BigInt index = 0;
   double       value = 0.0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(ij_x, 1, &index, &value), 0);
   ASSERT_EQ_DOUBLE(value, 0.25, 1e-10);
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
   ASSERT_EQ(hypre_SolverDestroy(native)((HYPRE_Solver)&nk), 0);
#else
   hypredrv_NestedKrylovDestroy(&nk);
#endif
   ASSERT_NULL(nk.runtime);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_FALSE(hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_SOLVER);
   ASSERT_NULL(nk.runtime);
   hypredrv_ErrorCodeResetAll();

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);

   hypredrv_IntArrayDestroy(&dofmap);
}

static void
test_nested_krylov_null_and_invalid_paths(void)
{
   hypredrv_NestedKrylovSetDefaultArgs(NULL);

   /* hypredrv_NestedKrylovSetArgsFromYAML(NULL, node): hits !args branch. */
   {
      YAMLnode *solver = hypredrv_YAMLnodeCreate("gmres", "", 0);
      add_yaml_child(solver, "max_iter", "2", 1);
      hypredrv_NestedKrylovSetArgsFromYAML(NULL, solver);
      hypredrv_YAMLnodeDestroy(solver);
   }

   NestedKrylov_args nk;
   hypredrv_NestedKrylovSetDefaultArgs(&nk);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

   nk.is_set = 0;
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(nk.runtime);

   hypredrv_NestedKrylovSetDefaultArgs(&nk);
   nk.is_set = 1;
   nk.solver_method = SOLVER_GMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_GMRES, &nk.solver);
   nk.solver.gmres.max_iter = 5;
   /* No runtime has been created. */
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, NULL, NULL, NULL), 0);
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, NULL, NULL, NULL), 0);

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSetup(NULL, NULL, NULL, NULL), 0);
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSolve(NULL, NULL, NULL, NULL), 0);

   /* Reject an invalid method before creating its otherwise valid preconditioner. */
   configure_nested_solver(&nk, (solver_t)999);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_BOOMERAMG;
   hypredrv_PreconArgsSetDefaultsForMethod(nk.precon_method, &nk.precon);
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_SOLVER);
   ASSERT_NULL(nk.runtime);
   hypredrv_ErrorCodeResetAll();

   hypredrv_NestedKrylovDestroy(NULL);
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
run_nested_lifecycle_precon_ilu(solver_t method)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, method);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_ILU;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_ILU, &nk.precon);

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_IntArrayDestroy(&dofmap);
      return;
   }
   ASSERT_NOT_NULL(nk.runtime);

   HYPRE_IJMatrix ij_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_test_ijvector_1x1(0.0);

   void *par_A = NULL, *par_b = NULL, *par_x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &par_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &par_x), 0);

   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);
   ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);

   hypredrv_NestedKrylovDestroy(&nk);

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
   hypredrv_IntArrayDestroy(&dofmap);
}
#endif

#if HYPRE_CHECK_MIN_VERSION(30100, 1)
static void
run_nested_lifecycle_precon_fsai(solver_t method)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, method);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_FSAI;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_FSAI, &nk.precon);

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_IntArrayDestroy(&dofmap);
      return;
   }
   ASSERT_NOT_NULL(nk.runtime);

   HYPRE_IJMatrix ij_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_test_ijvector_1x1(0.0);

   void *par_A = NULL, *par_b = NULL, *par_x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &par_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &par_x), 0);

   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);
   ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);

   hypredrv_NestedKrylovDestroy(&nk);

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
   hypredrv_IntArrayDestroy(&dofmap);
}
#endif

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static void
run_nested_lifecycle_precon_schwarz(solver_t method)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, method);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_SCHWARZ;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_SCHWARZ, &nk.precon);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);

   HYPRE_IJMatrix ij_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_test_ijvector_1x1(0.0);

   void *par_A = NULL, *par_b = NULL, *par_x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &par_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &par_x), 0);

   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);
   ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);

   hypredrv_NestedKrylovDestroy(&nk);
   ASSERT_NULL(nk.runtime);

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
}
#endif

/* Every solver method must use its own YAML schema through the shared table. */
static void
test_nested_krylov_solver_method_parsing(void)
{
   const char *names[] = {"pcg", "gmres", "fgmres", "bicgstab"};
   const solver_t methods[] = {SOLVER_PCG, SOLVER_GMRES, SOLVER_FGMRES, SOLVER_BICGSTAB};
   for (size_t i = 0; i < sizeof(methods) / sizeof(methods[0]); i++)
   {
      NestedKrylov_args nk;
      hypredrv_NestedKrylovSetDefaultArgs(&nk);
      YAMLnode *node = hypredrv_YAMLnodeCreate(names[i], "", 0);
      add_yaml_child(node, "max_iter", "17", 1);
      hypredrv_ErrorCodeResetAll();
      hypredrv_NestedKrylovSetArgsFromYAML(&nk, node);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(nk.solver_method, methods[i]);
      switch (methods[i])
      {
         case SOLVER_PCG: ASSERT_EQ(nk.solver.pcg.max_iter, 17); break;
         case SOLVER_GMRES: ASSERT_EQ(nk.solver.gmres.max_iter, 17); break;
         case SOLVER_FGMRES: ASSERT_EQ(nk.solver.fgmres.max_iter, 17); break;
         case SOLVER_BICGSTAB: ASSERT_EQ(nk.solver.bicgstab.max_iter, 17); break;
      }
      hypredrv_NestedKrylovDestroy(&nk);
      hypredrv_YAMLnodeDestroy(node);
   }
}

/* The reserved preconditioner can appear anywhere without changing the tree. */
static void
test_nested_krylov_yaml_preserves_children(void)
{
   for (int precon_pos = 0; precon_pos < 3; precon_pos++)
   {
      NestedKrylov_args nk;
      hypredrv_NestedKrylovSetDefaultArgs(&nk);
      YAMLnode *node = hypredrv_YAMLnodeCreate("gmres", "", 0);
      YAMLnode *children[3];
      for (int i = 0; i < 3; i++)
      {
         if (i == precon_pos)
         {
            children[i] = add_yaml_child(node, "preconditioner", "amg", 1);
         }
         else if (i == (precon_pos + 1) % 3)
         {
            children[i] = add_yaml_child(node, "max_iter", "17", 1);
         }
         else
         {
            children[i] = add_yaml_child(node, "print_level", "0", 1);
         }
      }
      hypredrv_ErrorCodeResetAll();
      hypredrv_NestedKrylovSetArgsFromYAML(&nk, node);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(nk.solver.gmres.max_iter, 17);
      ASSERT_EQ(nk.precon_method, PRECON_BOOMERAMG);
      ASSERT_TRUE(nk.has_precon);
      ASSERT_TRUE(node->children == children[0]);
      ASSERT_TRUE(children[0]->next == children[1]);
      ASSERT_TRUE(children[1]->next == children[2]);
      ASSERT_NULL(children[2]->next);
      for (int i = 0; i < 3; i++)
      {
         ASSERT_TRUE(children[i]->parent == node);
         ASSERT_EQ(children[i]->valid, YAML_NODE_VALID);
      }
      hypredrv_NestedKrylovDestroy(&nk);
      hypredrv_PreconArgsDestroyOwnedConfig(nk.precon_method, &nk.precon);
      hypredrv_YAMLnodeDestroy(node);
   }
}

/* MGR may set up a nested solver without vectors; the work vectors created for
 * that must follow the matrix row range and be released cleanly. */
static void
test_nested_krylov_setup_without_vectors(void)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, SOLVER_GMRES);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);

   HYPRE_IJMatrix ij_A  = create_test_ijmatrix_1x1(4.0);
   void          *par_A = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);

   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A, NULL, NULL),
             0);
   ASSERT_TRUE(nk.base.is_setup);

   hypredrv_NestedKrylovDestroy(&nk);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
}

/* Missing operators must be rejected before hypre can dereference them. */
static void
test_nested_krylov_precon_missing_operators(void)
{
   const precon_t methods[] = {PRECON_AMS, PRECON_ADS};
   for (size_t i = 0; i < sizeof(methods) / sizeof(methods[0]); i++)
   {
      NestedKrylov_args nk;
      configure_nested_solver(&nk, SOLVER_GMRES);
      nk.has_precon = 1;
      nk.precon_method = methods[i];
      hypredrv_PreconArgsSetDefaultsForMethod(methods[i], &nk.precon);
      hypredrv_ErrorCodeResetAll();
      hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(nk.runtime);
      ASSERT_FALSE(nk.base.is_setup);
      ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, NULL, NULL, NULL), 0);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_PRECON);
      ASSERT_FALSE(nk.base.is_setup);
      hypredrv_NestedKrylovDestroy(&nk);
      hypredrv_ErrorCodeResetAll();
      HYPRE_ClearAllErrors();
   }
}

/* A nested MGR receives its reduced operator only when setup is called. */
static void
test_nested_mgr_rejects_short_marker_storage(void)
{
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   NestedKrylov_args nk;
   configure_nested_solver(&nk, SOLVER_GMRES);
   nk.has_precon = 1;
   nk.precon_method = PRECON_MGR;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &nk.precon);
   nk.precon.mgr.num_levels = 2;
   nk.precon.mgr.level[0].f_dofs.size = 1;
   nk.precon.mgr.level[0].f_dofs.data[0] = 0;
   nk.precon.mgr.level[0].f_relaxation.type = 7;
   nk.precon.mgr.level[0].g_relaxation.type = -1;
   nk.precon.mgr.coarsest_level.type = 0;
   const int map[] = {0, 1};
   IntArray *dofmap = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, map, &dofmap);
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);
   HYPRE_IJMatrix matrix = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 2, 0, 2, &matrix), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(matrix), 0);
   HYPRE_Int ncols[] = {1, 1, 1};
   HYPRE_BigInt rows[] = {0, 1, 2};
   HYPRE_Real vals[] = {2.0, 2.0, 2.0};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(matrix, 3, ncols, rows, rows, vals), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(matrix), 0);
   void *par_matrix = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(matrix, &par_matrix), 0);
   ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_matrix,
                                        NULL, NULL), 0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_FALSE(nk.base.is_setup);
   hypredrv_NestedKrylovDestroy(&nk);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJMatrixDestroy(matrix);
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
#endif
}

static void
test_nested_krylov_solver_matrix_no_precon(void)
{
   run_nested_lifecycle(SOLVER_PCG, 0);
   run_nested_lifecycle(SOLVER_GMRES, 0);
   run_nested_lifecycle(SOLVER_FGMRES, 0);
   run_nested_lifecycle(SOLVER_BICGSTAB, 0);
}

static void
test_nested_krylov_solver_matrix_with_amg_precon(void)
{
   run_nested_lifecycle(SOLVER_PCG, 1);
   run_nested_lifecycle(SOLVER_GMRES, 1);
   run_nested_lifecycle(SOLVER_FGMRES, 1);
   run_nested_lifecycle(SOLVER_BICGSTAB, 1);
}

static void
test_nested_krylov_precon_ilu_matrix(void)
{
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   run_nested_lifecycle_precon_ilu(SOLVER_GMRES);
#endif
}

static void
test_nested_krylov_precon_fsai_matrix(void)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 1)
   run_nested_lifecycle_precon_fsai(SOLVER_GMRES);
#endif
}

static void
test_nested_krylov_precon_schwarz_matrix(void)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   run_nested_lifecycle_precon_schwarz(SOLVER_GMRES);
#endif
}

static void
test_nested_krylov_destroy_clears_mgr_runtime_state(void)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   return;
#endif
   NestedKrylov_args nk;
   configure_nested_solver(&nk, SOLVER_FGMRES);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_MGR;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &nk.precon);

   nk.precon.mgr.num_levels = 2;
   nk.precon.mgr.level[0].f_dofs.size    = 1;
   nk.precon.mgr.level[0].f_dofs.data[0] = 0;
   nk.precon.mgr.level[0].f_relaxation.type = 2;
   nk.precon.mgr.level[0].g_relaxation.type = -1;
   nk.precon.mgr.coarsest_level.type = 0;
   nk.precon.mgr.coarsest_level.amg.max_iter = 1;
   nk.precon.mgr.level[0].f_relaxation.amg.max_iter = 1;

   IntArray *dofmap = NULL;
   const int map[1] = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);

   hypredrv_NestedKrylovDestroy(&nk);
   ASSERT_NULL(nk.runtime);
   ASSERT_NULL(nk.precon.mgr.frelax[0]);
   ASSERT_NULL(nk.precon.mgr.csolver);
   ASSERT_EQ(nk.precon.mgr.csolver_type, -1);

   hypredrv_IntArrayDestroy(&dofmap);
}

static void
test_nested_krylov_matched_schur_requires_fgmres(void)
{
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
   return;
#else
   const int map[2] = {0, 1};
   IntArray *dofmap = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, map, &dofmap);
   ASSERT_NOT_NULL(dofmap);

   NestedKrylov_args nk;

   configure_nested_matched_schur(&nk, SOLVER_GMRES);
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_SOLVER);
   ASSERT_NULL(nk.runtime);

   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   configure_nested_matched_schur(&nk, SOLVER_FGMRES);
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(nk.runtime);

   hypredrv_NestedKrylovDestroy(&nk);
   hypredrv_IntArrayDestroy(&dofmap);
#endif
}

static void
test_nested_krylov_rejected_precon_keeps_live_solver(void)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, SOLVER_GMRES);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_BOOMERAMG;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_BOOMERAMG, &nk.precon);
   nk.precon.amg.max_iter = 1;

   IntArray *dofmap = NULL;
   const int map[]  = {0};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
   HYPRE_IJMatrix ij_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_test_ijvector_1x1(0.0);
   void          *A = NULL, *b = NULL, *x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &x), 0);
   hypredrv_ErrorStateReset();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, A, b, x), 0);
   const void *original_runtime = nk.runtime;

   const char *replacements[] = {
      "  preconditioner: unknown_precon\n",
      "  preconditioner:\n",
      "  preconditioner:\n    amg:\n    ilu:\n",
      "  preconditioner:\n    amg:\n      max_iter: invalid\n",
      "  preconditioner:\n    ilu:\n      type: unknown_ilu\n",
      "  preconditioner:\n    amg:\n      unknown_option: 1\n",
      "  preconditioner:\n    mgr:\n      level:\n        0:\n"
      "          f_dofs: [0]\n          f_relaxation:\n            gmres:\n"
      "              max_iter: 2\n      unknown_option: 1\n",
   };
   for (size_t i = 0; i < sizeof(replacements) / sizeof(replacements[0]); i++)
   {
      char text[1024];
      snprintf(text, sizeof(text), "fgmres:\n  max_iter: 3\n%s", replacements[i]);
      YAMLtree *tree = NULL;
      hypredrv_ErrorStateReset();
      hypredrv_YAMLtreeBuild(2, text, &tree);
      ASSERT_NOT_NULL(tree);
      hypredrv_NestedKrylovSetArgsFromYAML(&nk, tree->root->children);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_PTR_EQ(nk.runtime, original_runtime);
      ASSERT_TRUE(nk.base.is_setup);
      ASSERT_TRUE(nk.has_precon);
      ASSERT_EQ(nk.solver_method, SOLVER_GMRES);
      ASSERT_EQ(nk.solver.gmres.max_iter, 20);
      ASSERT_EQ(nk.precon_method, PRECON_BOOMERAMG);
      ASSERT_EQ(nk.precon.amg.max_iter, 1);
      hypredrv_YAMLtreeDestroy(&tree);

      hypredrv_ErrorStateReset();
      HYPRE_ClearAllErrors();
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues(x, 0.0), 0);
      ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, NULL, b, x), 0);
      HYPRE_BigInt index = 0;
      double       value = 0.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(ij_x, 1, &index, &value), 0);
      ASSERT_EQ_DOUBLE(value, 0.25, 1e-10);
   }

   YAMLnode *accepted = hypredrv_YAMLnodeCreate("fgmres", "", 0);
   add_yaml_child(accepted, "max_iter", "3", 1);
   add_yaml_child(accepted, "preconditioner", "ilu", 1);
   hypredrv_ErrorStateReset();
   hypredrv_NestedKrylovSetArgsFromYAML(&nk, accepted);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(nk.runtime);
   ASSERT_FALSE(nk.base.is_setup);
   ASSERT_EQ(nk.solver_method, SOLVER_FGMRES);
   ASSERT_EQ(nk.precon_method, PRECON_ILU);
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, A, b, x), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues(x, 0.0), 0);
   ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, NULL, b, x), 0);
   HYPRE_BigInt index = 0;
   double       value = 0.0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(ij_x, 1, &index, &value), 0);
   ASSERT_EQ_DOUBLE(value, 0.25, 1e-10);
   hypredrv_YAMLnodeDestroy(accepted);

   YAMLnode *solver_only = hypredrv_YAMLnodeCreate("gmres", "", 0);
   add_yaml_child(solver_only, "max_iter", "4", 1);
   hypredrv_NestedKrylovSetArgsFromYAML(&nk, solver_only);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(nk.precon_method, PRECON_ILU);
   ASSERT_TRUE(nk.has_precon);
   ASSERT_NULL(nk.runtime);
   hypredrv_YAMLnodeDestroy(solver_only);

   hypredrv_NestedKrylovDestroy(&nk);
   hypredrv_PreconArgsDestroyOwnedConfig(nk.precon_method, &nk.precon);
   hypredrv_IntArrayDestroy(&dofmap);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   TEST_HYPRE_INIT();

   RUN_TEST(test_nested_krylov_null_and_invalid_paths);
   RUN_TEST(test_nested_krylov_solver_matrix_no_precon);
   RUN_TEST(test_nested_krylov_solver_method_parsing);
   RUN_TEST(test_nested_krylov_yaml_preserves_children);
   RUN_TEST(test_nested_krylov_setup_without_vectors);
   RUN_TEST(test_nested_krylov_precon_missing_operators);
   RUN_TEST(test_nested_mgr_rejects_short_marker_storage);
   RUN_TEST(test_nested_krylov_solver_matrix_with_amg_precon);
   RUN_TEST(test_nested_krylov_rejected_precon_keeps_live_solver);
   RUN_TEST(test_nested_krylov_precon_ilu_matrix);
   RUN_TEST(test_nested_krylov_precon_fsai_matrix);
   RUN_TEST(test_nested_krylov_precon_schwarz_matrix);
   RUN_TEST(test_nested_krylov_destroy_clears_mgr_runtime_state);
   RUN_TEST(test_nested_krylov_matched_schur_requires_fgmres);

   TEST_HYPRE_FINALIZE();

   MPI_Finalize();
   return 0;
}
