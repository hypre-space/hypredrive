/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/krylov.h"
#include "internal/precon.h"
#include "internal/solver.h"
#include "internal/yaml.h"
#include "test_helpers.h"

static YAMLnode *
add_yaml_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

static HYPRE_IJMatrix
create_ijmatrix_1x1(double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows    = 1;
   HYPRE_Int    ncols[1] = {1};
   HYPRE_BigInt rows[1]  = {0};
   HYPRE_BigInt cols[1]  = {0};
   double       values[1] = {diag};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static HYPRE_IJVector
create_ijvector_1x1(double value)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[1] = {0};
   double       val[1] = {value};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 1, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
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

   HYPRE_Solver inner = NULL;
   IntArray *dofmap = NULL;

   if (use_precon_amg)
   {
      const int map[1] = {0};
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, map, &dofmap);
      ASSERT_NOT_NULL(dofmap);
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL, &inner);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(inner);

   HYPRE_IJMatrix ij_A = create_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_ijvector_1x1(0.0);

   void *par_A = NULL, *par_b = NULL, *par_x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(ij_A, &par_A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_b, &par_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(ij_x, &par_x), 0);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, (HYPRE_Matrix)par_A,
                                        (HYPRE_Vector)par_b, (HYPRE_Vector)par_x),
             0);

   hypredrv_NestedKrylovDestroy(&nk);
   ASSERT_NULL(nk.base_solver);
   ASSERT_NULL(nk.precon_obj);

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);

   hypredrv_IntArrayDestroy(&dofmap);
}

static void
test_nested_krylov_null_and_invalid_paths(void)
{
   hypredrv_NestedKrylovSetDefaultArgs(NULL);

   /* hypredrv_NestedKrylovSetArgsFromYAML(NULL, node): hits !args branch (krylov.c). */
   {
      YAMLnode *solver = hypredrv_YAMLnodeCreate("gmres", "", 0);
      add_yaml_child(solver, "max_iter", "2", 1);
      hypredrv_NestedKrylovSetArgsFromYAML(NULL, solver);
      hypredrv_YAMLnodeDestroy(solver);
   }

   NestedKrylov_args nk;
   hypredrv_NestedKrylovSetDefaultArgs(&nk);

   HYPRE_Solver out = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, NULL, NULL, NULL, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

   nk.is_set = 0;
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL, &out);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(out);

   hypredrv_NestedKrylovSetDefaultArgs(&nk);
   nk.is_set = 1;
   nk.solver_method = SOLVER_GMRES;
   hypredrv_SolverArgsSetDefaultsForMethod(SOLVER_GMRES, &nk.solver);
   nk.solver.gmres.max_iter = 5;
   /* base_solver intentionally unset */
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&nk, NULL, NULL, NULL), 0);
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSolve((HYPRE_Solver)&nk, NULL, NULL, NULL), 0);

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSetup(NULL, NULL, NULL, NULL), 0);
   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_NestedKrylovSolve(NULL, NULL, NULL, NULL), 0);

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

   HYPRE_Solver inner = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL, &inner);
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_IntArrayDestroy(&dofmap);
      return;
   }
   ASSERT_NOT_NULL(inner);

   HYPRE_IJMatrix ij_A = create_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_ijvector_1x1(0.0);

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

   HYPRE_Solver inner = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL, &inner);
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_IntArrayDestroy(&dofmap);
      return;
   }
   ASSERT_NOT_NULL(inner);

   HYPRE_IJMatrix ij_A = create_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_ijvector_1x1(0.0);

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

#if HYPREDRV_HAVE_EXPERIMENTAL
static void
run_nested_lifecycle_precon_schwarz(solver_t method)
{
   NestedKrylov_args nk;
   configure_nested_solver(&nk, method);
   nk.has_precon    = 1;
   nk.precon_method = PRECON_SCHWARZ;
   hypredrv_PreconArgsSetDefaultsForMethod(PRECON_SCHWARZ, &nk.precon);

   HYPRE_Solver inner = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, NULL, NULL, &inner);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(inner);

   HYPRE_IJMatrix ij_A = create_ijmatrix_1x1(4.0);
   HYPRE_IJVector ij_b = create_ijvector_1x1(1.0);
   HYPRE_IJVector ij_x = create_ijvector_1x1(0.0);

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
   ASSERT_NULL(nk.base_solver);
   ASSERT_NULL(nk.precon_obj);

   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_x), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(ij_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(ij_A), 0);
}
#endif

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
#if HYPREDRV_HAVE_EXPERIMENTAL
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

   HYPRE_Solver inner = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_NestedKrylovCreate(MPI_COMM_SELF, &nk, dofmap, NULL, &inner);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(inner);
   ASSERT_NOT_NULL(nk.precon_obj);

   hypredrv_NestedKrylovDestroy(&nk);
   ASSERT_NULL(nk.base_solver);
   ASSERT_NULL(nk.precon_obj);
   ASSERT_NULL(nk.precon.mgr.frelax[0]);
   ASSERT_NULL(nk.precon.mgr.csolver);
   ASSERT_EQ(nk.precon.mgr.csolver_type, -1);

   hypredrv_IntArrayDestroy(&dofmap);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   TEST_HYPRE_INIT();

   RUN_TEST(test_nested_krylov_null_and_invalid_paths);
   RUN_TEST(test_nested_krylov_solver_matrix_no_precon);
   RUN_TEST(test_nested_krylov_solver_matrix_with_amg_precon);
   RUN_TEST(test_nested_krylov_precon_ilu_matrix);
   RUN_TEST(test_nested_krylov_precon_fsai_matrix);
   RUN_TEST(test_nested_krylov_precon_schwarz_matrix);
   RUN_TEST(test_nested_krylov_destroy_clears_mgr_runtime_state);

   TEST_HYPRE_FINALIZE();

   MPI_Finalize();
   return 0;
}
