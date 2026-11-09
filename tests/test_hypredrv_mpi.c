/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include "HYPREDRV.h"
#include "_hypre_utilities.h"
#include "internal/error.h"
#include "internal/krylov.h"
#include "logging.h"
#include "object.h"
#include "vector_tags.h"
#include "_hypre_parcsr_mv.h"
#include "test_helpers.h"

static HYPREDRV_t
create_distributed_test_object(const char *scaling_type)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char scaling_block[96] = "";
   if (scaling_type)
   {
      snprintf(scaling_block, sizeof(scaling_block),
               "  scaling:\n    enabled: 1\n    type: %s\n", scaling_type);
   }

   char yaml[512];
   snprintf(yaml, sizeof(yaml),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "%s"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            scaling_block);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);

   HYPRE_BigInt row_ptr[2] = {0, 1};
   HYPRE_BigInt cols[1]    = {(HYPRE_BigInt)rank};
   HYPRE_Real   values[1]  = {2.0};
   HYPRE_Real   rhs[1]     = {1.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(
                obj, (HYPRE_BigInt)rank, (HYPRE_BigInt)rank, row_ptr, cols, values),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, (HYPRE_BigInt)rank,
                                                  (HYPRE_BigInt)rank, rhs),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   return obj;
}

static void
inject_rank_local_hypre_error(int failing_rank)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == failing_rank)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "injected rank-local HYPRE failure");
   }
}

static void
assert_collective_hypre_failure(uint32_t code)
{
   uint32_t code_min = 0, code_max = 0;
   MPI_Allreduce(&code, &code_min, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&code, &code_max, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
   ASSERT_EQ_U32(code_min, code_max);
   ASSERT_TRUE(code & ERROR_HYPRE_INTERNAL);
}

static void
assert_local_system_restored(HYPREDRV_t obj)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   struct hypredrv_struct *state        = (struct hypredrv_struct *)obj;
   HYPRE_Int               ncols        = 1;
   HYPRE_BigInt            index        = (HYPRE_BigInt)rank;
   HYPRE_Complex           matrix_value = 0.0, rhs_value = 0.0;
   ASSERT_EQ(
      HYPRE_IJMatrixGetValues(state->mat_A.handle, 1, &ncols, &index, &index, &matrix_value), 0);
   ASSERT_EQ(HYPRE_IJVectorGetValues(state->vec_b.handle, 1, &index, &rhs_value), 0);
   ASSERT_EQ_DOUBLE((double)matrix_value, 2.0, 1.0e-12);
   ASSERT_EQ_DOUBLE((double)rhs_value, 1.0, 1.0e-12);
}

static void
test_PreconSetup_synchronizes_rank_local_hypre_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);

   inject_rank_local_hypre_error(1);
   uint32_t code = HYPREDRV_PreconSetup(obj);
   assert_collective_hypre_failure(code);
   ASSERT_FALSE(((struct hypredrv_struct *)obj)->precon_is_setup);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_LinearSolverSetup_synchronizes_rank_local_hypre_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);

   inject_rank_local_hypre_error(0);
   uint32_t code = HYPREDRV_LinearSolverSetup(obj);
   assert_collective_hypre_failure(code);
   ASSERT_FALSE(((struct hypredrv_struct *)obj)->precon_is_setup);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_reused_precon_remains_setup_after_solver_setup_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->precon_is_setup);

   /* With no timestep metadata this policy reuses the already-setup
    * preconditioner, while the outer solver setup still executes. */
   state->iargs->precon_reuse.enabled      = 1;
   state->iargs->precon_reuse.per_timestep = 1;
   inject_rank_local_hypre_error(0);
   uint32_t code = HYPREDRV_LinearSolverSetup(obj);
   assert_collective_hypre_failure(code);
   ASSERT_TRUE(state->precon_is_setup);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
test_scaled_setup_restores_system_after_rank_local_hypre_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object("rhs_l2");
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);

   inject_rank_local_hypre_error(0);
   uint32_t code = HYPREDRV_LinearSolverSetup(obj);
   assert_collective_hypre_failure(code);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_FALSE(state->precon_is_setup);
   ASSERT_FALSE(state->scaling_ctx->is_applied);
   assert_local_system_restored(obj);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_scaled_solve_restores_system_after_rank_local_hypre_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object("rhs_l2");
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->scaling_ctx->is_applied);

   inject_rank_local_hypre_error(0);
   uint32_t code = HYPREDRV_LinearSolverApply(obj);
   assert_collective_hypre_failure(code);
   ASSERT_FALSE(state->scaling_ctx->is_applied);
   assert_local_system_restored(obj);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

/* A dofmap that is invalid on one rank only must fail setup on every rank
 * instead of leaving the others blocked in the scaling collectives. */
static void
test_dofmap_scaling_rank_local_size_mismatch_fails_collectively(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   HYPREDRV_t obj     = create_distributed_test_object("dofmap_mag");
   int        dofs[2] = {0, 0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, (rank == 1) ? 2 : 1, dofs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);

   uint32_t code = HYPREDRV_LinearSolverSetup(obj);
   ASSERT_TRUE(code != ERROR_NONE);

   hypredrv_ErrorStateReset();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}
#endif

/* A short marker buffer on one rank must stop every rank before MGR setup. */
static void
test_mgr_short_marker_storage_fails_collectively(void)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (int standalone = 0; standalone < 2; standalone++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
      char yaml[] = "general:\n  statistics: off\n"
                    "linear_system:\n  init_guess_mode: zeros\n"
                    "solver:\n  gmres:\n    max_iter: 5\n"
                    "preconditioner:\n  mgr:\n    level:\n      0:\n"
                    "        f_dofs: [0]\n        f_relaxation: jacobi\n"
                    "        g_relaxation: none\n    coarsest_level: amg\n";
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      HYPRE_BigInt first = 2 * (HYPRE_BigInt)rank;
      HYPRE_BigInt ptr[] = {0, 1, 2}, cols[] = {first, first + 1};
      HYPRE_Real values[] = {2.0, 2.0}, rhs[] = {1.0, 1.0};
      int labels[] = {0, 1};
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, first, first + 1, ptr, cols, values), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, first, first + 1, rhs), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, rank == 1 ? 1 : 2, labels), ERROR_NONE);
      if (standalone)
      {
         ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
      }
      else
      {
         ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      }
      uint32_t code = standalone ? HYPREDRV_PreconSetup(obj) : HYPREDRV_LinearSolverSetup(obj);
      uint32_t low = 0, high = 0;
      MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
      ASSERT_EQ_U32(low, high);
      ASSERT_TRUE(code & ERROR_INVALID_VAL);
      ASSERT_FALSE(obj->precon_is_setup);
      hypredrv_ErrorStateReset();
      HYPRE_ClearAllErrors();
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
#endif
}

/* A missing operator on one rank must reject every setup path before hypre
 * enters collective work. The other rank supplies real borrowed handles. */
static void
test_precon_operator_preflight_is_collective(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   PreconOperators supplied = {obj->mat_A.handle, obj->mat_A.handle,
                               {obj->vec_b.handle, obj->vec_b.handle,
                                obj->vec_b.handle}, NULL};
   const precon_t methods[] = {PRECON_AMS, PRECON_ADS};
   const solver_t solvers[] = {SOLVER_PCG, SOLVER_GMRES, SOLVER_FGMRES, SOLVER_BICGSTAB};
   for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++)
   {
      for (int path = 0; path < 5; path++)
      {
         precon_args args;
         hypredrv_PreconArgsSetDefaultsForMethod(methods[m], &args);
         HYPRE_Precon precon = NULL;
         hypredrv_PreconCreate(MPI_COMM_WORLD, methods[m], &args, NULL, NULL,
                               &precon, NULL, 0, rank ? &supplied : NULL);
         ASSERT_FALSE(hypredrv_ErrorCodeActive());
         ASSERT_NOT_NULL(precon);
         ASSERT_EQ(precon->operators_ok, rank ? 1 : 0);
         HYPRE_Solver solver = NULL;
         if (path == 0)
         {
            hypredrv_PreconSetup(methods[m], precon, obj->mat_A.handle);
         }
         else
         {
            solver_args solver_args;
            hypredrv_SolverArgsSetDefaultsForMethod(solvers[path - 1], &solver_args);
            hypredrv_SolverCreate(MPI_COMM_WORLD, solvers[path - 1], &solver_args, &solver);
            hypredrv_SolverSetupWithReuse(methods[m], solvers[path - 1], precon, solver,
                                          obj->mat_A.handle, obj->vec_b.handle,
                                          obj->vec_x.handle, NULL, 0);
         }
         uint32_t code = hypredrv_ErrorCodeGet(), low = 0, high = 0;
         MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
         ASSERT_EQ_U32(low, high);
         ASSERT_TRUE(code & ERROR_MISSING_PRECON);
         ASSERT_FALSE(precon->is_setup);
         hypredrv_ErrorStateReset();
         if (solver)
         {
            hypredrv_SolverDestroy(solvers[path - 1], &solver);
         }
         hypredrv_PreconDestroy(methods[m], &args, &precon, NULL, 0);
      }
   }
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_nested_setup_preflight_is_collective(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   void *matrix = NULL, *rhs = NULL, *solution = NULL;
   HYPRE_IJMatrixGetObject(obj->mat_A.handle, &matrix);
   HYPRE_IJVectorGetObject(obj->vec_b.handle, &rhs);
   HYPRE_IJVectorGetObject(obj->vec_x.handle, &solution);
   for (solver_t method = SOLVER_PCG; method <= SOLVER_BICGSTAB; method++)
   {
      NestedKrylov_args args;
      hypredrv_NestedKrylovSetDefaultArgs(&args);
      args.is_set = 1;
      args.solver_method = method;
      hypredrv_SolverArgsSetDefaultsForMethod(method, &args.solver);
      args.has_precon = 1;
      args.precon_method = PRECON_BOOMERAMG;
      hypredrv_PreconArgsSetDefaultsForMethod(args.precon_method, &args.precon);
      HYPRE_Solver solver = NULL;
      hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, &args, NULL, NULL, &solver);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(solver);
      ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&args,
                   rank ? (HYPRE_Matrix)matrix : NULL, (HYPRE_Vector)rhs,
                   (HYPRE_Vector)solution), 0);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_FALSE(args.is_setup);
      ASSERT_FALSE(args.precon_obj->is_setup);
      hypredrv_ErrorStateReset();
      hypredrv_NestedKrylovDestroy(&args);
   }
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_independent_self_solve_and_recycled_log_comm(void)
{
   int rank = 0, size = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   for (int i = 0; i < 8; i++)
   {
      MPI_Comm split = MPI_COMM_NULL;
      MPI_Comm_split(MPI_COMM_WORLD, 0, (i % 2) ? size - rank : rank, &split);
      int expected = -1;
      MPI_Comm_rank(split, &expected);
      ASSERT_EQ(hypredrv_LogRankFromComm(split), expected);
      MPI_Comm_free(&split);
   }
   /* Other ranks wait in a different collective. Any accidental WORLD
    * collective in the driver or nested MGR solvers deadlocks this test. */
   if (rank == 0)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      char yaml[] =
         "general:\n  statistics: off\nlinear_system:\n  init_guess_mode: zeros\n"
         "solver:\n  fgmres:\n    max_iter: 5\n"
         "preconditioner:\n"
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         "  mgr:\n    level:\n      0:\n        f_dofs: [0]\n"
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
         "        f_relaxation:\n          gmres:\n            max_iter: 1\n            "
         "preconditioner: amg\n"
         "        g_relaxation:\n          gmres:\n            max_iter: 1\n            "
         "preconditioner: amg\n"
         "    coarsest_level:\n      gmres:\n        max_iter: 1\n        "
         "preconditioner: amg\n";
#else
         "        f_relaxation: jacobi\n        g_relaxation: none\n"
         "    coarsest_level:\n      amg:\n        print_level: 0\n";
#endif
#else
         "  amg:\n    print_level: 0\n";
#endif
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      HYPRE_BigInt rows[] = {0, 1, 2}, cols[] = {0, 1};
      HYPRE_Real   values[] = {2.0, 3.0}, rhs[] = {4.0, 9.0};
      int          dofs[] = {0, 1};
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 1, rows, cols, values),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 1, rhs), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, dofs), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
      ASSERT_TRUE(obj->iargs->precon.mgr.comm == MPI_COMM_SELF);
#endif
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

static void
test_rank_local_schedule_error_is_collective(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   char path[256];
   test_scratch_path(path, sizeof(path), "hypredrive-schedule-mpi-XXXXXX");
   int  fd     = mkstemp(path);
   ASSERT_TRUE(fd >= 0);
   FILE *fp = fdopen(fd, "w");
   ASSERT_NOT_NULL(fp);
   fputs(rank == 0 ? "1\n0 0\n" : "0\n", fp);
   fclose(fp);
   char yaml[512];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: on\nlinear_system:\n  timestep_filename: %s\n",
            path);
   char    *argv[] = {yaml};
   uint32_t code   = HYPREDRV_InputArgsParse(1, argv, obj);
   uint32_t low = 0, high = 0;
   MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
   ASSERT_NE(low, ERROR_NONE);
   ASSERT_EQ_U32(low, high);
   unlink(path);
   hypredrv_ErrorStateReset();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_tagged_reduction_with_empty_rank(void)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0) && !defined(HYPRE_USING_GPU)
   int rank = 0, nprocs = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char *argv[] = {"solver:\n  gmres:\n    print_level: 3\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJVector rhs = NULL;
   HYPRE_BigInt lower = rank ? rank - 1 : 0, upper = rank - 1;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, lower, upper, &rhs), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(rhs, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(rhs), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(rhs), 0);
   void *object = NULL;
   ASSERT_EQ(HYPRE_IJVectorGetObject(rhs, &object), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)object, 1.0), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   int label = 1;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, rank ? 1 : 0, rank ? &label : NULL), ERROR_NONE);
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj->vector_tags);
   HYPRE_Int num_tags = 0;
   HYPRE_Complex *norms = NULL;
   ASSERT_EQ(hypre_ParVectorInnerProdTagged((hypre_ParVector *)object,
              (hypre_ParVector *)object, &num_tags, &norms), 0);
   ASSERT_EQ(num_tags, 2);
   ASSERT_NOT_NULL(norms);
   ASSERT_EQ_DOUBLE(norms[0], (double)(nprocs - 1), 1e-12);
   hypre_TFree(norms, HYPRE_MEMORY_HOST);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(hypre_ParVectorTags((hypre_ParVector *)object));
   HYPRE_IJVectorDestroy(rhs);
#endif
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   int nprocs = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs >= 2)
   {
      RUN_TEST(test_tagged_reduction_with_empty_rank);
   RUN_TEST(test_independent_self_solve_and_recycled_log_comm);
      RUN_TEST(test_rank_local_schedule_error_is_collective);
      RUN_TEST(test_mgr_short_marker_storage_fails_collectively);
      RUN_TEST(test_precon_operator_preflight_is_collective);
      RUN_TEST(test_nested_setup_preflight_is_collective);
      RUN_TEST(test_PreconSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_LinearSolverSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_reused_precon_remains_setup_after_solver_setup_error);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
      RUN_TEST(test_scaled_setup_restores_system_after_rank_local_hypre_error);
      RUN_TEST(test_scaled_solve_restores_system_after_rank_local_hypre_error);
      RUN_TEST(test_dofmap_scaling_rank_local_size_mismatch_fails_collectively);
#endif
   }

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
