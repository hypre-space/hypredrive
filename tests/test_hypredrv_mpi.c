/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include "_hypre_utilities.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "object.h"
#include "test_helpers.h"

static HYPREDRV_t
create_distributed_test_object(int scaling_enabled)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

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
            scaling_enabled ? "  scaling:\n    enabled: 1\n    type: rhs_l2\n" : "");
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);

   HYPRE_BigInt row_ptr[2] = {0, 1};
   HYPRE_BigInt cols[1]    = {(HYPRE_BigInt)rank};
   HYPRE_Real   values[1]  = {2.0};
   HYPRE_Real   rhs[1]     = {1.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(
                obj, (HYPRE_BigInt)rank, (HYPRE_BigInt)rank, row_ptr, cols, values),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(
                obj, (HYPRE_BigInt)rank, (HYPRE_BigInt)rank, rhs),
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

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   HYPRE_Int               ncols = 1;
   HYPRE_BigInt            index = (HYPRE_BigInt)rank;
   HYPRE_Complex           matrix_value = 0.0, rhs_value = 0.0;
   ASSERT_EQ(HYPRE_IJMatrixGetValues(state->mat_A, 1, &ncols, &index, &index,
                                     &matrix_value),
             0);
   ASSERT_EQ(HYPRE_IJVectorGetValues(state->vec_b, 1, &index, &rhs_value), 0);
   ASSERT_EQ_DOUBLE((double)matrix_value, 2.0, 1.0e-12);
   ASSERT_EQ_DOUBLE((double)rhs_value, 1.0, 1.0e-12);
}

static void
test_PreconSetup_synchronizes_rank_local_hypre_error(void)
{
   HYPREDRV_t obj = create_distributed_test_object(0);
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
   HYPREDRV_t obj = create_distributed_test_object(0);
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
   HYPREDRV_t obj = create_distributed_test_object(0);
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
   HYPREDRV_t obj = create_distributed_test_object(1);
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
   HYPREDRV_t obj = create_distributed_test_object(1);
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
#endif

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   int nprocs = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs >= 2)
   {
      RUN_TEST(test_PreconSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_LinearSolverSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_reused_precon_remains_setup_after_solver_setup_error);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
      RUN_TEST(test_scaled_setup_restores_system_after_rank_local_hypre_error);
      RUN_TEST(test_scaled_solve_restores_system_after_rank_local_hypre_error);
#endif
   }

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
