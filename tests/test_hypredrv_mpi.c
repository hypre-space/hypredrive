/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include "HYPREDRV.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_utilities.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/mgr_internal.h"
#include "internal/solver.h"
#include "logging.h"
#include "object.h"
#include "test_helpers.h"
#include "test_solution_helpers.h"
#include "vector_tags.h"

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

#if HYPRE_CHECK_MIN_VERSION(22600, 0)
static void
test_mtx_matrix_is_partitioned_across_ranks(void)
{
   int rank = 0, nprocs = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (nprocs != 2)
   {
      return;
   }

   char filename[PATH_MAX] = {0};
   if (rank == 0)
   {
      test_scratch_path(filename, sizeof(filename), "matrix-market.mtx");
   }
   MPI_Bcast(filename, (int)sizeof(filename), MPI_CHAR, 0, MPI_COMM_WORLD);

   int write_ok = 1;
   if (rank == 0)
   {
      FILE *fp = fopen(filename, "w");
      if (!fp)
      {
         write_ok = 0;
      }
      else
      {
         fputs("%%MatrixMarket matrix coordinate real symmetric\n"
               "4 4 6\n"
               "1 1 4.0\n"
               "2 2 5.0\n"
               "3 3 6.0\n"
               "4 4 7.0\n"
               "1 2 -1.0\n"
               "3 4 -2.0\n",
               fp);
         write_ok = (fclose(fp) == 0);
      }
   }
   MPI_Bcast(&write_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
   ASSERT_EQ(write_ok, 1);
   MPI_Barrier(MPI_COMM_WORLD);

   char yaml[PATH_MAX + 512];
   snprintf(yaml, sizeof(yaml),
            "general:\n"
            "  statistics: 0\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  type: mtx\n"
            "  matrix_filename: %s\n"
            "  rhs_mode: randsol\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 20\n"
            "    relative_tol: 1.0e-10\n"
            "preconditioner:\n"
            "  fsai:\n"
            "    max_steps: 20\n",
            filename);
   char      *yaml_argv[] = {yaml};
   HYPREDRV_t obj         = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, yaml_argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(hypredrv_LinearSystemMatrixGetNumRows(state->mat_A.handle), 4);
   ASSERT_EQ(hypredrv_LinearSystemMatrixGetNumNonzeros(state->mat_A.handle), 8);

   HYPRE_BigInt ilower = 0, iupper = -1, jlower = 0, jupper = -1;
   ASSERT_EQ(HYPRE_IJMatrixGetLocalRange(state->mat_A.handle, &ilower, &iupper, &jlower,
                                         &jupper),
             0);
   ASSERT_EQ(ilower, (HYPRE_BigInt)(rank * 2));
   ASSERT_EQ(iupper, (HYPRE_BigInt)(rank * 2 + 1));

   for (HYPRE_BigInt row = ilower; row <= iupper; row++)
   {
      HYPRE_Int     ncols     = 2;
      HYPRE_BigInt  cols[2]   = {row, row ^ 1};
      HYPRE_Complex values[2] = {0.0, 0.0};
      ASSERT_EQ(
         HYPRE_IJMatrixGetValues(state->mat_A.handle, 1, &ncols, &row, cols, values), 0);
      ASSERT_EQ_DOUBLE((double)values[0], 4.0 + (double)row, 0.0);
      ASSERT_EQ_DOUBLE((double)values[1], row < 2 ? -1.0 : -2.0, 0.0);
   }

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0)
   {
      unlink(filename);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}
#endif

static void
test_configured_construction_errors_are_collective(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj          = create_distributed_test_object(NULL);
   char       prefix[1024] = {0};
   if (!rank) test_scratch_path(prefix, sizeof(prefix), "configured-mpi");
   MPI_Bcast(prefix, sizeof(prefix), MPI_CHAR, 0, MPI_COMM_WORLD);
   HYPRE_IJMatrix matrix = obj->mat_A.handle;
   ASSERT_EQ(HYPRE_IJMatrixPrint(matrix, prefix), 0);
   MPI_Barrier(MPI_COMM_WORLD);
   char yaml[2048];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\n  exec_policy: host\n"
            "linear_system:\n  matrix_filename: %s\n  rhs_mode: ones\n"
            "  init_guess_mode: zeros\nsolver: pcg\npreconditioner: amg\n",
            prefix);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   int initial_index = obj->current_system_index;
   if (rank == 1) hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ObjectLinearSystemBuild(obj), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
   ASSERT_EQ(obj->current_system_index, initial_index);
   hypredrv_ErrorStateReset();

   inject_rank_local_hypre_error(0);
   assert_collective_hypre_failure(HYPREDRV_LinearSystemReadMatrix(obj));
   ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
   ASSERT_PTR_EQ(obj->mat_M.handle, matrix);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   ASSERT_EQ(obj->current_system_index, initial_index + 1);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   HYPRE_Complex *solution = NULL;
   ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)solution[0], 0.5, 1e-12);

   /* A matrix missing on one rank is rejected before RHS generation on any rank. */
   HYPREDRV_t partial = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &partial), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(partial), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, partial), ERROR_NONE);
   if (rank)
   {
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(partial, (HYPRE_Matrix)obj->mat_A.handle),
                ERROR_NONE);
   }
   ASSERT_EQ_U32(HYPREDRV_LinearSystemSetRHS(partial, NULL), ERROR_INVALID_VAL);
   ASSERT_NULL(partial->vec_b.handle);
   ASSERT_EQ(HYPREDRV_Destroy(&partial), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   MPI_Barrier(MPI_COMM_WORLD);
}

static void
test_configured_matrix_ingress_is_collective(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t obj = create_distributed_test_object(NULL);
      if (locations[p])
      {
         char *argv[] = {"general:\n  statistics: off\n  exec_policy: device\n"
                         "solver: pcg\npreconditioner: amg\n"};
         ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      }
      char prefix[1024] = {0};
      if (!rank) test_scratch_path(prefix, sizeof(prefix), "matrix-ingress");
      MPI_Bcast(prefix, sizeof(prefix), MPI_CHAR, 0, MPI_COMM_WORLD);
      ASSERT_EQ(HYPRE_IJMatrixPrint(obj->mat_A.handle, prefix), 0);
      char binary_prefix[1100];
      snprintf(binary_prefix, sizeof(binary_prefix), "%s-binary", prefix);
      if (!rank)
      {
         char path[1200];
         snprintf(path, sizeof(path), "%s.00000.bin", binary_prefix);
         FILE *file = fopen(path, "wb");
         ASSERT_NOT_NULL(file);
         /* The inconsistent format must be rejected before this is decoded. */
         fputs("invalid binary matrix", file);
         fclose(file);
      }
      snprintf(obj->iargs->ls.matrix_filename, sizeof(obj->iargs->ls.matrix_filename),
               "%s", prefix);
      /* Force a separate file read rather than the main-matrix alias. */
      snprintf(obj->iargs->ls.precmat_filename, sizeof(obj->iargs->ls.precmat_filename),
               "%s", prefix);
      obj->iargs->ls.matrix_filename[0] = '\0';
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      input_args *args = obj->iargs;
      const struct
      {
         int operation; /* read, configured preconditioner matrix, or build */
         int missing;   /* arguments, source, file, type, path, main, binary format */
      } cases[] = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2},
                   {0, 3}, {1, 3}, {0, 4}, {1, 4}, {1, 5}, {0, 6}, {1, 6}};
      for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
      {
         snprintf(args->ls.matrix_filename, sizeof(args->ls.matrix_filename), "%s",
                  cases[c].operation == 1 ? "main-matrix" : prefix);
         LS_args        original_args = args->ls;
         HYPRE_IJMatrix matrix = obj->mat_A.handle, precmat = obj->mat_M.handle;
         bool           owned_matrix = obj->mat_A.owned, owned_precmat = obj->mat_M.owned;
         HYPRE_Precon   precon       = obj->precon;
         hypredrv_Solver  *solver       = obj->solver;
         int            system_index = obj->current_system_index;
         int            ls_counter   = obj->stats->ls_counter;
         const input_args *cache     = obj->precmat_sequence_cache_args;
         if (cases[c].missing == 5) args->ls.precmat_filename[0] = '\0';
         if (rank == 1)
         {
            char *filename = cases[c].operation == 1 ? args->ls.precmat_filename
                                                     : args->ls.matrix_filename;
            switch (cases[c].missing)
            {
               case 0:
                  obj->iargs = NULL;
                  break;
               case 1:
                  strcpy(cases[c].operation == 1 ? args->ls.precmat_sequence_filename
                                                 : args->ls.sequence_filename,
                         "/missing-matrix-sequence");
                  break;
               case 2:
                  strcat(filename, "-missing");
                  break;
               case 3:
                  args->ls.type = 3;
                  break;
               case 4:
                  memset(args->ls.dirname, 'd', sizeof(args->ls.dirname) - 1);
                  args->ls.dirname[sizeof(args->ls.dirname) - 1] = '\0';
                  break;
               case 5:
                  obj->mat_A.handle = NULL;
                  break;
               case 6:
                  strcpy(filename, binary_prefix);
                  break;
            }
         }
         uint32_t code = cases[c].operation == 0 ? HYPREDRV_LinearSystemReadMatrix(obj)
                         : cases[c].operation == 1
                            ? HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL)
                            : HYPREDRV_LinearSystemBuild(obj);
         if (rank == 1 && cases[c].missing == 0) obj->iargs = args;
         if (rank == 1 && cases[c].missing == 5) obj->mat_A.handle = matrix;
         args->ls     = original_args;
         uint32_t low = 0, high = 0;
         MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
         ASSERT_NE(code, ERROR_NONE);
         ASSERT_EQ_U32(low, high);
         ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
         ASSERT_PTR_EQ(obj->mat_M.handle, precmat);
         ASSERT_EQ(obj->mat_A.owned, owned_matrix);
         ASSERT_EQ(obj->mat_M.owned, owned_precmat);
         ASSERT_PTR_EQ(obj->precon, precon);
         ASSERT_PTR_EQ(obj->solver, solver);
         ASSERT_TRUE(obj->precon_is_setup);
         ASSERT_PTR_EQ(obj->precmat_sequence_cache_args, cache);
         ASSERT_EQ(obj->current_system_index, system_index);
         ASSERT_EQ(obj->stats->ls_counter, ls_counter);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         HYPRE_Complex *values = NULL;
         ASSERT_EQ(test_copy_solution_values(obj, &values), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)values[0], 0.5, 1e-12);
      }
      /* Existing-handle attachment is local even on a distributed object. */
      if (rank == 1)
      {
         ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)obj->mat_A.handle),
                   ERROR_NONE);
         HYPRE_IJMatrix matrix = obj->mat_M.handle;
         obj->iargs            = NULL;
         ASSERT_TRUE(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix) &
                     ERROR_INVALID_VAL);
         obj->iargs = args;
         ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix),
                   ERROR_NONE);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      /* Repair the input, read a separate preconditioner, then replace the system.
       * The common reader must also produce usable matrices in device memory. */
      ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL), ERROR_NONE);
      ASSERT_TRUE(obj->mat_M.handle != obj->mat_A.handle);
      ASSERT_TRUE(obj->mat_M.owned);
      snprintf(args->ls.matrix_filename, sizeof(args->ls.matrix_filename), "%s", prefix);
      ASSERT_EQ(HYPREDRV_LinearSystemReadMatrix(obj), ERROR_NONE);
      ASSERT_PTR_EQ(obj->mat_M.handle, obj->mat_A.handle);
      ASSERT_TRUE(obj->mat_A.owned);
      ASSERT_FALSE(obj->mat_M.owned);
#ifdef HYPRE_USING_GPU
      void *native_matrix = NULL;
      ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &native_matrix), 0);
      ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)native_matrix),
                locations[p] ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
#endif
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      HYPRE_Complex *values = NULL;
      ASSERT_EQ(test_copy_solution_values(obj, &values), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)values[0], 0.5, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_solver_workflow_preflight_is_collective(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   enum
   {
      PCREATE,
      SCREATE,
      PSETUP,
      SSETUP,
      SAPPLY,
      PAPPLY,
      PDESTROY,
      SDESTROY
   };
   uint32_t (*operations[])(HYPREDRV_t) = {
      HYPREDRV_PreconCreate,      HYPREDRV_LinearSolverCreate, HYPREDRV_PreconSetup,
      HYPREDRV_LinearSolverSetup, HYPREDRV_LinearSolverApply,  NULL,
      HYPREDRV_PreconDestroy,     HYPREDRV_LinearSolverDestroy};
   enum Missing
   {
      ARGS,
      SOLVER,
      PRECON,
      SETUP,
      A,
      M,
      RHS,
      X,
      EXTERNAL_RHS,
      EXTERNAL_X
   };
   const struct
   {
      int          operation;
      enum Missing missing;
   } cases[] = {
      {PCREATE, ARGS},      {SCREATE, ARGS},  {PSETUP, ARGS},   {SSETUP, ARGS},
      {SAPPLY, ARGS},       {PAPPLY, ARGS},   {PDESTROY, ARGS}, {SDESTROY, ARGS},
      {SSETUP, SOLVER},     {SAPPLY, SOLVER}, {PSETUP, PRECON}, {SSETUP, PRECON},
      {SAPPLY, PRECON},     {PAPPLY, PRECON}, {SAPPLY, SETUP},  {PAPPLY, SETUP},
      {PSETUP, A},          {SSETUP, M},      {SAPPLY, A},      {PAPPLY, A},
      {SSETUP, RHS},        {SAPPLY, RHS},    {SAPPLY, X},      {PAPPLY, EXTERNAL_RHS},
      {PAPPLY, EXTERNAL_X},
   };
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t obj = create_distributed_test_object(NULL);
      char       yaml[256];
      snprintf(
         yaml, sizeof(yaml),
         "general:\n  statistics: off\n  exec_policy: %s\n"
         "solver: pcg\npreconditioner:\n  reuse: adaptive\n  amg:\n    print_level: 0\n",
         locations[p] ? "device" : "host");
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      input_args    *args   = obj->iargs;
      hypredrv_Solver *solver = obj->solver;
      HYPRE_Precon   precon = obj->precon;
      HYPRE_IJMatrix matrix = obj->mat_A.handle, precmat = obj->mat_M.handle;
      HYPRE_IJVector rhs = obj->vec_b.handle, solution = obj->vec_x.handle;
      for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
      {
         PreconReuseState *history      = obj->precon_reuse_state;
         int               history_age  = test_reuse_decision(history, 0).age;
         HYPRE_IJVector   supplied_rhs = rhs, supplied_solution = solution;
         int              system_index = obj->current_system_index;
         if (rank == 1)
         {
            switch (cases[c].missing)
            {
               case ARGS:
                  obj->iargs = NULL;
                  break;
               case SOLVER:
                  obj->solver = NULL;
                  break;
               case PRECON:
                  obj->precon = NULL;
                  break;
               case SETUP:
                  obj->precon_is_setup = false;
                  break;
               case A:
                  obj->mat_A.handle = NULL;
                  break;
               case M:
                  obj->mat_M.handle = NULL;
                  break;
               case RHS:
                  obj->vec_b.handle = NULL;
                  break;
               case X:
                  obj->vec_x.handle = NULL;
                  break;
               case EXTERNAL_RHS:
                  supplied_rhs = NULL;
                  break;
               case EXTERNAL_X:
                  supplied_solution = NULL;
                  break;
            }
         }
         uint32_t code = cases[c].operation == PAPPLY
                            ? HYPREDRV_PreconApply(obj, (HYPRE_Vector)supplied_rhs,
                                                   (HYPRE_Vector)supplied_solution)
                            : operations[cases[c].operation](obj);
         if (rank == 1)
         {
            switch (cases[c].missing)
            {
               case ARGS:
                  obj->iargs = args;
                  break;
               case SOLVER:
                  obj->solver = solver;
                  break;
               case PRECON:
                  obj->precon = precon;
                  break;
               case SETUP:
                  obj->precon_is_setup = true;
                  break;
               case A:
                  obj->mat_A.handle = matrix;
                  break;
               case M:
                  obj->mat_M.handle = precmat;
                  break;
               case RHS:
                  obj->vec_b.handle = rhs;
                  break;
               case X:
                  obj->vec_x.handle = solution;
                  break;
               default:
                  break;
            }
         }
         ASSERT_NE(code, ERROR_NONE);
         uint32_t low = 0, high = 0;
         MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
         ASSERT_EQ_U32(low, high);
         ASSERT_PTR_EQ(obj->iargs, args);
         ASSERT_PTR_EQ(obj->solver, solver);
         ASSERT_PTR_EQ(obj->precon, precon);
         ASSERT_TRUE(obj->precon_is_setup);
         ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
         ASSERT_PTR_EQ(obj->mat_M.handle, precmat);
         ASSERT_PTR_EQ(obj->vec_b.handle, rhs);
         ASSERT_PTR_EQ(obj->vec_x.handle, solution);
         ASSERT_EQ(obj->current_system_index, system_index);
         ASSERT_PTR_EQ(obj->precon_reuse_state, history);
         ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, history_age);

         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         HYPRE_Complex *values = NULL;
         ASSERT_EQ(test_copy_solution_values(obj, &values), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)values[0], 0.5, 1e-12);
      }
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_default_solution_creation_preserves_configured_peers(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t obj = create_distributed_test_object(NULL);
      if (locations[p])
      {
         char *argv[] = {"general:\n  statistics: off\n  exec_policy: device\n"
                         "solver: pcg\npreconditioner: amg\n"};
         ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      }
      HYPRE_IJVector original_x = obj->vec_x.handle, original_guess = obj->vec_x0.handle;
      void          *native_guess = NULL;
      ASSERT_EQ(HYPRE_IJVectorGetObject(original_guess, &native_guess), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)native_guess, 7.0), 0);
      char prefix[1024] = {0};
      if (!rank) test_scratch_path(prefix, sizeof(prefix), "ensured-initial-guess");
      MPI_Bcast(prefix, sizeof(prefix), MPI_CHAR, 0, MPI_COMM_WORLD);
      ASSERT_EQ(HYPRE_IJVectorPrint(original_guess, prefix), 0);
      snprintf(obj->iargs->ls.x0_filename, sizeof(obj->iargs->ls.x0_filename), "%s%s",
               prefix, rank == 1 ? "-missing" : "");
      if (rank == 1)
      {
         HYPRE_IJVectorDestroy(obj->vec_x.handle);
         obj->vec_x.handle = NULL;
         obj->vec_x.owned  = false;
      }
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      HYPRE_Precon precon = obj->precon;
      hypredrv_Solver *solver = obj->solver;
      ASSERT_NE(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_PTR_EQ(obj->precon, precon);
      ASSERT_PTR_EQ(obj->solver, solver);
      ASSERT_FALSE(obj->precon_is_setup);
      ASSERT_PTR_EQ(obj->vec_x.handle, rank == 1 ? NULL : original_x);
      ASSERT_PTR_EQ(obj->vec_x0.handle, original_guess);
      snprintf(obj->iargs->ls.x0_filename, sizeof(obj->iargs->ls.x0_filename), "%s",
               prefix);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_NOT_NULL(obj->vec_x.handle);
      if (rank != 1)
      {
         ASSERT_PTR_EQ(obj->vec_x.handle, original_x);
         ASSERT_PTR_EQ(obj->vec_x0.handle, original_guess);
      }
      HYPRE_BigInt  row         = rank;
      HYPRE_Complex guess_value = 0.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x0.handle, 1, &row, &guess_value), 0);
      ASSERT_EQ_DOUBLE((double)guess_value, 7.0, 1e-12);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      HYPRE_Complex *solution = NULL;
      ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)solution[0], 0.5, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_rank_local_mgr_outer_solver_rejection_is_collective(void)
{
#if HYPREDRV_HAS_MGR_DEV_FEATURES
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (int outer = 0; outer < 2; outer++)
   {
      HYPREDRV_t obj            = create_distributed_test_object(NULL);
      obj->iargs->precon_method = PRECON_MGR;
      hypredrv_PreconArgsSetDefaultsForMethod(PRECON_MGR, &obj->iargs->precon);
      MGR_args *mgr                                         = &obj->iargs->precon.mgr;
      mgr->num_levels                                       = 2;
      mgr->level[0].f_dofs.size                             = 1;
      mgr->level[0].f_dofs.data[0]                          = 0;
      mgr->level[0].prolongation_type                       = 2;
      mgr->level[0].restriction_type                        = 0;
      mgr->level[0].coarse_level_type                       = 0;
      mgr->level[0].matched_f_backsolve                     = 2;
      mgr->level[0].f_relaxation.type                       = 2;
      mgr->level[0].f_relaxation.symmetric_diagonal_scaling = 1;
      mgr->level[0].g_relaxation.type                       = -1;
      int label                                             = 0;
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
      obj->iargs->solver_method = rank == 1 ? SOLVER_GMRES : SOLVER_FGMRES;
      hypredrv_SolverArgsSetDefaultsForMethod(obj->iargs->solver_method,
                                              &obj->iargs->solver);
      uint32_t code =
         outer ? HYPREDRV_LinearSolverCreate(obj) : HYPREDRV_PreconCreate(obj);
      ASSERT_TRUE(code & ERROR_INVALID_SOLVER);
      ASSERT_NULL(obj->precon);
      ASSERT_NULL(obj->solver);
      ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "fgmres"), ERROR_NONE);
      ASSERT_EQ(outer ? HYPREDRV_LinearSolverCreate(obj) : HYPREDRV_PreconCreate(obj),
                ERROR_NONE);
      ASSERT_NOT_NULL(obj->precon);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
#endif
}

static void
test_preconditioner_creation_recovers_rank_local_missing_handle(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   const int num_cases =
#ifdef HYPRE_USING_GPU
      6;
#else
      3;
#endif
   for (int c = 0; c < num_cases; c++)
   {
      bool       pending_setup = c % 3 == 1;
      HYPREDRV_t obj           = create_distributed_test_object(NULL);
      if (c >= 3)
      {
         char *argv[] = {"general:\n  statistics: off\n  exec_policy: device\n"
                         "linear_system:\n  init_guess_mode: zeros\n"
                         "solver: pcg\npreconditioner: amg\n"};
         ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      }
      /* Without timestep metadata this policy holds an existing handle. A
       * missing handle must still force a collective creation decision. */
      obj->iargs->precon_reuse.enabled      = 1;
      obj->iargs->precon_reuse.per_timestep = 1;
      ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
      if (!pending_setup)
      {
         ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      }
      /* Model a rank whose preceding native creation/setup attempt left no
       * consumer. Release its borrower before the preconditioner itself. */
      if (rank == 1)
      {
         if (c % 3 != 2)
         {
            hypredrv_SolverDestroy(&obj->solver);
            hypredrv_PreconDestroy(&obj->precon, obj->stats, 0);
            ASSERT_NULL(obj->precon);
         }
         else ASSERT_TRUE(obj->precon->setup_called);
         obj->precon_is_setup = false;
      }
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_NOT_NULL(obj->precon);
      ASSERT_FALSE(obj->precon_is_setup);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      HYPRE_Complex *solution = NULL;
      ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)solution[0], 0.5, 1e-12);
      if (rank == 1)
      {
         hypredrv_SolverDestroy(&obj->solver);
         hypredrv_PreconDestroy(&obj->precon, obj->stats, 0);
         obj->precon_is_setup = false;
      }
      ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
      ASSERT_NULL(obj->solver);
      ASSERT_TRUE(rank == 1 ? obj->precon == NULL : obj->precon != NULL);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_FALSE(obj->precon_is_setup);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)solution[0], 0.5, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_collective_reuse_bookkeeping_is_published_after_agreement(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj    = create_distributed_test_object(NULL);
   char      *argv[] = {
      "general:\n  statistics: off\n  exec_policy: host\n"
           "solver: pcg\npreconditioner:\n  reuse: adaptive\n  amg:\n    print_level: 0\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, 1);
   obj->iargs->precon_reuse.guards.max_reuse_solves        = rank == 1 ? 1 : 100;
   obj->iargs->precon_reuse.guards.min_reuse_solves        = rank == 1 ? 0 : 100;
   obj->iargs->precon_reuse.guards.rebuild_on_new_timestep = 0;
   obj->stats->level_current_id[0]                         = rank + 2;
   PreconReuseState *original                              = obj->precon_reuse_state;
   int               original_age = test_reuse_decision(original, 0).age;
   HYPRE_Precon     original_precon                        = obj->precon;
   hypredrv_Solver *original_solver                        = obj->solver;
   for (int phase = PRECON_REUSE_CREATE; phase <= PRECON_REUSE_DESTROY; phase++)
   {
      HYPREDRV_ErrorCodeClear();
      if (rank == 1) hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      ASSERT_FALSE(hypredrv_PreconReuseDecide(obj, (PreconReusePhase)phase));
      ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->precon, original_precon);
      ASSERT_PTR_EQ(obj->solver, original_solver);
      ASSERT_TRUE(obj->precon_is_setup);
      ASSERT_PTR_EQ(obj->precon_reuse_state, original);
      ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, original_age);
   }
   /* Only rank one asks to rebuild. The collective decision resets the history
    * everywhere and captures each rank's current annotation context. */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_FALSE(obj->precon_is_setup);
   ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, 0);

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, 1);
   ASSERT_NULL(strstr(test_reuse_decision(obj->precon_reuse_state, 1).summary,
                      "guard=solver_failure"));
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_collective_reuse_does_not_publish_failed_bad_decisions(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj    = create_distributed_test_object(NULL);
   char      *argv[] = {
      "general:\n  statistics: off\n  exec_policy: host\n"
           "solver: pcg\npreconditioner:\n  reuse:\n    type: adaptive\n    guards:\n"
           "      max_iteration_ratio: -1\n      bad_decisions_to_rebuild: 2\n"
           "  amg:\n    print_level: 0\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   PreconReuseObservation observation = {
      .system_index = 1, .iters = 1, .solve_succeeded = 1};
   hypredrv_PreconReuseStateRecordObservation(obj->precon_reuse_state, &observation);
   observation.system_index = 2;
   hypredrv_PreconReuseStateRecordObservation(obj->precon_reuse_state, &observation);
   observation.system_index = 3;
   observation.iters        = 1000;
   hypredrv_PreconReuseStateRecordObservation(obj->precon_reuse_state, &observation);
   for (int phase = PRECON_REUSE_CREATE; phase <= PRECON_REUSE_DESTROY; phase++)
   {
      if (rank == 1) hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      ASSERT_FALSE(hypredrv_PreconReuseDecide(obj, (PreconReusePhase)phase));
      ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      hypredrv_ErrorStateReset();
   }
   /* A rejected peer-wide decision must not count toward the bad-decision limit. */
   int                 next_system = obj->stats->ls_counter + 1;
   PreconReuseDecision decision =
      test_reuse_decision(obj->precon_reuse_state, next_system);
   ASSERT_FALSE(decision.should_rebuild);
   ASSERT_NOT_NULL(strstr(decision.summary, "status=hold"));
   decision = test_reuse_decision(obj->precon_reuse_state, next_system + 1);
   ASSERT_TRUE(decision.should_rebuild);
   ASSERT_NOT_NULL(strstr(decision.summary, "status=rebuild"));
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
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

#if HYPRE_CHECK_MIN_VERSION(30100, 0)
static int equil_child_calls, equil_probe_only, equil_setup_fail, equil_solve_fail;
static HYPRE_PtrToSolverFcn equil_child_setup, equil_child_solve;

static HYPRE_Int
equil_child_setup_probe(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                        HYPRE_Vector x)
{
   HYPRE_Int code = equil_child_setup(solver, A, b, x);
   int       rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (equil_setup_fail && rank == 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Injected rank-local child setup failure");
      return HYPRE_GetError();
   }
   return code;
}

static HYPRE_Int
equil_child_solve_probe(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                        HYPRE_Vector x)
{
   if (!equil_probe_only) return equil_child_solve(solver, A, b, x);
   equil_child_calls++;
   if (equil_solve_fail)
   {
      hypre_ParVectorSetConstantValues((hypre_ParVector *)x, 10.0);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 1)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Injected rank-local child solve failure");
         return HYPRE_GetError();
      }
   }
   return 0;
}

static void
equil_assert_failure_on_every_rank(HYPRE_Int code)
{
   int failed = code != 0, all_failed = 0;
   MPI_Allreduce(&failed, &all_failed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   ASSERT_TRUE(all_failed);
   HYPRE_ClearAllErrors();
}

static void
test_scaled_frelax_rejects_rank_local_solve_inputs(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   for (int empty = 0; empty < 2; empty++)
   {
      HYPREDRV_t     obj          = create_distributed_test_object(NULL);
      HYPRE_IJMatrix empty_matrix = NULL;
      HYPRE_IJVector empty_rhs    = NULL;
      if (empty)
      {
         HYPRE_BigInt  lower = rank ? rank - 1 : 0, upper = rank - 1;
         HYPRE_Int     ncols = 1;
         HYPRE_Complex value = 2.0, rhs = 1.0;
         ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, lower, upper, lower, upper,
                                        &empty_matrix),
                   0);
         ASSERT_EQ(HYPRE_IJMatrixSetObjectType(empty_matrix, HYPRE_PARCSR), 0);
         ASSERT_EQ(HYPRE_IJMatrixInitialize(empty_matrix), 0);
         if (rank)
            ASSERT_EQ(
               HYPRE_IJMatrixSetValues(empty_matrix, 1, &ncols, &lower, &lower, &value),
               0);
         ASSERT_EQ(HYPRE_IJMatrixAssemble(empty_matrix), 0);
         ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, lower, upper, &empty_rhs), 0);
         ASSERT_EQ(HYPRE_IJVectorSetObjectType(empty_rhs, HYPRE_PARCSR), 0);
         ASSERT_EQ(HYPRE_IJVectorInitialize(empty_rhs), 0);
         if (rank) ASSERT_EQ(HYPRE_IJVectorSetValues(empty_rhs, 1, &lower, &rhs), 0);
         ASSERT_EQ(HYPRE_IJVectorAssemble(empty_rhs), 0);
         ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)empty_matrix),
                   ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)empty_rhs), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      }
      HYPRE_Solver inner = NULL;
      ASSERT_EQ(HYPRE_BoomerAMGCreate(&inner), 0);
      ASSERT_EQ(HYPRE_BoomerAMGSetMaxIter(inner, 1), 0);
      ASSERT_EQ(HYPRE_BoomerAMGSetTol(inner, 0.0), 0);
      hypre_Solver *child      = (hypre_Solver *)inner;
      equil_child_setup        = hypre_SolverSetup(child);
      equil_child_solve        = hypre_SolverSolve(child);
      hypre_SolverSetup(child) = equil_child_setup_probe;
      hypre_SolverSolve(child) = equil_child_solve_probe;
      HYPRE_Solver wrapper = hypredrv_MGRFRelaxEquilWrapperCreate(MPI_COMM_WORLD, inner);
      ASSERT_NOT_NULL(wrapper);
      hypre_Solver *base = (hypre_Solver *)wrapper;
      void         *A = NULL, *b = NULL, *x = NULL;
      ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &A), 0);
      ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &b), 0);
      ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &x), 0);
      equil_probe_only = 0;
      equil_setup_fail = equil_solve_fail = 0;
      equil_assert_failure_on_every_rank(
         hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)b, (HYPRE_Vector)x));
      ASSERT_EQ(hypre_SolverSetup(base)(wrapper, (HYPRE_Matrix)A, NULL, NULL), 0);

      HYPRE_IJMatrix self_matrix = NULL;
      void          *self_A      = NULL;
      if (rank == 1)
      {
         HYPRE_BigInt  row   = 0;
         HYPRE_Int     ncols = 1;
         HYPRE_Complex value = 2.0;
         ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &self_matrix), 0);
         ASSERT_EQ(HYPRE_IJMatrixSetObjectType(self_matrix, HYPRE_PARCSR), 0);
         ASSERT_EQ(HYPRE_IJMatrixInitialize(self_matrix), 0);
         ASSERT_EQ(HYPRE_IJMatrixSetValues(self_matrix, 1, &ncols, &row, &row, &value),
                   0);
         ASSERT_EQ(HYPRE_IJMatrixAssemble(self_matrix), 0);
         ASSERT_EQ(HYPRE_IJMatrixGetObject(self_matrix, &self_A), 0);
      }
      for (int bad = 0; bad < 4; bad++)
      {
         void *input_A = A;
         if (rank == 1)
         {
            if (bad == 0) input_A = NULL;
            if (bad == 1)
               hypre_CSRMatrixData(hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A))[0] =
                  -2.0;
            if (bad == 2) input_A = self_A;
            if (bad == 3)
               hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                                 "Injected callback admission error");
         }
         HYPRE_Int code =
            hypre_SolverSetup(base)(wrapper, (HYPRE_Matrix)input_A, NULL, NULL);
         if (rank == 1)
            hypre_CSRMatrixData(hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A))[0] = 2.0;
         equil_assert_failure_on_every_rank(code);
      }
      if (self_matrix) ASSERT_EQ(HYPRE_IJMatrixDestroy(self_matrix), 0);
      equil_assert_failure_on_every_rank(
         hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)b, (HYPRE_Vector)x));
      equil_setup_fail = 1;
      equil_assert_failure_on_every_rank(
         hypre_SolverSetup(base)(wrapper, (HYPRE_Matrix)A, NULL, NULL));
      equil_assert_failure_on_every_rank(
         hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)b, (HYPRE_Vector)x));
      equil_setup_fail = 0;
      ASSERT_EQ(hypre_SolverSetup(base)(wrapper, (HYPRE_Matrix)A, NULL, NULL), 0);

      hypre_ParVector *par_b         = (hypre_ParVector *)b;
      hypre_Vector    *local         = hypre_ParVectorLocalVector(par_b);
      HYPRE_Int        original_size = hypre_VectorSize(local);
      HYPRE_BigInt     global_size   = hypre_ParVectorGlobalSize(par_b);
      HYPRE_BigInt     first_index   = hypre_ParVectorFirstIndex(par_b);
      HYPRE_Complex   *original_data = hypre_VectorData(local);
      for (int bad = 0; bad < 6; bad++)
      {
         equil_child_calls = 0;
         equil_probe_only  = 1;
         void *input_b = b, *input_x = x;
         if (rank == 1)
         {
            if (bad == 0) input_b = NULL;
            if (bad == 1) hypre_VectorSize(local)++;
            if (bad == 2) hypre_VectorData(local) = NULL;
            if (bad == 3) hypre_ParVectorGlobalSize(par_b)++;
            if (bad == 4) hypre_ParVectorFirstIndex(par_b)++;
            if (bad == 5) input_x = NULL;
         }
         HYPRE_Int code = hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)input_b,
                                                  (HYPRE_Vector)input_x);
         hypre_VectorSize(local)          = original_size;
         hypre_VectorData(local)          = original_data;
         hypre_ParVectorGlobalSize(par_b) = global_size;
         hypre_ParVectorFirstIndex(par_b) = first_index;
         int calls                        = 0;
         MPI_Allreduce(&equil_child_calls, &calls, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         ASSERT_EQ(calls, 0);
         equil_assert_failure_on_every_rank(code);
      }
      equil_solve_fail = 1;
      equil_assert_failure_on_every_rank(
         hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)b, (HYPRE_Vector)x));
      if (!empty || rank)
         ASSERT_EQ_DOUBLE(
            hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)x))[0], 0.0,
            1e-12);
      equil_probe_only = equil_solve_fail = 0;
      ASSERT_EQ(hypre_SolverSolve(base)(wrapper, NULL, (HYPRE_Vector)b, (HYPRE_Vector)x),
                0);
      HYPRE_Complex *solution = NULL;
      ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
      if (!empty || rank) ASSERT_EQ_DOUBLE(solution[0], 0.5, 1e-12);
      ASSERT_EQ(hypredrv_MGRFRelaxEquilWrapperDestroy(wrapper), 0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      if (empty_matrix) ASSERT_EQ(HYPRE_IJMatrixDestroy(empty_matrix), 0);
      if (empty_rhs) ASSERT_EQ(HYPRE_IJVectorDestroy(empty_rhs), 0);
   }
}

#endif

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
test_scaled_setup_preserves_local_handle_registration(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t     obj = create_distributed_test_object("dofmap_mag");
   HYPRE_BigInt   row = rank, columns[2] = {rank, rank ^ 1};
   HYPRE_Int      ncols           = 2;
   HYPRE_Complex  coefficients[2] = {4.0, -1.0};
   HYPRE_IJMatrix matrices[2]     = {NULL, NULL};
   for (int i = 0; i < 2; i++)
   {
      ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, row, row, row, row, &matrices[i]),
                0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrices[i], HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJMatrixInitialize(matrices[i]), 0);
      ASSERT_EQ(
         HYPRE_IJMatrixSetValues(matrices[i], 1, &ncols, &row, columns, coefficients), 0);
      ASSERT_EQ(HYPRE_IJMatrixAssemble(matrices[i]), 0);
      /* Each candidate is a complete distributed operator before registration. */
      void *native = NULL;
      ASSERT_EQ(HYPRE_IJMatrixGetObject(matrices[i], &native), 0);
      ASSERT_EQ(hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *)native), 0);
   }
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrices[0]), ERROR_NONE);
   const int label = 0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_FALSE(hypredrv_ScalingIsApplied(obj->scaling_ctx));

   /* These registrations must stay local even with off-process scaling factors. */
   if (rank == 1)
   {
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrices[1]),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)obj->vec_b.handle),
                ERROR_NONE);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   HYPRE_Complex values[2] = {0.0, 0.0};
   ASSERT_EQ(HYPRE_IJMatrixGetValues(matrices[0], 1, &ncols, &row, columns, values), 0);
   ASSERT_EQ_DOUBLE(values[0], 4.0, 1e-12);
   ASSERT_EQ_DOUBLE(values[1], -1.0, 1e-12);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, values), 0);
   ASSERT_EQ_DOUBLE(values[0], 1.0 / 3.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   for (int i = 0; i < 2; i++)
   {
      ASSERT_EQ(HYPRE_IJMatrixGetValues(matrices[i], 1, &ncols, &row, columns, values),
                0);
      ASSERT_EQ_DOUBLE(values[0], 4.0, 1e-12);
      ASSERT_EQ_DOUBLE(values[1], -1.0, 1e-12);
      ASSERT_EQ(HYPRE_IJMatrixDestroy(matrices[i]), 0);
   }
}

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
   ASSERT_TRUE(!state->scaling_ctx || !hypredrv_ScalingIsApplied(state->scaling_ctx));
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
   ASSERT_FALSE(hypredrv_ScalingIsApplied(state->scaling_ctx));

   inject_rank_local_hypre_error(0);
   uint32_t code = HYPREDRV_LinearSolverApply(obj);
   assert_collective_hypre_failure(code);
   ASSERT_FALSE(hypredrv_ScalingIsApplied(state->scaling_ctx));
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

/* Rejected replacements must leave every rank's installed F-solver usable. */
static void
test_frelax_admission_preserves_the_live_hierarchy(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t   obj   = create_distributed_test_object(NULL);
   HYPRE_BigInt first = 3 * rank, last = first + 2;
   HYPRE_BigInt row_ptr[4]  = {0, 1, 2, 3};
   HYPRE_BigInt rows[3]     = {first, first + 1, last};
   HYPRE_Real   diagonal[3] = {2.0, 2.0, 2.0}, rhs[3] = {1.0, 1.0, 1.0};
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetMatrixFromCSR(obj, first, last, row_ptr, rows, diagonal),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, first, last, rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   MGR_args args;
   hypredrv_MGRSetDefaultArgs(&args);
   args.num_levels                 = 3;
   args.relax_type                 = 0;
   args.level[0].f_dofs            = (StackIntArray){.data = {0}, .size = 1};
   args.level[1].f_dofs            = (StackIntArray){.data = {1}, .size = 1};
   args.level[1].f_relaxation.type = 0;
   MGRfrlx_args *config            = &args.level[0].f_relaxation;
   config->type                    = 2;
   IntArray *dofmap                = NULL;
   const int labels[3]             = {0, 1, 2};
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, 3, labels, &dofmap);
   args.dofmap         = dofmap;
   HYPRE_Solver parent = NULL;
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
   void *A = NULL, *b = NULL, *x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &x), 0);
   ASSERT_EQ(hypredrv_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
             0);
#if !HYPRE_CHECK_MIN_VERSION(30000, 0)
   /* Match the driver's hierarchy rebuild on releases whose native setup
    * cannot safely reuse its hierarchy arrays. */
   hypredrv_MGRDestroy(&args, &parent, 1);
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
#endif
   ASSERT_EQ(hypredrv_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
             0);
   HYPRE_Int saved_interp = config->amg.aggressive.prolongation_type;
   HYPRE_Solver original     = args.frelax[0];
   for (int path = 0; path < 6; path++)
   {
      hypredrv_ErrorStateReset();
      if (rank == 1)
      {
         switch (path)
         {
            case 0:
               config->type = 999;
               break;
            case 1:
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               break;
            case 2:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "F-relax admission probe");
               break;
            case 3:
               config->amg.aggressive.prolongation_type = 999;
               break;
            case 4:
               args.active_level_map[0] = MAX_MGR_LEVELS;
               break;
            case 5:
               config->symmetric_diagonal_scaling = 1;
               break;
         }
      }
      ASSERT_FALSE(hypredrv_MGRFRelaxAttach(&args, parent, 0, 1, NULL, 0));
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_PTR_EQ(args.frelax[0], original);
      config->type                             = 2;
      config->symmetric_diagonal_scaling       = 0;
      config->amg.aggressive.prolongation_type = saved_interp;
      args.active_level_map[0]                 = 0;
      hypredrv_ErrorStateReset();
      ASSERT_EQ(HYPRE_ClearAllErrors(), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
      ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
                0);
      HYPRE_Complex values[3] = {0.0, 0.0, 0.0};
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 3, rows, values), 0);
      for (int i = 0; i < 3; i++) ASSERT_EQ_DOUBLE(values[i], 0.5, 1e-12);
   }
   args.keep_frelax[0] = 1;
   hypredrv_MGRDestroy(&args, &parent, 1);
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
   ASSERT_PTR_EQ(args.frelax[0], original);
#else
   ASSERT_NULL(args.frelax[0]);
#endif
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
   ASSERT_PTR_EQ(args.frelax[0], original);
#endif
   ASSERT_EQ(hypredrv_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
             0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
   ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_Complex kept_values[3] = {0.0, 0.0, 0.0};
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 3, rows, kept_values), 0);
   for (int i = 0; i < 3; i++) ASSERT_EQ_DOUBLE(kept_values[i], 0.5, 1e-12);
   args.keep_frelax[0] = 0;
   hypredrv_MGRDestroy(&args, &parent, 1);
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
   hypredrv_IntArrayDestroy(&dofmap);
   args.dofmap       = NULL;
   HYPRE_Solver cold = args.frelax[0];
   ASSERT_TRUE(hypredrv_MGRFRelaxAttach(&args, parent, 0, 1, NULL, 0));
   ASSERT_TRUE(args.frelax[0] != cold);
   ASSERT_EQ(hypredrv_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
             0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
   ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_Complex values[3] = {0.0, 0.0, 0.0};
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 3, rows, values), 0);
   for (int i = 0; i < 3; i++) ASSERT_EQ_DOUBLE(values[i], 0.5, 1e-12);
   args.num_levels          = 1;
   args.num_active_levels   = 0;
   args.active_level_map[0] = MAX_MGR_LEVELS;
   hypredrv_MGRDestroy(&args, &parent, 1);
   ASSERT_NULL(parent);
   ASSERT_NULL(args.frelax[0]);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_grelax_replacement_preserves_the_live_hierarchy(void)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   MGR_args   args;
   hypredrv_MGRSetDefaultArgs(&args);
   args.num_levels                 = 2;
   args.level[0].f_dofs            = (StackIntArray){.data = {0}, .size = 1};
   args.level[0].f_relaxation.type = 2;
   MGRgrlx_args *config            = &args.level[0].g_relaxation;
   config->type                    = 20;
   hypredrv_AMGSetDefaultArgs(&config->amg);
   config->amg.max_iter = 1;
   IntArray *dofmap     = NULL;
   const int label      = rank % 2;
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, 1, &label, &dofmap);
   args.dofmap         = dofmap;
   HYPRE_Solver parent = NULL;
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
   void *A = NULL, *b = NULL, *x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &x), 0);
   ASSERT_EQ(HYPRE_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_Solver original     = args.grelax[0];
   HYPRE_Int    saved_interp = config->amg.aggressive.prolongation_type;
   for (int path = 0; path < 6; path++)
   {
      hypredrv_ErrorStateReset();
      if (rank == 1)
      {
         switch (path)
         {
            case 0:
               config->type = 999;
               break;
            case 1:
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               break;
            case 2:
               config->type = 16;
               break;
            case 3:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "G-relax admission probe");
               break;
            case 4:
               config->amg.aggressive.prolongation_type = 999;
               break;
            case 5:
               args.active_level_map[0] = MAX_MGR_LEVELS;
               break;
         }
      }
      ASSERT_FALSE(hypredrv_MGRGRelaxAttach(&args, parent, 0, 1, NULL, 0));
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_PTR_EQ(args.grelax[0], original);
      config->type                             = 20;
      config->amg.aggressive.prolongation_type = saved_interp;
      args.active_level_map[0]                 = 0;
      hypredrv_ErrorStateReset();
      ASSERT_EQ(HYPRE_ClearAllErrors(), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
      ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
                0);
      HYPRE_BigInt  row   = rank;
      HYPRE_Complex value = 0.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, &value), 0);
      ASSERT_EQ_DOUBLE(value, 0.5, 1e-12);
   }
   config->type = 16;
   hypredrv_ILUSetDefaultArgs(&config->ilu);
   ASSERT_TRUE(hypredrv_MGRGRelaxAttach(&args, parent, 0, 0, NULL, 0));
   ASSERT_TRUE(args.grelax[0] != original);
   HYPRE_Solver         replacement = args.grelax[0];
   hypredrv_SolverBase *base        = (hypredrv_SolverBase *)replacement;
   ASSERT_EQ(base->setup(replacement, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x),
             0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
   ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_BigInt  row   = rank;
   HYPRE_Complex value = 0.0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, &value), 0);
   ASSERT_EQ_DOUBLE(value, 0.5, 1e-12);
   hypredrv_MGRDestroy(&args, &parent, 1);
   ASSERT_NULL(parent);
   ASSERT_NULL(args.grelax[0]);
   hypredrv_MGRGRelaxDestroy(&replacement);
   ASSERT_NULL(replacement);
   hypredrv_IntArrayDestroy(&dofmap);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
#endif
}

static void
test_coarse_replacement_preserves_all_ranks_on_rejection(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   MGR_args   args;
   hypredrv_MGRSetDefaultArgs(&args);
   args.num_levels  = 1;
   IntArray *dofmap = NULL;
   const int label  = 0;
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, 1, &label, &dofmap);
   args.dofmap         = dofmap;
   HYPRE_Solver parent = NULL;
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
   void *A = NULL, *b = NULL, *x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &x), 0);
   ASSERT_EQ(HYPRE_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_Solver original     = args.csolver;
   HYPRE_Int    saved_interp = args.coarsest_level.amg.aggressive.prolongation_type;
   for (int path = 0; path < 5; path++)
   {
      hypredrv_ErrorStateReset();
      if (rank == 1)
      {
         switch (path)
         {
            case 0:
               args.coarsest_level.type = 999;
               break;
            case 1:
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               break;
            case 2:
               args.coarsest_level.type = 32;
               break;
            case 3:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "coarse admission probe");
               break;
            case 4:
               args.coarsest_level.amg.aggressive.prolongation_type = 999;
               break;
         }
      }
      ASSERT_FALSE(hypredrv_MGRCoarseSolverAttach(&args, parent, 1, NULL, 0));
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_PTR_EQ(args.csolver, original);
      ASSERT_EQ(args.csolver_type, 0);
      args.coarsest_level.type                             = 0;
      args.coarsest_level.amg.aggressive.prolongation_type = saved_interp;
      hypredrv_ErrorStateReset();
      ASSERT_EQ(HYPRE_ClearAllErrors(), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
      ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x),
                0);
      HYPRE_BigInt  row   = rank;
      HYPRE_Complex value = 0.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, &value), 0);
      ASSERT_EQ_DOUBLE(value, 0.5, 1e-12);
   }

   ASSERT_EQ(HYPRE_MGRDestroy(parent), 0);
   parent = NULL;
   if (rank == 1)
   {
      ASSERT_EQ(HYPRE_BoomerAMGDestroy(args.csolver), 0);
      args.csolver      = NULL;
      args.csolver_type = -1;
   }
   /* A missing cache on one rank requires every rank to create a candidate. */
   hypredrv_MGRCreate(MPI_COMM_WORLD, &args, &parent, NULL, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(parent);
   if (rank != 1) ASSERT_TRUE(args.csolver != original);
   HYPRE_Solver repaired            = args.csolver;
   args.coarsest_level.amg.max_iter = 0;
   ASSERT_TRUE(hypredrv_MGRCoarseSolverAttach(&args, parent, 1, NULL, 0));
   ASSERT_TRUE(args.csolver != repaired);
   ASSERT_EQ(args.coarsest_level.amg.max_iter, 1);
   ASSERT_EQ(HYPRE_MGRSetup(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
   ASSERT_EQ(HYPRE_MGRSolve(parent, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x),
             0);
   HYPRE_BigInt  row   = rank;
   HYPRE_Complex value = 0.0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, &value), 0);
   ASSERT_EQ_DOUBLE(value, 0.5, 1e-12);
   hypredrv_MGRDestroy(&args, &parent, 1);
   ASSERT_NULL(parent);
   ASSERT_NULL(args.csolver);
   hypredrv_IntArrayDestroy(&dofmap);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_nested_frelax_creation_is_collective(void)
{
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t obj = create_distributed_test_object(NULL);
   MGR_args   parent, child;
   hypredrv_MGRSetDefaultArgs(&parent);
   hypredrv_MGRSetDefaultArgs(&child);
   IntArray *dofmap = NULL;
   const int label  = rank ? 7 : 2;
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, 1, &label, &dofmap);
   parent.comm                      = MPI_COMM_WORLD;
   parent.num_levels                = 2;
   parent.dofmap                    = dofmap;
   parent.level[0].f_dofs           = (StackIntArray){.data = {7, 2}, .size = 2};
   parent.level[0].f_relaxation.mgr = &child;
   child.num_levels                 = 1;
   child.dofmap                     = dofmap;
   child.vec_nn                     = obj->vec_b.handle;
   void *A = NULL, *b = NULL, *x = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &A), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &x), 0);
   for (int path = 0; path < 7; path++)
   {
      hypredrv_ErrorStateReset();
      if (rank == 1)
      {
         switch (path)
         {
            case 0:
               parent.level[0].f_dofs.data[1] = 99;
               break;
            case 1:
               parent.level[0].f_relaxation.mgr = NULL;
               break;
            case 2:
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "nested admission probe");
               break;
            case 3:
               dofmap->data[0] = -1;
               break;
            case 4:
               child.coarsest_level.type = 999;
               break;
            case 5:
               child.num_levels = 0;
               break;
            case 6:
               child.num_levels      = 2;
               child.level[0].f_dofs = (StackIntArray){.data = {-1}, .size = 1};
               break;
         }
      }
      HYPRE_Solver solver = hypredrv_MGRNestedFRelaxCreate(&parent, 0, NULL, 0);
      ASSERT_NULL(solver);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_PTR_EQ(child.dofmap, dofmap);
      ASSERT_PTR_EQ(child.vec_nn, obj->vec_b.handle);
      ASSERT_NULL(child.csolver);
      ASSERT_NULL(child.point_marker_data);

      hypredrv_ErrorStateReset();
      ASSERT_EQ(HYPRE_ClearAllErrors(), 0);
      dofmap->data[0]                  = label;
      parent.level[0].f_dofs.data[1]   = 2;
      parent.level[0].f_relaxation.mgr = &child;
      child.coarsest_level.type        = 0;
      child.num_levels                 = 1;
      solver = hypredrv_MGRNestedFRelaxCreate(&parent, 0, NULL, 0);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(solver);
      hypredrv_SolverBase *base = (hypredrv_SolverBase *)solver;
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, 0.0), 0);
      ASSERT_EQ(base->setup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x),
                0);
      ASSERT_EQ(base->solve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x),
                0);
      HYPRE_BigInt  row   = rank;
      HYPRE_Complex value = 0.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_x.handle, 1, &row, &value), 0);
      ASSERT_EQ_DOUBLE(value, 0.5, 1e-12);
      HYPRE_Solver alias = solver;
      if (path == 4) ASSERT_EQ(base->destroy(solver), 0);
      hypredrv_MGRNestedFRelaxDestroy(&solver);
      hypredrv_MGRNestedFRelaxDestroy(&alias);
      ASSERT_NULL(solver);
      ASSERT_NULL(alias);
      ASSERT_NULL(child.csolver);
      ASSERT_NULL(child.point_marker_data);
   }
   hypredrv_IntArrayDestroy(&dofmap);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

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
   HYPRE_IJMatrix  shell    = NULL;
   if (!rank)
   {
      ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &shell), 0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(shell, HYPRE_PARCSR), 0);
   }
   PreconOperators missing  = supplied;
   missing.G                = shell;
   const precon_t methods[] = {PRECON_AMS, PRECON_ADS};
   const solver_t solvers[] = {SOLVER_PCG, SOLVER_GMRES, SOLVER_FGMRES, SOLVER_BICGSTAB};
   for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++)
   {
      for (int kind = 0; kind < 2; kind++)
      {
         for (int path = 0; path < 5; path++)
         {
            precon_args args;
            hypredrv_PreconArgsSetDefaultsForMethod(methods[m], &args);
            HYPRE_Precon precon = NULL;
            hypredrv_PreconCreate(MPI_COMM_WORLD, methods[m], &args, NULL, NULL, &precon,
                                  NULL, 0, rank ? &supplied : (kind ? &missing : NULL));
            ASSERT_FALSE(hypredrv_ErrorCodeActive());
            ASSERT_NOT_NULL(precon);
            ASSERT_EQ(precon->operators_ok, rank ? 1 : 0);
            hypredrv_Solver *solver = NULL;
            if (path == 0)
            {
               hypredrv_PreconSetup(precon, obj->mat_A.handle);
            }
            else
            {
               solver_args solver_args;
               hypredrv_SolverArgsSetDefaultsForMethod(solvers[path - 1], &solver_args);
               hypredrv_SolverCreate(MPI_COMM_WORLD, solvers[path - 1], &solver_args,
                                     &solver);
               hypredrv_SolverSetupWithReuse(solver, precon, obj->mat_A.handle,
                                             obj->vec_b.handle, obj->vec_x.handle, NULL,
                                             NULL, 0);
            }
            uint32_t code = hypredrv_ErrorCodeGet(), low = 0, high = 0;
            MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
            ASSERT_EQ_U32(low, high);
            ASSERT_TRUE(code & ERROR_MISSING_PRECON);
            ASSERT_FALSE(precon->setup_called);
            hypredrv_ErrorStateReset();
            if (solver)
            {
               hypredrv_SolverDestroy(&solver);
            }
            hypredrv_PreconDestroy(&precon, NULL, 0);
         }
      }
   }
   if (shell)
   {
      HYPRE_IJMatrixDestroy(shell);
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
   for (int use_precon = 0; use_precon <= 1; use_precon++)
   {
      for (solver_t method = SOLVER_PCG; method <= SOLVER_BICGSTAB; method++)
      {
         NestedKrylov_args args;
         hypredrv_NestedKrylovSetDefaultArgs(&args);
         args.is_set        = 1;
         args.solver_method = method;
         hypredrv_SolverArgsSetDefaultsForMethod(method, &args.solver);
         args.has_precon = use_precon;
         if (use_precon)
         {
            args.precon_method = PRECON_BOOMERAMG;
            hypredrv_PreconArgsSetDefaultsForMethod(args.precon_method, &args.precon);
         }
         ASSERT_TRUE(hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, &args, NULL, NULL));
         ASSERT_FALSE(hypredrv_ErrorCodeActive());
         ASSERT_NOT_NULL(args.runtime);
         ASSERT_NE(hypredrv_NestedKrylovSetup((HYPRE_Solver)&args,
                                              rank ? (HYPRE_Matrix)matrix : NULL,
                                              (HYPRE_Vector)rhs, (HYPRE_Vector)solution),
                   0);
         ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
         ASSERT_FALSE(args.base.is_setup);
         hypredrv_ErrorStateReset();
         ASSERT_NE(hypredrv_NestedKrylovSolve((HYPRE_Solver)&args, (HYPRE_Matrix)matrix,
                                              (HYPRE_Vector)rhs, (HYPRE_Vector)solution),
                   0);
         ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
         hypredrv_ErrorStateReset();
         hypredrv_NestedKrylovDestroy(&args);
      }
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

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
test_scaling_preparation_rejects_rank_local_errors_collectively(void)
{
   HYPREDRV_t obj  = create_distributed_test_object(NULL);
   int        rank = 0;
   MPI_Comm_rank(obj->comm, &rank);
   Scaling_args args;
   hypredrv_ScalingSetDefaultArgs(&args);
   args.enabled         = 1;
   Scaling_context *ctx = NULL;
   ASSERT_TRUE(hypredrv_ScalingPrepare(obj->comm, &args, &ctx, obj->mat_A.handle,
                                       obj->vec_b.handle, NULL));
   Scaling_context *original = ctx;
   const int        label    = 0;
   IntArray        *dofmap   = NULL;
   hypredrv_IntArrayBuild(obj->comm, 1, &label, &dofmap);
   DoubleArray *values = hypredrv_DoubleArrayCreate(1);

   for (int failure = 0; failure < 5; failure++)
   {
      args.enabled = failure == 0 && rank == 0 ? 0 : 1;
      args.type    = failure == 1 && rank == 1 ? SCALING_DOFMAP_MAG : SCALING_RHS_L2;
      if (failure == 3) args.type = SCALING_DOFMAP_CUSTOM;
      args.custom_values = values;
      values->data[0]    = rank ? 1.0 : 0.0;
      if (failure == 4)
      {
         if (!rank)
            hypredrv_ScalingApplyToSystem(ctx, obj->mat_A.handle, obj->mat_A.handle,
                                          obj->vec_b.handle, obj->vec_x.handle);
         args.enabled = 0;
      }
      ASSERT_FALSE(hypredrv_ScalingPrepare(
         obj->comm, &args, &ctx, obj->mat_A.handle,
         failure == 2 && !rank ? NULL : obj->vec_b.handle, dofmap));
      uint32_t code = hypredrv_ErrorCodeGet(), low = 0, high = 0;
      MPI_Allreduce(&code, &low, 1, MPI_UINT32_T, MPI_MIN, obj->comm);
      MPI_Allreduce(&code, &high, 1, MPI_UINT32_T, MPI_MAX, obj->comm);
      ASSERT_EQ_U32(low, high);
      ASSERT_NE(code, ERROR_NONE);
      ASSERT_PTR_EQ(ctx, original);
      hypredrv_ErrorStateReset();
      HYPRE_ClearAllErrors();
      hypredrv_ScalingUndoOnSystem(ctx);
      ASSERT_FALSE(hypredrv_ScalingIsApplied(ctx));
   }
   HYPRE_BigInt  row   = rank;
   HYPRE_Int     ncols = 1;
   HYPRE_Complex value = 0.0;
   ASSERT_EQ(HYPRE_IJMatrixGetValues(obj->mat_A.handle, 1, &ncols, &row, &row, &value),
             0);
   ASSERT_EQ_DOUBLE(value, 2.0, 1e-12);
   ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_b.handle, 1, &row, &value), 0);
   ASSERT_EQ_DOUBLE(value, 1.0, 1e-12);
   hypredrv_ScalingContextDestroy(&ctx);
   hypredrv_DoubleArrayDestroy(&values);
   hypredrv_IntArrayDestroy(&dofmap);
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
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      RUN_TEST(test_mtx_matrix_is_partitioned_across_ranks);
#endif
      RUN_TEST(test_configured_construction_errors_are_collective);
      RUN_TEST(test_tagged_reduction_with_empty_rank);
   RUN_TEST(test_independent_self_solve_and_recycled_log_comm);
      RUN_TEST(test_rank_local_schedule_error_is_collective);
      RUN_TEST(test_mgr_short_marker_storage_fails_collectively);
      RUN_TEST(test_nested_frelax_creation_is_collective);
      RUN_TEST(test_coarse_replacement_preserves_all_ranks_on_rejection);
      RUN_TEST(test_grelax_replacement_preserves_the_live_hierarchy);
      RUN_TEST(test_frelax_admission_preserves_the_live_hierarchy);
      RUN_TEST(test_precon_operator_preflight_is_collective);
      RUN_TEST(test_nested_setup_preflight_is_collective);
      RUN_TEST(test_PreconSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_preconditioner_creation_recovers_rank_local_missing_handle);
      RUN_TEST(test_default_solution_creation_preserves_configured_peers);
      RUN_TEST(test_configured_matrix_ingress_is_collective);
      RUN_TEST(test_solver_workflow_preflight_is_collective);
      RUN_TEST(test_rank_local_mgr_outer_solver_rejection_is_collective);
      RUN_TEST(test_collective_reuse_bookkeeping_is_published_after_agreement);
      RUN_TEST(test_collective_reuse_does_not_publish_failed_bad_decisions);
      RUN_TEST(test_LinearSolverSetup_synchronizes_rank_local_hypre_error);
      RUN_TEST(test_reused_precon_remains_setup_after_solver_setup_error);
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
      RUN_TEST(test_scaled_frelax_rejects_rank_local_solve_inputs);
#endif
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
      RUN_TEST(test_scaled_setup_preserves_local_handle_registration);
      RUN_TEST(test_scaling_preparation_rejects_rank_local_errors_collectively);
      RUN_TEST(test_scaled_setup_restores_system_after_rank_local_hypre_error);
      RUN_TEST(test_scaled_solve_restores_system_after_rank_local_hypre_error);
      RUN_TEST(test_dofmap_scaling_rank_local_size_mismatch_fails_collectively);
#endif
   }

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
