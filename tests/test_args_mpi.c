/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/error.h"
#include "object.h"
#include "test_helpers.h"

static void
test_public_config_replacement_preserves_live_state(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char schedule[PATH_MAX] = {0}, missing[PATH_MAX] = {0};
   if (!rank)
   {
      test_scratch_path(schedule, sizeof(schedule), "mpi-live-schedule.txt");
      test_scratch_path(missing, sizeof(missing), "mpi-missing-schedule.txt");
      FILE *fp = fopen(schedule, "w");
      ASSERT_NOT_NULL(fp);
      ASSERT_TRUE(fputs("1\n31 0\n", fp) >= 0);
      ASSERT_EQ(fclose(fp), 0);
   }
   MPI_Bcast(schedule, sizeof(schedule), MPI_CHAR, 0, MPI_COMM_WORLD);
   MPI_Bcast(missing, sizeof(missing), MPI_CHAR, 0, MPI_COMM_WORLD);
   char yaml[PATH_MAX + 512];
   snprintf(
      yaml, sizeof(yaml),
      "general:\n  name: mpi-live\n  exec_policy: host\n  statistics: on\n"
      "  use_millisec: on\n  num_repetitions: 2\n"
      "linear_system:\n  timestep_filename: %s\n  init_guess_mode: zeros\n"
      "solver: gmres\npreconditioner:\n  reuse: adaptive\n  amg:\n    print_level: 0\n",
      schedule);
   char      *argv[] = {yaml};
   HYPREDRV_t obj    = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_BigInt rows[] = {0, 1}, columns[] = {rank};
   HYPRE_Real   values[] = {5.0}, rhs[] = {10.0};
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetMatrixFromCSR(obj, rank, rank, rows, columns, values),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, rank, rank, rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_TRUE(test_reuse_decision(obj->precon_reuse_state, 0).age > 0);
   input_args  *original_args   = obj->iargs;
   hypredrv_Solver *original_solver = obj->solver;
   HYPRE_Precon original_precon = obj->precon;
   IntArray    *original_ids    = obj->precon_reuse_timesteps.ids;
   IntArray    *original_starts = obj->precon_reuse_timesteps.starts;
   const int    num_cases =
#ifdef HYPRE_USING_GPU
      4;
#else
      3;
#endif
   for (int failure = 0; failure < num_cases; failure++)
   {
      snprintf(yaml, sizeof(yaml),
               "general:\n  name: candidate\n  statistics: on\n  exec_policy: host\n"
               "  num_repetitions: 9\nlinear_system:\n  timestep_filename: %s\n"
               "solver: pcg\npreconditioner: amg\n",
               failure == 1 && rank == 1 ? missing : schedule);
      if (failure == 0 && rank == 1)
         snprintf(yaml, sizeof(yaml),
                  "solver:\n  pcg:\n    max_iter: invalid\npreconditioner: amg\n");
#ifdef HYPRE_USING_GPU
      if (failure == 3 && rank == 1)
         snprintf(yaml, sizeof(yaml),
                  "general:\n  exec_policy: device\nsolver: pcg\npreconditioner:\n"
                  "  amg:\n    interpolation:\n      prolongation_type: standard\n");
#endif
      PreconReuseState *history = obj->precon_reuse_state;
      size_t history_count      = test_reuse_decision(obj->precon_reuse_state, 0).age;
      bool                    invalid_vector = failure == 2 && rank == 1;
      ASSERT_NE(HYPREDRV_InputArgsParse(invalid_vector ? 0 : 1,
                                        invalid_vector ? NULL : argv, obj),
                ERROR_NONE);
      ASSERT_PTR_EQ(obj->iargs, original_args);
      ASSERT_PTR_EQ(obj->solver, original_solver);
      ASSERT_PTR_EQ(obj->precon, original_precon);
      ASSERT_TRUE(obj->precon_is_setup);
      ASSERT_PTR_EQ(obj->precon_reuse_timesteps.ids, original_ids);
      ASSERT_PTR_EQ(obj->precon_reuse_timesteps.starts, original_starts);
      ASSERT_EQ(original_ids->data[0], 31);
      ASSERT_PTR_EQ(obj->precon_reuse_state, history);
      ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, history_count);
      ASSERT_STREQ(obj->stats->object_name, "mpi-live");
      ASSERT_EQ(obj->stats->num_reps, 2);
      ASSERT_TRUE(obj->stats->use_millisec);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      HYPRE_Complex *solution = NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
   }
   snprintf(yaml, sizeof(yaml),
            "general:\n  name: mpi-replacement\n  exec_policy: host\n  statistics: off\n"
            "solver: pcg\npreconditioner: amg\n");
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_TRUE(obj->iargs != original_args);
   ASSERT_NULL(obj->solver);
   ASSERT_NULL(obj->precon);
   ASSERT_NULL(obj->precon_reuse_timesteps.ids);
   ASSERT_NULL(obj->precon_reuse_timesteps.starts);
   ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, 0);
   ASSERT_STREQ(obj->stats->object_name, "mpi-replacement");
   ASSERT_FALSE(obj->stats->use_millisec);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   HYPRE_Complex *solution = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_preset_and_variant_changes_are_collective(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   HYPREDRV_t fresh = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &fresh), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(fresh), ERROR_NONE);
   ASSERT_NE(HYPREDRV_InputArgsSetPreconVariant(fresh, 0), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_SolverPresetRegister(
         "mpi-rejected-initial", rank == 1 ? "pcg:\n  max_iter: invalid\n" : "pcg", NULL),
      ERROR_NONE);
   ASSERT_NE(HYPREDRV_InputArgsSetSolverPreset(fresh, "mpi-rejected-initial"),
             ERROR_NONE);
   ASSERT_NULL(fresh->iargs);
   ASSERT_NULL(fresh->precon_reuse_state);
   for (int kind = 0; kind < 2; kind++)
      for (int failure = 0; failure < 2; failure++)
      {
         const char *name = kind ? "gmres" : "poisson";
         if (rank == 1) name = failure ? "missing-mpi-preset" : NULL;
         uint32_t code = kind ? HYPREDRV_InputArgsSetSolverPreset(fresh, name)
                              : HYPREDRV_InputArgsSetPreconPreset(fresh, name);
         ASSERT_NE(code, ERROR_NONE);
         ASSERT_NULL(fresh->iargs);
         ASSERT_NULL(fresh->precon_reuse_state);
      }
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(fresh, "gmres"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(fresh, 0), ERROR_NONE);
   ASSERT_NOT_NULL(fresh->precon_reuse_state);
   HYPRE_BigInt rows[] = {0, 1}, columns[] = {rank};
   HYPRE_Real   values[] = {4.0}, rhs[] = {2.0};
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetMatrixFromCSR(fresh, rank, rank, rows, columns, values),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(fresh, rank, rank, rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(fresh), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(fresh), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(fresh), ERROR_NONE);
   HYPRE_Complex *solution = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(fresh, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE(solution[0], 0.5, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&fresh), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_SolverPresetRegister("mpi-bad-solver",
                                           "pcg:\n  max_iter: invalid\n", NULL),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SolverPresetRegister("mpi-mixed-solver",
                                           rank == 1 ? "pcg" : "gmres", NULL),
             ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconPresetRegister("mpi-mixed-precon", rank == 1 ? "ilu" : "amg", NULL),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconPresetRegister(
                "mpi-owned-precon",
                "mgr:\n  level:\n    0:\n      f_dofs: [0]\n      f_relaxation:\n"
                "        gmres:\n          max_iter: 2\n  coarsest_level: amg\n",
                NULL),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconPresetRegister(
                "mpi-bad-precon",
                "mgr:\n  level:\n    0:\n      f_dofs: [0]\n      f_relaxation:\n"
                "        gmres:\n          max_iter: invalid\n  coarsest_level: amg\n",
                NULL),
             ERROR_NONE);
#ifdef HYPRE_USING_GPU
   ASSERT_EQ(HYPREDRV_PreconPresetRegister(
                "mpi-host-only-precon",
                "amg:\n  interpolation:\n    prolongation_type: standard\n", NULL),
             ERROR_NONE);
#endif
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      char yaml[512];
      snprintf(yaml, sizeof(yaml),
               "general:\n  exec_policy: %s\n  statistics: off\n"
               "linear_system:\n  init_guess_mode: zeros\n"
               "solver:\n  gmres:\n    print_level: 3\n"
               "preconditioner:\n  reuse: adaptive\n  amg:\n"
               "    - max_iter: 1\n    - max_iter: 2\n",
               locations[p] ? "device" : "host");
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      HYPRE_BigInt rows[] = {0, 1}, columns[] = {rank};
      HYPRE_Real   values[] = {5.0}, rhs[] = {10.0};
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetMatrixFromCSR(obj, rank, rank, rows, columns, values),
         ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, rank, rank, rhs), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      int label = 0;
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      ASSERT_TRUE(test_reuse_decision(obj->precon_reuse_state, 0).age > 0);
      input_args          *original_args     = obj->iargs;
      precon_t            *original_methods  = obj->iargs->precon_methods;
      precon_args         *original_variants = obj->iargs->precon_variants;
      hypredrv_Solver     *original_solver   = obj->solver;
      HYPRE_Precon         original_precon   = obj->precon;
      hypredrv_VectorTags *original_tags     = obj->vector_tags;

      const int num_cases = locations[p] ? 13 : 12;
      for (int failure = 0; failure < num_cases; failure++)
      {
         PreconReuseState *history = obj->precon_reuse_state;
         size_t history_count      = test_reuse_decision(obj->precon_reuse_state, 0).age;
         uint32_t                code;
         if (failure < 3)
         {
            int index = 1;
            if (rank == 1) index = failure == 0 ? -1 : failure == 1 ? 2 : 0;
            code = HYPREDRV_InputArgsSetPreconVariant(obj, index);
         }
         else if (failure < 7)
         {
            const char *name = "pcg";
            if (failure == 6) name = "mpi-mixed-solver";
            else if (rank == 1)
               name = failure == 3   ? NULL
                      : failure == 4 ? "missing-mpi-solver"
                                     : "mpi-bad-solver";
            code = HYPREDRV_InputArgsSetSolverPreset(obj, name);
         }
         else if (failure < 11)
         {
            const char *name =
               failure == 8 && !locations[p] ? "mpi-owned-precon" : "poisson";
            if (failure == 10) name = "mpi-mixed-precon";
            else if (rank == 1)
               name = failure == 7   ? NULL
                      : failure == 8 ? "missing-mpi-precon"
                                     : "mpi-bad-precon";
            code = HYPREDRV_InputArgsSetPreconPreset(obj, name);
         }
         else if (failure == 11)
            code = rank == 1 ? HYPREDRV_InputArgsSetSolverPreset(obj, "pcg")
                             : HYPREDRV_InputArgsSetPreconPreset(obj, "poisson");
         else
            code = HYPREDRV_InputArgsSetPreconPreset(
               obj, rank == 1 ? "mpi-host-only-precon" : "poisson");
         ASSERT_NE(code, ERROR_NONE);
         ASSERT_PTR_EQ(obj->iargs, original_args);
         ASSERT_PTR_EQ(obj->iargs->precon_methods, original_methods);
         ASSERT_PTR_EQ(obj->iargs->precon_variants, original_variants);
         ASSERT_EQ(obj->iargs->active_precon_variant, 0);
         ASSERT_EQ(obj->iargs->solver_method, SOLVER_GMRES);
         ASSERT_EQ(obj->iargs->precon_method, PRECON_BOOMERAMG);
         ASSERT_EQ(obj->iargs->precon.amg.max_iter, 1);
         ASSERT_EQ(original_variants[1].amg.max_iter, 2);
         ASSERT_PTR_EQ(obj->solver, original_solver);
         ASSERT_PTR_EQ(obj->precon, original_precon);
         ASSERT_PTR_EQ(obj->vector_tags, original_tags);
         ASSERT_TRUE(obj->precon_is_setup);
         ASSERT_PTR_EQ(obj->precon_reuse_state, history);
         ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age,
                        history_count);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         HYPRE_Complex *solution = NULL;
         ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
      }
      ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
      ASSERT_PTR_EQ(obj->solver, original_solver);
      ASSERT_PTR_EQ(obj->precon, original_precon);
      ASSERT_PTR_EQ(obj->vector_tags, original_tags);
      for (int change = 0; change < 3; change++)
      {
         if (change == 0)
            ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "pcg"), ERROR_NONE);
         else if (change == 1)
            ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
         else ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);
         ASSERT_NULL(obj->solver);
         ASSERT_NULL(obj->vector_tags);
         if (change == 0) ASSERT_PTR_EQ(obj->precon, original_precon);
         else
         {
            ASSERT_NULL(obj->precon);
            ASSERT_FALSE(obj->precon_is_setup);
         }
         ASSERT_EQ(obj->iargs->active_precon_variant, change == 2 ? 1 : 0);
         ASSERT_EQ(obj->iargs->precon.amg.max_iter, change == 2 ? 2 : 1);
         ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         HYPRE_Complex *solution = NULL;
         ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
         /* Reading values moves x to the host. Both an existing solver and
          * standalone preconditioner apply must restore the execution space. */
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_PreconApply(obj, (HYPRE_Vector)obj->vec_b.handle,
                                        (HYPRE_Vector)obj->vec_x.handle),
                   ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
      }
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_config_file_root_only(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char original_dir[PATH_MAX], local_dir[PATH_MAX];
   ASSERT_NOT_NULL(getcwd(original_dir, sizeof(original_dir)));
   test_scratch_path(local_dir, sizeof(local_dir), "config-cwd");
   ASSERT_EQ(mkdir(local_dir, 0700), 0);
   ASSERT_EQ(chdir(local_dir), 0);
   if (!rank)
   {
      const char *names[]    = {"config.yml", "solver.yml"};
      const char *contents[] = {"include: solver.yml\npreconditioner: amg\n",
                                "solver:\n  pcg:\n    max_iter: 17\n"};
      for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
      {
         FILE *fp = fopen(names[i], "w");
         ASSERT_NOT_NULL(fp);
         ASSERT_TRUE(fputs(contents[i], fp) >= 0);
         ASSERT_EQ(fclose(fp), 0);
      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   char       *argv[] = {"config.yml"};
   input_args *args   = NULL;
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, 1, argv, &args, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver.pcg.max_iter, 17);
   hypredrv_InputArgsDestroy(&args);
   char *driver_argv[] = {"hypredrive", "-i", "config.yml", "-a", "--solver:pcg:max_iter",
                          "23"};
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, 6, driver_argv, &args,
                                         NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->solver.pcg.max_iter, 23);
   hypredrv_InputArgsDestroy(&args);
   if (!rank)
   {
      FILE *fp = fopen("empty.yml", "w");
      ASSERT_NOT_NULL(fp);
      ASSERT_EQ(fclose(fp), 0);
   }
   MPI_Barrier(MPI_COMM_WORLD);
   argv[0] = "empty.yml";
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, 1, argv, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_KEY);
   ASSERT_NULL(args);
   ASSERT_EQ(chdir(original_dir), 0);
}

static void
test_config_source_validation_and_recovery(void)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   char       *inline_argv[]     = {"solver: pcg\npreconditioner: amg\n"};
   char       *null_argv[]       = {NULL};
   char       *override_argv[]   = {inline_argv[0], "--general:statistics", "off"};
   char       *null_value_argv[] = {inline_argv[0], "--general:statistics", NULL};
   char       *null_key_argv[]   = {inline_argv[0], NULL, "off"};
   input_args *args              = NULL;
   for (int failure = 0; failure < 7; failure++)
   {
      int          argc   = failure < 5 ? 1 : 3;
      char       **argv   = failure < 5 ? inline_argv : override_argv;
      input_args **output = &args;
      if (rank == 1)
      {
         if (failure == 0) argc = 0;
         if (failure == 1) argc = -1;
         if (failure == 2) argv = NULL;
         if (failure == 3) argv = null_argv;
         if (failure == 4) output = NULL;
         if (failure == 5) argv = null_value_argv;
         if (failure == 6) argv = null_key_argv;
      }
      hypredrv_ErrorStateReset();
      hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, argc, argv, output,
                                            NULL);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(args);
      ASSERT_TRUE(hypredrv_ErrorReportIsCollective());
   }
   char missing[PATH_MAX] = {0};
   if (!rank) test_scratch_path(missing, sizeof(missing), "missing.yml");
   MPI_Bcast(missing, sizeof(missing), MPI_CHAR, 0, MPI_COMM_WORLD);
   char *file_argv[] = {missing};
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParseWithObjectName(
      MPI_COMM_WORLD, true, 1, rank == 1 ? file_argv : inline_argv, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_NULL(args);
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, 1, file_argv, &args, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
   ASSERT_NULL(args);
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParseWithObjectName(MPI_COMM_WORLD, true, 1, inline_argv, &args,
                                         NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args);
   hypredrv_InputArgsDestroy(&args);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_config_file_root_only);
   RUN_TEST(test_config_source_validation_and_recovery);
   RUN_TEST(test_public_config_replacement_preserves_live_state);
   RUN_TEST(test_preset_and_variant_changes_are_collective);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
