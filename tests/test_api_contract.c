/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#include "HYPREDRV.h"
#include "_hypre_parcsr_mv.h"
#include "internal/args.h"
#include "internal/yaml.h"
#include "object.h"
#include "test_ij_helpers.h"
#include "vector_tags.h"

static uint32_t
set_matrix(HYPREDRV_t obj)
{
   HYPRE_BigInt rows[] = {0, 1}, cols[] = {0};
   HYPRE_Real   values[] = {5.0};
   return HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, rows, cols, values);
}

static uint32_t
set_rhs(HYPREDRV_t obj)
{
   HYPRE_Real values[] = {7.0};
   return HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, values);
}

static uint32_t
set_generated_rhs(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetRHS(obj, NULL);
}
static uint32_t
set_guess(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetInitialGuess(obj, NULL);
}
static uint32_t
set_reference(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetReferenceSolution(
      obj, (HYPRE_Vector)create_test_ijvector_1x1(3.0));
}
static uint32_t
set_prec_matrix(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL);
}
static uint32_t
set_nullspace(HYPREDRV_t obj)
{
   HYPRE_Complex value = 1.0;
   return HYPREDRV_LinearSystemSetNullSpace(obj, 1, 1, &value);
}
static uint32_t
set_near_nullspace(HYPREDRV_t obj)
{
   HYPRE_Complex value = 1.0;
   return HYPREDRV_LinearSystemSetNearNullSpace(obj, 1, 1, &value);
}

static void
test_independent_api_error_scopes(void)
{
   const struct
   {
      const char *name;
      uint32_t (*call)(HYPREDRV_t);
   } cases[] = {{"build", HYPREDRV_LinearSystemBuild},
                {"read matrix", HYPREDRV_LinearSystemReadMatrix},
                {"matrix from CSR", set_matrix},
                {"RHS", set_generated_rhs},
                {"RHS from array", set_rhs},
                {"initial guess", set_guess},
                {"reference", set_reference},
                {"preconditioner matrix", set_prec_matrix},
                {"nullspace", set_nullspace},
                {"near-nullspace", set_near_nullspace}};
   char prefix[1024], filename[2048];
   test_scratch_path(prefix, sizeof(prefix), "api-contract-matrix");
   snprintf(filename, sizeof(filename), "%s.00000", prefix);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPRE_IJMatrixPrint(matrix, prefix), 0);
   HYPRE_IJMatrixDestroy(matrix);
   HYPREDRV_t failed = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &failed), ERROR_NONE);
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      char yaml[PATH_MAX + 256];
      snprintf(
         yaml, sizeof(yaml),
         "general:\n  statistics: off\nlinear_system:\n  matrix_filename: %s\n"
         "  rhs_mode: ones\n  init_guess_mode: zeros\nsolver: pcg\npreconditioner: amg\n",
         prefix);
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_EQ(set_matrix(obj), ERROR_NONE);
      ASSERT_EQ(set_rhs(obj), ERROR_NONE);
      ASSERT_EQ(set_guess(obj), ERROR_NONE);
      ASSERT_NE(HYPREDRV_InputArgsGetWarmup(failed, NULL), ERROR_NONE);
      fprintf(stderr, "API error-scope case: %s\n", cases[i].name);
      ASSERT_EQ(cases[i].call(obj), ERROR_NONE);
      ASSERT_NOT_NULL(obj->mat_A.handle);
      ASSERT_NOT_NULL(obj->vec_b.handle);
      ASSERT_NOT_NULL(obj->vec_x.handle);
      if (cases[i].call == set_matrix)
         ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(obj->mat_A.handle), 5.0, 1e-15);
      if (cases[i].call == set_rhs)
         ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_b.handle), 7.0, 1e-15);
      if (cases[i].call == set_reference) ASSERT_NOT_NULL(obj->vec_xref.handle);
      if (cases[i].call == set_prec_matrix)
         ASSERT_PTR_EQ(obj->mat_M.handle, obj->mat_A.handle);
      if (cases[i].call == set_nullspace) ASSERT_NOT_NULL(obj->vec_ns);
      if (cases[i].call == set_near_nullspace) ASSERT_NOT_NULL(obj->vec_nn);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
   ASSERT_EQ(HYPREDRV_Destroy(&failed), ERROR_NONE);
   unlink(filename);
}

/* A failed configured read must leave the installed, set-up system usable.
 * Creating the missing inputs afterwards also exercises recovery through Build. */
static void
test_configured_replacement_failure_and_recovery(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   int initial_index = obj->current_system_index;
   ASSERT_TRUE(HYPREDRV_LinearSystemBuild(obj) & ERROR_INVALID_VAL);
   ASSERT_EQ(obj->current_system_index, initial_index);
   char prefix[1024];
   test_scratch_path(prefix, sizeof(prefix), "configured-replacement");
   char yaml[8192];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\n  exec_policy: host\n"
            "linear_system:\n  matrix_filename: %s-A\n  rhs_filename: %s-b\n"
            "  xref_filename: %s-ref\n  precmat_filename: %s-M\n"
            "  dofmap_filename: %s-dof\n  init_guess_mode: zeros\n"
            "solver:\n  gmres:\n    print_level: 3\npreconditioner: amg\n",
            prefix, prefix, prefix, prefix, prefix);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_BigInt rows[] = {0, 1}, cols[] = {0};
   HYPRE_Real   values[] = {2.0}, rhs_values[] = {4.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, rows, cols, values),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, rhs_values), ERROR_NONE);
   HYPRE_IJMatrix precon = create_test_ijmatrix_1x1(4.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)precon), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_b.handle),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   int label = 0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   /* A caller may reuse the currently installed labels as the next input. */
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, obj->dofmap->data), ERROR_NONE);
   ASSERT_EQ(obj->dofmap->data[0], 0);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   HYPRE_IJMatrix       matrix = obj->mat_A.handle;
   HYPRE_IJVector       rhs    = obj->vec_b.handle;
   IntArray            *dofmap = obj->dofmap;
   hypredrv_VectorTags *tags   = obj->vector_tags;
   hypredrv_Solver     *solver = obj->solver;

   ASSERT_EQ_U32(HYPREDRV_LinearSystemSetDofmap(obj, -1, &label), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, -1, 2),
                 ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, -1), ERROR_INVALID_VAL);
   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ObjectLinearSystemBuild(obj), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(obj->current_system_index, initial_index);
   ASSERT_TRUE(HYPREDRV_LinearSystemReadMatrix(obj) & ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, NULL) & ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL) &
               ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL) & ERROR_FILE_NOT_FOUND);
   ASSERT_NE(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);
   ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
   ASSERT_PTR_EQ(obj->mat_M.handle, precon);
   ASSERT_PTR_EQ(obj->vec_b.handle, rhs);
   ASSERT_PTR_EQ(obj->vec_xref.handle, rhs);
   ASSERT_PTR_EQ(obj->dofmap, dofmap);
   ASSERT_PTR_EQ(obj->vector_tags, tags);
   ASSERT_PTR_EQ(obj->solver, solver);
   ASSERT_TRUE(obj->mat_A.owned && obj->mat_M.owned && obj->vec_b.owned);
   ASSERT_FALSE(obj->vec_xref.owned);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   HYPRE_Complex *solution = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   /* Supply real files at the same configured paths and build the next system. */
   char           filename[MAX_FILENAME_LENGTH + 7];
   HYPRE_IJMatrix input_matrix = create_test_ijmatrix_1x1(8.0);
   ASSERT_EQ(HYPRE_IJMatrixPrint(input_matrix, obj->iargs->ls.matrix_filename), 0);
   HYPRE_IJMatrixDestroy(input_matrix);
   input_matrix = create_test_ijmatrix_1x1(4.0);
   ASSERT_EQ(HYPRE_IJMatrixPrint(input_matrix, obj->iargs->ls.precmat_filename), 0);
   HYPRE_IJMatrixDestroy(input_matrix);
   HYPRE_IJVector input_vector = create_test_ijvector_1x1(16.0);
   ASSERT_EQ(HYPRE_IJVectorPrint(input_vector, obj->iargs->ls.rhs_filename), 0);
   HYPRE_IJVectorDestroy(input_vector);
   input_vector = create_test_ijvector_1x1(2.0);
   ASSERT_EQ(HYPRE_IJVectorPrint(input_vector, obj->iargs->ls.xref_filename), 0);
   HYPRE_IJVectorDestroy(input_vector);
   snprintf(filename, sizeof(filename), "%s.00000", obj->iargs->ls.dofmap_filename);
   FILE *fp = fopen(filename, "w");
   ASSERT_NOT_NULL(fp);
   ASSERT_TRUE(fputs("1\n0\n", fp) >= 0);
   ASSERT_EQ(fclose(fp), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   ASSERT_EQ(obj->current_system_index, initial_index + 1);
   ASSERT_TRUE(obj->mat_A.owned && obj->mat_M.owned && obj->vec_b.owned &&
               obj->vec_xref.owned);
   ASSERT_EQ_SIZE(obj->dofmap->size, 1);
   ASSERT_EQ(obj->dofmap->data[0], 0);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
#ifdef HYPRE_USING_GPU
   /* Rebuild from the same inputs on the device, including M and the reference. */
   char *device_argv[] = {yaml, "--general:exec_policy", "device"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(3, device_argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   void *native_A = NULL, *native_M = NULL, *native_b = NULL, *native_ref = NULL;
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &native_A), 0);
   ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_M.handle, &native_M), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &native_b), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_xref.handle, &native_ref), 0);
   ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)native_A),
             HYPRE_MEMORY_DEVICE);
   ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)native_M),
             HYPRE_MEMORY_DEVICE);
   ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native_b),
             HYPRE_MEMORY_DEVICE);
   ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native_ref),
             HYPRE_MEMORY_DEVICE);
   matrix                   = obj->mat_A.handle;
   precon                   = obj->mat_M.handle;
   rhs                      = obj->vec_b.handle;
   HYPRE_IJVector reference = obj->vec_xref.handle;
   const char    *inputs[]  = {obj->iargs->ls.matrix_filename,
                               obj->iargs->ls.precmat_filename, obj->iargs->ls.rhs_filename,
                               obj->iargs->ls.xref_filename};
   for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++)
   {
      snprintf(filename, sizeof(filename), "%s.00000", inputs[i]);
      ASSERT_EQ(unlink(filename), 0);
   }
   ASSERT_TRUE(HYPREDRV_LinearSystemReadMatrix(obj) & ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL) & ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, NULL) & ERROR_FILE_NOT_FOUND);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL) &
               ERROR_FILE_NOT_FOUND);
   ASSERT_PTR_EQ(obj->mat_A.handle, matrix);
   ASSERT_PTR_EQ(obj->mat_M.handle, precon);
   ASSERT_PTR_EQ(obj->vec_b.handle, rhs);
   ASSERT_PTR_EQ(obj->vec_xref.handle, reference);
   ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)native_A),
             HYPRE_MEMORY_DEVICE);
   ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native_b),
             HYPRE_MEMORY_DEVICE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &solution), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1e-12);
#endif
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
#ifdef HYPRE_USING_GPU
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
}

static void
test_setup_creates_default_solution(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char  yaml[] = "solver: pcg\npreconditioner: amg\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector rhs    = create_test_ijvector_1x1(4.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_NULL(obj->vec_x.handle);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj->vec_x.handle);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 2.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   HYPRE_IJMatrixDestroy(matrix);
   HYPRE_IJVectorDestroy(rhs);
}

static void
test_config_replacement_validates_before_retiring_live_state(void)
{
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      char schedule[PATH_MAX], missing[PATH_MAX], malformed[PATH_MAX];
      test_scratch_path(schedule, sizeof(schedule),
                        locations[p] ? "live-device.txt" : "live-host.txt");
      test_scratch_path(missing, sizeof(missing), "missing-config-schedule.txt");
      test_scratch_path(malformed, sizeof(malformed), "malformed-config-schedule.txt");
      FILE *fp = fopen(schedule, "w");
      ASSERT_NOT_NULL(fp);
      ASSERT_TRUE(fputs("2\n11 0\n22 1\n", fp) >= 0);
      ASSERT_EQ(fclose(fp), 0);
      fp = fopen(malformed, "w");
      ASSERT_NOT_NULL(fp);
      ASSERT_TRUE(fputs("1\n0 -1\n", fp) >= 0);
      ASSERT_EQ(fclose(fp), 0);

      char yaml[PATH_MAX + 512];
      snprintf(yaml, sizeof(yaml),
               "general:\n  name: live-config\n  exec_policy: %s\n  statistics: on\n"
               "  use_millisec: on\n  num_repetitions: 2\n"
               "linear_system:\n  init_guess_mode: zeros\n  timestep_filename: %s\n"
               "solver:\n  gmres:\n    print_level: 3\n"
               "preconditioner:\n  reuse: adaptive\n  amg:\n    print_level: 0\n",
               locations[p] ? "device" : "host", schedule);
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_EQ(set_matrix(obj), ERROR_NONE);
      ASSERT_EQ(set_rhs(obj), ERROR_NONE);
      ASSERT_EQ(set_guess(obj), ERROR_NONE);
      int label = 0;
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      ASSERT_TRUE(test_reuse_decision(obj->precon_reuse_state, 0).age > 0);
#if HYPRE_CHECK_MIN_VERSION(30000, 0) && !defined(HYPRE_USING_GPU)
      ASSERT_NOT_NULL(obj->vector_tags);
#endif
      input_args          *original_args   = obj->iargs;
      hypredrv_Solver     *original_solver = obj->solver;
      HYPRE_Precon         original_precon = obj->precon;
      hypredrv_VectorTags *original_tags   = obj->vector_tags;
      IntArray            *original_ids    = obj->precon_reuse_timesteps.ids;
      IntArray            *original_starts = obj->precon_reuse_timesteps.starts;
      ASSERT_NOT_NULL(original_ids);
      ASSERT_NOT_NULL(original_starts);

      char missing_yaml[PATH_MAX + 256], malformed_yaml[PATH_MAX + 256];
      snprintf(missing_yaml, sizeof(missing_yaml),
               "general:\n  name: rejected\n  statistics: on\n  num_repetitions: 9\n"
               "linear_system:\n  timestep_filename: %s\n"
               "solver: pcg\npreconditioner: amg\n",
               missing);
      snprintf(malformed_yaml, sizeof(malformed_yaml),
               "general:\n  name: rejected\n  statistics: on\n  num_repetitions: 9\n"
               "linear_system:\n  timestep_filename: %s\n"
               "solver: pcg\npreconditioner: amg\n",
               malformed);
      char *rejected[] = {
         NULL,
         "solver:\n  pcg:\n    max_iter: invalid\npreconditioner: amg\n",
         "solver: pcg\n  invalid: child\npreconditioner: amg\n",
         missing_yaml,
         malformed_yaml,
#ifdef HYPRE_USING_GPU
         "general:\n  exec_policy: device\nsolver: pcg\n"
         "preconditioner:\n  amg:\n    interpolation:\n      prolongation_type: "
         "standard\n",
#endif
      };
      for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++)
      {
         PreconReuseState *history = obj->precon_reuse_state;
         size_t history_count      = test_reuse_decision(obj->precon_reuse_state, 0).age;
         char                   *bad_argv[]    = {rejected[i]};
         ASSERT_NE(HYPREDRV_InputArgsParse(i ? 1 : 0, i ? bad_argv : NULL, obj),
                   ERROR_NONE);
         ASSERT_PTR_EQ(obj->iargs, original_args);
         ASSERT_PTR_EQ(obj->solver, original_solver);
         ASSERT_PTR_EQ(obj->precon, original_precon);
         ASSERT_PTR_EQ(obj->vector_tags, original_tags);
         ASSERT_TRUE(obj->precon_is_setup);
         ASSERT_PTR_EQ(obj->precon_reuse_timesteps.ids, original_ids);
         ASSERT_PTR_EQ(obj->precon_reuse_timesteps.starts, original_starts);
         ASSERT_EQ(original_ids->data[0], 11);
         ASSERT_EQ(original_starts->data[1], 1);
         ASSERT_PTR_EQ(obj->precon_reuse_state, history);
         ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age,
                        history_count);
         ASSERT_STREQ(obj->stats->object_name, "live-config");
         ASSERT_EQ(obj->stats->num_reps, 2);
         ASSERT_TRUE(obj->stats->use_millisec);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 1.4, 1e-12);
      }

      snprintf(yaml, sizeof(yaml),
               "general:\n  name: replacement\n  statistics: off\n  exec_policy: %s\n"
               "solver: pcg\npreconditioner: amg\n",
               locations[p] ? "device" : "host");
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_TRUE(obj->iargs != original_args);
      ASSERT_NULL(obj->solver);
      ASSERT_NULL(obj->precon);
      ASSERT_NULL(obj->vector_tags);
      ASSERT_FALSE(obj->precon_is_setup);
      ASSERT_NULL(obj->precon_reuse_timesteps.ids);
      ASSERT_NULL(obj->precon_reuse_timesteps.starts);
      ASSERT_EQ_SIZE(test_reuse_decision(obj->precon_reuse_state, 0).age, 0);
      ASSERT_STREQ(obj->stats->object_name, "replacement");
      ASSERT_EQ(obj->stats->num_reps, 1);
      ASSERT_FALSE(obj->stats->use_millisec);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 1.4, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_yaml_input_contract(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char text[] =
      "---\r\nsolver:\r\n  pcg:\r\n    max_iter: 50\r\npreconditioner: amg\r\n...\r\n";
   char *argv[] = {text};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 50);
   char path[PATH_MAX];
   test_scratch_path(path, sizeof(path), "crlf-contract.yml");
   FILE *fp = fopen(path, "wb");
   ASSERT_NOT_NULL(fp);
   ASSERT_EQ_SIZE(fwrite(text, 1, strlen(text), fp), strlen(text));
   ASSERT_EQ(fclose(fp), 0);
   argv[0] = path;
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 50);
   unlink(path);
   char bad_indent[] = "solver:\n  pcg:\n      max_iter: 50\n    relative_tol: "
                       "1e-6\npreconditioner: amg\n";
   argv[0]           = bad_indent;
   ASSERT_NE(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   char duplicates[] =
      "solver:\n  pcg:\n    max_iter: 50\n    max_iter: 60\npreconditioner: amg\n";
   argv[0] = duplicates;
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 60);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_internal_call_status(void)
{
   HYPRE_BigInt rowptr[] = {0, 1}, column[] = {0};
   HYPRE_Real   values[] = {2.0};
   HYPREDRV_t   obj      = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemSetMatrixFromCSR(obj, 0, 0, rowptr, column, values),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_LinearSystemSetRHSFromArray(obj, 0, 0, values), ERROR_NONE);
   Stats *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);
   ASSERT_EQ(hypredrv_StatsAnnotateWithId(stats, HYPREDRV_ANNOTATE_BEGIN, "matrix", -1),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_StatsAnnotateWithId(stats, HYPREDRV_ANNOTATE_END, "matrix", -1),
             ERROR_NONE);
   ASSERT_EQ(
      hypredrv_StatsAnnotateLevelWithId(stats, HYPREDRV_ANNOTATE_BEGIN, 0, "level", -1),
      ERROR_NONE);
   ASSERT_EQ(
      hypredrv_StatsAnnotateLevelWithId(stats, HYPREDRV_ANNOTATE_END, 0, "level", -1),
      ERROR_NONE);
   int count = -1;
   ASSERT_EQ(hypredrv_StatsLevelGetCountChecked(stats, 0, &count, "test"), ERROR_NONE);
   ASSERT_EQ(count, 1);
   hypredrv_StatsDestroy(&stats);
   ASSERT_EQ(hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, NULL, NULL, NULL, NULL,
                                                NULL, NULL, NULL, NULL, NULL, NULL),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemSetRHSFromArray(obj, 0, 0, NULL), ERROR_INVALID_VAL);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP | ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   hypredrv_ErrorStateReset();
}

static void
test_internal_solver_workflows_preserve_errors(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char  yaml[] = "general:\n  exec_policy: host\n  statistics: off\n"
                  "solver: pcg\npreconditioner: amg\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(set_matrix(obj), ERROR_NONE);
   ASSERT_EQ(set_rhs(obj), ERROR_NONE);
   ASSERT_EQ(set_guess(obj), ERROR_NONE);

   uint32_t (*operations[])(HYPREDRV_t) = {hypredrv_ObjectLinearSolverCreate,
                                           hypredrv_ObjectLinearSolverDestroy};
   for (size_t i = 0; i < sizeof(operations) / sizeof(operations[0]); i++)
   {
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ASSERT_TRUE(operations[i](obj) & ERROR_MISSING_DOFMAP);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_DOFMAP);
   }

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_attachment_aliases(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector coordinate = create_test_ijvector_1x1(0.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, NULL, (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char *argv[] = {"solver: pcg\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   /* Reattaching an owned preconditioner matrix must preserve its lifetime. */
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_TRUE(obj->mat_M.owned);
   ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(matrix), 2.0, 1e-12);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(obj->mat_A.handle), 2.0, 1e-12);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)create_test_ijvector_1x1(4.0)),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_x.handle),
      ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_x.handle),
      ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 0.0, 1e-12);
   HYPRE_IJVector reference = create_test_ijvector_1x1(3.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)reference),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)reference),
             ERROR_NONE);
   ASSERT_TRUE(obj->vec_xref.owned);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(reference), 3.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_optional_shared_tags(void)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0) && !defined(HYPRE_USING_GPU)
   const struct
   {
      solver_t solver;
      int      level;
      int      tagged;
   } cases[]      = {{SOLVER_GMRES, 0, 0},   {SOLVER_GMRES, 2, 0},  {SOLVER_GMRES, 3, 1},
                     {SOLVER_FGMRES, 1, 0},  {SOLVER_FGMRES, 2, 1}, {SOLVER_PCG, 3, 0},
                     {SOLVER_BICGSTAB, 3, 0}};
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char *argv[] = {"solver: gmres\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJVector rhs = create_test_ijvector_1x1(4.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)rhs),
             ERROR_NONE);
   int label = 0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   void *par = NULL;
   HYPRE_IJVectorGetObject(rhs, &par);
   hypre_Vector *local_rhs = hypre_ParVectorLocalVector((hypre_ParVector *)par);
   HYPRE_IJVectorGetObject(obj->vec_x.handle, &par);
   hypre_Vector *local_x      = hypre_ParVectorLocalVector((hypre_ParVector *)par);
   HYPRE_Int     original_tag = 0;
   HYPRE_IJVectorSetTags(rhs, 0, 1, &original_tag);
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      obj->iargs->solver_method            = cases[i].solver;
      obj->iargs->solver.gmres.print_level = cases[i].level;
      if (cases[i].solver == SOLVER_FGMRES)
      {
         obj->iargs->solver.fgmres.print_level = cases[i].level;
      }
      ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
      if (cases[i].tagged)
      {
         ASSERT_NOT_NULL(obj->vector_tags);
         ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), hypre_VectorTags(local_x));
         ASSERT_TRUE(hypre_VectorTags(local_rhs) != &original_tag);
         ASSERT_EQ(hypre_VectorOwnsTags(local_rhs), 0);
         /* Repeated setup must restore and reattach without losing metadata. */
         ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
      }
      else
      {
         ASSERT_NULL(obj->vector_tags);
         ASSERT_NULL(hypre_VectorTags(local_x));
         ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
      }
   }
   obj->iargs->solver_method            = SOLVER_GMRES;
   obj->iargs->solver.gmres.print_level = 3;
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   hypredrv_VectorTags *tags              = obj->vector_tags;
   const char          *invalid_presets[] = {NULL, "no_such_tagged_solver"};
   for (size_t i = 0; i < sizeof(invalid_presets) / sizeof(invalid_presets[0]); i++)
   {
      ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, invalid_presets[i]),
                ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vector_tags, tags);
      ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), hypre_VectorTags(local_x));
   }
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   ASSERT_NULL(obj->vector_tags);
   ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   obj->iargs->general.exec_policy = 1;
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   ASSERT_NULL(obj->vector_tags);
   obj->iargs->general.exec_policy = 0;
   int mismatched_labels[]         = {0, 1};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, mismatched_labels), ERROR_NONE);
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_INVALID_VAL);
   ASSERT_NULL(obj->vector_tags);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
   ASSERT_EQ(hypre_VectorOwnsTags(local_rhs), 0);
   HYPRE_IJVectorDestroy(rhs);
#endif
}

static void
test_preset_replacement_preserves_solver_lifetimes(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char  yaml[] = "general:\n  exec_policy: host\n  statistics: off\n"
                  "solver: pcg\npreconditioner: amg\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(set_matrix(obj), ERROR_NONE);
   ASSERT_EQ(set_rhs(obj), ERROR_NONE);
   ASSERT_EQ(set_guess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   hypredrv_Solver *old_solver = obj->solver;
   HYPRE_Precon old_precon = obj->precon;

   ASSERT_EQ(HYPREDRV_SolverPresetRegister("api_contract_bad_solver",
                                           "gmres:\n  unknown_option: 1\n", "unit"),
             ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_InputArgsSetSolverPreset(obj, "api_contract_bad_solver") &
               ERROR_INVALID_KEY);
   ASSERT_PTR_EQ(obj->solver, old_solver);
   ASSERT_PTR_EQ(obj->precon, old_precon);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_PCG);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 1.4, 1e-12);

   ASSERT_EQ(HYPREDRV_SolverPresetRegister("api_contract_gmres",
                                           "gmres:\n  max_iter: 5\n  print_level: 0\n",
                                           "unit"),
             ERROR_NONE);
   /* Destruction must use the old PCG method, before the GMRES tag is installed. */
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "api_contract_gmres"), ERROR_NONE);
   ASSERT_NULL(obj->solver);
   ASSERT_PTR_EQ(obj->precon, old_precon);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_GMRES);
   ASSERT_EQ(obj->iargs->solver.gmres.max_iter, 5);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 1.4, 1e-12);

   old_solver                = obj->solver;
   old_precon                = obj->precon;
   const char *bad_precons[] = {
      "amg:\n  unknown_option: 1\n",
      "ilu: unknown_ilu_type\n",
      "mgr: unexpected_scalar\n",
      "mgr:\n  level:\n    0:\n      f_dofs: [0]\n      f_relaxation:\n"
      "        gmres:\n          max_iter: 2\n  unknown_option: 1\n",
   };
   for (size_t i = 0; i < sizeof(bad_precons) / sizeof(bad_precons[0]); i++)
   {
      char name[64];
      snprintf(name, sizeof(name), "api_contract_bad_precon_%zu", i);
      ASSERT_EQ(HYPREDRV_PreconPresetRegister(name, bad_precons[i], "unit"), ERROR_NONE);
      ASSERT_NE(HYPREDRV_InputArgsSetPreconPreset(obj, name), ERROR_NONE);
      ASSERT_PTR_EQ(obj->solver, old_solver);
      ASSERT_PTR_EQ(obj->precon, old_precon);
      ASSERT_TRUE(obj->precon_is_setup);
   }

   /* A successful preconditioner replacement must also retire its borrowing solver. */
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   ASSERT_NULL(obj->solver);
   ASSERT_NULL(obj->precon);
   ASSERT_FALSE(obj->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 1.4, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_rejected_presets_preserve_unconfigured_object(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   const char *rejected[] = {NULL, "missing-unconfigured-preset"};
   for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++)
   {
      ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, rejected[i]), ERROR_INVALID_VAL);
      ASSERT_NULL(obj->iargs);
      ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, rejected[i]), ERROR_INVALID_VAL);
      ASSERT_NULL(obj->iargs);
   }
   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_InputArgsSetPreset(obj, HYPREDRV_PRESET_SOLVER, "gmres"),
             ERROR_MISSING_DOFMAP);
   ASSERT_NULL(obj->iargs);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "gmres"), ERROR_NONE);
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_InputArgsSetPreconVariant(obj, 0), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_GMRES);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_presets_configure_active_variants(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, NULL), ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, NULL), ERROR_INVALID_VAL);
   ASSERT_NULL(obj->iargs);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "pcg"), ERROR_NONE);
   ASSERT_NULL(obj->iargs->precon_methods);
   ASSERT_NULL(obj->iargs->precon_variants);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "does_not_exist"), ERROR_INVALID_VAL);
   ASSERT_NULL(obj->iargs->precon_methods);
   ASSERT_NULL(obj->iargs->precon_variants);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "elasticity-2D"), ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon.amg.coarsening.num_functions, 2);
   ASSERT_NOT_NULL(obj->iargs->precon_methods);
   ASSERT_NOT_NULL(obj->iargs->precon_variants);

   ASSERT_EQ(HYPREDRV_SolverPresetRegister(
                "api_contract_fgmres",
                "fgmres:\n  max_iter: 17\n  krylov_dim: 23\n  print_level: 0\n", "unit"),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "api_contract_fgmres"), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_FGMRES);
   ASSERT_EQ(obj->iargs->solver.fgmres.max_iter, 17);
   ASSERT_EQ(obj->iargs->solver.fgmres.krylov_dim, 23);
   ASSERT_EQ(obj->iargs->solver.fgmres.print_level, 0);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "does_not_exist"), ERROR_INVALID_VAL);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_FGMRES);
   ASSERT_EQ(obj->iargs->solver.fgmres.max_iter, 17);

   char  yaml[] = "general:\n  exec_policy: host\n  statistics: off\n"
                  "linear_system:\n  dof_labels: {velocity: 0, pressure: 1}\n"
                  "solver: gmres\npreconditioner:\n  amg:\n"
                  "    - coarsening:\n        num_functions: 2\n"
                  "    - coarsening:\n        num_functions: 3\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   ASSERT_EQ(obj->iargs->active_precon_variant, 1);
   ASSERT_EQ(obj->iargs->precon.amg.coarsening.num_functions, 1);
   ASSERT_EQ(obj->iargs->precon_variants[0].amg.coarsening.num_functions, 2);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, -1), ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 2), ERROR_INVALID_VAL);
   ASSERT_EQ(obj->iargs->active_precon_variant, 1);

   ASSERT_EQ(
      HYPREDRV_PreconPresetRegister("api_contract_scalar_ilu", "ilu: bj-ilut\n", "unit"),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "api_contract_scalar_ilu"),
             ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon_method, PRECON_ILU);
   ASSERT_EQ(obj->iargs->precon.ilu.type, 1);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(obj->iargs->precon.amg.coarsening.num_functions, 2);

   ASSERT_EQ(HYPREDRV_PreconPresetRegister(
                "api_contract_label_mgr",
                "mgr:\n  level:\n    0:\n      f_dofs: [velocity]\n      f_relaxation:\n"
                "        gmres:\n          max_iter: 2\n  coarsest_level: amg\n",
                "unit"),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "api_contract_label_mgr"),
             ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon_method, PRECON_MGR);
   ASSERT_EQ(obj->iargs->precon.mgr.level[0].f_dofs.size, 1);
   ASSERT_EQ(obj->iargs->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_NOT_NULL(obj->iargs->precon.mgr.level[0].f_relaxation.krylov);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon_method, PRECON_ILU);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(obj->iargs->precon_method, PRECON_MGR);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_device_preset_rejection_preserves_live_solver(void)
{
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char  yaml[] = "general:\n  exec_policy: device\n  statistics: off\n"
                  "solver: pcg\npreconditioner: amg\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(set_matrix(obj), ERROR_NONE);
   ASSERT_EQ(set_rhs(obj), ERROR_NONE);
   ASSERT_EQ(set_guess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   hypredrv_Solver *solver = obj->solver;
   HYPRE_Precon precon = obj->precon;
   ASSERT_EQ(HYPREDRV_PreconPresetRegister(
                "api_contract_host_only_amg",
                "amg:\n  interpolation:\n    prolongation_type: standard\n", "unit"),
             ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_InputArgsSetPreconPreset(obj, "api_contract_host_only_amg") &
               ERROR_INVALID_PRECON);
   ASSERT_PTR_EQ(obj->solver, solver);
   ASSERT_PTR_EQ(obj->precon, precon);
   ASSERT_TRUE(obj->precon_is_setup);
   ASSERT_EQ(obj->iargs->general.exec_policy, 1);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   void *solution_object = NULL;
   ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x.handle, &solution_object), 0);
   hypre_Vector *solution =
      hypre_ParVectorLocalVector((hypre_ParVector *)solution_object);
   HYPRE_Complex value = 0;
   hypre_TMemcpy(&value, hypre_VectorData(solution), HYPRE_Complex, 1, HYPRE_MEMORY_HOST,
                 hypre_VectorMemoryLocation(solution));
   ASSERT_EQ_DOUBLE((double)hypre_creal(value), 1.4, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
#endif
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_independent_api_error_scopes);
   RUN_TEST(test_configured_replacement_failure_and_recovery);
   RUN_TEST(test_setup_creates_default_solution);
   RUN_TEST(test_preset_replacement_preserves_solver_lifetimes);
   RUN_TEST(test_rejected_presets_preserve_unconfigured_object);
   RUN_TEST(test_presets_configure_active_variants);
   RUN_TEST(test_device_preset_rejection_preserves_live_solver);
   RUN_TEST(test_config_replacement_validates_before_retiring_live_state);
   RUN_TEST(test_yaml_input_contract);
   RUN_TEST(test_optional_shared_tags);
   RUN_TEST(test_internal_call_status);
   RUN_TEST(test_internal_solver_workflows_preserve_errors);
   RUN_TEST(test_attachment_aliases);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
