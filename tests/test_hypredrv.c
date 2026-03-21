#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPREDRV.h"
#include "args.h"
#include "containers.h"
#include "error.h"
#include "linsys.h"
#include "stats.h"
#include "test_helpers.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

extern uint32_t HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t obj,
                                                         int        num_local_blocks,
                                                         int        num_dof_types);

struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   int      nstates;
   int     *states;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix mat_A;
   HYPRE_IJMatrix mat_M;
   HYPRE_IJVector vec_b;
   HYPRE_IJVector vec_x;
   HYPRE_IJVector vec_x0;
   HYPRE_IJVector vec_xref;
   HYPRE_IJVector vec_nn;
   HYPRE_IJVector *vec_s;
   bool           owns_mat_M;
   bool           owns_vec_x;
   bool           owns_vec_x0;
   bool           owns_vec_xref;

   HYPRE_Precon precon;
   HYPRE_Solver solver;

   void *scaling_ctx;
   IntArray *precon_reuse_timestep_starts;

   Stats *stats;
};

static void
reset_state(void)
{
   /* Ensure we start each test with clean global error/message and hypre error state.
    * Note: hypre error flags can be sticky across calls; some code paths inspect
    * HYPRE_GetError() and will misdiagnose unrelated operations otherwise. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   HYPREDRV_Finalize();
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
}

static bool
setup_ps3d10pt7_paths(char matrix_path[PATH_MAX], char rhs_path[PATH_MAX])
{
   snprintf(matrix_path, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.A", HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.b", HYPREDRIVE_SOURCE_DIR);

   if (access(matrix_path, F_OK) != 0 || access(rhs_path, F_OK) != 0)
   {
      fprintf(stderr, "SKIP: missing data files: %s or %s\n", matrix_path, rhs_path);
      return false;
   }

   return true;
}

static HYPREDRV_t
create_initialized_obj(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);
   return obj;
}

static void
parse_yaml_into_obj(HYPREDRV_t obj, char *yaml_config)
{
   char *argv[] = {yaml_config};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_NONE);
}

static void
parse_minimal_library_yaml(HYPREDRV_t obj)
{
   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
}

static HYPRE_IJMatrix
create_test_ijmatrix_1x1(double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows     = 1;
   HYPRE_Int    ncols[1]  = {1};
   HYPRE_BigInt rows[1]   = {0};
   HYPRE_BigInt cols[1]   = {0};
   double       values[1] = {diag};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static HYPRE_IJVector
create_test_ijvector_1x1(double value)
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

#define ASSERT_HAS_FLAG(code, flag) ASSERT_TRUE(((code) & (flag)) != 0)

static void
test_HYPREDRV_all_api_init_guard(void)
{
   reset_state();

   HYPREDRV_t obj = NULL;
   uint32_t   code;

   /* Core lifecycle */
   code = HYPREDRV_Create(MPI_COMM_SELF, &obj);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(obj);

   code = HYPREDRV_Destroy(&obj);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Non-object utilities */
   code = HYPREDRV_PrintLibInfo(MPI_COMM_SELF, 1);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PrintSystemInfo(MPI_COMM_SELF);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PrintExitInfo(MPI_COMM_SELF, "hypredrive");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Args parsing / global options */
   code = HYPREDRV_InputArgsParse(0, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_SetGlobalOptions(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Input getters (return ints) */
   (void)HYPREDRV_InputArgsGetWarmup(NULL);
   (void)HYPREDRV_InputArgsGetNumRepetitions(NULL);
   (void)HYPREDRV_InputArgsGetNumLinearSystems(NULL);

   /* Linear system APIs (object methods) */
   code = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemReadMatrix(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetMatrix(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetRHS(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetInitialGuess(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   HYPRE_Complex *ptr = NULL;
   code              = HYPREDRV_LinearSystemGetSolutionValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemGetRHSValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   HYPRE_Matrix mat_out = NULL;
   code = HYPREDRV_LinearSystemGetMatrix(NULL, &mat_out);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   double norm = 0.0;
   code        = HYPREDRV_LinearSystemGetSolutionNorm(NULL, "l2", &norm);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   code = HYPREDRV_LinearSystemSetPrecMatrix(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetDofmap(NULL, 0, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetInterleavedDofmap(NULL, 0, 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetContiguousDofmap(NULL, 0, 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemReadDofmap(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemPrintDofmap(NULL, "dofmap.txt");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Solver / precon APIs */
   code = HYPREDRV_PreconCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconApply(NULL, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Stats/annotation APIs */
   code = HYPREDRV_StatsPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateBegin("x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateEnd("x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
}

static void
test_HYPREDRV_all_api_obj_guard(void)
{
   reset_state();
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   uint32_t code;

   /* Object methods should trip the OBJ guard when hypredrv == NULL */
   code = HYPREDRV_SetLibraryMode(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_SetGlobalOptions(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   HYPRE_Complex *ptr = NULL;
   code              = HYPREDRV_LinearSystemGetSolutionValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSystemGetRHSValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   HYPRE_Matrix mat_null = NULL;
   code = HYPREDRV_LinearSystemGetMatrix(NULL, &mat_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   int typed_iter_null = -1;
   double typed_t_null = -1.0;
   code = HYPREDRV_LinearSolverGetNumIter(NULL, &typed_iter_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverGetSetupTime(NULL, &typed_t_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverGetSolveTime(NULL, &typed_t_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StateVectorApplyCorrection(NULL, 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   double norm = 0.0;
   code        = HYPREDRV_LinearSystemGetSolutionNorm(NULL, "l2", &norm);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemPrintDofmap(NULL, "x");
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_PreconCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_PreconSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_PreconDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_StatsPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   /* Reset the global error state before finalizing */
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_requires_initialization_guard(void)
{
   reset_state();

   HYPREDRV_t obj = NULL;
   uint32_t   code;

   code = HYPREDRV_Create(MPI_COMM_SELF, &obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(obj);
   hypredrv_ErrorCodeResetAll();

   code = HYPREDRV_SetLibraryMode(obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   hypredrv_ErrorCodeResetAll();
}

static void
test_initialize_and_finalize_idempotent(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE); /* second call should be no-op */
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE); /* already finalized */
}

static void
test_create_parse_and_destroy(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  matrix_filename: %s\n"
                            "  rhs_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 5\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n",
                            matrix_path, rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   HYPRE_Complex *sol_data = NULL;
   HYPRE_Complex *rhs_data = NULL;
   HYPRE_Complex *rhs_expected = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data), ERROR_NONE);
   ASSERT_NOT_NULL(sol_data);
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_NOT_NULL(rhs_data);
   hypredrv_LinearSystemGetRHSValues(state->vec_b, &rhs_expected);
   ASSERT_NOT_NULL(rhs_expected);
   ASSERT_PTR_EQ(rhs_data, rhs_expected);
   ASSERT_TRUE(rhs_data != sol_data);

   HYPRE_Matrix mat_retrieved = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetMatrix(obj, &mat_retrieved), ERROR_NONE);
   ASSERT_NOT_NULL(mat_retrieved);
   ASSERT_PTR_EQ((void *)mat_retrieved, (void *)state->mat_A);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetMatrix(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_Vector vec_sol = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol);
   ASSERT_PTR_EQ(vec_sol, (HYPRE_Vector)state->vec_x);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_Vector vec_rhs = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs), ERROR_NONE);
   ASSERT_NOT_NULL(vec_rhs);
   ASSERT_PTR_EQ(vec_rhs, (HYPRE_Vector)state->vec_b);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Ensure we have a dofmap to work with */
   if (!state->dofmap || state->dofmap->size == 0)
   {
      const int bootstrap_map[] = {0, 1, 2, 3};
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 4, bootstrap_map), ERROR_NONE);
   }

   ASSERT_NOT_NULL(state->dofmap);
   const size_t dofmap_size = state->dofmap->size;
   ASSERT_TRUE(dofmap_size > 0);

   int *manual_dofmap = malloc(dofmap_size * sizeof(int));
   ASSERT_NOT_NULL(manual_dofmap);
   for (size_t i = 0; i < dofmap_size; i++)
   {
      manual_dofmap[i] = (int)((i * 3) % 11);
   }

   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, (int)dofmap_size, manual_dofmap),
             ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, dofmap_size);
   ASSERT_EQ(state->dofmap->data[0], manual_dofmap[0]);
   if (dofmap_size > 5)
   {
      ASSERT_EQ(state->dofmap->data[5], manual_dofmap[5]);
   }
   free(manual_dofmap);

   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 3, 2), ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, (size_t)(3 * 2));
   ASSERT_EQ(state->dofmap->data[0], 0);
   ASSERT_EQ(state->dofmap->data[1], 1);
   ASSERT_EQ(state->dofmap->data[2], 0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 3, 2), ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, (size_t)(3 * 2));
   ASSERT_EQ(state->dofmap->data[0], 0);
   ASSERT_EQ(state->dofmap->data[1], 1);
   ASSERT_EQ(state->dofmap->data[2], 2);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);

   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);

   /* Destroy again to exercise unknown-object branch */
   uint32_t code = HYPREDRV_Destroy(&obj);
   ASSERT_TRUE(code & ERROR_UNKNOWN_HYPREDRV_OBJ);
   hypredrv_ErrorCodeResetAll();

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconCreate_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Test reuse logic: first call should create */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Second call with reuse=1 should reuse (not create again) */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   /* Precon should still exist */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverApply_with_xref(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* xref_filename isn't a supported key in this build; cover the xref branch by
    * directly setting the internal pointer to a valid vector. */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->vec_xref               = state->vec_b;

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

   /* Test apply with xref - should compute error norm */
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_stats_level_apis(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Drive Stats using internal timer keys ("prec"/"solve") */
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_BEGIN, "prec");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_END, "prec");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_END, "solve");

   /* HYPREDRV_GetLastStat branches (deprecated) */
   int    iter = -1;
   double t    = -1.0;
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "iter", &iter), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "setup", &t), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "solve", &t), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_GetLastStat(obj, "unknown", &t) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Typed stat getters */
   int    typed_iter   = -1;
   double setup_time   = -1.0;
   double solve_time   = -1.0;
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &typed_iter), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSetupTime(obj, &setup_time), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSolveTime(obj, &solve_time), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetNumIter(obj, NULL) & ERROR_UNKNOWN);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetSetupTime(obj, NULL) & ERROR_UNKNOWN);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetSolveTime(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_state_vectors_and_eigspec_error_paths(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Cover dofmap API entrypoints + PrintDofmap error branches */
   int dofmap_one[1] = {0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, dofmap_one), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 1, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, 2), ERROR_NONE);

   ASSERT_TRUE(HYPREDRV_LinearSystemPrintDofmap(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   /* Temporarily hide the dofmap to exercise the ERROR_MISSING_DOFMAP branch,
    * but restore it so the object can cleanly destroy owned state. */
   IntArray *saved_dofmap = state->dofmap;
   state->dofmap          = NULL;
   char *tmp_dof          = CREATE_TEMP_FILE("tmp_dofmap.txt");
   ASSERT_TRUE(HYPREDRV_LinearSystemPrintDofmap(obj, tmp_dof) & ERROR_MISSING_DOFMAP);
   hypredrv_ErrorCodeResetAll();
   free(tmp_dof);
   state->dofmap = saved_dofmap;

   uint32_t print_code = HYPREDRV_LinearSystemPrint(obj);
   ASSERT_TRUE(print_code == ERROR_NONE ||
               (print_code & (ERROR_MISSING_DOFMAP | ERROR_UNKNOWN)));
   hypredrv_ErrorCodeResetAll();

   /* Create two state vectors with the same range as vec_x */
   HYPRE_BigInt ilower = 0, iupper = 0;
   HYPRE_IJVectorGetLocalRange(state->vec_x, &ilower, &iupper);

   HYPRE_IJVector vecs[2] = {NULL, NULL};
   for (int i = 0; i < 2; i++)
   {
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, ilower, iupper, &vecs[i]), 0);
      ASSERT_NOT_NULL(vecs[i]);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vecs[i], HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize(vecs[i]), 0);
      ASSERT_EQ(HYPRE_IJVectorAssemble(vecs[i]), 0);
   }

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 2, vecs), ERROR_NONE);

   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_NOT_NULL(data);

   ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 0, 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_StateVectorApplyCorrection(obj, -1) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorApplyCorrection(obj, 99) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Force xref path in LinearSolverApply */
   state->vec_xref = state->vec_b;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   /* Hit HYPREDRV_PreconApply entrypoint */
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
             ERROR_NONE);

   /* Eigenspectrum API entrypoint (exercises CHECK_INIT/CHECK_OBJ) */
   uint32_t eig = HYPREDRV_LinearSystemComputeEigenspectrum(obj);
   ASSERT_TRUE(eig == ERROR_NONE || (eig & ERROR_UNKNOWN));
   hypredrv_ErrorCodeResetAll();

   /* Force error branches in GetValues/Copy by nulling an internal vec_s entry */
   HYPRE_IJVector saved_state0 = state->vec_s[state->states[0]];
   state->vec_s[state->states[0]] = NULL;
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorGetValues(obj, 0, &data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorCopy(obj, 0, 1) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_s[state->states[0]] = saved_state0;

   /* vecs[] are now owned by hypredrv (Destroy() will destroy hypredrv->vec_s[i]). */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_SetGlobalOptions_exec_policy(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "  use_vendor_spgemm: on\n"
            "  use_vendor_spmv: on\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);

   /* Test SetGlobalOptions with host policy */
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconCreate_reuse_logic_variations(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   /* Test reuse=0 (always create) */
   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 0\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* With reuse=0, should always create */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Second call should also create (reuse=0 means no reuse) */
   HYPREDRV_PreconDestroy(obj);
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverCreate_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 2\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* First call should create (ls_id=0, (0+1) % (2+1) = 1, not 0, but first system) */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->solver);

   /* Second call with reuse=2: ls_id=1, (1+1) % 3 = 2, should reuse (not create) */
   HYPREDRV_LinearSolverDestroy(obj);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   /* Solver should be NULL because we're reusing */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Create precon */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* With reuse=1, ls_id=0: (0+1) % 2 = 1, should NOT destroy (ls_id must be > 0) */
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   /* Precon should still exist because ls_id=0 */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverDestroy_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 2\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Create precon and solver */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->solver);

   /* Test ls_id = 0: should NOT destroy (ls_id must be > 0) */
   if (state->stats) {
      state->stats->ls_counter = 1; /* ls_id = ls_counter - 1 = 0 */
   }
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   /* Solver should still exist */

   /* Test ls_id = 3 with reuse=2: should destroy ((3+1)%(2+1)=0) */
   if (state->stats) {
      state->stats->ls_counter = 4; /* ls_id = ls_counter - 1 = 3 */
   }
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   /* Solver should be destroyed */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_linear_system_ids(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    linear_system_ids: [0, 2]\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 1, not in [0,2], so object should be kept. */
   state->stats->ls_counter = 1;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 2, in [0,2], so object should be dropped now. */
   state->stats->ls_counter = 2;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_per_timestep(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   char *tmp_ts = CREATE_TEMP_FILE("tmp_timesteps_reuse.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "2\n");
   fprintf(tf, "0 0\n");
   fprintf(tf, "1 3\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 640];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    per_timestep: on\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path, tmp_ts);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 1, not a timestep start, so keep object. */
   state->stats->ls_counter = 1;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 3, timestep start, so destroy for recompute. */
   state->stats->ls_counter = 3;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_PreconDestroy_reuse_per_timestep_frequency(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   char *tmp_ts = CREATE_TEMP_FILE("tmp_timesteps_reuse_freq.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "4\n");
   fprintf(tf, "0 0\n");
   fprintf(tf, "1 3\n");
   fprintf(tf, "2 6\n");
   fprintf(tf, "3 9\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 700];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    per_timestep: on\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path, tmp_ts);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 3 is timestep index 1, so with frequency=1 keep object. */
   state->stats->ls_counter = 3;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 6 is timestep index 2, so with frequency=1 destroy object. */
   state->stats->ls_counter = 6;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_LinearSolverApply_error_cases(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);

   /* Test with uninitialized solver */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->solver = NULL;
   result = HYPREDRV_LinearSolverApply(obj);
   ASSERT_TRUE(result & ERROR_INVALID_SOLVER);

   /* Clear sticky error state before cleanup assertions */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_Annotate_functions(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test annotate functions - these should not error even with invalid obj */
   HYPREDRV_AnnotateBegin("test", 1);
   HYPREDRV_AnnotateEnd("test", 1);

   /* Test with NULL object */
   HYPREDRV_AnnotateBegin(NULL, 1);
   HYPREDRV_AnnotateEnd(NULL, 1);

   /* These wrappers intentionally tolerate unknown annotation keys, but they can set
    * sticky error flags; clear before cleanup assertions. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PrintLibInfo_PrintExitInfo(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test print functions */
   HYPREDRV_PrintLibInfo(MPI_COMM_SELF, 1);
   HYPREDRV_PrintExitInfo(MPI_COMM_SELF, "test_prog");

   /* Avoid MPI_COMM_NULL here: some MPI implementations abort on MPI_Comm_rank(NULL). */
   HYPREDRV_PrintExitInfo(MPI_COMM_SELF, NULL);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSystemResetInitialGuess_error_cases(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   /* Test with NULL x vector */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->vec_x = NULL;
   result = HYPREDRV_LinearSystemResetInitialGuess(obj);
   ASSERT_TRUE(result & ERROR_UNKNOWN);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSystemBuild_error_cases(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_misc_0hit_branches(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   snprintf(matrix_path, sizeof(matrix_path), "%s/data/ps3d10pt7/np1/IJ.out.A",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, sizeof(rhs_path), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);
   TEST_REQUIRE_FILE(matrix_path);
   TEST_REQUIRE_FILE(rhs_path);

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 1\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Exercise solution/RHS getters and a successful solution norm path */
   HYPRE_Complex *sol_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data), ERROR_NONE);
   ASSERT_NOT_NULL(sol_data);

   HYPRE_Complex *rhs_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_NOT_NULL(rhs_data);
   HYPRE_Complex *rhs_expected = NULL;
   hypredrv_LinearSystemGetRHSValues(state->vec_b, &rhs_expected);
   ASSERT_NOT_NULL(rhs_expected);
   ASSERT_PTR_EQ(rhs_data, rhs_expected);
   ASSERT_TRUE(rhs_data != sol_data);

   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionValues(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHSValues(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_IJVector saved_b = state->vec_b;
   state->vec_b           = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = saved_b;

   HYPRE_IJVector saved_x = state->vec_x;
   state->vec_x           = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_x = saved_x;

   /* GetSolution / GetRHS: happy path */
   HYPRE_Vector vec_sol_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol_out);
   ASSERT_PTR_EQ(vec_sol_out, (HYPRE_Vector)state->vec_x);

   HYPRE_Vector vec_rhs_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_rhs_out);
   ASSERT_PTR_EQ(vec_rhs_out, (HYPRE_Vector)state->vec_b);

   /* GetSolution / GetRHS: NULL output arg */
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* GetSolution when vec_x is NULL */
   HYPRE_IJVector restore_x = state->vec_x;
   state->vec_x             = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_x = restore_x;

   /* GetRHS when vec_b is NULL */
   HYPRE_IJVector restore_b = state->vec_b;
   state->vec_b             = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs_out) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = restore_b;

   /* SetSolution: NULL recreates from vec_b */
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   vec_sol_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol_out);

   /* SetSolution(NULL) with no vec_b → error */
   HYPRE_IJVector restore_b2 = state->vec_b;
   state->vec_b              = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = restore_b2;

   double ok_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "L2", &ok_norm), ERROR_NONE);

   /* Delegate printing to linsys */
   ASSERT_EQ(HYPREDRV_LinearSystemPrint(obj), ERROR_NONE);

   /* Cover dofmap family + ReadDofmap no-file default path */
   int dm[2] = {0, 1};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, dm), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 2, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 2, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);

   char *tmp_dof = CREATE_TEMP_FILE("tmp_api_dofmap.txt");
   ASSERT_EQ(HYPREDRV_LinearSystemPrintDofmap(obj, tmp_dof), ERROR_NONE);
   free(tmp_dof);

   /* Cover GetSolutionNorm error branches */
   double norm = 0.0;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionNorm(obj, NULL, &norm) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionNorm(obj, "L2", NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Cover AnnotateLevelBegin/End paths */
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(0, "lvl", 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(0, "lvl", 1), ERROR_NONE);

   /* Cover GetLastStat else-if branches + error branch (deprecated API) */
   int    it = -1;
   double t  = -1.0;
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "iter", &it), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "setup", &t), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_GetLastStat(obj, "solve", &t), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_GetLastStat(obj, "unknown", &t) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Typed stat getters - same data, type-safe */
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &it), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSetupTime(obj, &t), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSolveTime(obj, &t), ERROR_NONE);
   hypredrv_ErrorCodeResetAll();

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_variants(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  warmup: false\n"
                            "  num_repetitions: 1\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  type: ij\n"
                            "  rhs_filename: %s\n"
                            "  matrix_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    relative_tol: 1.0e-9\n"
                            "    max_iter: 500\n"
                            "    print_level: 3\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: HMIS\n"
                            "        strong_th: 0.25\n"
                            "      interpolation:\n"
                            "        prolongation_type: \"MM-ext+i\"\n"
                            "      relaxation:\n"
                            "        down_type: 16\n"
                            "        down_sweeps: 1\n"
                            "        up_type: 16\n"
                            "        up_sweeps: 1\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: PMIS\n"
                            "        strong_th: 0.5\n"
                            "      interpolation:\n"
                            "        prolongation_type: standard\n"
                            "      relaxation:\n"
                            "        down_type: 8\n"
                            "        down_sweeps: 2\n"
                            "        up_type: 8\n"
                            "        up_sweeps: 2\n",
                            rhs_path, matrix_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);

   /* Check that variants were parsed */
   int num_variants = HYPREDRV_InputArgsGetNumPreconVariants(obj);
   ASSERT_EQ(num_variants, 2);

   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* Test setting and using each variant */
   for (int v = 0; v < num_variants; v++)
   {
      ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, v), ERROR_NONE);

      /* Reset initial guess */
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

      /* Create and setup */
      ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

      /* Solve */
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

      /* Destroy */
      ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   }

   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_preset_yaml(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  matrix_filename: %s\n"
                            "  rhs_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 5\n"
                            "preconditioner:\n"
                            "  preset: poisson\n",
                            matrix_path, rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_preset_invalid(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);

   hypredrv_ErrorCodeResetAll();
   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetPreconPreset(obj, "not-a-preset"),
                   ERROR_INVALID_VAL);
   ASSERT_HAS_FLAG(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);

   /* Clear error state so Destroy() isn't masked by the expected preset failure. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_explicit_nonlib_take_ownership(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 == vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(state->mat_M == mat_M);
   ASSERT_TRUE(state->owns_vec_x0);
   ASSERT_TRUE(state->owns_vec_xref);
   ASSERT_TRUE(state->owns_mat_M);
   ASSERT_NOT_NULL(state->vec_x);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_explicit_library_mode_borrow(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 == vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(state->mat_M == mat_M);
   ASSERT_TRUE(!state->owns_vec_x0);
   ASSERT_TRUE(!state->owns_vec_xref);
   ASSERT_TRUE(!state->owns_mat_M);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_ref), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_x0), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_M), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_null_preserve_default_behavior(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 != vec_x0);
   ASSERT_TRUE(state->owns_vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(!state->owns_vec_xref);
   ASSERT_TRUE(state->mat_M == state->mat_A);
   ASSERT_TRUE(!state->owns_mat_M);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_ref), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_x0), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_M), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
run_hypredrv_lifecycle_and_guards(void)
{
   RUN_TEST(test_HYPREDRV_all_api_init_guard);
   RUN_TEST(test_HYPREDRV_all_api_obj_guard);
   RUN_TEST(test_requires_initialization_guard);
   RUN_TEST(test_initialize_and_finalize_idempotent);
}

static void
run_hypredrv_solver_and_reuse(void)
{
   RUN_TEST(test_create_parse_and_destroy);
   RUN_TEST(test_HYPREDRV_PreconCreate_reuse_logic);
   RUN_TEST(test_HYPREDRV_LinearSolverApply_with_xref);
   RUN_TEST(test_HYPREDRV_stats_level_apis);
   RUN_TEST(test_HYPREDRV_state_vectors_and_eigspec_error_paths);
   RUN_TEST(test_HYPREDRV_SetGlobalOptions_exec_policy);
   RUN_TEST(test_HYPREDRV_PreconCreate_reuse_logic_variations);
   RUN_TEST(test_HYPREDRV_LinearSolverCreate_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_LinearSolverDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_linear_system_ids);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep_frequency);
   RUN_TEST(test_HYPREDRV_LinearSolverApply_error_cases);
}

static void
run_hypredrv_misc_and_preconditioners(void)
{
   RUN_TEST(test_HYPREDRV_Annotate_functions);
   RUN_TEST(test_HYPREDRV_PrintLibInfo_PrintExitInfo);
   RUN_TEST(test_HYPREDRV_LinearSystemResetInitialGuess_error_cases);
   RUN_TEST(test_HYPREDRV_LinearSystemBuild_error_cases);
   RUN_TEST(test_HYPREDRV_misc_0hit_branches);
   RUN_TEST(test_HYPREDRV_preconditioner_variants);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_yaml);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_invalid);
   RUN_TEST(test_HYPREDRV_linear_system_setters_explicit_nonlib_take_ownership);
   RUN_TEST(test_HYPREDRV_linear_system_setters_explicit_library_mode_borrow);
   RUN_TEST(test_HYPREDRV_linear_system_setters_null_preserve_default_behavior);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   run_hypredrv_lifecycle_and_guards();
   run_hypredrv_solver_and_reuse();
   run_hypredrv_misc_and_preconditioners();

   MPI_Finalize();
   return 0;
}
