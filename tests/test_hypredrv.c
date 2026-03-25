#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPRE_utilities.h"
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
   bool         precon_is_setup;

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

static bool
setup_poromech2k_dir(char ls_dir[PATH_MAX])
{
   char matrix0[2 * PATH_MAX];
   char rhs0[2 * PATH_MAX];
   char dofmap0[2 * PATH_MAX];
   char matrix1[2 * PATH_MAX];
   char rhs1[2 * PATH_MAX];
   char dofmap1[2 * PATH_MAX];

   snprintf(ls_dir, PATH_MAX, "%s/data/poromech2k/np1/ls", HYPREDRIVE_SOURCE_DIR);
   snprintf(matrix0, sizeof(matrix0), "%s_00000/IJ.out.A.00000.bin", ls_dir);
   snprintf(rhs0, sizeof(rhs0), "%s_00000/IJ.out.b.00000.bin", ls_dir);
   snprintf(dofmap0, sizeof(dofmap0), "%s_00000/dofmap.out.00000", ls_dir);
   snprintf(matrix1, sizeof(matrix1), "%s_00001/IJ.out.A.00000.bin", ls_dir);
   snprintf(rhs1, sizeof(rhs1), "%s_00001/IJ.out.b.00000.bin", ls_dir);
   snprintf(dofmap1, sizeof(dofmap1), "%s_00001/dofmap.out.00000", ls_dir);

   if (access(matrix0, F_OK) != 0 || access(rhs0, F_OK) != 0 || access(dofmap0, F_OK) != 0 ||
       access(matrix1, F_OK) != 0 || access(rhs1, F_OK) != 0 || access(dofmap1, F_OK) != 0)
   {
      fprintf(stderr,
              "SKIP: missing poromech2k timestep files under %s_00000 or %s_00001\n",
              ls_dir, ls_dir);
      return false;
   }

   return true;
}

static HYPREDRV_t
create_initialized_obj(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
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
parse_yaml_file_into_obj(HYPREDRV_t obj, const char *yaml_config, const char *tmp_name)
{
   char *tmp_yaml = CREATE_TEMP_FILE(tmp_name);
   ASSERT_NOT_NULL(tmp_yaml);

   FILE *fp = fopen(tmp_yaml, "w");
   ASSERT_NOT_NULL(fp);
   ASSERT_TRUE(fputs(yaml_config, fp) >= 0);
   fclose(fp);

   char *argv[] = {tmp_yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_NONE);
   free(tmp_yaml);
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
}

static void
parse_library_reuse_yaml(HYPREDRV_t obj, const char *reuse_block)
{
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[1024];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "%s"
            "  amg:\n"
            "    print_level: 0\n",
            reuse_block ? reuse_block : "");
   parse_yaml_into_obj(obj, yaml_config);
}

static void
attach_library_scalar_system(HYPREDRV_t obj, HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b)
{
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
}

static void
run_library_linear_solve(HYPREDRV_t obj, const char *newton_name)
{
   if (newton_name)
   {
      ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 1, newton_name, -1), ERROR_NONE);
   }

   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   if (newton_name)
   {
      ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 1, newton_name, -1), ERROR_NONE);
   }
}

static size_t
count_substr(const char *haystack, const char *needle)
{
   size_t      count = 0;
   const char *ptr   = haystack;
   size_t      len   = strlen(needle);

   while ((ptr = strstr(ptr, needle)) != NULL)
   {
      count++;
      ptr += len;
   }

   return count;
}

static void
capture_eigspec_warning_output(HYPREDRV_t obj, int num_calls, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();

#ifdef __APPLE__
   if (!tmp)
   {
      char path[] = "/tmp/hypredrv_test_eigspec.txt";
      int  fd     = mkstemp(path);
      ASSERT_TRUE(fd != -1);
      tmp = fdopen(fd, "w+");
      unlink(path);
   }
#endif

   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   for (int i = 0; i < num_calls; i++)
   {
      ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
   }
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

typedef void (*CapturedStdoutFn)(void *);

static void
capture_stdout_output(CapturedStdoutFn fn, void *context, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();

#ifdef __APPLE__
   if (!tmp)
   {
      char path[] = "/tmp/hypredrv_test_stdout.txt";
      int  fd     = mkstemp(path);
      ASSERT_TRUE(fd != -1);
      tmp = fdopen(fd, "w+");
      unlink(path);
   }
#endif

   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_out = dup(fileno(stdout));
   ASSERT_TRUE(saved_out != -1);

   fflush(stdout);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stdout)) != -1);

   fn(context);
   fflush(stdout);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_out, fileno(stdout)) != -1);
   close(saved_out);
   fclose(tmp);
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

   /* Args parsing */
   code = HYPREDRV_InputArgsParse(0, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Input getters */
   code = HYPREDRV_InputArgsGetWarmup(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_ObjectSetName(NULL, "guard");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_InputArgsGetNumRepetitions(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_InputArgsGetNumLinearSystems(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

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
   code = HYPREDRV_AnnotateBegin(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateEnd(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateLevelBegin(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateLevelEnd(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelGetCount(NULL, 0, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelGetEntry(NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelPrint(NULL, 0);
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
   code = HYPREDRV_AnnotateBegin(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateEnd(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateLevelBegin(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateLevelEnd(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelGetCount(NULL, 0, &typed_iter_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelGetEntry(NULL, 0, 0, &typed_iter_null, NULL, NULL, NULL,
                                        NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelPrint(NULL, 0);
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

   /* Eigenspectrum API entrypoint: always succeeds (no-op when built without eigspec) */
   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);

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
test_HYPREDRV_InputArgsParse_exec_policy(void)
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

#if HYPRE_CHECK_MIN_VERSION(22100, 0)
   HYPRE_MemoryLocation   memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy  exec_policy     = HYPRE_EXEC_HOST;

   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

#if HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_GetMemoryLocation(&memory_location), 0);
   ASSERT_EQ(HYPRE_GetExecutionPolicy(&exec_policy), 0);
   ASSERT_EQ(memory_location, HYPRE_MEMORY_HOST);
   ASSERT_EQ(exec_policy, HYPRE_EXEC_HOST);
#endif

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSystemComputeEigenspectrum_warns_once_when_disabled(void)
{
#ifdef HYPREDRV_ENABLE_EIGSPEC
   return;
#else
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   char       buffer[1024];

   capture_eigspec_warning_output(obj, 2, buffer, sizeof(buffer));

   ASSERT_EQ(count_substr(buffer, "eigenspectrum support is disabled"), 1);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
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
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 4 is inside timestep 1, so keep reusing the current timestep preconditioner. */
   state->stats->ls_counter = 4;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 5 is still inside timestep 1, so keep reusing. */
   state->stats->ls_counter = 5;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 6 is the first system in timestep 2, so rebuild for the new timestep. */
   state->stats->ls_counter = 6;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_library_mode_reuse_per_timestep_frequency_with_object_annotations(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_library_reuse_yaml(obj,
                            "  reuse:\n"
                            "    enabled: yes\n"
                            "    per_timestep: on\n"
                            "    frequency: 1\n");

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-1");
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
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

   /* Test annotate functions */
   HYPREDRV_AnnotateBegin(obj, "test", 1);
   HYPREDRV_AnnotateEnd(obj, "test", 1);

   /* Test with NULL name */
   HYPREDRV_AnnotateBegin(obj, NULL, 1);
   HYPREDRV_AnnotateEnd(obj, NULL, 1);

   /* These wrappers intentionally tolerate unknown annotation keys, but they can set
    * sticky error flags; clear before cleanup assertions. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_object_scoped_annotation_isolation(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj1 = NULL;
   HYPREDRV_t obj2 = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj2), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj2, 0, "obj2-timestep", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj2, 0, "obj2-timestep", -1), ERROR_NONE);

   int count1 = -1;
   int count2 = -1;
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj1, 0, &count1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj2, 0, &count2), ERROR_NONE);
   ASSERT_EQ(count1, 0);
   ASSERT_EQ(count2, 1);

   int    entry_id     = -1;
   int    num_solves   = -1;
   int    linear_iters = -1;
   double setup_time   = -1.0;
   double solve_time   = -1.0;
   ASSERT_EQ(HYPREDRV_StatsLevelGetEntry(obj2, 0, 0, &entry_id, &num_solves,
                                           &linear_iters, &setup_time, &solve_time),
             ERROR_NONE);
   ASSERT_EQ(entry_id, 1);
   ASSERT_EQ(num_solves, 0);
   ASSERT_EQ(linear_iters, 0);
   ASSERT_EQ_DOUBLE(setup_time, 0.0, 1e-12);
   ASSERT_EQ_DOUBLE(solve_time, 0.0, 1e-12);

   /* Annotating obj1 should not affect obj2 */
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj1, 0, "obj1-timestep", 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj1, 0, "obj1-timestep", 0), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj1, 0, &count1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj2, 0, &count2), ERROR_NONE);
   ASSERT_EQ(count1, 1);
   ASSERT_EQ(count2, 1);

   ASSERT_EQ(HYPREDRV_StatsLevelPrint(obj2, 0), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_reuse_per_timestep_with_object_annotations(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_library_reuse_yaml(obj,
                            "  reuse:\n"
                            "    enabled: yes\n"
                            "    per_timestep: on\n"
                            "    frequency: 0\n");

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-1");
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   int timestep_count = -1;
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj, 0, &timestep_count), ERROR_NONE);
   ASSERT_EQ(timestep_count, 2);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_mgr_recreates_precon_on_new_timestep(void)
{
   reset_state();

   /* MGR with AMG f-relaxation + filter_functions crashes in hypre < 2.21.0 */
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 22100
   printf("SKIP: MGR AMG f-relaxation requires hypre >= 2.21.0\n");
   return;
#endif

   char ls_dir[PATH_MAX];
   if (!setup_poromech2k_dir(ls_dir))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[8192];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  dirname: %s\n"
            "  init_suffix: 0\n"
            "  last_suffix: 24\n"
            "  rhs_filename: IJ.out.b\n"
            "  matrix_filename: IJ.out.A\n"
            "  dofmap_filename: dofmap.out\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  fgmres:\n"
            "    max_iter: 100\n"
            "    krylov_dim: 30\n"
            "    print_level: 0\n"
            "    relative_tol: 1.0e-6\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    enabled: yes\n"
            "    per_timestep: on\n"
            "    frequency: 0\n"
            "  mgr:\n"
            "    max_iter: 1\n"
            "    tolerance: 0.0\n"
            "    print_level: 0\n"
            "    coarse_th: 1e-20\n"
            "    level:\n"
            "      0:\n"
            "        f_dofs: [0, 1, 2]\n"
            "        f_relaxation: amg\n"
            "          amg:\n"
            "            coarsening:\n"
            "              type: pmis\n"
            "              strong_th: 0.5\n"
            "              num_functions: 3\n"
            "              filter_functions: on\n"
            "        g_relaxation: none\n"
            "        restriction_type: injection\n"
            "        prolongation_type: blk-jacobi\n"
            "        coarse_level_type: non-galerkin\n"
            "      1:\n"
            "        f_dofs: [5]\n"
            "        f_relaxation: jacobi\n"
            "        g_relaxation: none\n"
            "        restriction_type: injection\n"
            "        prolongation_type: jacobi\n"
            "        coarse_level_type: rap\n"
            "      2:\n"
            "        f_dofs: [4]\n"
            "        f_relaxation: single\n"
            "        g_relaxation: ilu\n"
            "        restriction_type: columped\n"
            "        prolongation_type: injection\n"
            "        coarse_level_type: rap\n"
            "    coarsest_level:\n"
            "      amg:\n"
            "        max_iter: 1\n"
            "        tolerance: 0.0\n"
            "        relaxation:\n"
            "          down_type: l1-jacobi\n"
            "          up_type: l1-jacobi\n",
            ls_dir);
   parse_yaml_into_obj(obj, yaml_config);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   run_library_linear_solve(obj, NULL);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_TRUE(state->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* Regression guard: the first solve in a new timestep must mark the reused MGR
    * preconditioner dirty again before setup. */
   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_FALSE(state->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   /* LinearSystemBuild created mat_A/vec_b from files, but lib_mode
    * prevents Destroy from freeing them.  Clean up explicitly. */
   HYPRE_IJMatrixDestroy(state->mat_A);
   state->mat_A = NULL;
   HYPRE_IJVectorDestroy(state->vec_b);
   state->vec_b = NULL;

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

struct DestroyObjectContext
{
   HYPREDRV_t *obj_ptr;
};

static void
destroy_object_for_capture(void *context)
{
   struct DestroyObjectContext *destroy_context = (struct DestroyObjectContext *)context;
   ASSERT_EQ(HYPREDRV_Destroy(destroy_context->obj_ptr), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_destroy_prints_named_statistics_summary(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
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
   ASSERT_EQ(HYPREDRV_ObjectSetName(obj, "named-handle"), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct DestroyObjectContext destroy_context = {&obj};
   capture_stdout_output(destroy_object_for_capture, &destroy_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "STATISTICS SUMMARY for named-handle:") != NULL);
   ASSERT_TRUE(strstr(output, "|      0 |") != NULL);
   ASSERT_NULL(obj);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
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
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "lvl", 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "lvl", 1), ERROR_NONE);

   /* Typed stat getters */
   int    it = -1;
   double t  = -1.0;
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &it), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSetupTime(obj, &t), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSolveTime(obj, &t), ERROR_NONE);
   hypredrv_ErrorCodeResetAll();

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 1, "late", 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 1, "late", 0), ERROR_NONE);
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
   int num_variants = 0;
   ASSERT_EQ(HYPREDRV_InputArgsGetNumPreconVariants(obj, &num_variants), ERROR_NONE);
   ASSERT_EQ(num_variants, 2);

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
test_HYPREDRV_InputArgsParse_solver_yaml_string(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   char yaml_config[] =
      "solver:\n"
      "  gmres:\n"
      "    max_iter: 25\n"
      "    relative_tol: 1.0e-9\n"
      "    krylov_dim: 17\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_NOT_NULL(state->iargs);
   ASSERT_EQ(state->iargs->solver_method, SOLVER_GMRES);
   ASSERT_EQ(state->iargs->solver.gmres.max_iter, 25);
   ASSERT_EQ(state->iargs->solver.gmres.relative_tol, 1.0e-9);
   ASSERT_EQ(state->iargs->solver.gmres.krylov_dim, 17);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_InputArgsParse_solver_yaml_file(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   char initial_yaml[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 12\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, initial_yaml);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_NOT_NULL(state->iargs);
   ASSERT_EQ(state->iargs->solver_method, SOLVER_PCG);
   ASSERT_EQ(state->iargs->solver.pcg.max_iter, 12);

   char const yaml_file_config[] =
      "solver:\n"
      "  gmres:\n"
      "    max_iter: 25\n"
      "    relative_tol: 1.0e-9\n"
      "    krylov_dim: 17\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_file_into_obj(obj, yaml_file_config, "tmp_parse_solver_file.yml");

   ASSERT_EQ(state->iargs->solver_method, SOLVER_GMRES);
   ASSERT_EQ(state->iargs->solver.gmres.max_iter, 25);
   ASSERT_EQ(state->iargs->solver.gmres.relative_tol, 1.0e-9);
   ASSERT_EQ(state->iargs->solver.gmres.krylov_dim, 17);

   hypredrv_ErrorCodeResetAll();
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
   RUN_TEST(test_HYPREDRV_InputArgsParse_exec_policy);
   RUN_TEST(test_HYPREDRV_LinearSystemComputeEigenspectrum_warns_once_when_disabled);
   RUN_TEST(test_HYPREDRV_state_vectors_and_eigspec_error_paths);
   RUN_TEST(test_HYPREDRV_PreconCreate_reuse_logic_variations);
   RUN_TEST(test_HYPREDRV_LinearSolverCreate_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_LinearSolverDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_linear_system_ids);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep_frequency);
   RUN_TEST(test_HYPREDRV_library_mode_reuse_per_timestep_with_object_annotations);
   RUN_TEST(
      test_HYPREDRV_library_mode_reuse_per_timestep_frequency_with_object_annotations);
   RUN_TEST(test_HYPREDRV_library_mode_mgr_recreates_precon_on_new_timestep);
   RUN_TEST(test_HYPREDRV_library_mode_destroy_prints_named_statistics_summary);
   RUN_TEST(test_HYPREDRV_LinearSolverApply_error_cases);
}

static void
run_hypredrv_misc_and_preconditioners(void)
{
   RUN_TEST(test_HYPREDRV_Annotate_functions);
   RUN_TEST(test_HYPREDRV_object_scoped_annotation_isolation);
   RUN_TEST(test_HYPREDRV_PrintLibInfo_PrintExitInfo);
   RUN_TEST(test_HYPREDRV_LinearSystemResetInitialGuess_error_cases);
   RUN_TEST(test_HYPREDRV_LinearSystemBuild_error_cases);
   RUN_TEST(test_HYPREDRV_misc_0hit_branches);
   RUN_TEST(test_HYPREDRV_preconditioner_variants);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_yaml);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_invalid);
   RUN_TEST(test_HYPREDRV_InputArgsParse_solver_yaml_string);
   RUN_TEST(test_HYPREDRV_InputArgsParse_solver_yaml_file);
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
