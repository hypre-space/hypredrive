/******************************************************************************
 * Tests for the public runtime lifecycle and process environment.
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "test_helpers.h"

#define ASSERT_HAS_FLAG(code, flag) ASSERT_TRUE(((code) & (flag)) != 0)

static void
reset_lib_and_runtime(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   (void)HYPREDRV_Finalize();
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
}

static void
test_runtime_lifecycle_and_invalid_handles(void)
{
   reset_lib_and_runtime();

   HYPREDRV_t objects[3] = {NULL, NULL, NULL};
   ASSERT_HAS_FLAG(HYPREDRV_Create(MPI_COMM_SELF, &objects[0]),
                   ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(objects[0]);
   ASSERT_EQ_U32(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_HAS_FLAG(HYPREDRV_Create(MPI_COMM_SELF, NULL), ERROR_INVALID_VAL);
   ASSERT_HAS_FLAG(HYPREDRV_Destroy(NULL), ERROR_INVALID_VAL);
   ASSERT_HAS_FLAG(HYPREDRV_SetLibraryMode(NULL), ERROR_UNKNOWN_HYPREDRV_OBJ);

   HYPREDRV_t invalid = (HYPREDRV_t)(uintptr_t)1;
   ASSERT_HAS_FLAG(HYPREDRV_SetLibraryMode(invalid), ERROR_UNKNOWN_HYPREDRV_OBJ);
   ASSERT_HAS_FLAG(HYPREDRV_Destroy(&invalid), ERROR_UNKNOWN_HYPREDRV_OBJ);
   ASSERT_PTR_EQ(invalid, (HYPREDRV_t)(uintptr_t)1);

   for (int i = 0; i < 3; i++)
   {
      ASSERT_EQ_U32(HYPREDRV_Create(MPI_COMM_SELF, &objects[i]), ERROR_NONE);
      ASSERT_NOT_NULL(objects[i]);
   }

   /* Removing an interior object must leave its neighbors live. Reinitializing
    * an active runtime must keep those objects registered. */
   uintptr_t removed = (uintptr_t)objects[1];
   ASSERT_EQ_U32(HYPREDRV_Destroy(&objects[1]), ERROR_NONE);
   ASSERT_NULL(objects[1]);
   ASSERT_HAS_FLAG(HYPREDRV_SetLibraryMode((HYPREDRV_t)removed),
                   ERROR_UNKNOWN_HYPREDRV_OBJ);
   ASSERT_EQ_U32(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_SetLibraryMode(objects[0]), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_SetLibraryMode(objects[2]), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Destroy(&objects[0]), ERROR_NONE);
   ASSERT_NULL(objects[0]);

   /* Finalization owns the remaining object; its old handle stays invalid
    * after a fresh initialization, before any new allocation can reuse it. */
   uintptr_t finalized = (uintptr_t)objects[2];
   ASSERT_EQ_U32(HYPREDRV_Finalize(), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Initialize(), ERROR_NONE);
   invalid = (HYPREDRV_t)finalized;
   ASSERT_HAS_FLAG(HYPREDRV_SetLibraryMode(invalid), ERROR_UNKNOWN_HYPREDRV_OBJ);
   ASSERT_HAS_FLAG(HYPREDRV_Destroy(&invalid), ERROR_UNKNOWN_HYPREDRV_OBJ);
   ASSERT_EQ_U32(HYPREDRV_Create(MPI_COMM_SELF, &objects[0]), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Destroy(&objects[0]), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Finalize(), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Finalize(), ERROR_NONE);
}

#if HYPRE_CHECK_MIN_VERSION(23100, 0)
static const char *const gpu_aware_mpi_env_names[] = {
   "MV2_USE_CUDA",
   "MV2_USE_HIP",
   "MPIR_CVAR_ENABLE_GPU",
   "MPICH_GPU_SUPPORT_ENABLED",
};

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

   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

static void
clear_gpu_aware_mpi_environment(void)
{
   for (size_t i = 0;
        i < sizeof(gpu_aware_mpi_env_names) / sizeof(gpu_aware_mpi_env_names[0]); i++)
   {
      unsetenv(gpu_aware_mpi_env_names[i]);
   }
}

static void
run_initialize_finalize(void *context)
{
   (void)context;
   ASSERT_EQ_U32(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_RuntimeGpuAwareMPI_defaults_to_disabled(void)
{
   char output[4096];

   reset_lib_and_runtime();
   clear_gpu_aware_mpi_environment();
   setenv("HYPREDRV_LOG_LEVEL", "1", 1);

   capture_stderr_output(run_initialize_finalize, NULL, output, sizeof(output));

   ASSERT_NOT_NULL(strstr(
      output, "GPU-aware MPI disabled (no recognized environment variable is set to 1)"));
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_RuntimeGpuAwareMPI_recognizes_enable_environment(void)
{
   char output[4096];
   char expected[128];

   for (size_t i = 0;
        i < sizeof(gpu_aware_mpi_env_names) / sizeof(gpu_aware_mpi_env_names[0]); i++)
   {
      reset_lib_and_runtime();
      clear_gpu_aware_mpi_environment();
      setenv("HYPREDRV_LOG_LEVEL", "1", 1);
      setenv(gpu_aware_mpi_env_names[i], "1", 1);

      capture_stderr_output(run_initialize_finalize, NULL, output, sizeof(output));

      snprintf(expected, sizeof(expected), "GPU-aware MPI enabled by %s=1",
               gpu_aware_mpi_env_names[i]);
      ASSERT_NOT_NULL(strstr(output, expected));
   }

   clear_gpu_aware_mpi_environment();
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_RuntimeGpuAwareMPI_requires_exact_value_one(void)
{
   static const char *const disabled_values[] = {"", "0", "01", "true"};
   char                     output[4096];

   for (size_t i = 0; i < sizeof(disabled_values) / sizeof(disabled_values[0]); i++)
   {
      reset_lib_and_runtime();
      clear_gpu_aware_mpi_environment();
      setenv("HYPREDRV_LOG_LEVEL", "1", 1);
      setenv("MPICH_GPU_SUPPORT_ENABLED", disabled_values[i], 1);

      capture_stderr_output(run_initialize_finalize, NULL, output, sizeof(output));

      ASSERT_NOT_NULL(strstr(
         output,
         "GPU-aware MPI disabled (no recognized environment variable is set to 1)"));
   }

   clear_gpu_aware_mpi_environment();
   unsetenv("HYPREDRV_LOG_LEVEL");
}
#endif

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();
   reset_lib_and_runtime();

   RUN_TEST(test_runtime_lifecycle_and_invalid_handles);
#if HYPRE_CHECK_MIN_VERSION(23100, 0)
   RUN_TEST(test_RuntimeGpuAwareMPI_defaults_to_disabled);
   RUN_TEST(test_RuntimeGpuAwareMPI_recognizes_enable_environment);
   RUN_TEST(test_RuntimeGpuAwareMPI_requires_exact_value_one);
#endif

   MPI_Finalize();
   return 0;
}
