/******************************************************************************
 * Unit tests for src/runtime.c (runtime registry and finalize guards).
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "internal/error.h"
#include "object.h"
#include "runtime.h"
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
test_RuntimeObjectIsLive_null(void)
{
   ASSERT_FALSE(hypredrv_RuntimeObjectIsLive(NULL));
}

static void
test_RuntimeRegisterObject_requires_initialized(void)
{
   hypredrv_t *dummy = (hypredrv_t *)calloc(1, sizeof(hypredrv_t));
   uint32_t    code;

   ASSERT_NOT_NULL(dummy);
   hypredrv_ErrorCodeResetAll();
   code = hypredrv_RuntimeRegisterObject((HYPREDRV_t)dummy);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   free(dummy);
}

static void
test_RuntimeRegisterObject_null_and_duplicate(void)
{
   hypredrv_t *obj = (hypredrv_t *)calloc(1, sizeof(hypredrv_t));

   ASSERT_NOT_NULL(obj);
   TEST_HYPRE_INIT();
   ASSERT_EQ_U32(hypredrv_RuntimeInitialize(), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   ASSERT_HAS_FLAG(hypredrv_RuntimeRegisterObject(NULL), ERROR_UNKNOWN_HYPREDRV_OBJ);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ_U32(hypredrv_RuntimeRegisterObject((HYPREDRV_t)obj), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   ASSERT_HAS_FLAG(hypredrv_RuntimeRegisterObject((HYPREDRV_t)obj),
                   ERROR_UNKNOWN_HYPREDRV_OBJ);

   hypredrv_RuntimeUnregisterObject((HYPREDRV_t)obj);
   free(obj);
   hypredrv_RuntimeFinalizeState();
   TEST_HYPRE_FINALIZE();
}

static void
test_RuntimeDestroyAllLiveObjects_null_destroy_fn(void)
{
   TEST_HYPRE_INIT();
   ASSERT_EQ_U32(hypredrv_RuntimeInitialize(), ERROR_NONE);
   (void)hypredrv_RuntimeDestroyAllLiveObjects(NULL);
   hypredrv_RuntimeFinalizeState();
   TEST_HYPRE_FINALIZE();
}

static uint32_t
destroy_nested_finalize_sweep(HYPREDRV_t h)
{
   ASSERT_TRUE(hypredrv_RuntimeIsFinalizing());
   (void)hypredrv_RuntimeDestroyAllLiveObjects(destroy_nested_finalize_sweep);
   hypredrv_RuntimeUnregisterObject(h);
   free(h);
   return hypredrv_ErrorCodeGet();
}

static void
test_RuntimeDestroyAllLiveObjects_nested_while_finalizing(void)
{
   hypredrv_t *obj = (hypredrv_t *)calloc(1, sizeof(hypredrv_t));

   ASSERT_NOT_NULL(obj);
   TEST_HYPRE_INIT();
   ASSERT_EQ_U32(hypredrv_RuntimeInitialize(), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_RuntimeRegisterObject((HYPREDRV_t)obj), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_RuntimeDestroyAllLiveObjects(destroy_nested_finalize_sweep),
                 ERROR_NONE);
   hypredrv_RuntimeFinalizeState();
   TEST_HYPRE_FINALIZE();
}

static void
test_RuntimeFinalizeState_blocks_with_live_objects(void)
{
   hypredrv_t *obj = (hypredrv_t *)calloc(1, sizeof(hypredrv_t));
   uint32_t    code;

   ASSERT_NOT_NULL(obj);
   TEST_HYPRE_INIT();
   ASSERT_EQ_U32(hypredrv_RuntimeInitialize(), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_RuntimeRegisterObject((HYPREDRV_t)obj), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   code = hypredrv_RuntimeFinalizeState();
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN);
   ASSERT_TRUE(hypredrv_RuntimeIsInitialized());

   hypredrv_RuntimeUnregisterObject((HYPREDRV_t)obj);
   free(obj);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ_U32(hypredrv_RuntimeFinalizeState(), ERROR_NONE);
   TEST_HYPRE_FINALIZE();
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

   RUN_TEST(test_RuntimeObjectIsLive_null);
   RUN_TEST(test_RuntimeRegisterObject_requires_initialized);

   RUN_TEST(test_RuntimeRegisterObject_null_and_duplicate);
   RUN_TEST(test_RuntimeDestroyAllLiveObjects_null_destroy_fn);
   RUN_TEST(test_RuntimeDestroyAllLiveObjects_nested_while_finalizing);
   RUN_TEST(test_RuntimeFinalizeState_blocks_with_live_objects);
#if HYPRE_CHECK_MIN_VERSION(23100, 0)
   RUN_TEST(test_RuntimeGpuAwareMPI_defaults_to_disabled);
   RUN_TEST(test_RuntimeGpuAwareMPI_recognizes_enable_environment);
   RUN_TEST(test_RuntimeGpuAwareMPI_requires_exact_value_one);
#endif

   MPI_Finalize();
   return 0;
}
