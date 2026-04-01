/******************************************************************************
 * Unit tests for src/runtime.c (runtime registry and finalize guards).
 ******************************************************************************/

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

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
   ASSERT_HAS_FLAG(hypredrv_RuntimeRegisterObject((HYPREDRV_t)obj), ERROR_UNKNOWN_HYPREDRV_OBJ);

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
   ASSERT_EQ_U32(hypredrv_RuntimeDestroyAllLiveObjects(destroy_nested_finalize_sweep), ERROR_NONE);
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

   MPI_Finalize();
   return 0;
}
