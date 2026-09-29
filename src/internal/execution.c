/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "execution.h"
#include "internal/info.h"
#include "logging.h"
#include "object.h"

/*-----------------------------------------------------------------------------
 * Push general.* runtime settings (exec policy, memory pools) into hypre
 *-----------------------------------------------------------------------------*/

void
hypredrv_LogExecutionPolicy(HYPREDRV_t hypredrv)
{
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPRE execution policy: %s",
      hypredrv_ExecutionPolicyName(hypredrv->iargs->general.exec_policy));
}

int
hypredrv_ValidateDevicePreconditioner(HYPREDRV_t hypredrv, int device_requested,
                                      precon_t method, const precon_args *args)
{
#ifndef HYPRE_USING_GPU
   (void)hypredrv;
   (void)device_requested;
   (void)method;
   (void)args;
   return 1;
#else
   char reason[160];

   if (!device_requested ||
       hypredrv_PreconSupportsDevice(method, args, reason, sizeof(reason)))
   {
      return 1;
   }

   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd(
      "GPU execution requested, but the configured linear solver strategy is "
      "not available on GPUs: %s. Select a GPU-supported strategy or set "
      "general.exec_policy to host.",
      reason);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "rejecting unsupported GPU strategy: %s", reason);
   return 0;
#endif
}

uint32_t
hypredrv_ApplyGlobalRuntimeSettings(HYPREDRV_t hypredrv)
{
   if (!hypredrv || !hypredrv->iargs) /* GCOVR_EXCL_BR_LINE */
   {
      return ERROR_NONE;
   }

   if (hypredrv->iargs->general.exec_policy) /* GCOVR_EXCL_BR_LINE */
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#endif

#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      HYPRE_SetSpGemmUseVendor(hypredrv->iargs->general.use_vendor_spgemm);
      HYPRE_SetSpMVUseVendor(hypredrv->iargs->general.use_vendor_spmv);
      HYPRE_SetSpTransUseVendor(hypredrv->iargs->general.use_vendor_sptrans);
#endif

#ifdef HYPRE_USING_UMPIRE
      HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE");
      HYPRE_SetUmpireUMPoolName("HYPRE_UM");
      HYPRE_SetUmpireHostPoolName("HYPRE_HOST");
      HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED");

      HYPRE_SetUmpireDevicePoolSize(hypredrv->iargs->general.dev_pool_size);
      HYPRE_SetUmpireUMPoolSize(hypredrv->iargs->general.uvm_pool_size);
      HYPRE_SetUmpireHostPoolSize(hypredrv->iargs->general.host_pool_size);
      HYPRE_SetUmpirePinnedPoolSize(hypredrv->iargs->general.pinned_pool_size);
#endif
   }
   else
   {
#if HYPRE_CHECK_MIN_VERSION(22100, 0)
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif
   }

   return ERROR_NONE;
}

uint32_t
hypredrv_ApplyConfiguredDeviceInitialization(HYPREDRV_t hypredrv)
{
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(23100, 0)
   if (hypredrv && hypredrv->iargs && hypredrv->iargs->general.exec_policy &&
       !hypredrv->iargs->general.device_lazy_init)
   {
      uint32_t code = hypredrv_ApplyGlobalRuntimeSettings(hypredrv);
      if (code != ERROR_NONE)
      {
         return code;
      }

      HYPREDRV_LOG_OBJECTF(1, hypredrv,
                           "eager device initialization requested by "
                           "general.device_lazy_init=off");
      HYPRE_DeviceInitialize();
   }
#else
   (void)hypredrv;
#endif

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Migrate a user-supplied matrix/vector to the memory space of the exec policy
 *-----------------------------------------------------------------------------*/

void
hypredrv_PrepareExplicitObjectForConfiguredExecution(HYPREDRV_t hypredrv, void *obj,
                                                     int is_matrix)
{
   hypredrv_PrepareExplicitObjectsForConfiguredExecution(hypredrv, &obj, 1, is_matrix);
}

void
hypredrv_PrepareExplicitObjectsForConfiguredExecution(HYPREDRV_t   hypredrv,
                                                      void *const *objects,
                                                      size_t num_objects, int is_matrix)
{
#if !defined(HYPRE_USING_GPU) || !HYPRE_CHECK_MIN_VERSION(22000, 0)
   (void)hypredrv;
   (void)objects;
   (void)num_objects;
   (void)is_matrix;
#else
   HYPRE_MemoryLocation target_memory = HYPRE_MEMORY_HOST;

   if (!hypredrv || !hypredrv->iargs || !objects || num_objects == 0)
   {
      return;
   }

   size_t i = 0;
   while (i < num_objects && !objects[i])
   {
      i++;
   }
   if (i == num_objects)
   {
      return;
   }

   (void)hypredrv_ApplyGlobalRuntimeSettings(hypredrv);

   if (hypredrv->iargs->ls.exec_policy)
   {
      target_memory = HYPRE_MEMORY_DEVICE;
   }

   for (; i < num_objects; i++)
   {
      if (!objects[i])
      {
         continue;
      }
      if (is_matrix)
      {
         HYPRE_IJMatrixMigrate((HYPRE_IJMatrix)objects[i], target_memory);
      }
      else
      {
         HYPRE_IJVectorMigrate((HYPRE_IJVector)objects[i], target_memory);
      }
   }
#endif
}
