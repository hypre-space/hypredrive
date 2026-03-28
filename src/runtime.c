/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "runtime.h"
#include <stdlib.h>
#include "error.h"
#include "hypredrv_log.h"
#include "hypredrv_object.h"
#include "presets.h"

typedef struct
{
   bool       initialized;
   bool       finalize_in_progress;
   int        active_count;
   int        next_object_id;
   HYPREDRV_t live_head;
} RuntimeState;

static RuntimeState g_runtime_state = {
   .initialized          = false,
   .finalize_in_progress = false,
   .active_count         = 0,
   .next_object_id       = 1,
   .live_head            = NULL,
};

static bool
RuntimeListContains(HYPREDRV_t hypredrv)
{
   for (HYPREDRV_t cursor = g_runtime_state.live_head; cursor; cursor = cursor->next_live)
   {
      if (cursor == hypredrv)
      {
         return true;
      }
   }

   return false;
}

bool
hypredrv_RuntimeIsInitialized(void)
{
   return g_runtime_state.initialized;
}

bool
hypredrv_RuntimeIsFinalizing(void)
{
   return g_runtime_state.finalize_in_progress;
}

bool
hypredrv_RuntimeObjectIsLive(HYPREDRV_t hypredrv)
{
   if (hypredrv == NULL)
   {
      return false;
   }

   return RuntimeListContains(hypredrv);
}

int
hypredrv_RuntimeGetActiveCount(void)
{
   return g_runtime_state.active_count;
}

uint32_t
hypredrv_RuntimeInitialize(void)
{
   if (!g_runtime_state.initialized)
   {
      /* A fresh runtime initialization owns a fresh error-state view. */
      hypredrv_ErrorStateReset();
      hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "runtime initialize begin");

      /* Initialize hypre */
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Initialize();
#if HYPRE_CHECK_MIN_VERSION(23100, 0)
      HYPRE_DeviceInitialize();
#endif
#endif

#if HYPRE_CHECK_MIN_VERSION(23100, 16)
      const char *env_log_level = getenv("HYPRE_LOG_LEVEL");
      HYPRE_Int   log_level =
         (env_log_level) ? (HYPRE_Int)strtol(env_log_level, NULL, 10) : 0;

      HYPRE_SetLogLevel(log_level);
      hypredrv_LogCommf(2, MPI_COMM_WORLD, NULL, 0,
                        "forwarded HYPRE_LOG_LEVEL=%d to hypre", (int)log_level);
#endif

      g_runtime_state.initialized = true;
      hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "runtime initialize end");
   }

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_RuntimeRegisterObject(HYPREDRV_t hypredrv)
{
   if (!g_runtime_state.initialized)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED);
      return hypredrv_ErrorCodeGet();
   }

   if (!hypredrv)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      hypredrv_ErrorMsgAdd("Cannot register a NULL HYPREDRV object");
      return hypredrv_ErrorCodeGet();
   }

   if (RuntimeListContains(hypredrv))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      hypredrv_ErrorMsgAdd("HYPREDRV object is already registered");
      return hypredrv_ErrorCodeGet();
   }

   hypredrv->runtime_object_id = g_runtime_state.next_object_id++;

   hypredrv->next_live       = g_runtime_state.live_head;
   g_runtime_state.live_head = hypredrv;
   g_runtime_state.active_count++;
   hypredrv_LogObjectf(2, hypredrv, "registered object (active=%d)",
                       g_runtime_state.active_count);

   return hypredrv_ErrorCodeGet();
}

void
hypredrv_RuntimeUnregisterObject(HYPREDRV_t hypredrv)
{
   HYPREDRV_t *cursor = &g_runtime_state.live_head;

   while (*cursor)
   {
      if (*cursor == hypredrv)
      {
         *cursor = hypredrv->next_live;
         if (g_runtime_state.active_count > 0)
         {
            g_runtime_state.active_count--;
         }
         hypredrv_LogObjectf(2, hypredrv, "unregistered object (active=%d)",
                             g_runtime_state.active_count);
         break;
      }
      cursor = &(*cursor)->next_live;
   }

   if (hypredrv)
   {
      hypredrv->next_live = NULL;
   }
}

uint32_t
hypredrv_RuntimeDestroyAllLiveObjects(hypredrv_RuntimeDestroyFunc destroy_fn)
{
   if (!g_runtime_state.initialized || !destroy_fn)
   {
      return hypredrv_ErrorCodeGet();
   }

   if (g_runtime_state.finalize_in_progress)
   {
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "finalize sweep begin (active=%d)",
                     g_runtime_state.active_count);
   g_runtime_state.finalize_in_progress = true;

   while (g_runtime_state.live_head)
   {
      HYPREDRV_t hypredrv = g_runtime_state.live_head;
      hypredrv_LogObjectf(2, hypredrv, "auto-destroying live object during finalize");
      destroy_fn(hypredrv);
   }

   g_runtime_state.finalize_in_progress = false;
   hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "finalize sweep end (active=%d)",
                     g_runtime_state.active_count);

   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_RuntimeFinalizeState(void)
{
   if (g_runtime_state.initialized)
   {
      hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "runtime finalize begin");
      if (g_runtime_state.live_head)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd(
            "Runtime finalize requested with %d live HYPREDRV object(s)",
            g_runtime_state.active_count);
         hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0,
                           "runtime finalize blocked: %d live object(s) remain",
                           g_runtime_state.active_count);
         return hypredrv_ErrorCodeGet();
      }

      hypredrv_PresetFreeUserPresets();
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      HYPRE_Finalize();
#endif
      g_runtime_state.initialized = false;
      hypredrv_LogCommf(1, MPI_COMM_WORLD, NULL, 0, "runtime finalize end");
   }

   g_runtime_state.finalize_in_progress = false;
   g_runtime_state.active_count         = 0;
   g_runtime_state.next_object_id       = 1;
   g_runtime_state.live_head            = NULL;

   /* Do not leak message buffers across independent initialize/finalize cycles. */
   hypredrv_ErrorStateReset();
   hypredrv_LogReset();

   return hypredrv_ErrorCodeGet();
}
