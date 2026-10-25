/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_SLOTS_HEADER
#define HYPREDRV_SLOTS_HEADER

#include <stdbool.h>
#include "HYPRE_IJ_mv.h"

typedef struct
{
   HYPRE_IJMatrix handle;
   bool           owned;
} hypredrv_MatrixSlot;

typedef struct
{
   HYPRE_IJVector handle;
   bool           owned;
} hypredrv_VectorSlot;

static inline bool
hypredrv_AttachmentIsOwned(bool library_mode)
{
   return (bool)!library_mode;
}

static inline void
hypredrv_MatrixSlotRelease(hypredrv_MatrixSlot *slot, HYPRE_IJMatrix alias)
{
   if (slot->handle && slot->owned && slot->handle != alias)
   {
      HYPRE_IJMatrixDestroy(slot->handle);
   }
   slot->handle = NULL;
   slot->owned  = false;
}

/* Reattaching a live handle preserves the original ownership decision. */
static inline void
hypredrv_MatrixSlotSet(hypredrv_MatrixSlot *slot, HYPRE_IJMatrix handle, bool owned,
                       HYPRE_IJMatrix alias)
{
   if (slot->handle == handle)
   {
      return;
   }
   hypredrv_MatrixSlotRelease(slot, alias);
   slot->handle = handle;
   slot->owned  = (bool)(owned && handle != NULL);
}

static inline void
hypredrv_VectorSlotRelease(hypredrv_VectorSlot *slot, HYPRE_IJVector alias)
{
   if (slot->handle && slot->owned && slot->handle != alias)
   {
      HYPRE_IJVectorDestroy(slot->handle);
   }
   slot->handle = NULL;
   slot->owned  = false;
}

static inline void
hypredrv_VectorSlotSet(hypredrv_VectorSlot *slot, HYPRE_IJVector handle, bool owned,
                       HYPRE_IJVector alias)
{
   if (slot->handle == handle)
   {
      return;
   }
   hypredrv_VectorSlotRelease(slot, alias);
   slot->handle = handle;
   slot->owned  = (bool)(owned && handle != NULL);
}

#endif
