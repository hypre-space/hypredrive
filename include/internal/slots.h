/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_SLOTS_HEADER
#define HYPREDRV_SLOTS_HEADER

#include <stdbool.h>
#include <stddef.h>
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
hypredrv_MatrixSlotRelease(hypredrv_MatrixSlot *slot)
{
   if (slot->handle && slot->owned)
   {
      HYPRE_IJMatrixDestroy(slot->handle);
   }
   slot->handle = NULL;
   slot->owned  = false;
}

static inline void
hypredrv_VectorSlotRelease(hypredrv_VectorSlot *slot)
{
   if (slot->handle && slot->owned)
   {
      HYPRE_IJVectorDestroy(slot->handle);
   }
   slot->handle = NULL;
   slot->owned  = false;
}

#endif
