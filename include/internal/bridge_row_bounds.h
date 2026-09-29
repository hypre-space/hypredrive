/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_BRIDGE_ROW_BOUNDS_HEADER
#define HYPREDRV_BRIDGE_ROW_BOUNDS_HEADER

#include "internal/compatibility.h"

/* Each bridge supplies its own conversion callback so diagnostics and error
 * codes remain language-specific. Convert the start first, then the end. */
typedef uint32_t (*hypredrv_BridgeBigIntFromI64Fn)(int64_t value, const char *name,
                                                   HYPRE_BigInt *converted);

static inline uint32_t
hypredrv_BridgeRowBoundsFromI64(int64_t row_start_i64, int64_t row_end_i64,
                                HYPRE_BigInt *row_start, HYPRE_BigInt *row_end,
                                hypredrv_BridgeBigIntFromI64Fn convert)
{
   uint32_t code = convert(row_start_i64, "row_start", row_start);
   if (code != 0u)
   {
      return code;
   }
   return convert(row_end_i64, "row_end", row_end);
}

#endif /* HYPREDRV_BRIDGE_ROW_BOUNDS_HEADER */
