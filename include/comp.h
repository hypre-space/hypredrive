/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"

#ifdef HYPREDRV_USING_ZLIB
#include <zlib.h>
#endif

#ifdef HYPREDRV_USING_ZSTD
#include <zstd.h>
#endif

#ifdef HYPREDRV_USING_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

#ifdef HYPREDRV_USING_BLOSC
#include <blosc.h>
#endif

/* Enum to identify the compression algorithm */
typedef enum
{
   COMP_NONE,
   COMP_ZLIB,
   COMP_ZSTD,
   COMP_LZ4,
   COMP_LZ4HC,
   COMP_BLOSC
} comp_alg_t;
