/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_VECTOR_TAGS_HEADER
#define HYPREDRV_VECTOR_TAGS_HEADER

#include "HYPREDRV.h"

typedef struct hypredrv_VectorTags hypredrv_VectorTags;

uint32_t hypredrv_VectorTagsSet(HYPREDRV_t);
void     hypredrv_VectorTagsClear(HYPREDRV_t);

#endif
