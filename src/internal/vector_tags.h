/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_VECTOR_TAGS_HEADER
#define HYPREDRV_VECTOR_TAGS_HEADER

#include <stdint.h>
#include "internal/object_fwd.h"

typedef struct hypredrv_VectorTags hypredrv_VectorTags;

uint32_t hypredrv_VectorTagsSet(hypredrv_t *);
void     hypredrv_VectorTagsClear(hypredrv_t *);

#endif
