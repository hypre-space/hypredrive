/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef FIELD_HEADER
#define FIELD_HEADER

#include <stddef.h>
#include "internal/containers.h"
#include "internal/yaml.h"

typedef void (*SetterFnc)(void *, const YAMLnode *);

typedef struct FieldOffsetMap_struct
{
   const char *name;
   size_t      offset;
   size_t      size; /* sizeof the member (HYPRE_Int may be 64-bit) */
   SetterFnc   setter;
} FieldOffsetMap;

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

// clang-format off
/* `size` is taken for every member (pointers included); only integer members use it. */
#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _setter) \
   {#_field_name, offsetof(_st, _field_name), sizeof(((_st *)0)->_field_name), _setter} /* NOLINT(bugprone-sizeof-expression) */
// clang-format on

/*-----------------------------------------------------------------------------
 * Prototypes
 *-----------------------------------------------------------------------------*/

void hypredrv_FieldOffsetMapSet(const FieldOffsetMap *, void *, const YAMLnode *);
void hypredrv_FieldTypeIntSet(void *, const YAMLnode *);
void hypredrv_FieldTypeNonNegIntSet(void *, const YAMLnode *);
void hypredrv_FieldTypePositiveIntSet(void *, const YAMLnode *);
void hypredrv_FieldTypeIntArraySet(void *, const YAMLnode *);
void hypredrv_FieldTypeStackIntArraySet(void *, const YAMLnode *);
void hypredrv_FieldTypeDoubleSet(void *, const YAMLnode *);
void hypredrv_FieldTypeDoubleArraySet(void *, const YAMLnode *);
void hypredrv_FieldTypeCharSet(void *, const YAMLnode *);
void hypredrv_FieldTypeStringSet(void *, const YAMLnode *);
void hypredrv_FieldTypeNoopSet(void *, const YAMLnode *);

#endif /* FIELD_HEADER */
