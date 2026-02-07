/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef FIELD_HEADER
#define FIELD_HEADER

#include <stddef.h>
#include "containers.h"
#include "yaml.h"

typedef void (*SetterFnc)(void *, const YAMLnode *);

typedef struct FieldOffsetMap_struct
{
   const char *name;
   size_t      offset;
   SetterFnc   setter;
} FieldOffsetMap;

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

// clang-format off
#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _setter) \
   {#_field_name, offsetof(_st, _field_name), _setter}
// clang-format on

/*-----------------------------------------------------------------------------
 * Prototypes
 *-----------------------------------------------------------------------------*/

void FieldTypeIntSet(void *, const YAMLnode *);
void FieldTypeIntArraySet(void *, const YAMLnode *);
void FieldTypeStackIntArraySet(void *, const YAMLnode *);
void FieldTypeDoubleSet(void *, const YAMLnode *);
void FieldTypeDoubleArraySet(void *, const YAMLnode *);
void FieldTypeCharSet(void *, const YAMLnode *);
void FieldTypeStringSet(void *, const YAMLnode *);

#endif /* FIELD_HEADER */
