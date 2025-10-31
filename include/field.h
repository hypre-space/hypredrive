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

typedef void (*SetterFnc)(void *, YAMLnode *);

typedef struct FieldOffsetMap_struct
{
   const char *name;
   size_t      offset;
   SetterFnc   setter;
} FieldOffsetMap;

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _setter) \
   {                                                      \
      #_field_name, offsetof(_st, _field_name), _setter   \
   }

/*-----------------------------------------------------------------------------
 * Prototypes
 *-----------------------------------------------------------------------------*/

void FieldTypeIntSet(void *, YAMLnode *);
void FieldTypeIntArraySet(void *, YAMLnode *);
void FieldTypeStackIntArraySet(void *, YAMLnode *);
void FieldTypeDoubleSet(void *, YAMLnode *);
void FieldTypeCharSet(void *, YAMLnode *);
void FieldTypeStringSet(void *, YAMLnode *);

#endif /* FIELD_HEADER */
