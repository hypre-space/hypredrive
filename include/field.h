/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef FIELD_HEADER
#define FIELD_HEADER

#include <stddef.h>
#include "maps.h"
#include "yaml.h"

typedef void (*SetterFnc)(void*, YAMLnode*);

typedef struct FieldOffsetMap_struct
{
   const char  *name;
   size_t       offset;
   SetterFnc    setter;
} FieldOffsetMap;

#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _setter) \
   {#_field_name, offsetof(_st, _field_name), _setter}

/*-----------------------------------------------------------------------------
 * Prototypes
 *-----------------------------------------------------------------------------*/

void FieldTypeIntSet(void*, YAMLnode*);
void FieldTypeDoubleSet(void*, YAMLnode*);
void FieldTypeCharSet(void*, YAMLnode*);
void FieldTypeStringSet(void*, YAMLnode*);

#endif /* FIELD_HEADER */