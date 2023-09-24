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

//typedef void (*SetterFnc)(void* obj, const char* field_name, YAMLnode* node);
typedef void (*SetterFnc)(void*, YAMLnode*);

typedef enum FieldType_enum {
   FIELD_TYPE_INT,
   FIELD_TYPE_DOUBLE,
   FIELD_TYPE_CHAR,
   FIELD_TYPE_STRING,
   FIELD_TYPE_STRUCT
} FieldType;

typedef struct FieldOffsetMap_struct
{
   const char  *name;
   size_t       offset;
   FieldType    type;
   SetterFnc    setter;
} FieldOffsetMap;

#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _field_type, _setter) \
   {#_field_name, offsetof(_st, _field_name), _field_type, _setter}

#endif /* FIELD_HEADER */
