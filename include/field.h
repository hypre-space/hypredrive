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

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

#define FIELD_OFFSET_MAP_ENTRY(_st, _field_name, _setter) \
   {#_field_name, offsetof(_st, _field_name), _setter}

#define DEFINE_SET_FIELD_BY_NAME_FUNC(_func_name, _args_type, _map, _num_fields) \
   void _func_name(_args_type *_args, YAMLnode *_node) \
   { \
      for (size_t i = 0; i < (_num_fields); i++) \
      { \
         if (!strcmp((_map)[i].name, (_node)->key)) \
         { \
            (_map)[i].setter((void*)((char*)(_args) + (_map)[i].offset), (_node)); \
            return; \
         } \
      } \
   }

#define DEFINE_GET_VALID_KEYS_FUNC(_func_name, _num_fields, _map) \
   StrArray _func_name(void) \
   { \
      static const char* _keys[_num_fields]; \
      for (size_t i = 0; i < _num_fields; i++) \
      { \
         _keys[i] = _map[i].name; \
      } \
      return STR_ARRAY_CREATE(_keys); \
   }

#define DECLARE_GET_VALID_VALUES_FUNC(_prefix) \
   StrIntMapArray _prefix##GetValidValues(const char*); \

#define DECLARE_SET_DEFAULT_ARGS_FUNC(_prefix) \
   void _prefix##SetDefaultArgs(_prefix##_args*); \

#define DEFINE_SET_ARGS_FROM_YAML_FUNC(_prefix) \
   void _prefix##SetArgsFromYAML(_prefix##_args *args, YAMLnode *parent) \
   { \
      YAML_NODE_ITERATE(parent, child) \
      { \
         YAML_NODE_VALIDATE(child, \
                            _prefix##GetValidKeys, \
                            _prefix##GetValidValues); \
         \
         YAML_NODE_SET_FIELD(child, \
                             args, \
                             _prefix##SetFieldByName); \
      } \
   }

#define CALL_SET_DEFAULT_ARGS_FUNC(_prefix, _args) _prefix##SetDefaultArgs(_args)
#define CALL_SET_ARGS_FROM_YAML_FUNC(_prefix, _args, _yaml) _prefix##SetArgsFromYAML(_args, _yaml)
#define DEFINE_SET_ARGS_FUNC(_prefix) \
   void _prefix##SetArgs(void *vargs, YAMLnode *parent) \
   { \
      _prefix##_args *args = (_prefix##_args*) vargs; \
      CALL_SET_DEFAULT_ARGS_FUNC(_prefix, args); \
      CALL_SET_ARGS_FROM_YAML_FUNC(_prefix, args, parent); \
   }

/*-----------------------------------------------------------------------------
 * Prototypes
 *-----------------------------------------------------------------------------*/

void FieldTypeIntSet(void*, YAMLnode*);
void FieldTypeDoubleSet(void*, YAMLnode*);
void FieldTypeCharSet(void*, YAMLnode*);
void FieldTypeStringSet(void*, YAMLnode*);

#endif /* FIELD_HEADER */
