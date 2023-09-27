/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef GEN_MACROS_HEADER
#define GEN_MACROS_HEADER

#include "maps.h"
#include "yaml.h"

/**
 * @brief Defines a function to set a field by name.
 *
 * @details This macro generates a function that sets the field of a structure
 * based on the provided name. The function uses a map to find the correct field
 * and then assigns the appropriate value.
 *
 * @param _funcName Name of the generated function.
 * @param _argsType Type of the argument structure.
 * @param _map Name of the map containing field information.
 * @param _numFields Number of fields in the map.
 */
#define DEFINE_SET_FIELD_BY_NAME_FUNC(_funcName, _argsType, _map, _numFields) \
   void _funcName(_argsType *_args, YAMLnode *_node) \
   { \
      for (size_t i = 0; i < (_numFields); i++) \
      { \
         if (!strcmp((_map)[i].name, (_node)->key)) \
         { \
            (_map)[i].setter((void*)((char*)(_args) + (_map)[i].offset), (_node)); \
            return; \
         } \
      } \
   }

/**
 * @brief Defines a function to get valid keys.
 *
 * @details This macro generates a function that constructs a static array
 * of key names from a given map and returns it as a StrArray object type.
 *
 * @param _funcName Name of the generated function.
 * @param _numFields Number of fields in the map.
 * @param _map Name of the map containing field information.
 */
#define DEFINE_GET_VALID_KEYS_FUNC(_funcName, _numFields, _map) \
   StrArray _funcName(void) \
   { \
      static const char* _keys[_numFields]; \
      for (size_t i = 0; i < _numFields; i++) \
      { \
         _keys[i] = _map[i].name; \
      } \
      return STR_ARRAY_CREATE(_keys); \
   }

/**
 * @brief Declares a function to get valid values.
 *
 * @details This macro generates a forward declaration for a function
 * that retrieves valid values as a StrIntMapArray object type.
 *
 * @param _prefix Prefix used in the naming of the declared function.
 */
#define DECLARE_GET_VALID_VALUES_FUNC(_prefix) \
   StrIntMapArray _prefix##GetValidValues(const char*); \

/**
 * @brief Declares a function to set default arguments.
 *
 * @details This macro generates a forward declaration for a function
 * that sets default arguments for a structure holding.
 *
 * @param _prefix Prefix used in the naming of the declared function.
 */
#define DECLARE_SET_DEFAULT_ARGS_FUNC(_prefix) \
   void _prefix##SetDefaultArgs(_prefix##_args*); \

/**
 * @brief Defines a function to set arguments from a YAML node.
 *
 * @details This macro generates a function that iterates over child nodes,
 * validates them, and sets fields, indicated by the node's keys, by name.
 *
 * @param _prefix Prefix used in the naming of the generated function.
 */
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

/**
 * @brief Defines a function to set arguments of a generic struct.
 *
 * @details This macro generates a function that sets default arguments
 * and then sets arguments from a YAML node.
 *
 * @param _prefix Prefix used in the naming of the generated function.
 */
#define DEFINE_SET_ARGS_FUNC(_prefix) \
   void _prefix##SetArgs(void *vargs, YAMLnode *parent) \
   { \
      _prefix##_args *args = (_prefix##_args*) vargs; \
      CALL_SET_DEFAULT_ARGS_FUNC(_prefix, args); \
      CALL_SET_ARGS_FROM_YAML_FUNC(_prefix, args, parent); \
   }

/**
 * @brief Adds an entry in the field offset map.
 *
 * @details X-macro used to generate an entry in a field offset map array,
 *          associating a structure member with a setter. It is designed to work in
 *          conjunction with the DEFINE_FIELD_OFFSET_MAP macro
 *
 * @param _prefix The prefix to be concatenated with _field to form the structure member.
 * @param _field The field/member of the structure.
 * @param _setter The function or macro to set the value of the field/member.
 */
#define ADD_FIELD_OFFSET_ENTRY(_prefix, _field, _setter) \
   FIELD_OFFSET_MAP_ENTRY(_prefix##_args, _field, _setter),

/**
 * @brief A macro to define a field offset map for a given prefix.
 *
 * @details This macro generates a static array of FieldOffsetMap, containing
 *          entries created by ADD_FIELD_OFFSET_ENTRY macro, each associating
 *          a structure field with a setter.
 *
 * @param _prefix The prefix used to identify the structure and its fields/members.
 */
#define DEFINE_FIELD_OFFSET_MAP(_prefix) \
   static const FieldOffsetMap _prefix##_field_offset_map[] = \
   { \
      _prefix##_FIELDS(_prefix) \
   };

/**
 * @brief Calls a function to set default arguments.
 *
 * @details This macro abstracts the calling pattern of the function
 * that sets default arguments.
 *
 * @param _prefix Prefix used in the naming of the called function.
 * @param _args Pointer to the arguments structure.
 */
#define CALL_SET_DEFAULT_ARGS_FUNC(_prefix, _args) _prefix##SetDefaultArgs(_args)

/**
 * @brief Calls a function to set arguments from a YAML node.
 *
 * @details This macro abstracts the calling pattern of the function
 * that sets arguments based on a YAML node.
 *
 * @param _prefix Prefix used in the naming of the called function.
 * @param _args Pointer to the arguments structure.
 * @param _yaml Pointer to the YAML node structure.
 */
#define CALL_SET_ARGS_FROM_YAML_FUNC(_prefix, _args, _yaml) _prefix##SetArgsFromYAML(_args, _yaml)

/**
 * @def GENERATE_PREFIXED_COMPONENTS(prefix)
 *
 * @brief An X-macro macro for generating a series of component definitions, declarations,
 * and initializations based on the provided prefix.
 *
 * @details This is an aggregate macro that generates several utility functions, declarations
 * and a field offset map, all prefixed by the passed parameter. It is designed to reduce the
 * amount of redundant code needed when creating multiple sets of similar components.
 * The generated components include:
 *
 * - A (Field, Offset, SetterFnc) map.
 * - A function definition to set fields by name.
 * - A function definition to get valid keys.
 * - A function declaration for getting valid values.
 * - A function declaration for setting default arguments.
 * - A function to set arguments from YAML.
 * - A function to set arguments (from YAML + default).
 *
 * @param prefix The prefix to be appended to the names of the generated components.
 *
 * @note The prefix parameter should be a valid identifier and should correspond to the
 * intended naming convention for the generated components.
 *
 * Usage:
 * @code
 * GENERATE_PREFIXED_COMPONENTS(AMGint)
 * @endcode
 *
 * In this example, the macro will generate function definitions like `AMGintSetFieldByName`,
 * `AMGintGetValidKeys`, etc., and function declarations like AMGintGetValidValues(const char*),
 * AMGintSetDefaultArgs(void), and AMGintSetDefaultArgs(void, YAMLnode*)
 */
#define GENERATE_PREFIXED_COMPONENTS(prefix) \
   DEFINE_FIELD_OFFSET_MAP(prefix); \
   DEFINE_SET_FIELD_BY_NAME_FUNC(prefix##SetFieldByName, \
                                 prefix##_args, \
                                 prefix##_field_offset_map, \
                                 prefix##_NUM_FIELDS); \
   DEFINE_GET_VALID_KEYS_FUNC(prefix##GetValidKeys, \
                              prefix##_NUM_FIELDS, \
                              prefix##_field_offset_map); \
   DECLARE_GET_VALID_VALUES_FUNC(prefix); \
   DECLARE_SET_DEFAULT_ARGS_FUNC(prefix); \
   DEFINE_SET_ARGS_FROM_YAML_FUNC(prefix); \
   DEFINE_SET_ARGS_FUNC(prefix); \

#endif /* GEN_MACROS */
