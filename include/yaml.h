/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef YAML_HEADER
#define YAML_HEADER

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include "utils.h"

#define MAX_LINE_LENGTH 1024

/*-----------------------------------------------------------------------------
 * YAML validity enum
 *-----------------------------------------------------------------------------*/

typedef enum YAMLvalidity_enum
{
   YAML_NODE_UNKNOWN,
   YAML_NODE_VALID,
   YAML_NODE_INVALID_KEY,
   YAML_NODE_INVALID_VAL,
   YAML_NODE_UNEXPECTED_VAL,
   YAML_NODE_INVALID_INDENT,
   YAML_NODE_INVALID_DIVISOR,
} YAMLvalidity;

/*-----------------------------------------------------------------------------
 * YAML print mode enum (for printing the tree)
 *-----------------------------------------------------------------------------*/

typedef enum YAMLprintMode_enum
{
   YAML_PRINT_MODE_NO_CHECKING,
   YAML_PRINT_MODE_ONLY_VALID,
   YAML_PRINT_MODE_ANY,
} YAMLprintMode;

/*-----------------------------------------------------------------------------
 * YAML node struct
 *-----------------------------------------------------------------------------*/

typedef struct YAMLnode_struct
{
   char                   *key;
   char                   *val;
   char                   *mapped_val;
   int                     level;
   YAMLvalidity            valid;
   struct YAMLnode_struct *parent;
   struct YAMLnode_struct *children;
   struct YAMLnode_struct *next;
} YAMLnode;

/*-----------------------------------------------------------------------------
 * YAML tree struct
 *-----------------------------------------------------------------------------*/

typedef struct _YAMLtree
{
   YAMLnode *root;
   int       base_indent;
   bool      is_validated;
} YAMLtree;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void YAMLtextRead(const char *, const char *, int, int *, size_t *, char **);

YAMLtree *YAMLtreeCreate(int);
void      YAMLtreeDestroy(YAMLtree **);
void      YAMLtreeBuild(int, char *, YAMLtree **);
void      YAMLtreeUpdate(int, char **, YAMLtree *);
void      YAMLtreePrint(YAMLtree *, YAMLprintMode);
void      YAMLtreeValidate(YAMLtree *);

YAMLnode *YAMLnodeCreate(char *, char *, int);
void      YAMLnodeDestroy(YAMLnode *);
void      YAMLnodeAddChild(YAMLnode *, YAMLnode *);
void      YAMLnodeAppend(YAMLnode *, YAMLnode **);
void      YAMLnodeValidate(YAMLnode *);
void      YAMLnodePrint(YAMLnode *, YAMLprintMode);
YAMLnode *YAMLnodeFindByKey(YAMLnode *, const char *);
YAMLnode *YAMLnodeFindChildByKey(YAMLnode *, const char *);
char     *YAMLnodeFindChildValueByKey(YAMLnode *, const char *);

/*-----------------------------------------------------------------------------
 * Public macros
 *-----------------------------------------------------------------------------*/

#define YAML_NODE_VALIDATE_HELPER(_node, _map_array)                       \
   do                                                                      \
   {                                                                       \
      if (StrIntMapArrayDomainEntryExists(_map_array, _node->val))         \
      {                                                                    \
         int _mapped = StrIntMapArrayGetImage(_map_array, _node->val);     \
         int _length = snprintf(NULL, 0, "%d", _mapped) + 1;               \
         if (!_node->mapped_val)                                           \
         {                                                                 \
            _node->mapped_val = (char *)malloc(_length * sizeof(char));    \
         }                                                                 \
         else if (_length > (int)strlen(_node->mapped_val))                \
         {                                                                 \
            _node->mapped_val =                                            \
               (char *)realloc(_node->mapped_val, _length * sizeof(char)); \
         }                                                                 \
         snprintf(_node->mapped_val, _length, "%d", _mapped);              \
         _node->valid = YAML_NODE_VALID;                                   \
      }                                                                    \
      else                                                                 \
      {                                                                    \
         _node->valid = YAML_NODE_INVALID_VAL;                             \
      }                                                                    \
   } while (0);

#define YAML_NODE_VALIDATE(_node, _callA, _callB)                                  \
   if ((_node->valid != YAML_NODE_INVALID_DIVISOR) &&                              \
       (_node->valid != YAML_NODE_INVALID_INDENT))                                 \
   {                                                                               \
      StrArray       _keys = _callA();                                             \
      StrIntMapArray _map_array_type;                                              \
      /* printf(" 0_node->key: %s\n", _node->key); */                              \
      if (StrArrayEntryExists(_keys, _node->key))                                  \
      {                                                                            \
         /* printf(" 1_node->key: %s - level: %d\n", _node->key, _node->level); */ \
         StrIntMapArray _map_array_key = _callB(_node->key);                       \
         if (_map_array_key.size > 0)                                              \
         {                                                                         \
            YAML_NODE_VALIDATE_HELPER(_node, _map_array_key)                       \
         }                                                                         \
         else                                                                      \
         {                                                                         \
            if (!_node->mapped_val) _node->mapped_val = strdup(_node->val);        \
            _node->valid = YAML_NODE_VALID;                                        \
         }                                                                         \
      }                                                                            \
      else if ((_map_array_type = _callB("type")),                                 \
               !strcmp(_node->key, "type") && _map_array_type.size > 0)            \
      {                                                                            \
         /* printf(" 2_node->key: %s - level: %d\n", _node->key, _node->level); */ \
         YAML_NODE_VALIDATE_HELPER(_node, _map_array_type)                         \
      }                                                                            \
      else                                                                         \
      {                                                                            \
         /* printf(" 3_node->key: %s\n", _node->key); */                           \
         _node->valid = YAML_NODE_INVALID_KEY;                                     \
      }                                                                            \
   }

#define YAML_NODE_SET_FIELD(_node, _args, _call) \
   if (_node->valid == YAML_NODE_VALID)          \
   {                                             \
      _call(_args, _node);                       \
   }

#define YAML_NODE_GET_VALIDITY(_node) _node->valid
#define YAML_NODE_SET_VALID(_node) _node->valid = YAML_NODE_VALID
#define YAML_NODE_SET_INVALID_KEY(_node) _node->valid = YAML_NODE_INVALID_KEY
#define YAML_NODE_SET_INVALID_VAL(_node) _node->valid = YAML_NODE_INVALID_VAL
#define YAML_NODE_SET_INVALID_INDENT(_node) _node->valid = YAML_NODE_INVALID_INDENT
#define YAML_NODE_SET_INVALID_DIVISOR(_node) _node->valid = YAML_NODE_INVALID_DIVISOR
#define YAML_NODE_SET_VALID_IF_NO_VAL(_node) \
   _node->valid = (!strcmp(_node->val, "")) ? YAML_NODE_VALID : YAML_NODE_UNEXPECTED_VAL

#define YAML_NODE_ITERATE(_parent, _child)                           \
   for (YAMLnode * (_child) = (_parent)->children; (_child) != NULL; \
        (_child)            = (_child)->next)

#define YAML_NODE_DEBUG(_node)                                                           \
   do                                                                                    \
   {                                                                                     \
      printf("YAMLnode address = %p, {key = %s (%p), val = %s, mapped_val = %s, ",       \
             (void *)_node, _node->key, (void *)_node->key, _node->val,                  \
             _node->mapped_val);                                                         \
      printf("level = %d, valid = %d, parent = %p, children = %p, next = %p}\n",         \
             _node->level, _node->valid, (void *)_node->parent, (void *)_node->children, \
             (void *)_node->next);                                                       \
   } while (0)

#endif /* YAML_HEADER */
