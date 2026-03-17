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

enum
{
   MAX_LINE_LENGTH = 1024,
};

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
 *
 * Represents a key-value pair in the YAML tree. Important distinction:
 *
 *   - For flat entries like "solver: gmres":
 *       key = "solver", val = "gmres", children = NULL
 *
 *   - For nested entries like:
 *       ilu:
 *         type: bj-ilut
 *       key = "ilu", val = "" (empty!), children = [type node, ...]
 *
 * When processing nested structures, check the KEY to identify the block type,
 * not the VAL (which is empty for parent nodes with children).
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

typedef struct YAMLtree
{
   YAMLnode *root;
   int       base_indent;
   bool      is_validated;
} YAMLtree;

/*-----------------------------------------------------------------------------
 * Schema-driven validation helpers (used heavily by config-driven components)
 *-----------------------------------------------------------------------------*/

typedef StrArray (*YAMLGetValidKeysFunc)(void);
typedef StrIntMapArray (*YAMLGetValidValuesFunc)(const char *);
typedef void (*YAMLSetFieldByNameFunc)(void *, const YAMLnode *);

void hypredrv_YAMLnodeValidateSchema(YAMLnode *, YAMLGetValidKeysFunc,
                                     YAMLGetValidValuesFunc);
void hypredrv_YAMLSetArgsGeneric(void *, YAMLnode *, YAMLGetValidKeysFunc,
                                 YAMLGetValidValuesFunc, YAMLSetFieldByNameFunc);

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void hypredrv_YAMLtextRead(const char *, const char *, int, int *, size_t *, char **);

YAMLtree *hypredrv_YAMLtreeCreate(int);
void      hypredrv_YAMLtreeDestroy(YAMLtree **);
void      hypredrv_YAMLtreeBuild(int, const char *, YAMLtree **);
void      hypredrv_YAMLtreeUpdate(int, char **, YAMLtree *);
void      hypredrv_YAMLtreePrint(YAMLtree *, YAMLprintMode);
void      hypredrv_YAMLtreeValidate(YAMLtree *);

YAMLnode *hypredrv_YAMLnodeCreate(const char *, const char *, int);
void      hypredrv_YAMLnodeDestroy(YAMLnode *);
void      hypredrv_YAMLnodeAddChild(YAMLnode *, YAMLnode *);
void      hypredrv_YAMLnodeAppend(YAMLnode *, YAMLnode **);
void      hypredrv_YAMLnodeValidate(YAMLnode *);
void      hypredrv_YAMLnodePrint(YAMLnode *, YAMLprintMode);
YAMLnode *hypredrv_YAMLnodeFindByKey(YAMLnode *, const char *);
YAMLnode *hypredrv_YAMLnodeFindChildByKey(YAMLnode *, const char *);
char     *hypredrv_YAMLnodeFindChildValueByKey(YAMLnode *, const char *);
int       hypredrv_YAMLnodeCollectSequenceItems(YAMLnode *, YAMLnode ***);
void      hypredrv_YAMLtreeExpandIncludes(YAMLtree *tree, const char *base_dir);

/*-----------------------------------------------------------------------------
 * Public macros
 *-----------------------------------------------------------------------------*/

#define YAML_NODE_VALIDATE_HELPER(_node, _map_array)                               \
   do                                                                              \
   {                                                                               \
      if (hypredrv_StrIntMapArrayDomainEntryExists(_map_array, _node->val))        \
      {                                                                            \
         int _mapped = hypredrv_StrIntMapArrayGetImage(_map_array, _node->val);    \
         int _length = snprintf(NULL, 0, "%d", _mapped) + 1;                       \
         if (!_node->mapped_val)                                                   \
         {                                                                         \
            _node->mapped_val = (char *)malloc((size_t)_length * sizeof(char));    \
         }                                                                         \
         else if (_length > (int)strlen(_node->mapped_val))                        \
         {                                                                         \
            _node->mapped_val =                                                    \
               (char *)realloc(_node->mapped_val, (size_t)_length * sizeof(char)); \
         }                                                                         \
         snprintf(_node->mapped_val, (size_t)_length, "%d", _mapped);              \
         _node->valid = YAML_NODE_VALID;                                           \
      }                                                                            \
      else                                                                         \
      {                                                                            \
         _node->valid = YAML_NODE_INVALID_VAL;                                     \
      }                                                                            \
   } while (0);

#define YAML_NODE_VALIDATE(_node, _callA, _callB)                   \
   do                                                               \
   {                                                                \
      hypredrv_YAMLnodeValidateSchema((_node), (_callA), (_callB)); \
   } while (0)

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
