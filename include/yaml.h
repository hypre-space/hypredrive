/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef YAML_HEADER
#define YAML_HEADER

#include <limits.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "error.h"
#include "utils.h"

#define MAX_LINE_LENGTH 1024

/*-----------------------------------------------------------------------------
 * YAML validity enum
 *-----------------------------------------------------------------------------*/

typedef enum YAMLvalidity_enum {
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

typedef enum YAMLprintMode_enum {
   YAML_PRINT_MODE_NO_CHECKING,
   YAML_PRINT_MODE_ONLY_VALID,
   YAML_PRINT_MODE_ANY,
} YAMLprintMode;

/*-----------------------------------------------------------------------------
 * YAML node struct
 *-----------------------------------------------------------------------------*/

typedef struct YAMLnode_struct {
   char                     *key;
   char                     *val;
   char                     *mapped_val;
   int                       level;
   YAMLvalidity              valid;
   struct YAMLnode_struct   *parent;
   struct YAMLnode_struct   *children;
   struct YAMLnode_struct   *next;
} YAMLnode;

/*-----------------------------------------------------------------------------
 * YAML tree struct
 *-----------------------------------------------------------------------------*/

typedef struct YAMLtree_struct {
   YAMLnode  *root;
} YAMLtree;

/*-----------------------------------------------------------------------------
 * Public prototypes - TODO: change names (YAMLcreateTree -> YAMLTreeCreate)
 *-----------------------------------------------------------------------------*/

YAMLtree* YAMLtreeCreate(void);
void YAMLtreeDestroy(YAMLtree**);
void YAMLtreeBuild(char*, YAMLtree**);
void YAMLtreeUpdate(int, char**, YAMLtree*);
void YAMLtreePrint(YAMLtree*, YAMLprintMode);

YAMLnode* YAMLnodeCreate(char*, char*, int);
void YAMLnodeDestroy(YAMLnode*);
void YAMLnodeAddChild(YAMLnode*, YAMLnode*);
void YAMLnodeAppend(YAMLnode*, YAMLnode**);
void YAMLnodePrint(YAMLnode*, YAMLprintMode);
YAMLnode* YAMLnodeFindByKey(YAMLnode*, const char*);
YAMLnode* YAMLnodeFindChildByKey(YAMLnode*, const char*);
char* YAMLnodeFindChildValueByKey(YAMLnode*, const char*);

/*-----------------------------------------------------------------------------
 * Public macros
 *-----------------------------------------------------------------------------*/

#define YAML_NODE_VALIDATE(_node, _callA, _callB) \
   if ((_node->valid != YAML_NODE_INVALID_DIVISOR) && \
       (_node->valid != YAML_NODE_INVALID_INDENT)) \
   { \
      StrArray _keys = _callA(); \
      if (StrArrayEntryExists(_keys, _node->key)) \
      { \
         StrIntMapArray _map_array = _callB(_node->key); \
         if (_map_array.size > 0) \
         { \
            if (StrIntMapArrayDomainEntryExists(_map_array, _node->val)) \
            { \
               int _mapped = StrIntMapArrayGetImage(_map_array, _node->val); \
               int _length = snprintf(NULL, 0, "%d", _mapped) + 1; \
               if (!_node->mapped_val) \
               { \
                  _node->mapped_val = (char*) malloc(_length * sizeof(char)); \
               } \
               else if (_length > strlen(_node->mapped_val)) \
               { \
                  _node->mapped_val = (char*) realloc(_node->mapped_val, _length * sizeof(char)); \
               } \
               snprintf(_node->mapped_val, _length, "%d", _mapped); \
               _node->valid = YAML_NODE_VALID; \
            } \
            else \
            { \
               _node->valid = YAML_NODE_INVALID_VAL; \
            } \
         } \
         else \
         { \
            if (!_node->mapped_val) \
            { \
               _node->mapped_val = strdup(_node->val); \
            } \
            _node->valid = YAML_NODE_VALID; \
         } \
      } \
      else \
      { \
         _node->valid = YAML_NODE_INVALID_KEY; \
      } \
   }

#define YAML_NODE_SET_FIELD(_node, _args, _call) \
   if (_node->valid == YAML_NODE_VALID) \
   { \
      _call(_args, _node); \
   }

#define YAML_NODE_GET_VALIDITY(_node) _node->valid
#define YAML_NODE_SET_VALID(_node) _node->valid = YAML_NODE_VALID
#define YAML_NODE_SET_INVALID_INDENT(_node) _node->valid = YAML_NODE_INVALID_INDENT
#define YAML_NODE_SET_INVALID_DIVISOR(_node) _node->valid = YAML_NODE_INVALID_DIVISOR
#define YAML_NODE_SET_VALID_IF_NO_VAL(_node) \
    _node->valid = (!strcmp(_node->val, "")) ? YAML_NODE_VALID : YAML_NODE_UNEXPECTED_VAL

#define YAML_NODE_ITERATE(_parent, _child) \
    for (YAMLnode* (_child) = (_parent)->children; (_child) != NULL; (_child) = (_child)->next)

// TODO: Remove the following macros:


#define YAML_CALL_IF_OPEN() if (0) {}
#define YAML_CALL_IF_CLOSE(_node) else {}
#define YAML_SET_IF_OPEN() if (0) {}
#define YAML_SET_IF_CLOSE(_node) else { ErrorMsgAddUnknownKey(_node->key); }
#define YAML_SET_IF_CLOSE_(_node) else { ErrorMsgAddInvalidKeyValPair(_node->key, _node->val); }
#define YAML_SET_INTEGER_IF_KEY_MATCHES(_var, _key, _node) \
   else if (!strcmp(_node->key, _key)) { _var = atoi(_node->val); }
#define YAML_SET_INTARRAY_IF_KEY_MATCHES(_count, _array, _key, _node) \
   else if (!strcmp(_node->key, _key)) { StrToIntArray(_node->val, _count, _array); }
#define YAML_SET_REAL_IF_KEY_MATCHES(_var, _key, _node) \
   else if (!strcmp(_node->key, _key)) { _var = atof(_node->val); }
#define YAML_SET_STRING_IF_KEY_MATCHES(_var, _key, _node) \
   else if (!strcmp(_node->key, _key)) { _var = strdup(_node->val); }
#define YAML_SET_INTEGER_IF_VAL_MATCHES(_var, _val, _node) \
   else if (!strcmp(_node->val, _val)) { _var = atoi(_node->val); }
#define YAML_SET_INTEGER_IF_VAL_MATCHES_TWO(_var, _val0, _val1, _node) \
   else if (!strcmp(_node->val, _val0) || !strcmp(_node->val, #_val1)) { _var = _val1; }
#define YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(_var, _val0, _val1, _node) \
   else if (!strcmp(_node->val, _val0) || !strcmp(_node->val, #_val1)) { _var = _val1; }
#define YAML_SET_INTEGER_IF_KEY_EXISTS(_var, _key, _node) \
   {char *val = YAMLnodeFindChildValueByKey(_node, _key); if (val != NULL) { _var = atoi(val); }}
#define YAML_CALL_IF_KEY_MATCHES(_call, _args, _node, _key) \
   else if (!strcmp(_node->key, _key)) { _call(_args, _node); }
#define YAML_CALL_IF_VAL_MATCHES(_call, _args, _node, _val) \
   else if (!strcmp(_node->val, _val)) { _call(_args, _node); }
#define YAML_RETURN_IF_VAL_MATCHES(_node, _val) \
   if (!strcmp(_node->val, _val)) { return; }
#define YAML_INC_INTEGER_IF_KEY_EXISTS(_var, _key, p) \
   { YAMLnode* c = p->children; while (c) { if (!strcmp(c->key, _key)) { _var++; } c = c->next; } }
#endif
