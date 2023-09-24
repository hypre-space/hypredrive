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
   YAML_NODE_VALID,
   YAML_NODE_INVALID_KEY,
   YAML_NODE_INVALID_VAL,
} YAMLvalidity;

typedef enum YAMLmode_enum {
   YAML_MODE_ANY,
   YAML_MODE_VALID,
   YAML_MODE_INVALID,
} YAMLmode;

/*-----------------------------------------------------------------------------
 * YAML node struct
 *-----------------------------------------------------------------------------*/

typedef struct YAMLnode_struct {
   char                     *key;
   char                     *val;
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

YAMLtree* YAMLcreateTree(void);
int YAMLdestroyTree(YAMLtree**);
int YAMLbuildTree(char*, YAMLtree**);
int YAMLprintTree(YAMLtree*, YAMLmode);
/* int YAMLfindIntegerByKey(const YAMLdict *, const char*, int, int); */

YAMLnode* YAMLcreateNode(char*, char*, int);
int YAMLdestroyNode(YAMLnode*);
int YAMLaddChildNode(YAMLnode*, YAMLnode*);
int YAMLappendNode(YAMLnode*, YAMLnode**);
int YAMLprintNode(YAMLnode*, YAMLmode);
YAMLnode* YAMLfindNodeByKey(YAMLnode*, const char*);
YAMLnode* YAMLfindChildNodeByKey(YAMLnode*, const char*);
char* YAMLfindChildValueByKey(YAMLnode*, const char*);

int YAMLStringToIntArray(const char*, int*, int**);

/*-----------------------------------------------------------------------------
 * Public macros
 *-----------------------------------------------------------------------------*/

#define YAML_VALIDATE_NODE(_node, _callA, _callB) \
   { \
      StrArray _keys = _callA(); \
      if (StrArrayEntryExists(_keys, _node->key)) \
      { \
         StrIntMapArray _map_array = _callB(_node->key); \
         if (_map_array.size > 0) \
         { \
            if (StrIntMapArrayDomainEntryExists(_map_array, _node->val)) \
            { \
               _node->valid = YAML_NODE_VALID; \
            } \
            else \
            { \
               _node->valid = YAML_NODE_INVALID_VAL; \
            } \
         } \
         else \
         { \
            _node->valid = YAML_NODE_VALID; \
         } \
      } \
      else \
      { \
         _node->valid = YAML_NODE_INVALID_KEY; \
      } \
   }

#define YAML_CALL_IF_OPEN() if (0) {}
#define YAML_CALL_IF_CLOSE(_node) else {}
#define YAML_SET_IF_OPEN() if (0) {}
#define YAML_SET_IF_CLOSE(_node) else { ErrorMsgAddUnknownKey(_node->key); }
#define YAML_SET_IF_CLOSE_(_node) else { ErrorMsgAddInvalidKeyValPair(_node->key, _node->val); }
#define YAML_SET_INTEGER_IF_KEY_MATCHES(_var, _key, _node) \
   else if (!strcmp(_node->key, _key)) { _var = atoi(_node->val); }
#define YAML_SET_INTARRAY_IF_KEY_MATCHES(_count, _array, _key, _node) \
   else if (!strcmp(_node->key, _key)) { YAMLStringToIntArray(_node->val, _count, _array); }
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
   {char *val = YAMLfindChildValueByKey(_node, _key); if (val != NULL) { _var = atoi(val); }}
#define YAML_CALL_IF_KEY_MATCHES(_call, _args, _node, _key) \
   else if (!strcmp(_node->key, _key)) { _call(_args, _node); }
#define YAML_CALL_IF_VAL_MATCHES(_call, _args, _node, _val) \
   else if (!strcmp(_node->val, _val)) { _call(_args, _node); }
#define YAML_RETURN_IF_VAL_MATCHES(_node, _val) \
   if (!strcmp(_node->val, _val)) { return EXIT_SUCCESS; }
#define YAML_INC_INTEGER_IF_KEY_EXISTS(_var, _key, p) \
   { YAMLnode* c = p->children; while (c) { if (!strcmp(c->key, _key)) { _var++; } c = c->next; } }
#endif
