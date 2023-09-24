/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "yaml.h"

/*-----------------------------------------------------------------------------
 * YAMLcreateTree
 *-----------------------------------------------------------------------------*/

YAMLtree*
YAMLcreateTree(void)
{
   YAMLtree *tree;

   tree = malloc(sizeof(YAMLtree));
   tree->root = YAMLcreateNode("", "", -1);

   return tree;
}

/*-----------------------------------------------------------------------------
 * YAMLdestroyTree
 *-----------------------------------------------------------------------------*/

void
YAMLdestroyTree(YAMLtree** tree_ptr)
{
   YAMLtree *tree = *tree_ptr;

   if (tree)
   {
      YAMLdestroyNode(tree->root);
      free(tree);
      *tree_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * YAMLbuildTree
 *-----------------------------------------------------------------------------*/

void
YAMLbuildTree(char *text, YAMLtree **tree_ptr)
{
   YAMLnode   *node;
   YAMLnode   *parent;
   YAMLtree   *tree;

   char       *remaining;
   char       *line_ptr;
   char       *sep;
   char       *key, *val;
   char       *line;
   int         level;
   int         count, pos, indent;
   int         nlines;
   int         next;
   bool        divisor_is_ok;

   tree = YAMLcreateTree();
   remaining = text;
   nlines = 0; parent = tree->root;
   while ((line = strtok_r(remaining, "\n", &remaining)))
   {
      /* Increase line counter */
      nlines++;

      /* Remove trailing newline character */
      line[strcspn(line, "\n")] = '\0';

      /* Ignore empty lines and comments */
      if (line[0] == '\0' || line[0] == '#')
      {
         continue;
      }

      /* Search for comment at the end of the line and remove it */
      if ((line_ptr = strchr(line, '#')) != NULL)
      {
        *line_ptr = '\0';
      }

      /* Skip line if it contains blank spaces only */
      line_ptr = line; next = 1;
      while (*line_ptr)
      {
         if (*line_ptr != ' '  &&
             *line_ptr != '\n' &&
             *line_ptr != '\r')
         {
            next = 0;
            break;
         }
         line_ptr++;
      }
      if (next) continue;

      /* Compute indendation */
      pos = count = indent = 0;
      while (line[count] == ' ' || line[indent] == '\t')
      {
         if (line[count] == ' ')
         {
            indent++;
            pos++;
         }
         else
         {
            indent += (8 - pos % 8);
            pos += (8 - pos % 8);
         }
         count++;
      }

      /* Calculate node level */
      level = indent / 2;

      /* Check for divisor character */
      divisor_is_ok = ((sep = strchr(line, ':')) == NULL) ? false : true;

      /* Extract (key, val) pair */
      if (divisor_is_ok)
      {
         *sep = '\0';
         key  = line + indent;
         val  = sep + 1;
      }
      else
      {
         key = line + indent;
         val = line + strlen(line);
      }

      /* Trim leading spaces */
      while (*key == ' ') key++;
      while (*val == ' ') val++;

      /* Skip if found include keys */
      if (!strcmp(key, "preconditioner_include") ||
          !strcmp(key, "solver_include"))
      {
         continue;
      }

      /* Create node entry */
      node = YAMLcreateNode(key, val, level);

      /* Append entry to tree */
      YAMLappendNode(node, &parent);

      /* Set error code if indentation is incorrect */
      if (indent % 2 != 0)
      {
         YAML_NODE_SET_INVALID_INDENT(node);
      }

      /* Set error code if divisor character is incorrect */
      if (!divisor_is_ok)
      {
         YAML_NODE_SET_INVALID_DIVISOR(node);
      }
   }

   *tree_ptr = tree;
}

/*-----------------------------------------------------------------------------
 * YAMLprintTree
 *
 * Prints all nodes in a tree
 *-----------------------------------------------------------------------------*/

void
YAMLprintTree(YAMLtree *tree, YAMLmode validity)
{
   YAMLnode *child;
   int       i, divisor = 80;

   if (!tree)
   {
      ErrorCodeSet(ERROR_YAML_TREE_NULL);
      ErrorMsgAdd("Cannot print a void YAML tree!");
      return;
   }

   for (i = 0; i < divisor; i++) { printf("-"); } printf("\n");
   child = tree->root->children;
   while (child != NULL)
   {
      YAMLprintNode(child, validity);
      child = child->next;
   }
   for (i = 0; i < divisor; i++) { printf("-"); } printf("\n");
}

/******************************************************************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
 * YAMLcreateNode
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLcreateNode(char *key, char* val, int level)
{
   YAMLnode *node;

   node           = (YAMLnode*) malloc(sizeof(YAMLnode));
   node->level    = level;
   node->key      = strdup(key); // TODO: transform to lower case
   node->val      = strdup(val); // TODO: transform to lower case
   node->valid    = YAML_NODE_VALID; // We assume nodes are valid by default
   node->parent   = NULL;
   node->children = NULL;
   node->next     = NULL;

   return node;
}

/*-----------------------------------------------------------------------------
 * YAMLdestroyNode
 *
 * Destroys a node via depth-first search (DFS)
 *-----------------------------------------------------------------------------*/

int
YAMLdestroyNode(YAMLnode* node)
{
   YAMLnode  *child;
   YAMLnode  *next;

   if (node == NULL)
   {
      return EXIT_SUCCESS;
   }

   child = node->children;
   while (child != NULL)
   {
      next = child->next;
      if (YAMLdestroyNode(child) != EXIT_SUCCESS)
      {
         return EXIT_FAILURE;
      }
      child = next;
   }
   free(node->key);
   free(node->val);
   free(node);

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * YAMLaddChildNode
 *
 * Adds a "child" node as the first child of the "parent" node.
 *-----------------------------------------------------------------------------*/

int
YAMLaddChildNode(YAMLnode *parent, YAMLnode *child)
{
   YAMLnode *node;

   child->parent = parent;
   if (parent->children == NULL)
   {
      parent->children = child;
   }
   else
   {
      node = parent->children;
      while (node->next != NULL)
      {
         node = node->next;
      }
      node->next = child;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * YAMLappendNode
 *
 * Appends a node to the tree
 *-----------------------------------------------------------------------------*/

int
YAMLappendNode(YAMLnode *node, YAMLnode **previous_ptr)
{
   YAMLnode  *previous = *previous_ptr;
   int        previous_level = previous->level;

   if (node->level > previous_level)
   {
      /* Add child to current parent */
      YAMLaddChildNode(previous, node);
   }
   else if (node->level == previous_level)
   {
      /* Add sibling to children's list and keep the current parent */
      YAMLaddChildNode(previous->parent, node);
   }
   else
   {
      while (previous_level > node->level)
      {
         previous = previous->parent;
         previous_level--;
      }

      /* Add ancestor */
      YAMLaddChildNode(previous->parent, node);
   }

   /* Update pointer to previous node */
   *previous_ptr = node;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * YAMLprintNodeHelper
 *-----------------------------------------------------------------------------*/

static inline void
YAMLprintNodeHelper(YAMLnode *node, const char *cKey, const char *cVal, const char *suffix)
{
   int offset = 2 * node->level + (int) strlen(node->key);

   printf("%s%*s%s: %s%s%s%s%s\n",
          cKey, offset, node->key, TEXT_RESET,
          cVal, node->val, TEXT_RESET, suffix, TEXT_RESET);
}

/*-----------------------------------------------------------------------------
 * YAMLprintNode
 *-----------------------------------------------------------------------------*/

int
YAMLprintNode(YAMLnode *node, YAMLmode mode)
{
   YAMLnode *child;

   if (node)
   {
      switch (mode)
      {
         case YAML_MODE_ANY:
            if (node->valid == YAML_NODE_VALID)
            {
               YAMLprintNodeHelper(node, TEXT_GREEN, TEXT_GREEN, "");
            }
            else if (node->valid == YAML_NODE_INVALID_INDENT)
            {
               YAMLprintNodeHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX INDENTATION *");
               ErrorCodeSet(ERROR_YAML_INVALID_INDENT);
            }
            else if (node->valid == YAML_NODE_INVALID_DIVISOR)
            {
               YAMLprintNodeHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX DIVISOR *");
               ErrorCodeSet(ERROR_YAML_INVALID_DIVISOR);
            }
            else if (node->valid == YAML_NODE_INVALID_KEY)
            {
               YAMLprintNodeHelper(node, TEXT_REDBOLD, TEXT_YELLOWBOLD,
                                   TEXT_BOLD " <-- * FIX KEY *");
               ErrorCodeSet(ERROR_INVALID_KEY);
               ErrorCodeSet(ERROR_MAYBE_INVALID_VAL);
            }
            else if (node->valid == YAML_NODE_INVALID_VAL)
            {
               YAMLprintNodeHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX VALUE *");
               ErrorCodeSet(ERROR_INVALID_VAL);
            }
            break;

         case YAML_MODE_ONLY_VALID:
            if (node->valid == YAML_NODE_VALID)
            {
               YAMLprintNodeHelper(node, "", "", "");
            }
            break;

         case YAML_MODE_NONE:
         default:
            YAMLprintNodeHelper(node, "", "", "");
            break;
      }
      child = node->children;

      while (child != NULL)
      {
         YAMLprintNode(child, mode);
         child = child->next;
      }
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * YAMLfindNodeByKey
 *
 * Finds a node by key starting from the input "node" via DFS.
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLfindNodeByKey(YAMLnode* node, const char* key)
{
   YAMLnode *child;
   YAMLnode *found;

   if (node)
   {
      if (strcmp(node->key, key) == 0)
      {
         return node;
      }

      child = node->children;
      while (child)
      {
         found = YAMLfindNodeByKey(child, key);
         if (found)
         {
            return found;
         }
         child = child->next;
      }
   }

   return NULL;
}

/*-----------------------------------------------------------------------------
 * YAMLfindChildNodeByKey
 *
 * Finds a node by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLfindChildNodeByKey(YAMLnode* parent, const char* key)
{
   YAMLnode *child;

   if (parent)
   {
      child = parent->children;
      while (child)
      {
         if (strcmp(child->key, key) == 0)
         {
            return child;
         }
         child = child->next;
      }
   }

   return NULL;
}

/*-----------------------------------------------------------------------------
 * YAMLfindChildValueByKey
 *
 * Finds a value by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

char*
YAMLfindChildValueByKey(YAMLnode* parent, const char* key)
{
   YAMLnode *child;

   if (parent)
   {
      child = parent->children;
      while (child)
      {
         if (!strcmp(child->key, key))
         {
            return child->val;
         }
         child = child->next;
      }
   }

   return NULL;
}

/*-----------------------------------------------------------------------------
 * YAMLStringToIntArray
 *-----------------------------------------------------------------------------*/

int
YAMLStringToIntArray(const char* string, int *count_ptr, int **array_ptr)
{
   char   *buffer;
   char   *token;
   int    *array;
   int     count;

   /* Find number of elements in array */
   buffer = strdup(string);
   token  = strtok(buffer, "[], ");
   count  = 0;
   while (token)
   {
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);
   *count_ptr = count;

   /* Build array */
   buffer = strdup(string);
   array  = (int*) malloc(count*sizeof(int));
   token  = strtok(buffer, "[], ");
   count  = 0;
   while (token)
   {
      array[count] = atoi(token);
      count++;
      token = strtok(NULL, "[], ");
   }
   free(buffer);
   *array_ptr = array;

   return EXIT_SUCCESS;
}
