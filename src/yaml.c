/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "yaml.h"

/*-----------------------------------------------------------------------------
 * YAMLtreeCreate
 *-----------------------------------------------------------------------------*/

YAMLtree*
YAMLtreeCreate(void)
{
   YAMLtree *tree;

   tree = malloc(sizeof(YAMLtree));
   tree->root = YAMLnodeCreate("", "", -1);

   return tree;
}

/*-----------------------------------------------------------------------------
 * YAMLtreeDestroy
 *-----------------------------------------------------------------------------*/

void
YAMLtreeDestroy(YAMLtree** tree_ptr)
{
   YAMLtree *tree = *tree_ptr;

   if (tree)
   {
      YAMLnodeDestroy(tree->root);
      free(tree);
      *tree_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * YAMLtreeBuild
 *-----------------------------------------------------------------------------*/

void
YAMLtreeBuild(char *text, YAMLtree **tree_ptr)
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

   tree = YAMLtreeCreate();
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
      node = YAMLnodeCreate(key, val, level);

      /* Append entry to tree */
      YAMLnodeAppend(node, &parent);

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
 * YAMLtreeUpdate
 *
 * Update a YAML tree with information passed via command line
 *-----------------------------------------------------------------------------*/

void
YAMLtreeUpdate(int argc, char** argv, YAMLtree *tree)
{
   /* TODO */
   return;
}

/*-----------------------------------------------------------------------------
 * YAMLtreePrint
 *
 * Prints all nodes in a tree
 *-----------------------------------------------------------------------------*/

void
YAMLtreePrint(YAMLtree *tree, YAMLprintMode print_mode)
{
   YAMLnode *child;

   if (!tree)
   {
      ErrorCodeSet(ERROR_YAML_TREE_NULL);
      ErrorMsgAdd("Cannot print a void YAML tree!");
      return;
   }

   PRINT_DASHED_LINE(MAX_DIVISOR_LENGTH)
   child = tree->root->children;
   while (child != NULL)
   {
      YAMLnodePrint(child, print_mode);
      child = child->next;
   }
   PRINT_DASHED_LINE(MAX_DIVISOR_LENGTH)
}

/******************************************************************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
 * YAMLnodeCreate
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLnodeCreate(char *key, char* val, int level)
{
   YAMLnode *node;

   node             = (YAMLnode*) malloc(sizeof(YAMLnode));
   node->level      = level;
   node->key        = StrTrim(strdup(key));
   node->mapped_val = NULL;
   node->valid      = YAML_NODE_VALID; // We assume nodes are valid by default
   node->parent     = NULL;
   node->children   = NULL;
   node->next       = NULL;

   /* If the key contains "filename", "node->val" will be the same as "val".
      Otherwise, "node->val" will be set as "val" with all lowercase letters */
   if (strstr(key, "filename") != NULL)
   {
      node->val     = StrTrim(strdup(val));
   }
   else
   {
      node->val     = StrToLowerCase(StrTrim(strdup(val)));
   }

   return node;
}

/*-----------------------------------------------------------------------------
 * YAMLnodeDestroy
 *
 * Destroys a node via depth-first search (DFS)
 *-----------------------------------------------------------------------------*/

void
YAMLnodeDestroy(YAMLnode* node)
{
   YAMLnode  *child;
   YAMLnode  *next;

   if (node == NULL)
   {
      return;
   }

   child = node->children;
   while (child != NULL)
   {
      next = child->next;
      YAMLnodeDestroy(child);
      child = next;
   }
   free(node->key);
   free(node->val);
   free(node->mapped_val);
   free(node);
}

/*-----------------------------------------------------------------------------
 * YAMLnodeAddChild
 *
 * Adds a "child" node as the first child of the "parent" node.
 *-----------------------------------------------------------------------------*/

void
YAMLnodeAddChild(YAMLnode *parent, YAMLnode *child)
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
}

/*-----------------------------------------------------------------------------
 * YAMLnodeAppend
 *
 * Appends a node to the tree
 *-----------------------------------------------------------------------------*/

void
YAMLnodeAppend(YAMLnode *node, YAMLnode **previous_ptr)
{
   YAMLnode  *previous = *previous_ptr;
   int        previous_level = previous->level;

   if (node->level > previous_level)
   {
      /* Add child to current parent */
      YAMLnodeAddChild(previous, node);
   }
   else if (node->level == previous_level)
   {
      /* Add sibling to children's list and keep the current parent */
      YAMLnodeAddChild(previous->parent, node);
   }
   else
   {
      while (previous_level > node->level)
      {
         previous = previous->parent;
         previous_level--;
      }

      /* Add ancestor */
      YAMLnodeAddChild(previous->parent, node);
   }

   /* Update pointer to previous node */
   *previous_ptr = node;
}

/*-----------------------------------------------------------------------------
 * YAMLnodePrintHelper
 *-----------------------------------------------------------------------------*/

static inline void
YAMLnodePrintHelper(YAMLnode *node, const char *cKey, const char *cVal, const char *suffix)
{
   int offset = 2 * node->level + (int) strlen(node->key);

   printf("%s%*s%s: %s%s%s%s%s\n",
          cKey, offset, node->key, TEXT_RESET,
          cVal, node->val, TEXT_RESET, suffix, TEXT_RESET);
}

/*-----------------------------------------------------------------------------
 * YAMLnodePrint
 *-----------------------------------------------------------------------------*/

void
YAMLnodePrint(YAMLnode *node, YAMLprintMode print_mode)
{
   YAMLnode *child;

   if (node)
   {
      switch (print_mode)
      {
         case YAML_PRINT_MODE_ANY:
            if (node->valid == YAML_NODE_VALID)
            {
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_GREEN, "");
            }
            else if (node->valid == YAML_NODE_INVALID_INDENT)
            {
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX INDENTATION *");
               ErrorCodeSet(ERROR_YAML_INVALID_INDENT);
            }
            else if (node->valid == YAML_NODE_INVALID_DIVISOR)
            {
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX DIVISOR *");
               ErrorCodeSet(ERROR_YAML_INVALID_DIVISOR);
            }
            else if (node->valid == YAML_NODE_INVALID_KEY)
            {
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_YELLOWBOLD,
                                   TEXT_BOLD " <-- * FIX KEY *");
               ErrorCodeSet(ERROR_INVALID_KEY);
               ErrorCodeSet(ERROR_MAYBE_INVALID_VAL);
            }
            else if (node->valid == YAML_NODE_INVALID_VAL)
            {
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX VALUE *");
               ErrorCodeSet(ERROR_INVALID_VAL);
            }
            else if (node->valid == YAML_NODE_UNEXPECTED_VAL)
            {
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * FIX VALUE *");
               ErrorCodeSet(ERROR_UNEXPECTED_VAL);
            }
            break;

         case YAML_PRINT_MODE_ONLY_VALID:
            if (node->valid == YAML_NODE_VALID)
            {
               YAMLnodePrintHelper(node, "", "", "");
            }
            break;

         case YAML_PRINT_MODE_NO_CHECKING:
         default:
            YAMLnodePrintHelper(node, "", "", "");
            break;
      }
      child = node->children;

      while (child != NULL)
      {
         YAMLnodePrint(child, print_mode);
         child = child->next;
      }
   }
}

/*-----------------------------------------------------------------------------
 * YAMLnodeFindByKey
 *
 * Finds a node by key starting from the input "node" via DFS.
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLnodeFindByKey(YAMLnode* node, const char* key)
{
   YAMLnode *child;
   YAMLnode *found;

   if (node)
   {
      if (!strcmp(node->key, key))
      {
         return node;
      }

      child = node->children;
      while (child)
      {
         found = YAMLnodeFindByKey(child, key);
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
 * YAMLnodeFindChildByKey
 *
 * Finds a node by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

YAMLnode*
YAMLnodeFindChildByKey(YAMLnode* parent, const char* key)
{
   YAMLnode *child;

   if (parent)
   {
      child = parent->children;
      while (child)
      {
         if (!strcmp(child->key, key))
         {
            return child;
         }
         child = child->next;
      }
   }

   return NULL;
}

/*-----------------------------------------------------------------------------
 * YAMLnodeFindChildValueByKey
 *
 * Finds a value by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

char*
YAMLnodeFindChildValueByKey(YAMLnode* parent, const char* key)
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
