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

YAMLtree *
YAMLtreeCreate(int base_indent)
{
   YAMLtree *tree = NULL;

   tree               = malloc(sizeof(YAMLtree));
   tree->root         = YAMLnodeCreate("", "", -1);
   tree->base_indent  = base_indent;
   tree->is_validated = false; // Initialize validation flag

   return tree;
}

/*-----------------------------------------------------------------------------
 * YAMLtreeDestroy
 *-----------------------------------------------------------------------------*/

void
YAMLtreeDestroy(YAMLtree **tree_ptr)
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
 * YAMLtextRead
 *-----------------------------------------------------------------------------*/

void
YAMLtextRead(const char *dirname, const char *basename, int level, int *base_indent_ptr,
             size_t *length_ptr, char **text_ptr)
{
   FILE  *fp  = NULL;
   char  *key = NULL, *val = NULL, *sep = NULL;
   char   line[MAX_LINE_LENGTH];
   char   backup[MAX_LINE_LENGTH];
   char  *filename    = NULL;
   char  *new_text    = NULL;
   int    inner_level = 0, pos = 0;
   size_t num_whitespaces     = 2 * level;
   size_t new_length          = 0;
   int    base_indent         = *base_indent_ptr; // Track base indentation level
   int    prev_indent         = -1;               // Track previous indentation level
   bool   first_indented_line = true;

   /* Construct the whole filename */
   CombineFilename(dirname, basename, &filename);

   /* Open file */
   fp = fopen(filename, "r");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(filename);
      return;
   }
   free(filename);

   while (fgets(line, sizeof(line), fp))
   {
      /* Save the original line */
      strcpy(backup, line);

      /* Remove trailing newline character */
      line[strcspn(line, "\n")] = '\0';

      /* Remove comments at the end of valid line */
      char *comment_ptr = NULL;
      if ((comment_ptr = strchr(line, '#')) != NULL)
      {
         *comment_ptr = '\0';
         /* Also remove from backup to prevent comment from being included in output */
         comment_ptr = strchr(backup, '#');
         if (comment_ptr)
         {
            *comment_ptr       = '\n'; /* Preserve newline in backup */
            *(comment_ptr + 1) = '\0';
         }
      }

      /* Ignore empty lines and comments */
      if (line[0] == '\0' || line[0] == '#')
      {
         continue;
      }

      /* Check for divisor character */
      if ((sep = strchr(line, ':')) == NULL)
      {
         continue;
      }

      *sep = '\0';
      key  = line;
      val  = sep + 1;

      /* Trim leading spaces */
      while (*key == ' ')
      {
         key++;
      }
      while (*val == ' ')
      {
         val++;
      }

      /* Calculate indentation */
      pos = 0;
      while (line[pos] == ' ' || line[pos] == '\t')
      {
         if (line[pos] == '\t')
         {
            ErrorCodeSet(ERROR_YAML_MIXED_INDENT);
            ErrorMsgAdd("Tab characters are not allowed in YAML input!");
            fclose(fp);
            return;
         }
         pos++;
      }

      /* Skip lines with only whitespace */
      if (line[pos] == '\0')
      {
         continue;
      }

      /* Establish base indentation on first indented line */
      if (base_indent == -1 && pos > 0)
      {
         *base_indent_ptr = base_indent = pos;
         if (base_indent < 2)
         {
            ErrorCodeSet(ERROR_YAML_INVALID_BASE_INDENT);
            ErrorMsgAdd("Base indentation in YAML input must be at least 2 spaces");
            fclose(fp);
            return;
         }
      }

      /* Check indentation consistency */
      if (pos > 0)
      {
         if (base_indent > 0 && pos % base_indent != 0)
         {
            ErrorCodeSet(ERROR_YAML_INCONSISTENT_INDENT);
            ErrorMsgAdd("Inconsistent indentation detected in YAML input!");
            fclose(fp);
            return;
         }

         /* Check for indentation jumps */
         if (!first_indented_line && pos > prev_indent &&
             (pos - prev_indent) > base_indent)
         {
            ErrorCodeSet(ERROR_YAML_INVALID_INDENT_JUMP);
            ErrorMsgAdd("Invalid indentation jump detected in YAML input!");
            fclose(fp);
            return;
         }

         first_indented_line = false;
         prev_indent         = pos;
      }

      if (!strcmp(key, "include"))
      {
         /* Calculate node level based on actual indentation */
         inner_level = pos / (base_indent > 0 ? base_indent : 1);

         /* Recursively read the content of the included file */
         YAMLtextRead(dirname, val, inner_level, base_indent_ptr, length_ptr, text_ptr);
      }
      else
      {
         /* Use actual indentation spacing */
         num_whitespaces = (base_indent > 0 ? base_indent : 1) * level;

         /* Regular line, append it to the text */
         new_length = *length_ptr + strlen(backup) + num_whitespaces;
         new_text   = (char *)realloc(*text_ptr, new_length + 1);
         if (!new_text)
         {
            fclose(fp);
            return;
         }
         *text_ptr = new_text;

         /* Fill with base whitespaces */
         memset((*text_ptr) + (*length_ptr), ' ', num_whitespaces);

         /* Copy backup line */
         memcpy((*text_ptr) + (*length_ptr) + num_whitespaces, backup, strlen(backup));

         /* Set new null terminator */
         (*text_ptr)[new_length] = '\0';

         /* Update length pointer */
         *length_ptr = new_length;
      }
   }

   fclose(fp);
}

/*-----------------------------------------------------------------------------
 * YAMLtreeBuild
 *-----------------------------------------------------------------------------*/

void
YAMLtreeBuild(int base_indent, char *text, YAMLtree **tree_ptr)
{
   YAMLnode *node   = NULL;
   YAMLnode *parent = NULL;
   YAMLtree *tree   = NULL;

   char *remaining = NULL;
   char *line_ptr  = NULL;
   char *sep       = NULL;
   char *key = NULL, *val = NULL;
   char *line  = NULL;
   int   level = 0;
   int   count = 0, pos = 0, indent = 0;
   int   nlines        = 0;
   int   next          = 0;
   bool  divisor_is_ok = false;

   tree      = YAMLtreeCreate(base_indent);
   remaining = text;
   nlines    = 0;
   parent    = tree->root;
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
      line_ptr = line;
      next     = 1;
      while (*line_ptr)
      {
         if (*line_ptr != ' ' && *line_ptr != '\n' && *line_ptr != '\r')
         {
            next = 0;
            break;
         }
         line_ptr++;
      }
      if (next)
      {
         continue;
      }

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
            indent += (8 - (pos % 8));
            pos += (8 - (pos % 8));
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
      while (*key == ' ')
      {
         key++;
      }
      while (*val == ' ')
      {
         val++;
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
YAMLtreeUpdate(int argc, char **argv, YAMLtree *tree)
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
   YAMLnode *child = NULL;

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

/*-----------------------------------------------------------------------------
 * YAMLtreeValidate
 *
 * Validates all nodes in a tree and sets appropriate error codes
 *-----------------------------------------------------------------------------*/

void
YAMLtreeValidate(YAMLtree *tree)
{
   if (!tree)
   {
      ErrorCodeSet(ERROR_YAML_TREE_NULL);
      ErrorMsgAdd("Cannot validate a void YAML tree!");
      return;
   }

   YAMLnode *child = tree->root->children;
   while (child != NULL)
   {
      YAMLnodeValidate(child);
      child = child->next;
   }

   tree->is_validated = true; // Mark tree as validated
}

/******************************************************************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
 * YAMLnodeCreate
 *-----------------------------------------------------------------------------*/

YAMLnode *
YAMLnodeCreate(const char *key, const char *val, int level)
{
   YAMLnode *node = NULL;

   node             = (YAMLnode *)malloc(sizeof(YAMLnode));
   node->level      = level;
   node->key        = StrTrim(strdup((char *)key));
   node->mapped_val = NULL;
   node->valid      = YAML_NODE_UNKNOWN;
   node->parent     = NULL;
   node->children   = NULL;
   node->next       = NULL;

   /* If the key contains "name", "node->val" will be the same as "val".
      Otherwise, "node->val" will be set as "val" with all lowercase letters */
   if (strstr(key, "name"))
   {
      node->val = StrTrim(strdup((char *)val));
   }
   else
   {
      node->val = StrToLowerCase(StrTrim(strdup((char *)val)));
   }

   return node;
}

/*-----------------------------------------------------------------------------
 * YAMLnodeDestroy
 *
 * Destroys a node via depth-first search (DFS)
 *-----------------------------------------------------------------------------*/

void
YAMLnodeDestroy(YAMLnode *node)
{
   YAMLnode *child = NULL;
   YAMLnode *next  = NULL;

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
   YAMLnode *node = NULL;

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
   YAMLnode *previous       = *previous_ptr;
   int       previous_level = previous->level;

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
YAMLnodePrintHelper(const YAMLnode *node, const char *cKey, const char *cVal,
                    const char *suffix)
{
   int offset = (2 * node->level) + (int)strlen(node->key);

   printf("%s%*s%s: %s%s%s%s%s\n", cKey, offset, node->key, TEXT_RESET, cVal, node->val,
          TEXT_RESET, suffix, TEXT_RESET);
}

/*-----------------------------------------------------------------------------
 * YAMLnodeValidate
 *-----------------------------------------------------------------------------*/

void
YAMLnodeValidate(YAMLnode *node)
{
   if (!node)
   {
      return;
   }

   switch (node->valid)
   {
      case YAML_NODE_INVALID_INDENT:
         ErrorCodeSet(ERROR_YAML_INVALID_INDENT);
         break;

      case YAML_NODE_INVALID_DIVISOR:
         ErrorCodeSet(ERROR_YAML_INVALID_DIVISOR);
         break;

      case YAML_NODE_INVALID_KEY:
         ErrorCodeSet(ERROR_INVALID_KEY);
         ErrorCodeSet(ERROR_MAYBE_INVALID_VAL);
         break;

      case YAML_NODE_INVALID_VAL:
         ErrorCodeSet(ERROR_INVALID_VAL);
         break;

      case YAML_NODE_UNEXPECTED_VAL:
         ErrorCodeSet(ERROR_UNEXPECTED_VAL);
         break;

      default:
         break;
   }

   // Recursively validate children
   YAMLnode *child = node->children;
   while (child != NULL)
   {
      YAMLnodeValidate(child);
      child = child->next;
   }
}

/*-----------------------------------------------------------------------------
 * YAMLnodePrint
 *-----------------------------------------------------------------------------*/

void
YAMLnodePrint(YAMLnode *node, YAMLprintMode print_mode)
{
   if (!node)
   {
      return;
   }

   // Handle printing based on mode and validity
   switch (print_mode)
   {
      case YAML_PRINT_MODE_ANY:
         switch (node->valid)
         {
            case YAML_NODE_VALID:
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_GREEN, "");
               break;
            case YAML_NODE_INVALID_INDENT:
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * INVALID INDENTATION *");
               break;
            case YAML_NODE_INVALID_DIVISOR:
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * INVALID DIVISOR *");
               break;
            case YAML_NODE_INVALID_KEY:
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_YELLOWBOLD,
                                   TEXT_BOLD " <-- * INVALID KEY *");
               break;
            case YAML_NODE_INVALID_VAL:
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * INVALID VALUE *");
               break;
            case YAML_NODE_UNEXPECTED_VAL:
               YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                   TEXT_BOLD " <-- * UNEXPECTED VALUE *");
               break;
            default:
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_REDBOLD " <-- * INVALID ENTRY *");
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

   // Print children
   YAMLnode *child = node->children;
   while (child != NULL)
   {
      YAMLnodePrint(child, print_mode);
      child = child->next;
   }
}

/*-----------------------------------------------------------------------------
 * YAMLnodeFindByKey
 *
 * Finds a node by key starting from the input "node" via DFS.
 *-----------------------------------------------------------------------------*/

YAMLnode *
YAMLnodeFindByKey(YAMLnode *node, const char *key)
{
   YAMLnode *child = NULL;

   if (node)
   {
      if (!strcmp(node->key, key))
      {
         return node;
      }

      child = node->children;
      while (child)
      {
         YAMLnode *found = YAMLnodeFindByKey(child, key);
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

YAMLnode *
YAMLnodeFindChildByKey(YAMLnode *parent, const char *key)
{
   YAMLnode *child = NULL;

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

char *
YAMLnodeFindChildValueByKey(YAMLnode *parent, const char *key)
{
   YAMLnode *child = NULL;

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
