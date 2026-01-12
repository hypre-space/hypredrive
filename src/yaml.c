/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "yaml.h"

/*-----------------------------------------------------------------------------
 * Schema-driven validation helpers (shared by macro-generated parsers)
 *-----------------------------------------------------------------------------*/

static void
YAMLnodeValidateMap(YAMLnode *node, StrIntMapArray map_array)
{
   if (StrIntMapArrayDomainEntryExists(map_array, node->val))
   {
      int mapped = StrIntMapArrayGetImage(map_array, node->val);
      int length = snprintf(NULL, 0, "%d", mapped) + 1;

      if (!node->mapped_val)
      {
         node->mapped_val = (char *)malloc((size_t)length * sizeof(char));
      }
      else if (length > (int)strlen(node->mapped_val))
      {
         node->mapped_val =
            (char *)realloc(node->mapped_val, (size_t)length * sizeof(char));
      }

      snprintf(node->mapped_val, (size_t)length, "%d", mapped);
      node->valid = YAML_NODE_VALID;
   }
   else
   {
      node->valid = YAML_NODE_INVALID_VAL;
   }
}

void
YAMLnodeValidateSchema(YAMLnode *node, YAMLGetValidKeysFunc get_keys,
                       YAMLGetValidValuesFunc get_vals)
{
   if (!node)
   {
      return;
   }

   if ((node->valid == YAML_NODE_INVALID_DIVISOR) ||
       (node->valid == YAML_NODE_INVALID_INDENT))
   {
      return;
   }

   StrArray keys = get_keys();
   if (StrArrayEntryExists(keys, node->key))
   {
      StrIntMapArray map_array_key = get_vals(node->key);
      if (map_array_key.size > 0)
      {
         YAMLnodeValidateMap(node, map_array_key);
      }
      else
      {
         if (!node->mapped_val)
         {
            node->mapped_val = strdup(node->val);
         }
         node->valid = YAML_NODE_VALID;
      }
      return;
   }

   StrIntMapArray map_array_type = get_vals("type");
   if (!strcmp(node->key, "type") && map_array_type.size > 0)
   {
      YAMLnodeValidateMap(node, map_array_type);
      return;
   }

   node->valid = YAML_NODE_INVALID_KEY;
}

void
YAMLSetArgsGeneric(void *args, YAMLnode *parent, YAMLGetValidKeysFunc get_keys,
                   YAMLGetValidValuesFunc get_vals, YAMLSetFieldByNameFunc set_field)
{
   if (!parent)
   {
      return;
   }

   if (parent->children)
   {
      /* Case 1: Nested structure - iterate over children */
      for (YAMLnode *child = parent->children; child != NULL; child = child->next)
      {
         YAMLnodeValidateSchema(child, get_keys, get_vals);
         if (child->valid == YAML_NODE_VALID)
         {
            set_field(args, child);
         }
      }
   }
   else
   {
      /* Case 2: Flat value - treat val as "type" */
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      YAMLnodeValidateSchema(parent, get_keys, get_vals);
      if (parent->valid == YAML_NODE_VALID)
      {
         set_field(args, parent);
      }

      /* Restore original key */
      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

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
   size_t num_whitespaces     = 0;
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
      free(filename);
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
         num_whitespaces = (size_t)(base_indent > 0 ? base_indent : 1) * (size_t)level;

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
      divisor_is_ok = ((sep = strchr(line, ':')) != NULL);

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

/* Helpers for YAMLtreeUpdate */
static void
YAMLnodeDestroyChildren(YAMLnode *node)
{
   if (!node)
   {
      return;
   }

   YAMLnode *child = node->children;
   while (child)
   {
      YAMLnode *next = child->next;
      YAMLnodeDestroy(child);
      child = next;
   }
   node->children = NULL;
}

static void
YAMLnodeClearMappedVal(YAMLnode *node)
{
   if (node && node->mapped_val)
   {
      free(node->mapped_val);
      node->mapped_val = NULL;
   }
}

static void
YAMLnodeSetScalarValue(YAMLnode *node, const char *val)
{
   if (!node)
   {
      return;
   }

   YAMLnodeClearMappedVal(node);

   free(node->val);
   node->val = StrTrim(strdup(val ? val : ""));
   if (!strstr(node->key, "name"))
   {
      StrToLowerCase(node->val);
   }
   node->valid = YAML_NODE_UNKNOWN;
}

static void
YAMLnodeEnsureMapping(YAMLnode *node)
{
   if (!node)
   {
      return;
   }

   YAMLnodeClearMappedVal(node);

   if (strcmp(node->val, "") != 0)
   {
      free(node->val);
      node->val = strdup("");
   }
   node->valid = YAML_NODE_UNKNOWN;
}

static YAMLnode *
YAMLnodeGetOrCreateChild(YAMLnode *parent, const char *key)
{
   YAMLnode *child = YAMLnodeFindChildByKey(parent, key);
   if (!child)
   {
      child = YAMLnodeCreate(key, "", parent->level + 1);
      YAMLnodeAddChild(parent, child);
   }
   return child;
}

static int
YAMLargsFindFlagIndex(int argc, char **argv, const char *short_flag,
                      const char *long_flag)
{
   for (int i = 0; i < argc; i++)
   {
      if ((short_flag && !strcmp(argv[i], short_flag)) ||
          (long_flag && !strcmp(argv[i], long_flag)))
      {
         return i;
      }
   }
   return -1;
}

static int
YAMLargsFindConfigFileIndex(int argc, char **argv)
{
   for (int i = 0; i < argc; i++)
   {
      if (argv[i] && IsYAMLFilename(argv[i]))
      {
         return i;
      }
   }
   return -1;
}

void
YAMLtreeUpdate(int argc, char **argv, YAMLtree *tree)
{
   if (!tree || !tree->root)
   {
      ErrorCodeSet(ERROR_YAML_TREE_NULL);
      ErrorMsgAdd("Cannot update a void YAML tree!");
      return;
   }

   /* Support two calling conventions:
    *
    * 1) Legacy "pair list" mode (library / internal calls):
    *      argv = ["--path:to:key", "val", ...]
    *
    * 2) Driver "full argv" mode:
    *      argv contains many tokens including -q, config file, etc.
    *      Overrides are only parsed after -a/--args and stop at the YAML filename
    *      if it appears after -a (some test harnesses append it at the end).
    */
   int  args_flag_idx  = YAMLargsFindFlagIndex(argc, argv, "-a", "--args");
   int  cfg_idx        = YAMLargsFindConfigFileIndex(argc, argv);
   int  override_start = 0;
   int  override_end   = argc;
   bool full_argv_mode = (args_flag_idx >= 0);

   if (full_argv_mode)
   {
      override_start = args_flag_idx + 1;
      if (cfg_idx >= 0 && cfg_idx >= override_start)
      {
         override_end = cfg_idx;
      }
      if (override_start >= override_end)
      {
         return; /* nothing to do */
      }
   }
   else
   {
      /* No -a provided:
       * - If argv looks like a pure override pair list, parse it.
       * - Otherwise assume caller passed a full argv without overrides -> no-op. */
      bool pair_list_mode = false;
      if (argc > 0 && (argc % 2) == 0)
      {
         bool looks_like_pairs = true;
         for (int i = 0; i < argc; i += 2)
         {
            const char *k = argv[i];
            const char *v = argv[i + 1];
            if (!k || !v || strncmp(k, "--", 2) != 0)
            {
               looks_like_pairs = false;
               break;
            }
         }
         pair_list_mode = looks_like_pairs;
      }

      if (!pair_list_mode)
      {
         return;
      }
   }

   int n_override_tokens = override_end - override_start;

   /* In full-argv mode, enforce pairs after -a. In pair-list mode, enforce pairs too. */
   if ((n_override_tokens % 2) != 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid CLI overrides: expected pairs of '--path:to:key value'");
      return;
   }

   for (int i = override_start; i < override_end; i += 2)
   {
      const char *k = argv[i];
      if (!k)
      {
         continue;
      }

      if (strncmp(k, "--", 2) != 0)
      {
         /* In full argv mode, ignore non override tokens (we only parse pairs after -a
          * anyway), but keep strictness in pair-list mode. */
         if (!full_argv_mode)
         {
            ErrorCodeSet(ERROR_INVALID_KEY);
            ErrorMsgAdd("Invalid override key '%s' (expected --path:to:key)", k);
            return;
         }
         continue;
      }

      if (i + 1 >= argc)
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Missing value for override '%s'", k);
         return;
      }

      const char *v = argv[i + 1];
      if (!v || (strncmp(v, "--", 2) == 0))
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Missing value for override '%s'", k);
         return;
      }

      /* Apply override */
      {
         char *path = strdup(k);
         char *p    = path;

         if (!strncmp(p, "--", 2))
         {
            p += 2;
         }

         YAMLnode *cur  = tree->root;
         char     *save = NULL;
         char     *seg  = strtok_r(p, ":", &save);

         if (!seg || *seg == '\0')
         {
            ErrorCodeSet(ERROR_INVALID_KEY);
            ErrorMsgAdd("Invalid override path: '%s'", k);
            free(path);
            return;
         }

         while (seg)
         {
            const char *seg_const = seg; /* Use const pointer for read-only access */
            char       *next_seg  = strtok_r(NULL, ":", &save);
            if (next_seg) /* intermediate */
            {
               YAMLnode *child = YAMLnodeGetOrCreateChild(cur, seg_const);
               YAMLnodeEnsureMapping(child);
               cur = child;
               seg = next_seg;
            }
            else /* leaf */
            {
               YAMLnode *leaf = YAMLnodeFindChildByKey(cur, seg_const);
               if (!leaf)
               {
                  leaf = YAMLnodeCreate(seg_const, v, cur->level + 1);
                  YAMLnodeAddChild(cur, leaf);
               }
               else
               {
                  YAMLnodeDestroyChildren(leaf);
                  YAMLnodeSetScalarValue(leaf, v);
               }
               break;
            }
         }

         free(path);
      }
   }
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
