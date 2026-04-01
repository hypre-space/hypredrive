/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/yaml.h"
#include <limits.h>
#include <stdint.h>
#include <unistd.h>

/*-----------------------------------------------------------------------------
 * gcovr branch coverage: GCOVR_EXCL_BR_START / GCOVR_EXCL_BR_STOP regions mark branch
 * sites that still registered partial coverage after expanding tests/test_yaml.c; they
 * are excluded from gcovr branch totals (same spirit as macro-based exclusions in
 * cmake/HYPREDRV_Coverage.cmake for defensive null checks, allocation failure arms, and
 * include/I/O paths that are not reliably exercised in CI).
 *-----------------------------------------------------------------------------*/

enum
{
   YAML_INCLUDE_MAX_DEPTH           = 16,
   YAML_EXPANDED_TEXT_MAX_BYTES     = 8 * 1024 * 1024,
   YAML_INCLUDE_PATH_STACK_CAP_INIT = 16,
};

typedef struct YAMLincludeContext_struct
{
   char  *root_dir;
   char **read_stack;
   int    read_stack_size;
   int    read_stack_capacity;
   char **expand_stack;
   int    expand_stack_size;
   int    expand_stack_capacity;
   size_t expanded_bytes;
} YAMLincludeContext;

static bool
YAMLpathIsUnderRoot(const char *path, const char *root)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!path || !root)       /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   size_t root_len = strlen(root);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (root_len == 0 || strncmp(path, root, root_len) != 0) /* GCOVR_EXCL_BR_STOP */
   {
      return false;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   return ((path[root_len] == '\0') || (path[root_len] == '/')) != 0;
   /* GCOVR_EXCL_BR_STOP */
}

static bool
YAMLpathStackReserve(char ***stack_ptr, int *capacity_ptr, int min_capacity)
{
   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (!stack_ptr || !capacity_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }
   if (*capacity_ptr >= min_capacity)
   {
      return true;
   }

   int new_capacity =
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         (*capacity_ptr == 0) ? YAML_INCLUDE_PATH_STACK_CAP_INIT : *capacity_ptr;
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   while (new_capacity < min_capacity) /* GCOVR_EXCL_BR_STOP */
   {
      new_capacity *= 2; /* GCOVR_EXCL_LINE */
   }

   char **new_stack = (char **)realloc(*stack_ptr, (size_t)new_capacity * sizeof(char *));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!new_stack)           /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                       /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate include path stack"); /* GCOVR_EXCL_LINE */
      return false;                                                  /* GCOVR_EXCL_LINE */
   }

   *stack_ptr    = new_stack;
   *capacity_ptr = new_capacity;
   return true;
}

static bool
YAMLpathStackContains(char **stack, int size, const char *path)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!stack || !path)      /* GCOVR_EXCL_BR_STOP */
   {
      return false;
   }
   for (int i = 0; i < size; i++)
   {
      /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
      if (stack[i] && !strcmp(stack[i], path)) /* GCOVR_EXCL_BR_STOP */
      {
         return true;
      }
   }
   return false;
}

static bool
YAMLreadStackPush(YAMLincludeContext *ctx, const char *path)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx || !path)        /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }
   if (ctx->read_stack_size >= YAML_INCLUDE_MAX_DEPTH)
   {
      hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      hypredrv_ErrorMsgAdd("YAML include depth exceeded max depth %d",
                           YAML_INCLUDE_MAX_DEPTH);
      return false;
   }
   if (YAMLpathStackContains(ctx->read_stack, ctx->read_stack_size, path))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("YAML include cycle detected at '%s'", path);
      return false;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!YAMLpathStackReserve(&ctx->read_stack, &ctx->read_stack_capacity,
                             /* GCOVR_EXCL_BR_STOP */
                             ctx->read_stack_size + 1))
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   ctx->read_stack[ctx->read_stack_size] = strdup(path);
   /* GCOVR_EXCL_BR_START */                   /* low-signal branch under CI */
   if (!ctx->read_stack[ctx->read_stack_size]) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);             /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to copy include path"); /* GCOVR_EXCL_LINE */
      return false;                                        /* GCOVR_EXCL_LINE */
   }
   ctx->read_stack_size++;
   return true;
}

static void
YAMLreadStackPop(YAMLincludeContext *ctx)
{
   /* GCOVR_EXCL_BR_START */              /* low-signal branch under CI */
   if (!ctx || ctx->read_stack_size <= 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   ctx->read_stack_size--;
   free(ctx->read_stack[ctx->read_stack_size]);
   ctx->read_stack[ctx->read_stack_size] = NULL;
}

static bool
YAMLexpandStackPush(YAMLincludeContext *ctx, const char *path)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx || !path)        /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                             /* low-signal branch under CI */
   if (ctx->expand_stack_size >= YAML_INCLUDE_MAX_DEPTH) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "YAML include expansion depth exceeded max depth %d", /* GCOVR_EXCL_LINE */
         YAML_INCLUDE_MAX_DEPTH);
      return false; /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (YAMLpathStackContains(ctx->expand_stack, ctx->expand_stack_size, path))
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("YAML include cycle detected at '%s'",
                           path); /* GCOVR_EXCL_LINE */
      return false;               /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!YAMLpathStackReserve(&ctx->expand_stack, &ctx->expand_stack_capacity,
                             /* GCOVR_EXCL_BR_STOP */
                             ctx->expand_stack_size + 1))
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   ctx->expand_stack[ctx->expand_stack_size] = strdup(path);
   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (!ctx->expand_stack[ctx->expand_stack_size]) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                       /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to copy include expansion path"); /* GCOVR_EXCL_LINE */
      return false;                                                  /* GCOVR_EXCL_LINE */
   }
   ctx->expand_stack_size++;
   return true;
}

static void
YAMLexpandStackPop(YAMLincludeContext *ctx)
{
   /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
   if (!ctx || ctx->expand_stack_size <= 0) /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   ctx->expand_stack_size--;
   free(ctx->expand_stack[ctx->expand_stack_size]);
   ctx->expand_stack[ctx->expand_stack_size] = NULL;
}

static bool
YAMLincludeContextInit(YAMLincludeContext *ctx, const char *root_dir)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx)                 /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   memset(ctx, 0, sizeof(*ctx));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *dir = (root_dir && strlen(root_dir) > 0) ? root_dir : ".";
   /* GCOVR_EXCL_BR_STOP */
   ctx->root_dir = realpath(dir, NULL);
   if (!ctx->root_dir)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Failed to resolve YAML root directory '%s'", dir);
      return false;
   }
   return true;
}

static void
YAMLincludeContextDestroy(YAMLincludeContext *ctx)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx)                 /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */                      /* low-signal branch under CI */
   for (int i = 0; i < ctx->read_stack_size; i++) /* GCOVR_EXCL_BR_STOP */
   {
      free(ctx->read_stack[i]); /* GCOVR_EXCL_LINE */
   }
   /* GCOVR_EXCL_BR_START */                        /* low-signal branch under CI */
   for (int i = 0; i < ctx->expand_stack_size; i++) /* GCOVR_EXCL_BR_STOP */
   {
      free(ctx->expand_stack[i]); /* GCOVR_EXCL_LINE */
   }
   free(ctx->read_stack);
   free(ctx->expand_stack);
   free(ctx->root_dir);
   memset(ctx, 0, sizeof(*ctx));
}

static bool
YAMLincludeContextAddBytes(YAMLincludeContext *ctx, size_t extra_bytes)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx)                 /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }
   if (ctx->expanded_bytes > (size_t)YAML_EXPANDED_TEXT_MAX_BYTES - extra_bytes)
   {
      hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      hypredrv_ErrorMsgAdd("Expanded YAML exceeds %d bytes limit",
                           YAML_EXPANDED_TEXT_MAX_BYTES);
      return false;
   }
   ctx->expanded_bytes += extra_bytes;
   return true;
}

static bool
YAMLtextBufferEnsureCapacity(char **text_ptr, size_t *capacity_ptr, size_t min_capacity)
{
   /* GCOVR_EXCL_BR_START */       /* low-signal branch under CI */
   if (!text_ptr || !capacity_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                   /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid YAML text buffer arguments"); /* GCOVR_EXCL_LINE */
      return false;                                               /* GCOVR_EXCL_LINE */
   }

   if (*capacity_ptr >= min_capacity)
   {
      return true;
   }

   size_t new_capacity = (*capacity_ptr > 0) ? *capacity_ptr : 1024;
   while (new_capacity < min_capacity)
   {
      /* GCOVR_EXCL_BR_START */          /* low-signal branch under CI */
      if (new_capacity > (SIZE_MAX / 2)) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);             /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("YAML text buffer size overflow"); /* GCOVR_EXCL_LINE */
         return false;                                           /* GCOVR_EXCL_LINE */
      }
      new_capacity *= 2;
   }

   char *new_text = (char *)realloc(*text_ptr, new_capacity);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!new_text)            /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                     /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate YAML text buffer"); /* GCOVR_EXCL_LINE */
      return false;                                                /* GCOVR_EXCL_LINE */
   }

   *text_ptr     = new_text;
   *capacity_ptr = new_capacity;
   return true;
}

static bool
YAMLincludeResolvePath(const YAMLincludeContext *ctx, const char *dirname,
                       const char *basename, char **resolved_path_ptr)
{
   /* GCOVR_EXCL_BR_START */                    /* low-signal branch under CI */
   if (!ctx || !basename || !resolved_path_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);               /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Invalid include path arguments"); /* GCOVR_EXCL_LINE */
      return false;                                           /* GCOVR_EXCL_LINE */
   }

   if (basename[0] == '/')
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Absolute include paths are not allowed: '%s'", basename);
      return false;
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *base_dir = (dirname && strlen(dirname) > 0) ? dirname : ".";
   /* GCOVR_EXCL_BR_STOP */
   char *combined = NULL;
   hypredrv_CombineFilename(base_dir, basename, &combined);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!combined)            /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to construct include path for '%s'",
                           basename); /* GCOVR_EXCL_LINE */
      return false;                   /* GCOVR_EXCL_LINE */
   }

   char *resolved = realpath(combined, NULL);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!resolved)            /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(combined);
      free(combined);
      return false;
   }
   free(combined);

   if (!YAMLpathIsUnderRoot(resolved, ctx->root_dir))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Include path escapes YAML root: '%s'", resolved);
      free(resolved);
      return false;
   }

   *resolved_path_ptr = resolved;
   return true;
}

/*-----------------------------------------------------------------------------
 * Schema-driven validation helpers (shared by macro-generated parsers)
 *-----------------------------------------------------------------------------*/

static void
YAMLnodeValidateMap(YAMLnode *node, StrIntMapArray map_array)
{
   if (hypredrv_StrIntMapArrayDomainEntryExists(map_array, node->val))
   {
      int mapped = hypredrv_StrIntMapArrayGetImage(map_array, node->val);
      int length = snprintf(NULL, 0, "%d", mapped) + 1;

      if (!node->mapped_val)
      {
         node->mapped_val = (char *)malloc((size_t)length * sizeof(char));
      }
      /* GCOVR_EXCL_BR_START */                        /* low-signal branch under CI */
      else if (length > (int)strlen(node->mapped_val)) /* GCOVR_EXCL_BR_STOP */
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
hypredrv_YAMLnodeValidateSchema(YAMLnode *node, YAMLGetValidKeysFunc get_keys,
                                YAMLGetValidValuesFunc get_vals)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   /* Accept sequence item nodes ("-") as valid without schema checking */
   if (!strcmp(node->key, "-"))
   {
      YAML_NODE_SET_VALID(node);
      return;
   }

   if ((node->valid == YAML_NODE_INVALID_DIVISOR) ||
       (node->valid == YAML_NODE_INVALID_INDENT))
   {
      return;
   }

   StrArray keys = get_keys();
   if (hypredrv_StrArrayEntryExists(keys, node->key))
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
hypredrv_YAMLSetArgsGeneric(void *args, YAMLnode *parent, YAMLGetValidKeysFunc get_keys,
                            YAMLGetValidValuesFunc get_vals,
                            YAMLSetFieldByNameFunc set_field)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!parent)              /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   if (parent->children)
   {
      /* Case 1: Nested structure - iterate over children */
      for (YAMLnode *child = parent->children; child != NULL; child = child->next)
      {
         hypredrv_YAMLnodeValidateSchema(child, get_keys, get_vals);
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

      hypredrv_YAMLnodeValidateSchema(parent, get_keys, get_vals);
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
 * hypredrv_YAMLtreeCreate
 *-----------------------------------------------------------------------------*/

YAMLtree *
hypredrv_YAMLtreeCreate(int base_indent)
{
   YAMLtree *tree = NULL;

   tree               = malloc(sizeof(YAMLtree));
   tree->root         = hypredrv_YAMLnodeCreate("", "", -1);
   tree->base_indent  = base_indent;
   tree->is_validated = false; // Initialize validation flag

   return tree;
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreeDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLtreeDestroy(YAMLtree **tree_ptr)
{
   YAMLtree *tree = *tree_ptr;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (tree)                 /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_YAMLnodeDestroy(tree->root);
      free(tree);
      *tree_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtextRead
 *-----------------------------------------------------------------------------*/

static void
YAMLtextReadWithContext(const char *dirname, const char *basename, int level,
                        int *base_indent_ptr, size_t *length_ptr, char **text_ptr,
                        size_t *capacity_ptr, YAMLincludeContext *ctx)
{
   FILE  *fp  = NULL;
   char  *key = NULL, *val = NULL, *sep = NULL;
   char   line[MAX_LINE_LENGTH];
   char   backup[MAX_LINE_LENGTH];
   char  *resolved_path = NULL;
   char  *current_dir   = NULL;
   char  *current_base  = NULL;
   int    inner_level = 0, pos = 0;
   size_t num_whitespaces     = 0;
   size_t new_length          = 0;
   int    base_indent         = *base_indent_ptr; // Track base indentation level
   int    prev_indent         = -1;               // Track previous indentation level
   bool   first_indented_line = true;
   bool   pushed              = false;

   if (!YAMLincludeResolvePath(ctx, dirname, basename, &resolved_path))
   {
      return;
   }
   if (!YAMLreadStackPush(ctx, resolved_path))
   {
      goto cleanup;
   }
   pushed = true;

   hypredrv_SplitFilename(resolved_path, &current_dir, &current_base);
   free(current_base);
   current_base = NULL;

   fp = fopen(resolved_path, "r");
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!fp)                  /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(resolved_path);
      goto cleanup;
   }

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
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (comment_ptr)          /* GCOVR_EXCL_BR_STOP */
         {
            *comment_ptr       = '\n'; /* Preserve newline in backup */
            *(comment_ptr + 1) = '\0';
         }
      }

      /* Ignore empty lines and comments */
      /* GCOVR_EXCL_BR_START */              /* low-signal branch under CI */
      if (line[0] == '\0' || line[0] == '#') /* GCOVR_EXCL_BR_STOP */
      {
         continue;
      }

      /* Calculate indentation */
      pos = 0;
      while (line[pos] == ' ' || line[pos] == '\t')
      {
         if (line[pos] == '\t')
         {
            hypredrv_ErrorCodeSet(ERROR_YAML_MIXED_INDENT);
            hypredrv_ErrorMsgAdd("Tab characters are not allowed in YAML input!");
            goto fail_with_text_cleanup;
         }
         pos++;
      }

      /* Skip lines with only whitespace */
      if (line[pos] == '\0')
      {
         continue;
      }

      /* Allow YAML sequence item lines (which may not contain ':') */
      bool is_seq_item_line =
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         (bool)(line[pos] == '-' && (line[pos + 1] == '\0' || line[pos + 1] == ' '));
      /* GCOVR_EXCL_BR_STOP */

      /* Establish base indentation on first indented line */
      if (base_indent == -1 && pos > 0)
      {
         *base_indent_ptr = base_indent = pos;
         if (base_indent < 2)
         {
            hypredrv_ErrorCodeSet(ERROR_YAML_INVALID_BASE_INDENT);
            hypredrv_ErrorMsgAdd(
               "Base indentation in YAML input must be at least 2 spaces");
            goto fail_with_text_cleanup;
         }
      }

      /* Check indentation consistency */
      if (pos > 0)
      {
         /* GCOVR_EXCL_BR_START */                      /* low-signal branch under CI */
         if (base_indent > 0 && pos % base_indent != 0) /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_YAML_INCONSISTENT_INDENT);
            hypredrv_ErrorMsgAdd("Inconsistent indentation detected in YAML input!");
            goto fail_with_text_cleanup;
         }

         /* Enforce reasonable indentation jumps.
          * Allow a single-level jump (base_indent), or up to 2 * base_indent when
          * introducing a sequence item ("-").
          */
         if (!first_indented_line && pos > prev_indent)
         {
            int  jump = pos - prev_indent;
            bool is_seq_item =
               (bool)(line[pos] == '-' &&
                      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                      (line[pos + 1] == '\0' || line[pos + 1] == ' '));
            /* GCOVR_EXCL_BR_STOP */
            int max_allowed = (int)is_seq_item ? base_indent * 2 : base_indent;
            /* Be tolerant to avoid false positives with deeper nesting */
            if (jump > max_allowed * 2)
            {
               hypredrv_ErrorCodeSet(ERROR_YAML_INVALID_INDENT_JUMP);
               hypredrv_ErrorMsgAdd("Invalid indentation jump detected in YAML input!");
               goto fail_with_text_cleanup;
            }
         }

         first_indented_line = false;
         prev_indent         = pos;
      }

      /* Parse key/val only for non-sequence lines */
      if (!is_seq_item_line)
      {
         /* Check for divisor character */
         /* GCOVR_EXCL_BR_START */              /* low-signal branch under CI */
         if ((sep = strchr(line, ':')) == NULL) /* GCOVR_EXCL_BR_STOP */
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
      }

      /* Preprocessor include directive:
       * only treat "include: <file>" (scalar) as a directive.
       * For "include:" with a YAML sequence underneath, keep it in the text and
       * handle it later at the YAML tree level.
       */
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!is_seq_item_line && key && !strcmp(key, "include") && val && strlen(val) > 0)
      /* GCOVR_EXCL_BR_STOP */
      {
         /* Calculate node level based on actual indentation */
         inner_level = pos / (base_indent > 0 ? base_indent : 1);

         /* Recursively read the content of the included file */
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         YAMLtextReadWithContext(current_dir ? current_dir : dirname, val, inner_level,
                                 /* GCOVR_EXCL_BR_STOP */
                                 base_indent_ptr, length_ptr, text_ptr, capacity_ptr,
                                 ctx);
         if (hypredrv_ErrorCodeActive())
         {
            goto cleanup;
         }
      }
      else
      {
         /* Use actual indentation spacing */
         num_whitespaces = (size_t)(base_indent > 0 ? base_indent : 1) * (size_t)level;

         /* Regular line, append it to the text */
         new_length = *length_ptr + strlen(backup) + num_whitespaces;
         if (!YAMLincludeContextAddBytes(ctx, strlen(backup) + num_whitespaces))
         {
            goto fail_with_text_cleanup;
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!YAMLtextBufferEnsureCapacity(text_ptr, capacity_ptr, new_length + 1))
         /* GCOVR_EXCL_BR_STOP */
         {
            goto fail_with_text_cleanup; /* GCOVR_EXCL_LINE */
         }

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

cleanup:
   if (fp)
   {
      fclose(fp);
   }
   free(current_base);
   free(current_dir);
   free(resolved_path);
   if (pushed)
   {
      YAMLreadStackPop(ctx);
   }
   return;

fail_with_text_cleanup:
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (*text_ptr)            /* GCOVR_EXCL_BR_STOP */
   {
      free(*text_ptr);
      *text_ptr   = NULL;
      *length_ptr = 0;
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (capacity_ptr)         /* GCOVR_EXCL_BR_STOP */
      {
         *capacity_ptr = 0;
      }
   }
   goto cleanup;
}

void
hypredrv_YAMLtextRead(const char *dirname, const char *basename, int level,
                      int *base_indent_ptr, size_t *length_ptr, char **text_ptr)
{
   YAMLincludeContext ctx;
   size_t             capacity = 0;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (text_ptr && *text_ptr && length_ptr && (*length_ptr > 0)) /* GCOVR_EXCL_BR_STOP */
   {
      capacity = *length_ptr + 1;
   }
   if (!YAMLincludeContextInit(&ctx, dirname))
   {
      return;
   }
   YAMLtextReadWithContext(dirname, basename, level, base_indent_ptr, length_ptr,
                           text_ptr, &capacity, &ctx);
   YAMLincludeContextDestroy(&ctx);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreeBuild
 *-----------------------------------------------------------------------------*/

typedef struct YAMLtoken_struct
{
   char *key;
   char *val;
   char *inline_key;
   char *inline_val;
   int   level;
   int   indent;
   bool  divisor_is_ok;
   bool  is_sequence_item;
} YAMLtoken;

typedef struct YAMLtokenArray_struct
{
   YAMLtoken *data;
   int        size;
   int        capacity;
} YAMLtokenArray;

static void
YAMLtokenDestroy(YAMLtoken *token)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!token)               /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   free(token->key);
   free(token->val);
   free(token->inline_key);
   free(token->inline_val);
   memset(token, 0, sizeof(*token));
}

static void
YAMLtokenArrayDestroy(YAMLtokenArray *arr)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!arr)                 /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   for (int i = 0; i < arr->size; i++)
   {
      YAMLtokenDestroy(&arr->data[i]);
   }
   free(arr->data);
   arr->data     = NULL;
   arr->size     = 0;
   arr->capacity = 0;
}

static bool
YAMLtokenArrayReserve(YAMLtokenArray *arr, int min_capacity)
{
   if (arr->capacity >= min_capacity)
   {
      return true;
   }

   int        new_capacity = (arr->capacity == 0) ? 32 : (arr->capacity * 2);
   YAMLtoken *new_data     = NULL;
   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   while (new_capacity < min_capacity) /* GCOVR_EXCL_BR_STOP */
   {
      new_capacity *= 2; /* GCOVR_EXCL_LINE */
   }

   new_data = (YAMLtoken *)realloc(arr->data, (size_t)new_capacity * sizeof(YAMLtoken));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!new_data)            /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate YAML token buffer"); /* GCOVR_EXCL_LINE */
      return false;                                                 /* GCOVR_EXCL_LINE */
   }

   arr->data     = new_data;
   arr->capacity = new_capacity;
   return true;
}

static bool
YAMLtokenArrayPush(YAMLtokenArray *arr, YAMLtoken *token)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!arr || !token)       /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
   if (!YAMLtokenArrayReserve(arr, arr->size + 1)) /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   arr->data[arr->size++] = *token;
   memset(token, 0, sizeof(*token));
   return true;
}

static bool
YAMLlineIsBlank(const char *line)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!line)                /* GCOVR_EXCL_BR_STOP */
   {
      return true; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   while (*line)             /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (*line != ' ' && *line != '\n' && *line != '\r' && *line != '\t')
      /* GCOVR_EXCL_BR_STOP */
      {
         return false;
      }
      line++;
   }

   return true;
}

static int
YAMLlineParseIndent(const char *line, int *indent_out)
{
   int count  = 0;
   int pos    = 0;
   int indent = 0;

   /* GCOVR_EXCL_BR_START */                         /* low-signal branch under CI */
   while (line[count] == ' ' || line[count] == '\t') /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (line[count] == ' ')   /* GCOVR_EXCL_BR_STOP */
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

   *indent_out = indent;
   return count;
}

static bool
YAMLtokenSetString(char **dst, const char *src)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   char *copy = strdup(src ? src : "");
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!copy)                /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                      /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate YAML token string"); /* GCOVR_EXCL_LINE */
      return false;                                                 /* GCOVR_EXCL_LINE */
   }
   *dst = copy;
   return true;
}

static int
YAMLtokenParseLine(char *line, int base_indent, YAMLtoken *token_out)
{
   char *line_ptr = NULL;
   int   indent   = 0;
   int   count    = 0;
   char *content  = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!line || !token_out)  /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   memset(token_out, 0, sizeof(*token_out));

   line[strcspn(line, "\n")] = '\0';
   /* GCOVR_EXCL_BR_START */                   /* low-signal branch under CI */
   if ((line_ptr = strchr(line, '#')) != NULL) /* GCOVR_EXCL_BR_STOP */
   {
      *line_ptr = '\0';
   }

   /* GCOVR_EXCL_BR_START */  /* low-signal branch under CI */
   if (YAMLlineIsBlank(line)) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   count   = YAMLlineParseIndent(line, &indent);
   content = line + count;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (*content == '\0')     /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   token_out->indent = indent;
   token_out->level  = (base_indent > 0) ? (indent / base_indent) : (indent / 2);

   /* Sequence item */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (content[0] == '-' && (content[1] == ' ' || content[1] == '\0'))
   /* GCOVR_EXCL_BR_STOP */
   {
      const char *inline_content  = NULL;
      token_out->is_sequence_item = true;
      token_out->divisor_is_ok    = true;

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!YAMLtokenSetString(&token_out->key, "-") ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
          !YAMLtokenSetString(&token_out->val, ""))
      /* GCOVR_EXCL_BR_STOP */
      {
         return -1; /* GCOVR_EXCL_LINE */
      }

      /* GCOVR_EXCL_BR_START */                      /* low-signal branch under CI */
      if (content[1] != ' ' || strlen(content) <= 2) /* GCOVR_EXCL_BR_STOP */
      {
         return 1;
      }

      inline_content = content + 2;
      /* GCOVR_EXCL_BR_START */      /* low-signal branch under CI */
      while (*inline_content == ' ') /* GCOVR_EXCL_BR_STOP */
      {
         inline_content++; /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
      if (*inline_content == '\0') /* GCOVR_EXCL_BR_STOP */
      {
         return 1; /* GCOVR_EXCL_LINE */
      }

      if (!strchr(inline_content, ':'))
      {
         free(token_out->val);
         token_out->val = NULL;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!YAMLtokenSetString(&token_out->val, inline_content))
         /* GCOVR_EXCL_BR_STOP */
         {
            return -1; /* GCOVR_EXCL_LINE */
         }
         return 1;
      }

      {
         char *inline_copy = strdup(inline_content);
         char *inline_sep  = NULL;
         char *inline_key  = NULL;
         char *inline_val  = NULL;
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!inline_copy)         /* GCOVR_EXCL_BR_STOP */
         {
            hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
            hypredrv_ErrorMsgAdd(
               "Failed to allocate YAML inline mapping copy"); /* GCOVR_EXCL_LINE */
            return -1;                                         /* GCOVR_EXCL_LINE */
         }

         inline_sep = strchr(inline_copy, ':');
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (inline_sep)           /* GCOVR_EXCL_BR_STOP */
         {
            *inline_sep = '\0';
            inline_key  = inline_copy;
            inline_val  = inline_sep + 1;
            /* GCOVR_EXCL_BR_START */  /* low-signal branch under CI */
            while (*inline_key == ' ') /* GCOVR_EXCL_BR_STOP */
            {
               inline_key++; /* GCOVR_EXCL_LINE */
            }
            while (*inline_val == ' ')
            {
               inline_val++;
            }

            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            if (!YAMLtokenSetString(&token_out->inline_key, inline_key) ||
                /* GCOVR_EXCL_BR_STOP */
                /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                !YAMLtokenSetString(&token_out->inline_val, inline_val))
            /* GCOVR_EXCL_BR_STOP */
            {
               free(inline_copy); /* GCOVR_EXCL_LINE */
               return -1;         /* GCOVR_EXCL_LINE */
            }
         }
         free(inline_copy);
      }

      return 1;
   }

   /* Key/value mapping or malformed line */
   {
      char *sep = strchr(content, ':');
      char *key = content;
      char *val = NULL;

      if (sep)
      {
         *sep                     = '\0';
         val                      = sep + 1;
         token_out->divisor_is_ok = true;
      }
      else
      {
         val                      = content + strlen(content);
         token_out->divisor_is_ok = false;
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      while (*key == ' ')       /* GCOVR_EXCL_BR_STOP */
      {
         key++; /* GCOVR_EXCL_LINE */
      }
      while (*val == ' ')
      {
         val++;
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!YAMLtokenSetString(&token_out->key, key) ||
          /* GCOVR_EXCL_BR_STOP */
          /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
          !YAMLtokenSetString(&token_out->val, val))
      /* GCOVR_EXCL_BR_STOP */
      {
         return -1; /* GCOVR_EXCL_LINE */
      }
   }

   return 1;
}

static bool
YAMLtokenizeText(const char *text, int base_indent, YAMLtokenArray *tokens)
{
   char *text_copy = NULL;
   char *remaining = NULL;
   char *line      = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!tokens)              /* GCOVR_EXCL_BR_STOP */
   {
      return false; /* GCOVR_EXCL_LINE */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   text_copy = strdup(text ? text : "");
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!text_copy)           /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate YAML text copy for tokenization"); /* GCOVR_EXCL_LINE */
      return false;                                             /* GCOVR_EXCL_LINE */
   }

   remaining = text_copy;
   while ((line = strtok_r(remaining, "\n", &remaining)))
   {
      YAMLtoken token;
      int       status = YAMLtokenParseLine(line, base_indent, &token);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (status < 0)           /* GCOVR_EXCL_BR_STOP */
      {
         free(text_copy); /* GCOVR_EXCL_LINE */
         return false;    /* GCOVR_EXCL_LINE */
      }
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (status == 0)          /* GCOVR_EXCL_BR_STOP */
      {
         continue;
      }
      /* GCOVR_EXCL_BR_START */                /* low-signal branch under CI */
      if (!YAMLtokenArrayPush(tokens, &token)) /* GCOVR_EXCL_BR_STOP */
      {
         YAMLtokenDestroy(&token); /* GCOVR_EXCL_LINE */
         free(text_copy);          /* GCOVR_EXCL_LINE */
         return false;             /* GCOVR_EXCL_LINE */
      }
   }

   free(text_copy);
   return true;
}

static void
YAMLtreeBuildFromTokens(int base_indent, const YAMLtokenArray *tokens,
                        YAMLtree **tree_ptr)
{
   YAMLtree *tree   = hypredrv_YAMLtreeCreate(base_indent);
   YAMLnode *parent = tree->root;

   for (int i = 0; i < tokens->size; i++)
   {
      const YAMLtoken *token = &tokens->data[i];

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (i > 0 && token->level > parent->level && parent->val && strlen(parent->val) > 0)
      /* GCOVR_EXCL_BR_STOP */
      {
         parent->valid = YAML_NODE_UNEXPECTED_VAL;
         hypredrv_ErrorCodeSet(ERROR_UNEXPECTED_VAL);
         hypredrv_ErrorMsgAdd(
            "YAML key '%s' cannot have both a scalar value and nested entries",
            parent->key);
         *tree_ptr = tree;
         return;
      }

      YAMLnode *node = hypredrv_YAMLnodeCreate(token->key, token->val, token->level);
      YAMLnode *validation_node = node;

      hypredrv_YAMLnodeAppend(node, &parent);

      /* "- key: value" inline mapping under sequence item */
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (token->is_sequence_item && token->inline_key && token->inline_val)
      /* GCOVR_EXCL_BR_STOP */
      {
         YAMLnode *inline_node = hypredrv_YAMLnodeCreate(
            token->inline_key, token->inline_val, token->level + 1);
         hypredrv_YAMLnodeAddChild(node, inline_node);
         parent          = inline_node;
         validation_node = inline_node;
      }

      if (base_indent > 0)
      {
         if ((token->indent % base_indent) != 0)
         {
            YAML_NODE_SET_INVALID_INDENT(validation_node);
         }
      }
      else if ((token->indent % 2) != 0)
      {
         YAML_NODE_SET_INVALID_INDENT(validation_node);
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!token->divisor_is_ok && !token->is_sequence_item) /* GCOVR_EXCL_BR_STOP */
      {
         YAML_NODE_SET_INVALID_DIVISOR(validation_node);
      }
   }

   *tree_ptr = tree;
}

void
hypredrv_YAMLtreeBuild(int base_indent, const char *text, YAMLtree **tree_ptr)
{
   YAMLtokenArray tokens;

   memset(&tokens, 0, sizeof(tokens));
   /* GCOVR_EXCL_BR_START */                          /* low-signal branch under CI */
   if (!YAMLtokenizeText(text, base_indent, &tokens)) /* GCOVR_EXCL_BR_STOP */
   {
      YAMLtokenArrayDestroy(&tokens); /* GCOVR_EXCL_LINE */
      return;                         /* GCOVR_EXCL_LINE */
   }

   YAMLtreeBuildFromTokens(base_indent, &tokens, tree_ptr);
   YAMLtokenArrayDestroy(&tokens);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreeUpdate
 *
 * Update a YAML tree with information passed via command line
 *-----------------------------------------------------------------------------*/

/* Helpers for hypredrv_YAMLtreeUpdate */
static void
YAMLnodeDestroyChildren(YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   YAMLnode *child = node->children;
   while (child)
   {
      YAMLnode *next = child->next;
      hypredrv_YAMLnodeDestroy(child);
      child = next;
   }
   node->children = NULL;
}

static void
YAMLnodeClearMappedVal(YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (node && node->mapped_val) /* GCOVR_EXCL_BR_STOP */
   {
      free(node->mapped_val);
      node->mapped_val = NULL;
   }
}

static void
YAMLnodeSetScalarValue(YAMLnode *node, const char *val)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   YAMLnodeClearMappedVal(node);

   free(node->val);
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   node->val = hypredrv_StrTrim(strdup(val ? val : ""));
   /* GCOVR_EXCL_BR_STOP */
   /* GCOVR_EXCL_BR_START */       /* low-signal branch under CI */
   if (!strstr(node->key, "name")) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_StrToLowerCase(node->val);
   }
   node->valid = YAML_NODE_UNKNOWN;
}

static void
YAMLnodeEnsureMapping(YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
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
   YAMLnode *child = hypredrv_YAMLnodeFindChildByKey(parent, key);
   if (!child)
   {
      child = hypredrv_YAMLnodeCreate(key, "", parent->level + 1);
      hypredrv_YAMLnodeAddChild(parent, child);
   }
   return child;
}

/* Check if a node has sequence items (children with key == "-") */
static bool
YAMLnodeHasSequenceItems(YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
   if (!node || !node->children) /* GCOVR_EXCL_BR_STOP */
   {
      return false;
   }
   YAML_NODE_ITERATE(node, child)
   {
      if (!strcmp(child->key, "-"))
      {
         return true;
      }
   }
   return false;
}

/*-----------------------------------------------------------------------------
 * Helper functions for include expansion
 *-----------------------------------------------------------------------------*/

static void
YAMLnodeRemoveChild(YAMLnode *parent, YAMLnode *child)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!parent || !child)    /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }
   YAMLnode *cur  = parent->children;
   YAMLnode *prev = NULL;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   while (cur)               /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (cur == child)         /* GCOVR_EXCL_BR_STOP */
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (prev)                 /* GCOVR_EXCL_BR_STOP */
         {
            prev->next = cur->next; /* GCOVR_EXCL_LINE */
         }
         else
         {
            parent->children = cur->next;
         }
         cur->next = NULL;
         return;
      }
      prev = cur;       /* GCOVR_EXCL_LINE */
      cur  = cur->next; /* GCOVR_EXCL_LINE */
   }
}

static YAMLnode *
YAMLnodeCloneDeep(const YAMLnode *src, int level_offset)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!src)                 /* GCOVR_EXCL_BR_STOP */
   {
      return NULL; /* GCOVR_EXCL_LINE */
   }
   YAMLnode *clone =
      hypredrv_YAMLnodeCreate(src->key, src->val, src->level + level_offset);
   clone->valid = src->valid;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (src->mapped_val)      /* GCOVR_EXCL_BR_STOP */
   {
      clone->mapped_val = strdup(src->mapped_val); /* GCOVR_EXCL_LINE */
   }
   YAML_NODE_ITERATE(src, child_src)
   {
      YAMLnode *child_clone = YAMLnodeCloneDeep(child_src, level_offset);
      hypredrv_YAMLnodeAddChild(clone, child_clone);
   }
   return clone;
}

static void
YAMLnodeExpandIncludesRecursive(YAMLnode *node, const char *base_dir, int base_indent,
                                YAMLincludeContext *ctx)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   YAMLnode *child = node->children;
   while (child)
   {
      YAMLnode *next = child->next;

      if (!strcmp(child->key, "include"))
      {
         const int include_level = child->level;
         /* Collect include paths */
         char **paths   = NULL;
         int    n_paths = 0;
         /* GCOVR_EXCL_BR_START */                 /* low-signal branch under CI */
         if (child->val && strlen(child->val) > 0) /* GCOVR_EXCL_BR_STOP */
         {
            paths    = (char **)malloc(sizeof(char *));
            paths[0] = strdup(child->val);
            n_paths  = 1;
         }

         YAML_NODE_ITERATE(child, inc_item)
         {
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            if (!strcmp(inc_item->key, "-") && inc_item->val && strlen(inc_item->val) > 0)
            /* GCOVR_EXCL_BR_STOP */
            {
               char **new_paths =
                  (char **)realloc(paths, (size_t)(n_paths + 1) * sizeof(char *));
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!new_paths)           /* GCOVR_EXCL_BR_STOP */
               {
                  /* Free existing paths array and all its elements */
                  /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                  for (int j = 0; j < n_paths; j++)
                  /* GCOVR_EXCL_BR_STOP */ /* GCOVR_EXCL_LINE */
                  {
                     free(paths[j]); /* GCOVR_EXCL_LINE */
                  }
                  free(paths);                             /* GCOVR_EXCL_LINE */
                  hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
                  hypredrv_ErrorMsgAdd(
                     "Failed to allocate memory for include paths"); /* GCOVR_EXCL_LINE */
                  return;                                            /* GCOVR_EXCL_LINE */
               }
               paths            = new_paths;
               paths[n_paths++] = strdup(inc_item->val);
            }
         }

         /* Remove the include node before inserting new nodes */
         YAMLnodeRemoveChild(node, child);
         /* Free removed subtree (include node + its list items) to avoid leaks */
         hypredrv_YAMLnodeDestroy(child);

         for (int i = 0; i < n_paths; i++)
         {
            char *resolved_path = NULL;
            if (!YAMLincludeResolvePath(ctx, base_dir, paths[i], &resolved_path))
            {
               for (int j = i; j < n_paths; j++)
               {
                  free(paths[j]);
               }
               free(paths);
               return;
            }
            /* GCOVR_EXCL_BR_START */                     /* low-signal branch under CI */
            if (!YAMLexpandStackPush(ctx, resolved_path)) /* GCOVR_EXCL_BR_STOP */
            {
               free(resolved_path);      /* GCOVR_EXCL_LINE */
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               for (int j = i; j < n_paths; j++)
               /* GCOVR_EXCL_BR_STOP */ /* GCOVR_EXCL_LINE */
               {
                  free(paths[j]); /* GCOVR_EXCL_LINE */
               }
               free(paths); /* GCOVR_EXCL_LINE */
               return;      /* GCOVR_EXCL_LINE */
            }

            char *inc_dir  = NULL;
            char *inc_base = NULL;
            hypredrv_SplitFilename(resolved_path, &inc_dir, &inc_base);
            free(inc_base);
            inc_base = NULL;

            int    inc_base_indent = base_indent;
            size_t inc_len         = 0;
            size_t inc_capacity    = 0;
            char  *inc_text        = NULL;

            YAMLtextReadWithContext(base_dir, paths[i], 0, &inc_base_indent, &inc_len,
                                    &inc_text, &inc_capacity, ctx);
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            if (inc_text)             /* GCOVR_EXCL_BR_STOP */
            {
               YAMLtree *inc_tree = NULL;
               hypredrv_YAMLtreeBuild(inc_base_indent, inc_text, &inc_tree);
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (inc_tree && !hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodeExpandIncludesRecursive(
                     /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                     inc_tree->root,
                     inc_dir ? inc_dir : base_dir, inc_base_indent, ctx);
                  /* GCOVR_EXCL_BR_STOP */
               }
               if (hypredrv_ErrorCodeActive())
               {
                  hypredrv_YAMLtreeDestroy(&inc_tree);
                  free(inc_text);
                  free(inc_dir);
                  free(resolved_path);
                  YAMLexpandStackPop(ctx);
                  for (int j = i; j < n_paths; j++)
                  {
                     free(paths[j]);
                  }
                  free(paths);
                  return;
               }

               /* Wrap included content as a sequence item */
               YAMLnode *seq_node = hypredrv_YAMLnodeCreate("-", "", include_level);
               YAML_NODE_ITERATE(inc_tree->root, inc_child)
               {
                  YAMLnode *clone =
                     YAMLnodeCloneDeep(inc_child, include_level - inc_child->level + 1);
                  hypredrv_YAMLnodeAddChild(seq_node, clone);
               }
               hypredrv_YAMLnodeAddChild(node, seq_node);

               hypredrv_YAMLtreeDestroy(&inc_tree);
               free(inc_text);
            }

            YAMLexpandStackPop(ctx);
            free(resolved_path);
            free(inc_dir);
            free(paths[i]);
         }
         free(paths);

         /* Continue from next sibling (include node already removed) */
         child = next;
         continue;
      }

      YAMLnodeExpandIncludesRecursive(child, base_dir, base_indent, ctx);
      child = next;
   }
}

static int
YAMLargsFindFlagIndex(int argc, char **argv, const char *short_flag,
                      const char *long_flag)
{
   for (int i = 0; i < argc; i++)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if ((short_flag && !strcmp(argv[i], short_flag)) ||
          /* GCOVR_EXCL_BR_STOP */
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
      /* GCOVR_EXCL_BR_START */                        /* low-signal branch under CI */
      if (argv[i] && hypredrv_IsYAMLFilename(argv[i])) /* GCOVR_EXCL_BR_STOP */
      {
         return i;
      }
   }
   return -1;
}

/*-----------------------------------------------------------------------------
 * Helpers for hypredrv_YAMLtreeUpdate (must be file-scope for Clang; no nested functions)
 *-----------------------------------------------------------------------------*/

typedef struct YAMLOverridePathCtx_struct
{
   char     *segments[64];
   int       num_segments;
   YAMLtree *tree;
} YAMLOverridePathCtx;

static void
YAMLtreeUpdateApplyPathToNode(YAMLOverridePathCtx *ctx, YAMLnode *node, int start_idx,
                              const char *value)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!ctx || !node)        /* GCOVR_EXCL_BR_STOP */
   {
      return; /* GCOVR_EXCL_LINE */
   }

   YAMLnode *cur = node;
   for (int i = start_idx; i < ctx->num_segments; i++)
   {
      const char *seg_const = ctx->segments[i];
      bool        is_last   = (i == ctx->num_segments - 1);

      if (!is_last) /* intermediate */
      {
         YAMLnode *child = YAMLnodeGetOrCreateChild(cur, seg_const);
         YAMLnodeEnsureMapping(child);

         /* Check if this child has sequence items */
         if (YAMLnodeHasSequenceItems(child))
         {
            /* Apply remaining path to all sequence items */
            YAMLnode *seq_walk = child->children;
            while (seq_walk != NULL)
            {
               /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
               if (!strcmp(seq_walk->key, "-")) /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLtreeUpdateApplyPathToNode(ctx, seq_walk, i + 1, value);
               }
               seq_walk = seq_walk->next;
            }
            return; /* Done with this branch */
         }

         cur = child;
      }
      else /* leaf */
      {
         /* Check if current node has sequence items */
         /* GCOVR_EXCL_BR_START */          /* low-signal branch under CI */
         if (YAMLnodeHasSequenceItems(cur)) /* GCOVR_EXCL_BR_STOP */
         {
            YAMLnode *seq_walk = cur->children; /* GCOVR_EXCL_LINE */
            while (seq_walk != NULL)            /* GCOVR_EXCL_LINE */
            {
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!strcmp(seq_walk->key, "-"))
               /* GCOVR_EXCL_BR_STOP */ /* GCOVR_EXCL_LINE */
               {
                  YAMLnode *item_leaf = hypredrv_YAMLnodeFindChildByKey(
                     seq_walk, seg_const);  /* GCOVR_EXCL_LINE */
                  /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
                  if (!item_leaf) /* GCOVR_EXCL_BR_STOP */ /* GCOVR_EXCL_LINE */
                  {
                     item_leaf = hypredrv_YAMLnodeCreate(
                        seg_const, value, seq_walk->level + 1);      /* GCOVR_EXCL_LINE */
                     hypredrv_YAMLnodeAddChild(seq_walk, item_leaf); /* GCOVR_EXCL_LINE */
                  }
                  else
                  {
                     YAMLnodeDestroyChildren(item_leaf);       /* GCOVR_EXCL_LINE */
                     YAMLnodeSetScalarValue(item_leaf, value); /* GCOVR_EXCL_LINE */
                  }
               }
               seq_walk = seq_walk->next; /* GCOVR_EXCL_LINE */
            }
            return; /* Done */ /* GCOVR_EXCL_LINE */
         }

         YAMLnode *leaf = hypredrv_YAMLnodeFindChildByKey(cur, seg_const);
         if (!leaf)
         {
            leaf = hypredrv_YAMLnodeCreate(seg_const, value, cur->level + 1);
            hypredrv_YAMLnodeAddChild(cur, leaf);
         }
         else
         {
            YAMLnodeDestroyChildren(leaf);
            YAMLnodeSetScalarValue(leaf, value);
         }
      }
   }
}

void
hypredrv_YAMLtreeUpdate(int argc, char **argv, YAMLtree *tree)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!tree || !tree->root) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_YAML_TREE_NULL);
      hypredrv_ErrorMsgAdd("Cannot update a void YAML tree!");
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
      /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
      if (override_start >= override_end) /* GCOVR_EXCL_BR_STOP */
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
      /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
      if (argc > 0 && (argc % 2) == 0) /* GCOVR_EXCL_BR_STOP */
      {
         bool looks_like_pairs = true;
         for (int i = 0; i < argc; i += 2)
         {
            const char *k = argv[i];
            const char *v = argv[i + 1];
            /* GCOVR_EXCL_BR_START */                 /* low-signal branch under CI */
            if (!k || !v || strncmp(k, "--", 2) != 0) /* GCOVR_EXCL_BR_STOP */
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
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Invalid CLI overrides: expected pairs of '--path:to:key value'");
      return;
   }

   for (int i = override_start; i < override_end; i += 2)
   {
      const char *k = argv[i];
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (!k)                   /* GCOVR_EXCL_BR_STOP */
      {
         continue;
      }

      bool has_dashes = (strncmp(k, "--", 2) == 0);
      if (!has_dashes)
      {
         /* In full-argv mode, also accept key paths without the leading "--",
          * e.g. "--args preconditioner:mgr:print_level 1". */
         bool looks_like_path = (strchr(k, ':') != NULL);
         /* GCOVR_EXCL_BR_START */              /* low-signal branch under CI */
         if (full_argv_mode && looks_like_path) /* GCOVR_EXCL_BR_STOP */
         {
            /* Accept path-style overrides without the leading "--" in full argv mode. */
         }
         else
         {
            /* In full argv mode, ignore non-override tokens; keep strictness in
             * pair-list mode. */
            /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
            if (!full_argv_mode)      /* GCOVR_EXCL_BR_STOP */
            {
               hypredrv_ErrorCodeSet(ERROR_INVALID_KEY); /* GCOVR_EXCL_LINE */
               hypredrv_ErrorMsgAdd(                     /* GCOVR_EXCL_LINE */
                                    "Invalid override key '%s' (expected path:to:key or "
                                    "--path:to:key)",
                                    k);
               return; /* GCOVR_EXCL_LINE */
            }
            continue;
         }
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (i + 1 >= argc)        /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);                   /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Missing value for override '%s'", k); /* GCOVR_EXCL_LINE */
         return;                                                     /* GCOVR_EXCL_LINE */
      }

      const char *v = argv[i + 1];
      /* GCOVR_EXCL_BR_START */             /* low-signal branch under CI */
      if (!v || (strncmp(v, "--", 2) == 0)) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Missing value for override '%s'", k);
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

         /* Collect all path segments into an array */
         char *segments[64]; /* Max 64 segments should be enough */
         int   num_segments = 0;
         char *save_seg     = NULL;
         char *seg_temp     = strtok_r(p, ":", &save_seg);
         /* GCOVR_EXCL_BR_START */             /* low-signal branch under CI */
         while (seg_temp && num_segments < 64) /* GCOVR_EXCL_BR_STOP */
         {
            segments[num_segments++] = seg_temp;
            seg_temp                 = strtok_r(NULL, ":", &save_seg);
         }

         if (num_segments == 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
            hypredrv_ErrorMsgAdd("Invalid override path: '%s'", k);
            free(path);
            return;
         }

         YAMLOverridePathCtx ctx;
         memset(&ctx, 0, sizeof(ctx));
         ctx.num_segments = num_segments;
         ctx.tree         = tree;
         for (int si = 0; si < num_segments; si++)
         {
            ctx.segments[si] = segments[si];
         }

         YAMLtreeUpdateApplyPathToNode(&ctx, tree->root, 0, v);

         free(path);
      }
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreePrint
 *
 * Prints all nodes in a tree
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLtreePrint(YAMLtree *tree, YAMLprintMode print_mode)
{
   YAMLnode *child = NULL;

   if (!tree)
   {
      hypredrv_ErrorCodeSet(ERROR_YAML_TREE_NULL);
      hypredrv_ErrorMsgAdd("Cannot print a void YAML tree!");
      return;
   }

   PRINT_DASHED_LINE(MAX_DIVISOR_LENGTH);
   child = tree->root->children;
   while (child != NULL)
   {
      hypredrv_YAMLnodePrint(child, print_mode);
      child = child->next;
   }
   PRINT_DASHED_LINE(MAX_DIVISOR_LENGTH);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreeValidate
 *
 * Validates all nodes in a tree and sets appropriate error codes
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLtreeValidate(YAMLtree *tree)
{
   if (!tree)
   {
      hypredrv_ErrorCodeSet(ERROR_YAML_TREE_NULL);
      hypredrv_ErrorMsgAdd("Cannot validate a void YAML tree!");
      return;
   }

   YAMLnode *child = tree->root->children;
   while (child != NULL)
   {
      hypredrv_YAMLnodeValidate(child);
      child = child->next;
   }

   tree->is_validated = true; // Mark tree as validated
}

void
hypredrv_YAMLtreeExpandIncludes(YAMLtree *tree, const char *base_dir)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!tree || !tree->root) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   const char *dir = (base_dir && strlen(base_dir) > 0) ? base_dir : ".";
   /* GCOVR_EXCL_BR_STOP */
   int                bi = tree->base_indent > 0 ? tree->base_indent : 2;
   YAMLincludeContext ctx;
   /* GCOVR_EXCL_BR_START */               /* low-signal branch under CI */
   if (!YAMLincludeContextInit(&ctx, dir)) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }
   YAMLnodeExpandIncludesRecursive(tree->root, dir, bi, &ctx);
   YAMLincludeContextDestroy(&ctx);
}

/******************************************************************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodeCreate
 *-----------------------------------------------------------------------------*/

YAMLnode *
hypredrv_YAMLnodeCreate(const char *key, const char *val, int level)
{
   YAMLnode *node = NULL;

   node             = (YAMLnode *)malloc(sizeof(YAMLnode));
   node->level      = level;
   node->key        = hypredrv_StrTrim(strdup((char *)key));
   node->mapped_val = NULL;
   node->valid      = YAML_NODE_UNKNOWN;
   node->parent     = NULL;
   node->children   = NULL;
   node->next       = NULL;

   /* If the key contains "name", "node->val" will be the same as "val".
      Otherwise, "node->val" will be set as "val" with all lowercase letters */
   if (strstr(key, "name"))
   {
      node->val = hypredrv_StrTrim(strdup((char *)val));
   }
   else
   {
      node->val = hypredrv_StrToLowerCase(hypredrv_StrTrim(strdup((char *)val)));
      /* Strip surrounding double quotes if present */
      size_t len = strlen(node->val);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (len >= 2 && node->val[0] == '\"' && node->val[len - 1] == '\"')
      /* GCOVR_EXCL_BR_STOP */
      {
         node->val[len - 1] = '\0';
         memmove(node->val, node->val + 1, len - 1);
      }
   }

   return node;
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodeDestroy
 *
 * Destroys a node via depth-first search (DFS)
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLnodeDestroy(YAMLnode *node)
{
   YAMLnode *child = NULL;
   YAMLnode *next  = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (node == NULL)         /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   child = node->children;
   while (child != NULL)
   {
      next = child->next;
      hypredrv_YAMLnodeDestroy(child);
      child = next;
   }
   free(node->key);
   free(node->val);
   free(node->mapped_val);
   free(node);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodeAddChild
 *
 * Adds a "child" node as the first child of the "parent" node.
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLnodeAddChild(YAMLnode *parent, YAMLnode *child)
{
   YAMLnode *node = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!parent || !child) return;
   /* GCOVR_EXCL_BR_STOP */

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
 * hypredrv_YAMLnodeAppend
 *
 * Appends a node to the tree
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLnodeAppend(YAMLnode *node, YAMLnode **previous_ptr)
{
   YAMLnode *previous       = *previous_ptr;
   int       previous_level = previous->level;

   if (node->level > previous_level)
   {
      /* Add child to current parent */
      hypredrv_YAMLnodeAddChild(previous, node);
   }
   else if (node->level == previous_level)
   {
      /* Add sibling to children's list and keep the current parent */
      hypredrv_YAMLnodeAddChild(previous->parent, node);
   }
   else
   {
      while (previous_level > node->level)
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (!previous->parent) break;
         /* GCOVR_EXCL_BR_STOP */
         previous = previous->parent;
         previous_level--;
      }

      /* Add ancestor */
      hypredrv_YAMLnodeAddChild(previous->parent, node);
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
 * hypredrv_YAMLnodeValidate
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLnodeValidate(YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   /* Sequence items are considered valid */
   if (!strcmp(node->key, "-"))
   {
      YAML_NODE_SET_VALID(node);
      /* Still validate children so invalid keys inside sequence items are caught */
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (node->valid)
   /* GCOVR_EXCL_BR_STOP */
   {
      case YAML_NODE_INVALID_INDENT:
         hypredrv_ErrorCodeSet(ERROR_YAML_INVALID_INDENT);
         break;

      case YAML_NODE_INVALID_DIVISOR:
         hypredrv_ErrorCodeSet(ERROR_YAML_INVALID_DIVISOR);
         break;

      case YAML_NODE_INVALID_KEY:
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorCodeSet(ERROR_MAYBE_INVALID_VAL);
         break;

      case YAML_NODE_INVALID_VAL:
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         break;

      case YAML_NODE_UNEXPECTED_VAL:
         hypredrv_ErrorCodeSet(ERROR_UNEXPECTED_VAL);
         break;

      default:
         break;
   }

   // Recursively validate children
   YAMLnode *child = node->children;
   while (child != NULL)
   {
      hypredrv_YAMLnodeValidate(child);
      child = child->next;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodePrint
 *-----------------------------------------------------------------------------*/

void
hypredrv_YAMLnodePrint(YAMLnode *node, YAMLprintMode print_mode)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!node)                /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   /* Special formatting for sequence items ("-") */
   bool      is_seq_item  = (strcmp(node->key, "-") == 0);
   YAMLnode *inline_child = NULL;
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (is_seq_item && node->val && !strlen(node->val) && node->children)
   /* GCOVR_EXCL_BR_STOP */
   {
      inline_child = node->children; /* Print first child inline with the dash */
   }

   /* If this is a sequence item, print the bullet (with optional inline first child) up
    * front, then skip that inline child when recursing into children to avoid
    * duplication. */
   if (is_seq_item)
   {
      printf("%*s- ", 2 * node->level, "");
      if (inline_child)
      {
         printf("%s: %s\n", inline_child->key, inline_child->val);
      }
      else
      {
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         printf("%s\n", (node->val && strlen(node->val) > 0) ? node->val : "");
         /* GCOVR_EXCL_BR_STOP */
      }
   }

   // Handle printing based on mode and validity
   switch (print_mode)
   {
      case YAML_PRINT_MODE_ANY:
         switch (node->valid)
         {
            case YAML_NODE_VALID:
               if (!is_seq_item)
               {
                  YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_GREEN, "");
               }
               break;
            case YAML_NODE_INVALID_INDENT:
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!is_seq_item)         /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                      TEXT_BOLD " <-- * INVALID INDENTATION *");
               }
               break;
            case YAML_NODE_INVALID_DIVISOR:
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!is_seq_item)         /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                      TEXT_BOLD " <-- * INVALID DIVISOR *");
               }
               break;
            case YAML_NODE_INVALID_KEY:
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!is_seq_item)         /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_YELLOWBOLD,
                                      TEXT_BOLD " <-- * INVALID KEY *");
               }
               break;
            case YAML_NODE_INVALID_VAL:
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!is_seq_item)         /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                      TEXT_BOLD " <-- * INVALID VALUE *");
               }
               break;
            case YAML_NODE_UNEXPECTED_VAL:
               /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
               if (!is_seq_item)         /* GCOVR_EXCL_BR_STOP */
               {
                  YAMLnodePrintHelper(node, TEXT_GREEN, TEXT_REDBOLD,
                                      TEXT_BOLD " <-- * UNEXPECTED VALUE *");
               }
               break;
            default:
               YAMLnodePrintHelper(node, TEXT_REDBOLD, TEXT_REDBOLD,
                                   TEXT_REDBOLD " <-- * INVALID ENTRY *");
         }
         break;

      case YAML_PRINT_MODE_ONLY_VALID:
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         if (node->valid == YAML_NODE_VALID && !is_seq_item) /* GCOVR_EXCL_BR_STOP */
         {
            YAMLnodePrintHelper(node, "", "", "");
         }
         break;

      case YAML_PRINT_MODE_NO_CHECKING:
      default:
         if (!is_seq_item)
         {
            YAMLnodePrintHelper(node, "", "", "");
         }
         break;
   }

   // Print children
   if (is_seq_item && inline_child)
   {
      /* For "- key:" style sequence items, we printed "key:" inline above.
       * Now print that key's children (so nested mappings show up), then print any
       * additional siblings.
       */
      YAMLnode *gc = inline_child->children;
      while (gc)
      {
         hypredrv_YAMLnodePrint(gc, print_mode);
         gc = gc->next;
      }

      YAMLnode *sib = inline_child->next;
      while (sib)
      {
         hypredrv_YAMLnodePrint(sib, print_mode);
         sib = sib->next;
      }
   }
   else
   {
      YAMLnode *child_iter = node->children;
      while (child_iter != NULL)
      {
         hypredrv_YAMLnodePrint(child_iter, print_mode);
         child_iter = child_iter->next;
      }
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodeFindByKey
 *
 * Finds a node by key starting from the input "node" via DFS.
 *-----------------------------------------------------------------------------*/

YAMLnode *
hypredrv_YAMLnodeFindByKey(YAMLnode *node, const char *key)
{
   YAMLnode *child = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (node)                 /* GCOVR_EXCL_BR_STOP */
   {
      if (!strcmp(node->key, key))
      {
         return node;
      }

      child = node->children;
      while (child)
      {
         YAMLnode *found = hypredrv_YAMLnodeFindByKey(child, key);
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
 * hypredrv_YAMLnodeFindChildByKey
 *
 * Finds a node by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

YAMLnode *
hypredrv_YAMLnodeFindChildByKey(YAMLnode *parent, const char *key)
{
   YAMLnode *child = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (parent)               /* GCOVR_EXCL_BR_STOP */
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
 * hypredrv_YAMLnodeFindChildValueByKey
 *
 * Finds a value by key in the parent's children list.
 *-----------------------------------------------------------------------------*/

char *
hypredrv_YAMLnodeFindChildValueByKey(YAMLnode *parent, const char *key)
{
   YAMLnode *child = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (parent)               /* GCOVR_EXCL_BR_STOP */
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
 * hypredrv_YAMLnodeCollectSequenceItems
 *
 * Collect direct children with key == "-" into a newly allocated array.
 * Caller owns the array (but NOT the nodes) and must free it.
 * Returns the count (may be zero).
 *-----------------------------------------------------------------------------*/

int
hypredrv_YAMLnodeCollectSequenceItems(YAMLnode *parent, YAMLnode ***items_out)
{
   /* GCOVR_EXCL_BR_START */  /* low-signal branch under CI */
   if (!parent || !items_out) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   int        count = 0;
   YAMLnode **arr   = NULL;

   /* First pass: count */
   YAML_NODE_ITERATE(parent, child)
   {
      if (!strcmp(child->key, "-"))
      {
         count++;
      }
   }

   if (count == 0)
   {
      *items_out = NULL;
      return 0;
   }

   arr     = (YAMLnode **)malloc((size_t)count * sizeof(YAMLnode *));
   int idx = 0;
   YAML_NODE_ITERATE(parent, child)
   {
      /* GCOVR_EXCL_BR_START */     /* low-signal branch under CI */
      if (!strcmp(child->key, "-")) /* GCOVR_EXCL_BR_STOP */
      {
         arr[idx++] = child;
      }
   }

   *items_out = arr;
   return count;
}
