/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/presets.h"
#include "internal/compatibility.h"
#include "internal/error.h"
#include "internal/precon.h"
#include "internal/solver.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Registry entries own their strings; callers receive a separate YAML tree. */
typedef struct
{
   char               *name;
   char               *text;
   char               *help;
   hypredrv_PresetKind kind;
} Preset;

static const Preset g_presets[] = {
   {"poisson", "amg", "BoomerAMG for poisson", HYPREDRV_PRESET_PRECON},
   {"elasticity_2d", "amg:\n  coarsening:\n    num_functions: 2\n    strong_th: 0.8",
    "BoomerAMG for 2D elasticity", HYPREDRV_PRESET_PRECON},
   {"elasticity_3d", "amg:\n  coarsening:\n    num_functions: 3\n    strong_th: 0.8",
    "BoomerAMG for 3D elasticity", HYPREDRV_PRESET_PRECON},
};

/* Dynamic user-registered presets */
static Preset *g_user_presets     = NULL;
static size_t  g_num_user_presets = 0;
static size_t  g_cap_user_presets = 0;

/*-----------------------------------------------------------------------------
 * Normalize preset name in-place: lowercase and treat '-' as '_'.
 *-----------------------------------------------------------------------------*/

static void
normalize_preset_name(char *s)
{
   for (; *s; ++s)
   {
      *s = (char)tolower((unsigned char)*s);
      if (*s == '-')
      {
         *s = '_';
      }
   }
}

/* Preset named norm_name (already normalized) of the given kind (any if < 0) */
static const Preset *
PresetFindIn(const Preset *presets, size_t n, const char *norm_name, int kind)
{
   for (size_t i = 0; i < n; i++)
   {
      if ((kind < 0 || (int)presets[i].kind == kind) &&
          !strcmp(norm_name, presets[i].name))
      {
         return &presets[i];
      }
   }
   return NULL;
}

static const char *
PresetKindName(hypredrv_PresetKind kind)
{
   switch (kind)
   {
      case HYPREDRV_PRESET_PRECON:
         return "preconditioner";
      case HYPREDRV_PRESET_SOLVER:
         return "solver";
      default:
         return NULL;
   }
}

/* Validate one registration before publishing its owned strings. */
uint32_t
hypredrv_PresetRegisterTyped(const char *name, const char *yaml_text, const char *help,
                             hypredrv_PresetKind kind)
{
   const char *label = PresetKindName(kind);
   if (!label || !name || !*name || !yaml_text || !*yaml_text)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preset registration requires a solver/preconditioner kind "
                           "and nonempty name and YAML text");
      return ERROR_INVALID_VAL;
   }
   char *norm = strdup(name);
   if (!norm)
   {
      hypredrv_AllocationFailure("Failed to allocate preset name");
      return ERROR_ALLOCATION;
   }
   normalize_preset_name(norm);

   /* Reserve method names and numeric aliases accepted by the solver parser,
    * including spellings that normalize to a reserved name. */
   StrIntMapArray solvers = hypredrv_SolverGetValidTypeIntMap();
   if (kind == HYPREDRV_PRESET_SOLVER &&
       (hypredrv_StrIntMapArrayLookup(solvers, norm, NULL) ||
        (strcmp(name, norm) != 0 && hypredrv_StrIntMapArrayLookup(solvers, name, NULL))))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preset '%s' conflicts with a built-in solver", name);
      free(norm);
      return ERROR_INVALID_VAL;
   }
   const size_t nb = sizeof(g_presets) / sizeof(g_presets[0]);
   if (PresetFindIn(g_presets, nb, norm, -1) ||
       PresetFindIn(g_user_presets, g_num_user_presets, norm, -1))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preset '%s' is already defined", norm);
      free(norm);
      return ERROR_INVALID_VAL;
   }

   char *dup_text = strdup(yaml_text);
   char *dup_help = strdup(help ? help : "");
   if (!dup_text || !dup_help)
   {
      free(norm);
      free(dup_text);
      free(dup_help);
      hypredrv_AllocationFailure("Failed to allocate preset definition");
      return ERROR_ALLOCATION;
   }
   if (g_num_user_presets >= g_cap_user_presets)
   {
      size_t  new_cap = g_cap_user_presets ? g_cap_user_presets * 2 : 8;
      Preset *tmp     = realloc(g_user_presets, new_cap * sizeof(*tmp));
      if (!tmp)
      {
         free(norm);
         free(dup_text);
         free(dup_help);
         hypredrv_AllocationFailure("Failed to grow preset registry");
         return ERROR_ALLOCATION;
      }
      g_user_presets     = tmp;
      g_cap_user_presets = new_cap;
   }
   Preset *entry = &g_user_presets[g_num_user_presets++];
   entry->name   = norm;
   entry->text   = dup_text;
   entry->help   = dup_help;
   entry->kind   = kind;
   return ERROR_NONE;
}

/*-----------------------------------------------------------------------------
 * Free all user-registered presets. Safe to call multiple times.
 *-----------------------------------------------------------------------------*/

void
hypredrv_PresetFreeUserPresets(void)
{
   for (size_t i = 0; i < g_num_user_presets; i++)
   {
      free(g_user_presets[i].name);
      free(g_user_presets[i].text);
      free(g_user_presets[i].help);
   }
   free(g_user_presets);
   g_user_presets     = NULL;
   g_num_user_presets = 0;
   g_cap_user_presets = 0;
}

/*-----------------------------------------------------------------------------
 * Generate help string listing available presets (caller must free).
 *-----------------------------------------------------------------------------*/

/* Write the "    - name: help" lines of the presets of the given kind into out
 * (capacity cap; out may be NULL to measure) and return their length. */
static size_t
PresetHelpLines(const Preset *presets, size_t n, hypredrv_PresetKind kind, char *out,
                size_t cap)
{
   size_t len = 0;
   for (size_t i = 0; i < n; i++)
   {
      if (presets[i].kind != kind)
      {
         continue;
      }
      int w = snprintf(out ? out + len : NULL, out ? cap - len : 0, "    - %s: %s\n",
                       presets[i].name, presets[i].help);
      len += (w > 0) ? (size_t)w : 0;
   }
   return len;
}

char *
hypredrv_PresetHelpTyped(hypredrv_PresetKind kind)
{
   const size_t nb    = sizeof(g_presets) / sizeof(g_presets[0]);
   const char  *label = PresetKindName(kind);
   if (!label)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Unknown preset kind: %d", (int)kind);
      return NULL;
   }
   char header[64];
   snprintf(header, sizeof(header), "Available %s presets:\n", label);
   size_t head_len = strlen(header);
   size_t nb_len   = PresetHelpLines(g_presets, nb, kind, NULL, 0);
   size_t nu_len   = PresetHelpLines(g_user_presets, g_num_user_presets, kind, NULL, 0);
   size_t bytes    = head_len + nb_len + nu_len + 1;

   char *out = (char *)malloc(bytes);
   if (!out)
   {
      return NULL;
   }
   memcpy(out, header, head_len + 1);
   PresetHelpLines(g_presets, nb, kind, out + head_len, bytes - head_len);
   PresetHelpLines(g_user_presets, g_num_user_presets, kind, out + head_len + nb_len,
                   bytes - head_len - nb_len);
   return out;
}

/* Return an owned configuration tree after validating its single type node. */
YAMLnode *
hypredrv_PresetLoadTypeNode(const char *preset_name, hypredrv_PresetKind kind,
                            YAMLtree **tree_out, int *method_out)
{
   *tree_out = NULL;
   if (hypredrv_ErrorCodeActive()) return NULL;

   const char *label = PresetKindName(kind);
   if (!label)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Unknown preset kind: %d", (int)kind);
      return NULL;
   }
   char *norm = strdup(preset_name ? preset_name : "");
   if (!norm)
   {
      hypredrv_AllocationFailure("Failed to allocate preset lookup name");
      return NULL;
   }
   normalize_preset_name(norm);
   const Preset *preset =
      PresetFindIn(g_presets, sizeof(g_presets) / sizeof(g_presets[0]), norm, (int)kind);
   if (!preset)
      preset = PresetFindIn(g_user_presets, g_num_user_presets, norm, (int)kind);
   free(norm);
   if (!preset)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Unknown %s preset: '%s'", label,
                           preset_name ? preset_name : "");
      char *help = hypredrv_PresetHelpTyped(kind);
      if (help)
      {
         hypredrv_ErrorMsgAdd("%s", help);
         free(help);
      }
      return NULL;
   }

   char *text = strdup(preset->text);
   /* GCOVR_EXCL_START */
   if (!text)
   {
      hypredrv_AllocationFailure("Failed to allocate preset YAML text");
      return NULL;
   }
   /* GCOVR_EXCL_STOP */

   hypredrv_YAMLtreeBuild(2, text, tree_out);
   free(text);
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_YAMLtreeDestroy(tree_out);
      return NULL;
   }

   YAMLtree *tree = *tree_out;
   if (!tree || !tree->root || !tree->root->children || tree->root->children->next)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preset '%s' must expand to a single %s type", preset->name,
                           label);
      hypredrv_YAMLtreeDestroy(tree_out);
      return NULL;
   }

   StrIntMapArray valid_types = kind == HYPREDRV_PRESET_SOLVER
                                   ? hypredrv_SolverGetValidTypeIntMap()
                                   : hypredrv_PreconGetValidTypeIntMap();
   YAMLnode      *type_node   = tree->root->children;
   int            method;
   if (!hypredrv_StrIntMapArrayLookup(valid_types, type_node->key, &method))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
      hypredrv_ErrorMsgAdd("Unknown %s type: '%s'", label, type_node->key);
      hypredrv_YAMLtreeDestroy(tree_out);
      return NULL;
   }

   *method_out = method;
   return type_node;
}
