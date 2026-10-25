/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/presets.h"
#include "internal/compatibility.h"
#include "internal/error.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Pre-defined preconditioner variants */
static char preset_poisson[] = "amg"; /* Note: no ":" required */

static char preset_elasticity_2d[] = "amg:\n"
                                     "  coarsening:\n"
                                     "    num_functions: 2\n"
                                     "    strong_th: 0.8";

static char preset_elasticity_3d[] = "amg:\n"
                                     "  coarsening:\n"
                                     "    num_functions: 3\n"
                                     "    strong_th: 0.8";

static const hypredrv_Preset g_presets[] = {
   HYPREDRV_PRESET(poisson, HYPREDRV_PRESET_PRECON, "BoomerAMG for poisson"),
   HYPREDRV_PRESET(elasticity_2d, HYPREDRV_PRESET_PRECON, "BoomerAMG for 2D elasticity"),
   HYPREDRV_PRESET(elasticity_3d, HYPREDRV_PRESET_PRECON, "BoomerAMG for 3D elasticity"),
};

/* Dynamic user-registered presets */
static hypredrv_Preset *g_user_presets     = NULL;
static size_t           g_num_user_presets = 0;
static size_t           g_cap_user_presets = 0;

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
static const hypredrv_Preset *
PresetFindIn(const hypredrv_Preset *presets, size_t n, const char *norm_name,
             hypredrv_PresetKind kind)
{
   for (size_t i = 0; i < n; i++)
   {
      if ((kind < 0 || presets[i].kind == kind) && !strcmp(norm_name, presets[i].name))
      {
         return &presets[i];
      }
   }
   return NULL;
}

static const char *
PresetKindHeader(hypredrv_PresetKind kind)
{
   switch (kind)
   {
      case HYPREDRV_PRESET_PRECON:
         return "Available preconditioner presets:\n";
      case HYPREDRV_PRESET_SOLVER:
         return "Available solver presets:\n";
      default:
         return "Available presets:\n";
   }
}

/*-----------------------------------------------------------------------------
 * Register a new user-defined preset. Returns 0 on success, -1 on failure.
 *-----------------------------------------------------------------------------*/

int
hypredrv_PresetRegister(const char *name, const char *yaml_text, const char *help)
{
   return hypredrv_PresetRegisterTyped(name, yaml_text, help, HYPREDRV_PRESET_PRECON);
}

int
hypredrv_PresetRegisterTyped(const char *name, const char *yaml_text, const char *help,
                             hypredrv_PresetKind kind)
{
   if (!name || !*name || !yaml_text || !*yaml_text)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "hypredrv_PresetRegister: name and yaml_text must be non-NULL and non-empty");
      return -1;
   }

   char *norm = strdup(name);
   if (!norm)
   {
      hypredrv_AllocationFailure("Memory allocation failed");
      return -1;
   }
   normalize_preset_name(norm);

   /* Reject names already taken by a built-in or a user preset (of any kind) */
   const size_t nb = sizeof(g_presets) / sizeof(g_presets[0]);
   if (PresetFindIn(g_presets, nb, norm, HYPREDRV_PRESET_ANY))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "hypredrv_PresetRegister: preset '%s' conflicts with built-in preset", norm);
      free(norm);
      return -1;
   }
   if (PresetFindIn(g_user_presets, g_num_user_presets, norm, HYPREDRV_PRESET_ANY))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("hypredrv_PresetRegister: preset '%s' already registered",
                           norm);
      free(norm);
      return -1;
   }

   /* Grow array if needed */
   if (g_num_user_presets >= g_cap_user_presets)
   {
      size_t           new_cap = g_cap_user_presets ? g_cap_user_presets * 2 : 8;
      hypredrv_Preset *tmp =
         (hypredrv_Preset *)realloc(g_user_presets, new_cap * sizeof(hypredrv_Preset));
      if (!tmp)
      {
         hypredrv_AllocationFailure("Memory allocation failed");
         free(norm);
         return -1;
      }
      g_user_presets     = tmp;
      g_cap_user_presets = new_cap;
   }

   char *dup_text = strdup(yaml_text);
   char *dup_help = strdup(help ? help : "");
   if (!dup_text || !dup_help)
   {
      free(norm);
      free(dup_text);
      free(dup_help);
      hypredrv_AllocationFailure("Memory allocation failed");
      return -1;
   }

   g_user_presets[g_num_user_presets].name = norm;
   g_user_presets[g_num_user_presets].text = dup_text;
   g_user_presets[g_num_user_presets].help = dup_help;
   g_user_presets[g_num_user_presets].kind = kind;
   g_num_user_presets++;

   return 0;
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

char *
hypredrv_PresetHelp(void)
{
   return hypredrv_PresetHelpTyped(HYPREDRV_PRESET_PRECON);
}

/* Write the "    - name: help" lines of the presets of the given kind into out
 * (capacity cap; out may be NULL to measure) and return their length. */
static size_t
PresetHelpLines(const hypredrv_Preset *presets, size_t n, hypredrv_PresetKind kind,
                char *out, size_t cap)
{
   size_t len = 0;
   for (size_t i = 0; i < n; i++)
   {
      if (presets[i].kind != kind)
      {
         continue;
      }
      int w = snprintf(out ? out + len : NULL, out ? cap - len : 0, "    - %s: %s\n",
                       presets[i].name ? presets[i].name : "",
                       presets[i].help ? presets[i].help : "");
      len += (w > 0) ? (size_t)w : 0;
   }
   return len;
}

char *
hypredrv_PresetHelpTyped(hypredrv_PresetKind kind)
{
   const size_t nb       = sizeof(g_presets) / sizeof(g_presets[0]);
   const char  *header   = PresetKindHeader(kind);
   size_t       head_len = strlen(header);
   size_t       nb_len   = PresetHelpLines(g_presets, nb, kind, NULL, 0);
   size_t nu_len = PresetHelpLines(g_user_presets, g_num_user_presets, kind, NULL, 0);
   size_t bytes  = head_len + nb_len + nu_len + 1;

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

/*-----------------------------------------------------------------------------
 * Find preconditioner preset by name (case-insensitive).
 *-----------------------------------------------------------------------------*/

const hypredrv_Preset *
hypredrv_PresetFind(const char *name)
{
   return hypredrv_PresetFindTyped(name, HYPREDRV_PRESET_ANY);
}

const hypredrv_Preset *
hypredrv_PresetFindTyped(const char *name, hypredrv_PresetKind kind)
{
   if (!name)
   {
      return NULL;
   }

   char *lower = strdup(name);
   if (!lower)
   {
      return NULL;
   }

   normalize_preset_name(lower);

   const hypredrv_Preset *match =
      PresetFindIn(g_presets, sizeof(g_presets) / sizeof(g_presets[0]), lower, kind);
   if (!match)
   {
      match = PresetFindIn(g_user_presets, g_num_user_presets, lower, kind);
   }

   free(lower);
   return match;
}
