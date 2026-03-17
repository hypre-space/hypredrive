/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "presets.h"
#include "error.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Pre-defined preconditioner variants */
static const char preset_poisson[] = "amg"; /* Note: no ":" required */

static const char preset_elasticity_2d[] = "amg:\n"
                                           "  coarsening:\n"
                                           "    num_functions: 2\n"
                                           "    strong_th: 0.8";

static const char preset_elasticity_3d[] = "amg:\n"
                                           "  coarsening:\n"
                                           "    num_functions: 3\n"
                                           "    strong_th: 0.8";

static const hypredrv_Preset g_presets[] = {
   HYPREDRV_PRESET(poisson, "BoomerAMG for poisson"),
   HYPREDRV_PRESET(elasticity_2d, "BoomerAMG for 2D elasticity"),
   HYPREDRV_PRESET(elasticity_3d, "BoomerAMG for 3D elasticity"),
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

/*-----------------------------------------------------------------------------
 * Count number of pre-defined preconditioner variants (presets).
 *-----------------------------------------------------------------------------*/

static size_t
hypredrv_PresetCount(void)
{
   return (sizeof(g_presets) / sizeof(g_presets[0])) + g_num_user_presets;
}

/*-----------------------------------------------------------------------------
 * Register a new user-defined preset. Returns 0 on success, -1 on failure.
 *-----------------------------------------------------------------------------*/

int
hypredrv_PresetRegister(const char *name, const char *yaml_text, const char *help)
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
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return -1;
   }
   normalize_preset_name(norm);

   /* Check for duplicates in built-in presets */
   for (size_t i = 0; i < sizeof(g_presets) / sizeof(g_presets[0]); i++)
   {
      if (!strcmp(norm, g_presets[i].name))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "hypredrv_PresetRegister: preset '%s' conflicts with built-in preset", norm);
         free(norm);
         return -1;
      }
   }

   /* Check for duplicates in user presets */
   for (size_t i = 0; i < g_num_user_presets; i++)
   {
      if (!strcmp(norm, g_user_presets[i].name))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("hypredrv_PresetRegister: preset '%s' already registered",
                              norm);
         free(norm);
         return -1;
      }
   }

   /* Grow array if needed */
   if (g_num_user_presets >= g_cap_user_presets)
   {
      size_t           new_cap = g_cap_user_presets ? g_cap_user_presets * 2 : 8;
      hypredrv_Preset *tmp =
         (hypredrv_Preset *)realloc(g_user_presets, new_cap * sizeof(hypredrv_Preset));
      if (!tmp)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
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
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      return -1;
   }

   g_user_presets[g_num_user_presets].name = norm;
   g_user_presets[g_num_user_presets].text = dup_text;
   g_user_presets[g_num_user_presets].help = dup_help;
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
      free((char *)g_user_presets[i].name);
      free((char *)g_user_presets[i].text);
      free((char *)g_user_presets[i].help);
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
   const size_t nb     = sizeof(g_presets) / sizeof(g_presets[0]);
   const char  *header = "Available preconditioner presets:\n";

   size_t bytes = strlen(header) + 1;
   for (size_t i = 0; i < nb; i++)
   {
      const char *pname = g_presets[i].name ? g_presets[i].name : "";
      const char *phelp = g_presets[i].help ? g_presets[i].help : "";
      bytes +=
         strlen("    - ") + strlen(pname) + strlen(": ") + strlen(phelp) + strlen("\n");
   }
   for (size_t i = 0; i < g_num_user_presets; i++)
   {
      const char *pname = g_user_presets[i].name ? g_user_presets[i].name : "";
      const char *phelp = g_user_presets[i].help ? g_user_presets[i].help : "";
      bytes +=
         strlen("    - ") + strlen(pname) + strlen(": ") + strlen(phelp) + strlen("\n");
   }

   char *out = (char *)malloc(bytes);
   if (!out)
   {
      return NULL;
   }
   out[0] = '\0';

   (void)strcat(out, header);
   for (size_t i = 0; i < nb; i++)
   {
      char line[1024];
      (void)snprintf(line, sizeof(line), "    - %s: %s\n", g_presets[i].name,
                     g_presets[i].help);
      (void)strcat(out, line);
   }
   for (size_t i = 0; i < g_num_user_presets; i++)
   {
      char line[1024];
      (void)snprintf(line, sizeof(line), "    - %s: %s\n", g_user_presets[i].name,
                     g_user_presets[i].help);
      (void)strcat(out, line);
   }
   return out;
}

/*-----------------------------------------------------------------------------
 * Find preconditioner preset by name (case-insensitive).
 *-----------------------------------------------------------------------------*/

const hypredrv_Preset *
hypredrv_PresetFind(const char *name)
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

   /* Normalize: lowercase and treat '-' as '_' so we can use identifier-style keys. */
   for (char *p = lower; *p; ++p)
   {
      *p = (char)tolower((unsigned char)*p);
      if (*p == '-')
      {
         *p = '_';
      }
   }

   const hypredrv_Preset *match = NULL;
   for (size_t i = 0; i < (sizeof(g_presets) / sizeof(g_presets[0])); i++)
   {
      if (!strcmp(lower, g_presets[i].name))
      {
         match = &g_presets[i];
         break;
      }
   }

   if (!match)
   {
      for (size_t i = 0; i < g_num_user_presets; i++)
      {
         if (!strcmp(lower, g_user_presets[i].name))
         {
            match = &g_user_presets[i];
            break;
         }
      }
   }

   free(lower);
   return match;
}
