/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "presets.h"

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

/*-----------------------------------------------------------------------------
 * Count number of pre-defined preconditioner variants (presets).
 *-----------------------------------------------------------------------------*/

static size_t
hypredrv_PresetCount(void)
{
   return sizeof(g_presets) / sizeof(g_presets[0]);
}

/*-----------------------------------------------------------------------------
 * Generate help string listing available presets (caller must free).
 *-----------------------------------------------------------------------------*/

char *
hypredrv_PresetHelp(void)
{
   const size_t n      = hypredrv_PresetCount();
   const char  *header = "Available preconditioner presets:\n";

   size_t bytes = strlen(header) + 1;
   for (size_t i = 0; i < n; i++)
   {
      const char *name = g_presets[i].name ? g_presets[i].name : "";
      const char *help = g_presets[i].help ? g_presets[i].help : "";
      bytes +=
         strlen("    - ") + strlen(name) + strlen(": ") + strlen(help) + strlen("\n");
   }

   char *out = (char *)malloc(bytes);
   if (!out)
   {
      return NULL;
   }
   out[0] = '\0';

   (void)strcat(out, header);
   for (size_t i = 0; i < n; i++)
   {
      char line[1024];
      (void)snprintf(line, sizeof(line), "    - %s: %s\n", g_presets[i].name,
                     g_presets[i].help);
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

   free(lower);
   return match;
}
