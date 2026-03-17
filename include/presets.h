/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_PRESETS_HEADER
#define HYPREDRV_PRESETS_HEADER

#include <stddef.h>

typedef struct hypredrv_preset_struct
{
   const char *name; /* lowercase key */
   const char *text; /* YAML snippet that expands under `preconditioner:` */
   const char *help; /* short help text */
} hypredrv_Preset;

/* Convenience initializer macro (keeps fields consistent when adding presets). */
#define HYPREDRV_PRESET(_id, _help)                       \
   (hypredrv_Preset)                                      \
   {                                                      \
      .name = #_id, .text = preset_##_id, .help = (_help) \
   }

/* Returns NULL if name is unknown. Name matching is case-insensitive. */
const hypredrv_Preset *hypredrv_PresetFind(const char *name);

/* Returns a formatted help string listing available presets (caller must free). */
char *hypredrv_PresetHelp(void);

/* Registers a new user preset. Returns 0 on success, -1 on failure. */
int hypredrv_PresetRegister(const char *name, const char *yaml_text, const char *help);

/* Frees all user-registered presets. Called from HYPREDRV_Finalize(). */
void hypredrv_PresetFreeUserPresets(void);

#endif /* HYPREDRV_PRESETS_HEADER */
