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
   int         kind; /* hypredrv_PresetKind */
} hypredrv_Preset;

typedef enum
{
   HYPREDRV_PRESET_ANY    = -1,
   HYPREDRV_PRESET_PRECON = 0,
   HYPREDRV_PRESET_SOLVER = 1,
} hypredrv_PresetKind;

/* Convenience initializer macro (keeps fields consistent when adding presets). */
#define HYPREDRV_PRESET(_id, _kind, _help)                                 \
   (hypredrv_Preset)                                                       \
   {                                                                       \
      .name = #_id, .text = preset_##_id, .help = (_help), .kind = (_kind) \
   }

/* Returns NULL if name is unknown. Name matching is case-insensitive. */
const hypredrv_Preset *hypredrv_PresetFind(const char *name);
const hypredrv_Preset *hypredrv_PresetFindTyped(const char         *name,
                                                hypredrv_PresetKind kind);

/* Returns a formatted help string listing available presets (caller must free). */
char *hypredrv_PresetHelp(void);
char *hypredrv_PresetHelpTyped(hypredrv_PresetKind kind);

/* Registers a new user preset. Returns 0 on success, -1 on failure. */
int hypredrv_PresetRegister(const char *name, const char *yaml_text, const char *help);
int hypredrv_PresetRegisterTyped(const char *name, const char *yaml_text,
                                 const char *help, hypredrv_PresetKind kind);

/* Frees all user-registered presets. Called from HYPREDRV_Finalize(). */
void hypredrv_PresetFreeUserPresets(void);

#endif /* HYPREDRV_PRESETS_HEADER */
