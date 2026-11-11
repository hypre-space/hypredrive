/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_PRESETS_HEADER
#define HYPREDRV_PRESETS_HEADER

#include <stdint.h>
#include "internal/yaml.h"

typedef enum
{
   HYPREDRV_PRESET_PRECON = 0,
   HYPREDRV_PRESET_SOLVER = 1,
} hypredrv_PresetKind;

/* Load exactly one known solver/preconditioner type. The returned node is owned
 * by *tree_out; callers parse method options and then destroy that tree. Failure
 * returns NULL, destroys *tree_out, and preserves diagnostics. Prior failures
 * abort without loading. Names are case-insensitive, with '-' and '_' equivalent.
 * Registry storage stays private. */
YAMLnode *hypredrv_PresetLoadTypeNode(const char *, hypredrv_PresetKind, YAMLtree **,
                                      int *);

/* Returns a formatted help string listing available presets (caller must free). */
char *hypredrv_PresetHelpTyped(hypredrv_PresetKind kind);

/* Validate identity and copy a complete registration before publication. Return
 * this operation's status without resetting the caller's diagnostic scope. */
uint32_t hypredrv_PresetRegisterTyped(const char *, const char *, const char *,
                                      hypredrv_PresetKind);

/* Frees all user-registered presets. Called from HYPREDRV_Finalize(). */
void hypredrv_PresetFreeUserPresets(void);

#endif /* HYPREDRV_PRESETS_HEADER */
