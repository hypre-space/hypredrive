/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_EXECUTION_HEADER
#define HYPREDRV_EXECUTION_HEADER

#include <stddef.h>
#include "HYPREDRV.h"
#include "internal/precon.h"

void     hypredrv_LogExecutionPolicy(HYPREDRV_t hypredrv);
int      hypredrv_ValidateDevicePreconditioner(HYPREDRV_t hypredrv, int device_requested,
                                               precon_t method, const precon_args *args);
uint32_t hypredrv_ApplyGlobalRuntimeSettings(HYPREDRV_t hypredrv);
uint32_t hypredrv_ApplyConfiguredDeviceInitialization(HYPREDRV_t hypredrv);
void hypredrv_PrepareExplicitObjectForConfiguredExecution(HYPREDRV_t hypredrv, void *obj,
                                                          int is_matrix);
void hypredrv_PrepareExplicitObjectsForConfiguredExecution(HYPREDRV_t   hypredrv,
                                                           void *const *objects,
                                                           size_t       num_objects,
                                                           int          is_matrix);

#endif /* HYPREDRV_EXECUTION_HEADER */
