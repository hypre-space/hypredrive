/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_DIAGNOSTICS_HEADER
#define HYPREDRV_DIAGNOSTICS_HEADER

#include "HYPREDRV.h"
#include "internal/mgr.h"

void hypredrv_SetPendingSolvePathContext(HYPREDRV_t hypredrv);
void hypredrv_PrintStatsWithConfiguredDestination(HYPREDRV_t hypredrv, int print_level);
void hypredrv_MaybeDumpLinearSystem(HYPREDRV_t hypredrv, int stage);
void hypredrv_ReportSolveDiagnostics(HYPREDRV_t hypredrv, double *x_norm,
                                     double *xref_norm, double *e_norm);
#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
void hypredrv_LogMGRCachedHandles(HYPREDRV_t hypredrv, const MGR_args *mgr,
                                  const char *msg);
#endif

#endif /* HYPREDRV_DIAGNOSTICS_HEADER */
