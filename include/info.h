/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef INFO_HEADER
#define INFO_HEADER

#include <stdio.h>
#include <time.h>
#include "HYPRE.h"
#include "HYPRE_config.h"

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

void PrintUsage(const char*);
void PrintLibInfo(void);
void PrintExitInfo(const char*);

#endif /* INFO_HEADER */
