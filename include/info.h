/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
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
void PrintSystemInfo(void);
void PrintExitInfo(const char*);

#endif /* INFO_HEADER */
