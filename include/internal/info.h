/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef INFO_HEADER
#define INFO_HEADER

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_PrintUsage(const char *);
void hypredrv_PrintLibInfo(MPI_Comm, int);
void hypredrv_PrintSystemInfo(MPI_Comm);
void hypredrv_PrintExitInfo(MPI_Comm, const char *);

#endif /* INFO_HEADER */
