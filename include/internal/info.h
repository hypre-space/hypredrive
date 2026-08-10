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

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

void        hypredrv_PrintLibInfo(MPI_Comm, int);
void        hypredrv_PrintSystemInfo(MPI_Comm);
const char *hypredrv_ExecutionPolicyName(int);
void        hypredrv_PrintExecutionPolicy(MPI_Comm, int, FILE *);
void        hypredrv_PrintExitInfo(MPI_Comm, const char *);

#endif /* INFO_HEADER */
