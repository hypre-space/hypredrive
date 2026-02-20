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

void PrintUsage(const char *);
void PrintLibInfo(MPI_Comm, int);
void PrintSystemInfo(MPI_Comm);
void PrintExitInfo(MPI_Comm, const char *);

#endif /* INFO_HEADER */
