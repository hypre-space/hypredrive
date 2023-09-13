/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef LINSYS_HEADER
#define LINSYS_HEADER

#include "args.h"
#include "utils.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

int LinearSystemReadMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix*);
int LinearSystemSetRHS(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJVector*);
int LinearSystemSetInitialGuess(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector*);
int LinearSystemSetPrecMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJMatrix*);
int LinearSystemReadDofmap(MPI_Comm, LS_args*, HYPRE_IntArray*);

#endif
