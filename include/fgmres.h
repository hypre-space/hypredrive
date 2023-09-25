/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef FGMRES_HEADER
#define FGMRES_HEADER

#include "yaml.h"
#include "field.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * FGMRES solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct FGMRES_args_struct {
   HYPRE_Int     min_iter;
   HYPRE_Int     max_iter;
   HYPRE_Int     krylov_dim;
   HYPRE_Int     logging;
   HYPRE_Int     print_level;
   HYPRE_Real    relative_tol;
   HYPRE_Real    absolute_tol;
} FGMRES_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void FGMRESSetArgs(void*, YAMLnode*);
void FGMRESCreate(MPI_Comm, FGMRES_args*, HYPRE_Solver*);

#endif
