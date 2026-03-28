/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef FGMRES_HEADER
#define FGMRES_HEADER

#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "internal/field.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * FGMRES solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct FGMRES_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  krylov_dim;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
} FGMRES_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_FGMRESSetDefaultArgs(FGMRES_args *);
void hypredrv_FGMRESSetArgs(void *, const YAMLnode *);
void hypredrv_FGMRESCreate(MPI_Comm, const FGMRES_args *, HYPRE_Solver *);

#endif /* FGMRES_HEADER */
