/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef GMRES_HEADER
#define GMRES_HEADER

#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "field.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * GMRES solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct GMRES_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  stop_crit;
   HYPRE_Int  skip_real_res_check;
   HYPRE_Int  krylov_dim;
   HYPRE_Int  rel_change;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
   HYPRE_Real conv_fac_tol;
} GMRES_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void GMRESSetArgs(void *, const YAMLnode *);
void GMRESCreate(MPI_Comm, const GMRES_args *, HYPRE_Solver *);

#endif /* GMRES_HEADER */
