/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef BICGSTAB_HEADER
#define BICGSTAB_HEADER

#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "field.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * BiCGSTAB solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct BiCGSTAB_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  stop_crit;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
   HYPRE_Real conv_fac_tol;
} BiCGSTAB_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void BiCGSTABSetArgs(void *, YAMLnode *);
void BiCGSTABCreate(MPI_Comm, BiCGSTAB_args *, HYPRE_Solver *);

#endif /* BICGSTAB_HEADER */
