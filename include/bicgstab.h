/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef BICGSTAB_HEADER
#define BICGSTAB_HEADER

#include "yaml.h"
#include "field.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * BiCGSTAB solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct BiCGSTAB_args_struct {
   HYPRE_Int     min_iter;
   HYPRE_Int     max_iter;
   HYPRE_Int     stop_crit;
   HYPRE_Int     logging;
   HYPRE_Int     print_level;
   HYPRE_Real    relative_tol;
   HYPRE_Real    absolute_tol;
   HYPRE_Real    conv_fac_tol;
} BiCGSTAB_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void BiCGSTABSetArgs(void*, YAMLnode*);
void BiCGSTABCreate(MPI_Comm, BiCGSTAB_args*, HYPRE_Solver*);

#endif
