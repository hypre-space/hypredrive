/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PCG_HEADER
#define PCG_HEADER

#include "yaml.h"
#include "field.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * PCG solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct PCG_args_struct {
   HYPRE_Int     max_iter;
   HYPRE_Int     two_norm;
   HYPRE_Int     stop_crit;
   HYPRE_Int     rel_change;
   HYPRE_Int     print_level;
   HYPRE_Int     recompute_res;
   HYPRE_Real    relative_tol;
   HYPRE_Real    absolute_tol;
   HYPRE_Real    residual_tol;
   HYPRE_Real    conv_fac_tol;
} PCG_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void PCGSetArgs(void*, YAMLnode*);
void PCGCreate(MPI_Comm, PCG_args*, HYPRE_Solver*);

#endif /* PCG_HEADER */
