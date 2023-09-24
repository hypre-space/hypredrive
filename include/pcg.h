/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef PCG_HEADER
#define PCG_HEADER

#include "yaml.h"
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
   HYPRE_Real    rtol;
   HYPRE_Real    atol;
   HYPRE_Real    res_tol;
   HYPRE_Real    cf_tol;
} PCG_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void PCGSetArgs(void*, YAMLnode*);
void PCGCreate(MPI_Comm, PCG_args*, HYPRE_Solver*);

#endif
