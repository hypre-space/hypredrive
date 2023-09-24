/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef GMRES_HEADER
#define GMRES_HEADER

#include "yaml.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * GMRES solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct GMRES_args_struct {
   HYPRE_Int     min_iter;
   HYPRE_Int     max_iter;
   HYPRE_Int     stop_crit;
   HYPRE_Int     skip_real_res_check;
   HYPRE_Int     krylov_dimension;
   HYPRE_Int     rel_change;
   HYPRE_Int     logging;
   HYPRE_Int     print_level;
   HYPRE_Real    rtol;
   HYPRE_Real    atol;
   HYPRE_Real    cf_tol;
} GMRES_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void GMRESSetDefaultArgs(GMRES_args*);
void GMRESSetArgsFromYAML(void*, YAMLnode*);
void GMRESCreate(MPI_Comm, GMRES_args*, HYPRE_Solver*);

#endif
