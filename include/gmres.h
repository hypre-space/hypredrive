/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef GMRES_HEADER
#define GMRES_HEADER

#include "yaml.h"
#include "field.h"
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
   HYPRE_Int     krylov_dim;
   HYPRE_Int     rel_change;
   HYPRE_Int     logging;
   HYPRE_Int     print_level;
   HYPRE_Real    relative_tol;
   HYPRE_Real    absolute_tol;
   HYPRE_Real    conv_fac_tol;
} GMRES_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void GMRESSetArgs(void*, YAMLnode*);
void GMRESCreate(MPI_Comm, GMRES_args*, HYPRE_Solver*);

#endif
