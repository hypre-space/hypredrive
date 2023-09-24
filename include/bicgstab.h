/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef BICGSTAB_HEADER
#define BICGSTAB_HEADER

#include "yaml.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * BICGSTAB solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct BICGSTAB_args_struct {
   HYPRE_Int     min_iter;
   HYPRE_Int     max_iter;
   HYPRE_Int     stop_crit;
   HYPRE_Int     logging;
   HYPRE_Int     print_level;
   HYPRE_Real    rtol;
   HYPRE_Real    atol;
   HYPRE_Real    cf_tol;
} BICGSTAB_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void BiCGSTABSetDefaultArgs(BICGSTAB_args*);
void BiCGSTABSetArgsFromYAML(void*, YAMLnode*);
void BiCGSTABCreate(MPI_Comm, BICGSTAB_args*, HYPRE_Solver*);

#endif
