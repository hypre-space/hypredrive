/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef ILU_HEADER
#define ILU_HEADER

#include "yaml.h"
#include "HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * ILU preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct ILU_args_struct {
   HYPRE_Int     max_iter;
   HYPRE_Int     print_level;
   HYPRE_Int     type;
   HYPRE_Int     fill_level;
   HYPRE_Int     reordering;
   HYPRE_Int     tri_solve;
   HYPRE_Int     lower_jac_iters;
   HYPRE_Int     upper_jac_iters;
   HYPRE_Int     max_row_nnz;
   HYPRE_Int     schur_max_iter;
   HYPRE_Real    droptol;
} ILU_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void ILUSetArgs(void*, YAMLnode*);
void ILUCreate(ILU_args*, HYPRE_Solver*);

#endif
