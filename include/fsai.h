/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef FSAI_HEADER
#define FSAI_HEADER

#include "yaml.h"
#include "field.h"
#include "HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * FSAI preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct FSAI_args_struct {
   HYPRE_Int     max_iter;
   HYPRE_Int     print_level;
   HYPRE_Int     algo_type;
   HYPRE_Int     ls_type;
   HYPRE_Int     max_steps;
   HYPRE_Int     max_step_size;
   HYPRE_Int     max_nnz_row;
   HYPRE_Int     num_levels;
   HYPRE_Int     eig_max_iters;
   HYPRE_Real    threshold;
   HYPRE_Real    kap_tolerance;
   HYPRE_Real    tolerance;
} FSAI_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void FSAISetArgs(void*, YAMLnode*);
void FSAICreate(FSAI_args*, HYPRE_Solver*);

#endif /* FSAI_HEADER */
