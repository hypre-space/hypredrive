/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef SCHWARZ_HEADER
#define SCHWARZ_HEADER

#include "HYPRE_utilities.h"
#include "internal/field.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * Schwarz preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct Schwarz_args_struct
{
   HYPRE_Int  variant;
   HYPRE_Int  overlap;
   HYPRE_Int  domain_type;
   HYPRE_Int  num_functions;
   HYPRE_Int  use_nonsymm;
   HYPRE_Int  local_solver_type;
   HYPRE_Int  iluk_level_of_fill;
   HYPRE_Int  ilut_max_nnz_row;
   HYPRE_Int  max_iter;
   HYPRE_Int  print_level;
   HYPRE_Int  logging;
   HYPRE_Real relax_weight;
   HYPRE_Real ilut_droptol;
   HYPRE_Real tolerance;
} Schwarz_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_SchwarzSetDefaultArgs(Schwarz_args *);
void hypredrv_SchwarzSetArgs(void *, const YAMLnode *);
void hypredrv_SchwarzCreate(const Schwarz_args *, HYPRE_Solver *);

#endif /* SCHWARZ_HEADER */
