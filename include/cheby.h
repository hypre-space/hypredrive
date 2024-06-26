/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef CHEBY_HEADER
#define CHEBY_HEADER

#include "yaml.h"
#include "field.h"
#include "HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * Chebyshev smoother arguments struct
 *--------------------------------------------------------------------------*/

typedef struct Cheby_args_struct {
   HYPRE_Int    order;
   HYPRE_Int    eig_est;
   HYPRE_Int    variant;
   HYPRE_Int    scale;
   HYPRE_Real   fraction;
} Cheby_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void ChebySetDefaultArgs(Cheby_args*);
void ChebySetArgs(void*, YAMLnode*);

#endif /* CHEBY_HEADER */
