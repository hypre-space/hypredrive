/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef CHEBY_HEADER
#define CHEBY_HEADER

#include "HYPRE_parcsr_ls.h"
#include "internal/field.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * Chebyshev smoother arguments struct
 *--------------------------------------------------------------------------*/

typedef struct Cheby_args_struct
{
   HYPRE_Int  order;
   HYPRE_Int  eig_est;
   HYPRE_Int  variant;
   HYPRE_Int  scale;
   HYPRE_Real fraction;
} Cheby_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_ChebySetDefaultArgs(Cheby_args *);
void hypredrv_ChebySetArgs(void *, const YAMLnode *);

#endif /* CHEBY_HEADER */
