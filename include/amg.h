/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef AMG_HEADER
#define AMG_HEADER

#include "HYPRE_parcsr_ls.h"
#include "cheby.h"
#include "field.h"
#include "fsai.h"
#include "ilu.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * AMG complex smoother arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGsmt_args_struct
{
   HYPRE_Int type;
   HYPRE_Int num_levels;
   HYPRE_Int num_sweeps;
   FSAI_args fsai;
   ILU_args  ilu;
} AMGsmt_args;

/*--------------------------------------------------------------------------
 * AMG relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGrlx_args_struct
{
   HYPRE_Int  down_type;
   HYPRE_Int  up_type;
   HYPRE_Int  coarse_type;
   HYPRE_Int  down_sweeps;
   HYPRE_Int  up_sweeps;
   HYPRE_Int  coarse_sweeps;
   HYPRE_Int  num_sweeps;
   HYPRE_Int  order;
   HYPRE_Real weight;
   HYPRE_Real outer_weight;
   Cheby_args chebyshev;
} AMGrlx_args;

/*--------------------------------------------------------------------------
 * AMG aggressive coarsening arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGagg_args_struct
{
   HYPRE_Int  num_levels;
   HYPRE_Int  num_paths;
   HYPRE_Int  prolongation_type;
   HYPRE_Int  max_nnz_row;
   HYPRE_Int  P12_max_elements;
   HYPRE_Real P12_trunc_factor;
   HYPRE_Real trunc_factor;
} AMGagg_args;

/*--------------------------------------------------------------------------
 * AMG coarsening arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGcsn_args_struct
{
   HYPRE_Int  type;
   HYPRE_Int  rap2;
   HYPRE_Int  mod_rap2;
   HYPRE_Int  keep_transpose;
   HYPRE_Int  num_functions;
   HYPRE_Int  filter_functions;
   HYPRE_Int  nodal;
   HYPRE_Int  seq_amg_th;
   HYPRE_Int  min_coarse_size;
   HYPRE_Int  max_coarse_size;
   HYPRE_Int  max_levels;
   HYPRE_Real max_row_sum;
   HYPRE_Real strong_th;
} AMGcsn_args;

/*--------------------------------------------------------------------------
 * AMG interpolation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGint_args_struct
{
   HYPRE_Int  prolongation_type;
   HYPRE_Int  restriction_type;
   HYPRE_Int  max_nnz_row;
   HYPRE_Real trunc_factor;
} AMGint_args;

/*--------------------------------------------------------------------------
 * AMG preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMG_args_struct
{
   AMGint_args interpolation;
   AMGagg_args aggressive;
   AMGcsn_args coarsening;
   AMGrlx_args relaxation;
   AMGsmt_args smoother;

   HYPRE_Int  max_iter;
   HYPRE_Int  print_level;
   HYPRE_Real tolerance;

   HYPRE_Int       num_rbms;
   HYPRE_ParVector rbms[3];
} AMG_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void AMGSetDefaultArgs(AMG_args *);
void AMGSetArgs(void *, const YAMLnode *);
void AMGCreate(const AMG_args *, HYPRE_Solver *);
void AMGSetRBMs(AMG_args *, HYPRE_IJVector);

#endif /* AMG_HEADER */
