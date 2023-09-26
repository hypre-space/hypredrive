/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef AMG_HEADER
#define AMG_HEADER

#include "yaml.h"
#include "field.h"
#include "ilu.h"
#include "fsai.h"
#include "cheby.h"
#include "HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * AMG complex smoother arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGsmt_args_struct {
   HYPRE_Int    type;
   HYPRE_Int    num_levels;
   HYPRE_Int    num_sweeps;
   FSAI_args    fsai;
   ILU_args     ilu;
} AMGsmt_args;

/*--------------------------------------------------------------------------
 * AMG relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGrlx_args_struct {
   HYPRE_Int    down_type;
   HYPRE_Int    up_type;
   HYPRE_Int    coarse_type;
   HYPRE_Int    down_sweeps;
   HYPRE_Int    up_sweeps;
   HYPRE_Int    coarse_sweeps;
   HYPRE_Int    num_sweeps;
   HYPRE_Int    order;
   HYPRE_Real   weight;
   HYPRE_Real   outer_weight;
   Cheby_args   chebyshev;
} AMGrlx_args;

/*--------------------------------------------------------------------------
 * AMG aggressive coarsening arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGagg_args_struct {
   HYPRE_Int    num_levels;
   HYPRE_Int    num_paths;
   HYPRE_Int    prolongation_type;
   HYPRE_Int    max_nnz_row;
   HYPRE_Int    P12_max_elements;
   HYPRE_Real   P12_trunc_factor;
   HYPRE_Real   trunc_factor;
} AMGagg_args;

/*--------------------------------------------------------------------------
 * AMG coarsening arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGcsn_args_struct {
   HYPRE_Int    type;
   HYPRE_Int    rap2;
   HYPRE_Int    mod_rap2;
   HYPRE_Int    keep_transpose;
   HYPRE_Int    num_functions;
   HYPRE_Int    seq_amg_th;
   HYPRE_Int    min_coarse_size;
   HYPRE_Int    max_coarse_size;
   HYPRE_Int    max_levels;
   HYPRE_Real   strong_th;
} AMGcsn_args;

/*--------------------------------------------------------------------------
 * AMG interpolation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMGint_args_struct {
   HYPRE_Int    prolongation_type;
   HYPRE_Int    restriction_type;
   HYPRE_Int    max_nnz_row;
   HYPRE_Real   max_row_sum;
   HYPRE_Real   trunc_factor;
} AMGint_args;

/*--------------------------------------------------------------------------
 * AMG preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMG_args_struct {
   AMGint_args  interpolation;
   AMGagg_args  aggressive;
   AMGcsn_args  coarsening;
   AMGrlx_args  relaxation;
   AMGsmt_args  smoother;

   HYPRE_Int    max_iter;
   HYPRE_Int    print_level;
   HYPRE_Real   tolerance;
} AMG_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void AMGSetArgs(void*, YAMLnode*);
void AMGCreate(AMG_args*, HYPRE_Solver*);

#endif
