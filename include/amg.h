/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef AMG_HEADER
#define AMG_HEADER

#include "yaml.h"
#include "ilu.h"
#include "HYPRE_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * AMG preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct AMG_args_struct {
   HYPRE_Int  interp_type;
   HYPRE_Real trunc_factor;
   HYPRE_Int  pmax;
   HYPRE_Int  restrict_type;
   HYPRE_Int  rap2;
   HYPRE_Int  mod_rap2;
   HYPRE_Int  keep_transpose;

   HYPRE_Int  coarsen_type;
   HYPRE_Real strong_th;
   HYPRE_Real max_row_sum;
   HYPRE_Int  num_functions;

   HYPRE_Int  agg_num_levels;
   HYPRE_Int  agg_num_paths;
   HYPRE_Int  agg_interp_type;
   HYPRE_Real agg_trunc_factor;
   HYPRE_Real agg_P12_trunc_factor;
   HYPRE_Int  agg_pmax;
   HYPRE_Int  agg_P12_max_elements;

   HYPRE_Int  relax_num_sweeps;
   HYPRE_Int  relax_down;
   HYPRE_Int  relax_up;
   HYPRE_Int  relax_coarse;
   HYPRE_Int  relax_down_sweeps;
   HYPRE_Int  relax_up_sweeps;
   HYPRE_Int  relax_coarse_sweeps;
   HYPRE_Int  relax_order;
   HYPRE_Real relax_weight;
   HYPRE_Real relax_outer_weight;

   HYPRE_Int  cheby_order;
   HYPRE_Real cheby_fraction;
   HYPRE_Int  cheby_eig_est;
   HYPRE_Int  cheby_variant;
   HYPRE_Int  cheby_scale;

   HYPRE_Int  fsai_algo_type;
   HYPRE_Int  fsai_ls_type;
   HYPRE_Int  fsai_max_steps;
   HYPRE_Int  fsai_max_step_size;
   HYPRE_Int  fsai_max_nnz_row;
   HYPRE_Int  fsai_num_levels;
   HYPRE_Int  fsai_eig_max_iters;
   HYPRE_Real fsai_th;
   HYPRE_Real fsai_kap_tol;

   ILU_args   ilu;

   HYPRE_Int  smooth_type;
   HYPRE_Int  smooth_num_levels;
   HYPRE_Int  smooth_num_sweeps;

   HYPRE_Int  seq_th;
   HYPRE_Int  max_coarse_size;
   HYPRE_Int  min_coarse_size;
   HYPRE_Int  max_levels;

   HYPRE_Real tol;
   HYPRE_Int  max_iter;
   HYPRE_Int  print_level;
} AMG_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

int AMGSetDefaultArgs(AMG_args*);
int AMGSetArgsFromYAML(AMG_args*, YAMLnode*);
int AMGSetArgs(AMG_args*, YAMLnode*);
int AMGCreate(AMG_args*, HYPRE_Solver*);

#endif
