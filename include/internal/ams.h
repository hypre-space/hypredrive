/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef AMS_HEADER
#define AMS_HEADER

#include "HYPRE_parcsr_ls.h"
#include "HYPRE_IJ_mv.h"
#include "internal/field.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * AMS (auxiliary-space Maxwell solver) preconditioner arguments struct
 *
 * Flat mirror of the scalar options accepted by HYPRE_AMSSet*. The grouped
 * options (smoothing, Chebyshev, alpha/beta AMG) are stored as individual
 * fields here and re-assembled into the corresponding multi-argument hypre
 * calls in hypredrv_AMSCreate. Defaults match hypre's internal defaults
 * (hypre_AMSCreate), except max_iter/tolerance/print_level which are tuned
 * for preconditioner use (single cycle, quiet).
 *--------------------------------------------------------------------------*/

typedef struct AMS_args_struct
{
   HYPRE_Int  dimension;
   HYPRE_Int  max_iter;
   HYPRE_Int  print_level;
   HYPRE_Int  cycle_type;
   HYPRE_Real tolerance;

   /* Smoothing on the original matrix. Chebyshev smoothing is available by
    * selecting a Chebyshev relax_type; hypre exposes no public setter to tune
    * the Chebyshev order/fraction for AMS, so those are left at hypre defaults. */
   HYPRE_Int  relax_type;
   HYPRE_Int  relax_times;
   HYPRE_Real relax_weight;
   HYPRE_Real omega;

   HYPRE_Int proj_freq;

   /* AMG options for the vector Poisson problem (alpha / Pi space) */
   HYPRE_Int  alpha_coarsen_type;
   HYPRE_Int  alpha_agg_levels;
   HYPRE_Int  alpha_relax_type;
   HYPRE_Real alpha_strength_threshold;
   HYPRE_Int  alpha_interp_type;
   HYPRE_Int  alpha_Pmax;
   HYPRE_Int  alpha_coarse_relax_type;

   /* AMG options for the scalar Poisson problem (beta / G space) */
   HYPRE_Int  beta_coarsen_type;
   HYPRE_Int  beta_agg_levels;
   HYPRE_Int  beta_relax_type;
   HYPRE_Real beta_strength_threshold;
   HYPRE_Int  beta_interp_type;
   HYPRE_Int  beta_Pmax;
   HYPRE_Int  beta_coarse_relax_type;
} AMS_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_AMSSetDefaultArgs(AMS_args *);
void hypredrv_AMSSetArgs(void *, const YAMLnode *);
void hypredrv_AMSCreate(const AMS_args *, HYPRE_Solver *);

/* Inject the operator inputs AMS requires (discrete gradient + vertex
 * coordinate vectors). Arguments are IJ handles (may be NULL); ParCSR/ParVector
 * objects are extracted internally. Must be called after AMSCreate and before
 * setup. */
void hypredrv_AMSSetOperators(HYPRE_Solver, HYPRE_IJMatrix G, HYPRE_IJVector x,
                             HYPRE_IJVector y, HYPRE_IJVector z);

#endif /* AMS_HEADER */
