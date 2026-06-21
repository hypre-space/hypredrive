/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef ADS_HEADER
#define ADS_HEADER

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "internal/field.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * ADS (auxiliary-space divergence solver) preconditioner arguments struct
 *
 * Flat mirror of the scalar options accepted by HYPRE_ADSSet*. Grouped options
 * (smoothing, Chebyshev, AMS block, AMG block) are stored as individual fields
 * and re-assembled into the corresponding hypre calls in hypredrv_ADSCreate.
 * Defaults match hypre's internal defaults (hypre_ADSCreate), except
 * max_iter/tolerance/print_level which are tuned for preconditioner use.
 *--------------------------------------------------------------------------*/

typedef struct ADS_args_struct
{
   HYPRE_Int  max_iter;
   HYPRE_Int  print_level;
   HYPRE_Int  cycle_type;
   HYPRE_Real tolerance;

   /* Smoothing on the original matrix */
   HYPRE_Int  relax_type;
   HYPRE_Int  relax_times;
   HYPRE_Real relax_weight;
   HYPRE_Real omega;
   HYPRE_Int  cheby_order;
   HYPRE_Real cheby_fraction;

   /* AMS options for the auxiliary curl-curl problem */
   HYPRE_Int  ams_cycle_type;
   HYPRE_Int  ams_coarsen_type;
   HYPRE_Int  ams_agg_levels;
   HYPRE_Int  ams_relax_type;
   HYPRE_Real ams_strength_threshold;
   HYPRE_Int  ams_interp_type;
   HYPRE_Int  ams_Pmax;

   /* AMG options for the auxiliary vector Poisson problem */
   HYPRE_Int  amg_coarsen_type;
   HYPRE_Int  amg_agg_levels;
   HYPRE_Int  amg_relax_type;
   HYPRE_Real amg_strength_threshold;
   HYPRE_Int  amg_interp_type;
   HYPRE_Int  amg_Pmax;
} ADS_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void hypredrv_ADSSetDefaultArgs(ADS_args *);
void hypredrv_ADSSetArgs(void *, const YAMLnode *);
void hypredrv_ADSCreate(const ADS_args *, HYPRE_Solver *);

/* Inject the operator inputs ADS requires (discrete gradient + discrete curl +
 * vertex coordinate vectors). Arguments are IJ handles (may be NULL); the
 * underlying ParCSR/ParVector objects are extracted internally. Must be called
 * after ADSCreate and before setup. */
void hypredrv_ADSSetOperators(HYPRE_Solver, HYPRE_IJMatrix G, HYPRE_IJMatrix C,
                              HYPRE_IJVector x, HYPRE_IJVector y, HYPRE_IJVector z);

#endif /* ADS_HEADER */
