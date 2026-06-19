/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/ams.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping (defaults match hypre, except
 * max_iter/tolerance/print_level tuned for preconditioner use)
 *-----------------------------------------------------------------------------*/

#define AMS_FIELDS(_X, _p)                                             \
   _X(_p, dimension, hypredrv_FieldTypeIntSet, 3)                      \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 1)                       \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 0)                    \
   _X(_p, cycle_type, hypredrv_FieldTypeIntSet, 1)                     \
   _X(_p, tolerance, hypredrv_FieldTypeDoubleSet, 0.0)                 \
   _X(_p, relax_type, hypredrv_FieldTypeIntSet, 2)                     \
   _X(_p, relax_times, hypredrv_FieldTypeIntSet, 1)                    \
   _X(_p, relax_weight, hypredrv_FieldTypeDoubleSet, 1.0)              \
   _X(_p, omega, hypredrv_FieldTypeDoubleSet, 1.0)                     \
   _X(_p, proj_freq, hypredrv_FieldTypeIntSet, 5)                      \
   _X(_p, alpha_coarsen_type, hypredrv_FieldTypeIntSet, 10)            \
   _X(_p, alpha_agg_levels, hypredrv_FieldTypeIntSet, 1)               \
   _X(_p, alpha_relax_type, hypredrv_FieldTypeIntSet, 3)               \
   _X(_p, alpha_strength_threshold, hypredrv_FieldTypeDoubleSet, 0.25) \
   _X(_p, alpha_interp_type, hypredrv_FieldTypeIntSet, 0)              \
   _X(_p, alpha_Pmax, hypredrv_FieldTypeIntSet, 0)                     \
   _X(_p, alpha_coarse_relax_type, hypredrv_FieldTypeIntSet, 8)        \
   _X(_p, beta_coarsen_type, hypredrv_FieldTypeIntSet, 10)             \
   _X(_p, beta_agg_levels, hypredrv_FieldTypeIntSet, 1)                \
   _X(_p, beta_relax_type, hypredrv_FieldTypeIntSet, 3)                \
   _X(_p, beta_strength_threshold, hypredrv_FieldTypeDoubleSet, 0.25)  \
   _X(_p, beta_interp_type, hypredrv_FieldTypeIntSet, 0)               \
   _X(_p, beta_Pmax, hypredrv_FieldTypeIntSet, 0)                      \
   _X(_p, beta_coarse_relax_type, hypredrv_FieldTypeIntSet, 8)

/* Define num_fields macro */
#define AMS_NUM_FIELDS (sizeof(AMS_field_offset_map) / sizeof(AMS_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(AMS) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * AMSGetValidValues (accept raw hypre integers)
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_AMSGetValidValues(const char *key)
{
   (void)key;
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * hypredrv_AMSCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_AMSCreate(const AMS_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon = NULL;

   HYPRE_AMSCreate(&precon);

   HYPRE_AMSSetDimension(precon, args->dimension);
   HYPRE_AMSSetMaxIter(precon, args->max_iter);
   HYPRE_AMSSetTol(precon, args->tolerance);
   HYPRE_AMSSetCycleType(precon, args->cycle_type);
   HYPRE_AMSSetPrintLevel(precon, args->print_level);
   HYPRE_AMSSetSmoothingOptions(precon, args->relax_type, args->relax_times,
                                args->relax_weight, args->omega);
   HYPRE_AMSSetAlphaAMGOptions(precon, args->alpha_coarsen_type, args->alpha_agg_levels,
                               args->alpha_relax_type, args->alpha_strength_threshold,
                               args->alpha_interp_type, args->alpha_Pmax);
   HYPRE_AMSSetAlphaAMGCoarseRelaxType(precon, args->alpha_coarse_relax_type);
   HYPRE_AMSSetBetaAMGOptions(precon, args->beta_coarsen_type, args->beta_agg_levels,
                              args->beta_relax_type, args->beta_strength_threshold,
                              args->beta_interp_type, args->beta_Pmax);
   HYPRE_AMSSetBetaAMGCoarseRelaxType(precon, args->beta_coarse_relax_type);
   HYPRE_AMSSetProjectionFrequency(precon, args->proj_freq);

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * hypredrv_AMSSetOperators
 *-----------------------------------------------------------------------------*/

void
hypredrv_AMSSetOperators(HYPRE_Solver solver, HYPRE_IJMatrix G, HYPRE_IJVector x,
                         HYPRE_IJVector y, HYPRE_IJVector z)
{
   void *obj = NULL;

   if (G)
   {
      HYPRE_IJMatrixGetObject(G, &obj);
      HYPRE_AMSSetDiscreteGradient(solver, (HYPRE_ParCSRMatrix)obj);
   }

   if (x && y && z)
   {
      HYPRE_ParVector px = NULL, py = NULL, pz = NULL;

      HYPRE_IJVectorGetObject(x, &obj);
      px = (HYPRE_ParVector)obj;
      HYPRE_IJVectorGetObject(y, &obj);
      py = (HYPRE_ParVector)obj;
      HYPRE_IJVectorGetObject(z, &obj);
      pz = (HYPRE_ParVector)obj;

      HYPRE_AMSSetCoordinateVectors(solver, px, py, pz);
   }
}
