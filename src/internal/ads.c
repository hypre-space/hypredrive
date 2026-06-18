/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/ads.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping (defaults match hypre, except
 * max_iter/tolerance/print_level tuned for preconditioner use)
 *-----------------------------------------------------------------------------*/

#define ADS_FIELDS(_X, _p)                                           \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 1)                     \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 0)                  \
   _X(_p, cycle_type, hypredrv_FieldTypeIntSet, 1)                   \
   _X(_p, tolerance, hypredrv_FieldTypeDoubleSet, 0.0)               \
   _X(_p, relax_type, hypredrv_FieldTypeIntSet, 2)                   \
   _X(_p, relax_times, hypredrv_FieldTypeIntSet, 1)                  \
   _X(_p, relax_weight, hypredrv_FieldTypeDoubleSet, 1.0)           \
   _X(_p, omega, hypredrv_FieldTypeDoubleSet, 1.0)                   \
   _X(_p, cheby_order, hypredrv_FieldTypeIntSet, 2)                  \
   _X(_p, cheby_fraction, hypredrv_FieldTypeDoubleSet, 0.3)         \
   _X(_p, ams_cycle_type, hypredrv_FieldTypeIntSet, 11)             \
   _X(_p, ams_coarsen_type, hypredrv_FieldTypeIntSet, 10)           \
   _X(_p, ams_agg_levels, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, ams_relax_type, hypredrv_FieldTypeIntSet, 3)              \
   _X(_p, ams_strength_threshold, hypredrv_FieldTypeDoubleSet, 0.25) \
   _X(_p, ams_interp_type, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, ams_Pmax, hypredrv_FieldTypeIntSet, 0)                    \
   _X(_p, amg_coarsen_type, hypredrv_FieldTypeIntSet, 10)           \
   _X(_p, amg_agg_levels, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, amg_relax_type, hypredrv_FieldTypeIntSet, 3)             \
   _X(_p, amg_strength_threshold, hypredrv_FieldTypeDoubleSet, 0.25) \
   _X(_p, amg_interp_type, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, amg_Pmax, hypredrv_FieldTypeIntSet, 0)

/* Define num_fields macro */
#define ADS_NUM_FIELDS (sizeof(ADS_field_offset_map) / sizeof(ADS_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(ADS) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * ADSGetValidValues (accept raw hypre integers)
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_ADSGetValidValues(const char *key)
{
   (void)key;
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * hypredrv_ADSCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_ADSCreate(const ADS_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon = NULL;

   HYPRE_ADSCreate(&precon);

   HYPRE_ADSSetMaxIter(precon, args->max_iter);
   HYPRE_ADSSetTol(precon, args->tolerance);
   HYPRE_ADSSetCycleType(precon, args->cycle_type);
   HYPRE_ADSSetPrintLevel(precon, args->print_level);
   HYPRE_ADSSetSmoothingOptions(precon, args->relax_type, args->relax_times,
                                args->relax_weight, args->omega);
   HYPRE_ADSSetChebySmoothingOptions(precon, args->cheby_order, args->cheby_fraction);
   HYPRE_ADSSetAMSOptions(precon, args->ams_cycle_type, args->ams_coarsen_type,
                          args->ams_agg_levels, args->ams_relax_type,
                          args->ams_strength_threshold, args->ams_interp_type,
                          args->ams_Pmax);
   HYPRE_ADSSetAMGOptions(precon, args->amg_coarsen_type, args->amg_agg_levels,
                          args->amg_relax_type, args->amg_strength_threshold,
                          args->amg_interp_type, args->amg_Pmax);

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ADSSetOperators
 *-----------------------------------------------------------------------------*/

void
hypredrv_ADSSetOperators(HYPRE_Solver solver, HYPRE_IJMatrix G, HYPRE_IJMatrix C,
                        HYPRE_IJVector x, HYPRE_IJVector y, HYPRE_IJVector z)
{
   void *obj = NULL;

   if (G)
   {
      HYPRE_IJMatrixGetObject(G, &obj);
      HYPRE_ADSSetDiscreteGradient(solver, (HYPRE_ParCSRMatrix)obj);
   }

   if (C)
   {
      HYPRE_IJMatrixGetObject(C, &obj);
      HYPRE_ADSSetDiscreteCurl(solver, (HYPRE_ParCSRMatrix)obj);
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

      HYPRE_ADSSetCoordinateVectors(solver, px, py, pz);
   }
}
