/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/pcg.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define PCG_FIELDS(_X, _p)                                   \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 100)           \
   _X(_p, two_norm, hypredrv_FieldTypeIntSet, 1)             \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, rel_change, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, recompute_res, hypredrv_FieldTypeIntSet, 0)        \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, residual_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

/* Define num_fields macro */
#define PCG_NUM_FIELDS (sizeof(PCG_field_offset_map) / sizeof(PCG_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(PCG) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * PCGGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_PCGGetValidValues(const char *key)
{
   if (!strcmp(key, "two_norm") || !strcmp(key, "stop_crit") ||
       !strcmp(key, "rel_change"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }

   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * PCGCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_PCGCreate(MPI_Comm comm, const PCG_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver = NULL;

   HYPRE_ParCSRPCGCreate(comm, &solver);
   HYPRE_PCGSetMaxIter(solver, args->max_iter);
   HYPRE_PCGSetTwoNorm(solver, args->two_norm);
   HYPRE_PCGSetStopCrit(solver, args->stop_crit);
   HYPRE_PCGSetRelChange(solver, args->rel_change);
   HYPRE_PCGSetPrintLevel(solver, args->print_level);
   HYPRE_PCGSetRecomputeResidual(solver, args->recompute_res);
   HYPRE_PCGSetTol(solver, args->relative_tol);
   HYPRE_PCGSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_PCGSetResidualTol(solver, args->residual_tol);
   HYPRE_PCGSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}
