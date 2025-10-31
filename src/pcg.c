/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "pcg.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define PCG_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, two_norm, FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, stop_crit, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, rel_change, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, recompute_res, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relative_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, absolute_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, residual_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, conv_fac_tol, FieldTypeDoubleSet)

/* Define num_fields macro */
#define PCG_NUM_FIELDS (sizeof(PCG_field_offset_map) / sizeof(PCG_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS(PCG)

/*-----------------------------------------------------------------------------
 * PCGGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PCGGetValidValues(const char *key)
{
   if (!strcmp(key, "two_norm") || !strcmp(key, "stop_crit") ||
       !strcmp(key, "rel_change"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * PCGSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
PCGSetDefaultArgs(PCG_args *args)
{
   args->max_iter      = 100;
   args->two_norm      = 1;
   args->stop_crit     = 0;
   args->rel_change    = 0;
   args->print_level   = 1;
   args->recompute_res = 0;
   args->relative_tol  = 1.0e-6;
   args->absolute_tol  = 0.0;
   args->residual_tol  = 0.0;
   args->conv_fac_tol  = 0.0;
}

/*-----------------------------------------------------------------------------
 * PCGCreate
 *-----------------------------------------------------------------------------*/

void
PCGCreate(MPI_Comm comm, PCG_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

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
