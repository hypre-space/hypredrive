/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "gmres.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define GMRES_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, min_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, stop_crit, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, skip_real_res_check, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, krylov_dim, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, rel_change, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, logging, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relative_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, absolute_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, conv_fac_tol, FieldTypeDoubleSet)

/* Define num_fields macro */
#define GMRES_NUM_FIELDS (sizeof(GMRES_field_offset_map) / sizeof(GMRES_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object */
GENERATE_PREFIXED_COMPONENTS(GMRES)
DEFINE_VOID_GET_VALID_VALUES_FUNC(GMRES)

/*-----------------------------------------------------------------------------
 * GMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
GMRESSetDefaultArgs(GMRES_args *args)
{
   args->min_iter            = 0;
   args->max_iter            = 300;
   args->stop_crit           = 0;
   args->skip_real_res_check = 0;
   args->krylov_dim          = 30;
   args->rel_change          = 0;
   args->logging             = 1;
   args->print_level         = 1;
   args->relative_tol        = 1.0e-6;
   args->absolute_tol        = 0.0;
   args->conv_fac_tol        = 0.0;
}

/*-----------------------------------------------------------------------------
 * GMRESCreate
 *-----------------------------------------------------------------------------*/

void
GMRESCreate(MPI_Comm comm, GMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

   HYPRE_ParCSRGMRESCreate(comm, &solver);
   HYPRE_GMRESSetMinIter(solver, args->min_iter);
   HYPRE_GMRESSetMaxIter(solver, args->max_iter);
   HYPRE_GMRESSetStopCrit(solver, args->stop_crit);
   HYPRE_GMRESSetSkipRealResidualCheck(solver, args->skip_real_res_check);
   HYPRE_GMRESSetKDim(solver, args->krylov_dim);
   HYPRE_GMRESSetRelChange(solver, args->rel_change);
   HYPRE_GMRESSetLogging(solver, args->logging);
   HYPRE_GMRESSetPrintLevel(solver, args->print_level);
   HYPRE_GMRESSetTol(solver, args->relative_tol);
   HYPRE_GMRESSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_GMRESSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}
