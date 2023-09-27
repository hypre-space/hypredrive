/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "fgmres.h"
#include "gen_macros.h"

#define FGMRES_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, min_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, krylov_dim, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, logging, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relative_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, absolute_tol, FieldTypeDoubleSet)

/* Define num_fields macro */
#define FGMRES_NUM_FIELDS (sizeof(FGMRES_field_offset_map) / sizeof(FGMRES_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object */
GENERATE_PREFIXED_COMPONENTS(FGMRES)
DEFINE_VOID_GET_VALID_VALUES_FUNC(FGMRES)

/*-----------------------------------------------------------------------------
 * FGMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
FGMRESSetDefaultArgs(FGMRES_args *args)
{
   args->min_iter     = 0;
   args->max_iter     = 300;
   args->krylov_dim   = 30;
   args->logging      = 1;
   args->print_level  = 1;
   args->relative_tol = 1.0e-6;
   args->absolute_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * FGMRESCreate
 *-----------------------------------------------------------------------------*/

void
FGMRESCreate(MPI_Comm comm, FGMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

   HYPRE_ParCSRFlexGMRESCreate(comm, &solver);
   HYPRE_FlexGMRESSetMinIter(solver, args->min_iter);
   HYPRE_FlexGMRESSetMaxIter(solver, args->max_iter);
   HYPRE_FlexGMRESSetKDim(solver, args->krylov_dim);
   HYPRE_FlexGMRESSetLogging(solver, args->logging);
   HYPRE_FlexGMRESSetPrintLevel(solver, args->print_level);
   HYPRE_FlexGMRESSetTol(solver, args->relative_tol);
   HYPRE_FlexGMRESSetAbsoluteTol(solver, args->absolute_tol);

   *solver_ptr = solver;
}
