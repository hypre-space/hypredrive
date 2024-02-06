/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "bicgstab.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define BiCGSTAB_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, min_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, stop_crit, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, logging, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relative_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, absolute_tol, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, conv_fac_tol, FieldTypeDoubleSet)

/* Define num_fields macro */
#define BiCGSTAB_NUM_FIELDS (sizeof(BiCGSTAB_field_offset_map) /\
                             sizeof(BiCGSTAB_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object */
GENERATE_PREFIXED_COMPONENTS(BiCGSTAB)
DEFINE_VOID_GET_VALID_VALUES_FUNC(BiCGSTAB)

/*-----------------------------------------------------------------------------
 * BiCGSTABSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetDefaultArgs(BiCGSTAB_args *args)
{
   args->min_iter     = 0;
   args->max_iter     = 100;
   args->stop_crit    = 0;
   args->logging      = 1;
   args->print_level  = 1;
   args->relative_tol = 1.0e-6;
   args->absolute_tol = 0.0;
   args->conv_fac_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * BiCGSTABCreate
 *-----------------------------------------------------------------------------*/

void
BiCGSTABCreate(MPI_Comm comm, BiCGSTAB_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

   HYPRE_ParCSRBiCGSTABCreate(comm, &solver);
   HYPRE_BiCGSTABSetMinIter(solver, args->min_iter);
   HYPRE_BiCGSTABSetMaxIter(solver, args->max_iter);
   HYPRE_BiCGSTABSetStopCrit(solver, args->stop_crit);
   HYPRE_BiCGSTABSetLogging(solver, args->logging);
   HYPRE_BiCGSTABSetPrintLevel(solver, args->print_level);
   HYPRE_BiCGSTABSetTol(solver, args->relative_tol);
   HYPRE_BiCGSTABSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_BiCGSTABSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}
