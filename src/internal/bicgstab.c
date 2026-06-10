/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/bicgstab.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define BiCGSTAB_FIELDS(_X, _p)                              \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 100)           \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

/* Define num_fields macro */
#define BiCGSTAB_NUM_FIELDS \
   (sizeof(BiCGSTAB_field_offset_map) / sizeof(BiCGSTAB_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(BiCGSTAB)          // LCOV_EXCL_LINE
hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC(hypredrv_BiCGSTAB) // LCOV_EXCL_LINE

   /*-----------------------------------------------------------------------------
    * BiCGSTABCreate
    *-----------------------------------------------------------------------------*/

   void hypredrv_BiCGSTABCreate(MPI_Comm comm, const BiCGSTAB_args *args,
                                HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver = NULL;

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
