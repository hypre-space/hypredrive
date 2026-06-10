/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/fgmres.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define FGMRES_FIELDS(_X, _p)                                \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 300)           \
   _X(_p, krylov_dim, hypredrv_FieldTypeIntSet, 30)          \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)

/* Define num_fields macro */
#define FGMRES_NUM_FIELDS \
   (sizeof(FGMRES_field_offset_map) / sizeof(FGMRES_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(FGMRES)          // LCOV_EXCL_LINE
hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC(hypredrv_FGMRES) // LCOV_EXCL_LINE

   /*-----------------------------------------------------------------------------
    * FGMRESCreate
    *-----------------------------------------------------------------------------*/

   void hypredrv_FGMRESCreate(MPI_Comm comm, const FGMRES_args *args,
                              HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver = NULL;

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
