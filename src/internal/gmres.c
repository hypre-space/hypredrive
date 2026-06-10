/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/gmres.h"
#include "internal/gen_macros.h"
#include "internal/utils.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define GMRES_FIELDS(_X, _p)                                 \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 300)           \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, skip_real_res_check, hypredrv_FieldTypeIntSet, 0)  \
   _X(_p, krylov_dim, hypredrv_FieldTypeIntSet, 30)          \
   _X(_p, rel_change, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

/* Define num_fields macro */
#define GMRES_NUM_FIELDS \
   (sizeof(GMRES_field_offset_map) / sizeof(GMRES_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(GMRES) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * GMRESGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_GMRESGetValidValues(const char *key)
{
   if (!strcmp(key, "skip_real_res_check") || !strcmp(key, "rel_change"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * GMRESCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_GMRESCreate(MPI_Comm comm, const GMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver = NULL;

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

/*-----------------------------------------------------------------------------
 * GMRESSetRefSolution
 *-----------------------------------------------------------------------------*/

/* Pass the reference solution (when one is set) to hypre's GMRES for error
 * tracking; a no-op on hypre versions without HYPRE_ParCSRGMRESSetRefSolution. */
void
hypredrv_GMRESSetRefSolution(HYPRE_Solver solver, HYPRE_IJVector vec_xref)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void           *obj_ref = NULL;
   HYPRE_ParVector par_ref = NULL;

   if (vec_xref)
   {
      HYPRE_IJVectorGetObject(vec_xref, &obj_ref);
      par_ref = (HYPRE_ParVector)obj_ref;
   }

   HYPRE_ParCSRGMRESSetRefSolution(solver, par_ref);
#else
   (void)solver;
   (void)vec_xref;
#endif
}
