/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/ilu.h"
#include "internal/gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define ILU_FIELDS(_X, _p)                                   \
   _X(_p, max_iter, hypredrv_FieldTypeIntSet, 1)             \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 0)          \
   _X(_p, type, hypredrv_FieldTypeIntSet, 0)                 \
   _X(_p, fill_level, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, reordering, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, tri_solve, hypredrv_FieldTypeIntSet, 1)            \
   _X(_p, lower_jac_iters, hypredrv_FieldTypeIntSet, 5)      \
   _X(_p, upper_jac_iters, hypredrv_FieldTypeIntSet, 5)      \
   _X(_p, max_row_nnz, hypredrv_FieldTypeIntSet, 200)        \
   _X(_p, schur_max_iter, hypredrv_FieldTypeIntSet, 3)       \
   _X(_p, droptol, hypredrv_FieldTypeDoubleSet, 1.0e-02)     \
   _X(_p, nsh_droptol, hypredrv_FieldTypeDoubleSet, 1.0e-02) \
   _X(_p, tolerance, hypredrv_FieldTypeDoubleSet, 0.0)

/* Define num_fields macro */
#define ILU_NUM_FIELDS (sizeof(ILU_field_offset_map) / sizeof(ILU_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS_WITH_DEFAULTS(ILU) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * ILUGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_ILUGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {
         {"bj-iluk", 0},          {"bj-ilut", 1},       {"gmres-iluk", 10},
         {"gmres-ilut", 11},      {"nsh-iluk", 20},     {"nsh-ilut", 21},
         {"ras-iluk", 30},        {"ras-ilut", 31},     {"ddpq-gmres-iluk", 40},
         {"ddpq-gmres-ilut", 41}, {"rap-mod-ilu0", 50},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }

   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * hypredrv_ILUCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_ILUCreate(const ILU_args *args, HYPRE_Solver *precon_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)args;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("ILU requires hypre >= 2.19.0");
   *precon_ptr = NULL;
   return;
#else
   HYPRE_Solver precon = NULL;
   int          uses_schur_solver;
   int          uses_nsh;

   HYPRE_ILUCreate(&precon);

   HYPRE_ILUSetType(precon, args->type);
   HYPRE_ILUSetLevelOfFill(precon, args->fill_level);
   HYPRE_ILUSetLocalReordering(precon, args->reordering);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_ILUSetTriSolve(precon, args->tri_solve);
   HYPRE_ILUSetLowerJacobiIters(precon, args->lower_jac_iters);
   HYPRE_ILUSetUpperJacobiIters(precon, args->upper_jac_iters);
#endif
   HYPRE_ILUSetPrintLevel(precon, args->print_level);
   HYPRE_ILUSetMaxIter(precon, args->max_iter);
   HYPRE_ILUSetTol(precon, args->tolerance);
   HYPRE_ILUSetMaxNnzPerRow(precon, args->max_row_nnz);
   HYPRE_ILUSetDropThreshold(precon, args->droptol);

   uses_schur_solver =
      (args->type == 10 || args->type == 11 || args->type == 20 || args->type == 21 ||
       args->type == 40 || args->type == 41 || args->type == 50);
   uses_nsh = (args->type == 20 || args->type == 21);
   if (uses_schur_solver)
   {
      HYPRE_ILUSetSchurMaxIter(precon, args->schur_max_iter);
   }
   if (uses_nsh)
   {
      HYPRE_ILUSetNSHDropThreshold(precon, args->nsh_droptol);
   }

   *precon_ptr = precon;
#endif
}
