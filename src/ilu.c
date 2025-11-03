/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "ilu.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define ILU_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fill_level, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, reordering, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tri_solve, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, lower_jac_iters, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, upper_jac_iters, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_row_nnz, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, schur_max_iter, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, droptol, FieldTypeDoubleSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, nsh_droptol, FieldTypeDoubleSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet)

/* Define num_fields macro */
#define ILU_NUM_FIELDS (sizeof(ILU_field_offset_map) / sizeof(ILU_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS(ILU)

/*-----------------------------------------------------------------------------
 * ILUGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
ILUGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {
         {"bj-iluk", 0},          {"bj-ilut", 1},      {"gmres-iluk", 10},
         {"gmres-ilut", 11},      {"nsh-iluk", 20},    {"nsh-ilut", 21},
         {"ras-iluk", 30},        {"ras-ilut", 31},    {"ddpq-gmres-iluk", 40},
         {"ddpq-gmres-ilut", 41}, {"rap-mod-ilu0", 50}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
#if 0
   /* TODO: Fix these options */
   else if (!strcmp(key, "reordering") ||
            !strcmp(key, "tri_solve"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
#endif
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * ILUSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
ILUSetDefaultArgs(ILU_args *args)
{
   args->max_iter        = 1;
   args->print_level     = 0;
   args->type            = 0;
   args->fill_level      = 0;
   args->reordering      = 0;
   args->tri_solve       = 1;
   args->lower_jac_iters = 5;
   args->upper_jac_iters = 5;
   args->max_row_nnz     = 200;
   args->schur_max_iter  = 3;
   args->droptol         = 1.0e-02;
   args->nsh_droptol     = 1.0e-02;
   args->tolerance       = 0.0;
}

/*-----------------------------------------------------------------------------
 * ILUCreate
 *-----------------------------------------------------------------------------*/

void
ILUCreate(const ILU_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon;

   HYPRE_ILUCreate(&precon);

   HYPRE_ILUSetType(precon, args->type);
   HYPRE_ILUSetLevelOfFill(precon, args->fill_level);
   HYPRE_ILUSetLocalReordering(precon, args->reordering);
   HYPRE_ILUSetTriSolve(precon, args->tri_solve);
   HYPRE_ILUSetLowerJacobiIters(precon, args->lower_jac_iters);
   HYPRE_ILUSetUpperJacobiIters(precon, args->upper_jac_iters);
   HYPRE_ILUSetPrintLevel(precon, args->print_level);
   HYPRE_ILUSetMaxIter(precon, args->max_iter);
   HYPRE_ILUSetTol(precon, args->tolerance);
   HYPRE_ILUSetMaxNnzPerRow(precon, args->max_row_nnz);
   HYPRE_ILUSetDropThreshold(precon, args->droptol);
   HYPRE_ILUSetSchurMaxIter(precon, args->schur_max_iter);
   HYPRE_ILUSetNSHDropThreshold(precon, args->nsh_droptol);

   *precon_ptr = precon;
}
