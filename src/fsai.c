/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "fsai.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *-----------------------------------------------------------------------------*/

#define FSAI_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, algo_type, FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ls_type, FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_steps, FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_step_size, FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, eig_max_iters, FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, threshold, FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, kap_tolerance, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet)

/* Define num_fields macro */
#define FSAI_NUM_FIELDS (sizeof(FSAI_field_offset_map) / sizeof(FSAI_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS(FSAI)

/*-----------------------------------------------------------------------------
 * FSAIGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
FSAIGetValidValues(const char *key)
{
   if (!strcmp(key, "algo_type"))
   {
      static StrIntMap map[] = {{"bj-afsai", 1}, {"bj-afsai-omp", 2}, {"bj-sfsai", 3}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * FSAISetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
FSAISetDefaultArgs(FSAI_args *args)
{
   args->max_iter      = 1;
   args->print_level   = 0;
   args->algo_type     = 1;
   args->ls_type       = 1;
   args->max_steps     = 5;
   args->max_step_size = 3;
   args->max_nnz_row   = 15;
   args->num_levels    = 1;
   args->eig_max_iters = 5;
   args->threshold     = 1.0e-3;
   args->kap_tolerance = 1.0e-3;
   args->tolerance     = 0.0;
}

/*-----------------------------------------------------------------------------
 * FSAICreate
 *-----------------------------------------------------------------------------*/

void
FSAICreate(const FSAI_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon;

   HYPRE_FSAICreate(&precon);

   HYPRE_FSAISetAlgoType(precon, args->algo_type);
   HYPRE_FSAISetLocalSolveType(precon, args->ls_type);
   HYPRE_FSAISetMaxSteps(precon, args->max_steps);
   HYPRE_FSAISetMaxStepSize(precon, args->max_step_size);
   HYPRE_FSAISetMaxNnzRow(precon, args->max_nnz_row);
   HYPRE_FSAISetNumLevels(precon, args->num_levels);
   HYPRE_FSAISetThreshold(precon, args->threshold);
   HYPRE_FSAISetKapTolerance(precon, args->kap_tolerance);
   HYPRE_FSAISetMaxIterations(precon, args->max_iter);
   HYPRE_FSAISetTolerance(precon, args->tolerance);
   HYPRE_FSAISetPrintLevel(precon, args->print_level);

   *precon_ptr = precon;
}
