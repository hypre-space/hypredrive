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
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, hypredrv_FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, algo_type, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ls_type, hypredrv_FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_steps, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_step_size, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, eig_max_iters, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, threshold, hypredrv_FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, kap_tolerance, hypredrv_FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, hypredrv_FieldTypeDoubleSet)

/* Define num_fields macro */
#define FSAI_NUM_FIELDS (sizeof(FSAI_field_offset_map) / sizeof(FSAI_field_offset_map[0]))

/* Generate the various function declarations/definitions and the field_offset_map object
 */
GENERATE_PREFIXED_COMPONENTS(FSAI) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * FSAIGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_FSAIGetValidValues(const char *key)
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
hypredrv_FSAISetDefaultArgs(FSAI_args *args)
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
 * hypredrv_FSAICreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_FSAICreate(const FSAI_args *args, HYPRE_Solver *precon_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
   (void)args;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
   *precon_ptr = NULL;
   return;
#else
   HYPRE_Solver precon = NULL;

   HYPRE_FSAICreate(&precon);

   HYPRE_FSAISetAlgoType(precon, args->algo_type);
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
   HYPRE_FSAISetLocalSolveType(precon, args->ls_type);
   HYPRE_FSAISetMaxSteps(precon, args->max_steps);
   HYPRE_FSAISetMaxStepSize(precon, args->max_step_size);
   HYPRE_FSAISetMaxNnzRow(precon, args->max_nnz_row);
   HYPRE_FSAISetNumLevels(precon, args->num_levels);
   HYPRE_FSAISetThreshold(precon, args->threshold);
   HYPRE_FSAISetKapTolerance(precon, args->kap_tolerance);
#endif
   HYPRE_FSAISetMaxIterations(precon, args->max_iter);
   HYPRE_FSAISetTolerance(precon, args->tolerance);
   HYPRE_FSAISetPrintLevel(precon, args->print_level);

   *precon_ptr = precon;
#endif
}
