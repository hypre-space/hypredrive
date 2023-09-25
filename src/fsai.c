/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "fsai.h"

static const FieldOffsetMap fsai_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, algo_type, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, ls_type, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, max_steps, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, max_step_size, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, max_nnz_row, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, num_levels, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, eig_max_iters, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, threshold, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, kap_tolerance, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(FSAI_args, tolerance, FieldTypeDoubleSet)
};

#define FSAI_NUM_FIELDS (sizeof(fsai_field_offset_map) / sizeof(fsai_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * FSAISetFieldByName
 *-----------------------------------------------------------------------------*/

void
FSAISetFieldByName(FSAI_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < FSAI_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(fsai_field_offset_map[i].name, node->key))
      {
         fsai_field_offset_map[i].setter(
            (void*)((char*) args + fsai_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * FSAIGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
FSAIGetValidKeys(void)
{
   static const char* keys[FSAI_NUM_FIELDS];

   for (size_t i = 0; i < FSAI_NUM_FIELDS; i++)
   {
      keys[i] = fsai_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * FSAIGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
FSAIGetValidValues(const char* key)
{
   if (!strcmp(key, "algo_type"))
   {
      static StrIntMap map[] = {{"bj-afsai",     0},
                                {"bj-afsai-omp", 1},
                                {"bj-sfsai",     2}};

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
   args->max_iter        = 1;
   args->print_level     = 0;
   args->algo_type       = 1;
   args->ls_type         = 1;
   args->max_steps       = 5;
   args->max_step_size   = 3;
   args->max_nnz_row     = 15;
   args->num_levels      = 1;
   args->eig_max_iters   = 5;
   args->threshold       = 1.0e-3;
   args->kap_tolerance   = 1.0e-3;
   args->tolerance       = 0.0;
}

/*-----------------------------------------------------------------------------
 * FSAISetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
FSAISetArgsFromYAML(FSAI_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         FSAIGetValidKeys,
                         FSAIGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          FSAISetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * FSAISetArgs
 *-----------------------------------------------------------------------------*/

void
FSAISetArgs(void *vargs, YAMLnode *parent)
{
   FSAI_args *args = (FSAI_args*) vargs;

   FSAISetDefaultArgs(args);
   FSAISetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * FSAICreate
 *-----------------------------------------------------------------------------*/

void
FSAICreate(FSAI_args *args, HYPRE_Solver *precon_ptr)
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
