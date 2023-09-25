/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "ilu.h"

static const FieldOffsetMap ilu_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(ILU_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, type, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, fill_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, reordering, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, tri_solve, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, lower_jac_iters, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, upper_jac_iters, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, max_row_nnz, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, schur_max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, droptol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, nsh_droptol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(ILU_args, tolerance, FieldTypeDoubleSet)
};

#define ILU_NUM_FIELDS (sizeof(ilu_field_offset_map) / sizeof(ilu_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * ILUSetFieldByName
 *-----------------------------------------------------------------------------*/

void
ILUSetFieldByName(ILU_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < ILU_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(ilu_field_offset_map[i].name, node->key))
      {
         ilu_field_offset_map[i].setter(
            (void*)((char*) args + ilu_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * ILUGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
ILUGetValidKeys(void)
{
   static const char* keys[ILU_NUM_FIELDS];

   for (size_t i = 0; i < ILU_NUM_FIELDS; i++)
   {
      keys[i] = ilu_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * ILUGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
ILUGetValidValues(const char* key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"bj-iluk",          0},
                                {"bj-ilut",          1},
                                {"gmres-iluk",      10},
                                {"gmres-ilut",      11},
                                {"nsh-iluk",        20},
                                {"nsh-ilut",        21},
                                {"ras-iluk",        30},
                                {"ras-ilut",        31},
                                {"ddpq-gmres-iluk", 40},
                                {"ddpq-gmres-ilut", 41},
                                {"rap-mod-ilu0",    50}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "reordering") ||
            !strcmp(key, "tri_solve"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
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
   args->max_row_nnz     = 1000;
   args->schur_max_iter  = 3;
   args->droptol         = 1.0e-02;
   args->nsh_droptol     = 1.0e-02;
   args->tolerance       = 0.0;
}

/*-----------------------------------------------------------------------------
 * ILUSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
ILUSetArgsFromYAML(ILU_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         ILUGetValidKeys,
                         ILUGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          ILUSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * ILUSetArgs
 *-----------------------------------------------------------------------------*/

void
ILUSetArgs(void *vargs, YAMLnode *parent)
{
   ILU_args *args = (ILU_args*) vargs;

   ILUSetDefaultArgs(args);
   ILUSetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * ILUCreate
 *-----------------------------------------------------------------------------*/

void
ILUCreate(ILU_args *args, HYPRE_Solver *precon_ptr)
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
