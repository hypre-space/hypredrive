/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "fgmres.h"

static const FieldOffsetMap fgmres_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, min_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, krylov_dim, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, logging, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, relative_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(FGMRES_args, absolute_tol, FieldTypeDoubleSet),
};

#define FGMRES_NUM_FIELDS (sizeof(fgmres_field_offset_map) / sizeof(fgmres_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * FGMRESSetFieldByName
 *-----------------------------------------------------------------------------*/

void
FGMRESSetFieldByName(FGMRES_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < FGMRES_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(fgmres_field_offset_map[i].name, node->key))
      {
         fgmres_field_offset_map[i].setter(
            (void*)((char*) args + fgmres_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * FGMRESGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
FGMRESGetValidKeys(void)
{
   static const char* keys[FGMRES_NUM_FIELDS];

   for (size_t i = 0; i < FGMRES_NUM_FIELDS; i++)
   {
      keys[i] = fgmres_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * FGMRESGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
FGMRESGetValidValues(const char* key)
{
   /* Don't impose any restrictions, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * FGMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
FGMRESSetDefaultArgs(FGMRES_args *args)
{
   args->min_iter     = 0;
   args->max_iter     = 300;
   args->krylov_dim   = 30;
   args->logging      = 1;
   args->print_level  = 1;
   args->relative_tol = 1.0e-6;
   args->absolute_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * FGMRESSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
FGMRESSetArgsFromYAML(FGMRES_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         FGMRESGetValidKeys,
                         FGMRESGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          FGMRESSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * FGMRESSetArgs
 *-----------------------------------------------------------------------------*/

void
FGMRESSetArgs(void *vargs, YAMLnode *parent)
{
   FGMRES_args  *args = (FGMRES_args*) vargs;

   FGMRESSetDefaultArgs(args);
   FGMRESSetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * FGMRESCreate
 *-----------------------------------------------------------------------------*/

void
FGMRESCreate(MPI_Comm comm, FGMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

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
