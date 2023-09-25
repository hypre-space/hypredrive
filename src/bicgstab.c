/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "bicgstab.h"

static const FieldOffsetMap bicgstab_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, min_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, stop_crit, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, logging, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, relative_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, absolute_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(BiCGSTAB_args, conv_fac_tol, FieldTypeDoubleSet)
};

#define BICGSTAB_NUM_FIELDS (sizeof(bicgstab_field_offset_map) /\
                             sizeof(bicgstab_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * BiCGSTABSetFieldByName
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetFieldByName(BiCGSTAB_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < BICGSTAB_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(bicgstab_field_offset_map[i].name, node->key))
      {
         bicgstab_field_offset_map[i].setter(
            (void*)((char*) args + bicgstab_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * BiCGSTABGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
BiCGSTABGetValidKeys(void)
{
   static const char* keys[BICGSTAB_NUM_FIELDS];

   for (size_t i = 0; i < BICGSTAB_NUM_FIELDS; i++)
   {
      keys[i] = bicgstab_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * BiCGSTABGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
BiCGSTABGetValidValues(const char* key)
{
   /* Don't impose any restrictions, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * BiCGSTABSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetDefaultArgs(BiCGSTAB_args *args)
{
   args->min_iter     = 0;
   args->max_iter     = 100;
   args->stop_crit    = 0;
   args->logging      = 1;
   args->print_level  = 1;
   args->relative_tol = 1.0e-6;
   args->absolute_tol = 0.0;
   args->conv_fac_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * BiCGSTABSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetArgsFromYAML(BiCGSTAB_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         BiCGSTABGetValidKeys,
                         BiCGSTABGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          BiCGSTABSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * BiCGSTABSetArgs
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetArgs(void *vargs, YAMLnode *parent)
{
   BiCGSTAB_args  *args = (BiCGSTAB_args*) vargs;

   BiCGSTABSetDefaultArgs(args);
   BiCGSTABSetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * BiCGSTABCreate
 *-----------------------------------------------------------------------------*/

void
BiCGSTABCreate(MPI_Comm comm, BiCGSTAB_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

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
