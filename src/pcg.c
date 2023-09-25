/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "pcg.h"

static const FieldOffsetMap pcg_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(PCG_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, two_norm, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, stop_crit, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, rel_change, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, recompute_res, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, relative_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, absolute_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, residual_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(PCG_args, conv_fac_tol, FieldTypeDoubleSet)
};

#define PCG_NUM_FIELDS (sizeof(pcg_field_offset_map) / sizeof(pcg_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * PCGSetFieldByName
 *-----------------------------------------------------------------------------*/

void
PCGSetFieldByName(PCG_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < PCG_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(pcg_field_offset_map[i].name, node->key))
      {
         pcg_field_offset_map[i].setter(
            (void*)((char*) args + pcg_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * PCGGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
PCGGetValidKeys(void)
{
   static const char* keys[PCG_NUM_FIELDS];

   for (size_t i = 0; i < PCG_NUM_FIELDS; i++)
   {
      keys[i] = pcg_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * PCGGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PCGGetValidValues(const char* key)
{
   /* Don't impose any restrictions, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * PCGSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
PCGSetDefaultArgs(PCG_args *args)
{
   args->max_iter      = 100;
   args->two_norm      = 1;
   args->stop_crit     = 0;
   args->rel_change    = 0;
   args->print_level   = 1;
   args->recompute_res = 0;
   args->relative_tol  = 1.0e-6;
   args->absolute_tol  = 0.0;
   args->residual_tol  = 0.0;
   args->conv_fac_tol  = 0.0;
}

/*-----------------------------------------------------------------------------
 * PCGSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
PCGSetArgsFromYAML(PCG_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         PCGGetValidKeys,
                         PCGGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          PCGSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * PCGSetArgs
 *-----------------------------------------------------------------------------*/

void
PCGSetArgs(void *vargs, YAMLnode *parent)
{
   PCG_args  *args = (PCG_args*) vargs;

   PCGSetDefaultArgs(args);
   PCGSetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * PCGCreate
 *-----------------------------------------------------------------------------*/

void
PCGCreate(MPI_Comm comm, PCG_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

   HYPRE_ParCSRPCGCreate(comm, &solver);
   HYPRE_PCGSetMaxIter(solver, args->max_iter);
   HYPRE_PCGSetTwoNorm(solver, args->two_norm);
   HYPRE_PCGSetStopCrit(solver, args->stop_crit);
   HYPRE_PCGSetRelChange(solver, args->rel_change);
   HYPRE_PCGSetPrintLevel(solver, args->print_level);
   HYPRE_PCGSetRecomputeResidual(solver, args->recompute_res);
   HYPRE_PCGSetTol(solver, args->relative_tol);
   HYPRE_PCGSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_PCGSetResidualTol(solver, args->residual_tol);
   HYPRE_PCGSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}
