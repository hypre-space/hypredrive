/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "gmres.h"

static const FieldOffsetMap gmres_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, min_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, max_iter, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, stop_crit, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, skip_real_res_check, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, krylov_dim, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, rel_change, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, logging, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, print_level, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, relative_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, absolute_tol, FieldTypeDoubleSet),
   FIELD_OFFSET_MAP_ENTRY(GMRES_args, conv_fac_tol, FieldTypeDoubleSet)
};

#define GMRES_NUM_FIELDS (sizeof(gmres_field_offset_map) / sizeof(gmres_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * GMRESSetFieldByName
 *-----------------------------------------------------------------------------*/

void
GMRESSetFieldByName(GMRES_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < GMRES_NUM_FIELDS; i++)
   {
      /* Which field from the arguments list are we trying to set? */
      if (!strcmp(gmres_field_offset_map[i].name, node->key))
      {
         gmres_field_offset_map[i].setter(
            (void*)((char*) args + gmres_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * GMRESGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
GMRESGetValidKeys(void)
{
   static const char* keys[GMRES_NUM_FIELDS];

   for (size_t i = 0; i < GMRES_NUM_FIELDS; i++)
   {
      keys[i] = gmres_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * GMRESGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
GMRESGetValidValues(const char* key)
{
   /* Don't impose any restrictions, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * GMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
GMRESSetDefaultArgs(GMRES_args *args)
{
   args->min_iter            = 0;
   args->max_iter            = 300;
   args->stop_crit           = 0;
   args->skip_real_res_check = 0;
   args->krylov_dim          = 30;
   args->rel_change          = 0;
   args->logging             = 1;
   args->print_level         = 1;
   args->relative_tol        = 1.0e-6;
   args->absolute_tol        = 0.0;
   args->conv_fac_tol        = 0.0;
}

/*-----------------------------------------------------------------------------
 * GMRESSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
GMRESSetArgsFromYAML(GMRES_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         GMRESGetValidKeys,
                         GMRESGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          GMRESSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * GMRESSetArgs
 *-----------------------------------------------------------------------------*/

void
GMRESSetArgs(void *vargs, YAMLnode *parent)
{
   GMRES_args  *args = (GMRES_args*) vargs;

   GMRESSetDefaultArgs(args);
   GMRESSetArgsFromYAML(args, parent);
}

/*-----------------------------------------------------------------------------
 * GMRESCreate
 *-----------------------------------------------------------------------------*/

void
GMRESCreate(MPI_Comm comm, GMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

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
