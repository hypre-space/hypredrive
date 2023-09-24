/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "ilu.h"

/*-----------------------------------------------------------------------------
 * ILUSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
ILUSetDefaultArgs(ILU_args *args)
{
   args->max_iter = 1;
   args->print_level = 0;
   args->type = 0;
   args->fill_level = 0;
   args->reordering = 0;
   args->tri_solve = 1;
   args->lower_jac_iters = 5;
   args->upper_jac_iters = 5;
   args->max_row_nnz = 1000;
   args->schur_max_iter = 3;
   args->droptol = 1.0e-02;
}

/*-----------------------------------------------------------------------------
 * ILUSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
ILUSetArgsFromYAML(ILU_args *args, YAMLnode *parent)
{
   YAMLnode *child;

   if (!parent)
   {
      return;
   }

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->type, "type", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fill_level, "fill_level", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->reordering, "reordering", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->tri_solve, "tri_solve", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->lower_jac_iters, "lower_jac_iters", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->upper_jac_iters, "upper_jac_iters", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_row_nnz, "max_row_nnz", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->schur_max_iter, "schur_max_iter", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->droptol, "droptol", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
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

   *precon_ptr = precon;
}
