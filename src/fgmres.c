/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "fgmres.h"

/*-----------------------------------------------------------------------------
 * FGMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
FGMRESSetDefaultArgs(FGMRES_args *args)
{
   args->min_iter = 0;
   args->max_iter = 100;
   args->krylov_dimension = 30;
   args->logging = 1;
   args->print_level = 1;
   args->rtol = 1.0e-6;
   args->atol = 0.0;
}

/*-----------------------------------------------------------------------------
 * FGMRESSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
FGMRESSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   FGMRES_args *args = (FGMRES_args*) vargs;
   YAMLnode    *child;

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "min_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->krylov_dimension, "krylov_dimension", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->logging, "logging", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->rtol, "rtol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->atol, "atol", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }
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
   HYPRE_FlexGMRESSetKDim(solver, args->krylov_dimension);
   HYPRE_FlexGMRESSetLogging(solver, args->logging);
   HYPRE_FlexGMRESSetPrintLevel(solver, args->print_level);
   HYPRE_FlexGMRESSetTol(solver, args->rtol);
   HYPRE_FlexGMRESSetAbsoluteTol(solver, args->atol);

   *solver_ptr = solver;
}
