/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "bicgstab.h"

/*-----------------------------------------------------------------------------
 * BiCGSTABSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetDefaultArgs(BiCGSTAB_args *args)
{
   args->min_iter = 0;
   args->max_iter = 100;
   args->stop_crit = 0;
   args->logging = 1;
   args->print_level = 1;
   args->rtol = 1.0e-6;
   args->atol = 0.0;
   args->cf_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * BiCGSTABSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
BiCGSTABSetArgsFromYAML(BiCGSTAB_args *args, YAMLnode *parent)
{
   YAMLnode       *child;

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "min_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->stop_crit, "stop_crit", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->logging, "logging", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->rtol, "rtol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->atol, "atol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->cf_tol, "cf_tol", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
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
   HYPRE_BiCGSTABSetTol(solver, args->rtol);
   HYPRE_BiCGSTABSetAbsoluteTol(solver, args->atol);
   HYPRE_BiCGSTABSetConvergenceFactorTol(solver, args->cf_tol);

   *solver_ptr = solver;
}
