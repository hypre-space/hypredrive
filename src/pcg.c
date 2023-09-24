/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "pcg.h"

/*-----------------------------------------------------------------------------
 * PCGSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
PCGSetDefaultArgs(PCG_args *args)
{
   args->max_iter = 100;
   args->two_norm = 1;
   args->stop_crit = 0;
   args->rel_change = 0;
   args->print_level = 1;
   args->recompute_res = 0;
   args->rtol = 1.0e-6;
   args->atol = 0.0;
   args->res_tol = 0.0;
   args->cf_tol = 0.0;
}

/*-----------------------------------------------------------------------------
 * PCGSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
PCGSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   PCG_args  *args = (PCG_args*) vargs;
   YAMLnode  *child;

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->two_norm, "two_norm", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->stop_crit, "stop_crit", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->rel_change, "rel_change", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->recompute_res, "recompute_res", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->rtol, "rtol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->atol, "atol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->res_tol, "res_tol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->cf_tol, "cf_tol", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }
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
   HYPRE_PCGSetTol(solver, args->rtol);
   HYPRE_PCGSetAbsoluteTol(solver, args->atol);
   HYPRE_PCGSetResidualTol(solver, args->res_tol);
   HYPRE_PCGSetConvergenceFactorTol(solver, args->cf_tol);

   *solver_ptr = solver;
}
