/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "gmres.h"

/*-----------------------------------------------------------------------------
 * GMRESSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
GMRESSetDefaultArgs(GMRES_args *args)
{
   args->min_iter = 0;
   args->max_iter = 100;
   args->stop_crit = 0;
   args->skip_real_res_check = 0;
   args->krylov_dimension = 30;
   args->rel_change = 0;
   args->logging = 1;
   args->print_level = 1;
   args->rtol = 1.0e-6;
   args->atol = 0.0;
   args->cf_tol = 0.0;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * GMRESSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
GMRESSetArgsFromYAML(GMRES_args *args, YAMLnode *parent)
{
   YAMLnode *child;

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "min_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->stop_crit, "stop_crit", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->skip_real_res_check, "skip_real_res_check", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->krylov_dimension, "krylov_dimension", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->rel_change, "rel_change", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->logging, "logging", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->rtol, "rtol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->atol, "atol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->cf_tol, "cf_tol", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * GMRESCreate
 *-----------------------------------------------------------------------------*/

int
GMRESCreate(MPI_Comm comm, GMRES_args *args, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver solver;

   HYPRE_ParCSRGMRESCreate(comm, &solver);
   HYPRE_GMRESSetMinIter(solver, args->min_iter);
   HYPRE_GMRESSetMaxIter(solver, args->max_iter);
   HYPRE_GMRESSetStopCrit(solver, args->stop_crit);
   HYPRE_GMRESSetSkipRealResidualCheck(solver, args->skip_real_res_check);
   HYPRE_GMRESSetKDim(solver, args->krylov_dimension);
   HYPRE_GMRESSetRelChange(solver, args->rel_change);
   HYPRE_GMRESSetLogging(solver, args->logging);
   HYPRE_GMRESSetPrintLevel(solver, args->print_level);
   HYPRE_GMRESSetTol(solver, args->rtol);
   HYPRE_GMRESSetAbsoluteTol(solver, args->atol);
   HYPRE_GMRESSetConvergenceFactorTol(solver, args->cf_tol);

   *solver_ptr = solver;

   return EXIT_SUCCESS;
}
