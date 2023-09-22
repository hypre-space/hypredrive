/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "mgr.h"

/*-----------------------------------------------------------------------------
 * MGRSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
MGRSetDefaultArgs(MGR_args *args)
{
   args->max_iter = 1;
   args->num_levels = 0;
   args->print_level = 0;
   args->non_c_to_f = 1;
   args->pmax = 0;
   args->tol = 0.0;
   args->coarse_th = 0.0;
   args->relax_type = 7;

   for (int i = 0; i < MAX_MGR_LEVELS; i++)
   {
      args->num_f_dofs[i] = 0;
      args->f_dofs[i] = NULL;
      args->frelax_types[i] = 0;  /* TODO: replace with MGR_FRELAX_NONE? */
      args->grelax_types[i] = -1; /* TODO: replace with MGR_GRELAX_NONE? */
      args->num_grelax_sweeps[i] = 0;
      args->num_frelax_sweeps[i] = 0;
      args->prolongation_types[i] = 0;
      args->restriction_types[i] = 0;
      args->coarse_grid_types[i] = 0;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetFRelaxArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetFRelaxArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->lvl;

   /* Return if value is "none" */
   YAML_RETURN_IF_VAL_MATCHES(parent, "none")

   /* If f-relax key is present, the default number of sweeps is one */
   args->num_frelax_sweeps[lvl] = 1;
   YAML_SET_INTEGER_IF_KEY_EXISTS(args->num_frelax_sweeps[lvl], "sweeps", parent)

   /* Call specific SetArgs function */
   YAML_CALL_IF_OPEN()
   YAML_CALL_IF_VAL_MATCHES(AMGSetArgs, &args->frelax[lvl].amg, parent, "boomeramg")
   YAML_CALL_IF_CLOSE()

   /* Set F-relaxation codes */
   /* TODO: we shouldn't need separate logics involving relax_type and frelax_types[lvl] */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "jacobi", 7, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->frelax_types[lvl], "v(1,0)", 1, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->frelax_types[lvl], "boomeramg", 2, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "h-fgs", 3, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "h-bgs", 4, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "h-ssor", 6, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "hl1-ssor", 8, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->frelax_types[lvl], "ge", 9, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "l1-fgs", 13, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "l1-bgs", 14, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "chebyshev", 16, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->relax_type, "l1-jacobi", 18, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->frelax_types[lvl], "ge-piv", 99, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->frelax_types[lvl], "ge-inv", 199, parent)
   YAML_SET_IF_CLOSE_(parent)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetGRelaxArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetGRelaxArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->lvl;

   /* Return if value is "none" */
   YAML_RETURN_IF_VAL_MATCHES(parent, "none")

   /* If g-relax key is present, the default number of sweeps is one */
   args->num_grelax_sweeps[lvl] = 1;
   YAML_SET_INTEGER_IF_KEY_EXISTS(args->num_grelax_sweeps[lvl], "sweeps", parent)

   /* Call specific SetArgs function */
   YAML_CALL_IF_OPEN()
   YAML_CALL_IF_VAL_MATCHES(ILUSetArgs, &args->grelax[lvl].ilu, parent, "ilu")
   YAML_CALL_IF_CLOSE()

   /* Set Global relaxation codes */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->grelax_types[lvl], "blk-jacobi", 0, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->grelax_types[lvl], "jacobi", 1, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->grelax_types[lvl], "ilu", 16, parent)
   YAML_SET_IF_CLOSE_(parent)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetProlongationArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetProlongationArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->lvl;

   /* Set prolongation codes */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "injection", 0, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "l1-jacobi", 1, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "jacobi", 2, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "classical-mod", 3, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "approx-inv", 4, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->prolongation_types[lvl], "blk-jacobi", 12, parent)
   YAML_SET_IF_CLOSE_(parent)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetRestrictionArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetRestrictionArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->lvl;

   /* Set prolongation codes */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->restriction_types[lvl], "injection", 0, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->restriction_types[lvl], "jacobi", 2, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->restriction_types[lvl], "approx-inv", 3, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->restriction_types[lvl], "blk-jacobi", 12, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->restriction_types[lvl], "cpr-like", 13, parent)
   YAML_SET_IF_CLOSE_(parent)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetCoarseGridArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetCoarseGridArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->lvl;

   /* Set prolongation codes */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->coarse_grid_types[lvl], "rap", 0, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->coarse_grid_types[lvl], "non-galerkin", 1, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->coarse_grid_types[lvl], "cpr-like-diag", 2, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->coarse_grid_types[lvl], "cpr-like-bdiag", 3, parent)
   YAML_SET_INTEGER_IF_VAL_MATCHES_ANYTWO(args->coarse_grid_types[lvl], "approx-inv", 4, parent)
   YAML_SET_IF_CLOSE_(parent)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetCoarseSolverArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetCoarseSolverArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   HYPRE_Int lvl = args->num_levels - 1;

   /* If coarse_solver key is present, the default number of sweeps is one */
   args->num_grelax_sweeps[lvl] = 1;
   YAML_SET_INTEGER_IF_KEY_EXISTS(args->num_grelax_sweeps[lvl], "sweeps", parent)

   /* Call specific SetArgs functions */
   if (!strcmp(parent->val, "boomeramg") || !strcmp(parent->val, "0"))
   {
      AMGSetArgs(&args->grelax[lvl].amg, parent);
      args->grelax_types[lvl] = 0;
   }
   /* TODO: Add SuperLU */

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetLevelArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetLevelArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   YAMLnode *child;

   if (!parent)
   {
      return EXIT_SUCCESS;
   }

   /* Which MGR level am I? */
   args->lvl = atoi(parent->val);

   /* Set MGR level options */
   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTARRAY_IF_KEY_MATCHES(&args->num_f_dofs[args->lvl],
                                       &args->f_dofs[args->lvl], "f_dofs", child)
      YAML_CALL_IF_KEY_MATCHES(MGRSetFRelaxArgsFromYAML, args, child, "f_relaxation")
      YAML_CALL_IF_KEY_MATCHES(MGRSetGRelaxArgsFromYAML, args, child, "g_relaxation")
      YAML_CALL_IF_KEY_MATCHES(MGRSetProlongationArgsFromYAML, args, child, "prolongation")
      YAML_CALL_IF_KEY_MATCHES(MGRSetRestrictionArgsFromYAML, args, child, "restriction")
      YAML_CALL_IF_KEY_MATCHES(MGRSetCoarseGridArgsFromYAML, args, child, "coarse_grid")
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
MGRSetArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   YAMLnode *child;

   if (!parent)
   {
      return EXIT_SUCCESS;
   }

   /* Compute num_levels */
   YAML_INC_INTEGER_IF_KEY_EXISTS(args->num_levels, "level", parent)
   YAML_INC_INTEGER_IF_KEY_EXISTS(args->num_levels, "coarse_solver", parent)

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_REAL_IF_KEY_MATCHES(args->tol, "tol", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->coarse_th, "coarse_th", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->pmax, "pmax", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_CALL_IF_KEY_MATCHES(MGRSetLevelArgsFromYAML, args, child, "level")
      YAML_CALL_IF_KEY_MATCHES(MGRSetCoarseSolverArgsFromYAML, args, child, "coarse_solver")
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRDestroyArgs
 *-----------------------------------------------------------------------------*/

int
MGRDestroyArgs(MGR_args **args_ptr)
{
   MGR_args *args;

   args = *args_ptr;
   for (int i = 0; i < args->num_levels; i++)
   {
      free(args->f_dofs[i]);
   }

   *args_ptr = NULL;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * MGRCreate
 *-----------------------------------------------------------------------------*/

int
MGRCreate(MGR_args *args, HYPRE_IntArray *dofmap, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver   precon;
   HYPRE_Solver   csolver;
   HYPRE_Solver   frelax;
   HYPRE_Int      num_dofs = dofmap->num_unique_entries;
   HYPRE_Int      num_dofs_last = num_dofs;
   HYPRE_Int      num_levels = args->num_levels;
   HYPRE_Int      num_c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int     *c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int     *inactive_dofs;
   HYPRE_Int      lvl, i, j;

   /* Compute num_c_dofs and c_dofs */
   inactive_dofs = (HYPRE_Int*) calloc(num_dofs, sizeof(HYPRE_Int));
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      c_dofs[lvl] = (HYPRE_Int*) malloc(num_dofs * sizeof(HYPRE_Int));
      num_c_dofs[lvl] = num_dofs_last - args->num_f_dofs[lvl];

      for (i = 0; i < args->num_f_dofs[lvl]; i++)
      {
         inactive_dofs[args->f_dofs[lvl][i]] = 1;
         --num_dofs_last;
      }

      for (i = 0, j = 0; i < num_dofs; i++)
      {
         if (!inactive_dofs[i])
         {
            c_dofs[lvl][j++] = i;
         }
      }
   }

   /* Config preconditioner */
   HYPRE_MGRCreate(&precon);
   HYPRE_MGRSetCpointsByPointMarkerArray(precon, num_dofs, num_levels - 1,
                                         num_c_dofs, c_dofs, dofmap->data);
   HYPRE_MGRSetNonCpointsToFpoints(precon, args->non_c_to_f);
   HYPRE_MGRSetMaxIter(precon, args->max_iter);
   HYPRE_MGRSetTol(precon, args->tol);
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
   HYPRE_MGRSetLevelFRelaxType(precon, args->frelax_types);
   HYPRE_MGRSetRelaxType(precon, args->relax_type); /* TODO: we shouldn't need this */
   HYPRE_MGRSetLevelNumRelaxSweeps(precon, args->num_frelax_sweeps);
   HYPRE_MGRSetLevelSmoothIters(precon, args->num_grelax_sweeps);
   HYPRE_MGRSetLevelSmoothType(precon, args->grelax_types);
   HYPRE_MGRSetLevelInterpType(precon, args->prolongation_types);
   HYPRE_MGRSetLevelRestrictType(precon, args->restriction_types);
   HYPRE_MGRSetCoarseGridMethod(precon, args->coarse_grid_types);

   /* Config finest level f-relaxation */
   if (args->frelax_types[0] == 2)
   {
      AMGCreate(&args->frelax[0].amg, &frelax);
      HYPRE_MGRSetCoarseSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, frelax);
   }

   /* Config coarsest level solver */
   if (args->grelax_types[num_levels - 1] == 0)
   {
      AMGCreate(&args->grelax[num_levels - 1].amg, &csolver);
      HYPRE_MGRSetCoarseSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, csolver);
   }

   /* Set output pointer */
   *precon_ptr = precon;

   /* Free memory */
   free(inactive_dofs);
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      free(c_dofs[lvl]);
   }

   return EXIT_SUCCESS;
}
