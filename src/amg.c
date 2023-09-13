/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "amg.h"

/*-----------------------------------------------------------------------------
 * AMGSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
AMGSetDefaultArgs(AMG_args *args)
{
   args->interp_type = 6;
   args->trunc_factor = 0.0;
   args->pmax = 4;
   args->restrict_type = 0;
   args->rap2 = 0;
#if defined (HYPRE_USING_GPU)
   args->mod_rap2 = 1;
   args->keep_transpose = 1;
#else
   args->mod_rap2 = 0;
   args->keep_transpose = 0;
#endif

   args->coarsen_type = 8;
   args->strong_th = 0.25;
   args->max_row_sum = 0.9;
   args->num_functions = 1;

   args->agg_num_levels = 0;
   args->agg_num_paths = 1;
   args->agg_interp_type = 4;
   args->agg_trunc_factor = 0.0;
   args->agg_P12_trunc_factor = 0.0;
   args->agg_pmax = 0;
   args->agg_P12_max_elements = 0;

   args->relax_num_sweeps = 1;
   args->relax_down = 18;
   args->relax_up = 18;
   args->relax_coarse = 9;
   args->relax_down_sweeps = -1;
   args->relax_up_sweeps = -1;
   args->relax_coarse_sweeps = 1;
   args->relax_order = 1;
   args->relax_weight = 1.0;
   args->relax_outer_weight = 1.0;

   args->cheby_order = 2;
   args->cheby_fraction = 0.3;
   args->cheby_eig_est = 10;
   args->cheby_variant = 0;
   args->cheby_scale = 1;

   args->fsai_algo_type = 1;
   args->fsai_ls_type = 1;
   args->fsai_max_steps = 5;
   args->fsai_max_step_size = 3;
   args->fsai_max_nnz_row = 15;
   args->fsai_num_levels = 1;
   args->fsai_eig_max_iters = 5;
   args->fsai_th = 1.0e-3;
   args->fsai_kap_tol = 1.0e-3;

   ILUSetDefaultArgs(&args->ilu);

   args->smooth_type = 5;
   args->smooth_num_levels = 0;
   args->smooth_num_sweeps = 1;

   args->seq_th = 0;
   args->max_coarse_size = 9;
   args->min_coarse_size = 0;
   args->max_levels = 25;

   args->tol = 0.0;
   args->max_iter = 1;
   args->print_level = 0;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * AMGSetSmootherArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
AMGSetSmootherArgsFromYAML(AMG_args *args, YAMLnode *node)
{
   /* Call specific SetArgsFromYAML function */
   YAML_CALL_IF_OPEN()
   YAML_CALL_IF_VAL_MATCHES(ILUSetArgsFromYAML, &args->ilu, node, "ilu")
   YAML_CALL_IF_CLOSE()

   /* Set smoother codes */
   YAML_SET_IF_OPEN()
   YAML_SET_INTEGER_IF_VAL_MATCHES_TWO(args->smooth_type, "ilu", 5, node)
   YAML_SET_INTEGER_IF_VAL_MATCHES_TWO(args->smooth_type, "fsai", 4, node)
   YAML_SET_IF_CLOSE_(node)

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * AMGSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
AMGSetArgsFromYAML(AMG_args *args, YAMLnode *parent)
{
   YAMLnode *child;

   if (!parent)
   {
      return EXIT_SUCCESS;
   }

   child = parent->children;
   while (child)
   {
      YAML_SET_IF_OPEN()
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->interp_type, "interp_type", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->trunc_factor, "trunc_factor", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->pmax, "pmax", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->restrict_type, "restrict_type", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->rap2, "rap2", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->mod_rap2, "mod_rap2", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->keep_transpose, "keep_transpose", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->coarsen_type, "coarsen_type", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->strong_th, "strong_th", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->max_row_sum, "max_row_sum", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->num_functions, "num_functions", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->agg_num_levels, "agg_num_levels", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->agg_num_paths, "agg_num_paths", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->agg_interp_type, "agg_interp_type", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->agg_trunc_factor, "agg_trunc_factor", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->agg_P12_trunc_factor, "agg_P12_trunc_factor", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->agg_pmax, "agg_pmax", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->agg_P12_max_elements, "agg_P12_max_elements", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_num_sweeps, "relax_num_sweeps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_down, "relax_down", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_up, "relax_up", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_coarse, "relax_coarse", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_down_sweeps, "relax_down_sweeps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_up_sweeps, "relax_up_sweeps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_coarse_sweeps, "relax_coarse_sweeps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->relax_order, "relax_order", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->relax_weight, "relax_weight", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->relax_outer_weight, "relax_outer_weight", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->cheby_order, "cheby_order", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->cheby_fraction, "cheby_fraction", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->cheby_eig_est, "cheby_eig_est", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->cheby_variant, "cheby_variant", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->cheby_scale, "cheby_scale", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_algo_type, "fsai_algo_type", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_ls_type, "fsai_ls_type", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_max_steps, "fsai_max_steps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_max_step_size, "fsai_max_step_size", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_max_nnz_row, "fsai_max_nnz_row", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_num_levels, "fsai_num_levels", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->fsai_eig_max_iters, "fsai_eig_max_iters", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->fsai_th, "fsai_th", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->fsai_kap_tol, "fsai_kap_tol", child)

      YAML_CALL_IF_KEY_MATCHES(AMGSetSmootherArgsFromYAML, args, child, "smoother")

      YAML_SET_INTEGER_IF_KEY_MATCHES(args->smooth_type, "smooth_type", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->smooth_num_levels, "smooth_num_levels", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->smooth_num_sweeps, "smooth_num_sweeps", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->seq_th, "seq_th", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_coarse_size, "max_coarse_size", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->min_coarse_size, "min_coarse_size", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_levels, "max_levels", child)
      YAML_SET_REAL_IF_KEY_MATCHES(args->tol, "tol", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->max_iter, "max_iter", child)
      YAML_SET_INTEGER_IF_KEY_MATCHES(args->print_level, "print_level", child)
      YAML_SET_IF_CLOSE(child)

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * AMGSetArgs
 *-----------------------------------------------------------------------------*/

int
AMGSetArgs(AMG_args *args, YAMLnode *parent)
{
   AMGSetDefaultArgs(args);
   AMGSetArgsFromYAML(args, parent);

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * AMGCreate
 *-----------------------------------------------------------------------------*/

int
AMGCreate(AMG_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon;

   HYPRE_BoomerAMGCreate(&precon);
   HYPRE_BoomerAMGSetInterpType(precon, args->interp_type);
   HYPRE_BoomerAMGSetRestriction(precon, args->restrict_type);
   HYPRE_BoomerAMGSetCoarsenType(precon, args->coarsen_type);
   HYPRE_BoomerAMGSetTol(precon, args->tol);
   HYPRE_BoomerAMGSetStrongThreshold(precon, args->strong_th);
   HYPRE_BoomerAMGSetSeqThreshold(precon, args->seq_th);
   HYPRE_BoomerAMGSetMaxCoarseSize(precon, args->max_coarse_size);
   HYPRE_BoomerAMGSetMinCoarseSize(precon, args->min_coarse_size);
   HYPRE_BoomerAMGSetTruncFactor(precon, args->trunc_factor);
   HYPRE_BoomerAMGSetPMaxElmts(precon, args->pmax);
   HYPRE_BoomerAMGSetPrintLevel(precon, args->print_level);
   HYPRE_BoomerAMGSetNumSweeps(precon, args->relax_num_sweeps);
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relax_down, 1);
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relax_up, 2);
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relax_coarse, 3);
   HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relax_coarse_sweeps, 3);
   if (args->relax_down_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relax_down_sweeps, 1);
   }
   if (args->relax_up_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relax_up_sweeps, 2);
   }
   HYPRE_BoomerAMGSetChebyOrder(precon, args->cheby_order);
   HYPRE_BoomerAMGSetChebyFraction(precon, args->cheby_fraction);
   HYPRE_BoomerAMGSetChebyEigEst(precon, args->cheby_eig_est);
   HYPRE_BoomerAMGSetChebyVariant(precon, args->cheby_variant);
   HYPRE_BoomerAMGSetChebyScale(precon, args->cheby_scale);
   HYPRE_BoomerAMGSetRelaxOrder(precon, args->relax_order);
   HYPRE_BoomerAMGSetRelaxWt(precon, args->relax_weight);
   HYPRE_BoomerAMGSetOuterWt(precon, args->relax_outer_weight);
   HYPRE_BoomerAMGSetMaxLevels(precon, args->max_levels);
   HYPRE_BoomerAMGSetSmoothType(precon, args->smooth_type);
   HYPRE_BoomerAMGSetSmoothNumSweeps(precon, args->smooth_num_sweeps);
   HYPRE_BoomerAMGSetSmoothNumLevels(precon, args->smooth_num_levels);
   HYPRE_BoomerAMGSetMaxRowSum(precon, args->max_row_sum);
   HYPRE_BoomerAMGSetILUType(precon, args->ilu.type);
   HYPRE_BoomerAMGSetILULevel(precon, args->ilu.fill_level);
   HYPRE_BoomerAMGSetILUDroptol(precon, args->ilu.droptol);
   HYPRE_BoomerAMGSetILUMaxRowNnz(precon, args->ilu.max_row_nnz);
   HYPRE_BoomerAMGSetILUMaxIter(precon, args->smooth_num_sweeps);
   HYPRE_BoomerAMGSetFSAIAlgoType(precon, args->fsai_algo_type);
   HYPRE_BoomerAMGSetFSAILocalSolveType(precon, args->fsai_ls_type);
   HYPRE_BoomerAMGSetFSAIMaxSteps(precon, args->fsai_max_steps);
   HYPRE_BoomerAMGSetFSAIMaxStepSize(precon, args->fsai_max_step_size);
   HYPRE_BoomerAMGSetFSAIMaxNnzRow(precon, args->fsai_max_nnz_row);
   HYPRE_BoomerAMGSetFSAINumLevels(precon, args->fsai_num_levels);
   HYPRE_BoomerAMGSetFSAIThreshold(precon, args->fsai_th);
   HYPRE_BoomerAMGSetFSAIEigMaxIters(precon, args->fsai_eig_max_iters);
   HYPRE_BoomerAMGSetFSAIKapTolerance(precon, args->fsai_kap_tol);
   HYPRE_BoomerAMGSetNumFunctions(precon, args->num_functions);
   HYPRE_BoomerAMGSetAggNumLevels(precon, args->agg_num_levels);
   HYPRE_BoomerAMGSetAggInterpType(precon, args->agg_interp_type);
   HYPRE_BoomerAMGSetAggTruncFactor(precon, args->agg_trunc_factor);
   HYPRE_BoomerAMGSetAggP12TruncFactor(precon, args->agg_P12_trunc_factor);
   HYPRE_BoomerAMGSetAggPMaxElmts(precon, args->agg_pmax);
   HYPRE_BoomerAMGSetAggP12MaxElmts(precon, args->agg_P12_max_elements);
   HYPRE_BoomerAMGSetNumPaths(precon, args->agg_num_paths);
   HYPRE_BoomerAMGSetMaxIter(precon, args->max_iter);
   HYPRE_BoomerAMGSetRAP2(precon, args->rap2);
   HYPRE_BoomerAMGSetModuleRAP2(precon, args->mod_rap2);
   HYPRE_BoomerAMGSetKeepTranspose(precon, args->keep_transpose);

   *precon_ptr = precon;

   return EXIT_SUCCESS;
}
