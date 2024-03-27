/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "amg.h"
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mappings
 *-----------------------------------------------------------------------------*/

/* AMG's interpolation fields */
#define AMGint_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, trunc_factor, FieldTypeDoubleSet)

/* AMG's coarsening fields */
#define AMGcsn_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, rap2, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mod_rap2, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, keep_transpose, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_functions, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, seq_amg_th, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, min_coarse_size, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_coarse_size, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_levels, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_row_sum, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, strong_th, FieldTypeDoubleSet)

/* AMG's aggressive coarsening fields */
#define AMGagg_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_paths, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, trunc_factor, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, P12_max_elements, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, P12_trunc_factor, FieldTypeDoubleSet)

/* AMG's relaxation fields */
#define AMGrlx_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, down_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, up_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, down_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, up_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, order, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, weight, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, outer_weight, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, chebyshev, ChebySetArgs)

/* AMG's complex smoother fields */
#define AMGsmt_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, FSAISetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, ILUSetArgs)

/* AMG */
#define AMG_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, interpolation, AMGintSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, aggressive, AMGaggSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsening, AMGcsnSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relaxation, AMGrlxSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, smoother, AMGsmtSetArgs)

/* Define the prefix list */
#define GENERATE_PREFIXED_LIST_AMG \
   GENERATE_PREFIXED_COMPONENTS(AMGint) \
   GENERATE_PREFIXED_COMPONENTS(AMGcsn) \
   GENERATE_PREFIXED_COMPONENTS(AMGagg) \
   GENERATE_PREFIXED_COMPONENTS(AMGrlx) \
   GENERATE_PREFIXED_COMPONENTS(AMGsmt) \
   GENERATE_PREFIXED_COMPONENTS(AMG)

/* Define num_fields macros for each struct prefix */
#define AMGint_NUM_FIELDS (sizeof(AMGint_field_offset_map) / sizeof(AMGint_field_offset_map[0]))
#define AMGcsn_NUM_FIELDS (sizeof(AMGcsn_field_offset_map) / sizeof(AMGcsn_field_offset_map[0]))
#define AMGagg_NUM_FIELDS (sizeof(AMGagg_field_offset_map) / sizeof(AMGagg_field_offset_map[0]))
#define AMGrlx_NUM_FIELDS (sizeof(AMGrlx_field_offset_map) / sizeof(AMGrlx_field_offset_map[0]))
#define AMGsmt_NUM_FIELDS (sizeof(AMGsmt_field_offset_map) / sizeof(AMGsmt_field_offset_map[0]))
#define AMG_NUM_FIELDS    (sizeof(AMG_field_offset_map)    / sizeof(AMG_field_offset_map[0]))

/* Iterates over each prefix in the list and
   generates the various function declarations/definitions and field_offset_map object */
GENERATE_PREFIXED_LIST_AMG

/*-----------------------------------------------------------------------------
 * AMGintSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGintSetDefaultArgs(AMGint_args *args)
{
   args->prolongation_type = 6;
   args->restriction_type  = 0;
   args->max_nnz_row       = 4;
   args->trunc_factor      = 0.0;
}

/*-----------------------------------------------------------------------------
 * AMGcsnSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGcsnSetDefaultArgs(AMGcsn_args *args)
{
   args->type            = 10;
   args->rap2            = 0;
#if defined (HYPRE_USING_GPU)
   args->mod_rap2        = 1;
   args->keep_transpose  = 1;
#else
   args->mod_rap2        = 0;
   args->keep_transpose  = 0;
#endif
   args->num_functions   = 1;
   args->seq_amg_th      = 0;
   args->min_coarse_size = 0;
   args->max_coarse_size = 64;
   args->max_levels      = 25;
   args->max_row_sum     = 0.9;
   args->strong_th       = 0.25;
}

/*-----------------------------------------------------------------------------
 * AMGaggSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGaggSetDefaultArgs(AMGagg_args *args)
{
   args->num_levels        = 0;
   args->num_paths         = 1;
   args->prolongation_type = 4;
   args->max_nnz_row       = 0;
   args->P12_max_elements  = 0;
   args->P12_trunc_factor  = 0.0;
   args->trunc_factor      = 0.0;
}

/*-----------------------------------------------------------------------------
 * AMGrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGrlxSetDefaultArgs(AMGrlx_args *args)
{
   args->down_type     = 13;
   args->up_type       = 14;
   args->coarse_type   = 9;
   args->down_sweeps   = -1;
   args->up_sweeps     = -1;
   args->coarse_sweeps = 1;
   args->num_sweeps    = 1;
   args->order         = 0;
   args->weight        = 1.0;
   args->outer_weight  = 1.0;

   ChebySetDefaultArgs(&args->chebyshev);
}

/*-----------------------------------------------------------------------------
 * AMGsmtSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGsmtSetDefaultArgs(AMGsmt_args *args)
{
   args->type       = 5;
   args->num_levels = 0;
   args->num_sweeps = 1;
}

/*-----------------------------------------------------------------------------
 * AMGSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
AMGSetDefaultArgs(AMG_args *args)
{
   args->max_iter    = 1;
   args->print_level = 0;
   args->tolerance   = 0.0;

   AMGintSetDefaultArgs(&args->interpolation);
   AMGaggSetDefaultArgs(&args->aggressive);
   AMGcsnSetDefaultArgs(&args->coarsening);
   AMGrlxSetDefaultArgs(&args->relaxation);
   AMGsmtSetDefaultArgs(&args->smoother);
}

/*-----------------------------------------------------------------------------
 * AMGintGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
AMGintGetValidValues(const char* key)
{
   if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {{"mod_classical",         0},
                                {"least_squares",         1},
                                {"mod_classical_he",      2},
                                {"direct_sep_weights",    3},
                                {"multipass",             4},
                                {"multipass_sep_weights", 5},
                                {"extended+i",            6},
                                {"extended+i_C",          7},
                                {"standard",              8},
                                {"standard_sep_weights",  9},
                                {"blk_classical",        10},
                                {"blk_classical_diag",   11},
                                {"F_F",                  12},
                                {"F_F1",                 13},
                                {"extended",             14},
                                {"direct_sep_weights",   15},
                                {"MM_extended",          16},
                                {"MM_extended+i",        17},
                                {"MM_extended+e",        18},
                                {"blk_direct",           24},
                                {"one_point",           100}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "restriction_type"))
   {
      static StrIntMap map[] = {{"AIR_1",          1},
                                {"AIR_2",          2},
                                {"Neumann_AIR_0",  3},
                                {"Neumann_AIR_1",  4},
                                {"Neumann_AIR_2",  5},
                                {"AIR_1.5",       15}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/* TODO */
StrIntMapArray AMGcsnGetValidValues(const char* key) { return STR_INT_MAP_ARRAY_VOID(); }
StrIntMapArray AMGaggGetValidValues(const char* key) { return STR_INT_MAP_ARRAY_VOID(); }
StrIntMapArray AMGrlxGetValidValues(const char* key) { return STR_INT_MAP_ARRAY_VOID(); }
StrIntMapArray AMGsmtGetValidValues(const char* key) { return STR_INT_MAP_ARRAY_VOID(); }
StrIntMapArray AMGGetValidValues(const char* key) { return STR_INT_MAP_ARRAY_VOID(); }

/*-----------------------------------------------------------------------------
 * AMGCreate
 *-----------------------------------------------------------------------------*/

void
AMGCreate(AMG_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon;

   HYPRE_BoomerAMGCreate(&precon);
   HYPRE_BoomerAMGSetInterpType(precon, args->interpolation.prolongation_type);
   HYPRE_BoomerAMGSetRestriction(precon, args->interpolation.restriction_type);
   HYPRE_BoomerAMGSetCoarsenType(precon, args->coarsening.type);
   HYPRE_BoomerAMGSetTol(precon, args->tolerance);
   HYPRE_BoomerAMGSetStrongThreshold(precon, args->coarsening.strong_th);
   HYPRE_BoomerAMGSetSeqThreshold(precon, args->coarsening.seq_amg_th);
   HYPRE_BoomerAMGSetMaxCoarseSize(precon, args->coarsening.max_coarse_size);
   HYPRE_BoomerAMGSetMinCoarseSize(precon, args->coarsening.min_coarse_size);
   HYPRE_BoomerAMGSetTruncFactor(precon, args->interpolation.trunc_factor);
   HYPRE_BoomerAMGSetPMaxElmts(precon, args->interpolation.max_nnz_row);
   HYPRE_BoomerAMGSetPrintLevel(precon, args->print_level);
   HYPRE_BoomerAMGSetNumSweeps(precon, args->relaxation.num_sweeps);
   HYPRE_BoomerAMGSetChebyOrder(precon, args->relaxation.chebyshev.order);
   HYPRE_BoomerAMGSetChebyFraction(precon, args->relaxation.chebyshev.fraction);
   HYPRE_BoomerAMGSetChebyEigEst(precon, args->relaxation.chebyshev.eig_est);
   HYPRE_BoomerAMGSetChebyVariant(precon, args->relaxation.chebyshev.variant);
   HYPRE_BoomerAMGSetChebyScale(precon, args->relaxation.chebyshev.scale);
   HYPRE_BoomerAMGSetRelaxOrder(precon, args->relaxation.order);
   HYPRE_BoomerAMGSetRelaxWt(precon, args->relaxation.weight);
   HYPRE_BoomerAMGSetOuterWt(precon, args->relaxation.outer_weight);
   HYPRE_BoomerAMGSetMaxLevels(precon, args->coarsening.max_levels);
   HYPRE_BoomerAMGSetSmoothType(precon, args->smoother.type);
   HYPRE_BoomerAMGSetSmoothNumSweeps(precon, args->smoother.num_sweeps);
   HYPRE_BoomerAMGSetSmoothNumLevels(precon, args->smoother.num_levels);
   HYPRE_BoomerAMGSetMaxRowSum(precon, args->coarsening.max_row_sum);
   HYPRE_BoomerAMGSetILUType(precon, args->smoother.ilu.type);
   HYPRE_BoomerAMGSetILULocalReordering(precon, args->smoother.ilu.reordering);
   HYPRE_BoomerAMGSetILUTriSolve(precon, args->smoother.ilu.tri_solve);
   HYPRE_BoomerAMGSetILULowerJacobiIters(precon, args->smoother.ilu.lower_jac_iters);
   HYPRE_BoomerAMGSetILUUpperJacobiIters(precon, args->smoother.ilu.upper_jac_iters);
   HYPRE_BoomerAMGSetILULevel(precon, args->smoother.ilu.fill_level);
   HYPRE_BoomerAMGSetILUDroptol(precon, args->smoother.ilu.droptol);
   HYPRE_BoomerAMGSetILUMaxRowNnz(precon, args->smoother.ilu.max_row_nnz);
   HYPRE_BoomerAMGSetILUMaxIter(precon, args->smoother.num_sweeps);
   HYPRE_BoomerAMGSetFSAIAlgoType(precon, args->smoother.fsai.algo_type);
   HYPRE_BoomerAMGSetFSAILocalSolveType(precon, args->smoother.fsai.ls_type);
   HYPRE_BoomerAMGSetFSAIMaxSteps(precon, args->smoother.fsai.max_steps);
   HYPRE_BoomerAMGSetFSAIMaxStepSize(precon, args->smoother.fsai.max_step_size);
   HYPRE_BoomerAMGSetFSAIMaxNnzRow(precon, args->smoother.fsai.max_nnz_row);
   HYPRE_BoomerAMGSetFSAINumLevels(precon, args->smoother.fsai.num_levels);
   HYPRE_BoomerAMGSetFSAIThreshold(precon, args->smoother.fsai.threshold);
   HYPRE_BoomerAMGSetFSAIEigMaxIters(precon, args->smoother.fsai.eig_max_iters);
   HYPRE_BoomerAMGSetFSAIKapTolerance(precon, args->smoother.fsai.kap_tolerance);
   HYPRE_BoomerAMGSetNumFunctions(precon, args->coarsening.num_functions);
   HYPRE_BoomerAMGSetAggNumLevels(precon, args->aggressive.num_levels);
   HYPRE_BoomerAMGSetAggInterpType(precon, args->aggressive.prolongation_type);
   HYPRE_BoomerAMGSetAggTruncFactor(precon, args->aggressive.trunc_factor);
   HYPRE_BoomerAMGSetAggP12TruncFactor(precon, args->aggressive.P12_trunc_factor);
   HYPRE_BoomerAMGSetAggPMaxElmts(precon, args->aggressive.max_nnz_row);
   HYPRE_BoomerAMGSetAggP12MaxElmts(precon, args->aggressive.P12_max_elements);
   HYPRE_BoomerAMGSetNumPaths(precon, args->aggressive.num_paths);
   HYPRE_BoomerAMGSetMaxIter(precon, args->max_iter);
   HYPRE_BoomerAMGSetRAP2(precon, args->coarsening.rap2);
   HYPRE_BoomerAMGSetModuleRAP2(precon, args->coarsening.mod_rap2);
   HYPRE_BoomerAMGSetKeepTranspose(precon, args->coarsening.keep_transpose);

   /* Specific relaxation info (down, up, coarsest) */
   if (args->relaxation.down_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.down_type, 1);
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.down_sweeps, 1);
   }
   if (args->relaxation.up_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.up_type, 2);
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.up_sweeps, 2);
   }
   if (args->relaxation.coarse_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.coarse_sweeps, 3);
   }
   else
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.num_sweeps, 3);
   }
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.coarse_type, 3);

   *precon_ptr = precon;
}
