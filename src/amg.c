/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "amg.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_IJ_mv.h"     // For hypre_IJVectorGlobalNumRows
#include "_hypre_parcsr_mv.h" // For hypre_ParVectorComm, hypre_ParVectorInitialize_v2
#include "gen_macros.h"

/*-----------------------------------------------------------------------------
 * Define Field/Offset/Setter mappings
 *-----------------------------------------------------------------------------*/

/* AMG's interpolation fields */
#define AMGint_FIELDS(_prefix)                                         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, trunc_factor, FieldTypeDoubleSet)

/* AMG's coarsening fields */
#define AMGcsn_FIELDS(_prefix)                                        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, rap2, FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mod_rap2, FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, keep_transpose, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_functions, FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, filter_functions, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, nodal, FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, seq_amg_th, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, min_coarse_size, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_coarse_size, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_levels, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_row_sum, FieldTypeDoubleSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, strong_th, FieldTypeDoubleSet)

/* AMG's aggressive coarsening fields */
#define AMGagg_FIELDS(_prefix)                                           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_paths, FieldTypeIntSet)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_nnz_row, FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, trunc_factor, FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, P12_max_elements, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, P12_trunc_factor, FieldTypeDoubleSet)

/* AMG's relaxation fields */
#define AMGrlx_FIELDS(_prefix)                                       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, down_type, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, up_type, FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_type, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, down_sweeps, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, up_sweeps, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_sweeps, FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, order, FieldTypeIntSet)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, weight, FieldTypeDoubleSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, outer_weight, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, chebyshev, ChebySetArgs)

/* AMG's complex smoother fields */
#define AMGsmt_FIELDS(_prefix)                                  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, FSAISetArgs)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, ILUSetArgs)

/* AMG */
#define AMG_FIELDS(_prefix)                                       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, interpolation, AMGintSetArgs)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, aggressive, AMGaggSetArgs)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsening, AMGcsnSetArgs)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relaxation, AMGrlxSetArgs)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, smoother, AMGsmtSetArgs)

/* Define the prefix list */
#define GENERATE_PREFIXED_LIST_AMG      \
   GENERATE_PREFIXED_COMPONENTS(AMGint) \
   GENERATE_PREFIXED_COMPONENTS(AMGcsn) \
   GENERATE_PREFIXED_COMPONENTS(AMGagg) \
   GENERATE_PREFIXED_COMPONENTS(AMGrlx) \
   GENERATE_PREFIXED_COMPONENTS(AMGsmt) \
   GENERATE_PREFIXED_COMPONENTS(AMG)

/* Define num_fields macros for each struct prefix */
// clang-format off
#define AMGint_NUM_FIELDS (sizeof(AMGint_field_offset_map) / sizeof(AMGint_field_offset_map[0]))
#define AMGcsn_NUM_FIELDS (sizeof(AMGcsn_field_offset_map) / sizeof(AMGcsn_field_offset_map[0]))
#define AMGagg_NUM_FIELDS (sizeof(AMGagg_field_offset_map) / sizeof(AMGagg_field_offset_map[0]))
#define AMGrlx_NUM_FIELDS (sizeof(AMGrlx_field_offset_map) / sizeof(AMGrlx_field_offset_map[0]))
#define AMGsmt_NUM_FIELDS (sizeof(AMGsmt_field_offset_map) / sizeof(AMGsmt_field_offset_map[0]))
#define AMG_NUM_FIELDS    (sizeof(AMG_field_offset_map)    / sizeof(AMG_field_offset_map[0]))
// clang-format on

/* Iterates over each prefix in the list and
   generates the various function declarations/definitions and field_offset_map object */
GENERATE_PREFIXED_LIST_AMG             // LCOV_EXCL_LINE
DEFINE_VOID_GET_VALID_VALUES_FUNC(AMG) // LCOV_EXCL_LINE

   /*-----------------------------------------------------------------------------
    * AMGintSetDefaultArgs
    *-----------------------------------------------------------------------------*/

   void AMGintSetDefaultArgs(AMGint_args *args)
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
   args->rap2 = 0;
#ifdef HYPRE_USING_GPU
   args->mod_rap2       = 1;
   args->keep_transpose = 1;
   args->type           = 8;
#else
   args->mod_rap2       = 0;
   args->keep_transpose = 0;
   args->type           = 10;
#endif
   args->num_functions    = 1;
   args->filter_functions = 0;
   args->nodal            = 0;
   args->seq_amg_th       = 0;
   args->min_coarse_size  = 0;
   args->max_coarse_size  = 64;
   args->max_levels       = 25;
   args->max_row_sum      = 0.9;
   args->strong_th        = 0.25;
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
#ifdef HYPRE_USING_GPU
   args->down_type = 18;
   args->up_type   = 18;
#else
   args->down_type = 13;
   args->up_type   = 14;
#endif
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
   args->num_rbms    = 0;
   args->rbms[0]     = NULL;
   args->rbms[1]     = NULL;
   args->rbms[2]     = NULL;

   AMGintSetDefaultArgs(&args->interpolation);
   AMGaggSetDefaultArgs(&args->aggressive);
   AMGcsnSetDefaultArgs(&args->coarsening);
   AMGrlxSetDefaultArgs(&args->relaxation);
   AMGsmtSetDefaultArgs(&args->smoother);
}

/*-----------------------------------------------------------------------------
 * AMGintGetValidValues
 *-----------------------------------------------------------------------------*/

// clang-format off
StrIntMapArray
AMGintGetValidValues(const char *key)
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
                                {"extended+i_c",          7},
                                {"standard",              8},
                                {"standard_sep_weights",  9},
                                {"blk_classical",        10},
                                {"blk_classical_diag",   11},
                                {"f_f",                  12},
                                {"f_f1",                 13},
                                {"extended",             14},
                                {"direct_sep_weights",   15},
                                {"mm_extended",          16},
                                {"mm_extended+i",        17},
                                {"mm_extended+e",        18},
                                {"blk_direct",           24},
                                {"one_point",           100}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "restriction_type"))
   {
      static StrIntMap map[] = {{"p_transpose",    0},
                                {"air_1",          1},
                                {"air_2",          2},
                                {"neumann_air_0",  3},
                                {"neumann_air_1",  4},
                                {"neumann_air_2",  5},
                                {"air_1.5",       15}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * AMGcsnGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
AMGcsnGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"cljp",    0},
                                {"rs",      1},
                                {"rs3",     3},
                                {"falgout", 6},
                                {"pmis",    8},
                                {"hmis",   10}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "filter_functions") ||
       !strcmp(key, "nodal") ||
       !strcmp(key, "rap2") ||
       !strcmp(key, "mod_rap2") ||
       !strcmp(key, "keep_transpose"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * AMGaggGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
AMGaggGetValidValues(const char *key)
{
   if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {{"2_stage_extended+i", 1},
                                {"2_stage_standard",   2},
                                {"2_stage_extended",   3},
                                {"multipass",          4},
                                {"mm_extended",        5},
                                {"mm_extended+i",      6},
                                {"mm_extended+e",      7}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * AMGrlxGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
AMGrlxGetValidValues(const char *key)
{
   if (!strcmp(key, "down_type"))
   {
      static StrIntMap map[] = {{"jacobi_non_mv",  0},
                                {"forward-hgs",    3},
                                {"chaotic-hgs",    5},
                                {"hsgs",           6},
                                {"jacobi",         7},
                                {"l1-hsgs",        8},
                                {"2gs-it1",       11},
                                {"2gs-it2",       12},
                                {"forward-hl1gs", 13},
                                {"cg",            15},
                                {"chebyshev",     16},
                                {"l1-jacobi",     18},
                                {"l1sym-hgs",     89}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "up_type"))
   {
      static StrIntMap map[] = {{"jacobi_non_mv",   0},
                                {"backward-hgs",    4},
                                {"chaotic-hgs",     5},
                                {"hsgs",            6},
                                {"jacobi",          7},
                                {"l1-hsgs",         8},
                                {"2gs-it1",        11},
                                {"2gs-it2",        12},
                                {"backward-hl1gs", 14},
                                {"cg",             15},
                                {"chebyshev",      16},
                                {"l1-jacobi",      18},
                                {"l1sym-hgs",      89}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "coarse_type"))
   {
      static StrIntMap map[] = {{"jacobi_non_mv",   0},
                                {"hsgs",            6},
                                {"jacobi",          7},
                                {"l1-hsgs",         8},
                                {"ge",              9},
                                {"2gs-it1",        11},
                                {"2gs-it2",        12},
                                {"forward-hl1gs",  13},
                                {"backward-hl1gs", 14},
                                {"cg",             15},
                                {"chebyshev",      16},
                                {"l1-jacobi",      18},
                                {"l1sym-hgs",      89},
                                {"lu_piv",         99},
                                {"lu_inv",        199}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * AMGsmtGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
AMGsmtGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"fsai",      4},
                                {"ilu",       5},
                                {"schwarz",   6},
                                {"pilut",     7},
                                {"parasails", 8},
                                {"euclid",    9}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}
// clang-format on

/*-----------------------------------------------------------------------------
 * AMGSetRBMs
 *-----------------------------------------------------------------------------*/

void
AMGSetRBMs(AMG_args *args, HYPRE_IJVector vec_nn)
{
   HYPRE_BigInt   jlower = 0, jupper = 0;
   HYPRE_Int      num_entries = 0;
   HYPRE_Complex *values      = NULL;

   /* Sanity: check if the near null space vector is set
      We do not error out when NOT using nodal coarsening. */
   if (!vec_nn || !args->coarsening.nodal)
   {
      if (args->coarsening.nodal)
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("Near null space vectors (RBMs) required"
                     " for nodal coarsening, but not set");
      }
      return;
   }
   HYPRE_IJVectorGetLocalRange(vec_nn, &jlower, &jupper);
   num_entries = (HYPRE_Int)(jupper - jlower + 1);
   values      = (HYPRE_Complex *)malloc((size_t)num_entries * sizeof(HYPRE_Complex));

   /* Reset any previous RBMs */
   for (HYPRE_Int i = 0; i < args->num_rbms; i++)
   {
      HYPRE_ParVectorDestroy(args->rbms[i]);
      args->rbms[i] = NULL;
   }

   /* Create three RBMs */
   args->num_rbms = 3;
   for (HYPRE_Int i = 0; i < args->num_rbms; i++)
   {
      /* Allocate single-component parallel vector for this RBM */
      HYPRE_BigInt partitioning[2] = {jlower, jupper + 1};

      HYPRE_ParVectorCreate(hypre_ParVectorComm(vec_nn),
                            hypre_IJVectorGlobalNumRows(vec_nn), partitioning,
                            &args->rbms[i]);
      hypre_ParVectorInitialize_v2(args->rbms[i], HYPRE_MEMORY_HOST);

      /* Copy component data into host buffer */
      HYPRE_IJVectorSetComponent(vec_nn, 3 + i);
      HYPRE_IJVectorGetValues(vec_nn, num_entries, NULL, values);

      /* Fill entries */
      for (HYPRE_Int j = 0; j < num_entries; j++)
      {
         hypre_ParVectorEntryI(args->rbms[i], j) = values[j];
      }
   }

   /* Free memory */
   free(values);
}

/*-----------------------------------------------------------------------------
 * AMGCreate
 *-----------------------------------------------------------------------------*/

void
AMGCreate(const AMG_args *args, HYPRE_Solver *precon_ptr)
{
   HYPRE_Solver precon = NULL;

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

   /* Pre-smoothing */
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.down_type, 1);
   if (args->relaxation.down_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.down_sweeps, 1);
   }
   else
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.num_sweeps, 1);
   }

   /* Post-smoothing */
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.up_type, 2);
   if (args->relaxation.up_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.up_sweeps, 2);
   }
   else
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.num_sweeps, 2);
   }

   /* Coarsest level smoothing */
   HYPRE_BoomerAMGSetCycleRelaxType(precon, args->relaxation.coarse_type, 3);
   if (args->relaxation.coarse_sweeps > -1)
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.coarse_sweeps, 3);
   }
   else
   {
      HYPRE_BoomerAMGSetCycleNumSweeps(precon, args->relaxation.num_sweeps, 3);
   }

   if (args->coarsening.nodal)
   {
      HYPRE_BoomerAMGSetNumFunctions(precon, 3);
      HYPRE_BoomerAMGSetNodal(precon, 4); // Nodal coarsening based on row-sum norm
      HYPRE_BoomerAMGSetNodalDiag(precon, 1);
      HYPRE_BoomerAMGSetInterpVecVariant(precon, 2); // GM-2
      HYPRE_BoomerAMGSetInterpVecQMax(precon, 4);
      HYPRE_BoomerAMGSetSmoothInterpVectors(precon, 1);
      HYPRE_BoomerAMGSetInterpVectors(precon, args->num_rbms,
                                      (HYPRE_ParVector *)args->rbms);
   }

   *precon_ptr = precon;
}
