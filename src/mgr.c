/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "mgr.h"
#include "stats.h"
#include "gen_macros.h"

#define MGRcls_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, AMGSetArgs)

#define MGRfrlx_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, AMGSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, ILUSetArgs)

#define MGRgrlx_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, ILUSetArgs)

#define MGRlvl_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_dofs, FieldTypeStackIntArraySet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_level_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_relaxation, MGRfrlxSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, g_relaxation, MGRgrlxSetArgs)

#define MGR_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, non_c_to_f, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pmax, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relax_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_th, FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsest_level, MGRclsSetArgs)

#define MGRcls_NUM_FIELDS  (sizeof(MGRcls_field_offset_map)  / sizeof(MGRcls_field_offset_map[0]))
#define MGRfrlx_NUM_FIELDS (sizeof(MGRfrlx_field_offset_map) / sizeof(MGRfrlx_field_offset_map[0]))
#define MGRgrlx_NUM_FIELDS (sizeof(MGRgrlx_field_offset_map) / sizeof(MGRgrlx_field_offset_map[0]))
#define MGRlvl_NUM_FIELDS  (sizeof(MGRlvl_field_offset_map)  / sizeof(MGRlvl_field_offset_map[0]))
#define MGR_NUM_FIELDS     (sizeof(MGR_field_offset_map)     / sizeof(MGR_field_offset_map[0]))

/* Define the prefix list */
#define GENERATE_PREFIXED_LIST_MGR \
   GENERATE_PREFIXED_COMPONENTS(MGRcls) \
   GENERATE_PREFIXED_COMPONENTS(MGRfrlx) \
   GENERATE_PREFIXED_COMPONENTS(MGRgrlx) \
   GENERATE_PREFIXED_COMPONENTS(MGRlvl)

/* Iterates over each prefix in the list and
   generates the various function declarations/definitions and field_offset_map object */
GENERATE_PREFIXED_LIST_MGR

DEFINE_FIELD_OFFSET_MAP(MGR);
DEFINE_SET_FIELD_BY_NAME_FUNC(MGRSetFieldByName,
                              MGR_args,
                              MGR_field_offset_map,
                              MGR_NUM_FIELDS);
DEFINE_GET_VALID_KEYS_FUNC(MGRGetValidKeys,
                           MGR_NUM_FIELDS,
                           MGR_field_offset_map);
DECLARE_GET_VALID_VALUES_FUNC(MGR);
DECLARE_SET_DEFAULT_ARGS_FUNC(MGR);
DECLARE_SET_ARGS_FROM_YAML_FUNC(MGR);
DEFINE_SET_ARGS_FUNC(MGR);

/*-----------------------------------------------------------------------------
 * MGRclsSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRclsSetDefaultArgs(MGRcls_args *args)
{
   args->type = 0;

   AMGSetDefaultArgs(&args->amg);
}

/*-----------------------------------------------------------------------------
 * MGRfrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRfrlxSetDefaultArgs(MGRfrlx_args *args)
{
   args->type = 7;
   args->num_sweeps = 1;

   AMGSetDefaultArgs(&args->amg); args->amg.max_iter = 0;
   ILUSetDefaultArgs(&args->ilu); args->ilu.max_iter = 0;
}

/*-----------------------------------------------------------------------------
 * MGRgrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRgrlxSetDefaultArgs(MGRgrlx_args *args)
{
   args->type = 2;
   args->num_sweeps = 0;

   ILUSetDefaultArgs(&args->ilu); args->ilu.max_iter = 0;
}

/*-----------------------------------------------------------------------------
 * MGRlvlSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRlvlSetDefaultArgs(MGRlvl_args *args)
{
   args->f_dofs             = STACK_INTARRAY_CREATE();
   args->prolongation_type  = 0;
   args->restriction_type   = 0;
   args->coarse_level_type  = 0;

   MGRfrlxSetDefaultArgs(&args->f_relaxation);
   MGRgrlxSetDefaultArgs(&args->g_relaxation);
}

/*-----------------------------------------------------------------------------
 * MGRSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRSetDefaultArgs(MGR_args *args)
{
   args->max_iter = 1;
   args->num_levels = 0;
   args->print_level = 0;
   args->non_c_to_f = 1;
   args->pmax = 0;
   args->tolerance = 0.0;
   args->coarse_th = 0.0;
   args->relax_type = 7;

   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      MGRlvlSetDefaultArgs(&args->level[i]);
   }
   MGRclsSetDefaultArgs(&args->coarsest_level);
}

/*-----------------------------------------------------------------------------
 * MGRclsGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRclsGetValidValues(const char* key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"amg", 0}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRfrlxGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRfrlxGetValidValues(const char* key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"",        -1},
                                {"none",    -1},
                                {"single",   7},
                                {"jacobi",   7},
                                {"v(1,0)",   1},
                                {"amg",      2},
                                {"ilu",     16},
                                {"ge",       9},
                                {"ge-piv",  99},
                                {"ge-inv", 199}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRgrlxGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRgrlxGetValidValues(const char* key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"",           -1},
                                {"none",       -1},
                                {"blk-jacobi",  0},
                                {"blk-gs",      1},
                                {"mixed-gs",    2},
                                {"h-fgs",       3},
                                {"h-bgs",       4},
                                {"ch-gs",       5},
                                {"h-ssor",      6},
                                {"euclid",      8},
                                {"2stg-fgs",   11},
                                {"2stg-bgs",   12},
                                {"l1-hfgs",    13},
                                {"l1-hbgs",    14},
                                {"ilu",        16},
                                {"l1-hsgs",    88}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRlvlGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRlvlGetValidValues(const char* key)
{
   if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {{"injection",      0},
                                {"l1-jacobi",      1},
                                {"jacobi",         2},
                                {"classical-mod",  3},
                                {"approx-inv",     4},
                                {"blk-jacobi",    12}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "restriction_type"))
   {
      static StrIntMap map[] = {{"injection",   0},
                                {"jacobi",      2},
                                {"approx-inv",  3},
                                {"blk-jacobi", 12},
                                {"cpr-like",   13},
                                {"columped",   14}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "coarse_level_type"))
   {
      static StrIntMap map[] = {{"rap",            0},
                                {"non-galerkin",   1},
                                {"cpr-like-diag",  2},
                                {"cpr-like-bdiag", 3},
                                {"approx-inv",     4},
                                {"rai",            5}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "f_relaxation"))
   {
      return MGRfrlxGetValidValues("type");
   }
   else if (!strcmp(key, "g_relaxation"))
   {
      return MGRgrlxGetValidValues("type");
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRGetValidValues(const char* key)
{
   if (!strcmp(key, "relax_type"))
   {
      static StrIntMap map[] = {{"jacobi",     7},
                                {"h-fgs",      3},
                                {"h-bgs",      4},
                                {"ch-gs",      5},
                                {"h-ssor",     6},
                                {"hl1-ssor",   8},
                                {"l1-fgs",    13},
                                {"l1-bgs",    14},
                                {"chebyshev", 16},
                                {"l1-jacobi", 18}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
MGRSetArgsFromYAML(MGR_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      if (!strcmp(child->key, "level"))
      {
         YAML_NODE_ITERATE(child, grandchild)
         {
            int lvl = atoi(grandchild->key);

            if (lvl >= 0 && lvl < MAX_MGR_LEVELS - 1)
            {
               YAML_NODE_ITERATE(grandchild, great_grandchild)
               {
                  YAML_NODE_VALIDATE(great_grandchild,
                                     MGRlvlGetValidKeys,
                                     MGRlvlGetValidValues);

                  YAML_NODE_SET_FIELD(great_grandchild,
                                      &args->level[lvl],
                                      MGRlvlSetFieldByName);
               }

               args->num_levels++;
               YAML_NODE_SET_VALID(grandchild);
            }
            else
            {
               YAML_NODE_SET_INVALID_KEY(grandchild);
            }
         }
      }
      else
      {
         if (!strcmp(child->key, "coarsest_level"))
         {
            args->num_levels++;
         }

         YAML_NODE_VALIDATE(child,
                            MGRGetValidKeys,
                            MGRGetValidValues);

         YAML_NODE_SET_FIELD(child,
                             args,
                             MGRSetFieldByName);
      }
   }
}

/*-----------------------------------------------------------------------------
 * MGRConvertArgInt
 *-----------------------------------------------------------------------------*/

HYPRE_Int*
MGRConvertArgInt(MGR_args *args, const char* name)
{
   static HYPRE_Int buf[MAX_MGR_LEVELS - 1];

   /* Sanity check */
   if (args->num_levels >= (MAX_MGR_LEVELS - 1))
   {
      return NULL;
   }

   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, f_relaxation.type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, f_relaxation.num_sweeps)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, g_relaxation.type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, g_relaxation.num_sweeps)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, prolongation_type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, restriction_type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, coarse_level_type)

   /* If we haven't returned yet, return a NULL pointer */
   return NULL;
}

/*-----------------------------------------------------------------------------
 * MGRSetDofmap
 *-----------------------------------------------------------------------------*/

void
MGRSetDofmap(MGR_args *args, IntArray *dofmap)
{
   args->dofmap = dofmap;
}

/*-----------------------------------------------------------------------------
 * MGRCreate
 *-----------------------------------------------------------------------------*/

void
MGRCreate(MGR_args *args, HYPRE_Solver *precon_ptr, HYPRE_Solver *csolver_ptr)
{
   HYPRE_Solver   precon;
   HYPRE_Solver   csolver;
   HYPRE_Solver   frelax;
   HYPRE_Solver   grelax;
   IntArray      *dofmap;
   HYPRE_Int      num_dofs;
   HYPRE_Int      num_dofs_last;
   HYPRE_Int      num_levels;
   HYPRE_Int      num_c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int     *c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int     *inactive_dofs;
   HYPRE_Int      lvl, i, j;

   /* Sanity checks */
   if (!args->dofmap)
   {
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
      return;
   }

   /* Initialize variables */
   dofmap     = args->dofmap;
   num_dofs   = dofmap->g_unique_size;
   num_levels = args->num_levels;

   /* Compute num_c_dofs and c_dofs */
   num_dofs_last = num_dofs;
   inactive_dofs = (HYPRE_Int*) calloc(num_dofs, sizeof(HYPRE_Int));
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      c_dofs[lvl] = (HYPRE_Int*) malloc(num_dofs * sizeof(HYPRE_Int));
      num_c_dofs[lvl] = num_dofs_last - args->level[lvl].f_dofs.size;

      for (i = 0; i < (int) args->level[lvl].f_dofs.size; i++)
      {
         inactive_dofs[args->level[lvl].f_dofs.data[i]] = 1;
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
   HYPRE_MGRSetTol(precon, args->tolerance);
   HYPRE_MGRSetPrintLevel(precon, args->print_level);
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
   HYPRE_MGRSetRelaxType(precon, args->relax_type); /* TODO: we shouldn't need this */

   /* Set level parameters */
   HYPRE_MGRSetLevelFRelaxType(precon, MGRConvertArgInt(args, "f_relaxation:type"));
   HYPRE_MGRSetLevelNumRelaxSweeps(precon, MGRConvertArgInt(args, "f_relaxation:num_sweeps"));
   HYPRE_MGRSetLevelSmoothType(precon, MGRConvertArgInt(args, "g_relaxation:type"));
   HYPRE_MGRSetLevelSmoothIters(precon, MGRConvertArgInt(args, "g_relaxation:num_sweeps"));
   HYPRE_MGRSetLevelInterpType(precon, MGRConvertArgInt(args, "prolongation_type"));
   HYPRE_MGRSetLevelRestrictType(precon, MGRConvertArgInt(args, "restriction_type"));
   HYPRE_MGRSetCoarseGridMethod(precon, MGRConvertArgInt(args, "coarse_level_type"));

   /* Config finest level f-relaxation */
   if (args->level[0].f_relaxation.type == 2)
   {
      AMGCreate(&args->level[0].f_relaxation.amg, &frelax);
      HYPRE_MGRSetFSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, frelax);
   }

   /* Config f-relaxation at level > 0 */
   for (i = 1; i < num_levels; i++)
   {
      if (args->level[i].f_relaxation.type == 2)
      {
         AMGCreate(&args->level[i].f_relaxation.amg, &frelax);
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
#else
         HYPRE_MGRSetFSolverAtLevel(i, precon, frelax);
#endif
      }
   }

   /* Config global relaxation at level >= 0 */
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   for (i = 0; i < num_levels; i++)
   {
      if (args->level[i].g_relaxation.type == 16)
      {
         ILUCreate(&args->level[i].g_relaxation.ilu, &grelax);
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, i);
      }
   }
#endif

   /* Config coarsest level solver */
   if (args->coarsest_level.type == 0)
   {
      AMGCreate(&args->coarsest_level.amg, &csolver);
      HYPRE_MGRSetCoarseSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, csolver);
   }

   /* Set output pointers */
   *precon_ptr  = precon;
   *csolver_ptr = csolver;

   /* Free memory */
   free(inactive_dofs);
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      free(c_dofs[lvl]);
   }
}
