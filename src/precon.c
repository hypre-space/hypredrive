/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "precon.h"
#include <stdio.h>
#include <strings.h>
#include "HYPRE_parcsr_mv.h"
#include "gen_macros.h"
#include "nested_krylov.h"

#define Precon_FIELDS(_prefix)                        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, AMGSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, MGRSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, ILUSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, FSAISetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, reuse, FieldTypeIntSet)

DEFINE_FIELD_OFFSET_MAP(Precon)
#define Precon_NUM_FIELDS \
   (sizeof(Precon_field_offset_map) / sizeof(Precon_field_offset_map[0]))

DEFINE_SET_FIELD_BY_NAME_FUNC(PreconSetFieldByName, Precon_args, Precon_field_offset_map,
                              Precon_NUM_FIELDS)
DEFINE_GET_VALID_KEYS_FUNC(PreconGetValidKeys, Precon_NUM_FIELDS, Precon_field_offset_map)

/*-----------------------------------------------------------------------------
 * PreconGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PreconGetValidValues(const char *key)
{
   (void)key;
   /* The "preconditioner" entry does not hold values, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * PreconGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
PreconGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {{"amg", (int)PRECON_BOOMERAMG},
                             {"mgr", (int)PRECON_MGR},
                             {"ilu", (int)PRECON_ILU},
                             {"fsai", (int)PRECON_FSAI}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * PreconSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
PreconSetDefaultArgs(precon_args *args)
{
   args->reuse = 0;
}

void
PreconReuseSetDefaultArgs(PreconReuse_args *args)
{
   if (!args)
   {
      return;
   }

   args->enabled           = 0;
   args->frequency         = 0;
   args->linear_system_ids = NULL;
   args->per_timestep      = 0;
}

void
PreconReuseDestroyArgs(PreconReuse_args *args)
{
   if (!args)
   {
      return;
   }

   if (args->linear_system_ids)
   {
      IntArrayDestroy(&args->linear_system_ids);
   }

   args->frequency    = 0;
   args->per_timestep = 0;
   args->enabled      = 0;
}

static int
PreconReuseParseOnOff(const char *value, int *out)
{
   if (!value || !out)
   {
      return 0;
   }

   if (!strcasecmp(value, "on") || !strcasecmp(value, "yes") ||
       !strcasecmp(value, "true") || !strcmp(value, "1"))
   {
      *out = 1;
      return 1;
   }
   if (!strcasecmp(value, "off") || !strcasecmp(value, "no") ||
       !strcasecmp(value, "false") || !strcmp(value, "0"))
   {
      *out = 0;
      return 1;
   }

   return 0;
}

static int
PreconReuseIntArrayContains(const IntArray *arr, int value)
{
   if (!arr || !arr->data)
   {
      return 0;
   }

   for (size_t i = 0; i < arr->size; i++)
   {
      if (arr->data[i] == value)
      {
         return 1;
      }
   }

   return 0;
}

static int
PreconReuseFindTimestepIndex(const IntArray *starts, int ls_id)
{
   if (!starts || !starts->data)
   {
      return -1;
   }

   for (size_t i = 0; i < starts->size; i++)
   {
      if (starts->data[i] == ls_id)
      {
         return (int)i;
      }
   }

   return -1;
}

void
PreconReuseTimestepsClear(IntArray **timestep_starts)
{
   if (!timestep_starts)
   {
      return;
   }

   IntArrayDestroy(timestep_starts);
}

uint32_t
PreconReuseTimestepsLoad(const PreconReuse_args *args, const char *filename,
                         IntArray **timestep_starts)
{
   if (!args || !timestep_starts)
   {
      return ErrorCodeGet();
   }

   PreconReuseTimestepsClear(timestep_starts);

   if (!args->enabled || !args->per_timestep)
   {
      return ErrorCodeGet();
   }

   if (!filename || filename[0] == '\0')
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd(
         "preconditioner.reuse.per_timestep requires linear_system.timestep_filename");
      return ErrorCodeGet();
   }

   FILE *fp = fopen(filename, "r");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open timestep file: '%s'", filename);
      return ErrorCodeGet();
   }

   int total = 0;
   if (fscanf(fp, "%d", &total) != 1 || total <= 0)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid timestep file header in '%s'", filename);
      return ErrorCodeGet();
   }

   IntArray *starts = IntArrayCreate((size_t)total);
   if (!starts)
   {
      fclose(fp);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate timestep starts array");
      return ErrorCodeGet();
   }

   for (int i = 0; i < total; i++)
   {
      int timestep = 0;
      int ls_start = 0;
      if (fscanf(fp, "%d %d", &timestep, &ls_start) != 2 || ls_start < 0)
      {
         fclose(fp);
         IntArrayDestroy(&starts);
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Invalid timestep entry in '%s' at line %d", filename, i + 2);
         return ErrorCodeGet();
      }
      starts->data[i] = ls_start;
      (void)timestep;
   }

   fclose(fp);
   *timestep_starts = starts;
   return ErrorCodeGet();
}

int
PreconReuseShouldRecompute(const PreconReuse_args *args, const IntArray *timestep_starts,
                           int next_ls_id)
{
   if (!args)
   {
      return 1;
   }

   if (next_ls_id < 0)
   {
      next_ls_id = 0;
   }

   int freq = args->frequency;
   if (!args->enabled || freq < 0)
   {
      freq = 0;
   }

   if (args->enabled && args->linear_system_ids && args->linear_system_ids->size > 0)
   {
      return PreconReuseIntArrayContains(args->linear_system_ids, next_ls_id);
   }

   if (args->enabled && args->per_timestep)
   {
      int timestep_idx = PreconReuseFindTimestepIndex(timestep_starts, next_ls_id);
      if (timestep_idx < 0)
      {
         return 0;
      }
      return (timestep_idx % (freq + 1)) == 0;
   }

   return (next_ls_id % (freq + 1)) == 0;
}

void
PreconReuseSetArgsFromYAML(PreconReuse_args *args, YAMLnode *parent)
{
   if (!args || !parent)
   {
      return;
   }

   /* Shorthand: reuse: <int> */
   if (!parent->children && parent->val && strcmp(parent->val, "") != 0)
   {
      if (sscanf(parent->val, "%d", &args->frequency) != 1)
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Invalid preconditioner reuse frequency: '%s'", parent->val);
         YAML_NODE_SET_INVALID_VAL(parent);
         return;
      }
      args->enabled = 1;
      YAML_NODE_SET_VALID(parent);
      return;
   }

   int seen_enabled           = 0;
   int seen_frequency         = 0;
   int seen_linear_system_ids = 0;
   int seen_per_timestep      = 0;
   YAML_NODE_ITERATE(parent, child)
   {
      if (!strcmp(child->key, "enabled"))
      {
         const char *value = child->mapped_val ? child->mapped_val : child->val;
         if (!PreconReuseParseOnOff(value, &args->enabled))
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Invalid value for preconditioner.reuse.enabled: '%s'",
                        value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_enabled = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "frequency"))
      {
         const char *value = child->mapped_val ? child->mapped_val : child->val;
         if (!value || sscanf(value, "%d", &args->frequency) != 1 || args->frequency < 0)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Invalid value for preconditioner.reuse.frequency: '%s'",
                        value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_frequency = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "linear_system_ids") ||
               !strcmp(child->key, "linear_solver_ids"))
      {
         const char *value = child->mapped_val ? child->mapped_val : child->val;
         if (!value)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Invalid value for preconditioner.reuse.linear_system_ids");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }

         IntArray *ids = NULL;
         StrToIntArray(value, &ids);
         if (!ids)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Failed to parse preconditioner.reuse.linear_system_ids");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }

         IntArrayDestroy(&args->linear_system_ids);
         args->linear_system_ids = ids;
         seen_linear_system_ids  = 1;
         YAML_NODE_SET_VALID(child);
      }
      else if (!strcmp(child->key, "per_timestep"))
      {
         const char *value = child->mapped_val ? child->mapped_val : child->val;
         if (!PreconReuseParseOnOff(value, &args->per_timestep))
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Invalid value for preconditioner.reuse.per_timestep: '%s'",
                        value ? value : "");
            YAML_NODE_SET_INVALID_VAL(child);
            return;
         }
         seen_per_timestep = args->per_timestep ? 1 : 0;
         YAML_NODE_SET_VALID(child);
      }
      else
      {
         ErrorCodeSet(ERROR_INVALID_KEY);
         ErrorMsgAdd("Unknown key under preconditioner.reuse: '%s'", child->key);
         YAML_NODE_SET_INVALID_KEY(child);
         return;
      }
   }

   if (!seen_enabled)
   {
      args->enabled = 1;
   }

   if (seen_linear_system_ids && (seen_frequency || seen_per_timestep))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("preconditioner.reuse.linear_system_ids cannot be combined with "
                  "frequency or per_timestep");
      YAML_NODE_SET_INVALID_VAL(parent);
      return;
   }

   if (!args->enabled)
   {
      IntArrayDestroy(&args->linear_system_ids);
      args->frequency    = 0;
      args->per_timestep = 0;
   }
}

void
PreconArgsSetDefaultsForMethod(precon_t method, precon_args *args)
{
   if (!args)
   {
      return;
   }

   PreconSetDefaultArgs(args);

   switch (method)
   {
      case PRECON_BOOMERAMG:
         AMGSetDefaultArgs(&args->amg);
         break;
      case PRECON_MGR:
         MGRSetDefaultArgs(&args->mgr);
         break;
      case PRECON_ILU:
         ILUSetDefaultArgs(&args->ilu);
         break;
      case PRECON_FSAI:
         FSAISetDefaultArgs(&args->fsai);
         break;
      case PRECON_NONE:
      default:
         break;
   }
}

/*-----------------------------------------------------------------------------
 * PreconSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
PreconSetArgsFromYAML(precon_args *args, YAMLnode *parent)
{
   if (!parent || !parent->children)
   {
      return;
   }

   YAMLSetArgsGeneric((void *)args, parent, PreconGetValidKeys, PreconGetValidValues,
                      PreconSetFieldByName);
}

/*-----------------------------------------------------------------------------
 * PreconCreate
 *-----------------------------------------------------------------------------*/

void
PreconCreate(precon_t precon_method, precon_args *args, IntArray *dofmap,
             HYPRE_IJVector vec_nn, HYPRE_Precon *precon_ptr)
{
   HYPRE_Precon precon = malloc(sizeof(hypre_Precon));

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         AMGSetRBMs(&args->amg, vec_nn);
         AMGCreate(&args->amg, &precon->main);
         break;

      case PRECON_MGR:
         MGRSetDofmap(&args->mgr, dofmap);
         MGRSetNearNullSpace(&args->mgr, vec_nn);
         MGRCreate(&args->mgr, &precon->main);
         break;

      case PRECON_ILU:
         ILUCreate(&args->ilu, &precon->main);
         break;

      case PRECON_FSAI:
         FSAICreate(&args->fsai, &precon->main);
         break;

      case PRECON_NONE:
         break;

      default:
         ErrorCodeSet(ERROR_INVALID_PRECON);
         free(precon);
         *precon_ptr = NULL;
         return;
   }

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * PreconSetup
 *-----------------------------------------------------------------------------*/

void
PreconSetup(precon_t precon_method, HYPRE_Precon precon, HYPRE_IJMatrix A)
{
   void              *vA    = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   HYPRE_Solver       prec = precon->main;

   HYPRE_IJMatrixGetObject(A, &vA);
   par_A = (HYPRE_ParCSRMatrix)vA;

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         HYPRE_BoomerAMGSetup(prec, par_A, par_b, par_x);
         break;

      case PRECON_MGR:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_MGRSetup(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#endif
         break;

      case PRECON_ILU:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_ILUSetup(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("ILU requires hypre >= 2.19.0");
#endif
         break;

      case PRECON_FSAI:
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         HYPRE_FSAISetup(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
#endif
         break;

      case PRECON_NONE:
         break;

      default:
         ErrorCodeSet(ERROR_INVALID_PRECON);
         break;
   }

   // TODO: fix timing. Adjust LinearSolverSetup.
   // StatsTimerStop("prec");
}

/*-----------------------------------------------------------------------------
 * PreconApply
 *-----------------------------------------------------------------------------*/

void
PreconApply(precon_t precon_method, HYPRE_Precon precon, HYPRE_IJMatrix A,
            HYPRE_IJVector b, HYPRE_IJVector x)
{
   void              *vA = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   HYPRE_Solver       prec = precon->main;

   HYPRE_IJMatrixGetObject(A, &vA);
   par_A = (HYPRE_ParCSRMatrix)vA;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         HYPRE_BoomerAMGSolve(prec, par_A, par_b, par_x);
         break;

      case PRECON_MGR:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_MGRSolve(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#endif
         break;

      case PRECON_ILU:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_ILUSolve(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("ILU requires hypre >= 2.19.0");
#endif
         break;

      case PRECON_FSAI:
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         HYPRE_FSAISolve(prec, par_A, par_b, par_x);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
#endif
         break;

      case PRECON_NONE:
         break;

      default:
         ErrorCodeSet(ERROR_INVALID_PRECON);
         break;
   }

   // StatsTimerStop("prec_apply");
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------
 */

static void
PreconDestroyMGRSolver(MGR_args *mgr, HYPRE_Solver *solver_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)mgr;
   (void)solver_ptr;
   ErrorCodeSet(ERROR_INVALID_PRECON);
   ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#else
   if (!mgr || !solver_ptr || !*solver_ptr)
   {
      return;
   }

   HYPRE_MGRDestroy(*solver_ptr);
   *solver_ptr = NULL;

   /* TODO: should MGR free these internally? */
   if (mgr->coarsest_level.use_krylov && mgr->coarsest_level.krylov)
   {
      NestedKrylovDestroy(mgr->coarsest_level.krylov);
   }
   else if (mgr->csolver)
   {
      /* MGR does not destroy user-provided coarse solvers. */
      if (mgr->csolver_type == 0)
      {
         HYPRE_BoomerAMGDestroy(mgr->csolver);
      }
#if defined(HYPRE_USING_DSUPERLU)
      else if (mgr->csolver_type == 29)
      {
         HYPRE_MGRDirectSolverDestroy(mgr->csolver);
      }
#endif
      else if (mgr->csolver_type == 32)
      {
         HYPRE_ILUDestroy(mgr->csolver);
      }
   }
   mgr->csolver      = NULL;
   mgr->csolver_type = -1;

   int max_levels = (mgr->num_levels > 0) ? (mgr->num_levels - 1) : 0;
   for (int i = 0; i < max_levels; i++)
   {
      if (mgr->level[i].f_relaxation.use_krylov && mgr->level[i].f_relaxation.krylov)
      {
         mgr->level[i].f_relaxation.krylov->base_solver = NULL;
         NestedKrylovDestroy(mgr->level[i].f_relaxation.krylov);
      }
      else if (i == 0 && mgr->frelax[i])
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 11)
         /* hypre-master (>= 3.1.0 develop 11 observed) destroys user-provided
          * level-0 F-relax solvers inside HYPRE_MGRDestroy(). Avoid double free. */
         if (mgr->level[i].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
         {
            HYPRE_Solver nested_mgr_solver =
               MGRNestedFRelaxWrapperGetInner(mgr->frelax[i]);
            MGRNestedFRelaxWrapperFree(&mgr->frelax[i]);

            if (mgr->level[i].f_relaxation.mgr)
            {
               PreconDestroyMGRSolver(mgr->level[i].f_relaxation.mgr, &nested_mgr_solver);
            }
            else if (nested_mgr_solver)
            {
               HYPRE_MGRDestroy(nested_mgr_solver);
               nested_mgr_solver = NULL;
            }
         }
         else
         {
            mgr->frelax[i] = NULL;
         }
#else
         /* MGR does not destroy user-provided F-relaxation solvers at level 0. */
         if (mgr->level[i].f_relaxation.type == 2)
         {
            HYPRE_BoomerAMGDestroy(mgr->frelax[i]);
         }
         else if (mgr->level[i].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
         {
            HYPRE_Solver nested_mgr_solver =
               MGRNestedFRelaxWrapperGetInner(mgr->frelax[i]);
            MGRNestedFRelaxWrapperFree(&mgr->frelax[i]);

            if (mgr->level[i].f_relaxation.mgr)
            {
               PreconDestroyMGRSolver(mgr->level[i].f_relaxation.mgr, &nested_mgr_solver);
            }
            else if (nested_mgr_solver)
            {
               HYPRE_MGRDestroy(nested_mgr_solver);
               nested_mgr_solver = NULL;
            }
         }
#if defined(HYPRE_USING_DSUPERLU)
         else if (mgr->level[i].f_relaxation.type == 29)
         {
            HYPRE_MGRDirectSolverDestroy(mgr->frelax[i]);
         }
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
         else if (mgr->level[i].f_relaxation.type == 32)
         {
            HYPRE_ILUDestroy(mgr->frelax[i]);
         }
#endif
         mgr->frelax[i] = NULL;
#endif
      }

      if (mgr->level[i].g_relaxation.use_krylov && mgr->level[i].g_relaxation.krylov)
      {
         mgr->level[i].g_relaxation.krylov->base_solver = NULL;
         NestedKrylovDestroy(mgr->level[i].g_relaxation.krylov);
      }
   }
#endif
}

/*-----------------------------------------------------------------------------
 * PreconDestroy
 *-----------------------------------------------------------------------------*/

void
PreconDestroy(precon_t precon_method, precon_args *args, HYPRE_Precon *precon_ptr)
{
   HYPRE_Precon precon = *precon_ptr;

   if (!precon)
   {
      return;
   }

   if (precon->main)
   {
      switch (precon_method)
      {
         case PRECON_BOOMERAMG:
            for (HYPRE_Int i = 0; i < args->amg.num_rbms; i++)
            {
               HYPRE_ParVectorDestroy(args->amg.rbms[i]);
               args->amg.rbms[i] = NULL;
            }
            HYPRE_BoomerAMGDestroy(precon->main);
            break;

         case PRECON_MGR:
            PreconDestroyMGRSolver(&args->mgr, &precon->main);
            break;

         case PRECON_ILU:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
            HYPRE_ILUDestroy(precon->main);
#endif
            break;

         case PRECON_FSAI:
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
            HYPRE_FSAIDestroy(precon->main);
#endif
            break;

         case PRECON_NONE:
            break;
      }

      precon->main = NULL;
   }

   free(*precon_ptr);
   *precon_ptr = NULL;
}
