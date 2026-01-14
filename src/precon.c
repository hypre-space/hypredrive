/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "precon.h"
#include "HYPRE_parcsr_mv.h"
#include "gen_macros.h"

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
         HYPRE_MGRSetup(prec, par_A, par_b, par_x);
         break;

      case PRECON_ILU:
         HYPRE_ILUSetup(prec, par_A, par_b, par_x);
         break;

      case PRECON_FSAI:
         HYPRE_FSAISetup(prec, par_A, par_b, par_x);
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
         HYPRE_MGRSolve(prec, par_A, par_b, par_x);
         break;

      case PRECON_ILU:
         HYPRE_ILUSolve(prec, par_A, par_b, par_x);
         break;

      case PRECON_FSAI:
         HYPRE_FSAISolve(prec, par_A, par_b, par_x);
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
            HYPRE_MGRDestroy(precon->main);

            /* TODO: should MGR free these internally? */
            if (args->mgr.coarsest_level.type == 0)
            {
               HYPRE_BoomerAMGDestroy(args->mgr.csolver);
            }
#if defined(HYPRE_USING_DSUPERLU)
            else if (args->mgr.coarsest_level.type == 29)
            {
               HYPRE_MGRDirectSolverDestroy(args->mgr.csolver);
            }
#endif
            else if (args->mgr.coarsest_level.type == 32)
            {
               HYPRE_ILUDestroy(args->mgr.csolver);
            }
            args->mgr.csolver = NULL;

            if (args->mgr.level[0].f_relaxation.type == 2)
            {
               HYPRE_BoomerAMGDestroy(args->mgr.frelax[0]);
               args->mgr.frelax[0] = NULL;
            }
            else if (args->mgr.level[0].f_relaxation.type == 32)
            {
               HYPRE_ILUDestroy(args->mgr.frelax[0]);
               args->mgr.frelax[0] = NULL;
            }
            break;

         case PRECON_ILU:
            HYPRE_ILUDestroy(precon->main);
            break;

         case PRECON_FSAI:
            HYPRE_FSAIDestroy(precon->main);
            break;

         case PRECON_NONE:
            break;
      }

      precon->main = NULL;
   }

   free(*precon_ptr);
   *precon_ptr = NULL;
}
