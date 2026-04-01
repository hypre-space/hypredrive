/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/precon.h"
#include "HYPRE_parcsr_mv.h"
#include "internal/gen_macros.h"
#include "internal/krylov.h"
#include "logging.h"

#define Precon_FIELDS(_prefix)                                 \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, hypredrv_AMGSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, hypredrv_MGRSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, hypredrv_ILUSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, hypredrv_FSAISetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, reuse, hypredrv_FieldTypeIntSet)

/* GCOVR_EXCL_START */
DEFINE_FIELD_OFFSET_MAP(Precon)
#define Precon_NUM_FIELDS \
   (sizeof(Precon_field_offset_map) / sizeof(Precon_field_offset_map[0]))

DEFINE_SET_FIELD_BY_NAME_FUNC(hypredrv_PreconSetFieldByName, Precon_args,
                              Precon_field_offset_map, Precon_NUM_FIELDS)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_PreconGetValidKeys, Precon_NUM_FIELDS,
                           Precon_field_offset_map)
/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_PreconGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_PreconGetValidValues(const char *key)
{
   (void)key;
   /* The "preconditioner" entry does not hold values, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_PreconGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {
      {"amg", (int)PRECON_BOOMERAMG},
      {"mgr", (int)PRECON_MGR},
      {"ilu", (int)PRECON_ILU},
      {"fsai", (int)PRECON_FSAI},
   };

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconSetDefaultArgs(precon_args *args)
{
   args->reuse = 0;
}

void
hypredrv_PreconArgsSetDefaultsForMethod(precon_t method, precon_args *args)
{
   if (!args)
   {
      return;
   }

   hypredrv_PreconSetDefaultArgs(args);

   switch (method)
   {
      case PRECON_BOOMERAMG:
         hypredrv_AMGSetDefaultArgs(&args->amg);
         break;
      case PRECON_MGR:
         hypredrv_MGRSetDefaultArgs(&args->mgr);
         break;
      case PRECON_ILU:
         hypredrv_ILUSetDefaultArgs(&args->ilu);
         break;
      case PRECON_FSAI:
         hypredrv_FSAISetDefaultArgs(&args->fsai);
         break;
      case PRECON_NONE:
      default:
         break;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconSetArgsFromYAML(precon_args *args, YAMLnode *parent)
{
   if (!parent || !parent->children)
   {
      return;
   }

   hypredrv_YAMLSetArgsGeneric((void *)args, parent, hypredrv_PreconGetValidKeys,
                               hypredrv_PreconGetValidValues,
                               hypredrv_PreconSetFieldByName);
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconCreate(precon_t precon_method, precon_args *args, IntArray *dofmap,
                      HYPRE_IJVector vec_nn, HYPRE_Precon *precon_ptr)
{
   HYPRE_Precon precon = malloc(sizeof(hypre_Precon));
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!precon)              /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      *precon_ptr = NULL;
      return;
   }

   precon->main   = NULL;
   precon->method = precon_method;
   precon->stats  = NULL;

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         hypredrv_AMGSetRBMs(&args->amg, vec_nn);
         hypredrv_AMGCreate(&args->amg, &precon->main);
         break;

      case PRECON_MGR:
         hypredrv_MGRSetDofmap(&args->mgr, dofmap);
         hypredrv_MGRSetNearNullSpace(&args->mgr, vec_nn);
         hypredrv_MGRCreate(&args->mgr, &precon->main);
         break;

      case PRECON_ILU:
         hypredrv_ILUCreate(&args->ilu, &precon->main);
         break;

      case PRECON_FSAI:
         hypredrv_FSAICreate(&args->fsai, &precon->main);
         break;

      case PRECON_NONE:
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         free(precon);
         *precon_ptr = NULL;
         return;
   }

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconSetup
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconSetup(precon_t precon_method, HYPRE_Precon precon, HYPRE_IJMatrix A)
{
   void              *vA    = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;

   /* GCOVR_EXCL_BR_START */         /* low-signal branch under CI */
   if (precon_method == PRECON_NONE) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   if (!precon)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Preconditioner setup requested with null preconditioner");
      return;
   }

   if (!A)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner setup requested with null matrix");
      return;
   }

   HYPRE_Solver prec = precon->main;

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
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      case PRECON_ILU:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_ILUSetup(prec, par_A, par_b, par_x);
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("ILU requires hypre >= 2.19.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      case PRECON_FSAI:
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         HYPRE_FSAISetup(prec, par_A, par_b, par_x);
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         break;
   }

   // TODO: fix timing. Adjust LinearSolverSetup.
   // StatsTimerStop("prec");
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconApply
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconApply(precon_t precon_method, HYPRE_Precon precon, HYPRE_IJMatrix A,
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
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      case PRECON_ILU:
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
         HYPRE_ILUSolve(prec, par_A, par_b, par_x);
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("ILU requires hypre >= 2.19.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      case PRECON_FSAI:
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         HYPRE_FSAISolve(prec, par_A, par_b, par_x);
#else  /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
#endif /* GCOVR_EXCL_STOP */
         break;

      case PRECON_NONE:
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         break;
   }

   // StatsTimerStop("prec_apply");
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------
 */

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void PreconDestroyMGRSolver(MGR_args *, HYPRE_Solver *);

static void
DestroyNestedMGRFRelaxInnerSolver(MGR_args *mgr, int i,
                                  HYPRE_Solver *nested_mgr_solver_ptr)
{
   /* GCOVR_EXCL_BR_START */                              /* low-signal branch under CI */
   if (!nested_mgr_solver_ptr || !*nested_mgr_solver_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   /* GCOVR_EXCL_BR_START */           /* low-signal branch under CI */
   if (mgr->level[i].f_relaxation.mgr) /* GCOVR_EXCL_BR_STOP */
   {
      PreconDestroyMGRSolver(mgr->level[i].f_relaxation.mgr, nested_mgr_solver_ptr);
   }
   else
   {
      HYPRE_MGRDestroy(*nested_mgr_solver_ptr);
      *nested_mgr_solver_ptr = NULL;
   }
}

/* GCOVR_EXCL_START */
static void
DestroyNestedMGRFRelaxAtLevel(MGR_args *mgr, int i)
{
   HYPRE_Solver nested_mgr_solver =
      hypredrv_MGRNestedFRelaxWrapperGetInner(mgr->frelax[i]);
   hypredrv_MGRNestedFRelaxWrapperFree(&mgr->frelax[i]);
   DestroyNestedMGRFRelaxInnerSolver(mgr, i, &nested_mgr_solver);
}
/* GCOVR_EXCL_STOP */
#endif /* HYPRE_CHECK_MIN_VERSION(21900, 0) */

static void
PreconDestroyMGRSolver(MGR_args *mgr, HYPRE_Solver *solver_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)mgr;
   (void)solver_ptr;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("MGR requires hypre >= 2.19.0");
#else
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!mgr)                 /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   /* GCOVR_EXCL_BR_START */        /* low-signal branch under CI */
   if (!solver_ptr || !*solver_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */   /* low-signal branch under CI */
      if (mgr->point_marker_data) /* GCOVR_EXCL_BR_STOP */
      {
         free(mgr->point_marker_data);
         mgr->point_marker_data = NULL;
      }
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(30100, 11)
   HYPRE_Solver detached_nested_lvl0_frelax  = NULL;
   HYPRE_Solver detached_nested_lvl0_wrapper = NULL;
   if (mgr->num_levels > 1 && mgr->frelax[0] &&
       mgr->level[0].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
   {
      detached_nested_lvl0_wrapper = mgr->frelax[0];
      detached_nested_lvl0_frelax =
         hypredrv_MGRNestedFRelaxWrapperDetachInner(mgr->frelax[0]);
      mgr->frelax[0] = NULL;
   }
#endif

   HYPRE_MGRDestroy(*solver_ptr);
   *solver_ptr = NULL;

#if HYPRE_CHECK_MIN_VERSION(30100, 11)
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (detached_nested_lvl0_wrapper &&
       /* GCOVR_EXCL_BR_STOP */
       hypredrv_MGRNestedFRelaxWrapperIsLive(detached_nested_lvl0_wrapper))
   {
      hypredrv_MGRNestedFRelaxWrapperFree(&detached_nested_lvl0_wrapper);
   }
   if (detached_nested_lvl0_frelax)
   {
      DestroyNestedMGRFRelaxInnerSolver(mgr, 0, &detached_nested_lvl0_frelax);
   }
#endif

   if (mgr->point_marker_data)
   {
      free(mgr->point_marker_data);
      mgr->point_marker_data = NULL;
   }

   /* TODO: should MGR free these internally? */
   if (mgr->coarsest_level.use_krylov && mgr->coarsest_level.krylov)
   {
      hypredrv_NestedKrylovDestroy(mgr->coarsest_level.krylov);
   }
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   else if (mgr->csolver)    /* GCOVR_EXCL_BR_STOP */
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
      /* GCOVR_EXCL_BR_START */         /* low-signal branch under CI */
      else if (mgr->csolver_type == 32) /* GCOVR_EXCL_BR_STOP */
      {
         HYPRE_ILUDestroy(mgr->csolver);
      }
   }
   mgr->csolver      = NULL;
   mgr->csolver_type = -1;

   int max_levels = (mgr->num_levels > 0) ? (mgr->num_levels - 1) : 0;
   for (int i = 0; i < max_levels; i++)
   {
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (mgr->level[i].f_relaxation.use_krylov && mgr->level[i].f_relaxation.krylov)
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_NestedKrylovDestroy(mgr->level[i].f_relaxation.krylov);
      }
      else if (i == 0 && mgr->frelax[i])
      {
         /* MGR does not destroy user-provided F-relaxation solvers at level 0. */
         if (mgr->level[i].f_relaxation.type == 2)
         {
            HYPRE_BoomerAMGDestroy(mgr->frelax[i]);
         }
         /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
         else if (mgr->level[i].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
         /* GCOVR_EXCL_BR_STOP */
         {
            DestroyNestedMGRFRelaxAtLevel(mgr, i);
         }
#if defined(HYPRE_USING_DSUPERLU)
         /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
         else if (mgr->level[i].f_relaxation.type == 29) /* GCOVR_EXCL_BR_STOP */
         {
            HYPRE_MGRDirectSolverDestroy(mgr->frelax[i]);
         }
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
         /* GCOVR_EXCL_BR_START */                       /* low-signal branch under CI */
         else if (mgr->level[i].f_relaxation.type == 32) /* GCOVR_EXCL_BR_STOP */
         {
            HYPRE_ILUDestroy(mgr->frelax[i]);
         }
#endif
         mgr->frelax[i] = NULL;
      }

      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (mgr->level[i].g_relaxation.use_krylov && mgr->level[i].g_relaxation.krylov)
      /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_NestedKrylovDestroy(mgr->level[i].g_relaxation.krylov);
      }
   }
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconDestroy(precon_t precon_method, precon_args *args,
                       HYPRE_Precon *precon_ptr)
{
   int log_rank = -1;
   /* GCOVR_EXCL_BR_START */   /* low-signal branch under CI */
   if (hypredrv_LogEnabled(3)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(MPI_COMM_WORLD);
   }

   HYPRE_Precon precon = *precon_ptr;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!precon)              /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_START */
      HYPREDRV_LOGF(3, log_rank, NULL, 0,
                    "preconditioner destroy skipped: object already NULL");
      /* GCOVR_EXCL_STOP */
      return;
   }

   if (precon->main)
   {
      switch (precon_method)
      {
         case PRECON_BOOMERAMG:
            /* GCOVR_EXCL_START */
            HYPREDRV_LOGF(3, log_rank, NULL, 0,
                          "preconditioner destroy dispatch: method=boomeramg");
            /* GCOVR_EXCL_STOP */
            for (HYPRE_Int i = 0; i < args->amg.num_rbms; i++)
            {
               HYPRE_ParVectorDestroy(args->amg.rbms[i]);
               args->amg.rbms[i] = NULL;
            }
            HYPRE_BoomerAMGDestroy(precon->main);
            break;

         case PRECON_MGR:
            /* GCOVR_EXCL_START */
            HYPREDRV_LOGF(3, log_rank, NULL, 0,
                          "preconditioner destroy dispatch: method=mgr");
            /* GCOVR_EXCL_STOP */
            PreconDestroyMGRSolver(&args->mgr, &precon->main);
            break;

         case PRECON_ILU:
            /* GCOVR_EXCL_START */
            HYPREDRV_LOGF(3, log_rank, NULL, 0,
                          "preconditioner destroy dispatch: method=ilu");
            /* GCOVR_EXCL_STOP */
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
            HYPRE_ILUDestroy(precon->main);
#endif
            break;

         case PRECON_FSAI:
            /* GCOVR_EXCL_START */
            HYPREDRV_LOGF(3, log_rank, NULL, 0,
                          "preconditioner destroy dispatch: method=fsai");
            /* GCOVR_EXCL_STOP */
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
            HYPRE_FSAIDestroy(precon->main);
#endif
            break;

         case PRECON_NONE:
            /* GCOVR_EXCL_START */
            HYPREDRV_LOGF(3, log_rank, NULL, 0,
                          "preconditioner destroy dispatch: method=none");
            /* GCOVR_EXCL_STOP */
            break;
      }

      precon->main = NULL;
   }

   free(*precon_ptr);
   *precon_ptr = NULL;
}
