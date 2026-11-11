/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/precon.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
#include "internal/gen_macros.h"
#include "internal/mgr_internal.h"
#include "internal/solver.h"
#include "logging.h"
#include "object.h"

#include <stdio.h>

/* Agree on the registration operation before entering its validation phases.
 * In particular, an invalid level on one rank must not strand its peers. */
static uint32_t
PreconOperatorBegin(hypredrv_t *hypredrv, int kind, int level)
{
   int request[] = {kind, level}, minimum[2], maximum[2];
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, hypredrv->comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, hypredrv->comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1] || kind < 0 || kind > 3 ||
       (kind == 3 && (level < 0 || level >= MAX_MGR_LEVELS - 1)))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Operator registration and MGR level must agree across ranks; "
                           "MGR levels must be in [0, %d)",
                           MAX_MGR_LEVELS - 1);
      return ERROR_INVALID_VAL;
   }
   return ERROR_NONE;
}

static uint32_t
PreconOperatorRecordError(HYPRE_Int code, const char *operation)
{
   if (!code) return ERROR_NONE;
   char message[HYPRE_MAX_MSG_LEN];
   HYPRE_DescribeError(code, message);
   hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
   hypredrv_ErrorMsgAdd("%s: HYPRE call failed: %s", operation, message);
   return ERROR_HYPRE_INTERNAL;
}

static uint32_t
PreconOperatorStatusSync(MPI_Comm comm, uint32_t status)
{
   status |= hypredrv_HypreConsumeErrors(comm);
   uint32_t collective = ERROR_NONE;
   MPI_Allreduce(&status, &collective, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (collective != status)
   {
      hypredrv_ErrorCodeSet((hypredrv_error_t)collective);
      hypredrv_ErrorMsgAdd("Operator registration failed on another rank");
   }
   return collective;
}

/* A change on any rank invalidates the consumers on every rank. Krylov owns
 * callbacks into the preconditioner and must be released first. */
static void
PreconOperatorInvalidate(hypredrv_t *hypredrv, int changed, unsigned consumers)
{
   int any_changed = 0;
   MPI_Allreduce(&changed, &any_changed, 1, MPI_INT, MPI_MAX, hypredrv->comm);
   if (any_changed && hypredrv->precon && (consumers & (1u << hypredrv->precon->method)))
      hypredrv_ObjectDestroySolvers(hypredrv);
}

uint32_t
hypredrv_PreconSetDiscreteOperator(hypredrv_t *hypredrv, hypredrv_DiscreteOperator kind,
                                   HYPRE_IJMatrix matrix)
{
   uint32_t status = PreconOperatorBegin(hypredrv, kind, -1);
   if (!status && matrix) status = hypredrv_IJMatrixValidate(matrix, "Discrete operator");
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   hypredrv_ObjectPrepareExecution(hypredrv, matrix, 1);
   status = PreconOperatorRecordError(HYPRE_GetError(), "Discrete operator migration");
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   hypredrv_MatrixSlot *slot =
      kind == PRECON_OPERATOR_GRADIENT ? &hypredrv->mat_G : &hypredrv->mat_C;
   unsigned consumers =
      (1u << PRECON_ADS) | (kind == PRECON_OPERATOR_GRADIENT ? (1u << PRECON_AMS) : 0u);
   PreconOperatorInvalidate(hypredrv, slot->handle != matrix, consumers);
   hypredrv_ObjectReplaceMatrix(hypredrv, slot, matrix,
                                hypredrv_AttachmentIsOwned(hypredrv->lib_mode));
   return ERROR_NONE;
}

uint32_t
hypredrv_PreconSetCoordinates(hypredrv_t *hypredrv, HYPRE_IJVector x, HYPRE_IJVector y,
                              HYPRE_IJVector z)
{
   uint32_t         status    = PreconOperatorBegin(hypredrv, 2, -1);
   HYPRE_IJVector   vectors[] = {x, y, z};
   HYPRE_BigInt     first[3] = {0}, last[3] = {0};
   hypre_ParVector *par[3] = {NULL};
   int              base = -1, changed = 0;
   for (int i = 0; i < 3 && !status; i++)
   {
      changed |= vectors[i] != hypredrv->vec_coord[i].handle;
      if (!vectors[i]) continue;
      HYPRE_Int type   = -1;
      void     *object = NULL;
      status = PreconOperatorRecordError(HYPRE_IJVectorGetObjectType(vectors[i], &type),
                                         "Coordinates");
      if (!status)
         status = PreconOperatorRecordError(HYPRE_IJVectorGetObject(vectors[i], &object),
                                            "Coordinates");
      if (!status)
         status = PreconOperatorRecordError(
            HYPRE_IJVectorGetLocalRange(vectors[i], &first[i], &last[i]), "Coordinates");
      if (status) break;
      par[i] = (hypre_ParVector *)object;
      hypre_Vector *local =
         type == HYPRE_PARCSR && par[i] ? hypre_ParVectorLocalVector(par[i]) : NULL;
      if (!local || hypre_VectorSize(local) < 0 || hypre_VectorNumVectors(local) != 1 ||
          (hypre_VectorSize(local) > 0 && !hypre_VectorData(local)))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Coordinates must be initialized, single-component ParCSR IJVectors");
         status = ERROR_INVALID_VAL;
         break;
      }
      if (base < 0) base = i;
      if (first[i] != first[base] || last[i] != last[base] ||
          hypre_ParVectorGlobalSize(par[i]) != hypre_ParVectorGlobalSize(par[base]) ||
          hypre_VectorSize(local) !=
             hypre_VectorSize(hypre_ParVectorLocalVector(par[base])))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Coordinate vectors must have matching row distributions");
         status = ERROR_INVALID_VAL;
      }
   }
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   for (int i = 0; i < 3; i++) hypredrv_ObjectPrepareExecution(hypredrv, vectors[i], 0);
   status = PreconOperatorRecordError(HYPRE_GetError(), "Coordinate migration");
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   PreconOperatorInvalidate(hypredrv, changed, (1u << PRECON_AMS) | (1u << PRECON_ADS));
   const hypredrv_VectorUpdate updates[] = {
      {&hypredrv->vec_coord[0], x, hypredrv_AttachmentIsOwned(hypredrv->lib_mode)},
      {&hypredrv->vec_coord[1], y, hypredrv_AttachmentIsOwned(hypredrv->lib_mode)},
      {&hypredrv->vec_coord[2], z, hypredrv_AttachmentIsOwned(hypredrv->lib_mode)}};
   hypredrv_ObjectInstallVectors(hypredrv, 3, updates);
   return ERROR_NONE;
}

uint32_t
hypredrv_PreconSetCoarseSchur(hypredrv_t *hypredrv, int level, HYPRE_IJMatrix matrix)
{
   uint32_t status = PreconOperatorBegin(hypredrv, 3, level);
   if (!status && matrix)
   {
      status = hypredrv_IJMatrixValidate(matrix, "Coarse Schur operator");
      for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
      {
         if (i != level && matrix == hypredrv->mat_coarse_schur[i].handle)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "A coarse Schur matrix cannot be registered at multiple MGR levels");
            status |= ERROR_INVALID_VAL;
         }
      }
   }
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   hypredrv_ObjectPrepareExecution(hypredrv, matrix, 1);
   status =
      PreconOperatorRecordError(HYPRE_GetError(), "Coarse Schur operator migration");
   if ((status = PreconOperatorStatusSync(hypredrv->comm, status)) != ERROR_NONE)
      return status;
   PreconOperatorInvalidate(hypredrv, hypredrv->mat_coarse_schur[level].handle != matrix,
                            1u << PRECON_MGR);
   hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_coarse_schur[level], matrix,
                                hypredrv_AttachmentIsOwned(hypredrv->lib_mode));
   return ERROR_NONE;
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_PreconGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {
      {"amg", (int)PRECON_BOOMERAMG},
      {"jacobi", (int)PRECON_BOOMERAMG},
      {"gauss-seidel", (int)PRECON_BOOMERAMG},
      {"mgr", (int)PRECON_MGR},
      {"ilu", (int)PRECON_ILU},
      {"fsai", (int)PRECON_FSAI},
      {"ams", (int)PRECON_AMS},
      {"ads", (int)PRECON_ADS},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      {"schwarz", (int)PRECON_SCHWARZ},
#endif
   };

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/* Per-method hypre entry points. Index by precon_t; keep in sync with the enum. */
typedef HYPRE_Int (*PreconParFn)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector,
                                 HYPRE_ParVector);
typedef HYPRE_Int (*PreconDestroyFn)(HYPRE_Solver);

typedef struct PreconOps_struct
{
   const char     *name;
   SetterFnc       parse_args;
   size_t          args_offset;
   PreconParFn     setup;
   PreconParFn     solve;
   PreconDestroyFn destroy;
} PreconOps;

static const PreconOps precon_ops[] = {
   [PRECON_BOOMERAMG] = {.name        = "boomeramg",
                         .parse_args  = hypredrv_AMGSetArgsFromYAML,
                         .args_offset = offsetof(precon_args, amg),
                         .setup       = HYPRE_BoomerAMGSetup,
                         .solve       = HYPRE_BoomerAMGSolve,
                         .destroy     = HYPRE_BoomerAMGDestroy},
   [PRECON_MGR] =
      {
         .name        = "mgr",
         .parse_args  = hypredrv_MGRSetArgsFromYAML,
         .args_offset = offsetof(precon_args, mgr),
         .setup       = hypredrv_MGRSetup,
         .solve       = HYPRE_MGRSolve,
         .destroy     = NULL, /* MGR teardown includes its owned runtime state. */
      },
   [PRECON_ILU] =
      {
         .name        = "ilu",
         .parse_args  = hypredrv_ILUSetArgsFromYAML,
         .args_offset = offsetof(precon_args, ilu),
         .setup       = HYPRE_ILUSetup,
         .solve       = HYPRE_ILUSolve,
         .destroy     = HYPRE_ILUDestroy,
      },
   [PRECON_FSAI] =
      {
         .name        = "fsai",
         .parse_args  = hypredrv_FSAISetArgsFromYAML,
         .args_offset = offsetof(precon_args, fsai),
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         .setup   = HYPRE_FSAISetup,
         .solve   = HYPRE_FSAISolve,
         .destroy = HYPRE_FSAIDestroy,
#else
         .setup   = NULL,
         .solve   = NULL,
         .destroy = NULL,
#endif
      },
   [PRECON_AMS] = {.name        = "ams",
                   .parse_args  = hypredrv_AMSSetArgsFromYAML,
                   .args_offset = offsetof(precon_args, ams),
                   .setup       = HYPRE_AMSSetup,
                   .solve       = HYPRE_AMSSolve,
                   .destroy     = HYPRE_AMSDestroy},
   [PRECON_ADS] = {.name        = "ads",
                   .parse_args  = hypredrv_ADSSetArgsFromYAML,
                   .args_offset = offsetof(precon_args, ads),
                   .setup       = HYPRE_ADSSetup,
                   .solve       = HYPRE_ADSSolve,
                   .destroy     = HYPRE_ADSDestroy},
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   [PRECON_SCHWARZ] = {.name        = "schwarz",
                       .parse_args  = hypredrv_SchwarzSetArgsFromYAML,
                       .args_offset = offsetof(precon_args, schwarz),
                       .setup       = HYPRE_SchwarzSetup,
                       .solve       = HYPRE_SchwarzSolve,
                       .destroy     = HYPRE_SchwarzDestroy},
#endif
   [PRECON_NONE] = {.name = "none", .setup = NULL, .solve = NULL, .destroy = NULL},
};

/* Parser names follow the method table, with BoomerAMG's shorter YAML key.
 * Scalar aliases remain in the type map; reuse belongs to the section schema. */
StrArray
hypredrv_PreconGetValidKeys(void)
{
   static const char *keys[PRECON_NONE + 1];
   for (int i = 0; i < PRECON_NONE; i++)
      keys[i] = i == PRECON_BOOMERAMG ? "amg" : precon_ops[i].name;
   keys[PRECON_NONE] = "reuse";
   return STR_ARRAY_CREATE(keys);
}

static const PreconOps *
PreconOpsLookup(precon_t method)
{
   if ((unsigned)method >= (unsigned)(sizeof(precon_ops) / sizeof(precon_ops[0])))
   {
      return NULL;
   }

   return &precon_ops[method];
}

/* Setup has already passed collective admission. Keep native setup, timing and
 * its ownership transition together for standalone and both Krylov paths. */
HYPRE_Int
hypredrv_PreconKrylovSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x)
{
   HYPRE_Precon precon = (HYPRE_Precon)solver;
   if (!precon || precon->method == PRECON_NONE)
   {
      return 0;
   }
#if !HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (precon->method == PRECON_MGR && precon->setup_called)
   {
      /* Older MGR setup frees hierarchy arrays without resetting their
       * pointers. Rebuild the native hierarchy behind the stable adapter. */
      hypredrv_MGRDestroy(&precon->args->mgr, &precon->main, 1);
      precon->setup_called = 0;
      hypredrv_MGRCreate(precon->comm, &precon->args->mgr, &precon->main, precon->stats,
                         0);
      if (!precon->main || hypredrv_ErrorCodeActive()) return HYPRE_ERROR_GENERIC;
   }
#endif
   if (precon->stats)
   {
      hypredrv_StatsAnnotate(precon->stats, HYPREDRV_ANNOTATE_BEGIN, "prec");
   }
   HYPRE_Int ierr = precon->ops->setup(precon->main, A, b, x);
   if (precon->stats)
   {
      hypredrv_StatsAnnotate(precon->stats, HYPREDRV_ANNOTATE_END, "prec");
   }
   precon->setup_called = 1;
   return ierr;
}

HYPRE_Int
hypredrv_PreconKrylovSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                           HYPRE_ParVector x)
{
   HYPRE_Precon precon = (HYPRE_Precon)solver;
   if (!precon || precon->method == PRECON_NONE)
   {
      return 0;
   }
   if (precon->method == PRECON_MGR)
      return hypredrv_MGRSolve(&precon->args->mgr, precon->main, A, b, x);
   return precon->ops->solve(precon->main, A, b, x);
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
      case PRECON_AMS:
         hypredrv_AMSSetDefaultArgs(&args->ams);
         break;
      case PRECON_ADS:
         hypredrv_ADSSetDefaultArgs(&args->ads);
         break;
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      case PRECON_SCHWARZ:
         hypredrv_SchwarzSetDefaultArgs(&args->schwarz);
         break;
#endif
      case PRECON_NONE:
      default:
         break;
   }
}

void
hypredrv_PreconSetMethodArgs(precon_t method, precon_args *args, YAMLnode *body)
{
   const PreconOps *ops = PreconOpsLookup(method);
   if (!ops || !ops->parse_args || !args || !body)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid preconditioner method arguments");
      return;
   }

   /* Initialize ownership fields before setters such as MGR release/reset
    * configuration. Callers provide fresh storage, not a live configuration. */
   hypredrv_PreconArgsSetDefaultsForMethod(method, args);
   /* Scalar aliases (jacobi/gauss-seidel) belong to the value-only section
    * syntax; typed bodies use the method keys from the preconditioner schema. */
   if (strcmp(body->key, "-") != 0 &&
       !hypredrv_StrArrayEntryExists(hypredrv_PreconGetValidKeys(), body->key))
   {
      body->avail_keys = hypredrv_PreconGetValidKeys();
      YAML_NODE_SET_INVALID_KEY(body);
      return;
   }
   YAML_NODE_SET_VALID(body);
   if (method == PRECON_MGR && body->val[0])
   {
      YAML_NODE_SET_VALID_IF_NO_VAL(body);
      return;
   }
   if (body->children || body->val[0])
   {
      ops->parse_args((char *)args + ops->args_offset, body);
   }
}

void
hypredrv_PreconArgsSetDefaultsForName(precon_t method, const char *name,
                                      precon_args *args)
{
   hypredrv_PreconArgsSetDefaultsForMethod(method, args);

   if (!args || !name || method != PRECON_BOOMERAMG)
   {
      return;
   }

   HYPRE_Int relax_type = -1;
   if (!strcmp(name, "jacobi"))
   {
      relax_type = 0;
   }
   else if (!strcmp(name, "gauss-seidel"))
   {
      relax_type = 3;
   }

   if (relax_type < 0)
   {
      return;
   }

   args->amg.coarsening.max_levels    = 1;
   args->amg.relaxation.type          = relax_type;
   args->amg.relaxation.down_type     = relax_type;
   args->amg.relaxation.coarse_type   = relax_type;
   args->amg.relaxation.down_sweeps   = 1;
   args->amg.relaxation.up_sweeps     = 0;
   args->amg.relaxation.coarse_sweeps = 1;
}

void
hypredrv_PreconArgsDestroyOwnedConfig(precon_t method, precon_args *args)
{
   if (!args)
   {
      return;
   }

   /* Only MGR owns nested configuration (its nested Krylov/MGR arguments). */
   if (method == PRECON_MGR)
   {
      hypredrv_MGRDestroyNestedSolverArgs(&args->mgr);
   }
}

void
hypredrv_PreconArgsDestroyRuntimeState(precon_t method, precon_args *args)
{
   if (!args)
   {
      return;
   }

   if (method == PRECON_BOOMERAMG)
   {
      hypredrv_AMGDestroyRBMs(&args->amg);
   }

   /* Only MGR keeps runtime state (cached component solvers) in its args. */
   if (method == PRECON_MGR)
   {
      hypredrv_MGRDestroy(&args->mgr, NULL, 0);
      hypredrv_MGRForgetCachedSolvers(&args->mgr);
   }
}

static int MGRHasConfiguredComponentReuse(const MGR_args *);

static int
NestedKrylovHasConfiguredComponentReuse(const NestedKrylov_args *args)
{
   if (!args || !args->has_precon)
   {
      return 0;
   }

   return args->precon_method == PRECON_MGR &&
          MGRHasConfiguredComponentReuse(&args->precon.mgr);
}

static int
MGRHasConfiguredComponentReuse(const MGR_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (args->coarsest_level.reuse.present)
   {
      return 1;
   }

   if (args->coarsest_level.use_krylov &&
       NestedKrylovHasConfiguredComponentReuse(args->coarsest_level.krylov))
   {
      return 1;
   }

   int max_levels = hypredrv_MGRNumFineLevels(args);
   for (int i = 0; i < max_levels; i++)
   {
      const MGRlvl_args *level_args = &args->level[i];

      if (level_args->f_relaxation.reuse.present ||
          level_args->g_relaxation.reuse.present)
      {
         return 1;
      }

      if (level_args->f_relaxation.use_krylov &&
          NestedKrylovHasConfiguredComponentReuse(level_args->f_relaxation.krylov))
      {
         return 1;
      }

      if (level_args->g_relaxation.use_krylov &&
          NestedKrylovHasConfiguredComponentReuse(level_args->g_relaxation.krylov))
      {
         return 1;
      }

      if (level_args->f_relaxation.mgr &&
          MGRHasConfiguredComponentReuse(level_args->f_relaxation.mgr))
      {
         return 1;
      }
   }

   return 0;
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconCreate(MPI_Comm comm, precon_t precon_method, precon_args *args,
                      IntArray *dofmap, HYPRE_IJVector vec_nn, HYPRE_Precon *precon_ptr,
                      const Stats *stats, int next_ls_id, const PreconOperators *ops)
{
   if (!precon_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner creation requires an output pointer");
      return;
   }
   *precon_ptr = NULL;

   const PreconOps *method_ops = PreconOpsLookup(precon_method);
   if (!method_ops)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Invalid preconditioner method (%d)", (int)precon_method);
      return;
   }
   if (precon_method != PRECON_NONE && (!method_ops->setup || !method_ops->solve))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("FSAI requires hypre >= 2.25.0");
      return;
   }
   if (precon_method != PRECON_NONE && !args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner creation requires method arguments");
      return;
   }

   if (precon_method == PRECON_MGR && !MGRHasConfiguredComponentReuse(&args->mgr))
   {
      hypredrv_PreconArgsDestroyRuntimeState(precon_method, args);
   }

   HYPRE_Precon precon = malloc(sizeof(hypre_Precon)); /* GCOVR_EXCL_BR_LINE */
   if (!precon)                                        /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_AllocationFailure("Memory allocation failed");
      return;
   }

   precon->main         = NULL;
   precon->comm         = comm;
   precon->method       = precon_method;
   precon->ops          = method_ops;
   precon->setup_called = 0;
   precon->args         = args;
   precon->operators_ok = 1;
   precon->stats        = NULL;

   switch (precon_method)
   {
      case PRECON_BOOMERAMG:
         /* GM1 marks modes projected to an MGR subblock. Full-system vectors would
          * not match that block's partitioning during interpolation setup. */
         if (args->amg.interp_vec_variant != 1)
         {
            hypredrv_AMGSetRBMs(&args->amg, vec_nn);
         }
         hypredrv_AMGCreate(&args->amg, &precon->main);
         break;

      case PRECON_MGR:
         hypredrv_MGRSetDofmap(&args->mgr, dofmap);
         hypredrv_MGRSetNearNullSpace(&args->mgr, vec_nn);
         hypredrv_MGRSetCoarseSchur(&args->mgr, ops ? ops->coarse_schur : NULL);
         hypredrv_MGRCreate(comm, &args->mgr, &precon->main, stats, next_ls_id);
         break;

      case PRECON_ILU:
         hypredrv_ILUCreate(&args->ilu, &precon->main);
         break;

      case PRECON_FSAI:
         hypredrv_FSAICreate(&args->fsai, &precon->main);
         break;

      case PRECON_AMS:
         hypredrv_AMSCreate(&args->ams, &precon->main);
         break;

      case PRECON_ADS:
         hypredrv_ADSCreate(&args->ads, &precon->main);
         break;

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      case PRECON_SCHWARZ:
         hypredrv_SchwarzCreate(&args->schwarz, &precon->main);
         break;
#endif

      case PRECON_NONE:
         break;

      default:
         break;
   }

   /* If a sub-create failed, do not publish a half-built wrapper whose main handle
    * is NULL: a later PreconSetup/PreconApply would otherwise pass NULL straight
    * into hypre and crash. A NULL main for any method other than PRECON_NONE is the
    * reliable failure signal here (the process-global error state is sticky and may
    * reflect an unrelated earlier failure, so it must not be used for this check). */
   if (precon_method != PRECON_NONE && !precon->main)
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Preconditioner creation failed (method=%d)",
                              (int)precon_method);
      }
      hypredrv_PreconArgsDestroyRuntimeState(precon_method, args);
      free(precon);
      return;
   }

   if (precon_method == PRECON_AMS)
   {
      precon->operators_ok =
         ops ? hypredrv_AMSSetOperators(precon->main, ops->G, ops->coord[0],
                                        ops->coord[1], ops->coord[2])
             : 0;
   }
   else if (precon_method == PRECON_ADS)
   {
      precon->operators_ok =
         ops ? hypredrv_ADSSetOperators(precon->main, ops->G, ops->C, ops->coord[0],
                                        ops->coord[1], ops->coord[2])
             : 0;
   }

   *precon_ptr = precon;
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconSupportsDevice
 *-----------------------------------------------------------------------------*/

static int PreconMGRSupportsDeviceAtDepth(const MGR_args *, char *, size_t, int);

static int
PreconNestedKrylovMGRSupportsDevice(const NestedKrylov_args *args, char *reason,
                                    size_t reason_size, int depth)
{
   if (!args || !args->has_precon || args->precon_method != PRECON_MGR)
   {
      return 1;
   }
   return PreconMGRSupportsDeviceAtDepth(&args->precon.mgr, reason, reason_size,
                                         depth + 1);
}

static int
PreconMGRSupportsDeviceAtDepth(const MGR_args *args, char *reason, size_t reason_size,
                               int depth)
{
   if (!args || depth >= MAX_MGR_LEVELS)
   {
      return 1;
   }

   if (args->interp_sweeps > 0)
   {
      hypredrv_FormatReason(reason, reason_size,
                            "MGR P2 interpolation refinement is currently CPU-only");
      return 0;
   }

   int fine_levels = hypredrv_MGRNumFineLevels(args);
   for (int level = 0; level < fine_levels; level++)
   {
      const MGRfrlx_args *frelax = &args->level[level].f_relaxation;
      const MGRgrlx_args *grelax = &args->level[level].g_relaxation;

      if (args->level[level].matched_f_backsolve)
      {
         hypredrv_FormatReason(
            reason, reason_size, "MGR level %d matched %s is currently CPU-only", level,
            args->level[level].matched_f_backsolve == 2 ? "Schur GMRES(1)"
                                                        : "F backsolve");
         return 0;
      }
      if (args->level[level].matched_q)
      {
         hypredrv_FormatReason(
            reason, reason_size, "MGR level %d matched %s Q is currently CPU-only", level,
            args->level[level].matched_q == 2 ? "adaptive FSAI" : "sparse");
         return 0;
      }
      if (frelax->symmetric_diagonal_scaling)
      {
         hypredrv_FormatReason(
            reason, reason_size,
            "MGR level %d symmetric diagonal F-solver scaling is CPU-only", level);
         return 0;
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      if (frelax->type == MGR_SOLVER_TYPE_SCHWARZ)
      {
         hypredrv_FormatReason(reason, reason_size,
                               "MGR level %d Schwarz F-relaxation is not ported to GPUs",
                               level);
         return 0;
      }
      if (grelax->type == MGR_SOLVER_TYPE_SCHWARZ)
      {
         hypredrv_FormatReason(
            reason, reason_size,
            "MGR level %d Schwarz global relaxation is not ported to GPUs", level);
         return 0;
      }
#endif
      if ((frelax->mgr && !PreconMGRSupportsDeviceAtDepth(frelax->mgr, reason,
                                                          reason_size, depth + 1)) ||
          !PreconNestedKrylovMGRSupportsDevice(frelax->krylov, reason, reason_size,
                                               depth) ||
          !PreconNestedKrylovMGRSupportsDevice(grelax->krylov, reason, reason_size,
                                               depth))
      {
         return 0;
      }
   }

   return PreconNestedKrylovMGRSupportsDevice(args->coarsest_level.krylov, reason,
                                              reason_size, depth);
}

int
hypredrv_PreconSupportsDevice(precon_t precon_method, const precon_args *args,
                              char *reason, size_t reason_size)
{
   if (!args)
   {
      return 1;
   }

   if (precon_method == PRECON_BOOMERAMG)
   {
      int interp_type = args->amg.interpolation.prolongation_type;
      if (interp_type == 8 || interp_type == 9)
      {
         hypredrv_FormatReason(reason, reason_size,
                               "BoomerAMG interpolation type %d is not ported to GPUs",
                               interp_type);
         return 0;
      }
   }

   if (precon_method == PRECON_MGR &&
       !PreconMGRSupportsDeviceAtDepth(&args->mgr, reason, reason_size, 0))
   {
      return 0;
   }

   if (reason && reason_size)
   {
      reason[0] = '\0';
   }
   return 1;
}

int
hypredrv_PreconValidateSetup(HYPRE_Precon precon, HYPRE_ParCSRMatrix matrix)
{
   if (!precon)
   {
      return 1;
   }

   /* Hypre can ignore a setup callback's return value and continue into
    * collective work. Agree on failures before entering any such callback. */
   hypredrv_HypreConsumeErrors(precon->comm);
   if (!precon->operators_ok)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_PRECON);
      if (precon->method == PRECON_AMS)
      {
         hypredrv_ErrorMsgAdd("AMS setup requires a discrete gradient matrix and "
                              "coordinate vectors with native hypre objects");
      }
      else
      {
         hypredrv_ErrorMsgAdd(
            "ADS setup requires a discrete gradient matrix, a discrete "
            "curl matrix, and coordinate vectors with native hypre objects");
      }
   }
   if (!matrix)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner setup requires a matrix");
   }
   if (hypredrv_DistributedErrorStateSync(precon->comm))
   {
      return 0;
   }
   return precon->method != PRECON_MGR ||
          hypredrv_MGRValidatePointMarkerRows(&precon->args->mgr, matrix);
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconSetup
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconSetup(HYPRE_Precon precon, HYPRE_IJMatrix A)
{
   void              *vA    = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   if (!precon)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Preconditioner setup requested with null preconditioner");
      return;
   }

   if (precon->method == PRECON_NONE)
   {
      return;
   }

   if (A)
   {
      HYPRE_IJMatrixGetObject(A, &vA);
   }
   par_A = (HYPRE_ParCSRMatrix)vA;
   if (!hypredrv_PreconValidateSetup(precon, par_A))
   {
      return;
   }

#if HYPREDRV_HYPRE_RELEASE_NUMBER == 22800
   /* hypre 2.28.0 unconditionally reads the RHS vector's component count during
    * ILU/FSAI and MGR setup, although setup otherwise accepts NULL vectors. */
   if (precon->method == PRECON_ILU || precon->method == PRECON_FSAI ||
       precon->method == PRECON_MGR)
   {
      HYPRE_ParVectorCreate(hypre_ParCSRMatrixComm(par_A),
                            hypre_ParCSRMatrixGlobalNumRows(par_A), NULL, &par_b);
      HYPRE_ParVectorInitialize(par_b);
      par_x = par_b;
   }
#endif

   hypredrv_PreconKrylovSetup((HYPRE_Solver)precon, par_A, par_b, par_x);

   hypredrv_HypreConsumeErrors(hypre_ParCSRMatrixComm(par_A));
#if HYPREDRV_HYPRE_RELEASE_NUMBER == 22800
   if (par_b)
   {
      HYPRE_ParVectorDestroy(par_b);
   }
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconApply
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconApply(HYPRE_Precon precon, HYPRE_IJMatrix A, HYPRE_IJVector b,
                     HYPRE_IJVector x)
{
   void              *vA = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;

   if (!precon)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Preconditioner apply requested with null preconditioner");
      return;
   }

   if (precon->method == PRECON_NONE)
   {
      return;
   }

   HYPRE_IJMatrixGetObject(A, &vA);
   par_A = (HYPRE_ParCSRMatrix)vA;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;

   precon->ops->solve(precon->main, par_A, par_b, par_x);
}

/*-----------------------------------------------------------------------------
 * hypredrv_PreconDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_PreconDestroy(HYPRE_Precon *precon_ptr, const Stats *stats, int ls_id)
{
   if (!precon_ptr) return;
   int log_rank = -1;          /* GCOVR_EXCL_BR_LINE */
   if (hypredrv_LogEnabled(3)) /* GCOVR_EXCL_BR_LINE */
   {
      log_rank = hypredrv_LogRankFromComm(MPI_COMM_WORLD);
   }

   char        obj_name_buf[32];
   const char *obj_name =
      hypredrv_StatsGetLogObjectName(stats, obj_name_buf, sizeof(obj_name_buf));

   HYPRE_Precon precon = *precon_ptr;
   if (!precon) /* GCOVR_EXCL_BR_LINE */
   {
      /* GCOVR_EXCL_START */
      HYPREDRV_LOGF(3, log_rank, obj_name, ls_id,
                    "preconditioner destroy skipped: object already NULL");
      /* GCOVR_EXCL_STOP */
      return;
   }

   /* MGR teardown also cleans component solvers cached before an outer-MGR
    * create failure, even when there is no main handle. */
   if (precon->main || precon->method == PRECON_MGR)
   {
      HYPREDRV_LOGF(3, log_rank, obj_name, ls_id,
                    "preconditioner destroy dispatch: method=%s", precon->ops->name);
      if (precon->method == PRECON_MGR)
      {
         hypredrv_MGRDestroy(&precon->args->mgr, &precon->main, precon->setup_called);
      }
      else
      {
         if (precon->ops->destroy)
         {
            precon->ops->destroy(precon->main);
         }
         if (precon->method == PRECON_BOOMERAMG)
         {
            hypredrv_AMGDestroyRBMs(&precon->args->amg);
         }
      }
   }

   free(precon);
   *precon_ptr = NULL;
}
