/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "solver_ops.h"

#include <math.h>
#include "_hypre_IJ_mv.h"
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
#include "_hypre_krylov.h" // For hypre_BiCGSTABGetConverged
#endif
#include "_hypre_parcsr_mv.h"
#include "internal/gen_macros.h"
#include "internal/linsys.h"
#include "logging.h"

#if !HYPRE_CHECK_MIN_VERSION(30000, 0)
/* Before hypre 3.0.0 this prototype lives in hypre's internal "krylov.h",
   which is shadowed by this project's internal/krylov.h on the include
   path, so declare it directly. */
HYPRE_Int hypre_BiCGSTABGetConverged(void *bicgstab_vdata, HYPRE_Int *converged);
#endif

static HYPRE_Int
PreconSetupNoop(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 0;
}

static int
SolverLinearSystemID(const Stats *stats)
{
   return stats ? hypredrv_StatsGetLinearSystemID(stats) : 0;
}

static void
SolverPCGSetDefaults(solver_args *args)
{
   hypredrv_PCGSetDefaultArgs(&args->pcg);
}

static void
SolverPCGCreate(MPI_Comm comm, const solver_args *args, HYPRE_Solver *solver_ptr)
{
   hypredrv_PCGCreate(comm, &args->pcg, solver_ptr);
}

static void
SolverGMRESSetDefaults(solver_args *args)
{
   hypredrv_GMRESSetDefaultArgs(&args->gmres);
}

static void
SolverGMRESCreate(MPI_Comm comm, const solver_args *args, HYPRE_Solver *solver_ptr)
{
   hypredrv_GMRESCreate(comm, &args->gmres, solver_ptr);
}

static void
SolverFGMRESSetDefaults(solver_args *args)
{
   hypredrv_FGMRESSetDefaultArgs(&args->fgmres);
}

static void
SolverFGMRESCreate(MPI_Comm comm, const solver_args *args, HYPRE_Solver *solver_ptr)
{
   hypredrv_FGMRESCreate(comm, &args->fgmres, solver_ptr);
}

static void
SolverBiCGSTABSetDefaults(solver_args *args)
{
   hypredrv_BiCGSTABSetDefaultArgs(&args->bicgstab);
}

static void
SolverBiCGSTABCreate(MPI_Comm comm, const solver_args *args, HYPRE_Solver *solver_ptr)
{
   hypredrv_BiCGSTABCreate(comm, &args->bicgstab, solver_ptr);
}

static HYPRE_Int
SolverBiCGSTABGetConverged(HYPRE_Solver solver, HYPRE_Int *converged)
{
   /* hypre does not expose HYPRE_BiCGSTABGetConverged; use the internal API. */
   return hypre_BiCGSTABGetConverged(solver, converged);
}

static const SolverOps solver_ops[] = {
   [SOLVER_PCG] =
      {
         .set_defaults           = SolverPCGSetDefaults,
         .get_keys               = hypredrv_PCGGetValidKeys,
         .get_values             = hypredrv_PCGGetValidValues,
         .set_field              = hypredrv_PCGSetFieldByName,
         .create                 = SolverPCGCreate,
         .set_precond            = HYPRE_ParCSRPCGSetPrecond,
         .setup                  = HYPRE_ParCSRPCGSetup,
         .solve                  = HYPRE_ParCSRPCGSolve,
         .destroy                = HYPRE_ParCSRPCGDestroy,
         .get_num_iterations     = HYPRE_PCGGetNumIterations,
         .get_converged          = HYPRE_PCGGetConverged,
         .get_final_rel_res_norm = HYPRE_PCGGetFinalRelativeResidualNorm,
      },
   [SOLVER_GMRES] =
      {
         .set_defaults           = SolverGMRESSetDefaults,
         .get_keys               = hypredrv_GMRESGetValidKeys,
         .get_values             = hypredrv_GMRESGetValidValues,
         .set_field              = hypredrv_GMRESSetFieldByName,
         .create                 = SolverGMRESCreate,
         .set_precond            = HYPRE_ParCSRGMRESSetPrecond,
         .setup                  = HYPRE_ParCSRGMRESSetup,
         .solve                  = HYPRE_ParCSRGMRESSolve,
         .destroy                = HYPRE_ParCSRGMRESDestroy,
         .get_num_iterations     = HYPRE_GMRESGetNumIterations,
         .get_converged          = HYPRE_GMRESGetConverged,
         .get_final_rel_res_norm = HYPRE_GMRESGetFinalRelativeResidualNorm,
      },
   [SOLVER_FGMRES] =
      {
         .set_defaults           = SolverFGMRESSetDefaults,
         .get_keys               = hypredrv_FGMRESGetValidKeys,
         .get_values             = hypredrv_FGMRESGetValidValues,
         .set_field              = hypredrv_FGMRESSetFieldByName,
         .create                 = SolverFGMRESCreate,
         .set_precond            = HYPRE_ParCSRFlexGMRESSetPrecond,
         .setup                  = HYPRE_ParCSRFlexGMRESSetup,
         .solve                  = HYPRE_ParCSRFlexGMRESSolve,
         .destroy                = HYPRE_ParCSRFlexGMRESDestroy,
         .get_num_iterations     = HYPRE_FlexGMRESGetNumIterations,
         .get_converged          = HYPRE_FlexGMRESGetConverged,
         .get_final_rel_res_norm = HYPRE_FlexGMRESGetFinalRelativeResidualNorm,
      },
   [SOLVER_BICGSTAB] =
      {
         .set_defaults           = SolverBiCGSTABSetDefaults,
         .get_keys               = hypredrv_BiCGSTABGetValidKeys,
         .get_values             = hypredrv_BiCGSTABGetValidValues,
         .set_field              = hypredrv_BiCGSTABSetFieldByName,
         .create                 = SolverBiCGSTABCreate,
         .set_precond            = HYPRE_ParCSRBiCGSTABSetPrecond,
         .setup                  = HYPRE_ParCSRBiCGSTABSetup,
         .solve                  = HYPRE_ParCSRBiCGSTABSolve,
         .destroy                = HYPRE_ParCSRBiCGSTABDestroy,
         .get_num_iterations     = HYPRE_BiCGSTABGetNumIterations,
         .get_converged          = SolverBiCGSTABGetConverged,
         .get_final_rel_res_norm = HYPRE_BiCGSTABGetFinalRelativeResidualNorm,
      },
};

const SolverOps *
hypredrv_SolverOpsLookup(solver_t method)
{
   if ((unsigned)method >= (unsigned)(sizeof(solver_ops) / sizeof(solver_ops[0])) ||
       !solver_ops[method].setup)
   {
      return NULL;
   }

   return &solver_ops[method];
}

#define solver_FIELDS(_prefix)                                     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pcg, hypredrv_PCGSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, gmres, hypredrv_GMRESSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fgmres, hypredrv_FGMRESSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, bicgstab, hypredrv_BiCGSTABSetArgs)

DEFINE_FIELD_OFFSET_MAP(solver)
#define solver_NUM_FIELDS \
   (sizeof(solver_field_offset_map) / sizeof(solver_field_offset_map[0]))

static DEFINE_SET_FIELD_BY_NAME_FUNC(hypredrv_SolverSetFieldByName, solver_args,
                                     solver_field_offset_map, solver_NUM_FIELDS)
   DEFINE_GET_VALID_KEYS_FUNC(hypredrv_SolverGetValidKeys, solver_NUM_FIELDS,
                              solver_field_offset_map)

   /*-----------------------------------------------------------------------------
    * SolverGetValidTypeIntMap
    *-----------------------------------------------------------------------------*/

   StrIntMapArray hypredrv_SolverGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {
      {"pcg", (int)SOLVER_PCG},
      {"gmres", (int)SOLVER_GMRES},
      {"fgmres", (int)SOLVER_FGMRES},
      {"bicgstab", (int)SOLVER_BICGSTAB},
   };

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * SolverGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_SolverGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      return hypredrv_SolverGetValidTypeIntMap();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * SolverArgsSetDefaultsForMethod
 *
 * Convenience helper for callers that need per-method defaults without constructing
 * fake YAML nodes (e.g., value-only `solver: gmres`).
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverArgsSetDefaultsForMethod(solver_t method, solver_args *args)
{
   const SolverOps *ops;

   if (!args)
   {
      return;
   }

   ops = hypredrv_SolverOpsLookup(method);
   if (!ops)
   {
      return;
   }

   ops->set_defaults(args);
}

void
hypredrv_SolverSetMethodArgs(solver_t method, solver_args *args, YAMLnode *body,
                             const YAMLnode *ignored_child)
{
   const SolverOps *ops = hypredrv_SolverOpsLookup(method);
   if (!ops || !args || !body)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid solver method arguments");
      return;
   }

   ops->set_defaults(args);
   YAML_NODE_SET_VALID(body);
   if (body->children || body->val[0])
   {
      hypredrv_YAMLSetArgsGeneric(args, body, ops->get_keys, ops->get_values,
                                  ops->set_field, ignored_child);
   }
}

/*-----------------------------------------------------------------------------
 * SolverSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

DEFINE_SET_ARGS_FROM_YAML_FUNC(solver, hypredrv_Solver)

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverCreate(MPI_Comm comm, solver_t solver_method, solver_args *args,
                      HYPRE_Solver *solver_ptr)
{
   const SolverOps *ops;
   int              log_rank = -1;
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(comm);
   }
   if (!solver_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverCreate: solver_ptr is NULL");
      HYPREDRV_LOGF(2, log_rank, NULL, 0, "solver create failed: solver_ptr is NULL");
      return;
   }

   ops = hypredrv_SolverOpsLookup(solver_method);
   /* GCOVR_EXCL_BR_START */
   if (!ops) /* GCOVR_EXCL_BR_STOP */
   {
      *solver_ptr = NULL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverCreate: invalid solver method");
      HYPREDRV_LOGF(2, log_rank, NULL, 0,
                    "solver create failed: invalid solver method=%d", (int)solver_method);
      return;
   }

   ops->create(comm, args, solver_ptr);
}

/*-----------------------------------------------------------------------------
 * SolverSetup
 *
 * TODO: split this function into hypredrv_PreconSetup and SolverSetup
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverSetupWithReuse(precon_t precon_method, solver_t solver_method,
                              HYPRE_Precon precon, HYPRE_Solver solver, HYPRE_IJMatrix M,
                              HYPRE_IJVector b, HYPRE_IJVector x, Stats *stats,
                              int skip_precon_setup)
{
   MPI_Comm    log_comm = hypredrv_IJSystemComm(M, NULL, b, x);
   int         ls_id    = SolverLinearSystemID(stats);
   int         log_rank = -1;
   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(log_comm);
   }
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id,
                 "solver setup begin (solver=%d precon=%d skip_precon_setup=%d)",
                 (int)solver_method, (int)precon_method, skip_precon_setup);

   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSetup: solver is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver setup failed: solver is NULL");
      return;
   }

   if (!M || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverSetup: matrix or vector is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver setup failed: matrix or vector is NULL");
      return;
   }

   if (precon_method != PRECON_NONE && !precon)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "SolverSetup: precon is NULL but precon_method is not PRECON_NONE");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver setup failed: preconditioner is NULL");
      return;
   }

   void              *vM = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix par_M = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   const SolverOps   *ops = hypredrv_SolverOpsLookup(solver_method);

   HYPRE_IJMatrixGetObject(M, &vM);
   par_M = (HYPRE_ParCSRMatrix)vM;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;
   if (precon)
   {
      precon->stats = skip_precon_setup ? NULL : stats;
      if (skip_precon_setup)
      {
         HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id,
                       "solver setup: detached preconditioner stats (skip_precon_setup)");
      }
   }

   /* GCOVR_EXCL_BR_START */
   if (!ops) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSetup: invalid solver method");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver setup failed: invalid solver method=%d", (int)solver_method);
      return;
   }

   if (precon_method != PRECON_NONE && !hypredrv_PreconValidateSetup(precon, par_M))
   {
      return;
   }

   if (precon_method != PRECON_NONE)
   {
      ops->set_precond(solver, hypredrv_PreconKrylovSolve,
                       skip_precon_setup ? PreconSetupNoop : hypredrv_PreconKrylovSetup,
                       (HYPRE_Solver)precon);
   }
   ops->setup(solver, par_M, par_b, par_x);

   hypredrv_HypreConsumeErrors(log_comm);
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver setup end");
}

/*-----------------------------------------------------------------------------
 * SolverSolveOnly
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_SolverSolveOnly(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
                         HYPRE_IJVector b, HYPRE_IJVector x)
{
   MPI_Comm log_comm = hypredrv_IJSystemComm(A, NULL, b, x);
   int      ls_id    = 0;
   int      log_rank = -1;
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(log_comm);
   }

   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSolveOnly: solver is NULL");
      HYPREDRV_LOGF(2, log_rank, NULL, ls_id, "solver solve failed: solver is NULL");
      return -1;
   }

   if (!A || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverSolveOnly: matrix or vector is NULL");
      HYPREDRV_LOGF(2, log_rank, NULL, ls_id,
                    "solver solve failed: matrix or vector is NULL");
      return -1;
   }

   void              *vA = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   HYPRE_Int          iters = 0;
   const SolverOps   *ops   = hypredrv_SolverOpsLookup(solver_method);

   HYPRE_IJMatrixGetObject(A, &vA);
   par_A = (HYPRE_ParCSRMatrix)vA;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;

   /* GCOVR_EXCL_BR_START */
   if (!ops) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSolveOnly: invalid solver method");
      HYPREDRV_LOGF(2, log_rank, NULL, ls_id,
                    "solver solve failed: invalid solver method=%d", (int)solver_method);
      return -1;
   }

   ops->solve(solver, par_A, par_b, par_x);
   ops->get_num_iterations(solver, &iters);

   hypredrv_HypreConsumeErrors(log_comm);

   return iters;
}

/*-----------------------------------------------------------------------------
 * SolverApply
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverApply(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
                     HYPRE_IJVector b, HYPRE_IJVector x, Stats *stats)
{
   MPI_Comm    log_comm = hypredrv_IJSystemComm(A, NULL, b, x);
   int         ls_id    = SolverLinearSystemID(stats);
   int         log_rank = -1;
   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(log_comm);
   }
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver apply begin (solver=%d)",
                 (int)solver_method);

   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverApply: solver is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed: solver is NULL");
      return;
   }

   if (!A || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverApply: matrix or vector is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed: matrix or vector is NULL");
      return;
   }

   HYPRE_Int      iters  = 0;
   HYPRE_Complex  b_norm = NAN, r_norm = NAN, r0_norm = NAN;
   HYPRE_IJVector vec_r = NULL; /* residual work vector shared by both norms */

   /* Compute initial residual norm (absolute L2) before timing the solve */
   hypredrv_LinearSystemComputeResidualNormWork(A, b, x, "L2", &vec_r, &r0_norm);

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsInitialResNormSet(stats, r0_norm);

   iters = hypredrv_SolverSolveOnly(solver_method, solver, A, b, x);

   if (iters < 0)
   {
      if (vec_r)
      {
         HYPRE_IJVectorDestroy(vec_r);
      }
      hypredrv_StatsIterSet(stats, 0);
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed during solve");
      return;
   }

   hypredrv_StatsIterSet(stats, (int)iters);
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");

   /* Compute the real relative residual norm. Note this is not timed */
   hypredrv_LinearSystemComputeVectorNorm(b, "L2", &b_norm);
   hypredrv_LinearSystemComputeResidualNormWork(A, b, x, "L2", &vec_r, &r_norm);
   if (vec_r)
   {
      HYPRE_IJVectorDestroy(vec_r);
   }
   b_norm = (b_norm > 0.0) ? b_norm : 1.0;

   hypredrv_StatsRelativeResNormSet(stats, r_norm / b_norm);
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver apply end (iters=%d)",
                 (int)iters);
}

/*-----------------------------------------------------------------------------
 * SolverDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverDestroy(solver_t solver_method, HYPRE_Solver *solver_ptr)
{
   int log_rank = -1;
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(MPI_COMM_WORLD);
   }

   if (!solver_ptr)
   {
      HYPREDRV_LOGF(2, log_rank, NULL, 0, "solver destroy skipped: solver_ptr is NULL");
      return;
   }

   if (*solver_ptr)
   {
      const SolverOps *ops = hypredrv_SolverOpsLookup(solver_method);

      /* GCOVR_EXCL_BR_START */
      if (!ops) /* GCOVR_EXCL_BR_STOP */
      {
         HYPREDRV_LOGF(2, log_rank, NULL, 0,
                       "solver destroy skipped: invalid solver method=%d",
                       (int)solver_method);
         return;
      }

      ops->destroy(*solver_ptr);
      *solver_ptr = NULL;
   }
   else
   {
      HYPREDRV_LOGF(3, log_rank, NULL, 0, "solver destroy skipped: solver already NULL");
   }
}

/*-----------------------------------------------------------------------------
 * SolverGetConverged
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverGetConverged(solver_t solver_method, HYPRE_Solver solver,
                            HYPRE_Int *converged_ptr)
{
   HYPRE_Int        converged = 0;
   const SolverOps *ops;

   if (!solver || !converged_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverGetConverged: solver and converged_ptr must be"
                           " non-NULL");
      return;
   }

   ops = hypredrv_SolverOpsLookup(solver_method);
   /* GCOVR_EXCL_BR_START */
   if (!ops) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverGetConverged: invalid solver method");
      return;
   }

   ops->get_converged(solver, &converged);
   *converged_ptr = converged;
}

/*-----------------------------------------------------------------------------
 * SolverGetFinalRelativeResidualNorm
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverGetFinalRelativeResidualNorm(solver_t solver_method, HYPRE_Solver solver,
                                            HYPRE_Real *norm_ptr)
{
   HYPRE_Real       norm = 0.0;
   const SolverOps *ops;

   if (!solver || !norm_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverGetFinalRelativeResidualNorm: solver and norm_ptr"
                           " must be non-NULL");
      return;
   }

   ops = hypredrv_SolverOpsLookup(solver_method);
   /* GCOVR_EXCL_BR_START */
   if (!ops) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverGetFinalRelativeResidualNorm: invalid solver"
                           " method");
      return;
   }

   ops->get_final_rel_res_norm(solver, &norm);

   *norm_ptr = norm;
}
