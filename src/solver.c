/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/solver.h"

#include <math.h>
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"
#include "internal/gen_macros.h"
#include "logging.h"

#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
static HYPRE_Int
FSAISetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
FSAISolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define LOCAL_FSAI_SETUP FSAISetupStub
#define LOCAL_FSAI_SOLVE FSAISolveStub
#else
#define LOCAL_FSAI_SETUP HYPRE_FSAISetup
#define LOCAL_FSAI_SOLVE HYPRE_FSAISolve
#endif

#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
static HYPRE_Int
ILUSetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
             HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
ILUSolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
             HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define LOCAL_ILU_SETUP ILUSetupStub
#define LOCAL_ILU_SOLVE ILUSolveStub
#else
#define LOCAL_ILU_SETUP HYPRE_ILUSetup
#define LOCAL_ILU_SOLVE HYPRE_ILUSolve
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

static MPI_Comm
SolverCommFromMatrix(HYPRE_IJMatrix mat)
{
   if (!mat)
   {
      return MPI_COMM_NULL;
   }

   void *obj = NULL;
   HYPRE_IJMatrixGetObject(mat, &obj);
   /* GCOVR_EXCL_BR_START */
   if (!obj) /* GCOVR_EXCL_BR_STOP */
   {
      return MPI_COMM_NULL;
   }

   return hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *)obj);
}

static MPI_Comm
SolverCommFromVector(HYPRE_IJVector vec)
{
   if (!vec)
   {
      return MPI_COMM_NULL;
   }

   return hypre_IJVectorComm((hypre_IJVector *)vec);
}

static MPI_Comm
SolverCommResolve(HYPRE_IJMatrix A, HYPRE_IJVector b, HYPRE_IJVector x)
{
   MPI_Comm comm = SolverCommFromMatrix(A);
   if (comm != MPI_COMM_NULL)
   {
      return comm;
   }

   comm = SolverCommFromVector(b);
   if (comm != MPI_COMM_NULL)
   {
      return comm;
   }

   return SolverCommFromVector(x);
}

static int
SolverLinearSystemID(const Stats *stats)
{
   return stats ? hypredrv_StatsGetLinearSystemID(stats) : 0;
}

static const char *
SolverLogObjectName(const Stats *stats, char *buf, size_t buf_size)
{
   if (stats && stats->object_name[0] != '\0')
   {
      return stats->object_name;
   }
   if (stats && stats->runtime_object_id > 0 && buf && buf_size > 0)
   {
      snprintf(buf, buf_size, "obj-%d", stats->runtime_object_id);
      return buf;
   }
   return NULL;
}

static HYPRE_Int
PreconSetupDispatch(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                    HYPRE_ParVector x)
{
   HYPRE_Precon precon = (HYPRE_Precon)solver;

   /* GCOVR_EXCL_BR_START */
   if (!precon) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   if (precon->stats)
   {
      hypredrv_StatsAnnotate(precon->stats, HYPREDRV_ANNOTATE_BEGIN, "prec");
   }

   HYPRE_Int ierr = 0;
   switch (precon->method)
   {
      case PRECON_BOOMERAMG:
         ierr = HYPRE_BoomerAMGSetup(precon->main, A, b, x);
         break;

      case PRECON_MGR:
         ierr = HYPRE_MGRSetup(precon->main, A, b, x);
         break;

      case PRECON_ILU:
         ierr = LOCAL_ILU_SETUP(precon->main, A, b, x);
         break;

      case PRECON_FSAI:
         ierr = LOCAL_FSAI_SETUP(precon->main, A, b, x);
         break;

      /* GCOVR_EXCL_BR_START */
      case PRECON_NONE:
         /* GCOVR_EXCL_BR_STOP */
         break;
   }

   if (precon->stats)
   {
      hypredrv_StatsAnnotate(precon->stats, HYPREDRV_ANNOTATE_END, "prec");
   }

   return ierr;
}

static HYPRE_Int
PreconSolveDispatch(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                    HYPRE_ParVector x)
{
   HYPRE_Precon precon = (HYPRE_Precon)solver;

   /* GCOVR_EXCL_BR_START */
   if (!precon) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }

   switch (precon->method)
   {
      case PRECON_BOOMERAMG:
         return HYPRE_BoomerAMGSolve(precon->main, A, b, x);

      case PRECON_MGR:
         return HYPRE_MGRSolve(precon->main, A, b, x);

      case PRECON_ILU:
         return LOCAL_ILU_SOLVE(precon->main, A, b, x);

      case PRECON_FSAI:
         return LOCAL_FSAI_SOLVE(precon->main, A, b, x);

      /* GCOVR_EXCL_BR_START */
      case PRECON_NONE:
         /* GCOVR_EXCL_BR_STOP */
         return 0;
   }

   /* GCOVR_EXCL_BR_START */
   return 0;
   /* GCOVR_EXCL_BR_STOP */
}

#define Solver_FIELDS(_prefix)                                     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pcg, hypredrv_PCGSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, gmres, hypredrv_GMRESSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fgmres, hypredrv_FGMRESSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, bicgstab, hypredrv_BiCGSTABSetArgs)

DEFINE_FIELD_OFFSET_MAP(Solver)
#define Solver_NUM_FIELDS \
   (sizeof(Solver_field_offset_map) / sizeof(Solver_field_offset_map[0]))

DEFINE_SET_FIELD_BY_NAME_FUNC(hypredrv_SolverSetFieldByName, Solver_args,
                              Solver_field_offset_map, Solver_NUM_FIELDS)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_SolverGetValidKeys, Solver_NUM_FIELDS,
                           Solver_field_offset_map)

/*-----------------------------------------------------------------------------
 * SolverGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_SolverGetValidTypeIntMap(void)
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
   if (!args)
   {
      return;
   }

   switch (method)
   {
      case SOLVER_PCG:
         hypredrv_PCGSetDefaultArgs(&args->pcg);
         break;
      case SOLVER_GMRES:
         hypredrv_GMRESSetDefaultArgs(&args->gmres);
         break;
      case SOLVER_FGMRES:
         hypredrv_FGMRESSetDefaultArgs(&args->fgmres);
         break;
      case SOLVER_BICGSTAB:
         hypredrv_BiCGSTABSetDefaultArgs(&args->bicgstab);
         break;
      default:
         break;
   }
}

/*-----------------------------------------------------------------------------
 * SolverSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

DEFINE_SET_ARGS_FROM_YAML_FUNC(Solver, hypredrv_Solver)

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverCreate(MPI_Comm comm, solver_t solver_method, solver_args *args,
                      HYPRE_Solver *solver_ptr)
{
   int log_rank = -1;
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

   switch (solver_method)
   {
      case SOLVER_PCG:
         hypredrv_PCGCreate(comm, &args->pcg, solver_ptr);
         break;

      case SOLVER_GMRES:
         hypredrv_GMRESCreate(comm, &args->gmres, solver_ptr);
         break;

      case SOLVER_FGMRES:
         hypredrv_FGMRESCreate(comm, &args->fgmres, solver_ptr);
         break;

      case SOLVER_BICGSTAB:
         hypredrv_BiCGSTABCreate(comm, &args->bicgstab, solver_ptr);
         break;

      /* GCOVR_EXCL_BR_START */
      default:
         /* GCOVR_EXCL_BR_STOP */
         *solver_ptr = NULL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("SolverCreate: invalid solver method");
         HYPREDRV_LOGF(2, log_rank, NULL, 0,
                       "solver create failed: invalid solver method=%d",
                       (int)solver_method);
   }
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
   MPI_Comm    log_comm = SolverCommResolve(M, b, x);
   int         ls_id    = SolverLinearSystemID(stats);
   int         log_rank = -1;
   char        log_name_buf[32];
   const char *log_object_name =
      SolverLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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

   switch (solver_method)
   {
      case SOLVER_PCG:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRPCGSetPrecond(solver, PreconSolveDispatch,
                                      skip_precon_setup ? PreconSetupNoop
                                                        : PreconSetupDispatch,
                                      (HYPRE_Solver)precon);
         }
         HYPRE_ParCSRPCGSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_GMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRGMRESSetPrecond(solver, PreconSolveDispatch,
                                        skip_precon_setup ? PreconSetupNoop
                                                          : PreconSetupDispatch,
                                        (HYPRE_Solver)precon);
         }
         HYPRE_ParCSRGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRFlexGMRESSetPrecond(solver, PreconSolveDispatch,
                                            skip_precon_setup ? PreconSetupNoop
                                                              : PreconSetupDispatch,
                                            (HYPRE_Solver)precon);
         }
         HYPRE_ParCSRFlexGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRBiCGSTABSetPrecond(solver, PreconSolveDispatch,
                                           skip_precon_setup ? PreconSetupNoop
                                                             : PreconSetupDispatch,
                                           (HYPRE_Solver)precon);
         }
         HYPRE_ParCSRBiCGSTABSetup(solver, par_M, par_b, par_x);
         break;

      /* GCOVR_EXCL_BR_START */
      default:
         /* GCOVR_EXCL_BR_STOP */
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("SolverSetup: invalid solver method");
         HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                       "solver setup failed: invalid solver method=%d",
                       (int)solver_method);
         return;
   }

   /* Clear pending error codes from hypre */
   HYPRE_ClearAllErrors();
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver setup end");
}

void
hypredrv_SolverSetup(precon_t precon_method, solver_t solver_method, HYPRE_Precon precon,
                     HYPRE_Solver solver, HYPRE_IJMatrix M, HYPRE_IJVector b,
                     HYPRE_IJVector x, Stats *stats)
{
   hypredrv_SolverSetupWithReuse(precon_method, solver_method, precon, solver, M, b, x,
                                 stats, 0);
}

/*-----------------------------------------------------------------------------
 * SolverSolveOnly
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_SolverSolveOnly(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
                         HYPRE_IJVector b, HYPRE_IJVector x)
{
   MPI_Comm log_comm = SolverCommResolve(A, b, x);
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

   HYPRE_IJMatrixGetObject(A, &vA);
   par_A = (HYPRE_ParCSRMatrix)vA;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSolve(solver, par_A, par_b, par_x);
         HYPRE_PCGGetNumIterations(solver, &iters);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
         HYPRE_GMRESGetNumIterations(solver, &iters);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSolve(solver, par_A, par_b, par_x);
         HYPRE_FlexGMRESGetNumIterations(solver, &iters);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSolve(solver, par_A, par_b, par_x);
         HYPRE_BiCGSTABGetNumIterations(solver, &iters);
         break;

      /* GCOVR_EXCL_BR_START */
      default:
         /* GCOVR_EXCL_BR_STOP */
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("SolverSolveOnly: invalid solver method");
         HYPREDRV_LOGF(2, log_rank, NULL, ls_id,
                       "solver solve failed: invalid solver method=%d",
                       (int)solver_method);
         return -1;
   }

   /* Clear pending error codes from hypre */
   HYPRE_ClearAllErrors();

   return iters;
}

/*-----------------------------------------------------------------------------
 * SolverApply
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverApply(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
                     HYPRE_IJVector b, HYPRE_IJVector x, Stats *stats)
{
   MPI_Comm    log_comm = SolverCommResolve(A, b, x);
   int         ls_id    = SolverLinearSystemID(stats);
   int         log_rank = -1;
   char        log_name_buf[32];
   const char *log_object_name =
      SolverLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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

   HYPRE_Int     iters  = 0;
   HYPRE_Complex b_norm = NAN, r_norm = NAN, r0_norm = NAN;

   /* Compute initial residual norm (absolute L2) before timing the solve */
   hypredrv_LinearSystemComputeResidualNorm(A, b, x, "L2", &r0_norm);

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsInitialResNormSet(stats, r0_norm);

   iters = hypredrv_SolverSolveOnly(solver_method, solver, A, b, x);

   if (iters < 0)
   {
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
   hypredrv_LinearSystemComputeResidualNorm(A, b, x, "L2", &r_norm);
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
      switch (solver_method)
      {
         case SOLVER_PCG:
            HYPRE_ParCSRPCGDestroy(*solver_ptr);
            break;

         case SOLVER_GMRES:
            HYPRE_ParCSRGMRESDestroy(*solver_ptr);
            break;

         case SOLVER_FGMRES:
            HYPRE_ParCSRFlexGMRESDestroy(*solver_ptr);
            break;

         case SOLVER_BICGSTAB:
            HYPRE_ParCSRBiCGSTABDestroy(*solver_ptr);
            break;

         /* GCOVR_EXCL_BR_START */
         default:
            /* GCOVR_EXCL_BR_STOP */
            HYPREDRV_LOGF(2, log_rank, NULL, 0,
                          "solver destroy skipped: invalid solver method=%d",
                          (int)solver_method);
            return;
      }

      *solver_ptr = NULL;
   }
   else
   {
      HYPREDRV_LOGF(3, log_rank, NULL, 0, "solver destroy skipped: solver already NULL");
   }
}
