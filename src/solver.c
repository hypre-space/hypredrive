/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "solver.h"

#include <math.h>
#include "gen_macros.h"

#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
static HYPRE_Int
HYPREDRV_FSAISetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                       HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
HYPREDRV_FSAISolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                       HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define HYPREDRV_FSAI_SETUP HYPREDRV_FSAISetupStub
#define HYPREDRV_FSAI_SOLVE HYPREDRV_FSAISolveStub
#else
#define HYPREDRV_FSAI_SETUP HYPRE_FSAISetup
#define HYPREDRV_FSAI_SOLVE HYPRE_FSAISolve
#endif

#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
static HYPRE_Int
HYPREDRV_ILUSetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
HYPREDRV_ILUSolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define HYPREDRV_ILU_SETUP HYPREDRV_ILUSetupStub
#define HYPREDRV_ILU_SOLVE HYPREDRV_ILUSolveStub
#else
#define HYPREDRV_ILU_SETUP HYPRE_ILUSetup
#define HYPREDRV_ILU_SOLVE HYPRE_ILUSolve
#endif

static HYPRE_Int
HYPREDRV_PreconSetupNoop(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                         HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 0;
}

#define Solver_FIELDS(_prefix)                            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pcg, PCGSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, gmres, GMRESSetArgs)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fgmres, FGMRESSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, bicgstab, BiCGSTABSetArgs)

DEFINE_FIELD_OFFSET_MAP(Solver)
#define Solver_NUM_FIELDS \
   (sizeof(Solver_field_offset_map) / sizeof(Solver_field_offset_map[0]))

DEFINE_SET_FIELD_BY_NAME_FUNC(SolverSetFieldByName, Solver_args, Solver_field_offset_map,
                              Solver_NUM_FIELDS)
DEFINE_GET_VALID_KEYS_FUNC(SolverGetValidKeys, Solver_NUM_FIELDS, Solver_field_offset_map)

/*-----------------------------------------------------------------------------
 * SolverGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
SolverGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {{"pcg", (int)SOLVER_PCG},
                             {"gmres", (int)SOLVER_GMRES},
                             {"fgmres", (int)SOLVER_FGMRES},
                             {"bicgstab", (int)SOLVER_BICGSTAB}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * SolverGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
SolverGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      return SolverGetValidTypeIntMap();
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
SolverArgsSetDefaultsForMethod(solver_t method, solver_args *args)
{
   if (!args)
   {
      return;
   }

   switch (method)
   {
      case SOLVER_PCG:
         PCGSetDefaultArgs(&args->pcg);
         break;
      case SOLVER_GMRES:
         GMRESSetDefaultArgs(&args->gmres);
         break;
      case SOLVER_FGMRES:
         FGMRESSetDefaultArgs(&args->fgmres);
         break;
      case SOLVER_BICGSTAB:
         BiCGSTABSetDefaultArgs(&args->bicgstab);
         break;
      default:
         break;
   }
}

/*-----------------------------------------------------------------------------
 * SolverSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

DEFINE_SET_ARGS_FROM_YAML_FUNC(Solver)

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

void
SolverCreate(MPI_Comm comm, solver_t solver_method, solver_args *args,
             HYPRE_Solver *solver_ptr)
{
   if (!solver_ptr)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("SolverCreate: solver_ptr is NULL");
      return;
   }

   switch (solver_method)
   {
      case SOLVER_PCG:
         PCGCreate(comm, &args->pcg, solver_ptr);
         break;

      case SOLVER_GMRES:
         GMRESCreate(comm, &args->gmres, solver_ptr);
         break;

      case SOLVER_FGMRES:
         FGMRESCreate(comm, &args->fgmres, solver_ptr);
         break;

      case SOLVER_BICGSTAB:
         BiCGSTABCreate(comm, &args->bicgstab, solver_ptr);
         break;

      default:
         *solver_ptr = NULL;
         ErrorCodeSet(ERROR_INVALID_SOLVER);
         ErrorMsgAdd("SolverCreate: invalid solver method");
   }
}

/*-----------------------------------------------------------------------------
 * SolverSetup
 *
 * TODO: split this function into PreconSetup and SolverSetup
 *-----------------------------------------------------------------------------*/

void
SolverSetupWithReuse(precon_t precon_method, solver_t solver_method, HYPRE_Precon precon,
                     HYPRE_Solver solver, HYPRE_IJMatrix M, HYPRE_IJVector b,
                     HYPRE_IJVector x, Stats *stats, int skip_precon_setup)
{
   if (!solver)
   {
      ErrorCodeSet(ERROR_INVALID_SOLVER);
      ErrorMsgAdd("SolverSetup: solver is NULL");
      return;
   }

   if (!M || !b || !x)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("SolverSetup: matrix or vector is NULL");
      return;
   }

   if (precon_method != PRECON_NONE && !precon)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("SolverSetup: precon is NULL but precon_method is not PRECON_NONE");
      return;
   }

   StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "prec");

   void                   *vM = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix      par_M = NULL;
   HYPRE_ParVector         par_b = NULL, par_x = NULL;
   HYPRE_PtrToParSolverFcn setup_ptrs[] = {HYPRE_BoomerAMGSetup, HYPRE_MGRSetup,
                                           HYPREDRV_ILU_SETUP, HYPREDRV_FSAI_SETUP};
   HYPRE_PtrToParSolverFcn solve_ptrs[] = {HYPRE_BoomerAMGSolve, HYPRE_MGRSolve,
                                           HYPREDRV_ILU_SOLVE, HYPREDRV_FSAI_SOLVE};

   HYPRE_IJMatrixGetObject(M, &vM);
   par_M = (HYPRE_ParCSRMatrix)vM;
   HYPRE_IJVectorGetObject(b, &vb);
   par_b = (HYPRE_ParVector)vb;
   HYPRE_IJVectorGetObject(x, &vx);
   par_x = (HYPRE_ParVector)vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRPCGSetPrecond(solver, solve_ptrs[precon_method],
                                      skip_precon_setup ? HYPREDRV_PreconSetupNoop
                                                        : setup_ptrs[precon_method],
                                      precon->main);
         }
         HYPRE_ParCSRPCGSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_GMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                        skip_precon_setup ? HYPREDRV_PreconSetupNoop
                                                          : setup_ptrs[precon_method],
                                        precon->main);
         }
         HYPRE_ParCSRGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRFlexGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                            skip_precon_setup ? HYPREDRV_PreconSetupNoop
                                                              : setup_ptrs[precon_method],
                                            precon->main);
         }
         HYPRE_ParCSRFlexGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRBiCGSTABSetPrecond(solver, solve_ptrs[precon_method],
                                           skip_precon_setup ? HYPREDRV_PreconSetupNoop
                                                             : setup_ptrs[precon_method],
                                           precon->main);
         }
         HYPRE_ParCSRBiCGSTABSetup(solver, par_M, par_b, par_x);
         break;

      default:
         StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "prec");
         return;
   }

   /* Clear pending error codes from hypre */
   HYPRE_ClearAllErrors();

   StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "prec");
}

void
SolverSetup(precon_t precon_method, solver_t solver_method, HYPRE_Precon precon,
            HYPRE_Solver solver, HYPRE_IJMatrix M, HYPRE_IJVector b, HYPRE_IJVector x,
            Stats *stats)
{
   SolverSetupWithReuse(precon_method, solver_method, precon, solver, M, b, x, stats, 0);
}

/*-----------------------------------------------------------------------------
 * SolverSolveOnly
 *-----------------------------------------------------------------------------*/

HYPRE_Int
SolverSolveOnly(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
                HYPRE_IJVector b, HYPRE_IJVector x)
{
   if (!solver)
   {
      ErrorCodeSet(ERROR_INVALID_SOLVER);
      ErrorMsgAdd("SolverSolveOnly: solver is NULL");
      return -1;
   }

   if (!A || !b || !x)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("SolverSolveOnly: matrix or vector is NULL");
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

      default:
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
SolverApply(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
            HYPRE_IJVector b, HYPRE_IJVector x, Stats *stats)
{
   if (!solver)
   {
      ErrorCodeSet(ERROR_INVALID_SOLVER);
      ErrorMsgAdd("SolverApply: solver is NULL");
      return;
   }

   if (!A || !b || !x)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("SolverApply: matrix or vector is NULL");
      return;
   }

   HYPRE_Int     iters  = 0;
   HYPRE_Complex b_norm = NAN, r_norm = NAN, r0_norm = NAN;

   /* Compute initial residual norm (absolute L2) before timing the solve */
   LinearSystemComputeResidualNorm(A, b, x, "L2", &r0_norm);

   StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   StatsInitialResNormSet(stats, r0_norm);

   iters = SolverSolveOnly(solver_method, solver, A, b, x);

   if (iters < 0)
   {
      StatsIterSet(stats, 0);
      StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");
      return;
   }

   StatsIterSet(stats, (int)iters);
   StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");

   /* Compute the real relative residual norm. Note this is not timed */
   LinearSystemComputeVectorNorm(b, "L2", &b_norm);
   LinearSystemComputeResidualNorm(A, b, x, "L2", &r_norm);
   b_norm = (b_norm > 0.0) ? b_norm : 1.0;

   StatsRelativeResNormSet(stats, r_norm / b_norm);
}

/*-----------------------------------------------------------------------------
 * SolverDestroy
 *-----------------------------------------------------------------------------*/

void
SolverDestroy(solver_t solver_method, HYPRE_Solver *solver_ptr)
{
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

         default:
            return;
      }

      *solver_ptr = NULL;
   }
}
