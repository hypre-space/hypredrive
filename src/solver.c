/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "solver.h"

#include <math.h>
#include "gen_macros.h"

static const FieldOffsetMap solver_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(solver_args, pcg, PCGSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, gmres, GMRESSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, fgmres, FGMRESSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, bicgstab, BiCGSTABSetArgs),
};

#define SOLVER_NUM_FIELDS \
   (sizeof(solver_field_offset_map) / sizeof(solver_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * SolverSetFieldByName
 *-----------------------------------------------------------------------------*/

void
SolverSetFieldByName(solver_args *args, const YAMLnode *node)
{
   for (size_t i = 0; i < SOLVER_NUM_FIELDS; i++)
   {
      /* Which union type are we trying to set? */
      if (!strcmp(solver_field_offset_map[i].name, node->key))
      {
         solver_field_offset_map[i].setter(
            (void *)((char *)args + solver_field_offset_map[i].offset), node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * SolverGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
SolverGetValidKeys(void)
{
   static const char *keys[SOLVER_NUM_FIELDS];

   for (size_t i = 0; i < SOLVER_NUM_FIELDS; i++)
   {
      keys[i] = solver_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

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
SolverSetup(precon_t precon_method, solver_t solver_method, HYPRE_Precon precon,
            HYPRE_Solver solver, HYPRE_IJMatrix M, HYPRE_IJVector b, HYPRE_IJVector x)
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

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "prec");

   void                   *vM = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix      par_M = NULL;
   HYPRE_ParVector         par_b = NULL, par_x = NULL;
   HYPRE_PtrToParSolverFcn setup_ptrs[] = {HYPRE_BoomerAMGSetup, HYPRE_MGRSetup,
                                           HYPRE_ILUSetup, HYPRE_FSAISetup};
   HYPRE_PtrToParSolverFcn solve_ptrs[] = {HYPRE_BoomerAMGSolve, HYPRE_MGRSolve,
                                           HYPRE_ILUSolve, HYPRE_FSAISolve};

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
                                      setup_ptrs[precon_method], precon->main);
         }
         HYPRE_ParCSRPCGSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_GMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                        setup_ptrs[precon_method], precon->main);
         }
         HYPRE_ParCSRGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRFlexGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                            setup_ptrs[precon_method], precon->main);
         }
         HYPRE_ParCSRFlexGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         if (precon_method != PRECON_NONE)
         {
            HYPRE_ParCSRBiCGSTABSetPrecond(solver, solve_ptrs[precon_method],
                                           setup_ptrs[precon_method], precon->main);
         }
         HYPRE_ParCSRBiCGSTABSetup(solver, par_M, par_b, par_x);
         break;

      default:
         StatsAnnotate(HYPREDRV_ANNOTATE_END, "prec");
         return;
   }

   /* Clear pending error codes from hypre */
   HYPRE_ClearAllErrors();

   StatsAnnotate(HYPREDRV_ANNOTATE_END, "prec");
}

/*-----------------------------------------------------------------------------
 * SolverApply
 *-----------------------------------------------------------------------------*/

void
SolverApply(solver_t solver_method, HYPRE_Solver solver, HYPRE_IJMatrix A,
            HYPRE_IJVector b, HYPRE_IJVector x)
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

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "solve");

   void              *vA = NULL, *vb = NULL, *vx = NULL;
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
   HYPRE_Int          iters  = 0;
   HYPRE_Complex      b_norm = NAN, r_norm = NAN;

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
         StatsIterSet((int)iters);
         StatsAnnotate(HYPREDRV_ANNOTATE_END, "solve");
         return;
   }

   /* Clear pending error codes from hypre */
   HYPRE_ClearAllErrors();

   StatsIterSet((int)iters);
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "solve");

   /* Compute the real relative residual norm. Note this is not timed */
   LinearSystemComputeVectorNorm(b, "L2", &b_norm);
   LinearSystemComputeResidualNorm(A, b, x, "L2", &r_norm);
   b_norm = (b_norm > 0.0) ? b_norm : 1.0;

   StatsRelativeResNormSet(r_norm / b_norm);
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
