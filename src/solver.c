/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "solver.h"

/*-----------------------------------------------------------------------------
 * SolverSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
SolverSetDefaultArgs(solver_t solver_method, solver_args *args)
{
   switch (solver_method)
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
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
SolverSetArgsFromYAML(solver_t solver_method, solver_args *args, YAMLnode *node)
{
   switch (solver_method)
   {
      case SOLVER_PCG:
         PCGSetArgsFromYAML(&args->pcg, node);
         break;

      case SOLVER_GMRES:
         GMRESSetArgsFromYAML(&args->gmres, node);
         break;

      case SOLVER_FGMRES:
         FGMRESSetArgsFromYAML(&args->fgmres, node);
         break;

      case SOLVER_BICGSTAB:
         BiCGSTABSetArgsFromYAML(&args->bicgstab, node);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

int
SolverCreate(MPI_Comm comm, solver_t solver_method, solver_args *args, HYPRE_Solver *solver_ptr)
{
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
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverSetup
 *-----------------------------------------------------------------------------*/

int
SolverSetup(precon_t precon_method, solver_t solver_method,
            HYPRE_Solver precon, HYPRE_Solver solver,
            HYPRE_IJMatrix M, HYPRE_IJVector b, HYPRE_IJVector x)
{
   void                    *vM, *vb, *vx;
   HYPRE_ParCSRMatrix       par_M;
   HYPRE_ParVector          par_b, par_x;
   HYPRE_PtrToParSolverFcn  setup_ptrs[] = {HYPRE_BoomerAMGSetup,
                                            HYPRE_MGRSetup,
                                            HYPRE_ILUSetup};
   HYPRE_PtrToParSolverFcn  solve_ptrs[] = {HYPRE_BoomerAMGSolve,
                                            HYPRE_MGRSolve,
                                            HYPRE_ILUSolve};

   HYPRE_IJMatrixGetObject(M, &vM); par_M = (HYPRE_ParCSRMatrix) vM;
   HYPRE_IJVectorGetObject(b, &vb); par_b = (HYPRE_ParVector) vb;
   HYPRE_IJVectorGetObject(x, &vx); par_x = (HYPRE_ParVector) vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSetPrecond(solver,
                                   solve_ptrs[precon_method],
                                   setup_ptrs[precon_method],
                                   precon);
         HYPRE_ParCSRPCGSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSetPrecond(solver,
                                     solve_ptrs[precon_method],
                                     setup_ptrs[precon_method],
                                     precon);
         HYPRE_ParCSRGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSetPrecond(solver,
                                         solve_ptrs[precon_method],
                                         setup_ptrs[precon_method],
                                         precon);
         HYPRE_ParCSRFlexGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                        solve_ptrs[precon_method],
                                        setup_ptrs[precon_method],
                                        precon);
         HYPRE_ParCSRBiCGSTABSetup(solver, par_M, par_b, par_x);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverApply
 *-----------------------------------------------------------------------------*/

int
SolverApply(solver_t solver_method, HYPRE_Solver solver,
            HYPRE_IJMatrix A, HYPRE_IJVector b, HYPRE_IJVector x)
{
   void               *vA, *vb, *vx;
   HYPRE_ParCSRMatrix  par_A;
   HYPRE_ParVector     par_b, par_x;

   HYPRE_IJMatrixGetObject(A, &vA); par_A = (HYPRE_ParCSRMatrix) vA;
   HYPRE_IJVectorGetObject(b, &vb); par_b = (HYPRE_ParVector) vb;
   HYPRE_IJVectorGetObject(x, &vx); par_x = (HYPRE_ParVector) vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSolve(solver, par_A, par_b, par_x);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverDestroy
 *-----------------------------------------------------------------------------*/

int
SolverDestroy(solver_t solver_method, HYPRE_Solver *solver_ptr)
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
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   *solver_ptr = NULL;

   return EXIT_SUCCESS;
}
