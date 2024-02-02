/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "args.h"
#include "linsys.h"
#include "info.h"

int main(int argc, char **argv)
{
   MPI_Comm         comm = MPI_COMM_WORLD;
   int              myid;
   IntArray        *dofmap;
   input_args      *iargs;
   HYPRE_IJMatrix   mat_A;
   HYPRE_IJMatrix   mat_M;
   HYPRE_IJVector   rhs;
   HYPRE_IJVector   sol;
   HYPRE_IJVector   sol0;
   HYPRE_Solver     precon;
   HYPRE_Solver     solver;

   HYPRE_Int        i;

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();

   if (argc < 1)
   {
      if (!myid) PrintUsage(argv[0]);
      return EXIT_FAILURE;
   }

   /*-----------------------------------------------------------
    * Print libraries/driver info
    *-----------------------------------------------------------*/

   if (!myid) PrintLibInfo();

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   InputArgsParse(comm, argc, argv, &iargs);

   /*-----------------------------------------------------------
    * Set hypre's global options
    *-----------------------------------------------------------*/

   if (iargs->ls.exec_policy)
   {
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
      HYPRE_SetSpGemmUseVendor(0);
   }
   else
   {
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
   }

   /*-----------------------------------------------------------
    * Build and solve linear system(s)
    *-----------------------------------------------------------*/

   if (iargs->warmup)
   {
      printf("TODO: Perform warmup");
   }

   /* Build linear system */
   LinearSystemReadMatrix(comm, &iargs->ls, &mat_A);
   LinearSystemSetRHS(comm, &iargs->ls, mat_A, &rhs);
   LinearSystemSetInitialGuess(comm, &iargs->ls, mat_A, rhs, &sol0, &sol);
   LinearSystemSetPrecMatrix(comm, &iargs->ls, mat_A, &mat_M);
   LinearSystemReadDofmap(comm, &iargs->ls, &dofmap);

   /* Solve linear system */
   for (i = 0; i < iargs->num_repetitions; i++)
   {
      /* Reset initial guess */
      LinearSystemResetInitialGuess(sol0, sol);

      /* Setup phase */
      PreconCreate(iargs->precon_method, &iargs->precon, dofmap, &precon);
      SolverCreate(comm, iargs->solver_method, &iargs->solver, &solver);
      SolverSetup(iargs->precon_method, iargs->solver_method, precon, solver, mat_M, rhs, sol);

      /* Solve phase */
      SolverApply(iargs->solver_method, solver, mat_A, rhs, sol);

      /* Destroy phase */
      PreconDestroy(iargs->precon_method, &precon);
      SolverDestroy(iargs->solver_method, &solver);
   }

   /*-----------------------------------------------------------
    * Finalize driver
    *-----------------------------------------------------------*/

   if (mat_A != mat_M)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_IJMatrixDestroy(mat_A);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(sol);
   HYPRE_IJVectorDestroy(sol0);

   HYPRE_Finalize();
   MPI_Finalize();

   if (!myid) StatsPrint(iargs->statistics);
   if (!myid) PrintExitInfo(argv[0]);
   InputArgsDestroy(&iargs);

   return EXIT_SUCCESS;
}
