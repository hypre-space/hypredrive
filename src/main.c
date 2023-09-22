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

#define ANNOTATE_REGION_BEGIN()
#define ANNOTATE_REGION_END()
#define ANNOTATE_ITER_BEGIN(i)
#define ANNOTATE_ITER_END(i)

int main(int argc, char **argv)
{
   MPI_Comm         comm = MPI_COMM_WORLD;
   int              myid;
   input_args      *iargs;
   HYPRE_IJMatrix   mat_A;
   HYPRE_IJMatrix   mat_M;
   HYPRE_IJVector   rhs;
   HYPRE_IJVector   sol;
   HYPRE_IntArray   dofmap;
   HYPRE_Solver     precon;
   HYPRE_Solver     solver;

   HYPRE_Int        i;

   if (argc < 1)
   {
      fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
      fprintf(stderr, "  filename: config file in YAML format\n");
      return EXIT_FAILURE;
   }

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   HYPRE_Initialize();

   /*-----------------------------------------------------------
    * Print driver info
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      PrintInfo();
   }

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   InputArgsParse(comm, argc, argv, &iargs);

   /*-----------------------------------------------------------
    * Build and solve linear system(s)
    *-----------------------------------------------------------*/

   if (iargs->warmup)
   {
      printf("TODO: Perform warmup");
   }

   /* Build linear system */
   ANNOTATE_REGION_BEGIN();
   LinearSystemReadMatrix(comm, &iargs->ls, &mat_A);
   LinearSystemSetRHS(comm, &iargs->ls, mat_A, &rhs);
   LinearSystemSetInitialGuess(comm, &iargs->ls, mat_A, rhs, &sol);
   LinearSystemSetPrecMatrix(comm, &iargs->ls, mat_A, &mat_M);
   LinearSystemReadDofmap(comm, &iargs->ls, &dofmap);
   ANNOTATE_REGION_END();

   /* Solve linear system */
   for (i = 0; i < iargs->num_repetitions; i++)
   {
      /* Setup phase */
      ANNOTATE_ITER_BEGIN(i);
      PreconCreate(iargs->precon_method, &iargs->precon, &dofmap, &precon);
      SolverCreate(comm, iargs->solver_method, &iargs->solver, &solver);
      SolverSetup(iargs->precon_method, iargs->solver_method, precon, solver, mat_M, rhs, sol);
      ANNOTATE_ITER_END(i);

      /* Solve phase */
      ANNOTATE_ITER_BEGIN(i);
      SolverApply(iargs->solver_method, solver, mat_A, rhs, sol);
      ANNOTATE_ITER_END(i);

      /* Destroy phase */
      PreconDestroy(iargs->precon_method, &precon);
      SolverDestroy(iargs->solver_method, &solver);
   }

   /*-----------------------------------------------------------
    * Finalize driver
    *-----------------------------------------------------------*/

   InputArgsDestroy(&iargs);
   if (mat_A != mat_M)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_IJMatrixDestroy(mat_A);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(sol);

   HYPRE_Finalize();
   MPI_Finalize();

   if (!myid)
   {
      PrintExitInfo(argv[0]);
   }

   return EXIT_SUCCESS;
}
