/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include "hypredrive.h"

void
PrintUsage(const char *argv0)
{
   fprintf(stderr, "Usage: %s <filename>\n", argv0);
   fprintf(stderr, "  filename: config file in YAML format\n");
}

int main(int argc, char **argv)
{
   MPI_Comm      comm = MPI_COMM_WORLD;
   int           myid, nprocs, i, k;
   HYPREDRV_t    obj;

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);
   HYPREDRV_Initialize();
   HYPREDRV_Create(comm, &obj);

   if (argc < 1)
   {
      if (!myid) PrintUsage(argv[0]);
      return EXIT_FAILURE;
   }

   /*-----------------------------------------------------------
    * Print libraries/driver info
    *-----------------------------------------------------------*/

   if (!myid) HYPREDRV_PrintLibInfo();
   if (!myid) printf("Running on %d MPI rank%s\n", nprocs, nprocs > 1 ? "s" : "");

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   HYPREDRV_InputArgsParse(argc, argv, obj);

   /*-----------------------------------------------------------
    * Set hypre's global options and warmup
    *-----------------------------------------------------------*/

   HYPREDRV_SetGlobalOptions(obj);
   if (HYPREDRV_InputArgsGetWarmup(obj))
   {
      printf("TODO: Perform warmup");
   }

   /*-----------------------------------------------------------
    * Build and solve linear system(s)
    *-----------------------------------------------------------*/

   for (k = 0; k < HYPREDRV_InputArgsGetNumLinearSystems(obj); k++)
   {
      /* Build linear system (matrix, RHS, LHS, and auxiliary data) */
      HYPREDRV_LinearSystemBuild(obj);

      for (i = 0; i < HYPREDRV_InputArgsGetNumRepetitions(obj); i++)
      {
         /* Reset initial guess */
         HYPREDRV_LinearSystemResetInitialGuess(obj);

         /* Create phase */
         HYPREDRV_PreconCreate(obj);
         HYPREDRV_LinearSolverCreate(obj);

         /* Setup phase */
         HYPREDRV_LinearSolverSetup(obj);

         /* Solve phase */
         HYPREDRV_LinearSolverApply(obj);

         /* Destroy phase */
         HYPREDRV_PreconDestroy(obj);
         HYPREDRV_LinearSolverDestroy(obj);
      }
   }

   /*-----------------------------------------------------------
    * Finalize driver
    *-----------------------------------------------------------*/

   if (!myid) HYPREDRV_StatsPrint(obj);
   if (!myid) HYPREDRV_PrintExitInfo(argv[0]);

   HYPREDRV_Destroy(&obj);
   HYPREDRV_Finalize();
   MPI_Finalize();

   return EXIT_SUCCESS;
}
