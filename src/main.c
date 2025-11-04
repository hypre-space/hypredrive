/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include "HYPREDRV.h"
#include "HYPREDRV_config.h"

void
PrintUsage(const char *argv0)
{
   fprintf(stderr, "Usage: %s <filename>\n", argv0);
   fprintf(stderr, "  filename: config file in YAML format\n");
   fflush(stderr);
}

int
main(int argc, char **argv)
{
   MPI_Comm   comm = MPI_COMM_WORLD;
   int        myid = 0, i = 0, k = 0;
   HYPREDRV_t obj = NULL;

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &obj));

   if (argc < 1)
   {
      if (!myid)
      {
         PrintUsage(argv[0]);
      }
      MPI_Abort(comm, 1);
   }

   /*-----------------------------------------------------------
    * Print libraries/driver info
    *-----------------------------------------------------------*/

   HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
   HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   if (argc < 1)
   {
      if (!myid)
      {
         fprintf(stderr, "Need at least one input argument!\n");
      }
      MPI_Abort(comm, 1);
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc - 1, argv + 1, obj));

   /*-----------------------------------------------------------
    * Set hypre's global options and warmup
    *-----------------------------------------------------------*/

   HYPREDRV_SAFE_CALL(HYPREDRV_SetGlobalOptions(obj));
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
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(obj));

      /* Optionally compute full eigenspectrum */
#ifdef HYPREDRV_ENABLE_EIGSPEC
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemComputeEigenspectrum(obj));
#endif

      for (i = 0; i < HYPREDRV_InputArgsGetNumRepetitions(obj); i++)
      {
         /* Reset initial guess */
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemResetInitialGuess(obj));

         /* Create phase */
         HYPREDRV_SAFE_CALL(HYPREDRV_PreconCreate(obj));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverCreate(obj));

         /* Setup phase */
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverSetup(obj));

         /* Solve phase */
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverApply(obj));

         /* Destroy phase */
         HYPREDRV_SAFE_CALL(HYPREDRV_PreconDestroy(obj));
         HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(obj));
      }
   }

   /*-----------------------------------------------------------
    * Finalize driver
    *-----------------------------------------------------------*/

   if (!myid)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(obj));
   }
   HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&obj));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
