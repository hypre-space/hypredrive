/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>
#include "HYPREDRV.h"

void
PrintUsage(const char *argv0)
{
   fprintf(stderr, "Usage: %s <filename>\n", argv0);
   fprintf(stderr, "  filename: config file in YAML format\n");
}

int main(int argc, char **argv)
{
   MPI_Comm      comm = MPI_COMM_WORLD;
   int           myid, i;
   HYPREDRV_t    obj;

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   HYPRE_Initialize();
   HYPRE_DeviceInitialize();
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

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   HYPREDRV_InputArgsParse(argc, argv, obj);

   /*-----------------------------------------------------------
    * Set hypre's global options
    *-----------------------------------------------------------*/

   if (HYPREDRV_InputArgsGetExecPolicy(obj))
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

   if (HYPREDRV_InputArgsGetWarmup(obj))
   {
      printf("TODO: Perform warmup");
   }

   /* Build linear system */
   HYPREDRV_LinearSystemReadMatrix(obj);
   HYPREDRV_LinearSystemSetRHS(obj);
   HYPREDRV_LinearSystemSetInitialGuess(obj);
   HYPREDRV_LinearSystemSetPrecMatrix(obj);
   HYPREDRV_LinearSystemReadDofmap(obj);

   /* Solve linear system */
   for (i = 0; i < HYPREDRV_InputArgsGetNumRepetitions(obj); i++)
   {
      /* Reset initial guess */
      HYPREDRV_LinearSystemResetInitialGuess(obj);

      /* Setup phase */
      HYPREDRV_PreconCreate(obj);
      HYPREDRV_LinearSolverCreate(obj);
      HYPREDRV_LinearSolverSetup(obj);

      /* Solve phase */
      HYPREDRV_LinearSolverApply(obj);

      /* Destroy phase */
      HYPREDRV_PreconDestroy(obj);
      HYPREDRV_LinearSolverDestroy(obj);
   }

   /*-----------------------------------------------------------
    * Finalize driver
    *-----------------------------------------------------------*/

   if (!myid) HYPREDRV_StatsPrint(obj);
   if (!myid) HYPREDRV_PrintExitInfo(argv[0]);

   HYPREDRV_Destroy(&obj);
   HYPRE_Finalize();
   MPI_Finalize();

   return EXIT_SUCCESS;
}
