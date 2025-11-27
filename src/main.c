/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include "HYPREDRV.h"
#include "HYPREDRV_config.h"

static void
PrintUsage(const char *argv0)
{
   fprintf(stdout, "Usage: %s [options] <filename>\n", argv0);
   fprintf(stdout, "  filename: config file in YAML format\n");
   fprintf(stdout, "\nOptions:\n");
   fprintf(stdout, "  -h, --help       Show this help message\n");
   fprintf(stdout, "  -q, --quiet      Skip system information printout\n");
   fflush(stdout);
}

static int
HelpRequested(int argc, char **argv)
{
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         return 1;
      }
   }
   return 0;
}

static int
QuietModeRequested(int argc, char **argv)
{
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0)
      {
         return 1;
      }
   }
   return 0;
}

static char *
FindConfigFile(int argc, char **argv)
{
   /* Find the first non-option argument (config file) */
   for (int i = 1; i < argc; i++)
   {
      if (argv[i][0] != '-')
      {
         return argv[i];
      }
   }
   return NULL;
}

static void
RequireConfigArgumentOrAbort(int argc, char **argv, MPI_Comm comm, int myid)
{
   if (FindConfigFile(argc, argv) == NULL)
   {
      if (!myid)
      {
         PrintUsage(argv[0]);
      }
      MPI_Abort(comm, 1);
   }
}

int
main(int argc, char **argv)
{
   MPI_Comm   comm = MPI_COMM_WORLD;
   int        myid = 0;
   HYPREDRV_t obj  = NULL;

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
   if (HelpRequested(argc, argv))
   {
      if (!myid)
      {
         PrintUsage(argv[0]);
      }
      HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
      MPI_Abort(comm, 0);
   }
   RequireConfigArgumentOrAbort(argc, argv, comm, myid);

   /*-----------------------------------------------------------
    * Create driver object and print libraries/driver info
    *-----------------------------------------------------------*/

   int quiet_mode = QuietModeRequested(argc, argv);

   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &obj));
   HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm));
   if (!quiet_mode)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   RequireConfigArgumentOrAbort(argc, argv, comm, myid);
   char *config_file     = FindConfigFile(argc, argv);
   char *config_argv[2]  = {config_file, NULL};
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(1, config_argv, obj));

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

   for (int k = 0; k < HYPREDRV_InputArgsGetNumLinearSystems(obj); k++)
   {
      /* Build linear system (matrix, RHS, LHS, and auxiliary data) */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(obj));

      /* Optionally compute full eigenspectrum */
#ifdef HYPREDRV_ENABLE_EIGSPEC
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemComputeEigenspectrum(obj));
#endif

      for (int i = 0; i < HYPREDRV_InputArgsGetNumRepetitions(obj); i++)
      {
         /* (Optional) Annotate the entire solve iteration */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin("Run-%d", i));

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

         /* (Optional) Annotate the entire solve iteration */
         HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd("Run-%d", i));
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
