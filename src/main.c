/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "HYPREDRV.h"

static bool
LooksLikeYAMLFilename(const char *str)
{
   if (!str || *str == '\0')
   {
      return false;
   }

   /* Filenames should not contain spaces */
   if (strchr(str, ' ') != NULL)
   {
      return false;
   }

   const char *dot = strrchr(str, '.');
   if (!dot || dot == str)
   {
      return false;
   }

   const char *ext = dot + 1;
   return (strcmp(ext, "yaml") == 0 || strcmp(ext, "yml") == 0) != 0;
}

static void
PrintUsage(const char *argv0)
{
   fprintf(stdout, "Usage: %s [options] <filename>\n", argv0);
   fprintf(stdout, "  filename: config file in YAML format\n");
   fprintf(stdout, "\nOptions:\n");
   fprintf(stdout, "  -h, --help         Show this help message\n");
   fprintf(stdout, "  -q, --quiet        Skip system information printout\n");
   fprintf(stdout, "  -a, --args         Override YAML parameters from the CLI\n");
   fprintf(stdout, "  -p, --prec-preset  Override preconditioner with a preset\n");
   fprintf(stdout, "\nOverride syntax (after -a/--args):\n");
   fprintf(stdout, "  --path:to:key  <value>\n");
   fprintf(stdout, "Examples:\n");
   fprintf(stdout, "  %s input.yml -a --solver:pcg:print_level 1\n", argv0);
   fprintf(stdout, "  %s input.yml -a --preconditioner:amg:print_level 2\n", argv0);
   fprintf(stdout, "  %s input.yml -p poisson\n", argv0);
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

static const char *
FindPreconPreset(int argc, char **argv)
{
   for (int i = 1; i < argc - 1; i++)
   {
      if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prec-preset") == 0)
      {
         return argv[i + 1];
      }
   }
   return NULL;
}

static char *
FindConfigFile(int argc, char **argv)
{
   /* Find the first arg that looks like a YAML filename.
    * This avoids mis-detecting override values as the config file. */
   for (int i = 1; i < argc; i++)
   {
      if (LooksLikeYAMLFilename(argv[i]))
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
   HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   if (!quiet_mode)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   /*-----------------------------------------------------------
    * Parse input parameters
    *-----------------------------------------------------------*/

   RequireConfigArgumentOrAbort(argc, argv, comm, myid);
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc, argv, obj));

   /* If --precon-preset was given, override the preconditioner */
   const char *preset_name = FindPreconPreset(argc, argv);
   if (preset_name)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(obj, preset_name));
   }

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

   int num_linear_systems  = HYPREDRV_InputArgsGetNumLinearSystems(obj);
   int num_precon_variants = HYPREDRV_InputArgsGetNumPreconVariants(obj);

   for (int k = 0; k < num_linear_systems; k++)
   {
      /* Build linear system (matrix, RHS, LHS, and auxiliary data) */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(obj));

      /* Optionally compute full eigenspectrum */
#ifdef HYPREDRV_ENABLE_EIGSPEC
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemComputeEigenspectrum(obj));
#endif

      /* Loop over preconditioner variants */
      for (int v = 0; v < num_precon_variants; v++)
      {
         /* Set active variant */
         HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconVariant(obj, v));

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
