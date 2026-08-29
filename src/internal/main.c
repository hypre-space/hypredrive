/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include "HYPREDRV.h"
#include "HYPREDRV_utils.h"
#include "internal/help.h"
#include "internal/info.h"
#include "internal/utils.h"
#include "object.h"

static void
PrintUsage(const char *argv0)
{
   fprintf(stdout, "Usage: %s [options] <filename> [filename ...]\n", argv0);
   fprintf(stdout, "  filename: config file in YAML format\n");
   fprintf(stdout, "\nOptions:\n");
   fprintf(stdout, "  -h, --help         Show this help message\n");
   fprintf(stdout, "  -i, --info         Show system information\n");
   fprintf(stdout, "  -a, --args         Override YAML parameters from the CLI\n");
   fprintf(stdout, "  -p, --prec-preset  Override preconditioner with a preset\n");
   fprintf(stdout, "\nOverride syntax (after -a/--args):\n");
   fprintf(stdout, "  [--]path:to:key  <value>\n");
   fprintf(stdout, "Examples:\n");
   fprintf(stdout, "  %s input.yml -a --solver:pcg:print_level 1\n", argv0);
   fprintf(stdout, "  %s input.yml -a solver:pcg:print_level 1\n", argv0);
   fprintf(stdout, "  %s input.yml -a --preconditioner:amg:print_level 2\n", argv0);
   fprintf(stdout, "  %s input.yml -p poisson\n", argv0);
   fprintf(stdout, "  %s input1.yml input2.yml -q\n", argv0);
   fflush(stdout);
}

// clang-format off
static void
PrintBanner(void)
{
   static const int colors = 0;

   const char *hypre_lines[] = {
      "██╗  ██╗██╗   ██╗██████╗ ██████╗ ███████╗",
      "██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔════╝",
      "███████║ ╚████╔╝ ██████╔╝██████╔╝█████╗" "  ",
      "██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══██╗██╔══╝" "  ",
      "██║  ██║   ██║   ██║     ██║  ██║███████╗",
      "╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚══════╝",
   };
   const char *drive_lines[] = {
      "██████╗ ██████╗ ██╗██╗   ██╗███████╗",
      "██╔══██╗██╔══██╗██║██║   ██║██╔════╝",
      "██║  ██║██████╔╝██║██║   ██║█████╗" "  ",
      "██║  ██║██╔══██╗██║╚██╗ ██╔╝██╔══╝" "  ",
      "██████╔╝██║  ██║██║ ╚████╔╝ ███████╗",
      "╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝",
   };
   const int num_banner_lines = (int)(sizeof(hypre_lines) / sizeof(hypre_lines[0]));
   const int banner_width     = 87 - ((colors) ? 0 : 7);

   const char *H = colors ? "\033[1;38;2;175;36;41m" : "";
   const char *D = colors ? "\033[1;38;2;0;159;224m" : "";
   const char *M = colors ? "\033[1;30m" : "";
   const char *W = colors ? "\033[0m" : "";

   printf("\n");
   printf("%s  ┌", M);
   for (int i = 0; i < banner_width; i++)
   {
      printf("─");
   }
   printf("┐%s\n", W);

   for (int i = 0; i < num_banner_lines; i++)
   {
      printf("%s  │ %s%s%s %s%s%s │%s\n", M, H, hypre_lines[i], M, D, drive_lines[i], M, W);
   }

   printf("%s  └", M);
   for (int i = 0; i < banner_width; i++)
   {
      printf("─");
   }
   printf("┘%s\n", W);
   printf("\n");
}
// clang-format on

static int
InfoModeRequested(int argc, char **argv)
{
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--info") == 0)
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

static int
IsConfigFileArgument(const char *arg)
{
   return arg && hypredrv_IsYAMLFilename(arg);
}

static int
CountConfigFiles(int argc, char **argv)
{
   int count = 0;
   for (int i = 1; i < argc; i++)
   {
      if (IsConfigFileArgument(argv[i]))
      {
         count++;
      }
   }
   return count;
}

static int
BuildConfigArgv(int argc, char **argv, int config_index, char **config_argv)
{
   int config_argc            = 0;
   config_argv[config_argc++] = argv[0];

   for (int i = 1; i < argc; i++)
   {
      if (IsConfigFileArgument(argv[i]))
      {
         if (i == config_index)
         {
            config_argv[config_argc++] = argv[i];
         }
      }
      else
      {
         config_argv[config_argc++] = argv[i];
      }
   }

   config_argv[config_argc] = NULL;
   return config_argc;
}

static int
RequireConfigArgument(int argc, char **argv, int myid)
{
   if (CountConfigFiles(argc, argv) == 0)
   {
      if (!myid)
      {
         PrintUsage(argv[0]);
      }
      return 0;
   }

   return 1;
}

static void
RunSolveLoops(HYPREDRV_t obj)
{
   int num_linear_systems  = 0;
   int num_precon_variants = 0;
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumLinearSystems(obj, &num_linear_systems));
   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumPreconVariants(obj, &num_precon_variants));

   for (int k = 0; k < num_linear_systems; k++)
   {
      /* Build linear system (matrix, RHS, LHS, and auxiliary data) */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemBuild(obj));

#ifdef HYPREDRV_ENABLE_EIGSPEC
      /* Optionally compute full eigenspectrum (no-op if not built with eigspec) */
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemComputeEigenspectrum(obj));
#endif

      /* Loop over preconditioner variants */
      for (int v = 0; v < num_precon_variants; v++)
      {
         /* Set active variant */
         HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconVariant(obj, v));

         int num_reps = 0;
         HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsGetNumRepetitions(obj, &num_reps));
         for (int i = 0; i < num_reps; i++)
         {
            /* (Optional) Annotate the entire solve iteration. The id is appended by
             * the annotation API, so the name is a plain literal (not a format). */
            HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateBegin(obj, "Run", i));

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
            HYPREDRV_SAFE_CALL(HYPREDRV_AnnotateEnd(obj, "Run", i));
         }
      }
   }
}

static void
RunOneConfig(MPI_Comm comm, int myid, int argc, char **argv, int print_lib_info,
             int print_system_info)
{
   HYPREDRV_t obj = NULL;

   HYPREDRV_SAFE_CALL(HYPREDRV_Create(comm, &obj));
   if (print_lib_info)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));
   }
   if (print_system_info)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintSystemInfo(comm));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsParse(argc, argv, obj));

   const char *preset_name = FindPreconPreset(argc, argv);
   if (preset_name)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_InputArgsSetPreconPreset(obj, preset_name));
   }

   /* User-facing execution-policy output belongs to the command-line driver;
    * library API calls only emit the policy through the internal log. */
   hypredrv_PrintExecutionPolicy(comm, obj->iargs->general.exec_policy, stdout);

   RunSolveLoops(obj);

   /*-----------------------------------------------------------
    * Finalize this solve case
    *-----------------------------------------------------------*/

   if (!myid)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_StatsPrint(obj));
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Destroy(&obj));
}

int
main(int argc, char **argv)
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int      myid = 0;
   char     help_topic[512];

   /*-----------------------------------------------------------
    * Initialize driver
    *-----------------------------------------------------------*/

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   if (!myid)
   {
      PrintBanner();
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());
   if (hypredrv_HelpRequested(argc, argv, help_topic, sizeof(help_topic)))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_PrintLibInfo(comm, 1));

      int help_status = 0;
      if (!myid)
      {
         help_status = hypredrv_HelpPrint(stdout, argv[0], help_topic);
      }
      HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
      MPI_Finalize();
      return help_status;
   }
   if (!RequireConfigArgument(argc, argv, myid))
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
      MPI_Finalize();
      return 1;
   }

   const int config_count = CountConfigFiles(argc, argv);
   const int info_mode    = InfoModeRequested(argc, argv);
   int       case_number  = 0;

   for (int i = 1; i < argc; i++)
   {
      if (!IsConfigFileArgument(argv[i]))
      {
         continue;
      }

      char *config_argv[argc + 1];
      int   config_argc = BuildConfigArgv(argc, argv, i, config_argv);
      case_number++;

      if (config_count > 1 && !myid)
      {
         printf("\n=== hypredrive case %d/%d: %s ===\n", case_number, config_count,
                argv[i]);
      }

      RunOneConfig(comm, myid, config_argc, config_argv, case_number == 1,
                   info_mode && case_number == 1);
   }

   HYPREDRV_SAFE_CALL(HYPREDRV_PrintExitInfo(comm, argv[0]));
   HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());
   MPI_Finalize();

   return 0;
}
