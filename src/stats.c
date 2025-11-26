/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "stats.h"
#include <mpi.h>
#include <stdarg.h>
#include <string.h>

/* Global timings variable */
static Stats *global_stats = NULL;

/* Reallocation expansion factor */
#define REALLOC_EXPAND_FACTOR 16

/*--------------------------------------------------------------------------
 * Helper: Reallocate array with expansion
 *--------------------------------------------------------------------------*/

static void
ReallocateArray(void **ptr, size_t elem_size, int old_capacity, int new_capacity)
{
   void *new_ptr = realloc(*ptr, new_capacity * elem_size);
   if (new_ptr)
   {
      /* Zero out the newly allocated portion */
      memset((char *)new_ptr + (old_capacity * elem_size), 0,
             REALLOC_EXPAND_FACTOR * elem_size);
      *ptr = new_ptr;
   }
}

/*--------------------------------------------------------------------------
 * Helper: Ensure capacity for current counter
 *--------------------------------------------------------------------------*/

static void
EnsureCapacity(void)
{
   /* Need capacity > counter to have space for counter index */
   if (global_stats->counter + 1 >= global_stats->capacity)
   {
      int old_capacity = global_stats->capacity;
      int new_capacity = old_capacity + REALLOC_EXPAND_FACTOR;

      /* Reallocate all arrays with the same new capacity */
      ReallocateArray((void **)&global_stats->matrix, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->rhs, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->dofmap, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->prec, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->solve, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->rrnorms, sizeof(double), old_capacity, new_capacity);
      ReallocateArray((void **)&global_stats->iters, sizeof(int), old_capacity, new_capacity);

      /* Update capacity after all reallocations */
      global_stats->capacity = new_capacity;
   }
}

/*--------------------------------------------------------------------------
 * Helper: Start timer for vector entry
 *--------------------------------------------------------------------------*/

static void
StartVectorTimer(double *timer_array, int index)
{
   if (timer_array && index >= 0 && index < global_stats->capacity)
   {
      timer_array[index] -= MPI_Wtime();
   }
}

/*--------------------------------------------------------------------------
 * Helper: Stop timer for vector entry
 *--------------------------------------------------------------------------*/

static void
StopVectorTimer(double *timer_array, int index)
{
   if (timer_array && index >= 0 && index < global_stats->capacity)
   {
      timer_array[index] += MPI_Wtime();
   }
}

/*--------------------------------------------------------------------------
 * Helper: Start timer for scalar entry
 *--------------------------------------------------------------------------*/

static void
StartScalarTimer(double *timer)
{
   if (timer)
   {
      *timer -= MPI_Wtime();
   }
}

/*--------------------------------------------------------------------------
 * Helper: Stop timer for scalar entry
 *--------------------------------------------------------------------------*/

static void
StopScalarTimer(double *timer)
{
   if (timer)
   {
      *timer += MPI_Wtime();
   }
}

/*--------------------------------------------------------------------------
 * Helper: Handle annotation begin
 *--------------------------------------------------------------------------*/

static void
HandleAnnotationBegin(const char *name)
{
   /* Ignore "Run" annotation */
   if (strncmp(name, "Run", 3) == 0)
   {
      return;
   }
   EnsureCapacity();

   /* Update counters for special annotations */
   if (!strcmp(name, "reset_x0"))
   {
      global_stats->reps++;
      global_stats->counter =
         ((global_stats->ls_counter) * global_stats->num_reps) +
         (global_stats->reps - 1);
   }
   else if (!strcmp(name, "matrix") || !strcmp(name, "system"))
   {
      StartVectorTimer(global_stats->matrix, global_stats->counter);
   }
   else if (!strcmp(name, "rhs"))
   {
      StartVectorTimer(global_stats->rhs, global_stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      StartVectorTimer(global_stats->dofmap, global_stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      StartVectorTimer(global_stats->prec, global_stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      /* Increment linear system counter */
      global_stats->reps = 0;
      global_stats->ls_counter++;
      global_stats->counter = (global_stats->ls_counter - 1) * global_stats->num_reps;
      StartVectorTimer(global_stats->solve, global_stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      StartScalarTimer(&global_stats->reset_x0);
   }
   else if (!strcmp(name, "initialize"))
   {
      StartScalarTimer(&global_stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      StartScalarTimer(&global_stats->finalize);
   }
   else
   {
      /* Unknown annotation - set error but don't fail */
      ErrorCodeSet(ERROR_UNKNOWN_TIMING);
      ErrorMsgAdd("Unknown timer key: '%s'", name);
   }
}

/*--------------------------------------------------------------------------
 * Helper: Handle annotation end
 *--------------------------------------------------------------------------*/

static void
HandleAnnotationEnd(const char *name)
{
   /* Ignore "Run" annotation */
   if (strncmp(name, "Run", 3) == 0)
   {
      return;
   }

   /* Stop timers based on annotation name */
   if (!strcmp(name, "matrix") || !strcmp(name, "system"))
   {
      StopVectorTimer(global_stats->matrix, global_stats->counter);
   }
   else if (!strcmp(name, "rhs"))
   {
      StopVectorTimer(global_stats->rhs, global_stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      StopVectorTimer(global_stats->dofmap, global_stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      StopVectorTimer(global_stats->prec, global_stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      StopVectorTimer(global_stats->solve, global_stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      StopScalarTimer(&global_stats->reset_x0);
   }
   else if (!strcmp(name, "initialize"))
   {
      StopScalarTimer(&global_stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      StopScalarTimer(&global_stats->finalize);
   }
   else
   {
      /* Unknown annotation - set error but don't fail */
      ErrorCodeSet(ERROR_UNKNOWN_TIMING);
      ErrorMsgAdd("Unknown timer key: '%s'", name);
   }
}

/*--------------------------------------------------------------------------
 * Helper: Print table divisor line
 *--------------------------------------------------------------------------*/

static void
PrintDivisor(void)
{
   printf("+------------");
   for (int i = 0; i < 4; i++)
   {
      printf("+-------------");
   }
   printf("+-------------+\n");
}

/*--------------------------------------------------------------------------
 * Helper: Print table header
 *--------------------------------------------------------------------------*/

static void
PrintHeader(const char *scale)
{
   const char *top[]    = {"", "LS build", "setup", "solve", "relative", ""};
   const char *bottom[] = {"Entry", "times", "times", "times", "res. norm", "iters"};

   printf("|%11s ", top[0]);
   for (int i = 1; i < 6; i++)
   {
      printf("|%12s ", top[i]);
   }
   printf("|\n");

   printf("|%11s ", bottom[0]);
   printf("|%*s %*s ", 11 - (int)strlen(scale), bottom[1], (int)strlen(scale), scale);
   printf("|%*s %*s ", 11 - (int)strlen(scale), bottom[2], (int)strlen(scale), scale);
   printf("|%*s %*s ", 11 - (int)strlen(scale), bottom[3], (int)strlen(scale), scale);
   printf("|%12s ", bottom[4]);
   printf("|%12s ", bottom[5]);
   printf("|\n");
}

/*--------------------------------------------------------------------------
 * Helper: Determine if build times should be printed for this entry
 *--------------------------------------------------------------------------*/

static bool
ShouldPrintBuildTimes(int entry_index)
{
   /* Always print if num_systems is unknown or <= 1 */
   if (global_stats->num_systems < 0 || global_stats->num_systems <= 1)
   {
      return true;
   }

   /* For multiple systems, only print for first entry of each system */
   int entries_per_system = (global_stats->counter + 1) / global_stats->num_systems;
   return (entry_index % entries_per_system == 0);
}

/*--------------------------------------------------------------------------
 * Helper: Print single table entry
 *--------------------------------------------------------------------------*/

static void
PrintEntry(int entry_index)
{
   if (ShouldPrintBuildTimes(entry_index))
   {
      double build_time = global_stats->time_factor *
         (global_stats->dofmap[entry_index] +
          global_stats->matrix[entry_index] +
          global_stats->rhs[entry_index]);
      printf("| %10d | %11.3f | %11.3f | %11.3f | %11.2e |  %10d |\n",
             entry_index,
             build_time,
             global_stats->time_factor * global_stats->prec[entry_index],
             global_stats->time_factor * global_stats->solve[entry_index],
             global_stats->rrnorms[entry_index],
             global_stats->iters[entry_index]);
   }
   else
   {
      printf("| %10d |             | %11.3f | %11.3f | %11.2e |  %10d |\n",
             entry_index,
             global_stats->time_factor * global_stats->prec[entry_index],
             global_stats->time_factor * global_stats->solve[entry_index],
             global_stats->rrnorms[entry_index],
             global_stats->iters[entry_index]);
   }
}

/*--------------------------------------------------------------------------
 * StatsCreate
 *--------------------------------------------------------------------------*/

void
StatsCreate(void)
{
   if (global_stats)
   {
      return;
   }

   int capacity = REALLOC_EXPAND_FACTOR;

   global_stats = (Stats *)malloc(sizeof(Stats));
   memset(global_stats, 0, sizeof(Stats));

   global_stats->capacity     = capacity;
   global_stats->counter      = 0;
   global_stats->reps         = 0;
   global_stats->num_reps     = 1;
   global_stats->num_systems  = -1;  /* Unknown by default */
   global_stats->ls_counter   = 0;
   global_stats->level_depth  = 0;
   global_stats->use_millisec = false;
   global_stats->time_factor  = 1.0;

   /* Allocate timing arrays */
   global_stats->dofmap = (double *)calloc(capacity, sizeof(double));
   global_stats->matrix = (double *)calloc(capacity, sizeof(double));
   global_stats->rhs    = (double *)calloc(capacity, sizeof(double));
   global_stats->iters  = (int *)calloc(capacity, sizeof(int));
   global_stats->prec   = (double *)calloc(capacity, sizeof(double));
   global_stats->solve  = (double *)calloc(capacity, sizeof(double));
   global_stats->rrnorms = (double *)calloc(capacity, sizeof(double));

   /* Initialize level stack */
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      global_stats->level_stack[i].name = NULL;
      global_stats->level_stack[i].start_time = 0.0;
      global_stats->level_stack[i].level = -1;
   }
}

/*--------------------------------------------------------------------------
 * StatsDestroy
 *--------------------------------------------------------------------------*/

void
StatsDestroy(void)
{
   if (global_stats)
   {
      free(global_stats->dofmap);
      free(global_stats->matrix);
      free(global_stats->rhs);
      free(global_stats->iters);
      free(global_stats->prec);
      free(global_stats->solve);
      free(global_stats->rrnorms);

      free(global_stats);
      global_stats = NULL;
   }
}

/*--------------------------------------------------------------------------
 * StatsAnnotateV
 *--------------------------------------------------------------------------*/

void
StatsAnnotateV(HYPREDRV_AnnotateAction action, const char *name, va_list args)
{
   if (!global_stats)
   {
      return;
   }

   /* Format the name string if variadic arguments are provided */
   char    formatted_name[1024];
   va_list args_copy;
   va_copy(args_copy, args);
   vsnprintf(formatted_name, sizeof(formatted_name), name, args_copy);
   va_end(args_copy);

   if (action == HYPREDRV_ANNOTATE_BEGIN)
   {
      HYPREDRV_ANNOTATE_REGION_BEGIN("HYPREDRV_%.1014s", formatted_name)
      HandleAnnotationBegin(formatted_name);
   }
   else if (action == HYPREDRV_ANNOTATE_END)
   {
      HandleAnnotationEnd(formatted_name);
      HYPREDRV_ANNOTATE_REGION_END("HYPREDRV_%.1014s", formatted_name)
   }
}

/*--------------------------------------------------------------------------
 * StatsAnnotate
 *--------------------------------------------------------------------------*/

void
StatsAnnotate(HYPREDRV_AnnotateAction action, const char *name, ...)
{
   va_list args;
   va_start(args, name);
   StatsAnnotateV(action, name, args);
   va_end(args);
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelBegin - Hierarchical annotation with level
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelBegin(int level, const char *name, ...)
{
   if (!global_stats)
   {
      return;
   }

   if (level < 0 || level >= STATS_MAX_LEVELS)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Annotation level %d out of range [0, %d)", level, STATS_MAX_LEVELS);
      return;
   }

   /* Format the name string */
   char    formatted_name[1024];
   va_list args;
   va_start(args, name);
   vsnprintf(formatted_name, sizeof(formatted_name), name, args);
   va_end(args);

   /* Check if level is already active */
   if (global_stats->level_stack[level].name != NULL)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d already has active annotation '%s'", level,
                  global_stats->level_stack[level].name);
      return;
   }

   /* Push onto level stack - allocate memory for name */
   size_t name_len = strlen(formatted_name) + 1;
   global_stats->level_stack[level].name = (const char *)malloc(name_len);
   if (global_stats->level_stack[level].name)
   {
      memcpy((void *)global_stats->level_stack[level].name, formatted_name, name_len);
   }
   global_stats->level_stack[level].level = level;
   global_stats->level_stack[level].start_time = MPI_Wtime();
   if (level >= global_stats->level_depth)
   {
      global_stats->level_depth = level + 1;
   }

   /* Commenting out for now to avoid multiple caliper regions for each solve call */
#if 0
   /* Start Caliper region with hierarchical name */
   char full_name[2048];
   snprintf(full_name, sizeof(full_name), "HYPREDRV_L%d_%.1014s", level, formatted_name);
   HYPREDRV_ANNOTATE_REGION_BEGIN("%s", full_name)
#endif
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelEnd - End hierarchical annotation
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelEnd(int level, const char *name, ...)
{
   if (!global_stats)
   {
      return;
   }

   if (level < 0 || level >= STATS_MAX_LEVELS)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Annotation level %d out of range [0, %d)", level, STATS_MAX_LEVELS);
      return;
   }

   /* Format the name string */
   char    formatted_name[1024];
   va_list args;
   va_start(args, name);
   vsnprintf(formatted_name, sizeof(formatted_name), name, args);
   va_end(args);

   /* Check if level matches */
   if (global_stats->level_stack[level].name == NULL ||
       strcmp(global_stats->level_stack[level].name, formatted_name) != 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d annotation mismatch: expected '%s', got '%s'",
                  level,
                  global_stats->level_stack[level].name ? global_stats->level_stack[level].name : "NULL",
                  formatted_name);
      return;
   }

   /* Commenting out for now to avoid multiple caliper regions for each solve call */
#if 0
   /* Stop Caliper region */
   char full_name[2048];
   snprintf(full_name, sizeof(full_name), "HYPREDRV_L%d_%.1014s", level, formatted_name);
   HYPREDRV_ANNOTATE_REGION_END("%s", full_name)
#endif

   /* Pop from level stack - free allocated memory */
   if (global_stats->level_stack[level].name)
   {
      free((void *)global_stats->level_stack[level].name);
      global_stats->level_stack[level].name = NULL;
   }
   global_stats->level_stack[level].level = -1;

   /* Update depth if needed */
   if (level == global_stats->level_depth - 1)
   {
      while (global_stats->level_depth > 0 &&
             global_stats->level_stack[global_stats->level_depth - 1].name == NULL)
      {
         global_stats->level_depth--;
      }
   }
}

/*--------------------------------------------------------------------------
 * StatsTimerSetMilliseconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetMilliseconds(void)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->use_millisec = true;
   global_stats->time_factor  = 1000.0;
}

/*--------------------------------------------------------------------------
 * StatsTimerSetSeconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetSeconds(void)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->use_millisec = false;
   global_stats->time_factor  = 1.0;
}

/*--------------------------------------------------------------------------
 * StatsIterSet
 *--------------------------------------------------------------------------*/

void
StatsIterSet(int num_iters)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->iters[global_stats->counter] = num_iters;
}

/*--------------------------------------------------------------------------
 * StatsRelativeResNormSet
 *--------------------------------------------------------------------------*/

void
StatsRelativeResNormSet(double rrnorm)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->rrnorms[global_stats->counter] = rrnorm;
}

/*--------------------------------------------------------------------------
 * StatsPrint
 *--------------------------------------------------------------------------*/

void
StatsPrint(int print_level)
{
   if (!global_stats || print_level < 1)
   {
      return;
   }

   const char *scale = global_stats->use_millisec ? "[ms]" : "[s]";

   PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   PrintDivisor();
   PrintHeader(scale);
   PrintDivisor();

   /* Print statistics for each entry that had a solve */
   /* This filters out Newton iterations that broke before solving */
   int max_entry = global_stats->counter;
   for (int i = 0; i <= max_entry; i++)
   {
      /* Only print entries that had a solve (iterations > 0 or solve time > 0) */
      if (global_stats->iters[i] > 0 || global_stats->solve[i] > 0.0)
      {
         PrintEntry(i);
      }
   }

   PrintDivisor();
   printf("\n");
}

/*--------------------------------------------------------------------------
 * StatsGetLinearSystemID
 *--------------------------------------------------------------------------*/

int
StatsGetLinearSystemID(void)
{
   if (!global_stats)
   {
      return -1;
   }
   /* Return 0-based linear system ID */
   return global_stats->ls_counter - 1;
}

/*--------------------------------------------------------------------------
 * StatsSetNumReps
 *--------------------------------------------------------------------------*/

void
StatsSetNumReps(int num_reps)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->num_reps = num_reps;
}

/*--------------------------------------------------------------------------
 * StatsSetNumLinearSystems
 *--------------------------------------------------------------------------*/

void
StatsSetNumLinearSystems(int num_systems)
{
   if (!global_stats)
   {
      return;
   }
   global_stats->num_systems = num_systems;
}

/*--------------------------------------------------------------------------
 * StatsGetLastIter
 *--------------------------------------------------------------------------*/

int
StatsGetLastIter(void)
{
   if (!global_stats)
   {
      return -1;
   }
   return global_stats->iters[global_stats->counter];
}

/*--------------------------------------------------------------------------
 * StatsGetLastSetupTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSetupTime(void)
{
   if (!global_stats)
   {
      return 0.0;
   }
   return global_stats->prec[global_stats->counter] * global_stats->time_factor;
}

/*--------------------------------------------------------------------------
 * StatsGetLastSolveTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSolveTime(void)
{
   if (!global_stats)
   {
      return 0.0;
   }
   return global_stats->solve[global_stats->counter] * global_stats->time_factor;
}
