/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "stats.h"
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <string.h>

/* Active stats context (pointer to stats owned by a hypredrv_t object) */
static Stats *active_stats =
   NULL; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/* Reallocation expansion factor */
#define REALLOC_EXPAND_FACTOR 16

/*--------------------------------------------------------------------------
 * Helper: Reallocate array with expansion
 *--------------------------------------------------------------------------*/

static void
ReallocateArray(void **ptr, size_t elem_size, int old_capacity, int new_capacity)
{
   void *new_ptr = realloc(*ptr, (size_t)new_capacity * elem_size);
   if (new_ptr)
   {
      /* Zero out the newly allocated portion */
      memset((char *)new_ptr + ((size_t)old_capacity * elem_size), 0,
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
   if (active_stats->counter + 1 >= active_stats->capacity)
   {
      int old_capacity = active_stats->capacity;
      int new_capacity = old_capacity + REALLOC_EXPAND_FACTOR;

      /* Reallocate all arrays with the same new capacity */
      ReallocateArray((void **)&active_stats->matrix, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->rhs, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->dofmap, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->prec, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->solve, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->rrnorms, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->r0norms, sizeof(double), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->iters, sizeof(int), old_capacity,
                      new_capacity);
      ReallocateArray((void **)&active_stats->entry_ls_id, sizeof(int), old_capacity,
                      new_capacity);

      /* Update capacity after all reallocations */
      active_stats->capacity = new_capacity;
   }
}

/*--------------------------------------------------------------------------
 * Helper: Start timer for vector entry
 *--------------------------------------------------------------------------*/

static void
StartVectorTimer(double *timer_array, int index)
{
   if (timer_array && index >= 0 && index < active_stats->capacity)
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
   if (timer_array && index >= 0 && index < active_stats->capacity)
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

   /* Update counters for special annotations */
   if (!strcmp(name, "matrix") || !strcmp(name, "system"))
   {
      /* Start of a new linear system build:
       * - reset repetition counter
       * - reserve the next stats entry so build timers attach to the upcoming solve entry
       */
      active_stats->reps = 0;
      active_stats->counter++;
      active_stats->matrix_counter = active_stats->counter;
      EnsureCapacity();
      StartVectorTimer(active_stats->matrix, active_stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      /* Beginning of a solve entry (one repetition / one variant):
       * - advance entry index for all but the first repetition after a build
       * - keep indices contiguous: 0,1,2,... across repetitions and variants
       */
      if (active_stats->counter < 0)
      {
         active_stats->counter = 0;
      }
      else if (active_stats->reps > 0)
      {
         active_stats->counter++;
      }
      active_stats->reps++;
      EnsureCapacity();
      StartScalarTimer(&active_stats->reset_x0);
   }
   else if (!strcmp(name, "rhs"))
   {
      if (active_stats->counter < 0)
      {
         active_stats->counter = 0;
      }
      EnsureCapacity();
      StartVectorTimer(active_stats->rhs, active_stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      if (active_stats->counter < 0)
      {
         active_stats->counter = 0;
      }
      EnsureCapacity();
      StartVectorTimer(active_stats->dofmap, active_stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      if (active_stats->counter < 0)
      {
         active_stats->counter = 0;
      }
      EnsureCapacity();
      StartVectorTimer(active_stats->prec, active_stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      /* Increment linear system counter only on the first solve for a new system.
       * (reset_x0 has already set reps=1 for the first repetition)
       * Only increment if this is the first solve after the last "matrix" annotation.
       * We track this by comparing counter to matrix_counter. */
      if (active_stats->reps == 1 &&
          active_stats->counter == active_stats->matrix_counter)
      {
         active_stats->ls_counter++;
      }
      if (active_stats->counter < 0)
      {
         active_stats->counter = 0;
      }
      EnsureCapacity();
      StartVectorTimer(active_stats->solve, active_stats->counter);
      /* Tag this entry with its linear system id */
      if (active_stats->entry_ls_id && active_stats->counter < active_stats->capacity)
      {
         active_stats->entry_ls_id[active_stats->counter] = active_stats->ls_counter - 1;
      }
   }
   else if (!strcmp(name, "initialize"))
   {
      EnsureCapacity();
      StartScalarTimer(&active_stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      EnsureCapacity();
      StartScalarTimer(&active_stats->finalize);
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
      StopVectorTimer(active_stats->matrix, active_stats->counter);
   }
   else if (!strcmp(name, "rhs"))
   {
      StopVectorTimer(active_stats->rhs, active_stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      StopVectorTimer(active_stats->dofmap, active_stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      StopVectorTimer(active_stats->prec, active_stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      StopVectorTimer(active_stats->solve, active_stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      StopScalarTimer(&active_stats->reset_x0);
   }
   else if (!strcmp(name, "initialize"))
   {
      StopScalarTimer(&active_stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      StopScalarTimer(&active_stats->finalize);
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

enum
{
   STATS_ENTRY_WIDTH = 6,
   STATS_TIME_WIDTH  = 11,
   STATS_RES_WIDTH   = 10,
   STATS_ITERS_WIDTH = 6
};

static void
PrintDivisor(void)
{
   const int widths[] = {STATS_ENTRY_WIDTH, STATS_TIME_WIDTH, STATS_TIME_WIDTH,
                         STATS_TIME_WIDTH,  STATS_RES_WIDTH,  STATS_RES_WIDTH,
                         STATS_ITERS_WIDTH};
   const int count    = (int)(sizeof(widths) / sizeof(widths[0]));

   for (int i = 0; i < count; i++)
   {
      putchar('+');
      for (int j = 0; j < widths[i] + 2; j++)
      {
         putchar('-');
      }
   }
   printf("+\n");
}

/*--------------------------------------------------------------------------
 * Helper: Print table header
 *--------------------------------------------------------------------------*/

static void
PrintHeader(const char *scale)
{
   const char *top[]    = {"", "LS build", "setup", "solve", "initial", "relative", ""};
   const char *bottom[] = {"Entry",     "times",     "times", "times",
                           "res. norm", "res. norm", "iters"};
   char        time_label[32];

   snprintf(time_label, sizeof(time_label), "times %s", scale);

   printf("| %*s ", STATS_ENTRY_WIDTH, top[0]);
   printf("| %*s ", STATS_TIME_WIDTH, top[1]);
   printf("| %*s ", STATS_TIME_WIDTH, top[2]);
   printf("| %*s ", STATS_TIME_WIDTH, top[3]);
   printf("| %*s ", STATS_RES_WIDTH, top[4]);
   printf("| %*s ", STATS_RES_WIDTH, top[5]);
   printf("| %*s |\n", STATS_ITERS_WIDTH, top[6]);

   printf("| %*s ", STATS_ENTRY_WIDTH, bottom[0]);
   printf("| %*s ", STATS_TIME_WIDTH, time_label);
   printf("| %*s ", STATS_TIME_WIDTH, time_label);
   printf("| %*s ", STATS_TIME_WIDTH, time_label);
   printf("| %*s ", STATS_RES_WIDTH, bottom[4]);
   printf("| %*s ", STATS_RES_WIDTH, bottom[5]);
   printf("| %*s |\n", STATS_ITERS_WIDTH, bottom[6]);
}

/*--------------------------------------------------------------------------
 * Helper: Print single table entry with a given display index
 *--------------------------------------------------------------------------*/

static void
PrintEntryWithIndex(int entry_index, int display_index)
{
   double build_time = active_stats->time_factor * (active_stats->dofmap[entry_index] +
                                                    active_stats->matrix[entry_index] +
                                                    active_stats->rhs[entry_index]);
   bool   show_build = (build_time > 0.0);

   if (show_build)
   {
      printf("| %*d | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             display_index, STATS_TIME_WIDTH, 3, build_time, STATS_TIME_WIDTH, 3,
             active_stats->time_factor * active_stats->prec[entry_index],
             STATS_TIME_WIDTH, 3,
             active_stats->time_factor * active_stats->solve[entry_index],
             STATS_RES_WIDTH, 2, active_stats->r0norms[entry_index], STATS_RES_WIDTH, 2,
             active_stats->rrnorms[entry_index], STATS_ITERS_WIDTH,
             active_stats->iters[entry_index]);
   }
   else
   {
      printf("| %*d | %*s | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             display_index, STATS_TIME_WIDTH, "", STATS_TIME_WIDTH, 3,
             active_stats->time_factor * active_stats->prec[entry_index],
             STATS_TIME_WIDTH, 3,
             active_stats->time_factor * active_stats->solve[entry_index],
             STATS_RES_WIDTH, 2, active_stats->r0norms[entry_index], STATS_RES_WIDTH, 2,
             active_stats->rrnorms[entry_index], STATS_ITERS_WIDTH,
             active_stats->iters[entry_index]);
   }
}

/*--------------------------------------------------------------------------
 * StatsCreate - creates and returns a new Stats object
 *--------------------------------------------------------------------------*/

Stats *
StatsCreate(void)
{
   int capacity = REALLOC_EXPAND_FACTOR;

   Stats *stats = (Stats *)malloc(sizeof(Stats));
   memset(stats, 0, sizeof(Stats));

   stats->capacity = capacity;
   /* First solve entry should start at index 0, but we increment at the start of
    * a new build ("matrix"/"system"), so initialize to -1. */
   stats->counter        = -1;
   stats->reps           = 0;
   stats->num_reps       = 1;
   stats->num_systems    = -1; /* Unknown by default */
   stats->ls_counter     = 0;
   stats->matrix_counter = -1;
   stats->level_depth    = 0;
   stats->use_millisec   = false;
   stats->time_factor    = 1.0;

   /* Allocate timing arrays */
   stats->dofmap      = (double *)calloc((size_t)capacity, sizeof(double));
   stats->matrix      = (double *)calloc((size_t)capacity, sizeof(double));
   stats->rhs         = (double *)calloc((size_t)capacity, sizeof(double));
   stats->iters       = (int *)calloc((size_t)capacity, sizeof(int));
   stats->prec        = (double *)calloc((size_t)capacity, sizeof(double));
   stats->solve       = (double *)calloc((size_t)capacity, sizeof(double));
   stats->rrnorms     = (double *)calloc((size_t)capacity, sizeof(double));
   stats->r0norms     = (double *)calloc((size_t)capacity, sizeof(double));
   stats->entry_ls_id = (int *)calloc((size_t)capacity, sizeof(int));

   /* Initialize level stack */
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      stats->level_stack[i].name       = NULL;
      stats->level_stack[i].start_time = 0.0;
      stats->level_stack[i].level      = -1;
   }

   /* Initialize per-level statistics */
   stats->level_active = 0;
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      stats->level_count[i] = 0;
      stats->level_entries[i] =
         (LevelEntry *)calloc(STATS_TIMESTEP_CAPACITY, sizeof(LevelEntry));
      stats->level_current_id[i]  = 0;
      stats->level_solve_start[i] = 0;
   }

   return stats;
}

/*--------------------------------------------------------------------------
 * StatsSetContext - sets the active stats context for internal use
 *--------------------------------------------------------------------------*/

void
StatsSetContext(Stats *stats)
{
   active_stats = stats;
}

/*--------------------------------------------------------------------------
 * StatsGetContext - returns the active stats context
 *--------------------------------------------------------------------------*/

Stats *
StatsGetContext(void)
{
   return active_stats;
}

/*--------------------------------------------------------------------------
 * StatsDestroy - destroys a Stats object and sets pointer to NULL
 *--------------------------------------------------------------------------*/

void
StatsDestroy(Stats **stats_ptr)
{
   if (!stats_ptr || !*stats_ptr)
   {
      return;
   }

   Stats *stats = *stats_ptr;

   /* Clear active context if this is the active stats */
   if (active_stats == stats)
   {
      active_stats = NULL;
   }

   /* Free timing arrays */
   free(stats->dofmap);
   free(stats->matrix);
   free(stats->rhs);
   free(stats->iters);
   free(stats->prec);
   free(stats->solve);
   free(stats->rrnorms);
   free(stats->r0norms);
   free(stats->entry_ls_id);

   /* Free level entry arrays */
   for (int i = 0; i < STATS_MAX_LEVELS; i++)
   {
      free(stats->level_entries[i]);
   }

   /* Free the struct itself */
   free(stats);
   *stats_ptr = NULL;
}

/*--------------------------------------------------------------------------
 * StatsAnnotateV
 *--------------------------------------------------------------------------*/

void
StatsAnnotateV(HYPREDRV_AnnotateAction action, const char *name, va_list args)
{
   if (!active_stats)
   {
      return;
   }

   /* Format the name string if variadic arguments are provided */
   char formatted_name[1024];
   if (args)
   {
      va_list args_copy;
      va_copy(args_copy, args);
      vsnprintf(formatted_name, sizeof(formatted_name), name, args_copy);
      va_end(args_copy);
   }
   else
   {
      /* No variadic args - use name as-is */
      strncpy(formatted_name, name, sizeof(formatted_name) - 1);
      formatted_name[sizeof(formatted_name) - 1] = '\0';
   }

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
StatsAnnotate(HYPREDRV_AnnotateAction action, const char *name)
{
   StatsAnnotateV(action, name, NULL);
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelBegin - Hierarchical annotation with level
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelBegin(int level, const char *name)
{
   if (!active_stats)
   {
      return;
   }

   if (level < 0 || level >= STATS_MAX_LEVELS)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Annotation level %d out of range [0, %d)", level, STATS_MAX_LEVELS);
      return;
   }

   /* Use name as-is (caller should format before calling) */
   const char *formatted_name = name;

   /* Check if level is already active */
   if (active_stats->level_stack[level].name != NULL)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d already has active annotation '%s'", level,
                  active_stats->level_stack[level].name);
      return;
   }

   /* Push onto level stack - allocate memory for name */
   size_t name_len                       = strlen(formatted_name) + 1;
   active_stats->level_stack[level].name = (const char *)malloc(name_len);
   if (active_stats->level_stack[level].name)
   {
      memcpy((void *)active_stats->level_stack[level].name, formatted_name, name_len);
   }
   active_stats->level_stack[level].level      = level;
   active_stats->level_stack[level].start_time = MPI_Wtime();
   if (level >= active_stats->level_depth)
   {
      active_stats->level_depth = level + 1;
   }

   /* Record solve index at start for this level */
   active_stats->level_active |= (1 << level);
   active_stats->level_current_id[level]++;
   active_stats->level_solve_start[level] = active_stats->ls_counter;

   /* Commenting out for now to avoid multiple caliper regions for each solve call */
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelEnd - End hierarchical annotation
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Helper: Ensure level array capacity (uses fixed capacity, reallocates if needed)
 *--------------------------------------------------------------------------*/

static void
EnsureLevelCapacity(int level)
{
   /* Simple approach: reallocate if count reaches capacity */
   int count    = active_stats->level_count[level];
   int capacity = STATS_TIMESTEP_CAPACITY;

   /* Check if we need more space (count is power-of-2 threshold) */
   while (capacity <= count)
   {
      capacity *= 2;
   }

   if (count > 0 && (count & (count - 1)) == 0 && count >= STATS_TIMESTEP_CAPACITY)
   {
      /* count is a power of 2 and >= initial capacity, time to grow */
      int         new_capacity = count * 2;
      LevelEntry *new_ptr      = (LevelEntry *)realloc(
         active_stats->level_entries[level], (size_t)new_capacity * sizeof(LevelEntry));
      if (new_ptr)
      {
         active_stats->level_entries[level] = new_ptr;
      }
   }
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelEnd - End hierarchical annotation
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelEnd(int level, const char *name)
{
   if (!active_stats)
   {
      return;
   }

   if (level < 0 || level >= STATS_MAX_LEVELS)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Annotation level %d out of range [0, %d)", level, STATS_MAX_LEVELS);
      return;
   }

   /* Use name as-is (caller should format before calling) */
   const char *formatted_name = name;

   /* Check if level matches */
   if (active_stats->level_stack[level].name == NULL ||
       strcmp(active_stats->level_stack[level].name, formatted_name) != 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d annotation mismatch: expected '%s', got '%s'", level,
                  active_stats->level_stack[level].name
                     ? active_stats->level_stack[level].name
                     : "NULL",
                  formatted_name);
      return;
   }

   /* Save level entry if this level is active */
   if (active_stats->level_active & (1 << level))
   {
      EnsureLevelCapacity(level);

      int idx                                    = active_stats->level_count[level];
      active_stats->level_entries[level][idx].id = active_stats->level_current_id[level];
      active_stats->level_entries[level][idx].solve_start =
         active_stats->level_solve_start[level];
      active_stats->level_entries[level][idx].solve_end = active_stats->ls_counter;

      active_stats->level_count[level]++;
      active_stats->level_active &= ~(1 << level);
   }

   /* Commenting out for now to avoid multiple caliper regions for each solve call */

   /* Pop from level stack - free allocated memory */
   if (active_stats->level_stack[level].name)
   {
      free((void *)active_stats->level_stack[level].name);
      active_stats->level_stack[level].name = NULL;
   }
   active_stats->level_stack[level].level = -1;

   /* Update depth if needed */
   if (level == active_stats->level_depth - 1)
   {
      while (active_stats->level_depth > 0 &&
             active_stats->level_stack[active_stats->level_depth - 1].name == NULL)
      {
         active_stats->level_depth--;
      }
   }
}

/*--------------------------------------------------------------------------
 * StatsTimerSetMilliseconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetMilliseconds(void)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->use_millisec = true;
   active_stats->time_factor  = 1000.0;
}

/*--------------------------------------------------------------------------
 * StatsTimerSetSeconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetSeconds(void)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->use_millisec = false;
   active_stats->time_factor  = 1.0;
}

/*--------------------------------------------------------------------------
 * StatsIterSet
 *--------------------------------------------------------------------------*/

void
StatsIterSet(int num_iters)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->iters[active_stats->counter] = num_iters;
}

/*--------------------------------------------------------------------------
 * StatsInitialResNormSet
 *--------------------------------------------------------------------------*/

void
StatsInitialResNormSet(double r0norm)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->r0norms[active_stats->counter] = r0norm;
}

/*--------------------------------------------------------------------------
 * StatsRelativeResNormSet
 *--------------------------------------------------------------------------*/

void
StatsRelativeResNormSet(double rrnorm)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->rrnorms[active_stats->counter] = rrnorm;
}

/*--------------------------------------------------------------------------
 * StatsPrint
 *--------------------------------------------------------------------------*/

void
StatsPrint(int print_level)
{
   if (!active_stats || print_level < 1)
   {
      return;
   }

   const char *scale = ((int)active_stats->use_millisec) ? "[ms]" : "[s]";

   PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH);
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   PrintDivisor();
   PrintHeader(scale);
   PrintDivisor();

   /* Print statistics for each entry that had a solve */
   /* This filters out Newton iterations that broke before solving */
   /* Use a display index to avoid gaps in the entry column */
   int max_entry   = active_stats->counter;
   int display_idx = 0;
   for (int i = 0; i <= max_entry; i++)
   {
      /* Only print entries that had a solve (iterations > 0 or solve time > 0) */
      if (active_stats->iters[i] > 0 || active_stats->solve[i] > 0.0)
      {
         PrintEntryWithIndex(i, display_idx);
         display_idx++;
      }
   }

   /* Print aggregate rows inside the same table for print level > 1 */
   if (display_idx > 1 && print_level > 1)
   {
      double min_build = HUGE_VAL, max_build = 0.0, sum_build = 0.0, ssq_build = 0.0;
      double min_setup = HUGE_VAL, max_setup = 0.0, sum_setup = 0.0, ssq_setup = 0.0;
      double min_solve = HUGE_VAL, max_solve = 0.0, sum_solve = 0.0, ssq_solve = 0.0;
      double min_r0 = HUGE_VAL, max_r0 = 0.0, sum_r0 = 0.0, ssq_r0 = 0.0;
      double min_rr = HUGE_VAL, max_rr = 0.0, sum_rr = 0.0, ssq_rr = 0.0;
      int    min_iters = INT_MAX, max_iters = 0, sum_iters = 0;
      double ssq_iters = 0.0;

      for (int i = 0; i <= max_entry; i++)
      {
         if (active_stats->iters[i] <= 0 && active_stats->solve[i] <= 0.0)
         {
            continue;
         }

         double b =
            active_stats->time_factor *
            (active_stats->dofmap[i] + active_stats->matrix[i] + active_stats->rhs[i]);
         double s  = active_stats->time_factor * active_stats->prec[i];
         double v  = active_stats->time_factor * active_stats->solve[i];
         double r0 = active_stats->r0norms[i];
         double rr = active_stats->rrnorms[i];
         int    it = active_stats->iters[i];

         if (b < min_build) min_build = b;
         if (b > max_build) max_build = b;
         sum_build += b;
         ssq_build += b * b;

         if (s < min_setup) min_setup = s;
         if (s > max_setup) max_setup = s;
         sum_setup += s;
         ssq_setup += s * s;

         if (v < min_solve) min_solve = v;
         if (v > max_solve) max_solve = v;
         sum_solve += v;
         ssq_solve += v * v;

         if (r0 < min_r0) min_r0 = r0;
         if (r0 > max_r0) max_r0 = r0;
         sum_r0 += r0;
         ssq_r0 += r0 * r0;

         if (rr < min_rr) min_rr = rr;
         if (rr > max_rr) max_rr = rr;
         sum_rr += rr;
         ssq_rr += rr * rr;

         if (it < min_iters) min_iters = it;
         if (it > max_iters) max_iters = it;
         sum_iters += it;
         ssq_iters += (double)it * it;
      }

      int    n         = display_idx;
      double avg_build = sum_build / n;
      double avg_setup = sum_setup / n;
      double avg_solve = sum_solve / n;
      double avg_r0    = sum_r0 / n;
      double avg_rr    = sum_rr / n;
      double avg_iters = (double)sum_iters / n;

      double std_build = sqrt((ssq_build - (n * avg_build * avg_build)) / (n - 1));
      double std_setup = sqrt((ssq_setup - (n * avg_setup * avg_setup)) / (n - 1));
      double std_solve = sqrt((ssq_solve - (n * avg_solve * avg_solve)) / (n - 1));
      double std_r0    = sqrt((ssq_r0 - (n * avg_r0 * avg_r0)) / (n - 1));
      double std_rr    = sqrt((ssq_rr - (n * avg_rr * avg_rr)) / (n - 1));
      double std_iters = sqrt((ssq_iters - (n * avg_iters * avg_iters)) / (n - 1));

      PrintDivisor();
      printf("| %*s | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             "Min.", STATS_TIME_WIDTH, 3, min_build, STATS_TIME_WIDTH, 3, min_setup,
             STATS_TIME_WIDTH, 3, min_solve, STATS_RES_WIDTH, 2, min_r0, STATS_RES_WIDTH,
             2, min_rr, STATS_ITERS_WIDTH, min_iters);
      printf("| %*s | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             "Max.", STATS_TIME_WIDTH, 3, max_build, STATS_TIME_WIDTH, 3, max_setup,
             STATS_TIME_WIDTH, 3, max_solve, STATS_RES_WIDTH, 2, max_r0, STATS_RES_WIDTH,
             2, max_rr, STATS_ITERS_WIDTH, max_iters);
      printf("| %*s | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*.1f |\n",
             STATS_ENTRY_WIDTH, "Avg.", STATS_TIME_WIDTH, 3, avg_build, STATS_TIME_WIDTH,
             3, avg_setup, STATS_TIME_WIDTH, 3, avg_solve, STATS_RES_WIDTH, 2, avg_r0,
             STATS_RES_WIDTH, 2, avg_rr, STATS_ITERS_WIDTH, avg_iters);
      printf("| %*s | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*.1f |\n",
             STATS_ENTRY_WIDTH, "Std.", STATS_TIME_WIDTH, 3, std_build, STATS_TIME_WIDTH,
             3, std_setup, STATS_TIME_WIDTH, 3, std_solve, STATS_RES_WIDTH, 2, std_r0,
             STATS_RES_WIDTH, 2, std_rr, STATS_ITERS_WIDTH, std_iters);
      printf("| %*s | %*.*f | %*.*f | %*.*f | %*s | %*s | %*d |\n", STATS_ENTRY_WIDTH,
             "Total", STATS_TIME_WIDTH, 3, sum_build, STATS_TIME_WIDTH, 3, sum_setup,
             STATS_TIME_WIDTH, 3, sum_solve, STATS_RES_WIDTH, "", STATS_RES_WIDTH, "",
             STATS_ITERS_WIDTH, sum_iters);
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
   if (!active_stats)
   {
      return -1;
   }
   /* Return 0-based linear system ID */
   return active_stats->ls_counter - 1;
}

/*--------------------------------------------------------------------------
 * StatsSetNumReps
 *--------------------------------------------------------------------------*/

void
StatsSetNumReps(int num_reps)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->num_reps = num_reps;
}

/*--------------------------------------------------------------------------
 * StatsSetNumLinearSystems
 *--------------------------------------------------------------------------*/

void
StatsSetNumLinearSystems(int num_systems)
{
   if (!active_stats)
   {
      return;
   }
   active_stats->num_systems = num_systems;
}

/*--------------------------------------------------------------------------
 * StatsGetLastIter
 *--------------------------------------------------------------------------*/

int
StatsGetLastIter(void)
{
   if (!active_stats)
   {
      return -1;
   }
   return active_stats->iters[active_stats->counter];
}

/*--------------------------------------------------------------------------
 * StatsGetLastSetupTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSetupTime(void)
{
   if (!active_stats)
   {
      return 0.0;
   }
   return active_stats->prec[active_stats->counter] * active_stats->time_factor;
}

/*--------------------------------------------------------------------------
 * StatsGetLastSolveTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSolveTime(void)
{
   if (!active_stats)
   {
      return 0.0;
   }
   return active_stats->solve[active_stats->counter] * active_stats->time_factor;
}

/*--------------------------------------------------------------------------
 * StatsLevelGetCount - Get number of entries at a level
 *--------------------------------------------------------------------------*/

int
StatsLevelGetCount(int level)
{
   if (!active_stats || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return 0;
   }
   return active_stats->level_count[level];
}

/*--------------------------------------------------------------------------
 * StatsLevelGetEntry - Get level entry by index
 *--------------------------------------------------------------------------*/

int
StatsLevelGetEntry(int level, int index, LevelEntry *entry)
{
   if (!active_stats || !entry || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return -1;
   }

   if (index < 0 || index >= active_stats->level_count[level])
   {
      return -1;
   }

   *entry = active_stats->level_entries[level][index];
   return 0;
}

/*--------------------------------------------------------------------------
 * Helper: Compute aggregates from solve index range
 *--------------------------------------------------------------------------*/

static void
ComputeLevelStats(const LevelEntry *entry, int *num_solves, int *linear_iters,
                  double *setup_time, double *solve_time)
{
   int    n_solves = 0;
   int    l_iters  = 0;
   double s_time   = 0.0;
   double p_time   = 0.0;

   for (int i = entry->solve_start; i < entry->solve_end; i++)
   {
      l_iters += active_stats->iters[i];
      p_time += active_stats->prec[i];
      s_time += active_stats->solve[i];
   }
   n_solves = entry->solve_end - entry->solve_start;

   if (num_solves) *num_solves = n_solves;
   if (linear_iters) *linear_iters = l_iters;
   if (setup_time) *setup_time = p_time;
   if (solve_time) *solve_time = s_time;
}

/*--------------------------------------------------------------------------
 * StatsLevelPrint - Print statistics summary for a level
 *--------------------------------------------------------------------------*/

void
StatsLevelPrint(int level)
{
   if (!active_stats || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return;
   }

   int count = active_stats->level_count[level];
   if (count == 0)
   {
      return;
   }

   /* Compute totals */
   long long total_solves = 0;
   long long total_linear = 0;
   double    total_setup  = 0.0;
   double    total_solve  = 0.0;

   for (int i = 0; i < count; i++)
   {
      const LevelEntry *entry      = &active_stats->level_entries[level][i];
      int               num_solves = 0, linear_iters = 0;
      double            setup_time = 0.0, solve_time = 0.0;

      ComputeLevelStats(entry, &num_solves, &linear_iters, &setup_time, &solve_time);

      total_solves += num_solves;
      total_linear += linear_iters;
      total_setup += setup_time;
      total_solve += solve_time;
   }

   /* Print summary in original format */
   double avg_iters_per_solve =
      total_solves > 0 ? (double)total_linear / (double)total_solves : 0.0;
   double avg_setup_per_solve =
      total_solves > 0 ? total_setup / (double)total_solves : 0.0;
   double avg_solve_per_solve =
      total_solves > 0 ? total_solve / (double)total_solves : 0.0;

   double avg_iters_per_entry = count > 0 ? (double)total_linear / count : 0.0;
   double avg_setup_per_entry = count > 0 ? total_setup / count : 0.0;
   double avg_solve_per_entry = count > 0 ? total_solve / count : 0.0;

   printf("\n");
   printf("Aggregate Summary:\n");
   printf("--------------------------------------------------------------\n");
   printf("Total number of Non-linear iterations: %lld\n", total_solves);
   printf("Total number of linear iterations:     %lld\n", total_linear);
   printf("Avg. LS iterations:                    %.2f\n", avg_iters_per_solve);
   printf("Avg. LS times: (setup, solve, total):  %.4f, %.4f, %.4f\n",
          avg_setup_per_solve, avg_solve_per_solve,
          avg_setup_per_solve + avg_solve_per_solve);
   printf("Total LS times: (setup, solve, total): %.4f, %.4f, %.4f\n", total_setup,
          total_solve, total_setup + total_solve);
   printf("Avg. LS iterations per timestep:       %.2f\n", avg_iters_per_entry);
   printf("Avg. LS times per timestep: (s, s, t): %.4f, %.4f, %.4f\n",
          avg_setup_per_entry, avg_solve_per_entry,
          avg_setup_per_entry + avg_solve_per_entry);
   printf("--------------------------------------------------------------\n");
   printf("\n");
}
