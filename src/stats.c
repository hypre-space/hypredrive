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

/* Reallocation expansion factor */
enum
{
   REALLOC_EXPAND_FACTOR = 16
};

/*--------------------------------------------------------------------------
 * Helper: Reallocate array with expansion
 *--------------------------------------------------------------------------*/

static int
ReallocateArray(void **ptr, size_t elem_size, int old_capacity, int new_capacity)
{
   void *new_ptr = realloc(*ptr, (size_t)new_capacity * elem_size);
   if (!new_ptr)
   {
      return 0;
   }

   /* Zero out the newly allocated portion */
   memset((char *)new_ptr + ((size_t)old_capacity * elem_size), 0,
          REALLOC_EXPAND_FACTOR * elem_size);
   *ptr = new_ptr;
   return 1;
}

/*--------------------------------------------------------------------------
 * Helper: Ensure capacity for current counter
 *--------------------------------------------------------------------------*/

static int
EnsureCapacity(Stats *stats)
{
   /* Need capacity > counter to have space for counter index */
   if (stats->counter + 1 >= stats->capacity)
   {
      int old_capacity = stats->capacity;
      int new_capacity = old_capacity + REALLOC_EXPAND_FACTOR;
      int failed       = 0;

      /* Reallocate all arrays with the same new capacity */
      failed |= !ReallocateArray((void **)&stats->matrix, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->rhs, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->dofmap, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->prec, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->solve, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->rrnorms, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->r0norms, sizeof(double), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->iters, sizeof(int), old_capacity,
                                 new_capacity);
      failed |= !ReallocateArray((void **)&stats->entry_ls_id, sizeof(int), old_capacity,
                                 new_capacity);

      if (failed)
      {
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Stats capacity growth failed (%d -> %d)", old_capacity,
                     new_capacity);
         return 0;
      }

      /* Update capacity after all reallocations */
      stats->capacity = new_capacity;
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Helper: Start timer for vector entry
 *--------------------------------------------------------------------------*/

static void
StartVectorTimer(const Stats *stats, double *timer_array, int index)
{
   if (timer_array && index >= 0 && index < stats->capacity)
   {
      timer_array[index] -= MPI_Wtime();
   }
}

/*--------------------------------------------------------------------------
 * Helper: Stop timer for vector entry
 *--------------------------------------------------------------------------*/

static void
StopVectorTimer(const Stats *stats, double *timer_array, int index)
{
   if (timer_array && index >= 0 && index < stats->capacity)
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
HandleAnnotationBegin(Stats *stats, const char *name)
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
      stats->reps = 0;
      stats->counter++;
      stats->matrix_counter = stats->counter;
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartVectorTimer(stats, stats->matrix, stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      /* Beginning of a solve entry (one repetition / one variant):
       * - advance entry index for all but the first repetition after a build
       * - keep indices contiguous: 0,1,2,... across repetitions and variants
       */
      if (stats->counter < 0)
      {
         stats->counter = 0;
      }
      else if (stats->reps > 0)
      {
         stats->counter++;
      }
      stats->reps++;
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartScalarTimer(&stats->reset_x0);
   }
   else if (!strcmp(name, "rhs"))
   {
      if (stats->counter < 0)
      {
         stats->counter = 0;
      }
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartVectorTimer(stats, stats->rhs, stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      if (stats->counter < 0)
      {
         stats->counter = 0;
      }
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartVectorTimer(stats, stats->dofmap, stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      if (stats->counter < 0)
      {
         stats->counter = 0;
      }
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartVectorTimer(stats, stats->prec, stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      /* Increment linear system counter only on the first solve for a new system.
       * (reset_x0 has already set reps=1 for the first repetition)
       * Only increment if this is the first solve after the last "matrix" annotation.
       * We track this by comparing counter to matrix_counter. */
      if (stats->reps == 1 && stats->counter == stats->matrix_counter)
      {
         stats->ls_counter++;
      }
      if (stats->counter < 0)
      {
         stats->counter = 0;
      }
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartVectorTimer(stats, stats->solve, stats->counter);
      /* Tag this entry with its linear system id */
      if (stats->entry_ls_id && stats->counter < stats->capacity)
      {
         stats->entry_ls_id[stats->counter] = stats->ls_counter - 1;
      }
   }
   else if (!strcmp(name, "initialize"))
   {
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartScalarTimer(&stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      if (!EnsureCapacity(stats))
      {
         return;
      }
      StartScalarTimer(&stats->finalize);
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
HandleAnnotationEnd(Stats *stats, const char *name)
{
   /* Ignore "Run" annotation */
   if (strncmp(name, "Run", 3) == 0)
   {
      return;
   }

   /* Stop timers based on annotation name */
   if (!strcmp(name, "matrix") || !strcmp(name, "system"))
   {
      StopVectorTimer(stats, stats->matrix, stats->counter);
   }
   else if (!strcmp(name, "rhs"))
   {
      StopVectorTimer(stats, stats->rhs, stats->counter);
   }
   else if (!strcmp(name, "dofmap"))
   {
      StopVectorTimer(stats, stats->dofmap, stats->counter);
   }
   else if (!strcmp(name, "prec"))
   {
      StopVectorTimer(stats, stats->prec, stats->counter);
   }
   else if (!strcmp(name, "solve"))
   {
      StopVectorTimer(stats, stats->solve, stats->counter);
   }
   else if (!strcmp(name, "reset_x0"))
   {
      StopScalarTimer(&stats->reset_x0);
   }
   else if (!strcmp(name, "initialize"))
   {
      StopScalarTimer(&stats->initialize);
   }
   else if (!strcmp(name, "finalize"))
   {
      StopScalarTimer(&stats->finalize);
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
PrintEntryWithIndex(const Stats *stats, int entry_index, int display_index)
{
   double build_time =
      stats->time_factor *
      (stats->dofmap[entry_index] + stats->matrix[entry_index] + stats->rhs[entry_index]);
   bool show_build = (build_time > 0.0);

   if (show_build)
   {
      printf("| %*d | %*.*f | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             display_index, STATS_TIME_WIDTH, 3, build_time, STATS_TIME_WIDTH, 3,
             stats->time_factor * stats->prec[entry_index], STATS_TIME_WIDTH, 3,
             stats->time_factor * stats->solve[entry_index], STATS_RES_WIDTH, 2,
             stats->r0norms[entry_index], STATS_RES_WIDTH, 2, stats->rrnorms[entry_index],
             STATS_ITERS_WIDTH, stats->iters[entry_index]);
   }
   else
   {
      printf("| %*d | %*s | %*.*f | %*.*f | %*.*e | %*.*e | %*d |\n", STATS_ENTRY_WIDTH,
             display_index, STATS_TIME_WIDTH, "", STATS_TIME_WIDTH, 3,
             stats->time_factor * stats->prec[entry_index], STATS_TIME_WIDTH, 3,
             stats->time_factor * stats->solve[entry_index], STATS_RES_WIDTH, 2,
             stats->r0norms[entry_index], STATS_RES_WIDTH, 2, stats->rrnorms[entry_index],
             STATS_ITERS_WIDTH, stats->iters[entry_index]);
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
StatsAnnotateV(Stats *stats, HYPREDRV_AnnotateAction action, const char *name,
               va_list args)
{
   if (!stats)
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
      HandleAnnotationBegin(stats, formatted_name);
   }
   else if (action == HYPREDRV_ANNOTATE_END)
   {
      HandleAnnotationEnd(stats, formatted_name);
      HYPREDRV_ANNOTATE_REGION_END("HYPREDRV_%.1014s", formatted_name)
   }
}

/*--------------------------------------------------------------------------
 * StatsAnnotate
 *--------------------------------------------------------------------------*/

void
StatsAnnotate(Stats *stats, HYPREDRV_AnnotateAction action, const char *name)
{
   StatsAnnotateV(stats, action, name, NULL);
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelBegin - Hierarchical annotation with level
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelBegin(Stats *stats, int level, const char *name)
{
   if (!stats)
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
   if (stats->level_stack[level].name != NULL)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d already has active annotation '%s'", level,
                  stats->level_stack[level].name);
      return;
   }

   /* Push onto level stack - allocate memory for name */
   size_t name_len                = strlen(formatted_name) + 1;
   stats->level_stack[level].name = (const char *)malloc(name_len);
   if (stats->level_stack[level].name)
   {
      memcpy((void *)stats->level_stack[level].name, formatted_name, name_len);
   }
   stats->level_stack[level].level      = level;
   stats->level_stack[level].start_time = MPI_Wtime();
   if (level >= stats->level_depth)
   {
      stats->level_depth = level + 1;
   }

   /* Record solve index at start for this level */
   stats->level_active |= (1 << level);
   stats->level_current_id[level]++;
   stats->level_solve_start[level] = stats->ls_counter;

   /* Commenting out for now to avoid multiple caliper regions for each solve call */
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelEnd - End hierarchical annotation
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Helper: Ensure level array capacity (uses fixed capacity, reallocates if needed)
 *--------------------------------------------------------------------------*/

static void
EnsureLevelCapacity(Stats *stats, int level)
{
   /* Simple approach: reallocate if count reaches capacity */
   int count    = stats->level_count[level];
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
         stats->level_entries[level], (size_t)new_capacity * sizeof(LevelEntry));
      if (new_ptr)
      {
         stats->level_entries[level] = new_ptr;
      }
   }
}

/*--------------------------------------------------------------------------
 * StatsAnnotateLevelEnd - End hierarchical annotation
 *--------------------------------------------------------------------------*/

void
StatsAnnotateLevelEnd(Stats *stats, int level, const char *name)
{
   if (!stats)
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
   if (stats->level_stack[level].name == NULL ||
       strcmp(stats->level_stack[level].name, formatted_name) != 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Level %d annotation mismatch: expected '%s', got '%s'", level,
                  stats->level_stack[level].name ? stats->level_stack[level].name
                                                 : "NULL",
                  formatted_name);
      return;
   }

   /* Save level entry if this level is active */
   if (stats->level_active & (1 << level))
   {
      EnsureLevelCapacity(stats, level);

      int idx                                      = stats->level_count[level];
      stats->level_entries[level][idx].id          = stats->level_current_id[level];
      stats->level_entries[level][idx].solve_start = stats->level_solve_start[level];
      stats->level_entries[level][idx].solve_end   = stats->ls_counter;

      stats->level_count[level]++;
      stats->level_active &= ~(1 << level);
   }

   /* Commenting out for now to avoid multiple caliper regions for each solve call */

   /* Pop from level stack - free allocated memory */
   if (stats->level_stack[level].name)
   {
      free((void *)stats->level_stack[level].name);
      stats->level_stack[level].name = NULL;
   }
   stats->level_stack[level].level = -1;

   /* Update depth if needed */
   if (level == stats->level_depth - 1)
   {
      while (stats->level_depth > 0 &&
             stats->level_stack[stats->level_depth - 1].name == NULL)
      {
         stats->level_depth--;
      }
   }
}

/*--------------------------------------------------------------------------
 * StatsTimerSetMilliseconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetMilliseconds(Stats *stats)
{
   if (!stats)
   {
      return;
   }
   stats->use_millisec = true;
   stats->time_factor  = 1000.0;
}

/*--------------------------------------------------------------------------
 * StatsTimerSetSeconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetSeconds(Stats *stats)
{
   if (!stats)
   {
      return;
   }
   stats->use_millisec = false;
   stats->time_factor  = 1.0;
}

/*--------------------------------------------------------------------------
 * StatsIterSet
 *--------------------------------------------------------------------------*/

void
StatsIterSet(Stats *stats, int num_iters)
{
   if (!stats)
   {
      return;
   }
   stats->iters[stats->counter] = num_iters;
}

/*--------------------------------------------------------------------------
 * StatsInitialResNormSet
 *--------------------------------------------------------------------------*/

void
StatsInitialResNormSet(Stats *stats, double r0norm)
{
   if (!stats)
   {
      return;
   }
   stats->r0norms[stats->counter] = r0norm;
}

/*--------------------------------------------------------------------------
 * StatsRelativeResNormSet
 *--------------------------------------------------------------------------*/

void
StatsRelativeResNormSet(Stats *stats, double rrnorm)
{
   if (!stats)
   {
      return;
   }
   stats->rrnorms[stats->counter] = rrnorm;
}

/*--------------------------------------------------------------------------
 * StatsPrint
 *--------------------------------------------------------------------------*/

void
StatsPrint(const Stats *stats, int print_level)
{
   if (!stats || print_level < 1)
   {
      return;
   }

   const char *scale = ((int)stats->use_millisec) ? "[ms]" : "[s]";

   PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH);
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   PrintDivisor();
   PrintHeader(scale);
   PrintDivisor();

   /* Print statistics for each entry that had a solve */
   /* This filters out Newton iterations that broke before solving */
   /* Use a display index to avoid gaps in the entry column */
   int max_entry   = stats->counter;
   int display_idx = 0;
   for (int i = 0; i <= max_entry; i++)
   {
      /* Only print entries that had a solve (iterations > 0 or solve time > 0) */
      if (stats->iters[i] > 0 || stats->solve[i] > 0.0)
      {
         PrintEntryWithIndex(stats, i, display_idx);
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
         if (stats->iters[i] <= 0 && stats->solve[i] <= 0.0)
         {
            continue;
         }

         double b =
            stats->time_factor * (stats->dofmap[i] + stats->matrix[i] + stats->rhs[i]);
         double s  = stats->time_factor * stats->prec[i];
         double v  = stats->time_factor * stats->solve[i];
         double r0 = stats->r0norms[i];
         double rr = stats->rrnorms[i];
         int    it = stats->iters[i];

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
StatsGetLinearSystemID(const Stats *stats)
{
   if (!stats)
   {
      return -1;
   }
   /* Return 0-based linear system ID */
   return stats->ls_counter - 1;
}

/*--------------------------------------------------------------------------
 * StatsSetNumReps
 *--------------------------------------------------------------------------*/

void
StatsSetNumReps(Stats *stats, int num_reps)
{
   if (!stats)
   {
      return;
   }
   stats->num_reps = num_reps;
}

/*--------------------------------------------------------------------------
 * StatsSetNumLinearSystems
 *--------------------------------------------------------------------------*/

void
StatsSetNumLinearSystems(Stats *stats, int num_systems)
{
   if (!stats)
   {
      return;
   }
   stats->num_systems = num_systems;
}

/*--------------------------------------------------------------------------
 * StatsGetLastIter
 *--------------------------------------------------------------------------*/

int
StatsGetLastIter(const Stats *stats)
{
   if (!stats)
   {
      return -1;
   }
   return stats->iters[stats->counter];
}

/*--------------------------------------------------------------------------
 * StatsGetLastSetupTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSetupTime(const Stats *stats)
{
   if (!stats)
   {
      return 0.0;
   }
   return stats->prec[stats->counter] * stats->time_factor;
}

/*--------------------------------------------------------------------------
 * StatsGetLastSolveTime
 *--------------------------------------------------------------------------*/

double
StatsGetLastSolveTime(const Stats *stats)
{
   if (!stats)
   {
      return 0.0;
   }
   return stats->solve[stats->counter] * stats->time_factor;
}

/*--------------------------------------------------------------------------
 * StatsLevelGetCount - Get number of entries at a level
 *--------------------------------------------------------------------------*/

int
StatsLevelGetCount(const Stats *stats, int level)
{
   if (!stats || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return 0;
   }
   return stats->level_count[level];
}

/*--------------------------------------------------------------------------
 * StatsLevelGetEntry - Get level entry by index
 *--------------------------------------------------------------------------*/

int
StatsLevelGetEntry(const Stats *stats, int level, int index, LevelEntry *entry)
{
   if (!stats || !entry || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return -1;
   }

   if (index < 0 || index >= stats->level_count[level])
   {
      return -1;
   }

   *entry = stats->level_entries[level][index];
   return 0;
}

/*--------------------------------------------------------------------------
 * Helper: Compute aggregates from solve index range
 *--------------------------------------------------------------------------*/

static void
ComputeLevelStats(const Stats *stats, const LevelEntry *entry, int *num_solves,
                  int *linear_iters, double *setup_time, double *solve_time)
{
   int    n_solves = 0;
   int    l_iters  = 0;
   double s_time   = 0.0;
   double p_time   = 0.0;

   for (int i = entry->solve_start; i < entry->solve_end; i++)
   {
      l_iters += stats->iters[i];
      p_time += stats->prec[i];
      s_time += stats->solve[i];
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
StatsLevelPrint(const Stats *stats, int level)
{
   if (!stats || level < 0 || level >= STATS_MAX_LEVELS)
   {
      return;
   }

   int count = stats->level_count[level];
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
      const LevelEntry *entry      = &stats->level_entries[level][i];
      int               num_solves = 0, linear_iters = 0;
      double            setup_time = 0.0, solve_time = 0.0;

      ComputeLevelStats(stats, entry, &num_solves, &linear_iters, &setup_time,
                        &solve_time);

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
