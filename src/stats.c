/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "stats.h"

/* Global timings variable */
static Stats *global_stats = NULL;

/* Local macros */
#define REALLOC_EXPAND_FACTOR 16
#define REALLOC(_d, _t, _e) \
   do { \
      if (((_t)->counter + 1) >= (_t)->capacity) \
      { \
         (_t)->capacity += REALLOC_EXPAND_FACTOR; \
         _d *ptr = (_d*) realloc((void*) (_t)->_e, (_t)->capacity * sizeof(_d)); \
         memset(ptr + (_t)->capacity - REALLOC_EXPAND_FACTOR, 0, \
                REALLOC_EXPAND_FACTOR * sizeof(_d)); \
         (_t)->_e = (ptr) ? ptr : (_t)->_e; \
      } \
   } while (0);
#define STATS_TIMES_START_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_stats->_e -= MPI_Wtime(); return; }
#define STATS_TIMES_FINISH_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_stats->_e += MPI_Wtime(); return; }
#define STATS_TIMES_START_VEC_ENTRY(_e) \
   if (!strcmp(name, #_e)) \
   { \
      REALLOC(double, global_stats, _e) \
      global_stats->_e[global_stats->counter] -= MPI_Wtime(); \
      return; \
   }
#define STATS_TIMES_FINISH_VEC_ENTRY(_e) \
   if (!strcmp(name, #_e)) \
   { \
      global_stats->_e[global_stats->counter] += MPI_Wtime(); \
      return; \
   }
#define STATS_PRINT_DIVISOR() \
   printf("+------------"); \
   for (size_t i = 0; i < 4; i++) printf("+-------------"); \
   printf("+-------------+\n");
#define STATS_PRINT_HEADER(_t, _b) \
   printf("|%11s ", _t[0]); \
   for (size_t i = 1; i < 6; i++) printf("|%12s ", _t[i]); \
   printf("|\n"); \
   printf("|%11s ", _b[0]); \
   for (size_t i = 1; i < 6; i++) printf("|%12s ", _b[i]); \
   printf("|\n");
#define STATS_PRINT_ENTRY(_t, _n) \
   if (!(_n % (((_t)->counter + 1) / (_t)->num_systems))) \
   { \
      printf("| %10ld | %11.3f | %11.3f | %11.3f | %11.2e |  %10d |\n", \
             (_n), (_t)->dofmap[(_n)] + (_t)->matrix[(_n)] + (_t)->rhs[(_n)], \
             (_t)->prec[(_n)], (_t)->solve[(_n)], (_t)->rrnorms[(_n)], \
             (_t)->iters[(_n)]); \
   } \
   else \
   { \
      printf("| %10ld |             |", (_n)); \
      printf(" %11.3f | %11.3f | %11.2e |  %10d |\n", \
             (_t)->prec[(_n)], (_t)->solve[(_n)], \
             (_t)->rrnorms[(_n)], (_t)->iters[(_n)]); \
   }

/*--------------------------------------------------------------------------
 * StatsCreate
 *--------------------------------------------------------------------------*/

void
StatsCreate(void)
{
   int capacity = REALLOC_EXPAND_FACTOR;

   if (global_stats) return;

   global_stats = (Stats*) malloc(sizeof(Stats));

   global_stats->capacity    = capacity;
   global_stats->counter     = 0;
   global_stats->reps        = 0;
   global_stats->num_reps    = 1;
   global_stats->num_systems = 1;
   global_stats->ls_counter  = 0;

   /* Overall timers */
   global_stats->initialize  = 0.0;
   global_stats->finalize    = 0.0;
   global_stats->reset_x0    = 0.0;

   /* Linear system loading (1st stage) */
   global_stats->dofmap      = (double*) calloc(capacity, sizeof(double));
   global_stats->matrix      = (double*) calloc(capacity, sizeof(double));
   global_stats->rhs         = (double*) calloc(capacity, sizeof(double));

   /* Linear system solution (2nd stage) */
   global_stats->iters       = (int*)    calloc(capacity, sizeof(int));
   global_stats->prec        = (double*) calloc(capacity, sizeof(double));
   global_stats->solve       = (double*) calloc(capacity, sizeof(double));
   global_stats->rrnorms     = (double*) calloc(capacity, sizeof(double));
}

/*--------------------------------------------------------------------------
 * StatsDestroy
 *--------------------------------------------------------------------------*/

void
StatsDestroy(Stats **stats_ptr)
{
   if (*stats_ptr)
   {
      free((*stats_ptr)->dofmap);
      free((*stats_ptr)->matrix);
      free((*stats_ptr)->rhs);

      free((*stats_ptr)->iters);
      free((*stats_ptr)->prec);
      free((*stats_ptr)->solve);
      free((*stats_ptr)->rrnorms);

      free(*stats_ptr);
      *stats_ptr = NULL;
   }
}

/*--------------------------------------------------------------------------
 * StatsTimerStart
 *--------------------------------------------------------------------------*/

void
StatsTimerStart(const char *name)
{
   /* Increase internal counters */
   if (!strcmp(name, "reset_x0"))
   {
      global_stats->reps++;
      global_stats->counter = (global_stats->ls_counter - 1) * global_stats->num_reps +
                              (global_stats->reps - 1);
   }
   else if (!strcmp(name, "matrix"))
   {
      global_stats->reps = 0;
      global_stats->ls_counter++;
      global_stats->counter = (global_stats->ls_counter - 1) * global_stats->num_reps;
   }

   /* Compute entry counter */
   STATS_TIMES_START_VEC_ENTRY(matrix)
   STATS_TIMES_START_VEC_ENTRY(rhs)
   STATS_TIMES_START_VEC_ENTRY(dofmap)
   STATS_TIMES_START_VEC_ENTRY(prec)
   STATS_TIMES_START_VEC_ENTRY(solve)
   STATS_TIMES_START_ENTRY(reset_x0)
   STATS_TIMES_START_ENTRY(initialize)
   STATS_TIMES_START_ENTRY(finalize)

   /* Set an error code if we haven't returned yet */
   ErrorCodeSet(ERROR_UNKNOWN_TIMING);
}

/*--------------------------------------------------------------------------
 * StatsTimerFinish
 *--------------------------------------------------------------------------*/

void
StatsTimerFinish(const char *name)
{
   STATS_TIMES_FINISH_VEC_ENTRY(matrix)
   STATS_TIMES_FINISH_VEC_ENTRY(rhs)
   STATS_TIMES_FINISH_VEC_ENTRY(dofmap)
   STATS_TIMES_FINISH_VEC_ENTRY(prec)
   STATS_TIMES_FINISH_VEC_ENTRY(solve)
   STATS_TIMES_FINISH_ENTRY(reset_x0)
   STATS_TIMES_FINISH_ENTRY(initialize)
   STATS_TIMES_FINISH_ENTRY(finalize)

   /* Set an error code if we haven't returned yet */
   ErrorCodeSet(ERROR_UNKNOWN_TIMING);
}

/*--------------------------------------------------------------------------
 * StatsIterSet
 *--------------------------------------------------------------------------*/

void
StatsIterSet(int num_iters)
{
   REALLOC(int, global_stats, iters);
   global_stats->iters[global_stats->counter] = num_iters;
}

/*--------------------------------------------------------------------------
 * StatsRelativeResNormSet
 *--------------------------------------------------------------------------*/

void
StatsRelativeResNormSet(double rrnorm)
{
   REALLOC(double, global_stats, rrnorms);
   global_stats->rrnorms[global_stats->counter] = rrnorm;
}

/*--------------------------------------------------------------------------
 * StatsPrint
 *--------------------------------------------------------------------------*/

void
StatsPrint(int print_level)
{
   const char *top[] = {"", "LS build", "setup", "solve", "relative", ""};
   const char *bottom[] = {"Entry", "times", "times", "times", "res. norm", "iters"};

   if (!global_stats || print_level < 1)
   {
      StatsDestroy(&global_stats);
      return;
   }

   PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   STATS_PRINT_DIVISOR()
   STATS_PRINT_HEADER(top, bottom)
   STATS_PRINT_DIVISOR()

   /* Print statistics for each entry in the array */
   for (size_t i = 0; i < global_stats->counter + 1; i++)
   {
      //printf("%f %f %f\n", global_stats->dofmap[i], global_stats->matrix[i], global_stats->rhs[i]);
      STATS_PRINT_ENTRY(global_stats, i);
   }

   STATS_PRINT_DIVISOR()
   printf("\n");

   /* Destroy global stats variable */
   StatsDestroy(&global_stats);
}

/*--------------------------------------------------------------------------
 * StatsGetLinearSystemID
 *--------------------------------------------------------------------------*/

int
StatsGetLinearSystemID(void)
{
   return global_stats->ls_counter - 1;
}

/*--------------------------------------------------------------------------
 * StatsSetNumReps
 *--------------------------------------------------------------------------*/

void
StatsSetNumReps(int num_reps)
{
   global_stats->num_reps = num_reps;
}

/*--------------------------------------------------------------------------
 * StatsSetNumSystems
 *--------------------------------------------------------------------------*/

void
StatsSetNumLinearSystems(int num_systems)
{
   global_stats->num_systems = num_systems;
}
