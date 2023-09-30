/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "stats.h"

/* Global timings variable */
static Stats *global_stats = NULL;

/* Local macros */
#define REALLOC_EXPAND_FACTOR 16
#define REALLOC(_d, _t, _e, _n) \
   do { \
      if (((_t)->size[(_n)] + 1) >= (_t)->capacity[(_n)]) \
      { \
         (_t)->capacity[(_n)] += REALLOC_EXPAND_FACTOR; \
         _d *ptr = (_d*) realloc((void*) (_t)->_e, (_t)->capacity[(_n)] * sizeof(_d)); \
         memset(ptr + (_t)->capacity[(_n)] - REALLOC_EXPAND_FACTOR, 0, \
         REALLOC_EXPAND_FACTOR * sizeof(_d)); \
         (_t)->_e = (ptr) ? ptr : (_t)->_e; \
      } \
   } while (0);
#define STATS_ITERS_REALLOC(_t) \
   REALLOC(int, _t, iters, 5)
#define STATS_ITERS_SET(_i) \
   STATS_ITERS_REALLOC(global_stats) \
   global_stats->iters[global_stats->size[5]++] = (_i);
#define STATS_TIMES_REALLOC(_t, _e, _n) \
   REALLOC(double, _t, _e, _n)
#define STATS_TIMES_START_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_stats->_e -= MPI_Wtime(); return; }
#define STATS_TIMES_FINISH_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_stats->_e += MPI_Wtime(); return; }
#define STATS_TIMES_START_VEC_ENTRY(_e, _n) \
   if (!strcmp(name, #_e)) \
   { \
      STATS_TIMES_REALLOC(global_stats, _e, _n) \
      global_stats->_e[global_stats->size[(_n)]++] -= MPI_Wtime(); \
      return; \
   }
#define STATS_TIMES_FINISH_VEC_ENTRY(_e, _n) \
   if (!strcmp(name, #_e)) \
   { \
      global_stats->_e[global_stats->size[(_n)] - 1] += MPI_Wtime(); \
      return; \
   }
#define STATS_TIMES_PRINT_DIVISOR() \
   printf("+------------"); \
   for (size_t i = 0; i < 5; i++) printf("+-------------"); \
   printf("+-------------+\n");
#define STATS_TIMES_PRINT_HEADER(_t, _b) \
   printf("|%11s ", _t[0]); \
   for (size_t i = 1; i < 7; i++) printf("|%12s ", _t[i]); \
   printf("|\n"); \
   printf("|%11s ", _b[0]); \
   for (size_t i = 1; i < 7; i++) printf("|%12s ", _b[i]); \
   printf("|\n");
#define STATS_TIMES_PRINT_ENTRY(_t, _n) \
   if (!(_n % ((_t)->size[4] / (_t)->ls_counter))) \
   { \
      printf("| %10ld | %11.2e | %11.2e | %11.2e | %11.2e | %11.2e |  %10d |\n", \
             (_n), (_t)->dofmap[(_n)], (_t)->matrix[(_n)], (_t)->rhs[(_n)], \
             (_t)->prec[(_n)], (_t)->solve[(_n)], (_t)->iters[(_n)]); \
   } \
   else \
   { \
      printf("| %10ld |             |             |             |", (_n)); \
      printf(" %11.2e | %11.2e |  %10d |\n", \
             (_t)->prec[(_n)], (_t)->solve[(_n)], (_t)->iters[(_n)]); \
   }

/*--------------------------------------------------------------------------
 * StatsCreate
 *--------------------------------------------------------------------------*/

Stats*
StatsCreate(int capacity)
{
   Stats *stats = (Stats*) malloc(sizeof(Stats));

   for (size_t i = 0; i < STATS_TIMES_NUM_ENTRIES; i++)
   {
      stats->capacity[i] = capacity;
      stats->size[i]     = 0;
   }
   stats->ls_counter = 0;

   /* Library initialization/finalization */
   stats->initialize = 0.0;
   stats->finalize   = 0.0;

   /* Linear system loading (1st stage) */
   stats->dofmap = (double*) malloc(stats->capacity[0] * sizeof(double));
   stats->matrix = (double*) malloc(stats->capacity[1] * sizeof(double));
   stats->rhs    = (double*) malloc(stats->capacity[2] * sizeof(double));

   /* Linear system solution (2nd stage) */
   stats->iters  = (int*)    malloc(stats->capacity[3] * sizeof(int));
   stats->prec   = (double*) malloc(stats->capacity[4] * sizeof(double));
   stats->solve  = (double*) malloc(stats->capacity[5] * sizeof(double));

   return stats;
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

      free((*stats_ptr)->prec);
      free((*stats_ptr)->solve);

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
   if (!global_stats)
   {
      global_stats = StatsCreate(16);
   }

   /* Linear system loading counter (1st stage) */
   if (!strcmp(name, "matrix"))
   {
      global_stats->ls_counter++;
   }

   STATS_TIMES_START_VEC_ENTRY(matrix, 0)
   STATS_TIMES_START_VEC_ENTRY(rhs,    1)
   STATS_TIMES_START_VEC_ENTRY(dofmap, 2)
   STATS_TIMES_START_VEC_ENTRY(prec,   3)
   STATS_TIMES_START_VEC_ENTRY(solve,  4)
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
   if (!global_stats) return;

   STATS_TIMES_FINISH_VEC_ENTRY(matrix, 0)
   STATS_TIMES_FINISH_VEC_ENTRY(rhs,    1)
   STATS_TIMES_FINISH_VEC_ENTRY(dofmap, 2)
   STATS_TIMES_FINISH_VEC_ENTRY(prec,   3)
   STATS_TIMES_FINISH_VEC_ENTRY(solve,  4)
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
   if (!global_stats) return;

   STATS_ITERS_SET(num_iters)
}

/*--------------------------------------------------------------------------
 * StatsPrint
 *--------------------------------------------------------------------------*/

void
StatsPrint(int print_level)
{
   const char *top[] = {"", "dofmap", "matrix", "rhs", "setup", "solve", ""};
   const char *bottom[] = {"Entry", "times", "times", "times", "times", "times", "iters"};

   if (!global_stats || print_level < 1)
   {
      StatsDestroy(&global_stats);
      return;
   }

   for (size_t i = 0; i < 98; i++) printf("=");
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   STATS_TIMES_PRINT_DIVISOR()
   STATS_TIMES_PRINT_HEADER(top, bottom)
   STATS_TIMES_PRINT_DIVISOR()

   /* Print the timings for each entry in the array */
   for (size_t i = 0; i < global_stats->size[5]; i++)
   {
      STATS_TIMES_PRINT_ENTRY(global_stats, i);
   }

   /* Printing a divisor line */
   STATS_TIMES_PRINT_DIVISOR()
   printf("\n");

   /* Destroy global stats variable */
   StatsDestroy(&global_stats);
}
