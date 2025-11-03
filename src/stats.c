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
#define REALLOC(_d, _e)                                                                \
   do                                                                                  \
   {                                                                                   \
      _d *ptr =                                                                        \
         (_d *)realloc((void *)global_stats->_e, global_stats->capacity * sizeof(_d)); \
      memset(ptr + global_stats->capacity - REALLOC_EXPAND_FACTOR, 0,                  \
             REALLOC_EXPAND_FACTOR * sizeof(_d));                                      \
      global_stats->_e = (ptr) ? ptr : global_stats->_e;                               \
   } while (0);
#define STATS_TIMES_START_ENTRY(_e)    \
   if (!strcmp(name, #_e))             \
   {                                   \
      global_stats->_e -= MPI_Wtime(); \
      return;                          \
   }
#define STATS_TIMES_STOP_ENTRY(_e)     \
   if (!strcmp(name, #_e))             \
   {                                   \
      global_stats->_e += MPI_Wtime(); \
      return;                          \
   }
#define STATS_TIMES_START_VEC_ENTRY(_e)                       \
   if (!strcmp(name, #_e))                                    \
   {                                                          \
      global_stats->_e[global_stats->counter] -= MPI_Wtime(); \
      return;                                                 \
   }
#define STATS_TIMES_START_VEC_ENTRY_ALIAS(_e, _a)             \
   if (!strcmp(name, #_a))                                    \
   {                                                          \
      global_stats->_e[global_stats->counter] -= MPI_Wtime(); \
      return;                                                 \
   }
#define STATS_TIMES_STOP_VEC_ENTRY(_e)                        \
   if (!strcmp(name, #_e))                                    \
   {                                                          \
      global_stats->_e[global_stats->counter] += MPI_Wtime(); \
      return;                                                 \
   }
#define STATS_TIMES_STOP_VEC_ENTRY_ALIAS(_e, _a)              \
   if (!strcmp(name, #_a))                                    \
   {                                                          \
      global_stats->_e[global_stats->counter] += MPI_Wtime(); \
      return;                                                 \
   }
#define STATS_PRINT_DIVISOR()                               \
   printf("+------------");                                 \
   for (size_t i = 0; i < 4; i++) printf("+-------------"); \
   printf("+-------------+\n");
#define STATS_PRINT_HEADER(_s, _t, _b)                                    \
   printf("|%11s ", _t[0]);                                               \
   for (size_t i = 1; i < 6; i++) printf("|%12s ", _t[i]);                \
   printf("|\n");                                                         \
   printf("|%11s ", _b[0]);                                               \
   printf("|%*s %*s ", 11 - (int)strlen(_s), _b[1], (int)strlen(_s), _s); \
   printf("|%*s %*s ", 11 - (int)strlen(_s), _b[2], (int)strlen(_s), _s); \
   printf("|%*s %*s ", 11 - (int)strlen(_s), _b[3], (int)strlen(_s), _s); \
   printf("|%12s ", _b[4]);                                               \
   printf("|%12s ", _b[5]);                                               \
   printf("|\n");
#define STATS_PRINT_ENTRY(_t, _n)                                                        \
   if ((_t)->num_systems < 0 || !(_n % (((_t)->counter + 1) / (_t)->num_systems)))       \
   {                                                                                     \
      printf(                                                                            \
         "| %10d | %11.3f | %11.3f | %11.3f | %11.2e |  %10d |\n", (_n),                 \
         (_t)->time_factor *((_t)->dofmap[(_n)] + (_t)->matrix[(_n)] + (_t)->rhs[(_n)]), \
         (_t)->time_factor *((_t)->prec[(_n)]), (_t)->time_factor *((_t)->solve[(_n)]),  \
         (_t)->rrnorms[(_n)], (_t)->iters[(_n)]);                                        \
   }                                                                                     \
   else                                                                                  \
   {                                                                                     \
      printf("| %10d |             |", (_n));                                            \
      printf(" %11.3f | %11.3f | %11.2e |  %10d |\n",                                    \
             (_t)->time_factor *((_t)->prec[(_n)]),                                      \
             (_t)->time_factor *((_t)->solve[(_n)]), (_t)->rrnorms[(_n)],                \
             (_t)->iters[(_n)]);                                                         \
   }

/*--------------------------------------------------------------------------
 * StatsCreate
 *--------------------------------------------------------------------------*/

void
StatsCreate(void)
{
   int capacity = REALLOC_EXPAND_FACTOR;

   if (global_stats) return;

   global_stats = (Stats *)malloc(sizeof(Stats));

   global_stats->capacity     = capacity;
   global_stats->counter      = 0;
   global_stats->reps         = 0;
   global_stats->num_reps     = 1;
   global_stats->num_systems  = -1;
   global_stats->ls_counter   = 0;
   global_stats->use_millisec = false;
   global_stats->time_factor  = 1.0;

   /* Overall timers */
   global_stats->initialize = 0.0;
   global_stats->finalize   = 0.0;
   global_stats->reset_x0   = 0.0;

   /* Linear system loading (1st stage) */
   global_stats->dofmap = (double *)calloc(capacity, sizeof(double));
   global_stats->matrix = (double *)calloc(capacity, sizeof(double));
   global_stats->rhs    = (double *)calloc(capacity, sizeof(double));

   /* Linear system solution (2nd stage) */
   global_stats->iters   = (int *)calloc(capacity, sizeof(int));
   global_stats->prec    = (double *)calloc(capacity, sizeof(double));
   global_stats->solve   = (double *)calloc(capacity, sizeof(double));
   global_stats->rrnorms = (double *)calloc(capacity, sizeof(double));
}

/*--------------------------------------------------------------------------
 * StatsDestroy
 *--------------------------------------------------------------------------*/

void
StatsDestroy(void)
{
   if (global_stats)
   {
      free((global_stats)->dofmap);
      free((global_stats)->matrix);
      free((global_stats)->rhs);

      free((global_stats)->iters);
      free((global_stats)->prec);
      free((global_stats)->solve);
      free((global_stats)->rrnorms);

      free(global_stats);
      global_stats = NULL;
   }
}

/*--------------------------------------------------------------------------
 * StatsTimerStart
 *--------------------------------------------------------------------------*/

void
StatsTimerStart(const char *name)
{
   if (!global_stats) return;

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

   /* Reallocate arrays if needed */
   if (global_stats->counter >= global_stats->capacity)
   {
      global_stats->capacity += REALLOC_EXPAND_FACTOR;
      REALLOC(double, matrix);
      REALLOC(double, rhs);
      REALLOC(double, dofmap);
      REALLOC(double, prec);
      REALLOC(double, solve);
      REALLOC(double, rrnorms);
      REALLOC(int, iters);
   }

   STATS_TIMES_START_VEC_ENTRY_ALIAS(matrix, system)
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
   ErrorMsgAdd("Unknown timer key: '%s'", name);
}

/*--------------------------------------------------------------------------
 * StatsTimerStop
 *--------------------------------------------------------------------------*/

void
StatsTimerStop(const char *name)
{
   if (!global_stats) return;

   STATS_TIMES_STOP_VEC_ENTRY_ALIAS(matrix, system)
   STATS_TIMES_STOP_VEC_ENTRY(matrix)
   STATS_TIMES_STOP_VEC_ENTRY(rhs)
   STATS_TIMES_STOP_VEC_ENTRY(dofmap)
   STATS_TIMES_STOP_VEC_ENTRY(prec)
   STATS_TIMES_STOP_VEC_ENTRY(solve)
   STATS_TIMES_STOP_ENTRY(reset_x0)
   STATS_TIMES_STOP_ENTRY(initialize)
   STATS_TIMES_STOP_ENTRY(finalize)

   /* Set an error code if we haven't returned yet */
   ErrorCodeSet(ERROR_UNKNOWN_TIMING);
   ErrorMsgAdd("Unknown timer key: '%s'", name);
}

/*--------------------------------------------------------------------------
 * StatsTimerSetMilliseconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetMilliseconds(void)
{
   if (!global_stats) return;

   global_stats->use_millisec = true;
   global_stats->time_factor  = 1000.0;
}

/*--------------------------------------------------------------------------
 * StatsTimerSetSeconds
 *--------------------------------------------------------------------------*/

void
StatsTimerSetSeconds(void)
{
   if (!global_stats) return;

   global_stats->use_millisec = false;
   global_stats->time_factor  = 1.0;
}

/*--------------------------------------------------------------------------
 * StatsIterSet
 *--------------------------------------------------------------------------*/

void
StatsIterSet(int num_iters)
{
   if (!global_stats) return;

   global_stats->iters[global_stats->counter] = num_iters;
}

/*--------------------------------------------------------------------------
 * StatsRelativeResNormSet
 *--------------------------------------------------------------------------*/

void
StatsRelativeResNormSet(double rrnorm)
{
   if (!global_stats) return;

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

   const char *top[]    = {"", "LS build", "setup", "solve", "relative", ""};
   const char *bottom[] = {"Entry", "times", "times", "times", "res. norm", "iters"};
   const char *scale    = global_stats->use_millisec ? "[ms]" : "[s]";

   PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
   printf("\n\nSTATISTICS SUMMARY:\n\n");

   STATS_PRINT_DIVISOR()
   STATS_PRINT_HEADER(scale, top, bottom)
   STATS_PRINT_DIVISOR()

   /* Print statistics for each entry in the array */
   for (int i = 0; i < global_stats->counter + 1; i++)
   {
      STATS_PRINT_ENTRY(global_stats, i);
   }

   STATS_PRINT_DIVISOR()
   printf("\n");
}

/*--------------------------------------------------------------------------
 * StatsGetLinearSystemID
 *--------------------------------------------------------------------------*/

int
StatsGetLinearSystemID(void)
{
   if (!global_stats) return -1;
   return global_stats->ls_counter - 1;
}

/*--------------------------------------------------------------------------
 * StatsSetNumReps
 *--------------------------------------------------------------------------*/

void
StatsSetNumReps(int num_reps)
{
   if (!global_stats) return;
   global_stats->num_reps = num_reps;
}

/*--------------------------------------------------------------------------
 * StatsSetNumSystems
 *--------------------------------------------------------------------------*/

void
StatsSetNumLinearSystems(int num_systems)
{
   if (!global_stats) return;
   global_stats->num_systems = num_systems;
}
