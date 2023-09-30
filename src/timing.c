/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "timing.h"

/* Global timings variable */
static ExecTimes *global_timings = NULL;

/* Local macros */
#define REALLOC_EXPAND_FACTOR 16
#define EXEC_TIMES_REALLOC(_t, _e, _n) \
   if (((_t)->size[(_n)] + 1) >= (_t)->capacity[(_n)]) \
   { \
      (_t)->capacity[(_n)] += REALLOC_EXPAND_FACTOR; \
      double *ptr = (double*) realloc((void*) (_t)->_e, (_t)->capacity[(_n)] * sizeof(double)); \
      memset(ptr + (_t)->capacity[(_n)] - REALLOC_EXPAND_FACTOR, 0, \
      REALLOC_EXPAND_FACTOR * sizeof(int)); \
      (_t)->_e = (ptr) ? ptr : (_t)->_e; \
   }
#define EXEC_TIMES_START_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_timings->_e -= MPI_Wtime(); return; }
#define EXEC_TIMES_FINISH_ENTRY(_e) \
   if (!strcmp(name, #_e)) { global_timings->_e += MPI_Wtime(); return; }
#define EXEC_TIMES_START_VEC_ENTRY(_e, _n) \
   if (!strcmp(name, #_e)) \
   { \
      EXEC_TIMES_REALLOC(global_timings, _e, _n) \
      global_timings->_e[global_timings->size[(_n)]++] -= MPI_Wtime(); \
      return; \
   }
#define EXEC_TIMES_FINISH_VEC_ENTRY(_e, _n) \
   if (!strcmp(name, #_e)) \
   { \
      global_timings->_e[global_timings->size[(_n)] - 1] += MPI_Wtime(); \
      return; \
   }
#define EXEC_TIMES_PRINT_DIVISOR() \
   printf("+------------+-------------+-------------+-------------+-------------+-------------+\n");
#define EXEC_TIMES_PRINT_ENTRY(_t, _n) \
   if (!(_n % ((_t)->size[4] / (_t)->ls_counter))) \
   { \
      printf("| %10ld | %11.2e | %11.2e | %11.2e | %11.2e | %11.2e |\n", \
             (_n), (_t)->dofmap[(_n)], (_t)->matrix[(_n)], (_t)->rhs[(_n)], \
             (_t)->prec[(_n)], (_t)->solve[(_n)]); \
   } \
   else \
   { \
      printf("| %10ld |             |             |             |", (_n)); \
      printf(" %11.2e | %11.2e |\n", (_t)->prec[(_n)], (_t)->solve[(_n)]); \
   }

/*--------------------------------------------------------------------------
 * ExecTimesCreate
 *--------------------------------------------------------------------------*/

ExecTimes*
ExecTimesCreate(int capacity)
{
   ExecTimes *timings = (ExecTimes*) malloc(sizeof(ExecTimes));

   for (size_t i = 0; i < EXEC_TIMES_NUM_ENTRIES; i++)
   {
      timings->capacity[i] = capacity;
      timings->size[i]     = 0;
   }
   timings->ls_counter = 0;

   /* Library initialization/finalization */
   timings->initialize = 0.0;
   timings->finalize   = 0.0;

   /* Linear system loading (1st stage) */
   timings->dofmap = (double*) malloc(timings->capacity[0] * sizeof(double));
   timings->matrix = (double*) malloc(timings->capacity[1] * sizeof(double));
   timings->rhs    = (double*) malloc(timings->capacity[2] * sizeof(double));

   /* Linear system solution (2nd stage) */
   timings->prec   = (double*) malloc(timings->capacity[3] * sizeof(double));
   timings->solve  = (double*) malloc(timings->capacity[4] * sizeof(double));

   return timings;
}

/*--------------------------------------------------------------------------
 * ExecTimesDestroy
 *--------------------------------------------------------------------------*/

void
ExecTimesDestroy(ExecTimes **timings_ptr)
{
   if (*timings_ptr)
   {
      free((*timings_ptr)->dofmap);
      free((*timings_ptr)->matrix);
      free((*timings_ptr)->rhs);

      free((*timings_ptr)->prec);
      free((*timings_ptr)->solve);

      free(*timings_ptr);
      *timings_ptr = NULL;
   }
}

/*--------------------------------------------------------------------------
 * ExecTimesStart
 *--------------------------------------------------------------------------*/

void
ExecTimesStart(const char *name)
{
   if (!global_timings)
   {
      global_timings = ExecTimesCreate(16);
   }

   /* Linear system loading counter (1st stage) */
   if (!strcmp(name, "matrix"))
   {
      global_timings->ls_counter++;
   }

   EXEC_TIMES_START_VEC_ENTRY(matrix, 0)
   EXEC_TIMES_START_VEC_ENTRY(rhs,    1)
   EXEC_TIMES_START_VEC_ENTRY(dofmap, 2)
   EXEC_TIMES_START_VEC_ENTRY(prec,   3)
   EXEC_TIMES_START_VEC_ENTRY(solve,  4)
   EXEC_TIMES_START_ENTRY(initialize)
   EXEC_TIMES_START_ENTRY(finalize)

   /* Set an error code if we haven't returned yet */
   ErrorCodeSet(ERROR_UNKNOWN_TIMING);
}

/*--------------------------------------------------------------------------
 * ExecTimesFinish
 *--------------------------------------------------------------------------*/

void
ExecTimesFinish(const char *name)
{
   if (!global_timings) return;

   EXEC_TIMES_FINISH_VEC_ENTRY(matrix, 0)
   EXEC_TIMES_FINISH_VEC_ENTRY(rhs,    1)
   EXEC_TIMES_FINISH_VEC_ENTRY(dofmap, 2)
   EXEC_TIMES_FINISH_VEC_ENTRY(prec,   3)
   EXEC_TIMES_FINISH_VEC_ENTRY(solve,  4)
   EXEC_TIMES_FINISH_ENTRY(initialize)
   EXEC_TIMES_FINISH_ENTRY(finalize)

   /* Set an error code if we haven't returned yet */
   ErrorCodeSet(ERROR_UNKNOWN_TIMING);
}

/*--------------------------------------------------------------------------
 * ExecTimesPrint
 *--------------------------------------------------------------------------*/

void
ExecTimesPrint(void)
{
   if (!global_timings) return;

   EXEC_TIMES_PRINT_DIVISOR();
   printf("|            |      dofmap |      matrix |         rhs |       setup |       solve |\n");
   printf("|      Entry |       times |       times |       times |       times |       times |\n");
   EXEC_TIMES_PRINT_DIVISOR();

   /* Print the timings for each entry in the array */
   for (size_t i = 0; i < global_timings->size[4]; i++)
   {
      EXEC_TIMES_PRINT_ENTRY(global_timings, i);
   }

   /* Printing a divisor line */
   EXEC_TIMES_PRINT_DIVISOR();
}
