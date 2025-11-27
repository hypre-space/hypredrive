/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef STATS_HEADER
#define STATS_HEADER

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HYPREDRV_config.h"
#include "error.h"
#include "utils.h"

/* Maximum number of hierarchical annotation levels */
#define STATS_MAX_LEVELS 4

/* HYPREDRV_AnnotateAction enum - internal use only (not in public API) */
typedef enum
{
   HYPREDRV_ANNOTATE_BEGIN = 0,
   HYPREDRV_ANNOTATE_END   = 1
} HYPREDRV_AnnotateAction;

/*--------------------------------------------------------------------------
 * Caliper instrumentation macros
 *--------------------------------------------------------------------------*/

#ifdef HYPREDRV_USING_CALIPER

#ifdef __cplusplus
extern "C++"
{
#endif

#include <caliper/cali.h>

#ifdef __cplusplus
}
#endif

#define HYPREDRV_ANNOTATE_REGION_BEGIN(...)                                  \
   {                                                                         \
      char hypredrv__markname[1024];                                         \
      snprintf(hypredrv__markname, sizeof(hypredrv__markname), __VA_ARGS__); \
      CALI_MARK_BEGIN(hypredrv__markname);                                   \
   }

#define HYPREDRV_ANNOTATE_REGION_END(...)                                    \
   {                                                                         \
      char hypredrv__markname[1024];                                         \
      snprintf(hypredrv__markname, sizeof(hypredrv__markname), __VA_ARGS__); \
      CALI_MARK_END(hypredrv__markname);                                     \
   }

#else

#define HYPREDRV_ANNOTATE_REGION_BEGIN(...)
#define HYPREDRV_ANNOTATE_REGION_END(...)

#endif /* HYPREDRV_USING_CALIPER */

/*--------------------------------------------------------------------------
 * Hierarchical annotation context
 *--------------------------------------------------------------------------*/

typedef struct
{
   const char *name;
   double      start_time;
   int         level;
} AnnotationContext;

/*--------------------------------------------------------------------------
 * Stats struct
 *--------------------------------------------------------------------------*/

typedef struct Stats_struct
{
   /* Capacity and counters */
   int capacity;
   int counter;     /* Current entry index */
   int reps;        /* Current repetition counter */
   int ls_counter;  /* Linear system counter (increments on "matrix" annotation) */
   int num_reps;    /* Number of repetitions per linear system */
   int num_systems; /* Number of linear systems (-1 if unknown) */

   /* Hierarchical annotation stack */
   AnnotationContext level_stack[STATS_MAX_LEVELS];
   int               level_depth; /* Current depth in hierarchy (0 = no active levels) */

   /* Timing arrays (indexed by counter) */
   double *matrix;  /* Matrix assembly time */
   double *rhs;     /* RHS assembly time */
   double *dofmap;  /* DOF map setup time */
   double *prec;    /* Preconditioner setup time */
   double *solve;   /* Linear solver time */
   double *rrnorms; /* Relative residual norms */
   int    *iters;   /* Iteration counts */

   /* Global timers */
   double initialize;
   double finalize;
   double reset_x0;

   /* Output formatting */
   double time_factor;
   bool   use_millisec;
} Stats;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

/* Stats object lifecycle */
Stats *StatsCreate(void);
void   StatsDestroy(Stats **stats_ptr);
void   StatsSetContext(Stats *stats);
Stats *StatsGetContext(void);

/* Annotation functions */
void StatsAnnotate(HYPREDRV_AnnotateAction action, const char *name, ...);
void StatsAnnotateV(HYPREDRV_AnnotateAction action, const char *name, va_list args);
void StatsAnnotateLevelBegin(int level, const char *name, ...);
void StatsAnnotateLevelEnd(int level, const char *name, ...);

/* Timer configuration */
void StatsTimerSetMilliseconds(void);
void StatsTimerSetSeconds(void);

/* Statistics setters */
void StatsIterSet(int);
void StatsRelativeResNormSet(double);
void StatsSetNumReps(int);
void StatsSetNumLinearSystems(int);

/* Statistics getters */
int    StatsGetLinearSystemID(void);
int    StatsGetLastIter(void);
double StatsGetLastSetupTime(void);
double StatsGetLastSolveTime(void);

/* Output */
void StatsPrint(int);

#endif /* STATS_HEADER */
