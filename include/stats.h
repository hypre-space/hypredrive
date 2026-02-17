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
#include "error.h"
#include "utils.h"

/* Undefine autotools package macros from hypre */
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION
#include "HYPREDRV_config.h"

/* Maximum number of hierarchical annotation levels */
enum StatsConstants
{
   STATS_MAX_LEVELS        = 4,
   STATS_TIMESTEP_CAPACITY = 64
};

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
 * Per-level entry (stats computed on-demand from solve index range)
 *--------------------------------------------------------------------------*/

typedef struct
{
   int id;          /* 1-based entry ID within this level */
   int solve_start; /* First solve index for this entry */
   int solve_end;   /* One past last solve index */
} LevelEntry;

/*--------------------------------------------------------------------------
 * Stats struct
 *--------------------------------------------------------------------------*/

typedef struct Stats_struct
{
   /* Capacity and counters */
   int capacity;
   int counter;        /* Current entry index */
   int reps;           /* Current repetition counter */
   int ls_counter;     /* Linear system counter (increments on "matrix" annotation) */
   int matrix_counter; /* Counter value when last "matrix" annotation was called */
   int num_reps;       /* Number of repetitions per linear system */
   int num_systems;    /* Number of linear systems (-1 if unknown) */

   /* Hierarchical annotation stack */
   AnnotationContext level_stack[STATS_MAX_LEVELS];
   int               level_depth; /* Current depth in hierarchy (0 = no active levels) */

   /* Timing arrays (indexed by counter) */
   double *matrix;      /* Matrix assembly time */
   double *rhs;         /* RHS assembly time */
   double *dofmap;      /* DOF map setup time */
   double *prec;        /* Preconditioner setup time */
   double *solve;       /* Linear solver time */
   double *rrnorms;     /* Relative residual norms */
   double *r0norms;     /* Initial residual norms (absolute) */
   int    *iters;       /* Iteration counts */
   int    *entry_ls_id; /* Linear system id per entry (for build-time printing) */

   /* Global timers */
   double initialize;
   double finalize;
   double reset_x0;

   /* Output formatting */
   double time_factor;
   bool   use_millisec;

   /* Per-level statistics (stats computed on-demand from solve index range) */
   int         level_count[STATS_MAX_LEVELS];   /* Number of entries per level */
   LevelEntry *level_entries[STATS_MAX_LEVELS]; /* Array of entries per level */

   /* Current state per level */
   int level_active;                        /* Bitmask: which levels are active */
   int level_current_id[STATS_MAX_LEVELS];  /* Current entry ID per level */
   int level_solve_start[STATS_MAX_LEVELS]; /* Solve index when level began */
} Stats;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

/* Stats object lifecycle */
Stats *StatsCreate(void);
void   StatsDestroy(Stats **stats_ptr);

/* Annotation functions */
void StatsAnnotate(Stats *stats, HYPREDRV_AnnotateAction action, const char *name);
void StatsAnnotateV(Stats *stats, HYPREDRV_AnnotateAction action, const char *name,
                    va_list args);
void StatsAnnotateLevelBegin(Stats *stats, int level, const char *name);
void StatsAnnotateLevelEnd(Stats *stats, int level, const char *name);

/* Timer configuration */
void StatsTimerSetMilliseconds(Stats *stats);
void StatsTimerSetSeconds(Stats *stats);

/* Statistics setters */
void StatsIterSet(Stats *stats, int);
void StatsInitialResNormSet(Stats *stats, double);
void StatsRelativeResNormSet(Stats *stats, double);
void StatsSetNumReps(Stats *stats, int);
void StatsSetNumLinearSystems(Stats *stats, int);

/* Statistics getters */
int    StatsGetLinearSystemID(const Stats *stats);
int    StatsGetLastIter(const Stats *stats);
double StatsGetLastSetupTime(const Stats *stats);
double StatsGetLastSolveTime(const Stats *stats);

/* Level statistics (populated automatically from level annotations) */
int  StatsLevelGetCount(const Stats *stats, int level);
int  StatsLevelGetEntry(const Stats *stats, int level, int index, LevelEntry *entry);
void StatsLevelPrint(const Stats *stats, int level);

/* Output */
void StatsPrint(const Stats *stats, int);

#endif /* STATS_HEADER */
