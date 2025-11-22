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

enum
{
   STATS_NUM_ENTRIES = 7
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
 * Stats struct
 *--------------------------------------------------------------------------*/

typedef struct Stats_struct
{
   int capacity;
   int counter;
   int reps;
   int ls_counter;
   int num_reps;
   int num_systems;

   double *matrix;
   double *rhs;
   double *dofmap;

   int    *iters;
   double *prec;
   double *solve;
   double *rrnorms;

   double initialize;
   double finalize;
   double reset_x0;

   double time_factor;
   bool   use_millisec;
} Stats;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void StatsCreate(void);
void StatsDestroy(void);
void StatsAnnotate(HYPREDRV_AnnotateAction action, const char *name, ...);
void StatsAnnotateV(HYPREDRV_AnnotateAction action, const char *name, va_list args);
void StatsIterSet(int);
void StatsTimerSetMilliseconds(void);
void StatsTimerSetSeconds(void);
void StatsRelativeResNormSet(double);
void StatsPrint(int);
int  StatsGetLinearSystemID(void);
void StatsSetNumReps(int);
void StatsSetNumLinearSystems(int);

#endif /* STATS_HEADER */
