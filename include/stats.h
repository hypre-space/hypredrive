/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef STATS_HEADER
#define STATS_HEADER

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include "error.h"
#include "utils.h"

#define STATS_NUM_ENTRIES 7

/*--------------------------------------------------------------------------
 * Stats struct
 *--------------------------------------------------------------------------*/

typedef struct Stats_struct {
   int       capacity;
   int       counter;
   int       reps;
   int       ls_counter;
   int       num_reps;
   int       num_systems;

   double   *matrix;
   double   *rhs;
   double   *dofmap;

   int      *iters;
   double   *prec;
   double   *solve;
   double   *rrnorms;

   double    initialize;
   double    finalize;
   double    reset_x0;

   double    time_factor;
   bool      use_millisec;
} Stats;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void StatsCreate(void);
void StatsDestroy(void);
void StatsTimerStart(const char*);
void StatsTimerFinish(const char*);
void StatsIterSet(int);
void StatsTimerSetMilliseconds(void);
void StatsTimerSetSeconds(void);
void StatsRelativeResNormSet(double);
void StatsPrint(int);
int  StatsGetLinearSystemID(void);
void StatsSetNumReps(int);
void StatsSetNumLinearSystems(int);

#endif /* STATS_HEADER */
