/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
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
   int       capacity[STATS_NUM_ENTRIES];
   int       size[STATS_NUM_ENTRIES];
   int       ls_counter;

   double   *matrix;
   double   *rhs;
   double   *dofmap;

   int      *iters;
   double   *prec;
   double   *solve;
   double   *rrnorms;

   double    initialize;
   double    finalize;
} Stats;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void StatsTimerStart(const char*);
void StatsTimerFinish(const char*);
void StatsIterSet(int);
void StatsRelativeResNormSet(double);
void StatsPrint(int);

#endif /* STATS_HEADER */
