/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef ARGS_HEADER
#define ARGS_HEADER

#include <HYPRE_utilities.h>
#include "precon.h"
#include "solver.h"
#include "linsys.h"
#include "utils.h"

/*--------------------------------------------------------------------------
 * Input arguments struct
 *--------------------------------------------------------------------------*/

typedef struct input_args_struct {
   HYPRE_Int     warmup; /* TODO: move this to separate struct */
   HYPRE_Int     num_repetitions; /* TODO: move this to separate struct */

   LS_args       ls;

   solver_args   solver;
   solver_t      solver_method;

   precon_args   precon;
   precon_t      precon_method;
} input_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void InputArgsCreate(input_args**);
void InputArgsDestroy(input_args**);
void InputArgsParse(MPI_Comm, int, char**, input_args**);

#endif
