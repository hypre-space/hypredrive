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
#include "utils.h"

/*--------------------------------------------------------------------------
 * Linear system arguments struct
 *--------------------------------------------------------------------------*/

typedef struct LS_args_struct {
   char          matrix_filename[MAX_FILENAME_LENGTH];
   char          precmat_filename[MAX_FILENAME_LENGTH];
   char          rhs_filename[MAX_FILENAME_LENGTH];
   char          x0_filename[MAX_FILENAME_LENGTH];
   char          sol_filename[MAX_FILENAME_LENGTH];
   char          dofmap_filename[MAX_FILENAME_LENGTH];
   HYPRE_Int     num_partitions;
   HYPRE_Int     init_guess_mode;
   HYPRE_Int     rhs_mode;
} LS_args;

/*--------------------------------------------------------------------------
 * Input arguments struct
 *--------------------------------------------------------------------------*/

typedef struct input_args_struct {
   HYPRE_Int     warmup;
   HYPRE_Int     num_repetitions;

   LS_args       ls;

   solver_args   solver;
   solver_t      solver_method;

   precon_args   precon;
   precon_t      precon_method;
} input_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

int InputArgsCreate(const char*, const char*, input_args**);
int InputArgsDestroy(input_args**);
int InputArgsParse(MPI_Comm, int, char**, input_args**);

#endif
