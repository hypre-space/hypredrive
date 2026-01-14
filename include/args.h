/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef ARGS_HEADER
#define ARGS_HEADER

#include <HYPRE_utilities.h>
#include "linsys.h"
#include "precon.h"
#include "solver.h"
#include "utils.h"

/*--------------------------------------------------------------------------
 * Input arguments struct
 *--------------------------------------------------------------------------*/

typedef struct General_args_struct
{
   int    warmup;
   int    statistics;
   int    print_config_params;
   int    use_millisec;
   int    num_repetitions;
   double dev_pool_size;
   double uvm_pool_size;
   double host_pool_size;
   double pinned_pool_size;
} General_args;

typedef struct input_args_struct
{
   General_args general;

   LS_args ls;

   solver_args solver;
   solver_t    solver_method;

   precon_args precon;
   precon_t    precon_method;

   /* Preconditioner variants support */
   int          num_precon_variants;
   int          active_precon_variant;
   precon_t    *precon_methods;  /* Array of size num_precon_variants */
   precon_args *precon_variants; /* Array of size num_precon_variants */
} input_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void InputArgsCreate(bool, input_args **);
void InputArgsDestroy(input_args **);
void InputArgsRead(MPI_Comm, char *, int *, char **);
void InputArgsParse(MPI_Comm, bool, int, char **, input_args **);

#endif /* ARGS_HEADER */
