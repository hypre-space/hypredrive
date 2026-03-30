/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef ARGS_HEADER
#define ARGS_HEADER

#include <HYPRE_utilities.h>
#include <stdint.h>
#include "internal/linsys.h"
#include "internal/precon.h"
#include "internal/scaling.h"
#include "internal/solver.h"
#include "internal/utils.h"

/*--------------------------------------------------------------------------
 * Input arguments struct
 *--------------------------------------------------------------------------*/

typedef struct General_args_struct
{
   char   name[MAX_FILENAME_LENGTH];
   char   statistics_filename[MAX_FILENAME_LENGTH];
   int    warmup;
   int    statistics;
   int    print_config_params;
   int    use_millisec;
   int    exec_policy;
   int    use_vendor_spgemm;
   int    use_vendor_spmv;
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

   Scaling_args scaling;

   precon_args      precon;
   precon_t         precon_method;
   PreconReuse_args precon_reuse;

   /* Preconditioner variants support */
   int          num_precon_variants;
   int          active_precon_variant;
   precon_t    *precon_methods;  /* Array of size num_precon_variants */
   precon_args *precon_variants; /* Array of size num_precon_variants */
} input_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void hypredrv_InputArgsCreate(bool, input_args **);
void hypredrv_InputArgsDestroy(input_args **);
void hypredrv_InputArgsRead(MPI_Comm, char *, int *, char **);
void hypredrv_InputArgsParse(MPI_Comm, bool, int, char **, input_args **);
void hypredrv_InputArgsParseWithObjectName(MPI_Comm, bool, int, char **, input_args **,
                                           const char *);
void hypredrv_InputArgsApplyPreconPreset(input_args *, const char *, int);

#endif /* ARGS_HEADER */
