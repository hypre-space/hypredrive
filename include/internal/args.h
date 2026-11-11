/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef ARGS_HEADER
#define ARGS_HEADER

#include "internal/linsys.h"
#include "internal/object_fwd.h"
#include "internal/precon.h"
#include "internal/presets.h"
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
   int    device_lazy_init;
   int    exec_policy;
   int    use_vendor_spgemm;
   int    use_vendor_spmv;
   int    use_vendor_sptrans;
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

   /* Creation/parsing maintain a positive count and an active index in range.
    * Default-only args have no arrays yet; the active precon above is authoritative. */
   int          num_precon_variants;
   int          active_precon_variant;
   precon_t    *precon_methods;  /* Array of size num_precon_variants */
   precon_args *precon_variants; /* Array of size num_precon_variants */
} input_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

void hypredrv_InputArgsDestroy(input_args **);
/* Parse complete configuration text and publish only validated arguments.
 * Primary files are read on rank zero; source kind and failures are collective.
 * Valid output pointers are left NULL on failure. The diagnostic scope is kept. */
void hypredrv_InputArgsParseWithObjectName(MPI_Comm, bool, int, char **, input_args **,
                                           const char *);
/* Object configuration operations receive a live object from the public API.
 * Parsing stages complete arguments, schedule data and device compatibility
 * before retiring existing native consumers or updating statistics.
 * Variant and preset selection validate their inputs and selected methods
 * collectively before changing configuration or native consumers. Preset kind
 * is supplied by the public API. These operations preserve the diagnostic scope. */
uint32_t hypredrv_InputArgsParseObject(hypredrv_t *, int, char **);
uint32_t hypredrv_InputArgsSetPreconVariant(hypredrv_t *, int);
uint32_t hypredrv_InputArgsSetPreset(hypredrv_t *, hypredrv_PresetKind, const char *);

#endif /* ARGS_HEADER */
