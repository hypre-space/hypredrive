/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef PRECON_HEADER
#define PRECON_HEADER

#include "ilu.h"
#include "amg.h"
#include "mgr.h"

/*--------------------------------------------------------------------------
 * Preconditioner types enum
 *--------------------------------------------------------------------------*/

typedef enum precon_type_enum {
   PRECON_BOOMERAMG,
   PRECON_MGR,
   PRECON_ILU,
   PRECON_NONE
} precon_t;

/*--------------------------------------------------------------------------
 * Generic preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef union precon_args_union {
   MGR_args      mgr;
   AMG_args      amg;
   ILU_args      ilu;
} precon_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

int PreconSetDefaultArgs(precon_t, precon_args*);
int PreconSetArgsFromYAML(precon_t, precon_args*, YAMLnode*);
int PreconCreate(precon_t, precon_args*, HYPRE_IntArray*, HYPRE_Solver*);
int PreconDestroy(precon_t, HYPRE_Solver*);

#endif
