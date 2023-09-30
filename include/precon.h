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
#include "fsai.h"
#include "yaml.h"
#include "field.h"

/*--------------------------------------------------------------------------
 * Preconditioner types enum
 *--------------------------------------------------------------------------*/

typedef enum precon_type_enum {
   PRECON_BOOMERAMG,
   PRECON_MGR,
   PRECON_ILU,
   PRECON_FSAI,
   PRECON_NONE
} precon_t;

/*--------------------------------------------------------------------------
 * Generic preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef union precon_args_union {
   AMG_args      amg;
   MGR_args      mgr;
   ILU_args      ilu;
   FSAI_args     fsai;
} precon_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray PreconGetValidKeys(void);
StrIntMapArray PreconGetValidValues(const char*);
StrIntMapArray PreconGetValidTypeIntMap(void);

void PreconSetArgsFromYAML(precon_args*, YAMLnode*); /* TODO: change name to PreconSetArgs */
void PreconCreate(precon_t, precon_args*, IntArray*, HYPRE_Solver*);
void PreconDestroy(precon_t, HYPRE_Solver*);

#endif
