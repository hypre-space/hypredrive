/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PRECON_HEADER
#define PRECON_HEADER

#include "amg.h"
#include "field.h"
#include "fsai.h"
#include "ilu.h"
#include "mgr.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * Preconditioner types enum
 *--------------------------------------------------------------------------*/

typedef enum precon_type_enum
{
   PRECON_BOOMERAMG,
   PRECON_MGR,
   PRECON_ILU,
   PRECON_FSAI,
   PRECON_NONE
} precon_t;

/*--------------------------------------------------------------------------
 * Generic preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct precon_args_struct
{
   union
   {
      AMG_args  amg;
      MGR_args  mgr;
      ILU_args  ilu;
      FSAI_args fsai;
   };
   int reuse;
} precon_args;

/*--------------------------------------------------------------------------
 * Generic preconditioner struct
 *--------------------------------------------------------------------------*/

typedef struct hypre_Precon_struct
{
   HYPRE_Solver main;
} hypre_Precon;

typedef hypre_Precon *HYPRE_Precon;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray       PreconGetValidKeys(void);
StrIntMapArray PreconGetValidValues(const char *);
StrIntMapArray PreconGetValidTypeIntMap(void);
void           PreconSetDefaultArgs(precon_args *);

void PreconSetArgsFromYAML(precon_args *,
                           YAMLnode *); /* TODO: change name to PreconSetArgs */
void PreconCreate(precon_t, precon_args *, IntArray *, HYPRE_Precon *);
void PreconSetup(precon_t, HYPRE_Precon, HYPRE_IJMatrix);
void PreconApply(precon_t, HYPRE_Precon, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
void PreconDestroy(precon_t, precon_args *, HYPRE_Precon *);

#endif /* PRECON_HEADER */
