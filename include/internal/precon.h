/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PRECON_HEADER
#define PRECON_HEADER

#include <stdint.h>
#include "internal/amg.h"
#include "internal/field.h"
#include "internal/fsai.h"
#include "internal/ilu.h"
#include "internal/mgr.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * Preconditioner types enum
 *--------------------------------------------------------------------------*/

typedef enum precon_type_enum
{
   PRECON_BOOMERAMG,
   PRECON_MGR,
   PRECON_ILU,
   PRECON_FSAI,
   PRECON_NONE,
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

typedef precon_args Precon_args;

typedef struct PreconReuse_args_struct
{
   int       enabled;
   int       frequency;
   IntArray *linear_system_ids;
   int       per_timestep;
} PreconReuse_args;

/*--------------------------------------------------------------------------
 * Generic preconditioner struct
 *--------------------------------------------------------------------------*/

typedef struct hypre_Precon_struct
{
   HYPRE_Solver         main;
   precon_t             method;
   struct Stats_struct *stats;
} hypre_Precon;

typedef hypre_Precon *HYPRE_Precon;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray       hypredrv_PreconGetValidKeys(void);
StrIntMapArray hypredrv_PreconGetValidValues(const char *);
StrIntMapArray hypredrv_PreconGetValidTypeIntMap(void);
void           hypredrv_PreconSetDefaultArgs(precon_args *);
void           hypredrv_PreconArgsSetDefaultsForMethod(precon_t, precon_args *);
void           hypredrv_PreconReuseSetDefaultArgs(PreconReuse_args *);
void           hypredrv_PreconReuseDestroyArgs(PreconReuse_args *);
void           hypredrv_PreconReuseSetArgsFromYAML(PreconReuse_args *, YAMLnode *);
void           hypredrv_PreconReuseTimestepsClear(IntArray **);
uint32_t       hypredrv_PreconReuseTimestepsLoad(const PreconReuse_args *, const char *,
                                                 IntArray **);
int hypredrv_PreconReuseShouldRecompute(const PreconReuse_args *, const IntArray *,
                                        const struct Stats_struct *, int);

void hypredrv_PreconSetArgsFromYAML(precon_args *,
                                    YAMLnode *); /* TODO: change to PreconSetArgs */
void hypredrv_PreconCreate(precon_t, precon_args *, IntArray *, HYPRE_IJVector,
                           HYPRE_Precon *);
void hypredrv_PreconSetup(precon_t, HYPRE_Precon, HYPRE_IJMatrix);
void hypredrv_PreconApply(precon_t, HYPRE_Precon, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector);
void hypredrv_PreconDestroy(precon_t, precon_args *, HYPRE_Precon *);

#endif /* PRECON_HEADER */
