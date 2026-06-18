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
#include "internal/ads.h"
#include "internal/ams.h"
#include "internal/compatibility.h"
#include "internal/field.h"
#include "internal/fsai.h"
#include "internal/ilu.h"
#include "internal/mgr.h"
#include "internal/precon_reuse.h"
#include "internal/utils.h"
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
#include "internal/schwarz.h"
#endif
#include "internal/stats.h"
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
   PRECON_AMS,
   PRECON_ADS,
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   PRECON_SCHWARZ,
#endif
   PRECON_NONE,
} precon_t;

/*--------------------------------------------------------------------------
 * Operator inputs required by AMS/ADS (discrete gradient, discrete curl, and
 * vertex coordinate vectors). All handles are borrowed (owned by the caller).
 *--------------------------------------------------------------------------*/

typedef struct PreconOperators_struct
{
   HYPRE_IJMatrix G;        /* discrete gradient (AMS, ADS) */
   HYPRE_IJMatrix C;        /* discrete curl (ADS) */
   HYPRE_IJVector coord[3]; /* vertex coordinate vectors (AMS, ADS) */
} PreconOperators;

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
      AMS_args  ams;
      ADS_args  ads;
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      Schwarz_args schwarz;
#endif
   };
   int reuse;
} precon_args;

typedef precon_args Precon_args;
/*--------------------------------------------------------------------------
 * Generic preconditioner struct
 *--------------------------------------------------------------------------*/

typedef struct hypre_Precon_struct
{
   HYPRE_Solver         main;
   precon_t             method;
   HYPRE_Int            is_setup;
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
void           hypredrv_PreconArgsDestroyOwnedConfig(precon_t, precon_args *);
void           hypredrv_PreconArgsDestroyRuntimeState(precon_t, precon_args *);
void           hypredrv_PreconSetArgsFromYAML(precon_args *,
                                              YAMLnode *); /* TODO: change to PreconSetArgs */
void           hypredrv_PreconCreate(precon_t, precon_args *, IntArray *, HYPRE_IJVector,
                                     HYPRE_Precon *, const Stats *, int,
                                     const PreconOperators *);
void           hypredrv_PreconSetup(precon_t, HYPRE_Precon, HYPRE_IJMatrix);
void hypredrv_PreconApply(precon_t, HYPRE_Precon, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector);
void hypredrv_PreconDestroy(precon_t, precon_args *, HYPRE_Precon *, const Stats *, int);

#endif /* PRECON_HEADER */
