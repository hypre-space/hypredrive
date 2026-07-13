/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PRECON_HEADER
#define PRECON_HEADER

#include <stdint.h>
#include "internal/ads.h"
#include "internal/amg.h"
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

/*--------------------------------------------------------------------------
 * Generic preconditioner struct
 *--------------------------------------------------------------------------*/

typedef struct hypre_Precon_struct
{
   HYPRE_Solver main;
   precon_t     method;
   HYPRE_Int    is_setup;
   /* For AMS/ADS: 1 when every operator input the method requires (discrete
    * gradient / discrete curl / coordinate vectors) was supplied at create
    * time. Always 1 for methods that need no operator inputs. Setup must abort
    * with a clean error when this is 0, otherwise hypre dereferences NULL. */
   HYPRE_Int            operators_ok;
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
                                     HYPRE_Precon *, const Stats *, int, const PreconOperators *);
/* Returns 1 if the method needs externally supplied operator inputs
 * (AMS: discrete gradient + coordinates; ADS: + discrete curl), else 0. */
int hypredrv_PreconMethodRequiresOperators(precon_t);

/* Returns 1 if every operator input required by the method is present in ops
 * (ops may be NULL, which counts as "all missing" for AMS/ADS). Methods that
 * need no operators always return 1. */
int hypredrv_PreconOperatorsComplete(precon_t, const PreconOperators *);

/* Setup-time guard shared by the standalone and Krylov-embedded setup paths.
 * Returns nonzero (after raising an ERROR_MISSING_PRECON HYPREDRV error) when
 * the preconditioner must not be set up because its required operators are
 * missing; returns 0 when setup may proceed. */
int  hypredrv_PreconSetupOperatorGuard(HYPRE_Precon);
void hypredrv_PreconGetCallbacks(precon_t, HYPRE_PtrToParSolverFcn *,
                                 HYPRE_PtrToParSolverFcn *);

void hypredrv_PreconSetup(precon_t, HYPRE_Precon, HYPRE_IJMatrix);
void hypredrv_PreconApply(precon_t, HYPRE_Precon, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector);
void hypredrv_PreconDestroy(precon_t, precon_args *, HYPRE_Precon *, const Stats *, int);

#endif /* PRECON_HEADER */
