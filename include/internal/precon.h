/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef PRECON_HEADER
#define PRECON_HEADER

#include <stddef.h>
#include <stdint.h>
#include "internal/ads.h"
#include "internal/amg.h"
#include "internal/ams.h"
#include "internal/compatibility.h"
#include "internal/field.h"
#include "internal/fsai.h"
#include "internal/ilu.h"
#include "internal/mgr.h"
#include "internal/object_fwd.h"
#include "internal/precon_reuse.h"
#include "internal/slots.h"
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
   const hypredrv_MatrixSlot
      *coarse_schur; /* per-level app-provided MGR coarse operators */
} PreconOperators;

/* Driver operator registration is collective on a live object's communicator,
 * works before parsing, and preserves caller diagnostics. Method-specific
 * implementations remain in their AMS/ADS/MGR modules. */
typedef enum
{
   PRECON_OPERATOR_GRADIENT,
   PRECON_OPERATOR_CURL
} hypredrv_DiscreteOperator;
uint32_t hypredrv_PreconSetDiscreteOperator(hypredrv_t *, hypredrv_DiscreteOperator,
                                            HYPRE_IJMatrix);
uint32_t hypredrv_PreconSetCoordinates(hypredrv_t *, HYPRE_IJVector, HYPRE_IJVector,
                                       HYPRE_IJVector);
uint32_t hypredrv_PreconSetCoarseSchur(hypredrv_t *, int, HYPRE_IJMatrix);

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

struct PreconOps_struct;

typedef struct hypre_Precon_struct
{
   HYPRE_Solver main;
   MPI_Comm     comm; /* borrowed from creation, also used for setup preflight */
   precon_t     method;
   const struct PreconOps_struct *ops; /* validated at creation */
   /* Native setup has run, even if it reported failure. MGR teardown uses
    * this ownership fact; distributed workflow readiness is tracked separately. */
   HYPRE_Int setup_called;
   /* For AMS/ADS: 1 when every operator input the method requires (discrete
    * gradient / discrete curl / coordinate vectors) has a native hypre object
    * at create time. Always 1 for methods that need no operator inputs. Setup must abort
    * with a clean error when this is 0, otherwise hypre dereferences NULL. */
   HYPRE_Int operators_ok;
   /* AMG/MGR runtime state belongs to the configuration used at creation.
    * That storage must remain live until this wrapper is destroyed. */
   precon_args         *args;
   struct Stats_struct *stats;
} hypre_Precon;

typedef hypre_Precon *HYPRE_Precon;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray       hypredrv_PreconGetValidKeys(void);
StrIntMapArray hypredrv_PreconGetValidTypeIntMap(void);
void           hypredrv_PreconSetDefaultArgs(precon_args *);
/* Initialize fresh method arguments from a real YAML body (including a
 * sequence item); an empty body selects defaults. */
void hypredrv_PreconSetMethodArgs(precon_t, precon_args *, YAMLnode *);
void hypredrv_PreconArgsSetDefaultsForMethod(precon_t, precon_args *);
void hypredrv_PreconArgsSetDefaultsForName(precon_t, const char *, precon_args *);
void hypredrv_PreconArgsDestroyOwnedConfig(precon_t, precon_args *);
void hypredrv_PreconArgsDestroyRuntimeState(precon_t, precon_args *);
void hypredrv_PreconCreate(MPI_Comm, precon_t, precon_args *, IntArray *, HYPRE_IJVector,
                           HYPRE_Precon *, const Stats *, int, const PreconOperators *);
/* Returns nonzero when the configured preconditioner has a complete device
 * implementation. On failure, writes an actionable description to reason. */
int hypredrv_PreconSupportsDevice(precon_t, const precon_args *, char *reason,
                                  size_t reason_size);

/* Collective setup preflight on the preconditioner's creation communicator.
 * Checks operator inputs and the matrix before any rank enters hypre setup.
 * Returns nonzero only when all ranks may proceed. */
int hypredrv_PreconValidateSetup(HYPRE_Precon, HYPRE_ParCSRMatrix);
/* Setup callbacks follow collective PreconValidateSetup admission. They receive
 * the wrapper so timing and native ownership state apply equally to standalone,
 * ordinary Krylov and nested Krylov setup. */
HYPRE_Int hypredrv_PreconKrylovSetup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector,
                                     HYPRE_ParVector);
HYPRE_Int hypredrv_PreconKrylovSolve(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector,
                                     HYPRE_ParVector);

void hypredrv_PreconSetup(HYPRE_Precon, HYPRE_IJMatrix);
void hypredrv_PreconApply(HYPRE_Precon, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
void hypredrv_PreconDestroy(HYPRE_Precon *, const Stats *, int);

#endif /* PRECON_HEADER */
