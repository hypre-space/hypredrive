/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef SCALING_HEADER
#define SCALING_HEADER

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "internal/compatibility.h"
#include "internal/containers.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * Scaling type enum
 *--------------------------------------------------------------------------*/

typedef enum scaling_type_enum
{
   SCALING_RHS_L2,
   SCALING_DOFMAP_MAG,
   SCALING_DOFMAP_CUSTOM,
   SCALING_DOFMAP_ROW_CUSTOM,
   SCALING_DOFMAP_COL_CUSTOM,
   SCALING_DOFMAP_SIMILARITY_CUSTOM,
} scaling_type_t;

typedef enum scaling_vector_kind_enum
{
   SCALING_VECTOR_RHS,
   SCALING_VECTOR_UNKNOWN,
   /* A replacement guess follows the working solution's current scale. */
   SCALING_VECTOR_INITIAL_GUESS,
} scaling_vector_kind_t;

/*--------------------------------------------------------------------------
 * Scaling arguments struct
 *--------------------------------------------------------------------------*/

typedef struct Scaling_args_struct
{
   int            enabled;
   scaling_type_t type;
   DoubleArray   *custom_values; /* Diagonal weights for custom dofmap scaling */
} Scaling_args;

/*--------------------------------------------------------------------------
 * Scaling context (runtime state)
 *--------------------------------------------------------------------------*/

typedef struct Scaling_context_struct Scaling_context;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrIntMapArray hypredrv_ScalingGetValidValues(const char *);

void hypredrv_ScalingSetDefaultArgs(Scaling_args *);
void hypredrv_ScalingSetArgsFromYAML(void *, YAMLnode *);

void hypredrv_ScalingContextDestroy(Scaling_context **);
bool hypredrv_ScalingIsApplied(const Scaling_context *);

/* Prepare a fresh, validated runtime and publish it only after collective success.
 * Failure preserves the installed runtime. Restore applied scaling before preparing
 * another system. Disabled/unsupported scaling installs no runtime. */
bool hypredrv_ScalingPrepare(MPI_Comm, const Scaling_args *, Scaling_context **,
                             HYPRE_IJMatrix, HYPRE_IJVector, const IntArray *);
void hypredrv_ScalingApplyToVector(const Scaling_context *, HYPRE_IJVector,
                                   scaling_vector_kind_t);
void hypredrv_ScalingUndoOnVector(const Scaling_context *, HYPRE_IJVector,
                                  scaling_vector_kind_t);
void hypredrv_ScalingApplyToSystem(Scaling_context *, HYPRE_IJMatrix, HYPRE_IJMatrix,
                                   HYPRE_IJVector, HYPRE_IJVector);
/* Undo the exact handles bound by the preceding system apply. */
void hypredrv_ScalingUndoOnSystem(Scaling_context *);

#endif /* SCALING_HEADER */
