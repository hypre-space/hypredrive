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
#include "compatibility.h"
#include "containers.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * Scaling type enum
 *--------------------------------------------------------------------------*/

typedef enum scaling_type_enum
{
   SCALING_RHS_L2,
   SCALING_DOFMAP_MAG,
   SCALING_DOFMAP_CUSTOM
} scaling_type_t;

typedef enum scaling_vector_kind_enum
{
   SCALING_VECTOR_RHS,
   SCALING_VECTOR_UNKNOWN
} scaling_vector_kind_t;

/*--------------------------------------------------------------------------
 * Scaling arguments struct
 *--------------------------------------------------------------------------*/

typedef struct Scaling_args_struct
{
   int            enabled;
   scaling_type_t type;
   DoubleArray   *custom_values; /* Array of custom scaling values for dofmap_custom */
} Scaling_args;

/*--------------------------------------------------------------------------
 * Scaling context (runtime state)
 *--------------------------------------------------------------------------*/

typedef struct Scaling_context_struct
{
   int             enabled;
   scaling_type_t  type;
   int             is_applied;     /* 1 if scaling is currently applied to system */
   HYPRE_Complex   scalar_factor;  /* for rhs_l2 */
   HYPRE_ParVector scaling_vector; /* for dofmap */
   HYPRE_IJVector  scaling_ijvec;  /* IJ wrapper for scaling_vector */
} Scaling_context;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrIntMapArray ScalingGetValidValues(const char *);

void ScalingSetDefaultArgs(Scaling_args *);
void ScalingSetArgsFromYAML(void *, YAMLnode *);

void ScalingContextCreate(Scaling_context **);
void ScalingContextDestroy(Scaling_context **);

void ScalingCompute(MPI_Comm, Scaling_args *, Scaling_context *, HYPRE_IJMatrix,
                    HYPRE_IJVector, IntArray *);
void ScalingApplyToVector(const Scaling_context *, HYPRE_IJVector, scaling_vector_kind_t);
void ScalingUndoOnVector(const Scaling_context *, HYPRE_IJVector, scaling_vector_kind_t);
void ScalingApplyToSystem(Scaling_context *, HYPRE_IJMatrix, HYPRE_IJMatrix,
                          HYPRE_IJVector, HYPRE_IJVector);
void ScalingUndoOnSystem(Scaling_context *, HYPRE_IJMatrix, HYPRE_IJMatrix,
                         HYPRE_IJVector, HYPRE_IJVector);

#endif /* SCALING_HEADER */
