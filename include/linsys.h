/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef LINSYS_HEADER
#define LINSYS_HEADER

#include "yaml.h"
#include "stats.h"
#include "field.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE.h"

/*--------------------------------------------------------------------------
 * Linear system arguments struct
 *--------------------------------------------------------------------------*/

typedef struct LS_args_struct {
   char          matrix_filename[MAX_FILENAME_LENGTH];
   char          matrix_basename[MAX_FILENAME_LENGTH];
   char          precmat_filename[MAX_FILENAME_LENGTH];
   char          rhs_filename[MAX_FILENAME_LENGTH];
   char          rhs_basename[MAX_FILENAME_LENGTH];
   char          x0_filename[MAX_FILENAME_LENGTH];
   char          sol_filename[MAX_FILENAME_LENGTH];
   char          dofmap_filename[MAX_FILENAME_LENGTH];
   char          dofmap_basename[MAX_FILENAME_LENGTH];
   HYPRE_Int     digits_suffix;
   HYPRE_Int     init_suffix;
   HYPRE_Int     last_suffix;
   HYPRE_Int     init_guess_mode;
   HYPRE_Int     rhs_mode;
   HYPRE_Int     type;
   HYPRE_Int     precon_reuse;
   HYPRE_Int     exec_policy;
   HYPRE_Int     num_systems;
} LS_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

StrArray LinearSystemGetValidKeys(void);
StrIntMapArray LinearSystemGetValidValues(const char*);

void LinearSystemSetDefaultArgs(LS_args*);
void LinearSystemSetNumSystems(LS_args*);
void LinearSystemSetArgsFromYAML(LS_args*, YAMLnode*);
void LinearSystemReadMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix*);
void LinearSystemSetRHS(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJVector*);
void LinearSystemSetInitialGuess(MPI_Comm, LS_args*, HYPRE_IJMatrix,
                                 HYPRE_IJVector, HYPRE_IJVector*, HYPRE_IJVector*);
void LinearSystemResetInitialGuess(HYPRE_IJVector, HYPRE_IJVector);
void LinearSystemSetPrecMatrix(MPI_Comm, LS_args*, HYPRE_IJMatrix, HYPRE_IJMatrix*);
void LinearSystemReadDofmap(MPI_Comm, LS_args*, IntArray**);
void LinearSystemComputeRHSNorm(HYPRE_IJVector, HYPRE_Complex*);
void LinearSystemComputeResidualNorm(HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector, HYPRE_Complex*);

long long int LinearSystemMatrixGetNumRows(HYPRE_IJMatrix);
long long int LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix);

#endif /* LINSYS_HEADER */
