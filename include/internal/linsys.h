/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef LINSYS_HEADER
#define LINSYS_HEADER

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_utilities.h"
#include "internal/compatibility.h"
#include "internal/containers.h"
#include "internal/eigspec.h"
#include "internal/field.h"
#include "internal/stats.h"
#include "internal/yaml.h"

/*--------------------------------------------------------------------------
 * Linear system arguments struct
 *--------------------------------------------------------------------------*/

typedef struct LS_args_struct
{
   char      dirname[MAX_FILENAME_LENGTH];
   char      sequence_filename[MAX_FILENAME_LENGTH];
   char      matrix_filename[MAX_FILENAME_LENGTH];
   char      matrix_basename[MAX_FILENAME_LENGTH];
   char      precmat_filename[MAX_FILENAME_LENGTH];
   char      precmat_basename[MAX_FILENAME_LENGTH];
   char      rhs_filename[MAX_FILENAME_LENGTH];
   char      rhs_basename[MAX_FILENAME_LENGTH];
   char      x0_filename[MAX_FILENAME_LENGTH];
   char      sol_filename[MAX_FILENAME_LENGTH];
   char      xref_filename[MAX_FILENAME_LENGTH];
   char      xref_basename[MAX_FILENAME_LENGTH];
   char      timestep_filename[MAX_FILENAME_LENGTH];
   char      dofmap_filename[MAX_FILENAME_LENGTH];
   char      dofmap_basename[MAX_FILENAME_LENGTH];
   HYPRE_Int digits_suffix;
   HYPRE_Int init_suffix;
   HYPRE_Int last_suffix;
   IntArray *set_suffix;
   HYPRE_Int init_guess_mode;
   HYPRE_Int rhs_mode;
   HYPRE_Int type;
   HYPRE_Int exec_policy;
   HYPRE_Int num_systems;

   /* Eigenspectrum options */
   EigSpec_args eigspec;

   /* Optional symbolic DOF label map (may be NULL) */
   DofLabelMap *dof_labels;
} LS_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

StrArray       hypredrv_LinearSystemGetValidKeys(void);
StrIntMapArray hypredrv_LinearSystemGetValidValues(const char *);

void hypredrv_LinearSystemSetDefaultArgs(LS_args *);

void hypredrv_LinearSystemSetNearNullSpace(MPI_Comm, const LS_args *, HYPRE_IJMatrix, int,
                                           int, const HYPRE_Complex *, HYPRE_IJVector *);
void hypredrv_LinearSystemSetNumSystems(LS_args *);
int  hypredrv_LinearSystemGetSuffix(const LS_args *, int ls_id);
void hypredrv_LinearSystemSetArgsFromYAML(LS_args *, YAMLnode *);
void hypredrv_LinearSystemReadMatrix(MPI_Comm, const LS_args *, HYPRE_IJMatrix *,
                                     Stats *);
void hypredrv_LinearSystemSetRHS(MPI_Comm, const LS_args *, HYPRE_IJMatrix,
                                 HYPRE_IJVector *, HYPRE_IJVector *, Stats *);
void hypredrv_LinearSystemCreateWorkingSolution(MPI_Comm, const LS_args *, HYPRE_IJVector,
                                                HYPRE_IJVector *);
void hypredrv_LinearSystemSetInitialGuess(MPI_Comm, LS_args *, HYPRE_IJMatrix,
                                          HYPRE_IJVector, HYPRE_IJVector *,
                                          HYPRE_IJVector *, Stats *);
void hypredrv_LinearSystemSetReferenceSolution(MPI_Comm, const LS_args *,
                                               HYPRE_IJVector *, const Stats *);
void hypredrv_LinearSystemResetInitialGuess(HYPRE_IJVector, HYPRE_IJVector, Stats *);
void hypredrv_LinearSystemSetVectorTags(HYPRE_IJVector, IntArray *);
void hypredrv_LinearSystemSetPrecMatrix(MPI_Comm, const LS_args *, HYPRE_IJMatrix,
                                        HYPRE_IJMatrix *, const Stats *);
void hypredrv_LinearSystemReadDofmap(MPI_Comm, const LS_args *, IntArray **, Stats *);
void hypredrv_LinearSystemGetSolutionValues(HYPRE_IJVector, HYPRE_Complex **);
void hypredrv_LinearSystemGetRHSValues(HYPRE_IJVector, HYPRE_Complex **);
void hypredrv_LinearSystemComputeVectorNorm(HYPRE_IJVector, const char *, double *);
void hypredrv_LinearSystemComputeErrorNorm(HYPRE_IJVector, HYPRE_IJVector, const char *,
                                           double *);
void hypredrv_LinearSystemComputeResidualNorm(HYPRE_IJMatrix, HYPRE_IJVector,
                                              HYPRE_IJVector, const char *, double *);
void hypredrv_LinearSystemPrintData(MPI_Comm, LS_args *, HYPRE_IJMatrix, HYPRE_IJVector,
                                    const IntArray *);

long long int hypredrv_LinearSystemMatrixGetNumRows(HYPRE_IJMatrix);
long long int hypredrv_LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix);

void hypredrv_IJVectorReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJVector *);
void hypredrv_IJMatrixReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJMatrix *);

#endif /* LINSYS_HEADER */
