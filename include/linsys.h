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
#include "compatibility.h"
#include "containers.h"
#include "eigspec.h"
#include "field.h"
#include "stats.h"
#include "yaml.h"

/*--------------------------------------------------------------------------
 * Linear system arguments struct
 *--------------------------------------------------------------------------*/

typedef struct LS_args_struct
{
   char      dirname[MAX_FILENAME_LENGTH];
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
} LS_args;

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

StrArray       LinearSystemGetValidKeys(void);
StrIntMapArray LinearSystemGetValidValues(const char *);

void LinearSystemSetDefaultArgs(LS_args *);

void LinearSystemSetNearNullSpace(MPI_Comm, const LS_args *, HYPRE_IJMatrix, int, int,
                                  const HYPRE_Complex *, HYPRE_IJVector *);
void LinearSystemSetNumSystems(LS_args *);
int  LinearSystemGetSuffix(const LS_args *, int ls_id);
void LinearSystemSetArgsFromYAML(LS_args *, YAMLnode *);
void LinearSystemReadMatrix(MPI_Comm, const LS_args *, HYPRE_IJMatrix *, Stats *);
void LinearSystemSetRHS(MPI_Comm, const LS_args *, HYPRE_IJMatrix, HYPRE_IJVector *,
                        HYPRE_IJVector *, Stats *);
void LinearSystemSetInitialGuess(MPI_Comm, LS_args *, HYPRE_IJMatrix, HYPRE_IJVector,
                                 HYPRE_IJVector *, HYPRE_IJVector *, Stats *);
void LinearSystemSetReferenceSolution(MPI_Comm, const LS_args *, HYPRE_IJVector *,
                                      Stats *);
void LinearSystemResetInitialGuess(HYPRE_IJVector, HYPRE_IJVector, Stats *);
void LinearSystemSetVectorTags(HYPRE_IJVector, IntArray *);
void LinearSystemSetPrecMatrix(MPI_Comm, LS_args *, HYPRE_IJMatrix, HYPRE_IJMatrix *,
                               Stats *);
void LinearSystemReadDofmap(MPI_Comm, LS_args *, IntArray **, Stats *);
void LinearSystemGetSolutionValues(HYPRE_IJVector, HYPRE_Complex **);
void LinearSystemGetRHSValues(HYPRE_IJVector, HYPRE_Complex **);
void LinearSystemComputeVectorNorm(HYPRE_IJVector, const char *, double *);
void LinearSystemComputeErrorNorm(HYPRE_IJVector, HYPRE_IJVector, const char *, double *);
void LinearSystemComputeResidualNorm(HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector,
                                     const char *, double *);
void LinearSystemPrintData(MPI_Comm, LS_args *, HYPRE_IJMatrix, HYPRE_IJVector,
                           const IntArray *);

long long int LinearSystemMatrixGetNumRows(HYPRE_IJMatrix);
long long int LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix);

void IJVectorReadMultipartBinary(const char *, MPI_Comm, uint64_t, HYPRE_MemoryLocation,
                                 HYPRE_IJVector *);
void IJMatrixReadMultipartBinary(const char *, MPI_Comm, uint64_t, HYPRE_MemoryLocation,
                                 HYPRE_IJMatrix *);

#endif /* LINSYS_HEADER */
