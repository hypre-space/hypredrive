/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef LINSYS_HEADER
#define LINSYS_HEADER

#include <stddef.h>
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
 * Scheduled linear-system dump configuration
 *--------------------------------------------------------------------------*/

typedef enum PrintSystemType_enum
{
   PRINT_SYSTEM_TYPE_ALL = 0,
   PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS,
   PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS,
   PRINT_SYSTEM_TYPE_IDS,
   PRINT_SYSTEM_TYPE_RANGES,
   PRINT_SYSTEM_TYPE_ITERATIONS_OVER,
   PRINT_SYSTEM_TYPE_SETUP_TIME_OVER,
   PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER,
   PRINT_SYSTEM_TYPE_SELECTORS,
} PrintSystemType;

typedef enum PrintSystemStageBits_enum
{
   PRINT_SYSTEM_STAGE_BUILD_BIT = 1 << 0,
   PRINT_SYSTEM_STAGE_SETUP_BIT = 1 << 1,
   PRINT_SYSTEM_STAGE_APPLY_BIT = 1 << 2,
} PrintSystemStageBits;

typedef enum PrintSystemStageId_enum
{
   PRINT_SYSTEM_STAGE_BUILD = 0,
   PRINT_SYSTEM_STAGE_SETUP = 1,
   PRINT_SYSTEM_STAGE_APPLY = 2,
} PrintSystemStageId;

typedef enum PrintSystemBasis_enum
{
   PRINT_SYSTEM_BASIS_LINEAR_SYSTEM = 0,
   PRINT_SYSTEM_BASIS_TIMESTEP,
   PRINT_SYSTEM_BASIS_LEVEL,
   PRINT_SYSTEM_BASIS_ITERATIONS,
   PRINT_SYSTEM_BASIS_SETUP_TIME,
   PRINT_SYSTEM_BASIS_SOLVE_TIME,
} PrintSystemBasis;

typedef enum PrintSystemArtifactBits_enum
{
   PRINT_SYSTEM_ARTIFACT_MATRIX   = 1 << 0,
   PRINT_SYSTEM_ARTIFACT_PRECMAT  = 1 << 1,
   PRINT_SYSTEM_ARTIFACT_RHS      = 1 << 2,
   PRINT_SYSTEM_ARTIFACT_X0       = 1 << 3,
   PRINT_SYSTEM_ARTIFACT_XREF     = 1 << 4,
   PRINT_SYSTEM_ARTIFACT_SOLUTION = 1 << 5,
   PRINT_SYSTEM_ARTIFACT_DOFMAP   = 1 << 6,
   PRINT_SYSTEM_ARTIFACT_METADATA = 1 << 7,
} PrintSystemArtifactBits;

typedef struct IntRange_struct
{
   int begin;
   int end;
} IntRange;

typedef struct IntRangeArray_struct
{
   IntRange *data;
   size_t    size;
} IntRangeArray;

typedef struct DumpSelector_args_struct
{
   int           basis;
   int           level;
   int           every;
   double        threshold;
   IntArray     *ids;
   IntRangeArray ranges;
} DumpSelector_args;

typedef struct PrintSystem_args_struct
{
   int  enabled;
   int  type;
   int  stage_mask;
   int  artifacts;
   char output_dir[MAX_FILENAME_LENGTH];
   int  overwrite;
   int  next_dump_index;
   int  overwrite_prepared;

   int           every;
   double        threshold;
   IntArray     *ids;
   IntRangeArray ranges;

   DumpSelector_args *selectors;
   size_t             num_selectors;
} PrintSystem_args;

typedef struct PrintSystemContext_struct
{
   int    stage;
   int    system_index;
   int    timestep_index;
   int    last_iter;
   int    variant_index;
   int    repetition_index;
   int    stats_ls_id;
   double last_setup_time;
   double last_solve_time;
   int    level_ids[STATS_MAX_LEVELS];
} PrintSystemContext;

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

   /* Scheduled linear-system dump options */
   PrintSystem_args print_system;

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
void hypredrv_PrintSystemSetDefaultArgs(PrintSystem_args *);
void hypredrv_PrintSystemDestroyArgs(PrintSystem_args *);
void hypredrv_PrintSystemSetArgs(void *, const YAMLnode *);

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
                                          HYPRE_IJVector *, const Stats *);
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
uint32_t hypredrv_LinearSystemDumpScheduled(MPI_Comm, const LS_args *, HYPRE_IJMatrix,
                                            HYPRE_IJMatrix, HYPRE_IJVector,
                                            HYPRE_IJVector, HYPRE_IJVector,
                                            HYPRE_IJVector, const IntArray *,
                                            const PrintSystemContext *, const char *);

long long int hypredrv_LinearSystemMatrixGetNumRows(HYPRE_IJMatrix);
long long int hypredrv_LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix);

void hypredrv_IJVectorReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJVector *);
void hypredrv_IJMatrixReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJMatrix *);

#endif /* LINSYS_HEADER */
