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
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
#include "internal/compatibility.h"
#include "internal/containers.h"
#include "internal/eigspec.h"
#include "internal/field.h"
#include "internal/object_fwd.h"
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
   char      precmat_sequence_filename[MAX_FILENAME_LENGTH];
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
   HYPRE_Int precmat_sequence_system_id;
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
void hypredrv_PrintSystemSetArgs(void *, YAMLnode *);
int  hypredrv_PrintSystemNeedsTimestepSchedule(const PrintSystem_args *);

/* Requires a live object and parsed arguments. Validates and installs exact
 * (orthonormalized) or near modes collectively; failure preserves installed
 * modes. Returns this operation's status without resetting prior diagnostics. */
uint32_t hypredrv_LinearSystemSetNullSpaceModes(hypredrv_t *, bool, int, int,
                                                const HYPRE_Complex *);

void hypredrv_LinearSystemProjectOutNullSpace(HYPRE_IJVector, int, HYPRE_IJVector);
void hypredrv_LinearSystemSetNumSystems(LS_args *);
int  hypredrv_LinearSystemGetSuffix(const LS_args *, int ls_id);
void hypredrv_LinearSystemSetArgsFromYAML(LS_args *, YAMLnode *);
/* Matrix reads own source selection, collective ingress, fixed sequence caching,
 * and installation on a live object. Configured reads require parsed arguments
 * on every rank; rejected input preserves installed matrices and ownership. */
uint32_t hypredrv_LinearSystemReadMatrix(hypredrv_t *);
/* External system matrices are validated and installed locally, including
 * before arguments are parsed. NULL clears the system/preconditioner pair. */
uint32_t hypredrv_LinearSystemSetMatrix(hypredrv_t *, HYPRE_IJMatrix);
/* Array ingress receives a live object, validates the caller's buffers and
 * partition collectively, and installs only a complete replacement. Return
 * status describes this operation; existing diagnostics are preserved. */
uint32_t hypredrv_LinearSystemSetMatrixFromCSR(hypredrv_t *, HYPRE_BigInt, HYPRE_BigInt,
                                               const HYPRE_BigInt *, const HYPRE_BigInt *,
                                               const HYPRE_Real *);
uint32_t hypredrv_LinearSystemSetRHSFromArray(hypredrv_t *, HYPRE_BigInt, HYPRE_BigInt,
                                              const HYPRE_Real *);
/* Own vector construction and installation. Configured RHS/reference reads and
 * default guesses validate parsed arguments collectively before using them.
 * A configured reference no-op must be selected on every rank. File readers
 * agree on format and partition count once, before native assembly.
 * External attachments and default solution allocation also work before parsing.
 * Validate external storage on ingress; installed RHS vectors have one component,
 * and each shared handle has one owner. Rejected replacements preserve installed
 * vectors and ownership. */
uint32_t hypredrv_LinearSystemSetRHS(hypredrv_t *, HYPRE_IJVector);
uint32_t hypredrv_LinearSystemSetReferenceSolution(hypredrv_t *, HYPRE_IJVector);
uint32_t hypredrv_LinearSystemSetSolution(hypredrv_t *, HYPRE_IJVector);
typedef enum
{
   GUESS_REPLACE,
   GUESS_ENSURE_SOLUTION,
} hypredrv_GuessUpdate;
/* Ensure mode receives a NULL source and constructs only when some rank needs
 * a solution. All ranks participate; existing vector pairs are preserved. */
uint32_t hypredrv_LinearSystemSetInitialGuess(hypredrv_t *, hypredrv_GuessUpdate,
                                              HYPRE_IJVector);
uint32_t hypredrv_LinearSystemResetInitialGuess(hypredrv_t *);

void hypredrv_LinearSystemLogBlockFrobenius(MPI_Comm, HYPRE_IJMatrix, const IntArray *,
                                            const DofLabelMap *, const char *, int);
/* A non-NULL matrix is validated and registered locally. NULL selects the
 * collective configured/default workflow. Both require parsed arguments. */
uint32_t hypredrv_LinearSystemSetPrecMatrix(hypredrv_t *, HYPRE_IJMatrix);
typedef enum
{
   DOFMAP_FROM_ARRAY,
   DOFMAP_INTERLEAVED,
   DOFMAP_CONTIGUOUS,
   DOFMAP_FROM_INPUT
} hypredrv_DofmapSource;
/* Collective construction and installation on a live object. Array sources use
 * size and labels; patterns use block count and type count; configured input
 * ignores those arguments and requires parsed options. Failure preserves the
 * map, tags and native consumers. Prior diagnostics abort without being reset. */
uint32_t hypredrv_LinearSystemSetDofmap(hypredrv_t *, hypredrv_DofmapSource, int, int,
                                        const int *);
/* Validate a non-NULL application matrix once, before migration/attachment.
 * Returns this check's status without resetting the caller's diagnostics. */
uint32_t hypredrv_IJMatrixValidate(HYPRE_IJMatrix, const char *);
/* Borrow initialized local data in the requested memory space. UNDEFINED keeps
 * the vector in its current space. Empty ranks may return NULL without an error. */
HYPRE_Complex *hypredrv_IJVectorData(HYPRE_IJVector, HYPRE_MemoryLocation);
void hypredrv_LinearSystemComputeVectorNorm(HYPRE_IJVector, const char *, double *);
void hypredrv_LinearSystemComputeErrorNorm(HYPRE_IJVector, HYPRE_IJVector, const char *,
                                           double *);
void hypredrv_LinearSystemComputeResidualNorm(HYPRE_IJMatrix, HYPRE_IJVector,
                                              HYPRE_IJVector, const char *, double *);
MPI_Comm  hypredrv_IJMatrixComm(HYPRE_IJMatrix);
MPI_Comm  hypredrv_IJVectorComm(HYPRE_IJVector);
HYPRE_Int hypredrv_ParCSRMatrixMigrate(HYPRE_ParCSRMatrix, HYPRE_MemoryLocation);
HYPRE_Int hypredrv_IJMatrixMigrate(HYPRE_IJMatrix, HYPRE_MemoryLocation);
HYPRE_Int hypredrv_IJVectorMigrate(HYPRE_IJVector, HYPRE_MemoryLocation);
MPI_Comm  hypredrv_IJSystemComm(HYPRE_IJMatrix A, HYPRE_IJMatrix M, HYPRE_IJVector b,
                                HYPRE_IJVector x);
void      hypredrv_LinearSystemComputeResidualNormWork(HYPRE_IJMatrix, HYPRE_IJVector,
                                                       HYPRE_IJVector, const char *,
                                                       HYPRE_IJVector *, double *);
void hypredrv_LinearSystemLogBlockResidualNorms(MPI_Comm, HYPRE_IJMatrix, HYPRE_IJVector,
                                                HYPRE_IJVector, const IntArray *,
                                                const DofLabelMap *, const char *, int);
void hypredrv_LinearSystemPrintData(MPI_Comm, LS_args *, HYPRE_IJMatrix, HYPRE_IJVector,
                                    const IntArray *);
uint32_t hypredrv_LinearSystemDumpScheduled(MPI_Comm, LS_args *, HYPRE_IJMatrix,
                                            HYPRE_IJMatrix, HYPRE_IJVector,
                                            HYPRE_IJVector, HYPRE_IJVector,
                                            HYPRE_IJVector, const IntArray *,
                                            const PrintSystemContext *, const char *);

long long int hypredrv_LinearSystemMatrixGetNumRows(HYPRE_IJMatrix);
long long int hypredrv_LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix);

uint32_t hypredrv_LinearSystemLoadTimestepSchedule(const char *, IntArray **,
                                                   IntArray **);

void hypredrv_IJVectorReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJVector *);
void hypredrv_IJMatrixReadMultipartBinary(const char *, MPI_Comm, uint64_t,
                                          HYPRE_MemoryLocation, HYPRE_IJMatrix *);

/* One rank-local matrix part in the multipart part layout: `nnz` row and column
 * indices of `index_size` (4 or 8) bytes and `nnz` values of `value_size` (4 or
 * 8) bytes. The builder may convert the arrays in place; `label` names the part
 * in error messages. */
typedef struct
{
   uint64_t    nrows;
   uint64_t    nnz;
   uint64_t    index_size;
   uint64_t    value_size;
   void       *rows;
   void       *cols;
   void       *vals;
   const char *label;
} hypredrv_IJMatrixMemPart;

/* Supplies rank-local matrix part p to hypredrv_IJMatrixBuildFromSource. `load`
 * always fills the metadata (nrows, nnz, widths, label) and allocates (malloc)
 * the index arrays when `want` has HYPREDRV_PART_INDICES and the values when
 * it has HYPREDRV_PART_VALUES; the builder frees them. A values-only request
 * may receive a part whose indices are already loaded. Returns 0 with the
 * error state set on failure. */
enum
{
   HYPREDRV_PART_INDICES = 1,
   HYPREDRV_PART_VALUES  = 2,
};
typedef struct
{
   void    *ctx;
   uint32_t nparts;
   int (*load)(void *ctx, uint32_t p, int want, hypredrv_IJMatrixMemPart *part);
} hypredrv_IJMatrixPartSource;

void hypredrv_IJMatrixBuildFromSource(MPI_Comm, const hypredrv_IJMatrixPartSource *,
                                      HYPRE_MemoryLocation, HYPRE_IJMatrix *);

/* One rank-local vector part: `nrows` values of `value_size` (4 or 8) bytes,
 * convertible in place; `label` names it in error messages. */
typedef struct
{
   uint64_t    nrows;
   uint64_t    value_size;
   void       *vals;
   const char *label;
} hypredrv_IJVectorMemPart;

/* Supplies rank-local vector part p to hypredrv_IJVectorBuildFromSource: `load`
 * always fills nrows, value_size and label, and allocates (malloc) the values
 * when `want_values` is set; the builder frees them. Returns 0 on failure. */
typedef struct
{
   void    *ctx;
   uint32_t nparts;
   int (*load)(void *ctx, uint32_t p, int want_values, hypredrv_IJVectorMemPart *part);
} hypredrv_IJVectorPartSource;

void hypredrv_IJVectorBuildFromSource(MPI_Comm, const hypredrv_IJVectorPartSource *,
                                      HYPRE_MemoryLocation, HYPRE_IJVector *);

#endif /* LINSYS_HEADER */
