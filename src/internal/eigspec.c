/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/* Add internal hypre headers */
#include "_hypre_IJ_mv.h" // hypre_IJMatrixComm
#include "_hypre_parcsr_mv.h"

/* Undefine autotools package macros from hypre */
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#include <limits.h>
#include <math.h>
#include "internal/eigspec.h"
#include "internal/error.h"
#include "internal/gen_macros.h"
#include "internal/linsys.h"
#include "internal/stats.h"
#include "internal/utils.h"
#include "logging.h"
#include "object.h"

/*
 * Eigenspectrum (debug/analysis) functionality
 *
 * This implementation is intended primarily for debugging and exploratory analysis on
 * small linear systems. It explicitly builds the full dense matrix and computes the full
 * eigenspectrum using LAPACK (dgeev for general, dsyev for symmetric/Hermitian problems).
 * As such, it is NOT designed for production-scale runs or large problem sizes.
 *
 * Characteristics and limitations
 * - Data movement: A host copy of the distributed operator is gathered and expanded
 * to a dense column-major array; the caller's matrix storage stays unchanged.
 * - Complexity: memory O(n^2) doubles, time O(n^3). This quickly becomes prohibitive as n
 * grows.
 * - Practical guidance: the dense array alone requires ~8*n^2 bytes (double precision).
 * For example, n=10,000 needs ~1 GiB for A alone, plus LAPACK work arrays and
 * (optionally) eigenvectors. The preconditioned case forms B = M^{-1}A explicitly,
 * requiring another dense n^2 buffer.
 * - Scope: LAPACK and output run on rank zero. All ranks participate in gathering
 * and in applying the distributed preconditioner.
 *
 * Preconditioned spectrum (B = M^{-1}A)
 * - When enabled, we construct B = M^{-1}A by applying the existing preconditioner as a
 * solve to each column of A. This path is meant to help diagnose preconditioner behavior
 * and conditioning. It is not a scalable path for routine use (use a distributed
 * eigensolver instead).
 *
 * Numerics and outputs
 * - General (non-symmetric) problems may return complex-conjugate pairs. We write
 * eigenvalues as text: one "Re Im" pair per eigenvalue for general problems, or one
 * real value for symmetric problems. Eigenvectors are stored to a binary file,
 * one eigenvector at a time (real or interleaved complex as Re,Im pairs
 * reconstructed from LAPACK's real-workspace output for dgeev).
 *
 * Build-time and configuration
 * - This feature is compiled only when enabled (e.g., -DHYPREDRV_ENABLE_EIGSPEC=ON).
 * - Runtime control is via the YAML eigenspectrum block
 *   (enable, vectors, hermitian, preconditioned, output_prefix).
 * - A guard rail rejects matrices with n larger than a conservative limit to avoid large
 * memory usage allocations.
 *
 */

/*--------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *--------------------------------------------------------------------------*/

#define EigSpec_FIELDS(_prefix)                                              \
   ADD_FIELD_OFFSET_ENTRY(_prefix, enable, hypredrv_FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, vectors, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, hermitian, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, preconditioned, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, output_prefix, hypredrv_FieldTypeStringSet)

#define EigSpec_NUM_FIELDS \
   (sizeof(EigSpec_field_offset_map) / sizeof(EigSpec_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(EigSpec) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_EigSpecGetValidValues(const char *key)
{
   if (!strcmp(key, "enable") || !strcmp(key, "vectors") || !strcmp(key, "hermitian") ||
       !strcmp(key, "preconditioned"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_EigSpecSetDefaultArgs(EigSpec_args *args)
{
   args->enable         = 0;
   args->vectors        = 0;
   args->hermitian      = 0;
   args->preconditioned = 0; /* A */
   snprintf(args->output_prefix, MAX_FILENAME_LENGTH, "%s", "eig");
}

#ifdef HYPREDRV_ENABLE_EIGSPEC

/*-----------------------------------------------------------------------------
 * LAPACK Fortran symbols
 *-----------------------------------------------------------------------------*/

extern void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr,
                   double *wi, double *vl, int *ldvl, double *vr, int *ldvr, double *work,
                   int *lwork, int *info);
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w,
                   double *work, int *lwork, int *info);

static uint32_t
EigspecCloseOutput(FILE *fp, const char *path)
{
   int ok = !ferror(fp);
   ok     = (fclose(fp) == 0) && ok;
   if (!ok)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Could not write '%s'", path);
   }
   return ok ? ERROR_NONE : ERROR_UNKNOWN;
}

/* Operator data and the buffers that hypredrv_EigSpecCompute releases at cleanup. */
typedef struct
{
   const EigSpec_args *eargs;
   MPI_Comm            comm;
   int                 myid;
   HYPRE_IJMatrix      mat_A;
   hypre_ParCSRMatrix *par_A, *host_A, *gather_A;
   hypre_CSRMatrix    *seq_A;
   HYPRE_IJVector      rhs, solution;
   HYPRE_Complex      *rhs_values, *solution_values;
   double             *A_cm, *B_cm;
   double             *wr, *wi, *VR_cm, *work;
   HYPRE_BigInt        nrows, ncols;
   HYPRE_BigInt        ilower, iupper, jlower, jupper;
   int                 n, local_rows, local_cols;
} EigSpecContext;

static void
EigSpecHeaderPrint(const EigSpec_args *eargs, int myid)
{
   if (!myid)
   {
      printf("[EigenSpectrum] | mode=%s | vectors=%s | prefix='%s'\n",
             eargs->hermitian ? "Hermitian" : "General", eargs->vectors ? "on" : "off",
             eargs->output_prefix[0] ? eargs->output_prefix : "eig");
      fflush(stdout);
   }
}

/* Validate the external operator before gathering or allocating dense storage.
 * The size bound also makes n*n safe as an MPI count and a LAPACK dimension.
 * Rank-local; the caller reduces the status. */
static uint32_t
EigSpecMatrixValidate(EigSpecContext *ctx)
{
   uint32_t status = ERROR_NONE;

   if (!ctx->mat_A)
   {
      status = ERROR_FILE_NOT_FOUND;
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Matrix not built yet for eigenspectrum computation");
   }
   else
   {
      void     *matrix_object = NULL;
      HYPRE_Int ierr          = HYPRE_IJMatrixGetObject(ctx->mat_A, &matrix_object);
      ctx->par_A              = (hypre_ParCSRMatrix *)matrix_object;
      if (ierr || !ctx->par_A ||
          HYPRE_ParCSRMatrixGetDims((HYPRE_ParCSRMatrix)ctx->par_A, &ctx->nrows,
                                    &ctx->ncols) ||
          HYPRE_IJMatrixGetLocalRange(ctx->mat_A, &ctx->ilower, &ctx->iupper,
                                      &ctx->jlower, &ctx->jupper))
      {
         status = ERROR_HYPRE_INTERNAL;
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
         hypredrv_ErrorMsgAdd("Could not inspect the eigenspectrum matrix");
      }
      else if (ctx->nrows != ctx->ncols || ctx->nrows <= 0 || ctx->nrows > 20000)
      {
         status = ERROR_OUT_OF_BOUNDS;
         hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
         hypredrv_ErrorMsgAdd("Eigenspectrum requires a square matrix with 1 to 20000 "
                              "rows: got %lld x %lld",
                              (long long)ctx->nrows, (long long)ctx->ncols);
      }
      else if ((size_t)ctx->nrows > SIZE_MAX / sizeof(double) / (size_t)ctx->nrows)
      {
         status = ERROR_INVALID_VAL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Eigenspectrum dense allocation would overflow size_t");
      }
   }
#ifdef HYPRE_COMPLEX
   status |= ERROR_INVALID_VAL;
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd(
      "The dense LAPACK eigenspectrum requires real matrix coefficients");
#endif
   return status;
}

/* Prepare the actual preconditioner once. A failed create/setup must not be
 * lost to another public entry point or produce a spectrum of a zero matrix. */
static uint32_t
EigSpecPreconPrepare(hypredrv_t *hypredrv, const EigSpecContext *ctx)
{
   uint32_t status = ERROR_NONE;

   if (ctx->eargs->preconditioned)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: preconditioned operator");
      status = hypredrv_ObjectPreconCreate(hypredrv);
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, ctx->comm);
      if (status)
      {
         return status;
      }
      status = hypredrv_ObjectPreconSetup(hypredrv);
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: unpreconditioned operator");
   }
   return status;
}

/* Older hypre gather implementations require host storage. Clone device
 * matrices instead of migrating the caller's operator and invalidating any
 * cached preconditioner data or changing its execution space. */
static uint32_t
EigSpecHostCopy(EigSpecContext *ctx)
{
   uint32_t            status   = ERROR_NONE;
   hypre_ParCSRMatrix *par_A    = ctx->par_A;
   hypre_ParCSRMatrix *host_A   = NULL;
   hypre_ParCSRMatrix *gather_A = par_A;
#ifdef HYPRE_USING_GPU
   if (hypre_ParCSRMatrixMemoryLocation(par_A) != HYPRE_MEMORY_HOST)
   {
      host_A   = hypre_ParCSRMatrixClone_v2(par_A, 1, HYPRE_MEMORY_HOST);
      gather_A = host_A;
      if (!host_A)
      {
         status =
            hypredrv_AllocationFailure("Could not clone eigenspectrum matrix to host");
      }
   }
#endif
   ctx->host_A   = host_A;
   ctx->gather_A = gather_A;
   return status;
}

/* Gathers the sequential CSR matrix and allocates the dense copy (rank-local
 * checks; the caller reduces the status). */
static uint32_t
EigSpecGather(EigSpecContext *ctx)
{
   uint32_t status = ERROR_NONE;

   ctx->seq_A = hypre_ParCSRMatrixToCSRMatrixAll(ctx->gather_A);
   if (ctx->host_A)
   {
      hypre_ParCSRMatrixDestroy(ctx->host_A);
      ctx->host_A = NULL;
   }
   if ((!ctx->seq_A && ctx->local_rows > 0) ||
       (ctx->seq_A && hypre_CSRMatrixNumRows(ctx->seq_A) != ctx->n))
   {
      status = ERROR_HYPRE_INTERNAL;
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("Failed to gather the eigenspectrum matrix");
   }
   ctx->A_cm = (double *)calloc((size_t)ctx->n * (size_t)ctx->n, sizeof(double));
   if (!ctx->A_cm)
   {
      status |=
         hypredrv_AllocationFailure("Could not allocate eigenspectrum dense matrix");
   }
   return status;
}

static void
EigSpecDenseFill(EigSpecContext *ctx)
{
   const int            n      = ctx->n;
   const HYPRE_Int     *di     = hypre_CSRMatrixI(ctx->seq_A);
   const HYPRE_Int     *dj     = hypre_CSRMatrixJ(ctx->seq_A);
   const HYPRE_Complex *values = hypre_CSRMatrixData(ctx->seq_A);
   for (int i = 0; i < n; i++)
   {
      for (HYPRE_Int entry = di[i]; entry < di[i + 1]; entry++)
      {
         ctx->A_cm[(size_t)i + ((size_t)dj[entry] * (size_t)n)] =
            (double)hypre_creal(values[entry]);
      }
   }
}

/* Fills the dense matrix on the lowest rank holding the gathered CSR and
 * broadcasts it (collective; every rank returns the same status). */
static uint32_t
EigSpecDenseBroadcast(EigSpecContext *ctx)
{
   int gather_rank = ctx->seq_A ? ctx->myid : INT_MAX;
   MPI_Allreduce(MPI_IN_PLACE, &gather_rank, 1, MPI_INT, MPI_MIN, ctx->comm);
   if (gather_rank == INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("No rank received the eigenspectrum matrix");
      return ERROR_HYPRE_INTERNAL;
   }
   if (ctx->myid == gather_rank)
   {
      EigSpecDenseFill(ctx);
   }
   if (ctx->seq_A)
   {
      hypre_CSRMatrixDestroy(ctx->seq_A);
      ctx->seq_A = NULL;
   }
   /* A rank with no rows can legitimately receive no gathered CSR. It still
    * participates in preconditioning and may be the LAPACK/output rank. */
   MPI_Bcast(ctx->A_cm, ctx->n * ctx->n, MPI_DOUBLE, gather_rank, ctx->comm);
   return ERROR_NONE;
}

static uint32_t
EigSpecPreconBuffersCreate(EigSpecContext *ctx)
{
   ctx->B_cm       = (double *)calloc((size_t)ctx->n * (size_t)ctx->n, sizeof(double));
   ctx->rhs_values = (HYPRE_Complex *)hypredrv_Calloc(
      (size_t)ctx->local_rows, sizeof(HYPRE_Complex), "eigenspectrum RHS");
   ctx->solution_values = (HYPRE_Complex *)hypredrv_Calloc(
      (size_t)ctx->local_cols, sizeof(HYPRE_Complex), "eigenspectrum solution");
   if (!ctx->B_cm || !ctx->rhs_values || !ctx->solution_values)
   {
      return hypredrv_AllocationFailure(
         "Could not allocate preconditioned eigenspectrum buffers");
   }
   return ERROR_NONE;
}

static uint32_t
EigSpecWorkVectorsCreate(EigSpecContext *ctx)
{
   HYPRE_MemoryLocation location = hypre_ParCSRMatrixMemoryLocation(ctx->par_A);
   HYPRE_Int ierr = HYPRE_IJVectorCreate(ctx->comm, ctx->ilower, ctx->iupper, &ctx->rhs);
   if (!ierr) ierr = HYPRE_IJVectorSetObjectType(ctx->rhs, HYPRE_PARCSR);
   if (!ierr) ierr = HYPRE_IJVectorInitialize_v2(ctx->rhs, location);
   if (!ierr)
      ierr = HYPRE_IJVectorCreate(ctx->comm, ctx->jlower, ctx->jupper, &ctx->solution);
   if (!ierr) ierr = HYPRE_IJVectorSetObjectType(ctx->solution, HYPRE_PARCSR);
   if (!ierr) ierr = HYPRE_IJVectorInitialize_v2(ctx->solution, location);
   if (ierr)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("Could not initialize eigenspectrum work vectors");
      return ERROR_HYPRE_INTERNAL;
   }
   return ERROR_NONE;
}

static uint32_t
EigSpecWorkVectorsAssemble(EigSpecContext *ctx, void **rhs_object, void **solution_object)
{
   HYPRE_Int ierr = HYPRE_IJVectorAssemble(ctx->rhs);
   ierr |= HYPRE_IJVectorAssemble(ctx->solution);
   ierr |= HYPRE_IJVectorGetObject(ctx->rhs, rhs_object);
   ierr |= HYPRE_IJVectorGetObject(ctx->solution, solution_object);
   if (ierr || !*rhs_object || !*solution_object)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("Could not assemble eigenspectrum work vectors");
      return ERROR_HYPRE_INTERNAL;
   }
   return ERROR_NONE;
}

/* Forms B = M^{-1} A column by column. Each preconditioner apply is followed
 * by a status reduction, so every rank leaves the loop at the same column. */
static uint32_t
EigSpecPreconApplyColumns(hypredrv_t *hypredrv, EigSpecContext *ctx, void *rhs_object,
                          void *solution_object)
{
   const int     n         = ctx->n;
   hypre_Vector *local_rhs = hypre_ParVectorLocalVector((hypre_ParVector *)rhs_object);
   hypre_Vector *local_solution =
      hypre_ParVectorLocalVector((hypre_ParVector *)solution_object);
   for (int column = 0; column < n; column++)
   {
      for (int row = 0; row < ctx->local_rows; row++)
      {
         ctx->rhs_values[row] =
            (HYPRE_Complex)ctx
               ->A_cm[(size_t)ctx->ilower + (size_t)row + ((size_t)column * (size_t)n)];
      }
      if (ctx->local_rows)
      {
         hypre_TMemcpy(hypre_VectorData(local_rhs), ctx->rhs_values, HYPRE_Complex,
                       ctx->local_rows, hypre_VectorMemoryLocation(local_rhs),
                       HYPRE_MEMORY_HOST);
      }
      HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)solution_object, 0.0);
      uint32_t status = hypredrv_ObjectPreconApply(hypredrv, ctx->rhs, ctx->solution);
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, ctx->comm);
      if (status)
      {
         return status;
      }
      if (ctx->local_cols)
      {
         hypre_TMemcpy(ctx->solution_values, hypre_VectorData(local_solution),
                       HYPRE_Complex, ctx->local_cols, HYPRE_MEMORY_HOST,
                       hypre_VectorMemoryLocation(local_solution));
      }
      for (int row = 0; row < ctx->local_cols; row++)
      {
         ctx->B_cm[(size_t)ctx->jlower + (size_t)row + ((size_t)column * (size_t)n)] =
            (double)hypre_creal(ctx->solution_values[row]);
      }
   }
   return ERROR_NONE;
}

/* Replaces the dense A with B = M^{-1} A (collective; every rank returns the
 * same status). */
static uint32_t
EigSpecPreconOperatorBuild(hypredrv_t *hypredrv, EigSpecContext *ctx)
{
   void    *rhs_object = NULL, *solution_object = NULL;
   uint32_t status = EigSpecPreconBuffersCreate(ctx);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, ctx->comm);
   if (status)
   {
      return status;
   }
   status = EigSpecWorkVectorsCreate(ctx);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, ctx->comm);
   if (status)
   {
      return status;
   }
   status = EigSpecWorkVectorsAssemble(ctx, &rhs_object, &solution_object);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, ctx->comm);
   if (status)
   {
      return status;
   }
   status = EigSpecPreconApplyColumns(hypredrv, ctx, rhs_object, solution_object);
   if (status)
   {
      return status;
   }
   MPI_Allreduce(MPI_IN_PLACE, ctx->B_cm, ctx->n * ctx->n, MPI_DOUBLE, MPI_SUM,
                 ctx->comm);
   free(ctx->A_cm);
   ctx->A_cm = ctx->B_cm;
   ctx->B_cm = NULL;
   return ERROR_NONE;
}

static uint32_t
EigSpecLapackBuffersCreate(EigSpecContext *ctx)
{
   const EigSpec_args *eargs = ctx->eargs;

   ctx->wr = (double *)malloc((size_t)ctx->n * sizeof(double));
   if (!eargs->hermitian)
   {
      ctx->wi = (double *)malloc((size_t)ctx->n * sizeof(double));
      if (eargs->vectors)
      {
         ctx->VR_cm = (double *)malloc((size_t)ctx->n * (size_t)ctx->n * sizeof(double));
      }
   }
   if (!ctx->wr || (!eargs->hermitian && (!ctx->wi || (eargs->vectors && !ctx->VR_cm))))
   {
      return hypredrv_AllocationFailure(
         "Could not allocate LAPACK eigenvalue/vector buffers");
   }
   return ERROR_NONE;
}

/* Calls dsyev or dgeev on the dense operator; lwork = -1 is the workspace
 * query. Returns LAPACK's info. */
static int
EigSpecLapackCall(EigSpecContext *ctx, double *work, int lwork)
{
   char jobz = ctx->eargs->vectors ? 'V' : 'N', jobvl = 'N', uplo = 'U';
   int  n = ctx->n, lda = n, ldv = n, info = 0;
   if (ctx->eargs->hermitian)
   {
      dsyev_(&jobz, &uplo, &n, ctx->A_cm, &lda, ctx->wr, work, &lwork, &info);
   }
   else
   {
      dgeev_(&jobvl, &jobz, &n, ctx->A_cm, &lda, ctx->wr, ctx->wi, NULL, &ldv, ctx->VR_cm,
             &ldv, work, &lwork, &info);
   }
   return info;
}

static uint32_t
EigSpecLapackSolve(EigSpecContext *ctx)
{
   const char *routine      = ctx->eargs->hermitian ? "dsyev" : "dgeev";
   double      optimal_work = 0.0;
   int         info         = EigSpecLapackCall(ctx, &optimal_work, -1);
   if (info || !isfinite(optimal_work) || optimal_work < 1.0 || optimal_work > INT_MAX ||
       optimal_work > (double)(SIZE_MAX / sizeof(double)))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Invalid %s workspace query: info=%d, size=%.17g", routine,
                           info, optimal_work);
      return ERROR_UNKNOWN;
   }
   int lwork = (int)optimal_work;
   ctx->work = (double *)malloc((size_t)lwork * sizeof(double));
   if (!ctx->work)
   {
      return hypredrv_AllocationFailure("Could not allocate LAPACK workspace");
   }
   info = EigSpecLapackCall(ctx, ctx->work, lwork);
   if (info)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("%s failed with info=%d", routine, info);
      return ERROR_UNKNOWN;
   }
   return ERROR_NONE;
}

static FILE *
EigSpecOutputOpen(const char *path, const char *mode)
{
   FILE *fp = fopen(path, mode);
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open '%s' for writing", path);
   }
   return fp;
}

static uint32_t
EigSpecValuesWrite(const EigSpecContext *ctx, const char *prefix)
{
   char path[MAX_FILENAME_LENGTH + 32];
   snprintf(path, sizeof(path), "%s.values.txt", prefix);
   FILE *fp = EigSpecOutputOpen(path, "w");
   if (!fp)
   {
      return ERROR_FILE_NOT_FOUND;
   }
   for (int i = 0; i < ctx->n; i++)
   {
      if (ctx->wi) fprintf(fp, "%.17g %.17g\n", ctx->wr[i], ctx->wi[i]);
      else fprintf(fp, "%.17g\n", ctx->wr[i]);
   }
   return EigspecCloseOutput(fp, path);
}

/* LAPACK stores a complex conjugate pair in adjacent real columns. Real
 * eigenvalues, and a column with no partner, get a zero imaginary part. */
static void
EigSpecComplexVectorsWrite(FILE *fp, int n, const double *VR_cm, const double *wi)
{
   for (int ev = 0; ev < n;)
   {
      if (wi[ev] > 0.0 && ev + 1 < n)
      {
         for (int conjugate = 0; conjugate < 2; conjugate++)
         {
            for (int i = 0; i < n; i++)
            {
               double pair[2] = {VR_cm[i + (ev * n)],
                                 (conjugate ? -1.0 : 1.0) * VR_cm[i + ((ev + 1) * n)]};
               fwrite(pair, sizeof(double), 2, fp);
            }
         }
         ev += 2;
      }
      else
      {
         for (int i = 0; i < n; i++)
         {
            double pair[2] = {VR_cm[i + (ev * n)], 0.0};
            fwrite(pair, sizeof(double), 2, fp);
         }
         ev++;
      }
   }
}

static uint32_t
EigSpecVectorsWrite(const EigSpecContext *ctx, const char *prefix)
{
   char path[MAX_FILENAME_LENGTH + 32];
   snprintf(path, sizeof(path), "%s.vectors.bin", prefix);
   FILE *fp = EigSpecOutputOpen(path, "wb");
   if (!fp)
   {
      return ERROR_FILE_NOT_FOUND;
   }
   if (ctx->eargs->hermitian)
   {
      fwrite(ctx->A_cm, sizeof(double), (size_t)ctx->n * (size_t)ctx->n, fp);
   }
   else
   {
      EigSpecComplexVectorsWrite(fp, ctx->n, ctx->VR_cm, ctx->wi);
   }
   return EigspecCloseOutput(fp, path);
}

/* Rank-local: LAPACK on the assembled dense operator, then the output files. */
static uint32_t
EigSpecSolveAndWrite(EigSpecContext *ctx)
{
   uint32_t status = EigSpecLapackBuffersCreate(ctx);
   if (status)
   {
      return status;
   }
   status = EigSpecLapackSolve(ctx);
   if (status)
   {
      return status;
   }
   const char *prefix = ctx->eargs->output_prefix[0] ? ctx->eargs->output_prefix : "eig";
   status             = EigSpecValuesWrite(ctx, prefix);
   if (status || !ctx->eargs->vectors)
   {
      return status;
   }
   return EigSpecVectorsWrite(ctx, prefix);
}

static void
EigSpecContextFree(EigSpecContext *ctx)
{
   if (ctx->seq_A) hypre_CSRMatrixDestroy(ctx->seq_A);
   if (ctx->host_A) hypre_ParCSRMatrixDestroy(ctx->host_A);
   if (ctx->rhs) HYPRE_IJVectorDestroy(ctx->rhs);
   if (ctx->solution) HYPRE_IJVectorDestroy(ctx->solution);
   free(ctx->rhs_values);
   free(ctx->solution_values);
   free(ctx->A_cm);
   free(ctx->B_cm);
   free(ctx->wr);
   free(ctx->wi);
   free(ctx->VR_cm);
   free(ctx->work);
}

/* Sets every error bit in `status` that this rank's error state lacks. */
static void
EigSpecRemoteErrorsAdopt(uint32_t status)
{
   uint32_t missing = status & ~hypredrv_ErrorCodeGet();
   for (uint32_t bit = 1; bit; bit <<= 1)
   {
      if (missing & bit) hypredrv_ErrorCodeSet((hypredrv_error_t)bit);
   }
}

#else

static void
EigSpecDisabledWarn(hypredrv_t *hypredrv)
{
   static bool warned_eigspec_disabled = false;
   if (!warned_eigspec_disabled && !hypredrv->mypid)
   {
      fprintf(stderr,
              "[HYPREDRV] Warning: HYPREDRV_LinearSystemComputeEigenspectrum called "
              "but eigenspectrum support is disabled. "
              "Reconfigure with -DHYPREDRV_ENABLE_EIGSPEC=ON to enable it.\n");
      fflush(stderr);
      warned_eigspec_disabled = true; /* GCOVR_EXCL_BR_LINE */
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum support disabled in build; emitted warning");
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv, "eigenspectrum support disabled in build; warning already emitted");
   }
}

#endif /* HYPREDRV_ENABLE_EIGSPEC */

uint32_t
hypredrv_EigSpecCompute(hypredrv_t *hypredrv)
{
   uint32_t status = ERROR_NONE;
#ifdef HYPREDRV_ENABLE_EIGSPEC
   const EigSpec_args *eargs = &hypredrv->iargs->ls.eigspec;
   MPI_Comm            comm  = hypredrv->comm;
   EigSpecContext      ctx   = {.eargs  = eargs,
                                .comm   = comm,
                                .myid   = hypredrv->mypid,
                                .mat_A  = hypredrv->mat_A.handle,
                                .iupper = -1,
                                .jupper = -1};
   bool                timed = false;

   if (!eargs->enable)
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "eigenspectrum computation skipped: linear_system.eigspec.enable=0");
      goto done;
   }
   EigSpecHeaderPrint(eargs, ctx.myid);

   status = EigSpecMatrixValidate(&ctx);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   ctx.n          = (int)ctx.nrows;
   ctx.local_rows = (int)(ctx.iupper - ctx.ilower + 1);
   ctx.local_cols = (int)(ctx.jupper - ctx.jlower + 1);

   status = EigSpecPreconPrepare(hypredrv, &ctx);
   if (status)
   {
      goto cleanup;
   }

   status = hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   timed  = (status == ERROR_NONE);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }

   status = EigSpecHostCopy(&ctx);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   status = EigSpecGather(&ctx);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   status = EigSpecDenseBroadcast(&ctx);
   if (status)
   {
      goto cleanup;
   }

   if (eargs->preconditioned)
   {
      status = EigSpecPreconOperatorBuild(hypredrv, &ctx);
      if (status)
      {
         goto cleanup;
      }
   }

   /* LAPACK and file output need only one copy of the already assembled operator.
    * Other ranks join the final status reduction, including root-only I/O errors. */
   if (!ctx.myid)
   {
      status = EigSpecSolveAndWrite(&ctx);
   }

cleanup:
   EigSpecContextFree(&ctx);
   if (timed)
   {
      status |= hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");
   }
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   /* Populate remote failures without resetting any diagnostics already held by
    * this caller. The return value reports only this operation's own status. */
   EigSpecRemoteErrorsAdopt(status);
done:
#else
   EigSpecDisabledWarn(hypredrv);
#endif
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)", status);
   return status;
}
