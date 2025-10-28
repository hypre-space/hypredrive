/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "linsys.h"
#include "stats.h"
#include "error.h"
#include "utils.h"
#include "eigspec.h"
#include "gen_macros.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h" // hypre_IJMatrixComm

/*
 * Eigenspectrum (debug/analysis) functionality
 *
 * This implementation is intended primarily for debugging and exploratory analysis on
 * small linear systems. It explicitly builds the full dense matrix and computes the full
 * eigenspectrum using LAPACK (dgeev for general, dsyev for symmetric/Hermitian problems).
 * As such, it is NOT designed for production-scale runs or large problem sizes.
 *
 * Characteristics and limitations
 * - Data movement: The parallel ParCSR matrix is gathered to a single rank as a sequential
 *   CSR and then expanded to a dense column-major array for LAPACK.
 * - Complexity: memory O(n^2) doubles, time O(n^3). This quickly becomes prohibitive as n grows.
 * - Practical guidance: the dense array alone requires ~8*n^2 bytes (double precision). For
 *   example, n=10,000 needs ~1 GiB for A alone, plus LAPACK work arrays and (optionally) eigenvectors.
 *   The preconditioned case forms B = M^{-1}A explicitly, requiring another dense n^2 buffer.
 * - Scope: single program instance (gathers on one rank). There is no distributed eigensolver here.
 *
 * Preconditioned spectrum (B = M^{-1}A)
 * - When enabled, we construct B = M^{-1}A by applying the existing preconditioner as a solve
 *   to each column of A. This path is meant to help diagnose preconditioner behavior and conditioning.
 *   It is not a scalable path for routine use (use a distributed eigensolver instead).
 *
 * Numerics and outputs
 * - General (non-symmetric) problems may return complex-conjugate pairs. We write eigenvalues as text:
 *   one line per eigenvalue, either "Re Im" or just "Re" when purely real. If eigenvectors are requested,
 *   they are stored to a binary file in row-major layout (real or interleaved complex as Re,Im pairs
 *   reconstructed from LAPACK’s real-workspace output for dgeev).
 *
 * Build-time and configuration
 * - This feature is compiled only when enabled (e.g., -DHYPREDRV_ENABLE_EIGSPEC=ON).
 * - Runtime control is via the YAML eigenspectrum block
 *   (enable, vectors, hermitian, preconditioned, output_prefix).
 * - A guard rail rejects matrices with n larger than a conservative limit to avoid large memory usage
 *   allocations.
 *
 * In short: this file is intentionally simple and dense-LAPACK-centric to provide a convenient
 * debugging tool for small n. For large problems, adopt iterative or distributed eigensolvers
 * instead.
 */

/*--------------------------------------------------------------------------
 * Define Field/Offset/Setter mapping
 *--------------------------------------------------------------------------*/

#define EigSpec_FIELDS(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, enable,         FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, vectors,        FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, hermitian,      FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, preconditioned, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, output_prefix,  FieldTypeStringSet)

#define EigSpec_NUM_FIELDS (sizeof(EigSpec_field_offset_map) / sizeof(EigSpec_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(EigSpec)

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

StrIntMapArray
EigSpecGetValidValues(const char* key)
{
   if (!strcmp(key, "enable") ||
       !strcmp(key, "vectors") ||
       !strcmp(key, "hermitian") ||
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
EigSpecSetDefaultArgs(EigSpec_args *args)
{
   args->enable         = 0;
   args->vectors        = 0;
   args->hermitian      = 0;
   args->preconditioned = 0; /* A */
   snprintf(args->output_prefix, MAX_FILENAME_LENGTH, "%s", "eig");
}

#if defined(HYPREDRV_ENABLE_EIGSPEC)

/*-----------------------------------------------------------------------------
 * LAPACK Fortran symbols
 *-----------------------------------------------------------------------------*/

extern void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,
                   double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr,
                   double *work, int *lwork, int *info);
extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda,
                   double *w, double *work, int *lwork, int *info);

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_IJMatrixToDense(HYPRE_IJMatrix mat_A, int *n_ptr, double **A_cm_ptr)
{
   void                *obj_A = NULL;
   hypre_CSRMatrix     *seq_A = NULL;
   HYPRE_ParCSRMatrix   par_A = NULL;
   HYPRE_BigInt         nrows, ncols;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix) obj_A;

   /* Ensure on host memory for CPU LAPACK */
   hypre_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_HOST);

   HYPRE_ParCSRMatrixGetDims(par_A, &nrows, &ncols);
   if (nrows != ncols)
   {
      ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      ErrorMsgAdd("Eigenspectrum requires square matrix: got %lld x %lld", (long long) nrows, (long long) ncols);
      return ErrorCodeGet();
   }
   if (nrows <= 0)
   {
      ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      ErrorMsgAdd("Matrix has non-positive size: %lld", (long long) nrows);
      return ErrorCodeGet();
   }

   /* Gather on a single rank */
   seq_A = hypre_ParCSRMatrixToCSRMatrixAll(par_A);

   const int     n  = hypre_CSRMatrixNumRows(seq_A);
   const int    *di = hypre_CSRMatrixI(seq_A);
   const int    *dj = hypre_CSRMatrixJ(seq_A);
   const double *da = (const double*) hypre_CSRMatrixData(seq_A);
   size_t        elems;

   if ((double) n > sqrt((double) (((size_t) -1) / sizeof(double))))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Requested dense allocation would overflow size_t for n=%d", n);
      return ErrorCodeGet();
   }
   elems = (size_t) n * (size_t) n;

   /* Column-major dense matrix for LAPACK */
   double *A_cm = (double*) calloc(elems, sizeof(double));
   if (!A_cm)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate %zu bytes for dense matrix", elems * sizeof(double));
      return ErrorCodeGet();
   }

   /* Fill nonzero coefficients (owned columns) */
   for (int i = 0; i < n; i++)
   {
      for (int p = di[i]; p < di[i + 1]; p++)
      {
         A_cm[i + dj[p] * n] = da[p];
      }
   }

   *n_ptr = n;
   *A_cm_ptr = A_cm;
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_WriteValuesASCII(const char *prefix, int n, const double *wr, const double *wi)
{
   char  path[MAX_FILENAME_LENGTH];
   FILE *fp;

   snprintf(path, sizeof(path), "%s.values.txt", prefix);
   fp = fopen(path, "w");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open '%s' for writing", path);
      return ErrorCodeGet();
   }

   for (int i = 0; i < n; i++)
   {
      if (wi)
      {
         fprintf(fp, "%.17g %.17g\n", wr[i], wi[i]);
      }
      else
      {
         fprintf(fp, "%.17g\n", wr[i]);
      }
   }
   fclose(fp);
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Write vectors in row major format
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_WriteRealVectorsBin(const char *prefix, int n, const double *vecs_cm)
{
   char  path[MAX_FILENAME_LENGTH];
   FILE *fp;
   snprintf(path, sizeof(path), "%s.vectors.bin", prefix);
   fp = fopen(path, "wb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open '%s' for writing", path);
      return ErrorCodeGet();
   }

   /* Convert column-major eigenvectors (each column is a vector) to row-major rows */
   for (int ev = 0; ev < n; ev++)
   {
      for (int i = 0; i < n; i++)
      {
         double v = vecs_cm[i + ev * n];
         fwrite(&v, sizeof(double), 1, fp);
      }
   }

   fclose(fp);
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Write vectors in row major format
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_WriteComplexVectorsBin(const char *prefix, int n, const double *VR_cm, const double *wi)
{
   char  path[MAX_FILENAME_LENGTH];
   FILE *fp;
   snprintf(path, sizeof(path), "%s.vectors.bin", prefix);
   fp = fopen(path, "wb");
   if (!fp)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Could not open '%s' for writing", path);
      return ErrorCodeGet();
   }

   for (int ev = 0; ev < n; )
   {
      if (wi[ev] == 0.0)
      {
         for (int i = 0; i < n; i++)
         {
            double re = VR_cm[i + ev * n];
            double im = 0.0;
            fwrite(&re, sizeof(double), 1, fp);
            fwrite(&im, sizeof(double), 1, fp);
         }
         ev++;
      }
      else if (wi[ev] > 0.0 && ev + 1 < n)
      {
         /* First vector of the conjugate pair: v = vR + i vI */
         for (int i = 0; i < n; i++)
         {
            double re = VR_cm[i + ev * n];
            double im = VR_cm[i + (ev + 1) * n];
            fwrite(&re, sizeof(double), 1, fp);
            fwrite(&im, sizeof(double), 1, fp);
         }
         /* Second vector: conjugate v* = vR - i vI */
         for (int i = 0; i < n; i++)
         {
            double re = VR_cm[i + ev * n];
            double im = -VR_cm[i + (ev + 1) * n];
            fwrite(&re, sizeof(double), 1, fp);
            fwrite(&im, sizeof(double), 1, fp);
         }
         ev += 2;
      }
      else
      {
         /* Unexpected ordering; write zeros for safety and advance */
         for (int i = 0; i < n; i++)
         {
            double re = VR_cm[i + ev * n];
            double im = 0.0;
            fwrite(&re, sizeof(double), 1, fp);
            fwrite(&im, sizeof(double), 1, fp);
         }
         ev++;
      }
   }

   fclose(fp);
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_EigSpecComputeGeneral(int n, double *A_cm, int want_vectors,
                               double **wr_out, double **wi_out, double **vecs_cm_out)
{
   int      lda = n;
   int      ldv = n;
   char     jobvl = 'N';
   char     jobvr = want_vectors ? 'V' : 'N';
   int      info  = 0;
   double  *wr = (double*) malloc((size_t) n * sizeof(double));
   double  *wi = (double*) malloc((size_t) n * sizeof(double));
   double  *VR = want_vectors ? (double*) malloc((size_t) n * (size_t) n * sizeof(double)) : NULL;
   double  *VL = NULL;
   double   wkopt;
   int      lwork = -1;

   if (!wr || !wi || (want_vectors && !VR))
   {
      free(wr); free(wi); free(VR);
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Allocation failure for dgeev outputs (n=%d)", n);
      return ErrorCodeGet();
   }

   /* Workspace query */
   dgeev_(&jobvl, &jobvr, &n, A_cm, &lda, wr, wi, VL, &ldv, VR, &ldv, &wkopt, &lwork, &info);
   lwork = (int) wkopt;
   double *work = (double*) malloc((size_t) lwork * sizeof(double));
   if (!work)
   {
      free(wr); free(wi); free(VR);
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Allocation failure for dgeev workspace (lwork=%d)", lwork);
      return ErrorCodeGet();
   }

   /* Compute eigenvalues (+vectors) in-place on A_cm */
   dgeev_(&jobvl, &jobvr, &n, A_cm, &lda, wr, wi, VL, &ldv, VR, &ldv, work, &lwork, &info);
   free(work);
   if (info != 0)
   {
      free(wr); free(wi); free(VR);
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dgeev failed with info=%d", info);
      return ErrorCodeGet();
   }

   *wr_out = wr;
   *wi_out = wi;
   *vecs_cm_out = VR;
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static uint32_t
hypredrv_EigSpecComputeSymmetric(int n, double *A_cm, int want_vectors,
                                 double **w_out, double **vecs_cm_out)
{
   int     lda = n;
   char    jobz = want_vectors ? 'V' : 'N';
   char    uplo = 'U';
   int     info = 0;
   double *w = (double*) malloc((size_t) n * sizeof(double));
   double  wkopt;
   int     lwork = -1;

   if (!w)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Allocation failure for dsyev eigenvalues (n=%d)", n);
      return ErrorCodeGet();
   }

   /* Workspace query */
   dsyev_(&jobz, &uplo, &n, A_cm, &lda, w, &wkopt, &lwork, &info);
   lwork = (int) wkopt;
   double *work = (double*) malloc((size_t) lwork * sizeof(double));
   if (!work)
   {
      free(w);
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Allocation failure for dsyev workspace (lwork=%d)", lwork);
      return ErrorCodeGet();
   }

   /* Compute; A_cm overwritten with eigenvectors if jobz='V' */
   dsyev_(&jobz, &uplo, &n, A_cm, &lda, w, work, &lwork, &info);
   free(work);
   if (info != 0)
   {
      free(w);
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dsyev failed with info=%d", info);
      return ErrorCodeGet();
   }

   *w_out = w;
   *vecs_cm_out = want_vectors ? A_cm : NULL;
   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_EigSpecCompute(const EigSpec_args *eargs,
                        void *imat_A,
                        void *precon_ctx,
                        hypredrv_PreconApplyFn precon_apply)
{
   if (!eargs || !eargs->enable)
   {
      return ErrorCodeGet();
   }

   HYPRE_IJMatrix mat_A = (HYPRE_IJMatrix) imat_A;
   if (!mat_A)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Matrix not built yet for eigenspectrum computation");
      return ErrorCodeGet();
   }
   MPI_Comm comm = hypre_IJMatrixComm((hypre_IJMatrix*) mat_A);

   /* Use "solve" timer bucket */
   StatsTimerStart("solve");

   /* Convert sparse matrix to dense (column-major) */
   int     n = 0;
   double *A_cm = NULL;
   hypredrv_IJMatrixToDense(mat_A, &n, &A_cm);
   if (ErrorCodeActive())
   {
      free(A_cm);
      StatsTimerStop("solve");
      return ErrorCodeGet();
   }

   /* Guard rails for very large n */
   const int n_guard = 20000;
   if (n > n_guard)
   {
      free(A_cm);
      ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      ErrorMsgAdd("Eigenspectrum guard: n=%d exceeds limit=%d", n, n_guard);
      StatsTimerStop("solve");
      return ErrorCodeGet();
   }

   /* Build preconditioned dense matrix B = M^{-1} A if requested */
   if (eargs->preconditioned)
   {
      HYPRE_BigInt jlower, jupper;
      HYPRE_IJVector e_i = NULL, t = NULL, col = NULL;

      HYPRE_IJMatrixGetLocalRange(mat_A, &jlower, &jupper, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, &e_i);
      HYPRE_IJVectorSetObjectType(e_i, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(e_i, HYPRE_MEMORY_HOST);
      HYPRE_IJVectorAssemble(e_i);
      HYPRE_IJVectorCreate(comm, jlower, jupper, &t);
      HYPRE_IJVectorSetObjectType(t, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(t, HYPRE_MEMORY_HOST);
      HYPRE_IJVectorAssemble(t);
      HYPRE_IJVectorCreate(comm, jlower, jupper, &col);
      HYPRE_IJVectorSetObjectType(col, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(col, HYPRE_MEMORY_HOST);
      HYPRE_IJVectorAssemble(col);

      double *B_cm = (double*) calloc((size_t) n * (size_t) n, sizeof(double));
      int loc_n = (int) (jupper - jlower + 1);
      for (int i = 0; i < n; i++)
      {
         /* Set t to A(:,i) using A_cm and apply preconditioner solve: col = M^{-1} t */
         void *obj_t, *obj_c;
         HYPRE_ParVector par_t, par_c;
         HYPRE_IJVectorGetObject(t, &obj_t);
         par_t = (HYPRE_ParVector) obj_t;
         hypre_Vector *v_loc_t = hypre_ParVectorLocalVector(par_t);
         double *tdata = hypre_VectorData(v_loc_t);
         for (int r = 0; r < loc_n; r++) tdata[r] = A_cm[(int)(jlower) + r + i * n];

         HYPRE_IJVectorGetObject(col, &obj_c);
         par_c = (HYPRE_ParVector) obj_c;
         hypre_Vector *v_loc_c = hypre_ParVectorLocalVector(par_c);
         double *cdata = hypre_VectorData(v_loc_c);
         for (int r = 0; r < loc_n; r++) cdata[r] = 0.0;

         if (precon_apply)
         {
            precon_apply(precon_ctx, (void*) t, (void*) col);
         }

         for (int r = 0; r < loc_n; r++)
         {
            int global_r = (int) jlower + r;
            B_cm[global_r + i * n] = cdata[r];
         }
      }

      free(A_cm);
      A_cm = B_cm;

      HYPRE_IJVectorDestroy(e_i);
      HYPRE_IJVectorDestroy(t);
      HYPRE_IJVectorDestroy(col);
   }

   if (eargs->hermitian)
   {
      double *w = NULL;      /* eigenvalues */
      double *Z_cm = NULL;   /* eigenvectors in column-major (returned in A_cm) */
      hypredrv_EigSpecComputeSymmetric(n, A_cm, eargs->vectors, &w, &Z_cm);
      if (!ErrorCodeActive())
      {
         const char *prefix = eargs->output_prefix[0] ? eargs->output_prefix : "eig";
         hypredrv_WriteValuesASCII(prefix, n, w, NULL);
         if (eargs->vectors && Z_cm)
         {
            hypredrv_WriteRealVectorsBin(prefix, n, Z_cm);
         }
      }
      free(w);
      free(A_cm);
   }
   else
   {
      double *wr = NULL;     /* real parts */
      double *wi = NULL;     /* imaginary parts */
      double *VR_cm = NULL;  /* right eigenvectors in column-major */
      hypredrv_EigSpecComputeGeneral(n, A_cm, eargs->vectors, &wr, &wi, &VR_cm);
      if (!ErrorCodeActive())
      {
         const char *prefix = eargs->output_prefix[0] ? eargs->output_prefix : "eig";
         hypredrv_WriteValuesASCII(prefix, n, wr, wi);
         if (eargs->vectors && VR_cm)
         {
            hypredrv_WriteComplexVectorsBin(prefix, n, VR_cm, wi);
         }
      }
      free(wr);
      free(wi);
      free(VR_cm);
      free(A_cm);
   }

   StatsTimerStop("solve");

   return ErrorCodeGet();
}

#endif /* HYPREDRV_ENABLE_EIGSPEC */
