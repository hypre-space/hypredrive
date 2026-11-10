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

#endif /* HYPREDRV_ENABLE_EIGSPEC */

uint32_t
hypredrv_EigSpecCompute(hypredrv_t *hypredrv)
{
   uint32_t status = ERROR_NONE;
#ifdef HYPREDRV_ENABLE_EIGSPEC
   const EigSpec_args *eargs = &hypredrv->iargs->ls.eigspec;
   MPI_Comm            comm  = hypredrv->comm;
   int                 myid  = hypredrv->mypid;
   HYPRE_IJMatrix      mat_A = hypredrv->mat_A.handle;
   hypre_ParCSRMatrix *par_A = NULL, *host_A = NULL;
   hypre_CSRMatrix    *seq_A = NULL;
   HYPRE_IJVector      rhs = NULL, solution = NULL;
   HYPRE_Complex      *rhs_values = NULL, *solution_values = NULL;
   double             *A_cm = NULL, *B_cm = NULL;
   double             *wr = NULL, *wi = NULL, *VR_cm = NULL, *work = NULL;
   HYPRE_BigInt        nrows = 0, ncols = 0;
   HYPRE_BigInt        ilower = 0, iupper = -1, jlower = 0, jupper = -1;
   int                 n = 0, local_rows = 0, local_cols = 0;
   bool                timed = false;

   if (!eargs->enable)
   {
      HYPREDRV_LOG_OBJECTF(
         2, hypredrv,
         "eigenspectrum computation skipped: linear_system.eigspec.enable=0");
      goto done;
   }
   if (!myid)
   {
      printf("[EigenSpectrum] | mode=%s | vectors=%s | prefix='%s'\n",
             eargs->hermitian ? "Hermitian" : "General", eargs->vectors ? "on" : "off",
             eargs->output_prefix[0] ? eargs->output_prefix : "eig");
      fflush(stdout);
   }

   /* Validate the external operator before gathering or allocating dense storage.
    * The size bound also makes n*n safe as an MPI count and a LAPACK dimension. */
   if (!mat_A)
   {
      status = ERROR_FILE_NOT_FOUND;
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Matrix not built yet for eigenspectrum computation");
   }
   else
   {
      void     *matrix_object = NULL;
      HYPRE_Int ierr          = HYPRE_IJMatrixGetObject(mat_A, &matrix_object);
      par_A                   = (hypre_ParCSRMatrix *)matrix_object;
      if (ierr || !par_A ||
          HYPRE_ParCSRMatrixGetDims((HYPRE_ParCSRMatrix)par_A, &nrows, &ncols) ||
          HYPRE_IJMatrixGetLocalRange(mat_A, &ilower, &iupper, &jlower, &jupper))
      {
         status = ERROR_HYPRE_INTERNAL;
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
         hypredrv_ErrorMsgAdd("Could not inspect the eigenspectrum matrix");
      }
      else if (nrows != ncols || nrows <= 0 || nrows > 20000)
      {
         status = ERROR_OUT_OF_BOUNDS;
         hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
         hypredrv_ErrorMsgAdd("Eigenspectrum requires a square matrix with 1 to 20000 "
                              "rows: got %lld x %lld",
                              (long long)nrows, (long long)ncols);
      }
      else if ((size_t)nrows > SIZE_MAX / sizeof(double) / (size_t)nrows)
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
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   n          = (int)nrows;
   local_rows = (int)(iupper - ilower + 1);
   local_cols = (int)(jupper - jlower + 1);

   /* Prepare the actual preconditioner once. A failed create/setup must not be
    * lost to another public entry point or produce a spectrum of a zero matrix. */
   if (eargs->preconditioned)
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: preconditioned operator");
      status = hypredrv_ObjectPreconCreate(hypredrv);
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
      if (status)
      {
         goto cleanup;
      }
      status = hypredrv_ObjectPreconSetup(hypredrv);
      if (status)
      {
         goto cleanup;
      }
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "eigenspectrum computation path: unpreconditioned operator");
   }

   status = hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   timed  = (status == ERROR_NONE);
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }

   /* Older hypre gather implementations require host storage. Clone device
    * matrices instead of migrating the caller's operator and invalidating any
    * cached preconditioner data or changing its execution space. */
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
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   seq_A = hypre_ParCSRMatrixToCSRMatrixAll(gather_A);
   if (host_A)
   {
      hypre_ParCSRMatrixDestroy(host_A);
      host_A = NULL;
   }
   if ((!seq_A && local_rows > 0) || (seq_A && hypre_CSRMatrixNumRows(seq_A) != n))
   {
      status = ERROR_HYPRE_INTERNAL;
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("Failed to gather the eigenspectrum matrix");
   }
   A_cm = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
   if (!A_cm)
   {
      status |=
         hypredrv_AllocationFailure("Could not allocate eigenspectrum dense matrix");
   }
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (status)
   {
      goto cleanup;
   }
   int gather_rank = seq_A ? myid : INT_MAX;
   MPI_Allreduce(MPI_IN_PLACE, &gather_rank, 1, MPI_INT, MPI_MIN, comm);
   if (gather_rank == INT_MAX)
   {
      status = ERROR_HYPRE_INTERNAL;
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("No rank received the eigenspectrum matrix");
      goto cleanup;
   }
   if (myid == gather_rank)
   {
      const HYPRE_Int     *di     = hypre_CSRMatrixI(seq_A);
      const HYPRE_Int     *dj     = hypre_CSRMatrixJ(seq_A);
      const HYPRE_Complex *values = hypre_CSRMatrixData(seq_A);
      for (int i = 0; i < n; i++)
      {
         for (HYPRE_Int entry = di[i]; entry < di[i + 1]; entry++)
         {
            A_cm[(size_t)i + (size_t)dj[entry] * (size_t)n] =
               (double)hypre_creal(values[entry]);
         }
      }
   }
   if (seq_A)
   {
      hypre_CSRMatrixDestroy(seq_A);
      seq_A = NULL;
   }
   /* A rank with no rows can legitimately receive no gathered CSR. It still
    * participates in preconditioning and may be the LAPACK/output rank. */
   MPI_Bcast(A_cm, n * n, MPI_DOUBLE, gather_rank, comm);

   if (eargs->preconditioned)
   {
      B_cm       = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
      rhs_values = (HYPRE_Complex *)hypredrv_Calloc(
         (size_t)local_rows, sizeof(HYPRE_Complex), "eigenspectrum RHS");
      solution_values = (HYPRE_Complex *)hypredrv_Calloc(
         (size_t)local_cols, sizeof(HYPRE_Complex), "eigenspectrum solution");
      if (!B_cm || !rhs_values || !solution_values)
      {
         status = hypredrv_AllocationFailure(
            "Could not allocate preconditioned eigenspectrum buffers");
      }
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
      if (status)
      {
         goto cleanup;
      }
      HYPRE_MemoryLocation location = hypre_ParCSRMatrixMemoryLocation(par_A);
      HYPRE_Int            ierr     = HYPRE_IJVectorCreate(comm, ilower, iupper, &rhs);
      if (!ierr) ierr = HYPRE_IJVectorSetObjectType(rhs, HYPRE_PARCSR);
      if (!ierr) ierr = HYPRE_IJVectorInitialize_v2(rhs, location);
      if (!ierr) ierr = HYPRE_IJVectorCreate(comm, jlower, jupper, &solution);
      if (!ierr) ierr = HYPRE_IJVectorSetObjectType(solution, HYPRE_PARCSR);
      if (!ierr) ierr = HYPRE_IJVectorInitialize_v2(solution, location);
      if (ierr)
      {
         status = ERROR_HYPRE_INTERNAL;
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
         hypredrv_ErrorMsgAdd("Could not initialize eigenspectrum work vectors");
      }
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
      if (status)
      {
         goto cleanup;
      }
      ierr = HYPRE_IJVectorAssemble(rhs);
      ierr |= HYPRE_IJVectorAssemble(solution);
      void *rhs_object = NULL, *solution_object = NULL;
      ierr |= HYPRE_IJVectorGetObject(rhs, &rhs_object);
      ierr |= HYPRE_IJVectorGetObject(solution, &solution_object);
      if (ierr || !rhs_object || !solution_object)
      {
         status = ERROR_HYPRE_INTERNAL;
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
         hypredrv_ErrorMsgAdd("Could not assemble eigenspectrum work vectors");
      }
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
      if (status)
      {
         goto cleanup;
      }
      hypre_Vector *local_rhs = hypre_ParVectorLocalVector((hypre_ParVector *)rhs_object);
      hypre_Vector *local_solution =
         hypre_ParVectorLocalVector((hypre_ParVector *)solution_object);
      for (int column = 0; column < n; column++)
      {
         for (int row = 0; row < local_rows; row++)
         {
            rhs_values[row] = (HYPRE_Complex)
               A_cm[(size_t)ilower + (size_t)row + (size_t)column * (size_t)n];
         }
         if (local_rows)
         {
            hypre_TMemcpy(hypre_VectorData(local_rhs), rhs_values, HYPRE_Complex,
                          local_rows, hypre_VectorMemoryLocation(local_rhs),
                          HYPRE_MEMORY_HOST);
         }
         HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)solution_object, 0.0);
         status = hypredrv_ObjectPreconApply(hypredrv, rhs, solution);
         MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
         if (status)
         {
            goto cleanup;
         }
         if (local_cols)
         {
            hypre_TMemcpy(solution_values, hypre_VectorData(local_solution),
                          HYPRE_Complex, local_cols, HYPRE_MEMORY_HOST,
                          hypre_VectorMemoryLocation(local_solution));
         }
         for (int row = 0; row < local_cols; row++)
         {
            B_cm[(size_t)jlower + (size_t)row + (size_t)column * (size_t)n] =
               (double)hypre_creal(solution_values[row]);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, B_cm, n * n, MPI_DOUBLE, MPI_SUM, comm);
      free(A_cm);
      A_cm = B_cm;
      B_cm = NULL;
   }

   /* LAPACK and file output need only one copy of the already assembled operator.
    * Other ranks join the final status reduction, including root-only I/O errors. */
   if (!myid)
   {
      wr = (double *)malloc((size_t)n * sizeof(double));
      if (!eargs->hermitian)
      {
         wi = (double *)malloc((size_t)n * sizeof(double));
         if (eargs->vectors)
         {
            VR_cm = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
         }
      }
      if (!wr || (!eargs->hermitian && (!wi || (eargs->vectors && !VR_cm))))
      {
         status = hypredrv_AllocationFailure(
            "Could not allocate LAPACK eigenvalue/vector buffers");
         goto cleanup;
      }
      char   jobz = eargs->vectors ? 'V' : 'N', jobvl = 'N', uplo = 'U';
      int    lda = n, ldv = n, info = 0, lwork = -1;
      double optimal_work = 0.0;
      if (eargs->hermitian)
      {
         dsyev_(&jobz, &uplo, &n, A_cm, &lda, wr, &optimal_work, &lwork, &info);
      }
      else
      {
         dgeev_(&jobvl, &jobz, &n, A_cm, &lda, wr, wi, NULL, &ldv, VR_cm, &ldv,
                &optimal_work, &lwork, &info);
      }
      if (info || !isfinite(optimal_work) || optimal_work < 1.0 ||
          optimal_work > INT_MAX || optimal_work > (double)(SIZE_MAX / sizeof(double)))
      {
         status = ERROR_UNKNOWN;
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Invalid %s workspace query: info=%d, size=%.17g",
                              eargs->hermitian ? "dsyev" : "dgeev", info, optimal_work);
         goto cleanup;
      }
      lwork = (int)optimal_work;
      work  = (double *)malloc((size_t)lwork * sizeof(double));
      if (!work)
      {
         status = hypredrv_AllocationFailure("Could not allocate LAPACK workspace");
         goto cleanup;
      }
      if (eargs->hermitian)
      {
         dsyev_(&jobz, &uplo, &n, A_cm, &lda, wr, work, &lwork, &info);
      }
      else
      {
         dgeev_(&jobvl, &jobz, &n, A_cm, &lda, wr, wi, NULL, &ldv, VR_cm, &ldv, work,
                &lwork, &info);
      }
      if (info)
      {
         status = ERROR_UNKNOWN;
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("%s failed with info=%d",
                              eargs->hermitian ? "dsyev" : "dgeev", info);
         goto cleanup;
      }
      const char *prefix = eargs->output_prefix[0] ? eargs->output_prefix : "eig";
      char        path[MAX_FILENAME_LENGTH + 32];
      snprintf(path, sizeof(path), "%s.values.txt", prefix);
      FILE *fp = fopen(path, "w");
      if (!fp)
      {
         status = ERROR_FILE_NOT_FOUND;
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Could not open '%s' for writing", path);
         goto cleanup;
      }
      for (int i = 0; i < n; i++)
      {
         if (wi) fprintf(fp, "%.17g %.17g\n", wr[i], wi[i]);
         else fprintf(fp, "%.17g\n", wr[i]);
      }
      status = EigspecCloseOutput(fp, path);
      if (status)
      {
         goto cleanup;
      }
      if (eargs->vectors)
      {
         snprintf(path, sizeof(path), "%s.vectors.bin", prefix);
         fp = fopen(path, "wb");
         if (!fp)
         {
            status = ERROR_FILE_NOT_FOUND;
            hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
            hypredrv_ErrorMsgAdd("Could not open '%s' for writing", path);
            goto cleanup;
         }
         if (eargs->hermitian)
         {
            fwrite(A_cm, sizeof(double), (size_t)n * (size_t)n, fp);
         }
         else
         {
            /* LAPACK stores a complex conjugate pair in adjacent real columns. */
            for (int ev = 0; ev < n;)
            {
               if (wi[ev] == 0.0)
               {
                  for (int i = 0; i < n; i++)
                  {
                     double pair[2] = {VR_cm[i + ev * n], 0.0};
                     fwrite(pair, sizeof(double), 2, fp);
                  }
                  ev++;
               }
               else if (wi[ev] > 0.0 && ev + 1 < n)
               {
                  for (int conjugate = 0; conjugate < 2; conjugate++)
                  {
                     for (int i = 0; i < n; i++)
                     {
                        double pair[2] = {VR_cm[i + ev * n], (conjugate ? -1.0 : 1.0) *
                                                                VR_cm[i + (ev + 1) * n]};
                        fwrite(pair, sizeof(double), 2, fp);
                     }
                  }
                  ev += 2;
               }
               else
               {
                  for (int i = 0; i < n; i++)
                  {
                     double pair[2] = {VR_cm[i + ev * n], 0.0};
                     fwrite(pair, sizeof(double), 2, fp);
                  }
                  ev++;
               }
            }
         }
         status = EigspecCloseOutput(fp, path);
      }
   }

cleanup:
   if (seq_A) hypre_CSRMatrixDestroy(seq_A);
   if (host_A) hypre_ParCSRMatrixDestroy(host_A);
   if (rhs) HYPRE_IJVectorDestroy(rhs);
   if (solution) HYPRE_IJVectorDestroy(solution);
   free(rhs_values);
   free(solution_values);
   free(A_cm);
   free(B_cm);
   free(wr);
   free(wi);
   free(VR_cm);
   free(work);
   if (timed)
   {
      status |= hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "solve");
   }
   MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_UINT32_T, MPI_BOR, comm);
   /* Populate remote failures without resetting any diagnostics already held by
    * this caller. The return value reports only this operation's own status. */
   uint32_t missing = status & ~hypredrv_ErrorCodeGet();
   for (uint32_t bit = 1; bit; bit <<= 1)
   {
      if (missing & bit) hypredrv_ErrorCodeSet((hypredrv_error_t)bit);
   }
done:
#else
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
#endif
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPREDRV_LinearSystemComputeEigenspectrum end (code=0x%x)", status);
   return status;
}
