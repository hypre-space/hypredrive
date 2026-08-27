/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdint.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_utilities.h" // for hypre_TAlloc, hypre_TMemcpy, hypre_TFree
#include "internal/utils.h"

enum
{
   IJMATRIX_MAX_PART_NNZ   = 200u * 1000u * 1000u,
   IJMATRIX_MAX_PART_NROWS = 200u * 1000u * 1000u,
};

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert((size_t)IJMATRIX_MAX_PART_NNZ <= SIZE_MAX / sizeof(HYPRE_BigInt),
               "IJ matrix part nnz fits HYPRE_BigInt allocation");
_Static_assert((size_t)IJMATRIX_MAX_PART_NNZ <= SIZE_MAX / sizeof(HYPRE_Complex),
               "IJ matrix part nnz fits HYPRE_Complex allocation");
_Static_assert((size_t)IJMATRIX_MAX_PART_NROWS <= SIZE_MAX / sizeof(HYPRE_Int),
               "IJ matrix part nrows fits HYPRE_Int allocation");
_Static_assert((HYPRE_BigInt)-1 < 0,
               "IJ matrix index validation requires signed HYPRE_BigInt");
#else
typedef char
   hypredrv_matrix_requires_signed_hypre_bigint[((HYPRE_BigInt)-1 < 0) ? 1 : -1];
#endif

/* Host staging buffers for one part, plus the arrays actually handed to hypre
 * (identical to the host buffers unless the matrix is device-resident). */
typedef struct
{
   HYPRE_BigInt  *h_rows;
   HYPRE_BigInt  *h_cols;
   HYPRE_Complex *h_vals;
   HYPRE_BigInt  *rows;
   HYPRE_BigInt  *cols;
   HYPRE_Complex *vals;
} IJMatrixEntryBuffers;

static int
IJMatrixValidateHeader(const uint64_t *header, const char *filename)
{
   uint64_t nrows = 0;

   /* GCOVR_EXCL_START */
   if (!header)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null matrix part header");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   /* GCOVR_EXCL_START */
   if (header[8] < header[7])
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd(
         "Invalid matrix row range in %s: row_upper (%llu) < row_lower (%llu)",
         filename ? filename : "(unknown)", (unsigned long long)header[8],
         (unsigned long long)header[7]);
      return 0;
   }

   nrows = header[8] - header[7] + 1u;
   if (nrows > (uint64_t)IJMATRIX_MAX_PART_NROWS)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix row count exceeds per-part limit in %s (%llu rows)",
                           filename ? filename : "(unknown)", (unsigned long long)nrows);
      return 0;
   }
   if (header[6] > (uint64_t)IJMATRIX_MAX_PART_NNZ)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix nnz exceeds per-part limit in %s (%llu entries)",
                           filename ? filename : "(unknown)",
                           (unsigned long long)header[6]);
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   return 1;
}

static int
IJMatrixPartNnzMatchesPrepass(size_t nnzs_max, uint64_t part_nnz, const char *filename)
{
   if (part_nnz > (uint64_t)nnzs_max)
   {
      /* GCOVR_EXCL_START */
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix part nnz exceeds pre-scan maximum at %s",
                           filename ? filename : "(unknown)");
      return 0;
      /* GCOVR_EXCL_STOP */
   }
   return 1;
}

static int
IJMatrixValidateEntry(HYPRE_BigInt row, HYPRE_BigInt col, uint64_t nrows, uint64_t ncols,
                      const char *filename)
{
   if (row < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Detected negative matrix row %lld while reading %s",
                           (long long)row, filename ? filename : "(unknown)");
      return 0;
   }
   if (col < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Detected negative matrix column %lld while reading %s",
                           (long long)col, filename ? filename : "(unknown)");
      return 0;
   }
   if ((uint64_t)row >= nrows)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Detected out-of-bounds matrix row %llu while reading %s",
                           (unsigned long long)row, filename ? filename : "(unknown)");
      return 0;
   }
   if ((uint64_t)col >= ncols)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Detected out-of-bounds matrix column %llu while reading %s",
                           (unsigned long long)col, filename ? filename : "(unknown)");
      return 0;
   }

   return 1;
}

static int
IJMatrixRejectNonfiniteCoefficient(const char *filename)
{
   hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
   hypredrv_ErrorMsgAdd("Detected non-finite matrix coefficient while reading %s",
                        filename ? filename : "(unknown)");
   return 0;
}

/* Data-type widths accepted for on-disk row/column index arrays. */
static int
IJMatrixIndexDtypeIsValid(uint64_t isize)
{
   return (isize == sizeof(HYPRE_BigInt) || isize == sizeof(uint32_t) ||
           isize == sizeof(uint64_t));
}

/* Opens part `partid` of a multipart matrix, then reads and validates its
 * 11-word header. Returns a stream positioned just past the header, or NULL
 * with the error state set (the stream is closed on every failure path).
 * `missing_is_not_found` selects the error reported when the file cannot be
 * opened; `check_prepass` additionally cross-checks the part nnz against the
 * pre-scan maximum `nnzs_max`. */
static FILE *
IJMatrixOpenPart(const char *prefixname, uint32_t partid, char *filename,
                 size_t filename_size, uint64_t *header, size_t nnzs_max,
                 int check_prepass, int missing_is_not_found)
{
   FILE *fp = NULL;

   snprintf(filename, filename_size, "%s.%05d.bin", prefixname, (int)partid);
   fp = fopen(filename, "rb");
   if (!fp)
   {
      if (missing_is_not_found)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
      }
      /* GCOVR_EXCL_START */
      else
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not read header from %s", filename);
      }
      /* GCOVR_EXCL_STOP */
      return NULL;
   }

   if (fread(header, sizeof(uint64_t), 11, fp) != 11)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read header from %s", filename);
      fclose(fp);
      return NULL;
   }

   if (!IJMatrixValidateHeader(header, filename) ||
       (check_prepass && !IJMatrixPartNnzMatchesPrepass(nnzs_max, header[6], filename)))
   {
      fclose(fp);
      return NULL;
   }

   return fp;
}

/* Reads `nnz` indices stored as `isize`-byte unsigned integers into `dst`,
 * widening them to HYPRE_BigInt. `scratch` must hold at least `nnz` elements of
 * `isize` bytes; it is unused when the on-disk width already matches. */
static int
IJMatrixReadIndexArray(FILE *fp, HYPRE_BigInt *dst, uint64_t nnz, uint64_t isize,
                       void *scratch, const char *filename, const char *what)
{
   if (isize == sizeof(HYPRE_BigInt))
   {
      if (fread(dst, sizeof(HYPRE_BigInt), nnz, fp) != nnz)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not read %s indices from %s", what, filename);
         return 0;
      }
      return 1;
   }

   /* Alternate on-disk index widths are build-/format-dependent and are not
    * exercised by the default test corpus. */
   /* GCOVR_EXCL_START */
   /* LCOV_EXCL_START */
   if (fread(scratch, (size_t)isize, nnz, fp) != nnz)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read %s indices from %s", what, filename);
      return 0;
   }

   if (isize == sizeof(uint32_t))
   {
      const uint32_t *src = (const uint32_t *)scratch;

      for (size_t i = 0; i < nnz; i++)
      {
         dst[i] = (HYPRE_BigInt)src[i];
      }
   }
   else
   {
      const uint64_t *src = (const uint64_t *)scratch;

      for (size_t i = 0; i < nnz; i++)
      {
         dst[i] = (HYPRE_BigInt)src[i];
      }
   }

   return 1;
   /* LCOV_EXCL_STOP */
   /* GCOVR_EXCL_STOP */
}

/* Reads the row and column index arrays of one part into `h_rows`/`h_cols`. */
static int
IJMatrixReadIndexPair(FILE *fp, const uint64_t *header, size_t nnzs_max,
                      HYPRE_BigInt *h_rows, HYPRE_BigInt *h_cols, const char *filename)
{
   const uint64_t isize   = header[1];
   const uint64_t nnz     = header[6];
   void          *scratch = NULL;
   int            status  = 0;

   if (!IJMatrixIndexDtypeIsValid(isize))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid row/col data type size %lld at %s", (long long)isize,
                           filename);
      return 0;
   }

   /* Nothing to decode: zero-length reads consume no bytes from the stream. */
   if (nnz == 0 || !h_rows || !h_cols)
   {
      return 1;
   }

   /* GCOVR_EXCL_START */
   if (isize != sizeof(HYPRE_BigInt))
   {
      scratch = malloc((size_t)nnzs_max * (size_t)isize);
      if (!scratch)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate uint%llu index buffer for %s",
                              (unsigned long long)isize * 8u, filename);
         return 0;
      }
   }
   /* GCOVR_EXCL_STOP */

   status = IJMatrixReadIndexArray(fp, h_rows, nnz, isize, scratch, filename, "row") &&
            IJMatrixReadIndexArray(fp, h_cols, nnz, isize, scratch, filename, "column");

   free(scratch);

   return status;
}

/* Reads the coefficient array of one part into `h_vals`, widening from the
 * on-disk float/double representation and rejecting non-finite entries. */
static int
IJMatrixReadCoefficients(FILE *fp, const uint64_t *header, size_t nnzs_max,
                         HYPRE_Complex *h_vals, const char *filename)
{
   const uint64_t vsize  = header[2];
   const uint64_t nnz    = header[6];
   void          *buffer = NULL;
   int            status = 1;

   /* GCOVR_EXCL_BR_START */
   if (vsize != sizeof(float) && vsize != sizeof(double)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid coefficient data type size %lld at %s",
                           (long long)vsize, filename);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (nnz == 0 || !h_vals) /* GCOVR_EXCL_BR_STOP */
   {
      return 1;
   }

   buffer = malloc((size_t)nnzs_max * (size_t)vsize);
   /* GCOVR_EXCL_BR_START */
   if (!buffer || fread(buffer, (size_t)vsize, nnz, fp) != nnz) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read coeficients from %s", filename);
      free(buffer);
      return 0;
   }

   /* GCOVR_EXCL_BR_START */
   if (vsize == sizeof(float)) /* GCOVR_EXCL_BR_STOP */
   {
      const float *src = (const float *)buffer;

      for (size_t i = 0; i < nnz; i++)
      {
         if (!hypredrv_FloatIsFinite(src[i]))
         {
            status = IJMatrixRejectNonfiniteCoefficient(filename);
            break;
         }
         h_vals[i] = (HYPRE_Complex)src[i];
      }
   }
   else
   {
      const double *src = (const double *)buffer;

      for (size_t i = 0; i < nnz; i++)
      {
         if (!hypredrv_DoubleIsFinite(src[i]))
         {
            status = IJMatrixRejectNonfiniteCoefficient(filename);
            break;
         }
         h_vals[i] = (HYPRE_Complex)src[i];
      }
   }

   free(buffer);

   return status;
}

/* First pass: reads every part header to accumulate this rank's local row count
 * and the largest per-part nnz, which bounds the entry read buffers. */
static int
IJMatrixScanParts(const char *prefixname, const uint32_t *partids, uint32_t nparts,
                  uint64_t *nrows_sum_out, size_t *nnzs_max_out)
{
   char     filename[1024];
   uint64_t header[11];
   uint64_t nrows_sum = 0;
   size_t   nnzs_max  = 0;

   for (uint32_t part = 0; part < nparts; part++)
   {
      uint64_t part_nrows = 0;
      FILE *fp = IJMatrixOpenPart(prefixname, partids[part], filename, sizeof(filename),
                                  header, 0, 0, 1);

      if (!fp)
      {
         return 0;
      }
      fclose(fp);

      part_nrows = (uint64_t)(header[8] - header[7] + 1u);
      /* GCOVR_EXCL_START */
      if (nrows_sum > UINT64_MAX - part_nrows)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Matrix local row count overflow while reading %s",
                              filename);
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      nrows_sum += part_nrows;
      nnzs_max = ((size_t)header[6] > nnzs_max) ? (size_t)header[6] : nnzs_max;
   }

   *nrows_sum_out = nrows_sum;
   *nnzs_max_out  = nnzs_max;

   return 1;
}

/* Builds this rank's slice of the global part id map. */
static int
IJMatrixBuildPartIds(uint64_t first_part, uint32_t nparts, uint32_t **partids_out)
{
   uint32_t *partids = NULL;

   /* GCOVR_EXCL_START */
   if (nparts > (uint32_t)(SIZE_MAX / sizeof(uint32_t)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix part id count exceeds allocation bounds");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   partids = (uint32_t *)malloc(nparts * sizeof(uint32_t));
   /* GCOVR_EXCL_START */
   if (nparts > 0 && !partids)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate matrix part id map (%u entries)", nparts);
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   for (uint32_t part = 0; part < nparts; part++)
   {
      partids[part] = (uint32_t)(first_part + part);
   }

   *partids_out = partids;

   return 1;
}

/* Allocates the host staging buffers sized by the pre-scan maximum part nnz. */
static int
IJMatrixAllocEntryBuffers(size_t nnzs_max, IJMatrixEntryBuffers *buf)
{
   if (nnzs_max == 0)
   {
      return 1;
   }

   buf->h_rows = (HYPRE_BigInt *)malloc(nnzs_max * sizeof(HYPRE_BigInt));
   buf->h_cols = (HYPRE_BigInt *)malloc(nnzs_max * sizeof(HYPRE_BigInt));
   buf->h_vals = (HYPRE_Complex *)malloc(nnzs_max * sizeof(HYPRE_Complex));

   /* GCOVR_EXCL_START */
   if (!buf->h_rows || !buf->h_cols || !buf->h_vals)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate matrix read buffers (%zu entries)",
                           nnzs_max);
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   return 1;
}

/* Tallies one part's entries into the per-local-row diagonal/off-diagonal counts. */
static int
IJMatrixCountPartSparsity(const HYPRE_BigInt *h_rows, const HYPRE_BigInt *h_cols,
                          uint64_t nnz, uint64_t nrows, HYPRE_BigInt ilower,
                          HYPRE_BigInt iupper, uint64_t nrows_sum, HYPRE_Int *dsizes,
                          HYPRE_Int *osizes, const char *filename)
{
   /* GCOVR_EXCL_BR_START */
   if (!h_rows || !h_cols) /* GCOVR_EXCL_BR_STOP */
   {
      return 1;
   }

   /* TODO: add threading */
   for (size_t i = 0; i < nnz; i++)
   {
      const HYPRE_BigInt row       = h_rows[i];
      const HYPRE_BigInt col       = h_cols[i];
      size_t             local_row = 0;

      /* Multipart IJ matrices are created as square matrices in this reader. */
      if (!IJMatrixValidateEntry(row, col, nrows, nrows, filename))
      {
         return 0;
      }
      if (row < ilower || row > iupper)
      {
         /* This row does not belong to the current rank. Skipping it... */
         continue;
      }

      local_row = (size_t)(row - ilower);
      /* GCOVR_EXCL_START */
      if (local_row >= nrows_sum)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd(
            "Matrix local row index exceeds precompute bounds while reading %s",
            filename);
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      if (col >= ilower && col <= iupper)
      {
         dsizes[local_row]++;
      }
      else
      {
         osizes[local_row]++;
      }
   }

   return 1;
}

/* Host path only: replays every part's index arrays to count diagonal and
 * off-diagonal entries per local row, then pre-sizes the IJ matrix. */
static int
IJMatrixPrecomputeHostSparsity(HYPRE_IJMatrix mat, const char *prefixname,
                               const uint32_t *partids, uint32_t nparts, size_t nnzs_max,
                               IJMatrixEntryBuffers *buf, uint64_t nrows_sum,
                               uint64_t nrows, HYPRE_BigInt ilower, HYPRE_BigInt iupper)
{
   char       filename[1024];
   uint64_t   header[11];
   HYPRE_Int *dsizes = NULL;
   HYPRE_Int *osizes = NULL;
   int        status = 0;

   /* GCOVR_EXCL_START */
   if (nrows_sum > (uint64_t)SIZE_MAX / sizeof(HYPRE_Int))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix row count exceeds host precompute bounds");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   dsizes = (HYPRE_Int *)calloc(nrows_sum, sizeof(HYPRE_Int));
   osizes = (HYPRE_Int *)calloc(nrows_sum, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (!dsizes || !osizes)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate matrix host sparsity buffers");
      goto done;
   }
   /* GCOVR_EXCL_STOP */

   for (uint32_t part = 0; part < nparts; part++)
   {
      /* Second-pass reopen failures are not exercised deterministically. */
      FILE *fp = IJMatrixOpenPart(prefixname, partids[part], filename, sizeof(filename),
                                  header, nnzs_max, 1, 0);

      if (!fp)
      {
         goto done;
      }

      /* Read row and column indices */
      if (!IJMatrixReadIndexPair(fp, header, nnzs_max, buf->h_rows, buf->h_cols,
                                 filename))
      {
         fclose(fp);
         goto done;
      }
      fclose(fp);

      if (!IJMatrixCountPartSparsity(buf->h_rows, buf->h_cols, header[6], nrows, ilower,
                                     iupper, nrows_sum, dsizes, osizes, filename))
      {
         goto done;
      }
   }

   /* Pre-allocating the sparsity pattern */
   HYPRE_IJMatrixSetDiagOffdSizes(mat, dsizes, osizes);
   status = 1;

done:
   free(dsizes);
   free(osizes);

   return status;
}

/* Copies one part's staged entries to device memory when the matrix lives there. */
/* GCOVR_EXCL_START */
static void
IJMatrixStageEntriesToDevice(IJMatrixEntryBuffers *buf, uint64_t nnz)
{
#ifdef HYPRE_USING_GPU
   if (buf->rows != buf->h_rows)
   {
      hypre_TMemcpy(buf->rows, buf->h_rows, HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(buf->cols, buf->h_cols, HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(buf->vals, buf->h_vals, HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
   }
#else
   (void)buf;
   (void)nnz;
#endif
}
/* GCOVR_EXCL_STOP */

/* Third pass: reads one part's indices and coefficients and hands them to hypre. */
static int
IJMatrixSetPartValues(HYPRE_IJMatrix mat, const char *prefixname, uint32_t partid,
                      size_t nnzs_max, uint64_t nrows, IJMatrixEntryBuffers *buf)
{
   char      filename[1024];
   uint64_t  header[11];
   HYPRE_Int nvalues = 0;
   FILE     *fp = IJMatrixOpenPart(prefixname, partid, filename, sizeof(filename), header,
                                   nnzs_max, 1, 0);

   if (!fp)
   {
      return 0;
   }

   /* Read row and column indices */
   if (!IJMatrixReadIndexPair(fp, header, nnzs_max, buf->h_rows, buf->h_cols, filename))
   {
      fclose(fp);
      return 0;
   }

   /* Validate entries before reading values or passing indices to hypre.
    * This reader currently constructs square IJ matrices, so the global
    * row count is also the valid global column count. */
   for (size_t i = 0; i < header[6]; i++)
   {
      if (!IJMatrixValidateEntry(buf->h_rows[i], buf->h_cols[i], nrows, nrows, filename))
      {
         fclose(fp);
         return 0;
      }
   }

   /* Read matrix coefficients */
   if (!IJMatrixReadCoefficients(fp, header, nnzs_max, buf->h_vals, filename))
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);

   IJMatrixStageEntriesToDevice(buf, header[6]);

   nvalues = (HYPRE_Int)header[6]; /* NOLINT(cppcoreguidelines-narrowing-conversions) */
   HYPRE_IJMatrixSetValues(mat, nvalues, NULL, buf->rows, buf->cols, buf->vals);

   return 1;
}

/* Rank-collective agreement point: returns nonzero only when every rank in
 * `comm` is still error-free, so a per-rank failure cannot leave peers blocked
 * in the collective calls that follow. */
static int
IJMatrixAllRanksOk(MPI_Comm comm)
{
   int local_ok = hypredrv_ErrorCodeActive() ? 0 : 1;

   MPI_Allreduce(MPI_IN_PLACE, &local_ok, 1, MPI_INT, MPI_MIN, comm);

   return local_ok;
}

void
hypredrv_IJMatrixReadMultipartBinary(const char *prefixname, MPI_Comm comm,
                                     uint64_t             g_nparts,
                                     HYPRE_MemoryLocation memory_location,
                                     HYPRE_IJMatrix      *mat_ptr)
{
   int      nprocs = 0, myid = 0;
   uint32_t nparts       = 0;
   uint64_t local_nparts = 0, first_part = 0;

   uint64_t nrows        = 0;
   uint64_t nrows_sum    = 0;
   uint64_t nrows_offset = 0;
   size_t   nnzs_max     = 0;

   uint32_t *partids = NULL;

   HYPRE_IJMatrix       mat    = NULL;
   HYPRE_BigInt         ilower = 0, iupper = 0;
   IJMatrixEntryBuffers buf = {NULL, NULL, NULL, NULL, NULL, NULL};

   *mat_ptr = NULL;

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   hypredrv_MultipartRange(g_nparts, nprocs, myid, &first_part, &local_nparts);
   nparts = (uint32_t)local_nparts;
   if (g_nparts < (size_t)nprocs)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid number of parts!");
      return;
   }

   if (!hypredrv_BinaryPathPrefixIsSafe(prefixname))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid matrix data path prefix");
      return;
   }

   /* 1b) Compute partids array */
   if (!IJMatrixBuildPartIds(first_part, nparts, &partids))
   {
      return;
   }

   /* 2) Read nrows/nnz for each part. A failure here is reported through the
    * error state and settled collectively just below, so peers never diverge. */
   (void)IJMatrixScanParts(prefixname, partids, nparts, &nrows_sum, &nnzs_max);
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   /* 3) Build IJMatrix */
   MPI_Allreduce(&nrows_sum, &nrows, 1, MPI_UINT64_T, MPI_SUM, comm);
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + (HYPRE_BigInt)nrows_sum - 1);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);

   /* 4) Fill entries. Both steps below report failures through the error state;
    * the collective agreement afterwards keeps all ranks on the same path. */
   if (IJMatrixAllocEntryBuffers(nnzs_max, &buf) && memory_location == HYPRE_MEMORY_HOST)
   {
      /* 4a) Pre-compute the sparsity pattern when storing on host memory */
      (void)IJMatrixPrecomputeHostSparsity(mat, prefixname, partids, nparts, nnzs_max,
                                           &buf, nrows_sum, nrows, ilower, iupper);
   }
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   /* Allocate matrix on the final memory */
   HYPRE_IJMatrixInitialize_v2(mat, memory_location);

   /* Allocate device variables */
   /* GCOVR_EXCL_START */
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      buf.rows = hypre_TAlloc(HYPRE_BigInt, nnzs_max, memory_location);
      buf.cols = hypre_TAlloc(HYPRE_BigInt, nnzs_max, memory_location);
      buf.vals = hypre_TAlloc(HYPRE_Complex, nnzs_max, memory_location);
   }
   else
#endif
   /* GCOVR_EXCL_STOP */
   {
      buf.rows = buf.h_rows;
      buf.cols = buf.h_cols;
      buf.vals = buf.h_vals;
   }

   /* Set matrix values */
   for (uint32_t part = 0; part < nparts; part++)
   {
      if (!IJMatrixSetPartValues(mat, prefixname, partids[part], nnzs_max, nrows, &buf))
      {
         break;
      }
   }
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   HYPRE_IJMatrixAssemble(mat);
   *mat_ptr = mat;

cleanup:
   /* Free memory */
   free(partids);
   free(buf.h_rows);
   free(buf.h_cols);
   free(buf.h_vals);
   /* GCOVR_EXCL_START */
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(buf.rows, HYPRE_MEMORY_DEVICE);
      hypre_TFree(buf.cols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(buf.vals, HYPRE_MEMORY_DEVICE);
   }
#endif
   /* GCOVR_EXCL_STOP */
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_STOP */
   {
      /* GCOVR_EXCL_BR_START */
      if (mat) /* GCOVR_EXCL_BR_STOP */
      {
         HYPRE_IJMatrixDestroy(mat);
      }
      *mat_ptr = NULL;
   }
}
