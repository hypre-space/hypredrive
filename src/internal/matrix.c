/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <limits.h>
#include <stdint.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_utilities.h" // for hypre_TAlloc, hypre_TMemcpy, hypre_TFree
#include "internal/linsys.h"
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

/* Data-type widths accepted for on-disk row/column index arrays. */
static int
IJMatrixIndexDtypeIsValid(uint64_t isize)
{
   return (isize == sizeof(HYPRE_BigInt) || isize == sizeof(uint32_t) ||
           isize == sizeof(uint64_t));
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

/* Converts one index array of `nnz` entries stored with `isize` bytes to
 * HYPRE_BigInt, in place when narrowing or equal (each write lands at or
 * before its read), otherwise into a new array replacing *data. */
static int
IJMatrixIndexArrayToBigInt(void **data, uint64_t nnz, uint64_t isize)
{
   if (isize == sizeof(HYPRE_BigInt) || nnz == 0 || !*data)
   {
      return 1;
   }

   HYPRE_BigInt *dst = (isize > sizeof(HYPRE_BigInt))
                          ? (HYPRE_BigInt *)*data
                          : (HYPRE_BigInt *)malloc((size_t)nnz * sizeof(HYPRE_BigInt));
   if (!dst) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd(
         "Failed to allocate matrix index conversion buffer"); /* GCOVR_EXCL_LINE */
      return 0;                                                /* GCOVR_EXCL_LINE */
   }
   for (size_t i = 0; i < (size_t)nnz; i++)
   {
      uint64_t value = 0;
      if (isize == sizeof(uint32_t))
      {
         uint32_t v32 = 0;
         memcpy(&v32, (const unsigned char *)*data + (i * sizeof(uint32_t)), sizeof(v32));
         value = v32;
      }
      else
      {
         memcpy(&value, (const unsigned char *)*data + (i * sizeof(uint64_t)),
                sizeof(value));
      }
      dst[i] = (HYPRE_BigInt)value;
   }
   if ((void *)dst != *data)
   {
      free(*data);
      *data = dst;
   }
   else
   {
      /* Narrowed in place: give back the unused tail of the buffer. */
      void *shrunk = realloc(*data, (size_t)nnz * sizeof(HYPRE_BigInt));
      if (shrunk) /* GCOVR_EXCL_BR_LINE */
      {
         *data = shrunk;
      }
   }
   return 1;
}

/* Makes part->rows/cols HYPRE_BigInt arrays (see IJMatrixIndexArrayToBigInt). */
static int
IJMatrixIndicesToBigInt(hypredrv_IJMatrixMemPart *part)
{
   return IJMatrixIndexArrayToBigInt(&part->rows, part->nnz, part->index_size) &&
          IJMatrixIndexArrayToBigInt(&part->cols, part->nnz, part->index_size);
}

/*-----------------------------------------------------------------------------
 * Source-driven IJ matrix builder
 *
 * Parts come from a hypredrv_IJMatrixPartSource one at a time over three
 * passes: metadata (row range), indices (validation, plus host sparsity
 * pre-sizing), then values (inserted through device staging buffers for
 * device matrices). A rank owning a single part keeps its indices from the
 * second pass, so they are loaded only once. Collective over `comm`.
 *-----------------------------------------------------------------------------*/

static void
IJMatrixMemPartFree(hypredrv_IJMatrixMemPart *part)
{
   free(part->rows);
   free(part->cols);
   free(part->vals);
   part->rows = part->cols = part->vals = NULL;
}

/* Calls the source, guaranteeing the error state is set when it fails (so the
 * collective status checks see every rank-local failure). */
static int
IJMatrixSourceLoad(const hypredrv_IJMatrixPartSource *src, uint32_t p, int want,
                   hypredrv_IJMatrixMemPart *part)
{
   if (src->load(src->ctx, p, want, part))
   {
      return 1;
   }
   if (!hypredrv_ErrorCodeActive())
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not load matrix part %u", (unsigned)p);
   }
   return 0;
}

/* Loads part p and checks it against the metadata from the first pass. */
static int
IJMatrixLoadPart(const hypredrv_IJMatrixPartSource *src, uint32_t p, int want,
                 const hypredrv_IJMatrixMemPart *meta, hypredrv_IJMatrixMemPart *part)
{
   if (!IJMatrixSourceLoad(src, p, want, part))
   {
      IJMatrixMemPartFree(part);
      return 0;
   }
   if (part->nnz != meta->nnz || part->nrows != meta->nrows ||
       part->index_size != meta->index_size || part->value_size != meta->value_size)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Matrix part changed between read passes in %s",
                           meta->label ? meta->label : "(unknown)");
      IJMatrixMemPartFree(part);
      return 0;
   }
   return 1;
}

void
hypredrv_IJMatrixBuildFromSource(MPI_Comm comm, const hypredrv_IJMatrixPartSource *src,
                                 HYPRE_MemoryLocation memory_location,
                                 HYPRE_IJMatrix      *mat_ptr)
{
   const uint32_t            nparts    = src->nparts;
   hypredrv_IJMatrixMemPart *meta      = NULL;
   hypredrv_IJMatrixMemPart  cached    = {0};
   uint64_t                  nrows_sum = 0, nrows = 0, nrows_offset = 0, nnz_max = 0;
   HYPRE_BigInt              ilower = 0, iupper = 0;
   HYPRE_IJMatrix            mat    = NULL;
   HYPRE_Int                *dsizes = NULL, *osizes = NULL;
   const int                 host   = (memory_location == HYPRE_MEMORY_HOST);
   HYPRE_BigInt             *d_rows = NULL, *d_cols = NULL;
   HYPRE_Complex            *d_vals = NULL;

   *mat_ptr = NULL;
#ifndef HYPRE_USING_GPU
   (void)d_rows;
   (void)d_cols;
   (void)d_vals;
#endif

   /* 1) Metadata: validate every part and size the local row range. */
   meta = (hypredrv_IJMatrixMemPart *)calloc(nparts ? (size_t)nparts : 1u, sizeof(*meta));
   if (!meta) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                     /* GCOVR_EXCL_LINE */
      hypredrv_ErrorMsgAdd("Failed to allocate matrix part list"); /* GCOVR_EXCL_LINE */
   }
   for (uint32_t p = 0; meta && p < nparts && !hypredrv_ErrorCodeActive(); p++)
   {
      if (!IJMatrixSourceLoad(src, p, 0, &meta[p]))
      {
         break;
      }
      if (!IJMatrixIndexDtypeIsValid(meta[p].index_size) ||
          (meta[p].value_size != sizeof(float) && meta[p].value_size != sizeof(double)) ||
          meta[p].nnz > (uint64_t)IJMATRIX_MAX_PART_NNZ ||
          meta[p].nrows > (uint64_t)IJMATRIX_MAX_PART_NROWS)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid matrix part metadata in %s",
                              meta[p].label ? meta[p].label : "(unknown)");
         break;
      }
      nrows_sum += meta[p].nrows;
      nnz_max = (meta[p].nnz > nnz_max) ? meta[p].nnz : nnz_max;
   }
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   MPI_Allreduce(&nrows_sum, &nrows, 1, MPI_UINT64_T, MPI_SUM, comm);
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + (HYPRE_BigInt)nrows_sum - 1);
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);

   /* 2) Indices: validate entries; host matrices are also pre-sized. */
   if (host)
   {
      dsizes = (HYPRE_Int *)calloc(nrows_sum ? (size_t)nrows_sum : 1u, sizeof(HYPRE_Int));
      osizes = (HYPRE_Int *)calloc(nrows_sum ? (size_t)nrows_sum : 1u, sizeof(HYPRE_Int));
      if (!dsizes || !osizes) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate matrix host sparsity buffers"); /* GCOVR_EXCL_LINE */
      }
   }
   for (uint32_t p = 0; p < nparts && !hypredrv_ErrorCodeActive(); p++)
   {
      hypredrv_IJMatrixMemPart part = {0};

      if (!IJMatrixLoadPart(src, p, HYPREDRV_PART_INDICES, &meta[p], &part) ||
          !IJMatrixIndicesToBigInt(&part))
      {
         IJMatrixMemPartFree(&part);
         break;
      }
      const HYPRE_BigInt *rows = (const HYPRE_BigInt *)part.rows;
      const HYPRE_BigInt *cols = (const HYPRE_BigInt *)part.cols;
      if (host)
      {
         (void)IJMatrixCountPartSparsity(rows, cols, part.nnz, nrows, ilower, iupper,
                                         nrows_sum, dsizes, osizes, part.label);
      }
      else
      {
         for (size_t k = 0; k < (size_t)part.nnz; k++)
         {
            if (!IJMatrixValidateEntry(rows[k], cols[k], nrows, nrows, part.label))
            {
               break;
            }
         }
      }
      if (nparts == 1)
      {
         cached = part; /* reused by the values pass */
      }
      else
      {
         IJMatrixMemPartFree(&part);
      }
   }
   if (host && !hypredrv_ErrorCodeActive())
   {
      HYPRE_IJMatrixSetDiagOffdSizes(mat, dsizes, osizes);
   }
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   /* 3) Values: validate/widen (in place when widths match) and insert. */
   HYPRE_IJMatrixInitialize_v2(mat, memory_location);
#ifdef HYPRE_USING_GPU
   if (!host)
   {
      d_rows = hypre_TAlloc(HYPRE_BigInt, nnz_max, memory_location);
      d_cols = hypre_TAlloc(HYPRE_BigInt, nnz_max, memory_location);
      d_vals = hypre_TAlloc(HYPRE_Complex, nnz_max, memory_location);
   }
#endif
   for (uint32_t p = 0; p < nparts && !hypredrv_ErrorCodeActive(); p++)
   {
      hypredrv_IJMatrixMemPart part = {0};
      HYPRE_Complex           *wide = NULL, *vals = NULL;
      int                      ok = 0;

      if (nparts == 1)
      {
         part   = cached;
         cached = (hypredrv_IJMatrixMemPart){0};
         ok     = IJMatrixSourceLoad(src, p, HYPREDRV_PART_VALUES, &part);
      }
      else
      {
         /* The cached single part already holds HYPRE_BigInt indices. */
         ok = IJMatrixLoadPart(src, p, HYPREDRV_PART_INDICES | HYPREDRV_PART_VALUES,
                               &meta[p], &part) &&
              IJMatrixIndicesToBigInt(&part);
      }
      if (!ok)
      {
         IJMatrixMemPartFree(&part);
         break;
      }
      if (part.nnz == 0)
      {
         IJMatrixMemPartFree(&part);
         continue;
      }

      HYPRE_BigInt *rows = (HYPRE_BigInt *)part.rows;
      HYPRE_BigInt *cols = (HYPRE_BigInt *)part.cols;
      /* Indices re-loaded for multi-part ranks are re-validated. */
      for (size_t k = 0; nparts > 1 && k < (size_t)part.nnz; k++)
      {
         if (!IJMatrixValidateEntry(rows[k], cols[k], nrows, nrows, part.label))
         {
            break;
         }
      }
      vals = (HYPRE_Complex *)part.vals;
#if !defined(HYPRE_COMPLEX)
      if (part.value_size != sizeof(HYPRE_Complex))
#endif
      {
         wide = (HYPRE_Complex *)malloc((size_t)part.nnz * sizeof(HYPRE_Complex));
         vals = wide;
      }
      /* GCOVR_EXCL_BR_START */
      if (!hypredrv_ErrorCodeActive() && vals &&
          hypredrv_ConvertCoefficients(part.vals, part.value_size, part.nnz, vals,
                                       "matrix", part.label))
      /* GCOVR_EXCL_BR_STOP */
      {
         HYPRE_BigInt  *set_rows = rows, *set_cols = cols;
         HYPRE_Complex *set_vals = vals;
#ifdef HYPRE_USING_GPU
         /* GCOVR_EXCL_START */
         if (!host)
         {
            hypre_TMemcpy(d_rows, rows, HYPRE_BigInt, part.nnz, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            hypre_TMemcpy(d_cols, cols, HYPRE_BigInt, part.nnz, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            hypre_TMemcpy(d_vals, vals, HYPRE_Complex, part.nnz, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            set_rows = d_rows;
            set_cols = d_cols;
            set_vals = d_vals;
         }
         /* GCOVR_EXCL_STOP */
#endif
         HYPRE_IJMatrixSetValues(mat, (HYPRE_Int)part.nnz, NULL, set_rows, set_cols,
                                 set_vals);
      }
      else if (!vals) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION); /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd(
            "Failed to allocate matrix value buffer"); /* GCOVR_EXCL_LINE */
      }
      free(wide);
      IJMatrixMemPartFree(&part);
   }
   if (!IJMatrixAllRanksOk(comm))
   {
      goto cleanup;
   }

   HYPRE_IJMatrixAssemble(mat);
   *mat_ptr = mat;
   mat      = NULL;

cleanup:
#ifdef HYPRE_USING_GPU
   hypre_TFree(d_rows, memory_location);
   hypre_TFree(d_cols, memory_location);
   hypre_TFree(d_vals, memory_location);
#endif
   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }
   IJMatrixMemPartFree(&cached);
   free(meta);
   free(dsizes);
   free(osizes);
}

/*-----------------------------------------------------------------------------
 * Multipart binary files: one part file per stored part,
 * "<prefix>.<partid:05d>.bin" = 11-word header, rows, cols, values.
 *-----------------------------------------------------------------------------*/

typedef struct
{
   const char *prefixname;
   uint64_t    first_part;
   char        filename[1024];
} IJMatrixFileSource;

/* Opens part `partid` and reads/validates its 11-word header. Returns a stream
 * positioned just past the header, or NULL with the error state set. */
static FILE *
IJMatrixOpenPart(IJMatrixFileSource *fs, uint32_t partid, uint64_t *header)
{
   FILE *fp = NULL;

   snprintf(fs->filename, sizeof(fs->filename), "%s.%05d.bin", fs->prefixname,
            (int)partid);
   fp = fopen(fs->filename, "rb");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(fs->filename);
      return NULL;
   }

   if (fread(header, sizeof(uint64_t), 11, fp) != 11)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read header from %s", fs->filename);
      fclose(fp);
      return NULL;
   }

   if (!IJMatrixValidateHeader(header, fs->filename))
   {
      fclose(fp);
      return NULL;
   }

   return fp;
}

/* Reads `count` entries of `width` bytes into a new array (*out). */
static int
IJMatrixReadRaw(FILE *fp, uint64_t count, uint64_t width, void **out, const char *what,
                const char *filename)
{
   *out = NULL;
   if (count == 0)
   {
      return 1;
   }
   *out = malloc((size_t)count * (size_t)width);
   if (!*out || fread(*out, (size_t)width, (size_t)count, fp) != count)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read %s from %s", what, filename);
      free(*out);
      *out = NULL;
      return 0;
   }
   return 1;
}

static int
IJMatrixFileLoad(void *ctx, uint32_t p, int want, hypredrv_IJMatrixMemPart *part)
{
   IJMatrixFileSource *fs = (IJMatrixFileSource *)ctx;
   uint64_t            header[11];
   FILE               *fp = IJMatrixOpenPart(fs, (uint32_t)(fs->first_part + p), header);
   int                 ok = (fp != NULL);

   if (ok)
   {
      part->nrows      = header[8] - header[7] + 1u;
      part->nnz        = header[6];
      part->index_size = header[1];
      part->value_size = header[2];
      part->label      = fs->filename;
   }
   if (ok && want && !IJMatrixIndexDtypeIsValid(part->index_size))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid row/col data type size %lld at %s",
                           (long long)part->index_size, fs->filename);
      ok = 0;
   }
   if (ok && (want & HYPREDRV_PART_INDICES))
   {
      /* GCOVR_EXCL_BR_START */
      ok = IJMatrixReadRaw(fp, part->nnz, part->index_size, &part->rows, "row indices",
                           fs->filename) &&
           IJMatrixReadRaw(fp, part->nnz, part->index_size, &part->cols, "column indices",
                           fs->filename);
      /* GCOVR_EXCL_BR_STOP */
   }
   else if (ok && (want & HYPREDRV_PART_VALUES))
   {
      /* Indices are already in memory: skip them on disk. */
      const uint64_t index_bytes = 2u * part->nnz * part->index_size;
      void          *skip        = NULL;
      if (index_bytes <= (uint64_t)LONG_MAX)
      {
         ok = (fseek(fp, (long)index_bytes, SEEK_CUR) == 0);
      }
      else /* GCOVR_EXCL_START */
      {
         ok = IJMatrixReadRaw(fp, 2u * part->nnz, part->index_size, &skip, "indices",
                              fs->filename);
         free(skip);
      } /* GCOVR_EXCL_STOP */
   }
   if (ok && (want & HYPREDRV_PART_VALUES))
   {
      if (part->value_size != sizeof(float) && part->value_size != sizeof(double))
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid coefficient data type size %lld at %s",
                              (long long)part->value_size, fs->filename);
         ok = 0;
      }
      else
      {
         ok = IJMatrixReadRaw(fp, part->nnz, part->value_size, &part->vals, "coeficients",
                              fs->filename);
      }
   }
   if (fp)
   {
      fclose(fp);
   }
   return ok;
}

void
hypredrv_IJMatrixReadMultipartBinary(const char *prefixname, MPI_Comm comm,
                                     uint64_t             g_nparts,
                                     HYPRE_MemoryLocation memory_location,
                                     HYPRE_IJMatrix      *mat_ptr)
{
   int                nprocs = 0, myid = 0;
   uint64_t           local_nparts = 0;
   IJMatrixFileSource fs           = {prefixname, 0, {0}};

   *mat_ptr = NULL;
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   hypredrv_MultipartRange(g_nparts, nprocs, myid, &fs.first_part, &local_nparts);
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

   hypredrv_IJMatrixPartSource src = {&fs, (uint32_t)local_nparts, IJMatrixFileLoad};
   hypredrv_IJMatrixBuildFromSource(comm, &src, memory_location, mat_ptr);
}
