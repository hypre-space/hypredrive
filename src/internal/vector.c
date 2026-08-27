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
   IJVECTOR_MAX_PART_NROWS = 200u * 1000u * 1000u,
};

static int
IJVectorValidateHeader(const uint64_t *header, const char *filename)
{
   /* LCOV_EXCL_START */
   if (!header)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null vector part header");
      return 0;
   }
   /* LCOV_EXCL_STOP */

   if (header[5] > (uint64_t)IJVECTOR_MAX_PART_NROWS)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector row count exceeds per-part limit in %s (%llu rows)",
                           filename ? filename : "(unknown)",
                           (unsigned long long)header[5]);
      return 0;
   }
   /* Per-part row cap is far below SIZE_MAX/sizeof(coeff); keep overflow guard for
    * hypothetical builds without the cap, but do not count it toward coverage. */
#ifdef HYPRE_COMPLEX
   /* LCOV_EXCL_START */
   if (header[5] > (uint64_t)SIZE_MAX / sizeof(HYPRE_Complex) ||
       header[5] > (uint64_t)SIZE_MAX / sizeof(double))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part sizes overflow allocation bounds in %s",
                           filename ? filename : "(unknown)");
      return 0;
   }
   /* LCOV_EXCL_STOP */
#else
   /* LCOV_EXCL_START */
   if (header[5] > (uint64_t)SIZE_MAX / sizeof(double))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part sizes overflow allocation bounds in %s",
                           filename ? filename : "(unknown)");
      return 0;
   }
   /* LCOV_EXCL_STOP */
#endif

   return 1;
}

static int
IJVectorPartRowsMatchesPrepass(uint64_t nrows_max, uint64_t part_rows,
                               const char *filename)
{
   if (part_rows > nrows_max)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part row count exceeds pre-scan maximum at %s",
                           filename ? filename : "(unknown)");
      return 0;
   }
   return 1;
}

static int
IJVectorRejectNonfiniteCoefficient(const char *filename)
{
   hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
   hypredrv_ErrorMsgAdd("Detected non-finite vector coefficient while reading %s",
                        filename ? filename : "(unknown)");
   return 0;
}

/* Host staging buffers for one part, plus the arrays handed to hypre (identical
 * to the host buffers unless the vector is device-resident). */
typedef struct
{
   HYPRE_BigInt  *h_indices;
   HYPRE_Complex *h_vals;
   HYPRE_BigInt  *indices;
   HYPRE_Complex *vals;
} IJVectorEntryBuffers;

/* Opens part `partid`, then reads and validates its 8-word header. Returns a
 * stream positioned just past the header, or NULL with the error state set (the
 * stream is closed on every failure path). `missing_is_not_found` selects the
 * error reported when the file cannot be opened; `check_prepass` additionally
 * cross-checks the part row count against the pre-scan maximum. */
static FILE *
IJVectorOpenPart(const char *prefixname, uint32_t partid, char *filename,
                 size_t filename_size, uint64_t *header, uint64_t nrows_max,
                 int check_prepass)
{
   FILE *fp = NULL;

   snprintf(filename, filename_size, "%s.%05d.bin", prefixname, (int)partid);
   fp = fopen(filename, "rb");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(filename);
      return NULL;
   }

   if (fread(header, sizeof(uint64_t), 8, fp) != 8)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read header from %s", filename);
      fclose(fp);
      return NULL;
   }

   if (!IJVectorValidateHeader(header, filename) ||
       (check_prepass && !IJVectorPartRowsMatchesPrepass(nrows_max, header[5], filename)))
   {
      fclose(fp);
      return NULL;
   }

   return fp;
}

/* First pass: reads every part header to accumulate this rank's local row count
 * and the largest per-part row count, which bounds the read buffers. */
static int
IJVectorScanParts(const char *prefixname, const uint32_t *partids, uint32_t nparts,
                  uint64_t *nrows_sum_out, uint64_t *nrows_max_out)
{
   char     filename[1024];
   uint64_t header[8];
   uint64_t nrows_sum = 0;
   uint64_t nrows_max = 0;

   for (uint32_t part = 0; part < nparts; part++)
   {
      FILE *fp = IJVectorOpenPart(prefixname, partids[part], filename, sizeof(filename),
                                  header, 0, 0);

      if (!fp)
      {
         return 0;
      }
      fclose(fp);

      /* LCOV_EXCL_START */
      if (nrows_sum > UINT64_MAX - header[5])
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Vector local row count overflow while reading %s",
                              filename);
         return 0;
      }
      /* LCOV_EXCL_STOP */
      nrows_sum += header[5];
      nrows_max = (header[5] > nrows_max) ? header[5] : nrows_max;
   }

   *nrows_sum_out = nrows_sum;
   *nrows_max_out = nrows_max;

   return 1;
}

/* Builds this rank's slice of the global part id map. */
static int
IJVectorBuildPartIds(uint64_t first_part, uint32_t nparts, uint32_t **partids_out)
{
   uint32_t *partids = NULL;

   if (nparts > (uint32_t)(SIZE_MAX / sizeof(uint32_t)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part id count exceeds allocation bounds");
      return 0;
   }

   partids = (uint32_t *)malloc(nparts * sizeof(uint32_t));
   /* LCOV_EXCL_START */
   if (nparts > 0 && !partids)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate vector part id map (%u entries)", nparts);
      return 0;
   }
   /* LCOV_EXCL_STOP */

   for (uint32_t part = 0; part < nparts; part++)
   {
      partids[part] = (uint32_t)(first_part + part);
   }

   *partids_out = partids;

   return 1;
}

/* Reads one part's coefficients into `h_vals`, widening from the on-disk
 * float/double representation and rejecting non-finite entries. */
static int
IJVectorReadCoefficients(FILE *fp, const uint64_t *header, uint64_t nrows_max,
                         HYPRE_Complex *h_vals, const char *filename)
{
   const uint64_t vsize  = header[1];
   const uint64_t nrows  = header[5];
   void          *buffer = NULL;
   int            status = 1;

   if (vsize != sizeof(float) && vsize != sizeof(double))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid coefficient data type size %lld at %s",
                           (long long)vsize, filename);
      return 0;
   }

   if (nrows == 0 || !h_vals)
   {
      return 1;
   }

   buffer = malloc((size_t)nrows_max * (size_t)vsize);
   if (!buffer || fread(buffer, (size_t)vsize, nrows, fp) != nrows)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read coeficients from %s", filename);
      free(buffer);
      return 0;
   }

   if (vsize == sizeof(float))
   {
      const float *src = (const float *)buffer;

      for (size_t i = 0; i < nrows; i++)
      {
         if (!hypredrv_FloatIsFinite(src[i]))
         {
            status = IJVectorRejectNonfiniteCoefficient(filename);
            break;
         }
         h_vals[i] = (HYPRE_Complex)src[i];
      }
   }
   else
   {
      const double *src = (const double *)buffer;

      for (size_t i = 0; i < nrows; i++)
      {
         if (!hypredrv_DoubleIsFinite(src[i]))
         {
            status = IJVectorRejectNonfiniteCoefficient(filename);
            break;
         }
         h_vals[i] = (HYPRE_Complex)src[i];
      }
   }

   free(buffer);

   return status;
}

/* Copies one part's staged entries to device memory when the vector lives there. */
static void
IJVectorStageEntriesToDevice(IJVectorEntryBuffers *buf, uint64_t nrows)
{
#ifdef HYPRE_USING_GPU
   if (buf->vals != buf->h_vals)
   {
      hypre_TMemcpy(buf->vals, buf->h_vals, HYPRE_Complex, nrows, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(buf->indices, buf->h_indices, HYPRE_BigInt, nrows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
#else
   (void)buf;
   (void)nrows;
#endif
}

/* Second pass: reads one part's coefficients and hands them to hypre.
 *
 * Each runtime rank can own several consecutive stored parts when the
 * communicator is smaller than g_nparts. Explicit indices preserve the
 * concatenation offset; indices=NULL would restart at ilower for every part and
 * overwrite the values loaded from preceding parts. */
static int
IJVectorSetPartValues(HYPRE_IJVector vec, const char *prefixname, uint32_t partid,
                      uint64_t nrows_max, uint64_t nrows_sum, HYPRE_BigInt ilower,
                      IJVectorEntryBuffers *buf, uint64_t *local_row_offset)
{
   char      filename[1024];
   uint64_t  header[8];
   HYPRE_Int nvalues = 0;
   FILE     *fp = IJVectorOpenPart(prefixname, partid, filename, sizeof(filename), header,
                                   nrows_max, 1);

   if (!fp)
   {
      return 0;
   }

   /* Read vector coefficients */
   if (!IJVectorReadCoefficients(fp, header, nrows_max, buf->h_vals, filename))
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);

   if (header[5] > nrows_sum || *local_row_offset > nrows_sum - header[5])
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part rows exceed the pre-scanned local range at %s",
                           filename);
      return 0;
   }

   for (uint64_t i = 0; i < header[5]; i++)
   {
      buf->h_indices[i] = ilower + (HYPRE_BigInt)(*local_row_offset) + (HYPRE_BigInt)i;
   }

   IJVectorStageEntriesToDevice(buf, header[5]);

   nvalues = (HYPRE_Int)header[5]; /* NOLINT(cppcoreguidelines-narrowing-conversions) */
   HYPRE_IJVectorSetValues(vec, nvalues, buf->indices, buf->vals);
   *local_row_offset += header[5];

   return 1;
}

/* Rank-collective agreement point: returns nonzero only when every rank in
 * `comm` is still error-free, so a per-rank failure cannot leave peers blocked
 * in the collective calls that follow. */
static int
IJVectorAllRanksOk(MPI_Comm comm)
{
   int local_ok = hypredrv_ErrorCodeActive() ? 0 : 1;

   MPI_Allreduce(MPI_IN_PLACE, &local_ok, 1, MPI_INT, MPI_MIN, comm);

   return local_ok;
}

void
hypredrv_IJVectorReadMultipartBinary(const char *prefixname, MPI_Comm comm,
                                     uint64_t             g_nparts,
                                     HYPRE_MemoryLocation memory_location,
                                     HYPRE_IJVector      *vec_ptr)
{
   int      nprocs = 0, myid = 0;
   uint32_t nparts       = 0;
   uint64_t local_nparts = 0, first_part = 0;

   uint64_t nrows_sum = 0, nrows_max = 0, nrows_offset = 0, local_row_offset = 0;

   uint32_t *partids = NULL;

   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_IJVector       vec = NULL;
   IJVectorEntryBuffers buf = {NULL, NULL, NULL, NULL};

   *vec_ptr = NULL;

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
      hypredrv_ErrorMsgAdd("Invalid vector data path prefix");
      return;
   }

   /* 1b) Compute partids array */
   if (!IJVectorBuildPartIds(first_part, nparts, &partids))
   {
      return;
   }

   /* 2) Read nrows for each part. A failure here is reported through the error
    * state and settled collectively just below, so peers never diverge. */
   (void)IJVectorScanParts(prefixname, partids, nparts, &nrows_sum, &nrows_max);
   if (!IJVectorAllRanksOk(comm))
   {
      goto cleanup;
   }

   /* 3) Build IJVector */
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + (HYPRE_BigInt)nrows_sum - 1);

   HYPRE_IJVectorCreate(comm, ilower, iupper, &vec);
   HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec, memory_location);

   /* Allocate variables */
   buf.h_indices =
      (nrows_max > 0) ? (HYPRE_BigInt *)malloc(nrows_max * sizeof(HYPRE_BigInt)) : NULL;
   buf.h_vals =
      (nrows_max > 0) ? (HYPRE_Complex *)malloc(nrows_max * sizeof(HYPRE_Complex)) : NULL;
   /* LCOV_EXCL_START */
   if (nrows_max > 0 && (!buf.h_indices || !buf.h_vals))
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate vector read buffers (%llu rows)",
                           (unsigned long long)nrows_max);
      goto cleanup;
   }
   /* LCOV_EXCL_STOP */
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      buf.indices = hypre_TAlloc(HYPRE_BigInt, nrows_max, memory_location);
      buf.vals    = hypre_TAlloc(HYPRE_Complex, nrows_max, memory_location);
      /* LCOV_EXCL_START */
      if (nrows_max > 0 && (!buf.indices || !buf.vals))
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate device vector read buffers (%llu rows)",
                              (unsigned long long)nrows_max);
         goto cleanup;
      }
      /* LCOV_EXCL_STOP */
   }
   else
#endif
   {
      buf.indices = buf.h_indices;
      buf.vals    = buf.h_vals;
   }

   /* 4) Fill entries */
   for (uint32_t part = 0; part < nparts; part++)
   {
      if (!IJVectorSetPartValues(vec, prefixname, partids[part], nrows_max, nrows_sum,
                                 ilower, &buf, &local_row_offset))
      {
         break;
      }
   }
   if (!IJVectorAllRanksOk(comm))
   {
      goto cleanup;
   }

   HYPRE_IJVectorAssemble(vec);
   *vec_ptr = vec;

cleanup:
   /* Free memory */
   free(partids);
   free(buf.h_indices);
   free(buf.h_vals);
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(buf.indices, HYPRE_MEMORY_DEVICE);
      hypre_TFree(buf.vals, HYPRE_MEMORY_DEVICE);
   }
#endif
   if (hypredrv_ErrorCodeActive())
   {
      if (vec)
      {
         HYPRE_IJVectorDestroy(vec);
      }
      *vec_ptr = NULL;
   }
}
