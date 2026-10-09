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
#include "internal/linsys.h"
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

/*-----------------------------------------------------------------------------
 * Source-driven IJ vector builder
 *
 * Parts come from a hypredrv_IJVectorPartSource one at a time: a metadata pass
 * sizes the local row range, then a values pass validates/widens each part's
 * values (in place when widths match) and inserts them at the part's
 * concatenation offset, through device staging buffers for device vectors.
 * Collective over `comm`.
 *-----------------------------------------------------------------------------*/

static int
IJVectorSourceLoad(const hypredrv_IJVectorPartSource *src, uint32_t p, int want_values,
                   hypredrv_IJVectorMemPart *part)
{
   if (src->load(src->ctx, p, want_values, part))
   {
      return 1;
   }
   if (!hypredrv_ErrorCodeActive())
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not load vector part %u", (unsigned)p);
   }
   return 0;
}

void
hypredrv_IJVectorBuildFromSource(MPI_Comm comm, const hypredrv_IJVectorPartSource *src,
                                 HYPRE_MemoryLocation memory_location,
                                 HYPRE_IJVector      *vec_ptr)
{
   const uint32_t nparts    = src->nparts;
   uint64_t       nrows_sum = 0, nrows_max = 0, nrows_offset = 0, row = 0;
   HYPRE_BigInt   ilower = 0, iupper = 0;
   HYPRE_IJVector vec       = NULL;
   HYPRE_BigInt  *indices   = NULL;
   HYPRE_Complex *wide      = NULL;
   HYPRE_BigInt  *d_indices = NULL;
   HYPRE_Complex *d_vals    = NULL;

   *vec_ptr = NULL;
#ifndef HYPRE_USING_GPU
   (void)d_indices;
   (void)d_vals;
#endif

   /* 1) Metadata: validate every part and size the local row range. */
   for (uint32_t p = 0; p < nparts; p++)
   {
      hypredrv_IJVectorMemPart meta = {0};
      if (!IJVectorSourceLoad(src, p, 0, &meta))
      {
         break;
      }
      if ((meta.value_size != sizeof(float) && meta.value_size != sizeof(double)) ||
          meta.nrows > (uint64_t)IJVECTOR_MAX_PART_NROWS)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid vector part metadata in %s",
                              meta.label ? meta.label : "(unknown)");
         break;
      }
      nrows_sum += meta.nrows;
      nrows_max = (meta.nrows > nrows_max) ? meta.nrows : nrows_max;
   }
   if (!hypredrv_ErrorCodeActive() && nrows_max > 0)
   {
      indices = (HYPRE_BigInt *)malloc((size_t)nrows_max * sizeof(HYPRE_BigInt));
      wide    = (HYPRE_Complex *)malloc((size_t)nrows_max * sizeof(HYPRE_Complex));
      /* GCOVR_EXCL_BR_START */
      if (!indices || !wide) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);                   /* GCOVR_EXCL_LINE */
         hypredrv_ErrorMsgAdd("Failed to allocate vector buffers"); /* GCOVR_EXCL_LINE */
      }
#ifdef HYPRE_USING_GPU
      if (memory_location == HYPRE_MEMORY_DEVICE)
      {
         d_indices = hypre_TAlloc(HYPRE_BigInt, nrows_max, memory_location);
         d_vals    = hypre_TAlloc(HYPRE_Complex, nrows_max, memory_location);
      }
#endif
   }
   if (!IJVectorAllRanksOk(comm))
   {
      goto cleanup;
   }

   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + (HYPRE_BigInt)nrows_sum - 1);
   HYPRE_IJVectorCreate(comm, ilower, iupper, &vec);
   HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec, memory_location);

   /* 2) Values: explicit indices keep each part at its concatenation offset. */
   for (uint32_t p = 0; p < nparts && !hypredrv_ErrorCodeActive(); p++)
   {
      hypredrv_IJVectorMemPart part = {0};
      HYPRE_Complex           *vals = wide;

      if (!IJVectorSourceLoad(src, p, 1, &part))
      {
         free(part.vals);
         break;
      }
      if (part.nrows > nrows_max || row > nrows_sum - part.nrows)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Vector part rows exceed the pre-scanned local range at %s",
                              part.label ? part.label : "(unknown)");
         free(part.vals);
         break;
      }
      if (part.nrows == 0)
      {
         free(part.vals);
         continue;
      }
#if !defined(HYPRE_COMPLEX)
      if (part.value_size == sizeof(HYPRE_Complex))
      {
         vals = (HYPRE_Complex *)part.vals;
      }
#endif
      if (!hypredrv_ConvertCoefficients(part.vals, part.value_size, part.nrows, vals,
                                        "vector", part.label))
      {
         free(part.vals);
         break;
      }
      for (uint64_t i = 0; i < part.nrows; i++)
      {
         indices[i] = ilower + (HYPRE_BigInt)(row + i);
      }

      HYPRE_BigInt  *set_indices = indices;
      HYPRE_Complex *set_vals    = vals;
#ifdef HYPRE_USING_GPU
      /* GCOVR_EXCL_START */
      if (memory_location == HYPRE_MEMORY_DEVICE)
      {
         hypre_TMemcpy(d_indices, indices, HYPRE_BigInt, part.nrows, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         hypre_TMemcpy(d_vals, vals, HYPRE_Complex, part.nrows, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
         set_indices = d_indices;
         set_vals    = d_vals;
      }
      /* GCOVR_EXCL_STOP */
#endif
      HYPRE_IJVectorSetValues(vec, (HYPRE_Int)part.nrows, set_indices, set_vals);
      row += part.nrows;
      free(part.vals);
   }
   if (!IJVectorAllRanksOk(comm))
   {
      goto cleanup;
   }

   HYPRE_IJVectorAssemble(vec);
   *vec_ptr = vec;
   vec      = NULL;

cleanup:
#ifdef HYPRE_USING_GPU
   hypre_TFree(d_indices, memory_location);
   hypre_TFree(d_vals, memory_location);
#endif
   if (vec)
   {
      HYPRE_IJVectorDestroy(vec);
   }
   free(indices);
   free(wide);
}

/*-----------------------------------------------------------------------------
 * Multipart binary files: "<prefix>.<partid:05d>.bin" = 8-word header, values.
 *-----------------------------------------------------------------------------*/

typedef struct
{
   const char *prefixname;
   uint64_t    first_part;
   char        filename[1024];
} IJVectorFileSource;

static int
IJVectorFileLoad(void *ctx, uint32_t p, int want_values, hypredrv_IJVectorMemPart *part)
{
   IJVectorFileSource *fs = (IJVectorFileSource *)ctx;
   uint64_t            header[8];
   FILE               *fp = NULL;
   int                 ok = 0;

   snprintf(fs->filename, sizeof(fs->filename), "%s.%05d.bin", fs->prefixname,
            (int)(fs->first_part + p));
   fp = fopen(fs->filename, "rb");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(fs->filename);
      return 0;
   }
   if (fread(header, sizeof(uint64_t), 8, fp) != 8)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Could not read header from %s", fs->filename);
   }
   else if (IJVectorValidateHeader(header, fs->filename))
   {
      part->nrows      = header[5];
      part->value_size = header[1];
      part->label      = fs->filename;
      ok               = 1;
      if (want_values && part->nrows > 0)
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
            part->vals = malloc((size_t)part->nrows * (size_t)part->value_size);
            if (!part->vals || fread(part->vals, (size_t)part->value_size,
                                     (size_t)part->nrows, fp) != part->nrows)
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               hypredrv_ErrorMsgAdd("Could not read coeficients from %s", fs->filename);
               free(part->vals);
               part->vals = NULL;
               ok         = 0;
            }
         }
      }
   }
   fclose(fp);
   return ok;
}

void
hypredrv_IJVectorReadMultipartBinary(const char *prefixname, MPI_Comm comm,
                                     uint64_t             g_nparts,
                                     HYPRE_MemoryLocation memory_location,
                                     HYPRE_IJVector      *vec_ptr)
{
   int                nprocs = 0, myid = 0;
   uint64_t           local_nparts = 0;
   IJVectorFileSource fs           = {prefixname, 0, {0}};

   *vec_ptr = NULL;
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
      hypredrv_ErrorMsgAdd("Invalid vector data path prefix");
      return;
   }

   hypredrv_IJVectorPartSource src = {&fs, (uint32_t)local_nparts, IJVectorFileLoad};
   hypredrv_IJVectorBuildFromSource(comm, &src, memory_location, vec_ptr);
}
