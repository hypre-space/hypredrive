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

static void
IJVectorInitializeCompat(HYPRE_IJVector vec, HYPRE_MemoryLocation memory_location)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 21900
   HYPRE_IJVectorInitialize_v2(vec, memory_location);
#else
   (void)memory_location;
   HYPRE_IJVectorInitialize(vec);
#endif
}

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

void
hypredrv_IJVectorReadMultipartBinary(const char *prefixname, MPI_Comm comm,
                                     uint64_t             g_nparts,
                                     HYPRE_MemoryLocation memory_location,
                                     HYPRE_IJVector      *vec_ptr)
{
   int      nprocs = 0, myid = 0;
   uint32_t nparts = 0;
   uint64_t part   = 0;
   uint32_t offset = 0;

   char     filename[1024];
   uint64_t header[11];

   uint64_t nrows_sum = 0, nrows_max = 0, nrows_offset = 0;

   uint32_t *partids = NULL;
   FILE     *fp      = NULL;

   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_IJVector       vec    = NULL;
   HYPRE_Complex       *h_vals = NULL;
   const HYPRE_Complex *vals   = NULL;
#ifdef HYPRE_USING_GPU
   HYPRE_Complex *d_vals = NULL;
#endif

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = (uint32_t)(g_nparts / (uint64_t)nprocs);
   nparts += (myid < ((int)g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < (size_t)nprocs)
   {
      *vec_ptr = NULL;
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid number of parts!");
      return;
   }

   if (!hypredrv_BinaryPathPrefixIsSafe(prefixname))
   {
      *vec_ptr = NULL;
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid vector data path prefix");
      return;
   }

   /* 1b) Compute partids array */
   if (nparts > (uint32_t)(SIZE_MAX / sizeof(uint32_t)))
   {
      *vec_ptr = NULL;
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Vector part id count exceeds allocation bounds");
      return;
   }
   partids = (uint32_t *)malloc(nparts * sizeof(uint32_t));
   /* LCOV_EXCL_START */
   if (nparts > 0 && !partids)
   {
      *vec_ptr = NULL;
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate vector part id map (%u entries)", nparts);
      return;
   }
   /* LCOV_EXCL_STOP */
   offset = ((uint32_t)myid) * nparts;
   offset += (myid < ((int)g_nparts % nprocs)) ? (uint32_t)myid
                                               : (uint32_t)((int)g_nparts % nprocs);
   for (part = 0; part < nparts; part++)
   {
      partids[part] = (uint32_t)(offset + part);
   }

   /* 2) Read nrows/nnz for each part */
   nrows_max = nrows_sum = 0;
   for (part = 0; part < nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      if (!fp)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
         goto cleanup;
      }

      /* Read header contents */
      if (fread(header, sizeof(uint64_t), 8, fp) != 8)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         fp = NULL;
         goto cleanup;
      }
      fclose(fp);
      fp = NULL;

      if (!IJVectorValidateHeader(header, filename))
      {
         goto cleanup;
      }
      /* LCOV_EXCL_START */
      if (nrows_sum > UINT64_MAX - header[5])
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Vector local row count overflow while reading %s",
                              filename);
         goto cleanup;
      }
      /* LCOV_EXCL_STOP */
      nrows_sum += header[5];
      nrows_max = (header[5] > nrows_max) ? header[5] : nrows_max;
   }

   /* 3) Build IJVector */
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + (HYPRE_BigInt)nrows_sum - 1);

   HYPRE_IJVectorCreate(comm, ilower, iupper, &vec);
   HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
   IJVectorInitializeCompat(vec, memory_location);

   /* Allocate variables */
   h_vals =
      (nrows_max > 0) ? (HYPRE_Complex *)malloc(nrows_max * sizeof(HYPRE_Complex)) : NULL;
   /* LCOV_EXCL_START */
   if (nrows_max > 0 && !h_vals)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate vector read buffer (%llu rows)",
                           (unsigned long long)nrows_max);
      goto cleanup;
   }
   /* LCOV_EXCL_STOP */
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      vals = d_vals = hypre_TAlloc(HYPRE_Complex, nrows_max, memory_location);
   }
   else
#endif
   {
      vals = h_vals;
   }

   /* 4) Fill entries */
   for (part = 0; part < nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      /* Second-pass fopen failure mirrors pass 1 but is not reachable once pass 1
       * succeeded on the same files in a single-threaded run. */
      /* LCOV_EXCL_START */
      if (!fp)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
         goto cleanup;
      }
      /* LCOV_EXCL_STOP */

      if (fread(header, sizeof(uint64_t), 8, fp) != 8) /* LCOV_EXCL_START */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         fp = NULL;
         goto cleanup;
      }
      /* LCOV_EXCL_STOP */

      if (!IJVectorValidateHeader(header, filename)) /* LCOV_EXCL_START */
      {
         fclose(fp);
         fp = NULL;
         goto cleanup;
      }
      /* LCOV_EXCL_STOP */
      if (!IJVectorPartRowsMatchesPrepass(nrows_max, header[5], filename))
      {
         fclose(fp);
         fp = NULL;
         goto cleanup;
      }

      /* Read vector coefficients */
      if (header[1] == sizeof(float))
      {
         float *buffer = NULL;
         if (header[5] > 0)
         {
            buffer = (float *)malloc((size_t)nrows_max * sizeof(float));
            if (!buffer || fread(buffer, sizeof(float), header[5], fp) != header[5])
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               hypredrv_ErrorMsgAdd("Could not read coeficients from %s", filename);
               fclose(fp);
               fp = NULL;
               free(buffer);
               goto cleanup;
            }
         }

         for (size_t i = 0; h_vals && i < header[5]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else if (header[1] == sizeof(double))
      {
         double *buffer = NULL;
         if (header[5] > 0)
         {
            buffer = (double *)malloc((size_t)nrows_max * sizeof(double));
            if (!buffer || fread(buffer, sizeof(double), header[5], fp) != header[5])
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               hypredrv_ErrorMsgAdd("Could not read coeficients from %s", filename);
               fclose(fp);
               fp = NULL;
               free(buffer);
               goto cleanup;
            }
         }

         for (size_t i = 0; h_vals && i < header[5]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid coefficient data type size %lld at %s", header[1],
                              filename);
         fclose(fp);
         fp = NULL;
         goto cleanup;
      }
      fclose(fp);
      fp = NULL;

#ifdef HYPRE_USING_GPU
      if (vals != h_vals)
      {
         hypre_TMemcpy(d_vals, h_vals, HYPRE_Complex, header[5], HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
#endif

      HYPRE_IJVectorSetValues(vec, (HYPRE_BigInt)header[5], NULL, vals);
   }

   HYPRE_IJVectorAssemble(vec);
   *vec_ptr = vec;

cleanup:
   /* Free memory */
   /* LCOV_EXCL_START */
   if (fp)
   {
      fclose(fp);
   }
   /* LCOV_EXCL_STOP */
   free(partids);
   free(h_vals);
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
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
