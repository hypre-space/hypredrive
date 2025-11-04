/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPRE_IJ_mv.h"
#include "_hypre_utilities.h" // for hypre_TAlloc, hypre_TMemcpy, hypre_TFree
#include "utils.h"

void
IJVectorReadMultipartBinary(const char *prefixname, MPI_Comm comm, uint64_t g_nparts,
                            HYPRE_MemoryLocation memory_location, HYPRE_IJVector *vec_ptr)
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

   HYPRE_IJVector       vec    = NULL;
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_Complex       *h_vals = NULL, *d_vals = NULL;
   const HYPRE_Complex *vals = NULL;

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = g_nparts / (uint64_t)nprocs;
   nparts += (myid < ((int)g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < (size_t)nprocs)
   {
      *vec_ptr = NULL;
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid number of parts!");
      return;
   }

   /* 1b) Compute partids array */
   partids = (uint32_t *)malloc(nparts * sizeof(uint32_t));
   offset  = ((uint32_t)myid) * nparts;
   offset += (myid < ((int)g_nparts % nprocs)) ? (uint32_t)myid
                                               : (uint32_t)((int)g_nparts % nprocs);
   for (part = 0; part < nparts; part++)
   {
      partids[part] = offset + part;
   }

   /* 2) Read nrows/nnz for each part */
   nrows_max = nrows_sum = 0;
   for (part = 0; part < nparts; part++)
   {
      sprintf(filename, "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      if (!fp)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(filename);
         return;
      }

      /* Read header contents */
      if (fread(header, sizeof(uint64_t), 8, fp) != 8)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         return;
      }
      fclose(fp);

      nrows_sum += header[5];
      nrows_max = (header[5] > nrows_max) ? header[5] : nrows_max;
   }

   /* 3) Build IJVector */
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + nrows_sum - 1);

   HYPRE_IJVectorCreate(comm, ilower, iupper, &vec);
   HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec, memory_location);

   /* Allocate variables */
   h_vals = (HYPRE_Complex *)malloc(nrows_max * sizeof(HYPRE_Complex));
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
      sprintf(filename, "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      if (fread(header, sizeof(uint64_t), 8, fp) != 8)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         return;
      }

      /* Read vector coefficients */
      if (header[1] == sizeof(float))
      {
         float *buffer = (float *)malloc(header[5] * sizeof(float));

         if (fread(buffer, sizeof(float), header[5], fp) != header[5])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[5]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else if (header[1] == sizeof(double))
      {
         double *buffer = (double *)malloc(header[5] * sizeof(double));

         if (fread(buffer, sizeof(double), header[5], fp) != header[5])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[5]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid coefficient data type size %lld at %s", header[1],
                     filename);
         fclose(fp);
         return;
      }
      fclose(fp);

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

   /* Free memory */
   free(partids);
   free(h_vals);
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
   }
#endif
}
