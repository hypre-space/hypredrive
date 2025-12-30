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
IJMatrixReadMultipartBinary(const char *prefixname, MPI_Comm comm, uint64_t g_nparts,
                            HYPRE_MemoryLocation memory_location, HYPRE_IJMatrix *mat_ptr)
{
   int      nprocs = 0, myid = 0;
   uint32_t nparts = 0;
   uint64_t part   = 0;
   uint32_t offset = 0;

   char     filename[1024];
   uint64_t header[11];

   uint64_t nrows     = 0;
   uint64_t nrows_sum = 0, nrows_offset = 0;
   uint64_t nnzs_max = 0;

   uint32_t *partids = NULL;
   FILE     *fp      = NULL;

   HYPRE_IJMatrix       mat    = NULL;
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_BigInt        *h_rows = NULL;
   HYPRE_BigInt        *h_cols = NULL;
   HYPRE_Complex       *h_vals = NULL;
   const HYPRE_Int     *rows   = NULL;
   const HYPRE_Int     *cols   = NULL;
   const HYPRE_Complex *vals   = NULL;
#ifdef HYPRE_USING_GPU
   HYPRE_BigInt  *d_rows = NULL;
   HYPRE_BigInt  *d_cols = NULL;
   HYPRE_Complex *d_vals = NULL;
#endif

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = g_nparts / (uint64_t)nprocs;
   nparts += (myid < ((int)g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < (size_t)nprocs)
   {
      *mat_ptr = NULL;
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
   nrows_sum = nnzs_max = 0;
   for (part = 0; part < nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      if (!fp)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(filename);
         goto cleanup;
      }

      /* Read header contents */
      if (fread(header, sizeof(uint64_t), 11, fp) != 11)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         goto cleanup;
      }
      fclose(fp);

      nrows_sum += (uint64_t)(header[8] - header[7] + 1);
      nnzs_max = ((uint64_t)header[6] > nnzs_max) ? (uint64_t)header[6] : nnzs_max;
   }

   // printf("[%d]: nrows_sum: %lld, nnz_max: %lld!\n", myid, nrows_sum, nnzs_max);
   // fflush(stdout); MPI_Barrier(MPI_COMM_WORLD);

   /* 3) Build IJMatrix */
   MPI_Allreduce(&nrows_sum, &nrows, 1, MPI_UINT64_T, MPI_SUM, comm);
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
   ilower = (HYPRE_BigInt)(nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt)(ilower + nrows_sum - 1);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);

   /* 4) Fill entries */
   h_rows =
      (nnzs_max > 0) ? (HYPRE_BigInt *)malloc(nnzs_max * sizeof(HYPRE_BigInt)) : NULL;
   h_cols =
      (nnzs_max > 0) ? (HYPRE_BigInt *)malloc(nnzs_max * sizeof(HYPRE_BigInt)) : NULL;
   h_vals =
      (nnzs_max > 0) ? (HYPRE_Complex *)malloc(nnzs_max * sizeof(HYPRE_Complex)) : NULL;

   /* 4a) Pre-compute the sparsity pattern when storing on host memory */
   if (memory_location == HYPRE_MEMORY_HOST)
   {
      HYPRE_Int *dsizes = (HYPRE_Int *)calloc(nrows_sum, sizeof(HYPRE_Int));
      HYPRE_Int *osizes = (HYPRE_Int *)calloc(nrows_sum, sizeof(HYPRE_Int));

      for (part = 0; part < nparts; part++)
      {
         snprintf(filename, sizeof(filename), "%s.%05d.bin", prefixname,
                  (int)partids[part]);
         fp = fopen(filename, "rb");
         if (!fp || fread(header, sizeof(uint64_t), 11, fp) != 11)
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read header from %s", filename);
            if (fp) fclose(fp);
            free(dsizes);
            free(osizes);
            goto cleanup;
         }

         /* Read row and column indices */
         if (header[1] == sizeof(HYPRE_BigInt) && h_rows != NULL && h_cols != NULL &&
             header[6] > 0)
         {
            if (fread(h_rows, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read row indices from %s", filename);
               fclose(fp);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }

            if (fread(h_cols, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read column indices from %s", filename);
               fclose(fp);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }
         }
         else if (header[1] == sizeof(uint32_t))
         {
            uint32_t *buffer = (uint32_t *)malloc(header[6] * sizeof(uint32_t));

            if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read row indices from %s", filename);
               fclose(fp);
               free(buffer);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }

            for (size_t i = 0; h_rows && i < header[6]; i++)
            {
               h_rows[i] = (HYPRE_BigInt)buffer[i];
            }

            if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read column indices from %s", filename);
               fclose(fp);
               free(buffer);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }

            for (size_t i = 0; h_cols && i < header[6]; i++)
            {
               h_cols[i] = (HYPRE_BigInt)buffer[i];
            }

            free(buffer);
         }
         else if (header[1] == sizeof(uint64_t))
         {
            uint64_t *buffer = (uint64_t *)malloc(header[6] * sizeof(uint64_t));

            if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read row indices from %s", filename);
               fclose(fp);
               free(buffer);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }

            for (size_t i = 0; h_rows && i < header[6]; i++)
            {
               h_rows[i] = (HYPRE_BigInt)buffer[i];
            }

            if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
            {
               ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
               ErrorMsgAdd("Could not read column indices from %s", filename);
               fclose(fp);
               free(buffer);
               free(dsizes);
               free(osizes);
               goto cleanup;
            }

            for (size_t i = 0; h_cols && i < header[6]; i++)
            {
               h_cols[i] = (HYPRE_BigInt)buffer[i];
            }

            free(buffer);
         }
         else
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Invalid row/col data type size %lld at %s", header[1], filename);
            fclose(fp);
            free(dsizes);
            free(osizes);
            goto cleanup;
         }
         fclose(fp);

         /* TODO: add threading */
         for (size_t i = 0; i < header[6]; i++)
         {
            if (!h_rows)
            {
               break;
            }
            int64_t row = h_rows[i];
            int64_t col = h_cols[i];

            /* Check if (row, col) pair makes sense */
            if (row < 0)
            {
               printf("[%d]: Warning! Detected negative row: %llu\n", myid,
                      (unsigned long long)row);
               fflush(stdout);
            }
            else if (col < 0)
            {
               printf("[%d]: Warning! Detected negative column: %llu\n", myid,
                      (unsigned long long)col);
               fflush(stdout);
            }
            else if (row >= nrows)
            {
               printf("[%d]: Warning! Detected out-of-bounds row: %llu\n", myid,
                      (unsigned long long)row);
               fflush(stdout);
            }
            else if (col >= nrows)
            {
               printf("[%d]: Warning! Detected out-of-bounds column: %llu\n", myid,
                      (unsigned long long)col);
               fflush(stdout);
            }
            else if (row < ilower || row > iupper)
            {
               /* This row does not belong to the current rank. Skipping it... */
               continue;
            }

            if (dsizes && osizes)
            {
               if (col >= ilower && col <= iupper)
               {
                  dsizes[row - ilower]++;
               }
               else
               {
                  osizes[row - ilower]++;
               }
            }
         }
      }

      /* Pre-allocating the sparsity pattern */
      HYPRE_IJMatrixSetDiagOffdSizes(mat, dsizes, osizes);
      free(dsizes);
      free(osizes);
   }

   /* Allocate matrix on the final memory */
   HYPRE_IJMatrixInitialize_v2(mat, memory_location);

   /* Allocate device variables */
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      rows = d_rows = hypre_TAlloc(HYPRE_BigInt, nnzs_max, memory_location);
      cols = d_cols = hypre_TAlloc(HYPRE_BigInt, nnzs_max, memory_location);
      vals = d_vals = hypre_TAlloc(HYPRE_Complex, nnzs_max, memory_location);
   }
   else
#endif
   {
      rows = h_rows;
      cols = h_cols;
      vals = h_vals;
   }

   /* Set matrix values */
   for (part = 0; part < nparts; part++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d.bin", prefixname, (int)partids[part]);
      fp = fopen(filename, "rb");
      if (!fp || fread(header, sizeof(uint64_t), 11, fp) != 11)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not read header from %s", filename);
         if (fp) fclose(fp);
         goto cleanup;
      }

      /* Read row and column indices */
      if (header[1] == sizeof(HYPRE_BigInt) && h_rows != NULL && h_cols != NULL &&
          header[6] > 0)
      {
         if (fread(h_rows, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            goto cleanup;
         }

         if (fread(h_cols, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            goto cleanup;
         }
      }
      else if (header[1] == sizeof(uint32_t))
      {
         uint32_t *buffer = (uint32_t *)malloc(header[6] * sizeof(uint32_t));

         if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; i < header[6]; i++)
         {
            h_rows[i] = (HYPRE_BigInt)buffer[i];
         }

         if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; i < header[6]; i++)
         {
            h_cols[i] = (HYPRE_BigInt)buffer[i];
         }

         free(buffer);
      }
      else if (header[1] == sizeof(uint64_t))
      {
         uint64_t *buffer = (uint64_t *)malloc(header[6] * sizeof(uint64_t));

         if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; i < header[6]; i++)
         {
            h_rows[i] = (HYPRE_BigInt)buffer[i];
         }

         if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; i < header[6]; i++)
         {
            h_cols[i] = (HYPRE_BigInt)buffer[i];
         }

         free(buffer);
      }
      else
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid row/col data type size %lld at %s", header[1], filename);
         fclose(fp);
         goto cleanup;
      }

#ifdef HYPRE_USING_GPU
      if (rows != h_rows)
      {
         hypre_TMemcpy(d_rows, h_rows, HYPRE_BigInt, header[6], HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
      if (cols != h_cols)
      {
         hypre_TMemcpy(d_cols, h_cols, HYPRE_BigInt, header[6], HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
#endif

      /* Read matrix coefficients */
      if (header[2] == sizeof(float))
      {
         float *buffer = NULL;
         if (header[6] > 0)
         {
            buffer = (float *)malloc(header[6] * sizeof(float));
         }

         if (!buffer || fread(buffer, sizeof(float), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; h_vals && i < header[6]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else if (header[2] == sizeof(double))
      {
         double *buffer = NULL;
         if (header[6] > 0)
         {
            buffer = (double *)malloc(header[6] * sizeof(double));
         }

         if (!buffer || fread(buffer, sizeof(double), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            free(buffer);
            goto cleanup;
         }

         for (size_t i = 0; h_vals && i < header[6]; i++)
         {
            h_vals[i] = (HYPRE_Complex)buffer[i];
         }

         free(buffer);
      }
      else
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid coefficient data type size %lld at %s", header[2],
                     filename);
         fclose(fp);
         goto cleanup;
      }
      fclose(fp);

#ifdef HYPRE_USING_GPU
      if (vals != h_vals)
      {
         hypre_TMemcpy(d_vals, h_vals, HYPRE_Complex, header[6], HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
#endif

      HYPRE_IJMatrixSetValues(mat, (HYPRE_BigInt)header[6], NULL, rows, cols, vals);
   }

   HYPRE_IJMatrixAssemble(mat);
   *mat_ptr = mat;

cleanup:
   /* Free memory */
   free(partids);
   free(h_rows);
   free(h_cols);
   free(h_vals);
#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(d_rows, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_cols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
   }
#endif
   if (ErrorCodeActive())
   {
      if (mat)
      {
         HYPRE_IJMatrixDestroy(mat);
      }
      *mat_ptr = NULL;
   }
}
