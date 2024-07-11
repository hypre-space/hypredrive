/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "utils.h"
#include "HYPRE_IJ_mv.h"

void
IJMatrixReadMultipartBinary(const char     *prefixname,
                            MPI_Comm        comm,
                            uint64_t        g_nparts,
                            HYPRE_IJMatrix *mat_ptr)
{
   int               nprocs, myid;
   uint32_t          nparts;
   uint64_t          part;
   uint32_t          offset;

   char              filename[1024];
   uint64_t          header[11];

   uint64_t          nrows_sum, nrows_offset;
   uint64_t          nnzs_max;

   uint32_t         *partids;
   FILE             *fp;

   HYPRE_IJMatrix    mat;
   HYPRE_BigInt      ilower, iupper;
   HYPRE_BigInt     *rows, *cols;
   HYPRE_Complex    *vals;

   /* 1a) Find number of parts per processor */
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myid);
   nparts = g_nparts / (uint64_t) nprocs;
   nparts += (myid < ((int) g_nparts % nprocs)) ? 1 : 0;
   if (g_nparts < (size_t) nprocs)
   {
      *mat_ptr = NULL;
      ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      ErrorMsgAdd("Invalid number of parts!");
      return;
   }

   /* 1b) Compute partids array */
   partids = (uint32_t*) malloc(nparts * sizeof(uint32_t));
   offset  = ((uint32_t) myid) * nparts;
   offset  += (myid < ((int) g_nparts % nprocs)) ? (uint32_t) myid : (uint32_t) ((int) g_nparts % nprocs);
   for (part = 0; part < nparts; part++)
   {
      partids[part] = offset + part;
   }

   /* 2) Read nrows/nnz for each part */
   nrows_sum = nnzs_max = 0;
   for (part = 0; part < nparts; part++)
   {
      sprintf(filename, "%s.%05d.bin", prefixname, partids[part]);
      fp = fopen(filename, "rb");
      if (!fp)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(filename);
         fclose(fp);
         return;
      }

      /* Read header contents */
      if (fread(header, sizeof(uint64_t), 11, fp) != 11)
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Could not read header from %s", filename);
         fclose(fp);
         return;
      }
      fclose(fp);

      nrows_sum += (uint64_t) (header[8] - header[7] + 1);
      nnzs_max  = ((uint64_t) header[6] > nnzs_max) ? (uint64_t) header[6] : nnzs_max;
   }

   /* 3) Build IJMatrix */
   MPI_Scan(&nrows_sum, &nrows_offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
   ilower = (HYPRE_BigInt) (nrows_offset - nrows_sum);
   iupper = (HYPRE_BigInt) (ilower + nrows_sum - 1);

   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize_v2(mat, HYPRE_MEMORY_HOST);

   /* 4) Fill entries */
   rows = (HYPRE_BigInt*) malloc(nnzs_max * sizeof(HYPRE_BigInt));
   cols = (HYPRE_BigInt*) malloc(nnzs_max * sizeof(HYPRE_BigInt));
   vals = (HYPRE_Complex*) malloc(nnzs_max * sizeof(HYPRE_Complex));
   for (part = 0; part < nparts; part++)
   {
      sprintf(filename, "%s.%05d.bin", prefixname, partids[part]);
      fp = fopen(filename, "rb");
      fread(header, sizeof(uint64_t), 11, fp) == 11;

      /* Read row and column indices */
      if (header[1] == sizeof(HYPRE_BigInt))
      {
         if (fread(rows, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            return;
         }

         if (fread(cols, sizeof(HYPRE_BigInt), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            return;
         }
      }
      else if (header[1] == sizeof(uint32_t))
      {
         uint32_t* buffer = (uint32_t*) malloc(header[6] * sizeof(uint32_t));

         if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) rows[i] = (HYPRE_BigInt) buffer[i];

         if (fread(buffer, sizeof(uint32_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) cols[i] = (HYPRE_BigInt) buffer[i];

         free(buffer);
      }
      else if (header[1] == sizeof(uint64_t))
      {
         uint64_t* buffer = (uint64_t*) malloc(header[6] * sizeof(uint64_t));

         if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read row indices from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) rows[i] = (HYPRE_BigInt) buffer[i];

         if (fread(buffer, sizeof(uint64_t), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read column indices from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) cols[i] = (HYPRE_BigInt) buffer[i];

         free(buffer);
      }
      else
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid row/col data type size %lld at %s", header[1], filename);
         fclose(fp);
         return;
      }

      /* Read matrix coefficients */
      if (header[2] == sizeof(HYPRE_Complex))
      {
         if (fread(vals, sizeof(HYPRE_Complex), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            return;
         }
      }
      else if (header[2] == sizeof(float))
      {
         float* buffer = (float*) malloc(header[6] * sizeof(float));

         if (fread(buffer, sizeof(float), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) vals[i] = (HYPRE_Complex) buffer[i];

         free(buffer);
      }
      else if (header[2] == sizeof(double))
      {
         double* buffer = (double*) malloc(header[6] * sizeof(double));

         if (fread(buffer, sizeof(double), header[6], fp) != header[6])
         {
            ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
            ErrorMsgAdd("Could not read coeficients from %s", filename);
            fclose(fp);
            return;
         }

         for (size_t i = 0; i < header[6]; i++) vals[i] = (HYPRE_Complex) buffer[i];

         free(buffer);
      }
      else
      {
         ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         ErrorMsgAdd("Invalid coefficient data type size %lld at %s", header[2], filename);
         fclose(fp);
         return;
      }

      HYPRE_IJMatrixSetValues(mat, (HYPRE_BigInt) header[6], NULL, rows, cols, vals);
   }

   HYPRE_IJMatrixAssemble(mat);
   *mat_ptr = mat;

   /* Free memory */
   free(partids);
   free(rows);
   free(cols);
   free(vals);
}
