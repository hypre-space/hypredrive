/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "HYPREDRV.h"

_Static_assert(((HYPRE_BigInt)-1) < (HYPRE_BigInt)0,
               "HYPREDRV Fortran bridge requires signed HYPRE_BigInt");
_Static_assert(sizeof(HYPRE_Real) == sizeof(double),
               "HYPREDRV Fortran bridge requires HYPRE_Real to use the C double ABI");

static uint32_t
hypredrv_fortran_bigint_from_i64(int64_t value, const char *name, HYPRE_BigInt *out)
{
   HYPRE_BigInt converted = (HYPRE_BigInt)value;
   if ((int64_t)converted != value)
   {
      char message[256];
      int  n = snprintf(message, sizeof(message),
                        "%s=%lld is outside the active HYPRE_BigInt range", name,
                        (long long)value);
      if (n < 0 || (size_t)n >= sizeof(message))
      {
         (void)snprintf(message, sizeof(message),
                        "value is outside the active HYPRE_BigInt range");
      }
      return HYPREDRV_ErrorInvalidValue(message);
   }
   *out = converted;
   return HYPREDRV_SUCCESS;
}

size_t
HYPREDRV_FortranBigIntSize(void)
{
   return sizeof(HYPRE_BigInt);
}

uint32_t
HYPREDRV_FortranCreate(MPI_Fint fcomm, HYPREDRV_t *hypredrv_ptr)
{
   return HYPREDRV_Create(MPI_Comm_f2c(fcomm), hypredrv_ptr);
}

uint32_t
HYPREDRV_FortranPrintLibInfo(MPI_Fint fcomm, int print_datetime)
{
   return HYPREDRV_PrintLibInfo(MPI_Comm_f2c(fcomm), print_datetime);
}

uint32_t
HYPREDRV_FortranPrintSystemInfo(MPI_Fint fcomm)
{
   return HYPREDRV_PrintSystemInfo(MPI_Comm_f2c(fcomm));
}

uint32_t
HYPREDRV_FortranPrintExitInfo(MPI_Fint fcomm, const char *argv0)
{
   return HYPREDRV_PrintExitInfo(MPI_Comm_f2c(fcomm), argv0);
}

uint32_t
HYPREDRV_FortranLinearSystemSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start_in,
                                             int64_t row_end_in, const int64_t *indptr_in,
                                             int64_t           indptr_len,
                                             const int64_t    *col_indices_in,
                                             int64_t           col_indices_len,
                                             const HYPRE_Real *data, int64_t data_len)
{
   HYPRE_BigInt row_start;
   HYPRE_BigInt row_end;
   uint32_t     ierr;

   ierr = hypredrv_fortran_bigint_from_i64(row_start_in, "row_start", &row_start);
   if (ierr)
   {
      return ierr;
   }
   ierr = hypredrv_fortran_bigint_from_i64(row_end_in, "row_end", &row_end);
   if (ierr)
   {
      return ierr;
   }

   if (indptr_len <= 0 || col_indices_len < 0 || data_len < 0 ||
       col_indices_len != data_len)
   {
      return HYPREDRV_ErrorInvalidValue("invalid Fortran CSR buffer lengths");
   }

   int64_t nrows = row_end_in - row_start_in + 1;
   if (nrows <= 0 || indptr_len != nrows + 1)
   {
      return HYPREDRV_ErrorInvalidValue(
         "Fortran CSR indptr length does not match row range");
   }

   if (indptr_in[0] != 0 || indptr_in[indptr_len - 1] != col_indices_len)
   {
      return HYPREDRV_ErrorInvalidValue(
         "Fortran CSR input must use normalized indptr[0]=0 and matching nnz length");
   }

   HYPRE_BigInt *indptr = NULL;
   HYPRE_BigInt *cols   = NULL;

   indptr = (HYPRE_BigInt *)malloc((size_t)indptr_len * sizeof(*indptr));
   if (!indptr)
   {
      return HYPREDRV_ErrorInvalidValue(
         "failed to allocate Fortran CSR indptr conversion buffer");
   }
   for (int64_t i = 0; i < indptr_len; i++)
   {
      ierr = hypredrv_fortran_bigint_from_i64(indptr_in[i], "indptr", &indptr[i]);
      if (ierr)
      {
         free(indptr);
         return ierr;
      }
   }

   if (col_indices_len > 0)
   {
      cols = (HYPRE_BigInt *)malloc((size_t)col_indices_len * sizeof(*cols));
      if (!cols)
      {
         free(indptr);
         return HYPREDRV_ErrorInvalidValue(
            "failed to allocate Fortran CSR column conversion buffer");
      }
      for (int64_t i = 0; i < col_indices_len; i++)
      {
         ierr =
            hypredrv_fortran_bigint_from_i64(col_indices_in[i], "col_indices", &cols[i]);
         if (ierr)
         {
            free(cols);
            free(indptr);
            return ierr;
         }
      }
   }

   ierr = HYPREDRV_LinearSystemSetMatrixFromCSR(hypredrv, row_start, row_end, indptr,
                                                cols, data);
   free(cols);
   free(indptr);
   return ierr;
}

uint32_t
HYPREDRV_FortranLinearSystemSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start_in,
                                            int64_t row_end_in, const HYPRE_Real *values,
                                            int64_t values_len)
{
   HYPRE_BigInt row_start;
   HYPRE_BigInt row_end;
   uint32_t     ierr;

   ierr = hypredrv_fortran_bigint_from_i64(row_start_in, "row_start", &row_start);
   if (ierr)
   {
      return ierr;
   }
   ierr = hypredrv_fortran_bigint_from_i64(row_end_in, "row_end", &row_end);
   if (ierr)
   {
      return ierr;
   }

   int64_t nrows = row_end_in - row_start_in + 1;
   if (nrows <= 0 || values_len != nrows)
   {
      return HYPREDRV_ErrorInvalidValue("Fortran RHS length does not match row range");
   }

   return HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end, values);
}

uint32_t
HYPREDRV_FortranLinearSystemGetSolutionLength(HYPREDRV_t hypredrv, int64_t *length_out)
{
   HYPRE_BigInt length;
   uint32_t     ierr = HYPREDRV_LinearSystemGetSolutionLength(hypredrv, &length);
   if (ierr)
   {
      return ierr;
   }
   *length_out = (int64_t)length;
   return HYPREDRV_SUCCESS;
}
