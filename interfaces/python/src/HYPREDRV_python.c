/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/**
 * @file HYPREDRV_python.c
 * @brief Python-extension bridge for ABI-variable HYPRE scalar and index types.
 *
 * Cython declarations need concrete C types, while HYPRE can be configured with
 * different widths for HYPRE_BigInt and HYPRE_Real. This file keeps the
 * ABI-sensitive casts in C, compiled against the active HYPRE headers, and
 * exposes fixed-width and opaque-buffer entry points to the Cython extension.
 *
 * The file is intentionally compiled only into the Python extension target. It
 * is not part of the core libHYPREDRV build when Python support is disabled.
 */

#include <stdint.h>
#include <stdio.h>

#include "HYPREDRV_python.h"

/**
 * @brief Convert a Python ABI int64 row bound into HYPRE_BigInt.
 *
 * @param value     Fixed-width value received from the Cython extension.
 * @param name      Field name used in diagnostics.
 * @param converted Output HYPRE_BigInt value.
 *
 * @return 0 on success; a HYPREDRV error code when the active HYPRE_BigInt type
 * cannot represent @p value.
 */
static uint32_t
HYPREDRV_PythonBigIntFromInt64(int64_t value, const char *name,
                               HYPRE_BigInt *converted)
{
   HYPRE_BigInt tmp = (HYPRE_BigInt)value;
   if ((int64_t)tmp != value)
   {
      char message[256];
      int  written = snprintf(message, sizeof(message),
                              "%s (%lld) is outside the active HYPRE_BigInt range",
                              name, (long long)value);
      if (written < 0 || written >= (int)sizeof(message))
      {
         return HYPREDRV_ErrorInvalidValue(
            "Python integer is outside the active HYPRE_BigInt range");
      }
      return HYPREDRV_ErrorInvalidValue(message);
   }

   *converted = tmp;
   return 0;
}

/**
 * @brief Return sizeof(HYPRE_BigInt) for the linked HYPRE build.
 */
size_t
HYPREDRV_PythonIndexSize(void)
{
   return sizeof(HYPRE_BigInt);
}

/**
 * @brief Return sizeof(HYPRE_Real) for the linked HYPRE build.
 */
size_t
HYPREDRV_PythonRealSize(void)
{
   return sizeof(HYPRE_Real);
}

/**
 * @brief Return sizeof(HYPRE_Complex) for the linked HYPRE build.
 */
size_t
HYPREDRV_PythonSolutionEntrySize(void)
{
   return sizeof(HYPRE_Complex);
}

/**
 * @brief Create a serial HYPREDRV object using MPI_COMM_SELF.
 */
uint32_t
HYPREDRV_PythonCreateWithSelf(HYPREDRV_t *hypredrv_ptr)
{
   return HYPREDRV_Create(MPI_COMM_SELF, hypredrv_ptr);
}

/**
 * @brief Create a HYPREDRV object from a Fortran MPI communicator handle.
 */
uint32_t
HYPREDRV_PythonCreateFromFortranComm(int fortran_comm, HYPREDRV_t *hypredrv_ptr)
{
   return HYPREDRV_Create(MPI_Comm_f2c((MPI_Fint)fortran_comm), hypredrv_ptr);
}

/**
 * @brief Create a distributed HYPREDRV object using MPI_COMM_WORLD.
 */
uint32_t
HYPREDRV_PythonCreateWithWorld(HYPREDRV_t *hypredrv_ptr)
{
   return HYPREDRV_Create(MPI_COMM_WORLD, hypredrv_ptr);
}

/**
 * @brief Install a CSR matrix from Python-owned opaque buffers.
 */
uint32_t
HYPREDRV_PythonSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start,
                                int64_t row_end, const void *indptr,
                                const void *col_indices, const void *data)
{
   HYPRE_BigInt hypre_row_start = 0;
   HYPRE_BigInt hypre_row_end   = 0;
   uint32_t     code            = 0;

   code = HYPREDRV_PythonBigIntFromInt64(row_start, "row_start", &hypre_row_start);
   if (code != 0)
   {
      return code;
   }
   code = HYPREDRV_PythonBigIntFromInt64(row_end, "row_end", &hypre_row_end);
   if (code != 0)
   {
      return code;
   }

   return HYPREDRV_LinearSystemSetMatrixFromCSR(
      hypredrv, hypre_row_start, hypre_row_end, (const HYPRE_BigInt *)indptr,
      (const HYPRE_BigInt *)col_indices, (const HYPRE_Real *)data);
}

/**
 * @brief Install a right-hand-side vector from a Python-owned opaque buffer.
 */
uint32_t
HYPREDRV_PythonSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start,
                               int64_t row_end, const void *values)
{
   HYPRE_BigInt hypre_row_start = 0;
   HYPRE_BigInt hypre_row_end   = 0;
   uint32_t     code            = 0;

   code = HYPREDRV_PythonBigIntFromInt64(row_start, "row_start", &hypre_row_start);
   if (code != 0)
   {
      return code;
   }
   code = HYPREDRV_PythonBigIntFromInt64(row_end, "row_end", &hypre_row_end);
   if (code != 0)
   {
      return code;
   }

   return HYPREDRV_LinearSystemSetRHSFromArray(
      hypredrv, hypre_row_start, hypre_row_end, (const HYPRE_Real *)values);
}

/**
 * @brief Return an opaque pointer to the current solution values.
 */
uint32_t
HYPREDRV_PythonGetSolutionValues(HYPREDRV_t hypredrv, const void **sol_data,
                                 size_t *sol_length)
{
   if (!sol_data || !sol_length)
   {
      return HYPREDRV_ErrorInvalidValue(
         "HYPREDRV_PythonGetSolutionValues: sol_data and sol_length must be non-NULL");
   }

   HYPRE_Complex *src        = NULL;
   HYPRE_BigInt   length_big = 0;
   uint32_t       code       = HYPREDRV_LinearSystemGetSolutionLength(hypredrv,
                                                                      &length_big);
   if (code != 0)
   {
      return code;
   }
   if (length_big < 0)
   {
      return HYPREDRV_ErrorInvalidValue(
         "HYPREDRV_PythonGetSolutionValues: solution length is negative");
   }
   code        = HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &src);
   *sol_data   = (const void *)src;
   *sol_length = (size_t)length_big;
   return code;
}
