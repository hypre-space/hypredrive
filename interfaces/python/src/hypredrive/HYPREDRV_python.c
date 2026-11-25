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
#include <string.h>

#include "HYPREDRV_python.h"
#include "internal/bridge_row_bounds.h"

static int HYPREDRV_python_mpi_owned = 0;

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
HYPREDRV_PythonBigIntFromInt64(int64_t value, const char *name, HYPRE_BigInt *converted)
{
   if (!hypredrv_BigIntFromI64(value, converted))
   {
      char message[256];
      int  written = snprintf(message, sizeof(message),
                              "%s (%lld) is outside the active HYPRE_BigInt range", name,
                              (long long)value);
      if (written < 0 || written >= (int)sizeof(message))
      {
         return HYPREDRV_ErrorInvalidValue(
            "Python integer is outside the active HYPRE_BigInt range");
      }
      return HYPREDRV_ErrorInvalidValue(message);
   }

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
 * @brief Initialize MPI with the same MPI library linked into hypredrive.
 */
uint32_t
HYPREDRV_PythonMPIInitialize(void)
{
   int initialized = 0;
   int finalized   = 0;
   MPI_Finalized(&finalized);
   if (finalized)
   {
      return HYPREDRV_ErrorInvalidValue(
         "HYPREDRV_PythonMPIInitialize: MPI has already been finalized");
   }
   MPI_Initialized(&initialized);
   if (initialized)
   {
      return 0;
   }

   int provided = 0;
   int mpi_ierr = MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
   if (mpi_ierr != MPI_SUCCESS)
   {
      return HYPREDRV_ErrorInvalidValue(
         "HYPREDRV_PythonMPIInitialize: MPI_Init_thread failed");
   }
   if (provided < MPI_THREAD_SERIALIZED)
   {
      return HYPREDRV_ErrorInvalidValue(
         "HYPREDRV_PythonMPIInitialize: MPI implementation did not provide the "
         "requested MPI_THREAD_SERIALIZED thread level");
   }
   HYPREDRV_python_mpi_owned = 1;
   return 0;
}

/**
 * @brief Finalize MPI if HYPREDRV_PythonMPIInitialize initialized it.
 */
uint32_t
HYPREDRV_PythonMPIFinalize(void)
{
   if (!HYPREDRV_python_mpi_owned)
   {
      return 0;
   }

   int finalized = 0;
   MPI_Finalized(&finalized);
   if (!finalized)
   {
      int mpi_ierr = MPI_Finalize();
      if (mpi_ierr != MPI_SUCCESS)
      {
         return HYPREDRV_ErrorInvalidValue(
            "HYPREDRV_PythonMPIFinalize: MPI_Finalize failed");
      }
   }
   HYPREDRV_python_mpi_owned = 0;
   return 0;
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

/* MPI_Comm is never exposed to Cython: its representation (int, pointer, ...)
 * depends on the MPI implementation. Fortran handles may be negative (MPICH sets
 * the high bit on user communicators), so MPI_COMM_SELF is selected explicitly. */
static MPI_Comm
HYPREDRV_PythonComm(int use_self, int fortran_comm)
{
   return use_self ? MPI_COMM_SELF : MPI_Comm_f2c((MPI_Fint)fortran_comm);
}

/** @brief HYPREDRV_PrintLibInfo on a communicator given by its Fortran handle. */
uint32_t
HYPREDRV_PythonPrintLibInfo(int use_self, int fortran_comm, int print_datetime)
{
   return HYPREDRV_PrintLibInfo(HYPREDRV_PythonComm(use_self, fortran_comm),
                                print_datetime);
}

/** @brief HYPREDRV_PrintSystemInfo on a communicator given by its Fortran handle. */
uint32_t
HYPREDRV_PythonPrintSystemInfo(int use_self, int fortran_comm)
{
   return HYPREDRV_PrintSystemInfo(HYPREDRV_PythonComm(use_self, fortran_comm));
}

/** @brief HYPREDRV_PrintExitInfo on a communicator given by its Fortran handle. */
uint32_t
HYPREDRV_PythonPrintExitInfo(int use_self, int fortran_comm, const char *argv0)
{
   return HYPREDRV_PrintExitInfo(HYPREDRV_PythonComm(use_self, fortran_comm), argv0);
}

/**
 * @brief Install a CSR matrix from Python-owned opaque buffers.
 */
uint32_t
HYPREDRV_PythonSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start, int64_t row_end,
                                const void *indptr, const void *col_indices,
                                const void *data)
{
   HYPRE_BigInt hypre_row_start = 0;
   HYPRE_BigInt hypre_row_end   = 0;
   uint32_t     code =
      hypredrv_BridgeRowBoundsFromI64(row_start, row_end, &hypre_row_start,
                                      &hypre_row_end, HYPREDRV_PythonBigIntFromInt64);
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
HYPREDRV_PythonSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start, int64_t row_end,
                               const void *values)
{
   HYPRE_BigInt hypre_row_start = 0;
   HYPRE_BigInt hypre_row_end   = 0;
   uint32_t     code =
      hypredrv_BridgeRowBoundsFromI64(row_start, row_end, &hypre_row_start,
                                      &hypre_row_end, HYPREDRV_PythonBigIntFromInt64);
   if (code != 0)
   {
      return code;
   }

   return HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, hypre_row_start, hypre_row_end,
                                               (const HYPRE_Real *)values);
}

/** @brief Copy local solution values into a caller-owned host array. */
uint32_t
HYPREDRV_PythonCopySolutionValues(HYPREDRV_t hypredrv, void *sol_data, size_t sol_length)
{
   HYPRE_BigInt count = (HYPRE_BigInt)sol_length;
   if (count < 0 || (size_t)count != sol_length)
   {
      return HYPREDRV_ErrorInvalidValue(
         "solution length exceeds the active HYPRE_BigInt range");
   }
   return HYPREDRV_LinearSystemCopySolutionValues(hypredrv, count,
                                                  (HYPRE_Complex *)sol_data);
}

/** @brief Return the local solution length using the Python bridge's size type. */
uint32_t
HYPREDRV_PythonGetSolutionLength(HYPREDRV_t hypredrv, size_t *sol_length)
{
   if (!sol_length) return HYPREDRV_ErrorInvalidValue("solution length output is NULL");
   HYPRE_BigInt count = 0;
   uint32_t     code  = HYPREDRV_LinearSystemGetSolutionLength(hypredrv, &count);
   if (code) return code;
   if (count < 0 || (uintmax_t)count > SIZE_MAX)
      return HYPREDRV_ErrorInvalidValue("solution length exceeds size_t range");
   *sol_length = (size_t)count;
   return HYPREDRV_SUCCESS;
}

/** @brief Local length of an IJ vector using the Python bridge's size type. */
uint32_t
HYPREDRV_PythonVectorLocalLength(HYPRE_IJVector vec, size_t *length)
{
   if (!vec || !length)
      return HYPREDRV_ErrorInvalidValue("vector or length output is NULL");
   HYPRE_BigInt ilower = 0, iupper = -1;
   if (HYPRE_IJVectorGetLocalRange(vec, &ilower, &iupper))
   {
      HYPRE_ClearAllErrors();
      return HYPREDRV_ErrorInvalidValue("failed to query the vector local range");
   }
   HYPRE_BigInt count = iupper - ilower + 1;
   if (count < 0 || (uintmax_t)count > SIZE_MAX)
      return HYPREDRV_ErrorInvalidValue("vector length exceeds size_t range");
   *length = (size_t)count;
   return HYPREDRV_SUCCESS;
}

/** @brief Local row range of the driver's system matrix as fixed-width values. */
uint32_t
HYPREDRV_PythonGetMatrixRowRange(HYPREDRV_t hypredrv, int64_t *row_start,
                                 int64_t *row_end)
{
   if (!row_start || !row_end)
      return HYPREDRV_ErrorInvalidValue("row range output is NULL");
   HYPRE_Matrix mat  = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetMatrix(hypredrv, &mat);
   if (code) return code;
   HYPRE_BigInt ilower = 0, iupper = -1, jlower = 0, jupper = -1;
   if (!mat || HYPRE_IJMatrixGetLocalRange((HYPRE_IJMatrix)mat, &ilower, &iupper, &jlower,
                                           &jupper))
   {
      HYPRE_ClearAllErrors();
      return HYPREDRV_ErrorInvalidValue("failed to query the matrix local range");
   }
   *row_start = (int64_t)ilower;
   *row_end   = (int64_t)iupper;
   return HYPREDRV_SUCCESS;
}

/** @brief Local RHS length using the Python bridge's size type. */
uint32_t
HYPREDRV_PythonGetRHSLength(HYPREDRV_t hypredrv, size_t *length)
{
   HYPRE_Vector rhs  = NULL;
   uint32_t     code = HYPREDRV_LinearSystemGetRHS(hypredrv, &rhs);
   if (code) return code;
   return HYPREDRV_PythonVectorLocalLength((HYPRE_IJVector)rhs, length);
}

/**
 * @brief Copy the local RHS into caller-owned host storage of exactly
 * @p length HYPRE_Complex entries (the RHS local length).
 */
uint32_t
HYPREDRV_PythonCopyRHSValues(HYPREDRV_t hypredrv, void *dst, size_t length)
{
   size_t   n    = 0;
   uint32_t code = HYPREDRV_PythonGetRHSLength(hypredrv, &n);
   if (code) return code;
   if (n != length) return HYPREDRV_ErrorInvalidValue("RHS copy length mismatch");
#ifdef HYPRE_USING_GPU
   /* HYPREDRV_LinearSystemGetRHSValues reports no memory location, so the raw
    * pointer may be device memory on GPU builds. */
   (void)dst;
   return HYPREDRV_ErrorInvalidValue(
      "RHS values are not readable on GPU HYPRE builds; use get_rhs_handle()");
#else
   HYPRE_Complex *data = NULL;
   code                = HYPREDRV_LinearSystemGetRHSValues(hypredrv, &data);
   if (code) return code;
   if (n && (!data || !dst))
      return HYPREDRV_ErrorInvalidValue("RHS data is not available");
   if (n) memcpy(dst, data, n * sizeof(HYPRE_Complex));
   return HYPREDRV_SUCCESS;
#endif
}

/**
 * @brief Copy @p length entries of state vector @p index into caller-owned host
 * storage. The caller sizes @p length from the registered state vectors.
 */
uint32_t
HYPREDRV_PythonCopyStateVectorValues(HYPREDRV_t hypredrv, int index, void *dst,
                                     size_t length)
{
   HYPRE_Complex *data = NULL;
   uint32_t       code = HYPREDRV_StateVectorGetValues(hypredrv, index, &data);
   if (code) return code;
   if (length && (!data || !dst))
      return HYPREDRV_ErrorInvalidValue("state vector data is not available");
   if (length) memcpy(dst, data, length * sizeof(HYPRE_Complex));
   return HYPREDRV_SUCCESS;
}
