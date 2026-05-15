/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_PYTHON_HEADER
#define HYPREDRV_PYTHON_HEADER

#include <stddef.h>
#include <stdint.h>

#include "HYPREDRV.h"

#ifdef __cplusplus
extern "C"
{
#endif

   /**
    * @brief Return the runtime size of HYPRE_BigInt.
    *
    * The Python extension uses this value to select the matching NumPy integer
    * dtype at import time instead of assuming a particular HYPRE integer ABI.
    */
   size_t HYPREDRV_PythonIndexSize(void);

   /**
    * @brief Return the runtime size of HYPRE_Real.
    *
    * The Python extension uses this value to select the matching NumPy floating
    * dtype at import time.
    */
   size_t HYPREDRV_PythonRealSize(void);

   /**
    * @brief Return the runtime size of HYPRE_Complex.
    *
    * This is used to reject unsupported complex HYPRE builds before a solve is
    * attempted.
    */
   size_t HYPREDRV_PythonSolutionEntrySize(void);

   /**
    * @brief Initialize MPI for Python serial use when no owner has done so.
    *
    * If MPI is already initialized (for example by mpi4py), this is a no-op.
    * If this bridge initializes MPI, HYPREDRV_PythonMPIFinalize() will finalize
    * it later.
    */
   uint32_t HYPREDRV_PythonMPIInitialize(void);

   /**
    * @brief Finalize MPI only if this bridge initialized it.
    */
   uint32_t HYPREDRV_PythonMPIFinalize(void);

   /**
    * @brief Create a HYPREDRV object on MPI_COMM_SELF for serial Python use.
    */
   uint32_t HYPREDRV_PythonCreateWithSelf(HYPREDRV_t *hypredrv_ptr);

   /**
    * @brief Create a HYPREDRV object from a Fortran MPI communicator handle.
    *
    * mpi4py exposes communicators through Comm.py2f(); this bridge converts
    * that handle back to a C MPI_Comm before calling HYPREDRV_Create().
    */
   uint32_t HYPREDRV_PythonCreateFromFortranComm(int fortran_comm,
                                                 HYPREDRV_t *hypredrv_ptr);

   /**
    * @brief Create a HYPREDRV object on MPI_COMM_WORLD.
    */
   uint32_t HYPREDRV_PythonCreateWithWorld(HYPREDRV_t *hypredrv_ptr);

   /**
    * @brief Install a CSR matrix from Python-owned opaque buffers.
    *
    * @param hypredrv    HYPREDRV object.
    * @param row_start   Inclusive global first row as a fixed-width Python ABI value.
    * @param row_end     Inclusive global last row as a fixed-width Python ABI value.
    * @param indptr      Buffer with HYPRE_BigInt entries.
    * @param col_indices Buffer with HYPRE_BigInt entries.
    * @param data        Buffer with HYPRE_Real entries.
    *
    * @return HYPREDRV error code. Out-of-range row bounds are rejected before
    * casting to HYPRE_BigInt.
    */
   uint32_t HYPREDRV_PythonSetMatrixFromCSR(HYPREDRV_t hypredrv, int64_t row_start,
                                            int64_t row_end, const void *indptr,
                                            const void *col_indices,
                                            const void *data);

   /**
    * @brief Install a right-hand-side vector from a Python-owned opaque buffer.
    *
    * @return HYPREDRV error code. Out-of-range row bounds are rejected before
    * casting to HYPRE_BigInt.
    */
   uint32_t HYPREDRV_PythonSetRHSFromArray(HYPREDRV_t hypredrv, int64_t row_start,
                                           int64_t row_end, const void *values);

   /**
    * @brief Return an opaque pointer to the current solution values.
    *
    * The returned pointer is owned by the HYPREDRV object and remains valid
    * until the solution vector is replaced or the object is destroyed.
    * @param sol_length Output local solution length in scalar entries.
    */
   uint32_t HYPREDRV_PythonGetSolutionValues(HYPREDRV_t hypredrv,
                                             const void **sol_data,
                                             size_t *sol_length);

#ifdef __cplusplus
}
#endif

#endif /* HYPREDRV_PYTHON_HEADER */
