#ifndef HYPREDRIVE_PYTHON_NATIVE_ABI_H
#define HYPREDRIVE_PYTHON_NATIVE_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "HYPREDRV.h"
#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "mpi.h"

/*
 * Cython needs concrete base types for C typedef declarations, but HYPRE
 * allows several of those typedefs to vary by build configuration. Keep the
 * ABI-sensitive casts in C, compiled against the actual HYPRE headers, and
 * expose only fixed-width / opaque-buffer entry points to Cython.
 */

static inline size_t
hypredrive_PythonIndexSize(void)
{
    return sizeof(HYPRE_BigInt);
}

static inline size_t
hypredrive_PythonRealSize(void)
{
    return sizeof(HYPRE_Real);
}

static inline size_t
hypredrive_PythonSolutionEntrySize(void)
{
    return sizeof(HYPRE_Complex);
}

static inline uint32_t
hypredrive_PythonCreateWithSelf(HYPREDRV_t *hypredrv_ptr)
{
    return HYPREDRV_Create(MPI_COMM_SELF, hypredrv_ptr);
}

static inline uint32_t
hypredrive_PythonCreateFromFortranComm(int fortran_comm,
                                       HYPREDRV_t *hypredrv_ptr)
{
    return HYPREDRV_Create(MPI_Comm_f2c((MPI_Fint) fortran_comm), hypredrv_ptr);
}

static inline uint32_t
hypredrive_PythonCreateWithWorld(HYPREDRV_t *hypredrv_ptr)
{
    return HYPREDRV_Create(MPI_COMM_WORLD, hypredrv_ptr);
}

static inline uint32_t
hypredrive_PythonSetMatrixFromCSR(HYPREDRV_t hypredrv,
                                  int64_t row_start,
                                  int64_t row_end,
                                  const void *indptr,
                                  const void *col_indices,
                                  const void *data)
{
    return HYPREDRV_LinearSystemSetMatrixFromCSR(
        hypredrv,
        (HYPRE_BigInt) row_start,
        (HYPRE_BigInt) row_end,
        (const HYPRE_BigInt *) indptr,
        (const HYPRE_BigInt *) col_indices,
        (const HYPRE_Real *) data);
}

static inline uint32_t
hypredrive_PythonSetRHSFromArray(HYPREDRV_t hypredrv,
                                 int64_t row_start,
                                 int64_t row_end,
                                 const void *values)
{
    return HYPREDRV_LinearSystemSetRHSFromArray(
        hypredrv,
        (HYPRE_BigInt) row_start,
        (HYPRE_BigInt) row_end,
        (const HYPRE_Real *) values);
}

static inline uint32_t
hypredrive_PythonGetSolutionValues(HYPREDRV_t hypredrv,
                                   const void **sol_data)
{
    HYPRE_Complex *src = NULL;
    uint32_t code = HYPREDRV_LinearSystemGetSolutionValues(hypredrv, &src);
    *sol_data = (const void *) src;
    return code;
}

#endif /* HYPREDRIVE_PYTHON_NATIVE_ABI_H */
