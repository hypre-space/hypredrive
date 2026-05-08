# cython: language_level=3
"""C declarations for the subset of HYPRE / hypredrive used by the Python binding.

We deliberately keep this surface narrow: only the public entry points needed
to drive a single end-to-end solve from Python. Anything more is pulled in
behind explicit feature work.
"""

from libc.stdint cimport uint32_t


cdef extern from "mpi.h" nogil:
    ctypedef struct ompi_communicator_t   # opaque; OpenMPI uses pointer-comm
    ctypedef void *MPI_Comm
    MPI_Comm MPI_COMM_SELF
    MPI_Comm MPI_COMM_WORLD


cdef extern from "HYPRE_utilities.h" nogil:
    ctypedef int       HYPRE_Int
    ctypedef long long HYPRE_BigInt   # may be int / long long depending on build
    ctypedef double    HYPRE_Real
    ctypedef double    HYPRE_Complex


cdef extern from "HYPRE.h" nogil:
    ctypedef void *HYPRE_Vector
    ctypedef void *HYPRE_Matrix


cdef extern from "HYPREDRV.h" nogil:
    ctypedef struct hypredrv_struct
    ctypedef hypredrv_struct *HYPREDRV_t

    uint32_t HYPREDRV_Initialize()
    uint32_t HYPREDRV_Finalize()

    uint32_t HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *hypredrv_ptr)
    uint32_t HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
    uint32_t HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_InputArgsParse(int argc, char **argv,
                                     HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_LinearSystemSetMatrixFromCSR(
        HYPREDRV_t hypredrv,
        HYPRE_BigInt row_start,
        HYPRE_BigInt row_end,
        const HYPRE_BigInt *indptr,
        const HYPRE_BigInt *col_indices,
        const HYPRE_Real   *data)

    uint32_t HYPREDRV_LinearSystemSetRHSFromArray(
        HYPREDRV_t hypredrv,
        HYPRE_BigInt row_start,
        HYPRE_BigInt row_end,
        const HYPRE_Real *values)

    uint32_t HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv,
                                                  HYPRE_Vector vec)
    uint32_t HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv,
                                                    HYPRE_Complex **sol_data)
    uint32_t HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv,
                                                  const char *norm_type,
                                                  double *norm)

    uint32_t HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)

    void HYPREDRV_ErrorCodeDescribe(uint32_t error_code)
