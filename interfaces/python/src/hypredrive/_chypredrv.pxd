# cython: language_level=3
"""C declarations for the subset of HYPRE / hypredrive used by the Python binding.

We deliberately keep this surface narrow: only the public entry points needed
to drive a single end-to-end solve from Python. Anything more is pulled in
behind explicit feature work.
"""

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint32_t, uintptr_t


cdef extern from "HYPRE.h" nogil:
    ctypedef void *HYPRE_Vector
    ctypedef void *HYPRE_Matrix
    ctypedef void *HYPRE_IJVector
    ctypedef double HYPRE_Complex
    ctypedef int HYPRE_MemoryLocation
    cdef enum:
        HYPRE_MEMORY_HOST


cdef extern from "HYPREDRV.h" nogil:
    ctypedef struct hypredrv_struct
    ctypedef hypredrv_struct *HYPREDRV_t

    uint32_t HYPREDRV_Initialize()
    uint32_t HYPREDRV_Finalize()

    uint32_t HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
    uint32_t HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_InputArgsParse(int argc, char **argv,
                                     HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv,
                                                  HYPRE_Vector vec)
    uint32_t HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv,
                                            int size,
                                            const int *dofmap)
    uint32_t HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv,
                                                  const char *norm_type,
                                                  double *norm)

    uint32_t HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverSetRelativeTol(HYPREDRV_t hypredrv, double tol)
    uint32_t HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSolverGetNumIter(HYPREDRV_t hypredrv, int *iters)
    uint32_t HYPREDRV_LinearSolverGetConverged(HYPREDRV_t hypredrv,
                                               int *converged)
    uint32_t HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(
        HYPREDRV_t hypredrv, double *norm)
    uint32_t HYPREDRV_LinearSolverGetSetupTime(HYPREDRV_t hypredrv,
                                               double *seconds)
    uint32_t HYPREDRV_LinearSolverGetSolveTime(HYPREDRV_t hypredrv,
                                               double *seconds)

    uint32_t HYPREDRV_AnnotateBegin(HYPREDRV_t hypredrv, const char *name,
                                    int id)
    uint32_t HYPREDRV_AnnotateEnd(HYPREDRV_t hypredrv, const char *name,
                                  int id)

    void HYPREDRV_ErrorCodeDescribe(uint32_t error_code)
    void HYPREDRV_ErrorCodeClear()
    uint32_t HYPREDRV_ErrorInvalidValue(const char *message)


    uint32_t HYPREDRV_ObjectSetName(HYPREDRV_t hypredrv, const char *name)
    uint32_t HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv, int *warmup)
    uint32_t HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv,
                                                 int *num_reps)
    uint32_t HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv,
                                                   int *num_ls)
    uint32_t HYPREDRV_InputArgsGetNumPreconVariants(HYPREDRV_t hypredrv,
                                                    int *num_variants)
    uint32_t HYPREDRV_InputArgsSetPreconVariant(HYPREDRV_t hypredrv,
                                                int variant_idx)
    uint32_t HYPREDRV_InputArgsSetPreconPreset(HYPREDRV_t hypredrv,
                                               const char *preset)
    uint32_t HYPREDRV_InputArgsSetSolverPreset(HYPREDRV_t hypredrv,
                                               const char *preset)
    uint32_t HYPREDRV_SolverPresetRegister(const char *name,
                                           const char *yaml_text,
                                           const char *help)
    uint32_t HYPREDRV_PreconPresetRegister(const char *name,
                                           const char *yaml_text,
                                           const char *help)

    uint32_t HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv,
                                            HYPRE_Matrix mat_A)
    uint32_t HYPREDRV_LinearSystemSetDiscreteGradient(HYPREDRV_t hypredrv,
                                                      HYPRE_Matrix G)
    uint32_t HYPREDRV_LinearSystemSetDiscreteCurl(HYPREDRV_t hypredrv,
                                                  HYPRE_Matrix C)
    uint32_t HYPREDRV_LinearSystemSetCoordinates(HYPREDRV_t hypredrv,
                                                 HYPRE_Vector x,
                                                 HYPRE_Vector y,
                                                 HYPRE_Vector z)
    uint32_t HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
    uint32_t HYPREDRV_LinearSystemSetSolution(HYPREDRV_t hypredrv,
                                              HYPRE_Vector vec)
    uint32_t HYPREDRV_LinearSystemSetReferenceSolution(HYPREDRV_t hypredrv,
                                                       HYPRE_Vector vec)
    uint32_t HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv,
                                                HYPRE_Matrix mat)
    uint32_t HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv,
                                                       int num_local_blocks,
                                                       int num_dof_types)
    uint32_t HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv,
                                                      int num_local_blocks,
                                                      int num_dof_types)
    uint32_t HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t hypredrv,
                                              const char *filename)
    uint32_t HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv,
                                                   int num_entries,
                                                   int num_components,
                                                   const HYPRE_Complex *values)
    uint32_t HYPREDRV_LinearSystemSetNullSpace(HYPREDRV_t hypredrv,
                                               int num_entries,
                                               int num_components,
                                               const HYPRE_Complex *values)
    uint32_t HYPREDRV_LinearSystemSetCoarseSchur(HYPREDRV_t hypredrv, int level,
                                                 HYPRE_Matrix mat_S)
    uint32_t HYPREDRV_LinearSystemGetSolutionValues(
        HYPREDRV_t hypredrv, HYPRE_Complex **sol_data,
        HYPRE_MemoryLocation *memory_location)
    uint32_t HYPREDRV_LinearSystemGetSolution(HYPREDRV_t hypredrv,
                                              HYPRE_Vector *vec)
    uint32_t HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv,
                                               HYPRE_Complex **rhs_data)
    uint32_t HYPREDRV_LinearSystemGetRHS(HYPREDRV_t hypredrv, HYPRE_Vector *vec)
    uint32_t HYPREDRV_LinearSystemGetMatrix(HYPREDRV_t hypredrv,
                                            HYPRE_Matrix *mat)
    uint32_t HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates,
                                     HYPRE_IJVector *vecs)
    uint32_t HYPREDRV_StateVectorGetValues(HYPREDRV_t hypredrv, int index,
                                           HYPRE_Complex **data_ptr)
    uint32_t HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv, int index_in,
                                      int index_out)
    uint32_t HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv,
                                                 int state_idx)

    uint32_t HYPREDRV_PreconCreate(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_PreconSetup(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b,
                                  HYPRE_Vector vec_x)
    uint32_t HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)

    uint32_t HYPREDRV_StatsPrint(HYPREDRV_t hypredrv)
    uint32_t HYPREDRV_AnnotateLevelBegin(HYPREDRV_t hypredrv, int level,
                                         const char *name, int id)
    uint32_t HYPREDRV_AnnotateLevelEnd(HYPREDRV_t hypredrv, int level,
                                       const char *name, int id)
    uint32_t HYPREDRV_StatsLevelGetCount(HYPREDRV_t hypredrv, int level,
                                         int *count)
    uint32_t HYPREDRV_StatsLevelGetEntry(HYPREDRV_t hypredrv, int level,
                                         int index, int *entry_id,
                                         int *num_solves, int *linear_iters,
                                         double *setup_time, double *solve_time)
    uint32_t HYPREDRV_StatsLevelPrint(HYPREDRV_t hypredrv, int level)


cdef extern from "HYPREDRV_python.h" nogil:
    uint32_t HYPREDRV_PythonPrintLibInfo(int fortran_comm, int print_datetime)
    uint32_t HYPREDRV_PythonPrintSystemInfo(int fortran_comm)
    uint32_t HYPREDRV_PythonPrintExitInfo(int fortran_comm, const char *argv0)
    size_t HYPREDRV_PythonIndexSize()
    size_t HYPREDRV_PythonRealSize()
    size_t HYPREDRV_PythonSolutionEntrySize()
    uint32_t HYPREDRV_PythonMPIInitialize()
    uint32_t HYPREDRV_PythonMPIFinalize()

    uint32_t HYPREDRV_PythonCreateWithSelf(HYPREDRV_t *hypredrv_ptr)
    uint32_t HYPREDRV_PythonCreateFromFortranComm(
        int fortran_comm,
        HYPREDRV_t *hypredrv_ptr)

    uint32_t HYPREDRV_PythonSetMatrixFromCSR(
        HYPREDRV_t hypredrv,
        int64_t row_start,
        int64_t row_end,
        const void *indptr,
        const void *col_indices,
        const void *data)

    uint32_t HYPREDRV_PythonSetRHSFromArray(
        HYPREDRV_t hypredrv,
        int64_t row_start,
        int64_t row_end,
        const void *values)

    uint32_t HYPREDRV_PythonCopySolutionValues(
        HYPREDRV_t hypredrv,
        void *sol_data,
        size_t sol_length)

    uint32_t HYPREDRV_PythonGetSolutionLength(HYPREDRV_t hypredrv, size_t *sol_length)
