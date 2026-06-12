! Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
! HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: MIT

module hypredrive
   use, intrinsic :: iso_c_binding
   implicit none
   private

   integer, parameter, public :: hypredrv_error_kind = c_int32_t
   integer, parameter, public :: hypredrv_bigint_kind = c_int64_t
   integer, parameter, public :: hypredrv_real_kind = c_double
   integer(hypredrv_error_kind), parameter, public :: HYPREDRV_SUCCESS = 0_c_int32_t

   public :: HYPREDRV_Initialize, HYPREDRV_Finalize, HYPREDRV_ErrorCodeDescribe
   public :: HYPREDRV_ErrorInvalidValue, HYPREDRV_Create, HYPREDRV_Destroy
   public :: HYPREDRV_PrintLibInfo, HYPREDRV_PrintSystemInfo, HYPREDRV_PrintExitInfo
   public :: HYPREDRV_InputArgsParse, HYPREDRV_InputArgsParseYaml
   public :: HYPREDRV_SetLibraryMode, HYPREDRV_ObjectSetName
   public :: HYPREDRV_InputArgsGetWarmup, HYPREDRV_InputArgsGetNumRepetitions
   public :: HYPREDRV_InputArgsGetNumLinearSystems, HYPREDRV_InputArgsGetNumPreconVariants
   public :: HYPREDRV_InputArgsSetPreconVariant, HYPREDRV_InputArgsSetPreconPreset
   public :: HYPREDRV_InputArgsSetSolverPreset, HYPREDRV_SolverPresetRegister
   public :: HYPREDRV_PreconPresetRegister, HYPREDRV_LinearSystemBuild
   public :: HYPREDRV_LinearSystemReadMatrix, HYPREDRV_LinearSystemSetMatrix
   public :: HYPREDRV_LinearSystemSetRHS, HYPREDRV_LinearSystemSetMatrixFromCSR
   public :: HYPREDRV_LinearSystemSetRHSFromArray, HYPREDRV_LinearSystemSetInitialGuess
   public :: HYPREDRV_LinearSystemSetSolution, HYPREDRV_LinearSystemSetReferenceSolution
   public :: HYPREDRV_LinearSystemResetInitialGuess, HYPREDRV_LinearSystemSetPrecMatrix
   public :: HYPREDRV_LinearSystemSetDofmap, HYPREDRV_LinearSystemSetInterleavedDofmap
   public :: HYPREDRV_LinearSystemSetContiguousDofmap, HYPREDRV_LinearSystemReadDofmap
   public :: HYPREDRV_LinearSystemPrintDofmap, HYPREDRV_LinearSystemPrint
   public :: HYPREDRV_LinearSystemSetNearNullSpace, HYPREDRV_LinearSystemSetNullSpace
   public :: HYPREDRV_LinearSystemGetSolutionValues
   public :: HYPREDRV_LinearSystemGetSolutionLength, HYPREDRV_LinearSystemGetSolutionNorm
   public :: HYPREDRV_LinearSystemGetSolution, HYPREDRV_LinearSystemGetRHSValues
   public :: HYPREDRV_LinearSystemGetRHS, HYPREDRV_LinearSystemGetMatrix
   public :: HYPREDRV_StateVectorSet, HYPREDRV_StateVectorGetValues
   public :: HYPREDRV_StateVectorCopy, HYPREDRV_StateVectorUpdateAll
   public :: HYPREDRV_StateVectorApplyCorrection, HYPREDRV_PreconCreate
   public :: HYPREDRV_LinearSolverCreate, HYPREDRV_PreconSetup
   public :: HYPREDRV_LinearSolverSetup, HYPREDRV_LinearSolverApply
   public :: HYPREDRV_PreconApply, HYPREDRV_PreconDestroy, HYPREDRV_LinearSolverDestroy
   public :: HYPREDRV_StatsPrint, HYPREDRV_AnnotateBegin, HYPREDRV_AnnotateEnd
   public :: HYPREDRV_AnnotateLevelBegin, HYPREDRV_AnnotateLevelEnd
   public :: HYPREDRV_LinearSystemComputeEigenspectrum, HYPREDRV_LinearSolverGetNumIter
   public :: HYPREDRV_LinearSolverGetConverged
   public :: HYPREDRV_LinearSolverGetFinalRelativeResidualNorm
   public :: HYPREDRV_LinearSolverGetSetupTime, HYPREDRV_LinearSolverGetSolveTime
   public :: HYPREDRV_StatsLevelGetCount, HYPREDRV_StatsLevelGetEntry
   public :: HYPREDRV_StatsLevelPrint, HYPREDRV_BigIntSize, HYPREDRV_Check
   public :: HYPREDRV_ToCString

   interface
      function HYPREDRV_BigIntSize() bind(C, name="HYPREDRV_FortranBigIntSize") result(size)
         import :: c_size_t
         integer(c_size_t) :: size
      end function

      function HYPREDRV_Initialize() bind(C, name="HYPREDRV_Initialize") result(ierr)
         import :: c_int32_t
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_Finalize() bind(C, name="HYPREDRV_Finalize") result(ierr)
         import :: c_int32_t
         integer(c_int32_t) :: ierr
      end function

      subroutine HYPREDRV_ErrorCodeDescribe(error_code) bind(C, name="HYPREDRV_ErrorCodeDescribe")
         import :: c_int32_t
         integer(c_int32_t), value :: error_code
      end subroutine

      function HYPREDRV_ErrorInvalidValue(message) bind(C, name="HYPREDRV_ErrorInvalidValue") result(ierr)
         import :: c_char, c_int32_t
         character(kind=c_char), intent(in) :: message(*)
         integer(c_int32_t) :: ierr
      end function

      subroutine c_exit(status) bind(C, name="exit")
         import :: c_int
         integer(c_int), value :: status
      end subroutine c_exit

      function HYPREDRV_Create(fcomm, hypredrv) bind(C, name="HYPREDRV_FortranCreate") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         integer(c_int), value :: fcomm
         type(c_ptr) :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_Destroy(hypredrv) bind(C, name="HYPREDRV_Destroy") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), intent(inout) :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PrintLibInfo(fcomm, print_datetime) bind(C, name="HYPREDRV_FortranPrintLibInfo") result(ierr)
         import :: c_int, c_int32_t
         integer(c_int), value :: fcomm
         integer(c_int), value :: print_datetime
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PrintSystemInfo(fcomm) bind(C, name="HYPREDRV_FortranPrintSystemInfo") result(ierr)
         import :: c_int, c_int32_t
         integer(c_int), value :: fcomm
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PrintExitInfo(fcomm, argv0) bind(C, name="HYPREDRV_FortranPrintExitInfo") result(ierr)
         import :: c_char, c_int, c_int32_t
         integer(c_int), value :: fcomm
         character(kind=c_char), intent(in) :: argv0(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsParse(argc, argv, hypredrv) bind(C, name="HYPREDRV_InputArgsParse") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         integer(c_int), value :: argc
         type(c_ptr) :: argv(*)
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function c_HYPREDRV_InputArgsParseYaml(hypredrv, yaml_text) bind(C, name="HYPREDRV_FortranInputArgsParseYaml") result(ierr)
         import :: c_char, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: yaml_text(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_SetLibraryMode(hypredrv) bind(C, name="HYPREDRV_SetLibraryMode") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_ObjectSetName(hypredrv, name) bind(C, name="HYPREDRV_ObjectSetName") result(ierr)
         import :: c_char, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: name(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsGetWarmup(hypredrv, warmup) bind(C, name="HYPREDRV_InputArgsGetWarmup") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: warmup
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsGetNumRepetitions(hypredrv, num_reps) &
         bind(C, name="HYPREDRV_InputArgsGetNumRepetitions") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: num_reps
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsGetNumLinearSystems(hypredrv, num_ls) &
         bind(C, name="HYPREDRV_InputArgsGetNumLinearSystems") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: num_ls
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsGetNumPreconVariants(hypredrv, num_variants) &
         bind(C, name="HYPREDRV_InputArgsGetNumPreconVariants") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: num_variants
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsSetPreconVariant(hypredrv, variant_idx) &
         bind(C, name="HYPREDRV_InputArgsSetPreconVariant") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: variant_idx
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsSetPreconPreset(hypredrv, preset) bind(C, name="HYPREDRV_InputArgsSetPreconPreset") result(ierr)
         import :: c_char, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: preset(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_InputArgsSetSolverPreset(hypredrv, preset) bind(C, name="HYPREDRV_InputArgsSetSolverPreset") result(ierr)
         import :: c_char, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: preset(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_SolverPresetRegister(name, yaml_text, help) bind(C, name="HYPREDRV_SolverPresetRegister") result(ierr)
         import :: c_char, c_int32_t
         character(kind=c_char), intent(in) :: name(*), yaml_text(*), help(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PreconPresetRegister(name, yaml_text, help) bind(C, name="HYPREDRV_PreconPresetRegister") result(ierr)
         import :: c_char, c_int32_t
         character(kind=c_char), intent(in) :: name(*), yaml_text(*), help(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemBuild(hypredrv) bind(C, name="HYPREDRV_LinearSystemBuild") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemReadMatrix(hypredrv) bind(C, name="HYPREDRV_LinearSystemReadMatrix") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetMatrix(hypredrv, mat_A) bind(C, name="HYPREDRV_LinearSystemSetMatrix") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, mat_A
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetRHS(hypredrv, vec) bind(C, name="HYPREDRV_LinearSystemSetRHS") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, vec
         integer(c_int32_t) :: ierr
      end function

      function c_HYPREDRV_LinearSystemSetMatrixFromCSR(hypredrv, row_start, row_end, &
                                                       indptr, indptr_len, col_indices, &
                                                       col_indices_len, data, data_len) &
         bind(C, name="HYPREDRV_FortranLinearSystemSetMatrixFromCSR") result(ierr)
         import :: c_double, c_int32_t, c_int64_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int64_t), value :: row_start, row_end, indptr_len, col_indices_len, data_len
         integer(c_int64_t), intent(in) :: indptr(*), col_indices(*)
         real(c_double), intent(in) :: data(*)
         integer(c_int32_t) :: ierr
      end function

      function c_HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end, &
                                                      values, values_len) &
         bind(C, name="HYPREDRV_FortranLinearSystemSetRHSFromArray") result(ierr)
         import :: c_double, c_int32_t, c_int64_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int64_t), value :: row_start, row_end, values_len
         real(c_double), intent(in) :: values(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetInitialGuess(hypredrv, vec) bind(C, name="HYPREDRV_LinearSystemSetInitialGuess") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, vec
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetSolution(hypredrv, vec) bind(C, name="HYPREDRV_LinearSystemSetSolution") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, vec
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetReferenceSolution(hypredrv, vec) &
         bind(C, name="HYPREDRV_LinearSystemSetReferenceSolution") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, vec
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemResetInitialGuess(hypredrv) bind(C, name="HYPREDRV_LinearSystemResetInitialGuess") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetPrecMatrix(hypredrv, mat) bind(C, name="HYPREDRV_LinearSystemSetPrecMatrix") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, mat
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetDofmap(hypredrv, size, dofmap) bind(C, name="HYPREDRV_LinearSystemSetDofmap") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: size
         integer(c_int), intent(in) :: dofmap(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetInterleavedDofmap(hypredrv, num_local_blocks, &
                                                         num_dof_types) &
         bind(C, name="HYPREDRV_LinearSystemSetInterleavedDofmap") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: num_local_blocks, num_dof_types
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetContiguousDofmap(hypredrv, num_local_blocks, &
                                                        num_dof_types) &
         bind(C, name="HYPREDRV_LinearSystemSetContiguousDofmap") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: num_local_blocks, num_dof_types
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemReadDofmap(hypredrv) bind(C, name="HYPREDRV_LinearSystemReadDofmap") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemPrintDofmap(hypredrv, filename) bind(C, name="HYPREDRV_LinearSystemPrintDofmap") result(ierr)
         import :: c_char, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: filename(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemPrint(hypredrv) bind(C, name="HYPREDRV_LinearSystemPrint") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetNearNullSpace(hypredrv, num_entries, &
                                                     num_components, values) &
         bind(C, name="HYPREDRV_LinearSystemSetNearNullSpace") result(ierr)
         import :: c_double, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: num_entries, num_components
         real(c_double), intent(in) :: values(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemSetNullSpace(hypredrv, num_entries, &
                                                 num_components, values) &
         bind(C, name="HYPREDRV_LinearSystemSetNullSpace") result(ierr)
         import :: c_double, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: num_entries, num_components
         real(c_double), intent(in) :: values(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetSolutionValues(hypredrv, sol_data) &
         bind(C, name="HYPREDRV_LinearSystemGetSolutionValues") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         type(c_ptr) :: sol_data
         integer(c_int32_t) :: ierr
      end function

      function c_HYPREDRV_LinearSystemGetSolutionLength(hypredrv, length) &
         bind(C, name="HYPREDRV_FortranLinearSystemGetSolutionLength") result(ierr)
         import :: c_int32_t, c_int64_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int64_t) :: length
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetSolutionNorm(hypredrv, norm_type, norm) &
         bind(C, name="HYPREDRV_LinearSystemGetSolutionNorm") result(ierr)
         import :: c_char, c_double, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: norm_type(*)
         real(c_double) :: norm
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetSolution(hypredrv, vec) bind(C, name="HYPREDRV_LinearSystemGetSolution") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         type(c_ptr) :: vec
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetRHSValues(hypredrv, rhs_data) bind(C, name="HYPREDRV_LinearSystemGetRHSValues") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         type(c_ptr) :: rhs_data
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetRHS(hypredrv, vec) bind(C, name="HYPREDRV_LinearSystemGetRHS") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         type(c_ptr) :: vec
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemGetMatrix(hypredrv, mat) bind(C, name="HYPREDRV_LinearSystemGetMatrix") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         type(c_ptr) :: mat
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StateVectorSet(hypredrv, nstates, vecs) bind(C, name="HYPREDRV_StateVectorSet") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: nstates
         type(c_ptr), intent(in) :: vecs(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StateVectorGetValues(hypredrv, index, data_ptr) bind(C, name="HYPREDRV_StateVectorGetValues") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: index
         type(c_ptr) :: data_ptr
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StateVectorCopy(hypredrv, index_in, index_out) bind(C, name="HYPREDRV_StateVectorCopy") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: index_in, index_out
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StateVectorUpdateAll(hypredrv) bind(C, name="HYPREDRV_StateVectorUpdateAll") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StateVectorApplyCorrection(hypredrv, state_idx) &
         bind(C, name="HYPREDRV_StateVectorApplyCorrection") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: state_idx
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PreconCreate(hypredrv) bind(C, name="HYPREDRV_PreconCreate") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverCreate(hypredrv) bind(C, name="HYPREDRV_LinearSolverCreate") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PreconSetup(hypredrv) bind(C, name="HYPREDRV_PreconSetup") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverSetup(hypredrv) bind(C, name="HYPREDRV_LinearSolverSetup") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverApply(hypredrv) bind(C, name="HYPREDRV_LinearSolverApply") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PreconApply(hypredrv, vec_b, vec_x) bind(C, name="HYPREDRV_PreconApply") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv, vec_b, vec_x
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_PreconDestroy(hypredrv) bind(C, name="HYPREDRV_PreconDestroy") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverDestroy(hypredrv) bind(C, name="HYPREDRV_LinearSolverDestroy") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StatsPrint(hypredrv) bind(C, name="HYPREDRV_StatsPrint") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_AnnotateBegin(hypredrv, name, id) bind(C, name="HYPREDRV_AnnotateBegin") result(ierr)
         import :: c_char, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: name(*)
         integer(c_int), value :: id
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_AnnotateEnd(hypredrv, name, id) bind(C, name="HYPREDRV_AnnotateEnd") result(ierr)
         import :: c_char, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         character(kind=c_char), intent(in) :: name(*)
         integer(c_int), value :: id
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_AnnotateLevelBegin(hypredrv, level, name, id) bind(C, name="HYPREDRV_AnnotateLevelBegin") result(ierr)
         import :: c_char, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: level, id
         character(kind=c_char), intent(in) :: name(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_AnnotateLevelEnd(hypredrv, level, name, id) bind(C, name="HYPREDRV_AnnotateLevelEnd") result(ierr)
         import :: c_char, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: level, id
         character(kind=c_char), intent(in) :: name(*)
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSystemComputeEigenspectrum(hypredrv) &
         bind(C, name="HYPREDRV_LinearSystemComputeEigenspectrum") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverGetNumIter(hypredrv, iters) bind(C, name="HYPREDRV_LinearSolverGetNumIter") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: iters
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverGetConverged(hypredrv, converged) &
         bind(C, name="HYPREDRV_LinearSolverGetConverged") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int) :: converged
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(hypredrv, norm) &
         bind(C, name="HYPREDRV_LinearSolverGetFinalRelativeResidualNorm") result(ierr)
         import :: c_double, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         real(c_double) :: norm
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverGetSetupTime(hypredrv, seconds) bind(C, name="HYPREDRV_LinearSolverGetSetupTime") result(ierr)
         import :: c_double, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         real(c_double) :: seconds
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_LinearSolverGetSolveTime(hypredrv, seconds) bind(C, name="HYPREDRV_LinearSolverGetSolveTime") result(ierr)
         import :: c_double, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         real(c_double) :: seconds
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StatsLevelGetCount(hypredrv, level, count) bind(C, name="HYPREDRV_StatsLevelGetCount") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: level
         integer(c_int) :: count
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StatsLevelGetEntry(hypredrv, level, index, entry_id, &
                                           num_solves, linear_iters, setup_time, solve_time) &
         bind(C, name="HYPREDRV_StatsLevelGetEntry") result(ierr)
         import :: c_double, c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: level, index
         integer(c_int) :: entry_id, num_solves, linear_iters
         real(c_double) :: setup_time, solve_time
         integer(c_int32_t) :: ierr
      end function

      function HYPREDRV_StatsLevelPrint(hypredrv, level) bind(C, name="HYPREDRV_StatsLevelPrint") result(ierr)
         import :: c_int, c_int32_t, c_ptr
         type(c_ptr), value :: hypredrv
         integer(c_int), value :: level
         integer(c_int32_t) :: ierr
      end function
   end interface

contains

   subroutine HYPREDRV_ToCString(text, c_text)
      character(len=*), intent(in) :: text
      character(kind=c_char), allocatable, intent(out) :: c_text(:)
      integer :: i, n
      n = len(text)
      allocate (character(kind=c_char) :: c_text(n + 1))
      do i = 1, n
         c_text(i) = text(i:i)
      end do
      c_text(n + 1) = c_null_char
   end subroutine HYPREDRV_ToCString

   function HYPREDRV_InputArgsParseYaml(hypredrv, yaml_text) result(ierr)
      type(c_ptr), value :: hypredrv
      character(len=*), intent(in) :: yaml_text
      integer(c_int32_t) :: ierr
      character(kind=c_char), allocatable :: c_yaml(:)

      call HYPREDRV_ToCString(yaml_text, c_yaml)
      ierr = c_HYPREDRV_InputArgsParseYaml(hypredrv, c_yaml)
   end function HYPREDRV_InputArgsParseYaml

   ! These wrappers intentionally avoid the Fortran 2008 CONTIGUOUS attribute so
   ! the public module remains Fortran 2003-compatible. For strided actual
   ! arguments, compilers may pass a packed temporary; callers should pass
   ! contiguous allocatable arrays for large systems to avoid that copy.
   function HYPREDRV_LinearSystemSetMatrixFromCSR(hypredrv, row_start, row_end, indptr, col_indices, data, nnz) result(ierr)
      type(c_ptr), value :: hypredrv
      integer(c_int64_t), value :: row_start, row_end
      integer(c_int64_t), intent(in) :: indptr(:), col_indices(:)
      real(c_double), intent(in) :: data(:)
      integer(c_int64_t), intent(in), optional :: nnz
      integer(c_int32_t) :: ierr
      integer(c_int64_t) :: data_len

      data_len = int(size(data), c_int64_t)
      if (present(nnz)) data_len = nnz
      if (data_len < 0_c_int64_t .or. data_len > int(size(col_indices), c_int64_t) .or. &
          data_len > int(size(data), c_int64_t)) then
         ierr = HYPREDRV_ErrorInvalidValue('Fortran CSR nnz exceeds col_indices/data length'//c_null_char)
         return
      end if
      ierr = c_HYPREDRV_LinearSystemSetMatrixFromCSR(hypredrv, row_start, row_end, &
                                                     indptr, int(size(indptr), c_int64_t), &
                                                     col_indices, data_len, data, data_len)
   end function HYPREDRV_LinearSystemSetMatrixFromCSR

   function HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end, values) result(ierr)
      type(c_ptr), value :: hypredrv
      integer(c_int64_t), value :: row_start, row_end
      real(c_double), intent(in) :: values(:)
      integer(c_int32_t) :: ierr

      ierr = c_HYPREDRV_LinearSystemSetRHSFromArray(hypredrv, row_start, row_end, &
                                                    values, int(size(values), c_int64_t))
   end function HYPREDRV_LinearSystemSetRHSFromArray

   function HYPREDRV_LinearSystemGetSolutionLength(hypredrv, length) result(ierr)
      type(c_ptr), value :: hypredrv
      integer(c_int64_t), intent(out) :: length
      integer(c_int32_t) :: ierr

      ierr = c_HYPREDRV_LinearSystemGetSolutionLength(hypredrv, length)
   end function HYPREDRV_LinearSystemGetSolutionLength

   subroutine HYPREDRV_Check(ierr, message)
      integer(c_int32_t), intent(in) :: ierr
      character(len=*), intent(in), optional :: message
      character(len=32) :: rank_text
      character(len=48) :: prefix
      integer :: env_status

      if (ierr == HYPREDRV_SUCCESS) return
      prefix = ''
      call get_environment_variable('PMI_RANK', rank_text, status=env_status)
      if (env_status /= 0) call get_environment_variable('OMPI_COMM_WORLD_RANK', rank_text, status=env_status)
      if (env_status == 0) prefix = '[rank '//trim(rank_text)//']'
      if (present(message)) then
         if (len_trim(prefix) > 0) then
            write (*, '(a,1x,a)') trim(prefix), trim(message)
         else
            write (*, '(a)') trim(message)
         end if
      end if
      call HYPREDRV_ErrorCodeDescribe(ierr)
      if (len_trim(prefix) > 0) then
         write (*, '(a,1x,a)') trim(prefix), 'HYPREDRV call failed'
      else
         write (*, '(a)') 'HYPREDRV call failed'
      end if
      call c_exit(1_c_int)
   end subroutine HYPREDRV_Check

end module hypredrive
