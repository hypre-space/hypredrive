program test_csr_mpi
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   interface
      function HYPREDRV_FortranTestCreateVector(fcomm, row_start, row_end, value, vec) &
         bind(C, name="HYPREDRV_FortranTestCreateVector") result(ierr)
         import :: c_double, c_int, c_int32_t, c_int64_t, c_ptr
         integer(c_int), value :: fcomm
         integer(c_int64_t), value :: row_start, row_end
         real(c_double), value :: value
         type(c_ptr) :: vec
         integer(c_int32_t) :: ierr
      end function HYPREDRV_FortranTestCreateVector

      function HYPREDRV_FortranTestDestroyVector(vec) &
         bind(C, name="HYPREDRV_FortranTestDestroyVector") result(ierr)
         import :: c_int32_t, c_ptr
         type(c_ptr), value :: vec
         integer(c_int32_t) :: ierr
      end function HYPREDRV_FortranTestDestroyVector
   end interface

   integer, parameter :: n = 8
   integer :: ierr, rank, nproc, i, local_n, pos
   integer(c_int) :: iters, converged, stats_count, stats_entry, stats_solves, stats_iters
   integer(c_int64_t) :: row_start, row_end, global_row, solution_len
   integer(c_int64_t), allocatable :: indptr(:), cols(:), empty_indptr(:)
   integer(c_int64_t), allocatable :: empty_cols(:), bad_indptr(:), bad_cols(:)
   integer(c_int), allocatable :: dofmap(:)
   real(c_double), allocatable :: data(:), rhs(:), empty_data(:), bad_data(:), overflow_rhs(:), near_null(:)
   real(c_double) :: norm, final_res_norm, setup_time, solve_time, stats_setup_time, stats_solve_time
   type(c_ptr) :: drv, mat, rhs_vec, sol_vec, rhs_values, sol_values, state_values
   type(c_ptr) :: state_vecs(2)
   character(len=:), allocatable :: yaml

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)

   row_start = int((rank*n)/nproc, c_int64_t)
   row_end = int(((rank + 1)*n)/nproc - 1, c_int64_t)
   local_n = int(row_end - row_start + 1, kind=kind(local_n))
   if (local_n <= 0) stop 'test requires at least one row per rank'

   allocate (indptr(local_n + 1), cols(3*local_n), data(3*local_n), rhs(local_n))
   pos = 0
   indptr(1) = 0_c_int64_t
   do i = 1, local_n
      global_row = row_start + int(i - 1, c_int64_t)
      if (global_row > 0_c_int64_t) then
         pos = pos + 1
         cols(pos) = global_row - 1_c_int64_t
         data(pos) = -1.0_c_double
      end if
      pos = pos + 1
      cols(pos) = global_row
      data(pos) = 2.0_c_double
      if (global_row < int(n - 1, c_int64_t)) then
         pos = pos + 1
         cols(pos) = global_row + 1_c_int64_t
         data(pos) = -1.0_c_double
      end if
      indptr(i + 1) = int(pos, c_int64_t)
      rhs(i) = 1.0_c_double
   end do

   yaml = 'solver:'//new_line('a')// &
          '  pcg:'//new_line('a')// &
          '    max_iter: 100'//new_line('a')// &
          '    relative_tol: 1.0e-8'//new_line('a')// &
          'preconditioner:'//new_line('a')// &
          '  amg:'//new_line('a')// &
          '    max_iter: 1'//new_line('a')// &
          '    tolerance: 0.0'//new_line('a')

   call HYPREDRV_Check(HYPREDRV_Initialize())
   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
   call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml))
   call HYPREDRV_Check(HYPREDRV_SetLibraryMode(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols(:pos), data(:pos)))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end, rhs))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetPrecMatrix(drv, c_null_ptr))
   allocate (dofmap(local_n))
   dofmap = 0_c_int
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetInterleavedDofmap(drv, int(local_n, c_int), 1_c_int))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetContiguousDofmap(drv, int(local_n, c_int), 1_c_int))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetDofmap(drv, int(local_n, c_int), dofmap))
   call HYPREDRV_Check(HYPREDRV_LinearSystemPrintDofmap(drv, 'fortran_test_dofmap'//c_null_char))
   allocate (near_null(local_n))
   near_null = 1.0_c_double
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetNearNullSpace(drv, int(local_n, c_int), 1_c_int, near_null))
   deallocate (near_null)
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetInitialGuess(drv, c_null_ptr))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetReferenceSolution(drv, c_null_ptr))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetMatrix(drv, mat))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetRHS(drv, rhs_vec))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetRHSValues(drv, rhs_values))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolution(drv, sol_vec))
   if (.not. c_associated(mat) .or. .not. c_associated(rhs_vec) .or. .not. c_associated(rhs_values) .or. &
       .not. c_associated(sol_vec)) then
      stop 'invalid matrix/vector accessor pointer'
   end if
   call HYPREDRV_Check(HYPREDRV_FortranTestCreateVector(int(MPI_COMM_WORLD, c_int), &
                                                        row_start, row_end, 1.0_c_double, state_vecs(1)))
   call HYPREDRV_Check(HYPREDRV_FortranTestCreateVector(int(MPI_COMM_WORLD, c_int), &
                                                        row_start, row_end, 0.0_c_double, state_vecs(2)))
   call HYPREDRV_Check(HYPREDRV_StateVectorSet(drv, 2_c_int, state_vecs))
   call HYPREDRV_Check(HYPREDRV_StateVectorGetValues(drv, 0_c_int, state_values))
   if (.not. c_associated(state_values)) stop 'invalid state-vector values pointer'
   call HYPREDRV_Check(HYPREDRV_StateVectorCopy(drv, 0_c_int, 1_c_int))
   call HYPREDRV_Check(HYPREDRV_StateVectorUpdateAll(drv))
   call HYPREDRV_Check(HYPREDRV_PreconCreate(drv))
   call HYPREDRV_Check(HYPREDRV_PreconSetup(drv))
   call HYPREDRV_Check(HYPREDRV_PreconApply(drv, rhs_vec, sol_vec))
   call HYPREDRV_Check(HYPREDRV_PreconDestroy(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverCreate(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverSetup(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverApply(drv))
   call HYPREDRV_Check(HYPREDRV_StateVectorApplyCorrection(drv, 0_c_int))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolution(drv, sol_vec))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionValues(drv, sol_values))
   if (.not. c_associated(sol_vec) .or. .not. c_associated(sol_values)) stop 'invalid solution accessor pointer'
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionNorm(drv, 'l2'//c_null_char, norm))
   if (norm <= 0.0_c_double) stop 'invalid solution norm'
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetNumIter(drv, iters))
   if (iters <= 0) stop 'invalid iteration count'
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetConverged(drv, converged))
   if (converged /= 1) stop 'solver did not converge'
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(drv, final_res_norm))
   if (final_res_norm < 0.0_c_double) stop 'invalid final relative residual norm'
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetSetupTime(drv, setup_time))
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetSolveTime(drv, solve_time))
   if (setup_time < 0.0_c_double .or. solve_time < 0.0_c_double) stop 'invalid solver timing'
   call HYPREDRV_Check(HYPREDRV_StatsLevelGetCount(drv, 0_c_int, stats_count))
   if (stats_count > 0) then
      call HYPREDRV_Check(HYPREDRV_StatsLevelGetEntry(drv, 0_c_int, 0_c_int, stats_entry, &
                                                      stats_solves, stats_iters, stats_setup_time, stats_solve_time))
      if (stats_entry < 0 .or. stats_solves < 0 .or. stats_iters < 0) stop 'invalid stats entry'
   end if
   if (rank == 0) call HYPREDRV_Check(HYPREDRV_StatsLevelPrint(drv, 0_c_int))
   if (rank == 0) call HYPREDRV_Check(HYPREDRV_StatsPrint(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionLength(drv, solution_len))
   if (solution_len /= int(local_n, c_int64_t)) stop 'invalid solution length'
   call HYPREDRV_Check(HYPREDRV_LinearSystemComputeEigenspectrum(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverDestroy(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSystemResetInitialGuess(drv))
   call accept_success_or_error(HYPREDRV_LinearSystemReadDofmap(drv))

   allocate (empty_indptr(1), empty_cols(0), empty_data(0))
   empty_indptr = 0_c_int64_t
   call expect_error(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, 0_c_int64_t, -1_c_int64_t, &
                                                           empty_indptr, empty_cols, empty_data), 'empty CSR slab')
   deallocate (empty_indptr, empty_cols, empty_data)

   allocate (bad_indptr(1), bad_cols(1), bad_data(1))
   bad_indptr = 0_c_int64_t
   bad_cols = 0_c_int64_t
   bad_data = 1.0_c_double
   call expect_error(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, 0_c_int64_t, 0_c_int64_t, &
                                                           bad_indptr, bad_cols, bad_data), 'malformed CSR length')
   call expect_error(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, 0_c_int64_t, 0_c_int64_t, &
                                                           bad_indptr, bad_cols, bad_data, nnz=2_c_int64_t), &
                     'oversized CSR nnz')
   deallocate (bad_indptr, bad_cols, bad_data)

   if (HYPREDRV_BigIntSize() < int(bit_size(row_start)/8, c_size_t)) then
      allocate (overflow_rhs(1))
      overflow_rhs = 1.0_c_double
      call expect_error(HYPREDRV_LinearSystemSetRHSFromArray(drv, huge(row_start), huge(row_start), &
                                                             overflow_rhs), 'BigInt overflow')
      deallocate (overflow_rhs)
   end if
   allocate (overflow_rhs(1))
   overflow_rhs = 1.0_c_double
   call expect_error(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_start - 1_c_int64_t, &
                                                          overflow_rhs), 'always-on invalid RHS range')
   deallocate (overflow_rhs)

   call HYPREDRV_Check(HYPREDRV_FortranTestDestroyVector(state_vecs(1)))
   call HYPREDRV_Check(HYPREDRV_FortranTestDestroyVector(state_vecs(2)))
   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   call HYPREDRV_Check(HYPREDRV_Finalize())

   call MPI_Finalize(ierr)
contains
   subroutine accept_success_or_error(ierr)
      integer(c_int32_t), intent(in) :: ierr

      if (ierr < 0_c_int32_t) stop 'invalid HYPREDRV error code'
   end subroutine accept_success_or_error

   subroutine expect_error(ierr, label)
      integer(c_int32_t), intent(in) :: ierr
      character(len=*), intent(in) :: label

      if (ierr == HYPREDRV_SUCCESS) then
         write (*, '(a,a)') 'expected Fortran bridge error for ', trim(label)
         stop 'expected Fortran bridge error'
      end if
   end subroutine expect_error
end program test_csr_mpi
