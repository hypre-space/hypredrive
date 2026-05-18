program test_lifecycle
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr
   type(c_ptr) :: drv
   type(c_ptr) :: argv(3)
   character(kind=c_char), allocatable, target :: c_yaml(:), c_key(:), c_value(:)
   integer(c_int) :: warmup, num_reps, num_ls, num_variants

   call MPI_Init(ierr)
   call HYPREDRV_Check(HYPREDRV_Initialize())
   call HYPREDRV_Check(HYPREDRV_PrintLibInfo(int(MPI_COMM_WORLD, c_int), 0_c_int))
   call HYPREDRV_Check(HYPREDRV_PrintSystemInfo(int(MPI_COMM_WORLD, c_int)))
   call HYPREDRV_Check(HYPREDRV_PrintExitInfo(int(MPI_COMM_WORLD, c_int), 'test_fortran_lifecycle'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_SolverPresetRegister('fortran_solver_smoke'//c_null_char, &
                                                     'pcg:'//new_line('a')//'  max_iter: 4'//new_line('a')//c_null_char, &
                                                     'Fortran solver preset smoke test'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_PreconPresetRegister('fortran_precon_smoke'//c_null_char, &
                                                     'amg:'//new_line('a')//'  max_iter: 1'//new_line('a')//c_null_char, &
                                                     'Fortran preconditioner preset smoke test'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
   if (.not. c_associated(drv)) stop 'HYPREDRV_Create returned null handle'
   call HYPREDRV_ToCString('solver:   '//new_line('a')// &
                           '  pcg:   '//new_line('a')// &
                           '    max_iter: 4   '//new_line('a')// &
                           'preconditioner:   '//new_line('a')// &
                           '  amg:   '//new_line('a')// &
                           '    max_iter: 1   '//new_line('a')// &
                           '    tolerance: 0.0   '//new_line('a'), c_yaml)
   call HYPREDRV_ToCString('--solver:pcg:max_iter', c_key)
   call HYPREDRV_ToCString('7', c_value)
   argv(1) = c_loc(c_yaml(1))
   argv(2) = c_loc(c_key(1))
   argv(3) = c_loc(c_value(1))
   call HYPREDRV_Check(HYPREDRV_InputArgsParse(3_c_int, argv, drv))
   call HYPREDRV_Check(HYPREDRV_SetLibraryMode(drv))
   call HYPREDRV_Check(HYPREDRV_ObjectSetName(drv, 'fortran_lifecycle'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_InputArgsSetSolverPreset(drv, 'fortran_solver_smoke'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_InputArgsSetPreconPreset(drv, 'fortran_precon_smoke'//c_null_char))
   call HYPREDRV_Check(HYPREDRV_InputArgsSetPreconVariant(drv, 0_c_int))
   call HYPREDRV_Check(HYPREDRV_InputArgsGetWarmup(drv, warmup))
   call HYPREDRV_Check(HYPREDRV_InputArgsGetNumRepetitions(drv, num_reps))
   call HYPREDRV_Check(HYPREDRV_InputArgsGetNumLinearSystems(drv, num_ls))
   call HYPREDRV_Check(HYPREDRV_InputArgsGetNumPreconVariants(drv, num_variants))
   if (warmup < 0 .or. num_reps < 0 .or. num_ls < 0 .or. num_variants < 0) then
      stop 'invalid input-argument getter value'
   end if
   call HYPREDRV_Check(HYPREDRV_AnnotateBegin(drv, 'initialize'//c_null_char, -1_c_int))
   call HYPREDRV_Check(HYPREDRV_AnnotateEnd(drv, 'initialize'//c_null_char, -1_c_int))
   call HYPREDRV_Check(HYPREDRV_AnnotateLevelBegin(drv, 0_c_int, 'initialize-level'//c_null_char, -1_c_int))
   call HYPREDRV_Check(HYPREDRV_AnnotateLevelEnd(drv, 0_c_int, 'initialize-level'//c_null_char, -1_c_int))
   call accept_success_or_error(HYPREDRV_LinearSystemBuild(drv))
   call accept_success_or_error(HYPREDRV_LinearSystemReadMatrix(drv))
   call accept_success_or_error(HYPREDRV_LinearSystemSetMatrix(drv, c_null_ptr))
   call accept_success_or_error(HYPREDRV_LinearSystemSetRHS(drv, c_null_ptr))
   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   if (c_associated(drv)) stop 'HYPREDRV_Destroy did not clear handle'
   call HYPREDRV_Check(HYPREDRV_Finalize())
   call MPI_Finalize(ierr)
contains
   subroutine accept_success_or_error(ierr)
      integer(c_int32_t), intent(in) :: ierr

      if (ierr < 0_c_int32_t) stop 'invalid HYPREDRV error code'
   end subroutine accept_success_or_error
end program test_lifecycle
