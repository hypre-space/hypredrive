program yaml_solve_mpi
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr, rank
   type(c_ptr) :: drv
   character(len=:), allocatable :: yaml
   character(len=512) :: matrix_file, rhs_file
   character(len=512) :: cli_arg
   character(len=512), allocatable :: hypredrv_args(:)
   integer :: nargs, iarg, tail_start

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

   ! Collect hypredrive YAML overrides given after -a/--args, e.g.
   ! ./hypredrive-fortran-yaml -a --solver:pcg:max_iter 10
   nargs = command_argument_count()
   tail_start = 0
   do iarg = 1, nargs
      call get_command_argument(iarg, cli_arg)
      if (trim(cli_arg) == '-a' .or. trim(cli_arg) == '--args') then
         tail_start = iarg + 1
         exit
      end if
   end do
   if (tail_start > 0 .and. tail_start <= nargs) then
      allocate (hypredrv_args(nargs - tail_start + 1))
      do iarg = tail_start, nargs
         call get_command_argument(iarg, hypredrv_args(iarg - tail_start + 1))
      end do
   else
      allocate (hypredrv_args(0))
   end if

   call HYPREDRV_Check(HYPREDRV_Initialize())
   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))

   write (matrix_file, '(a)') 'data/ps3d10pt7/np1/IJ.out.A'
   write (rhs_file, '(a)') 'data/ps3d10pt7/np1/IJ.out.b'
   yaml = 'linear_system:'//new_line('a')// &
          '  matrix_filename: '//trim(matrix_file)//new_line('a')// &
          '  rhs_filename: '//trim(rhs_file)//new_line('a')// &
          'solver:'//new_line('a')// &
          '  pcg:'//new_line('a')// &
          '    max_iter: 100'//new_line('a')// &
          '    relative_tol: 1.0e-8'//new_line('a')// &
          'preconditioner:'//new_line('a')// &
          '  amg:'//new_line('a')// &
          '    max_iter: 1'//new_line('a')// &
          '    tolerance: 0.0'//new_line('a')

   call HYPREDRV_Check(HYPREDRV_InputArgsParseYamlArgs(drv, yaml, hypredrv_args))
   call HYPREDRV_Check(HYPREDRV_LinearSystemBuild(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverCreate(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverSetup(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverApply(drv))
   call HYPREDRV_Check(HYPREDRV_StatsPrint(drv))
   call HYPREDRV_Check(HYPREDRV_LinearSolverDestroy(drv))
   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   call HYPREDRV_Check(HYPREDRV_Finalize())

   if (rank == 0) print *, 'Fortran YAML solve completed'
   call MPI_Finalize(ierr)
end program yaml_solve_mpi
