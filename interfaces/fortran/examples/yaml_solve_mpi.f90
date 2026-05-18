program yaml_solve_mpi
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr, rank
   type(c_ptr) :: drv
   character(len=:), allocatable :: yaml
   character(len=512) :: matrix_file, rhs_file

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

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

   call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml))
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
