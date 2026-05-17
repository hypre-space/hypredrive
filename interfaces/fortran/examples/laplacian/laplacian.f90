program laplacian
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr, rank, nproc, isolve
   integer(c_int) :: iters
   integer(c_int64_t) :: row_start, row_end, nnz
   integer(c_int64_t), allocatable :: indptr(:), cols(:)
   real(c_double), allocatable :: data(:), rhs(:)
   real(c_double) :: norm
   type(c_ptr) :: drv
   character(len=:), allocatable :: yaml

   integer :: n(3) = [10, 10, 10]
   integer :: p(3) = [1, 1, 1]
   integer :: stencil = 7
   integer :: nsolve = 5
   integer :: verbose = 3
   real(c_double) :: c(3) = [1.0_c_double, 1.0_c_double, 1.0_c_double]
   character(len=256) :: solver_yaml = ''
   character(len=256) :: input_file = ''

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)

   call get_command_argument(1, input_file)
   call read_parameters(trim(input_file), rank, n, p, c, stencil, nsolve, verbose, solver_yaml)
   call validate_parameters(rank, nproc, n, p, stencil, nsolve)

   call HYPREDRV_Check(HYPREDRV_Initialize())
   if (iand(verbose, 1) /= 0) call HYPREDRV_Check(HYPREDRV_PrintLibInfo(int(MPI_COMM_WORLD, c_int), 1_c_int))
   if (iand(verbose, 2) /= 0) call HYPREDRV_Check(HYPREDRV_PrintSystemInfo(int(MPI_COMM_WORLD, c_int)))

   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
   call HYPREDRV_Check(HYPREDRV_SetLibraryMode(drv))

   if (len_trim(solver_yaml) > 0) then
      call read_text_file(trim(solver_yaml), yaml)
   else
      yaml = default_yaml()
   end if
   call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml))

   if (rank == 0) call print_setup(n, p, c, stencil, nsolve, verbose, solver_yaml)
   if (rank == 0 .and. verbose > 0) print '(a)', 'Assembling 3D Laplacian system...'
   call build_laplacian_7pt(rank, n, p, c, row_start, row_end, indptr, cols, data, rhs, nnz)

   call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols, data, nnz))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end, rhs))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetInitialGuess(drv, c_null_ptr))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetPrecMatrix(drv, c_null_ptr))

   if (iand(verbose, 4) /= 0) call HYPREDRV_Check(HYPREDRV_LinearSystemPrint(drv))

   do isolve = 1, nsolve
      if (rank == 0) print '(a,i0,a,i0,a)', 'Solve ', isolve, '/', nsolve, '...'
      if (isolve > 1) call HYPREDRV_Check(HYPREDRV_LinearSystemResetInitialGuess(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverCreate(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverSetup(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverApply(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverDestroy(drv))
   end do

   call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionNorm(drv, 'l2'//c_null_char, norm))
   call HYPREDRV_Check(HYPREDRV_LinearSolverGetNumIter(drv, iters))
   if (rank == 0) print '(a,es12.5,a,i0)', 'Solution l2 norm: ', norm, ', iterations: ', iters
   if (rank == 0 .and. iand(verbose, 1) /= 0) call HYPREDRV_Check(HYPREDRV_StatsPrint(drv))

   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   call HYPREDRV_Check(HYPREDRV_Finalize())
   call MPI_Finalize(ierr)

contains

   subroutine read_parameters(filename, rank, n, p, c, stencil, nsolve, verbose, solver_yaml)
      character(len=*), intent(in) :: filename
      integer, intent(in) :: rank
      integer, intent(inout) :: n(3), p(3), stencil, nsolve, verbose
      real(c_double), intent(inout) :: c(3)
      character(len=*), intent(inout) :: solver_yaml
      integer :: unit, ios
      namelist /laplacian/ n, p, c, stencil, nsolve, verbose, solver_yaml

      if (len_trim(filename) == 0) return
      unit = free_unit()
      open (unit=unit, file=filename, status='old', action='read', iostat=ios)
      if (ios /= 0) then
         if (rank == 0) print '(a,a)', 'Could not open input file: ', trim(filename)
         stop 'failed to open namelist input'
      end if
      read (unit, nml=laplacian, iostat=ios)
      close (unit)
      if (ios /= 0) then
         if (rank == 0) print '(a,a)', 'Could not parse namelist input: ', trim(filename)
         stop 'failed to parse namelist input'
      end if
   end subroutine read_parameters

   subroutine validate_parameters(rank, nproc, n, p, stencil, nsolve)
      integer, intent(in) :: rank, nproc, n(3), p(3), stencil, nsolve

      if (any(n < 2)) then
         if (rank == 0) print '(a)', 'Each grid dimension must be at least 2.'
         stop 'invalid grid dimensions'
      end if
      if (any(p < 1)) then
         if (rank == 0) print '(a)', 'Each processor-grid dimension must be positive.'
         stop 'invalid processor grid'
      end if
      if (product(p) /= nproc) then
         if (rank == 0) print '(a,i0,a,3(i0,1x),a,i0)', 'MPI ranks (', nproc, ') do not match p = ', p, 'product ', product(p)
         stop 'processor grid mismatch'
      end if
      if (any(n < p)) then
         if (rank == 0) print '(a)', 'This example requires at least one grid point per rank in each partitioned direction.'
         stop 'empty local block'
      end if
      if (stencil /= 7) then
         if (rank == 0) print '(a)', 'The Fortran Laplacian example currently supports stencil = 7.'
         stop 'unsupported stencil'
      end if
      if (nsolve < 1) then
         if (rank == 0) print '(a)', 'nsolve must be positive.'
         stop 'invalid nsolve'
      end if
   end subroutine validate_parameters

   subroutine print_setup(n, p, c, stencil, nsolve, verbose, solver_yaml)
      integer, intent(in) :: n(3), p(3), stencil, nsolve, verbose
      real(c_double), intent(in) :: c(3)
      character(len=*), intent(in) :: solver_yaml

      print '(a)', ''
      print '(a)', '====================================================='
      print '(a)', '              Fortran Laplacian Problem Setup'
      print '(a)', '====================================================='
      print '(a,3(i0,1x))', 'Grid dimensions:      ', n
      print '(a,3(i0,1x))', 'Processor topology:   ', p
      print '(a,3(es10.3,1x))', 'Diffusion coeffs:     ', c
      print '(a,i0,a)', 'Discretization:       ', stencil, '-point stencil'
      print '(a,z0)', 'Verbosity level:      0x', verbose
      print '(a,i0)', 'Number of solves:     ', nsolve
      if (len_trim(solver_yaml) > 0) print '(a,a)', 'Solver YAML:          ', trim(solver_yaml)
      print '(a)', '====================================================='
      print '(a)', ''
   end subroutine print_setup

   subroutine build_laplacian_7pt(rank, n, p, c, row_start, row_end, indptr, cols, data, rhs, nnz)
      integer, intent(in) :: rank, n(3), p(3)
      real(c_double), intent(in) :: c(3)
      integer(c_int64_t), intent(out) :: row_start, row_end, nnz
      integer(c_int64_t), allocatable, intent(out) :: indptr(:), cols(:)
      real(c_double), allocatable, intent(out) :: data(:), rhs(:)
      integer(c_int64_t), allocatable :: starts(:, :)
      integer :: coords(3), local_n(3), ix, iy, iz
      integer(c_int64_t) :: gx, gy, gz, row, pos, row_local
      integer(c_int64_t) :: local_size, max_nnz, global_size

      call compute_starts(n, p, starts)
      call rank_to_coords(rank, p, coords)
      local_n = [int(starts(coords(1) + 1, 1) - starts(coords(1), 1)), &
                 int(starts(coords(2) + 1, 2) - starts(coords(2), 2)), &
                 int(starts(coords(3) + 1, 3) - starts(coords(3), 3))]
      local_size = int(local_n(1), c_int64_t)*int(local_n(2), c_int64_t)* &
                   int(local_n(3), c_int64_t)
      max_nnz = 7_c_int64_t*local_size
      global_size = int(n(1), c_int64_t)*int(n(2), c_int64_t)*int(n(3), c_int64_t)
      row_start = grid_to_index(starts(coords(1), 1), starts(coords(2), 2), starts(coords(3), 3), coords, n, starts)
      row_end = row_start + local_size - 1_c_int64_t

      allocate (indptr(local_size + 1_c_int64_t), cols(max_nnz), data(max_nnz), rhs(local_size))
      rhs = 0.0_c_double
      indptr(1) = 0_c_int64_t
      pos = 0
      row_local = 0
      do iz = 0, local_n(3) - 1
         gz = starts(coords(3), 3) + int(iz, c_int64_t)
         do iy = 0, local_n(2) - 1
            gy = starts(coords(2), 2) + int(iy, c_int64_t)
            do ix = 0, local_n(1) - 1
               gx = starts(coords(1), 1) + int(ix, c_int64_t)
               row = grid_to_index(gx, gy, gz, coords, n, starts)
               row_local = row_local + 1

               call append_entry(pos, cols, data, row, 2.0_c_double*(c(1) + c(2) + c(3)))
               call add_neighbor(pos, cols, data, gx - 1_c_int64_t, gy, gz, n, p, starts, -c(1))
               call add_neighbor(pos, cols, data, gx + 1_c_int64_t, gy, gz, n, p, starts, -c(1))
               call add_neighbor(pos, cols, data, gx, gy - 1_c_int64_t, gz, n, p, starts, -c(2))
               call add_neighbor(pos, cols, data, gx, gy + 1_c_int64_t, gz, n, p, starts, -c(2))
               call add_neighbor(pos, cols, data, gx, gy, gz - 1_c_int64_t, n, p, starts, -c(3))
               call add_neighbor(pos, cols, data, gx, gy, gz + 1_c_int64_t, n, p, starts, -c(3))

               if (gy == 0_c_int64_t) rhs(row_local) = 1.0_c_double
               indptr(row_local + 1) = int(pos, c_int64_t)
            end do
         end do
      end do

      if (any(cols(:pos) < 0_c_int64_t) .or. any(cols(:pos) >= global_size)) then
         print '(a,2(i0,1x),a,i0)', 'Invalid generated column range: ', minval(cols(:pos)), &
            maxval(cols(:pos)), 'global rows: ', global_size
         stop 'invalid generated column index'
      end if
      nnz = pos
   end subroutine build_laplacian_7pt

   subroutine compute_starts(n, p, starts)
      integer, intent(in) :: n(3), p(3)
      integer(c_int64_t), allocatable, intent(out) :: starts(:, :)
      integer :: dim, base, rem, coord

      allocate (starts(0:maxval(p), 3))
      starts = 0_c_int64_t
      do dim = 1, 3
         base = n(dim)/p(dim)
         rem = n(dim) - base*p(dim)
         do coord = 0, p(dim)
            starts(coord, dim) = int(base*coord + min(coord, rem), c_int64_t)
         end do
      end do
   end subroutine compute_starts

   subroutine rank_to_coords(rank, p, coords)
      integer, intent(in) :: rank, p(3)
      integer, intent(out) :: coords(3)

      coords(1) = mod(rank, p(1))
      coords(2) = mod(rank/p(1), p(2))
      coords(3) = rank/(p(1)*p(2))
   end subroutine rank_to_coords

   subroutine owner_coords(gx, gy, gz, p, starts, coords)
      integer(c_int64_t), intent(in) :: gx, gy, gz
      integer, intent(in) :: p(3)
      integer(c_int64_t), intent(in) :: starts(0:, :)
      integer, intent(out) :: coords(3)
      integer :: dim, coord
      integer(c_int64_t) :: g(3)

      g = [gx, gy, gz]
      do dim = 1, 3
         coords(dim) = -1
         do coord = 0, p(dim) - 1
            if (g(dim) >= starts(coord, dim) .and. g(dim) < starts(coord + 1, dim)) then
               coords(dim) = coord
               exit
            end if
         end do
      end do
   end subroutine owner_coords

   integer(c_int64_t) function grid_to_index(gx, gy, gz, coords, n, starts) result(idx)
      integer(c_int64_t), intent(in) :: gx, gy, gz
      integer, intent(in) :: coords(3), n(3)
      integer(c_int64_t), intent(in) :: starts(0:, :)
      integer :: rx, ry, rz
      integer(c_int64_t) :: nx, ny, nz

      idx = 0_c_int64_t
      do rz = 0, coords(3) - 1
         nz = starts(rz + 1, 3) - starts(rz, 3)
         idx = idx + int(n(1), c_int64_t)*int(n(2), c_int64_t)*nz
      end do
      do ry = 0, coords(2) - 1
         ny = starts(ry + 1, 2) - starts(ry, 2)
         nz = starts(coords(3) + 1, 3) - starts(coords(3), 3)
         idx = idx + int(n(1), c_int64_t)*ny*nz
      end do
      do rx = 0, coords(1) - 1
         nx = starts(rx + 1, 1) - starts(rx, 1)
         ny = starts(coords(2) + 1, 2) - starts(coords(2), 2)
         nz = starts(coords(3) + 1, 3) - starts(coords(3), 3)
         idx = idx + nx*ny*nz
      end do

      nx = starts(coords(1) + 1, 1) - starts(coords(1), 1)
      ny = starts(coords(2) + 1, 2) - starts(coords(2), 2)
      idx = idx + ((gz - starts(coords(3), 3))*ny + &
                   (gy - starts(coords(2), 2)))*nx + &
            (gx - starts(coords(1), 1))
   end function grid_to_index

   subroutine add_neighbor(pos, cols, data, gx, gy, gz, n, p, starts, value)
      integer(c_int64_t), intent(inout) :: pos
      integer(c_int64_t), intent(inout) :: cols(:)
      real(c_double), intent(inout) :: data(:)
      integer(c_int64_t), intent(in) :: gx, gy, gz
      integer, intent(in) :: n(3), p(3)
      integer(c_int64_t), intent(in) :: starts(0:, :)
      real(c_double), intent(in) :: value
      integer :: owner(3)

      if (gx < 0_c_int64_t .or. gx >= int(n(1), c_int64_t)) return
      if (gy < 0_c_int64_t .or. gy >= int(n(2), c_int64_t)) return
      if (gz < 0_c_int64_t .or. gz >= int(n(3), c_int64_t)) return
      call owner_coords(gx, gy, gz, p, starts, owner)
      call append_entry(pos, cols, data, grid_to_index(gx, gy, gz, owner, n, starts), value)
   end subroutine add_neighbor

   subroutine append_entry(pos, cols, data, col, value)
      integer(c_int64_t), intent(inout) :: pos
      integer(c_int64_t), intent(inout) :: cols(:)
      real(c_double), intent(inout) :: data(:)
      integer(c_int64_t), intent(in) :: col
      real(c_double), intent(in) :: value

      pos = pos + 1
      cols(pos) = col
      data(pos) = value
   end subroutine append_entry

   function default_yaml() result(text)
      character(len=:), allocatable :: text

      text = 'solver:'//new_line('a')// &
             '  pcg:'//new_line('a')// &
             '    max_iter: 100'//new_line('a')// &
             '    relative_tol: 1.0e-8'//new_line('a')// &
             'preconditioner:'//new_line('a')// &
             '  amg:'//new_line('a')// &
             '    max_iter: 1'//new_line('a')// &
             '    tolerance: 0.0'//new_line('a')
   end function default_yaml

   subroutine read_text_file(filename, text)
      character(len=*), intent(in) :: filename
      character(len=:), allocatable, intent(out) :: text
      character(len=1024) :: line
      integer :: unit, ios, line_len, total_len, used

      total_len = 0
      unit = free_unit()
      open (unit=unit, file=filename, status='old', action='read', iostat=ios)
      if (ios /= 0) stop 'failed to open solver YAML file'
      do
         read (unit, '(a)', iostat=ios) line
         if (ios /= 0) exit
         total_len = total_len + len_trim(line) + 1
      end do
      close (unit)

      allocate (character(len=total_len) :: text)
      if (total_len == 0) return

      used = 0
      unit = free_unit()
      open (unit=unit, file=filename, status='old', action='read', iostat=ios)
      if (ios /= 0) stop 'failed to open solver YAML file'
      do
         read (unit, '(a)', iostat=ios) line
         if (ios /= 0) exit
         line_len = len_trim(line)
         if (line_len > 0) text(used + 1:used + line_len) = trim(line)
         used = used + line_len
         text(used + 1:used + 1) = new_line('a')
         used = used + 1
      end do
      close (unit)
   end subroutine read_text_file

   integer function free_unit() result(unit)
      logical :: opened

      do unit = 10, 999
         inquire (unit=unit, opened=opened)
         if (.not. opened) return
      end do
      stop 'no free Fortran I/O unit'
   end function free_unit

end program laplacian
