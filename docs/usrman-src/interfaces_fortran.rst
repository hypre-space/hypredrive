.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _FortranInterface:

Fortran Interface
=================

The Fortran interface provides thin bindings to the Fortran-compatible subset of
the public ``HYPREDRV_`` C API. It supports Fortran applications that use MPI and
need hypredrive without C glue code. APIs that require C pointer-to-array
contracts not represented by this module, including the discrete gradient, curl,
coordinate, and coarse-Schur setters, remain available through the C and C++
interfaces.

Prerequisites
-------------

The interface requires these components:

- A C compiler.
- A Fortran 2003 compiler.
- MPI C and Fortran wrappers, such as ``mpicc`` and ``mpifort``.
- CMake.
- A HYPRE build that is compatible with `hypredrive`.

The interface requires real-valued HYPRE with the C ``double`` ABI for
``HYPRE_Real``. Configuration rejects single-precision and long-double builds. The
examples use ``mpif.h`` for compiler and MPI compatibility. The build also requires an
MPI Fortran wrapper and library.

Build
-----

CMake disables the optional interface by default:

.. code-block:: bash

   cmake -S . -B build-fortran \
     -DHYPREDRV_ENABLE_FORTRAN=ON \
     -DHYPREDRV_ENABLE_TESTING=ON \
     -DHYPREDRV_ENABLE_EXAMPLES=ON
   cmake --build build-fortran --parallel

The ``laplacian-fortran`` driver is available whenever
``HYPREDRV_ENABLE_FORTRAN=ON``, so the core Fortran test coverage does not
depend on ``HYPREDRV_ENABLE_EXAMPLES``. The ``fortran-test`` target runs small
namelist-driven Laplacian smoke cases: serial, 2-rank x/z partitions, a 4-rank
x/y partition, anisotropic coefficients, and external solver YAML.

Build only the examples:

.. code-block:: bash

   cmake --build build-fortran --target fortran-examples

CMake installs the ``laplacian-fortran`` driver when
``HYPREDRV_ENABLE_FORTRAN=ON``. This driver is a user example and the
reproducible smoke driver for the Fortran tests.

Installed use
-------------

Save the minimal example below as ``minimal.f90`` and use this ``CMakeLists.txt``
in the same directory to link an installed hypredrive:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.23)
   project(minimal LANGUAGES C Fortran)
   find_package(MPI COMPONENTS Fortran REQUIRED)
   find_package(HYPREDRV CONFIG REQUIRED)
   add_executable(minimal minimal.f90)
   target_link_libraries(minimal PRIVATE HYPREDRV::Fortran MPI::MPI_Fortran)

Configure this application with ``-DCMAKE_PREFIX_PATH=/path/to/hypredrive/install``,
then build and run ``minimal``.

Fortran module files are compiler-specific. Use the same Fortran compiler family
for hypredrive and downstream code. A different compiler can fail to read
``hypredrive.mod``. CMake installs this module as ``include/hypredrive.mod`` under
the installation prefix.

Installed hypredrive executables use relative runtime paths for prefix-local
libraries. External HYPRE or MPI libraries must also be on the runtime search
path (for example, ``LD_LIBRARY_PATH`` on Linux).

Minimal example
---------------

This serial example solves ``A x = b`` for ``A = [[2, -1], [-1, 2]]`` and
``b = [1, 1]``, giving ``x = [1, 1]``.

.. code-block:: fortran

   program minimal
      use, intrinsic :: iso_c_binding
      use hypredrive
      implicit none
      include 'mpif.h'

      integer :: ierr
      type(c_ptr) :: drv, solution
      real(c_double), pointer :: x(:)
      integer(c_int64_t) :: indptr(3) = [0_c_int64_t, 2_c_int64_t, 4_c_int64_t]
      integer(c_int64_t) :: cols(4) = [0_c_int64_t, 1_c_int64_t, 0_c_int64_t, 1_c_int64_t]
      real(c_double) :: values(4) = [2.0_c_double, -1.0_c_double, -1.0_c_double, 2.0_c_double]
      real(c_double) :: rhs(2) = [1.0_c_double, 1.0_c_double]

      call MPI_Init(ierr)
      call HYPREDRV_Check(HYPREDRV_Initialize())
      call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_SELF, c_int), drv))
      call HYPREDRV_Check(HYPREDRV_SetLibraryMode(drv))
      call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, &
         'general: {statistics: 0}' // new_line('a') // &
         'solver: pcg' // new_line('a') // 'preconditioner: amg' // new_line('a')))
      call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, &
         0_c_int64_t, 1_c_int64_t, indptr, cols, values))
      call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, 0_c_int64_t, 1_c_int64_t, rhs))
      call HYPREDRV_Check(HYPREDRV_LinearSystemSetInitialGuess(drv, c_null_ptr))
      call HYPREDRV_Check(HYPREDRV_LinearSolverCreate(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverSetup(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSolverApply(drv))
      call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionValues(drv, solution))
      call c_f_pointer(solution, x, [2])
      print *, x
      call HYPREDRV_Check(HYPREDRV_Destroy(drv))
      call HYPREDRV_Check(HYPREDRV_Finalize())
      call MPI_Finalize(ierr)
   end program minimal

The solution pointer is a view of driver-owned storage; use or copy it before
destroying the driver. The CSR and RHS helpers copy the input arrays.

The binding accepts MPI communicators as Fortran handles and converts them to C
communicators. The examples use ``mpif.h`` instead of
``use mpi`` for compatibility with MPI installations whose ``mpi.mod`` was built
by another Fortran compiler.

The helper ``HYPREDRV_InputArgsParseYaml`` accepts ordinary Fortran strings and
parses them through the same YAML input path as the C API:

.. code-block:: fortran

   yaml = 'solver:' // new_line('a') // &
          '  pcg:' // new_line('a') // &
          '    relative_tol: 1.0e-8' // new_line('a') // &
          'preconditioner:' // new_line('a') // &
          '  amg:' // new_line('a') // &
          '    max_iter: 1' // new_line('a') // &
          '    tolerance: 0.0' // new_line('a')

   call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml))

The ``HYPREDRV_InputArgsParseYamlArgs`` helper also accepts command-line interface (CLI)
override tokens. These tokens override the YAML configuration. One example is
``args = [character(len=32) :: '--solver:pcg:max_iter', '100']``.

CSR assembly
------------

The convenience CSR path accepts zero-based global row and column indices, using
Fortran arrays for local CSR slabs:

.. code-block:: fortran

   integer(c_int64_t), allocatable :: indptr(:), cols(:)
   real(c_double), allocatable :: data(:), rhs(:)

   call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols, data))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end, rhs))

Normalize ``indptr``. Set ``indptr(1)`` to zero. Set the final entry to
``size(cols)``. Confirm that ``size(cols)`` equals ``size(data)``. The convenience
helper rejects empty local CSR slabs.

The Fortran 2003 binding accepts assumed-shape arrays. For large systems, pass contiguous
allocatable arrays. This prevents packing copies for strided sections.

Examples
--------

Examples are in ``interfaces/fortran/examples``:

- ``yaml_solve_mpi.f90`` reads matrix/RHS files and solves using YAML options.
- ``laplacian/laplacian.f90`` assembles a distributed 3D Laplacian directly from
  Fortran CSR arrays and reads problem settings from a namelist file.

Run the Laplacian example with MPI:

.. code-block:: bash

   mpiexec -n 2 ./build-fortran/laplacian-fortran interfaces/fortran/examples/laplacian/default.nml

Both examples accept hypredrive command-line overrides after ``-a``/``--args``,
for example ``-a --solver:pcg:max_iter 50``.

API coverage
------------

The module provides procedures for the Fortran-compatible public ``HYPREDRV_``
functions in ``include/HYPREDRV.h``. These procedures cover lifecycle control,
errors, input, presets, linear systems, solvers, state vectors, statistics,
annotations, and timing. The module also provides small Fortran helpers, such as
``HYPREDRV_Check`` and
``HYPREDRV_BigIntSize``.

The Fortran interface preserves the C API ownership rules. Call
``HYPREDRV_SetLibraryMode`` before attaching borrowed HYPRE objects to
retain caller ownership. Objects created by the CSR and array helpers remain
driver-owned. Raw array accessors such
as ``HYPREDRV_LinearSystemGetSolutionValues`` return C-owned host pointers. Use
``c_f_pointer`` when you need a Fortran view. Do not free these pointers from Fortran.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - C API group
     - Fortran module procedures
   * - Lifecycle/info
     - ``HYPREDRV_Initialize``, ``HYPREDRV_Finalize``, ``HYPREDRV_Create``, ``HYPREDRV_Destroy``, ``HYPREDRV_PrintLibInfo``, ``HYPREDRV_PrintSystemInfo``, ``HYPREDRV_PrintExitInfo``
   * - Errors/helpers
     - ``HYPREDRV_ErrorCodeDescribe``, ``HYPREDRV_ErrorInvalidValue``, ``HYPREDRV_BigIntSize``, ``HYPREDRV_Check``, ``HYPREDRV_ToCString``
   * - Object/input/options
     - ``HYPREDRV_SetLibraryMode``, ``HYPREDRV_ObjectSetName``, ``HYPREDRV_InputArgsParse``, ``HYPREDRV_InputArgsParseYaml``, ``HYPREDRV_InputArgsParseYamlArgs``, ``HYPREDRV_InputArgsGet*``, ``HYPREDRV_InputArgsSet*``
   * - Presets
     - ``HYPREDRV_SolverPresetRegister``, ``HYPREDRV_PreconPresetRegister``, ``HYPREDRV_InputArgsSetSolverPreset``, ``HYPREDRV_InputArgsSetPreconPreset``
   * - Linear system
     - ``HYPREDRV_LinearSystemBuild``, ``HYPREDRV_LinearSystemReadMatrix``, ``HYPREDRV_LinearSystemSetMatrix*``, ``HYPREDRV_LinearSystemSetRHS*``, ``HYPREDRV_LinearSystemSetInitialGuess``, ``HYPREDRV_LinearSystemResetInitialGuess``, ``HYPREDRV_LinearSystemSetSolution``, ``HYPREDRV_LinearSystemSetReferenceSolution``, ``HYPREDRV_LinearSystemSetPrecMatrix``, ``HYPREDRV_LinearSystemPrint``
   * - Dofmaps/nullspace
     - ``HYPREDRV_LinearSystemSetDofmap``, ``HYPREDRV_LinearSystemSetInterleavedDofmap``, ``HYPREDRV_LinearSystemSetContiguousDofmap``, ``HYPREDRV_LinearSystemReadDofmap``, ``HYPREDRV_LinearSystemPrintDofmap``, ``HYPREDRV_LinearSystemSetNearNullSpace``, ``HYPREDRV_LinearSystemSetNullSpace``
   * - Accessors
     - ``HYPREDRV_LinearSystemGetSolution*``, ``HYPREDRV_LinearSystemGetRHS*``, ``HYPREDRV_LinearSystemGetMatrix``, ``HYPREDRV_LinearSystemGetSolutionNorm``
   * - State vectors
     - ``HYPREDRV_StateVectorSet``, ``HYPREDRV_StateVectorGetValues``, ``HYPREDRV_StateVectorCopy``, ``HYPREDRV_StateVectorUpdateAll``, ``HYPREDRV_StateVectorApplyCorrection``
   * - Solver/preconditioner
     - ``HYPREDRV_PreconCreate``, ``HYPREDRV_PreconSetup``, ``HYPREDRV_PreconApply``, ``HYPREDRV_PreconDestroy``, ``HYPREDRV_LinearSolverCreate``, ``HYPREDRV_LinearSolverSetup``, ``HYPREDRV_LinearSolverApply``, ``HYPREDRV_LinearSolverDestroy``
   * - Stats/timing
     - ``HYPREDRV_StatsPrint``, ``HYPREDRV_StatsLevelGetCount``, ``HYPREDRV_StatsLevelGetEntry``, ``HYPREDRV_StatsLevelPrint``, ``HYPREDRV_LinearSolverGetNumIter``, ``HYPREDRV_LinearSolverGetConverged``, ``HYPREDRV_LinearSolverGetFinalRelativeResidualNorm``, ``HYPREDRV_LinearSolverGetSetupTime``, ``HYPREDRV_LinearSolverGetSolveTime``
   * - Annotations/eigenspectrum
     - ``HYPREDRV_Annotate*``, ``HYPREDRV_LinearSystemComputeEigenspectrum``

Testing
-------

Use CMake's convenience target for local validation:

.. code-block:: bash

   cmake --build build-fortran --target fortran-test --parallel

The target tests lifecycle control and complete ``argv`` input parsing. It also tests
``HYPREDRV_Check`` failures, MPI CSR assembly, solves, and namelist-driven Laplacian cases.

Notes
-----

The C header defines ``HYPREDRV_SUCCESS`` as a macro. The Fortran module defines
it separately as a parameter because ``bind(C)`` interfaces cannot import C
preprocessor macros.

The current CI job exercises GNU Fortran on Ubuntu. The project plans future CI
coverage for Intel, NVHPC, Cray, and macOS Fortran stacks.

Limitations
-----------

- The first implementation targets real-valued, double-precision HYPRE builds.
- CSR/RHS convenience helpers use ``integer(c_int64_t)`` and ``real(c_double)``
  on the Fortran side and convert through a small C bridge.
- The convenience helper rejects empty local CSR slabs.
- The interface is thin. It is not an object-oriented Fortran redesign of
  `hypredrive`.
