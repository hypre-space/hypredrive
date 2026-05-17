.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _FortranInterface:

Fortran Interface
=================

The Fortran interface provides thin bindings to the public ``HYPREDRV_`` C API.
It is intended for Fortran applications that already use MPI and want to drive
hypredrive without writing C glue code.

Prerequisites
-------------

Prerequisites are a C compiler, a Fortran 2003 compiler, MPI C and Fortran
wrappers (``mpicc``/``mpifort`` or equivalents), CMake, and a HYPRE build
compatible with hypredrive. The examples use ``mpif.h`` for broad compiler/MPI
compatibility, but the build still requires an MPI Fortran wrapper and library.

Build
-----

The interface is optional and disabled by default:

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

The ``laplacian-fortran`` driver is installed whenever
``HYPREDRV_ENABLE_FORTRAN=ON``. It is both a user-facing example and the
reproducible smoke driver used by the Fortran tests.

Installed use
-------------

Installed Fortran consumers can use the exported CMake target:

.. code-block:: cmake

   find_package(MPI COMPONENTS Fortran REQUIRED)
   find_package(HYPREDRV REQUIRED)
   target_link_libraries(my_solver PRIVATE HYPREDRV::Fortran MPI::MPI_Fortran)

Fortran module files are compiler-specific. Build downstream code with the same
Fortran compiler family used to install hypredrive, otherwise the compiler may
not be able to read ``hypredrive.mod``. The module is installed as
``include/hypredrive.mod`` under the installation prefix.

Installed executables use a relative runtime search path so prefix-local
libraries in ``lib`` are found without setting ``LD_LIBRARY_PATH``.

Usage model
-----------

Applications use the ``hypredrive`` module and opaque ``type(c_ptr)`` handles:

.. code-block:: fortran

   program driver
      use, intrinsic :: iso_c_binding
      use hypredrive
      implicit none
      include 'mpif.h'

      integer :: ierr
      type(c_ptr) :: drv

      call MPI_Init(ierr)
      call HYPREDRV_Check(HYPREDRV_Initialize())
      call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
      ! Install matrix/RHS and solve here.
      call HYPREDRV_Check(HYPREDRV_Destroy(drv))
      call HYPREDRV_Check(HYPREDRV_Finalize())
      call MPI_Finalize(ierr)
   end program driver

MPI communicators are passed as Fortran MPI handles and converted to C
communicators inside the binding. The examples use ``mpif.h`` instead of
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

CSR assembly
------------

The convenience CSR path accepts zero-based global row and column indices, using
Fortran arrays for local CSR slabs:

.. code-block:: fortran

   integer(c_int64_t), allocatable :: indptr(:), cols(:)
   real(c_double), allocatable :: data(:), rhs(:)

   call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols, data))
   call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end, rhs))

``indptr`` must be normalized: ``indptr(1) == 0`` and the final entry must equal
``size(cols)`` and ``size(data)``. Empty local CSR slabs are rejected by the
convenience helper. The Fortran 2003 binding accepts ordinary assumed-shape
arrays; pass contiguous allocatable arrays for large systems to avoid
compiler-generated packing copies for strided sections.

Examples
--------

Examples are in ``interfaces/fortran/examples``:

- ``yaml_solve_mpi.f90`` reads matrix/RHS files and solves using YAML options.
- ``laplacian/laplacian.f90`` assembles a distributed 3D Laplacian directly from
  Fortran CSR arrays and reads problem settings from a namelist file.

Run the Laplacian example with MPI:

.. code-block:: bash

   mpiexec -n 2 ./build-fortran/laplacian-fortran interfaces/fortran/examples/laplacian/default.nml

API coverage
------------

The module exposes procedures for the public ``HYPREDRV_`` functions in
``include/HYPREDRV.h``. This includes lifecycle, error handling, input parsing,
presets, matrix/RHS setup, solver and preconditioner lifecycle, state vectors,
statistics, annotations, timing accessors, and small Fortran helper routines
such as ``HYPREDRV_Check`` and ``HYPREDRV_BigIntSize``.

The Fortran interface preserves the C API ownership rules. Borrowed HYPRE
objects remain caller-owned unless the corresponding C function documents that
hypredrive takes ownership.

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
     - ``HYPREDRV_SetLibraryMode``, ``HYPREDRV_ObjectSetName``, ``HYPREDRV_InputArgsParse``, ``HYPREDRV_InputArgsParseYaml``, ``HYPREDRV_InputArgsGet*``, ``HYPREDRV_InputArgsSet*``
   * - Presets
     - ``HYPREDRV_SolverPresetRegister``, ``HYPREDRV_PreconPresetRegister``, ``HYPREDRV_InputArgsSetSolverPreset``, ``HYPREDRV_InputArgsSetPreconPreset``
   * - Linear system
     - ``HYPREDRV_LinearSystemBuild``, ``HYPREDRV_LinearSystemReadMatrix``, ``HYPREDRV_LinearSystemSetMatrix*``, ``HYPREDRV_LinearSystemSetRHS*``, ``HYPREDRV_LinearSystemSetInitialGuess``, ``HYPREDRV_LinearSystemResetInitialGuess``, ``HYPREDRV_LinearSystemSetSolution``, ``HYPREDRV_LinearSystemSetReferenceSolution``, ``HYPREDRV_LinearSystemSetPrecMatrix``, ``HYPREDRV_LinearSystemPrint``
   * - Dofmaps/nullspace
     - ``HYPREDRV_LinearSystemSetDofmap``, ``HYPREDRV_LinearSystemSetInterleavedDofmap``, ``HYPREDRV_LinearSystemSetContiguousDofmap``, ``HYPREDRV_LinearSystemReadDofmap``, ``HYPREDRV_LinearSystemPrintDofmap``, ``HYPREDRV_LinearSystemSetNearNullSpace``
   * - Accessors
     - ``HYPREDRV_LinearSystemGetSolution*``, ``HYPREDRV_LinearSystemGetRHS*``, ``HYPREDRV_LinearSystemGetMatrix``, ``HYPREDRV_LinearSystemGetSolutionNorm``
   * - State vectors
     - ``HYPREDRV_StateVectorSet``, ``HYPREDRV_StateVectorGetValues``, ``HYPREDRV_StateVectorCopy``, ``HYPREDRV_StateVectorUpdateAll``, ``HYPREDRV_StateVectorApplyCorrection``
   * - Solver/preconditioner
     - ``HYPREDRV_PreconCreate``, ``HYPREDRV_PreconSetup``, ``HYPREDRV_PreconApply``, ``HYPREDRV_PreconDestroy``, ``HYPREDRV_LinearSolverCreate``, ``HYPREDRV_LinearSolverSetup``, ``HYPREDRV_LinearSolverApply``, ``HYPREDRV_LinearSolverDestroy``
   * - Stats/timing
     - ``HYPREDRV_StatsPrint``, ``HYPREDRV_StatsLevelGetCount``, ``HYPREDRV_StatsLevelGetEntry``, ``HYPREDRV_StatsLevelPrint``, ``HYPREDRV_LinearSolverGetNumIter``, ``HYPREDRV_LinearSolverGetSetupTime``, ``HYPREDRV_LinearSolverGetSolveTime``
   * - Annotations/eigenspectrum
     - ``HYPREDRV_Annotate*``, ``HYPREDRV_LinearSystemComputeEigenspectrum``

Testing
-------

Use CMake's convenience target for local validation:

.. code-block:: bash

   cmake --build build-fortran --target fortran-test --parallel

The test target includes lifecycle checks and full ``argv`` input parsing from
``test_lifecycle.f90``, expected ``HYPREDRV_Check`` failure behavior from
``test_check_failure.f90``, MPI CSR assembly/solve coverage, and the
namelist-driven Laplacian cases.

Notes
-----

``HYPREDRV_SUCCESS`` is defined separately as a C macro and as a Fortran module
parameter because Fortran ``bind(C)`` interfaces cannot import C preprocessor
macros.

The current CI job exercises GNU Fortran on Ubuntu. Other compiler families
such as Intel, NVHPC, Cray, and macOS Fortran stacks are intended future CI
coverage.

Limitations
-----------

- The first implementation targets real-valued HYPRE builds.
- CSR/RHS convenience helpers use ``integer(c_int64_t)`` and ``real(c_double)``
  on the Fortran side and convert through a small C bridge.
- Empty local CSR slabs are rejected by the convenience helper.
- The interface is intentionally thin; it is not an object-oriented Fortran
  redesign of hypredrive.
