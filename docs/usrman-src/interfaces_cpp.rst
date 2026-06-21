.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _CppInterface:

C++ Interface
=============

The C++ interface is a header-only C++17 wrapper over the public ``HYPREDRV_`` C
API. It provides RAII ownership for ``HYPREDRV_t``, throwing error handling, a
YAML-driven configuration path, and an idiomatic snake_case method mirror of the
C API.

Build
-----

.. code-block:: bash

   cmake -S . -B build-cpp \
     -DHYPREDRV_ENABLE_CPP=ON \
     -DHYPREDRV_ENABLE_TESTING=ON \
     -DHYPREDRV_ENABLE_EXAMPLES=ON
   cmake --build build-cpp --target cpp-test --parallel

Usage
-----

.. code-block:: cpp

   #include <hypredrive.hpp>

   hypredrive::initialize();
   hypredrive::driver drv(MPI_COMM_SELF);
   drv.set_library_mode();
   drv.parse_yaml(R"yaml(
   general:
     statistics: 0
   solver:
     pcg:
       max_iter: 100
       relative_tol: 1.0e-8
   preconditioner:
     amg:
       print_level: 0
   )yaml");
   drv.set_matrix_from_csr(row_start, row_end, indptr, cols, values);
   drv.set_rhs_from_array(row_start, row_end, rhs);
   drv.set_zero_initial_guess();
   drv.solve();
   hypredrive::finalize();

``hypredrive::driver`` is move-only and destroys its underlying ``HYPREDRV_t`` in
its destructor. Methods throw ``hypredrive::error`` when the underlying C API
returns a nonzero status. The original status is available through
``error::code()``. Call ``hypredrive::describe_error(e.code())`` if a caught
exception should also print the C-side diagnostic. Explicit
``destroy_linear_solver()`` calls are only needed to release solver state before
the driver goes out of scope.

Configuration
-------------

Solver configuration is intentionally YAML-driven. The C++ wrapper does not
mirror the full solver option schema with C++ setters, because the YAML schema is
the project-owned source of truth and supports new HYPREDRV options immediately.
Use ``parse_yaml()`` for YAML from a string, string literal, input stream, or
file path; use ``parse_args(argc, argv)`` for the same command-line style parsing
exposed by the C API. String-like inputs are treated as file paths when they look
like paths, otherwise as inline YAML text. YAML text is passed to the C parser as
a C string, so embedded NUL bytes are rejected.

Example
-------

The ``laplacian-cpp`` example assembles a distributed 3D 7-point Laplacian and
solves it through the wrapper:

.. code-block:: bash

   mpiexec -n 4 ./build-cpp/laplacian-cpp -n 16 16 16 -P 2 2 1 -s 7

Pass a custom YAML file with ``-i/--input options.yml``.

API correspondence
------------------

The C++ interface is a thin throwing wrapper over the public C API. The table
below maps each C++ entry point to the C API it forwards to. C API names link
to the generated API reference.

.. BEGIN CXX API TABLE

.. list-table:: C++ and C API correspondence
   :header-rows: 1
   :widths: 45 55

   * - C++ API
     - C API
   * - ``hypredrive::initialize``
     - :cpp:func:`HYPREDRV_Initialize`
   * - ``hypredrive::finalize``
     - :cpp:func:`HYPREDRV_Finalize`
   * - ``hypredrive::describe_error``
     - :cpp:func:`HYPREDRV_ErrorCodeDescribe`
   * - ``hypredrive::throw_invalid_value``
     - :cpp:func:`HYPREDRV_ErrorInvalidValue`
   * - ``hypredrive::print_lib_info``
     - :cpp:func:`HYPREDRV_PrintLibInfo`
   * - ``hypredrive::print_system_info``
     - :cpp:func:`HYPREDRV_PrintSystemInfo`
   * - ``hypredrive::print_exit_info``
     - :cpp:func:`HYPREDRV_PrintExitInfo`
   * - ``hypredrive::register_solver_preset``
     - :cpp:func:`HYPREDRV_SolverPresetRegister`
   * - ``hypredrive::register_precon_preset``
     - :cpp:func:`HYPREDRV_PreconPresetRegister`
   * - ``hypredrive::driver::driver``
     - :cpp:func:`HYPREDRV_Create`
   * - ``hypredrive::driver::~driver``
     - :cpp:func:`HYPREDRV_Destroy`
   * - ``hypredrive::driver::destroy``
     - :cpp:func:`HYPREDRV_Destroy`
   * - ``hypredrive::driver::parse_args``
     - :cpp:func:`HYPREDRV_InputArgsParse`
   * - ``hypredrive::driver::parse_yaml``
     - :cpp:func:`HYPREDRV_InputArgsParse`
   * - ``hypredrive::driver::set_library_mode``
     - :cpp:func:`HYPREDRV_SetLibraryMode`
   * - ``hypredrive::driver::set_object_name``
     - :cpp:func:`HYPREDRV_ObjectSetName`
   * - ``hypredrive::driver::get_warmup``
     - :cpp:func:`HYPREDRV_InputArgsGetWarmup`
   * - ``hypredrive::driver::get_num_repetitions``
     - :cpp:func:`HYPREDRV_InputArgsGetNumRepetitions`
   * - ``hypredrive::driver::get_num_linear_systems``
     - :cpp:func:`HYPREDRV_InputArgsGetNumLinearSystems`
   * - ``hypredrive::driver::get_num_precon_variants``
     - :cpp:func:`HYPREDRV_InputArgsGetNumPreconVariants`
   * - ``hypredrive::driver::set_precon_variant``
     - :cpp:func:`HYPREDRV_InputArgsSetPreconVariant`
   * - ``hypredrive::driver::set_precon_preset``
     - :cpp:func:`HYPREDRV_InputArgsSetPreconPreset`
   * - ``hypredrive::driver::set_solver_preset``
     - :cpp:func:`HYPREDRV_InputArgsSetSolverPreset`
   * - ``hypredrive::driver::build_system``
     - :cpp:func:`HYPREDRV_LinearSystemBuild`
   * - ``hypredrive::driver::read_matrix``
     - :cpp:func:`HYPREDRV_LinearSystemReadMatrix`
   * - ``hypredrive::driver::set_matrix``
     - :cpp:func:`HYPREDRV_LinearSystemSetMatrix`
   * - ``hypredrive::driver::set_rhs``
     - :cpp:func:`HYPREDRV_LinearSystemSetRHS`
   * - ``hypredrive::driver::set_matrix_from_csr``
     - :cpp:func:`HYPREDRV_LinearSystemSetMatrixFromCSR`
   * - ``hypredrive::driver::set_rhs_from_array``
     - :cpp:func:`HYPREDRV_LinearSystemSetRHSFromArray`
   * - ``hypredrive::driver::set_initial_guess``
     - :cpp:func:`HYPREDRV_LinearSystemSetInitialGuess`
   * - ``hypredrive::driver::set_zero_initial_guess``
     - :cpp:func:`HYPREDRV_LinearSystemSetInitialGuess`
   * - ``hypredrive::driver::set_solution``
     - :cpp:func:`HYPREDRV_LinearSystemSetSolution`
   * - ``hypredrive::driver::set_reference_solution``
     - :cpp:func:`HYPREDRV_LinearSystemSetReferenceSolution`
   * - ``hypredrive::driver::reset_initial_guess``
     - :cpp:func:`HYPREDRV_LinearSystemResetInitialGuess`
   * - ``hypredrive::driver::set_prec_matrix``
     - :cpp:func:`HYPREDRV_LinearSystemSetPrecMatrix`
   * - ``hypredrive::driver::set_dofmap``
     - :cpp:func:`HYPREDRV_LinearSystemSetDofmap`
   * - ``hypredrive::driver::set_interleaved_dofmap``
     - :cpp:func:`HYPREDRV_LinearSystemSetInterleavedDofmap`
   * - ``hypredrive::driver::set_contiguous_dofmap``
     - :cpp:func:`HYPREDRV_LinearSystemSetContiguousDofmap`
   * - ``hypredrive::driver::read_dofmap``
     - :cpp:func:`HYPREDRV_LinearSystemReadDofmap`
   * - ``hypredrive::driver::print_dofmap``
     - :cpp:func:`HYPREDRV_LinearSystemPrintDofmap`
   * - ``hypredrive::driver::print_system``
     - :cpp:func:`HYPREDRV_LinearSystemPrint`
   * - ``hypredrive::driver::set_near_null_space``
     - :cpp:func:`HYPREDRV_LinearSystemSetNearNullSpace`
   * - ``hypredrive::driver::set_null_space``
     - :cpp:func:`HYPREDRV_LinearSystemSetNullSpace`
   * - ``hypredrive::driver::get_solution_values_raw``
     - :cpp:func:`HYPREDRV_LinearSystemGetSolutionValues`
   * - ``hypredrive::driver::get_solution_length``
     - :cpp:func:`HYPREDRV_LinearSystemGetSolutionLength`
   * - ``hypredrive::driver::get_solution_values``
     - :cpp:func:`HYPREDRV_LinearSystemGetSolutionValues`
   * - ``hypredrive::driver::get_solution_norm``
     - :cpp:func:`HYPREDRV_LinearSystemGetSolutionNorm`
   * - ``hypredrive::driver::get_solution``
     - :cpp:func:`HYPREDRV_LinearSystemGetSolution`
   * - ``hypredrive::driver::get_rhs_values``
     - :cpp:func:`HYPREDRV_LinearSystemGetRHSValues`
   * - ``hypredrive::driver::get_rhs``
     - :cpp:func:`HYPREDRV_LinearSystemGetRHS`
   * - ``hypredrive::driver::get_matrix``
     - :cpp:func:`HYPREDRV_LinearSystemGetMatrix`
   * - ``hypredrive::driver::set_state_vector``
     - :cpp:func:`HYPREDRV_StateVectorSet`
   * - ``hypredrive::driver::get_state_vector_values``
     - :cpp:func:`HYPREDRV_StateVectorGetValues`
   * - ``hypredrive::driver::copy_state_vector``
     - :cpp:func:`HYPREDRV_StateVectorCopy`
   * - ``hypredrive::driver::update_all_state_vectors``
     - :cpp:func:`HYPREDRV_StateVectorUpdateAll`
   * - ``hypredrive::driver::apply_state_vector_correction``
     - :cpp:func:`HYPREDRV_StateVectorApplyCorrection`
   * - ``hypredrive::driver::create_precon``
     - :cpp:func:`HYPREDRV_PreconCreate`
   * - ``hypredrive::driver::create_linear_solver``
     - :cpp:func:`HYPREDRV_LinearSolverCreate`
   * - ``hypredrive::driver::setup_precon``
     - :cpp:func:`HYPREDRV_PreconSetup`
   * - ``hypredrive::driver::setup_linear_solver``
     - :cpp:func:`HYPREDRV_LinearSolverSetup`
   * - ``hypredrive::driver::apply_linear_solver``
     - :cpp:func:`HYPREDRV_LinearSolverApply`
   * - ``hypredrive::driver::solve``
     - :cpp:func:`HYPREDRV_LinearSolverCreate`, :cpp:func:`HYPREDRV_LinearSolverSetup`, :cpp:func:`HYPREDRV_LinearSolverApply`
   * - ``hypredrive::driver::apply_precon``
     - :cpp:func:`HYPREDRV_PreconApply`
   * - ``hypredrive::driver::destroy_precon``
     - :cpp:func:`HYPREDRV_PreconDestroy`
   * - ``hypredrive::driver::destroy_linear_solver``
     - :cpp:func:`HYPREDRV_LinearSolverDestroy`
   * - ``hypredrive::driver::print_stats``
     - :cpp:func:`HYPREDRV_StatsPrint`
   * - ``hypredrive::driver::begin_annotation``
     - :cpp:func:`HYPREDRV_AnnotateBegin`
   * - ``hypredrive::driver::end_annotation``
     - :cpp:func:`HYPREDRV_AnnotateEnd`
   * - ``hypredrive::driver::begin_level_annotation``
     - :cpp:func:`HYPREDRV_AnnotateLevelBegin`
   * - ``hypredrive::driver::end_level_annotation``
     - :cpp:func:`HYPREDRV_AnnotateLevelEnd`
   * - ``hypredrive::driver::compute_eigenspectrum``
     - :cpp:func:`HYPREDRV_LinearSystemComputeEigenspectrum`
   * - ``hypredrive::driver::get_linear_solver_num_iter``
     - :cpp:func:`HYPREDRV_LinearSolverGetNumIter`
   * - ``hypredrive::driver::get_linear_solver_converged``
     - :cpp:func:`HYPREDRV_LinearSolverGetConverged`
   * - ``hypredrive::driver::get_linear_solver_final_relative_residual_norm``
     - :cpp:func:`HYPREDRV_LinearSolverGetFinalRelativeResidualNorm`
   * - ``hypredrive::driver::get_linear_solver_setup_time``
     - :cpp:func:`HYPREDRV_LinearSolverGetSetupTime`
   * - ``hypredrive::driver::get_linear_solver_solve_time``
     - :cpp:func:`HYPREDRV_LinearSolverGetSolveTime`
   * - ``hypredrive::driver::get_stats_level_count``
     - :cpp:func:`HYPREDRV_StatsLevelGetCount`
   * - ``hypredrive::driver::get_stats_level_entry``
     - :cpp:func:`HYPREDRV_StatsLevelGetEntry`
   * - ``hypredrive::driver::print_stats_level``
     - :cpp:func:`HYPREDRV_StatsLevelPrint`

.. END CXX API TABLE

``hypredrive::driver::get_solution_values_raw`` returns the same
HYPREDRV-owned host pointer as ``HYPREDRV_LinearSystemGetSolutionValues``. On
GPU builds, calling it migrates/synchronizes the solution to host memory before
the pointer is returned.

Installation
------------

When ``HYPREDRV_ENABLE_CPP=ON``, CMake installs ``hypredrive.hpp`` and
exports the ``HYPREDRV::CXX`` target. Downstream projects should link that target
instead of manually adding include directories.
