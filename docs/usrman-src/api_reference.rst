.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _APIReference:

API Reference
=============

This chapter documents every public function that ``libHYPREDRV`` exports. Doxygen
comments in ``include/HYPREDRV.h`` generate the detailed reference.

Why use ``libHYPREDRV``?
------------------------

The ``hypredrive-cli`` executable supports file-based workflows. It reads a matrix and a
YAML configuration, and then it calculates a solution. ``libHYPREDRV`` supports applications
that already have HYPRE matrices and vectors in memory. A YAML string or file configures
the HYPRE solver stack at run time.

Typical benefits:

- **Run-time configuration.** Change solvers and preconditioners without compilation.
  Different code paths can use different YAML strings.
- **Small integration layer.** Three to five ``HYPREDRV_*`` calls replace many direct HYPRE
  calls for common solver setups.
- **Statistics and timing.** The library produces structured timing data and keeps
  statistics across repetitions.
- **Preconditioner reuse.** Keep one preconditioner for multiple right-hand sides or time
  steps without another setup phase.

Usage Notes
-----------

For a full library-mode walkthrough and complete code examples, see
:ref:`LibraryExamples`.

Follow these two rules when you use the application programming interface (API) directly:

- Call ``HYPREDRV_SetLibraryMode`` before ``HYPREDRV_InputArgsParse`` when your application
  owns the HYPRE matrices and vectors.
- Do not call ``HYPREDRV_PreconDestroy`` after ``HYPREDRV_LinearSolverDestroy`` unless you
  manage the preconditioner separately. The solver destroy function also destroys its
  preconditioner.

An embedded application can keep multiple ``HYPREDRV_t`` objects active. Use the applicable
handle with the annotation and hierarchical statistics APIs. The ``HYPREDRV_Annotate*`` and
``HYPREDRV_StatsLevel*`` functions record or query state for that handle.

Reference by Topic
------------------

The following groups organize the public API by workflow stage. The generated
Doxygen section contains detailed documentation for each function.

Core Type
~~~~~~~~~

- :cpp:type:`HYPREDRV_t` - Opaque handle that owns the hypredrive configuration, objects, and runtime state.

Lifecycle and Setup
~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_Initialize` - Initialize hypredrive and its underlying HYPRE runtime state.
- :cpp:func:`HYPREDRV_Create` - Create a new hypredrive object bound to an MPI communicator.
- :cpp:func:`HYPREDRV_SetLibraryMode` - Mark externally provided HYPRE objects as borrowed rather than owned.
- :cpp:func:`HYPREDRV_ObjectSetName` - Attach or clear an optional display name for a specific object.
- :cpp:func:`HYPREDRV_Destroy` - Destroy a hypredrive object and release its managed resources.
- :cpp:func:`HYPREDRV_Finalize` - Auto-destroy any remaining live handles and tear down global runtime state.
- :cpp:func:`HYPREDRV_ErrorCodeDescribe` - Print a human-readable description for a hypredrive error code.
- :cpp:func:`HYPREDRV_PrintLibInfo` - Print version and startup information for hypredrive and hypre.
- :cpp:func:`HYPREDRV_PrintSystemInfo` - Print detected machine and software environment information.
- :cpp:func:`HYPREDRV_PrintExitInfo` - Print shutdown information for a driver-style run.

Input Parsing and Presets
~~~~~~~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_InputArgsParse` - Parse YAML or argv-style input and apply the resulting configuration.
- :cpp:func:`HYPREDRV_InputArgsGetWarmup` - Get the warmup setting.
- :cpp:func:`HYPREDRV_InputArgsGetNumRepetitions` - Get the configured number of repetitions.
- :cpp:func:`HYPREDRV_InputArgsGetNumLinearSystems` - Get the configured number of linear systems.
- :cpp:func:`HYPREDRV_InputArgsGetNumPreconVariants` - Get the number of preconditioner variants.
- :cpp:func:`HYPREDRV_InputArgsSetPreconVariant` - Select the active preconditioner variant by index.
- :cpp:func:`HYPREDRV_InputArgsSetPreconPreset` - Apply a named preconditioner preset to the active variant.
- :cpp:func:`HYPREDRV_InputArgsSetSolverPreset` - Apply a named solver preset without reparsing YAML.
- :cpp:func:`HYPREDRV_SolverPresetRegister` - Register a custom named solver preset from YAML text.
- :cpp:func:`HYPREDRV_PreconPresetRegister` - Register a custom named preconditioner preset from YAML text.

Linear System Setup
~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_LinearSystemBuild` - Build the matrix and vectors from the parsed input configuration.
- :cpp:func:`HYPREDRV_LinearSystemComputeEigenspectrum` - Compute the eigenspectrum of the current matrix when enabled.
- :cpp:func:`HYPREDRV_LinearSystemReadMatrix` - Read the system matrix from file input.
- :cpp:func:`HYPREDRV_LinearSystemSetMatrix` - Attach a user-provided system matrix.
- :cpp:func:`HYPREDRV_LinearSystemSetRHS` - Attach a user-provided right-hand side vector.
- :cpp:func:`HYPREDRV_LinearSystemSetInitialGuess` - Set or rebuild the initial guess vector.
- :cpp:func:`HYPREDRV_LinearSystemSetSolution` - Set the vector that receives the solver result.
- :cpp:func:`HYPREDRV_LinearSystemSetReferenceSolution` - Set the reference solution used for error-aware workflows.
- :cpp:func:`HYPREDRV_LinearSystemResetInitialGuess` - Restore the initial guess to its configured original state.
- :cpp:func:`HYPREDRV_LinearSystemSetPrecMatrix` - Set the matrix used for preconditioner setup.
- :cpp:func:`HYPREDRV_LinearSystemSetDofmap` - Set an explicit local degree-of-freedom map.
- :cpp:func:`HYPREDRV_LinearSystemSetInterleavedDofmap` - Build an interleaved degree-of-freedom map.
- :cpp:func:`HYPREDRV_LinearSystemSetContiguousDofmap` - Build a contiguous degree-of-freedom map.
- :cpp:func:`HYPREDRV_LinearSystemReadDofmap` - Read the degree-of-freedom map from file input.
- :cpp:func:`HYPREDRV_LinearSystemPrintDofmap` - Write the current degree-of-freedom map to text output.
- :cpp:func:`HYPREDRV_LinearSystemPrint` - Print the current matrix, RHS, and DOF map to files.
- :cpp:func:`HYPREDRV_LinearSystemSetNearNullSpace` - Attach near-nullspace vectors such as rigid-body modes.
- :cpp:func:`HYPREDRV_LinearSystemSetNullSpace` - Attach exact null space modes that
  hypredrive projects out of computed solutions.

Linear System Accessors
~~~~~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_LinearSystemGetSolutionValues` - Get a host pointer to the local solution values array.
- :cpp:func:`HYPREDRV_LinearSystemGetSolutionNorm` - Compute a named norm of the current solution vector.
- :cpp:func:`HYPREDRV_LinearSystemGetSolution` - Get the solution vector object.
- :cpp:func:`HYPREDRV_LinearSystemGetRHSValues` - Get a pointer to the local RHS values array.
- :cpp:func:`HYPREDRV_LinearSystemGetRHS` - Get the right-hand side vector object.
- :cpp:func:`HYPREDRV_LinearSystemGetMatrix` - Get the system matrix object.

State Vectors
~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_StateVectorSet` - Register a set of state vectors for multi-state workflows.
- :cpp:func:`HYPREDRV_StateVectorGetValues` - Get the local values pointer for a state vector.
- :cpp:func:`HYPREDRV_StateVectorCopy` - Copy one logical state vector into another.
- :cpp:func:`HYPREDRV_StateVectorUpdateAll` - Rotate or update the registered state vectors after a step.
- :cpp:func:`HYPREDRV_StateVectorApplyCorrection` - Add the current solver correction into a target state vector.

Solver and Preconditioner
~~~~~~~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_PreconCreate` - Create the configured preconditioner object.
- :cpp:func:`HYPREDRV_LinearSolverCreate` - Create the configured linear solver object.
- :cpp:func:`HYPREDRV_PreconSetup` - Set up the preconditioner for standalone application.
- :cpp:func:`HYPREDRV_LinearSolverSetup` - Set up the solver and its associated preconditioner.
- :cpp:func:`HYPREDRV_LinearSolverApply` - Run the configured solver on the current linear system.
- :cpp:func:`HYPREDRV_PreconApply` - Apply the configured preconditioner as an operator.
- :cpp:func:`HYPREDRV_PreconDestroy` - Destroy the current preconditioner object.
- :cpp:func:`HYPREDRV_LinearSolverDestroy` - Destroy the current solver and its associated preconditioner.

Statistics and Timing
~~~~~~~~~~~~~~~~~~~~~

- :cpp:func:`HYPREDRV_StatsLevelGetCount` - Get how many hierarchical stats entries exist at a level.
- :cpp:func:`HYPREDRV_StatsLevelGetEntry` - Get one hierarchical stats entry by level and index.
- :cpp:func:`HYPREDRV_StatsLevelPrint` - Print the hierarchical stats summary for a level.
- :cpp:func:`HYPREDRV_StatsPrint` - Print the collected statistics for the current object.
- :cpp:func:`HYPREDRV_LinearSolverGetNumIter` - Get the iteration count from the last solve.
- :cpp:func:`HYPREDRV_LinearSolverGetConverged` - Get the convergence flag from the last solve.
- :cpp:func:`HYPREDRV_LinearSolverGetFinalRelativeResidualNorm` - Get the final relative residual norm from the last solve.
- :cpp:func:`HYPREDRV_LinearSolverGetSetupTime` - Get the setup time from the last solve.
- :cpp:func:`HYPREDRV_LinearSolverGetSolveTime` - Get the solve time from the last solve.

Annotation
~~~~~~~~~~

- :cpp:func:`HYPREDRV_AnnotateBegin` - Begin a named annotation region.
- :cpp:func:`HYPREDRV_AnnotateEnd` - End a named annotation region.
- :cpp:func:`HYPREDRV_AnnotateLevelBegin` - Begin a hierarchical annotation region at a given level.
- :cpp:func:`HYPREDRV_AnnotateLevelEnd` - End a hierarchical annotation region at a given level.

See :ref:`LibraryExamples` for worked examples that use this API.

Detailed API Reference
----------------------

.. doxygengroup:: HYPREDRV
   :project: hypredrive
   :content-only:
