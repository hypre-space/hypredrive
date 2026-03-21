.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _APIReference:

API Reference
=============

This chapter documents every public function exported by ``libHYPREDRV``. The reference
below is generated directly from the Doxygen comments in ``include/HYPREDRV.h``.

Why use ``libHYPREDRV``?
------------------------

The ``hypredrive`` executable is well-suited for file-based workflows: provide a matrix, a
YAML configuration, and get a solution. ``libHYPREDRV`` serves the complementary use case:
your application already has an assembled ``HYPRE_IJMatrix``/``HYPRE_IJVector`` in memory
and needs a low-overhead path to HYPRE's solver stack, configurable at runtime from a YAML
string or file.

Typical benefits:

- **Runtime configurability** — swap solvers and preconditioners without recompiling; pass
  different YAML strings in different code paths.
- **Minimal boilerplate** — three to five ``HYPREDRV_*`` calls replace dozens of direct
  HYPRE API calls for common solver setups.
- **Statistics and timing** — built-in timing infrastructure produces structured output for
  benchmarking; reuse statistics across repetitions.
- **Preconditioner reuse** — attach a long-lived preconditioner object across multiple
  right-hand sides or timesteps without rebuilding the setup phase.

Object Lifecycle
----------------

A typical library-mode program follows this order:

.. code-block:: text

   HYPREDRV_Initialize()
   HYPREDRV_Create(comm, &h)
   HYPREDRV_SetLibraryMode(h)           ← must precede InputArgsParse
   HYPREDRV_InputArgsParse(1, &yaml_str, h)  ← yaml_str is a char* YAML string (or a filename)
   [assemble HYPRE_IJMatrix / HYPRE_IJVector]
   HYPREDRV_LinearSystemSetMatrix(h, A)
   HYPREDRV_LinearSystemSetRHS(h, b)
   HYPREDRV_LinearSystemSetInitialGuess(h, x0)  ← pass NULL for default
   HYPREDRV_PreconCreate(h)
   HYPREDRV_LinearSolverCreate(h)
   HYPREDRV_LinearSolverSetup(h)
   HYPREDRV_LinearSolverApply(h)
   HYPREDRV_LinearSolverDestroy(h)      ← also destroys preconditioner
   HYPREDRV_StatsPrint(h)
   HYPREDRV_Destroy(&h)
   HYPREDRV_Finalize()

Key rules:

- **``HYPREDRV_SetLibraryMode``** must be called before ``HYPREDRV_InputArgsParse``. It
  signals that the application owns the externally created HYPRE objects (matrices, vectors)
  passed via ``LinearSystemSet*``: hypredrive borrows them and will never destroy them.
  Without this call (driver mode), hypredrive takes ownership and frees objects on
  replacement or teardown.
- **``HYPREDRV_InputArgsParse``** parses YAML and applies solver/preconditioner settings. In
  driver mode (without ``SetLibraryMode``) it also sets the HYPRE memory and execution
  policy from the ``general.exec_policy`` field.
- **``HYPREDRV_LinearSolverDestroy``** destroys both the solver and the associated
  preconditioner. There is no need to call ``HYPREDRV_PreconDestroy`` separately after this
  call. Only call ``HYPREDRV_PreconDestroy`` directly when you want to tear down the
  preconditioner while keeping the solver alive.

Function Groups
---------------

The full API is organized into the following groups:

- **Lifecycle** — ``Initialize``, ``Create``, ``Destroy``, ``Finalize``, ``SetLibraryMode``
- **Input parsing** — ``InputArgsParse``, ``InputArgsSet*``, ``InputArgsGet*``,
  ``PreconPresetRegister``
- **Linear system** — ``LinearSystemSet*``, ``LinearSystemGet*``, ``LinearSystemBuild``,
  ``LinearSystemReset*``
- **Solver** — ``LinearSolverCreate``, ``LinearSolverSetup``, ``LinearSolverApply``,
  ``LinearSolverDestroy``, ``LinearSolverGet*``
- **Preconditioner** — ``PreconCreate``, ``PreconSetUp``, ``PreconApply``, ``PreconDestroy``
- **Statistics** — ``StatsPrint``, ``StatsLevel*``
- **Annotation** (Caliper) — ``AnnotateBegin``, ``AnnotateEnd``

See :ref:`LibraryExamples` for worked examples that use this API.

Full Reference
--------------

.. doxygengroup:: HYPREDRV
   :project: hypredrive
   :content-only:
