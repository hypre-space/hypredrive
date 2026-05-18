.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _MatlabInterface:

MATLAB/Octave Interface
=======================

The MATLAB/Octave interface provides a portable ``hypredrive_solve.m`` wrapper backed
by a compiled MEX entry point named ``hypredrive_mex``. It accepts a real double
sparse matrix, a real double dense right-hand side, and optional solver options.
It returns the dense solution vector and an optional information struct.

The interface is intentionally serial in its first version. MATLAB/Octave sparse
storage is converted from CSC to CSR inside the MEX wrapper, then hypredrive
solves on ``MPI_COMM_SELF``. OpenMP-enabled host HYPRE builds are supported, but
MATLAB/Octave MPI workflows are not exposed by this interface.

Prerequisites
-------------

- MATLAB with ``mex`` or GNU Octave with ``mkoctfile --mex``.
- CMake with the ``FindMatlab`` module when building for MATLAB.
- A real-valued HYPRE build where ``HYPRE_Real`` uses the C ``double`` ABI.
- Non-complex HYPRE. Single-precision, long-double, and complex HYPRE builds are
  rejected when ``HYPREDRV_ENABLE_MATLAB=ON``.
- The MEX file links to the same MPI implementation used to build HYPRE and
  hypredrive. MATLAB can ship or preload its own MPI runtime, so MPI ABI
  mismatches are the most common runtime failure. Prefer launching MATLAB or
  Octave from an environment where the intended MPI ``bin`` and ``lib``
  directories are first on ``PATH`` and the platform runtime library path.

Build
-----

The interface is optional and disabled by default. The same option enables both
runtimes; if MATLAB and Octave are both available, both MEX files are built:

.. code-block:: bash

   cmake -S . -B build-matlab \
     -DHYPREDRV_ENABLE_MATLAB=ON \
     -DHYPREDRV_ENABLE_TESTING=ON
   cmake --build build-matlab --parallel

If MATLAB is not discoverable automatically, pass
``-DMatlab_ROOT_DIR=/path/to/MATLAB``. For Octave, ensure ``mkoctfile`` is on
``PATH``.

Usage
-----

Add the wrapper and the runtime-specific MEX directory to the MATLAB path:

.. code-block:: matlab

   addpath("/path/to/source/interfaces/matlab/src")
   addpath("/path/to/build-matlab/interfaces/matlab/matlab")

For Octave, use the Octave MEX directory:

.. code-block:: matlab

   addpath("/path/to/source/interfaces/matlab/src")
   addpath("/path/to/build-matlab/lib/octave")

The build-tree paths differ because MATLAB MEX files are produced by
``FindMatlab``, while Octave MEX files are produced by ``mkoctfile`` into the
common library tree.

For an installed prefix, add ``/path/to/install/lib/matlab`` in MATLAB or
``/path/to/install/lib/octave`` in Octave. To make that persistent, put the
corresponding ``addpath`` command in MATLAB's ``startup.m`` or Octave's startup
file.

As a one-step setup from an installed prefix, run:

.. code-block:: matlab

   run("/path/to/install/lib/matlab/hypredrive_setup.m")

or, for Octave:

.. code-block:: matlab

   run("/path/to/install/lib/octave/hypredrive_setup.m")

Solve a sparse system:

.. code-block:: matlab

   n = 64;
   e = ones(n, 1);
   A = spdiags([-e, 2*e, -e], -1:1, n, n);
   b = ones(n, 1);

   [x, info] = hypredrive_solve(A, b);
   relres = norm(b - A*x) / norm(b)
   info.iterations

Custom solver options should usually be built with ``hypredrive_options``:

.. code-block:: matlab

   opts = hypredrive_options( ...
       'solver', 'pcg', ...
       'preconditioner', 'amg', ...
       'pcg', struct('max_iter', 200, 'relative_tol', 1.0e-10), ...
       'amg', struct('print_level', 0));

   [x, info] = hypredrive_solve(A, b, opts);

The struct form mirrors hypredrive YAML:

.. code-block:: matlab

   opts = struct();
   opts.solver.pcg.max_iter = 200;
   opts.solver.pcg.relative_tol = 1.0e-10;
   opts.preconditioner.amg.print_level = 0;

   [x, info] = hypredrive_solve(A, b, opts);

MATLAB/Octave usage is quiet by default: ``hypredrive_options`` adds
``general.statistics = 0``, and the no-options path uses the same policy.
Request the HYPREDRV statistics table explicitly when needed:

.. code-block:: matlab

   opts = hypredrive_options(struct('general', struct('statistics', 1)));

Raw YAML text is still accepted for advanced users. The default options are PCG
with AMG preconditioning and statistics disabled.

Information struct
------------------

When requested, the second output is a struct with these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``iterations``
     - Solver iteration count reported by hypredrive.
   * - ``setup_time``
     - Linear-solver setup time in seconds.
   * - ``solve_time``
     - Linear-solver apply/solve time in seconds.
   * - ``solution_norm``
     - L2 norm of the returned solution vector.

Testing
-------

When testing is enabled, the ``matlab-test`` target runs a small serial
Laplacian smoke test for each available runtime:

.. code-block:: bash

   cmake --build build-matlab --target matlab-test

Installation
------------

MATLAB files are installed under ``lib/matlab`` and Octave files are installed
under ``lib/octave``:

.. code-block:: bash

   cmake --install build-matlab --prefix /path/to/install

Add ``/path/to/install/lib/matlab`` to MATLAB's path, or
``/path/to/install/lib/octave`` to Octave's path, before calling ``hypredrive_solve``.

On Unix-like systems, installed MEX files use a relative runtime search path
back to the prefix-local library directory. If you relocate the MEX file
manually, keep it one directory below the installed library directory or set the
platform runtime library path so ``libHYPREDRV`` and HYPRE are visible.

Installed examples are placed under ``share/matlab/examples``. For example,
``share/matlab/examples/laplacian1d.m`` solves a small serial 1D Laplacian.

Troubleshooting
---------------

- ``libmpi`` load errors usually mean MATLAB/Octave is not seeing the same MPI
  runtime used by HYPRE. Start the runtime from a shell where the intended MPI
  installation is active.
- ``Invalid MEX-file`` errors usually mean a missing shared-library dependency.
  Check the MEX file with ``ldd`` on Linux or ``otool -L`` on macOS.
- If options fail to parse, build them with ``hypredrive_options`` first and
  print the resulting YAML string before calling ``hypredrive_solve``.
