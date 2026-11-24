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

The first interface version is serial. The MEX wrapper converts MATLAB/Octave
sparse storage from CSC to CSR. Then hypredrive solves on ``MPI_COMM_SELF``.
The interface supports OpenMP-enabled host HYPRE builds, but it does not expose
MATLAB/Octave MPI workflows.

Prerequisites
-------------

- MATLAB with ``mex`` or GNU Octave with ``mkoctfile --mex``.
- CMake with the ``FindMatlab`` module when building for MATLAB.
- A real-valued HYPRE build where ``HYPRE_Real`` uses the C ``double`` ABI.
- Non-complex HYPRE. Single-precision, long-double, and complex HYPRE builds are
  rejected when ``HYPREDRV_ENABLE_MATLAB=ON``.
- The MEX file links to the MPI implementation that built HYPRE and `hypredrive`.
  MATLAB can load a different MPI run time. An MPI ABI mismatch then causes a common
  run-time failure. Start MATLAB or Octave in the intended MPI environment. Put its
  ``bin`` and ``lib`` directories first in the applicable search paths.

Build
-----

CMake disables the optional interface by default. The same option enables both run
times. If MATLAB and Octave are available, CMake builds both MEX files:

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

   addpath("/path/to/source/interfaces/matlab-octave/src")
   addpath("/path/to/build-matlab/interfaces/matlab/matlab")

For Octave, use the Octave MEX directory:

.. code-block:: matlab

   addpath("/path/to/source/interfaces/matlab-octave/src")
   addpath("/path/to/build-matlab/lib/octave")

The build-tree paths differ because ``FindMatlab`` produces MATLAB MEX files.
``mkoctfile`` produces Octave MEX files in the common library tree.

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

Minimal example
---------------

This serial example solves ``A x = b`` for ``A = [[2, -1], [-1, 2]]`` and
``b = [1, 1]``, giving ``x = [1, 1]``.

After adding the wrapper and MEX directories to the path as above:

.. code-block:: matlab

   A = sparse([2.0, -1.0; -1.0, 2.0]);
   b = ones(2, 1);
   [x, info] = hypredrive_solve(A, b);
   disp(x)  % [1; 1]
   disp(info.converged)

Solver options
--------------

Use ``hypredrive_options`` to build custom solver options:

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

Examples
--------

The examples are regular MATLAB/Octave scripts that call ``hypredrive_solve``:

.. code-block:: matlab

   addpath("/path/to/source/interfaces/matlab-octave/examples")

   laplacian(16, 16, 16)
   elasticity(16, 16, 16)

``laplacian.m`` assembles 1D, 2D, or 3D finite-difference Poisson problems.
``elasticity.m`` assembles 1D, 2D, or 3D linear-elasticity problems and uses
the matching elasticity AMG preset. A test or script can call
``build_laplacian.m`` or ``build_elasticity.m`` when it needs only the matrix and
right-hand side.

Handle-based API
----------------

Besides the one-shot solve, ``hypredrive_mex('command', ...)`` exposes every public
``HYPREDRV_`` function; the one-shot form ``hypredrive_mex(A, b, yaml)`` is
unchanged. Drivers, IJ matrices, and IJ vectors are opaque scalar ``uint64``
handles that are validated on every call: stale, destroyed, forged, or wrong-kind
handles raise ``hypredrive:InvalidHandle``. ``hypredrive_mex('commands')`` lists
the available commands.

.. code-block:: matlab

   drv = hypredrive_mex('create', hypredrive_options('solver', 'pcg', ...
                        'preconditioner', 'amg', 'amg', struct('print_level', 0)));
   hypredrive_mex('set_matrix_csr', drv, A);
   hypredrive_mex('set_rhs_array', drv, b);
   hypredrive_mex('solver_create', drv);
   hypredrive_mex('solver_setup', drv);
   hypredrive_mex('solver_apply', drv);
   x    = hypredrive_mex('copy_solution_values', drv);
   info = hypredrive_mex('solver_info', drv);
   hypredrive_mex('destroy', drv);

Lifetime rules: ``destroy`` on a driver also invalidates borrowed handles issued
from it (``get_matrix``, ``get_rhs``, ``get_solution``). Destroying an owned
matrix or vector that is still attached to a live driver raises
``hypredrive:ObjectInUse``. ``destroy_all`` and the ``mexAtExit`` hook release
every handle, drivers first.

The full command-to-C-function table is in ``interfaces/matlab-octave/README.md``.
Commands cover input arguments and presets, linear-system setters and getters
(sparse matrices and dense vectors in MATLAB-native form, or raw IJ handles for
the ``HYPRE_Matrix``/``HYPRE_Vector`` APIs), dof maps, null spaces, state vectors,
preconditioner and solver control, solver status queries, statistics, and
annotations. Only real-double, host-memory, serial use is supported. The
interface was syntax-checked but not run on MATLAB or Octave when it was added;
run ``matlab-test`` to validate it.

Information struct
------------------

When requested, the second output is a struct with these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``iterations``
     - Solver iteration count reported by hypredrive.
   * - ``converged``
     - Logical convergence flag from the last solve (true when the solver reached its tolerance).
   * - ``final_res_norm``
     - Final relative residual norm from the last solve.
   * - ``setup_time``
     - Linear-solver setup time in seconds.
   * - ``solve_time``
     - Linear-solver apply/solve time in seconds.
   * - ``solution_norm``
     - L2 norm of the returned solution vector.

Testing
-------

When you enable testing, the ``matlab-test`` target runs the MEX wrapper smoke
test plus small serial Laplacian and elasticity examples for each available
runtime:

.. code-block:: bash

   cmake --build build-matlab --target matlab-test

Installation
------------

CMake installs MATLAB files under ``lib/matlab`` and Octave files under
``lib/octave``:

.. code-block:: bash

   cmake --install build-matlab --prefix /path/to/install

Add ``/path/to/install/lib/matlab`` to MATLAB's path, or
``/path/to/install/lib/octave`` to Octave's path, before calling ``hypredrive_solve``.

On Unix-like systems, installed MEX files use a relative run-time search path. This path
points to the library directory in the installation prefix. After you move a MEX file,
keep it one directory below this library directory. Alternatively, add ``libHYPREDRV``
and HYPRE to the platform run-time library path.

CMake installs examples under ``share/matlab/examples``. This directory includes
``laplacian.m``, ``elasticity.m``, ``build_laplacian.m``, and
``build_elasticity.m``.

Troubleshooting
---------------

- ``libmpi`` load errors usually mean MATLAB/Octave is not seeing the same MPI
  runtime used by HYPRE. Start the runtime from a shell where the intended MPI
  installation is active.
- ``Invalid MEX-file`` errors usually mean a missing shared-library dependency.
  Check the MEX file with ``ldd`` on Linux or ``otool -L`` on macOS.
- If options fail to parse, build them with ``hypredrive_options`` first and
  print the resulting YAML string before calling ``hypredrive_solve``.
