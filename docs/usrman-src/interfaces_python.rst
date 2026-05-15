.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _PythonInterface:

Python Interface
================

The Python interface provides thin bindings to the hypredrive C library. It accepts
solver options as Python dictionaries, YAML strings, or YAML files; assembles sparse
matrices from CSR data or SciPy CSR matrices; runs the solve lifecycle; and returns the
local solution as a NumPy array.

The heavy lifting still happens in libHYPREDRV and HYPRE. The Python layer mainly handles
input normalization, dtype checks, YAML conversion, and result extraction.

Prerequisites
-------------

- Python 3.9 or newer.
- NumPy.
- Cython 3.0 or newer when building from source.
- An MPI implementation with compiler wrappers available at build time.
- ``mpi4py`` for distributed solves from Python.
- SciPy when passing ``scipy.sparse.csr_matrix`` objects directly.

Installation
------------

The recommended package path is to build from ``interfaces/python`` with
``scikit-build-core``. This keeps Python packaging independent from ordinary C builds
while still linking against libHYPREDRV.

Against an installed hypredrive:

.. code-block:: bash

   cmake --install build --prefix $HOME/opt/hypredrive
   pip install ./interfaces/python \
     --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive

Against an in-tree development build:

.. code-block:: bash

   cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DHYPREDRV_ENABLE_TESTING=OFF
   cmake --build build --parallel
   pip install -e ./interfaces/python \
     --config-settings=cmake.define.HYPREDRV_DIR=$PWD/build

The top-level CMake build can also build the Python extension directly:

.. code-block:: bash

   cmake -S . -B build -DHYPREDRV_ENABLE_PYTHON=ON
   cmake --build build --target _native --parallel

This mode is intended for developer and CI builds. It requires Python, NumPy, and Cython
at configure/build time, so ``HYPREDRV_ENABLE_PYTHON`` is disabled by default.

Optional dependencies can be installed through package extras:

.. code-block:: bash

   pip install ./interfaces/python[mpi]
   pip install ./interfaces/python[scipy]
   pip install -e ./interfaces/python[test]

Quick start
-----------

One-shot solve
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import scipy.sparse as sp
   import hypredrive as hd

   n = 64
   diag_main = 2.0 * np.ones(n)
   diag_off = -np.ones(n - 1)
   A = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")
   b = np.ones(n)

   options = hd.configure(
       solver="pcg",
       preconditioner="amg",
       pcg={"max_iter": 100, "relative_tol": 1.0e-8},
       amg={"print_level": 0},
   )

   result = hd.solve(A, b, options=options)

   print("solution norm:", result.solution_norm)
   print("first entries:", result.x[:5])

Reusable driver
~~~~~~~~~~~~~~~

Use ``HypreDrive`` when one process needs to solve multiple related systems and reuse the
driver lifecycle.

.. code-block:: python

   import hypredrive as hd

   with hd.HypreDrive(options="my_config.yaml") as drv:
       for step in range(num_steps):
           drv.set_matrix_from_csr(build_matrix(step))
           drv.set_rhs(build_rhs(step))
           drv.solve()
           x = drv.get_solution()

Distributed solve with MPI
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``mpi4py``, each rank provides its local CSR slab. Row bounds are global and
inclusive; column indices are global.

.. code-block:: python

   from mpi4py import MPI
   import hypredrive as hd

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()

   indptr, col_indices, data, rhs, row_start, row_end = build_local_slab(rank, size)

   with hd.HypreDrive(options=opts, comm=comm) as drv:
       drv.set_matrix_from_csr(
           indptr,
           col_indices,
           data,
           row_start=row_start,
           row_end=row_end,
       )
       drv.set_rhs(rhs)
       drv.solve()
       x_local = drv.get_solution()

Configuration input
-------------------

Anywhere ``options`` is accepted, pass one of:

.. list-table::
   :header-rows: 1

   * - Input
     - Behavior
   * - ``dict``
     - Converted to YAML in memory and parsed by hypredrive.
   * - ``str`` containing a newline
     - Treated as a YAML document.
   * - ``str`` or ``pathlib.Path`` naming an existing file
     - File contents are loaded and parsed.
   * - ``None``
     - Uses the Python binding's minimal default YAML.

The accepted YAML keys and solver/preconditioner options are the same as the CLI; see
:ref:`InputFileStructure`.

Testing
-------

After installing the package in editable mode with test dependencies:

.. code-block:: bash

   pip install -e ./interfaces/python[test]
   python -m pytest interfaces/python/tests/test_solve_serial.py -v

MPI tests must be launched under an MPI process manager:

.. code-block:: bash

   mpirun -np 2 python -m pytest interfaces/python/tests/test_solve_mpi.py -v

Tests that require the native extension or ``mpi4py`` skip when those optional runtime
components are unavailable.

Current limitations
-------------------

- The Python interface currently targets real-valued solves.
- Solution data is copied back to host NumPy arrays.
- GPU/device execution is not exposed as a Python-native data path.
- Result metadata is intentionally small; the one-shot API currently exposes the solution
  array and solution norm.
- ``mpi4py`` integration is optional and uses ``Comm.py2f()`` plus the C-side
  ``MPI_Comm_f2c`` bridge.
