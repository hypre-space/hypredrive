.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Interfaces_Julia:

Julia Interface
===============

The Julia interface is an in-tree package under ``interfaces/julia``. It supports source
trees and installation prefixes. A separate Julia package repository is not required.
The interface requires Julia 1.9 or newer. The package and module use the Julia name
``HypreDrive``.

Build
-----

Enable the interface with CMake:

.. code-block:: bash

   cmake -S . -B build-julia \
     -DHYPREDRV_ENABLE_JULIA=ON \
     -DHYPREDRV_ENABLE_TESTING=ON
   cmake --build build-julia --target julia-test --parallel

To validate that an installed HYPREDRV tree exports a usable Julia bridge target:

.. code-block:: bash

   cmake --build build-julia --target julia-install-consumer-test --parallel

The CMake build creates a small ``libHYPREDRV_Julia`` bridge library. The bridge hides
HYPRE application binary interface (ABI) details from Julia. These details include the
``HYPRE_BigInt`` width and real precision. Julia loads this bridge with ``Libdl``. Thus,
CMake always builds the bridge as a shared library.

External static HYPRE archives require
``-fPIC``. Automatically retrieved HYPRE builds use position-independent code.

CMake installs the bridge under ``lib/julia``. Thus, language-specific shared libraries
do not add files to the top-level library directory. Its install RPATH includes that
directory and the parent ``lib`` directory so it can find ``libHYPREDRV`` from the
same prefix without ``LD_LIBRARY_PATH``.

CMake prints the Julia executable it found. If Julia is not found, the ``julia-test``
target fails with a clear message when run.

Use from Julia
--------------

Develop the in-tree package:

.. code-block:: julia

   import Pkg
   Pkg.develop(path="interfaces/julia")

From outside the repository root, Julia's subdirectory package support can install
the package sources directly:

.. code-block:: julia

   import Pkg
   Pkg.add(Pkg.PackageSpec(url="https://github.com/hypre-space/hypredrive.git",
                           subdir="interfaces/julia"))

This installs the Julia sources only. Until a binary artifact/JLL is published, users
still need a compatible HYPREDRV build or install prefix.

Point the package at the CMake-built bridge library:

.. code-block:: bash

   export HYPREDRV_LIBRARY=$PWD/build-julia/interfaces/julia/lib/libHYPREDRV_Julia.so

On macOS, use the ``.dylib`` extension instead of ``.so``.

After you install hypredrive, you can set ``HYPREDRV_DIR`` instead:

.. code-block:: bash

   export HYPREDRV_DIR=/path/to/hypredrive/install

For Julia, ``HYPREDRV_DIR`` identifies the HYPREDRV installation prefix. It does not
identify the CMake package directory that contains ``HYPREDRVConfig.cmake``. The package
searches ``lib/julia``, ``lib``, ``lib64/julia``, and ``lib64`` below that prefix.

Without these environment variables, the package first searches ``Artifacts.toml`` for
``hypredrive_mpi_trampoline``. It then searches the source tree and calls
``Libdl.find_library``. The artifact contains the HYPREDRV Julia bridge for MPItrampoline.
The ``MPItrampoline_jll`` package supplies MPItrampoline.

Example
-------

.. code-block:: julia

   using HypreDrive
   using SparseArrays

   n = 64
   A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
   b = ones(n)

   x, info = hypredrive_solve(A, b)
   println(info.iterations)

Use Julia keyword arguments to construct options instead of hand-written YAML:

.. code-block:: julia

   opts = hypredrive_options(
       solver=:pcg,
       preconditioner=:amg,
       pcg=(max_iter=200, relative_tol=1.0e-10),
       amg=(print_level=0,),
   )

   x, info = hypredrive_solve(A, b; options=opts)

The generic aliases ``solve``, ``solve_mpi``, ``initialize``, and ``shutdown`` remain
available for qualified calls such as ``HypreDrive.solve(...)``. The package does
not export them.

MPI
---

The package also supports MPI execution through ``hypredrive_solve_mpi``:

.. code-block:: bash

   mpiexec -n 2 julia --project=interfaces/julia interfaces/julia/test/mpi.jl

``hypredrive_solve_mpi(A, b)`` creates the HYPREDRV driver on ``MPI_COMM_WORLD``,
partitions the rows of the global Julia sparse matrix across ranks, and returns each
rank's local solution segment. This convenience API is useful for correctness tests and
integration work.

The explicit MPI helper names ``hypredrive_mpi_world_rank``,
``hypredrive_mpi_world_size``, and ``hypredrive_mpi_world_sum`` operate on
``MPI_COMM_WORLD``. The older qualified aliases remain available but are not exported.

``Pkg.test()`` runs only the serial package tests. CMake and CTest run the MPI
test because it requires ``mpiexec``.

Standalone Laplacian example
----------------------------

The standalone example follows the C Laplacian driver's command-line shape and assembles
only the local CSR slab on each rank:

.. code-block:: bash

   mpiexec -n 2 julia --project=interfaces/julia \
     interfaces/julia/examples/laplacian.jl \
     -n 12 12 12 -P 2 1 1 -s 7 -ns 2

Use ``-i options.yml`` to provide a hypredrive YAML solver configuration. Without
``-i``, the example uses quiet PCG+AMG defaults. The ``-P`` topology controls the 3D
block partition used to assemble local matrix rows.

Standalone Darcy example
------------------------

The standalone Darcy example follows the C mixed RT0/P0 Darcy driver and uses
the Julia MPI CSR interface with a field dofmap for MGR:

.. code-block:: bash

   mpiexec -n 2 julia --project=interfaces/julia \
     interfaces/julia/examples/darcy.jl \
     -n 4 3 1 -P 1 2 1 -g y -v 1

For constant permeability, the example reports the relative pressure L2 error
against the analytic pressure-drop solution. It also accepts heterogeneous
SPE10-style permeability files and VTK output:

.. code-block:: bash

   mpiexec -n 2 julia --project=interfaces/julia \
     interfaces/julia/examples/darcy.jl \
     -n 8 8 4 -P 1 1 2 \
     --K-file data/spe10_case2a/spe_perm.dat \
     --K-file-grid 60 220 85 \
     --K-file-k-order top-down \
     -g y -v 1 \
     --output darcy_spe10_julia.vti

Use ``-i options.yml`` to provide a solver configuration. Alternatively, put
`hypredrive` command-line overrides after ``-a`` or ``--args``. For example, use
``-a --solver:gmres:max_iter 100``. The ``laplacian.jl`` and ``laplacian1d.jl``
examples accept the same overrides. The ``-P`` topology controls the Cartesian MPI
rank grid. Set the product of its entries to the MPI size.

Scope
-----

The interface uses ``MPI_COMM_SELF`` for serial solves and ``MPI_COMM_WORLD`` for MPI
solves. It supports real-valued, double-precision HYPRE builds. The interface rejects
complex, single-precision, and long-double builds at run time. One Julia process cannot
safely contain multiple independent owners of the process-wide MPI state.

Release maintainers can register the package in Julia's General registry from
``interfaces/julia``. Use a normal Julia semantic version such as ``0.2.0`` for a
registry release. Do not use a development version string. The fallback distribution
model uses a source tree or installation prefix. Users build HYPREDRV from this
checkout or install HYPREDRV separately and set
``HYPREDRV_DIR`` or ``HYPREDRV_LIBRARY``.

Julia artifacts permit binary releases from this repository. The initial policy uses
MPItrampoline. GitHub Releases stores the Linux x86_64 glibc archives. The
``hypredrive_mpi_trampoline`` entry in ``interfaces/julia/Artifacts.toml`` identifies
these archives.

Release maintainers run the ``Julia Artifacts`` workflow on the release branch. Enable
``update_artifacts_toml`` and set ``release_tag`` to the new tag. Create the tag from the
resulting commit. Before publication, the tag workflow verifies the applicable URLs in
``Artifacts.toml``. The Julia interface ``binary`` tools contain the release recipe.

Local BinaryBuilder runs require a working container runner. If Docker cleanup fails at
``sudo chown``, use the GitHub workflow. A machine with passwordless ``sudo`` or
unprivileged container support is another option.

The project does not currently ship artifacts for other platforms or MPI implementations.
This includes macOS, Linux aarch64, musl, Windows, OpenMPI, and custom MPI. Users on these
systems build from source or use an installation prefix. ``MPItrampoline_jll`` is a normal
dependency. Thus, a clean Julia environment can preload the artifact without extension
activation.
