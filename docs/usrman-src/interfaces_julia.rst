.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Interfaces_Julia:

Julia Interface
===============

The Julia interface is an in-tree package under ``interfaces/julia``. It supports
source-tree and install-prefix use today and can be registered from this monorepo
subdirectory later; a separate Julia-package repository is not required. Julia 1.9
or newer is required. The package and module name follow Julia convention: use
``using HypreDrive``.

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

The CMake build creates a small ``libHYPREDRV_Julia`` bridge library. The bridge keeps
Julia away from HYPRE ABI details such as ``HYPRE_BigInt`` width and real precision.
Because Julia loads this bridge with ``Libdl``, it is always built as a shared library.
External static HYPRE archives must therefore be compiled with ``-fPIC``. Auto-fetched
HYPRE builds are configured with position-independent code when the Julia interface is
enabled.

The bridge is installed under ``lib/julia`` so language-specific shared libraries do
not pollute the top-level library directory. Its install RPATH includes both that
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

After installing hypredrive, ``HYPREDRV_DIR`` may be set instead:

.. code-block:: bash

   export HYPREDRV_DIR=/path/to/hypredrive/install

For the Julia package, ``HYPREDRV_DIR`` means the HYPREDRV install prefix, not the
CMake package directory containing ``HYPREDRVConfig.cmake``. The Julia package
searches ``lib/julia``, ``lib``, ``lib64/julia``, and ``lib64`` under that prefix.
If neither environment variable is set, it falls back to the Julia artifact named
``hypredrive_mpi_trampoline`` in ``Artifacts.toml``, then the source-tree build
search, and finally ``Libdl.find_library``. The artifact contains the HYPREDRV
Julia bridge built against MPItrampoline; MPItrampoline itself comes from
``MPItrampoline_jll``.

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

Options can be constructed as Julia keyword arguments instead of hand-written YAML:

.. code-block:: julia

   opts = hypredrive_options(
       solver=:pcg,
       preconditioner=:amg,
       pcg=(max_iter=200, relative_tol=1.0e-10),
       amg=(print_level=0,),
   )

   x, info = hypredrive_solve(A, b; options=opts)

The generic aliases ``solve``, ``solve_mpi``, ``initialize``, and ``shutdown`` remain
available for qualified calls such as ``HypreDrive.solve(...)``, but they are
intentionally not exported.

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

``Pkg.test()`` runs the serial package tests only. The MPI test is wired through
CMake/CTest because it requires ``mpiexec``.

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

Scope
-----

The interface supports serial solves via ``MPI_COMM_SELF`` and MPI solves via
``MPI_COMM_WORLD``. It supports standard double-precision, real-valued HYPRE builds.
Complex, single-precision, and long-double HYPRE builds are rejected at runtime with a
clear error. The package uses process-global MPI state; do not mix multiple independent
MPI owners inside one Julia process.

The package can be registered in Julia's General registry from ``interfaces/julia``.
Registry releases should use normal Julia semver such as ``0.2.0``, not development
version strings. The fallback distribution model is source/install-prefix based:
users build HYPREDRV from this checkout or install HYPREDRV separately and set
``HYPREDRV_DIR`` or ``HYPREDRV_LIBRARY``.

Binary releases can be distributed from this monorepo through Julia artifacts without
moving the package to a separate repository. The first artifact policy is
MPItrampoline: Linux x86_64 glibc release tarballs are hosted on GitHub Releases and bound into
``interfaces/julia/Artifacts.toml`` under the artifact name
``hypredrive_mpi_trampoline``. Release maintainers should run the ``Julia
Artifacts`` workflow manually on the release branch with ``update_artifacts_toml``
enabled and ``release_tag`` set to the future tag name, then create the tag from
the resulting commit. The tag workflow verifies that ``Artifacts.toml`` already
contains URLs for that tag before publishing release assets. The release recipe
lives in the Julia interface's ``binary`` tooling. Local BinaryBuilder runs require a working container runner;
if Docker cleanup fails at a ``sudo chown`` step, use the GitHub ``Julia
Artifacts`` workflow or run on a machine with passwordless sudo/unprivileged
container support. macOS, Linux aarch64, musl, Windows, OpenMPI, and custom-MPI
artifacts are not shipped until explicit additional artifact flavors are added;
those users should continue to use source/install-prefix mode.
``MPItrampoline_jll`` is a normal dependency, not a weak dependency, so artifact
preloading works in a clean Julia environment without extension activation.
