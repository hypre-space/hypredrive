.. Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (MIT)

.. _Interfaces_Rust:

Rust Interface
==============

The ``hypredrive`` Cargo crate lives in ``interfaces/rust``. It wraps the public
``HYPREDRV_`` C API with process and driver lifetime management, rank-local CSR
input, YAML configuration, and owned host solutions. MPI interoperability uses
the ``mpi`` crate (rsmpi). Private C bindings are generated from the selected
HYPREDRV, HYPRE, and MPI headers.

Requirements and build
----------------------

Use Rust 1.82 or newer, Cargo, CMake, libclang, an MPI C compiler, and a shared
HYPREDRV library. The crate uses ``mpi`` 0.8 and ``bindgen`` 0.71.
HYPRE must use real, double-precision scalars. The default,
BIGINT, and MIXEDINT index configurations are supported. Binding generation
rejects complex, single-precision, and long-double builds. The public Rust
``BigInt`` alias follows ``HYPRE_BigInt`` from the selected headers. Use checked
conversions for application indices.

Configure from the repository root with a dedicated build directory.

.. code-block:: bash

   rust_build_dir=$(mktemp -d /tmp/hypredrive-rust-build-XXXXXXXX)
   cmake -S . -B "$rust_build_dir" \
     -DBUILD_SHARED_LIBS=ON \
     -DHYPREDRV_ENABLE_RUST=ON \
     -DHYPREDRV_ENABLE_TESTING=ON \
     -DHYPREDRV_DATASETS= \
     -DCMAKE_C_COMPILER=mpicc
   cmake --build "$rust_build_dir" --target rust-test --parallel

``rust-build`` builds the Rust crate, examples, and smoke executable.
``rust-test`` runs the serial and MPI CTest checks after building them. The
crate is staged in the CMake build directory, together with its committed
lockfile, so the source checkout can be read-only. Cargo output is beneath
``$rust_build_dir/interfaces/rust/cargo-target``. Multi-config generators
insert the configuration name before ``cargo-target``. CTest runs the
compiled smoke executable, with one MPI initialization on each process's
main thread. Cargo runs with ``--locked``, so ``interfaces/rust/Cargo.lock``
must stay in step with ``Cargo.toml``. Refresh it deliberately with
``cargo update --manifest-path interfaces/rust/Cargo.toml`` and commit the result. Set ``HYPREDRV_USE_MPIEXEC_FOR_SINGLE_RANK_TESTS=ON`` to use the
MPI launcher for the one-rank test too.
The empty ``HYPREDRV_DATASETS`` selection keeps the full native build
independent of downloaded example datasets. The Rust examples and tests
assemble their own input.

CMake passes its live native target's library path, include directories,
compile definitions, and MPI compiler to Cargo. The Rust option defaults to
``OFF``. Rust-disabled builds perform their usual native configuration without
Rust tool discovery.

Minimal example
---------------

The serial example solves ``A x = b`` with ``A = [[2, -1], [-1, 2]]`` and
``b = [1, 1]``, producing ``x = [1, 1]``.

.. literalinclude:: ../../interfaces/rust/examples/minimal.rs
   :language: rust

The application initializes MPI with rsmpi. ``Runtime`` borrows its
``Universe`` and each ``HypreDrive`` borrows the runtime and its communicator.
These borrows enforce cleanup of drivers before HYPREDRV finalization and MPI
finalization. The runtime and its drivers are ``!Send`` and ``!Sync``. Only one
Rust runtime may be active per process, including across threads. The runtime
requires exclusive ownership of HYPREDRV's process-global state relative to
other native callers.

Run the built examples with the matching MPI installation. Direct launches
need the loader environment in the current shell. The environment CMake sets
for tests applies only to those test processes. On Linux, include the build's
library directory and any separate HYPRE or MPI library directories in
``LD_LIBRARY_PATH``. On macOS, use the corresponding loader paths or
``DYLD_LIBRARY_PATH``.

.. code-block:: bash

   cmake --build "$rust_build_dir" --target rust-build --parallel
   export LD_LIBRARY_PATH="$rust_build_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   "$rust_build_dir/interfaces/rust/cargo-target/debug/examples/minimal"
   mpiexec -n 2 "$rust_build_dir/interfaces/rust/cargo-target/debug/examples/laplacian" 8 10 12

The Laplacian example assembles each rank's contiguous range of rows for a
3D seven-point stencil. The default grid is 10 × 10 × 10. Pass one side length
for a cube or three lengths for a rectangular grid. The example reuses the
matrix and driver for a second right-hand side.

With Linux sanitizer builds, CTest preloads the native compiler's sanitizer
runtimes into each Rust test process. For a direct launch, set ``LD_PRELOAD``
to the same runtimes, with the address sanitizer first. For example, when
the native build uses GCC through ``mpicc``.

.. code-block:: bash

   LD_PRELOAD="$(mpicc -print-file-name=libasan.so):$(mpicc -print-file-name=libubsan.so)" \
     "$rust_build_dir/interfaces/rust/cargo-target/debug/examples/minimal"

Installed-library use
---------------------

An external Cargo application can use a normal shared HYPREDRV installation,
including one configured with the Rust CMake option disabled. Install to a
separate prefix and keep Cargo output in its own temporary directory. Build
the native installation targets before installing.

.. code-block:: bash

   rust_install_dir=$(mktemp -d /tmp/hypredrive-rust-install-XXXXXXXX)
   cmake --build "$rust_build_dir" --parallel
   cmake --install "$rust_build_dir" --prefix "$rust_install_dir"
   export MPICC=$(command -v mpicc)
   export CMAKE_PREFIX_PATH="$rust_install_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
   export CARGO_TARGET_DIR=$(mktemp -d /tmp/hypredrive-rust-cargo-XXXXXXXX)
   export LD_LIBRARY_PATH="$rust_install_dir/lib:$rust_install_dir/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

``MPICC`` must identify the same MPI installation used by HYPREDRV and HYPRE.
Use a fresh ``CARGO_TARGET_DIR`` when switching MPI installations. The crate
checks the compiled rsmpi communicator ABI against the selected MPI headers.
rsmpi consults ``MPI_PKG_CONFIG``, then ``CRAY_MPICH_DIR``, before ``MPICC``.
Either may be set, provided it selects the MPI installation used by HYPREDRV;
the compiled-ABI check rejects a mismatched communicator size, handle kind, or
MPI version, but not every difference between two installations of the same
ABI. The ``MPICC`` consistency check is skipped when either variable is set.
Alternatively, set ``HYPREDRV_DIR`` to the directory containing
``HYPREDRVConfig.cmake``, typically ``lib/cmake/HYPREDRV`` or
``lib64/cmake/HYPREDRV`` under the prefix. Set ``LIBCLANG_PATH`` if libclang is
outside its normal search locations.

Add path dependencies to the external application's ``Cargo.toml``.

.. code-block:: toml

   [dependencies]
   hypredrive = { path = "/path/to/hypredrive/interfaces/rust" }
   mpi = { version = "0.8", default-features = false }

Then run ``cargo build`` and ``cargo run`` in that application. The crate's
own examples also work through standalone Cargo discovery.

.. code-block:: bash

   cargo build --manifest-path interfaces/rust/Cargo.toml --examples
   cargo run --manifest-path interfaces/rust/Cargo.toml --example minimal
   mpiexec -n 2 "$CARGO_TARGET_DIR/debug/examples/laplacian"

The build script configures a metadata-only CMake project that requests the
crate's own major and minor version with ``find_package(HYPREDRV CONFIG REQUIRED)``. CMake resolves the native
dependencies, verifies the HYPRE type sizes and MPI communicator ABI against
the producer's metadata, and supplies the matching headers and library path.
If the installation predates this ABI metadata, rebuild and reinstall
hypredrive. Generated bindings stay in Cargo's ``OUT_DIR``.

The executable also needs runtime access to HYPREDRV, HYPRE, and MPI shared
libraries. Cargo's link-time search path does not set the loader's runtime
search path. The Linux example above sets ``LD_LIBRARY_PATH``. Add any separate
HYPRE or MPI library directories. On macOS use the corresponding loader paths
or ``DYLD_LIBRARY_PATH``.

Configuration and results
-------------------------

Create a configured driver with
``HypreDrive::from_yaml(&runtime, &comm, yaml)`` or
``HypreDrive::from_file(&runtime, &comm, path)``. Both use the native YAML
parser. ``from_yaml`` requires byte-identical text on all ranks and checks
agreement before parsing. ``from_file`` takes a UTF-8 path without spaces and
with a ``.yaml`` or ``.yml`` extension, matched case-insensitively. Rank zero
reads the file through the native parser and broadcasts
its contents. Relative paths in YAML retain their configuration-directory
semantics. Embedded NUL bytes are rejected. Drivers require an MPI
intracommunicator, including split communicators.

``set_matrix_csr(row_start, row_end, indptr, columns, values)`` accepts inclusive
global row bounds, global column indices, and host values. Every rank must own
at least one row. The wrapper validates row arithmetic, slice lengths,
nonnegative monotone offsets, and terminal bounds. Offset CSR slabs with a
nonzero first offset are accepted when the column and value buffers contain
the preceding entries. The C implementation checks the global row partition
and column range.

``set_rhs(values)`` uses the installed matrix's local range and validates the
RHS length. Native assembly copies input data, so the caller can release or
reuse its buffers after each call. ``set_initial_guess()`` constructs the
configured guess for the next solve. ``reset_initial_guess()`` selects that
guess again. ``solve()`` uses the configured guess automatically on the first
solve for a new matrix. A second ``set_rhs()`` preserves
the existing guess and solution. Use ``reset_initial_guess()`` to restart from
that guess, or ``set_initial_guess()`` to regenerate an RHS-dependent guess.
Replacing the matrix requires another ``set_rhs()`` call.

``solve()`` creates, sets up, and applies the native solver. It copies the
solution to host storage through the public C API and returns a ``SolveResult``.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``x: Vec<f64>``
     - Owned host values for this rank's local solution.
   * - ``iterations: i32``
     - Iteration count.
   * - ``converged: bool``
     - Whether the solver converged. A repeated or warm-started solve that
       already satisfies the relative tolerance counts as converged.
   * - ``final_res_norm: f64``
     - Final relative residual norm reported by the native solver.
   * - ``setup_time: f64``
     - Native solver setup time.
   * - ``solve_time: f64``
     - Native solver apply time.

Times are measured per solve and are in seconds by default. With the YAML
option ``general.use_millisec`` enabled, times are in milliseconds.

Nonconvergence returns ``converged=false``. A nonzero native API status returns
``Error`` with the operation name available through ``operation()`` and the
original ``u32`` status through ``status()``. Rust validation errors have no
native status. Scope cleanup also releases partially constructed drivers.

MPI call contract
-----------------

Call distributed driver operations in matching order on every rank of the
supplied intracommunicator and pass byte-identical inline YAML to
``from_yaml``. The wrapper agrees on Rust-side validation failures before
entering collective native calls. Every rank must participate even when only
one rank has invalid input.

The harness-free smoke executable exercises serial and distributed solves,
split communicators, RHS reuse, invalid CSR/RHS data, checked index arithmetic,
malformed YAML, a validation failure on one rank, and cleanup after failed
construction. Run it with each intended MPI installation and HYPRE index
configuration.
