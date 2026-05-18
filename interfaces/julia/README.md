# HYPREDRV Julia interface

This is an in-tree Julia package for using an installed or build-tree HYPREDRV library from Julia. Julia 1.9 or newer is required. The package and module name follow Julia convention, so use `using HypreDrive`.

## Build

Configure HYPREDRV with Julia enabled:

```bash
cmake -S . -B build-julia \
  -DHYPREDRV_ENABLE_JULIA=ON \
  -DHYPREDRV_ENABLE_TESTING=ON
cmake --build build-julia --target julia-test --parallel
```

To explicitly validate install-tree consumption of the Julia bridge CMake target:

```bash
cmake --build build-julia --target julia-install-consumer-test --parallel
```

Because Julia loads the interface with `Libdl`, the CMake build creates
`libHYPREDRV_Julia` as a shared bridge library. If you link against an external
static HYPRE archive, that archive must have been compiled with `-fPIC`; otherwise
the bridge cannot be linked. Auto-fetched HYPRE builds are configured with PIC
when the Julia interface is enabled.

The bridge installs to the same library directory as `libHYPREDRV` so exported
CMake targets and runtime search paths follow the normal HYPREDRV install
layout. Artifact tarballs may still store the bridge under `lib/julia`; the
Julia loader searches both locations.

CMake prints the Julia executable it found. If Julia is not found at configure time,
the `julia-test` target fails with a clear message when run.

## Use from the source tree

```julia
import Pkg
Pkg.develop(path="interfaces/julia")
```

From a checkout outside the repository root, use Julia's subdirectory package
support:

```julia
import Pkg
Pkg.add(Pkg.PackageSpec(url="https://github.com/hypre-space/hypredrive.git",
                        subdir="interfaces/julia"))
```

This installs the Julia package sources only. Until a binary artifact/JLL is
published for the user's platform, users still need a compatible HYPREDRV build
or install prefix.

Point the package at the bridge library built by CMake:

```bash
export HYPREDRV_LIBRARY=$PWD/build-julia/interfaces/julia/lib/libHYPREDRV_Julia.so
```

On macOS the extension is `.dylib` instead of `.so`.

Alternatively, after `cmake --install build-julia`, set `HYPREDRV_PREFIX` to
the install prefix. `HYPREDRV_DIR` is still accepted for compatibility, but
`HYPREDRV_PREFIX` avoids the CMake convention where `<Pkg>_DIR` means the
directory containing `HYPREDRVConfig.cmake`. The package searches `lib/julia`,
`lib`, `lib64/julia`, and `lib64` below that prefix before falling back to
`Libdl.find_library`.

If neither environment variable is set, the package next checks the Julia
artifact named `hypredrive_mpi_trampoline` in `Artifacts.toml`. The artifact
contains the HYPREDRV Julia bridge built against MPItrampoline; MPItrampoline
itself comes from `MPItrampoline_jll`. The committed `Artifacts.toml` is empty
until release tarballs are bound, so source checkouts fall through to the
build-tree and `Libdl.find_library` fallbacks.

## Example

```julia
using HypreDrive
using SparseArrays

n = 64
A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
b = ones(n)

x, info = hypredrive_solve(A, b)
println(info.iterations)
```

`hypredrive_solve` accepts Julia sparse or dense matrices and uses quiet PCG+AMG defaults. Custom options can be built in Julia style:

```julia
opts = hypredrive_options(
    solver=:pcg,
    preconditioner=:amg,
    pcg=(max_iter=200, relative_tol=1.0e-10),
    amg=(print_level=0,),
)

x, info = hypredrive_solve(A, b; options=opts)
```

The generic aliases `solve`, `solve_mpi`, `initialize`, and `shutdown` remain available for qualified calls such as `HypreDrive.solve(...)`, but they are intentionally not exported.

## MPI

Run the same package under `mpiexec` and call `hypredrive_solve_mpi`:

```bash
mpiexec -n 2 julia --project=interfaces/julia interfaces/julia/test/mpi.jl
```

`hypredrive_solve_mpi(A, b)` creates the driver on `MPI_COMM_WORLD`, partitions the rows of the global Julia matrix across ranks, and returns each rank's local solution segment. This convenience API is intended for correctness and integration workflows.

The explicit MPI helper names `hypredrive_mpi_world_rank`,
`hypredrive_mpi_world_size`, and `hypredrive_mpi_world_sum` operate on
`MPI_COMM_WORLD`. The older qualified aliases remain available but are not
exported.

`Pkg.test()` runs the serial package tests only. The MPI test is intentionally wired through CMake/CTest because it requires `mpiexec`:

```bash
cmake --build build-julia --target julia-test --parallel
```

## Standalone Laplacian example

The standalone example mirrors the C Laplacian driver's command-line shape and assembles only the local CSR slab on each rank:

```bash
mpiexec -n 2 julia --project=interfaces/julia \
  interfaces/julia/examples/laplacian.jl \
  -n 12 12 12 -P 2 1 1 -s 7 -ns 2
```

Use `-i options.yml` to provide a hypredrive YAML solver configuration. Without
`-i`, the example uses quiet PCG+AMG defaults. The `-P` topology controls the 3D
block partition used to assemble local matrix rows.

## Scope

The package is ready to be registered from this monorepo subdirectory; a separate
repository is not required. Registering it in Julia's General registry should use
`interfaces/julia` as the package subdirectory and normal Julia semver releases
such as `0.2.0`, not development-version strings.

The current fallback distribution model is source/install-prefix based: users
either build HYPREDRV from this checkout or install HYPREDRV separately and set
`HYPREDRV_PREFIX` or `HYPREDRV_LIBRARY`. `HYPREDRV_DIR` is accepted only as a
compatibility alias. Binary releases can be distributed from this
monorepo through Julia artifacts without moving the package out of this repository.
The first artifact policy is MPItrampoline, with Linux x86_64 glibc tarballs
hosted on GitHub Releases and bound into `Artifacts.toml`. macOS, Linux aarch64,
musl, Windows, OpenMPI, and custom-MPI artifacts are not shipped until explicit
additional artifact flavors are added.
`MPItrampoline_jll` is a normal package dependency. The module loads it lazily:
source-tree and install-prefix workflows do not need to instantiate the artifact
dependency, while artifact-backed installs preload MPItrampoline before opening
`libHYPREDRV_Julia`.

Release maintainers must build and bind the Linux x86_64 artifact through the
`Julia Artifacts` workflow before creating the release tag:

1. Run the workflow manually on the release branch with the workflow input
   `update_artifacts_toml=true` and `release_tag` set to the future tag name.
2. Let the workflow commit the populated `interfaces/julia/Artifacts.toml`.
   The workflow pushes with a pull/rebase retry loop; protected branches must
   allow the GitHub Actions bot to push this maintainer-generated commit.
3. Create the tag from that commit.

The tag workflow verifies that `Artifacts.toml` already contains URLs for the
exact tag before uploading release assets. If the manual binding step was skipped
or used the wrong tag name, the `release-assets` job fails intentionally instead
of silently shipping an empty or stale artifact table.

For local maintainer debugging, build the tarball with:

```bash
julia -e 'using Pkg; Pkg.add("BinaryBuilder")'
HYPREDRV_BINARY_GIT_SHA=$(git rev-parse HEAD) \
  julia interfaces/julia/binary/build_tarballs.jl --verbose --deploy=local
```

Local BinaryBuilder runs require a working container runner. If Docker cleanup
fails at a `sudo chown` step, use the GitHub `Julia Artifacts` workflow or run on
a machine with passwordless sudo/unprivileged container support.

After uploading the produced tarball to a GitHub Release, bind it with:

```bash
julia interfaces/julia/binary/bind_artifact.jl \
  /path/to/hypredrive-tarball.tar.gz \
  "https://github.com/hypre-space/hypredrive/releases/download/TAG/TARBALL_NAME"
```

OpenMPI or custom-MPI builds should continue to use source/install-prefix mode
until explicit additional artifact flavors are added.

The package uses process-global MPI state; do not mix multiple independent MPI
owners inside one Julia process.
