# hypredrive MATLAB/Octave interface

This interface builds a MEX function for MATLAB and GNU Octave. Users call the
portable `hypredrive_solve.m` wrapper; the compiled entry point is
`hypredrive_mex`.

The first version is intentionally serial and one-shot:

- MATLAB or Octave provides one real double sparse matrix `A` and one real double dense vector `b`.
- The MEX wrapper converts CSC sparse storage to the CSR format used by hypredrive.
- The solve runs on `MPI_COMM_SELF`.
- The wrapper returns a dense solution vector and an optional info struct.

It is intended for host CPU HYPRE builds. OpenMP-enabled HYPRE is fine. MATLAB
or Octave MPI workflows are not part of this interface; use the C, Fortran, or
Python interfaces for distributed-memory runs.

## Requirements

- MATLAB with `mex`, or GNU Octave with `mkoctfile --mex`.
- CMake with `FindMatlab` when building the MATLAB MEX.
- A real-valued HYPRE build where `HYPRE_Real` uses the C `double` ABI.
- Non-complex HYPRE. `HYPRE_ENABLE_SINGLE`, `HYPRE_ENABLE_LONG_DOUBLE`, and `HYPRE_ENABLE_COMPLEX` are rejected for this interface.
- The MEX file links to the same MPI implementation used to build HYPRE and
  hypredrive. MATLAB can ship or preload its own MPI runtime, so MPI ABI
  mismatches are the most common runtime failure. Prefer launching MATLAB or
  Octave from an environment where the intended MPI `bin` and `lib` directories
  are first on `PATH` and the platform runtime library path.

## Build

The same CMake option enables both runtimes. If both MATLAB and Octave are
available, both MEX files are built.

```bash
cmake -S . -B build-matlab \
  -DHYPREDRV_ENABLE_MATLAB=ON \
  -DHYPREDRV_ENABLE_TESTING=ON
cmake --build build-matlab --parallel
```

If MATLAB is not in the default search path, pass `-DMatlab_ROOT_DIR=/path/to/MATLAB`.
If Octave is not in the default search path, ensure `mkoctfile` is on `PATH`.

Useful explicit targets:

```bash
cmake --build build-matlab --target hypredrive-matlab
cmake --build build-matlab --target hypredrive-octave
```

## Use

Add the wrapper and the runtime-specific MEX directory to the MATLAB path:

```matlab
addpath("/path/to/source/interfaces/matlab-octave/src")
addpath("/path/to/build-matlab/interfaces/matlab/matlab")
```

For Octave, use the Octave MEX directory instead:

```matlab
addpath("/path/to/source/interfaces/matlab-octave/src")
addpath("/path/to/build-matlab/lib/octave")
```

The build-tree paths differ because MATLAB MEX files are produced by
`FindMatlab`, while Octave MEX files are produced by `mkoctfile` into the common
library tree.

For an installed prefix, add `/path/to/install/lib/matlab` in MATLAB or
`/path/to/install/lib/octave` in Octave. To make that persistent, put the
corresponding `addpath` command in MATLAB's `startup.m` or Octave's startup
file. As a one-step setup from an installed prefix, run:

```matlab
run("/path/to/install/lib/matlab/hypredrive_setup.m")
```

or, for Octave:

```matlab
run("/path/to/install/lib/octave/hypredrive_setup.m")
```

Solve a sparse system:

```matlab
n = 64;
e = ones(n, 1);
A = spdiags([-e, 2*e, -e], -1:1, n, n);
b = ones(n, 1);

[x, info] = hypredrive_solve(A, b);
norm(b - A*x) / norm(b)
info.iterations
```

Custom solver options should usually be built with `hypredrive_options`:

```matlab
opts = hypredrive_options( ...
    'solver', 'pcg', ...
    'preconditioner', 'amg', ...
    'pcg', struct('max_iter', 200, 'relative_tol', 1.0e-10), ...
    'amg', struct('print_level', 0));

[x, info] = hypredrive_solve(A, b, opts);
```

The struct form mirrors hypredrive YAML:

```matlab
opts = struct();
opts.solver.pcg.max_iter = 200;
opts.solver.pcg.relative_tol = 1.0e-10;
opts.preconditioner.amg.print_level = 0;

[x, info] = hypredrive_solve(A, b, opts);
```

MATLAB/Octave usage is quiet by default: `hypredrive_options` adds
`general.statistics = 0`, and the no-options path uses the same policy. Request
the HYPREDRV statistics table explicitly when needed:

```matlab
opts = hypredrive_options(struct('general', struct('statistics', 1)));
```

Raw YAML text is still accepted for advanced users. The default options are PCG
with AMG preconditioning and statistics disabled.

## Examples

The examples are regular MATLAB/Octave scripts that call `hypredrive_solve`:

```matlab
addpath("/path/to/source/interfaces/matlab-octave/examples")

laplacian(16, 16, 16)
elasticity(16, 16, 16)
```

`laplacian.m` assembles 1D, 2D, or 3D finite-difference Poisson problems.
`elasticity.m` assembles 1D, 2D, or 3D linear-elasticity problems and uses the
matching elasticity AMG preset. The helper files `build_laplacian.m` and
`build_elasticity.m` can also be called directly when a test or script needs
just the matrix and right-hand side.

The optional `info` output has these fields:

| Field | Meaning |
| --- | --- |
| `iterations` | Solver iteration count reported by hypredrive. |
| `converged` | Logical convergence flag from the last solve (true when the solver reached its tolerance). |
| `final_res_norm` | Final relative residual norm from the last solve. |
| `setup_time` | Linear-solver setup time in seconds. |
| `solve_time` | Linear-solver apply/solve time in seconds. |
| `solution_norm` | L2 norm of the returned solution vector. |

## Test

When `HYPREDRV_ENABLE_TESTING=ON`, the optional test target runs the MEX wrapper
smoke test plus small serial Laplacian and elasticity examples for each
available runtime:

```bash
cmake --build build-matlab --target matlab-test
```

The target exists when `HYPREDRV_ENABLE_MATLAB=ON` and MATLAB or Octave is found.

## Install

```bash
cmake --install build-matlab --prefix /path/to/install
```

MATLAB files are installed under `lib/matlab`. Octave files are installed under
`lib/octave`. The example scripts and assembly helpers are installed under
`share/matlab/examples`.
On Unix-like systems, installed MEX files use a relative runtime search path
back to the prefix-local library directory. If you relocate the MEX file
manually, keep it one directory below the installed library directory or set the
platform runtime library path so `libHYPREDRV` and HYPRE are visible.

## Troubleshooting

- `libmpi` load errors usually mean MATLAB/Octave is not seeing the same MPI
  runtime used by HYPRE. Start the runtime from a shell where the intended MPI
  installation is active.
- `Invalid MEX-file` errors usually mean a missing shared-library dependency.
  Check the MEX file with `ldd` on Linux or `otool -L` on macOS.
- If options fail to parse, build them with `hypredrive_options` first and print
  the resulting YAML string before calling `hypredrive_solve`.
