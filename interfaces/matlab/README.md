# hypredrive MATLAB/Octave interface

This interface builds a MEX function for MATLAB and GNU Octave. Users call the
portable `hypredrive.m` wrapper; the compiled entry point is `hypredrive_mex`.

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
addpath("/path/to/source/interfaces/matlab/src")
addpath("/path/to/build-matlab/interfaces/matlab/matlab")
```

For Octave, use the Octave MEX directory instead:

```matlab
addpath("/path/to/source/interfaces/matlab/src")
addpath("/path/to/build-matlab/interfaces/matlab/octave")
```

Solve a sparse system:

```matlab
n = 64;
e = ones(n, 1);
A = spdiags([-e, 2*e, -e], -1:1, n, n);
b = ones(n, 1);

[x, info] = hypredrive(A, b);
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

[x, info] = hypredrive(A, b, opts);
```

The struct form mirrors hypredrive YAML:

```matlab
opts = struct();
opts.solver.pcg.max_iter = 200;
opts.solver.pcg.relative_tol = 1.0e-10;
opts.preconditioner.amg.print_level = 0;

[x, info] = hypredrive(A, b, opts);
```

MATLAB/Octave usage is quiet by default: `hypredrive_options` adds
`general.statistics = 0`, and the no-options path uses the same policy. Request
the HYPREDRV statistics table explicitly when needed:

```matlab
opts = hypredrive_options(struct('general', struct('statistics', 1)));
```

Raw YAML text is still accepted for advanced users. The default options are PCG
with AMG preconditioning and statistics disabled.

## Test

When `HYPREDRV_ENABLE_TESTING=ON`, the optional test target runs a small 1D
Laplacian smoke test for each available runtime:

```bash
cmake --build build-matlab --target matlab-test
```

The target exists when `HYPREDRV_ENABLE_MATLAB=ON` and MATLAB or Octave is found.

## Install

```bash
cmake --install build-matlab --prefix /path/to/install
```

MATLAB files are installed under `lib/matlab`. Octave files are installed under
`lib/octave`. Examples are installed under `share/matlab/examples`.
On Unix-like systems, installed MEX files use a relative runtime search path
back to the prefix-local `lib` directory.
