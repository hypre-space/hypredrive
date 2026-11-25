# hypredrive MATLAB/Octave interface

This interface builds a MEX function for MATLAB and GNU Octave. Users call the
portable `hypredrive_solve.m` wrapper; the compiled entry point is
`hypredrive_mex`.

The interface is serial. `hypredrive_solve` is the one-shot path (see
[Handle-based API](#handle-based-api) for incremental use):

- MATLAB or Octave provides one real double sparse matrix `A` and one real double dense vector `b`.
- The MEX wrapper converts CSC sparse storage to the CSR format used by hypredrive.
- The solve runs on `MPI_COMM_SELF`.
- The wrapper returns a dense solution vector and an optional info struct.

It is intended for host CPU HYPRE builds. OpenMP-enabled HYPRE is fine. MATLAB
or Octave MPI workflows are not part of this interface; use the C, C++, Fortran,
Python, or Julia interfaces for distributed-memory runs.

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

## Minimal example

This serial example solves `A x = b` for `A = [[2, -1], [-1, 2]]` and
`b = [1, 1]`, giving `x = [1, 1]`.

After adding the wrapper and MEX directories to the path as above:

```matlab
A = sparse([2.0, -1.0; -1.0, 2.0]);
b = ones(2, 1);
[x, info] = hypredrive_solve(A, b);
disp(x)  % [1; 1]
disp(info.converged)
```

## Solver options

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

## Handle-based API

Besides the one-shot solve, `hypredrive_mex('command', ...)` exposes every
public `HYPREDRV_` function. The one-shot form `hypredrive_mex(A, b, yaml)`
(numeric first argument) is unchanged, and `hypredrive_solve` still uses it.

Conventions:

- Drivers, IJ matrices and IJ vectors are opaque scalar `uint64` handles. They
  are never raw pointers: every call looks the handle up in a table, and a
  stale, destroyed, forged, or wrong-kind handle raises
  `hypredrive:InvalidHandle`. Handle ids are never reused.
- Drivers are created on `MPI_COMM_SELF` in library mode, so matrices and
  vectors attached to a driver remain owned by the interface.
- `hypredrive_mex('commands')` returns the command names as a cell array.
- Vectors are real double columns; matrices are real double sparse (CSC is
  converted to CSR). Near-null-space and null-space modes are
  `num_entries`-by-`num_components` dense matrices. Indices passed to
  `state_*` and `stats_*` are zero-based, as in the C API.
- Failures raise `hypredrive:HypreDriveError` (after the C error description is
  printed and the global error state cleared), or one of
  `hypredrive:InvalidHandle`, `InvalidArgument`, `InvalidMatrix`,
  `InvalidNumInputs`, `UnknownCommand`, `ObjectInUse`.

Handle lifetime:

- `destroy(h)` on a driver destroys it and invalidates borrowed handles issued
  from it (`get_matrix`, `get_rhs`, `get_solution`).
- `destroy(h)` on an owned matrix/vector (`matrix_from_sparse`,
  `vector_from_array`) that a live driver currently holds raises
  `hypredrive:ObjectInUse`; replace it in the driver or destroy the driver
  first. Objects a driver has replaced, rejected, or only received through
  `precon_apply` can be destroyed. Borrowed handles cannot be destroyed
  individually.
- `state_set` requires all state vectors to share one local length.
- `destroy_all` releases everything (drivers first), and the same cleanup runs at
  `mexAtExit`. `clear mex` therefore does not leak. `is_valid(h)` and
  `num_handles` help with checking.

Example:

```matlab
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
```

### Command reference

| Command | C API |
| --- | --- |
| `h = create([yaml])` / `destroy(h)` | `HYPREDRV_Create`, `HYPREDRV_SetLibraryMode`, `HYPREDRV_Destroy` |
| `parse(h, yaml)` | `HYPREDRV_InputArgsParse` |
| `error_clear`, `n = error_invalid_value(msg)` | `HYPREDRV_ErrorCodeClear`, `HYPREDRV_ErrorInvalidValue` |
| `print_lib_info([dt])`, `print_system_info`, `print_exit_info([argv0])` | `HYPREDRV_PrintLibInfo`, `HYPREDRV_PrintSystemInfo`, `HYPREDRV_PrintExitInfo` |
| `set_name(h, name)` | `HYPREDRV_ObjectSetName` |
| `get_warmup`, `get_num_repetitions`, `get_num_linear_systems`, `get_num_precon_variants` `(h)` | `HYPREDRV_InputArgsGet*` |
| `set_precon_variant(h, i)`, `set_precon_preset(h, s)`, `set_solver_preset(h, s)` | `HYPREDRV_InputArgsSet*` |
| `register_solver_preset(name, yaml[, help])`, `register_precon_preset(...)` | `HYPREDRV_SolverPresetRegister`, `HYPREDRV_PreconPresetRegister` |
| `hA = matrix_from_sparse(A)`, `hv = vector_from_array(v)`, `vector_get_values(hv)`, `vector_set_values(hv, v)` | raw IJ objects for the `HYPRE_Matrix`/`HYPRE_Vector` APIs |
| `build`, `read_matrix`, `read_dofmap`, `print`, `compute_eigenspectrum` `(h)` | `HYPREDRV_LinearSystemBuild`, `ReadMatrix`, `ReadDofmap`, `Print`, `ComputeEigenspectrum` |
| `set_matrix_csr(h, A)`, `set_rhs_array(h, b)` | `HYPREDRV_LinearSystemSetMatrixFromCSR`, `SetRHSFromArray` |
| `set_matrix`, `set_prec_matrix`, `set_discrete_gradient`, `set_discrete_curl` `(h, hA)` | `HYPREDRV_LinearSystemSetMatrix`, `SetPrecMatrix`, `SetDiscreteGradient`, `SetDiscreteCurl` |
| `set_rhs`, `set_initial_guess`, `set_solution`, `set_reference_solution` `(h, hv)` | `HYPREDRV_LinearSystemSetRHS`, `SetInitialGuess`, `SetSolution`, `SetReferenceSolution` (`[]` clears) |
| `set_coordinates(h, hx[, hy[, hz]])`, `set_coarse_schur(h, level, hA)` | `HYPREDRV_LinearSystemSetCoordinates`, `SetCoarseSchur` |
| `reset_initial_guess(h)` | `HYPREDRV_LinearSystemResetInitialGuess` |
| `set_dofmap(h, ints)`, `set_interleaved_dofmap(h, nblk, ntypes)`, `set_contiguous_dofmap(h, nblk, ntypes)`, `print_dofmap(h, file)` | `HYPREDRV_LinearSystemSetDofmap`, `SetInterleavedDofmap`, `SetContiguousDofmap`, `PrintDofmap` |
| `set_near_null_space(h, M)`, `set_null_space(h, M)` | `HYPREDRV_LinearSystemSetNearNullSpace`, `SetNullSpace` |
| `get_matrix(h)`, `get_rhs(h)`, `get_solution(h)` | `HYPREDRV_LinearSystemGetMatrix`, `GetRHS`, `GetSolution` (borrowed handles) |
| `get_solution_values(h)`, `copy_solution_values(h)`, `get_rhs_values(h)`, `get_solution_length(h)`, `get_solution_norm(h[, type])` | `HYPREDRV_LinearSystemGetSolutionValues`, `CopySolutionValues`, `GetRHSValues`, `GetSolutionLength`, `GetSolutionNorm` |
| `state_set(h, uint64([hv1 hv2 ...]))`, `state_get_values(h, i)`, `state_copy(h, in, out)`, `state_update_all(h)`, `state_apply_correction(h, i)` | `HYPREDRV_StateVector*` |
| `precon_create`, `precon_setup`, `precon_destroy` `(h)`, `precon_apply(h, hb, hx)` | `HYPREDRV_PreconCreate`, `PreconSetup`, `PreconDestroy`, `PreconApply` |
| `solver_create`, `solver_setup`, `solver_apply`, `solver_destroy` `(h)`, `solver_set_relative_tol(h, tol)` | `HYPREDRV_LinearSolverCreate`, `Setup`, `Apply`, `Destroy`, `SetRelativeTol` |
| `solver_num_iter`, `solver_converged`, `solver_final_res_norm`, `solver_setup_time`, `solver_solve_time`, `solver_info` `(h)` | `HYPREDRV_LinearSolverGet*` (`solver_info` returns the same struct as the one-shot `info`) |
| `stats_print(h)`, `stats_level_count(h, lvl)`, `stats_level_entry(h, lvl, i)`, `stats_level_print(h, lvl)` | `HYPREDRV_StatsPrint`, `StatsLevelGetCount`, `GetEntry`, `Print` |
| `annotate_begin/end(h, name[, id])`, `annotate_level_begin/end(h, lvl, name, id)` | `HYPREDRV_AnnotateBegin/End`, `AnnotateLevelBegin/End` |

Only the host-memory, real-double, serial (`MPI_COMM_SELF`) case is supported.
`get_solution_values` errors if the solution lives in device memory; use
`copy_solution_values` instead.

## Test

When `HYPREDRV_ENABLE_TESTING=ON`, the optional test target runs the MEX wrapper
smoke test, the handle-API test (`test_hypredrive_api.m`), plus small serial Laplacian and elasticity examples for each
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
