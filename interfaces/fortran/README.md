# hypredrive Fortran interface

Fortran bindings for the public `HYPREDRV_` C API. The interface is thin by
intent: lifecycle, options, linear-system setup, solver calls, statistics, and
annotation routines keep the same names and ownership rules as the C library.

## Prerequisites

Build prerequisites are a C compiler, a Fortran 2003 compiler, MPI C and
Fortran wrappers (`mpicc`/`mpifort` or equivalents), CMake, and a HYPRE build
compatible with hypredrive. The current Fortran interface requires a real-valued
HYPRE build where `HYPRE_Real` uses the C `double` ABI; single-precision and
long-double HYPRE builds are rejected at configure time.

## Build

The Fortran interface is optional and disabled by default:

```bash
cmake -S . -B build-fortran \
  -DHYPREDRV_ENABLE_FORTRAN=ON \
  -DHYPREDRV_ENABLE_TESTING=ON \
  -DHYPREDRV_ENABLE_EXAMPLES=ON
cmake --build build-fortran --parallel
```

Run the Fortran tests:

```bash
cmake --build build-fortran --target fortran-test
```

The `laplacian-fortran` driver is available whenever `HYPREDRV_ENABLE_FORTRAN=ON`,
so the core Fortran test coverage does not depend on `HYPREDRV_ENABLE_EXAMPLES`.
The `fortran-test` target runs small namelist-driven Laplacian smoke cases:
serial, 2-rank x/z partitions, a 4-rank x/y partition, anisotropic coefficients,
and external solver YAML.

Build only the examples:

```bash
cmake --build build-fortran --target fortran-examples
```

The `laplacian-fortran` driver is installed whenever `HYPREDRV_ENABLE_FORTRAN=ON`.
It is both a user-facing example and the reproducible smoke driver used by the
Fortran tests.

## Installed use

The install exports a CMake target for downstream Fortran applications:

```cmake
find_package(MPI COMPONENTS Fortran REQUIRED)
find_package(HYPREDRV REQUIRED)
target_link_libraries(my_solver PRIVATE HYPREDRV::Fortran MPI::MPI_Fortran)
```

Fortran module files are compiler-specific. Build downstream code with the same
Fortran compiler family used to install hypredrive, otherwise the compiler may
not be able to read `hypredrive.mod`. The module is installed as
`include/hypredrive.mod` under the installation prefix.

Installed executables use a relative runtime search path so prefix-local
libraries in `lib` are found without setting `LD_LIBRARY_PATH`.

## Design

Use the module:

```fortran
use hypredrive
```

Handles are opaque C pointers:

```fortran
type(c_ptr) :: drv
```

MPI communicators are passed as Fortran MPI handles. The examples use `mpif.h`
instead of `use mpi` so they work with compiler/MPI stacks where `mpi.mod` is
compiler-specific. The binding converts the communicator to C internally:

```fortran
program driver
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr
   type(c_ptr) :: drv

   call MPI_Init(ierr)
   call HYPREDRV_Check(HYPREDRV_Initialize())
   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
   ! Install matrix/RHS and solve here.
   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   call HYPREDRV_Check(HYPREDRV_Finalize())
   call MPI_Finalize(ierr)
end program driver
```

String helpers are provided for the common YAML path:

```fortran
call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml_text))
```

For APIs that take C strings directly, pass null-terminated strings:

```fortran
call HYPREDRV_Check(HYPREDRV_LinearSystemGetSolutionNorm(drv, 'l2' // c_null_char, norm))
```

CSR and RHS array helpers use portable Fortran-side types and validate lengths
before calling the C library:

```fortran
integer(c_int64_t), allocatable :: indptr(:), cols(:)
real(c_double), allocatable :: data(:), rhs(:)

call HYPREDRV_Check(HYPREDRV_LinearSystemSetMatrixFromCSR(drv, row_start, row_end, indptr, cols, data))
call HYPREDRV_Check(HYPREDRV_LinearSystemSetRHSFromArray(drv, row_start, row_end, rhs))
```

The CSR helper expects normalized CSR slabs with `indptr(1) == 0` and
`indptr(size(indptr)) == size(cols) == size(data)`. Column indices and row bounds
are zero-based global indices, matching HYPRE IJ conventions. The Fortran 2003
binding accepts ordinary assumed-shape arrays; pass contiguous allocatable arrays
for large systems to avoid compiler-generated packing copies for strided
sections.

## Minimal example

```fortran
program solve
   use, intrinsic :: iso_c_binding
   use hypredrive
   implicit none
   include 'mpif.h'

   integer :: ierr
   type(c_ptr) :: drv
   character(len=:), allocatable :: yaml

   call MPI_Init(ierr)

   yaml = 'solver:' // new_line('a') // &
          '  pcg:' // new_line('a') // &
          '    relative_tol: 1.0e-8' // new_line('a') // &
          'preconditioner:' // new_line('a') // &
          '  amg:' // new_line('a') // &
          '    max_iter: 1' // new_line('a') // &
          '    tolerance: 0.0' // new_line('a')

   call HYPREDRV_Check(HYPREDRV_Initialize())
   call HYPREDRV_Check(HYPREDRV_Create(int(MPI_COMM_WORLD, c_int), drv))
   call HYPREDRV_Check(HYPREDRV_InputArgsParseYaml(drv, yaml))
   ! Install matrix/RHS or call HYPREDRV_LinearSystemBuild here.
   call HYPREDRV_Check(HYPREDRV_Destroy(drv))
   call HYPREDRV_Check(HYPREDRV_Finalize())

   call MPI_Finalize(ierr)
end program solve
```

## Examples

Examples live in `interfaces/fortran/examples`.

- `yaml_solve_mpi.f90` reads the same data files used by the C examples.
- `laplacian/laplacian.f90` assembles a distributed 3D Laplacian from Fortran
  CSR arrays and reads problem settings from namelist files in the same folder.

Run the Laplacian example with MPI:

```bash
mpiexec -n 2 ./build-fortran/laplacian-fortran interfaces/fortran/examples/laplacian/default.nml
```

## API coverage

The module exposes Fortran procedures for the public `HYPREDRV_` APIs in
`include/HYPREDRV.h`, including:

- lifecycle and error handling;
- object creation/destruction;
- YAML/input argument parsing;
- preset registration and selection;
- linear-system build, matrix/RHS injection, dofmaps, near-nullspace, and data
  accessors;
- state-vector operations;
- preconditioner and solver lifecycle;
- statistics and timing accessors;
- annotation hooks;
- optional eigenspectrum entry point.

The Fortran procedures intentionally preserve C ownership semantics. Objects
borrowed from HYPRE remain caller-owned unless the corresponding C API says
otherwise.

| C API group | Fortran module procedures |
| --- | --- |
| Lifecycle/info | `HYPREDRV_Initialize`, `HYPREDRV_Finalize`, `HYPREDRV_Create`, `HYPREDRV_Destroy`, `HYPREDRV_PrintLibInfo`, `HYPREDRV_PrintSystemInfo`, `HYPREDRV_PrintExitInfo` |
| Errors/helpers | `HYPREDRV_ErrorCodeDescribe`, `HYPREDRV_ErrorInvalidValue`, `HYPREDRV_BigIntSize`, `HYPREDRV_Check`, `HYPREDRV_ToCString` |
| Object/input/options | `HYPREDRV_SetLibraryMode`, `HYPREDRV_ObjectSetName`, `HYPREDRV_InputArgsParse`, `HYPREDRV_InputArgsParseYaml`, `HYPREDRV_InputArgsGet*`, `HYPREDRV_InputArgsSet*` |
| Presets | `HYPREDRV_SolverPresetRegister`, `HYPREDRV_PreconPresetRegister`, `HYPREDRV_InputArgsSetSolverPreset`, `HYPREDRV_InputArgsSetPreconPreset` |
| Linear system | `HYPREDRV_LinearSystemBuild`, `HYPREDRV_LinearSystemReadMatrix`, `HYPREDRV_LinearSystemSetMatrix*`, `HYPREDRV_LinearSystemSetRHS*`, `HYPREDRV_LinearSystemSetInitialGuess`, `HYPREDRV_LinearSystemResetInitialGuess`, `HYPREDRV_LinearSystemSetSolution`, `HYPREDRV_LinearSystemSetReferenceSolution`, `HYPREDRV_LinearSystemSetPrecMatrix`, `HYPREDRV_LinearSystemPrint` |
| Dofmaps/nullspace | `HYPREDRV_LinearSystemSetDofmap`, `HYPREDRV_LinearSystemSetInterleavedDofmap`, `HYPREDRV_LinearSystemSetContiguousDofmap`, `HYPREDRV_LinearSystemReadDofmap`, `HYPREDRV_LinearSystemPrintDofmap`, `HYPREDRV_LinearSystemSetNearNullSpace`, `HYPREDRV_LinearSystemSetNullSpace` |
| Accessors | `HYPREDRV_LinearSystemGetSolution*`, `HYPREDRV_LinearSystemGetRHS*`, `HYPREDRV_LinearSystemGetMatrix`, `HYPREDRV_LinearSystemGetSolutionNorm` |
| State vectors | `HYPREDRV_StateVectorSet`, `HYPREDRV_StateVectorGetValues`, `HYPREDRV_StateVectorCopy`, `HYPREDRV_StateVectorUpdateAll`, `HYPREDRV_StateVectorApplyCorrection` |
| Solver/preconditioner | `HYPREDRV_PreconCreate`, `HYPREDRV_PreconSetup`, `HYPREDRV_PreconApply`, `HYPREDRV_PreconDestroy`, `HYPREDRV_LinearSolverCreate`, `HYPREDRV_LinearSolverSetup`, `HYPREDRV_LinearSolverApply`, `HYPREDRV_LinearSolverDestroy` |
| Stats/timing | `HYPREDRV_StatsPrint`, `HYPREDRV_StatsLevelGetCount`, `HYPREDRV_StatsLevelGetEntry`, `HYPREDRV_StatsLevelPrint`, `HYPREDRV_LinearSolverGetNumIter`, `HYPREDRV_LinearSolverGetConverged`, `HYPREDRV_LinearSolverGetFinalRelativeResidualNorm`, `HYPREDRV_LinearSolverGetSetupTime`, `HYPREDRV_LinearSolverGetSolveTime` |
| Annotations/eigenspectrum | `HYPREDRV_Annotate*`, `HYPREDRV_LinearSystemComputeEigenspectrum` |

## Testing

The `fortran-test` target includes lifecycle checks and full `argv` input
parsing from `test_lifecycle.f90`, expected `HYPREDRV_Check` failure behavior
from `test_check_failure.f90`, MPI CSR assembly/solve coverage, and the
namelist-driven Laplacian cases.

## Notes

`HYPREDRV_SUCCESS` is defined separately as a C macro and as a Fortran module
parameter because Fortran `bind(C)` interfaces cannot import C preprocessor
macros.

The current CI job exercises GNU Fortran on Ubuntu. Other compiler families such
as Intel, NVHPC, Cray, and macOS Fortran stacks are intended future CI coverage.

## Limitations

- The first interface targets real-valued, double-precision HYPRE builds.
- Matrix and RHS convenience helpers use `integer(c_int64_t)` for global indices
  and `real(c_double)` for values, with conversion in the C bridge.
- Empty local CSR slabs are rejected by the convenience helper.
- The interface is not an object-oriented Fortran redesign; it is a thin binding
  plus small portability helpers.
