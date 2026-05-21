# hypredrive C++ interface

The C++ interface is a header-only C++17 convenience layer over the public
`HYPREDRV_` C API. It does not introduce a separate solver implementation or ABI;
C++ applications still link to `HYPREDRV::HYPREDRV` through the exported
`HYPREDRV::CXX` target.

## Build

```bash
cmake -S . -B build-cpp \
  -DHYPREDRV_ENABLE_CPP=ON \
  -DHYPREDRV_ENABLE_TESTING=ON \
  -DHYPREDRV_ENABLE_EXAMPLES=ON
cmake --build build-cpp --target cpp-test --parallel
```

The public header is `hypredrive.hpp`.

## Use

```cpp
#include <hypredrive.hpp>

hypredrive::initialize();
hypredrive::driver drv(MPI_COMM_SELF);
drv.set_library_mode();
drv.parse_yaml(R"yaml(
general:
  statistics: 0
solver:
  pcg:
    max_iter: 100
    relative_tol: 1.0e-8
preconditioner:
  amg:
    print_level: 0
)yaml");
drv.set_matrix_from_csr(row_start, row_end, indptr, cols, values);
drv.set_rhs_from_array(row_start, row_end, rhs);
drv.set_zero_initial_guess();
drv.solve();
hypredrive::finalize();
```

The wrapper throws `hypredrive::error` for nonzero HYPREDRV status codes. The
exception stores the original status via `error::code()`. Call
`hypredrive::describe_error(e.code())` if you want HYPREDRV to print the C-side
diagnostic for a caught exception.

`hypredrive::driver` destroys its underlying `HYPREDRV_t` in its destructor, so
normal scope exit is enough. Call `destroy_linear_solver()` only when solver
state needs to be released before the driver itself is destroyed.

## Configuration

Solver configuration is intentionally YAML-driven. The C++ wrapper does not
mirror the full solver option schema with C++ setters, because the YAML schema is
the project-owned source of truth and supports new HYPREDRV options immediately.
Use `parse_yaml()` for YAML from a string, string literal, input stream, or file
path; use `parse_args(argc, argv)` for the same command-line style parsing
exposed by the C API. String-like inputs are treated as file paths when they look
like paths, otherwise as inline YAML text. YAML text is passed to the C parser as
a C string, so embedded NUL bytes are rejected.

## API coverage

The `driver` class mirrors the handle-based public `HYPREDRV_` API using
snake_case method names. Free functions cover global APIs such as initialization,
finalization, print helpers, and preset registration. Raw HYPRE object ownership
is unchanged from the C API; the C++ wrapper does not take ownership unless the C
API does.

## Example

`interfaces/cpp/examples/laplacian.cpp` assembles a distributed 3D 7-point
Laplacian directly in C++ and solves it through the wrapper:

```bash
mpiexec -n 4 ./build-cpp/laplacian-cpp -n 16 16 16 -P 2 2 1 -s 7

Pass a custom YAML file with `-i/--input options.yml`.
```
