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

## Minimal example

This serial example solves `A x = b` for `A = [[2, -1], [-1, 2]]` and
`b = [1, 1]`, giving `x = [1, 1]`.

```cpp
#include <hypredrive.hpp>
#include <iostream>

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   hypredrive::initialize();
   {
      hypredrive::driver drv(MPI_COMM_SELF);
      drv.set_library_mode();
      drv.parse_yaml("general: {statistics: 0}\n"
                     "solver: pcg\npreconditioner: amg\n");
      const hypredrive::bigint indptr[] = {0, 2, 4};
      const hypredrive::bigint cols[] = {0, 1, 0, 1};
      const hypredrive::real values[] = {2.0, -1.0, -1.0, 2.0};
      const hypredrive::real rhs[] = {1.0, 1.0};
      drv.set_matrix_from_csr(0, 1, indptr, cols, values);
      drv.set_rhs_from_array(0, 1, rhs);
      drv.set_zero_initial_guess();
      drv.solve();
      const auto x = drv.get_solution_values();
      std::cout << x[0] << " " << x[1] << "\n";
   } // Destroy the driver before finalizing hypredrive and MPI.
   hypredrive::finalize();
   MPI_Finalize();
}
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

## Installed use

Build with `HYPREDRV_ENABLE_CPP=ON` and install hypredrive. Save the minimal
example as `minimal.cpp` and add this `CMakeLists.txt` beside it:

```cmake
cmake_minimum_required(VERSION 3.23)
project(minimal LANGUAGES C CXX)
find_package(HYPREDRV CONFIG REQUIRED)
add_executable(minimal minimal.cpp)
target_link_libraries(minimal PRIVATE HYPREDRV::CXX)
```

Configure this application with `-DCMAKE_PREFIX_PATH=/path/to/hypredrive/install`,
then build and run `minimal`. `HYPREDRV::CXX` supplies the C++17 requirement,
headers, and library dependencies.

With HYPRE or MPI in a nonstandard install prefix, ensure their shared libraries
are on the runtime search path (for example, `LD_LIBRARY_PATH` on Linux).

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
```

Pass a custom YAML file with `-i/--input options.yml`.
