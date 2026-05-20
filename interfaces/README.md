# hypredrive interfaces

Language interfaces for hypredrive live here. Each interface is kept separate
from the core C library so it can own its packaging, examples, tests, and
developer tooling.

## Fortran

The Fortran interface is in [`fortran/`](fortran/). It provides thin bindings to
the public `HYPREDRV_` C API, MPI examples, and CTest smoke tests.

Enable it with:

```bash
cmake -S . -B build-fortran -DHYPREDRV_ENABLE_FORTRAN=ON
cmake --build build-fortran --parallel
```

See [`fortran/README.md`](fortran/README.md) for API, build, test, and example
details.

## Python

The Python interface is in [`python/`](python/). It provides NumPy/SciPy input
support, optional MPI usage through `mpi4py`, developer benchmarks, and
experimental Linux/macOS MPI wheel artifacts.

For source builds against an installed hypredrive:

```bash
python -m pip install ./interfaces/python \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive
```

See [`python/README.md`](python/README.md) for wheel, source-build, in-tree
CMake, test, MPI, and benchmark details.

## MATLAB/Octave

The MATLAB/Octave interface is in [`matlab-octave/`](matlab-octave/). It provides a serial
MEX function plus a portable `.m` wrapper that accepts MATLAB or Octave sparse
matrices, converts them to hypredrive CSR, and solves on `MPI_COMM_SELF`.

Enable it with:

```bash
cmake -S . -B build-matlab -DHYPREDRV_ENABLE_MATLAB=ON
cmake --build build-matlab --parallel
```

See [`matlab-octave/README.md`](matlab-octave/README.md) for build, usage,
test, and install details.

## Julia

The Julia interface is in [`julia/`](julia/). It is an in-tree Julia package
for source-tree and install-prefix use via `Pkg.develop(path="interfaces/julia")`.
It builds a small ABI bridge library so Julia does not call HYPRE-width-dependent
C entry points directly.

Enable it with:

```bash
cmake -S . -B build-julia -DHYPREDRV_ENABLE_JULIA=ON -DHYPREDRV_ENABLE_TESTING=ON
cmake --build build-julia --target julia-test --parallel
```

See [`julia/README.md`](julia/README.md) for package development, library discovery,
options, tests, and example details.
