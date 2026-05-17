# hypredrive interfaces

Language interfaces for hypredrive live here. Each interface is kept separate
from the core C library so it can own its packaging, examples, tests, and
developer tooling.

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
