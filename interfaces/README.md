# hypredrive interfaces

Language interfaces for hypredrive live here. Each interface is kept separate
from the core C library so it can own its packaging, examples, tests, and
developer tooling.

## Python

The Python interface is in [`python/`](python/). It provides NumPy/SciPy input
support, optional MPI usage through `mpi4py`, developer benchmarks, and
experimental Linux/macOS wheel artifacts.

For source builds against an installed hypredrive:

```bash
python -m pip install ./interfaces/python \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive
```

See [`python/README.md`](python/README.md) for wheel, source-build, in-tree
CMake, test, MPI, and benchmark details.
