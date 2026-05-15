# hypredrive interfaces

Language interfaces for hypredrive live here. Each interface is kept separate
from the core C library so it can own its packaging, examples, tests, and
developer tooling.

## Python

The Python interface is in [`python/`](python/). It provides NumPy/SciPy input
support, optional MPI usage through `mpi4py`, and developer benchmarks.

Start with:

```bash
python -m pip install ./interfaces/python
```

See [`python/README.md`](python/README.md) for build, test, MPI, and benchmark
details.
