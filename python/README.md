# hypredrive (Python interface)

Python bindings for [hypredrive][hypredrive], a C library and CLI driver
that solves sparse linear systems with [HYPRE][hypre] via YAML input.

The bindings let you:

* configure solver / preconditioner from a Python `dict`, a YAML string,
  or a YAML file path;
* assemble the matrix as CSR (NumPy arrays or a `scipy.sparse.csr_matrix`)
  and the RHS as a NumPy array, on either a single rank or distributed
  across MPI ranks via `mpi4py`;
* run setup + apply and pull the solution back as a NumPy array.

The bindings are deliberately thin: the heavy lifting (CSR -> ParCSR
assembly, AMG setup, Krylov iteration) all happens in the C library. This
keeps the Python footprint small and ensures behavior is identical to
running the CLI driver on the same YAML configuration.

[hypredrive]: https://github.com/hypre-space/hypredrive
[hypre]: https://github.com/hypre-space/hypre

## Installation

You need:

* Python ≥ 3.9, NumPy, and (for build) [Cython][cython] ≥ 3.0
  (pulled in by `pip` automatically via PEP 517 build requirements);
* an MPI implementation with `mpicc`;
* the hypredrive C library, either installed (i.e. there is an
  `HYPREDRVConfig.cmake` somewhere on `CMAKE_PREFIX_PATH`) or built in
  tree.

[cython]: https://cython.org

### Against an installed hypredrive

```bash
cmake --install build --prefix $HOME/opt/hypredrive
pip install ./python \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive
```

### Against an in-tree development build

```bash
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DHYPREDRV_ENABLE_TESTING=OFF
cmake --build build --parallel
pip install -e ./python \
  --config-settings=cmake.define.HYPREDRV_DIR=$PWD/build
```

The `HYPREDRV_DIR` form points scikit-build-core at the build directory
where `HYPREDRVConfig.cmake` is generated; no install step needed.

### Optional MPI integration

`mpi4py` is an optional dependency. Install it to drive distributed
solves:

```bash
pip install ./python[mpi]
```

## Quick start

### One-shot solve

```python
import numpy as np
import scipy.sparse as sp
import hypredrive as hd

# Build a 1D Poisson system in Python.
n = 64
diag_main = 2.0 * np.ones(n)
diag_off = -np.ones(n - 1)
A = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")
b = np.ones(n)

result = hd.solve(
    A,
    b,
    options={
        "solver": {"pcg": {"max_iter": 100, "relative_tol": 1.0e-8}},
        "preconditioner": {"amg": {"print_level": 0}},
    },
)
print("solution norm:", result.solution_norm)
print("first few entries:", result.x[:5])
```

### Reusable driver

`HypreDrive` is the object-oriented entry point; reuse it across multiple
solves to amortize hypredrive setup costs.

```python
with hd.HypreDrive(options="my_config.yaml") as drv:
    for step in range(num_steps):
        drv.set_matrix_from_csr(build_matrix(step))
        drv.set_rhs(build_rhs(step))
        drv.solve()
        results.append(drv.get_solution())
```

### Distributed (MPI)

```python
from mpi4py import MPI
import numpy as np
import hypredrive as hd

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myid = comm.Get_rank()

# Each rank assembles its rank-local CSR slab. Global rows are
# [row_start, row_end] inclusive; column indices are global.
indptr, col_indices, data, b_local, row_start, row_end = build_local_slab(myid, nprocs)

with hd.HypreDrive(options=opts, comm=comm) as drv:
    drv.set_matrix_from_csr(
        indptr, col_indices, data,
        row_start=row_start, row_end=row_end,
    )
    drv.set_rhs(b_local)
    drv.solve()
    x_local = drv.get_solution()
```

## Configuration

Anywhere `options` is accepted you can pass:

| Shape                                 | Behavior                                      |
| ------------------------------------- | --------------------------------------------- |
| `dict`                                | Translated to YAML in memory and parsed       |
| `str` containing a newline            | Treated as a YAML literal                     |
| `str` / `pathlib.Path` to a real file | File contents loaded and parsed               |
| `None`                                | Minimal default (statistics off, all defaults)|

The supported keys are exactly those documented for the YAML CLI; see
`docs/usrman-src/` in the main repository.

## Testing

After installing the package in editable mode and the optional test
dependencies (``pip install -e ./python[test]``), run:

```bash
python -m pytest python/tests/test_solve_serial.py -v
```

MPI integration tests must be launched under a process manager so that
``MPI.COMM_WORLD`` has multiple ranks:

```bash
mpirun -np 2 python -m pytest python/tests/test_solve_mpi.py -v
```

If you run ``test_solve_mpi.py`` without ``mpirun``, tests skip when only
one rank is available.

## Limitations (v1)

* The Python layer does not currently expose iteration counts or final
  residuals. Use `result.solution_norm` for a coarse sanity check.
* GPU / device execution is not exposed through the Python API yet
  (`exec_policy: device` may parse but the solution copy is
  host-resident).
* The `mpi4py` integration goes through `MPI_Comm_f2c`, so the binding
  works against any mpi4py version that exposes `Comm.py2f()`.
