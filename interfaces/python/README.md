# hypredrive Python interface

Python bindings for [hypredrive][hypredrive], backed by the same C/HYPRE
library used by the CLI.

[hypredrive]: https://github.com/hypre-space/hypredrive

## Install

From a full hypredrive checkout:

```bash
cd interfaces/python
python -m pip install .
```

This builds the Python extension and bundles the in-tree HYPREDRV/HYPRE
libraries automatically.

For editable development:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install .[scipy]
python -m pip install .[mpi]
python -m pip install -e .[test]
```

To build against an installed hypredrive instead of bundling the in-tree
libraries:

```bash
cmake --install build --prefix $HOME/opt/hypredrive
python -m pip install . \
  --config-settings=cmake.define.HYPREDRV_PYTHON_BUNDLE_CORE=OFF \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive
```

## Wheel artifacts

GitHub Actions can build experimental MPI wheel artifacts for Linux and macOS.
These wheels bundle host-only `libHYPREDRV` and `libHYPRE`, but they do not
bundle an MPI runtime.

On pull requests, the wheel workflow runs only when the PR has the
`Run Python Wheels` label. It can also be started manually with
`workflow_dispatch` from the
[Python Wheels][python-wheels-workflow] workflow page.

Each artifact is tied to an MPI flavor:

* `mpich` wheels require an MPICH-compatible runtime.
* `openmpi` wheels require an OpenMPI-compatible runtime.

Download the wheel artifact from the GitHub Actions `Python Wheels` workflow
run first. GitHub stores artifacts as zip files, so unzip the artifact before
installing the wheel:

```bash
python -m venv .venv
source .venv/bin/activate

unzip hypredrive-wheels-*.zip -d wheelhouse
python -m pip install wheelhouse/hypredrive-*.whl
```

Use a source install instead when you need a custom HYPRE build, GPU support,
BIGINT/MIXEDINT, vendor MPI, or downstream-packager control over shared
libraries.

At runtime, the package records how it was built:

```python
import hypredrive as hd
print(hd.BUILD_INFO)
```

[python-wheels-workflow]: https://github.com/hypre-space/hypredrive/actions/workflows/python-wheels.yml

## In-tree CMake build

The top-level project can build and test the Python extension as a developer
convenience:

```bash
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DHYPREDRV_ENABLE_PYTHON=ON
cmake --build build --target _core --parallel
cmake --build build --target python-test
```

This path does not replace Python packaging. It is useful for CI and local
development where the C library and Python extension should be built together.

## Example

```python
import numpy as np
import scipy.sparse as sp
import hypredrive as hd

n = 64
A = sp.diags(
    [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
    [-1, 0, 1],
    format="csr",
)
b = np.ones(n)

options = hd.configure(
    solver="pcg",
    preconditioner="amg",
    pcg={"relative_tol": 1.0e-8},
)

result = hd.solve(A, b, options=options)
print(result.solution_norm)
```

## MPI

Serial solves do not need `mpi4py`. Distributed Python solves use
`mpi4py.MPI.Comm`; hypredrive forwards the underlying communicator to the C
library rather than providing a separate Python MPI wrapper.

```python
from mpi4py import MPI
import hypredrive as hd

comm = MPI.COMM_WORLD
indptr, cols, data, b, row_start, row_end = build_local_slab(comm)

with hd.HypreDrive(options=options, comm=comm) as drv:
    drv.set_matrix_from_csr(indptr, cols, data, row_start=row_start, row_end=row_end)
    drv.set_rhs(b)
    drv.solve()
    x_local = drv.get_solution()
```

For raw CSR input, `row_start` and `row_end` are the inclusive global row range
owned by the rank, and `cols` contains global column indices.

## Tests

```bash
python -m pytest interfaces/python/tests/test_solve_serial.py -v
mpirun -np 2 python -m pytest \
  interfaces/python/tests/test_solve_mpi.py \
  interfaces/python/tests/test_laplacian_example_mpi.py -v
```

## Benchmarks

For developer benchmarks, prefer the CMake targets when using an in-tree build:

```bash
python -m pip install cython numpy scipy pyamg
cmake -S . -B build-bench \
  -DHYPREDRV_ENABLE_PYTHON=ON \
  -DPython_EXECUTABLE=$VIRTUAL_ENV/bin/python
cmake --build build-bench --target python-benchmark
```

The editable install path also works from a venv:

```bash
python -m pip install -e ./interfaces/python[bench] \
  --config-settings=cmake.define.HYPREDRV_DIR=$PWD/build
python interfaces/python/benchmarks/compare_laplacian.py
```

## Notes

* `options` may be a Python `dict`, YAML string, YAML file path, or `None`.
* SciPy sparse inputs are converted to CSR; pass CSR directly to avoid that
  conversion.
* Python arrays are copied/coerced to the HYPRE scalar and index types used by
  the linked C library.
* GPU/device execution is not exposed as a Python-native data path yet.
* Examples live in `interfaces/python/examples`.
* Developer benchmarks live in `interfaces/python/benchmarks`.
