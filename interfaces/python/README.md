# hypredrive Python interface

Python bindings for [hypredrive][hypredrive], backed by the same C/HYPRE
library used by the CLI.

[hypredrive]: https://github.com/hypre-space/hypredrive

## Install

Run the following source-build commands from the repository root:

```bash
python -m pip install ./interfaces/python
```

This builds the Python extension and bundles the in-tree HYPREDRV/HYPRE
libraries automatically.

For editable development:

```bash
python -m pip install -e ./interfaces/python
```

Optional extras:

```bash
python -m pip install "./interfaces/python[scipy]"
python -m pip install "./interfaces/python[mpi]"
python -m pip install -e "./interfaces/python[test]"
```

To build against an installed hypredrive instead of bundling the in-tree
libraries:

```bash
cmake --install build --prefix $HOME/opt/hypredrive
python -m pip install ./interfaces/python \
  --config-settings=cmake.define.HYPREDRV_PYTHON_BUNDLE_CORE=OFF \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/opt/hypredrive
```

## Wheel artifacts

GitHub Actions builds MPI wheel artifacts for Linux, macOS, and Windows. Each
wheel bundles host-only `libHYPREDRV` and `libHYPRE`, but it does not bundle an
MPI runtime.

On pull requests, the wheel workflow runs when the PR has the
`Run Python Wheels` label or the shared `Generate Artifacts` label. The shared
label also runs the Julia artifact workflow; use the component-specific label
when only Python wheels are needed. It can also be started manually with
`workflow_dispatch` from the
[Python Wheels][python-wheels-workflow] workflow page.

On a `release-<version>` pull request, use `Prepare Release` to build and stage
both Python wheels and Julia archives on a draft release, refresh Julia's
artifact bindings, and validate the final candidate. `Run CI` keeps its existing
behavior. Tagging and publishing remain manual.

Each artifact is tied to an MPI flavor:

* `mpich` wheels require an MPICH-compatible runtime.
* `openmpi` wheels require an OpenMPI-compatible runtime.
* `msmpi` wheels are for Windows and require the Microsoft MPI runtime.

The release workflow builds CPython 3.10 through 3.14 wheels for MPICH and
OpenMPI on Linux x86_64/aarch64, MPICH and OpenMPI on macOS x86_64/arm64, and
the Windows x86_64 `msmpi` flavor. The
Windows distribution is named `hypredrive-msmpi`, while its import name is
still `hypredrive`. The Microsoft MPI runtime is intentionally external to the
wheel; install it before using distributed solves. The Windows CI uses the
binary `mpi4py` distribution rather than compiling `mpi4py` against a
different MPI implementation. Windows ARM64 is not included because the
Microsoft MPI build used by this wheel has no ARM64 runtime.

On a version tag, the workflow uploads all wheels directly to that tag's
GitHub Release. Pull-request and manually dispatched runs expose the wheels as
GitHub Actions artifacts instead.

Download the wheel artifact from the GitHub Actions `Python Wheels` workflow
run first. GitHub stores artifacts as zip files, so unzip the artifact before
installing the wheel:

```bash
python -m venv .venv
source .venv/bin/activate

unzip hypredrive-wheels-*.zip -d wheelhouse
python -m pip install --find-links=wheelhouse hypredrive
```

On Windows, use `hypredrive-msmpi` instead of `hypredrive` in the install
command. `--find-links` lets pip select a wheel compatible with your Python
version while fetching dependencies from the package index.

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
cmake --build build --target _core hypredrive_python_sources --parallel
cmake --build build --target python-test
```

Save the minimal example below as `minimal.py` and run it against this build
from the repository root:

```bash
PYTHONPATH="$PWD/build/interfaces/python/python" python minimal.py
```

This path does not replace Python packaging. It is useful for CI and local
development where the C library and Python extension should be built together.

## Minimal example

This serial example solves `A x = b` for `A = [[2, -1], [-1, 2]]` and
`b = [1, 1]`, giving `x = [1, 1]`.

Install the `scipy` extra for this example. `hd.solve` manages initialization
and driver cleanup and returns a copy of the solution in `result.x`.

```python
import numpy as np
import scipy.sparse as sp
import hypredrive as hd

A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
b = np.ones(2)
options = hd.configure(solver="pcg", preconditioner="amg")
result = hd.solve(A, b, options=options)
print(result.x)  # [1. 1.]
print(result.converged)
```

Check `result.converged`; failure to reach the solver tolerance does not raise
an exception. Customize options with `hd.configure`, for example:

```python
options = hd.configure(
    solver="pcg", preconditioner="amg", pcg={"relative_tol": 1.0e-8},
)
result = hd.solve(A, b, options=options)
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
  interfaces/python/tests/laplacian/test_example_mpi.py -v
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
* Examples live in `interfaces/python/examples/{laplacian,darcy}`.
* Developer benchmarks live in `interfaces/python/benchmarks`.
