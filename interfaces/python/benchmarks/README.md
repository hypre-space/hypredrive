# hypredrive Python benchmarks

Developer benchmarks for the Python interface. These scripts are not tests:
they report timings and residuals, and they should not be used as pass/fail CI
criteria.

Install optional serial benchmark dependencies:

```bash
python3 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install --upgrade pip
python -m pip install cython numpy scipy pyamg
```

When using an in-tree hypredrive build, configure CMake from the same activated
environment and use the benchmark target:

```bash
cmake -S . -B build -DHYPREDRV_ENABLE_PYTHON=ON
cmake --build build --target python-benchmark
```

The Python extension must be built and run with the same Python/NumPy ABI. If
you are using a virtual environment, either configure CMake while the venv is
active or pass the interpreter explicitly:

```bash
cmake -S . -B build-bench \
  -DHYPREDRV_ENABLE_PYTHON=ON \
  -DPython_EXECUTABLE=$VIRTUAL_ENV/bin/python
cmake --build build-bench --target python-benchmark
```

Run a serial reference comparison against PyAMG:

```bash
python interfaces/python/benchmarks/compare_laplacian.py --repeat 3
python interfaces/python/benchmarks/compare_laplacian.py --json
python interfaces/python/benchmarks/compare_laplacian.py \
  --pyamg-method sa --pyamg-method ruge-stuben --pyamg-method rootnode
```

This comparison uses the same SciPy CSR matrix, RHS, tolerance, and zero initial
guess. Hypredrive runs BoomerAMG-preconditioned PCG. PyAMG methods are also
used as PCG preconditioners through `ml.solve(..., accel="cg")`, not as
standalone multigrid solvers. PyAMG is a serial reference baseline here; it is
not a distributed performance competitor for hypredrive.

The serial comparison defaults to PyAMG smoothed aggregation (`pyamg-sa+pcg`),
Ruge-Stuben (`pyamg-rs+pcg`), root-node, and pairwise methods. AIR can be added
with `--pyamg-method air`.

Install optional MPI benchmark dependencies:

```bash
python -m pip install scipy mpi4py
```

Run a hypredrive-only MPI scaling benchmark:

```bash
mpiexec -n 4 python interfaces/python/benchmarks/scaling_laplacian.py \
  --n 24 24 24 --P 2 2 1 --repeat 3
```

Use `--json` on either script for machine-readable records suitable for later
plotting.
