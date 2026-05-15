"""End-to-end MPI smoke tests for HypreDrive (distributed CSR + MPI communicator).

These mirror :mod:`test_solve_serial` but assemble each rank's local CSR slab
of a 1D Laplacian and pass ``mpi4py.MPI.COMM_WORLD`` into
:class:`hypredrive.HypreDrive` / :func:`hypredrive.solve`.

**How to run**

Serial pytest only exercises one rank, so you must launch the MPI process
group explicitly::

    mpirun -np 2 python -m pytest python/tests/test_solve_mpi.py -v

If ``mpi4py`` or MPI is unavailable, the module is skipped. Tests that need
a multi-rank world skip when ``MPI.COMM_WORLD.Get_size() < 2``.

If the native extension is missing or broken, imports follow the same
``importorskip`` behavior as :mod:`test_solve_serial`.
"""

from __future__ import annotations

import numpy as np
import pytest

hd = pytest.importorskip("hypredrive")
pytest.importorskip("mpi4py")

from mpi4py import MPI  # noqa: E402


def _require_parallel_world():
    """Skip single-rank runs (e.g. plain ``pytest`` without ``mpirun``)."""
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("MPI parallel tests need at least 2 ranks (use mpirun -np 2 …)")


def _partition_rows(n: int, nprocs: int, rank: int) -> tuple[int, int]:
    """Contiguous row range [row_start, row_end] inclusive (same rule as C tests)."""
    base = n // nprocs
    rem = n % nprocs
    local_n = base + (1 if rank < rem else 0)
    start = rank * base + (rank if rank < rem else rem)
    row_start = int(start)
    row_end = int(start + local_n - 1)
    return row_start, row_end


def _build_local_laplacian_csr(
    n_global: int, row_start: int, row_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CSR for rows [row_start, row_end] of the 1D (-1,2,-1) Laplacian, rhs all ones."""
    bigint = hd.BIGINT_DTYPE
    real = hd.REAL_DTYPE
    nrows = row_end - row_start + 1
    cols: list[int] = []
    data: list[float] = []
    for i in range(nrows):
        grow = row_start + i
        if grow > 0:
            cols.append(grow - 1)
            data.append(-1.0)
        cols.append(grow)
        data.append(2.0)
        if grow < n_global - 1:
            cols.append(grow + 1)
            data.append(-1.0)
    indptr = np.zeros(nrows + 1, dtype=bigint)
    for i in range(nrows):
        grow = row_start + i
        nnz_i = 1
        if grow > 0:
            nnz_i += 1
        if grow < n_global - 1:
            nnz_i += 1
        indptr[i + 1] = indptr[i] + nnz_i
    cols_arr = np.asarray(cols, dtype=bigint)
    data_arr = np.asarray(data, dtype=real)
    rhs = np.ones(nrows, dtype=real)
    return indptr, cols_arr, data_arr, rhs


def _dense_poisson_reference(n: int) -> np.ndarray:
    """Global solution for 1D Poisson with Dirichlet (implicit from structure) — dense ref."""
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a[i, i] = 2.0
        if i > 0:
            a[i, i - 1] = -1.0
        if i < n - 1:
            a[i, i + 1] = -1.0
    b = np.ones(n, dtype=np.float64)
    return np.linalg.solve(a, b)


def _gather_x_on_root(comm: MPI.Comm, local_x: np.ndarray) -> np.ndarray | None:
    """Concatenate per-rank solution slabs on rank 0 (rank order = ascending rank)."""
    rank = comm.Get_rank()
    parts = comm.gather(local_x, root=0)
    if rank == 0:
        return np.concatenate(parts)
    return None


def test_one_shot_solve_mpi(base_options):
    """Distributed :func:`hypredrive.solve` matches the global dense reference."""
    _require_parallel_world()
    comm = MPI.COMM_WORLD
    n_global = 32
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    rs, re = _partition_rows(n_global, nprocs, rank)
    indptr, cols, data, rhs = _build_local_laplacian_csr(n_global, rs, re)

    result = hd.solve(
        (indptr, cols, data),
        rhs,
        options=base_options,
        comm=comm,
        row_start=rs,
        row_end=re,
    )
    assert isinstance(result, hd.SolveResult)
    assert result.x.shape == (re - rs + 1,)
    assert result.x.dtype == hd.REAL_DTYPE
    assert np.all(result.x > 0.0)

    x_global = _gather_x_on_root(comm, result.x)
    if rank == 0:
        ref = _dense_poisson_reference(n_global)
        np.testing.assert_allclose(x_global, ref, rtol=1e-5, atol=1e-7)
        assert result.solution_norm == pytest.approx(
            float(np.linalg.norm(ref)), rel=1e-5
        )


def test_driver_lifecycle_mpi(base_options):
    """Two consecutive solves on one :class:`hypredrive.HypreDrive` in parallel."""
    _require_parallel_world()
    comm = MPI.COMM_WORLD
    n_global = 32
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    rs, re = _partition_rows(n_global, nprocs, rank)
    indptr, cols, data, rhs = _build_local_laplacian_csr(n_global, rs, re)

    with hd.HypreDrive(options=base_options, comm=comm) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=rs, row_end=re)
        drv.set_rhs(rhs)
        drv.solve()
        x1 = drv.get_solution()
        drv.set_rhs(rhs)
        drv.solve()
        x2 = drv.get_solution()

    np.testing.assert_allclose(x1, x2, rtol=1e-9, atol=1e-12)

    g1 = _gather_x_on_root(comm, x1)
    if rank == 0:
        ref = _dense_poisson_reference(n_global)
        np.testing.assert_allclose(g1, ref, rtol=1e-5, atol=1e-7)
