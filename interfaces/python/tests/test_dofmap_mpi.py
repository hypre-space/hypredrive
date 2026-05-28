"""MPI smoke tests for MGR with an explicit DOF map.

Each rank assembles its local slab of the 2-DOF interleaved block system
(see ``_block_2dof_csr_local`` in conftest), passes a local labels array
to :meth:`hypredrive.HypreDrive.set_dofmap`, solves with GMRES+MGR, and
gathers the global solution on rank 0 for comparison against a dense
reference.

How to run::

    mpirun -np 2 python -m pytest interfaces/python/tests/test_dofmap_mpi.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
import hypredrive as hd
pytest.importorskip("mpi4py")

from mpi4py import MPI  # noqa: E402

from conftest import _block_2dof_csr_local, _block_2dof_dense_reference  # noqa: E402


def _require_parallel_world():
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("MPI dofmap tests need at least 2 ranks (use mpirun -np 2 …)")


def _partition_blocks(n_blocks: int, nprocs: int, rank: int) -> tuple[int, int]:
    base = n_blocks // nprocs
    rem = n_blocks % nprocs
    local_n = base + (1 if rank < rem else 0)
    start = rank * base + (rank if rank < rem else rem)
    return int(start), int(start + local_n - 1)


def _gather_x_on_root(comm: MPI.Comm, local_x: np.ndarray) -> np.ndarray | None:
    rank = comm.Get_rank()
    parts = comm.gather(local_x, root=0)
    if rank == 0:
        return np.concatenate(parts)
    return None


def test_mgr_solve_explicit_dofmap_mpi(mgr_options):
    _require_parallel_world()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    n_blocks_global = 16
    block_start, block_end = _partition_blocks(n_blocks_global, nprocs, rank)
    indptr, cols, data, labels, local_nrows = _block_2dof_csr_local(
        n_blocks_global=n_blocks_global,
        block_start=block_start,
        block_end=block_end,
        bigint=hd.BIGINT_DTYPE,
        real=hd.REAL_DTYPE,
    )
    row_start = 2 * block_start
    row_end = 2 * block_end + 1
    rhs = np.ones(local_nrows, dtype=hd.REAL_DTYPE)

    with hd.HypreDrive(options=mgr_options, comm=comm) as drv:
        drv.set_matrix_from_csr(
            indptr, cols, data, row_start=row_start, row_end=row_end,
        )
        drv.set_rhs(rhs)
        drv.set_dofmap(labels)
        drv.solve()
        x_local = drv.get_solution()

    assert x_local.shape == (local_nrows,)
    x_global = _gather_x_on_root(comm, x_local)
    if rank == 0:
        x_ref = _block_2dof_dense_reference(n_blocks_global)
        np.testing.assert_allclose(x_global, x_ref, rtol=1e-6, atol=1e-8)


def test_mgr_driver_lifecycle_mpi(mgr_options):
    """Two consecutive solves on one driver in parallel; labels survive."""
    _require_parallel_world()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    n_blocks_global = 16
    block_start, block_end = _partition_blocks(n_blocks_global, nprocs, rank)
    indptr, cols, data, labels, local_nrows = _block_2dof_csr_local(
        n_blocks_global=n_blocks_global,
        block_start=block_start,
        block_end=block_end,
        bigint=hd.BIGINT_DTYPE,
        real=hd.REAL_DTYPE,
    )
    row_start = 2 * block_start
    row_end = 2 * block_end + 1
    rhs = np.ones(local_nrows, dtype=hd.REAL_DTYPE)

    with hd.HypreDrive(options=mgr_options, comm=comm) as drv:
        drv.set_matrix_from_csr(
            indptr, cols, data, row_start=row_start, row_end=row_end,
        )
        drv.set_rhs(rhs)
        drv.set_dofmap(labels)
        drv.solve()
        x1 = drv.get_solution()
        drv.set_rhs(rhs)
        drv.solve()
        x2 = drv.get_solution()

    np.testing.assert_allclose(x1, x2, rtol=1e-12, atol=1e-14)
    g1 = _gather_x_on_root(comm, x1)
    if rank == 0:
        x_ref = _block_2dof_dense_reference(n_blocks_global)
        np.testing.assert_allclose(g1, x_ref, rtol=1e-6, atol=1e-8)
