"""MPI smoke tests for the Python Laplacian example helpers.

Unlike ``test_solve_mpi.py``, this exercises the example path used by users:
Cartesian rank layout, 3D CSR assembly, ``Allgatherv`` of the distributed
solution, and a global residual reduction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
pytest.importorskip("mpi4py")

from mpi4py import MPI  # noqa: E402


def _require_parallel_world() -> None:
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("Laplacian example MPI test needs at least 2 ranks")


def _load_laplacian_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "laplacian.py"
    spec = importlib.util.spec_from_file_location("hypredrive_laplacian_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stencil", [7, 19])
def test_laplacian_example_distributed_residual(stencil: int) -> None:
    _require_parallel_world()
    example = _load_laplacian_example()
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    grid = example.make_grid(comm, (4, 3, 2), (nprocs, 1, 1))

    indptr, cols, data, rhs = example.build_csr(
        grid,
        stencil,
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )

    import hypredrive as hd

    with hd.HypreDrive(options=example.default_options(), comm=grid.comm) as drv:
        drv.set_matrix_from_csr(
            indptr,
            cols,
            data,
            row_start=grid.row_start,
            row_end=grid.row_end,
        )
        drv.set_rhs(rhs)
        drv.solve()
        x_local = drv.get_solution()
        solution_norm = drv.solution_norm()

    assert x_local.shape == rhs.shape
    assert np.all(np.isfinite(x_local))
    assert np.isfinite(solution_norm)

    x_global = example.gather_solution(grid, x_local)
    local_r2 = example.residual_norm(indptr, cols, data, rhs, x_global)
    residual = np.sqrt(grid.comm.allreduce(local_r2, op=MPI.SUM))
    assert residual < 1.0e-6
