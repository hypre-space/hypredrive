"""Solve a 2D Laplacian A x = b distributed across MPI ranks via hypredrive.

Each rank assembles only its own rows of the global Laplacian as a
rectangular SciPy CSR matrix of shape ``(n_local, N)``: row indices are
local (``[0, n_local)``) while column indices are global. hypredrive's
Python interface accepts this directly when ``row_start`` / ``row_end``
are provided, and assembles the underlying HYPRE IJ matrix internally.

Run as:

    mpirun -np 4 python examples_python/laplacian2d_mpi.py
"""

from __future__ import annotations

from mpi4py import MPI

import numpy as np
import scipy.sparse as sp

import hypredrive as hd


def build_local_csr_slab(nx: int, ny: int, row_start: int, row_end: int):
    """Return the rank-local Laplacian slab as a SciPy CSR plus its RHS.

    The slab has shape ``(n_local, N)`` where ``N = nx*ny`` is the global
    number of unknowns and ``n_local = row_end - row_start + 1``. Global row
    ``r = row_start + k`` is stored at local row ``k``; column indices stay
    in the global numbering so the matrix can be matched against a global
    solution vector. Dirichlet boundaries are handled implicitly by omitting
    off-grid neighbors.
    """
    N = nx * ny
    n_local = row_end - row_start + 1
    indptr = [0]
    cols: list[int] = []
    data: list[float] = []
    for r in range(row_start, row_end + 1):
        ix, iy = r % nx, r // nx
        cols.append(r);          data.append(4.0)
        if ix > 0:      cols.append(r - 1);  data.append(-1.0)
        if ix < nx - 1: cols.append(r + 1);  data.append(-1.0)
        if iy > 0:      cols.append(r - nx); data.append(-1.0)
        if iy < ny - 1: cols.append(r + nx); data.append(-1.0)
        indptr.append(len(cols))
    A_local = sp.csr_matrix((data, cols, indptr), shape=(n_local, N))
    b_local = np.ones(n_local)
    return A_local, b_local


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    nx = ny = 32
    N = nx * ny

    counts = np.array(
        [N // nprocs + (1 if r < N % nprocs else 0) for r in range(nprocs)],
        dtype=np.int32,
    )
    displs = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)
    row_start = int(displs[rank])
    row_end = int(displs[rank] + counts[rank] - 1)

    A_local, b_local = build_local_csr_slab(nx, ny, row_start, row_end)

    options = {
        "general": {"statistics": False, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "pcg": {"max_iter": 200, "relative_tol": 1.0e-8, "print_level": 0},
        },
        "preconditioner": {"amg": {"print_level": 0}},
    }

    with hd.HypreDrive(options=options, comm=comm) as drv:
        drv.set_matrix_from_csr(A_local, row_start=row_start, row_end=row_end)
        drv.set_rhs(b_local)
        drv.solve()
        x_local = drv.get_solution()

    # Gather the full solution and compute the global residual norm. A_local
    # has global column indices, so multiplying by the gathered x_global gives
    # the correct local block of A x.
    x_global = np.empty(N, dtype=hd.REAL_DTYPE)
    comm.Allgatherv(x_local, [x_global, counts, displs, MPI.DOUBLE])

    r_local = b_local - A_local @ x_global
    res_norm = float(np.sqrt(comm.allreduce(float(r_local @ r_local), op=MPI.SUM)))

    if rank == 0:
        print(f"ranks          : {nprocs}")
        print(f"grid           : {nx} x {ny} ({N} unknowns)")
        print(f"rows per rank  : {counts.tolist()}")
        print(f"||b - A x||_2  : {res_norm:.6e}")
        print(f"x[0:5] (rank 0): {x_local[:5]}")


if __name__ == "__main__":
    main()
