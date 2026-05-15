"""Solve a 2D Laplacian problem A x = b through the hypredrive Python interface.

Model problem: 5-point finite-difference Laplacian on an n x n grid with
Dirichlet boundaries (handled implicitly by omitting off-grid neighbors).
The system is solved with PCG preconditioned by BoomerAMG.

Run as:

    mpirun -np 1 python interfaces/python/examples/laplacian2d_seq.py

(``mpirun`` is required because hypredrive's C library does not call
``MPI_Init`` itself; we trigger it by importing ``mpi4py.MPI``.)
"""

from __future__ import annotations

import mpi4py.MPI  # noqa: F401  -- side effect: calls MPI_Init

import numpy as np
import scipy.sparse as sp

import hypredrive as hd


def build_laplacian_2d(n: int) -> sp.csr_matrix:
    """Return the n*n-by-n*n 2D Laplacian (5-point stencil) as a CSR matrix."""
    e = np.ones(n)
    T = sp.diags([-e[:-1], 2.0 * e, -e[:-1]], [-1, 0, 1])
    return sp.kronsum(T, T, format="csr")


def main() -> None:
    n = 32
    A = build_laplacian_2d(n)
    b = np.ones(A.shape[0])

    result = hd.solve(
        A,
        b,
        options={
            "general": {"statistics": False, "exec_policy": "host"},
            "linear_system": {"init_guess_mode": "zeros"},
            "solver": {
                "pcg": {"max_iter": 200, "relative_tol": 1.0e-8, "print_level": 0},
            },
            "preconditioner": {"amg": {"print_level": 0}},
        },
    )

    residual_norm = float(np.linalg.norm(b - A @ result.x))
    print(f"grid           : {n} x {n} ({A.shape[0]} unknowns)")
    print(f"solution norm  : {result.solution_norm:.6e}")
    print(f"||b - A x||_2  : {residual_norm:.6e}")
    print(f"first 5 entries: {result.x[:5]}")


if __name__ == "__main__":
    main()
