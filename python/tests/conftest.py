"""Pytest fixtures shared across the hypredrive Python test suite."""

from __future__ import annotations

import mpi4py.MPI  # noqa: F401  -- import side-effect: calls MPI_Init
import numpy as np
import pytest

import hypredrive as hd


@pytest.fixture(scope="session")
def base_options() -> dict:
    """A minimal, deterministic PCG+AMG configuration.

    Used as the default for solve-cycle tests so we exercise a real solver
    rather than relying on hypredrive's silent defaults (which may change).
    """
    return {
        "general": {"statistics": False, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "pcg": {
                "max_iter": 100,
                "relative_tol": 1.0e-8,
                "print_level": 0,
            }
        },
        "preconditioner": {"amg": {"print_level": 0}},
    }


@pytest.fixture
def laplacian_1d():
    """Return a CSR representation of the 1D Laplacian and an all-ones RHS.

    The matrix is small (32 rows) so tests stay fast; PCG+AMG converges in
    a handful of iterations, which keeps the test deterministic across
    HYPRE versions.
    """
    n = 32
    indptr = np.zeros(n + 1, dtype=hd.BIGINT_DTYPE)
    cols: list[int] = []
    data: list[float] = []
    for i in range(n):
        if i > 0:
            cols.append(i - 1)
            data.append(-1.0)
        cols.append(i)
        data.append(2.0)
        if i < n - 1:
            cols.append(i + 1)
            data.append(-1.0)
        indptr[i + 1] = len(cols)
    cols_arr = np.asarray(cols, dtype=hd.BIGINT_DTYPE)
    data_arr = np.asarray(data, dtype=hd.REAL_DTYPE)
    rhs = np.ones(n, dtype=hd.REAL_DTYPE)
    return indptr, cols_arr, data_arr, rhs, n
