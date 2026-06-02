"""Pytest fixtures shared across the hypredrive Python test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def base_options() -> dict:
    """A minimal, deterministic PCG+AMG configuration.

    Used as the default for solve-cycle tests so we exercise a real solver
    rather than relying on hypredrive's silent defaults (which may change).

    Function-scoped so a test that mutates a nested entry does not leak the
    change into siblings via shared dict state.
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
def mgr_options() -> dict:
    """GMRES + MGR (1 level, jacobi F-relaxation, AMG coarse).

    Configured against the 2-DOF interleaved block system in
    :func:`block_2dof_csr`: F-points = label 1 (the ``v`` DOF), so MGR
    eliminates ``v`` and hands the Schur-complement system to AMG.

    Function-scoped so a test that tweaks ``solver.gmres.max_iter`` (etc.)
    does not poison sibling tests via the shared mutable dict.
    """
    return {
        "general": {"statistics": False, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": "zeros"},
        "solver": {
            "gmres": {
                "max_iter": 200,
                "krylov_dim": 30,
                "relative_tol": 1.0e-10,
                "absolute_tol": 0.0,
                "print_level": 0,
            }
        },
        "preconditioner": {
            "mgr": {
                "tolerance": 0.0,
                "max_iter": 1,
                "print_level": 0,
                "coarse_th": 0.0,
                "level": {
                    "0": {
                        "f_dofs": [1],
                        "f_relaxation": "jacobi",
                        "g_relaxation": "none",
                        "restriction_type": "injection",
                        "prolongation_type": "jacobi",
                        "coarse_level_type": "rap",
                    },
                },
                "coarsest_level": {
                    "amg": {
                        "tolerance": 0.0,
                        "max_iter": 1,
                        "print_level": 0,
                    },
                },
            }
        },
    }


def _block_2dof_csr_local(
    n_blocks_global: int,
    block_start: int,
    block_end: int,
    bigint,
    real,
):
    """Build the local CSR slab + labels for the 2-DOF interleaved system.

    Block ``k`` contributes two rows ``2*k`` (label 0, "u") and ``2*k+1``
    (label 1, "v"). The matrix is
        A_uu = standard 1D Laplacian (-1, 2, -1) over the blocks,
        A_vv = 2 I (diagonal),
        A_uv = A_vu = 0.1 I (intra-block coupling).
    The structure is SPD, which keeps the dense reference inexpensive,
    and gives MGR an actual Schur complement to coarsen.
    """
    import numpy as _np

    eps = 0.1
    local_blocks = block_end - block_start + 1
    nrows = 2 * local_blocks
    cols: list[int] = []
    data: list[float] = []
    indptr = _np.zeros(nrows + 1, dtype=bigint)
    labels: list[int] = []
    for i in range(local_blocks):
        k = block_start + i  # global block index
        # u_k (row 2*k, label 0)
        if k > 0:
            cols.append(2 * (k - 1))
            data.append(-1.0)
        cols.append(2 * k)
        data.append(2.0)
        if k < n_blocks_global - 1:
            cols.append(2 * (k + 1))
            data.append(-1.0)
        cols.append(2 * k + 1)
        data.append(eps)
        indptr[2 * i + 1] = len(cols)
        labels.append(0)
        # v_k (row 2*k+1, label 1)
        cols.append(2 * k)
        data.append(eps)
        cols.append(2 * k + 1)
        data.append(2.0)
        indptr[2 * i + 2] = len(cols)
        labels.append(1)
    cols_arr = _np.asarray(cols, dtype=bigint)
    data_arr = _np.asarray(data, dtype=real)
    labels_arr = _np.asarray(labels, dtype=_np.intc)
    return indptr, cols_arr, data_arr, labels_arr, nrows


def _block_2dof_dense_reference(n_blocks: int):
    """Dense reference solution for the 2-DOF system with RHS = ones."""
    import numpy as _np

    eps = 0.1
    n = 2 * n_blocks
    a = _np.zeros((n, n), dtype=_np.float64)
    for k in range(n_blocks):
        a[2 * k, 2 * k] = 2.0
        if k > 0:
            a[2 * k, 2 * (k - 1)] = -1.0
        if k < n_blocks - 1:
            a[2 * k, 2 * (k + 1)] = -1.0
        a[2 * k, 2 * k + 1] = eps
        a[2 * k + 1, 2 * k + 1] = 2.0
        a[2 * k + 1, 2 * k] = eps
    b = _np.ones(n, dtype=_np.float64)
    return _np.linalg.solve(a, b)


@pytest.fixture
def block_2dof_system():
    """Whole-system 2-DOF block fixture for serial dofmap tests.

    Returns ``(indptr, cols, data, labels, rhs, n_blocks, x_ref)`` where
    the CSR / labels span all rows and ``x_ref`` is the dense reference
    solution for cross-checking.
    """
    hd = pytest.importorskip("hypredrive")
    n_blocks = 16
    indptr, cols, data, labels, nrows = _block_2dof_csr_local(
        n_blocks_global=n_blocks,
        block_start=0,
        block_end=n_blocks - 1,
        bigint=hd.BIGINT_DTYPE,
        real=hd.REAL_DTYPE,
    )
    rhs = np.ones(nrows, dtype=hd.REAL_DTYPE)
    x_ref = _block_2dof_dense_reference(n_blocks)
    return indptr, cols, data, labels, rhs, n_blocks, x_ref


@pytest.fixture
def laplacian_1d():
    """Return a CSR representation of the 1D Laplacian and an all-ones RHS.

    The matrix is small (32 rows) so tests stay fast; PCG+AMG converges in
    a handful of iterations, which keeps the test deterministic across
    HYPRE versions.
    """
    hd = pytest.importorskip("hypredrive")
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
