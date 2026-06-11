"""Regression tests for ``linear_system.init_guess_mode`` in library mode.

These pin down two historical silent failures:

* ``previous`` copied from a working solution that had already been
  destroyed and recreated (zeroed) inside SetInitialGuess, so it behaved
  as ``zeros``;
* the Python driver never copied x0 into the working solution
  (``HYPREDRV_LinearSystemResetInitialGuess``), so *no* non-zero mode
  reached the solver.

Both manifested as "the solve converges anyway, just from zeros", which
solution-correctness checks cannot catch — hence these tests assert on
iteration counts, not only on solutions.
"""

from __future__ import annotations

import numpy as np
import pytest

hd = pytest.importorskip("hypredrive")


def _with_init_guess(options: dict, mode: str) -> dict:
    linear_system = {**options.get("linear_system", {}), "init_guess_mode": mode}
    return {**options, "linear_system": linear_system}


def _gmres_amg_options(mode: str) -> dict:
    """GMRES+AMG with the given init-guess mode.

    The iteration-count assertions below need GMRES: it accepts an
    already-converged initial guess at iteration 0, whereas hypre's PCG
    only checks convergence inside the first iteration and reports 1.
    """
    return {
        "general": {"statistics": False, "exec_policy": "host"},
        "linear_system": {"init_guess_mode": mode},
        "solver": {
            "gmres": {
                "max_iter": 100,
                "relative_tol": 1.0e-8,
                "print_level": 0,
            }
        },
        "preconditioner": {"amg": {"print_level": 0}},
    }


def test_previous_mode_starts_from_prior_solution(laplacian_1d):
    """Re-solving an identical system from ``previous`` takes 0 iterations."""
    indptr, cols, data, rhs, n = laplacian_1d
    options = _gmres_amg_options("previous")
    with hd.HypreDrive(options=options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        first_iters = drv.last_iterations
        x1 = drv.get_solution().copy()

        drv.set_rhs(rhs)
        drv.solve()
        second_iters = drv.last_iterations
        x2 = drv.get_solution()

    assert first_iters > 0
    # The previous solution already satisfies the stopping criterion, so the
    # solver must accept it without iterating. Anything > 0 here means the
    # previous solution never reached the iterate.
    assert second_iters == 0
    np.testing.assert_allclose(x2, x1, rtol=1e-10, atol=1e-12)


def test_previous_mode_first_solve_falls_back_to_zeros(laplacian_1d, base_options):
    """With no prior solution, ``previous`` must fall back to zeros, not fail."""
    indptr, cols, data, rhs, n = laplacian_1d
    options = _with_init_guess(base_options, "previous")
    with hd.HypreDrive(options=options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        assert drv.last_iterations > 0
        assert drv.solution_norm() > 0.0


def test_previous_mode_size_change_falls_back(laplacian_1d, base_options):
    """A system of a different size must not reuse the stale solution."""
    indptr, cols, data, rhs, n = laplacian_1d
    options = _with_init_guess(base_options, "previous")
    small_indptr = np.array([0, 1, 2], dtype=hd.BIGINT_DTYPE)
    small_cols = np.array([0, 1], dtype=hd.BIGINT_DTYPE)
    small_data = np.array([2.0, 4.0], dtype=hd.REAL_DTYPE)
    small_rhs = np.array([8.0, 16.0], dtype=hd.REAL_DTYPE)

    with hd.HypreDrive(options=options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()

        drv.set_matrix_from_csr(
            (small_indptr, small_cols, small_data), row_start=0, row_end=1
        )
        drv.set_rhs(small_rhs)
        drv.solve()
        np.testing.assert_allclose(
            drv.get_solution(), np.array([4.0, 4.0]), atol=1e-6
        )


def test_ones_mode_reaches_the_solver(base_options):
    """``ones`` must reach the iterate: 0 iterations when ones is exact.

    Guards the x0-to-working-solution copy independently of ``previous``:
    the 1D Laplacian with ``rhs = A @ ones`` has the all-ones exact
    solution, so a honored ``ones`` initial guess has zero residual.
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
    rhs = np.zeros(n, dtype=hd.REAL_DTYPE)
    rhs[0] = 1.0
    rhs[n - 1] = 1.0  # A @ ones

    options = _gmres_amg_options("ones")
    with hd.HypreDrive(options=options) as drv:
        drv.set_matrix_from_csr(indptr, cols_arr, data_arr, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        assert drv.last_iterations == 0
        np.testing.assert_allclose(drv.get_solution(), np.ones(n), atol=1e-8)
