"""End-to-end serial smoke tests for HypreDrive.

These rely on a working libHYPREDRV being discoverable at runtime; if the
extension module fails to import for any reason (e.g. wrong RPATH), the
whole module is skipped so an unconfigured CI does not produce noise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
import hypredrive as hd


def test_one_shot_solve(laplacian_1d, base_options):
    indptr, cols, data, rhs, n = laplacian_1d
    result = hd.solve(
        (indptr, cols, data),  # not a scipy matrix, so we pass tuple-style
        rhs,
        options=base_options,
        row_start=0,
        row_end=n - 1,
    )
    assert isinstance(result, hd.SolveResult)
    assert result.x.shape == (n,)
    assert result.x.dtype == hd.REAL_DTYPE
    # Solution is the integral of a 1D Poisson with Dirichlet endpoints; all
    # entries are positive.
    assert np.all(result.x > 0.0)
    # PCG+AMG should agree with our independent l2 norm computation.
    assert result.solution_norm == pytest.approx(
        float(np.linalg.norm(result.x)), rel=1e-6
    )
    same_values = hd.SolveResult(x=result.x.copy(), solution_norm=result.solution_norm)
    np.testing.assert_allclose(same_values.x, result.x)
    assert same_values.solution_norm == result.solution_norm
    assert result != same_values
    with pytest.raises(TypeError):
        hash(result)


def test_one_shot_row_range_must_be_paired(laplacian_1d, base_options):
    indptr, cols, data, rhs, n = laplacian_1d
    with pytest.raises(TypeError, match="provided together"):
        hd.solve((indptr, cols, data), rhs, options=base_options, row_start=0)
    with pytest.raises(TypeError, match="provided together"):
        hd.solve((indptr, cols, data), rhs, options=base_options, row_end=n - 1)


def test_driver_lifecycle(laplacian_1d, base_options):
    indptr, cols, data, rhs, n = laplacian_1d
    with hd.HypreDrive(options=base_options) as drv:
        # Pass the (indptr, cols, data) triple directly.
        drv.set_matrix_from_csr(
            indptr, cols, data, row_start=0, row_end=n - 1
        )
        drv.set_rhs(rhs)
        drv.solve()
        x1 = drv.get_solution()
        with pytest.raises(ValueError, match="kind must be one of"):
            drv.solution_norm("bad")
        # A second solve on the same driver must succeed and produce the
        # same answer (deterministic under fixed YAML).
        drv.set_rhs(rhs)
        drv.solve()
        x2 = drv.get_solution()
    np.testing.assert_allclose(x1, x2, rtol=1e-9, atol=1e-12)


def test_dtype_coercion(base_options):
    # Pass non-canonical dtypes; HypreDrive must coerce silently.
    n = 4
    indptr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    cols = np.array([0, 1, 2, 3], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rhs = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)

    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        x = drv.get_solution()
    # Solving a diagonal system with diag(1,2,3,4) and rhs=(1,4,9,16) gives
    # x = (1, 2, 3, 4).
    np.testing.assert_allclose(x, np.array([1.0, 2.0, 3.0, 4.0]), atol=1e-6)


def test_invalid_inputs_raise(base_options):
    with hd.HypreDrive(options=base_options) as drv:
        with pytest.raises(ValueError):
            drv.set_matrix_from_csr(
                np.array([0], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.REAL_DTYPE),
                row_start=0,
                row_end=5,  # claims 6 rows but indptr has 1 entry
            )
        with pytest.raises(ValueError, match="monotonically"):
            drv.set_matrix_from_csr(
                np.array([0, 2, 1], dtype=hd.BIGINT_DTYPE),
                np.array([0, 1], dtype=hd.BIGINT_DTYPE),
                np.array([1.0, 1.0], dtype=hd.REAL_DTYPE),
                row_start=0,
                row_end=1,
            )
        with pytest.raises(ValueError, match="HYPRE_BigInt range"):
            drv.set_matrix_from_csr(
                np.array([0], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.REAL_DTYPE),
                row_start=np.iinfo(hd.BIGINT_DTYPE).max + 1,
                row_end=np.iinfo(hd.BIGINT_DTYPE).max + 1,
            )


def test_explicit_initialize_finalize_is_noop_when_redundant(base_options, laplacian_1d):
    indptr, cols, data, rhs, n = laplacian_1d
    hd.initialize()
    hd.initialize()  # second call must be a no-op
    assert hd.is_initialized()
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        # Driver close happens at __exit__.
    # Don't call finalize() here: leave it to atexit so the process can run
    # multiple tests that each create their own HypreDrive.
