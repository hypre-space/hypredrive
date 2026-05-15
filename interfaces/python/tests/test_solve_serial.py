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
from hypredrive import _core


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


def test_scipy_sparse_inputs(base_options):
    sp = pytest.importorskip("scipy.sparse")

    n = 8
    diagonals = [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)]
    offsets = [-1, 0, 1]
    rhs = np.ones(n, dtype=hd.REAL_DTYPE)

    csr = sp.diags(diagonals, offsets, format="csr")
    csc = csr.tocsc()

    csr_result = hd.solve(csr, rhs, options=base_options)
    csc_result = hd.solve(csc, rhs, options=base_options)

    np.testing.assert_allclose(csc_result.x, csr_result.x, rtol=1e-8, atol=1e-10)
    assert csc_result.solution_norm == pytest.approx(csr_result.solution_norm, rel=1e-8)

    slab_result = hd.solve(csr, rhs, options=base_options, row_start=0, row_end=n - 1)
    assert slab_result.x.shape == (n,)

    with pytest.raises(TypeError, match="provided together"):
        with hd.HypreDrive(options=base_options) as drv:
            drv.set_matrix_from_csr(csr, row_start=10)
    with pytest.raises(TypeError, match="provided together"):
        with hd.HypreDrive(options=base_options) as drv:
            drv.set_matrix_from_csr(csr, row_end=n - 1)


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


def test_replacing_matrix_mid_lifecycle(base_options):
    indptr = np.array([0, 1, 2], dtype=hd.BIGINT_DTYPE)
    cols = np.array([0, 1], dtype=hd.BIGINT_DTYPE)
    data1 = np.array([2.0, 4.0], dtype=hd.REAL_DTYPE)
    data2 = np.array([4.0, 8.0], dtype=hd.REAL_DTYPE)
    rhs = np.array([8.0, 16.0], dtype=hd.REAL_DTYPE)

    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr((indptr, cols, data1), row_start=0, row_end=1)
        drv.set_rhs(rhs)
        drv.solve()
        np.testing.assert_allclose(drv.get_solution(), np.array([4.0, 4.0]), atol=1e-6)

        drv.set_matrix_from_csr((indptr, cols, data2), row_start=0, row_end=1)
        drv.set_rhs(rhs)
        drv.solve()
        np.testing.assert_allclose(drv.get_solution(), np.array([2.0, 2.0]), atol=1e-6)


def test_core_rejects_oversized_solution_copy(laplacian_1d, base_options):
    indptr, cols, data, rhs, n = laplacian_1d
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        drv.set_rhs(rhs)
        drv.solve()
        out = np.empty(n + 1, dtype=hd.REAL_DTYPE)
        with pytest.raises(ValueError, match="exceeds local solution length"):
            drv._core.copy_solution(out)


def test_core_rejects_solution_copy_before_solve(base_options):
    hd.initialize()
    core = _core.HypreDriveCore()
    try:
        core.parse_yaml(hd.options_to_yaml(base_options).encode("utf-8"))
        out = np.empty(1, dtype=hd.REAL_DTYPE)
        with pytest.raises(hd.HypreDriveError):
            core.copy_solution(out)
    finally:
        core.close()


def test_core_bridge_reports_bigint_overflow(base_options):
    if hd.BIGINT_DTYPE.itemsize >= np.dtype(np.int64).itemsize:
        pytest.skip("active HYPRE_BigInt already spans int64")

    hd.initialize()
    core = _core.HypreDriveCore()
    try:
        core.parse_yaml(hd.options_to_yaml(base_options).encode("utf-8"))
        indptr = np.array([0], dtype=hd.BIGINT_DTYPE)
        cols = np.array([], dtype=hd.BIGINT_DTYPE)
        data = np.array([], dtype=hd.REAL_DTYPE)
        too_large = np.iinfo(np.int64).max
        with pytest.raises(_core.HypreDriveError) as exc:
            core.set_matrix_from_csr(too_large, too_large, indptr, cols, data)
        assert exc.value.code != 0
    finally:
        core.close()


def test_high_level_rejects_bigint_scalar_out_of_range(base_options):
    with hd.HypreDrive(options=base_options) as drv:
        with pytest.raises(ValueError, match="HYPRE_BigInt range"):
            drv.set_matrix_from_csr(
                np.array([0], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.BIGINT_DTYPE),
                np.array([], dtype=hd.REAL_DTYPE),
                row_start=np.iinfo(hd.BIGINT_DTYPE).max + 1,
                row_end=np.iinfo(hd.BIGINT_DTYPE).max + 1,
            )


def test_nonzero_indptr_offset_supported(base_options):
    indptr = np.array([2, 3], dtype=hd.BIGINT_DTYPE)
    cols = np.array([99, 99, 0], dtype=hd.BIGINT_DTYPE)
    data = np.array([-1.0, -1.0, 3.0], dtype=hd.REAL_DTYPE)
    rhs = np.array([6.0], dtype=hd.REAL_DTYPE)

    result = hd.solve((indptr, cols, data), rhs, options=base_options, row_start=0, row_end=0)
    np.testing.assert_allclose(result.x, np.array([2.0]), atol=1e-6)


def test_rhs_range_mismatch_rejected_by_core_layer(base_options):
    indptr = np.array([0, 1, 2], dtype=hd.BIGINT_DTYPE)
    cols = np.array([0, 1], dtype=hd.BIGINT_DTYPE)
    data = np.array([1.0, 1.0], dtype=hd.REAL_DTYPE)
    rhs = np.array([1.0], dtype=hd.REAL_DTYPE)

    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr((indptr, cols, data), row_start=0, row_end=1)
        with pytest.raises(hd.HypreDriveError):
            drv.set_rhs(rhs, row_start=0, row_end=0)


def test_sequence_of_mappings_config_parses(base_options):
    options = {
        **base_options,
        "preconditioner": {
            "amg": [
                {"coarsening": {"type": "HMIS", "strong_th": 0.25}},
                {"coarsening": {"type": "PMIS", "strong_th": 0.5}},
            ]
        },
    }
    hd.initialize()
    core = _core.HypreDriveCore()
    try:
        core.parse_yaml(hd.options_to_yaml(options).encode("utf-8"))
    finally:
        core.close()


def test_repeated_solve_with_preconditioner_reuse(laplacian_1d, base_options):
    indptr, cols, data, rhs, n = laplacian_1d
    options = {**base_options, "preconditioner": {
            "amg": {"print_level": 0},
            "reuse": {"enabled": "yes", "frequency": 2},
        },
    }
    with hd.HypreDrive(options=options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=n - 1)
        for _ in range(3):
            drv.set_rhs(rhs)
            drv.solve()
            assert drv.solution_norm() > 0.0


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
