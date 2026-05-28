"""Serial smoke tests for MGR with an explicit DOF map.

These exercise :meth:`hypredrive.HypreDrive.set_dofmap` against a synthetic
2-DOF interleaved block system (see ``block_2dof_system`` in conftest).
Configured GMRES + MGR with label 1 as F-points must converge to the dense
reference solution.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
import hypredrive as hd


def test_mgr_solve_explicit_dofmap(block_2dof_system, mgr_options):
    indptr, cols, data, labels, rhs, n_blocks, x_ref = block_2dof_system
    nrows = 2 * n_blocks

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        drv.set_rhs(rhs)
        drv.set_dofmap(labels)
        drv.solve()
        x = drv.get_solution()

    assert x.shape == (nrows,)
    assert x.dtype == hd.REAL_DTYPE
    np.testing.assert_allclose(x, x_ref, rtol=1e-6, atol=1e-8)


def test_mgr_without_dofmap_raises(block_2dof_system, mgr_options):
    indptr, cols, data, _labels, rhs, n_blocks, _x_ref = block_2dof_system
    nrows = 2 * n_blocks

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        drv.set_rhs(rhs)
        # Intentionally skip set_dofmap; MGR setup must fail.
        with pytest.raises(hd.HypreDriveError):
            drv.solve()


def test_set_dofmap_length_mismatch_rejected(block_2dof_system, mgr_options):
    indptr, cols, data, _labels, _rhs, n_blocks, _x_ref = block_2dof_system
    nrows = 2 * n_blocks

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        too_short = np.zeros(nrows - 1, dtype=np.intc)
        with pytest.raises(ValueError, match="does not match local row count"):
            drv.set_dofmap(too_short)


def test_set_dofmap_before_matrix_rejected(mgr_options):
    with hd.HypreDrive(options=mgr_options) as drv:
        with pytest.raises(RuntimeError, match="no matrix set"):
            drv.set_dofmap(np.zeros(4, dtype=np.intc))


def test_set_dofmap_dtype_coercion(block_2dof_system, mgr_options):
    indptr, cols, data, labels, rhs, n_blocks, x_ref = block_2dof_system
    nrows = 2 * n_blocks

    # Pass labels as a plain Python list and as int64; both must be coerced
    # to the C ``int`` width before reaching the binding.
    list_labels = [int(v) for v in labels.tolist()]
    int64_labels = np.asarray(labels, dtype=np.int64)

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        drv.set_rhs(rhs)
        drv.set_dofmap(list_labels)
        drv.solve()
        x_from_list = drv.get_solution()

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        drv.set_rhs(rhs)
        drv.set_dofmap(int64_labels)
        drv.solve()
        x_from_int64 = drv.get_solution()

    np.testing.assert_allclose(x_from_list, x_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x_from_int64, x_from_list, rtol=1e-12, atol=1e-14)


def test_dofmap_repeated_solve_preserves_labels(block_2dof_system, mgr_options):
    indptr, cols, data, labels, rhs, n_blocks, x_ref = block_2dof_system
    nrows = 2 * n_blocks

    with hd.HypreDrive(options=mgr_options) as drv:
        drv.set_matrix_from_csr(indptr, cols, data, row_start=0, row_end=nrows - 1)
        drv.set_rhs(rhs)
        drv.set_dofmap(labels)
        drv.solve()
        x1 = drv.get_solution()
        # Second solve without re-setting the dofmap: must succeed and
        # produce the same answer (deterministic under fixed YAML).
        drv.set_rhs(rhs)
        drv.solve()
        x2 = drv.get_solution()

    np.testing.assert_allclose(x1, x_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x2, x1, rtol=1e-12, atol=1e-14)
