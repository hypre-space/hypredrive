"""Validation tests for the mixed Darcy example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("hypredrive.driver")
pytest.importorskip("mpi4py")


def _load_darcy_mixed_example():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "darcy"
        / "darcy_mixed.py"
    )
    spec = importlib.util.spec_from_file_location("hypredrive_darcy_mixed", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("nx", "ny", "nz"),
    [
        (1, 16, 1),
        (1, 16, 16),
        (16, 1, 16),
    ],
)
def test_non_prefix_active_meshes_rejected(nx: int, ny: int, nz: int) -> None:
    example = _load_darcy_mixed_example()

    with pytest.raises(SystemExit, match="active dimensions must be a prefix"):
        example.validate_supported_mesh_axes(nx, ny, nz)


@pytest.mark.parametrize(
    ("nx", "ny", "nz"),
    [
        (16, 1, 1),
        (16, 16, 1),
        (16, 16, 16),
    ],
)
def test_prefix_active_meshes_accepted(nx: int, ny: int, nz: int) -> None:
    example = _load_darcy_mixed_example()

    example.validate_supported_mesh_axes(nx, ny, nz)


def test_parallel_bc_pinned_rows_are_zeroed_without_touching_free_rows() -> None:
    example = _load_darcy_mixed_example()
    mesh = example.Mesh(nx=3, ny=2, nz=1, hx=1.0 / 3.0, hy=0.5, hz=1.0)
    lay = example.ParallelLayout(mesh, Px=1, Py=1, Pz=1)
    off = 0
    total = int(lay.total[0])
    values = np.arange(1, total * total + 1, dtype=np.float64)
    A = sp.csr_matrix(values.reshape(total, total))
    rhs = np.ones(total, dtype=np.float64)

    A_new, rhs_new = example._apply_parallel_bc(
        A, rhs.copy(), mesh, lay, off, total, axis=0
    )

    pinned_parts = []
    for high in (False, True):
        gidx, _, _ = example._boundary_faces(mesh, lay, axis=1, high=high)
        pinned_parts.append(gidx - off)
    pinned = np.concatenate(pinned_parts)
    pinned_mask = np.zeros(total, dtype=bool)
    pinned_mask[pinned] = True

    dense = A_new.toarray()
    for row in pinned:
        expected = np.zeros(total, dtype=np.float64)
        expected[row] = 1.0
        np.testing.assert_allclose(dense[row], expected)
    np.testing.assert_allclose(dense[~pinned_mask], A.toarray()[~pinned_mask])
    np.testing.assert_allclose(rhs_new[pinned], 0.0)
