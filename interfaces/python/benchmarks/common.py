"""Shared helpers for hypredrive Python benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` once and return ``(result, seconds)``."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def laplacian_matrix(dim: int, n: int):
    """Build a 2D or 3D finite-difference Laplacian as a SciPy CSR matrix."""
    import scipy.sparse as sp

    if dim not in {2, 3}:
        raise ValueError("dim must be 2 or 3")
    if n <= 1:
        raise ValueError("n must be greater than 1")

    e = np.ones(n)
    t = sp.diags([-e[:-1], 2.0 * e, -e[:-1]], [-1, 0, 1], format="csr")
    if dim == 2:
        return sp.kronsum(t, t, format="csr")
    return sp.kronsum(sp.kronsum(t, t, format="csr"), t, format="csr")


def rhs_for_matrix(matrix) -> np.ndarray:
    """Return the default benchmark RHS for ``matrix``."""
    return np.ones(matrix.shape[0], dtype=np.float64)


def residual_norm(matrix, x: np.ndarray, b: np.ndarray) -> float:
    """Compute ``||b - A x||_2`` independently of any solver backend."""
    return float(np.linalg.norm(b - matrix @ x))


def print_table(records: list[dict[str, Any]]) -> None:
    """Print a compact text table for benchmark records."""
    columns = [
        "backend",
        "setup_s",
        "solve_s",
        "total_s",
        "residual_l2",
        "solution_l2",
        "iterations",
    ]
    widths = {col: max(14, len(col)) for col in columns}
    for record in records:
        for col in columns:
            value = record.get(col)
            if isinstance(value, float):
                text = f"{value:.6e}"
            elif value is None:
                text = "-"
            else:
                text = str(value)
            widths[col] = max(widths[col], len(text))

    print(" ".join(f"{col:>{widths[col]}}" for col in columns))
    for record in records:
        values = []
        for col in columns:
            value = record.get(col)
            if isinstance(value, float):
                values.append(f"{value:{widths[col]}.6e}")
            elif value is None:
                values.append(f"{'-':>{widths[col]}}")
            else:
                values.append(f"{str(value):>{widths[col]}}")
        print(" ".join(values))


def emit_records(records: list[dict[str, Any]], json_path: str | None) -> None:
    """Emit benchmark records as text or JSON."""
    if json_path:
        text = json.dumps(records, indent=2, sort_keys=True)
        if json_path == "-":
            print(text)
        else:
            Path(json_path).write_text(text + "\n", encoding="utf-8")
        return
    print_table(records)
