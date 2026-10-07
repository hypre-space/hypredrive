"""Process-lifecycle tests that need a fresh interpreter.

MPI cannot be re-initialized after ``MPI_Finalize``, so anything that calls
``hypredrive.finalize()`` runs in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("hypredrive.driver")


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )


_SOLVE_PRELUDE = """
    import numpy as np
    import hypredrive as hd
    from hypredrive import _core

    n = 16
    indptr = np.zeros(n + 1, dtype=hd.BIGINT_DTYPE)
    cols, vals = [], []
    for i in range(n):
        for j, v in ((i - 1, -1.0), (i, 2.0), (i + 1, -1.0)):
            if 0 <= j < n:
                cols.append(j)
                vals.append(v)
        indptr[i + 1] = len(cols)
    opts = {
        "general": {"statistics": False},
        "solver": {"pcg": {"print_level": 0}},
        "preconditioner": {"amg": {"print_level": 0}},
    }
    drv = hd.HypreDrive(options=opts)
    drv.set_matrix_from_csr(indptr, cols, vals, row_start=0, row_end=n - 1)
    drv.set_rhs(np.ones(n))
    drv.solve()
"""


def test_finalize_closes_live_drivers():
    proc = _run(
        _SOLVE_PRELUDE
        + """
    assert len(_core._live_cores) == 1
    hd.finalize()
    assert len(_core._live_cores) == 0
    try:
        drv.get_solution()
    except RuntimeError as exc:
        assert "closed" in str(exc), exc
    else:
        raise AssertionError("driver still usable after finalize()")
    drv.close()  # already released by finalize(); must stay a no-op
    print("ok")
    """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_driver_alive_at_exit_is_released_by_atexit_hook():
    # The module-level driver outlives the atexit finalize hook; it must be
    # destroyed by that hook rather than during interpreter teardown, after
    # MPI has been finalized.
    # atexit runs hooks LIFO, so registering the probe before hypredrive's
    # first initialize() makes it run after the finalize hook.
    proc = _run(
        """
    import atexit, sys

    def probe():
        core = sys.modules["hypredrive._core"]
        print("live after finalize:", len(core._live_cores))

    atexit.register(probe)
    """
        + _SOLVE_PRELUDE
    )
    assert proc.returncode == 0, proc.stderr
    assert "live after finalize: 0" in proc.stdout
