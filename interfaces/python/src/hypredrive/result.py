"""Result container for one-shot ``hypredrive.solve`` calls.

We use a plain frozen dataclass rather than something more elaborate
because the structure is essentially read-only once the C solve returns.
Dataclass equality is disabled because NumPy array value equality returns
arrays, not a scalar truth value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(eq=False, frozen=True)
class SolveResult:
    """Outcome of a single hypredrive linear-system solve.

    Attributes
    ----------
    x:
        Local-rank solution slab as a NumPy ``float64`` array of length
        ``row_end - row_start + 1``. The caller already owns this buffer:
        we copy out of HYPRE storage so subsequent solves do not mutate it.
    solution_norm:
        Convenience l2 norm of ``x``, computed inside the C library so the
        value is consistent with what the CLI prints. Useful for cheap
        smoke checks ("did anything happen?") without inspecting ``x``.
    iterations:
        Number of solver iterations.
    converged:
        Whether the solver reached its tolerance. ``False`` means it stopped
        early, e.g. at the maximum iteration count; :func:`hypredrive.solve`
        does not raise in that case, so check this flag.
    final_res_norm:
        Final relative residual norm reported by the solver.
    setup_time, solve_time:
        Preconditioner/solver setup and apply times, in seconds (milliseconds
        when ``general.use_millisec`` is enabled).
    """

    x: np.ndarray
    solution_norm: float
    iterations: Optional[int] = None
    converged: Optional[bool] = None
    final_res_norm: Optional[float] = None
    setup_time: Optional[float] = None
    solve_time: Optional[float] = None
    __hash__ = None
