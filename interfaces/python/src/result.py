"""Result container for one-shot ``hypredrive.solve`` calls.

We use a plain frozen dataclass rather than something more elaborate
because the structure is essentially read-only once the C solve returns.
Dataclass equality is disabled because NumPy array value equality returns
arrays, not a scalar truth value.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """

    x: np.ndarray
    solution_norm: float
    __hash__ = None
