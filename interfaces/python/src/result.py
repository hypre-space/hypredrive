"""Result container for one-shot ``hypredrive.solve`` calls.

We use a plain frozen dataclass rather than something more elaborate
because the structure is essentially read-only once the C solve returns.
Users typically only consume ``x``; the diagnostic fields are best-effort
and may be ``None`` if the solver does not expose the corresponding
counter through the current C API surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
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
        Reserved for future use. Always ``None`` in v1; will be wired up
        when the C API exposes a stable getter.
    """

    x: np.ndarray
    solution_norm: float
    iterations: Optional[int] = None
