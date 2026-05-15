"""Python interface to hypredrive.

Top-level public API:

* :class:`HypreDrive` -- stateful object-oriented driver.
* :func:`solve` -- one-shot helper that wraps the full lifecycle.
* :func:`initialize` / :func:`finalize` -- explicit lifecycle hooks.
* :class:`HypreDriveError` -- exception type raised on C-level failure.
* :class:`SolveResult` -- dataclass returned by :func:`solve`.

Module-level dtypes :data:`BIGINT_DTYPE` and :data:`REAL_DTYPE` describe
the types HYPRE was compiled with; user numpy arrays are coerced to these
before being passed to the native layer.
"""

from __future__ import annotations

from .driver import (
    BIGINT_DTYPE,
    HypreDrive,
    REAL_DTYPE,
    solve,
)
from .errors import HypreDriveError
from .options import OptionsLike, configure, normalize_options, options_to_yaml
from .result import SolveResult
from .session import finalize, initialize, is_initialized

__all__ = [
    "BIGINT_DTYPE",
    "HypreDrive",
    "HypreDriveError",
    "OptionsLike",
    "REAL_DTYPE",
    "SolveResult",
    "configure",
    "finalize",
    "initialize",
    "is_initialized",
    "normalize_options",
    "options_to_yaml",
    "solve",
]

__version__ = "0.3.0.dev0"
