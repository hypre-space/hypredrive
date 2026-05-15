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

import importlib

from .driver import BIGINT_DTYPE, REAL_DTYPE
from .options import OptionsLike, configure, normalize_options, options_to_yaml
from .result import SolveResult

_DRIVER_EXPORTS = {"HypreDrive", "solve"}
_SESSION_EXPORTS = {"finalize", "initialize", "is_initialized"}

BIGINT_DTYPE = None
HypreDrive = None
HypreDriveError = None
REAL_DTYPE = None
finalize = None
initialize = None
is_initialized = None
solve = None

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


def __getattr__(name: str):
    if name in _DRIVER_EXPORTS:
        driver = importlib.import_module(".driver", __name__)
        value = getattr(driver, name)
    elif name in _SESSION_EXPORTS:
        session = importlib.import_module(".session", __name__)
        value = getattr(session, name)
    elif name == "HypreDriveError":
        from .errors import HypreDriveError as value
    else:
        raise AttributeError(f"module 'hypredrive' has no attribute {name!r}")
    globals()[name] = value
    return value
