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

try:
    from ._build_info import BUILD_INFO, BUNDLED_CORE, DISTRIBUTION_NAME, MPI_FLAVOR
except ImportError:  # pragma: no cover - only for direct source-tree imports
    DISTRIBUTION_NAME = "hypredrive"
    MPI_FLAVOR = "source"
    BUNDLED_CORE = False
    BUILD_INFO = {
        "distribution_name": DISTRIBUTION_NAME,
        "mpi_flavor": MPI_FLAVOR,
        "bundled_core": BUNDLED_CORE,
    }

from .errors import HypreDriveError
from .options import OptionsLike, configure, normalize_options, options_to_yaml
from .result import SolveResult

_DRIVER_EXPORTS = {"BIGINT_DTYPE", "HypreDrive", "REAL_DTYPE", "solve"}
_SESSION_EXPORTS = {"finalize", "initialize", "is_initialized"}

__all__ = [
    "BIGINT_DTYPE",
    "BUILD_INFO",
    "BUNDLED_CORE",
    "DISTRIBUTION_NAME",
    "HypreDrive",
    "HypreDriveError",
    "MPI_FLAVOR",
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


def initialize(*args, **kwargs):
    session = importlib.import_module(".session", __name__)
    return session.initialize(*args, **kwargs)


def finalize(*args, **kwargs):
    session = importlib.import_module(".session", __name__)
    return session.finalize(*args, **kwargs)


def is_initialized(*args, **kwargs):
    session = importlib.import_module(".session", __name__)
    return session.is_initialized(*args, **kwargs)


def __getattr__(name: str):
    if name in _DRIVER_EXPORTS:
        driver = importlib.import_module(".driver", __name__)
        value = getattr(driver, name)
    elif name in _SESSION_EXPORTS:
        session = importlib.import_module(".session", __name__)
        value = getattr(session, name)
    else:
        raise AttributeError(f"module 'hypredrive' has no attribute {name!r}")
    globals()[name] = value
    return value
