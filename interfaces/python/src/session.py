"""Process-wide hypredrive runtime lifecycle.

``HYPREDRV_Initialize`` and ``HYPREDRV_Finalize`` must bracket every
hypredrive-using region in a process. The C library tolerates redundant
calls, but it does *not* tolerate finalize-before-MPI_Finalize ordering
when an MPI runtime is in use, nor does it tolerate Python interpreter
teardown destroying handles after finalize.

We address both via a tiny module-level state machine:

* ``initialize()`` is idempotent and lazy. The first ``HypreDrive`` that
  needs it triggers it; user code may also call it explicitly to control
  ordering relative to ``mpi4py``'s ``Init``.
* ``initialize()`` initializes MPI through the same native MPI library linked
  into hypredrive if no other owner has initialized it yet. This avoids relying
  on a possibly ABI-mismatched ``mpi4py`` wheel for serial use.
* Re-entrant calls are no-ops, so embedding hypredrive inside a larger
  Python application that initializes/finalizes its own MPI is safe.
"""

from __future__ import annotations

import atexit
import threading

from . import _core

_lock = threading.Lock()
_initialized = False


def _atexit_finalize() -> None:
    """``atexit`` hook: only finalize if the user did not already do so."""
    finalize()


def initialize() -> None:
    """Initialize the hypredrive runtime if it has not been initialized yet.

    Safe to call multiple times. The first invocation registers an
    ``atexit`` hook so the runtime is torn down even if the user forgets
    to call :func:`finalize` explicitly.
    """
    global _initialized
    with _lock:
        if _initialized:
            return
        _core._initialize()
        _initialized = True
        # Register at first init rather than at module import so that
        # we never queue an atexit hook in processes that import
        # hypredrive but never use it (rare, but keeps the import
        # side-effect-free).
        atexit.register(_atexit_finalize)


def finalize() -> None:
    """Tear down the hypredrive runtime. Safe to call multiple times."""
    global _initialized
    with _lock:
        if not _initialized:
            return
        _core._finalize()
        _initialized = False


def is_initialized() -> bool:
    """Return ``True`` between :func:`initialize` and :func:`finalize`."""
    return _initialized
