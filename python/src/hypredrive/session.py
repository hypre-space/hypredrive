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
* ``finalize()`` is registered with ``atexit`` *before* mpi4py's own
  finalize hook runs. This is achieved by importing ``mpi4py.MPI`` only
  on demand from the high-level driver; if the user has already imported
  mpi4py we simply piggyback on that ordering.
* Re-entrant calls are no-ops, so embedding hypredrive inside a larger
  Python application that initializes/finalizes its own MPI is safe.
"""

from __future__ import annotations

import atexit
import threading

from . import _native

_lock = threading.Lock()
_initialized = False
_atexit_registered = False


def _atexit_finalize() -> None:
    """``atexit`` hook: only finalize if the user did not already do so."""
    finalize()


def initialize() -> None:
    """Initialize the hypredrive runtime if it has not been initialized yet.

    Safe to call multiple times. The first invocation registers an
    ``atexit`` hook so the runtime is torn down even if the user forgets
    to call :func:`finalize` explicitly.
    """
    global _initialized, _atexit_registered
    with _lock:
        if _initialized:
            return
        _native._initialize()
        _initialized = True
        if not _atexit_registered:
            # Register at first init rather than at module import so that
            # we never queue an atexit hook in processes that import
            # hypredrive but never use it (rare, but keeps the import
            # side-effect-free).
            atexit.register(_atexit_finalize)
            _atexit_registered = True


def finalize() -> None:
    """Tear down the hypredrive runtime. Safe to call multiple times."""
    global _initialized
    with _lock:
        if not _initialized:
            return
        _native._finalize()
        _initialized = False


def is_initialized() -> bool:
    """Return ``True`` between :func:`initialize` and :func:`finalize`."""
    return _initialized
