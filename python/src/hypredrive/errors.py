"""Error types for the hypredrive Python interface.

We re-export the Cython-defined ``HypreDriveError`` so users can ``except``
it without reaching into the private ``_native`` module. Keeping a single
exception type avoids forcing user code to handle a sprawl of subclasses;
the ``code`` attribute on the exception carries the original C error
bitfield for callers that want to discriminate.
"""

from __future__ import annotations

from ._native import HypreDriveError

__all__ = ["HypreDriveError"]
