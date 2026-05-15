"""Error types for the hypredrive Python interface.

This module is intentionally pure Python so ``import hypredrive`` does not
eagerly import the native extension. The Cython layer imports and raises the
same ``HypreDriveError`` class defined here.
"""

from __future__ import annotations


class HypreDriveError(RuntimeError):
    """Exception raised when a hypredrive C call returns a nonzero error code.

    The ``code`` attribute carries the bitfield exactly as returned by the
    library; ``HYPREDRV_ErrorCodeDescribe`` will have already printed a
    human-readable description to stderr by the time this is raised.
    """

    def __init__(self, code: int, msg: str):
        super().__init__(f"{msg} (code=0x{int(code):08x})")
        self.code = int(code)

__all__ = ["HypreDriveError"]
