"""Validate NumPy buffers before their addresses cross the native boundary."""

from __future__ import annotations

import numpy as np


def validate_native_array(array, name, dtype, *, writable=False):
    """Require a one-dimensional, aligned native buffer of the exact C type."""
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    if array.dtype != np.dtype(dtype) or not array.dtype.isnative:
        raise ValueError(f"{name} dtype must be native {np.dtype(dtype)}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError(f"{name} must be aligned")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")


def validate_csr(row_start, row_end, indptr, col_indices, data):
    """Check lengths and offsets before C reads the CSR pointers.

    The C API accepts nonzero initial offsets into the columns/values arrays.
    These arrays therefore need storage through ``indptr[-1]``, not merely
    ``indptr[-1] - indptr[0]`` entries.
    """
    nrows = int(row_end) - int(row_start) + 1
    if nrows <= 0:
        raise ValueError(f"row_end ({row_end}) must be >= row_start ({row_start})")
    if indptr.shape[0] != nrows + 1:
        raise ValueError(
            f"indptr length {indptr.shape[0]} does not match nrows+1 = {nrows + 1}"
        )
    start, end = int(indptr[0]), int(indptr[-1])
    if start < 0:
        raise ValueError(f"indptr[0] ({start}) must be nonnegative")
    # Comparing adjacent entries avoids signed overflow in np.diff.
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr entries must be monotonically nondecreasing")
    if col_indices.shape[0] < end or data.shape[0] < end:
        raise ValueError(f"col_indices/data must have at least indptr[-1]={end} entries")


def validate_rhs(row_start, row_end, values):
    nrows = int(row_end) - int(row_start) + 1
    if nrows <= 0:
        raise ValueError(f"row_end ({row_end}) must be >= row_start ({row_start})")
    if values.shape[0] != nrows:
        raise ValueError(
            f"RHS length {values.shape[0]} does not match row range size {nrows}"
        )
