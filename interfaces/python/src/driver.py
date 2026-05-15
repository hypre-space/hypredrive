"""High-level Python interface to hypredrive.

The :class:`HypreDrive` class is a thin orchestration layer over the
private Cython binding. Its job is twofold:

* normalize the various input shapes a user is likely to supply
  (``scipy.sparse.csr_matrix`` vs. raw ``(indptr, indices, data)`` slabs;
  Python ``dict`` of options vs. YAML literal vs. file path);
* enforce a small set of correctness invariants the C API does not
  natively check (e.g. dtypes match the platform's ``HYPRE_BigInt``,
  RHS length matches the matrix row range).

Anything beyond that (the actual solve, error reporting, ownership) is
delegated to the C library. The class is intentionally not a thin
``ctypes`` mirror of every C entry point: that would couple Python users
to internal naming choices and make later refactors painful.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import _native, session
from .options import OptionsLike, normalize_options
from .result import SolveResult

__all__ = ["HypreDrive", "solve", "BIGINT_DTYPE", "REAL_DTYPE"]


# ---------------------------------------------------------------------------
# Platform-derived dtypes
#
# HYPRE_BigInt is either ``int32`` or ``int64`` depending on whether HYPRE
# was built with mixed-int support. The exact dtype is observable at native
# build time and exposed through ``_native._hypre_bigint_size``; we surface
# it as a module-level numpy dtype so user code does not have to guess.
# ---------------------------------------------------------------------------

def _resolve_bigint_dtype() -> np.dtype:
    size = _native._hypre_bigint_size()
    if size == 4:
        return np.dtype(np.int32)
    if size == 8:
        return np.dtype(np.int64)
    raise RuntimeError(
        f"Unsupported HYPRE_BigInt size {size} reported by the native module"
    )


def _resolve_real_dtype() -> np.dtype:
    size = _native._hypre_real_size()
    if size == 8:
        return np.dtype(np.float64)
    if size == 4:
        return np.dtype(np.float32)
    raise RuntimeError(
        f"Unsupported HYPRE_Real size {size} reported by the native module"
    )


BIGINT_DTYPE: np.dtype = _resolve_bigint_dtype()
REAL_DTYPE: np.dtype = _resolve_real_dtype()


def _as_bigint_array(arr: Any, name: str) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=BIGINT_DTYPE)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {a.shape}")
    return a


def _as_real_array(arr: Any, name: str) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=REAL_DTYPE)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {a.shape}")
    return a


def _validate_bigint_scalar(value: Any, name: str) -> int:
    ivalue = int(value)
    info = np.iinfo(BIGINT_DTYPE)
    if ivalue < int(info.min) or ivalue > int(info.max):
        raise ValueError(
            f"{name} ({ivalue}) is outside the HYPRE_BigInt range "
            f"[{int(info.min)}, {int(info.max)}]"
        )
    return ivalue


# ---------------------------------------------------------------------------
# HypreDrive: object-oriented entry point
# ---------------------------------------------------------------------------

class HypreDrive:
    """Stateful driver wrapping a single ``HYPREDRV_t`` handle.

    Typical usage::

        with hypredrive.HypreDrive(options=...) as drv:
            drv.set_matrix_from_csr(A_csr)
            drv.set_rhs(b)
            drv.solve()
            x = drv.get_solution()

    The driver always operates in *library mode*: it expects the matrix
    and RHS to be supplied via ``set_matrix_from_csr`` / ``set_rhs``. If
    you want the legacy CLI behavior of reading from a YAML file with
    on-disk matrix/RHS data, use the C driver directly.
    """

    def __init__(
        self,
        options: OptionsLike = None,
        comm: Any = None,
    ) -> None:
        session.initialize()
        self._core: Optional[_native.HypreDriveCore] = _native.HypreDriveCore(
            comm=comm,
            library_mode=True,
        )
        self._row_start: Optional[int] = None
        self._row_end: Optional[int] = None
        self._rhs_set: bool = False

        # Always parse a YAML configuration up front so the C side has a
        # solver/preconditioner method selected before we hand over data.
        yaml_text = normalize_options(options)
        self._core.parse_yaml(yaml_text.encode("utf-8"))

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "HypreDrive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HYPREDRV handle. Idempotent."""
        if self._core is not None:
            self._core.close()
            self._core = None

    # ------------------------------------------------------------------
    # Data ingest
    # ------------------------------------------------------------------

    def set_matrix_from_csr(
        self,
        matrix_or_indptr: Any,
        col_indices: Any = None,
        data: Any = None,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
    ) -> None:
        """Install a system matrix from CSR.

        Two call shapes are accepted:

        1. ``set_matrix_from_csr(scipy_csr_matrix)``: the ``indptr``,
           ``indices``, ``data`` triple is taken from the SciPy object.
           ``row_start`` defaults to ``0`` and ``row_end`` to
           ``shape[0] - 1``. If an explicit row range is supplied with a
           SciPy matrix, the matrix is interpreted as this rank's local slab
           and must have ``shape[0] == row_end - row_start + 1``.
        2. ``set_matrix_from_csr(indptr, col_indices, data,
           row_start=..., row_end=...)``: distributed mode, useful when
           each MPI rank owns a slab of rows.
        """
        self._require_open()

        if col_indices is None and data is None:
            if isinstance(matrix_or_indptr, tuple) and len(matrix_or_indptr) == 3:
                indptr, indices, values = matrix_or_indptr
            elif hasattr(matrix_or_indptr, "tocsr"):
                csr = matrix_or_indptr.tocsr()
                indptr, indices, values = csr.indptr, csr.indices, csr.data
                if row_start is None:
                    row_start = 0
                if row_end is None:
                    row_end = csr.shape[0] - 1
            else:
                raise TypeError(
                    "set_matrix_from_csr expects a scipy.sparse matrix or "
                    "the (indptr, col_indices, data) triple"
                )
        else:
            indptr, indices, values = matrix_or_indptr, col_indices, data

        if row_start is None or row_end is None:
            raise TypeError(
                "row_start and row_end are required when passing raw "
                "(indptr, col_indices, data) arrays"
            )

        indptr_arr = _as_bigint_array(indptr, "indptr")
        indices_arr = _as_bigint_array(indices, "col_indices")
        values_arr = _as_real_array(values, "data")

        row_start_i = _validate_bigint_scalar(row_start, "row_start")
        row_end_i = _validate_bigint_scalar(row_end, "row_end")

        nrows = row_end_i - row_start_i + 1
        if nrows < 0:
            raise ValueError(
                f"row_end ({row_end}) must be >= row_start ({row_start})"
            )
        if indptr_arr.shape[0] != nrows + 1:
            raise ValueError(
                f"indptr length {indptr_arr.shape[0]} does not match "
                f"nrows+1 = {nrows + 1}"
            )
        indptr_start = int(indptr_arr[0])
        indptr_end = int(indptr_arr[-1])
        if indptr_start < 0:
            raise ValueError(
                f"indptr[0] ({indptr_start}) must be nonnegative"
            )
        if indptr_end < indptr_start:
            raise ValueError(
                f"indptr[-1] ({indptr_end}) must be >= indptr[0] "
                f"({indptr_start})"
            )
        if np.any(indptr_arr[1:] < indptr_arr[:-1]):
            raise ValueError("indptr entries must be monotonically nondecreasing")
        required_entries = indptr_end
        if (
            indices_arr.shape[0] < required_entries
            or values_arr.shape[0] < required_entries
        ):
            raise ValueError(
                "col_indices/data must have at least indptr[-1]="
                f"{required_entries} entries"
            )

        self._core.set_matrix_from_csr(
            row_start_i, row_end_i, indptr_arr, indices_arr, values_arr,
        )
        self._row_start = row_start_i
        self._row_end = row_end_i

    def set_rhs(
        self,
        values: Any,
        row_start: Optional[int] = None,
        row_end: Optional[int] = None,
    ) -> None:
        """Install the right-hand side vector.

        If the matrix has already been registered via
        :meth:`set_matrix_from_csr`, ``row_start`` / ``row_end`` are
        inferred from it; the caller may still override for clarity.
        """
        self._require_open()
        values_arr = _as_real_array(values, "values")

        if row_start is None:
            row_start = self._row_start
        if row_end is None:
            row_end = self._row_end
        if row_start is None or row_end is None:
            raise RuntimeError(
                "RHS row range unknown; call set_matrix_from_csr first or "
                "pass row_start / row_end explicitly"
            )

        row_start_i = _validate_bigint_scalar(row_start, "row_start")
        row_end_i = _validate_bigint_scalar(row_end, "row_end")
        nrows = row_end_i - row_start_i + 1
        if values_arr.shape[0] != nrows:
            raise ValueError(
                f"RHS length {values_arr.shape[0]} does not match "
                f"row range size {nrows}"
            )

        self._core.set_rhs_from_array(row_start_i, row_end_i, values_arr)
        self._rhs_set = True

    # ------------------------------------------------------------------
    # Solve cycle
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Run setup + apply on the configured solver/preconditioner."""
        self._require_open()
        if self._row_start is None:
            raise RuntimeError("solve(): no matrix set; call set_matrix_from_csr")
        if not self._rhs_set:
            raise RuntimeError("solve(): no RHS set; call set_rhs")

        # Build x0 (zeros by default, controlled by the YAML init_guess_mode).
        self._core.setup_initial_guess()

        self._core.solver_create()
        try:
            self._core.solver_setup()
            self._core.solver_apply()
        finally:
            # Destroy the solver immediately so a subsequent solve cycle on the
            # same HypreDrive object starts from a clean slate, even if setup or
            # apply failed. The C handles for the linear system itself are retained
            # for inspection.
            self._core.solver_destroy()

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_solution(self) -> np.ndarray:
        """Return the local-rank solution slab as a NumPy array."""
        self._require_open()
        if self._row_start is None or self._row_end is None:
            raise RuntimeError("get_solution(): solve() has not been run yet")
        nrows = self._row_end - self._row_start + 1
        out = np.empty(nrows, dtype=REAL_DTYPE)
        self._core.copy_solution(out)
        return out

    def solution_norm(self, kind: str = "l2") -> float:
        """Return the norm of the current solution as computed by HYPRE."""
        self._require_open()
        if kind not in {"l1", "l2", "linf"}:
            raise ValueError("kind must be one of 'l1', 'l2', or 'linf'")
        return self._core.solution_norm(kind)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._core is None:
            raise RuntimeError("HypreDrive is closed")


# ---------------------------------------------------------------------------
# One-shot solve helper
# ---------------------------------------------------------------------------

def solve(
    A: Any,
    b: Any,
    options: OptionsLike = None,
    comm: Any = None,
    *,
    row_start: Optional[int] = None,
    row_end: Optional[int] = None,
) -> SolveResult:
    """Configure, solve, and tear down a single linear system.

    This is sugar over the :class:`HypreDrive` lifecycle for the common
    case where the caller does not need to reuse the driver across solves.

    Parameters
    ----------
    A:
        ``scipy.sparse.csr_matrix`` (or anything with a ``.tocsr()``
        method) on a single rank, or any object accepted by
        :meth:`HypreDrive.set_matrix_from_csr`.
    b:
        Local-rank RHS values.
    options:
        Solver/preconditioner configuration. See
        :func:`hypredrive.options.normalize_options` for accepted shapes.
    comm:
        Optional ``mpi4py.MPI.Comm``. Defaults to ``MPI_COMM_SELF``.
    row_start, row_end:
        Inclusive local row range, required when ``A`` is not a SciPy
        matrix.
    """
    if (row_start is None) != (row_end is None):
        raise TypeError("row_start and row_end must be provided together")

    with HypreDrive(options=options, comm=comm) as drv:
        if row_start is None or row_end is None:
            drv.set_matrix_from_csr(A)
        else:
            drv.set_matrix_from_csr(A, row_start=row_start, row_end=row_end)
        drv.set_rhs(b, row_start=row_start, row_end=row_end)
        drv.solve()
        x = drv.get_solution()
        norm = drv.solution_norm("l2")
    return SolveResult(x=x, solution_norm=norm)
