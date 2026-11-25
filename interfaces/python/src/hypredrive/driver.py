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

import contextlib
from typing import Any, Optional, Sequence

import numpy as np

from . import _core, session
from ._buffers import validate_csr, validate_rhs
from .options import OptionsLike, normalize_options
from .result import SolveResult

__all__ = ["HypreDrive", "solve", "register_solver_preset",
           "register_precon_preset", "print_lib_info", "print_system_info",
           "print_exit_info", "clear_error", "invalid_value_error", "BIGINT_DTYPE", "REAL_DTYPE"]


# ---------------------------------------------------------------------------
# Platform-derived dtypes
#
# HYPRE_BigInt is either ``int32`` or ``int64`` depending on whether HYPRE
# was built with mixed-int support. The exact dtype is observable at native
# build time and exposed through ``_core._hypre_bigint_size``; we surface
# it as a module-level numpy dtype so user code does not have to guess.
# ---------------------------------------------------------------------------

def _resolve_bigint_dtype() -> np.dtype:
    size = _core._hypre_bigint_size()
    if size == 4:
        return np.dtype(np.int32)
    if size == 8:
        return np.dtype(np.int64)
    raise RuntimeError(
        f"Unsupported HYPRE_BigInt size {size} reported by the native module"
    )


def _resolve_real_dtype() -> np.dtype:
    size = _core._hypre_real_size()
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
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {a.shape}")
    if a.dtype == BIGINT_DTYPE or a.size == 0:
        return np.require(a, dtype=BIGINT_DTYPE, requirements=["C", "A"])
    # numpy's dtype conversion would silently truncate floats and wrap integers
    # that do not fit (e.g. int64 indices on a 32-bit HYPRE_BigInt build).
    if a.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers, got dtype {a.dtype}")
    info = np.iinfo(BIGINT_DTYPE)
    lo, hi = int(a.min()), int(a.max())
    if lo < int(info.min) or hi > int(info.max):
        raise ValueError(
            f"{name} has values in [{lo}, {hi}], outside the HYPRE_BigInt range "
            f"[{int(info.min)}, {int(info.max)}]"
        )
    return np.require(a, dtype=BIGINT_DTYPE, requirements=["C", "A"])


def _as_real_array(arr: Any, name: str) -> np.ndarray:
    a = np.require(arr, dtype=REAL_DTYPE, requirements=["C", "A"])
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
        input_args: Optional[Sequence[str]] = None,
    ) -> None:
        session.initialize()
        self._core: Optional[_core.HypreDriveCore] = _core.HypreDriveCore(
            comm=comm,
            library_mode=True,
        )
        self._row_start: Optional[int] = None
        self._row_end: Optional[int] = None
        self._rhs_set: bool = False
        self._native_system: bool = False
        # Native LinearSystemBuild installs the configured guess (e.g. the
        # previous solution); the next solve() must not rebuild it.
        self._guess_from_build: bool = False
        self._last_iterations: Optional[int] = None
        self._last_converged: Optional[bool] = None
        self._last_final_res_norm: Optional[float] = None
        self._last_setup_time: Optional[float] = None
        self._last_solve_time: Optional[float] = None

        # Always parse a YAML configuration up front so the C side has a
        # solver/preconditioner method selected before we hand over data.
        # ``input_args`` supplies CLI-style overrides applied on top of the
        # options, e.g. ["--solver:pcg:max_iter", "100"] (a leading "-a"
        # token is tolerated).
        yaml_text = normalize_options(options)
        extra = list(input_args) if input_args else None
        self._core.parse_yaml(yaml_text.encode("utf-8"), extra)

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
                if (row_start is None) != (row_end is None):
                    raise TypeError(
                        "row_start and row_end must be provided together "
                        "when passing a SciPy sparse matrix"
                    )
                if getattr(matrix_or_indptr, "format", None) == "csr":
                    csr = matrix_or_indptr
                else:
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

        # Both entry points use the same shape/offset contract. The native C
        # API cannot check Python buffer lengths, so the core checks it again
        # for callers that bypass this high-level driver.
        validate_csr(row_start_i, row_end_i, indptr_arr, indices_arr, values_arr)

        # "matrix" marks the start of a new linear system in the stats
        # context (HYPREDRV_AnnotateBegin docs); without it every solve on
        # a reused driver collapses into the first statistics entry.
        with self._annotated("matrix"):
            self._core.set_matrix_from_csr(
                row_start_i, row_end_i, indptr_arr, indices_arr, values_arr,
            )
        self._row_start = row_start_i
        self._row_end = row_end_i
        self._guess_from_build = False

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
        validate_rhs(row_start_i, row_end_i, values_arr)

        with self._annotated("rhs"):
            self._core.set_rhs_from_array(row_start_i, row_end_i, values_arr)
        self._rhs_set = True
        self._guess_from_build = False

    def set_dofmap(self, labels: Any) -> None:
        """Install per-row DOF labels for MGR.

        ``labels`` is a 1-D sequence (NumPy array, list, etc.) of small
        non-negative integer codes — one per local row — telling MGR which
        block each row belongs to. The length must match the local row
        range previously registered by :meth:`set_matrix_from_csr`.

        The dofmap is required when the configured preconditioner is MGR
        and is otherwise ignored by the C layer. Calling this method
        replaces any previously installed dofmap.
        """
        self._require_open()
        if self._row_start is None or self._row_end is None:
            raise RuntimeError(
                "set_dofmap(): no matrix set; call set_matrix_from_csr first"
            )
        # Range-check before the dtype cast: np.ascontiguousarray to intc
        # silently wraps int64 values outside the int32 range and treats
        # negative sentinels as valid labels, both of which produce
        # nonsensical MGR partitions far away from this entry point.
        labels_in = np.asarray(labels)
        if labels_in.size:
            labels_range = labels_in
            if not (
                np.issubdtype(labels_in.dtype, np.integer)
                or np.issubdtype(labels_in.dtype, np.bool_)
            ):
                try:
                    labels_float = np.asarray(labels_in, dtype=np.float64)
                except (TypeError, ValueError):
                    raise ValueError("labels must be integer codes") from None
                if not np.isfinite(labels_float).all():
                    raise ValueError("labels must be finite integer codes")
                if not np.equal(labels_float, np.trunc(labels_float)).all():
                    raise ValueError("labels must be integer codes")
                labels_range = labels_float
            info = np.iinfo(np.intc)
            lo = int(labels_range.min())
            hi = int(labels_range.max())
            if lo < 0:
                raise ValueError(
                    f"labels must be non-negative; got minimum {lo}"
                )
            if hi > int(info.max):
                raise ValueError(
                    f"labels max ({hi}) exceeds C int range [0, {int(info.max)}]"
                )
        labels_arr = np.require(labels_in, dtype=np.intc, requirements=["C", "A"])
        if labels_arr.ndim != 1:
            raise ValueError(
                f"labels must be a 1-D array, got shape {labels_arr.shape}"
            )
        nrows = self._row_end - self._row_start + 1
        if labels_arr.shape[0] != nrows:
            raise ValueError(
                f"labels length {labels_arr.shape[0]} does not match local "
                f"row count {nrows}"
            )
        with self._annotated("dofmap"):
            self._core.set_dofmap(labels_arr)

    # ------------------------------------------------------------------
    # Stats / Caliper annotations
    # ------------------------------------------------------------------

    def annotate_begin(self, name: str, id: int = -1) -> None:
        """Begin a stats/Caliper annotation region.

        Reserved names ("system"/"matrix", "rhs", "dofmap", "prec",
        "solve", "reset_x0", ...) feed the statistics context directly;
        see the ``HYPREDRV_AnnotateBegin`` C documentation. The driver
        already emits "matrix"/"rhs"/"dofmap"/"reset_x0" from its data
        ingest and solve methods, so manual annotation is only needed for
        custom regions (e.g. Caliper-only "Run-*" spans).
        """
        self._require_open()
        self._core.annotate_begin(name.encode("utf-8"), id)

    def annotate_end(self, name: str, id: int = -1) -> None:
        """End a stats/Caliper annotation region (pairs annotate_begin)."""
        self._require_open()
        self._core.annotate_end(name.encode("utf-8"), id)

    @contextlib.contextmanager
    def _annotated(self, name: str):
        encoded = name.encode("utf-8")
        self._core.annotate_begin(encoded, -1)
        try:
            yield
        finally:
            self._core.annotate_end(encoded, -1)

    # ------------------------------------------------------------------
    # Solve cycle
    # ------------------------------------------------------------------

    def set_relative_tol(self, tol: float) -> None:
        """Set the Krylov relative tolerance used by subsequent :meth:`solve` calls.

        Overrides the tolerance from the options. ``tol`` must be in ``[0, 1)``.
        """
        self._require_open()
        self._core.solver_set_relative_tol(float(tol))

    def solve(self) -> None:
        """Run setup + apply on the configured solver/preconditioner."""
        self._require_open()
        if self._row_start is None and not self._native_system:
            raise RuntimeError("solve(): no matrix set; call set_matrix_from_csr")
        if not self._rhs_set and not self._native_system:
            raise RuntimeError("solve(): no RHS set; call set_rhs")

        # Reset before we start so the last_* properties never report stale
        # values from a previous successful solve when this one fails partway.
        self._last_iterations = None
        self._last_converged = None
        self._last_final_res_norm = None
        self._last_setup_time = None
        self._last_solve_time = None

        # Build x0 (controlled by the YAML init_guess_mode), then copy it into
        # the working solution — without the reset the solve always starts
        # from the zeroed working vector regardless of mode. ResetInitialGuess
        # annotates the "reset_x0" stats region internally (wrapping it
        # manually would count the repetition twice).
        if self._guess_from_build:
            self._guess_from_build = False
        else:
            self._core.setup_initial_guess()
        self._core.reset_initial_guess()

        self._core.solver_create()
        try:
            self._core.solver_setup()
            self._core.solver_apply()
            self._last_iterations = self._core.solver_iterations()
            # Convergence info is read from the live solver object, so it must
            # be captured before solver_destroy runs in the finally block.
            self._last_converged = self._core.solver_converged()
            self._last_final_res_norm = self._core.solver_final_res_norm()
            self._last_setup_time = self._core.solver_setup_time()
            self._last_solve_time = self._core.solver_solve_time()
        finally:
            # Destroy the solver immediately so a subsequent solve cycle on the
            # same HypreDrive object starts from a clean slate, even if setup or
            # apply failed. The C handles for the linear system itself are retained
            # for inspection.
            self._core.solver_destroy()

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    @property
    def last_iterations(self) -> Optional[int]:
        """Number of iterations taken by the most recent :meth:`solve` call.

        ``None`` until a successful solve. Reset to ``None`` at the start of
        every :meth:`solve`, so a failed solve always reports ``None`` here
        rather than the previous solve's count.
        """
        return self._last_iterations

    @property
    def last_converged(self) -> Optional[bool]:
        """Whether the most recent :meth:`solve` reached its tolerance.

        ``False`` means the solver stopped without converging (e.g., it hit
        the maximum number of iterations). ``None`` until a successful solve;
        reset to ``None`` at the start of every :meth:`solve`.
        """
        return self._last_converged

    @property
    def last_final_res_norm(self) -> Optional[float]:
        """Final relative residual norm from the most recent :meth:`solve`.

        ``None`` until a successful solve; reset to ``None`` at the start of
        every :meth:`solve`.
        """
        return self._last_final_res_norm

    @property
    def last_setup_time(self) -> Optional[float]:
        """Solver/preconditioner setup time from the most recent :meth:`solve`.

        Unit matches whatever the C library was configured to report:
        seconds by default, milliseconds when the YAML option
        ``general.use_millisec`` is set to ``yes``. Callers that mix the two
        modes must inspect their own YAML to label printouts correctly.
        ``None`` until the first successful solve.
        """
        return self._last_setup_time

    @property
    def last_solve_time(self) -> Optional[float]:
        """Solver-apply (Krylov iteration) time from the most recent :meth:`solve`.

        Returned in the same units as :attr:`last_setup_time` — see that
        property for the unit/YAML coupling. ``None`` until the first
        successful solve.
        """
        return self._last_solve_time

    def get_solution(self) -> np.ndarray:
        """Return the local-rank solution slab as a NumPy array."""
        self._require_open()
        if self._row_start is None or self._row_end is None:
            if self._native_system:
                return self._core.get_solution_values()
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
    # Object naming and parsed-input queries
    # ------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        """Name this object (``HYPREDRV_ObjectSetName``) for stats/log output."""
        self._require_open()
        self._core.object_set_name(str(name))

    @property
    def warmup(self) -> int:
        """Number of warmup runs from the parsed input."""
        self._require_open()
        return self._core.get_warmup()

    @property
    def num_repetitions(self) -> int:
        """Number of repetitions from the parsed input."""
        self._require_open()
        return self._core.get_num_repetitions()

    @property
    def num_linear_systems(self) -> int:
        """Number of linear systems described by the parsed input."""
        self._require_open()
        return self._core.get_num_linear_systems()

    @property
    def num_precon_variants(self) -> int:
        """Number of preconditioner variants in the parsed input."""
        self._require_open()
        return self._core.get_num_precon_variants()

    def set_precon_variant(self, variant_idx: int) -> None:
        """Select the active preconditioner variant (0-based)."""
        self._require_open()
        self._core.set_precon_variant(int(variant_idx))

    def set_precon_preset(self, preset: str) -> None:
        """Select a registered/built-in preconditioner preset by name."""
        self._require_open()
        self._core.set_precon_preset(str(preset))

    def set_solver_preset(self, preset: str) -> None:
        """Select a registered/built-in solver preset by name."""
        self._require_open()
        self._core.set_solver_preset(str(preset))

    # ------------------------------------------------------------------
    # Linear system: file-based build/read/print
    # ------------------------------------------------------------------

    def build_linear_system(self) -> None:
        """Build matrix/RHS/solution from the parsed ``linear_system`` input."""
        self._require_open()
        self._core.linear_system_build()
        self._adopt_native_matrix()
        self._guess_from_build = True

    def read_matrix(self) -> None:
        """Read the matrix from files named in the parsed input."""
        self._require_open()
        self._core.read_matrix()
        self._adopt_native_matrix()
        self._guess_from_build = False

    def read_dofmap(self) -> None:
        """Read the DOF map from the file named in the parsed input."""
        self._require_open()
        self._core.read_dofmap()

    def print_dofmap(self, filename: str) -> None:
        """Write the DOF map to ``filename`` (``filename.NNNNN`` on MPI)."""
        self._require_open()
        self._core.print_dofmap(str(filename))

    def print_linear_system(self) -> None:
        """Dump matrix, RHS and DOF map to ASCII files."""
        self._require_open()
        self._core.print_linear_system()

    def compute_eigenspectrum(self) -> None:
        """Compute the eigenspectrum of the system (``HYPREDRV_LinearSystemComputeEigenspectrum``)."""
        self._require_open()
        self._core.compute_eigenspectrum()

    def _adopt_native_matrix(self) -> None:
        # The matrix was replaced natively: refresh the cached local row range
        # so get_solution(), set_rhs() and set_dofmap() size against it, and
        # leave solve() preconditions to the C layer.
        self._row_start, self._row_end = self._core.matrix_row_range()
        self._native_system = True

    # ------------------------------------------------------------------
    # DOF maps by block structure, null spaces
    # ------------------------------------------------------------------

    def set_interleaved_dofmap(self, num_local_blocks: int, num_dof_types: int) -> None:
        """Set a dofmap ``[0..t-1, 0..t-1, ...]`` (types interleaved per block)."""
        self._require_open()
        self._core.set_interleaved_dofmap(int(num_local_blocks), int(num_dof_types))

    def set_contiguous_dofmap(self, num_local_blocks: int, num_dof_types: int) -> None:
        """Set a dofmap with each DOF type stored contiguously."""
        self._require_open()
        self._core.set_contiguous_dofmap(int(num_local_blocks), int(num_dof_types))

    def _set_space(self, which: str, values: Any, num_components: Optional[int]) -> None:
        arr = np.asarray(values)
        if arr.ndim == 2:
            # (num_components, num_entries): modes stored back-to-back.
            num_components = arr.shape[0] if num_components is None else num_components
            num_entries = arr.shape[1]
            arr = arr.reshape(-1)
        else:
            if num_components is None:
                raise TypeError("num_components is required for 1-D values")
            if int(num_components) <= 0 or arr.size % int(num_components):
                raise ValueError("values length must be a multiple of num_components")
            num_entries = arr.size // int(num_components)
        arr = _as_real_array(arr, "values")
        getattr(self._core, which)(int(num_entries), int(num_components), arr)

    def set_near_null_space(self, values: Any, num_components: Optional[int] = None) -> None:
        """Provide near-null-space modes (e.g. rigid-body modes).

        ``values`` is either a 2-D ``(num_components, num_entries)`` array or a
        1-D array of ``num_components`` modes stored back-to-back. Data is
        copied; the matrix must be set first.
        """
        self._require_open()
        self._set_space("set_near_null_space", values, num_components)

    def set_null_space(self, values: Any, num_components: Optional[int] = None) -> None:
        """Provide exact null-space modes (same layout as :meth:`set_near_null_space`)."""
        self._require_open()
        self._set_space("set_null_space", values, num_components)

    # ------------------------------------------------------------------
    # Raw HYPRE handles
    #
    # HYPRE_Matrix / HYPRE_Vector arguments are opaque pointers passed as
    # Python ints (the address of a HYPRE_IJMatrix / HYPRE_IJVector, e.g.
    # ``ctypes.c_void_p.value``); ``None`` means NULL. Lifetime follows the
    # HYPREDRV.h docs: the driver always runs in library mode, so
    # externally supplied handles are *borrowed* and must outlive their use
    # (destroy them after closing the driver). Handles returned by the
    # ``*_handle`` getters are borrowed from the driver: never free them,
    # and do not use them after the driver is closed.
    # ------------------------------------------------------------------

    def set_matrix_handle(self, mat: Any) -> None:
        """Attach an assembled ``HYPRE_IJMatrix`` address as the system matrix."""
        self._require_open()
        # Native SetMatrix does not annotate; "matrix" starts a new stats entry.
        with self._annotated("matrix"):
            attached = self._core.set_matrix_handle(mat)
        self._guess_from_build = False
        if attached:
            self._adopt_native_matrix()
        else:  # cleared: no matrix, so no row range
            self._row_start = self._row_end = None
            self._native_system = False

    def set_prec_matrix_handle(self, mat: Any) -> None:
        """Attach a separate preconditioning matrix (``None`` clears it)."""
        self._require_open()
        self._core.set_prec_matrix_handle(mat)

    def set_discrete_gradient_handle(self, G: Any) -> None:
        """Attach the discrete gradient matrix for AMS/ADS."""
        self._require_open()
        self._core.set_discrete_gradient_handle(G)

    def set_discrete_curl_handle(self, C: Any) -> None:
        """Attach the discrete curl matrix for ADS."""
        self._require_open()
        self._core.set_discrete_curl_handle(C)

    def set_coarse_schur_handle(self, level: int, mat: Any) -> None:
        """Attach a user coarse Schur operator for an MGR level (``None`` clears)."""
        self._require_open()
        self._core.set_coarse_schur_handle(int(level), mat)

    def set_coordinates_handle(self, x: Any, y: Any, z: Any) -> None:
        """Attach vertex coordinate vectors (``HYPRE_IJVector`` addresses) for AMS/ADS."""
        self._require_open()
        self._core.set_coordinates_handle(x, y, z)

    def set_rhs_handle(self, vec: Any) -> None:
        """Attach a ``HYPRE_IJVector`` address as the RHS."""
        self._require_open()
        self._core.set_rhs_handle(vec)
        self._rhs_set = True
        self._guess_from_build = False

    def set_solution_handle(self, vec: Any) -> None:
        """Attach a ``HYPRE_IJVector`` address as the solution vector."""
        self._require_open()
        self._core.set_solution_handle(vec)

    def set_reference_solution_handle(self, vec: Any) -> None:
        """Attach a reference solution vector (``None`` clears)."""
        self._require_open()
        self._core.set_reference_solution_handle(vec)

    def get_matrix_handle(self) -> int:
        """Return the borrowed system-matrix handle as an integer address."""
        self._require_open()
        return self._core.get_matrix_handle()

    def get_rhs_handle(self) -> int:
        """Return the borrowed RHS handle as an integer address."""
        self._require_open()
        return self._core.get_rhs_handle()

    def get_solution_handle(self) -> int:
        """Return the borrowed solution handle as an integer address."""
        self._require_open()
        return self._core.get_solution_handle()

    def get_solution_values(self) -> np.ndarray:
        """Return a copy of the native local solution (host memory only)."""
        self._require_open()
        return self._core.get_solution_values()

    def get_rhs_values(self) -> np.ndarray:
        """Return a copy of the native local RHS (host HYPRE builds only)."""
        self._require_open()
        return self._core.get_rhs_values()

    # ------------------------------------------------------------------
    # State vectors (time stepping / Newton)
    # ------------------------------------------------------------------

    def set_state_vectors(self, handles: Sequence[Any]) -> None:
        """Register ``HYPRE_IJVector`` addresses as state vectors.

        All state vectors must share one local length. The caller keeps
        ownership; destroy them after closing the driver.
        """
        self._require_open()
        self._core.state_vector_set(list(handles))

    def state_vector_values(self, index: int) -> np.ndarray:
        """Return a copy of the local data of state vector ``index``."""
        self._require_open()
        return self._core.state_vector_get_values(int(index))

    def state_vector_copy(self, index_in: int, index_out: int) -> None:
        """Copy state vector ``index_in`` into ``index_out``."""
        self._require_open()
        self._core.state_vector_copy(int(index_in), int(index_out))

    def state_vector_update_all(self) -> None:
        """Cycle the logical state indices (new current state at index 0)."""
        self._require_open()
        self._core.state_vector_update_all()

    def state_vector_apply_correction(self, state_idx: int = 0) -> None:
        """Add the last linear-solver solution to state vector ``state_idx``."""
        self._require_open()
        self._core.state_vector_apply_correction(int(state_idx))

    # ------------------------------------------------------------------
    # Standalone preconditioner
    # ------------------------------------------------------------------

    def precon_create(self) -> None:
        """Create the preconditioner (destroyed automatically on close)."""
        self._require_open()
        self._core.precon_create()

    def precon_setup(self) -> None:
        """Set up the standalone preconditioner."""
        self._require_open()
        self._core.precon_setup()

    def precon_apply(self, vec_b: Any, vec_x: Any) -> None:
        """Apply the preconditioner: ``vec_x = M^{-1} vec_b`` (HYPRE_Vector addresses)."""
        self._require_open()
        self._core.precon_apply(vec_b, vec_x)

    def precon_destroy(self) -> None:
        """Destroy the standalone preconditioner."""
        self._require_open()
        self._core.precon_destroy()

    # ------------------------------------------------------------------
    # Statistics by level
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Print the statistics summary."""
        self._require_open()
        self._core.stats_print()

    def annotate_level_begin(self, level: int, name: str, id: int = -1) -> None:
        """Begin a leveled annotation (feeds :meth:`stats_level_entry`)."""
        self._require_open()
        self._core.annotate_level_begin(int(level), name.encode("utf-8"), int(id))

    def annotate_level_end(self, level: int, name: str, id: int = -1) -> None:
        """End a leveled annotation."""
        self._require_open()
        self._core.annotate_level_end(int(level), name.encode("utf-8"), int(id))

    def stats_level_count(self, level: int) -> int:
        """Number of entries recorded at ``level``."""
        self._require_open()
        return self._core.stats_level_count(int(level))

    def stats_level_entry(self, level: int, index: int) -> dict:
        """Return ``{entry_id, num_solves, linear_iters, setup_time, solve_time}``."""
        self._require_open()
        eid, ns, it, st, sv = self._core.stats_level_entry(int(level), int(index))
        return {
            "entry_id": eid,
            "num_solves": ns,
            "linear_iters": it,
            "setup_time": st,
            "solve_time": sv,
        }

    def print_stats_level(self, level: int) -> None:
        """Print the statistics table for ``level``."""
        self._require_open()
        self._core.stats_level_print(int(level))

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
    input_args: Optional[Sequence[str]] = None,
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
    input_args:
        CLI-style override tokens applied on top of ``options``, e.g.
        ``["--solver:pcg:max_iter", "100"]``.
    """
    if (row_start is None) != (row_end is None):
        raise TypeError("row_start and row_end must be provided together")

    with HypreDrive(options=options, comm=comm, input_args=input_args) as drv:
        if row_start is None or row_end is None:
            drv.set_matrix_from_csr(A)
        else:
            drv.set_matrix_from_csr(A, row_start=row_start, row_end=row_end)
        drv.set_rhs(b, row_start=row_start, row_end=row_end)
        drv.solve()
        return SolveResult(
            x=drv.get_solution(),
            solution_norm=drv.solution_norm("l2"),
            iterations=drv.last_iterations,
            converged=drv.last_converged,
            final_res_norm=drv.last_final_res_norm,
            setup_time=drv.last_setup_time,
            solve_time=drv.last_solve_time,
        )


# ---------------------------------------------------------------------------
# Module-level (driver-independent) entry points
# ---------------------------------------------------------------------------

def register_solver_preset(name: str, yaml_text: str, help: str = "") -> None:
    """Register a named solver preset (``HYPREDRV_SolverPresetRegister``)."""
    session.initialize()
    _core._solver_preset_register(name, yaml_text, help)


def register_precon_preset(name: str, yaml_text: str, help: str = "") -> None:
    """Register a named preconditioner preset (``HYPREDRV_PreconPresetRegister``)."""
    session.initialize()
    _core._precon_preset_register(name, yaml_text, help)


def print_lib_info(comm: Any = None, print_datetime: bool = False) -> None:
    """Print library version/build info (``HYPREDRV_PrintLibInfo``)."""
    session.initialize()
    _core._print_lib_info(comm, bool(print_datetime))


def print_system_info(comm: Any = None) -> None:
    """Print host/system info (``HYPREDRV_PrintSystemInfo``)."""
    session.initialize()
    _core._print_system_info(comm)


def print_exit_info(comm: Any = None, argv0: str = "hypredrive") -> None:
    """Print the exit banner (``HYPREDRV_PrintExitInfo``)."""
    session.initialize()
    _core._print_exit_info(comm, argv0)


def clear_error() -> None:
    """Clear the recorded HYPREDRV error chain (``HYPREDRV_ErrorCodeClear``)."""
    _core._error_code_clear()


def invalid_value_error(message: str) -> int:
    """Record an invalid-value error and return its code (``HYPREDRV_ErrorInvalidValue``)."""
    return _core._error_invalid_value(message)
