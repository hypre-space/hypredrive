# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython binding to a small subset of the hypredrive C API.

This module exposes the minimum surface required by the high-level
``hypredrive.HypreDrive`` driver. End users should prefer that class; the
``HypreDriveCore`` type and module-level helpers below are explicitly
private (their names start with ``_``).

The binding follows three principles:

* Numpy buffer ownership stays with Python. HYPRE copies once during IJ
  assembly; we never alias caller buffers across the call boundary.
* Errors are translated into ``HypreDriveError`` with the original numeric
  code intact, so user code can pattern-match on specific failure modes.
* MPI communicators arrive as ``mpi4py`` ``Comm`` objects (or ``None`` for
  ``MPI_COMM_SELF``). We call ``MPI_Comm_f2c`` so the binding does not
  compile-time depend on a particular mpi4py version.
"""

from libc.stdint cimport int64_t, uint32_t, intptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cimport cython
import numpy as np
cimport numpy as cnp

from . cimport _native as _c

cnp.import_array()


cdef extern from "mpi.h" nogil:
    _c.MPI_Comm MPI_Comm_f2c(int comm)


# ---------------------------------------------------------------------------
# Module-level capability discovery
# ---------------------------------------------------------------------------

def _hypre_bigint_size():
    """Return ``sizeof(HYPRE_BigInt)`` as observed at native build time."""
    return _c.hypredrive_PythonIndexSize()


def _hypre_real_size():
    """Return ``sizeof(HYPRE_Real)`` as observed at native build time."""
    return _c.hypredrive_PythonRealSize()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

class HypreDriveError(RuntimeError):
    """Exception raised when a hypredrive C call returns a nonzero error code.

    The ``code`` attribute carries the bitfield exactly as returned by the
    library; ``HYPREDRV_ErrorCodeDescribe`` will have already printed a
    human-readable description to stderr by the time this is raised.
    """

    def __init__(self, code, msg):
        super().__init__(f"{msg} (code=0x{int(code):08x})")
        self.code = int(code)


cdef inline void _check(uint32_t code, str what):
    """Raise ``HypreDriveError`` if ``code`` is nonzero.

    The C library already records and prints a structured description; we
    just need to surface a Python-level exception so callers can ``except``.
    """
    if code != 0:
        # Side-effect: prints the description chain + clears it.
        _c.HYPREDRV_ErrorCodeDescribe(code)
        raise HypreDriveError(code, what)


# ---------------------------------------------------------------------------
# Module-wide initialize / finalize
# ---------------------------------------------------------------------------

def _initialize():
    """Call ``HYPREDRV_Initialize``. Idempotent at the C level."""
    _check(_c.HYPREDRV_Initialize(), "HYPREDRV_Initialize")


def _finalize():
    """Call ``HYPREDRV_Finalize``. Idempotent at the C level.

    Errors here are swallowed: by the time we reach finalize the interpreter
    is shutting down and there is no useful action a caller could take.
    """
    cdef uint32_t code = _c.HYPREDRV_Finalize()
    if code != 0:
        # Best effort: still describe to stderr without raising.
        _c.HYPREDRV_ErrorCodeDescribe(code)


# ---------------------------------------------------------------------------
# Communicator resolution
# ---------------------------------------------------------------------------

cdef _c.MPI_Comm _resolve_comm(object comm) except *:
    """Translate an mpi4py ``Comm`` or ``None`` into an ``MPI_Comm`` handle.

    We avoid a compile-time dependency on mpi4py by going through the
    Fortran handle bridge ``Comm.py2f()`` -> ``MPI_Comm_f2c``.
    """
    cdef int fortran_handle
    if comm is None:
        return _c.MPI_COMM_SELF
    if not hasattr(comm, "py2f"):
        raise TypeError(
            "comm must be an mpi4py.MPI.Comm (or None for MPI_COMM_SELF); "
            f"got {type(comm).__name__}"
        )
    fortran_handle = int(comm.py2f())
    return MPI_Comm_f2c(fortran_handle)


# ---------------------------------------------------------------------------
# HypreDriveCore: thin wrapper around HYPREDRV_t
# ---------------------------------------------------------------------------

cdef class HypreDriveCore:
    """Owns one ``HYPREDRV_t`` handle and forwards calls to the C API.

    The high-level ``HypreDrive`` Python class composes one of these and
    layers on input validation, dict-to-YAML conversion, numpy result
    extraction, and context-manager semantics.
    """

    cdef _c.HYPREDRV_t _handle
    cdef bint _alive
    cdef bint _solver_created

    def __cinit__(self, object comm=None, bint library_mode=True):
        cdef _c.MPI_Comm c
        self._handle = NULL
        self._alive = False
        self._solver_created = False

        c = _resolve_comm(comm)
        _check(_c.HYPREDRV_Create(c, &self._handle), "HYPREDRV_Create")
        self._alive = True
        if library_mode:
            _check(_c.HYPREDRV_SetLibraryMode(self._handle),
                   "HYPREDRV_SetLibraryMode")

    def __dealloc__(self):
        self.close()

    cpdef close(self):
        """Tear down the underlying handle. Safe to call multiple times."""
        if self._alive:
            if self._solver_created:
                # Best-effort cleanup; ignore errors at teardown.
                _c.HYPREDRV_LinearSolverDestroy(self._handle)
                self._solver_created = False
            _c.HYPREDRV_Destroy(&self._handle)
            self._handle = NULL
            self._alive = False

    # ------------------------------------------------------------------
    # Configuration: YAML in-memory
    # ------------------------------------------------------------------

    def parse_yaml(self, bytes yaml_text):
        """Configure solver/preconditioner from an in-memory YAML document.

        ``HYPREDRV_InputArgsParse`` accepts a YAML string in ``argv[0]`` (it
        falls back from filename-on-disk to literal-text when the path does
        not resolve), so we go through that same entry point and skip
        round-tripping through a temp file.
        """
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        cdef bytes payload = yaml_text  # keep a reference alive for the duration
        cdef char *argv0 = payload
        cdef char **argv = <char **>malloc(sizeof(char *))
        if argv == NULL:
            raise MemoryError()
        argv[0] = argv0
        try:
            _check(
                _c.HYPREDRV_InputArgsParse(1, argv, self._handle),
                "HYPREDRV_InputArgsParse",
            )
        finally:
            free(argv)

    # ------------------------------------------------------------------
    # Linear-system data ingest
    # ------------------------------------------------------------------

    def set_matrix_from_csr(
        self,
        cnp.npy_int64 row_start,
        cnp.npy_int64 row_end,
        cnp.ndarray indptr,
        cnp.ndarray col_indices,
        cnp.ndarray data,
    ):
        """Pass per-rank CSR slabs into the C layer.

        Caller must guarantee ``indptr``, ``col_indices``, ``data`` are
        C-contiguous and have the dtype matching ``HYPRE_BigInt`` /
        ``HYPRE_Real`` for index and value buffers respectively. The
        Python-level ``HypreDrive.set_matrix_from_csr`` performs that
        normalization before calling here.
        """
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")

        _check(
            _c.hypredrive_PythonSetMatrixFromCSR(
                self._handle,
                <int64_t>row_start,
                <int64_t>row_end,
                <const void *>cnp.PyArray_DATA(indptr),
                <const void *>cnp.PyArray_DATA(col_indices),
                <const void *>cnp.PyArray_DATA(data),
            ),
            "HYPREDRV_LinearSystemSetMatrixFromCSR",
        )

    def set_rhs_from_array(
        self,
        cnp.npy_int64 row_start,
        cnp.npy_int64 row_end,
        cnp.ndarray values,
    ):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")

        _check(
            _c.hypredrive_PythonSetRHSFromArray(
                self._handle,
                <int64_t>row_start,
                <int64_t>row_end,
                <const void *>cnp.PyArray_DATA(values),
            ),
            "HYPREDRV_LinearSystemSetRHSFromArray",
        )

    def set_initial_guess_zero(self):
        """Build the working solution / x0 buffers using the YAML default."""
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        _check(
            _c.HYPREDRV_LinearSystemSetInitialGuess(self._handle, NULL),
            "HYPREDRV_LinearSystemSetInitialGuess",
        )

    # ------------------------------------------------------------------
    # Solve cycle
    # ------------------------------------------------------------------

    def solver_create(self):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverCreate(self._handle),
               "HYPREDRV_LinearSolverCreate")
        self._solver_created = True

    def solver_setup(self):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverSetup(self._handle),
               "HYPREDRV_LinearSolverSetup")

    def solver_apply(self):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverApply(self._handle),
               "HYPREDRV_LinearSolverApply")

    def solver_destroy(self):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        if self._solver_created:
            _check(_c.HYPREDRV_LinearSolverDestroy(self._handle),
                   "HYPREDRV_LinearSolverDestroy")
            self._solver_created = False

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def copy_solution(self, cnp.ndarray out):
        """Copy the local solution slab into ``out`` (a contiguous float64 array).

        We always copy: HYPRE owns the underlying HYPRE_ParVector storage and
        may (e.g. on GPU builds) keep it in device memory; surfacing a
        device pointer to NumPy would be a footgun for v1.
        """
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        cdef const void *src = NULL
        _check(
            _c.hypredrive_PythonGetSolutionValues(self._handle, &src),
            "HYPREDRV_LinearSystemGetSolutionValues",
        )
        if src == NULL:
            raise RuntimeError("hypredrive returned a NULL solution pointer")
        cdef Py_ssize_t n = out.shape[0]
        cdef size_t scalar_size = _c.hypredrive_PythonSolutionEntrySize()
        if scalar_size != <size_t>cnp.PyArray_ITEMSIZE(out):
            raise RuntimeError(
                "complex-valued HYPRE builds are not supported by the "
                "real-valued Python binding"
            )
        memcpy(cnp.PyArray_DATA(out), src,
               <size_t>n * scalar_size)

    def solution_norm(self, str norm_type):
        if not self._alive:
            raise RuntimeError("HypreDriveCore is closed")
        cdef bytes norm_bytes = norm_type.encode("ascii")
        cdef double value = 0.0
        _check(
            _c.HYPREDRV_LinearSystemGetSolutionNorm(self._handle,
                                                    <const char *>norm_bytes,
                                                    &value),
            "HYPREDRV_LinearSystemGetSolutionNorm",
        )
        return float(value)
