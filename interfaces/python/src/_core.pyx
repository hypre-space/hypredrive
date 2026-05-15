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
from libc.string cimport memcpy

cimport cython
import numpy as np
cimport numpy as cnp
from hypredrive.errors import HypreDriveError

# The source package is intentionally flattened under interfaces/python/src,
# so _core.pxd is a sibling file at Cython time rather than under a physical
# hypredrive/ package directory.
cimport _core as _c

cnp.import_array()

cdef size_t _HYPREDRIVE_PYTHON_REAL_SIZE = _c.HYPREDRV_PythonRealSize()
cdef size_t _HYPREDRIVE_PYTHON_SOLUTION_ENTRY_SIZE = (
    _c.HYPREDRV_PythonSolutionEntrySize()
)
if _HYPREDRIVE_PYTHON_SOLUTION_ENTRY_SIZE != _HYPREDRIVE_PYTHON_REAL_SIZE:
    raise ImportError(
        "complex-valued HYPRE builds are not supported by the real-valued "
        "hypredrive Python binding"
    )


# ---------------------------------------------------------------------------
# Module-level capability discovery
# ---------------------------------------------------------------------------

def _hypre_bigint_size():
    """Return ``sizeof(HYPRE_BigInt)`` as observed at native build time."""
    return _c.HYPREDRV_PythonIndexSize()


def _hypre_real_size():
    """Return ``sizeof(HYPRE_Real)`` as observed at native build time."""
    return _c.HYPREDRV_PythonRealSize()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

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
    _check(_c.HYPREDRV_PythonMPIInitialize(), "HYPREDRV_PythonMPIInitialize")
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
    code = _c.HYPREDRV_PythonMPIFinalize()
    if code != 0:
        _c.HYPREDRV_ErrorCodeDescribe(code)


# ---------------------------------------------------------------------------
# Communicator resolution
# ---------------------------------------------------------------------------

cdef uint32_t _create_driver(object comm, _c.HYPREDRV_t *handle) except? 1:
    """Create a driver from an mpi4py ``Comm`` or ``None``.

    We avoid a compile-time dependency on mpi4py by going through the
    Fortran handle bridge ``Comm.py2f()``; the C shim converts it to the
    ABI-specific ``MPI_Comm`` representation.
    """
    cdef int fortran_handle
    if comm is None:
        return _c.HYPREDRV_PythonCreateWithSelf(handle)
    if not hasattr(comm, "py2f"):
        raise TypeError(
            "comm must be an mpi4py.MPI.Comm (or None for MPI_COMM_SELF); "
            f"got {type(comm).__name__}"
        )
    fortran_handle = int(comm.py2f())
    return _c.HYPREDRV_PythonCreateFromFortranComm(fortran_handle, handle)


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
    cdef bint _solver_created

    def __cinit__(self, object comm=None, bint library_mode=True):
        self._handle = NULL
        self._solver_created = False

        _check(_create_driver(comm, &self._handle), "HYPREDRV_Create")
        if library_mode:
            _check(_c.HYPREDRV_SetLibraryMode(self._handle),
                   "HYPREDRV_SetLibraryMode")

    def __dealloc__(self):
        if self._handle != NULL:
            if self._solver_created:
                _c.HYPREDRV_LinearSolverDestroy(self._handle)
                self._solver_created = False
            _c.HYPREDRV_Destroy(&self._handle)
            self._handle = NULL

    cpdef close(self):
        """Tear down the underlying handle. Safe to call multiple times."""
        cdef uint32_t first_code = 0
        cdef uint32_t code = 0
        if self._handle != NULL:
            if self._solver_created:
                code = _c.HYPREDRV_LinearSolverDestroy(self._handle)
                self._solver_created = False
                if first_code == 0:
                    first_code = code
            code = _c.HYPREDRV_Destroy(&self._handle)
            self._handle = NULL
            if first_code == 0:
                first_code = code
            _check(first_code, "HypreDriveCore.close")

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
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        cdef bytes payload = yaml_text  # keep a reference alive for the duration
        cdef const char *payload_data = payload
        cdef char *argv[2]
        argv[0] = <char *>payload_data
        argv[1] = NULL
        _check(
            _c.HYPREDRV_InputArgsParse(1, argv, self._handle),
            "HYPREDRV_InputArgsParse",
        )

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
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        # These checks intentionally duplicate the high-level driver:
        # HypreDriveCore is the trust boundary for direct _core callers.
        if not cnp.PyArray_IS_C_CONTIGUOUS(indptr):
            raise ValueError("indptr must be C-contiguous")
        if not cnp.PyArray_IS_C_CONTIGUOUS(col_indices):
            raise ValueError("col_indices must be C-contiguous")
        if not cnp.PyArray_IS_C_CONTIGUOUS(data):
            raise ValueError("data must be C-contiguous")
        if <size_t>cnp.PyArray_ITEMSIZE(indptr) != _c.HYPREDRV_PythonIndexSize():
            raise ValueError("indptr dtype does not match HYPRE_BigInt")
        if <size_t>cnp.PyArray_ITEMSIZE(col_indices) != _c.HYPREDRV_PythonIndexSize():
            raise ValueError("col_indices dtype does not match HYPRE_BigInt")
        if <size_t>cnp.PyArray_ITEMSIZE(data) != _HYPREDRIVE_PYTHON_REAL_SIZE:
            raise ValueError("data dtype does not match HYPRE_Real")

        _check(
            _c.HYPREDRV_PythonSetMatrixFromCSR(
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
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        if not cnp.PyArray_IS_C_CONTIGUOUS(values):
            raise ValueError("values must be C-contiguous")
        if <size_t>cnp.PyArray_ITEMSIZE(values) != _HYPREDRIVE_PYTHON_REAL_SIZE:
            raise ValueError("values dtype does not match HYPRE_Real")

        _check(
            _c.HYPREDRV_PythonSetRHSFromArray(
                self._handle,
                <int64_t>row_start,
                <int64_t>row_end,
                <const void *>cnp.PyArray_DATA(values),
            ),
            "HYPREDRV_LinearSystemSetRHSFromArray",
        )

    def setup_initial_guess(self):
        """Set up x0 according to ``linear_system.init_guess_mode``.

        Passing NULL delegates the actual initial-guess policy to the C layer.
        """
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        _check(
            _c.HYPREDRV_LinearSystemSetInitialGuess(self._handle, NULL),
            "HYPREDRV_LinearSystemSetInitialGuess",
        )

    # ------------------------------------------------------------------
    # Solve cycle
    # ------------------------------------------------------------------

    def solver_create(self):
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverCreate(self._handle),
               "HYPREDRV_LinearSolverCreate")
        self._solver_created = True

    def solver_setup(self):
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverSetup(self._handle),
               "HYPREDRV_LinearSolverSetup")

    def solver_apply(self):
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        _check(_c.HYPREDRV_LinearSolverApply(self._handle),
               "HYPREDRV_LinearSolverApply")

    def solver_iterations(self):
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        cdef int iters = 0
        _check(_c.HYPREDRV_LinearSolverGetNumIter(self._handle, &iters),
               "HYPREDRV_LinearSolverGetNumIter")
        return int(iters)

    def solver_destroy(self):
        if self._handle == NULL:
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
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        cdef const void *src = NULL
        cdef size_t src_length = 0
        _check(
            _c.HYPREDRV_PythonGetSolutionValues(self._handle, &src, &src_length),
            "HYPREDRV_LinearSystemGetSolutionValues",
        )
        if src == NULL:
            raise RuntimeError("hypredrive returned a NULL solution pointer")
        cdef Py_ssize_t n = out.shape[0]
        if n < 0:
            raise ValueError("output length must be nonnegative")
        if <size_t>n > src_length:
            raise ValueError(
                f"output length {n} exceeds local solution length {src_length}"
            )
        cdef size_t scalar_size = _HYPREDRIVE_PYTHON_SOLUTION_ENTRY_SIZE
        memcpy(cnp.PyArray_DATA(out), src,
               <size_t>n * scalar_size)

    def solution_norm(self, str norm_type):
        if self._handle == NULL:
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
