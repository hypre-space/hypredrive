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

from libc.stdint cimport int64_t, uint32_t, intptr_t, uintptr_t
from libc.stdlib cimport malloc, free

cimport cython
import numpy as np
cimport numpy as cnp
import weakref

from hypredrive.errors import HypreDriveError
from hypredrive._buffers import validate_csr, validate_native_array, validate_rhs

from hypredrive cimport _chypredrv as _c

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

# Handles that have not been closed yet; _finalize() releases them before
# tearing down the runtime.
_live_cores = weakref.WeakSet()


def _initialize():
    """Call ``HYPREDRV_Initialize``. Idempotent at the C level."""
    _check(_c.HYPREDRV_PythonMPIInitialize(), "HYPREDRV_PythonMPIInitialize")
    _check(_c.HYPREDRV_Initialize(), "HYPREDRV_Initialize")


def _finalize():
    """Call ``HYPREDRV_Finalize``. Idempotent at the C level.

    Any ``HypreDriveCore`` still alive is closed first: destroying a handle
    after ``HYPREDRV_Finalize``/``MPI_Finalize`` (e.g. a module-level driver
    collected during interpreter teardown) is undefined behavior.

    Errors here are swallowed: by the time we reach finalize the interpreter
    is shutting down and there is no useful action a caller could take.
    """
    cdef HypreDriveCore core
    for core in list(_live_cores):
        _live_cores.discard(core)
        core._release()
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
# Opaque-handle helpers and global (driver-independent) entry points
# ---------------------------------------------------------------------------

cdef void *_as_ptr(object handle, str name) except? NULL:
    """Convert ``None`` or an integer address into an opaque ``void *``.

    HYPRE_Matrix / HYPRE_Vector arguments are raw HYPRE handles (e.g. the
    address of a ``HYPRE_IJMatrix``/``HYPRE_IJVector``). Python callers pass
    them as integers (``ctypes.c_void_p.value``, a C extension's pointer,
    ...); ``None`` maps to ``NULL``. The caller owns the lifetime rules
    documented for the corresponding HYPREDRV_ function.
    """
    cdef uintptr_t addr
    if handle is None:
        return NULL
    if hasattr(handle, "value") and not isinstance(handle, int):
        handle = handle.value  # ctypes.c_void_p and friends
        if handle is None:
            return NULL
    try:
        addr = <uintptr_t>int(handle)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(
            f"{name} must be None or an integer pointer address; "
            f"got {type(handle).__name__}"
        ) from None
    return <void *>addr


cdef int _fortran_comm(object comm) except -2:
    """Fortran handle of ``comm``; -1 selects ``MPI_COMM_SELF``."""
    cdef int fortran_handle = -1
    if comm is not None:
        if not hasattr(comm, "py2f"):
            raise TypeError(
                "comm must be an mpi4py.MPI.Comm (or None for MPI_COMM_SELF); "
                f"got {type(comm).__name__}"
            )
        fortran_handle = int(comm.py2f())
        if fortran_handle < 0:
            raise ValueError("invalid communicator handle")
    return fortran_handle


def _error_code_clear():
    """Call ``HYPREDRV_ErrorCodeClear``."""
    _c.HYPREDRV_ErrorCodeClear()


def _error_invalid_value(str message):
    """Call ``HYPREDRV_ErrorInvalidValue``; return the resulting error code."""
    cdef bytes raw = message.encode("utf-8")
    return int(_c.HYPREDRV_ErrorInvalidValue(<const char *>raw))


def _print_lib_info(object comm=None, bint print_datetime=False):
    cdef int c = _fortran_comm(comm)
    _check(_c.HYPREDRV_PythonPrintLibInfo(c, 1 if print_datetime else 0),
           "HYPREDRV_PrintLibInfo")


def _print_system_info(object comm=None):
    cdef int c = _fortran_comm(comm)
    _check(_c.HYPREDRV_PythonPrintSystemInfo(c), "HYPREDRV_PrintSystemInfo")


def _print_exit_info(object comm=None, str argv0="hypredrive"):
    cdef int c = _fortran_comm(comm)
    cdef bytes raw = argv0.encode("utf-8")
    _check(_c.HYPREDRV_PythonPrintExitInfo(c, <const char *>raw),
           "HYPREDRV_PrintExitInfo")


def _solver_preset_register(str name, str yaml_text, str help_text=""):
    cdef bytes n = name.encode("utf-8")
    cdef bytes y = yaml_text.encode("utf-8")
    cdef bytes h = help_text.encode("utf-8")
    _check(_c.HYPREDRV_SolverPresetRegister(
               <const char *>n, <const char *>y, <const char *>h),
           "HYPREDRV_SolverPresetRegister")


def _precon_preset_register(str name, str yaml_text, str help_text=""):
    cdef bytes n = name.encode("utf-8")
    cdef bytes y = yaml_text.encode("utf-8")
    cdef bytes h = help_text.encode("utf-8")
    _check(_c.HYPREDRV_PreconPresetRegister(
               <const char *>n, <const char *>y, <const char *>h),
           "HYPREDRV_PreconPresetRegister")


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
    cdef bint _precon_created
    cdef object __weakref__

    def __cinit__(self, object comm=None, bint library_mode=True):
        self._handle = NULL
        self._solver_created = False
        self._precon_created = False

        _check(_create_driver(comm, &self._handle), "HYPREDRV_Create")
        _live_cores.add(self)
        if library_mode:
            _check(_c.HYPREDRV_SetLibraryMode(self._handle),
                   "HYPREDRV_SetLibraryMode")

    def __dealloc__(self):
        self._release()

    cdef uint32_t _release(self):
        """Destroy the solver and handle; return the first nonzero error code."""
        cdef uint32_t first_code = 0
        cdef uint32_t code = 0
        if self._handle == NULL:
            return 0
        if self._precon_created:
            first_code = _c.HYPREDRV_PreconDestroy(self._handle)
            self._precon_created = False
        if self._solver_created:
            code = _c.HYPREDRV_LinearSolverDestroy(self._handle)
            if first_code == 0:
                first_code = code
            self._solver_created = False
        code = _c.HYPREDRV_Destroy(&self._handle)
        self._handle = NULL
        if first_code == 0:
            first_code = code
        return first_code

    cdef int _require_open(self) except -1:
        if self._handle == NULL:
            raise RuntimeError("HypreDriveCore is closed")
        return 0

    cpdef close(self):
        """Tear down the underlying handle. Safe to call multiple times."""
        _live_cores.discard(self)
        _check(self._release(), "HypreDriveCore.close")

    # ------------------------------------------------------------------
    # Configuration: YAML in-memory
    # ------------------------------------------------------------------

    def parse_yaml(self, bytes yaml_text, list extra_args=None):
        """Configure solver/preconditioner from an in-memory YAML document.

        ``HYPREDRV_InputArgsParse`` accepts a YAML string in ``argv[0]`` (it
        falls back from filename-on-disk to literal-text when the path does
        not resolve), so we go through that same entry point and skip
        round-tripping through a temp file.

        ``extra_args`` supplies CLI-style override tokens appended after
        ``argv[0]``, e.g. ``[b"--solver:pcg:max_iter", b"100"]`` (``str``
        items are encoded as UTF-8; a leading ``"-a"``/``"--args"`` token is
        dropped so the C side always receives an unambiguous
        ``--path:to:key value`` pair list).
        """
        self._require_open()
        cdef bytes payload = yaml_text  # keep a reference alive for the duration
        cdef const char *payload_data = payload
        # keep encoded byte strings alive across the C call
        cdef list encoded = []
        cdef Py_ssize_t i, n = 0
        cdef char **argv
        if extra_args:
            for item in extra_args:
                encoded.append(
                    item if isinstance(item, bytes) else str(item).encode("utf-8")
                )
            if encoded and encoded[0] in (b"-a", b"--args"):
                encoded = encoded[1:]
            n = len(encoded)
        argv = <char **>malloc(<size_t>(n + 2) * sizeof(char *))
        if argv == NULL:
            raise MemoryError("failed to allocate argv for HYPREDRV_InputArgsParse")
        try:
            argv[0] = <char *>payload_data
            for i in range(n):
                argv[i + 1] = <char *><bytes>encoded[i]
            argv[n + 1] = NULL
            _check(
                _c.HYPREDRV_InputArgsParse(<int>(n + 1), argv, self._handle),
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

        The high-level driver normalizes the arrays; this boundary also
        validates direct callers before passing their addresses into C.
        """
        self._require_open()
        index_dtype = np.dtype(f"int{8 * _c.HYPREDRV_PythonIndexSize()}")
        real_dtype = np.dtype(f"float{8 * _HYPREDRIVE_PYTHON_REAL_SIZE}")
        validate_native_array(indptr, "indptr", index_dtype)
        validate_native_array(col_indices, "col_indices", index_dtype)
        validate_native_array(data, "data", real_dtype)
        validate_csr(row_start, row_end, indptr, col_indices, data)

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
        self._require_open()
        real_dtype = np.dtype(f"float{8 * _HYPREDRIVE_PYTHON_REAL_SIZE}")
        validate_native_array(values, "values", real_dtype)
        validate_rhs(row_start, row_end, values)

        _check(
            _c.HYPREDRV_PythonSetRHSFromArray(
                self._handle,
                <int64_t>row_start,
                <int64_t>row_end,
                <const void *>cnp.PyArray_DATA(values),
            ),
            "HYPREDRV_LinearSystemSetRHSFromArray",
        )

    def set_dofmap(self, cnp.ndarray labels):
        """Install per-row DOF labels for MGR.

        ``labels`` must be a C-contiguous, aligned 1-D array of C ints.
        The high-level driver also checks it against the local row count.
        """
        self._require_open()
        validate_native_array(labels, "labels", np.intc)
        cdef Py_ssize_t n = labels.shape[0]
        if n > <Py_ssize_t>0x7fffffff:
            raise ValueError("dofmap length exceeds C int range")
        _check(
            _c.HYPREDRV_LinearSystemSetDofmap(
                self._handle,
                <int>n,
                <const int *>cnp.PyArray_DATA(labels),
            ),
            "HYPREDRV_LinearSystemSetDofmap",
        )

    def setup_initial_guess(self):
        """Set up x0 according to ``linear_system.init_guess_mode``.

        Passing NULL delegates the actual initial-guess policy to the C layer.
        """
        self._require_open()
        _check(
            _c.HYPREDRV_LinearSystemSetInitialGuess(self._handle, NULL),
            "HYPREDRV_LinearSystemSetInitialGuess",
        )

    def reset_initial_guess(self):
        """Copy x0 into the working solution (annotates "reset_x0" internally)."""
        self._require_open()
        _check(
            _c.HYPREDRV_LinearSystemResetInitialGuess(self._handle),
            "HYPREDRV_LinearSystemResetInitialGuess",
        )

    # ------------------------------------------------------------------
    # Stats / Caliper annotations
    # ------------------------------------------------------------------

    def annotate_begin(self, bytes name, int id=-1):
        """Begin a stats/Caliper annotation region (see HYPREDRV_AnnotateBegin)."""
        self._require_open()
        _check(
            _c.HYPREDRV_AnnotateBegin(self._handle, <const char *>name, id),
            "HYPREDRV_AnnotateBegin",
        )

    def annotate_end(self, bytes name, int id=-1):
        """End a stats/Caliper annotation region (see HYPREDRV_AnnotateEnd)."""
        self._require_open()
        _check(
            _c.HYPREDRV_AnnotateEnd(self._handle, <const char *>name, id),
            "HYPREDRV_AnnotateEnd",
        )

    # ------------------------------------------------------------------
    # Solve cycle
    # ------------------------------------------------------------------

    def solver_create(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSolverCreate(self._handle),
               "HYPREDRV_LinearSolverCreate")
        self._solver_created = True

    def solver_setup(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSolverSetup(self._handle),
               "HYPREDRV_LinearSolverSetup")

    def solver_apply(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSolverApply(self._handle),
               "HYPREDRV_LinearSolverApply")

    def solver_set_relative_tol(self, double tol):
        self._require_open()
        _check(_c.HYPREDRV_LinearSolverSetRelativeTol(self._handle, tol),
               "HYPREDRV_LinearSolverSetRelativeTol")

    def solver_iterations(self):
        self._require_open()
        cdef int iters = 0
        _check(_c.HYPREDRV_LinearSolverGetNumIter(self._handle, &iters),
               "HYPREDRV_LinearSolverGetNumIter")
        return int(iters)

    def solver_converged(self):
        self._require_open()
        cdef int converged = 0
        _check(_c.HYPREDRV_LinearSolverGetConverged(self._handle, &converged),
               "HYPREDRV_LinearSolverGetConverged")
        return bool(converged)

    def solver_final_res_norm(self):
        self._require_open()
        cdef double norm = 0.0
        _check(_c.HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(
                   self._handle, &norm),
               "HYPREDRV_LinearSolverGetFinalRelativeResidualNorm")
        return float(norm)

    def solver_setup_time(self):
        self._require_open()
        cdef double seconds = 0.0
        _check(_c.HYPREDRV_LinearSolverGetSetupTime(self._handle, &seconds),
               "HYPREDRV_LinearSolverGetSetupTime")
        return float(seconds)

    def solver_solve_time(self):
        self._require_open()
        cdef double seconds = 0.0
        _check(_c.HYPREDRV_LinearSolverGetSolveTime(self._handle, &seconds),
               "HYPREDRV_LinearSolverGetSolveTime")
        return float(seconds)

    def solver_destroy(self):
        self._require_open()
        if self._solver_created:
            _check(_c.HYPREDRV_LinearSolverDestroy(self._handle),
                   "HYPREDRV_LinearSolverDestroy")
            self._solver_created = False

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def copy_solution(self, cnp.ndarray out):
        """Copy the local solution slab into ``out`` (a contiguous float64 array).

        Device values are synchronized and copied without changing the native
        solution vector's memory location or data pointer.
        """
        self._require_open()
        real_dtype = np.dtype(f"float{8 * _HYPREDRIVE_PYTHON_SOLUTION_ENTRY_SIZE}")
        validate_native_array(out, "out", real_dtype, writable=True)
        cdef Py_ssize_t n = out.shape[0]
        cdef size_t local_length = 0
        _check(_c.HYPREDRV_PythonGetSolutionLength(self._handle, &local_length),
               "HYPREDRV_LinearSystemGetSolutionLength")
        if <size_t>n > local_length:
            raise ValueError(f"output length {n} exceeds local solution length {local_length}")
        _check(
            _c.HYPREDRV_PythonCopySolutionValues(
                self._handle, cnp.PyArray_DATA(out), <size_t>n),
            "HYPREDRV_LinearSystemCopySolutionValues",
        )

    def solution_norm(self, str norm_type):
        self._require_open()
        cdef bytes norm_bytes = norm_type.encode("ascii")
        cdef double value = 0.0
        _check(
            _c.HYPREDRV_LinearSystemGetSolutionNorm(self._handle,
                                                    <const char *>norm_bytes,
                                                    &value),
            "HYPREDRV_LinearSystemGetSolutionNorm",
        )
        return float(value)

    # ------------------------------------------------------------------
    # Remaining HYPREDRV_ entry points
    # ------------------------------------------------------------------

    cdef bytes _enc(self, str s):
        return s.encode("utf-8")

    def object_set_name(self, str name):
        self._require_open()
        cdef bytes raw = self._enc(name)
        _check(_c.HYPREDRV_ObjectSetName(self._handle, <const char *>raw),
               "HYPREDRV_ObjectSetName")

    def get_warmup(self):
        self._require_open()
        cdef int v = 0
        _check(_c.HYPREDRV_InputArgsGetWarmup(self._handle, &v),
               "HYPREDRV_InputArgsGetWarmup")
        return int(v)

    def get_num_repetitions(self):
        self._require_open()
        cdef int v = 0
        _check(_c.HYPREDRV_InputArgsGetNumRepetitions(self._handle, &v),
               "HYPREDRV_InputArgsGetNumRepetitions")
        return int(v)

    def get_num_linear_systems(self):
        self._require_open()
        cdef int v = 0
        _check(_c.HYPREDRV_InputArgsGetNumLinearSystems(self._handle, &v),
               "HYPREDRV_InputArgsGetNumLinearSystems")
        return int(v)

    def get_num_precon_variants(self):
        self._require_open()
        cdef int v = 0
        _check(_c.HYPREDRV_InputArgsGetNumPreconVariants(self._handle, &v),
               "HYPREDRV_InputArgsGetNumPreconVariants")
        return int(v)

    def set_precon_variant(self, int variant_idx):
        self._require_open()
        _check(_c.HYPREDRV_InputArgsSetPreconVariant(self._handle, variant_idx),
               "HYPREDRV_InputArgsSetPreconVariant")

    def set_precon_preset(self, str preset):
        self._require_open()
        cdef bytes raw = self._enc(preset)
        _check(_c.HYPREDRV_InputArgsSetPreconPreset(self._handle,
                                                    <const char *>raw),
               "HYPREDRV_InputArgsSetPreconPreset")

    def set_solver_preset(self, str preset):
        self._require_open()
        cdef bytes raw = self._enc(preset)
        _check(_c.HYPREDRV_InputArgsSetSolverPreset(self._handle,
                                                    <const char *>raw),
               "HYPREDRV_InputArgsSetSolverPreset")

    # -- linear system (file / handle based) ---------------------------

    def linear_system_build(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemBuild(self._handle),
               "HYPREDRV_LinearSystemBuild")

    def read_matrix(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemReadMatrix(self._handle),
               "HYPREDRV_LinearSystemReadMatrix")

    def set_matrix_handle(self, object mat):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetMatrix(
                   self._handle, <_c.HYPRE_Matrix>_as_ptr(mat, "mat")),
               "HYPREDRV_LinearSystemSetMatrix")

    def set_prec_matrix_handle(self, object mat):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetPrecMatrix(
                   self._handle, <_c.HYPRE_Matrix>_as_ptr(mat, "mat")),
               "HYPREDRV_LinearSystemSetPrecMatrix")

    def set_discrete_gradient_handle(self, object mat):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetDiscreteGradient(
                   self._handle, <_c.HYPRE_Matrix>_as_ptr(mat, "G")),
               "HYPREDRV_LinearSystemSetDiscreteGradient")

    def set_discrete_curl_handle(self, object mat):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetDiscreteCurl(
                   self._handle, <_c.HYPRE_Matrix>_as_ptr(mat, "C")),
               "HYPREDRV_LinearSystemSetDiscreteCurl")

    def set_coarse_schur_handle(self, int level, object mat):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetCoarseSchur(
                   self._handle, level, <_c.HYPRE_Matrix>_as_ptr(mat, "mat_S")),
               "HYPREDRV_LinearSystemSetCoarseSchur")

    def set_coordinates_handle(self, object x, object y, object z):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetCoordinates(
                   self._handle,
                   <_c.HYPRE_Vector>_as_ptr(x, "x"),
                   <_c.HYPRE_Vector>_as_ptr(y, "y"),
                   <_c.HYPRE_Vector>_as_ptr(z, "z")),
               "HYPREDRV_LinearSystemSetCoordinates")

    def set_rhs_handle(self, object vec):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetRHS(
                   self._handle, <_c.HYPRE_Vector>_as_ptr(vec, "vec")),
               "HYPREDRV_LinearSystemSetRHS")

    def set_solution_handle(self, object vec):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetSolution(
                   self._handle, <_c.HYPRE_Vector>_as_ptr(vec, "vec")),
               "HYPREDRV_LinearSystemSetSolution")

    def set_reference_solution_handle(self, object vec):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetReferenceSolution(
                   self._handle, <_c.HYPRE_Vector>_as_ptr(vec, "vec")),
               "HYPREDRV_LinearSystemSetReferenceSolution")

    def get_matrix_handle(self):
        self._require_open()
        cdef _c.HYPRE_Matrix mat = NULL
        _check(_c.HYPREDRV_LinearSystemGetMatrix(self._handle, &mat),
               "HYPREDRV_LinearSystemGetMatrix")
        return <uintptr_t>mat

    def get_rhs_handle(self):
        self._require_open()
        cdef _c.HYPRE_Vector vec = NULL
        _check(_c.HYPREDRV_LinearSystemGetRHS(self._handle, &vec),
               "HYPREDRV_LinearSystemGetRHS")
        return <uintptr_t>vec

    def get_solution_handle(self):
        self._require_open()
        cdef _c.HYPRE_Vector vec = NULL
        _check(_c.HYPREDRV_LinearSystemGetSolution(self._handle, &vec),
               "HYPREDRV_LinearSystemGetSolution")
        return <uintptr_t>vec

    def set_interleaved_dofmap(self, int num_local_blocks, int num_dof_types):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetInterleavedDofmap(
                   self._handle, num_local_blocks, num_dof_types),
               "HYPREDRV_LinearSystemSetInterleavedDofmap")

    def set_contiguous_dofmap(self, int num_local_blocks, int num_dof_types):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemSetContiguousDofmap(
                   self._handle, num_local_blocks, num_dof_types),
               "HYPREDRV_LinearSystemSetContiguousDofmap")

    def read_dofmap(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemReadDofmap(self._handle),
               "HYPREDRV_LinearSystemReadDofmap")

    def print_dofmap(self, str filename):
        self._require_open()
        cdef bytes raw = self._enc(filename)
        _check(_c.HYPREDRV_LinearSystemPrintDofmap(self._handle,
                                                   <const char *>raw),
               "HYPREDRV_LinearSystemPrintDofmap")

    def print_linear_system(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemPrint(self._handle),
               "HYPREDRV_LinearSystemPrint")

    def compute_eigenspectrum(self):
        self._require_open()
        _check(_c.HYPREDRV_LinearSystemComputeEigenspectrum(self._handle),
               "HYPREDRV_LinearSystemComputeEigenspectrum")

    def set_near_null_space(self, int num_entries, int num_components,
                            object values):
        """``values``: C-contiguous HYPRE_Real array of length
        ``num_entries * num_components`` (copied by HYPREDRV), or ``None``."""
        self._require_open()
        return self._set_space(num_entries, num_components, values, True)

    def set_null_space(self, int num_entries, int num_components,
                       object values):
        self._require_open()
        return self._set_space(num_entries, num_components, values, False)

    cdef _set_space(self, int num_entries, int num_components, object values,
                    bint near):
        cdef const _c.HYPRE_Complex *ptr = NULL
        cdef cnp.ndarray arr
        if num_entries < 0 or num_components <= 0:
            raise ValueError("num_entries must be >= 0 and num_components > 0")
        if values is not None:
            arr = values
            real_dtype = np.dtype(f"float{8 * _HYPREDRIVE_PYTHON_REAL_SIZE}")
            validate_native_array(arr, "values", real_dtype)
            if arr.ndim != 1 or arr.shape[0] != <Py_ssize_t>num_entries * num_components:
                raise ValueError(
                    "values must be 1-D with num_entries * num_components entries"
                )
            ptr = <const _c.HYPRE_Complex *>cnp.PyArray_DATA(arr)
        if near:
            _check(_c.HYPREDRV_LinearSystemSetNearNullSpace(
                       self._handle, num_entries, num_components, ptr),
                   "HYPREDRV_LinearSystemSetNearNullSpace")
        else:
            _check(_c.HYPREDRV_LinearSystemSetNullSpace(
                       self._handle, num_entries, num_components, ptr),
                   "HYPREDRV_LinearSystemSetNullSpace")

    # -- native value access (copies; never aliases HYPRE memory) -------

    cdef Py_ssize_t _local_length(self) except -1:
        cdef size_t n = 0
        _check(_c.HYPREDRV_PythonGetSolutionLength(self._handle, &n),
               "HYPREDRV_LinearSystemGetSolutionLength")
        return <Py_ssize_t>n

    cdef object _copy_host(self, const _c.HYPRE_Complex *p, Py_ssize_t n):
        cdef cnp.ndarray out = np.empty(
            n, dtype=np.dtype(f"float{8 * _HYPREDRIVE_PYTHON_REAL_SIZE}"))
        cdef Py_ssize_t i
        cdef double *dst = <double *>cnp.PyArray_DATA(out)
        for i in range(n):
            dst[i] = p[i]
        return out

    def get_solution_values(self):
        """Copy of the native local solution (host memory only)."""
        self._require_open()
        cdef _c.HYPRE_Complex *p = NULL
        cdef _c.HYPRE_MemoryLocation loc = 0
        _check(_c.HYPREDRV_LinearSystemGetSolutionValues(self._handle, &p, &loc),
               "HYPREDRV_LinearSystemGetSolutionValues")
        if loc != _c.HYPRE_MEMORY_HOST:
            raise RuntimeError(
                "native solution lives in device memory; use copy_solution()"
            )
        return self._copy_host(p, self._local_length())

    def get_rhs_values(self, Py_ssize_t n):
        """Copy of the native local RHS (host memory only)."""
        self._require_open()
        cdef _c.HYPRE_Complex *p = NULL
        _check(_c.HYPREDRV_LinearSystemGetRHSValues(self._handle, &p),
               "HYPREDRV_LinearSystemGetRHSValues")
        if p == NULL:
            raise RuntimeError("RHS values are not available")
        return self._copy_host(p, n)

    # -- state vectors ----------------------------------------------------

    def state_vector_set(self, list handles):
        self._require_open()
        cdef Py_ssize_t i, n = len(handles)
        cdef _c.HYPRE_IJVector *vecs = NULL
        if n == 0 or n > 0x7fffffff:
            raise ValueError("handles must hold between 1 and INT_MAX handles")
        vecs = <_c.HYPRE_IJVector *>malloc(<size_t>n * sizeof(_c.HYPRE_IJVector))
        if vecs == NULL:
            raise MemoryError("failed to allocate state vector array")
        try:
            for i in range(n):
                vecs[i] = <_c.HYPRE_IJVector>_as_ptr(handles[i], "handles[%d]" % i)
            _check(_c.HYPREDRV_StateVectorSet(self._handle, <int>n, vecs),
                   "HYPREDRV_StateVectorSet")
        finally:
            free(vecs)

    def state_vector_get_values(self, int index, Py_ssize_t n):
        """Copy of the local data of state vector ``index`` (host memory)."""
        self._require_open()
        cdef _c.HYPRE_Complex *p = NULL
        _check(_c.HYPREDRV_StateVectorGetValues(self._handle, index, &p),
               "HYPREDRV_StateVectorGetValues")
        if p == NULL:
            raise RuntimeError("state vector data is not available")
        return self._copy_host(p, n)

    def state_vector_copy(self, int index_in, int index_out):
        self._require_open()
        _check(_c.HYPREDRV_StateVectorCopy(self._handle, index_in, index_out),
               "HYPREDRV_StateVectorCopy")

    def state_vector_update_all(self):
        self._require_open()
        _check(_c.HYPREDRV_StateVectorUpdateAll(self._handle),
               "HYPREDRV_StateVectorUpdateAll")

    def state_vector_apply_correction(self, int state_idx):
        self._require_open()
        _check(_c.HYPREDRV_StateVectorApplyCorrection(self._handle, state_idx),
               "HYPREDRV_StateVectorApplyCorrection")

    # -- standalone preconditioner ---------------------------------------

    def precon_create(self):
        self._require_open()
        _check(_c.HYPREDRV_PreconCreate(self._handle), "HYPREDRV_PreconCreate")
        self._precon_created = True

    def precon_setup(self):
        self._require_open()
        _check(_c.HYPREDRV_PreconSetup(self._handle), "HYPREDRV_PreconSetup")

    def precon_apply(self, object vec_b, object vec_x):
        self._require_open()
        _check(_c.HYPREDRV_PreconApply(
                   self._handle,
                   <_c.HYPRE_Vector>_as_ptr(vec_b, "vec_b"),
                   <_c.HYPRE_Vector>_as_ptr(vec_x, "vec_x")),
               "HYPREDRV_PreconApply")

    def precon_destroy(self):
        self._require_open()
        _check(_c.HYPREDRV_PreconDestroy(self._handle), "HYPREDRV_PreconDestroy")
        self._precon_created = False

    # -- stats -------------------------------------------------------------

    def stats_print(self):
        self._require_open()
        _check(_c.HYPREDRV_StatsPrint(self._handle), "HYPREDRV_StatsPrint")

    def annotate_level_begin(self, int level, bytes name, int id=-1):
        self._require_open()
        _check(_c.HYPREDRV_AnnotateLevelBegin(
                   self._handle, level, <const char *>name, id),
               "HYPREDRV_AnnotateLevelBegin")

    def annotate_level_end(self, int level, bytes name, int id=-1):
        self._require_open()
        _check(_c.HYPREDRV_AnnotateLevelEnd(
                   self._handle, level, <const char *>name, id),
               "HYPREDRV_AnnotateLevelEnd")

    def stats_level_count(self, int level):
        self._require_open()
        cdef int count = 0
        _check(_c.HYPREDRV_StatsLevelGetCount(self._handle, level, &count),
               "HYPREDRV_StatsLevelGetCount")
        return int(count)

    def stats_level_entry(self, int level, int index):
        self._require_open()
        cdef int eid = 0, ns = 0, it = 0
        cdef double st = 0.0, sv = 0.0
        _check(_c.HYPREDRV_StatsLevelGetEntry(
                   self._handle, level, index, &eid, &ns, &it, &st, &sv),
               "HYPREDRV_StatsLevelGetEntry")
        return (int(eid), int(ns), int(it), float(st), float(sv))

    def stats_level_print(self, int level):
        self._require_open()
        _check(_c.HYPREDRV_StatsLevelPrint(self._handle, level),
               "HYPREDRV_StatsLevelPrint")
