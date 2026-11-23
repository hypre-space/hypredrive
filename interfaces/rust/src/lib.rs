// Copyright (c) 2024, Lawrence Livermore National Security, LLC.
// See the top-level LICENSE and NOTICE files for details.

//! A small, owned Rust interface to the public hypredrive C API.
//!
//! Initialize MPI with [`mpi::initialize`] first, then create a [`Runtime`].
//! Drivers borrow both the runtime and their MPI communicator. All driver
//! operations must be called in matching order by every rank in that communicator.
//! Rust input validation is agreed collectively before entering native operations.
//!
//! The runtime has exclusive ownership of hypredrive's process-global state.
//! Other native callers must coordinate their entire hypredrive lifetime with it.
//! The runtime and its drivers stay on the MPI initialization thread.

mod ffi {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code
    )]

    use mpi::ffi::MPI_Comm;

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    trait CommRepresentation {
        const IS_INT: bool;
    }

    impl CommRepresentation for std::ffi::c_int {
        const IS_INT: bool = true;
    }

    impl CommRepresentation for std::ffi::c_uint {
        const IS_INT: bool = true;
    }

    impl<T> CommRepresentation for *mut T {
        const IS_INT: bool = false;
    }

    impl<T> CommRepresentation for *const T {
        const IS_INT: bool = false;
    }

    const fn is_integer_handle<T: CommRepresentation>(_: fn(MPI_Comm) -> T) -> bool {
        T::IS_INT
    }

    // Check the MPI ABI Cargo actually compiled, even if mpi-sys was cached.
    const _: () = assert!(
        std::mem::size_of::<MPI_Comm>() == HYPREDRV_RUST_MPI_COMM_SIZE as usize
            && std::mem::align_of::<MPI_Comm>() == HYPREDRV_RUST_MPI_COMM_ALIGN as usize
            && is_integer_handle(|comm| comm.0) == (HYPREDRV_RUST_MPI_COMM_IS_INT != 0)
            && mpi::ffi::MPI_VERSION as i64 == HYPREDRV_RUST_MPI_VERSION as i64
            && mpi::ffi::MPI_SUBVERSION as i64 == HYPREDRV_RUST_MPI_SUBVERSION as i64,
        "mpi-sys ABI differs from the selected MPI headers. Use a fresh CARGO_TARGET_DIR for this MPI installation."
    );
}

use mpi::collective::SystemOperation;
use mpi::environment::Universe;
use mpi::raw::AsRaw;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Root};
use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The configured HYPRE global index type, obtained from its native headers.
///
/// Use `BigInt::try_from(index)` to convert application indices without truncation.
pub type BigInt = ffi::HYPRE_BigInt;

static RUNTIME_ACTIVE: AtomicBool = AtomicBool::new(false);

/// An input, runtime, or native API failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    operation: &'static str,
    status: Option<u32>,
    message: String,
}

impl Error {
    /// The operation that failed.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// The original native `u32` status, when a native call failed.
    ///
    /// On a rank whose call succeeded, this is a failing peer's status. Rust
    /// validation and runtime ownership errors have no native status.
    pub fn status(&self) -> Option<u32> {
        self.status
    }

    fn invalid(operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            operation,
            status: None,
            message: message.into(),
        }
    }

    fn native(operation: &'static str, status: u32) -> Self {
        Self {
            operation,
            status: Some(status),
            message: format!("native API status 0x{status:08x}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} failed: {}", self.operation, self.message)
    }
}

impl std::error::Error for Error {}

/// A process-level hypredrive lifetime that borrows the initialized MPI universe.
///
/// Only one Rust runtime can be active in a process. Create it on the thread
/// that initialized MPI, and drop all drivers before dropping it. This type is
/// neither `Send` nor `Sync` because native calls share unsynchronized state.
pub struct Runtime<'mpi> {
    _universe: &'mpi Universe,
    _thread_bound: PhantomData<Rc<()>>,
}

impl<'mpi> Runtime<'mpi> {
    /// Initialize hypredrive after MPI initialization.
    pub fn new(universe: &'mpi Universe) -> Result<Self, Error> {
        // Reject other threads before touching the process-wide reservation, so
        // a caller that is about to fail cannot block the legitimate thread.
        let mut is_main = 0;
        // The borrowed Universe keeps MPI initialized throughout this call.
        let mpi_status = unsafe { mpi::ffi::MPI_Is_thread_main(&mut is_main) };
        if mpi_status != mpi::ffi::MPI_SUCCESS as i32 || is_main == 0 {
            return Err(Error::invalid(
                "Runtime::new",
                "the runtime must be created on the MPI initialization thread",
            ));
        }
        if RUNTIME_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::invalid(
                "Runtime::new",
                "a hypredrive runtime is already active in this process",
            ));
        }

        // The atomic reservation excludes every other Rust native caller.
        let status = unsafe { ffi::HYPREDRV_Initialize() };
        if status != 0 {
            describe_error(status);
            // Finalize is valid even after partially successful initialization.
            describe_error(unsafe { ffi::HYPREDRV_Finalize() });
            RUNTIME_ACTIVE.store(false, Ordering::Release);
            return Err(Error::native("HYPREDRV_Initialize", status));
        }
        Ok(Self {
            _universe: universe,
            _thread_bound: PhantomData,
        })
    }
}

impl Drop for Runtime<'_> {
    fn drop(&mut self) {
        // Drivers borrow this runtime and have already been dropped.
        describe_error(unsafe { ffi::HYPREDRV_Finalize() });
        RUNTIME_ACTIVE.store(false, Ordering::Release);
    }
}

/// Owned local solution values and statistics for the last solve.
#[derive(Clone, Debug)]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub iterations: i32,
    pub converged: bool,
    pub final_res_norm: f64,
    pub setup_time: f64,
    pub solve_time: f64,
}

/// A configured driver borrowing a runtime and an MPI intracommunicator.
///
/// Every method is collective on the supplied communicator. Call methods in
/// matching order on all ranks, including calls that fail Rust validation.
/// Native matrix assembly copies CSR/RHS buffers before returning.
pub struct HypreDrive<'a, C: Communicator> {
    handle: ffi::HYPREDRV_t,
    _runtime: &'a Runtime<'a>,
    communicator: &'a C,
    _native_communicator: SimpleCommunicator,
    rows: Option<(BigInt, BigInt, usize)>,
    rhs_ready: bool,
    guess_ready: bool,
    reset_pending: bool,
}

impl<'a, C: Communicator> HypreDrive<'a, C> {
    /// Create a driver from YAML text, parsed by the native configuration parser.
    ///
    /// Every rank must supply identical YAML text.
    pub fn from_yaml(
        runtime: &'a Runtime<'a>,
        communicator: &'a C,
        yaml: &str,
    ) -> Result<Self, Error> {
        require_intracommunicator(communicator, "from_yaml")?;
        // A newline unambiguously selects inline input in the native parser.
        let configuration = CString::new(format!("{yaml}\n"))
            .map_err(|_| Error::invalid("from_yaml", "YAML contains an embedded NUL"));
        let configuration = agree_validation(communicator, "from_yaml", configuration)?;
        require_identical_yaml(communicator, configuration.as_bytes())?;
        Self::configured(runtime, communicator, vec![configuration])
    }

    /// Create a driver from a `.yaml` or `.yml` file read by the native parser.
    ///
    /// Rank zero reads the file and broadcasts it. Relative paths in the YAML
    /// retain the native parser's configuration-directory semantics.
    /// File extensions are case-insensitive. Paths must be UTF-8 and contain no
    /// spaces, as required by the native parser's filename recognition.
    pub fn from_file(
        runtime: &'a Runtime<'a>,
        communicator: &'a C,
        path: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        require_intracommunicator(communicator, "from_file")?;
        let configuration = (|| {
            let path = path.as_ref();
            let name = path
                .to_str()
                .ok_or_else(|| Error::invalid("from_file", "path must be UTF-8"))?;
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !extension.eq_ignore_ascii_case("yaml") && !extension.eq_ignore_ascii_case("yml") {
                return Err(Error::invalid(
                    "from_file",
                    "configuration file must have a .yaml or .yml extension",
                ));
            }
            if name.contains(' ') {
                return Err(Error::invalid(
                    "from_file",
                    "the native parser requires a configuration path without spaces",
                ));
            }
            CString::new(name)
                .map_err(|_| Error::invalid("from_file", "path contains an embedded NUL"))
        })();
        let configuration = agree_validation(communicator, "from_file", configuration)?;
        // A separate argv[0] keeps unusual filename characters out of the
        // native parser's inline-YAML detection.
        Self::configured(
            runtime,
            communicator,
            vec![CString::new("hypredrive-rust").unwrap(), configuration],
        )
    }

    fn configured(
        runtime: &'a Runtime<'a>,
        communicator: &'a C,
        arguments: Vec<CString>,
    ) -> Result<Self, Error> {
        // Keep a native communicator alive even when safe Rust code forgets
        // the driver and releases the source communicator before finalization.
        // A forgotten duplicate survives until MPI finalization.
        let native_communicator = communicator.duplicate();
        let mut driver = Self {
            handle: ptr::null_mut(),
            _runtime: runtime,
            communicator,
            _native_communicator: native_communicator,
            rows: None,
            rhs_ready: false,
            guess_ready: false,
            reset_pending: false,
        };
        // The communicator remains borrowed until after the native handle dies.
        let status = unsafe {
            ffi::HYPREDRV_Create(driver._native_communicator.as_raw(), &mut driver.handle)
        };
        driver.check("HYPREDRV_Create", status)?;
        let status = unsafe { ffi::HYPREDRV_SetLibraryMode(driver.handle) };
        driver.check("HYPREDRV_SetLibraryMode", status)?;
        let mut argv: Vec<_> = arguments.iter().map(|s| s.as_ptr().cast_mut()).collect();
        // The C API consumes these strings during parsing and retains no pointers.
        let status = unsafe {
            ffi::HYPREDRV_InputArgsParse(argv.len() as i32, argv.as_mut_ptr(), driver.handle)
        };
        driver.check("HYPREDRV_InputArgsParse", status)?;
        Ok(driver)
    }

    /// Copy a rank-local CSR slab with inclusive global row bounds.
    ///
    /// Every rank must own at least one row. Columns are global indices. Offsets
    /// may start above zero, provided the complete referenced prefix is present
    /// in both `columns` and `values`. Native code validates global partitions
    /// and referenced column indices.
    pub fn set_matrix_csr(
        &mut self,
        row_start: BigInt,
        row_end: BigInt,
        indptr: &[BigInt],
        columns: &[BigInt],
        values: &[f64],
    ) -> Result<(), Error> {
        let rows = agree_validation(
            self.communicator,
            "set_matrix_csr",
            validate_csr(row_start, row_end, indptr, columns, values),
        )?;
        // Validation proves every native CSR read is inside a supplied slice.
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetMatrixFromCSR(
                self.handle,
                row_start,
                row_end,
                indptr.as_ptr(),
                columns.as_ptr(),
                values.as_ptr(),
            )
        };
        self.check("HYPREDRV_LinearSystemSetMatrixFromCSR", status)?;
        self.rows = Some((row_start, row_end, rows));
        self.rhs_ready = false;
        self.guess_ready = false;
        self.reset_pending = false;
        // CSR assembly preserves the old matrix on failure. Advance the native
        // system only after its replacement has been installed successfully.
        self.annotate(c"matrix")
    }

    /// Copy the local RHS for the current matrix's row range.
    ///
    /// Replacing the RHS preserves the existing working solution and guess.
    /// Call [`Self::reset_initial_guess`] or [`Self::set_initial_guess`] to reset it.
    pub fn set_rhs(&mut self, values: &[f64]) -> Result<(), Error> {
        let range = self
            .rows
            .ok_or_else(|| Error::invalid("set_rhs", "set the matrix before setting the RHS"));
        let range = range.and_then(|range| {
            if values.len() != range.2 {
                return Err(Error::invalid(
                    "set_rhs",
                    "RHS length must match the local matrix row count",
                ));
            }
            Ok(range)
        });
        let (row_start, row_end, _) = agree_validation(self.communicator, "set_rhs", range)?;
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetRHSFromArray(
                self.handle,
                row_start,
                row_end,
                values.as_ptr(),
            )
        };
        self.check("HYPREDRV_LinearSystemSetRHSFromArray", status)?;
        self.rhs_ready = true;
        Ok(())
    }

    /// Construct the configured initial guess and apply it on the next solve.
    pub fn set_initial_guess(&mut self) -> Result<(), Error> {
        self.require_system("set_initial_guess")?;
        let status =
            unsafe { ffi::HYPREDRV_LinearSystemSetInitialGuess(self.handle, ptr::null_mut()) };
        self.check("HYPREDRV_LinearSystemSetInitialGuess", status)?;
        self.guess_ready = true;
        self.reset_pending = true;
        Ok(())
    }

    /// Restore the configured initial guess on the next solve.
    pub fn reset_initial_guess(&mut self) -> Result<(), Error> {
        let validation = if self.guess_ready {
            Ok(())
        } else {
            Err(Error::invalid(
                "reset_initial_guess",
                "set the initial guess or solve before resetting it",
            ))
        };
        agree_validation(self.communicator, "reset_initial_guess", validation)?;
        self.reset_pending = true;
        Ok(())
    }

    /// Create, set up and apply the configured solver, returning owned host data.
    ///
    /// A missing initial guess is constructed according to the YAML settings.
    /// Nonconvergence is returned as `Ok` with `converged == false`.
    pub fn solve(&mut self) -> Result<SolveResult, Error> {
        self.require_system("solve")?;
        if !self.guess_ready {
            self.set_initial_guess()?;
        }
        if self.reset_pending {
            // Apply the requested guess exactly once per solve. Repeated
            // setter/reset calls before solving must not consume native solve
            // entries or skip the first repetition of a new matrix.
            let status = unsafe { ffi::HYPREDRV_LinearSystemResetInitialGuess(self.handle) };
            self.check("HYPREDRV_LinearSystemResetInitialGuess", status)?;
            self.reset_pending = false;
        } else {
            // Start a fresh native statistics entry while retaining the current
            // working solution as a warm start. Native ResetInitialGuess already
            // records this entry when an actual reset was requested.
            self.annotate(c"reset_x0")?;
        }
        let status = unsafe { ffi::HYPREDRV_LinearSolverCreate(self.handle) };
        self.check("HYPREDRV_LinearSolverCreate", status)?;
        let result = (|| {
            let status = unsafe { ffi::HYPREDRV_LinearSolverSetup(self.handle) };
            self.check("HYPREDRV_LinearSolverSetup", status)?;
            let status = unsafe { ffi::HYPREDRV_LinearSolverApply(self.handle) };
            self.check("HYPREDRV_LinearSolverApply", status)?;

            let mut length: BigInt = 0;
            let status =
                unsafe { ffi::HYPREDRV_LinearSystemGetSolutionLength(self.handle, &mut length) };
            self.check("HYPREDRV_LinearSystemGetSolutionLength", status)?;
            let allocation = (|| {
                let length = usize::try_from(length)
                    .map_err(|_| Error::invalid("solve", "solution length exceeds usize range"))?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(length)
                    .map_err(|error| Error::invalid("solve", error.to_string()))?;
                values.resize(length, 0.0);
                Ok(values)
            })();
            let mut x = agree_validation(self.communicator, "solve", allocation)?;
            let status = unsafe {
                ffi::HYPREDRV_LinearSystemCopySolutionValues(self.handle, length, x.as_mut_ptr())
            };
            self.check("HYPREDRV_LinearSystemCopySolutionValues", status)?;

            let mut iterations = 0;
            let mut converged = 0;
            let mut final_res_norm = 0.0;
            let mut setup_time = 0.0;
            let mut solve_time = 0.0;
            let status =
                unsafe { ffi::HYPREDRV_LinearSolverGetNumIter(self.handle, &mut iterations) };
            self.check("HYPREDRV_LinearSolverGetNumIter", status)?;
            let status =
                unsafe { ffi::HYPREDRV_LinearSolverGetConverged(self.handle, &mut converged) };
            self.check("HYPREDRV_LinearSolverGetConverged", status)?;
            let status = unsafe {
                ffi::HYPREDRV_LinearSolverGetFinalRelativeResidualNorm(
                    self.handle,
                    &mut final_res_norm,
                )
            };
            self.check("HYPREDRV_LinearSolverGetFinalRelativeResidualNorm", status)?;
            let status =
                unsafe { ffi::HYPREDRV_LinearSolverGetSetupTime(self.handle, &mut setup_time) };
            self.check("HYPREDRV_LinearSolverGetSetupTime", status)?;
            let status =
                unsafe { ffi::HYPREDRV_LinearSolverGetSolveTime(self.handle, &mut solve_time) };
            self.check("HYPREDRV_LinearSolverGetSolveTime", status)?;
            Ok(SolveResult {
                x,
                iterations,
                converged: converged != 0,
                final_res_norm,
                setup_time,
                solve_time,
            })
        })();
        // Release the per-solve Krylov solver and apply the preconditioner reuse
        // teardown policy on both success and failure. All ranks enter together
        // because every step above synchronizes status across the communicator.
        let destroy_status = unsafe { ffi::HYPREDRV_LinearSolverDestroy(self.handle) };
        match result {
            Ok(solve_result) => {
                self.check("HYPREDRV_LinearSolverDestroy", destroy_status)?;
                Ok(solve_result)
            }
            Err(error) => {
                describe_error(destroy_status);
                Err(error)
            }
        }
    }

    fn require_system(&self, operation: &'static str) -> Result<(), Error> {
        let validation = if self.rows.is_some() && self.rhs_ready {
            Ok(())
        } else {
            Err(Error::invalid(operation, "set the matrix and RHS first"))
        };
        agree_validation(self.communicator, operation, validation)
    }

    fn annotate(&self, name: &CStr) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_AnnotateBegin(self.handle, name.as_ptr(), -1) };
        self.check("HYPREDRV_AnnotateBegin", status)?;
        let status = unsafe { ffi::HYPREDRV_AnnotateEnd(self.handle, name.as_ptr(), -1) };
        self.check("HYPREDRV_AnnotateEnd", status)
    }

    fn check(&self, operation: &'static str, status: u32) -> Result<(), Error> {
        // Preserve diagnostics before any later HYPREDRV call resets its scope.
        describe_error(status);
        let mut collective_status = 0;
        self.communicator
            .all_reduce_into(&status, &mut collective_status, SystemOperation::max());
        if collective_status == 0 {
            Ok(())
        } else {
            Err(Error::native(
                operation,
                if status != 0 {
                    status
                } else {
                    collective_status
                },
            ))
        }
    }
}

impl<C: Communicator> Drop for HypreDrive<'_, C> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Destruction is local and valid during partial construction.
            describe_error(unsafe { ffi::HYPREDRV_Destroy(&mut self.handle) });
        }
    }
}

fn describe_error(status: u32) {
    if status != 0 {
        // This diagnostic helper prints and returns without aborting or panicking.
        unsafe { ffi::HYPREDRV_ErrorCodeDescribe(status) };
    }
}

fn require_intracommunicator<C: Communicator>(
    communicator: &C,
    operation: &'static str,
) -> Result<(), Error> {
    if communicator.test_inter() {
        return Err(Error::invalid(
            operation,
            "hypredrive requires an MPI intracommunicator",
        ));
    }
    Ok(())
}

fn require_identical_yaml<C: Communicator>(communicator: &C, bytes: &[u8]) -> Result<(), Error> {
    let length = agree_validation(
        communicator,
        "from_yaml",
        u64::try_from(bytes.len())
            .map_err(|_| Error::invalid("from_yaml", "YAML length exceeds u64 range")),
    )?;
    let mut root_length = length;
    let root = communicator.process_at_rank(0);
    root.broadcast_into(&mut root_length);
    let length_validation = if root_length == length {
        Ok(())
    } else {
        Err(Error::invalid(
            "from_yaml",
            "YAML text must be identical on every MPI rank",
        ))
    };
    agree_validation(communicator, "from_yaml", length_validation)?;

    // Exact, bounded-size comparisons avoid hash collisions and MPI count
    // overflow for configurations larger than an MPI int-sized buffer.
    let mut block = [0_u8; 4096];
    let mut identical = true;
    for local in bytes.chunks(block.len()) {
        let received = &mut block[..local.len()];
        if communicator.rank() == 0 {
            received.copy_from_slice(local);
        }
        root.broadcast_into(received);
        identical &= received == local;
    }
    let validation = if identical {
        Ok(())
    } else {
        Err(Error::invalid(
            "from_yaml",
            "YAML text must be identical on every MPI rank",
        ))
    };
    agree_validation(communicator, "from_yaml", validation)
}

fn agree_validation<C: Communicator, T>(
    communicator: &C,
    operation: &'static str,
    validation: Result<T, Error>,
) -> Result<T, Error> {
    let failed = i32::from(validation.is_err());
    let mut any_failed = 0;
    communicator.all_reduce_into(&failed, &mut any_failed, SystemOperation::max());
    if any_failed != 0 && validation.is_ok() {
        return Err(Error::invalid(
            operation,
            "validation failed on another MPI rank",
        ));
    }
    validation
}

fn validate_csr(
    row_start: BigInt,
    row_end: BigInt,
    indptr: &[BigInt],
    columns: &[BigInt],
    values: &[f64],
) -> Result<usize, Error> {
    let invalid = |message| Error::invalid("set_matrix_csr", message);
    if row_start < 0 || row_end < row_start || row_end.checked_add(1).is_none() {
        return Err(invalid(
            "row bounds must be nonnegative, nonempty and representable",
        ));
    }
    let count = row_end
        .checked_sub(row_start)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| invalid("local row count overflows BigInt"))?;
    ffi::HYPRE_Int::try_from(count)
        .map_err(|_| invalid("local row count exceeds HYPRE_Int range"))?;
    let rows =
        usize::try_from(count).map_err(|_| invalid("local row count exceeds usize range"))?;
    let offsets = rows
        .checked_add(1)
        .ok_or_else(|| invalid("CSR offset length exceeds usize range"))?;
    if indptr.len() != offsets {
        return Err(invalid(
            "indptr length must equal the local row count plus one",
        ));
    }
    if columns.len() != values.len() {
        return Err(invalid("column and value slice lengths must match"));
    }
    if indptr[0] < 0 || indptr.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(invalid("CSR offsets must be nonnegative and monotone"));
    }
    let terminal = usize::try_from(indptr[rows])
        .map_err(|_| invalid("terminal CSR offset exceeds usize range"))?;
    if terminal > columns.len() || terminal > values.len() {
        return Err(invalid(
            "terminal CSR offset exceeds the column/value buffers",
        ));
    }
    let nonzeros = indptr[rows]
        .checked_sub(indptr[0])
        .ok_or_else(|| invalid("local nonzero count overflows BigInt"))?;
    ffi::HYPRE_Int::try_from(nonzeros)
        .map_err(|_| invalid("local nonzero count exceeds HYPRE_Int range"))?;
    Ok(rows)
}
