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

/// A raw HYPRE matrix handle (`HYPRE_Matrix`, an assembled ParCSR `HYPRE_IJMatrix`).
pub type Matrix = ffi::HYPRE_Matrix;

/// A raw HYPRE vector handle (`HYPRE_Vector`, an initialized ParCSR `HYPRE_IJVector`).
pub type Vector = ffi::HYPRE_Vector;

/// A raw `HYPRE_IJVector` handle, as registered by [`HypreDrive::set_state_vectors`].
pub type IJVector = ffi::HYPRE_IJVector;

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

impl Runtime<'_> {
    /// Clear the process-wide native error state recorded by earlier failures.
    pub fn clear_error_codes(&self) {
        unsafe { ffi::HYPREDRV_ErrorCodeClear() };
    }

    /// Build the native "invalid value" status carrying `message` and return it.
    ///
    /// The returned `u32` can be passed to application error reporting. A
    /// message containing an embedded NUL is rejected.
    pub fn error_invalid_value(&self, message: &str) -> Result<u32, Error> {
        let message = CString::new(message)
            .map_err(|_| Error::invalid("error_invalid_value", "message contains an embedded NUL"))?;
        Ok(unsafe { ffi::HYPREDRV_ErrorInvalidValue(message.as_ptr()) })
    }

    /// Print hypredrive and HYPRE version information on `comm`.
    ///
    /// Collective on `comm`; `print_datetime` adds a date/time header.
    pub fn print_lib_info<C: Communicator>(
        &self,
        comm: &C,
        print_datetime: bool,
    ) -> Result<(), Error> {
        require_intracommunicator(comm, "print_lib_info")?;
        let status =
            unsafe { ffi::HYPREDRV_PrintLibInfo(comm.as_raw(), i32::from(print_datetime)) };
        check_status(comm, "HYPREDRV_PrintLibInfo", status)
    }

    /// Print information about the host system on `comm`. Collective on `comm`.
    pub fn print_system_info<C: Communicator>(&self, comm: &C) -> Result<(), Error> {
        require_intracommunicator(comm, "print_system_info")?;
        let status = unsafe { ffi::HYPREDRV_PrintSystemInfo(comm.as_raw()) };
        check_status(comm, "HYPREDRV_PrintSystemInfo", status)
    }

    /// Print the closing banner naming the application `argv0`. Collective on `comm`.
    pub fn print_exit_info<C: Communicator>(&self, comm: &C, argv0: &str) -> Result<(), Error> {
        require_intracommunicator(comm, "print_exit_info")?;
        let name = CString::new(argv0)
            .map_err(|_| Error::invalid("print_exit_info", "name contains an embedded NUL"));
        let name = agree_validation(comm, "print_exit_info", name)?;
        let status = unsafe { ffi::HYPREDRV_PrintExitInfo(comm.as_raw(), name.as_ptr()) };
        check_status(comm, "HYPREDRV_PrintExitInfo", status)
    }

    /// Register a named solver preset from a YAML snippet expanding under `solver:`.
    ///
    /// The registry is process-global and the native call copies its strings.
    /// Register the same presets on every rank before selecting them with
    /// [`HypreDrive::set_solver_preset`]. `help` is an optional description.
    pub fn register_solver_preset(
        &self,
        name: &str,
        yaml_text: &str,
        help: Option<&str>,
    ) -> Result<(), Error> {
        register_preset("register_solver_preset", name, yaml_text, help, false)
    }

    /// Register a named preconditioner preset from a YAML snippet expanding
    /// under `preconditioner:`. See [`Self::register_solver_preset`].
    pub fn register_precon_preset(
        &self,
        name: &str,
        yaml_text: &str,
        help: Option<&str>,
    ) -> Result<(), Error> {
        register_preset("register_precon_preset", name, yaml_text, help, true)
    }
}

fn register_preset(
    operation: &'static str,
    name: &str,
    yaml_text: &str,
    help: Option<&str>,
    precon: bool,
) -> Result<(), Error> {
    let nul = |_| Error::invalid(operation, "argument contains an embedded NUL");
    let name = CString::new(name).map_err(nul)?;
    let yaml_text = CString::new(yaml_text).map_err(nul)?;
    let help = help.map(CString::new).transpose().map_err(nul)?;
    let help_ptr = help.as_ref().map_or(ptr::null(), |text| text.as_ptr());
    let (label, status) = if precon {
        (
            "HYPREDRV_PreconPresetRegister",
            unsafe {
                ffi::HYPREDRV_PreconPresetRegister(name.as_ptr(), yaml_text.as_ptr(), help_ptr)
            },
        )
    } else {
        (
            "HYPREDRV_SolverPresetRegister",
            unsafe {
                ffi::HYPREDRV_SolverPresetRegister(name.as_ptr(), yaml_text.as_ptr(), help_ptr)
            },
        )
    };
    describe_error(status);
    if status == 0 {
        Ok(())
    } else {
        Err(Error::native(label, status))
    }
}

/// Statistics recorded for one hierarchical annotation entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatsEntry {
    /// One-based entry identifier.
    pub entry_id: i32,
    /// Number of solves in the entry.
    pub num_solves: i32,
    /// Total linear iterations.
    pub linear_iters: i32,
    /// Total setup time in the configured statistics units.
    pub setup_time: f64,
    /// Total solve time in the configured statistics units.
    pub solve_time: f64,
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
    external_matrix: bool,
    external_rhs: bool,
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
        require_identical_input(
            communicator,
            "from_yaml",
            "YAML text",
            configuration.as_bytes(),
        )?;
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
        Self::from_file_with_overrides(runtime, communicator, path, &[])
    }

    /// Read a YAML file and apply native `--path:to:key value` overrides.
    ///
    /// Pass alternating keys and values, with identical tokens on every rank.
    /// File handling is the same as in [`Self::from_file`].
    pub fn from_file_with_overrides(
        runtime: &'a Runtime<'a>,
        communicator: &'a C,
        path: impl AsRef<Path>,
        overrides: &[&str],
    ) -> Result<Self, Error> {
        require_intracommunicator(communicator, "from_file")?;
        let arguments = (|| {
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
            let configuration = CString::new(name)
                .map_err(|_| Error::invalid("from_file", "path contains an embedded NUL"))?;
            if overrides.len() % 2 != 0
                || overrides.chunks_exact(2).any(|pair| {
                    let key = pair[0];
                    !key.starts_with("--") && !key.contains(':')
                })
            {
                return Err(Error::invalid(
                    "from_file",
                    "expected pairs of '--path:to:key value'",
                ));
            }
            if i32::try_from(overrides.len() + 3).is_err() {
                return Err(Error::invalid("from_file", "too many YAML overrides"));
            }
            let overrides = overrides
                .iter()
                .map(|arg| {
                    CString::new(*arg).map_err(|_| {
                        Error::invalid("from_file", "YAML override contains an embedded NUL")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((configuration, overrides))
        })();
        let (configuration, overrides) = agree_validation(communicator, "from_file", arguments)?;
        let override_bytes: Vec<_> = overrides
            .iter()
            .flat_map(|arg| arg.as_bytes_with_nul().iter().copied())
            .collect();
        require_identical_input(communicator, "from_file", "YAML overrides", &override_bytes)?;
        // A separate argv[0] keeps unusual filename characters out of the
        // native parser's inline-YAML detection.
        let mut arguments = vec![c"hypredrive-rust".to_owned(), configuration];
        if !overrides.is_empty() {
            arguments.push(c"-a".to_owned());
            arguments.extend(overrides);
        }
        Self::configured(runtime, communicator, arguments)
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
            external_matrix: false,
            external_rhs: false,
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
        self.external_matrix = false;
        self.external_rhs = false;
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

    /// Set the Krylov relative tolerance used by subsequent solves.
    ///
    /// Overrides the tolerance from the YAML configuration. Every rank must
    /// supply the same `tol` in `[0, 1)`.
    pub fn set_relative_tol(&mut self, tol: f64) -> Result<(), Error> {
        let operation = "set_relative_tol";
        let mut root_tol = tol;
        self.communicator
            .process_at_rank(0)
            .broadcast_into(&mut root_tol);
        let validation = if !(0.0..1.0).contains(&tol) {
            Err(Error::invalid(operation, "tolerance must be in [0, 1)"))
        } else if tol != root_tol {
            Err(Error::invalid(operation, "tolerance must match on every rank"))
        } else {
            Ok(())
        };
        self.agreed(operation, validation)?;
        let status = unsafe { ffi::HYPREDRV_LinearSolverSetRelativeTol(self.handle, tol) };
        self.check("HYPREDRV_LinearSolverSetRelativeTol", status)
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


    /// Set or clear (`None`) the display name used in statistics banners.
    pub fn set_object_name(&mut self, name: Option<&str>) -> Result<(), Error> {
        let name = match name {
            Some(text) => Some(self.agreed_cstring("set_object_name", text)?),
            None => None,
        };
        let name_ptr = name.as_ref().map_or(ptr::null(), |text| text.as_ptr());
        let status = unsafe { ffi::HYPREDRV_ObjectSetName(self.handle, name_ptr) };
        self.check("HYPREDRV_ObjectSetName", status)
    }

    /// The configured number of warmup runs.
    pub fn warmup(&self) -> Result<i32, Error> {
        let mut value = 0;
        let status = unsafe { ffi::HYPREDRV_InputArgsGetWarmup(self.handle, &mut value) };
        self.check("HYPREDRV_InputArgsGetWarmup", status)?;
        Ok(value)
    }

    /// The configured number of solve repetitions.
    pub fn num_repetitions(&self) -> Result<i32, Error> {
        let mut value = 0;
        let status = unsafe { ffi::HYPREDRV_InputArgsGetNumRepetitions(self.handle, &mut value) };
        self.check("HYPREDRV_InputArgsGetNumRepetitions", status)?;
        Ok(value)
    }

    /// The configured number of linear systems.
    pub fn num_linear_systems(&self) -> Result<i32, Error> {
        let mut value = 0;
        let status = unsafe { ffi::HYPREDRV_InputArgsGetNumLinearSystems(self.handle, &mut value) };
        self.check("HYPREDRV_InputArgsGetNumLinearSystems", status)?;
        Ok(value)
    }

    /// The number of configured preconditioner variants.
    pub fn num_precon_variants(&self) -> Result<i32, Error> {
        let mut value = 0;
        let status =
            unsafe { ffi::HYPREDRV_InputArgsGetNumPreconVariants(self.handle, &mut value) };
        self.check("HYPREDRV_InputArgsGetNumPreconVariants", status)?;
        Ok(value)
    }

    /// Activate the zero-based preconditioner variant `variant`.
    pub fn set_precon_variant(&mut self, variant: i32) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_InputArgsSetPreconVariant(self.handle, variant) };
        self.check("HYPREDRV_InputArgsSetPreconVariant", status)
    }

    /// Select a built-in or registered preconditioner preset for the active variant.
    ///
    /// A successful change releases the native solver and preconditioner;
    /// [`Self::solve`] recreates them.
    pub fn set_precon_preset(&mut self, preset: &str) -> Result<(), Error> {
        let preset = self.agreed_cstring("set_precon_preset", preset)?;
        let status =
            unsafe { ffi::HYPREDRV_InputArgsSetPreconPreset(self.handle, preset.as_ptr()) };
        self.check("HYPREDRV_InputArgsSetPreconPreset", status)
    }

    /// Select a built-in (for example `"pcg"`) or registered solver preset.
    ///
    /// A successful change releases the native solver; [`Self::solve`] recreates it.
    pub fn set_solver_preset(&mut self, preset: &str) -> Result<(), Error> {
        let preset = self.agreed_cstring("set_solver_preset", preset)?;
        let status =
            unsafe { ffi::HYPREDRV_InputArgsSetSolverPreset(self.handle, preset.as_ptr()) };
        self.check("HYPREDRV_InputArgsSetSolverPreset", status)
    }

    /// Build the matrix, RHS and guess from the YAML `linear_system` section.
    ///
    /// The system is then usable by [`Self::solve`] without the CSR setters.
    pub fn build_linear_system(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemBuild(self.handle) };
        self.check("HYPREDRV_LinearSystemBuild", status)?;
        self.rows = None;
        self.external_matrix = true;
        self.external_rhs = true;
        self.guess_ready = false;
        self.reset_pending = false;
        self.annotate(c"matrix")
    }

    /// Read only the matrix from the files configured in YAML.
    ///
    /// Installs a matrix for [`Self::solve`]. Afterwards call
    /// [`Self::set_rhs_vector`] with a null handle to construct the RHS from the YAML input.
    pub fn read_matrix(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemReadMatrix(self.handle) };
        self.check("HYPREDRV_LinearSystemReadMatrix", status)?;
        self.rows = None;
        self.external_matrix = true;
        self.rhs_ready = false;
        self.external_rhs = false;
        self.guess_ready = false;
        self.reset_pending = false;
        self.annotate(c"matrix")
    }

    /// Attach a raw assembled ParCSR matrix as the system (and default preconditioning) matrix.
    ///
    /// A null handle clears both attachments. The matrix is borrowed: the
    /// driver runs in library mode and never destroys it.
    ///
    /// # Safety
    ///
    /// A non-null `matrix` must be a valid, assembled `HYPRE_IJMatrix` of
    /// ParCSR type, created on a communicator congruent with this driver's,
    /// and it must outlive this driver or be replaced/cleared before it is
    /// destroyed. Call collectively with consistent matrices on all ranks.
    pub unsafe fn set_matrix(&mut self, matrix: Matrix) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetMatrix(self.handle, matrix) };
        self.check("HYPREDRV_LinearSystemSetMatrix", status)?;
        self.rows = None;
        self.external_matrix = !matrix.is_null();
        self.guess_ready = false;
        self.reset_pending = false;
        self.annotate(c"matrix")
    }

    /// Set the matrix used to build the preconditioner.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::set_matrix`] for the handle's validity and lifetime.
    pub unsafe fn set_precon_matrix(&mut self, matrix: Matrix) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetPrecMatrix(self.handle, matrix) };
        self.check("HYPREDRV_LinearSystemSetPrecMatrix", status)
    }

    /// Set the discrete gradient for AMS/ADS preconditioners (null clears it).
    ///
    /// # Safety
    ///
    /// A non-null handle must be an assembled ParCSR `HYPRE_IJMatrix` that
    /// outlives this driver (borrowed in library mode). Call collectively.
    pub unsafe fn set_discrete_gradient(&mut self, gradient: Matrix) -> Result<(), Error> {
        let status =
            unsafe { ffi::HYPREDRV_LinearSystemSetDiscreteGradient(self.handle, gradient) };
        self.check("HYPREDRV_LinearSystemSetDiscreteGradient", status)
    }

    /// Set the discrete curl for the ADS preconditioner (null clears it).
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::set_discrete_gradient`].
    pub unsafe fn set_discrete_curl(&mut self, curl: Matrix) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetDiscreteCurl(self.handle, curl) };
        self.check("HYPREDRV_LinearSystemSetDiscreteCurl", status)
    }

    /// Set the vertex coordinate vectors for AMS/ADS (null clears one).
    ///
    /// # Safety
    ///
    /// Non-null handles must be initialized, single-component ParCSR
    /// `HYPRE_IJVector`s with matching row distributions that outlive this
    /// driver (borrowed in library mode). Call collectively.
    pub unsafe fn set_coordinates(&mut self, x: Vector, y: Vector, z: Vector) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetCoordinates(self.handle, x, y, z) };
        self.check("HYPREDRV_LinearSystemSetCoordinates", status)
    }

    /// Provide a user coarse Schur-complement matrix for MGR level `level` (null clears it).
    ///
    /// # Safety
    ///
    /// A non-null handle must be an assembled ParCSR `HYPRE_IJMatrix` sized
    /// for that level's coarse DOF set and must outlive this driver. All ranks
    /// must register the same level.
    pub unsafe fn set_coarse_schur(&mut self, level: i32, matrix: Matrix) -> Result<(), Error> {
        let status =
            unsafe { ffi::HYPREDRV_LinearSystemSetCoarseSchur(self.handle, level, matrix) };
        self.check("HYPREDRV_LinearSystemSetCoarseSchur", status)
    }

    /// Attach a raw RHS vector (null constructs one from the YAML input).
    ///
    /// # Safety
    ///
    /// A non-null handle must be an initialized, single-component ParCSR
    /// `HYPRE_IJVector` matching the matrix rows that outlives this driver.
    pub unsafe fn set_rhs_vector(&mut self, rhs: Vector) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetRHS(self.handle, rhs) };
        self.check("HYPREDRV_LinearSystemSetRHS", status)?;
        self.external_rhs = true;
        Ok(())
    }

    /// Attach a raw solution vector (null lets the driver allocate one from the RHS).
    ///
    /// # Safety
    ///
    /// A non-null handle must be an initialized ParCSR `HYPRE_IJVector` with the
    /// matrix's row distribution that outlives this driver.
    pub unsafe fn set_solution_vector(&mut self, solution: Vector) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetSolution(self.handle, solution) };
        self.check("HYPREDRV_LinearSystemSetSolution", status)
    }

    /// Attach a raw reference solution used for error reporting (null clears it).
    ///
    /// # Safety
    ///
    /// Same requirements as [`Self::set_solution_vector`].
    pub unsafe fn set_reference_solution(&mut self, reference: Vector) -> Result<(), Error> {
        let status =
            unsafe { ffi::HYPREDRV_LinearSystemSetReferenceSolution(self.handle, reference) };
        self.check("HYPREDRV_LinearSystemSetReferenceSolution", status)
    }

    /// Borrow the driver's system matrix handle.
    ///
    /// # Safety
    ///
    /// The returned handle is owned by the driver. Do not destroy it. It is
    /// invalidated by operations that replace the matrix and by dropping the driver.
    pub unsafe fn matrix(&self) -> Result<Matrix, Error> {
        let mut matrix: Matrix = ptr::null_mut();
        let status = unsafe { ffi::HYPREDRV_LinearSystemGetMatrix(self.handle, &mut matrix) };
        self.check("HYPREDRV_LinearSystemGetMatrix", status)?;
        Ok(matrix)
    }

    /// Borrow the driver's RHS vector handle.
    ///
    /// # Safety
    ///
    /// The handle is owned by the driver; do not destroy it, and do not use it
    /// after the RHS is replaced or the driver is dropped.
    pub unsafe fn rhs_vector(&self) -> Result<Vector, Error> {
        let mut vector: Vector = ptr::null_mut();
        let status = unsafe { ffi::HYPREDRV_LinearSystemGetRHS(self.handle, &mut vector) };
        self.check("HYPREDRV_LinearSystemGetRHS", status)?;
        Ok(vector)
    }

    /// Borrow the driver's solution vector handle.
    ///
    /// # Safety
    ///
    /// The handle is owned by the driver (unless supplied through
    /// [`Self::set_solution_vector`]); do not destroy it, and do not use it
    /// after the solution is replaced or the driver is dropped.
    pub unsafe fn solution_vector(&self) -> Result<Vector, Error> {
        let mut vector: Vector = ptr::null_mut();
        let status = unsafe { ffi::HYPREDRV_LinearSystemGetSolution(self.handle, &mut vector) };
        self.check("HYPREDRV_LinearSystemGetSolution", status)?;
        Ok(vector)
    }

    /// Borrow the native local RHS data pointer.
    ///
    /// # Safety
    ///
    /// The pointer addresses this rank's local entries, possibly in device
    /// memory. It is owned by the driver and must not be freed. It is
    /// invalidated by replacing the RHS or dropping the driver.
    pub unsafe fn rhs_values_ptr(&self) -> Result<*mut f64, Error> {
        let mut data: *mut f64 = ptr::null_mut();
        let status = unsafe { ffi::HYPREDRV_LinearSystemGetRHSValues(self.handle, &mut data) };
        self.check("HYPREDRV_LinearSystemGetRHSValues", status)?;
        Ok(data)
    }

    /// Borrow the native local solution data pointer and whether it is device memory.
    ///
    /// # Safety
    ///
    /// The pointer is owned by the driver, must not be freed, and is
    /// invalidated by solver setup/apply or replacing the solution. A pointer
    /// reported as device memory must not be dereferenced on the host. Use
    /// [`Self::solve`] results for host copies.
    pub unsafe fn solution_values_ptr(&self) -> Result<(*mut f64, bool), Error> {
        let mut data: *mut f64 = ptr::null_mut();
        // HYPRE_MEMORY_HOST is zero in every bindgen enum representation.
        let mut location: ffi::HYPRE_MemoryLocation = unsafe { std::mem::zeroed() };
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemGetSolutionValues(self.handle, &mut data, &mut location)
        };
        self.check("HYPREDRV_LinearSystemGetSolutionValues", status)?;
        // HYPRE_MEMORY_DEVICE is 1; HOST is 0 and UNDEFINED is -1.
        Ok((data, (location as i32) == 1))
    }

    /// Compute the `"L1"`, `"L2"` or `"inf"` norm of the current solution.
    pub fn solution_norm(&self, norm_type: &str) -> Result<f64, Error> {
        let norm_type = self.agreed_cstring("solution_norm", norm_type)?;
        let mut norm = 0.0;
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemGetSolutionNorm(self.handle, norm_type.as_ptr(), &mut norm)
        };
        self.check("HYPREDRV_LinearSystemGetSolutionNorm", status)?;
        Ok(norm)
    }

    /// Install a local DOF map; entries label each local row's DOF type.
    ///
    /// The array is copied. An empty slice is allowed on ranks without rows.
    pub fn set_dofmap(&mut self, dofmap: &[i32]) -> Result<(), Error> {
        let size = self.agreed_int("set_dofmap", dofmap.len())?;
        let data = if dofmap.is_empty() { ptr::null() } else { dofmap.as_ptr() };
        let status = unsafe { ffi::HYPREDRV_LinearSystemSetDofmap(self.handle, size, data) };
        self.check("HYPREDRV_LinearSystemSetDofmap", status)
    }

    /// Install an interleaved DOF map (`u0 v0 p0 u1 v1 p1 ...`).
    pub fn set_interleaved_dofmap(
        &mut self,
        num_local_blocks: i32,
        num_dof_types: i32,
    ) -> Result<(), Error> {
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetInterleavedDofmap(
                self.handle,
                num_local_blocks,
                num_dof_types,
            )
        };
        self.check("HYPREDRV_LinearSystemSetInterleavedDofmap", status)
    }

    /// Install a contiguous DOF map (all `u`, then all `v`, then all `p`).
    pub fn set_contiguous_dofmap(
        &mut self,
        num_local_blocks: i32,
        num_dof_types: i32,
    ) -> Result<(), Error> {
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetContiguousDofmap(
                self.handle,
                num_local_blocks,
                num_dof_types,
            )
        };
        self.check("HYPREDRV_LinearSystemSetContiguousDofmap", status)
    }

    /// Read the DOF map from the source configured in YAML.
    pub fn read_dofmap(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemReadDofmap(self.handle) };
        self.check("HYPREDRV_LinearSystemReadDofmap", status)
    }

    /// Write the local DOF map to `filename` (suffixed with the rank when run on several ranks).
    pub fn print_dofmap(&self, filename: &str) -> Result<(), Error> {
        let filename = self.agreed_cstring("print_dofmap", filename)?;
        let status =
            unsafe { ffi::HYPREDRV_LinearSystemPrintDofmap(self.handle, filename.as_ptr()) };
        self.check("HYPREDRV_LinearSystemPrintDofmap", status)
    }

    /// Write the matrix, RHS and DOF map (when present) to ASCII files.
    pub fn print_linear_system(&self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemPrint(self.handle) };
        self.check("HYPREDRV_LinearSystemPrint", status)
    }

    /// Install near-null-space modes (for example rigid-body modes).
    ///
    /// `values` holds `num_components` contiguous blocks of `num_entries`
    /// values (component-major) and is copied. `num_entries` must match the
    /// local matrix column range and `num_components` must agree on all ranks.
    pub fn set_near_null_space(
        &mut self,
        num_entries: usize,
        num_components: usize,
        values: &[f64],
    ) -> Result<(), Error> {
        let (entries, components, data) =
            self.agreed_modes("set_near_null_space", num_entries, num_components, values)?;
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetNearNullSpace(self.handle, entries, components, data)
        };
        self.check("HYPREDRV_LinearSystemSetNearNullSpace", status)
    }

    /// Install exact null-space modes projected out of every solution.
    ///
    /// Layout and validation match [`Self::set_near_null_space`]. Passing
    /// `num_components == 0` with empty `values` clears the modes.
    pub fn set_null_space(
        &mut self,
        num_entries: usize,
        num_components: usize,
        values: &[f64],
    ) -> Result<(), Error> {
        let (entries, components, data) =
            self.agreed_modes("set_null_space", num_entries, num_components, values)?;
        let status = unsafe {
            ffi::HYPREDRV_LinearSystemSetNullSpace(self.handle, entries, components, data)
        };
        self.check("HYPREDRV_LinearSystemSetNullSpace", status)
    }

    fn agreed_modes(
        &self,
        operation: &'static str,
        num_entries: usize,
        num_components: usize,
        values: &[f64],
    ) -> Result<(i32, i32, *const f64), Error> {
        let checked = (|| {
            let entries = i32::try_from(num_entries)
                .map_err(|_| Error::invalid(operation, "entry count exceeds the native int range"))?;
            let components = i32::try_from(num_components).map_err(|_| {
                Error::invalid(operation, "component count exceeds the native int range")
            })?;
            let expected = num_entries
                .checked_mul(num_components)
                .ok_or_else(|| Error::invalid(operation, "mode size overflows usize"))?;
            if values.len() != expected {
                return Err(Error::invalid(
                    operation,
                    "values length must equal num_entries * num_components",
                ));
            }
            Ok((entries, components))
        })();
        let (entries, components) = self.agreed(operation, checked)?;
        let data = if values.is_empty() { ptr::null() } else { values.as_ptr() };
        Ok((entries, components, data))
    }

    /// Register the state vectors managed by the `state_vector_*` methods.
    ///
    /// The handle array is copied. HYPREDRV never destroys the vectors.
    ///
    /// # Safety
    ///
    /// Every handle must be an initialized `HYPRE_IJVector` with the linear
    /// system's row range, valid until after this driver is dropped.
    pub unsafe fn set_state_vectors(&mut self, vectors: &[IJVector]) -> Result<(), Error> {
        let count = self.agreed_int("set_state_vectors", vectors.len())?;
        let status = unsafe {
            ffi::HYPREDRV_StateVectorSet(self.handle, count, vectors.as_ptr().cast_mut())
        };
        self.check("HYPREDRV_StateVectorSet", status)
    }

    /// Borrow the local data pointer of the state vector at logical `index`.
    ///
    /// # Safety
    ///
    /// The pointer is owned by the vector, addresses this rank's local entries
    /// (possibly device memory), and is invalidated by operations that
    /// reallocate the vector or by destroying it.
    pub unsafe fn state_vector_values_ptr(&self, index: i32) -> Result<*mut f64, Error> {
        let mut data: *mut f64 = ptr::null_mut();
        let status = unsafe { ffi::HYPREDRV_StateVectorGetValues(self.handle, index, &mut data) };
        self.check("HYPREDRV_StateVectorGetValues", status)?;
        Ok(data)
    }

    /// Copy state vector `index_in` into `index_out`.
    pub fn state_vector_copy(&mut self, index_in: i32, index_out: i32) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_StateVectorCopy(self.handle, index_in, index_out) };
        self.check("HYPREDRV_StateVectorCopy", status)
    }

    /// Rotate the logical state indices by one position without copying data.
    pub fn state_vector_update_all(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_StateVectorUpdateAll(self.handle) };
        self.check("HYPREDRV_StateVectorUpdateAll", status)
    }

    /// Add the last linear solver solution to the state vector at `state_idx`.
    pub fn state_vector_apply_correction(&mut self, state_idx: i32) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_StateVectorApplyCorrection(self.handle, state_idx) };
        self.check("HYPREDRV_StateVectorApplyCorrection", status)
    }

    /// Create the preconditioner without creating the Krylov solver.
    pub fn precon_create(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_PreconCreate(self.handle) };
        self.check("HYPREDRV_PreconCreate", status)
    }

    /// Set up only the preconditioner, for standalone use with [`Self::precon_apply`].
    pub fn precon_setup(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_PreconSetup(self.handle) };
        self.check("HYPREDRV_PreconSetup", status)
    }

    /// Apply the preconditioner to `b`, writing the result to `x`.
    ///
    /// # Safety
    ///
    /// Both handles must be initialized ParCSR `HYPRE_IJVector`s compatible
    /// with the matrix. Wrap standalone applications in a `"solve"` annotation
    /// when statistics are enabled.
    pub unsafe fn precon_apply(&mut self, b: Vector, x: Vector) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_PreconApply(self.handle, b, x) };
        self.check("HYPREDRV_PreconApply", status)
    }

    /// Destroy the preconditioner (the solver also destroys it).
    pub fn precon_destroy(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_PreconDestroy(self.handle) };
        self.check("HYPREDRV_PreconDestroy", status)
    }

    /// Print the native statistics summary.
    pub fn print_stats(&self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_StatsPrint(self.handle) };
        self.check("HYPREDRV_StatsPrint", status)
    }

    /// Begin a hierarchical annotation region at `level` (0 to 9).
    ///
    /// A nonnegative `id` is appended to the name; a negative one is ignored.
    /// End it with [`Self::annotate_level_end`] using the same arguments.
    pub fn annotate_level_begin(&mut self, level: i32, name: &str, id: i32) -> Result<(), Error> {
        let name = self.agreed_cstring("annotate_level_begin", name)?;
        let status = unsafe {
            ffi::HYPREDRV_AnnotateLevelBegin(self.handle, level, name.as_ptr(), id)
        };
        self.check("HYPREDRV_AnnotateLevelBegin", status)
    }

    /// End the hierarchical annotation region begun with the same arguments.
    pub fn annotate_level_end(&mut self, level: i32, name: &str, id: i32) -> Result<(), Error> {
        let name = self.agreed_cstring("annotate_level_end", name)?;
        let status =
            unsafe { ffi::HYPREDRV_AnnotateLevelEnd(self.handle, level, name.as_ptr(), id) };
        self.check("HYPREDRV_AnnotateLevelEnd", status)
    }

    /// Compute the eigenspectrum of the system (single rank; no-op without native support).
    pub fn compute_eigenspectrum(&mut self) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_LinearSystemComputeEigenspectrum(self.handle) };
        self.check("HYPREDRV_LinearSystemComputeEigenspectrum", status)
    }

    /// Number of entries recorded at annotation `level`.
    pub fn stats_level_count(&self, level: i32) -> Result<i32, Error> {
        let mut count = 0;
        let status = unsafe { ffi::HYPREDRV_StatsLevelGetCount(self.handle, level, &mut count) };
        self.check("HYPREDRV_StatsLevelGetCount", status)?;
        Ok(count)
    }

    /// Statistics of the zero-based entry `index` at annotation `level`.
    pub fn stats_level_entry(&self, level: i32, index: i32) -> Result<StatsEntry, Error> {
        let mut entry = StatsEntry {
            entry_id: 0,
            num_solves: 0,
            linear_iters: 0,
            setup_time: 0.0,
            solve_time: 0.0,
        };
        let status = unsafe {
            ffi::HYPREDRV_StatsLevelGetEntry(
                self.handle,
                level,
                index,
                &mut entry.entry_id,
                &mut entry.num_solves,
                &mut entry.linear_iters,
                &mut entry.setup_time,
                &mut entry.solve_time,
            )
        };
        self.check("HYPREDRV_StatsLevelGetEntry", status)?;
        Ok(entry)
    }

    /// Print the statistics table for annotation `level`.
    pub fn print_stats_level(&self, level: i32) -> Result<(), Error> {
        let status = unsafe { ffi::HYPREDRV_StatsLevelPrint(self.handle, level) };
        self.check("HYPREDRV_StatsLevelPrint", status)
    }

    fn require_system(&self, operation: &'static str) -> Result<(), Error> {
        let validation = if (self.rows.is_some() || self.external_matrix)
            && (self.rhs_ready || self.external_rhs)
        {
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
        check_status(self.communicator, operation, status)
    }

    /// Agree on a locally computed validation result across the communicator.
    fn agreed<T>(&self, operation: &'static str, validation: Result<T, Error>) -> Result<T, Error> {
        agree_validation(self.communicator, operation, validation)
    }

    /// Convert text to a `CString`, agreeing on NUL failures across ranks.
    fn agreed_cstring(&self, operation: &'static str, text: &str) -> Result<CString, Error> {
        let text = CString::new(text)
            .map_err(|_| Error::invalid(operation, "text contains an embedded NUL"));
        self.agreed(operation, text)
    }

    /// Convert a length or count to the native `int`, agreeing across ranks.
    fn agreed_int(&self, operation: &'static str, value: usize) -> Result<i32, Error> {
        let value = i32::try_from(value)
            .map_err(|_| Error::invalid(operation, "count exceeds the native int range"));
        self.agreed(operation, value)
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

fn check_status<C: Communicator>(
    communicator: &C,
    operation: &'static str,
    status: u32,
) -> Result<(), Error> {
    // Preserve diagnostics before any later HYPREDRV call resets its scope.
    describe_error(status);
    let mut collective_status = 0;
    communicator.all_reduce_into(&status, &mut collective_status, SystemOperation::max());
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

fn require_identical_input<C: Communicator>(
    communicator: &C,
    operation: &'static str,
    input: &str,
    bytes: &[u8],
) -> Result<(), Error> {
    let mismatch = || {
        Error::invalid(
            operation,
            format!("{input} must be identical on every MPI rank"),
        )
    };
    let length = agree_validation(
        communicator,
        operation,
        u64::try_from(bytes.len())
            .map_err(|_| Error::invalid(operation, format!("{input} length exceeds u64 range"))),
    )?;
    let mut root_length = length;
    let root = communicator.process_at_rank(0);
    root.broadcast_into(&mut root_length);
    let length_validation = if root_length == length {
        Ok(())
    } else {
        Err(mismatch())
    };
    agree_validation(communicator, operation, length_validation)?;

    // Exact, bounded-size comparisons avoid hash collisions and MPI count
    // overflow for inputs larger than an MPI int-sized buffer.
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
    let validation = if identical { Ok(()) } else { Err(mismatch()) };
    agree_validation(communicator, operation, validation)
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
