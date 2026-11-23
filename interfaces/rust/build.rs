use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Native {
    library: PathBuf,
    link_dir: PathBuf,
    mpi_compiler: PathBuf,
    includes: Vec<String>,
    definitions: Vec<String>,
}

fn read_metadata(path: &Path) -> Native {
    println!("cargo:rerun-if-changed={}", path.display());
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Cannot read native metadata {}: {error}", path.display()));
    let mut library = None;
    let mut link_dir = None;
    let mut mpi_compiler = None;
    let mut includes = Vec::new();
    let mut definitions = Vec::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("Invalid native metadata line in {}", path.display()));
        if value.is_empty() {
            continue;
        }
        match key {
            "library" => library = Some(PathBuf::from(value)),
            "link_dir" => link_dir = Some(PathBuf::from(value)),
            "mpi_compiler" => mpi_compiler = Some(PathBuf::from(value)),
            "include" => includes.push(value.to_owned()),
            "define" => definitions.push(value.to_owned()),
            "config" => println!("cargo:rerun-if-changed={value}"),
            _ => panic!("Unknown native metadata key {key}"),
        }
    }
    Native {
        library: library.expect("Native metadata must contain library"),
        link_dir: link_dir.expect("Native metadata must contain link_dir"),
        mpi_compiler: mpi_compiler.expect("Native metadata must contain mpi_compiler"),
        includes,
        definitions,
    }
}

fn discover(manifest: &Path, out_dir: &Path) -> PathBuf {
    let build_dir = out_dir.join("native-discovery");
    // Start from a clean discovery directory so changed generators, toolchains,
    // compilers, or package prefixes cannot collide with a cached CMakeCache.
    if build_dir.exists() {
        let _ = fs::remove_dir_all(&build_dir);
    }
    let mut command = Command::new(env::var_os("CMAKE").unwrap_or_else(|| "cmake".into()));
    command
        .arg("-S")
        .arg(manifest)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // The crate version is the single source of the package compatibility
        // request. Installed 0.x packages are compatible within a minor series.
        .arg(format!(
            "-DHYPREDRV_RUST_REQUIRED_VERSION={}.{}",
            env::var("CARGO_PKG_VERSION_MAJOR").unwrap(),
            env::var("CARGO_PKG_VERSION_MINOR").unwrap()
        ));
    if let Some(package_dir) = env::var_os("HYPREDRV_DIR").filter(|value| !value.is_empty()) {
        command.arg(format!(
            "-DHYPREDRV_DIR={}",
            PathBuf::from(package_dir).display()
        ));
    }
    if let Some(compiler) = env::var_os("MPICC").filter(|value| !value.is_empty()) {
        command.arg(format!(
            "-DMPI_C_COMPILER={}",
            PathBuf::from(compiler).display()
        ));
    }
    let output = command
        .output()
        .expect("Cannot execute CMake to find shared libHYPREDRV");
    if !output.status.success() {
        panic!(
            "CMake could not discover shared libHYPREDRV. Set HYPREDRV_DIR to its CMake \
             package directory or CMAKE_PREFIX_PATH to its installation prefix.\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    build_dir.join("native-Release.txt")
}

fn executable_path(command: &Path) -> PathBuf {
    if command.components().count() > 1 {
        return fs::canonicalize(command).unwrap_or_else(|error| {
            panic!("Cannot resolve MPI compiler {}: {error}", command.display())
        });
    }
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(command))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| fs::canonicalize(candidate).ok())
        .unwrap_or_else(|| panic!("Cannot find MPI compiler {} on PATH", command.display()))
}

fn main() {
    for name in [
        "HYPREDRV_RUST_METADATA",
        "HYPREDRV_DIR",
        "HYPRE_DIR",
        "HYPRE_ROOT",
        "CMAKE_PREFIX_PATH",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_GENERATOR",
        "CMAKE",
        "CC",
        "MPICC",
        "MPI_PKG_CONFIG",
        "CRAY_MPICH_DIR",
        "PATH",
        "LIBCLANG_PATH",
        "BINDGEN_EXTRA_CLANG_ARGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("CMakeLists.txt").display()
    );
    let metadata = env::var_os("HYPREDRV_RUST_METADATA")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| discover(&manifest, &out_dir));
    let native = read_metadata(&metadata);

    // rsmpi consults MPI_PKG_CONFIG, then CRAY_MPICH_DIR, before MPICC, so
    // MPICC selects MPI only when neither is set. The generated ABI checks are
    // separate and validate whatever MPI rsmpi ends up using, because Cargo can
    // reuse mpi-sys bindings previously generated with another MPI.
    let rsmpi_uses_mpicc =
        env::var_os("MPI_PKG_CONFIG").is_none() && env::var_os("CRAY_MPICH_DIR").is_none();
    let selected_mpicc = env::var_os("MPICC")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "mpicc".into());
    if rsmpi_uses_mpicc
        && executable_path(Path::new(&selected_mpicc)) != executable_path(&native.mpi_compiler)
    {
        panic!(
            "MPI compiler mismatch. Set MPICC={} before running Cargo so rsmpi and \
             hypredrive use the same MPI installation.",
            native.mpi_compiler.display()
        );
    }
    if !native.library.is_file() {
        panic!(
            "Shared native library {} is missing. Build or install hypredrive first.",
            native.library.display()
        );
    }
    println!("cargo:rerun-if-changed={}", native.library.display());
    println!(
        "cargo:rustc-link-search=native={}",
        native.link_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=HYPREDRV");

    let functions = [
        "Initialize",
        "Finalize",
        "Create",
        "Destroy",
        "SetLibraryMode",
        "AnnotateBegin",
        "AnnotateEnd",
        "InputArgsParse",
        "ErrorCodeDescribe",
        "LinearSystemSetMatrixFromCSR",
        "LinearSystemSetRHSFromArray",
        "LinearSystemSetInitialGuess",
        "LinearSystemResetInitialGuess",
        "LinearSolverCreate",
        "LinearSolverSetup",
        "LinearSolverApply",
        "LinearSolverDestroy",
        "LinearSystemGetSolutionLength",
        "LinearSystemCopySolutionValues",
        "LinearSolverGetNumIter",
        "LinearSolverGetConverged",
        "LinearSolverGetFinalRelativeResidualNorm",
        "LinearSolverGetSetupTime",
        "LinearSolverGetSolveTime",
    ];
    let mut bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_str().unwrap())
        .clang_arg("-std=c11")
        .allowlist_function(format!("HYPREDRV_({})", functions.join("|")))
        .allowlist_type("HYPREDRV_t")
        .allowlist_type("HYPRE_(BigInt|Int|Real|Complex)")
        .allowlist_var("HYPREDRV_RUST_MPI_.*")
        .blocklist_type("MPI_Comm")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    for include in native.includes {
        bindings = bindings.clang_arg(format!("-I{include}"));
    }
    for definition in native.definitions {
        bindings = bindings.clang_arg(format!("-D{definition}"));
    }
    bindings
        .generate()
        .expect("Cannot generate HYPREDRV bindings. Check libclang and the matching native headers/configuration.")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Cannot write generated bindings to Cargo OUT_DIR");
}
