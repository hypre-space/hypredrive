// Copyright (c) 2024, Lawrence Livermore National Security, LLC.
// See the top-level LICENSE and NOTICE files for details.

// This executable owns one MPI initialization on its main thread. CTest runs
// the built executable directly, both serially and under mpiexec.
use hypredrive::{BigInt, HypreDrive, Runtime, SolveResult};
use mpi::collective::SystemOperation;
use mpi::environment::{Threading, Universe};
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::*;
use std::error::Error;
use std::io;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const YAML: &str = "general:\n  statistics: off\nsolver:\n  pcg:\n    max_iter: 100\n    relative_tol: 1.0e-12\n    print_level: 0\npreconditioner:\n  amg:\n    print_level: 0\n";

struct Slab {
    first: BigInt,
    last: BigInt,
    indptr: Vec<BigInt>,
    columns: Vec<BigInt>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

fn laplacian(comm: &impl Communicator) -> TestResult<Slab> {
    let size = usize::try_from(comm.size())?;
    let rank = usize::try_from(comm.rank())?;
    let n = 32.max(2 * size);
    let rows = n / size + usize::from(rank < n % size);
    let first = rank * (n / size) + rank.min(n % size);
    let last = first + rows - 1;
    let mut slab = Slab {
        first: BigInt::try_from(first)?,
        last: BigInt::try_from(last)?,
        indptr: vec![0],
        columns: Vec::new(),
        values: Vec::new(),
        rhs: Vec::with_capacity(rows),
    };
    for row in first..=last {
        let mut value = 2.0;
        if row > 0 {
            slab.columns.push(BigInt::try_from(row - 1)?);
            slab.values.push(-1.0);
            value -= 1.0;
        }
        slab.columns.push(BigInt::try_from(row)?);
        slab.values.push(2.0);
        if row + 1 < n {
            slab.columns.push(BigInt::try_from(row + 1)?);
            slab.values.push(-1.0);
            value -= 1.0;
        }
        slab.indptr.push(BigInt::try_from(slab.columns.len())?);
        slab.rhs.push(value);
    }
    Ok(slab)
}

fn check(comm: &impl Communicator, condition: bool, message: &str) -> TestResult {
    // Agree before returning, so a failed assertion cannot strand another rank
    // in the next collective operation.
    let mut all_pass = 0_i32;
    comm.all_reduce_into(&i32::from(condition), &mut all_pass, SystemOperation::min());
    if all_pass == 0 {
        return Err(io::Error::other(message).into());
    }
    Ok(())
}

fn check_solution(
    comm: &impl Communicator,
    result: &SolveResult,
    rows: usize,
    expected: f64,
) -> TestResult {
    check(comm, result.converged, "solver did not converge")?;
    check(
        comm,
        result.x.len() == rows,
        "solution length does not match local rows",
    )?;
    check(
        comm,
        result.x.iter().all(|x| (x - expected).abs() < 1.0e-8),
        "solution differs from the known solution",
    )?;
    check(
        comm,
        result.iterations >= 0
            && result.final_res_norm.is_finite()
            && result.final_res_norm < 1.0e-8
            && result.setup_time.is_finite()
            && result.setup_time >= 0.0
            && result.solve_time.is_finite()
            && result.solve_time >= 0.0,
        "invalid solver statistics",
    )
}

fn serial_minimal(runtime: &Runtime<'_>) -> TestResult {
    let comm = SimpleCommunicator::self_comm();
    let mut driver = HypreDrive::from_yaml(runtime, &comm, YAML)?;
    driver.set_matrix_csr(0, 1, &[0, 2, 4], &[0, 1, 0, 1], &[2.0, -1.0, -1.0, 2.0])?;
    driver.set_rhs(&[1.0, 1.0])?;
    driver.set_initial_guess()?;
    let result = driver.solve()?;
    check_solution(&comm, &result, 2, 1.0)?;
    let residual =
        (2.0 * result.x[0] - result.x[1] - 1.0).hypot(2.0 * result.x[1] - result.x[0] - 1.0);
    check(
        &comm,
        residual < 1.0e-10,
        "two-row system residual is too large",
    )
}

fn initial_guess_and_statistics(runtime: &Runtime<'_>) -> TestResult {
    let comm = SimpleCommunicator::self_comm();
    let yaml = "general:\n  statistics: off\nlinear_system:\n  init_guess_mode: ones\nsolver:\n  gmres:\n    relative_tol: 1.0e-12\npreconditioner: jacobi\n";
    for explicit in [false, true] {
        let mut driver = HypreDrive::from_yaml(runtime, &comm, yaml)?;
        driver.set_matrix_csr(0, 1, &[0, 2, 4], &[0, 1, 0, 1], &[2.0, -1.0, -1.0, 2.0])?;
        driver.set_rhs(&[1.0, 1.0])?;
        if explicit {
            driver.set_initial_guess()?;
            driver.reset_initial_guess()?;
            driver.set_initial_guess()?;
            driver.reset_initial_guess()?;
        }
        let result = driver.solve()?;
        check(
            &comm,
            result.iterations == 0
                && result.converged
                && result.x.iter().all(|x| (x - 1.0).abs() < 1.0e-12),
            "the configured exact initial guess was not applied",
        )?;
    }

    let path = std::env::temp_dir().join(format!(
        "hypredrive-rust-statistics-{}.txt",
        std::process::id()
    ));
    let yaml = format!(
        "general:\n  statistics: on\n  statistics_filename: {}\nsolver:\n  gmres:\n    relative_tol: 1.0e-12\npreconditioner: jacobi\n",
        path.display()
    );
    {
        let mut driver = HypreDrive::from_yaml(runtime, &comm, &yaml)?;
        driver.set_matrix_csr(0, 1, &[0, 2, 4], &[0, 1, 0, 1], &[2.0, -1.0, -1.0, 2.0])?;
        driver.set_rhs(&[1.0, 1.0])?;
        driver.set_initial_guess()?;
        driver.reset_initial_guess()?;
        driver.set_initial_guess()?;
        driver.reset_initial_guess()?;
        for repetition in 0..4 {
            let result = driver.solve()?;
            check(
                &comm,
                result.x.iter().all(|x| (x - 1.0).abs() < 1.0e-12)
                    && result.converged
                    && ((repetition == 0 && result.iterations > 0)
                        || (repetition > 0 && result.iterations == 0)),
                "a repeated solve discarded its warm start",
            )?;
        }
        // Replacing the matrix must start a new native system and invalidate
        // the old vectors, even when its local layout is identical.
        driver.set_matrix_csr(0, 1, &[0, 2, 4], &[0, 1, 0, 1], &[3.0, -1.0, -1.0, 3.0])?;
        driver.set_rhs(&[2.0, 2.0])?;
        check_solution(&comm, &driver.solve()?, 2, 1.0)?;
    }
    // Native statistics are flushed when the driver is dropped. Distinct rows
    // prove warm-start solves use separate timer accumulators.
    let statistics = std::fs::read_to_string(&path)?;
    std::fs::remove_file(&path)?;
    let rows: Vec<_> = statistics
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('|');
            fields.next()?;
            fields.next()?.trim().parse::<usize>().ok()
        })
        .collect();
    check(
        &comm,
        rows == [0, 1, 2, 3, 4],
        "repeated solves did not produce distinct native statistics rows",
    )
}

fn solve_on_comm(runtime: &Runtime<'_>, comm: &impl Communicator) -> TestResult {
    let slab = laplacian(comm)?;
    let mut driver = HypreDrive::from_yaml(runtime, comm, YAML)?;

    // Leading storage is deliberately outside the consumed CSR slab.
    let mut columns = vec![-1];
    columns.extend_from_slice(&slab.columns);
    let mut values = vec![f64::NAN];
    values.extend_from_slice(&slab.values);
    let indptr: Vec<BigInt> = slab.indptr.iter().map(|offset| offset + 1).collect();
    driver.set_matrix_csr(slab.first, slab.last, &indptr, &columns, &values)?;
    driver.set_rhs(&slab.rhs)?;
    driver.set_initial_guess()?;
    let first = driver.solve()?;
    check_solution(comm, &first, slab.rhs.len(), 1.0)?;

    let second_rhs: Vec<_> = slab.rhs.iter().map(|value| 2.0 * value).collect();
    driver.set_rhs(&second_rhs)?;
    driver.reset_initial_guess()?;
    let second = driver.solve()?;
    check_solution(comm, &second, slab.rhs.len(), 2.0)?;
    check(
        comm,
        first.x.iter().all(|x| (x - 1.0).abs() < 1.0e-8),
        "a second solve changed the previously copied solution",
    )
}

fn validation(runtime: &Runtime<'_>, comm: &impl Communicator) -> TestResult {
    let slab = laplacian(comm)?;
    let mut driver = HypreDrive::from_yaml(runtime, comm, YAML)?;
    check(
        comm,
        driver.set_rhs(&slab.rhs).is_err(),
        "RHS before matrix was accepted",
    )?;
    check(
        comm,
        driver.solve().is_err(),
        "solve before matrix was accepted",
    )?;
    check(
        comm,
        driver.set_initial_guess().is_err(),
        "initial guess before RHS was accepted",
    )?;

    // Each failure is supplied on rank zero only. Peers must return an error
    // before any native assembly collective, even with otherwise valid input.
    let mut offsets = slab.indptr.clone();
    if comm.rank() == 0 {
        offsets.pop();
    }
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, slab.last, &offsets, &slab.columns, &slab.values)
            .is_err(),
        "short CSR row-pointer array was accepted",
    )?;
    let mut offsets = slab.indptr.clone();
    if comm.rank() == 0 {
        offsets[0] = -1;
    }
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, slab.last, &offsets, &slab.columns, &slab.values)
            .is_err(),
        "negative CSR offset was accepted",
    )?;
    let mut offsets = slab.indptr.clone();
    if comm.rank() == 0 {
        offsets[1] = offsets[2] + 1;
    }
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, slab.last, &offsets, &slab.columns, &slab.values)
            .is_err(),
        "nonmonotone CSR offsets were accepted",
    )?;
    let mut offsets = slab.indptr.clone();
    if comm.rank() == 0 {
        *offsets.last_mut().unwrap() += 1;
    }
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, slab.last, &offsets, &slab.columns, &slab.values)
            .is_err(),
        "out-of-bounds terminal CSR offset was accepted",
    )?;
    let values = if comm.rank() == 0 {
        &slab.values[..slab.values.len() - 1]
    } else {
        &slab.values
    };
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, slab.last, &slab.indptr, &slab.columns, values)
            .is_err(),
        "short CSR values were accepted",
    )?;
    let (first, last) = if comm.rank() == 0 {
        (BigInt::MAX, BigInt::MAX)
    } else {
        (slab.first, slab.last)
    };
    check(
        comm,
        driver
            .set_matrix_csr(first, last, &slab.indptr, &slab.columns, &slab.values)
            .is_err(),
        "overflowing row range was accepted",
    )?;
    let last = if comm.rank() == 0 {
        slab.first - 1
    } else {
        slab.last
    };
    check(
        comm,
        driver
            .set_matrix_csr(slab.first, last, &slab.indptr, &slab.columns, &slab.values)
            .is_err(),
        "empty local row range was accepted",
    )?;

    driver.set_matrix_csr(
        slab.first,
        slab.last,
        &slab.indptr,
        &slab.columns,
        &slab.values,
    )?;
    let rhs = if comm.rank() == 0 {
        &slab.rhs[..slab.rhs.len() - 1]
    } else {
        &slab.rhs
    };
    check(comm, driver.set_rhs(rhs).is_err(), "short RHS was accepted")?;
    driver.set_rhs(&slab.rhs)?;
    driver.set_initial_guess()?;

    // Column/partition checks belong to the C implementation. A native failure
    // must preserve the original u32 status and the installed valid matrix.
    let mut columns = slab.columns.clone();
    if comm.rank() == 0 {
        columns[0] = BigInt::MAX;
    }
    let failure =
        driver.set_matrix_csr(slab.first, slab.last, &slab.indptr, &columns, &slab.values);
    check(
        comm,
        failure.as_ref().err().is_some_and(|error| {
            error.status().is_some_and(|status| status != 0)
                && error.operation() == "HYPREDRV_LinearSystemSetMatrixFromCSR"
        }),
        "native error lost its operation or status",
    )?;
    let result = driver.solve()?;
    check_solution(comm, &result, slab.rhs.len(), 1.0)
}

fn configuration(runtime: &Runtime<'_>, comm: &impl Communicator) -> TestResult {
    let malformed = HypreDrive::from_yaml(runtime, comm, "solver: [\n");
    check(
        comm,
        malformed.as_ref().err().is_some_and(|error| {
            error.status().is_some_and(|status| status != 0)
                && error.operation() == "HYPREDRV_InputArgsParse"
        }),
        "uniform malformed YAML did not preserve its native parser status",
    )?;
    if comm.size() > 1 {
        let yaml = if comm.rank() == comm.size() - 1 {
            "solver: [\n"
        } else {
            YAML
        };
        let malformed = HypreDrive::from_yaml(runtime, comm, yaml);
        check(
            comm,
            malformed
                .as_ref()
                .err()
                .is_some_and(|error| error.status().is_none() && error.operation() == "from_yaml"),
            "malformed YAML on a nonroot rank escaped collective validation",
        )?;
        // Equal lengths exercise byte comparison in addition to length agreement.
        let changed = YAML.replace("max_iter: 100", "max_iter: 101");
        let yaml = if comm.rank() == comm.size() - 1 {
            changed.as_str()
        } else {
            YAML
        };
        let different = HypreDrive::from_yaml(runtime, comm, yaml);
        check(
            comm,
            different
                .as_ref()
                .err()
                .is_some_and(|error| error.status().is_none() && error.operation() == "from_yaml"),
            "differing valid YAML was accepted across MPI ranks",
        )?;
    }
    let yaml = if comm.rank() == 0 {
        "solver: [\n"
    } else {
        YAML
    };
    check(
        comm,
        HypreDrive::from_yaml(runtime, comm, yaml).is_err(),
        "malformed YAML on one rank was accepted",
    )?;
    let yaml = if comm.rank() == 0 {
        "solver: pcg\0\n"
    } else {
        YAML
    };
    check(
        comm,
        HypreDrive::from_yaml(runtime, comm, yaml).is_err(),
        "embedded NUL on one rank was accepted",
    )?;
    // A successful construction after the failed ones exercises partial cleanup.
    drop(HypreDrive::from_yaml(runtime, comm, YAML)?);

    let path = std::env::temp_dir().join(format!(
        "hypredrive-rust-smoke-{}-{}.yml",
        std::process::id(),
        comm.rank()
    ));
    std::fs::write(&path, YAML)?;
    drop(HypreDrive::from_file(runtime, comm, &path)?);
    let selected_path = if comm.rank() == 0 {
        path.with_extension("missing.yml")
    } else {
        path.clone()
    };
    let missing = HypreDrive::from_file(runtime, comm, &selected_path);
    std::fs::remove_file(&path)?;
    check(
        comm,
        missing.is_err(),
        "missing YAML file on one rank was accepted",
    )
}

fn nonconvergence(runtime: &Runtime<'_>, comm: &impl Communicator) -> TestResult {
    let slab = laplacian(comm)?;
    let yaml = "general:\n  statistics: off\nsolver:\n  pcg:\n    max_iter: 1\n    relative_tol: 1.0e-14\npreconditioner: jacobi\n";
    let mut driver = HypreDrive::from_yaml(runtime, comm, yaml)?;
    driver.set_matrix_csr(
        slab.first,
        slab.last,
        &slab.indptr,
        &slab.columns,
        &slab.values,
    )?;
    driver.set_rhs(&slab.rhs)?;
    driver.set_initial_guess()?;
    let result = driver.solve()?;
    check(
        comm,
        !result.converged && result.iterations == 1 && result.final_res_norm > 1.0e-14,
        "nonconvergence was not returned as a successful SolveResult",
    )
}

fn forgotten_driver(runtime: &Runtime<'_>, comm: &impl Communicator) -> TestResult {
    let source = comm.duplicate();
    let slab = laplacian(&source)?;
    let mut driver = HypreDrive::from_yaml(runtime, &source, YAML)?;
    driver.set_matrix_csr(
        slab.first,
        slab.last,
        &slab.indptr,
        &slab.columns,
        &slab.values,
    )?;
    driver.set_rhs(&slab.rhs)?;
    let result = driver.solve()?;
    check_solution(&source, &result, slab.rhs.len(), 1.0)?;
    // Forgetting is safe Rust. The source communicator can then be released
    // before Runtime finalizes the native object retained in its registry.
    // The driver's owned duplicate stays valid until MPI finalization.
    std::mem::forget(driver);
    drop(source);
    Ok(())
}

fn run(universe: &Universe, world: &impl Communicator, threading: Threading) -> TestResult {
    // No runtime is active here, so only the MPI thread-affinity check can reject
    // a runtime requested from another thread.
    let wrong_thread_error = if threading != Threading::Single {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    Runtime::new(universe)
                        .err()
                        .map(|error| error.to_string())
                        .unwrap_or_default()
                })
                .join()
                .unwrap()
        })
    } else {
        String::from("MPI initialization thread")
    };
    check(
        world,
        wrong_thread_error.contains("MPI initialization thread"),
        "a runtime on another thread was not rejected by the MPI thread check",
    )?;
    {
        let runtime = Runtime::new(universe)?;
        check(
            world,
            Runtime::new(universe).is_err(),
            "second active runtime was accepted",
        )?;
        let other_thread_rejected = if threading != Threading::Single {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| Runtime::new(universe).is_err())
                    .join()
                    .unwrap()
            })
        } else {
            true
        };
        check(
            world,
            other_thread_rejected,
            "second active runtime on another thread was accepted",
        )?;
        serial_minimal(&runtime)?;
        initial_guess_and_statistics(&runtime)?;
        configuration(&runtime, world)?;
        validation(&runtime, world)?;
        solve_on_comm(&runtime, world)?;
        nonconvergence(&runtime, world)?;
        let split = world
            .split_by_color(Color::with_value(world.rank() % 2))
            .unwrap();
        solve_on_comm(&runtime, &split)?;
        forgotten_driver(&runtime, world)?;
    }
    // The previous runtime released the process-wide ownership guard.
    let runtime = Runtime::new(universe)?;
    serial_minimal(&runtime)
}

fn main() {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the smoke executable initializes MPI once");
    let world = universe.world();
    if let Err(error) = run(&universe, &world, threading) {
        eprintln!("rank {}: {error}", world.rank());
        world.abort(1);
    }
    if world.rank() == 0 {
        println!(
            "Rust interface smoke checks passed on {} MPI ranks",
            world.size()
        );
    }
}
