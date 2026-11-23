// Copyright (c) 2024, Lawrence Livermore National Security, LLC.
// See the top-level LICENSE and NOTICE files for details.

use hypredrive::{BigInt, HypreDrive, Runtime};
use mpi::collective::SystemOperation;
use mpi::traits::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().expect("MPI must be initialized once by this application");
    let comm = universe.world();
    let runtime = Runtime::new(&universe)?;
    let n: usize = std::env::args()
        .nth(1)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(32);
    let size = usize::try_from(comm.size())?;
    let rank = usize::try_from(comm.rank())?;
    if n < size {
        return Err("the global row count must be at least the MPI process count".into());
    }
    let _global_rows = BigInt::try_from(n)?;
    let rows = n / size + usize::from(rank < n % size);
    let first = rank * (n / size) + rank.min(n % size);
    let last = first + rows - 1;

    // Assemble only this rank's rows. Column indices are global.
    let mut indptr = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    let mut rhs = Vec::with_capacity(rows);
    for row in first..=last {
        let mut value = 2.0;
        if row > 0 {
            columns.push(BigInt::try_from(row - 1)?);
            values.push(-1.0);
            value -= 1.0;
        }
        columns.push(BigInt::try_from(row)?);
        values.push(2.0);
        if row + 1 < n {
            columns.push(BigInt::try_from(row + 1)?);
            values.push(-1.0);
            value -= 1.0;
        }
        indptr.push(BigInt::try_from(columns.len())?);
        rhs.push(value);
    }

    let mut driver = HypreDrive::from_yaml(
        &runtime,
        &comm,
        "general:\n  statistics: off\nsolver:\n  pcg:\n    max_iter: 100\n    relative_tol: 1.0e-12\npreconditioner: amg\n",
    )?;
    driver.set_matrix_csr(
        BigInt::try_from(first)?,
        BigInt::try_from(last)?,
        &indptr,
        &columns,
        &values,
    )?;
    driver.set_rhs(&rhs)?;
    driver.set_initial_guess()?;
    let result = driver.solve()?;
    let local_error = result.x.iter().map(|x| (x - 1.0).abs()).fold(0.0, f64::max);
    let mut error = 0.0_f64;
    comm.all_reduce_into(&local_error, &mut error, SystemOperation::max());
    assert!(result.converged && error < 1.0e-8);
    if comm.rank() == 0 {
        println!(
            "rows = {n}, ranks = {size}, iterations = {}",
            result.iterations
        );
        println!("maximum solution error = {error:.3e}");
    }

    // Reuse the native matrix and driver with a second right-hand side.
    let second_rhs: Vec<_> = rhs.iter().map(|value| 2.0 * value).collect();
    driver.set_rhs(&second_rhs)?;
    driver.reset_initial_guess()?;
    let second = driver.solve()?;
    let local_error = second.x.iter().map(|x| (x - 2.0).abs()).fold(0.0, f64::max);
    comm.all_reduce_into(&local_error, &mut error, SystemOperation::max());
    assert!(second.converged && error < 1.0e-8);
    Ok(())
}
