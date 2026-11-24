// Copyright (c) 2024, Lawrence Livermore National Security, LLC.
// See the top-level LICENSE and NOTICE files for details.

use hypredrive::{BigInt, HypreDrive, Runtime};
use mpi::collective::SystemOperation;
use mpi::traits::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = std::env::args()
        .skip(1)
        .map(|arg| arg.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let [nx, ny, nz] = match dimensions.as_slice() {
        [] => [10; 3],
        [n] => [*n; 3],
        [nx, ny, nz] => [*nx, *ny, *nz],
        _ => return Err("usage: laplacian [n | nx ny nz]".into()),
    };
    if nx == 0 || ny == 0 || nz == 0 {
        return Err("grid dimensions must be positive".into());
    }
    let plane = nx.checked_mul(ny).ok_or("grid dimensions overflow")?;
    let global_rows = plane.checked_mul(nz).ok_or("grid dimensions overflow")?;
    BigInt::try_from(global_rows)?;

    let universe = mpi::initialize().expect("MPI must be initialized once by this application");
    let comm = universe.world();
    let size = usize::try_from(comm.size())?;
    let rank = usize::try_from(comm.rank())?;
    if global_rows < size {
        return Err("the global row count must be at least the MPI process count".into());
    }
    let runtime = Runtime::new(&universe)?;
    let rows = global_rows / size + usize::from(rank < global_rows % size);
    let first = rank * (global_rows / size) + rank.min(global_rows % size);
    let last = first + rows - 1;

    // Flatten the grid with x varying fastest. Assemble only this rank's rows.
    let mut indptr = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    let mut rhs = Vec::with_capacity(rows);
    for row in first..=last {
        let (x, y, z) = (row % nx, (row / nx) % ny, row / plane);
        columns.push(BigInt::try_from(row)?);
        values.push(6.0);
        // Each missing neighbor contributes one to the RHS for the exact solution x = 1.
        let mut row_sum = 6.0;
        for (coordinate, extent, stride) in [(x, nx, 1), (y, ny, nx), (z, nz, plane)] {
            if coordinate > 0 {
                columns.push(BigInt::try_from(row - stride)?);
                values.push(-1.0);
                row_sum -= 1.0;
            }
            if coordinate + 1 < extent {
                columns.push(BigInt::try_from(row + stride)?);
                values.push(-1.0);
                row_sum -= 1.0;
            }
        }
        indptr.push(BigInt::try_from(columns.len())?);
        rhs.push(row_sum);
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
            "grid = {nx} x {ny} x {nz}, rows = {global_rows}, ranks = {size}, iterations = {}",
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
