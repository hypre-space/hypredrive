// Copyright (c) 2024, Lawrence Livermore National Security, LLC.
// See the top-level LICENSE and NOTICE files for details.

use hypredrive::{BigInt, HypreDrive, Runtime};
use mpi::topology::SimpleCommunicator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().expect("MPI must be initialized once by this application");
    let comm = SimpleCommunicator::self_comm();
    let runtime = Runtime::new(&universe)?;
    let mut driver = HypreDrive::from_yaml(
        &runtime,
        &comm,
        "general:\n  statistics: off\nsolver:\n  pcg:\n    relative_tol: 1.0e-12\npreconditioner: amg\n",
    )?;

    let indptr: [BigInt; 3] = [0, 2, 4];
    let columns: [BigInt; 4] = [0, 1, 0, 1];
    driver.set_matrix_csr(0, 1, &indptr, &columns, &[2.0, -1.0, -1.0, 2.0])?;
    driver.set_rhs(&[1.0, 1.0])?;
    driver.set_initial_guess()?;
    let result = driver.solve()?;
    assert!(result.converged);
    assert!(result.x.iter().all(|x| (x - 1.0).abs() < 1.0e-10));
    assert!(result.final_res_norm < 1.0e-10);
    println!("x = {:?}, iterations = {}", result.x, result.iterations);

    // Locals drop in reverse order: driver, runtime, communicator, then MPI.
    Ok(())
}
