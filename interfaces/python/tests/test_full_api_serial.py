"""Serial tests for the remaining HYPREDRV_ API wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
import hypredrive as hd


def _laplacian(n=20):
    import scipy.sparse as sp

    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")


def test_module_level_functions():
    hd.initialize()
    hd.print_lib_info()
    hd.print_system_info()
    hd.print_exit_info(argv0="prog")
    hd.clear_error()
    code = hd.invalid_value_error("python test")
    assert code != 0
    hd.clear_error()


def test_presets_and_input_queries(base_options):
    hd.register_solver_preset(
        "pytest_pcg", "pcg:\n  max_iter: 50\n  relative_tol: 1.0e-8\n", "pytest")
    hd.register_precon_preset("pytest_amg", "amg:\n  print_level: 0\n", "pytest")
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_name("pytest-driver")
        assert drv.warmup >= 0
        assert drv.num_repetitions >= 1
        assert drv.num_linear_systems >= 1
        assert drv.num_precon_variants >= 1
        drv.set_precon_variant(0)
        drv.set_solver_preset("pytest_pcg")
        drv.set_precon_preset("pytest_amg")
        with pytest.raises(hd.HypreDriveError):
            drv.set_solver_preset("no_such_preset_xyz")
        hd.clear_error()


def test_native_handles_and_values(base_options):
    A = _laplacian()
    b = np.ones(A.shape[0])
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(A)
        drv.set_rhs(b)
        assert drv.get_matrix_handle() != 0
        assert drv.get_rhs_handle() != 0
        np.testing.assert_allclose(drv.get_rhs_values(), b)
        drv.solve()
        assert drv.get_solution_handle() != 0
        x = drv.get_solution()
        np.testing.assert_allclose(drv.get_solution_values(), x)
        # Re-attaching our own borrowed handles is allowed.
        drv.set_reference_solution_handle(None)
        drv.set_prec_matrix_handle(None)
        drv.set_coarse_schur_handle(0, None)
        drv.set_coordinates_handle(None, None, None)
        drv.set_discrete_gradient_handle(None)
        drv.set_discrete_curl_handle(None)


def test_native_handle_setters_refresh_rows(base_options):
    A20, A40 = _laplacian(20), _laplacian(40)
    # ``donor`` owns the 40-row handles and must outlive ``drv``.
    with hd.HypreDrive(options=base_options) as donor:
        donor.set_matrix_from_csr(A40)
        donor.set_rhs(np.arange(1.0, 41.0))
        with hd.HypreDrive(options=base_options) as drv:
            drv.set_matrix_from_csr(A20)
            drv.set_rhs(np.ones(20))
            drv.solve()
            assert drv.get_solution().shape == (20,)
            drv.set_matrix_handle(donor.get_matrix_handle())
            drv.set_rhs_handle(donor.get_rhs_handle())
            np.testing.assert_allclose(drv.get_rhs_values(), np.arange(1.0, 41.0))
            drv.solve()
            x = drv.get_solution()
            assert x.shape == (40,)
            np.testing.assert_allclose(A40 @ x, np.arange(1.0, 41.0), rtol=1e-6)
            drv.set_matrix_handle(None)  # clearing is allowed
            with pytest.raises(RuntimeError):
                drv.solve()


_PS3D = (Path(__file__).resolve().parents[3] / "data" / "ps3d10pt7" / "np1")


@pytest.mark.skipif(not (_PS3D / "IJ.out.A.00000").exists(),
                    reason="ps3d10pt7 dataset not downloaded")
def test_build_linear_system_keeps_previous_guess():
    opts = (
        "linear_system:\n"
        f"  matrix_filename: {_PS3D / 'IJ.out.A'}\n"
        f"  rhs_filename: {_PS3D / 'IJ.out.b'}\n"
        "  init_guess_mode: previous\n"
        "solver: pcg\npreconditioner: amg\n"
    )
    with hd.HypreDrive(options=opts) as drv:
        drv.build_linear_system()
        drv.solve()
        first = drv.last_iterations
        assert drv.get_solution().shape == drv.get_rhs_values().shape == (1000,)
        # The rebuilt system starts from the previous solution.
        drv.build_linear_system()
        drv.solve()
        assert drv.last_converged is True
        assert drv.last_iterations < first


def test_state_vectors_sized_from_vectors(base_options):
    with hd.HypreDrive(options=base_options) as small, \
            hd.HypreDrive(options=base_options) as big:
        small.set_matrix_from_csr(_laplacian(4))
        small.set_rhs(np.arange(1.0, 5.0))
        big.set_matrix_from_csr(_laplacian(16))
        big.set_rhs(np.ones(16))
        with hd.HypreDrive(options=base_options) as drv:
            drv.set_matrix_from_csr(_laplacian(16))
            drv.set_rhs(np.ones(16))
            drv.solve()
            with pytest.raises(ValueError):
                drv.set_state_vectors([small.get_rhs_handle(), big.get_rhs_handle()])
            # A state shorter than the system is read with its own length.
            drv.set_state_vectors([small.get_rhs_handle()])
            np.testing.assert_allclose(drv.state_vector_values(0),
                                       np.arange(1.0, 5.0))


def test_print_info_accepts_duplicated_comm():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD.Dup()
    try:
        hd.print_lib_info(comm=comm)
        hd.print_system_info(comm=comm)
    finally:
        comm.Free()


def test_handle_type_validation(base_options):
    with hd.HypreDrive(options=base_options) as drv:
        with pytest.raises(TypeError):
            drv.set_rhs_handle("not a pointer")


def test_standalone_precon(base_options):
    A = _laplacian()
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(A)
        drv.set_rhs(np.ones(A.shape[0]))
        drv.solve()  # materializes the solution vector
        drv.precon_create()
        drv.precon_setup()
        rhs = drv.get_rhs_handle()
        sol = drv.get_solution_handle()
        drv.precon_apply(rhs, sol)
        assert np.isfinite(drv.get_solution_values()).all()
        drv.precon_destroy()
        drv.precon_create()  # destroyed automatically on close


def test_null_spaces_validation(base_options):
    A = _laplacian()
    n = A.shape[0]
    modes = np.ones((1, n))
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(A)
        drv.set_near_null_space(modes)
        drv.set_null_space(modes.reshape(-1), num_components=1)
        with pytest.raises(TypeError):
            drv.set_null_space(np.ones(n))
        with pytest.raises(ValueError):
            drv.set_null_space(np.ones(n), num_components=3)


def test_dofmap_variants_and_printing(base_options, tmp_path):
    A = _laplacian(20)
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(A)
        drv.set_interleaved_dofmap(10, 2)
        drv.print_dofmap(str(tmp_path / "dofmap"))
        assert any(tmp_path.glob("dofmap*"))
        drv.set_contiguous_dofmap(10, 2)


def test_level_stats(base_options):
    A = _laplacian()
    with hd.HypreDrive(options=base_options) as drv:
        drv.set_matrix_from_csr(A)
        drv.set_rhs(np.ones(A.shape[0]))
        drv.annotate_level_begin(0, "timestep", 1)
        drv.solve()
        drv.annotate_level_end(0, "timestep", 1)
        assert drv.stats_level_count(0) == 1
        entry = drv.stats_level_entry(0, 0)
        assert entry["entry_id"] >= 1
        assert set(entry) == {"entry_id", "num_solves", "linear_iters",
                              "setup_time", "solve_time"}
        drv.print_stats_level(0)
        drv.print_stats()
        with pytest.raises(hd.HypreDriveError):
            drv.stats_level_count(999)
        hd.clear_error()


def test_closed_driver_rejects_calls(base_options):
    drv = hd.HypreDrive(options=base_options)
    drv.close()
    with pytest.raises(RuntimeError):
        drv.print_stats()
    with pytest.raises(RuntimeError):
        drv.get_rhs_handle()
