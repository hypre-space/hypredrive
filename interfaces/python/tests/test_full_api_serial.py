"""Serial tests for the remaining HYPREDRV_ API wrappers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypredrive.driver")
import hypredrive as hd
from hypredrive import _core


def _laplacian(n=20):
    import scipy.sparse as sp

    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")


def test_module_level_functions(capfd):
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
