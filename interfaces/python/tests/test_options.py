"""Tests for the options helper.

These run without invoking the C library, so they're cheap CI smoke tests
that catch regressions in the dict->YAML emission.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hypredrive.options import configure, normalize_options, options_to_yaml


def test_package_import_does_not_eagerly_load_core_or_driver():
    code = (
        "import hypredrive, sys; "
        "assert 'hypredrive._core' not in sys.modules; "
        "assert 'hypredrive.driver' not in sys.modules; "
        "assert 'hypredrive.session' not in sys.modules; "
        "from hypredrive.options import configure; "
        "assert configure(solver='pcg')['solver'] == {'pcg': {}}"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_simple_dict_emits_indented_yaml():
    yaml = options_to_yaml(
        {
            "general": {"statistics": False},
            "solver": {"pcg": {"max_iter": 100}},
        }
    )
    assert "general:\n" in yaml
    assert "  statistics: off" in yaml
    assert "solver:\n" in yaml
    assert "  pcg:\n" in yaml
    assert "    max_iter: 100" in yaml


def test_configure_builds_method_options():
    options = configure(
        solver="pcg",
        preconditioner="amg",
        pcg={"relative_tol": 1.0e-8},
    )
    assert options == {
        "solver": {"pcg": {"relative_tol": 1.0e-8}},
        "preconditioner": {"amg": {}},
    }


def test_configure_builds_example_style_full_config():
    options = configure(
        general={"use_millisec": True, "dev_pool_size": 0.01},
        linear_system={
            "rhs_filename": "data/ps3d10pt7/np1/IJ.out.b",
            "matrix_filename": "data/ps3d10pt7/np1/IJ.out.A",
        },
        solver="pcg",
        preconditioner="amg",
    )
    assert options == {
        "general": {"use_millisec": True, "dev_pool_size": 0.01},
        "linear_system": {
            "rhs_filename": "data/ps3d10pt7/np1/IJ.out.b",
            "matrix_filename": "data/ps3d10pt7/np1/IJ.out.A",
        },
        "solver": {"pcg": {}},
        "preconditioner": {"amg": {}},
    }
    yaml = options_to_yaml(options)
    assert "use_millisec: on" in yaml
    assert "dev_pool_size: 0.01" in yaml
    assert "solver:\n  pcg:\n" in yaml
    assert "preconditioner:\n  amg:\n" in yaml


def test_configure_preserves_nested_mgr_options():
    mgr = {
        "max_iter": 1,
        "tolerance": 0.0,
        "level": {
            "0": {
                "f_dofs": [0, 1, 2],
                "f_relaxation": {
                    "amg": {
                        "coarsening": {
                            "type": "pmis",
                            "strong_th": 0.5,
                            "num_functions": 3,
                            "filter_functions": True,
                        }
                    }
                },
                "g_relaxation": "none",
                "restriction_type": "injection",
            },
            "1": {
                "f_dofs": [5],
                "f_relaxation": "jacobi",
                "g_relaxation": "none",
            },
        },
        "coarsest_level": {
            "amg": {
                "max_iter": 1,
                "relaxation": {
                    "down_type": "l1-jacobi",
                    "up_type": "l1-jacobi",
                },
            }
        },
    }
    options = configure(
        solver="fgmres",
        preconditioner="mgr",
        fgmres={"max_iter": 100, "krylov_dim": 30, "relative_tol": 1.0e-6},
        mgr=mgr,
    )
    assert options["solver"]["fgmres"]["krylov_dim"] == 30
    assert options["preconditioner"]["mgr"]["level"]["0"]["f_dofs"] == [0, 1, 2]
    assert (
        options["preconditioner"]["mgr"]["level"]["0"]["f_relaxation"]["amg"]
        ["coarsening"]["filter_functions"]
        is True
    )
    yaml = options_to_yaml(options)
    assert "fgmres:" in yaml
    assert "mgr:" in yaml
    assert "f_dofs: 0,1,2" in yaml
    assert "filter_functions: on" in yaml


def test_configure_preserves_list_valued_amg_options():
    amg = [
        {
            "coarsening": {"type": "HMIS", "strong_th": 0.25},
            "interpolation": {"prolongation_type": "MM-ext+i"},
            "relaxation": {
                "down_type": 16,
                "down_sweeps": 1,
                "up_type": 16,
                "up_sweeps": 1,
            },
        },
        {
            "coarsening": {"type": "PMIS", "strong_th": 0.5},
            "interpolation": {"prolongation_type": "direct_sep_weights"},
            "relaxation": {
                "down_type": 8,
                "down_sweeps": 2,
                "up_type": 8,
                "up_sweeps": 2,
            },
        },
    ]
    options = configure(
        solver="pcg",
        preconditioner="amg",
        pcg={"relative_tol": 1.0e-9, "max_iter": 500},
        amg=amg,
    )
    assert options["preconditioner"]["amg"] == amg
    yaml = options_to_yaml(options)
    assert "amg:\n    - coarsening:" in yaml
    assert "strong_th: 0.25" in yaml
    assert "prolongation_type: MM-ext+i" in yaml


def test_configure_rejects_unused_option_blocks():
    with pytest.raises(ValueError, match="unused method option"):
        configure(solver="pcg", gmres={"max_iter": 10})


def test_configure_rejects_non_mapping_sections():
    with pytest.raises(TypeError, match="general options"):
        configure(general="use_millisec: on")  # type: ignore[arg-type]


def test_configure_rejects_non_mapping_method_options():
    with pytest.raises(TypeError, match="pcg options"):
        configure(solver="pcg", pcg=1.0)


def test_bool_renders_on_off():
    assert "x: on" in options_to_yaml({"x": True})
    assert "x: off" in options_to_yaml({"x": False})


def test_list_joined_with_commas():
    yaml = options_to_yaml({"linear_system": {"set_suffix": [1, 2, 5]}})
    assert "set_suffix: 1,2,5" in yaml


def test_empty_sequence_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        options_to_yaml({"linear_system": {"set_suffix": []}})


def test_empty_method_option_sequence_rejected_on_emit():
    options = configure(solver="pcg", preconditioner="amg", amg=[])
    with pytest.raises(ValueError, match="must not be empty"):
        options_to_yaml(options)


def test_unknown_key_type_rejected():
    with pytest.raises(TypeError):
        options_to_yaml({1: "bad"})


def test_normalize_none_returns_minimal_yaml():
    text = normalize_options(None)
    assert "general:" in text


def test_normalize_str_passthrough():
    raw = "general:\n  statistics: off\n"
    assert normalize_options(raw) == raw


def test_normalize_path_reads_file(tmp_path):
    fp = tmp_path / "opts.yml"
    fp.write_text("general:\n  statistics: on\n", encoding="utf-8")
    assert normalize_options(fp).strip() == "general:\n  statistics: on".strip()


def test_normalize_missing_pathlike_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_options(tmp_path / "missing.yml")


def test_normalize_missing_path_string_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_options(str(tmp_path / "missing.yml"))


def test_normalize_single_line_yaml_literal_still_passes_through():
    assert normalize_options("general:") == "general:"
