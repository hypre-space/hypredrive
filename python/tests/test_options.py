"""Tests for the options helper.

These run without invoking the C library, so they're cheap CI smoke tests
that catch regressions in the dict->YAML emission.
"""

from __future__ import annotations

import pytest

from hypredrive.options import normalize_options, options_to_yaml


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


def test_bool_renders_on_off():
    assert "x: on" in options_to_yaml({"x": True})
    assert "x: off" in options_to_yaml({"x": False})


def test_list_joined_with_commas():
    yaml = options_to_yaml({"linear_system": {"set_suffix": [1, 2, 5]}})
    assert "set_suffix: 1,2,5" in yaml


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
