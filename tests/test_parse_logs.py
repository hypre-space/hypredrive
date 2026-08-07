#!/usr/bin/env python3
"""Tests for the mode-based HypreDrive log parser."""

from __future__ import annotations

import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SOURCE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from parse_logs import parse_snapshots  # noqa: E402


class BlockNormParserTests(unittest.TestCase):
    def test_bare_numeric_labels(self) -> None:
        log = """\
[obj-4] [ls=9] matrix block Frobenius norms: blocks=2 matrix_norm=5 ignored_nnz=0
block Frobenius row 0: 0=1(nnz=1) 2=2(nnz=2)
block Frobenius row 2: 0=3(nnz=3) 2=4(nnz=4)
"""
        snapshot = parse_snapshots(StringIO(log), "bare.log")[0]
        self.assertEqual(snapshot.labels(), ["0", "2"])
        self.assertEqual(snapshot.norm_matrix(), [[1.0, 2.0], [3.0, 4.0]])

    def test_named_labels_may_contain_whitespace(self) -> None:
        log = """\
matrix block Frobenius norms: blocks=2 matrix_norm=5 ignored_nnz=0
block Frobenius row fluid velocity(id=0): fluid velocity(id=0)=1(nnz=1) pore pressure(id=2)=2(nnz=2)
block Frobenius row pore pressure(id=2): fluid velocity(id=0)=3(nnz=3) pore pressure(id=2)=4(nnz=4)
"""
        snapshot = parse_snapshots(StringIO(log), "named.log")[0]
        self.assertEqual(snapshot.labels(), ["fluid velocity", "pore pressure"])
        self.assertEqual(snapshot.nnz_matrix(), [[1, 2], [3, 4]])

    def test_mode_entry_point_dispatches_block_norms(self) -> None:
        log = """\
matrix block Frobenius norms: blocks=1 matrix_norm=1 ignored_nnz=0
block Frobenius row 0: 0=1(nnz=1)
"""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "parse_logs.py"),
                "--mode",
                "block_norms",
                "-",
                "--format",
                "csv",
            ],
            input=log,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("row/column,0", result.stdout)
        self.assertIn("0,1.000000e+00", result.stdout)


if __name__ == "__main__":
    unittest.main()
