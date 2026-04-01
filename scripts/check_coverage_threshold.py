#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

"""Fail if gcovr coverage-summary.json (or coverage.xml) is below thresholds."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET


def _load_json_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rates_from_xml(path: str) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    line_r = float(root.get("line-rate", 0.0))
    branch_r = float(root.get("branch-rate", 0.0))
    return line_r, branch_r


def main() -> int:
    p = argparse.ArgumentParser(
        description="Check gcovr coverage against minimum line/branch/function rates."
    )
    p.add_argument(
        "summary_json",
        nargs="?",
        default="coverage-summary.json",
        help="Path to gcovr --json-summary output (default: coverage-summary.json)",
    )
    p.add_argument(
        "--xml",
        dest="xml_path",
        default=None,
        help="Optional coverage.xml; line/branch rates are taken from XML when given "
        "(consistent with Cobertura root attributes).",
    )
    p.add_argument(
        "--min-line",
        type=float,
        default=0.90,
        help="Minimum line coverage (default: 0.90)",
    )
    p.add_argument(
        "--min-branch",
        type=float,
        default=0.90,
        help="Minimum branch coverage (default: 0.90)",
    )
    p.add_argument(
        "--min-function",
        type=float,
        default=0.95,
        help="Minimum function coverage from JSON summary (default: 0.95)",
    )
    args = p.parse_args()

    try:
        summary = _load_json_summary(args.summary_json)
    except OSError as e:
        sys.stderr.write(f"Error: could not read {args.summary_json}: {e}\n")
        return 2
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: invalid JSON in {args.summary_json}: {e}\n")
        return 2

    func_pct = float(summary.get("function_percent", 0.0)) / 100.0
    line_pct = float(summary.get("line_percent", 0.0)) / 100.0
    branch_pct = float(summary.get("branch_percent", 0.0)) / 100.0

    if args.xml_path:
        try:
            xml_line, xml_branch = _rates_from_xml(args.xml_path)
            line_pct = xml_line
            branch_pct = xml_branch
        except (OSError, ET.ParseError) as e:
            sys.stderr.write(f"Error: could not use XML {args.xml_path}: {e}\n")
            return 2

    ok = True
    lines = [
        f"Coverage: lines={line_pct * 100:.2f}% branches={branch_pct * 100:.2f}% "
        f"functions={func_pct * 100:.2f}%"
    ]

    if line_pct + 1e-9 < args.min_line:
        ok = False
        lines.append(
            f"FAIL: line coverage {line_pct * 100:.2f}% < {args.min_line * 100:.0f}%"
        )
    if branch_pct + 1e-9 < args.min_branch:
        ok = False
        lines.append(
            f"FAIL: branch coverage {branch_pct * 100:.2f}% < {args.min_branch * 100:.0f}%"
        )
    if func_pct + 1e-9 < args.min_function:
        ok = False
        lines.append(
            f"FAIL: function coverage {func_pct * 100:.2f}% < "
            f"{args.min_function * 100:.0f}%"
        )

    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
