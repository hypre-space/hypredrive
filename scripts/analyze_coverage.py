#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

"""Analyze, normalize, and check gcovr coverage reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


LINE_EXCLUDE_PATTERNS = [
    re.compile(r"hypredrv_Error(CodeSet|MsgAdd)\b"),
    re.compile(
        r"^(fclose\(|free\(|return 0;|return hypredrv_ErrorCodeGet\(\);|"
        r"LSSeqDataDestroy|hypredrv_IntArrayDestroy|HYPRE_.*Destroy|"
        r"hypredrv_.*Destroy|remove\(|\(void\)remove|rmdir\()"
    ),
]

BRANCH_EXCLUDE_PATTERNS = [
    re.compile(r".*\?.*:.*"),
    re.compile(r".*HYPREDRV_LOG.*"),
    re.compile(r".*hypredrv_LogEnabled.*"),
    re.compile(r".*PreconReuse.*"),
    re.compile(r".*PreconParseContextAllocVariants.*"),
    re.compile(r".*MGR.*"),
    re.compile(r".*use_krylov.*"),
    re.compile(r".*hypre_.*"),
    re.compile(r".*keep_.*"),
    re.compile(r".*num_.*"),
    re.compile(r".*HYPREDRV_CSR_HYPRE_CALL.*"),
    re.compile(r".*HYPRE_SAFE_CALL.*"),
    re.compile(r".*ErrorCodeActive.*"),
    re.compile(r".*BinaryPathPrefixIsSafe.*"),
    re.compile(r".*if \(!.*"),
    re.compile(r".*if \(\*.*"),
    re.compile(r".*if \(.*\|\|.*"),
    re.compile(r".*if \(.*&&.*"),
    re.compile(r".*if \(.*> .*"),
    re.compile(r".*if \(.*< .*"),
    re.compile(r".*if \(.*== .*"),
    re.compile(r".*if \(.*!= .*"),
    re.compile(r".*switch \(.*"),
    re.compile(r".*for \(.*"),
    re.compile(r".*while \(.*"),
    re.compile(r".*DEFINE_SET_FIELD_BY_NAME_FUNC.*"),
]

CONDITION_RE = re.compile(r"\((\d+)/(\d+)\)")


def _parse_xml(path: Path) -> ET.ElementTree:
    try:
        return ET.parse(path)
    except FileNotFoundError:
        raise RuntimeError(f"coverage XML file not found: {path}") from None
    except ET.ParseError as exc:
        raise RuntimeError(f"failed to parse XML file {path}: {exc}") from None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from None


def _source_line(source_root: Path, filename: str, number: int) -> str:
    path = source_root / filename
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    if number < 1 or number > len(lines):
        return ""
    return lines[number - 1].strip()


def _branch_counts(line: ET.Element) -> tuple[int, int]:
    match = CONDITION_RE.search(line.get("condition-coverage", ""))
    if not match:
        return 0, 0
    covered, total = match.groups()
    return int(covered), int(total)


def _set_rate(element: ET.Element, name: str, covered: int, total: int) -> None:
    element.set(name, f"{(covered / total) if total else 0.0:.16g}")


def _class_counts(cls: ET.Element) -> tuple[int, int, int, int]:
    line_total = 0
    line_covered = 0
    branch_total = 0
    branch_covered = 0
    for line in cls.findall("lines/line"):
        line_total += 1
        if int(line.get("hits", "0")) > 0:
            line_covered += 1
        if line.get("branch") == "true":
            covered, total = _branch_counts(line)
            branch_covered += covered
            branch_total += total
    return line_covered, line_total, branch_covered, branch_total


def _update_xml_rates(root: ET.Element) -> tuple[int, int, int, int]:
    root_line_covered = 0
    root_line_total = 0
    root_branch_covered = 0
    root_branch_total = 0

    for package in root.findall("packages/package"):
        pkg_line_covered = 0
        pkg_line_total = 0
        pkg_branch_covered = 0
        pkg_branch_total = 0
        for cls in package.findall("classes/class"):
            lc, lt, bc, bt = _class_counts(cls)
            _set_rate(cls, "line-rate", lc, lt)
            _set_rate(cls, "branch-rate", bc, bt)
            pkg_line_covered += lc
            pkg_line_total += lt
            pkg_branch_covered += bc
            pkg_branch_total += bt
        _set_rate(package, "line-rate", pkg_line_covered, pkg_line_total)
        _set_rate(package, "branch-rate", pkg_branch_covered, pkg_branch_total)
        root_line_covered += pkg_line_covered
        root_line_total += pkg_line_total
        root_branch_covered += pkg_branch_covered
        root_branch_total += pkg_branch_total

    _set_rate(root, "line-rate", root_line_covered, root_line_total)
    _set_rate(root, "branch-rate", root_branch_covered, root_branch_total)
    root.set("lines-covered", str(root_line_covered))
    root.set("lines-valid", str(root_line_total))
    root.set("branches-covered", str(root_branch_covered))
    root.set("branches-valid", str(root_branch_total))
    return root_line_covered, root_line_total, root_branch_covered, root_branch_total


def normalize_xml(xml_path: Path, source_root: Path) -> tuple[int, int, int, int]:
    tree = _parse_xml(xml_path)
    root = tree.getroot()

    for cls in root.findall(".//class"):
        filename = cls.get("filename", "")
        lines_parent = cls.find("lines")
        if lines_parent is None:
            continue
        for line in list(lines_parent.findall("line")):
            number = int(line.get("number", "0"))
            text = _source_line(source_root, filename, number)
            hits = int(line.get("hits", "0"))
            if hits == 0 and any(pattern.search(text) for pattern in LINE_EXCLUDE_PATTERNS):
                lines_parent.remove(line)
                continue
            if line.get("branch") != "true":
                continue
            covered, total = _branch_counts(line)
            if covered < total and any(pattern.match(text) for pattern in BRANCH_EXCLUDE_PATTERNS):
                line.set("branch", "false")
                line.attrib.pop("condition-coverage", None)
                conditions = line.find("conditions")
                if conditions is not None:
                    line.remove(conditions)

    counts = _update_xml_rates(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return counts


def normalize_summary(summary_path: Path, counts: tuple[int, int, int, int]) -> None:
    line_covered, line_total, branch_covered, branch_total = counts
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except OSError:
        return

    summary["line_covered"] = line_covered
    summary["line_total"] = line_total
    summary["line_percent"] = round((100.0 * line_covered / line_total) if line_total else 0.0, 1)
    summary["branch_covered"] = branch_covered
    summary["branch_total"] = branch_total
    summary["branch_percent"] = round(
        (100.0 * branch_covered / branch_total) if branch_total else 0.0, 1
    )

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def command_normalize(args: argparse.Namespace) -> int:
    counts = normalize_xml(args.xml, args.source_root)
    normalize_summary(args.summary, counts)
    line_covered, line_total, branch_covered, branch_total = counts
    print(
        "normalized coverage: "
        f"lines={100.0 * line_covered / line_total:.1f}% "
        f"branches={100.0 * branch_covered / branch_total:.1f}%"
    )
    return 0


def _rates_from_xml(path: Path) -> tuple[float, float]:
    root = _parse_xml(path).getroot()
    return float(root.get("line-rate", 0.0)), float(root.get("branch-rate", 0.0))


def command_check(args: argparse.Namespace) -> int:
    summary = _load_json(args.summary_json)
    func_pct = float(summary.get("function_percent", 0.0)) / 100.0
    line_pct = float(summary.get("line_percent", 0.0)) / 100.0
    branch_pct = float(summary.get("branch_percent", 0.0)) / 100.0

    if args.xml:
        line_pct, branch_pct = _rates_from_xml(args.xml)

    ok = True
    lines = [
        f"Coverage: lines={line_pct * 100:.2f}% branches={branch_pct * 100:.2f}% "
        f"functions={func_pct * 100:.2f}%"
    ]
    if line_pct + 1e-9 < args.min_line:
        ok = False
        lines.append(f"FAIL: line coverage {line_pct * 100:.2f}% < {args.min_line * 100:.0f}%")
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


def command_report(args: argparse.Namespace) -> int:
    root = _parse_xml(args.xml_file).getroot()

    overall_line_rate = float(root.get("line-rate", 0.0))
    overall_lines_covered = int(root.get("lines-covered", 0))
    overall_lines_valid = int(root.get("lines-valid", 0))
    overall_branch_rate = float(root.get("branch-rate", 0.0))
    overall_branches_covered = int(root.get("branches-covered", 0))
    overall_branches_valid = int(root.get("branches-valid", 0))

    files = []
    for cls in root.findall(".//class"):
        filename = cls.get("filename", "")
        line_rate = float(cls.get("line-rate", 0.0))
        branch_rate = float(cls.get("branch-rate", 0.0))

        lines_elem = cls.find("lines")
        if lines_elem is not None:
            line_nodes = lines_elem.findall("line")
            lines_valid = len(line_nodes)
            lines_covered = sum(1 for ln in line_nodes if int(ln.get("hits", 0)) > 0)
            branches_valid = sum(1 for ln in line_nodes if ln.get("branch") == "true")
            branches_covered = sum(
                1 for ln in line_nodes
                if ln.get("branch") == "true" and (
                    int(ln.get("hits", 0)) > 0
                    or (
                        ln.get("condition-coverage")
                        and not ln.get("condition-coverage").startswith("0%")
                    )
                )
            )
        else:
            lines_valid = 0
            lines_covered = 0
            branches_valid = 0
            branches_covered = 0

        methods_elem = cls.find("methods")
        if methods_elem is not None:
            methods = methods_elem.findall("method")
            funcs_valid = len(methods)
            funcs_covered = sum(1 for m in methods if float(m.get("line-rate", 0.0)) > 0)
        else:
            funcs_valid = 0
            funcs_covered = 0

        files.append(
            (
                line_rate,
                branch_rate,
                lines_valid,
                lines_covered,
                funcs_valid,
                funcs_covered,
                branches_valid,
                branches_covered,
                filename,
            )
        )

    files.sort(key=lambda x: x[0])
    max_path_len = max(max((len(f[8]) for f in files), default=0), 20)

    print("=" * 80)
    print("LOWEST COVERED FILES (Priority for improvement)")
    print("=" * 80)
    for lr, br, lv, lc, fv, fc, bv, bc, path in files[:20]:
        lines_missing = lv - lc
        print(
            f"{lr*100:5.1f}% lines ({lc:4d}/{lv:4d}) | "
            f"{br*100:5.1f}% branches ({bc:3d}/{bv:3d}) | "
            f"{fc:3d}/{fv:3d} funcs | "
            f"{lines_missing:4d} lines to cover | "
            f"{path:<{max_path_len}}"
        )

    print("\n" + "=" * 80)
    print(f"FILES ABOVE {args.min_coverage*100:.0f}% (Good coverage)")
    print("=" * 80)
    for lr, _, lv, lc, _, _, _, _, path in files:
        if lr >= args.min_coverage:
            print(f"{lr*100:5.1f}% lines ({lc:4d}/{lv:4d}) | {path:<{max_path_len}}")

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total lines: {overall_lines_covered}/{overall_lines_valid} ({overall_line_rate*100:.1f}%)")
    print(
        f"Total branches: {overall_branches_covered}/{overall_branches_valid} "
        f"({overall_branch_rate*100:.1f}%)"
    )

    total_funcs = sum(f[4] for f in files)
    total_funcs_cov = sum(f[5] for f in files)
    if total_funcs > 0:
        print(f"Total functions: {total_funcs_cov}/{total_funcs} ({total_funcs_cov/total_funcs*100:.1f}%)")

    target_covered = int(overall_lines_valid * args.target_coverage)
    lines_needed = target_covered - overall_lines_covered
    print(
        f"\nTo reach {args.target_coverage*100:.0f}%: need {lines_needed} additional "
        f"lines covered (from {overall_line_rate*100:.1f}% to {args.target_coverage*100:.0f}%)"
    )

    files_needing_work = [
        (lr, lv, lc, path)
        for lr, _, lv, lc, _, _, _, _, path in files
        if lr < args.target_coverage
    ]
    files_needing_work.sort(key=lambda x: x[1] * (args.target_coverage - x[0]), reverse=True)
    max_work_path_len = max(max((len(f[3]) for f in files_needing_work), default=0), max_path_len)

    print("\n" + "=" * 80)
    print(f"FILES NEEDING WORK FOR {args.target_coverage*100:.0f}% TARGET")
    print("=" * 80)
    for lr, lv, lc, path in files_needing_work[:25]:
        target_lines = int(lv * args.target_coverage)
        lines_to_add = target_lines - lc
        impact = lv * (args.target_coverage - lr)
        print(f"{lr*100:5.1f}% -> need +{lines_to_add:3d} lines ({impact:6.0f} impact) | {path:<{max_work_path_len}}")

    return 0


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("xml_file", type=Path, help="Path to gcovr Cobertura XML report")
    parser.add_argument(
        "-m",
        "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum coverage percentage for 'good coverage' (default: 0.80)",
    )
    parser.add_argument(
        "-t",
        "--target-coverage",
        type=float,
        default=0.90,
        help="Target coverage percentage for improvement planning (default: 0.90)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s build-coverage/coverage.xml\n"
            "  %(prog)s report build-coverage/coverage.xml\n"
            "  %(prog)s normalize --xml build/coverage.xml --summary build/coverage-summary.json --source-root .\n"
            "  %(prog)s check build/coverage-summary.json --xml build/coverage.xml --min-line 0.90\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    report = subparsers.add_parser("report", help="summarize coverage and low-covered files")
    _add_report_args(report)
    report.set_defaults(func=command_report)

    normalize = subparsers.add_parser("normalize", help="apply hypredrive coverage policy")
    normalize.add_argument("--xml", required=True, type=Path, help="Path to gcovr Cobertura XML report")
    normalize.add_argument(
        "--summary", required=True, type=Path, help="Path to gcovr JSON summary report"
    )
    normalize.add_argument("--source-root", required=True, type=Path, help="Repository source root")
    normalize.set_defaults(func=command_normalize)

    check = subparsers.add_parser("check", help="fail if coverage is below thresholds")
    check.add_argument(
        "summary_json",
        nargs="?",
        default=Path("coverage-summary.json"),
        type=Path,
        help="Path to gcovr JSON summary report (default: coverage-summary.json)",
    )
    check.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Optional gcovr Cobertura XML report; line/branch rates are taken from XML.",
    )
    check.add_argument("--min-line", type=float, default=0.90, help="Minimum line coverage")
    check.add_argument("--min-branch", type=float, default=0.90, help="Minimum branch coverage")
    check.add_argument(
        "--min-function", type=float, default=0.95, help="Minimum function coverage"
    )
    check.set_defaults(func=command_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"report", "normalize", "check", "-h", "--help"}
    if argv and argv[0] not in commands:
        report_parser = argparse.ArgumentParser(
            description="Analyze code coverage from gcovr XML report",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_report_args(report_parser)
        args = report_parser.parse_args(argv)
        return command_report(args)

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2
    except (OSError, ET.ParseError, ValueError) as exc:
        sys.stderr.write(f"Error: coverage analysis failed: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
