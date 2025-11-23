#!/usr/bin/env python3
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

import xml.etree.ElementTree as ET
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Analyze code coverage from gcovr XML report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build-coverage/coverage.xml
  %(prog)s path/to/coverage.xml
        """
    )
    parser.add_argument(
        "xml_file",
        type=str,
        help="Path to coverage XML file (e.g., from gcovr --xml)"
    )
    parser.add_argument(
        "-m", "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum coverage percentage for 'good coverage' (default: 0.80)"
    )
    parser.add_argument(
        "-t", "--target-coverage",
        type=float,
        default=0.90,
        help="Target coverage percentage for improvement planning (default: 0.90)"
    )

    args = parser.parse_args()

    try:
        root = ET.parse(args.xml_file).getroot()
    except FileNotFoundError:
        sys.stderr.write(f"Error: Coverage XML file not found: {args.xml_file}\n")
        sys.exit(1)
    except ET.ParseError as e:
        sys.stderr.write(f"Error: Failed to parse XML file: {e}\n")
        sys.exit(1)

    # Get overall stats from root
    overall_line_rate = float(root.get("line-rate", 0.0))
    overall_lines_covered = int(root.get("lines-covered", 0))
    overall_lines_valid = int(root.get("lines-valid", 0))
    overall_branch_rate = float(root.get("branch-rate", 0.0))
    overall_branches_covered = int(root.get("branches-covered", 0))
    overall_branches_valid = int(root.get("branches-valid", 0))

    files = []
    for cls in root.findall(".//class"):
        filename = cls.get("filename")
        line_rate = float(cls.get("line-rate", 0.0))
        branch_rate = float(cls.get("branch-rate", 0.0))

        # Count lines and branches from <lines> elements
        lines_elem = cls.find("lines")
        if lines_elem is not None:
            line_nodes = lines_elem.findall("line")
            lines_valid = len(line_nodes)
            lines_covered = sum(1 for ln in line_nodes if int(ln.get("hits", 0)) > 0)

            # Count branches (lines with branch="true")
            branches_valid = sum(1 for ln in line_nodes if ln.get("branch") == "true")
            # Branches covered: those with condition-coverage > 0% or hits > 0
            branches_covered = sum(
                1 for ln in line_nodes
                if ln.get("branch") == "true" and (
                    int(ln.get("hits", 0)) > 0 or
                    (ln.get("condition-coverage") and not ln.get("condition-coverage").startswith("0%"))
                )
            )
        else:
            lines_valid = 0
            lines_covered = 0
            branches_valid = 0
            branches_covered = 0

        # Count functions (methods)
        methods_elem = cls.find("methods")
        if methods_elem is not None:
            methods = methods_elem.findall("method")
            funcs_valid = len(methods)
            funcs_covered = sum(1 for m in methods if float(m.get("line-rate", 0.0)) > 0)
        else:
            funcs_valid = 0
            funcs_covered = 0

        files.append((line_rate, branch_rate, lines_valid, lines_covered,
                      funcs_valid, funcs_covered, branches_valid, branches_covered, filename))

    files.sort(key=lambda x: x[0])

    # Calculate column widths for alignment
    max_path_len = max(len(f[8]) for f in files) if files else 0
    max_path_len = max(max_path_len, 20)  # Minimum width for path column

    print("=" * 80)
    print("LOWEST COVERED FILES (Priority for improvement)")
    print("=" * 80)
    for lr, br, lv, lc, fv, fc, bv, bc, path in files[:20]:
        lines_missing = lv - lc
        print(f"{lr*100:5.1f}% lines ({lc:4d}/{lv:4d}) | "
              f"{br*100:5.1f}% branches ({bc:3d}/{bv:3d}) | "
              f"{fc:3d}/{fv:3d} funcs | "
              f"{lines_missing:4d} lines to cover | "
              f"{path:<{max_path_len}}")

    print("\n" + "=" * 80)
    print(f"FILES ABOVE {args.min_coverage*100:.0f}% (Good coverage)")
    print("=" * 80)
    for lr, br, lv, lc, fv, fc, bv, bc, path in files:
        if lr >= args.min_coverage:
            print(f"{lr*100:5.1f}% lines ({lc:4d}/{lv:4d}) | {path:<{max_path_len}}")

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total lines: {overall_lines_covered}/{overall_lines_valid} ({overall_line_rate*100:.1f}%)")
    print(f"Total branches: {overall_branches_covered}/{overall_branches_valid} ({overall_branch_rate*100:.1f}%)")

    total_funcs = sum(f[4] for f in files)
    total_funcs_cov = sum(f[5] for f in files)
    if total_funcs > 0:
        print(f"Total functions: {total_funcs_cov}/{total_funcs} ({total_funcs_cov/total_funcs*100:.1f}%)")

    # Calculate lines needed for target coverage
    target_covered = int(overall_lines_valid * args.target_coverage)
    lines_needed = target_covered - overall_lines_covered
    print(f"\nTo reach {args.target_coverage*100:.0f}%: need {lines_needed} additional lines covered (from {overall_line_rate*100:.1f}% to {args.target_coverage*100:.0f}%)")

    # Identify files that need work (below target)
    files_needing_work = [(lr, lv, lc, path) for lr, _, lv, lc, _, _, _, _, path in files if lr < args.target_coverage]
    files_needing_work.sort(key=lambda x: x[1] * (args.target_coverage - x[0]), reverse=True)  # Sort by impact

    # Calculate max path length for files needing work
    max_work_path_len = max(len(f[3]) for f in files_needing_work) if files_needing_work else max_path_len
    max_work_path_len = max(max_work_path_len, max_path_len)

    print("\n" + "=" * 80)
    print(f"FILES NEEDING WORK FOR {args.target_coverage*100:.0f}% TARGET")
    print("=" * 80)
    for lr, lv, lc, path in files_needing_work[:25]:
        target_lines = int(lv * args.target_coverage)
        lines_to_add = target_lines - lc
        impact = lv * (args.target_coverage - lr)
        print(f"{lr*100:5.1f}% -> need +{lines_to_add:3d} lines ({impact:6.0f} impact) | {path:<{max_work_path_len}}")


if __name__ == "__main__":
    main()
