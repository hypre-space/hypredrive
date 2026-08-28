#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

# Maximum cyclomatic complexity allowed for a single function. Lowering this is
# a deliberate ratchet: drop it once the tree is comfortably below the new value
# so the gate keeps complexity from creeping back up.
DEFAULT_THRESHOLD=20

ROOT_DIR=""
THRESHOLD="${DEFAULT_THRESHOLD}"
REPORT_DIR=""
QUIET=0
PATHS=()

usage() {
  cat <<'EOF'
Usage: scripts/check_complexity.sh [--root <repo>] [--threshold <N>]
                                   [--report <dir>] [--quiet] [-- <paths>...]

Tracks per-function cyclomatic complexity (CCN) with lizard and fails when any
function exceeds the allowed threshold.

Defaults:
  --root       repository root inferred from this script's location
  --threshold  20 (see DEFAULT_THRESHOLD in this script)
  paths        src (all C sources under the library)

Options:
  --root <repo>       Repository root to analyze.
  --threshold <N>     Fail if any function has CCN > N.
  --report <dir>      Write complexity.csv and complexity.xml into <dir>.
  --quiet             Only print violations and the final verdict.
  -- <paths>...       Analyze these paths instead of the default.

Exit status:
  0 if every function is within the threshold
  1 if one or more functions exceed it
  2 for script usage / environment errors (e.g. lizard not installed)

Requires: lizard (pip install lizard) and python3.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)      [ $# -ge 2 ] || die "--root requires a value";      ROOT_DIR="$2"; shift 2 ;;
    --threshold) [ $# -ge 2 ] || die "--threshold requires a value"; THRESHOLD="$2"; shift 2 ;;
    --report)    [ $# -ge 2 ] || die "--report requires a value";    REPORT_DIR="$2"; shift 2 ;;
    --quiet)     QUIET=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    --)          shift; PATHS=("$@"); break ;;
    *)           usage >&2; die "unknown argument: $1" ;;
  esac
done

case "${THRESHOLD}" in
  ''|*[!0-9]*) die "--threshold must be a positive integer (got '${THRESHOLD}')" ;;
esac

if [ -z "${ROOT_DIR}" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
[ -d "${ROOT_DIR}" ] || die "root directory not found: ${ROOT_DIR}"
cd "${ROOT_DIR}"

if [ ${#PATHS[@]} -eq 0 ]; then
  PATHS=(src)
fi
for p in "${PATHS[@]}"; do
  [ -e "${p}" ] || die "path not found: ${p}"
done

# lizard ships both a console script and a runnable module; accept either.
if command -v lizard > /dev/null 2>&1; then
  LIZARD=(lizard)
elif python3 -c "import lizard" > /dev/null 2>&1; then
  LIZARD=(python3 -m lizard)
else
  die "lizard is not installed. Install it with: python3 -m pip install lizard"
fi

CSV_FILE="$(mktemp)"
trap 'rm -f "${CSV_FILE}"' EXIT

# `-l c` keeps the C parser from being applied to unrelated languages; the CSV
# carries every function so the summary below never re-runs the analysis.
if ! "${LIZARD[@]}" "${PATHS[@]}" -l c --csv > "${CSV_FILE}" 2> /dev/null; then
  die "lizard failed to analyze: ${PATHS[*]}"
fi

if [ -n "${REPORT_DIR}" ]; then
  mkdir -p "${REPORT_DIR}"
  cp "${CSV_FILE}" "${REPORT_DIR}/complexity.csv"
  chmod 0644 "${REPORT_DIR}/complexity.csv"
  "${LIZARD[@]}" "${PATHS[@]}" -l c --xml > "${REPORT_DIR}/complexity.xml" 2> /dev/null || true
fi

QUIET="${QUIET}" THRESHOLD="${THRESHOLD}" python3 - "${CSV_FILE}" <<'PY'
import csv, os, sys

threshold = int(os.environ["THRESHOLD"])
quiet = os.environ["QUIET"] == "1"

rows = []
with open(sys.argv[1], newline="") as fh:
    for r in csv.reader(fh):
        if len(r) < 11:
            continue
        rows.append({
            "nloc": int(r[0]), "ccn": int(r[1]), "params": int(r[3]),
            "file": r[6], "func": r[7], "start": r[9],
        })

if not rows:
    print("ERROR: lizard reported no functions; check the analyzed paths", file=sys.stderr)
    raise SystemExit(2)

ccns = [r["ccn"] for r in rows]
violations = sorted((r for r in rows if r["ccn"] > threshold),
                    key=lambda r: -r["ccn"])

if not quiet:
    print(f"Analyzed {len(rows)} functions (threshold: CCN <= {threshold})")
    print(f"  mean CCN {sum(ccns) / len(ccns):.2f}   max CCN {max(ccns)}"
          f"   max NLOC {max(r['nloc'] for r in rows)}"
          f"   max params {max(r['params'] for r in rows)}")
    for t in (10, 15, 20, threshold):
        print(f"  CCN > {t:<3} {sum(1 for c in ccns if c > t):>5} functions")

    worst = sorted(rows, key=lambda r: -r["ccn"])[:10]
    print("\nMost complex functions:")
    print(f"  {'CCN':>4}  {'NLOC':>5}  LOCATION")
    for r in worst:
        print(f"  {r['ccn']:>4}  {r['nloc']:>5}  {r['file']}:{r['start']} {r['func']}")

if violations:
    print(f"\nERROR: {len(violations)} function(s) exceed CCN {threshold}:", file=sys.stderr)
    for r in violations:
        print(f"  {r['file']}:{r['start']}: {r['func']} has CCN {r['ccn']}", file=sys.stderr)
    print("\nSplit these into smaller helpers, or raise the threshold in "
          "scripts/check_complexity.sh if the increase is intentional.", file=sys.stderr)
    raise SystemExit(1)

print(f"\ncomplexity: OK (no function exceeds CCN {threshold})")
PY
