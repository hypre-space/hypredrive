#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/
#
# Dry-run portability check for every script shipped in the repository.
#
# For the current operating system it verifies that:
#   * every shell script parses                 (bash -n)
#   * every shell CLI answers --help            (exit 0, no real run)
#   * every Python script byte-compiles         (python3 -m py_compile)
#   * every standalone Python CLI answers --help (exit 0)
#
# It performs no heavy work: no MPI launches, no builds, no rendering. The
# example postprocess scripts import pyvista/imageio lazily, so --help only needs
# lightweight deps (numpy, matplotlib, pandas, plotly, anytree, pyyaml).
#
# Scripts that import the compiled `hypredrive` module or `mpi4py` at top level
# are byte-compiled but their --help is skipped here: running them requires the
# built+installed package, which the dedicated Python-interface / wheel CI covers.
#
# This backs the "Code Portability / Linux" and "Code Portability / MacOS" CI
# jobs and is equally meant to be run by hand:
#
#     bash scripts/check_portability.sh
#
# Override the interpreter with PYTHON=/path/to/python3. Exit status is non-zero
# if any check fails. Written to run under bash 3.2 (macOS) with a BSD userland.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
OS="$(uname -s)"

# Keep byte-compiled artifacts out of the source tree (Python 3.8+). Cleaned on exit.
PYC_DIR="${TMPDIR:-/tmp}/hypredrive-portability-pyc.$$"
export PYTHONPYCACHEPREFIX="$PYC_DIR"
trap 'rm -rf "$PYC_DIR"' EXIT

# Scripts whose runtime is intentionally Linux-only (they drive HPC schedulers
# such as Slurm via srun/sbatch). Skipped entirely off Linux.
is_linux_only() {
    case "$1" in
        scripts/job_launcher.py) return 0 ;;
        *) return 1 ;;
    esac
}

pass=0
fail=0
skip=0
failed_list=""

ok()      { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad()     { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); failed_list="${failed_list}
  - $1"; }
skipped() { printf '  SKIP  %s\n' "$1"; skip=$((skip + 1)); }

# run_check <label> <command...>: run capturing output; PASS on exit 0, else FAIL
# with the captured output indented for context.
run_check() {
    local label="$1"; shift
    local output
    if output="$("$@" 2>&1)"; then
        ok "$label"
    else
        bad "$label"
        printf '%s\n' "$output" | sed 's/^/        | /'
    fi
}

# list_scripts <ext>: tracked scripts of the given extension (falls back to find
# outside a git checkout). Using the git index keeps build trees, fetched deps,
# and stray nested checkouts out of scope.
list_scripts() {
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git ls-files -- "*.$1"
    else
        find . -type f -name "*.$1" \
            -not -path './.git/*' -not -path './build*' -not -path './install*' \
            -not -path '*/_deps/*' -not -path './hypre/*' -not -path './hypredrive/*' \
            -not -path '*/__pycache__/*' | sed 's|^\./||'
    fi
}

has_argparse()    { grep -qE 'import argparse|from argparse' "$1"; }
# Top-level (column 0) import of a build/MPI-coupled module.
needs_built_pkg() { grep -qE '^(import|from)[[:space:]]+(hypredrive|mpi4py)\b' "$1"; }

printf '=== Code Portability dry-run ===\n'
printf 'OS      : %s\n' "$(uname -srm)"
printf 'bash    : %s\n' "${BASH_VERSION:-unknown}"
printf 'python  : %s\n' "$($PYTHON --version 2>&1)"

printf '\n-- Shell scripts: syntax (bash -n) --\n'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    run_check "bash -n      $f" bash -n "$f"
done < <(list_scripts sh)

printf '\n-- reproduce.sh: --help smoke (exit 0, no run) --\n'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in */reproduce.sh) ;; *) continue ;; esac
    run_check "--help       $f" bash "$f" --help
done < <(list_scripts sh)

printf '\n-- Python scripts: byte-compile (py_compile) --\n'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    if [ "$OS" != "Linux" ] && is_linux_only "$f"; then
        skipped "py_compile   $f (Linux-only)"; continue
    fi
    run_check "py_compile   $f" "$PYTHON" -m py_compile "$f"
done < <(list_scripts py)

printf '\n-- Python shebangs: portable interpreter (env python3, not a python2 path) --\n'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    line1="$(head -1 "$f")"
    case "$line1" in
        '#!'*) ;;        # has a shebang to validate
        *) continue ;;   # library module without a shebang
    esac
    case "$line1" in
        *python2*|*/python)
            bad "shebang      $f  ('$line1' -> use '#!/usr/bin/env python3')" ;;
        *)
            ok "shebang      $f  ($line1)" ;;
    esac
done < <(list_scripts py)

printf '\n-- Python CLIs: --help smoke (exit 0) --\n'
while IFS= read -r f; do
    [ -n "$f" ] || continue
    has_argparse "$f" || continue
    if [ "$OS" != "Linux" ] && is_linux_only "$f"; then
        skipped "--help       $f (Linux-only)"; continue
    fi
    if needs_built_pkg "$f"; then
        skipped "--help       $f (needs built hypredrive/mpi4py)"; continue
    fi
    run_check "--help       $f" "$PYTHON" "$f" --help
done < <(list_scripts py)

printf '\n=== Summary: %d passed, %d failed, %d skipped ===\n' "$pass" "$fail" "$skip"
if [ "$fail" -ne 0 ]; then
    printf 'Failed checks:%s\n' "$failed_list"
    exit 1
fi
printf 'All script portability checks passed.\n'
exit 0
