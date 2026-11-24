#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

ROOT_DIR=""

usage() {
  cat <<'EOF2'
Usage: scripts/check_interface_coverage.sh [--root <repo>]

Checks that every public HYPREDRV_ API in include/HYPREDRV.h is exposed by each
language interface (C++, Fortran, Python, Rust, Julia, MATLAB/Octave). A new
public API must be bound in all of them before it can be merged. Comments and
documentation mentions do not count as bindings.

Options:
  --root <repo>  Repository root (default: git rev-parse --show-toplevel)
EOF2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) [[ $# -ge 2 ]] || { echo "Missing value for --root" >&2; exit 2; }
            ROOT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ROOT_DIR" ]] || ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT_DIR" && -d "$ROOT_DIR" ]] || { echo "Pass --root <repo>." >&2; exit 2; }
cd "$ROOT_DIR"

APIS=$(bash scripts/list_public_apis.sh --root "$ROOT_DIR" | grep -E '^HYPREDRV_' || true)
[[ -n "$APIS" ]] || { echo "No public APIs found" >&2; exit 2; }

# Strip comments so that documentation mentions do not count as bindings.
strip_c()   { sed -E 's://.*$::' "$@" | perl -0pe 's:/\*.*?\*/::gs'; }
strip_f90() { sed -E 's:![^'"'"'"]*$::' "$@"; }
strip_hash(){ sed -E 's:(^|[^"'"'"'])#.*$:\1:' "$@"; }

# interface name | source files (space separated) | how a binding for API <name> looks
# The pattern receives the full API (HYPREDRV_Foo) and the short name (Foo).
INTERFACES=(cpp fortran python rust julia matlab)
sources() {
  case "$1" in
    cpp)     ls interfaces/cpp/include/*.hpp ;;
    fortran) ls interfaces/fortran/src/*.f90 ;;
    python)  ls interfaces/python/src/hypredrive/*.pxd interfaces/python/src/hypredrive/*.pyx ;;
    rust)    ls interfaces/rust/build.rs interfaces/rust/src/*.rs ;;
    julia)   ls interfaces/julia/src/*.c interfaces/julia/src/*.jl ;;
    matlab)  ls interfaces/matlab-octave/src/*.c ;;
  esac
}
text_of() {
  # shellcheck disable=SC2046
  case "$1" in
    cpp|matlab) strip_c $(sources "$1") ;;
    fortran)    strip_f90 $(sources "$1") ;;
    python)     strip_hash $(sources "$1") ;;
    rust)       sed -E 's:^[[:space:]]*//.*$::' $(sources "$1") ;;
    julia)      { strip_c $(ls interfaces/julia/src/*.c); strip_hash $(ls interfaces/julia/src/*.jl); } ;;
  esac
}
pattern_for() {
  local iface="$1" api="$2" short="${2#HYPREDRV_}"
  case "$iface" in
    cpp|matlab|python|fortran) echo "\\b${api}\\b" ;;
    rust)  echo "(ffi::${api}\\b|\"${short}\")" ;;     # call site or bindgen allowlist
    julia) echo "\\b(HYPREDRV_Julia${short}|${api})\\b" ;;
  esac
}

CURRENT=$(mktemp); TEXTFILE=$(mktemp); trap 'rm -f "$CURRENT" "$TEXTFILE"' EXIT
for iface in "${INTERFACES[@]}"; do
  # Grep a file (not a pipe): with pipefail, grep -q exiting early would SIGPIPE
  # the writer and report spurious misses.
  text_of "$iface" > "$TEXTFILE"
  while IFS= read -r api; do
    pat=$(pattern_for "$iface" "$api")
    grep -Eq "$pat" "$TEXTFILE" || echo "$iface $api" >> "$CURRENT"
  done <<< "$APIS"
done
LC_ALL=C sort -o "$CURRENT" "$CURRENT" 2>/dev/null || true
touch "$CURRENT"

if [[ -s "$CURRENT" ]]; then
  echo "Public APIs missing from an interface (every HYPREDRV_ API must be bound in every interface):" >&2
  sed 's/^/  /' "$CURRENT" >&2
  exit 1
fi
echo "Interface coverage OK: all public APIs are exposed by ${INTERFACES[*]}"
