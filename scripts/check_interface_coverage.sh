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

APIS=$(bash scripts/list_public_apis.sh --root . | sed -n '/^HYPREDRV_/p')
[[ -n "$APIS" ]] || { echo "No public APIs found" >&2; exit 2; }

# Strip comments so that documentation mentions do not count as bindings.
strip_c()   { sed -E 's://.*$::' "$@" | perl -0pe 's:/\*.*?\*/::gs'; }
strip_f90() { sed -E 's:![^'"'"'"]*$::' "$@"; }
strip_hash(){ sed -E 's:(^|[^"'"'"'])#.*$:\1:' "$@"; }

text_of() {
  case "$1" in
    cpp)     strip_c interfaces/cpp/include/*.hpp ;;
    fortran) strip_f90 interfaces/fortran/src/*.f90 ;;
    python)  strip_c interfaces/python/src/hypredrive/*.c
             strip_hash interfaces/python/src/hypredrive/*.pyx ;;
    rust)    strip_c interfaces/rust/src/*.rs ;;
    julia)   strip_c interfaces/julia/src/*.c; strip_hash interfaces/julia/src/*.jl ;;
    matlab)  strip_c interfaces/matlab-octave/src/*.c ;;
  esac
}

INTERFACES=(cpp fortran python rust julia matlab)
MISSING=""
for iface in "${INTERFACES[@]}"; do
  # Python and Rust declarations alone do not make a callable binding.
  prefix='\b'
  [[ "$iface" != rust ]] || prefix='ffi::'
  bound=$(text_of "$iface" | { grep -oE "${prefix}HYPREDRV_[A-Za-z0-9_]+" || [[ $? -eq 1 ]]; } |
    sed 's/^ffi:://' | LC_ALL=C sort -u)
  missing=$(LC_ALL=C comm -23 <(printf '%s\n' "$APIS") <(printf '%s\n' "$bound"))
  [[ -n "$missing" ]] || continue
  while IFS= read -r api; do
    MISSING+="  $iface $api"$'\n'
  done <<< "$missing"
done

if [[ -n "$MISSING" ]]; then
  echo "Public APIs missing from an interface (every HYPREDRV_ API must be bound in every interface):" >&2
  printf '%s' "$MISSING" >&2
  exit 1
fi
echo "Interface coverage OK: all public APIs are exposed by ${INTERFACES[*]}"
