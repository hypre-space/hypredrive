#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

# --------------------------------------------------------------------------
# check_symbol_prefix.sh
#
# Verify that every *defined, global* symbol in libHYPREDRV uses one of the
# approved prefixes (HYPREDRV_ or hypredrv_).  This catches issues the
# source-level check_private_prefix.sh cannot: macro-generated symbols,
# global data, and anything else the linker exposes.
#
# Usage:
#   scripts/check_symbol_prefix.sh <path-to-libHYPREDRV.a | .so>
#   scripts/check_symbol_prefix.sh --build-dir <cmake-build-dir>
#
# Exit codes:
#   0  all symbols are properly prefixed
#   1  violations found
#   2  usage / environment error
# --------------------------------------------------------------------------

QUIET=0
LIB_PATH=""
BUILD_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/check_symbol_prefix.sh [OPTIONS] [<library>]

Check that all defined global symbols in libHYPREDRV use proper prefixes.

Positional:
  <library>        Path to libHYPREDRV.a or libHYPREDRV.so*

Options:
  --build-dir DIR  CMake build directory (auto-discovers the library)
  --quiet          Suppress informational output; only print violations
  -h, --help       Show this help

Approved prefixes:
  HYPREDRV_    (public API)
  hypredrv_    (internal symbols)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --build-dir" >&2; exit 2; }
      BUILD_DIR="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      LIB_PATH="$1"
      shift
      ;;
  esac
done

# --- Resolve library path ------------------------------------------------

if [[ -n "$BUILD_DIR" && -z "$LIB_PATH" ]]; then
  for candidate in \
      "${BUILD_DIR}/lib/libHYPREDRV.a" \
      "${BUILD_DIR}/libHYPREDRV.a" \
      "${BUILD_DIR}/lib/libHYPREDRV.so" \
      "${BUILD_DIR}/libHYPREDRV.so"; do
    if [[ -f "$candidate" ]]; then
      LIB_PATH="$candidate"
      break
    fi
  done
  if [[ -z "$LIB_PATH" ]]; then
    # Try glob for versioned .so
    LIB_PATH="$(find "$BUILD_DIR" -maxdepth 2 -name 'libHYPREDRV.so*' -type f 2>/dev/null | head -1 || true)"
  fi
fi

if [[ -z "$LIB_PATH" || ! -f "$LIB_PATH" ]]; then
  echo "ERROR: Cannot find libHYPREDRV library." >&2
  echo "Provide a path or use --build-dir <dir>." >&2
  exit 2
fi

if ! command -v nm &>/dev/null; then
  echo "ERROR: 'nm' not found in PATH." >&2
  exit 2
fi

[[ "$QUIET" -eq 0 ]] && echo "Checking symbols in: ${LIB_PATH}"

# --- Extract defined global symbols --------------------------------------
#
# nm output format:  <addr> <type> <name>
#   Uppercase type letter = global (external linkage)
#   We want: T (text), D (data), B (bss), R (rodata), S (small objects)
#   We skip: U (undefined/imported), lowercase (local/static), W/w (weak)
#
# For .so files, use -D to check dynamic symbols (what's actually exported).
# For .a files, use plain nm (all global symbols).

NM_FLAGS=""
if [[ "$LIB_PATH" == *.so* ]]; then
  NM_FLAGS="-D"
fi

# Extract global defined symbols, one per line: "name"
mapfile -t SYMBOLS < <(
  nm $NM_FLAGS "$LIB_PATH" 2>/dev/null \
    | grep -E '^[0-9a-f]+ [TDBRS] ' \
    | awk '{print $3}' \
    | LC_ALL=C sort -u
)

if [[ "${#SYMBOLS[@]}" -eq 0 ]]; then
  [[ "$QUIET" -eq 0 ]] && echo "No global defined symbols found (empty library?)."
  exit 0
fi

[[ "$QUIET" -eq 0 ]] && echo "Found ${#SYMBOLS[@]} global defined symbol(s)."

# --- Check each symbol against approved prefixes --------------------------

declare -a VIOLATIONS=()

for sym in "${SYMBOLS[@]}"; do
  case "$sym" in
    HYPREDRV_*|hypredrv_*)
      # Properly prefixed
      ;;
    __odr_asan.*|__asan_*|__ubsan_*|__tsan_*|__msan_*)
      # Compiler sanitizer instrumentation symbols
      ;;
    *)
      VIOLATIONS+=("$sym")
      ;;
  esac
done

# --- Report ---------------------------------------------------------------

if [[ "${#VIOLATIONS[@]}" -gt 0 ]]; then
  echo
  echo "Symbol prefix violations (${#VIOLATIONS[@]}):"
  for v in "${VIOLATIONS[@]}"; do
    # Try to locate the symbol's origin object file (static archives only)
    origin=""
    if [[ "$LIB_PATH" == *.a ]]; then
      origin="$(nm -A "$LIB_PATH" 2>/dev/null \
                | grep -E " [TDBRS] ${v}$" \
                | head -1 \
                | sed 's/.*:\(.*\.o\):.*/\1/' || true)"
    fi
    if [[ -n "$origin" ]]; then
      echo "  - ${v}  (from ${origin})"
    else
      echo "  - ${v}"
    fi
  done
  echo
  echo "FAIL: All global symbols must use HYPREDRV_ or hypredrv_ prefix."
  exit 1
fi

[[ "$QUIET" -eq 0 ]] && echo "OK: all global symbols are properly prefixed."
exit 0
