#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

ROOT_DIR=""
CHECK_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/list_public_apis.sh [--root <repo>] [--check]

Generates a sorted list of all public HYPREDRV_ API function names by parsing
include/HYPREDRV.h for function declarations. Output is one name per line,
followed by a summary line.

Options:
  --root <repo>  Repository root (default: git rev-parse --show-toplevel)
  --check        Validate that all public APIs start with HYPREDRV_ and
                 accept a HYPREDRV_t parameter (with a small exempt list
                 for global/utility functions). Exit 1 on any violation.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "Missing value for --root" >&2; exit 2; }
      ROOT_DIR="$2"
      shift 2
      ;;
    --check)
      CHECK_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi

if [[ -z "$ROOT_DIR" || ! -d "$ROOT_DIR" ]]; then
  echo "Could not determine repository root. Pass --root <repo>." >&2
  exit 2
fi

HEADER="${ROOT_DIR}/include/HYPREDRV.h"
if [[ ! -f "$HEADER" ]]; then
  echo "Header not found: $HEADER" >&2
  exit 2
fi

APIS=$(grep -oE 'HYPREDRV_[A-Za-z0-9_]+ *\(' "$HEADER" \
  | sed 's/ *($//' \
  | grep -v -E '^HYPREDRV_(SAFE_CALL|operation)$' \
  | sort -u)

if [[ $CHECK_MODE -eq 1 ]]; then
  VIOLATIONS=0

  # 1. Every public API must start with HYPREDRV_
  while IFS= read -r api; do
    [[ -z "$api" ]] && continue
    if [[ "$api" != HYPREDRV_* ]]; then
      echo "Violation: public API must start with HYPREDRV_: $api" >&2
      VIOLATIONS=$((VIOLATIONS + 1))
    fi
  done <<< "$APIS"

  # 2. Every public API must accept a HYPREDRV_t parameter, except for a small
  #    set of global/utility functions that legitimately operate without one.
  EXEMPT_RE='^HYPREDRV_(Initialize|Finalize|ErrorCodeDescribe|ErrorInvalidValue|Create|PrintLibInfo|PrintSystemInfo|PrintExitInfo|PreconPresetRegister|SolverPresetRegister)$'
  while IFS= read -r api; do
    [[ -z "$api" ]] && continue
    echo "$api" | grep -qE "$EXEMPT_RE" && continue
    # Find the declaration line (skip Doxygen comment lines that reference the API)
    DECL_LINE=$(grep -nE "${api} *\(" "$HEADER" \
      | grep -v ':[[:space:]]*\*' \
      | head -1 | cut -d: -f1)
    [[ -z "$DECL_LINE" ]] && continue
    # Check the declaration line and the next for HYPREDRV_t (handles multi-line signatures)
    SIG=$(sed -n "${DECL_LINE},$((DECL_LINE + 1))p" "$HEADER")
    if ! echo "$SIG" | grep -q 'HYPREDRV_t'; then
      echo "Violation: $api must accept a HYPREDRV_t parameter" >&2
      VIOLATIONS=$((VIOLATIONS + 1))
    fi
  done <<< "$APIS"

  COUNT=$(echo "$APIS" | grep -c . || true)
  echo "Total: $COUNT public APIs"
  if [[ $VIOLATIONS -gt 0 ]]; then
    exit 1
  fi
  exit 0
fi

echo "$APIS"
COUNT=$(echo "$APIS" | grep -c . || true)
echo "Total: $COUNT public APIs"
