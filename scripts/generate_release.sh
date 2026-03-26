#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/
#
# generate_release.sh — Bump version strings and optionally tag a release.
#
# Usage:
#   scripts/generate_release.sh <NEW_VERSION> [--tag] [--dry-run]
#
# Arguments:
#   NEW_VERSION   SemVer string, e.g. 0.2.0
#   --tag         Create and push an annotated git tag after updating files
#   --dry-run     Print what would change without modifying any file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -- helpers ----------------------------------------------------------------

usage() {
  sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

die() { echo "ERROR: $*" >&2; exit 1; }

info() { echo "  $*"; }

# -- argument parsing --------------------------------------------------------

NEW_VERSION=""
DO_TAG=""
DRY_RUN=""

for arg in "$@"; do
  case "$arg" in
    --tag)     DO_TAG=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) usage ;;
    -*)        die "Unknown option: $arg" ;;
    *)
      [[ -n "$NEW_VERSION" ]] && die "Unexpected extra argument: $arg"
      NEW_VERSION="$arg"
      ;;
  esac
done

[[ -z "$NEW_VERSION" ]] && { echo "Error: NEW_VERSION is required."; usage; }

# Validate SemVer (MAJOR.MINOR.PATCH, no pre-release suffix required for now)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  die "NEW_VERSION must be MAJOR.MINOR.PATCH (e.g. 0.2.0), got: $NEW_VERSION"
fi

# -- locate files ------------------------------------------------------------

CMAKE_FILE="${REPO_ROOT}/CMakeLists.txt"
CONF_PY="${REPO_ROOT}/docs/usrman-src/conf.py"
CHANGELOG="${REPO_ROOT}/CHANGELOG"

for f in "$CMAKE_FILE" "$CONF_PY" "$CHANGELOG"; do
  [[ -f "$f" ]] || die "Expected file not found: $f"
done

# Detect current version from CMakeLists.txt
CURRENT_VERSION=$(grep -m1 '^project(hypredrive VERSION' "$CMAKE_FILE" \
  | sed 's/.*VERSION \([0-9][0-9.]*\).*/\1/')
[[ -z "$CURRENT_VERSION" ]] && die "Could not detect current version in $CMAKE_FILE"

echo
echo "Releasing hypredrive ${CURRENT_VERSION} → ${NEW_VERSION}${DRY_RUN:+ (dry run)}"
echo

# -- check CHANGELOG has an entry for the new version -----------------------

TODAY=$(date +%Y/%m/%d)
CHANGELOG_HEADER="Version ${NEW_VERSION} released ${TODAY}"

if ! grep -q "^Version ${NEW_VERSION} released" "$CHANGELOG"; then
  if [[ -n "$DRY_RUN" ]]; then
    info "CHANGELOG: would prepend '${CHANGELOG_HEADER}' entry (none found)"
  else
    die "CHANGELOG has no entry for Version ${NEW_VERSION}. Add release notes first."
  fi
fi

# -- apply changes -----------------------------------------------------------

apply_sed() {
  local file="$1" pattern="$2" replacement="$3"
  if [[ -n "$DRY_RUN" ]]; then
    info "$(basename "$file"): s|${pattern}|${replacement}|"
  else
    sed -i "s|${pattern}|${replacement}|" "$file"
    info "$(basename "$file"): updated"
  fi
}

# 1. CMakeLists.txt
apply_sed "$CMAKE_FILE" \
  "project(hypredrive VERSION ${CURRENT_VERSION} LANGUAGES C)" \
  "project(hypredrive VERSION ${NEW_VERSION} LANGUAGES C)"

# 2. conf.py fallback release/version strings
apply_sed "$CONF_PY" \
  "os.environ.get('HYPREDRV_DOCS_RELEASE', '[^']*')" \
  "os.environ.get('HYPREDRV_DOCS_RELEASE', '${NEW_VERSION}')"

echo

# -- git tag -----------------------------------------------------------------

if [[ -n "$DO_TAG" ]]; then
  if [[ -n "$DRY_RUN" ]]; then
    info "git: would create annotated tag v${NEW_VERSION} and push"
  else
    cd "$REPO_ROOT"
    git add "$CMAKE_FILE" "$CONF_PY"
    git commit -m "Bump version to ${NEW_VERSION}"
    git tag -a "v${NEW_VERSION}" -m "hypredrive ${NEW_VERSION}"
    echo "  Tagged v${NEW_VERSION}. Push with: git push origin v${NEW_VERSION}"
  fi
else
  if [[ -z "$DRY_RUN" ]]; then
    echo "Files updated. Next steps:"
    echo "  1. Update CHANGELOG with release notes for ${NEW_VERSION}"
    echo "  2. Commit: git add CMakeLists.txt docs/usrman-src/conf.py CHANGELOG"
    echo "  3. Tag:    git tag -a v${NEW_VERSION} -m 'hypredrive ${NEW_VERSION}'"
    echo "  4. Push:   git push && git push origin v${NEW_VERSION}"
  fi
fi
