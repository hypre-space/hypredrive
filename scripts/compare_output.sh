#!/bin/bash
# Compare actual output against reference output
# This script normalizes timestamps and paths before comparison

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <actual_output> <reference_output>"
    exit 1
fi

ACTUAL="$1"
REFERENCE="$2"

if [ ! -f "$ACTUAL" ]; then
    echo "Error: Actual output file '$ACTUAL' not found"
    exit 1
fi

if [ ! -f "$REFERENCE" ]; then
    echo "Error: Reference output file '$REFERENCE' not found"
    exit 1
fi

# Create temporary files for normalized output
ACTUAL_NORM=$(mktemp)
REFERENCE_NORM=$(mktemp)
trap "rm -f '$ACTUAL_NORM' '$REFERENCE_NORM'" EXIT

# Normalize the output files
# 1. Replace date/time patterns
# 2. Replace HYPRE version strings
# 3. Replace paths
# 4. Replace HYPREDRIVE_PATH placeholder
sed -E \
    -e 's/[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/YYYY-MM-DD HH:MM:SS/g' \
    -e 's/HYPRE_[A-Z_]*: [^[:space:]]*/HYPRE_VERSION_GOES_HERE/g' \
    -e 's|[/a-zA-Z0-9_.-]+/hypredrive|\${HYPREDRIVE_PATH}/hypredrive|g' \
    -e 's|Date and time: YYYY-MM-DD HH:MM:SS|Date and time: YYYY-MM-DD HH:MM:SS|g' \
    "$ACTUAL" > "$ACTUAL_NORM"

sed -E \
    -e 's/[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}/YYYY-MM-DD HH:MM:SS/g' \
    -e 's/HYPRE_[A-Z_]*: [^[:space:]]*/HYPRE_VERSION_GOES_HERE/g' \
    -e 's|[/a-zA-Z0-9_.-]+/hypredrive|\${HYPREDRIVE_PATH}/hypredrive|g' \
    -e 's|Date and time: YYYY-MM-DD HH:MM:SS|Date and time: YYYY-MM-DD HH:MM:SS|g' \
    "$REFERENCE" > "$REFERENCE_NORM"

# Compare normalized outputs
if diff -u "$REFERENCE_NORM" "$ACTUAL_NORM" > /dev/null; then
    echo "Output matches reference"
    exit 0
else
    echo "Output differs from reference:"
    diff -u "$REFERENCE_NORM" "$ACTUAL_NORM" || true
    exit 1
fi

