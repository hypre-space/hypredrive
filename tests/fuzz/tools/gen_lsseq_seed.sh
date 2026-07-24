#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <hypredrive-cli> <hypredrive-lsseq> <output.bin>" >&2
    exit 2
fi

cli=$1
packer=$2
output=$3
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/hypredrv-lsseq-seed.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT

"$cli" --matrix-filename "IJ.out.A" --rhs-filename "IJ.out.b" \
    "${HYPREDRIVE_SOURCE_DIR:-.}/examples/ex7.yml" >/dev/null

"$packer" pack --input-dir . --output "$tmpdir/seed.bin" >/dev/null
cp "$tmpdir/seed.bin" "$output"
