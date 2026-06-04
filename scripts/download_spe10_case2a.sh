#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/download_spe10_case2a.sh [destination]

Downloads and unpacks the SPE10 model 2 porosity/permeability archive.
The default destination is data/spe10_case2a, which is ignored by git.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

for tool in curl unzip; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Error: ${tool} is required" >&2
        exit 1
    fi
done

url="${SPE10_CASE2A_URL:-https://www.spe.org/web/csp/datasets/por_perm_case2a.zip}"
dest="${1:-data/spe10_case2a}"
tmpdir="$(mktemp -d)"
archive="${tmpdir}/por_perm_case2a.zip"

cleanup() {
    rm -rf "${tmpdir}"
}
trap cleanup EXIT

mkdir -p "${dest}"

echo "Downloading ${url}"
curl --fail --location --retry 3 --output "${archive}" "${url}"

echo "Unpacking into ${dest}"
unzip -o -q "${archive}" -d "${dest}"

if [[ ! -s "${dest}/spe_perm.dat" || ! -s "${dest}/spe_phi.dat" ]]; then
    echo "Error: expected spe_perm.dat and spe_phi.dat in ${dest}" >&2
    exit 1
fi

echo "SPE10 case 2a permeability: ${dest}/spe_perm.dat"
echo "SPE10 case 2a porosity:     ${dest}/spe_phi.dat"
