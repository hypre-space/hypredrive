#!/usr/bin/env bash
set -euo pipefail

: "${HYPRE_PREFIX:?Set HYPRE_PREFIX to the HYPRE installation shared by both builds}"
repo=${GITHUB_REPOSITORY:-hypre-space/hypredrive}
tag=$(gh release view --repo "$repo" --json tagName --jq .tagName)
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]

scratch=$(mktemp -d "${RUNNER_TEMP:-/tmp}/hypredrive-abi-XXXXXXXX")
baseline="$scratch/baseline"
mkdir "$baseline"
gh api "repos/$repo/tarball/$tag" | tar -xz --strip-components=1 -C "$baseline"

build_library() {
    cmake -S "$1" -B "$2" -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=mpicc -DCMAKE_PREFIX_PATH="$HYPRE_PREFIX" \
        -DBUILD_SHARED_LIBS=ON
    cmake --build "$2" --target HYPREDRV --parallel
}

build_library "$baseline" "$scratch/released"
build_library "$PWD" "$scratch/candidate"

released_abi=$(sed -En 's/^set\(HYPREDRV_ABI_VERSION ([0-9]+)\)$/\1/p' "$baseline/cmake/HYPREDRV_ABI.cmake")
candidate_abi=$(sed -En 's/^set\(HYPREDRV_ABI_VERSION ([0-9]+)\)$/\1/p' cmake/HYPREDRV_ABI.cmake)
[[ "$released_abi" =~ ^[0-9]+$ && "$candidate_abi" =~ ^[0-9]+$ ]]

# Public headers declare opaque handles. Ignore layouts defined only in
# implementation files while comparing exported functions and public types.
status=0
abidiff --no-added-syms --no-linkage-name --drop-private-types \
    --header-file1 "$baseline/include/HYPREDRV.h" \
    --header-file1 "$baseline/include/HYPREDRV_utils.h" \
    --header-file2 "$PWD/include/HYPREDRV.h" \
    --header-file2 "$PWD/include/HYPREDRV_utils.h" \
    "$scratch/released/lib/libHYPREDRV.so" "$scratch/candidate/lib/libHYPREDRV.so" || status=$?
if (( status & 3 )); then
    exit "$status"
fi
if (( status & 12 )) && (( candidate_abi <= released_abi )); then
    echo "::error::Review the public ABI changes against $tag and increment HYPREDRV_ABI_VERSION if incompatible."
    exit 1
fi
