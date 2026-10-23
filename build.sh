#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIGURE_TIMEOUT="${CONFIGURE_TIMEOUT:-300s}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-symbolize=1:print_stacktrace=1:abort_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:abort_on_error=1}"

usage() {
    cat <<'USAGE'
Usage: ./build.sh [all|analysis|coverage|v310|v220] [CMake options...]

By default, run all four configurations in a unique directory under /tmp.
Set HYPREDRV_BUILD_ROOT to reuse an existing temporary build tree.
Analysis includes ASan/UBSan, tests, clang-tidy, cppcheck, formatting, and docs.
Coverage runs tests once before generating its report; sanitizers are disabled.
The v220 configuration builds in build-v220-check and runs unit and driver tests.

Overrides: CONFIGURE_TIMEOUT, CMAKE_BUILD_PARALLEL_LEVEL, ANALYSIS_CC, COVERAGE_CC,
HYPRE_VERSION, HYPRE_310_ROOT, HYPRE_220_ROOT, HYPREDRV_BUILD_ROOT. Extra CMake options are passed last.
USAGE
}

PROFILE="${1:-all}"
if (($#)); then shift; fi
case "$PROFILE" in
    all|analysis|coverage|v310|v220) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
CMAKE_OPTIONS=("$@")
HYPREDRV_BUILD_ROOT="${HYPREDRV_BUILD_ROOT:-$(mktemp -d /tmp/hypredrive-build-XXXXXXXX)}"

build_configuration() {
    local suffix="$1"
    shift
    HYPREDRV_BUILD="$HYPREDRV_BUILD_ROOT/build-$suffix"
    local install_dir="$HYPREDRV_BUILD_ROOT/install-$suffix"

    timeout "$CONFIGURE_TIMEOUT" cmake -S "$SOURCE_DIR" -B "$HYPREDRV_BUILD" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_VERBOSE_MAKEFILE=OFF \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DHYPREDRV_ENABLE_TESTING=ON \
        -DHYPREDRV_ENABLE_EXAMPLES=ON \
        -DHYPREDRV_ENABLE_COVERAGE=OFF \
        -DHYPREDRV_ENABLE_ANALYSIS=OFF \
        -DHYPREDRV_ENABLE_COMPRESSION=OFF \
        -DHYPREDRV_ENABLE_DOCS=OFF \
        -DCLANG_FORMAT=clang-format-18 \
        "$@" "${CMAKE_OPTIONS[@]}"
    cmake --build "$HYPREDRV_BUILD" --parallel
    cmake --install "$HYPREDRV_BUILD"
}

run_profile() {
    case "$1" in
        analysis)
            build_configuration analysis-relwithdebinfo \
                -DHYPRE_VERSION="${HYPRE_VERSION:-master}" \
                -DCMAKE_C_COMPILER="${ANALYSIS_CC:-gcc-14}" \
                -DHYPREDRV_ENABLE_ANALYSIS=ON \
                -DHYPREDRV_ENABLE_COMPRESSION=ON \
                -DHYPREDRV_ENABLE_DOCS=ON
            ctest --test-dir "$HYPREDRV_BUILD" --output-on-failure
            for target in clang-tidy cppcheck format docs; do
                cmake --build "$HYPREDRV_BUILD" --target "$target"
            done
            ;;
        coverage)
            build_configuration coverage-relwithdebinfo \
                -DHYPRE_VERSION="${HYPRE_VERSION:-master}" \
                -DCMAKE_C_COMPILER="${COVERAGE_CC:-clang-23}" \
                -DBUILD_SHARED_LIBS=ON \
                -DHYPREDRV_ENABLE_COVERAGE=ON \
                -DHYPREDRV_ENABLE_COMPRESSION=ON
            # The coverage target builds and runs the suite itself.
            cmake --build "$HYPREDRV_BUILD" --target coverage
            ;;
        v310)
            build_configuration v3.1.0-relwithdebinfo \
                -DHYPRE_ROOT="${HYPRE_310_ROOT:-$SOURCE_DIR/../hypre-3.1.0/install-cpu}"
            ctest --test-dir "$HYPREDRV_BUILD" --output-on-failure
            ;;
        v220)
            local hypre_root="${HYPRE_220_ROOT:-$SOURCE_DIR/../hypre-2.20.0/src/hypre}"
            build_configuration v220-check \
                -DHYPRE_INCLUDE_DIRS="$hypre_root/include" \
                -DHYPRE_LIBRARY="$hypre_root/lib/libHYPRE.a"
            ctest --test-dir "$HYPREDRV_BUILD" -L unit --output-on-failure
            ctest --test-dir "$HYPREDRV_BUILD" -L hypredrive --output-on-failure
            ;;
    esac
}

if [[ "$PROFILE" == all ]]; then
    for profile in analysis coverage v310 v220; do
        run_profile "$profile"
    done
else
    run_profile "$PROFILE"
fi
