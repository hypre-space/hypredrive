#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

SECONDS=0

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  script_dir="${PWD}"
fi
submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
mode="${1:-all}"

repo_url="${HYPREDRV_REPO_URL:-https://github.com/hypre-space/hypredrive.git}"
repo_branch="${HYPREDRV_BRANCH:-master}"

default_source_dir() {
  if git -C "${submit_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '%s\n' "${submit_dir}"
  elif git -C "${script_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '%s\n' "${submit_dir}"
  else
    printf '%s\n' "${submit_dir}/hypredrive-regression"
  fi
}

source_dir="${HYPREDRV_SOURCE_DIR:-$(default_source_dir)}"

build_type="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
generator="${CMAKE_GENERATOR:-Ninja}"
suffix="${HYPREDRV_SUFFIX:-relwithdebinfo}"
build_dir="${HYPREDRV_BUILD_DIR:-${source_dir}/build-${suffix}}"
hypre_version="${HYPRE_VERSION:-master}"
load_modules_mode="${HYPREDRV_LOAD_MODULES:-auto}"

fetch_elapsed=0
build_elapsed=0
test_elapsed=0
current_phase=""

get_machine_name() {
  printf '%s\n' "${HOST:-}"
}

classify_machine() {
  case "${1}" in
    dane*)
      printf '%s\n' 'dane'
      ;;
    matrix*)
      printf '%s\n' 'matrix'
      ;;
    tioga*)
      printf '%s\n' 'tioga'
      ;;
    tuolumne*)
      printf '%s\n' 'tuolumne'
      ;;
    tux-gfx1100*)
      printf '%s\n' 'tux-gfx1100'
      ;;
    tux-sm120*)
      printf '%s\n' 'tux-sm120'
      ;;
    tux*)
      printf '%s\n' 'tux'
      ;;
    *)
      printf '%s\n' ''
      ;;
  esac
}

machine_name="$(get_machine_name)"
cluster_name="$(classify_machine "${machine_name}")"

default_ctest_timeout() {
  case "${cluster_name}" in
    matrix|tioga|tuolumne|tux-gfx1100|tux-sm120)
      printf '%s\n' '40'
      ;;
    *)
      printf '%s\n' '10'
      ;;
  esac
}

ctest_timeout="${CTEST_TIMEOUT:-$(default_ctest_timeout)}"

usage() {
  cat <<'EOF'
Usage: ./fetch-build-test_hypredrive.sh [all|fetch-only|build-only|test-only|build-test]

Modes:
  all         Configure/fetch, build, then test.
  fetch-only  Clone/configure only. Use this on a login node to pre-fetch sources and dependencies.
  build-only  Build an existing configured tree without fetching.
  test-only   Run ctest on an existing build tree.
  build-test  Build and test an existing configured tree without fetching.

Environment:
  HYPREDRV_SOURCE_DIR  Checkout to use or create. Default: launch directory,
                       or a hypredrive-regression subdirectory when cloning.
  HYPREDRV_REPO_URL    Hypredrive git URL. Default: official GitHub repo.
  HYPREDRV_BRANCH      Hypredrive branch/tag to clone. Default: master.
  HYPREDRV_CMAKE_ARGS  Extra CMake arguments appended after cluster defaults.
  HYPREDRV_GPU_TEST_DEVICE_IDS
                       Comma-separated GPU IDs for CTest resource allocation.
  HYPREDRV_GPU_TEST_RESOURCE_SLOTS
                       GPU slots per physical device. Defaults to enough total
                       slots for four MPI ranks to run on visible devices.
  HYPREDRV_GPU_TEST_TIMEOUT
                       Per-GPU-test timeout in seconds. Default: 60.
  HYPREDRV_GPU_TEST_AUTODETECT_HIP_DEVICES
                       Set ON to opt into rocminfo device enumeration.
  HYPREDRV_GPU_TEST_APPLY_VISIBILITY
                       Set OFF to keep CTest scheduling without changing GPU
                       visibility variables (useful for SYCL selectors).
  HYPREDRV_SYCL_DEVICE_CODE_SPLIT
                       Auto-fetched HYPRE SYCL split mode (default: per_source).
  HYPREDRV_SYCL_DEVICE_SELECTOR
                       ONEAPI_DEVICE_SELECTOR value for SYCL tests.
  HOST                 Machine selector used for module/CMake mappings.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

format_seconds() {
  local total="$1"
  local hours minutes seconds
  hours=$((total / 3600))
  minutes=$(((total % 3600) / 60))
  seconds=$((total % 60))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

on_error() {
  local status=$?
  echo
  echo "================================================================================"
  echo "fetch-build-test_hypredrive.sh failed"
  printf 'Mode         : %s\n' "${mode}"
  printf 'Phase        : %s\n' "${current_phase:-unknown}"
  printf 'Elapsed      : %s\n' "$(format_seconds "${SECONDS}")"
  printf 'Exit status  : %s\n' "${status}"
  echo "================================================================================"
  exit "${status}"
}

trap on_error ERR

load_module_init() {
  if type module >/dev/null 2>&1; then
    return
  fi

  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck source=/usr/share/Modules/init/bash
    source /usr/share/Modules/init/bash
  fi
}

get_cluster_modules() {
  if [[ -n "${HYPREDRV_MODULES:-}" ]]; then
    printf '%s\n' "${HYPREDRV_MODULES}"
    return
  fi

  case "${cluster_name}" in
    dane)
      printf '%s\n' 'ninja cmake/3.30 gcc/13'
      ;;
    matrix)
      printf '%s\n' 'ninja cmake/3.30 gcc/13 cuda/12.9'
      ;;
    tioga)
      printf '%s\n' 'ninja cmake/3.29 cce/20 rocm/6.4.3'
      ;;
    tuolumne)
      printf '%s\n' 'ninja cmake/3.29 cce/20 rocm/6.4.3'
      ;;
    tux|tux-gfx1100|tux-sm120)
      printf '%s\n' ''
      ;;
    *)
      printf '%s\n' ''
      ;;
  esac
}

get_cluster_cmake_args() {
  case "${cluster_name}" in
    dane)
      printf '%s\n' '-DMPIEXEC_EXECUTABLE=/usr/global/tools/flux_wrappers/bin/srun'
      ;;
    tux)
      printf '%s\n' ''
      ;;
    tux-gfx1100)
      printf '%s\n' '-DHYPRE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100'
      ;;
    tux-sm120)
      printf '%s\n' '-DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120'
      ;;
    matrix)
      printf '%s\n' '-DMPIEXEC_EXECUTABLE=/usr/global/tools/flux_wrappers/bin/srun -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90'
      ;;
    tioga)
      printf '%s\n' '-DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DMPIEXEC_EXECUTABLE=/usr/global/tools/flux_wrappers/bin/srun -DHYPRE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON'
      ;;
    tuolumne)
      printf '%s\n' '-DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DMPIEXEC_EXECUTABLE=/usr/global/tools/flux_wrappers/bin/srun -DHYPRE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON'
      ;;
    *)
      printf '%s\n' ''
      ;;
  esac
}

load_modules() {
  if [[ "${load_modules_mode}" == "OFF" ]]; then
    return
  fi

  load_module_init

  if ! type module >/dev/null 2>&1; then
    if [[ "${load_modules_mode}" == "ON" ]]; then
      echo "module command is unavailable but HYPREDRV_LOAD_MODULES=ON" >&2
      return 1
    fi
    log "Module environment not available; continuing without module load."
    return
  fi

  local modules
  modules="$(get_cluster_modules)"
  if [[ -z "${modules}" ]]; then
    log "No module mapping for HOST='${machine_name:-unset}' (class='${cluster_name:-unset}'); continuing as-is."
    return
  fi

  log "Loading modules for HOST='${machine_name:-unset}' (class='${cluster_name:-custom}'): ${modules}"
  # shellcheck disable=SC2086
  module load ${modules}
}

supports_cmake_fresh() {
  cmake --help 2>/dev/null | grep -q -- '--fresh'
}

requires_configured_tree() {
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    echo "Configured build tree not found at ${build_dir}" >&2
    echo "Run './fetch-build-test_hypredrive.sh fetch-only' first." >&2
    exit 1
  fi
}

ensure_hypredrive_checkout() {
  if git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return
  fi

  if [[ -e "${source_dir}" ]] && [[ -n "$(find "${source_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Source directory exists and is not a git checkout: ${source_dir}" >&2
    echo "Set HYPREDRV_SOURCE_DIR to an empty directory or an existing hypredrive checkout." >&2
    return 1
  fi

  log "Cloning hypredrive branch '${repo_branch}' into ${source_dir}"
  mkdir -p "$(dirname "${source_dir}")"
  git clone --depth 1 --branch "${repo_branch}" "${repo_url}" "${source_dir}"
}

run_fetch() {
  local start="${SECONDS}"
  current_phase="fetch"

  ensure_hypredrive_checkout
  rm -rf "${build_dir}"
  local cluster_cmake_args extra_cmake_args
  cluster_cmake_args="$(get_cluster_cmake_args)"
  extra_cmake_args="${HYPREDRV_CMAKE_ARGS:-}"

  local -a args=(
    -DCMAKE_BUILD_TYPE="${build_type}"
    -DCMAKE_VERBOSE_MAKEFILE=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DHYPREDRV_ENABLE_TESTING=ON
    -DHYPREDRV_ENABLE_EXAMPLES=ON
    -DHYPREDRV_ENABLE_COVERAGE=ON
    -DHYPREDRV_ENABLE_ANALYSIS=ON
    -S "${source_dir}"
    -B "${build_dir}"
    -G "${generator}"
  )

  if [[ -n "${HYPRE_ROOT:-}" ]]; then
    args+=(-DHYPRE_ROOT="${HYPRE_ROOT}")
  else
    args+=(-DHYPRE_VERSION="${hypre_version}")
    args+=(-DFETCHCONTENT_UPDATES_DISCONNECTED=OFF)
    args+=(-DFETCHCONTENT_FULLY_DISCONNECTED=OFF)
  fi

  if supports_cmake_fresh; then
    args+=(--fresh)
  fi

  if [[ -n "${cluster_cmake_args}" ]]; then
    log "Applying CMake args for HOST='${machine_name:-unset}' (class='${cluster_name:-custom}'): ${cluster_cmake_args}"
    # shellcheck disable=SC2206
    args+=(${cluster_cmake_args})
  fi

  for gpu_cache_name in \
    HYPREDRV_GPU_TEST_DEVICE_IDS \
    HYPREDRV_GPU_TEST_PARALLEL_LEVEL \
    HYPREDRV_GPU_TEST_RESOURCE_SLOTS \
    HYPREDRV_GPU_TEST_TIMEOUT \
    HYPREDRV_GPU_TEST_APPLY_VISIBILITY \
    HYPREDRV_GPU_VISIBLE_DEVICES_ENV \
    HYPREDRV_GPU_TEST_LAUNCHER \
    HYPREDRV_GPU_TEST_AUTODETECT_HIP_DEVICES \
    HYPREDRV_SYCL_DEVICE_CODE_SPLIT \
    HYPREDRV_SYCL_DEVICE_SELECTOR; do
    if [[ -n "${!gpu_cache_name:-}" ]]; then
      args+=("-D${gpu_cache_name}=${!gpu_cache_name}")
    fi
  done

  if [[ -n "${extra_cmake_args}" ]]; then
    log "Applying extra CMake args: ${extra_cmake_args}"
    # shellcheck disable=SC2206
    args+=(${extra_cmake_args})
  fi

  log "Starting fetch"
  cmake "${args[@]}"
  fetch_elapsed=$((SECONDS - start))
  log "Finished fetch ($(format_seconds "${fetch_elapsed}"))"
}

run_build() {
  local start="${SECONDS}"
  current_phase="build"

  requires_configured_tree

  log "Starting build"
  cmake --build "${build_dir}" --parallel
  build_elapsed=$((SECONDS - start))
  log "Finished build ($(format_seconds "${build_elapsed}"))"
}

get_gpu_resource_spec() {
  local cache_file="${build_dir}/CMakeCache.txt"
  local configured_spec=""

  if [[ -f "${cache_file}" ]]; then
    configured_spec="$(sed -n \
      's/^HYPREDRV_GPU_TEST_RESOURCE_SPEC_FILE:FILEPATH=//p' \
      "${cache_file}")"
  fi

  if [[ -n "${configured_spec}" ]]; then
    if [[ ! -f "${configured_spec}" ]]; then
      echo "Configured GPU resource specification is missing: ${configured_spec}" >&2
      return 1
    fi
    printf '%s\n' "${configured_spec}"
  elif [[ -f "${build_dir}/hypredrive-ctest-gpus.json" ]]; then
    printf '%s\n' "${build_dir}/hypredrive-ctest-gpus.json"
  fi
}

get_gpu_test_timeout() {
  local cache_file="${build_dir}/CMakeCache.txt"
  if [[ -f "${cache_file}" ]]; then
    sed -n \
      's/^HYPREDRV_GPU_TEST_TIMEOUT:STRING=//p' \
      "${cache_file}"
  fi
}

get_gpu_test_parallel_level() {
  local cache_file="${build_dir}/CMakeCache.txt"
  if [[ -f "${cache_file}" ]]; then
    sed -n \
      's/^HYPREDRV_GPU_TEST_PARALLEL_LEVEL_EFFECTIVE:INTERNAL=//p' \
      "${cache_file}"
  fi
}

run_test() {
  local start="${SECONDS}"
  current_phase="test"

  requires_configured_tree

  log "Starting test"
  local resource_spec effective_timeout configured_gpu_timeout configured_gpu_parallel
  resource_spec="$(get_gpu_resource_spec)"
  effective_timeout="${ctest_timeout}"
  if [[ -z "${CTEST_TIMEOUT:-}" && -n "${resource_spec}" ]]; then
    configured_gpu_timeout="$(get_gpu_test_timeout)"
    if [[ "${configured_gpu_timeout}" =~ ^[1-9][0-9]*$ ]]; then
      effective_timeout="${configured_gpu_timeout}"
    fi
  fi
  local -a ctest_args=(
    --test-dir "${build_dir}"
    --output-on-failure
    --timeout "${effective_timeout}"
  )
  if [[ -n "${resource_spec}" ]]; then
    ctest_args+=(--resource-spec-file "${resource_spec}")
    log "Using GPU resource specification: ${resource_spec}"
    log "Using GPU test timeout: ${effective_timeout}s"
    if [[ -z "${CTEST_PARALLEL_LEVEL:-}" ]]; then
      configured_gpu_parallel="$(get_gpu_test_parallel_level)"
      if [[ "${configured_gpu_parallel}" =~ ^[1-9][0-9]*$ ]]; then
        ctest_args+=("-j${configured_gpu_parallel}")
        log "Using GPU CTest parallel level: ${configured_gpu_parallel}"
      fi
    fi
  fi
  ctest "${ctest_args[@]}"
  test_elapsed=$((SECONDS - start))
  log "Finished test ($(format_seconds "${test_elapsed}"))"
}

print_summary() {
  echo
  echo "================================================================================"
  echo "fetch-build-test_hypredrive.sh completed successfully"
  printf 'Mode        : %s\n' "${mode}"
  printf 'Submit dir  : %s\n' "${submit_dir}"
  printf 'Source dir  : %s\n' "${source_dir}"
  printf 'Build dir   : %s\n' "${build_dir}"
  printf 'Fetch time  : %s\n' "$(format_seconds "${fetch_elapsed}")"
  printf 'Build time  : %s\n' "$(format_seconds "${build_elapsed}")"
  printf 'Test time   : %s\n' "$(format_seconds "${test_elapsed}")"
  printf 'Total time  : %s\n' "$(format_seconds "${SECONDS}")"
  echo "================================================================================"
}

main() {
  case "${mode}" in
    all|fetch-only|build-only|test-only|build-test)
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac

  load_modules

  case "${mode}" in
    all)
      run_fetch
      run_build
      run_test
      ;;
    fetch-only)
      run_fetch
      ;;
    build-only)
      run_build
      ;;
    test-only)
      run_test
      ;;
    build-test)
      run_build
      run_test
      ;;
  esac

  print_summary
}

main "$@"
