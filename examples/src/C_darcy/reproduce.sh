#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: examples/src/C_darcy/reproduce.sh [options]

Reproduce the SPE10 mixed Darcy example:
  1. download/unpack the SPE10 case 2a data if needed,
  2. build the darcy example if its executable is missing,
  3. run the 3D C Darcy benchmark,
  4. generate SPE10 permeability/pressure figures.

Options:
  --figure-mode <layer|3d|both>
                   Figure(s) to generate (default: layer, or FIGURE_MODE)
  --skip-run        Do not build or run the C benchmark.
  --skip-figure     Do not generate PNG figures.
  -h, --help        Print this help.

Environment overrides:
  BUILD_DIR        Build directory for the auto-build (default: build)
  DARCY_BIN        Prebuilt darcy executable to use instead of auto-building
                   (default: ${BUILD_DIR}/darcy)
  HYPRE_ROOT       Existing HYPRE install for the auto-build; if unset, hypre
                   is fetched and built automatically by CMake.
  CMAKE_BUILD_TYPE Build type for the auto-build (default: Release)
  MPIEXEC          MPI launcher (default: mpirun)
  NP               MPI ranks for the C benchmark (default: 16)
  NXYZ             C benchmark grid (default: "60 220 85")
  PGRID            C benchmark rank grid (default: "1 4 4")
  SPE10_DATA_DIR   Dataset directory (default: data/spe10_case2a)
  OUT_DIR          Output/log directory (default: examples/src/C_darcy/reproduce-out)
  PYTHON           Python interpreter for postprocess.py (default: python3)
  RESULT_FILE      C Darcy VTK output path (default: ${OUT_DIR}/darcy_spe10.vti)
  FIGURE_MODE      Figure mode: layer, 3d, or both (default: layer)
  FIGURE_PATH      Layer figure path (default: docs/usrman-src/figures/spe10_darcy_fields.png)
  FIGURE_3D_PATH   3D figure path (default: docs/usrman-src/figures/spe10_darcy_3d.png)
  SPE10_LAYER      Physical z-layer for the layer figure, 0-based (default: 35)
EOF
}

skip_run=0
skip_figure=0
figure_mode="${FIGURE_MODE:-layer}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --figure-mode)
            if [[ $# -lt 2 ]]; then
                echo "Error: --figure-mode requires layer, 3d, or both" >&2
                exit 1
            fi
            figure_mode="$2"
            shift 2
            ;;
        --skip-run)
            skip_run=1
            shift
            ;;
        --skip-figure)
            skip_figure=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

build_dir="${BUILD_DIR:-${repo_root}/build}"
darcy_bin="${DARCY_BIN:-${build_dir}/darcy}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
mpiexec_cmd="${MPIEXEC:-mpirun}"
np="${NP:-16}"
nxyz="${NXYZ:-60 220 85}"
pgrid="${PGRID:-1 4 4}"
data_dir="${SPE10_DATA_DIR:-${repo_root}/data/spe10_case2a}"
out_dir="${OUT_DIR:-${script_dir}/reproduce-out}"
python_cmd="${PYTHON:-python3}"
result_file="${RESULT_FILE:-${out_dir}/darcy_spe10.vti}"
figure_path="${FIGURE_PATH:-${repo_root}/docs/usrman-src/figures/spe10_darcy_fields.png}"
figure_3d_path="${FIGURE_3D_PATH:-${repo_root}/docs/usrman-src/figures/spe10_darcy_3d.png}"
spe10_layer="${SPE10_LAYER:-35}"

case "${figure_mode}" in
    layer|3d|both) ;;
    *)
        echo "Error: FIGURE_MODE/--figure-mode must be layer, 3d, or both" >&2
        exit 1
        ;;
esac

mkdir -p "${out_dir}"

if [[ ! -s "${data_dir}/spe_perm.dat" ]]; then
    "${repo_root}/scripts/download_spe10_case2a.sh" "${data_dir}"
fi

build_darcy() {
    if ! command -v cmake >/dev/null 2>&1; then
        echo "Error: cmake is required to build the darcy example." >&2
        echo "Install cmake, or set DARCY_BIN=/path/to/prebuilt/darcy." >&2
        exit 1
    fi

    echo "Darcy executable not found; building it in ${build_dir}"
    local -a cfg_args=(
        -S "${repo_root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE="${build_type}"
        -DHYPREDRV_ENABLE_EXAMPLES=ON
    )
    if [[ -n "${HYPRE_ROOT:-}" ]]; then
        cfg_args+=(-DHYPRE_ROOT="${HYPRE_ROOT}")
    fi
    cmake "${cfg_args[@]}"
    cmake --build "${build_dir}" --target darcy --parallel
}

if [[ "${skip_run}" -eq 0 ]]; then
    if [[ ! -x "${darcy_bin}" ]]; then
        if [[ -n "${DARCY_BIN:-}" ]]; then
            echo "Error: DARCY_BIN is set but not executable: ${darcy_bin}" >&2
            exit 1
        fi
        build_darcy
        if [[ ! -x "${darcy_bin}" ]]; then
            echo "Error: build did not produce an executable at ${darcy_bin}" >&2
            exit 1
        fi
    fi

    log_path="${out_dir}/darcy_spe10.log"
    echo "Running C Darcy SPE10 benchmark; log: ${log_path}"
    # shellcheck disable=SC2086
    "${mpiexec_cmd}" -np "${np}" "${darcy_bin}" \
        -n ${nxyz} \
        -P ${pgrid} \
        --K-file "${data_dir}/spe_perm.dat" \
        --K-file-grid 60 220 85 \
        --K-file-k-order top-down \
        --output "${result_file}" \
        -g y -v 1 | tee "${log_path}"
fi

if [[ "${skip_figure}" -eq 0 ]]; then
    echo "Generating SPE10 figure mode '${figure_mode}'"
    mkdir -p "$(dirname "${figure_path}")"
    mkdir -p "$(dirname "${figure_3d_path}")"
    result_input="${result_file}"
    result_base="${result_file%.*}"
    if [[ -s "${result_base}.pvti" ]]; then
        result_input="${result_base}.pvti"
    fi
    # shellcheck disable=SC2086
    "${python_cmd}" "${script_dir}/postprocess.py" \
        --result-file "${result_input}" \
        --mode "${figure_mode}" \
        --layer "${spe10_layer}" \
        --figure-path "${figure_path}" \
        --figure-3d-path "${figure_3d_path}"
fi
