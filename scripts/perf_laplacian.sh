#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./perf_laplacian [options]

Runs perf+FlameGraph profiling for the standalone laplacian driver built
against two hypre versions (defaults: v2.30.0 and v3.0.0).

Options:
  -a|--version-a <ver>    Baseline hypre version (default: v2.30.0)
  -b|--version-b <ver>    Comparison hypre version (default: v3.0.0)
  -c|--version-c <ver>    Optional third hypre version (default: unset)
  --build-type <type>     CMake build type (default: RelWithDebInfo)
  --n <nx> <ny> <nz>      Grid dimensions (default: 100 100 100)
  --pgrid <Px> <Py> <Pz>  Processor grid (default: 1 1 1)
  --nsolve <n>            Number of solves (default: 5)
  --stencil <s>           Stencil (7|19|27|125) (default: 7)
  --mpi-np <n>            MPI ranks (default: 1)
  --mpi-list <list>       Comma/space-separated MPI ranks
  --auto-pgrid            Derive Px Py Pz per MPI ranks (near-cube)
  # Slurm: uses srun and -n when SLURM_* is set or host matches dane/tioga/tuo
  --yaml <file>           YAML solver config (optional)
  --verbose <n>           Verbosity bitset for laplacian (default: 1)
  --perf-out <dir>        Output directory prefix (timestamp appended)
  --call-graph <mode>     perf callgraph mode (dwarf|fp) (default: dwarf)
  --freq <hz>             perf sampling frequency (default: 99)
  --svg-width <px>        FlameGraph width in pixels (default: 2400)
  --font-size <px>        FlameGraph font size (default: 12)
  --minwidth <px>         FlameGraph min frame width (default: 1)
  --scaling <mode>        strong|weak (default: strong)
  --skip-build            Skip CMake builds (reuse existing build dirs)
  --no-warmup             Skip warmup run
  --no-report             Skip perf report text output
  --no-perf               Disable perf stat/record/flamegraphs
  --caliper               Use Caliper instead of perf (sets CALI_CONFIG)
  --help                  Show this help
Environment overrides:
  CFLAGS, MPIEXEC, MPIEXEC_ARGS, FLAMEGRAPH_DIR, PERF_EVENTS,
  PERF_STAT_EXTRA, PERF_RECORD_EXTRA
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION_A="v2.30.0"
VERSION_B="v3.0.0"
VERSION_C=""
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
CFLAGS="${CFLAGS:--fno-omit-frame-pointer -fno-optimize-sibling-calls}"

MPI_NP="${MPI_NP:-1}"
MPI_LIST="${MPI_LIST:-}"
AUTO_PGRID="${AUTO_PGRID:-0}"
PGRID="${PGRID:-1 1 1}"
NXYZ="${NXYZ:-100 100 100}"
NSOLVE="${NSOLVE:-5}"
STENCIL="${STENCIL:-7}"
VERBOSE="${VERBOSE:-1}"
YAML_INPUT="${YAML_INPUT:-}"
SCALING_MODE="${SCALING_MODE:-strong}"

PERF_OUT="${PERF_OUT:-$ROOT_DIR/perf-out/laplacian}"
PERF_FREQ="${PERF_FREQ:-99}"
PERF_CALLGRAPH="${PERF_CALLGRAPH:-dwarf}"
PERF_EVENTS="${PERF_EVENTS:-task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults}"
PERF_STAT_EXTRA="${PERF_STAT_EXTRA:-}"
PERF_RECORD_EXTRA="${PERF_RECORD_EXTRA:-}"
SVG_WIDTH="${SVG_WIDTH:-2400}"
FONT_SIZE="${FONT_SIZE:-12}"
MIN_WIDTH="${MIN_WIDTH:-1}"
GENERATE_REPORT="${GENERATE_REPORT:-1}"
PERF_ENABLED="${PERF_ENABLED:-1}"
CALIPER_ENABLED="${CALIPER_ENABLED:-0}"
CALI_CONFIG="${CALI_CONFIG:-runtime-report,max_column_width=200,calc.inclusive,output=stdout,mpi-report}"

FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$ROOT_DIR/FlameGraph}"
SKIP_BUILD="${SKIP_BUILD:-0}"
WARMUP="${WARMUP:-1}"

MPIEXEC="${MPIEXEC:-}"
MPIEXEC_NP_FLAG="${MPIEXEC_NP_FLAG:--np}"
MPIEXEC_ARGS="${MPIEXEC_ARGS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--version-a)
      [[ $# -ge 2 ]] || fail "missing value for -a or --version-a"
      VERSION_A="$2"
      shift 2
      ;;
    -b|--version-b)
      [[ $# -ge 2 ]] || fail "missing value for -b or --version-b"
      VERSION_B="$2"
      shift 2
      ;;
    -c|--version-c)
      [[ $# -ge 2 ]] || fail "missing value for -c or --version-c"
      VERSION_C="$2"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || fail "missing value for --build-type"
      BUILD_TYPE="$2"
      shift 2
      ;;
    --n)
      [[ $# -ge 4 ]] || fail "missing values for --n"
      NXYZ="$2 $3 $4"
      shift 4
      ;;
    --pgrid)
      [[ $# -ge 4 ]] || fail "missing values for --pgrid"
      PGRID="$2 $3 $4"
      shift 4
      ;;
    --nsolve)
      [[ $# -ge 2 ]] || fail "missing value for --nsolve"
      NSOLVE="$2"
      shift 2
      ;;
    --stencil)
      [[ $# -ge 2 ]] || fail "missing value for --stencil"
      STENCIL="$2"
      shift 2
      ;;
    --mpi-np)
      [[ $# -ge 2 ]] || fail "missing value for --mpi-np"
      MPI_NP="$2"
      shift 2
      ;;
    --mpi-list)
      [[ $# -ge 2 ]] || fail "missing value for --mpi-list"
      MPI_LIST="$2"
      shift 2
      ;;
    --auto-pgrid)
      AUTO_PGRID=1
      shift
      ;;
    --yaml)
      [[ $# -ge 2 ]] || fail "missing value for --yaml"
      YAML_INPUT="$2"
      shift 2
      ;;
    --verbose)
      [[ $# -ge 2 ]] || fail "missing value for --verbose"
      VERBOSE="$2"
      shift 2
      ;;
    --perf-out)
      [[ $# -ge 2 ]] || fail "missing value for --perf-out"
      PERF_OUT="$2"
      shift 2
      ;;
    --call-graph)
      [[ $# -ge 2 ]] || fail "missing value for --call-graph"
      PERF_CALLGRAPH="$2"
      shift 2
      ;;
    --freq)
      [[ $# -ge 2 ]] || fail "missing value for --freq"
      PERF_FREQ="$2"
      shift 2
      ;;
    --svg-width)
      [[ $# -ge 2 ]] || fail "missing value for --svg-width"
      SVG_WIDTH="$2"
      shift 2
      ;;
    --font-size)
      [[ $# -ge 2 ]] || fail "missing value for --font-size"
      FONT_SIZE="$2"
      shift 2
      ;;
    --minwidth)
      [[ $# -ge 2 ]] || fail "missing value for --minwidth"
      MIN_WIDTH="$2"
      shift 2
      ;;
    --scaling)
      [[ $# -ge 2 ]] || fail "missing value for --scaling"
      SCALING_MODE="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --no-warmup)
      WARMUP=0
      shift
      ;;
    --no-report)
      GENERATE_REPORT=0
      shift
      ;;
    --no-perf)
      PERF_ENABLED=0
      shift
      ;;
    --caliper)
      CALIPER_ENABLED=1
      PERF_ENABLED=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

if [[ "${PERF_ENABLED}" -eq 1 ]]; then
  command -v perf >/dev/null 2>&1 || fail "perf not found in PATH"
  command -v perl >/dev/null 2>&1 || fail "perl not found in PATH"
fi


resolve_mpi_exec() {
  if [[ -n "${MPIEXEC}" ]]; then
    if [[ "${MPIEXEC}" == "srun" && "${MPIEXEC_NP_FLAG}" == "-np" ]]; then
      MPIEXEC_NP_FLAG="-n"
    fi
    return
  fi

  local host
  host="$(hostname -s 2>/dev/null || hostname)"

  if command -v srun >/dev/null 2>&1; then
    if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_NTASKS:-}" || "${host}" =~ ^(dane|tioga|tuo)[0-9]*$ ]]; then
      MPIEXEC="srun"
      MPIEXEC_NP_FLAG="-n"
      return
    fi
  fi

  if command -v mpirun >/dev/null 2>&1; then
    MPIEXEC="mpirun"
  elif command -v mpiexec >/dev/null 2>&1; then
    MPIEXEC="mpiexec"
  else
    fail "mpirun/mpiexec/srun not found (set MPIEXEC to override)"
  fi
}

resolve_mpi_exec

if [[ "${PERF_ENABLED}" -eq 1 ]]; then
  if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
    perf_paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
    if [[ "$perf_paranoid" -gt 1 ]]; then
      echo "warning: perf_event_paranoid=${perf_paranoid} may block sampling." >&2
      echo "         Try: sudo sysctl -w kernel.perf_event_paranoid=1" >&2
    fi
  fi
  if [[ -r /proc/sys/kernel/kptr_restrict ]]; then
    kptr_restrict="$(cat /proc/sys/kernel/kptr_restrict)"
    if [[ "$kptr_restrict" -gt 0 ]]; then
      echo "warning: kptr_restrict=${kptr_restrict} may hide kernel symbols." >&2
      echo "         Try: sudo sysctl -w kernel.kptr_restrict=0" >&2
    fi
  fi
fi

read -r -a NXYZ_ARR <<< "$NXYZ"
[[ ${#NXYZ_ARR[@]} -eq 3 ]] || fail "--n requires 3 integers"

case "$SCALING_MODE" in
  strong|weak)
    ;;
  *)
    fail "invalid --scaling mode: ${SCALING_MODE} (use strong or weak)"
    ;;
esac

calc_pgrid() {
  local np="$1"
  local best_x=1
  local best_y=1
  local best_z="$np"
  local best_score=999999
  local x y z rem max min score

  for ((x=1; x<=np; x++)); do
    (( np % x == 0 )) || continue
    rem=$((np / x))
    for ((y=x; y<=rem; y++)); do
      (( rem % y == 0 )) || continue
      z=$((rem / y))
      (( y <= z )) || continue
      max=$z
      min=$x
      score=$((max - min))
      if (( score < best_score )); then
        best_score=$score
        best_x=$x
        best_y=$y
        best_z=$z
      fi
    done
  done

  PGRID_ARR=("$best_x" "$best_y" "$best_z")
}

prepare_run() {
  local np="$1"
  MPI_NP="$np"

  if [[ "$AUTO_PGRID" -eq 1 ]]; then
    calc_pgrid "$MPI_NP"
    PGRID="${PGRID_ARR[*]}"
  else
    read -r -a PGRID_ARR <<< "$PGRID"
    [[ ${#PGRID_ARR[@]} -eq 3 ]] || fail "--pgrid requires 3 integers"
  fi

  if (( PGRID_ARR[0] * PGRID_ARR[1] * PGRID_ARR[2] != MPI_NP )); then
    fail "pgrid product (${PGRID_ARR[*]}) must equal MPI ranks (${MPI_NP})"
  fi

  GLOBAL_NXYZ_ARR=("${NXYZ_ARR[@]}")
  if [[ "$SCALING_MODE" == "weak" ]]; then
    GLOBAL_NXYZ_ARR=(
      $((NXYZ_ARR[0] * PGRID_ARR[0]))
      $((NXYZ_ARR[1] * PGRID_ARR[1]))
      $((NXYZ_ARR[2] * PGRID_ARR[2]))
    )
  fi
}

MPI_LIST_ARR=()
if [[ -n "$MPI_LIST" ]]; then
  MPI_LIST="${MPI_LIST//,/ }"
  read -r -a MPI_LIST_ARR <<< "$MPI_LIST"
  [[ ${#MPI_LIST_ARR[@]} -gt 0 ]] || fail "--mpi-list requires at least one rank"
  for np in "${MPI_LIST_ARR[@]}"; do
    [[ "$np" =~ ^[0-9]+$ ]] || fail "invalid MPI rank in --mpi-list: $np"
  done
fi

MPIEXEC_ARGS_ARR=()
if [[ -n "$MPIEXEC_ARGS" ]]; then
  read -r -a MPIEXEC_ARGS_ARR <<< "$MPIEXEC_ARGS"
fi

summarize_times_row() {
  local version="$1"
  local mode="$2"
  local log_file="${RUN_OUT}/${version}/run.log"
  local out_file="${RUN_OUT}/summary_times.txt"
  local col=4

  if [[ "$mode" == "solve" ]]; then
    col=5
  fi

  if [[ ! -f "$log_file" ]]; then
    printf "%-12s | %10s | %10s | %10s | %4s\n" \
      "$version" "NA" "NA" "NA" "0" >> "${out_file}"
    return
  fi

  awk -F'|' -v ver="${version}" -v col="${col}" '
    function trim(s) { gsub(/^[ 	]+|[ 	]+$/, "", s); return s }
    $0 ~ /^\|/ {
      idx = trim($2)
      val = trim($(col))
      if (idx ~ /^[0-9]+$/ && val != "")
      {
        v = val + 0.0
        if (count == 0 || v < vmin) vmin = v
        if (count == 0 || v > vmax) vmax = v
        sum += v
        count++
      }
    }
    END {
      if (count > 0)
      {
        printf("%-12s | %10.6f | %10.6f | %10.6f | %4d\n",
               ver, sum/count, vmin, vmax, count)
      }
      else
      {
        printf("%-12s | %10s | %10s | %10s | %4s\n",
               ver, "NA", "NA", "NA", "0")
      }
    }
  ' "${log_file}" >> "${out_file}"
}


append_scaling_summary() {
  local np="$1"
  local pgrid="$2"
  local summary_file="${RUN_OUT}/summary_times.txt"
  local setup_file="${PERF_OUT}/summary_scaling_setup.txt"
  local solve_file="${PERF_OUT}/summary_scaling_solve.txt"

  if [[ ! -f "$setup_file" ]]; then
    printf "%4s | %-9s | %-10s | %10s | %10s | %10s | %4s\n" \
      "np" "pgrid" "version" "AVG" "MIN" "MAX" "n" > "${setup_file}"
    printf "%4s-+-%-9s-+-%-10s-+-%10s-+-%10s-+-%10s-+-%4s\n" \
      "----" "---------" "----------" "----------" "----------" "----------" "----" >> "${setup_file}"
  fi

  if [[ ! -f "$solve_file" ]]; then
    printf "%4s | %-9s | %-10s | %10s | %10s | %10s | %4s\n" \
      "np" "pgrid" "version" "AVG" "MIN" "MAX" "n" > "${solve_file}"
    printf "%4s-+-%-9s-+-%-10s-+-%10s-+-%10s-+-%10s-+-%4s\n" \
      "----" "---------" "----------" "----------" "----------" "----------" "----" >> "${solve_file}"
  fi

  awk -F'|' -v np="${np}" -v pgrid="${pgrid}" '
    function trim(s) { gsub(/^[ 	]+|[ 	]+$/, "", s); return s }
    /^Setup / {section="setup"; next}
    /^Solve / {section="solve"; next}
    $0 ~ /\|/ {
      v = trim($1)
      if (v == "version" || v ~ /^-+$/ || v == "") next
      avg = trim($2); min = trim($3); max = trim($4); n = trim($5)
      if (section == "setup")
        printf("%4s | %-9s | %-10s | %10s | %10s | %10s | %4s\n",
               np, pgrid, v, avg, min, max, n)
    }
  ' "${summary_file}" >> "${setup_file}"

  awk -F'|' -v np="${np}" -v pgrid="${pgrid}" '
    function trim(s) { gsub(/^[ 	]+|[ 	]+$/, "", s); return s }
    /^Setup / {section="setup"; next}
    /^Solve / {section="solve"; next}
    $0 ~ /\|/ {
      v = trim($1)
      if (v == "version" || v ~ /^-+$/ || v == "") next
      avg = trim($2); min = trim($3); max = trim($4); n = trim($5)
      if (section == "solve")
        printf("%4s | %-9s | %-10s | %10s | %10s | %10s | %4s\n",
               np, pgrid, v, avg, min, max, n)
    }
  ' "${summary_file}" >> "${solve_file}"
}





setup_flamegraph() {
  local fg_dir="$1"
  local fg_needed=("flamegraph.pl" "stackcollapse-perf.pl" "difffolded.pl")

  local have_all=1
  for f in "${fg_needed[@]}"; do
    if [[ ! -f "${fg_dir}/${f}" ]]; then
      have_all=0
    fi
  done

  if [[ "$have_all" -eq 0 ]]; then
    local fallback="${ROOT_DIR}/tools/FlameGraph"
    if [[ "$fg_dir" != "$fallback" ]]; then
      fg_dir="$fallback"
    fi
    if [[ ! -d "$fg_dir" ]]; then
      command -v git >/dev/null 2>&1 || fail "git not found and FlameGraph missing"
      git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$fg_dir"
    fi
  fi

  for f in "${fg_needed[@]}"; do
    [[ -f "${fg_dir}/${f}" ]] || fail "FlameGraph script not found: ${fg_dir}/${f}"
  done

  FLAMEGRAPH_DIR="$fg_dir"
}

build_version() {
  local version="$1"
  local build_dir="${ROOT_DIR}/build-${version}"
  local need_rebuild=0

  if [[ -x "${build_dir}/laplacian" ]]; then
    if [[ "${CALIPER_ENABLED}" -eq 1 ]]; then
      if ! grep -q "HYPREDRV_ENABLE_CALIPER:BOOL=ON" "${build_dir}/CMakeCache.txt" 2>/dev/null; then
        echo "==> Build exists for ${version} but without Caliper, rebuilding"
        need_rebuild=1
      else
        echo "==> Build exists for ${version}, skipping"
        return
      fi
    else
      echo "==> Build exists for ${version}, skipping"
      return
    fi
  fi

  local caliper_flag="-DHYPREDRV_ENABLE_CALIPER=OFF"
  local hypre_caliper_flag="-DHYPRE_ENABLE_CALIPER=OFF"
  if [[ "${CALIPER_ENABLED}" -eq 1 ]]; then
    caliper_flag="-DHYPREDRV_ENABLE_CALIPER=ON"
    hypre_caliper_flag="-DHYPRE_ENABLE_CALIPER=ON"
  fi

  cmake -DCMAKE_C_COMPILER=clang \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_C_FLAGS="${CFLAGS}" \
        -DHYPRE_VERSION="${version}" \
        -DHYPREDRV_ENABLE_EXAMPLES=ON \
        ${caliper_flag} \
        ${hypre_caliper_flag} \
        -B "${build_dir}" -S "${ROOT_DIR}" --fresh
  cmake --build "${build_dir}" --parallel
}



run_perf_for_version() {
  local version="$1"
  local build_dir="${ROOT_DIR}/build-${version}"
  local exe="${build_dir}/laplacian"
  local out_dir="${RUN_OUT}/${version}"

  [[ -x "$exe" ]] || fail "missing executable: ${exe}"
  mkdir -p "$out_dir"

  local cmd=(
    "${MPIEXEC}"
    "${MPIEXEC_ARGS_ARR[@]}"
    "${MPIEXEC_NP_FLAG}" "${MPI_NP}"
    "${exe}"
    -P "${PGRID_ARR[@]}"
    -n "${GLOBAL_NXYZ_ARR[@]}"
    -s "${STENCIL}"
    -ns "${NSOLVE}"
    -v "${VERBOSE}"
  )
  if [[ -n "$YAML_INPUT" ]]; then
    cmd+=(-i "$YAML_INPUT")
  fi

  printf '%q ' "${cmd[@]}" > "${out_dir}/cmd.txt"
  printf '\n' >> "${out_dir}/cmd.txt"

  local env_prefix=()
  if [[ "${CALIPER_ENABLED}" -eq 1 ]]; then
    env_prefix=("env" "CALI_CONFIG=${CALI_CONFIG}")
  fi

  if [[ "$WARMUP" -eq 1 ]]; then
    (cd "${build_dir}" && "${env_prefix[@]}" "${cmd[@]}" > "${out_dir}/warmup.log" 2>&1)
  fi

  if [[ "${PERF_ENABLED}" -eq 1 ]]; then
    if ! (cd "${build_dir}" && perf stat -x, -o "${out_dir}/perf_stat.csv" \
          -e "${PERF_EVENTS}" ${PERF_STAT_EXTRA} -- "${cmd[@]}" \
          > "${out_dir}/run.log" 2> "${out_dir}/perf_stat.log"); then
      echo "perf stat failed for ${version}. Check ${out_dir}/perf_stat.log" >&2
      exit 1
    fi

    if ! (cd "${build_dir}" && perf record -o "${out_dir}/perf.data" \
          -F "${PERF_FREQ}" --call-graph "${PERF_CALLGRAPH}" \
          ${PERF_RECORD_EXTRA} -- "${cmd[@]}" \
          > "${out_dir}/record.log" 2> "${out_dir}/record.err"); then
      echo "perf record failed for ${version}. Check ${out_dir}/record.err" >&2
      exit 1
    fi

    if [[ "${GENERATE_REPORT}" -eq 1 ]]; then
      perf report --stdio -i "${out_dir}/perf.data" \
        --no-children --percent-limit 1 \
        > "${out_dir}/perf_report.txt" 2> "${out_dir}/perf_report.err" || true
    fi

    perf script -i "${out_dir}/perf.data" > "${out_dir}/perf.script"
    perl "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" "${out_dir}/perf.script" \
      > "${out_dir}/perf.folded"
    perl "${FLAMEGRAPH_DIR}/flamegraph.pl" \
      --title "laplacian ${version}" --countname "samples" \
      --width "${SVG_WIDTH}" --fontsize "${FONT_SIZE}" --minwidth "${MIN_WIDTH}" \
      "${out_dir}/perf.folded" > "${out_dir}/flame.svg"
  else
    if ! (cd "${build_dir}" && "${env_prefix[@]}" "${cmd[@]}" > "${out_dir}/run.log" 2>&1); then
      echo "run failed for ${version}. Check ${out_dir}/run.log" >&2
      exit 1
    fi
  fi
}




if [[ "${PERF_ENABLED}" -eq 1 ]]; then
  setup_flamegraph "${FLAMEGRAPH_DIR}"
fi
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
PERF_OUT="${PERF_OUT%/}-${TIMESTAMP}"
mkdir -p "${PERF_OUT}"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  build_version "${VERSION_A}"
  build_version "${VERSION_B}"
  if [[ -n "${VERSION_C}" ]]; then
    build_version "${VERSION_C}"
  fi
fi

run_suite() {
  local np="$1"
  local pgrid_str="$2"

  run_perf_for_version "${VERSION_A}"
  run_perf_for_version "${VERSION_B}"
  if [[ -n "${VERSION_C}" ]]; then
    run_perf_for_version "${VERSION_C}"
  fi

  summary_times="${RUN_OUT}/summary_times.txt"
  echo "scaling=${SCALING_MODE} pgrid=${PGRID_ARR[*]} n=${GLOBAL_NXYZ_ARR[*]} mpi_np=${MPI_NP}" > "${summary_times}"
  for version in "${VERSION_A}" "${VERSION_B}" "${VERSION_C}"; do
    [[ -n "${version}" ]] || continue
    if [[ -f "${RUN_OUT}/${version}/cmd.txt" ]]; then
      echo "cmd ${version}: $(tr -d '\n' < ${RUN_OUT}/${version}/cmd.txt)" >> "${summary_times}"
    fi
  done
  echo "" >> "${summary_times}"
  echo "Setup (seconds)" >> "${summary_times}"
  printf "%-12s | %10s | %10s | %10s | %4s\n" \
    "version" "AVG" "MIN" "MAX" "n" >> "${summary_times}"
  printf "%-12s-+-%10s-+-%10s-+-%10s-+-%4s\n" \
    "------------" "----------" "----------" "----------" "----" >> "${summary_times}"
  summarize_times_row "${VERSION_A}" "setup"
  summarize_times_row "${VERSION_B}" "setup"
  if [[ -n "${VERSION_C}" ]]; then
    summarize_times_row "${VERSION_C}" "setup"
  fi
  echo "" >> "${summary_times}"
  echo "Solve (seconds)" >> "${summary_times}"
  printf "%-12s | %10s | %10s | %10s | %4s\n" \
    "version" "AVG" "MIN" "MAX" "n" >> "${summary_times}"
  printf "%-12s-+-%10s-+-%10s-+-%10s-+-%4s\n" \
    "------------" "----------" "----------" "----------" "----" >> "${summary_times}"
  summarize_times_row "${VERSION_A}" "solve"
  summarize_times_row "${VERSION_B}" "solve"
  if [[ -n "${VERSION_C}" ]]; then
    summarize_times_row "${VERSION_C}" "solve"
  fi

  if [[ "${PERF_ENABLED}" -eq 1 ]]; then
    diff_flag=""
    if perl "${FLAMEGRAPH_DIR}/flamegraph.pl" --help 2>&1 | grep -q -- '--diff'; then
      diff_flag="--diff"
    fi

    diff_ab_folded="${RUN_OUT}/diff_ab.folded"
    perl "${FLAMEGRAPH_DIR}/difffolded.pl" \
      "${RUN_OUT}/${VERSION_A}/perf.folded" \
      "${RUN_OUT}/${VERSION_B}/perf.folded" \
      > "${diff_ab_folded}"
    perl "${FLAMEGRAPH_DIR}/flamegraph.pl" ${diff_flag} \
      --title "laplacian ${VERSION_B} vs ${VERSION_A}" --countname "samples" \
      --width "${SVG_WIDTH}" --fontsize "${FONT_SIZE}" --minwidth "${MIN_WIDTH}" \
      "${diff_ab_folded}" > "${RUN_OUT}/diff_ab.svg"

    if [[ -n "${VERSION_C}" ]]; then
      diff_ac_folded="${RUN_OUT}/diff_ac.folded"
      perl "${FLAMEGRAPH_DIR}/difffolded.pl" \
        "${RUN_OUT}/${VERSION_A}/perf.folded" \
        "${RUN_OUT}/${VERSION_C}/perf.folded" \
        > "${diff_ac_folded}"
      perl "${FLAMEGRAPH_DIR}/flamegraph.pl" ${diff_flag} \
        --title "laplacian ${VERSION_C} vs ${VERSION_A}" --countname "samples" \
        --width "${SVG_WIDTH}" --fontsize "${FONT_SIZE}" --minwidth "${MIN_WIDTH}" \
        "${diff_ac_folded}" > "${RUN_OUT}/diff_ac.svg"

      diff_bc_folded="${RUN_OUT}/diff_bc.folded"
      perl "${FLAMEGRAPH_DIR}/difffolded.pl" \
        "${RUN_OUT}/${VERSION_B}/perf.folded" \
        "${RUN_OUT}/${VERSION_C}/perf.folded" \
        > "${diff_bc_folded}"
      perl "${FLAMEGRAPH_DIR}/flamegraph.pl" ${diff_flag} \
        --title "laplacian ${VERSION_C} vs ${VERSION_B}" --countname "samples" \
        --width "${SVG_WIDTH}" --fontsize "${FONT_SIZE}" --minwidth "${MIN_WIDTH}" \
        "${diff_bc_folded}" > "${RUN_OUT}/diff_bc.svg"
    fi

    summary_csv="${RUN_OUT}/summary.csv"
    echo "version,event,value,unit" > "${summary_csv}"
    for version in "${VERSION_A}" "${VERSION_B}" "${VERSION_C}"; do
      [[ -n "${version}" ]] || continue
      stat_file="${RUN_OUT}/${version}/perf_stat.csv"
      if [[ -f "$stat_file" ]]; then
        awk -F, -v ver="${version}" '
          $0 ~ /^#/ { next }
          NF < 3 { next }
          $1 ~ /not supported/ { next }
          $1 ~ /not counted/ { next }
          {
            gsub(/^[ 	]+|[ 	]+$/, "", $1)
            gsub(/^[ 	]+|[ 	]+$/, "", $2)
            gsub(/^[ 	]+|[ 	]+$/, "", $3)
            printf "%s,%s,%s,%s\n", ver, $3, $1, $2
          }
        ' "$stat_file" >> "${summary_csv}"
      fi
    done
  fi

  echo "Done (np=${np}, pgrid=${pgrid_str})."
  if [[ "${PERF_ENABLED}" -eq 1 ]]; then
    echo "FlameGraphs:"
    echo "  ${RUN_OUT}/${VERSION_A}/flame.svg"
    echo "  ${RUN_OUT}/${VERSION_B}/flame.svg"
    echo "  ${RUN_OUT}/diff_ab.svg"
    if [[ -n "${VERSION_C}" ]]; then
      echo "  ${RUN_OUT}/${VERSION_C}/flame.svg"
      echo "  ${RUN_OUT}/diff_ac.svg"
      echo "  ${RUN_OUT}/diff_bc.svg"
    fi
    echo "Summary:"
    echo "  ${summary_csv}"
  else
    if [[ "${CALIPER_ENABLED}" -eq 1 ]]; then
      echo "Caliper enabled (--caliper): perf stats/flamegraphs skipped."
    else
      echo "Perf disabled (--no-perf): no perf stats/flamegraphs generated."
    fi
  fi
  echo "Timing summary:"
  echo "  ${summary_times}"
  if [[ "${GENERATE_REPORT}" -eq 1 && "${PERF_ENABLED}" -eq 1 ]]; then
    echo "Perf report:"
    echo "  ${RUN_OUT}/${VERSION_A}/perf_report.txt"
    echo "  ${RUN_OUT}/${VERSION_B}/perf_report.txt"
  fi

  if [[ -n "$MPI_LIST" ]]; then
    append_scaling_summary "$np" "$pgrid_str"
  fi
}


if [[ ${#MPI_LIST_ARR[@]} -gt 0 ]]; then
  for np in "${MPI_LIST_ARR[@]}"; do
    prepare_run "$np"
    RUN_OUT="${PERF_OUT}/np${MPI_NP}"
    mkdir -p "${RUN_OUT}"
    run_suite "$MPI_NP" "${PGRID_ARR[0]}x${PGRID_ARR[1]}x${PGRID_ARR[2]}"
  done

  if [[ -f "${PERF_OUT}/summary_scaling_setup.txt" || -f "${PERF_OUT}/summary_scaling_solve.txt" ]]; then
    {
      if [[ -f "${PERF_OUT}/summary_scaling_setup.txt" ]]; then
        echo "Setup (seconds)"
        cat "${PERF_OUT}/summary_scaling_setup.txt"
        echo ""
      fi
      if [[ -f "${PERF_OUT}/summary_scaling_solve.txt" ]]; then
        echo "Solve (seconds)"
        cat "${PERF_OUT}/summary_scaling_solve.txt"
      fi
    } > "${PERF_OUT}/summary_scaling.txt"

    echo "Scaling summary:"
    echo "  ${PERF_OUT}/summary_scaling.txt"
  fi
else
  prepare_run "$MPI_NP"
  RUN_OUT="${PERF_OUT}"
  run_suite "$MPI_NP" "${PGRID_ARR[0]}x${PGRID_ARR[1]}x${PGRID_ARR[2]}"
fi
