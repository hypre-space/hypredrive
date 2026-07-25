#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${HYPREDRV_SCALING_RESULTS_DIR:-${ROOT_DIR}/results/node-scaling}"
START_UNKNOWNS=100000
NSOLVE=5
VERBOSE=1

# Conservative starting points for each machine. These are tuning limits, not
# guaranteed out-of-memory boundaries. Override one with --max-unknowns.
declare -A LAP7_CAPS=(
  [dane]=128000000
  [matrix]=90000000
  [tuo-cpu]=128000000
  [tuo-gpu-cpx]=128000000
  [tuo-gpu-spx]=128000000
  [polaris-cpu]=128000000
  [polaris-gpu]=45000000
  [aurora]=128000000
  [frontier-cpu]=128000000
  [frontier-gpu]=128000000
  [tioga-cpu]=128000000
  [tioga-gpu]=128000000
)
declare -A LAP27_CAPS=(
  [dane]=96000000
  [matrix]=45000000
  [tuo-cpu]=64000000
  [tuo-gpu-cpx]=64000000
  [tuo-gpu-spx]=64000000
  [polaris-cpu]=64000000
  [polaris-gpu]=12000000
  [aurora]=64000000
  [frontier-cpu]=64000000
  [frontier-gpu]=64000000
  [tioga-cpu]=64000000
  [tioga-gpu]=64000000
)
declare -A ELAST_CAPS=(
  [dane]=64000000
  [matrix]=26000000
  [tuo-cpu]=32000000
  [tuo-gpu-cpx]=32000000
  [tuo-gpu-spx]=32000000
  [polaris-cpu]=32000000
  [polaris-gpu]=12000000
  [aurora]=32000000
  [frontier-cpu]=32000000
  [frontier-gpu]=32000000
  [tioga-cpu]=32000000
  [tioga-gpu]=32000000
)

usage() {
  cat <<'EOF'
Usage: scripts/node_scaling.sh -m MACHINE -p PROBLEM [options]

Run a single-node problem-size scaling study inside an interactive allocation.

Required:
  -m, --machine MACHINE   dane | matrix | tuo-cpu | tuo-gpu-cpx |
                          tuo-gpu-spx | polaris-cpu | polaris-gpu |
                          aurora | frontier-cpu | frontier-gpu |
                          tioga-cpu | tioga-gpu
  -p, --problem PROBLEM   lap-7 | lap-27 | elast

Options:
  -e, --executable PATH   Use PATH instead of the driver in --build-dir
      --build-dir DIR     Directory containing laplacian and elasticity
                          (default: build)
      --max-unknowns N    Replace the conservative machine/problem size cap
      --dry-run           Print commands without requiring an allocation
  -h, --help              Show this help

Environment:
  HYPREDRV_SCALING_RESULTS_DIR
                          Output root (default: results/node-scaling)

Examples:
  scripts/node_scaling.sh -m dane -p lap-7
  scripts/node_scaling.sh -m tuo-gpu-cpx -p lap-27 --max-unknowns 80000000
  scripts/node_scaling.sh -m polaris-gpu -p elast --dry-run
  scripts/node_scaling.sh -m aurora -p lap-7 --dry-run
  scripts/node_scaling.sh -m frontier-gpu -p lap-27 --dry-run
  scripts/node_scaling.sh -m tioga-gpu -p elast --dry-run
  scripts/node_scaling.sh -m dane -p lap-7 -e install/bin/laplacian

Notes:
  - The script uses one complete compute node and never requests an allocation.
  - tuo-gpu-cpx requires an allocation created with --amd-gpumode=CPX.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

shell_command() {
  local result
  printf -v result '%q ' "$@"
  printf '%s' "${result% }"
}

machine=""
problem=""
build_dir="${ROOT_DIR}/build"
executable_override=""
max_unknowns=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--machine)
      [[ $# -ge 2 ]] || die "missing value for $1"
      machine="$2"
      shift 2
      ;;
    -p|--problem)
      [[ $# -ge 2 ]] || die "missing value for $1"
      problem="$2"
      shift 2
      ;;
    -e|--executable)
      [[ $# -ge 2 ]] || die "missing value for $1"
      executable_override="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || die "missing value for --build-dir"
      build_dir="$2"
      shift 2
      ;;
    --max-unknowns)
      [[ $# -ge 2 ]] || die "missing value for --max-unknowns"
      max_unknowns="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "${machine}" ]] || die "-m/--machine is required"
[[ -n "${problem}" ]] || die "-p/--problem is required"

if [[ "${build_dir}" != /* ]]; then
  build_dir="${ROOT_DIR}/${build_dir}"
fi

launcher=()
rank_wrapper=()
ranks=0
px=0
py=0
pz=0
scheduler=""
resource_description=""
gpu_aware_env=""

case "${machine}" in
  dane)
    ranks=112
    px=7
    py=4
    pz=4
    scheduler="slurm"
    resource_description="112 CPU cores"
    launcher=(srun --nodes=1 --ntasks=112 --exclusive)
    ;;
  matrix)
    ranks=4
    px=2
    py=2
    pz=1
    scheduler="slurm"
    resource_description="4 NVIDIA H100 GPUs"
    launcher=(srun --nodes=1 --ntasks=4 --exclusive --gpus-per-task=1)
    rank_wrapper=()
    # rank_wrapper=(
    #   bash -c
    #   'export CUDA_VISIBLE_DEVICES="${SLURM_LOCAL_ID}"; exec "$@"'
    #   bash
    # )
    gpu_aware_env="MV2_USE_CUDA=0"
    ;;
  tuo-cpu)
    ranks=80
    px=5
    py=4
    pz=4
    scheduler="flux"
    resource_description="80 CPU cores"
    launcher=(flux run --nodes=1 --ntasks=80 --exclusive)
    ;;
  tuo-gpu-cpx)
    ranks=24
    px=4
    py=3
    pz=2
    scheduler="flux"
    resource_description="24 logical AMD MI300A GPUs in CPX mode"
    launcher=(flux run --nodes=1 --ntasks=24 --exclusive)
    gpu_aware_env="MPICH_GPU_SUPPORT_ENABLED=1"
    ;;
  tuo-gpu-spx)
    ranks=4
    px=2
    py=2
    pz=1
    scheduler="flux"
    resource_description="4 AMD MI300A GPUs in SPX mode"
    launcher=(flux run --nodes=1 --ntasks=4 --exclusive)
    gpu_aware_env="MPICH_GPU_SUPPORT_ENABLED=1"
    ;;
  tioga-cpu)
    ranks=64
    px=4
    py=4
    pz=4
    scheduler="flux"
    resource_description="64 AMD EPYC CPU cores"
    launcher=(flux run --nodes=1 --ntasks=64 --exclusive)
    ;;
  tioga-gpu)
    ranks=8
    px=2
    py=2
    pz=2
    scheduler="flux"
    resource_description="8 AMD MI250X GPU devices"
    launcher=(flux run --nodes=1 --ntasks=8 --exclusive)
    gpu_aware_env="MPICH_GPU_SUPPORT_ENABLED=1"
    ;;
  polaris-cpu)
    ranks=32
    px=4
    py=4
    pz=2
    scheduler="pbs"
    resource_description="32 physical CPU cores"
    launcher=(mpiexec -n 32 --ppn 32 --depth=1 --cpu-bind depth)
    ;;
  polaris-gpu)
    ranks=4
    px=2
    py=2
    pz=1
    scheduler="pbs"
    resource_description="4 NVIDIA A100 GPUs"
    launcher=(mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth)
    rank_wrapper=(
      bash -c
      'gpu=$((3 - PMI_LOCAL_RANK % 4)); export CUDA_VISIBLE_DEVICES="${gpu}"; exec "$@"'
      bash
    )
    gpu_aware_env="MPICH_GPU_SUPPORT_ENABLED=1"
    ;;
  aurora)
    ranks=12
    px=3
    py=2
    pz=2
    scheduler="pbs"
    resource_description="12 Intel Data Center GPU Max tiles"
    launcher=(
      mpiexec -n 12 -ppn 12
      --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100
    )
    rank_wrapper=(gpu_tile_compact.sh)
    gpu_aware_env="MPIR_CVAR_ENABLE_GPU=1"
    ;;
  frontier-cpu)
    ranks=56
    px=7
    py=4
    pz=2
    scheduler="slurm"
    resource_description="56 allocatable physical CPU cores"
    launcher=(
      srun --nodes=1 --ntasks=56 --ntasks-per-node=56
      --cpus-per-task=1 --threads-per-core=1 --cpu-bind=threads
      --distribution=block:cyclic --exclusive
    )
    ;;
  frontier-gpu)
    ranks=8
    px=2
    py=2
    pz=2
    scheduler="slurm"
    resource_description="8 AMD MI250X GPU compute dies"
    launcher=(
      srun --nodes=1 --ntasks=8 --ntasks-per-node=8
      --cpus-per-task=7 --threads-per-core=1 --cpu-bind=threads
      --gpus-per-node=8 --gpus-per-task=1 --gpu-bind=closest --exclusive
    )
    gpu_aware_env="MPICH_GPU_SUPPORT_ENABLED=1"
    ;;
  *)
    die "unsupported machine: ${machine}"
    ;;
esac

driver_args=()
dofs_per_grid_point=1
case "${problem}" in
  lap-7)
    executable="${build_dir}/laplacian"
    driver_args=(-s 7)
    default_cap="${LAP7_CAPS[${machine}]}"
    ;;
  lap-27)
    executable="${build_dir}/laplacian"
    driver_args=(-s 27)
    default_cap="${LAP27_CAPS[${machine}]}"
    ;;
  elast)
    executable="${build_dir}/elasticity"
    driver_args=(--solver-preset elasticity_3D)
    dofs_per_grid_point=3
    default_cap="${ELAST_CAPS[${machine}]}"
    ;;
  *)
    die "unsupported problem: ${problem}"
    ;;
esac

if [[ -n "${executable_override}" ]]; then
  executable="${executable_override}"
fi

if [[ -n "${max_unknowns}" ]]; then
  [[ "${max_unknowns}" =~ ^[1-9][0-9]*$ ]] ||
    die "--max-unknowns must be a positive integer"
else
  max_unknowns="${default_cap}"
fi
((max_unknowns >= START_UNKNOWNS)) ||
  die "--max-unknowns must be at least ${START_UNKNOWNS}"

((px * py * pz == ranks)) ||
  die "internal error: processor grid does not match rank count"
[[ -x "${executable}" ]] || die "executable not found: ${executable}"

if ((dry_run == 0)); then
  case "${scheduler}" in
    slurm)
      [[ -n "${SLURM_JOB_ID:-}" ]] ||
        die "${machine} runs must start inside a Slurm allocation"
      command -v srun >/dev/null 2>&1 || die "srun was not found"
      ;;
    flux)
      [[ -n "${FLUX_JOB_ID:-}${FLUX_URI:-}" ]] ||
        die "${machine} runs must start inside a Flux allocation"
      command -v flux >/dev/null 2>&1 || die "flux was not found"
      ;;
    pbs)
      [[ -n "${PBS_JOBID:-}" ]] ||
        die "${machine} runs must start inside a PBS allocation"
      command -v mpiexec >/dev/null 2>&1 || die "mpiexec was not found"
      if [[ "${machine}" == "aurora" ]]; then
        command -v gpu_tile_compact.sh >/dev/null 2>&1 ||
          die "Aurora's gpu_tile_compact.sh was not found"
      fi
      ;;
  esac
fi

# Set NX, NY, NZ, and UNKNOWNS for a nominal near-cubic grid edge. Rounding
# each dimension to its processor-grid multiple keeps the driver partition valid.
set_grid() {
  local edge="$1"
  NX=$((((edge + px - 1) / px) * px))
  NY=$((((edge + py - 1) / py) * py))
  NZ=$((((edge + pz - 1) / pz) * pz))
  UNKNOWNS=$((dofs_per_grid_point * NX * NY * NZ))
}

# Find the first grid at or above 100K unknowns.
edge=1
while true; do
  set_grid "${edge}"
  ((UNKNOWNS >= START_UNKNOWNS)) && break
  edge=$((edge + 1))
done
((UNKNOWNS <= max_unknowns)) ||
  die "the size cap is too small for the first processor-grid-compatible problem"

# Grow each dimension by about 25%, which approximately doubles the volume.
edges=()
while true; do
  set_grid "${edge}"
  ((UNKNOWNS <= max_unknowns)) || break
  edges+=("${edge}")
  next_edge=$(((edge * 5 + 3) / 4))
  ((next_edge > edge)) || next_edge=$((edge + 1))
  edge="${next_edge}"
done

# Also include the largest nominal grid edge below the cap.
last_index=$((${#edges[@]} - 1))
last_edge="${edges[${last_index}]}"
set_grid "${last_edge}"
last_unknowns="${UNKNOWNS}"
max_edge="${last_edge}"
probe=$((last_edge + 1))
while true; do
  set_grid "${probe}"
  ((UNKNOWNS <= max_unknowns)) || break
  max_edge="${probe}"
  probe=$((probe + 1))
done
set_grid "${max_edge}"
if ((UNKNOWNS > last_unknowns)); then
  edges+=("${max_edge}")
fi

printf 'Single-node hypredrive scaling\n'
printf '  Machine:      %s (%s)\n' "${machine}" "${resource_description}"
printf '  Problem:      %s\n' "${problem}"
printf '  MPI layout:   %d ranks, %d x %d x %d\n' "${ranks}" "${px}" "${py}" "${pz}"
printf '  Size range:   %d to %d unknowns\n' "${START_UNKNOWNS}" "${max_unknowns}"
printf '  Executable:   %s\n' "${executable}"

if ((dry_run == 0)); then
  timestamp="$(date '+%Y-%m-%d_%H-%M-%S_%Z')"
  run_dir="${RESULTS_ROOT}/${machine}/${problem}/${timestamp}"
  if [[ -e "${run_dir}" ]]; then
    run_dir="${run_dir}_$$"
  fi
  mkdir -p "${run_dir}"
  commands_log="${run_dir}/commands.log"
  summary_file="${run_dir}/summary.tsv"
  metadata_file="${run_dir}/metadata.txt"

  git_revision="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  launcher_text="$(shell_command "${launcher[@]}")"
  {
    printf 'timestamp=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'machine=%s\n' "${machine}"
    printf 'problem=%s\n' "${problem}"
    printf 'resources=%s\n' "${resource_description}"
    printf 'mpi_ranks=%d\n' "${ranks}"
    printf 'processor_grid=%d %d %d\n' "${px}" "${py}" "${pz}"
    printf 'max_unknowns=%d\n' "${max_unknowns}"
    printf 'executable=%s\n' "${executable}"
    printf 'launcher=%s\n' "${launcher_text}"
    printf 'OMP_NUM_THREADS=1\n'
    [[ -z "${gpu_aware_env}" ]] || printf '%s\n' "${gpu_aware_env}"
    printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
    printf 'FLUX_JOB_ID=%s\n' "${FLUX_JOB_ID:-}"
    printf 'PBS_JOBID=%s\n' "${PBS_JOBID:-}"
  } >"${metadata_file}"
  printf 'unknowns\tnx\tny\tnz\tstatus\tstarted\tfinished\tlog\n' >"${summary_file}"
  printf '  Results:      %s\n' "${run_dir}"
else
  printf '  Mode:         dry run\n'
fi

final_status=0
for edge in "${edges[@]}"; do
  set_grid "${edge}"
  driver_command=(
    "${executable}"
    "${driver_args[@]}"
    -n "${NX}" "${NY}" "${NZ}"
    -P "${px}" "${py}" "${pz}"
    -ns "${NSOLVE}"
    -v "${VERBOSE}"
  )

  environment=(env OMP_NUM_THREADS=1)
  if [[ -n "${gpu_aware_env}" ]]; then
    environment+=("${gpu_aware_env}")
  fi
  command=(
    "${environment[@]}"
    "${launcher[@]}"
    "${rank_wrapper[@]}"
    "${driver_command[@]}"
  )
  command_text="$(shell_command "${command[@]}")"

  printf '\n[%d unknowns; grid %d x %d x %d]\n' "${UNKNOWNS}" "${NX}" "${NY}" "${NZ}"
  printf '%s\n' "${command_text}"
  if ((dry_run)); then
    continue
  fi

  log_name="run_${UNKNOWNS}.log"
  log_file="${run_dir}/${log_name}"
  printf '%s\n' "${command_text}" >>"${commands_log}"
  started="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  set +e
  "${command[@]}" 2>&1 | tee "${log_file}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  status="${pipeline_status[0]}"

  finished="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf '%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n' \
    "${UNKNOWNS}" "${NX}" "${NY}" "${NZ}" "${status}" \
    "${started}" "${finished}" "${log_name}" >>"${summary_file}"

  if ((status != 0)); then
    printf 'Run failed with status %d; stopping the scaling sweep.\n' "${status}" >&2
    final_status="${status}"
    break
  fi
done

if ((dry_run == 0)); then
  printf '\nResults saved to %s\n' "${run_dir}"
fi
exit "${final_status}"
