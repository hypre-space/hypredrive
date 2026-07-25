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
STRATEGY_DIR="${ROOT_DIR}/results"
START_UNKNOWNS=100000
NSOLVE=5
VERBOSE=1

# Write the baked-in base linear-solver strategies used by -i. Cluster-specific
# tweaks are applied later with -a (requires a YAML input file).
write_base_strategies() {
  mkdir -p "${STRATEGY_DIR}"

  cat >"${STRATEGY_DIR}/lap-7.yml" <<'EOF'
# Base strategy for lap-7 (7-point Laplacian): PCG + BoomerAMG (poisson).
general:
  use_vendor_spmv: on

solver: pcg

preconditioner:
  preset: poisson
EOF

  cat >"${STRATEGY_DIR}/lap-27.yml" <<'EOF'
# Base strategy for lap-27 (27-point Laplacian): PCG + BoomerAMG (poisson).
general:
  use_vendor_spmv: on

solver: pcg

preconditioner:
  preset: poisson
EOF

  cat >"${STRATEGY_DIR}/elast.yml" <<'EOF'
# Base strategy for elast (3D linear elasticity): PCG + elasticity AMG.
# Keep AMG coarsening options aligned with the elasticity_3D precon preset
# that the driver applies by default after YAML parse.
general:
  use_vendor_spmv: on

solver: pcg

preconditioner:
  amg:
    coarsening:
      num_functions: 3
      strong_th: 0.8
EOF
}

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
  [tuo-cpu]=96000000
  [tuo-gpu-cpx]=96000000
  [tuo-gpu-spx]=96000000
  [polaris-cpu]=64000000
  [polaris-gpu]=12000000
  [aurora]=64000000
  [frontier-cpu]=96000000
  [frontier-gpu]=96000000
  [tioga-cpu]=64000000
  [tioga-gpu]=64000000
)
declare -A ELAST_CAPS=(
  [dane]=64000000
  [matrix]=26000000
  [tuo-cpu]=64000000
  [tuo-gpu-cpx]=60000000
  [tuo-gpu-spx]=60000000
  [polaris-cpu]=32000000
  [polaris-gpu]=12000000
  [aurora]=32000000
  [frontier-cpu]=64000000
  [frontier-gpu]=64000000
  [tioga-cpu]=32000000
  [tioga-gpu]=32000000
)

usage() {
  cat <<'EOF'
Usage:
  scripts/node_scaling.sh -m MACHINE -p PROBLEM [options]
  scripts/node_scaling.sh --summary [RESULTS_DIR]
  scripts/node_scaling.sh -m MACHINE --pack
  scripts/node_scaling.sh [--plot1|--plot2] SUMMARY.tsv [-p PROBLEM]

Run a single-node problem-size scaling study inside an interactive allocation,
or post-process an existing results tree.

Required (run mode):
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
      --summary [DIR]     Post-process results only; write DIR/summary.tsv
                          from the newest dated run of each machine/problem
                          (default DIR: HYPREDRV_SCALING_RESULTS_DIR or
                          results/node-scaling)
      --pack              Tar all results for -m MACHINE (and MACHINE-*
                          family members) into
                          <results>/tarballs/<machine>.tar.gz
      --plot1 FILE        Plot total time vs problem size (log-log) from
                          aggregate summary FILE; all problems unless -p
                          is given; writes plot1-<problem>.png/.pdf
      --plot2 FILE        Side-by-side setup/solve times vs problem size
                          (log-log); all problems unless -p is given;
                          shared legend; writes plot2-<problem>.png/.pdf
  -h, --help              Show this help

Environment:
  HYPREDRV_SCALING_RESULTS_DIR
                          Output root (default: results/node-scaling)

Base strategies:
  results/lap-7.yml, results/lap-27.yml, and results/elast.yml are written
  from baked-in YAML and passed to the driver with -i. Selected machines add
  trailing -a overrides (AMD GPU configs disable vendor SpMV).

Examples:
  scripts/node_scaling.sh -m dane -p lap-7
  scripts/node_scaling.sh -m tuo-gpu-cpx -p lap-27 --max-unknowns 80000000
  scripts/node_scaling.sh -m polaris-gpu -p elast --dry-run
  scripts/node_scaling.sh -m aurora -p lap-7 --dry-run
  scripts/node_scaling.sh -m frontier-gpu -p lap-27 --dry-run
  scripts/node_scaling.sh -m tioga-gpu -p elast --dry-run
  scripts/node_scaling.sh -m dane -p lap-7 -e install/bin/laplacian
  scripts/node_scaling.sh --summary ~/workspace/hypredrive-node-scaling
  HYPREDRV_SCALING_RESULTS_DIR=~/workspace/hypredrive-node-scaling \\
    scripts/node_scaling.sh -m dane --pack
  HYPREDRV_SCALING_RESULTS_DIR=~/workspace/hypredrive-node-scaling \\
    scripts/node_scaling.sh -m frontier --pack
  HYPREDRV_SCALING_RESULTS_DIR=~/workspace/hypredrive-node-scaling \\
    scripts/node_scaling.sh -m tuo --pack
  scripts/node_scaling.sh \\
    --plot1 ~/workspace/hypredrive-node-scaling/summary.tsv
  scripts/node_scaling.sh -p lap-7 \\
    --plot2 ~/workspace/hypredrive-node-scaling/summary.tsv

Notes:
  - The script uses one complete compute node and never requests an allocation.
  - tuo-gpu-cpx requires an allocation created with --amd-gpumode=CPX.
  - --summary timings are the minimum setup/solve from each STATISTICS SUMMARY
    table; total = setup + solve.
  - --pack archives every dated run under matching trees: exact -m name and
    any <name>-* siblings (e.g. -m tuo packs tuo-cpu / tuo-gpu-cpx /
    tuo-gpu-spx when present; -m frontier packs frontier-cpu / frontier-gpu).
  - --plot1/--plot2 read an aggregate summary.tsv; omit -p to plot every
    problem present in the file.
EOF
}

# Resolve a summary path and print its problem names (preferred order first).
problems_from_summary() {
  local summary_file="$1"
  local preferred p
  declare -A seen=()

  summary_file="${summary_file/#\~/${HOME}}"
  if [[ "${summary_file}" != /* ]]; then
    summary_file="${ROOT_DIR}/${summary_file}"
  fi
  [[ -f "${summary_file}" ]] || die "summary file not found: ${summary_file}"

  preferred=(lap-7 lap-27 elast)
  while IFS=$'\t' read -r _mach prob _rest; do
    [[ "${_mach}" == "machine" ]] && continue
    [[ -n "${prob}" ]] || continue
    seen["${prob}"]=1
  done <"${summary_file}"

  for p in "${preferred[@]}"; do
    [[ -n "${seen[$p]+x}" ]] && printf '%s\n' "${p}"
  done
  for p in "${!seen[@]}"; do
    case "${p}" in
      lap-7|lap-27|elast) ;;
      *) printf '%s\n' "${p}" ;;
    esac
  done
}

# Extract min setup/solve (and total) from a driver log STATISTICS SUMMARY table.
# Prints: setup<TAB>solve<TAB>total   or nothing if the table is missing.
min_timings_from_log() {
  awk '
    /^\|[[:space:]]*[0-9]+[[:space:]]*\|/ {
      n = split($0, a, "|")
      if (n < 6) next
      setup = a[4]; solve = a[5]
      gsub(/[[:space:]]/, "", setup)
      gsub(/[[:space:]]/, "", solve)
      if (setup ~ /^[0-9]+([.][0-9]+)?$/) {
        if (min_setup == "" || setup + 0 < min_setup + 0) min_setup = setup
      }
      if (solve ~ /^[0-9]+([.][0-9]+)?$/) {
        if (min_solve == "" || solve + 0 < min_solve + 0) min_solve = solve
      }
    }
    END {
      if (min_setup != "" && min_solve != "")
        printf "%.6g\t%.6g\t%.6g\n", min_setup, min_solve, min_setup + min_solve
    }
  ' "$1"
}

# Scan RESULTS_DIR for machine/problem pairs, use the newest dated subfolder of
# each, and write RESULTS_DIR/summary.tsv with min setup/solve/total per size.
write_aggregate_summary() {
  local root="$1"
  local out tmp machine_dir problem_dir machine problem latest run_dir
  local log_file unknowns timings base

  [[ -d "${root}" ]] || die "results directory not found: ${root}"
  out="${root}/summary.tsv"
  tmp="$(mktemp)"

  for machine_dir in "${root}"/*/; do
    [[ -d "${machine_dir}" ]] || continue
    machine="${machine_dir%/}"
    machine="${machine##*/}"
    [[ "${machine}" == "tarballs" ]] && continue

    for problem_dir in "${machine_dir}"*/; do
      [[ -d "${problem_dir}" ]] || continue
      problem="${problem_dir%/}"
      problem="${problem##*/}"

      latest=""
      for d in "${problem_dir}"*/; do
        [[ -d "${d}" ]] || continue
        base="${d%/}"
        base="${base##*/}"
        if [[ -z "${latest}" || "${base}" > "${latest}" ]]; then
          latest="${base}"
        fi
      done
      [[ -n "${latest}" ]] || continue
      run_dir="${problem_dir}${latest}"

      printf '  %s / %s <- %s\n' "${machine}" "${problem}" "${latest}"
      for log_file in "${run_dir}"/run_*.log; do
        [[ -f "${log_file}" ]] || continue
        base="${log_file##*/}"
        unknowns="${base#run_}"
        unknowns="${unknowns%.log}"
        timings="$(min_timings_from_log "${log_file}")"
        [[ -n "${timings}" ]] || continue
        printf '%s\t%s\t%s\t%s\n' "${machine}" "${problem}" "${unknowns}" "${timings}"
      done >>"${tmp}"
    done
  done

  {
    printf 'machine\tproblem\tunknowns\tsetup\tsolve\ttotal\n'
    if [[ -s "${tmp}" ]]; then
      sort -t$'\t' -k1,1 -k2,2 -k3,3n "${tmp}"
    fi
  } >"${out}"
  rm -f "${tmp}"
  printf 'Wrote %s\n' "${out}"
}

# Tar all results for MACHINE (exact match and MACHINE-* family members)
# under RESULTS_DIR into RESULTS_DIR/tarballs/MACHINE.tar.gz.
pack_machine_results() {
  local root="$1"
  local machine="$2"
  local out_dir out d name
  local -a members=()

  root="${root/#\~/${HOME}}"
  if [[ "${root}" != /* ]]; then
    root="${ROOT_DIR}/${root}"
  fi
  out_dir="${root}/tarballs"
  out="${out_dir}/${machine}.tar.gz"

  for d in "${root}"/*/; do
    [[ -d "${d}" ]] || continue
    name="${d%/}"
    name="${name##*/}"
    [[ "${name}" == "tarballs" ]] && continue
    if [[ "${name}" == "${machine}" || "${name}" == "${machine}"-* ]]; then
      members+=("${name}")
      printf '  + %s\n' "${name}"
    fi
  done
  ((${#members[@]} > 0)) ||
    die "no results for machine/family '${machine}' under ${root}"

  mkdir -p "${out_dir}"
  tar -C "${root}" -czf "${out}" "${members[@]}"
  printf 'Wrote %s (%d trees)\n' "${out}" "${#members[@]}"
}

# Plot total time vs unknowns for one problem across all machines in SUMMARY.
plot1_total_vs_size() {
  local summary_file="$1"
  local problem="$2"
  local out_png

  summary_file="${summary_file/#\~/${HOME}}"
  if [[ "${summary_file}" != /* ]]; then
    summary_file="${ROOT_DIR}/${summary_file}"
  fi
  [[ -f "${summary_file}" ]] || die "summary file not found: ${summary_file}"
  [[ -n "${problem}" ]] || die "internal error: empty problem for --plot1"

  out_png="$(cd "$(dirname "${summary_file}")" && pwd)/plot1-${problem}.png"
  python3 - "${summary_file}" "${problem}" "${out_png}" <<'PY'
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

summary_path = Path(sys.argv[1])
problem = sys.argv[2]
out_png = Path(sys.argv[3])

PROBLEM_TITLE = {
    "lap-7": "7-point Laplacian",
    "lap-27": "27-point Laplacian",
    "elast": "3D Linear Elasticity",
}

# Presentation palette (Okabe–Ito) + distinct markers per known machine.
MACHINE_STYLE = {
    "dane":          ("#0072B2", "o"),
    "matrix":        ("#E69F00", "s"),
    "tuo-cpu":       ("#009E73", "^"),
    "tuo-gpu-cpx":   ("#56B4E9", "D"),
    "tuo-gpu-spx":   ("#CC79A7", "v"),
    "polaris-cpu":   ("#D55E00", "P"),
    "polaris-gpu":   ("#AA3377", "X"),
    "aurora":        ("#000000", "h"),
    "frontier-cpu":  ("#332288", "o"),
    "frontier-gpu":  ("#66CCEE", "s"),
    "tioga-cpu":     ("#117733", "^"),
    "tioga-gpu":     ("#999933", "D"),
}
FALLBACK_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#AA3377", "#000000", "#332288", "#66CCEE",
]
FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "p"]

rows = []
with summary_path.open() as fh:
    header = fh.readline().rstrip("\n").split("\t")
    idx = {name: i for i, name in enumerate(header)}
    for key in ("machine", "problem", "unknowns", "total"):
        if key not in idx:
            sys.exit(f"summary is missing column '{key}'")
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(idx.values()):
            continue
        if parts[idx["problem"]] != problem:
            continue
        rows.append(
            (
                parts[idx["machine"]],
                float(parts[idx["unknowns"]]),
                float(parts[idx["total"]]),
            )
        )

if not rows:
    sys.exit(f"no rows for problem '{problem}' in {summary_path}")

by_machine = {}
for machine, unknowns, total in rows:
    by_machine.setdefault(machine, []).append((unknowns, total))
for machine in by_machine:
    by_machine[machine].sort(key=lambda t: t[0])

machines = sorted(by_machine)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.linewidth": 1.35,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#9a9a9a",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#b0b0b0",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=160)
fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.28)

for i, machine in enumerate(machines):
    pts = by_machine[machine]
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    color, marker = MACHINE_STYLE.get(
        machine,
        (FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
         FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]),
    )
    ax.loglog(
        x,
        y,
        color=color,
        marker=marker,
        markersize=8.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.9,
        linewidth=2.4,
        label=machine,
        zorder=3,
    )

ax.grid(True, which="major", zorder=0)
ax.grid(True, which="minor", alpha=0.25, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.set_xlabel("Problem size (unknowns)", fontsize=16, labelpad=10)
ax.set_ylabel("Total time (s)", fontsize=16, labelpad=10)
ax.set_title(
    f"{PROBLEM_TITLE.get(problem, problem)}  ·  Single-node BoomerAMG-CG scaling",
    fontsize=18,
    fontweight="semibold",
    pad=14,
    color="#111111",
)
ax.tick_params(axis="both", which="major", labelsize=13, length=6, width=1.1)
ax.tick_params(axis="both", which="minor", labelsize=10, length=3.5, width=0.8)

handles, labels = ax.get_legend_handles_labels()
ncol = min(4, max(1, len(labels)))
leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.54, 0.105),
    ncol=ncol,
    fontsize=15,
    handlelength=2.4,
    columnspacing=1.4,
    handletextpad=0.6,
    borderaxespad=0.0,
    frameon=True,
    fancybox=True,
    edgecolor="#b0b0b0",
    framealpha=0.95,
    borderpad=0.8,
)
frame = leg.get_frame()
frame.set_linewidth(1.05)
frame.set_boxstyle("round", pad=0.45, rounding_size=0.35)

fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.25)
out_pdf = out_png.with_suffix(".pdf")
fig.savefig(out_pdf, dpi=220, bbox_inches="tight", pad_inches=0.25)
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
PY
}

# Side-by-side setup/solve vs unknowns for one problem across all machines.
plot2_setup_solve_vs_size() {
  local summary_file="$1"
  local problem="$2"
  local out_png

  summary_file="${summary_file/#\~/${HOME}}"
  if [[ "${summary_file}" != /* ]]; then
    summary_file="${ROOT_DIR}/${summary_file}"
  fi
  [[ -f "${summary_file}" ]] || die "summary file not found: ${summary_file}"
  [[ -n "${problem}" ]] || die "internal error: empty problem for --plot2"

  out_png="$(cd "$(dirname "${summary_file}")" && pwd)/plot2-${problem}.png"
  python3 - "${summary_file}" "${problem}" "${out_png}" <<'PY'
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

summary_path = Path(sys.argv[1])
problem = sys.argv[2]
out_png = Path(sys.argv[3])

PROBLEM_TITLE = {
    "lap-7": "7-point Laplacian",
    "lap-27": "27-point Laplacian",
    "elast": "3D Linear Elasticity",
}

MACHINE_STYLE = {
    "dane":          ("#0072B2", "o"),
    "matrix":        ("#E69F00", "s"),
    "tuo-cpu":       ("#009E73", "^"),
    "tuo-gpu-cpx":   ("#56B4E9", "D"),
    "tuo-gpu-spx":   ("#CC79A7", "v"),
    "polaris-cpu":   ("#D55E00", "P"),
    "polaris-gpu":   ("#AA3377", "X"),
    "aurora":        ("#000000", "h"),
    "frontier-cpu":  ("#332288", "o"),
    "frontier-gpu":  ("#66CCEE", "s"),
    "tioga-cpu":     ("#117733", "^"),
    "tioga-gpu":     ("#999933", "D"),
}
FALLBACK_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#AA3377", "#000000", "#332288", "#66CCEE",
]
FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "p"]

rows = []
with summary_path.open() as fh:
    header = fh.readline().rstrip("\n").split("\t")
    idx = {name: i for i, name in enumerate(header)}
    for key in ("machine", "problem", "unknowns", "setup", "solve"):
        if key not in idx:
            sys.exit(f"summary is missing column '{key}'")
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(idx.values()):
            continue
        if parts[idx["problem"]] != problem:
            continue
        rows.append(
            (
                parts[idx["machine"]],
                float(parts[idx["unknowns"]]),
                float(parts[idx["setup"]]),
                float(parts[idx["solve"]]),
            )
        )

if not rows:
    sys.exit(f"no rows for problem '{problem}' in {summary_path}")

by_machine = {}
for machine, unknowns, setup, solve in rows:
    by_machine.setdefault(machine, []).append((unknowns, setup, solve))
for machine in by_machine:
    by_machine[machine].sort(key=lambda t: t[0])

machines = sorted(by_machine)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.linewidth": 1.35,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#9a9a9a",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#b0b0b0",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2), dpi=160, sharey=False)
fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.28, wspace=0.18)

panel_specs = (
    (axes[0], 1, "Setup time (s)", "Setup"),
    (axes[1], 2, "Solve time (s)", "Solve"),
)
panel_box = dict(
    boxstyle="round,pad=0.4,rounding_size=0.3",
    facecolor="white",
    edgecolor="#b0b0b0",
    linewidth=1.05,
    alpha=0.95,
)

for ax, y_idx, ylabel, panel_title in panel_specs:
    for i, machine in enumerate(machines):
        pts = by_machine[machine]
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[y_idx] for p in pts], dtype=float)
        color, marker = MACHINE_STYLE.get(
            machine,
            (FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
             FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]),
        )
        ax.loglog(
            x,
            y,
            color=color,
            marker=marker,
            markersize=8.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=2.4,
            label=machine,
            zorder=3,
        )

    ax.grid(True, which="major", zorder=0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_xlabel("Problem size (unknowns)", fontsize=15, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
    ax.text(
        0.5,
        0.965,
        panel_title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color="#111111",
        bbox=panel_box,
        zorder=5,
    )
    ax.tick_params(axis="both", which="major", labelsize=12.5, length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=3.5, width=0.8)

fig.suptitle(
    f"{PROBLEM_TITLE.get(problem, problem)}  ·  Single-node BoomerAMG-CG scaling",
    fontsize=18,
    fontweight="semibold",
    y=0.97,
    color="#111111",
)

handles, labels = axes[0].get_legend_handles_labels()
ncol = min(5, max(1, len(labels)))
leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.525, 0.105),
    ncol=ncol,
    fontsize=15,
    handlelength=2.4,
    columnspacing=1.4,
    handletextpad=0.6,
    borderaxespad=0.0,
    frameon=True,
    fancybox=True,
    edgecolor="#b0b0b0",
    framealpha=0.95,
    borderpad=0.8,
)
frame = leg.get_frame()
frame.set_linewidth(1.05)
frame.set_boxstyle("round", pad=0.45, rounding_size=0.35)

fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.25)
out_pdf = out_png.with_suffix(".pdf")
fig.savefig(out_pdf, dpi=220, bbox_inches="tight", pad_inches=0.25)
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")
PY
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
summary_mode=0
summary_dir=""
pack_mode=0
plot1_mode=0
plot1_file=""
plot2_mode=0
plot2_file=""

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
    --summary)
      summary_mode=1
      if [[ $# -ge 2 && "$2" != -* ]]; then
        summary_dir="$2"
        shift 2
      else
        shift
      fi
      ;;
    --pack)
      pack_mode=1
      shift
      ;;
    --plot1)
      [[ $# -ge 2 ]] || die "missing value for --plot1"
      plot1_mode=1
      plot1_file="$2"
      shift 2
      ;;
    --plot2)
      [[ $# -ge 2 ]] || die "missing value for --plot2"
      plot2_mode=1
      plot2_file="$2"
      shift 2
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

if ((summary_mode)); then
  if [[ -z "${summary_dir}" ]]; then
    summary_dir="${RESULTS_ROOT}"
  fi
  summary_dir="${summary_dir/#\~/${HOME}}"
  if [[ "${summary_dir}" != /* ]]; then
    summary_dir="${ROOT_DIR}/${summary_dir}"
  fi
  write_aggregate_summary "${summary_dir}"
  exit 0
fi

if ((pack_mode)); then
  [[ -n "${machine}" ]] || die "-m/--machine is required with --pack"
  pack_machine_results "${RESULTS_ROOT}" "${machine}"
  exit 0
fi

if ((plot1_mode)); then
  if [[ -n "${problem}" ]]; then
    plot1_total_vs_size "${plot1_file}" "${problem}"
  else
    mapfile -t _plot_problems < <(problems_from_summary "${plot1_file}")
    ((${#_plot_problems[@]} > 0)) || die "no problems found in ${plot1_file}"
    for problem in "${_plot_problems[@]}"; do
      plot1_total_vs_size "${plot1_file}" "${problem}"
    done
  fi
  exit 0
fi

if ((plot2_mode)); then
  if [[ -n "${problem}" ]]; then
    plot2_setup_solve_vs_size "${plot2_file}" "${problem}"
  else
    mapfile -t _plot_problems < <(problems_from_summary "${plot2_file}")
    ((${#_plot_problems[@]} > 0)) || die "no problems found in ${plot2_file}"
    for problem in "${_plot_problems[@]}"; do
      plot2_setup_solve_vs_size "${plot2_file}" "${problem}"
    done
  fi
  exit 0
fi

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

write_base_strategies

driver_args=()
cli_overrides=()
dofs_per_grid_point=1
strategy_file=""
case "${problem}" in
  lap-7)
    executable="${build_dir}/laplacian"
    strategy_file="${STRATEGY_DIR}/lap-7.yml"
    driver_args=(-i "${strategy_file}" -s 7)
    default_cap="${LAP7_CAPS[${machine}]}"
    ;;
  lap-27)
    executable="${build_dir}/laplacian"
    strategy_file="${STRATEGY_DIR}/lap-27.yml"
    driver_args=(-i "${strategy_file}" -s 27)
    default_cap="${LAP27_CAPS[${machine}]}"
    ;;
  elast)
    executable="${build_dir}/elasticity"
    strategy_file="${STRATEGY_DIR}/elast.yml"
    driver_args=(-i "${strategy_file}")
    dofs_per_grid_point=3
    default_cap="${ELAST_CAPS[${machine}]}"
    ;;
  *)
    die "unsupported problem: ${problem}"
    ;;
esac

# Machine-specific hypredrive -a overrides. Must remain last on the driver
# command line (example drivers treat -a as consuming the rest of argv).
case "${machine}" in
  tuo-gpu-cpx|tuo-gpu-spx|frontier-gpu|tioga-gpu)
    # Prefer hypre's internal SpMV over rocSPARSE on AMD GPU nodes.
    cli_overrides=(-a --general:use_vendor_spmv off)
    ;;
esac

if [[ -n "${executable_override}" ]]; then
  executable="${executable_override}"
fi

[[ -f "${strategy_file}" ]] || die "strategy file was not written: ${strategy_file}"

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
printf '  Strategy:     %s\n' "${strategy_file}"
if ((${#cli_overrides[@]} > 0)); then
  printf '  CLI overrides:%s\n' "$(printf ' %q' "${cli_overrides[@]}")"
else
  printf '  CLI overrides:(none)\n'
fi
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
    printf 'strategy_file=%s\n' "${strategy_file}"
    if ((${#cli_overrides[@]} > 0)); then
      printf 'cli_overrides=%s\n' "$(shell_command "${cli_overrides[@]}")"
    else
      printf 'cli_overrides=\n'
    fi
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
    "${cli_overrides[@]}"
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
