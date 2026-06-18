#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

# Two studies for the definite Maxwell (AMS) example:
#
#   refine  (default)  Mesh-refinement study: the AMS-preconditioned PCG iteration
#                      count stays nearly constant in h while the discretization
#                      error against the manufactured solution decreases like O(h^2).
#
#   sweep              Coefficient sweep on a 64^3 mesh: fix muinv = 1 and vary sigma
#                      over six orders of magnitude (curl-dominated to mass-dominated).
#                      Renders a side-by-side iteration line plot comparing AMS to
#                      generic preconditioners (BoomerAMG, RAS(1)-ILU0) plus a stacked
#                      AMS setup/solve-time bar via analyze_statistics.py.

set -euo pipefail

MODE="${1:-refine}"
case "${MODE}" in
    refine|sweep) ;;
    -h|--help|help)
        echo "Usage: $0 [refine|sweep]"
        echo ""
        echo "Modes:"
        echo "  refine  (default)  mesh-refinement study (iters + discretization error)"
        echo "  sweep              fix muinv=1, sweep sigma over 6 orders of magnitude on a"
        echo "                     64^3 mesh; compare AMS vs AMG/RAS iters + AMS setup/solve"
        echo ""
        echo "Environment overrides:"
        echo "  MPIEXEC_EXECUTABLE   MPI launcher (default: mpirun)"
        echo "  MPIEXEC_NUMPROC_FLAG MPI rank flag (default: -np)"
        echo "  MPI_RANKS            Number of MPI ranks (default: 1)"
        echo "  PGRID                Processor grid 'Px Py Pz' (default: 1 1 1)"
        echo "  EXEC                 maxwell executable path (default: ./maxwell)"
        echo "  YAML                 YAML config (default: pcg-ams.yml)"
        echo "  STATS                analyze_statistics.py path"
        echo "  SWEEP_N              nodes per dim for the sweep (default: 65 = 64^3 cells)"
        echo "  SIGMA_VALUES         sigma values, muinv=1 (default: '0.001 0.01 0.1 1 10 100 1000')"
        echo "  OUT_PREFIX           output figure name prefix (default: maxwell_sigma_sweep)"
        exit 0 ;;
    *) echo "Unknown mode '${MODE}'. Use -h for help."; exit 1 ;;
esac

MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE:-mpirun}"
MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG:--np}"
MPI_RANKS="${MPI_RANKS:-1}"
PGRID="${PGRID:-1 1 1}"
EXEC="${EXEC:-./maxwell}"
YAML="${YAML:-pcg-ams.yml}"
STATS="${STATS:-../../../scripts/analyze_statistics.py}"

run() { ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${EXEC} -P ${PGRID} "$@"; }

# Column accessors for the statistics-summary table row "| 0 | | setup | solve | r0 | rr | iters |"
col_iters() { awk -F'|' '/^\| *[0-9]+ *\|/{gsub(/ /,"",$8); print $8}'; }
col_setup() { awk -F'|' '/^\| *[0-9]+ *\|/{gsub(/ /,"",$4); print $4}'; }
col_solve() { awk -F'|' '/^\| *[0-9]+ *\|/{gsub(/ /,"",$5); print $5}'; }

if [[ "${MODE}" == "refine" ]]; then
    SIZES=(9 17 33 65)
    echo "Definite Maxwell refinement study (AMS-preconditioned PCG)"
    printf "%-10s %-12s %-16s\n" "grid" "iters" "rel. error"
    for n in "${SIZES[@]}"; do
        out=$(run -i "${YAML}" -n "${n}" "${n}" "${n}" -v 1 2>&1 || true)
        iters=$(echo "${out}" | col_iters | tail -1)
        err=$(echo "${out}" | grep -i "Discretization error" | grep -oE "[0-9.]+e[-+][0-9]+" | tail -1 || true)
        printf "%-10s %-12s %-16s\n" "${n}^3" "${iters:-NA}" "${err:-NA}"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# sweep mode: fix muinv = 1 and vary sigma over six orders of magnitude. Since
# the conditioning depends on the curl-to-mass ratio sigma/muinv (= sigma here)
# and the sweep crosses 1 in both directions, a single sweep covers the curl-
# dominated and mass-dominated regimes. The iteration plot compares AMS against
# two generic baselines (BoomerAMG and RAS(1)-ILU0), which are not robust on
# H(curl) systems; the stacked time bar reports AMS setup/solve.
# ---------------------------------------------------------------------------
SWEEP_N="${SWEEP_N:-65}"
read -r -a SIGMA_VALUES <<< "${SIGMA_VALUES:-0.001 0.01 0.1 1 10 100 1000}"
OUT_PREFIX="${OUT_PREFIX:-maxwell_sigma_sweep}"
CELLS=$((SWEEP_N - 1))

# Solver configurations to compare: "yaml|legend". The legend doubles as the
# object name so each solver's per-sigma runs group into one line.
SOLVERS=(
    "${YAML}|PCG-AMS"
    "pcg-amg.yml|PCG-AMG"
    "gmres-ras1-ilu0.yml|GMRES-RAS1-ILU0"
)

echo "Definite Maxwell coefficient sweep (muinv=1), ${CELLS}^3 cells"
ITER_LOGS=()
NATIVE_LOG=""
for cfg in "${SOLVERS[@]}"; do
    yaml="${cfg%%|*}"
    name="${cfg##*|}"
    log="${OUT_PREFIX}_${name}.log"
    : > "${log}"
    printf "\n[%s]\n" "${name}"
    printf "%-10s %-10s %-12s %-12s\n" "sigma" "iters" "setup[s]" "solve[s]"
    for s in "${SIGMA_VALUES[@]}"; do
        out=$(run -i "${yaml}" --name "${name}" -n "${SWEEP_N}" "${SWEEP_N}" "${SWEEP_N}" -muinv 1 -sigma "${s}" -v 1 2>&1 || true)
        echo "${out}" >> "${log}"
        printf "%-10s %-10s %-12s %-12s\n" "${s}" \
            "$(echo "${out}" | col_iters)" "$(echo "${out}" | col_setup)" "$(echo "${out}" | col_solve)"
    done
    ITER_LOGS+=("${log}")
    [ "${name}" = "PCG-AMS" ] && NATIVE_LOG="${log}"
done

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import matplotlib" >/dev/null 2>&1; then
    echo "python3/matplotlib not available; skipping figures (data in ${OUT_PREFIX}_*.log)"
    exit 0
fi

echo "Generating figures via analyze_statistics.py ..."
# Left: iteration counts vs sigma, one line per solver (log-y spans the range).
MPLBACKEND=Agg python3 "${STATS}" -f "${ITER_LOGS[@]}" -m iters -t xval --log-x --log-y \
    --xvalues "${SIGMA_VALUES[@]}" -l '$\sigma$  ($\mu^{-1}=1$)' \
    -T 'Iterations vs $\sigma$: AMS vs generic preconditioners' \
    -s "${OUT_PREFIX}.png" >/dev/null 2>&1
# Right: stacked setup/solve time bar for the native AMS solver.
MPLBACKEND=Agg python3 "${STATS}" -f "${NATIVE_LOG}" -m bar+setup+solve \
    -ln "${SIGMA_VALUES[@]}" -l '$\sigma$  ($\mu^{-1}=1$)' \
    -T 'AMS setup/solve time vs $\sigma$' -s "${OUT_PREFIX}.png" >/dev/null 2>&1

if command -v convert >/dev/null 2>&1; then
    convert "iters_${OUT_PREFIX}.png" "stacked_bar_${OUT_PREFIX}.png" \
        -resize x600 +append "${OUT_PREFIX}_panel.png"
    echo "Wrote side-by-side panel: ${OUT_PREFIX}_panel.png"
else
    echo "ImageMagick 'convert' not found; wrote iters_${OUT_PREFIX}.png and stacked_bar_${OUT_PREFIX}.png"
fi
