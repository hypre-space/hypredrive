#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

set -euo pipefail

MODE="run-and-plot"

for arg in "$@"; do
    case "${arg}" in
        --plot-only)
            MODE="plot-only"
            ;;
        -h|--help)
            MODE="help"
            ;;
        *)
            echo "Unknown option: ${arg}"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

if [[ "${MODE}" == "help" ]]; then
    echo "Usage: $0"
    echo ""
    echo "Runs the three convection-diffusion solver configurations on an"
    echo "elongated duct partitioned along the flow direction, and produces"
    echo "per-linear-system iteration and total-time plots using"
    echo "scripts/analyze_statistics.py."
    echo ""
    echo "The time step grows geometrically across the transient, so each"
    echo "linear system is harder than the previous one: the plots show every"
    echo "solve of the sequence, as in the lid-driven cavity example."
    echo ""
    echo "Modes:"
    echo "  (default)   Run all variants and generate plots"
    echo "  --plot-only Skip runs and only generate plots from existing output logs"
    echo ""
    echo "Environment overrides:"
    echo "  MPIEXEC_EXECUTABLE   MPI launcher executable (default: mpirun)"
    echo "  MPIEXEC_NUMPROC_FLAG MPI rank flag (default: -np)"
    echo "  MPI_RANKS            Number of MPI ranks (default: 16; keep it"
    echo "                       within the machine's physical core count)"
    echo "  EXEC                 convdif executable path (default: ./convdif)"
    echo "  STATS                Statistics script path (default: ../../../scripts/analyze_statistics.py)"
    echo "  CONFIG_DIR           Directory holding the gmres-*.yml configs (default: script directory)"
    echo "  ARGS                 Common problem arguments (without -P)"
    echo ""
    echo "Outputs:"
    echo "  convdif_air.out convdif_amg.out convdif_ilu.out"
    echo "  convdif_512x64x64_iters.png"
    echo "  convdif_512x64x64_total.png"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE:-mpirun}"
MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG:--np}"
MPI_RANKS="${MPI_RANKS:-16}"
EXEC="${EXEC:-./convdif}"
STATS="${STATS:-${SCRIPT_DIR}/../../../scripts/analyze_statistics.py}"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}}"

# Elongated duct (~2.1M cells), convection-dominated, twelve time steps whose
# size doubles every step: the sequence sweeps the systems from mass-dominated
# (CFL < 1) to strongly convection-dominated (CFL > 1000). The domain is
# partitioned along the flow direction only (-P MPI_RANKS 1 1), the regime
# where sweep-based smoothers lose their flow-following character.
ARGS="${ARGS:--n 512 64 64 -L 8 -k 1e-4 -nt 12 -dt 0.01 -dtg 2 -v 1}"

CONFIGS=(
    "gmres-air.yml"
    "gmres-amg.yml"
    "gmres-ilu.yml"
)

OUT_FILES=(
    "convdif_air.out"
    "convdif_amg.out"
    "convdif_ilu.out"
)

METHODS=(
    "GMRES+AMG-AIR"
    "GMRES+AMG"
    "GMRES+ILU"
)

plot_results() {
    for f in "${OUT_FILES[@]}"; do
        if [[ ! -f "${f}" ]]; then
            echo "Missing required output log '${f}'. Run without --plot-only first."
            exit 1
        fi
    done

    echo "Generating per-linear-system plots..."
    PYTHONWARNINGS=ignore::UserWarning MPLBACKEND=Agg python3 "${STATS}" \
        -f "${OUT_FILES[@]}" -ln "${METHODS[@]}" -m "iters+total" \
        --style docs -s "convdif_512x64x64.png"

    # analyze_statistics.py prefixes the plot mode to the file name; rename to
    # the names the user manual embeds (figures/convdif_512x64x64_{iters,total}.png)
    mv "iters_convdif_512x64x64.png" "convdif_512x64x64_iters.png"
    mv "total_convdif_512x64x64.png" "convdif_512x64x64_total.png"
    echo "Done. Plots saved in current directory."
}

if [[ "${MODE}" == "plot-only" ]]; then
    echo "Plot-only mode: using existing output logs."
    plot_results
    exit 0
fi

# max_iter is raised above the configs' 100 so that reruns at higher rank
# counts (where block-Jacobi ILU degrades past 100 iterations) converge
# instead of hitting the cap; it does not change any converged result.
for i in "${!CONFIGS[@]}"; do
    config="${CONFIG_DIR}/${CONFIGS[$i]}"
    out="${OUT_FILES[$i]}"
    echo "Running ${CONFIGS[$i]} on ${MPI_RANKS} rank(s)..."
    "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${MPI_RANKS}" \
        "${EXEC}" -i "${config}" ${ARGS} -P "${MPI_RANKS}" 1 1 \
        -a --solver:gmres:max_iter 300 > "${out}" 2>&1
done

plot_results
