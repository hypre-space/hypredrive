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
    echo "Runs three elasticity solver preset configurations across"
    echo "8 mesh-size variants (~1e4 to ~1e6 DOFs) and produces"
    echo "separate iteration/setup/solve plots using scripts/analyze_statistics.py."
    echo ""
    echo "Modes:"
    echo "  (default)   Run all variants and generate plots"
    echo "  --plot-only Skip runs and only generate plots from existing output logs"
    echo ""
    echo "Environment overrides:"
    echo "  MPIEXEC_EXECUTABLE   MPI launcher executable (default: mpirun)"
    echo "  MPIEXEC_NUMPROC_FLAG MPI rank flag (default: -np)"
    echo "  MPI_RANKS            Number of MPI ranks (default: 1)"
    echo "  PGRID                Processor grid 'Px Py Pz' (default: 1 1 1)"
    echo "  EXEC                 Elasticity executable path (default: ./elasticity)"
    echo "  STATS                Statistics script path (default: ../../../scripts/analyze_statistics.py)"
    echo "  ARGS                 Common elasticity arguments (without -n/-P)"
    echo "  MAX_VARIANTS         Limit number of size variants (default: 8)"
    echo ""
    echo "Outputs:"
    echo "  elasticity_builtin.out"
    echo "  elasticity_sdc.out"
    echo "  elasticity_nodal.out"
    echo "  iters_elasticity_presets_dofs.png"
    echo "  setup_elasticity_presets_dofs.png"
    echo "  solve_elasticity_presets_dofs.png"
    exit 0
fi

MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE:-mpirun}"
MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG:--np}"
MPI_RANKS="${MPI_RANKS:-8}"
PGRID="${PGRID:-2 2 2}"
EXEC="${EXEC:-./elasticity}"
STATS="${STATS:-../../../scripts/analyze_statistics.py}"
ARGS="${ARGS:--L 3 1 1 -ns 1 -v 1}"
MAX_VARIANTS="${MAX_VARIANTS:-8}"

# 8 size variants spanning roughly 1e4 to 1e6 DOFs.
# DOFs = 3 * Nx * Ny * Nz with (Nx, Ny, Nz) listed below.
SIZE_VARIANTS=(
    "33 11 11"   # 11,979
    "39 13 13"   # 19,773
    "48 16 16"   # 36,864
    "60 20 20"   # 72,000
    "75 25 25"   # 140,625
    "93 31 31"   # 268,119
    "114 38 38"  # 493,848
    "144 48 48"  # 995,328
)

PRESETS=(
    "elasticity_3D"
    "elasticity_sdc_3D"
    "elasticity_nodal_3D"
)

OUT_FILES=(
    "elasticity_builtin.out"
    "elasticity_sdc.out"
    "elasticity_nodal.out"
)

METHODS=(
    "elasticity_3D"
    "elasticity_sdc_3D"
    "elasticity_nodal_3D"
)

plot_results() {
    for f in "${OUT_FILES[@]}"; do
        if [[ ! -f "${f}" ]]; then
            echo "Missing required output log '${f}'. Run without --plot-only first."
            exit 1
        fi
    done

    # Sanity-check that each source has enough size points.
    local min_points=2
    for f in "${OUT_FILES[@]}"; do
        local variant_points
        variant_points=$(awk '/Solving linear system #[0-9]+ with [0-9]+ rows and [0-9]+ nonzeros\.\.\./{c++} END{print c+0}' "${f}")
        if (( variant_points < min_points )); then
            echo "Error: '${f}' has only ${variant_points} size point(s)."
            echo "       Need at least ${min_points} points for a meaningful curve."
            echo "       Re-run full sweep (default) or increase MAX_VARIANTS."
            exit 1
        fi
    done

    echo "Generating plots (log-scale X axis, DOFs)..."

    echo "  Plotting: iterations vs DOFs -> iters_elasticity_presets_dofs.png"
    PYTHONWARNINGS=ignore::UserWarning MPLBACKEND=Agg python3 "${STATS}" -f "${OUT_FILES[@]}" -ln "${METHODS[@]}" -m "iters" \
        -t rows -l "DOFs (rows)" -s "elasticity_presets_dofs.png" --log-x -T "Linear solver iterations"

    echo "  Plotting: setup time vs DOFs -> setup_elasticity_presets_dofs.png"
    PYTHONWARNINGS=ignore::UserWarning MPLBACKEND=Agg python3 "${STATS}" -f "${OUT_FILES[@]}" -ln "${METHODS[@]}" -m "setup" \
        -t rows -l "DOFs (rows)" -s "elasticity_presets_dofs.png" --log-x --log-y -T "Linear solver setup time [s]"

    echo "  Plotting: solve time vs DOFs -> solve_elasticity_presets_dofs.png"
    PYTHONWARNINGS=ignore::UserWarning MPLBACKEND=Agg python3 "${STATS}" -f "${OUT_FILES[@]}" -ln "${METHODS[@]}" -m "solve" \
        -t rows -l "DOFs (rows)" -s "elasticity_presets_dofs.png" --log-x --log-y -T "Linear solver solve time [s]"

    echo "Done. Plots saved in current directory."
}

if [[ "${MODE}" == "plot-only" ]]; then
    echo "Plot-only mode: using existing output logs."
    plot_results
    exit 0
fi

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    outfile="${OUT_FILES[$i]}"
    : > "${outfile}"
    echo "Running ${preset} -> ${outfile}"

    for j in "${!SIZE_VARIANTS[@]}"; do
        if (( j >= MAX_VARIANTS )); then
            break
        fi

        read -r nx ny nz <<< "${SIZE_VARIANTS[$j]}"
        dofs=$((3 * nx * ny * nz))

        echo "  Variant $((j + 1)): N=${nx}x${ny}x${nz} (DOFs=${dofs})"
        {
            echo ""
            echo "===== Variant $((j + 1)): N=${nx}x${ny}x${nz}, DOFs=${dofs} ====="
            # Provide rows/nonzeros marker expected by analyze_statistics.py for -t rows mode.
            echo "Solving linear system #${j} with ${dofs} rows and 1 nonzeros..."
        } >> "${outfile}"

        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${EXEC} ${ARGS} \
            -P ${PGRID} -n "${nx}" "${ny}" "${nz}" --solver-preset "${preset}" \
            2>&1 | tee -a "${outfile}"
    done
done

plot_results
