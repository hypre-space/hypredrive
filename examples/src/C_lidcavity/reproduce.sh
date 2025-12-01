#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

# Default mode
MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --centerlines)
            MODE="centerlines"
            shift
            ;;
        --solvers)
            MODE="solvers"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--centerlines|--solvers]"
            echo ""
            echo "Modes:"
            echo "  --centerlines  Run validation with different Reynolds numbers and plot centerlines"
            echo "  --solvers      Run with different solver configurations and analyze statistics"
            echo ""
            echo "If no mode is specified, --solvers is used as default."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Default to solvers mode if not specified
if [[ -z "$MODE" ]]; then
    MODE="solvers"
fi

# Common configuration
# Allow override from environment variables
MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE:-mpirun}"
MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG:--np}"
EXEC="${EXEC:-./lidcavity}"

# ============================================================================
# CENTERLINES MODE: Validation with different Reynolds numbers
# ============================================================================
if [[ "$MODE" == "centerlines" ]]; then
    MPI_RANKS="${MPI_RANKS:-16}"
    POSTPROCESS="${POSTPROCESS:-./postprocess.py}"

    # Common arguments shared by all runs
    ARGS="-dt 1 -tf 500 -vis 1 -n 128 128 -P 4 4 -adt -reg -v 1"

    # List of Reynolds numbers
    RE_LIST=("1" "10" "100" "400" "1000" "3200" "5000" "7500")

    # Execution
    for RE in "${RE_LIST[@]}"; do
        # Generate output filename and results files
        filename="lidcavity_Re${RE}_128x128_4x4"
        outfile="${filename}.out"
        resfile="${filename}.pvd"

        # Run driver
        echo "Running: Re=${RE} (${resfile})"
        $MPIEXEC_EXECUTABLE $MPIEXEC_NUMPROC_FLAG $MPI_RANKS $EXEC $ARGS -Re $RE 2>&1 > "$outfile"

        # Plot centerline results
        $POSTPROCESS ${resfile} -c --plot --save lidcavity_128x128 --Re ${RE}
    done

# ============================================================================
# SOLVERS MODE: Different solver configurations
# ============================================================================
elif [[ "$MODE" == "solvers" ]]; then
    MPI_RANKS="${MPI_RANKS:-64}"
    STATS="${STATS:-../../../scripts/analyze_statistics.py}"

    # Common arguments shared by all runs
    ARGS="-P 8 8 -dt 0.01 -tf 50 -n 256 256 -v 1 -adt -reg"

    # List of input files (with extensions)
    CONFIG_FILES=(
        "fgmres-ilu0.yml"
        "fgmres-ilu1.yml"
        "fgmres-ilut_1e-2.yml"
        "fgmres-amg.yml"
        "fgmres-amg-ilut.yml"
    )

    # List of methods
    METHODS=(
        "ILUK(0)"
        "ILUK(1)"
        "ILUT(1e-2)"
        "AMG"
        "AMG+ILUT(1e-2)"
    )

    # Array to store output filenames for the post-process step
    OUT_FILES=()

    # Execution
    for config in "${CONFIG_FILES[@]}"; do
        # Generate output filename: replace .yml with .out
        outfile="${config%.yml}.out"

        # Store this output filename for later use
        OUT_FILES+=("$outfile")

        # Run driver
        echo "Running: $config -> $outfile"
        $MPIEXEC_EXECUTABLE $MPIEXEC_NUMPROC_FLAG $MPI_RANKS $EXEC $ARGS -i "$config" 2>&1 > "$outfile"
    done

    # Post process (iteration and execution time plots)
    python3 "$STATS" -f "${OUT_FILES[@]}" -ll "${METHODS[@]}" -m "iters+total"
else
    echo "Error: Unknown mode: $MODE"
    exit 1
fi
