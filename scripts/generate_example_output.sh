#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/
#
# This script generates the reference output for the configurations
# listed in the examples folder
#

generate_output() {
    local EXAMPLE_ID=$1

    case ${EXAMPLE_ID} in
        1|3|5|6|7|8)
            NP=1
            ;;
        2|4)
            NP=4
            ;;
        *)
            echo "Unknown example ID: ${EXAMPLE_ID}"
            exit 1
    esac

    EXAMPLE_FILE="${HYPREDRIVE_EXAMPLES_DIR}/ex${EXAMPLE_ID}.yml"
    REFERENCE_OUTPUT_FILE="${HYPREDRIVE_EXAMPLES_DIR}/refOutput/ex${EXAMPLE_ID}.txt"
    RUN="mpirun -np ${NP}"

    if [ ! -f ${EXAMPLE_FILE} ]; then
        echo "The example file does not exist: ${EXAMPLE_FILE}"
        exit 1
    fi

    echo -e "Generating reference output for example #${EXAMPLE_ID}..."

    # Generate output
    CMD="${RUN} ${DRIVER} ${EXAMPLE_FILE}"
    cd ${HYPREDRIVE_DIR}
    eval ${CMD} > ${CWD}/hypredrive.temp.out
    cd ${CWD}

    # Postprocess the output file
    sed -r '
        s/(Date and time: ).*/\1YYYY-MM-DD HH:MM:SS/g
        s/(Using HYPREDRV_DEVELOP_STRING: ).*/\1HYPREDRV_VERSION_GOES_HERE/g
        s/(Using HYPRE_DEVELOP_STRING: ).*/\1HYPRE_VERSION_GOES_HERE/g
        s|(.*/hypredrive)( done!)|\${HYPREDRIVE_PATH}/hypredrive\2|g
        /^=+ System Information =+$/,/^=+ System Information =+$/d
    ' "hypredrive.temp.out" > "${REFERENCE_OUTPUT_FILE}"

    rm -rf hypredrive.temp.out
}

# Check usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 example_ID | a | all"
    exit 1
fi

CWD="$(pwd)"
HYPREDRIVE_SCRIPT_DIR="$(realpath "$(dirname "$0")")"
HYPREDRIVE_DIR="$(dirname "$HYPREDRIVE_SCRIPT_DIR")"
HYPREDRIVE_EXAMPLES_DIR="${HYPREDRIVE_DIR}/examples"
DRIVER="${HYPREDRIVE_DIR}/hypredrive"

if [ ! -f ${DRIVER} ]; then
    echo "The hypredrive executable does not exist: ${DRIVER}"
    exit 1
fi

if [[ "$1" == "all" || "$1" == "a" ]]; then
    # Loop through all example IDs and generate output
    EXAMPLE_IDS=({1..8})

    for ID in "${EXAMPLE_IDS[@]}"; do
        generate_output "$ID"
    done
else
    # Process a single example ID
    generate_output "$1"
fi
