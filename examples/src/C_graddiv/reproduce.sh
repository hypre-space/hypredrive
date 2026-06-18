#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

# Mesh-refinement study for the definite grad-div example. For each grid it
# reports the ADS-preconditioned PCG iteration count (which should stay nearly
# constant -- ADS is uniform in h) and the discretization error against the
# manufactured solution (which should decrease with refinement).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo ""
    echo "Environment overrides:"
    echo "  MPIEXEC_EXECUTABLE   MPI launcher (default: mpirun)"
    echo "  MPIEXEC_NUMPROC_FLAG MPI rank flag (default: -np)"
    echo "  MPI_RANKS            Number of MPI ranks (default: 1)"
    echo "  PGRID                Processor grid 'Px Py Pz' (default: 1 1 1)"
    echo "  EXEC                 graddiv executable path (default: ./graddiv)"
    echo "  YAML                 YAML config (default: pcg-ads.yml)"
    exit 0
fi

MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE:-mpirun}"
MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG:--np}"
MPI_RANKS="${MPI_RANKS:-1}"
PGRID="${PGRID:-1 1 1}"
EXEC="${EXEC:-./graddiv}"
YAML="${YAML:-pcg-ads.yml}"

SIZES=(9 17 33 65)

echo "Definite grad-div refinement study (ADS-preconditioned PCG)"
printf "%-10s %-12s %-16s\n" "grid" "iters" "rel. error"

for n in "${SIZES[@]}"; do
    out=$(${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${EXEC} \
          -i "${YAML}" -P ${PGRID} -n "${n}" "${n}" "${n}" -v 1 2>&1 || true)
    # Iteration count is the last column of the statistics-summary table row.
    iters=$(echo "${out}" | awk -F'|' '/^\| *[0-9]+ *\|/{gsub(/ /,"",$8); print $8}' | tail -1)
    err=$(echo "${out}" | grep -i "Discretization error" | grep -oE "[0-9.]+e[-+][0-9]+" | tail -1 || true)
    printf "%-10s %-12s %-16s\n" "${n}^3" "${iters:-NA}" "${err:-NA}"
done
