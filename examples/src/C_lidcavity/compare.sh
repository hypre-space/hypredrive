#!/usr/bin/env bash
#/******************************************************************************
#* Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
#* HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#*
#* SPDX-License-Identifier: MIT
#******************************************************************************/

# Compare the Q1-Q1 (stabilized) and Q2-Q1 (Taylor-Hood) discretizations of the
# lid-driven cavity driver over a sweep of Reynolds and CFL numbers, reporting
# time steps, nonlinear/linear iteration counts, the final kinetic energy, and
# wall times.
#
# Both discretizations run at matched velocity resolution: Q2-Q1 with an N x N
# pressure grid has the same (2N-1) x (2N-1) velocity grid as Q1-Q1 run with
# -n (2N-1) (2N-1). Adaptive time stepping (-adt) is enabled: the runs start
# from a small time step (CFL = 0.1 on the velocity grid) and grow it until the
# target CFL cap of the sweep. Since the driver caps dt at max_cfl * h with its
# own grid spacing (the pressure grid for Q2-Q1), the script passes the target
# divided by two to Q2-Q1, so both discretizations share the same physical dt
# cap relative to the common velocity grid. Each discretization uses its
# built-in default solver configuration: FGMRES+AMG-ILU for Q1-Q1 and
# FGMRES+MGR (velocities as F points, pressure as C point) for Q2-Q1, where the
# unpinned Q2-Q1 pressure gauge is fixed by the null space projection. The
# kinetic energies are expected to agree to discretization accuracy (a few
# percent at these resolutions), not to machine precision.

EXE=""
N=64
TF=2.0
NP=16
PGRID="4 4"
RE_LIST="100 400"
CFL_LIST="1 10 100"
MPIEXEC_CMD="${MPIEXEC:-mpiexec}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exe)      EXE="$2";      shift 2 ;;
        -n)         N="$2";        shift 2 ;;
        -tf)        TF="$2";       shift 2 ;;
        -np)        NP="$2";       shift 2 ;;
        -P)         PGRID="$2";    shift 2 ;;
        --re)       RE_LIST="$2";  shift 2 ;;
        --cfl)      CFL_LIST="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --exe <path>   Path to the lidcavity binary (default: autodetect)"
            echo "  -n <N>         Q2-Q1 pressure grid size; Q1-Q1 uses 2N-1 (default: 64)"
            echo "  -tf <T>        Final simulation time (default: 2.0)"
            echo "  -np <np>       Number of MPI ranks (default: 16)"
            echo "  -P \"<Px Py>\"   Processor grid, required when np > 1 (default: \"4 4\")"
            echo "  --re \"<list>\"  Reynolds numbers to sweep (default: \"100 400\")"
            echo "  --cfl \"<list>\" Target CFL caps to sweep (default: \"1 10 100\")"
            exit 0
            ;;
        *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
    esac
done

# Autodetect the binary if not given
if [[ -z "${EXE}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in "${script_dir}"/../../../build*/lidcavity \
                     "${script_dir}"/../../../build*/examples/src/C_lidcavity/lidcavity; do
        if [[ -x "${candidate}" ]]; then EXE="${candidate}"; break; fi
    done
fi
if [[ ! -x "${EXE}" ]]; then
    echo "Error: lidcavity binary not found; pass it with --exe <path>"
    exit 1
fi

NV=$((2 * N - 1))
LAUNCH=("${MPIEXEC_CMD}" -n "${NP}")
PARGS=()
if [[ ${NP} -gt 1 ]]; then
    if [[ -z "${PGRID}" ]]; then
        echo "Error: -P \"<Px Py>\" is required when np > 1"
        exit 1
    fi
    read -r -a pgrid_arr <<< "${PGRID}"
    PARGS=(-P "${pgrid_arr[0]}" "${pgrid_arr[1]}")
fi

run_case() {
    local disc="$1" re="$2" dt="$3" cfl_cap="$4" logfile="$5"
    local disc_args=()
    local n_nodes

    if [[ "${disc}" == "q2q1" ]]; then
        disc_args=(-disc q2q1)
        n_nodes=${N}
        # the dt cap is max_cfl * h with h the pressure grid spacing (2 * h_v)
        cfl_cap=$(awk -v c="${cfl_cap}" 'BEGIN { printf "%g", c / 2 }')
    else
        n_nodes=${NV}
    fi

    local t0 t1
    t0=$(date +%s.%N)
    "${LAUNCH[@]}" "${EXE}" "${disc_args[@]}" -n "${n_nodes}" "${n_nodes}" \
        "${PARGS[@]}" -Re "${re}" -dt "${dt}" -tf "${TF}" -adt -cfl "${cfl_cap}" \
        -v 1 > "${logfile}" 2>&1
    local rc=$?
    t1=$(date +%s.%N)

    local steps newton lin_avg ke wall
    wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.2f", b - a }')
    if [[ ${rc} -ne 0 ]]; then
        printf "%-5s %-6s %-9s %-6s %-6s %-7s %-8s %-13s %-8s\n" \
               "${re}" "${cfl}" "${dt}" "${disc}" "-" "-" "-" "FAILED(rc=${rc})" "${wall}"
        return
    fi

    steps=$(sed -n 's/^Time step: *\([0-9]*\).*/\1/p' "${logfile}" | tail -1)
    newton=$(grep -c '^Time step:' "${logfile}")
    lin_avg=$(sed -n 's/.*Lin: *\([0-9]*\).*/\1/p' "${logfile}" |
              awk '{ s += $1; n++ } END { if (n) printf "%.1f", s / n; else print "-" }')
    ke=$(sed -n 's/^Final kinetic energy: *//p' "${logfile}" | tail -1)

    printf "%-5s %-6s %-9s %-6s %-6s %-6s %-7s %-13s %-8s\n" \
           "${re}" "${cfl}" "${dt}" "${disc}" "${steps:--}" "${newton}" \
           "${lin_avg}" "${ke:--}" "${wall}"
}

logdir=$(mktemp -d /tmp/lidcavity-compare-XXXXXX)
dt0=$(awk -v m="$((NV - 1))" 'BEGIN { printf "%.6f", 0.1 / m }')
echo ""
echo "Lid-driven cavity: Q1-Q1 (n = ${NV}) vs Q2-Q1 (n = ${N}) at matched velocity grids"
echo "tf = ${TF}, np = ${NP}, adaptive time stepping from dt0 = ${dt0} (CFL 0.1 on the"
echo "velocity grid, h_v = 1 / $((NV - 1))) up to the target CFL cap; logs in ${logdir}"
echo ""
printf "%-5s %-6s %-9s %-6s %-6s %-6s %-7s %-13s %-8s\n" \
       "Re" "CFL" "dt0" "disc" "steps" "newton" "lin/nl" "kinetic-en" "time[s]"
echo "-------------------------------------------------------------------------"
for re in ${RE_LIST}; do
    for cfl in ${CFL_LIST}; do
        run_case q1q1 "${re}" "${dt0}" "${cfl}" "${logdir}/q1q1_Re${re}_cfl${cfl}.log"
        run_case q2q1 "${re}" "${dt0}" "${cfl}" "${logdir}/q2q1_Re${re}_cfl${cfl}.log"
    done
done
echo ""
