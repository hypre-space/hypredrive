# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN OR NOT DEFINED PACKER_BIN OR NOT DEFINED SEQ_OUTPUT OR NOT DEFINED CONFIG_FILE)
  message(FATAL_ERROR "HYPREDRV_PackAndRunScript.cmake: required variables are missing")
endif()

if(NOT EXISTS "${LAUNCH_DIR}/data/poromech2k")
  message(STATUS "[test] Skipping example: required dataset not available: data/poromech2k")
  return()
endif()

message(STATUS "[test] Packing sequence into ${SEQ_OUTPUT}")
execute_process(
  COMMAND "${PACKER_BIN}"
          --dirname "${LAUNCH_DIR}/data/poromech2k/np1/ls"
          --matrix-filename "IJ.out.A"
          --rhs-filename "IJ.out.b"
          --dofmap-filename "dofmap.out"
          --init-suffix "0"
          --last-suffix "2"
          --digits-suffix "5"
          --algo "none"
          --output "${SEQ_OUTPUT}"
  WORKING_DIRECTORY "${LAUNCH_DIR}"
  RESULT_VARIABLE _pack_ret
  OUTPUT_VARIABLE _pack_out
  ERROR_VARIABLE _pack_err
)
if(NOT _pack_ret EQUAL 0)
  message(FATAL_ERROR "Packer failed with exit code ${_pack_ret}\n\nstdout:\n${_pack_out}\n\nstderr:\n${_pack_err}")
endif()

set(_target_args
  -q
  -a
  --linear_system:sequence_filename "${SEQ_OUTPUT}"
)

if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  execute_process(
    COMMAND "${MPIEXEC}" "${MPI_NUMPROC_FLAG}" "${MPI_NUMPROCS}" ${MPI_PREFLAGS}
            "${TARGET_BIN}" ${_target_args} "${CONFIG_FILE}" ${MPI_POSTFLAGS}
    WORKING_DIRECTORY "${LAUNCH_DIR}"
    RESULT_VARIABLE _run_ret
    OUTPUT_VARIABLE _run_out
    ERROR_VARIABLE _run_err
  )
else()
  execute_process(
    COMMAND "${TARGET_BIN}" ${_target_args} "${CONFIG_FILE}"
    WORKING_DIRECTORY "${LAUNCH_DIR}"
    RESULT_VARIABLE _run_ret
    OUTPUT_VARIABLE _run_out
    ERROR_VARIABLE _run_err
  )
endif()

if(NOT _run_ret EQUAL 0)
  message(FATAL_ERROR "hypredrive failed with exit code ${_run_ret}\n\nstdout:\n${_run_out}\n\nstderr:\n${_run_err}")
endif()

set(_combined "${_run_out}\n${_run_err}")
if(DEFINED REQUIRE_CONTAINS AND NOT REQUIRE_CONTAINS STREQUAL "")
  string(REPLACE "|" ";" _needles "${REQUIRE_CONTAINS}")
  foreach(_needle IN LISTS _needles)
    if(NOT _needle STREQUAL "")
      string(FIND "${_combined}" "${_needle}" _pos)
      if(_pos EQUAL -1)
        message(FATAL_ERROR "Missing required substring '${_needle}'\n\nOutput:\n${_combined}")
      endif()
    endif()
  endforeach()
endif()
