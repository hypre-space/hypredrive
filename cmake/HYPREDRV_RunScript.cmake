# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Wrapper to run hypredrive or example drivers with optional dataset checks.
#
# Inputs (passed via -D):
#   LAUNCH_DIR       : repo root to use as working directory
#   TARGET_BIN       : path to executable
#   MPIEXEC          : MPI launcher executable
#   MPI_NUMPROCS     : number of MPI processes
#   MPI_NUMPROC_FLAG : flag name for process count (e.g., -n or -np)
#   MPI_PREFLAGS     : extra flags before program
#   MPI_POSTFLAGS    : extra flags after program
#   CONFIG_FILE      : optional YAML config (enables dataset checks)
#   CONFIG_FILES     : optional '|' separated YAML configs for one batched run
#   TARGET_ARGS      : optional '|' separated list of extra arguments
#   REQUIRE_CONTAINS : optional '|' separated list of substrings that must appear in output
#   REQUIRE_PATHS    : optional '|' separated list of files/directories that must exist after run
#   PACKER_BIN       : optional linear-system sequence packer to run first
#   SEQ_OUTPUT       : output file required when PACKER_BIN is set
#
if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN)
  message(FATAL_ERROR "HYPREDRV_RunScript.cmake: LAUNCH_DIR and TARGET_BIN must be defined")
endif()
if(DEFINED PACKER_BIN AND NOT PACKER_BIN STREQUAL "" AND
   (NOT DEFINED SEQ_OUTPUT OR SEQ_OUTPUT STREQUAL "" OR
    NOT DEFINED CONFIG_FILE OR CONFIG_FILE STREQUAL ""))
  message(FATAL_ERROR
    "HYPREDRV_RunScript.cmake: PACKER_BIN requires SEQ_OUTPUT and CONFIG_FILE")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_TestLauncher.cmake")
hypredrv_prepare_test_launcher(_launcher_command _launcher_postflags)

set(_config_files "")
if(DEFINED CONFIG_FILE AND NOT CONFIG_FILE STREQUAL "")
  list(APPEND _config_files "${CONFIG_FILE}")
endif()
if(DEFINED CONFIG_FILES AND NOT CONFIG_FILES STREQUAL "")
  string(REPLACE "|" ";" _config_files_from_batch "${CONFIG_FILES}")
  foreach(_config_file IN LISTS _config_files_from_batch)
    if(NOT _config_file STREQUAL "")
      list(APPEND _config_files "${_config_file}")
    endif()
  endforeach()
endif()

# Parse configs to detect referenced dataset directories under 'data/<name>/...'
foreach(_config_file IN LISTS _config_files)
  file(READ "${_config_file}" _cfg_text)
  set(_matches "")
  # Match strings like data/ps3d10pt7 or data/compflow6k (first two path components)
  string(REGEX MATCHALL "data/[A-Za-z0-9_.-]+" _raw_matches "${_cfg_text}")
  if(_raw_matches)
    list(REMOVE_DUPLICATES _raw_matches)
    # Resolve to full paths and check existence/non-empty
    set(_missing_list "")
    foreach(_m IN LISTS _raw_matches)
      # Compute absolute dataset dir
      set(_abs_ds "${LAUNCH_DIR}/${_m}")
      if(EXISTS "${_abs_ds}" AND IS_DIRECTORY "${_abs_ds}")
        file(GLOB _ds_any "${_abs_ds}/*")
        list(LENGTH _ds_any _ds_n)
        if(_ds_n EQUAL 0)
          list(APPEND _missing_list "${_m} (empty)")
        endif()
      else()
        list(APPEND _missing_list "${_m} (missing)")
      endif()
    endforeach()
    if(_missing_list)
      string(REPLACE ";" ", " _missing_str "${_missing_list}")
      message(STATUS "[test] Skipping example: required dataset(s) not available: ${_missing_str}")
      message(STATUS "[test] To fetch datasets: cmake --build . --target data")
      return()
    endif()
  endif()
endforeach()

# Optionally prepare a packed linear-system sequence before launching the test.
if(DEFINED PACKER_BIN AND NOT PACKER_BIN STREQUAL "")
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
    message(FATAL_ERROR
      "Packer failed with exit code ${_pack_ret}\n\nstdout:\n${_pack_out}\n\nstderr:\n${_pack_err}")
  endif()
endif()

# Build argument list
set(_target_args "")
if(DEFINED PACKER_BIN AND NOT PACKER_BIN STREQUAL "")
  list(APPEND _target_args
    -a
    --linear_system:sequence_filename "${SEQ_OUTPUT}")
endif()
if(DEFINED TARGET_ARGS AND NOT TARGET_ARGS STREQUAL "")
  string(REPLACE "|" ";" _target_args_joined "${TARGET_ARGS}")
  foreach(_arg IN LISTS _target_args_joined)
    list(APPEND _target_args "${_arg}")
  endforeach()
endif()
list(APPEND _target_args ${_config_files})

# For debugging purposes only
message(STATUS "[test] TARGET_BIN=${TARGET_BIN}")
if(_launcher_command)
  message(STATUS "[test] MPIEXEC=${MPIEXEC} MPI_NUMPROC_FLAG=${MPI_NUMPROC_FLAG} MPI_NUMPROCS=${MPI_NUMPROCS}")
else()
  message(STATUS "[test] MPIEXEC not defined; running serially")
endif()
message(STATUS "[test] TARGET_ARGS=${_target_args}")

# Run executable (optionally via MPI)
set(_capture_output FALSE)
if(DEFINED REQUIRE_CONTAINS AND NOT REQUIRE_CONTAINS STREQUAL "")
  set(_capture_output TRUE)
endif()

if(_capture_output)
  if(_launcher_command)
    execute_process(
      COMMAND ${_launcher_command} "${TARGET_BIN}" ${_target_args} ${_launcher_postflags}
      WORKING_DIRECTORY "${LAUNCH_DIR}"
      RESULT_VARIABLE _ret
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
    )
  else()
    execute_process(
      COMMAND "${TARGET_BIN}" ${_target_args}
      WORKING_DIRECTORY "${LAUNCH_DIR}"
      RESULT_VARIABLE _ret
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
    )
  endif()

  if(NOT _ret EQUAL 0)
    message(FATAL_ERROR "Executable failed with exit code ${_ret}\n\nstdout:\n${_out}\n\nstderr:\n${_err}")
  endif()

  set(_combined "${_out}\n${_err}")

  # Check required substrings
  string(REPLACE "|" ";" _req_list "${REQUIRE_CONTAINS}")
  foreach(_needle IN LISTS _req_list)
    if(NOT _needle STREQUAL "")
      string(FIND "${_combined}" "${_needle}" _pos)
      if(_pos EQUAL -1)
        message(FATAL_ERROR "Missing required substring '${_needle}'\n\nOutput:\n${_combined}")
      endif()
    endif()
  endforeach()
else()
  if(_launcher_command)
    execute_process(
      COMMAND ${_launcher_command} "${TARGET_BIN}" ${_target_args} ${_launcher_postflags}
      WORKING_DIRECTORY "${LAUNCH_DIR}"
      RESULT_VARIABLE _ret
    )
  else()
    execute_process(
      COMMAND "${TARGET_BIN}" ${_target_args}
      WORKING_DIRECTORY "${LAUNCH_DIR}"
      RESULT_VARIABLE _ret
    )
  endif()

  if(NOT _ret EQUAL 0)
    message(FATAL_ERROR "Executable failed with exit code ${_ret}")
  endif()
endif()

if(DEFINED REQUIRE_PATHS AND NOT REQUIRE_PATHS STREQUAL "")
  string(REPLACE "|" ";" _req_path_list "${REQUIRE_PATHS}")
  foreach(_req_path IN LISTS _req_path_list)
    if(NOT _req_path STREQUAL "")
      if(IS_ABSOLUTE "${_req_path}")
        set(_resolved_req_path "${_req_path}")
      else()
        set(_resolved_req_path "${LAUNCH_DIR}/${_req_path}")
      endif()
      if(NOT EXISTS "${_resolved_req_path}")
        message(FATAL_ERROR "Missing required path '${_req_path}' (resolved to '${_resolved_req_path}')")
      endif()
    endif()
  endforeach()
endif()
