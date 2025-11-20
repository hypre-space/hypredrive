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
#   TARGET_ARGS      : optional semicolon-separated list of extra arguments
#
if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN)
  message(FATAL_ERROR "HYPREDRV_RunScript.cmake: LAUNCH_DIR and TARGET_BIN must be defined")
endif()

# Parse CONFIG_FILE to detect referenced dataset directories under 'data/<name>/...'
if(DEFINED CONFIG_FILE AND NOT CONFIG_FILE STREQUAL "")
  file(READ "${CONFIG_FILE}" _cfg_text)
  set(_matches "")
  # Match strings like data/ps3d10pt7 or data/compflow6k (first two path components)
  string(REGEX MATCHALL "data/[A-Za-z0-9_\-\.]+" _raw_matches "${_cfg_text}")
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
endif()

# Build argument list
set(_target_args "")
if(DEFINED TARGET_ARGS AND NOT TARGET_ARGS STREQUAL "")
  string(REPLACE "|" ";" _target_args_joined "${TARGET_ARGS}")
  foreach(_arg IN LISTS _target_args_joined)
    list(APPEND _target_args "${_arg}")
  endforeach()
endif()
if(DEFINED CONFIG_FILE AND NOT CONFIG_FILE STREQUAL "")
  list(APPEND _target_args "${CONFIG_FILE}")
endif()

# Run executable (optionally via MPI)
if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  execute_process(
    COMMAND "${MPIEXEC}" "${MPI_NUMPROC_FLAG}" "${MPI_NUMPROCS}" ${MPI_PREFLAGS} "${TARGET_BIN}" ${_target_args} ${MPI_POSTFLAGS}
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
