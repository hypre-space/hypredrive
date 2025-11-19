# Wrapper to run hypredrive only if required datasets exist; otherwise skip gracefully.
#
# Inputs (passed via -D):
#   LAUNCH_DIR       : repo root to use as working directory
#   TARGET_BIN       : path to hypredrive executable
#   MPIEXEC          : MPI launcher executable
#   MPI_NUMPROCS     : number of MPI processes
#   MPI_NUMPROC_FLAG : flag name for process count (e.g., -n or -np)
#   MPI_PREFLAGS     : extra flags before program
#   MPI_POSTFLAGS    : extra flags after program
#   CONFIG_FILE      : path to YAML config to run

if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN OR NOT DEFINED CONFIG_FILE)
  message(FATAL_ERROR "HYPREDRV_RunScript.cmake: LAUNCH_DIR, TARGET_BIN, and CONFIG_FILE must be defined")
endif()

# Parse CONFIG_FILE to detect referenced dataset directories under 'data/<name>/...'
file(READ "${CONFIG_FILE}" _cfg_text)
set(_matches "")
# Match strings like data/ps3d10pt7 or data/compflow6k (first two path components)
string(REGEX MATCHALL "data/[A-Za-z0-9_\\-\\.]+" _raw_matches "${_cfg_text}")
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
else()
  # No dataset references found; run the test directly
endif()

# Run hypredrive (optionally via MPI)
if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  execute_process(
    COMMAND "${MPIEXEC}" "${MPI_NUMPROC_FLAG}" "${MPI_NUMPROCS}" ${MPI_PREFLAGS} "${TARGET_BIN}" "${CONFIG_FILE}" ${MPI_POSTFLAGS}
    WORKING_DIRECTORY "${LAUNCH_DIR}"
    RESULT_VARIABLE _ret
  )
else()
  execute_process(
    COMMAND "${TARGET_BIN}" "${CONFIG_FILE}"
    WORKING_DIRECTORY "${LAUNCH_DIR}"
    RESULT_VARIABLE _ret
  )
endif()
if(NOT _ret EQUAL 0)
  message(FATAL_ERROR "hypredrive failed with exit code ${_ret}")
endif()


