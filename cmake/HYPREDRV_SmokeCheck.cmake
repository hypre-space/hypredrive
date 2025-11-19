# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Smoke test helper invoked by the 'check' custom target
# Inputs:
#   -DLAUNCH_DIR=...   top-level source directory
#   -DTARGET_BIN=...   path to hypredrive executable

if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN)
  message(FATAL_ERROR "HYPREDRV_Check.cmake: LAUNCH_DIR and TARGET_BIN must be defined")
endif()

set(_dataset_dir "${LAUNCH_DIR}/data/ps3d10pt7")
if(EXISTS "${_dataset_dir}" AND IS_DIRECTORY "${_dataset_dir}")
  file(GLOB _files "${_dataset_dir}/*")
  list(LENGTH _files _nfiles)
  if(_nfiles GREATER 0)
    execute_process(
      COMMAND "${TARGET_BIN}" "${LAUNCH_DIR}/examples/ex1.yml"
      WORKING_DIRECTORY "${LAUNCH_DIR}"
      RESULT_VARIABLE _ret
    )
    if(NOT _ret EQUAL 0)
      message(FATAL_ERROR "Smoke test failed with exit code ${_ret}")
    endif()
  else()
    message(STATUS "[check] Skipping smoke test: dataset 'data/ps3d10pt7' is empty.")
    message(STATUS "[check] To fetch datasets: cmake --build . --target data")
  endif()
else()
  message(STATUS "[check] Skipping smoke test: dataset 'data/ps3d10pt7' not found.")
  message(STATUS "[check] To fetch datasets: cmake --build . --target data")
endif()
