# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Smoke test helper invoked by the 'check' custom target.
# Inputs:
#   -DLAUNCH_DIR=...                        top-level source directory
#   -DTARGET_BIN=...                        path to hypredrive executable
#   -DTEST_RUNTIME_ENV_ASSIGNMENT=KEY=VAL   optional (e.g. LD_LIBRARY_PATH=...)
#   -DHYPREDRV_SMOKE_TEST_USES_SYCL=ON      run with a minimal env (SYCL workaround)

if(NOT DEFINED LAUNCH_DIR OR NOT DEFINED TARGET_BIN)
  message(FATAL_ERROR "HYPREDRV_SmokeCheck.cmake: LAUNCH_DIR and TARGET_BIN must be defined")
endif()

set(_dataset_dir "${LAUNCH_DIR}/data/ps3d10pt7")
file(GLOB _dataset_files "${_dataset_dir}/*")
if(NOT IS_DIRECTORY "${_dataset_dir}")
  message(STATUS "[check] Skipping smoke test: dataset 'data/ps3d10pt7' not found.")
  message(STATUS "[check] To fetch datasets: cmake --build . --target data")
  return()
endif()
if(NOT _dataset_files)
  message(STATUS "[check] Skipping smoke test: dataset 'data/ps3d10pt7' is empty.")
  message(STATUS "[check] To fetch datasets: cmake --build . --target data")
  return()
endif()

# Apply the hypredrive-computed runtime env (currently LD_LIBRARY_PATH or
# DYLD_LIBRARY_PATH, set during configure in HYPREDRV_Deps.cmake).
if(DEFINED TEST_RUNTIME_ENV_ASSIGNMENT AND
   TEST_RUNTIME_ENV_ASSIGNMENT MATCHES "^([^=]+)=(.*)$")
  set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
elseif(DEFINED TEST_RUNTIME_ENV_ASSIGNMENT AND NOT TEST_RUNTIME_ENV_ASSIGNMENT STREQUAL "")
  message(WARNING "[check] Ignoring malformed TEST_RUNTIME_ENV_ASSIGNMENT: ${TEST_RUNTIME_ENV_ASSIGNMENT}")
endif()

# Default a SYCL device selector if the caller didn't pick one.
if(HYPREDRV_SMOKE_TEST_USES_SYCL AND
   "$ENV{ONEAPI_DEVICE_SELECTOR}" STREQUAL "" AND
   "$ENV{SYCL_DEVICE_FILTER}" STREQUAL "")
  set(ENV{ONEAPI_DEVICE_SELECTOR} "opencl:gpu")
  message(STATUS "[check] ONEAPI_DEVICE_SELECTOR not set; using 'opencl:gpu' for SYCL smoke test.")
endif()

if(HYPREDRV_SMOKE_TEST_USES_SYCL)
  # oneAPI's setvars.sh exports a broad toolchain environment. The SYCL smoke
  # only needs a small runtime subset; inheriting the full compiler shell can
  # corrupt shutdown after an otherwise successful GPU solve.
  # TODO: root-cause the shutdown corruption and drop this env scrub.
  find_program(_env_program env REQUIRED)
  set(_smoke_env "")
  foreach(_name IN ITEMS
          PATH HOME TMPDIR TMP TEMP XDG_CACHE_HOME LD_LIBRARY_PATH
          ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER
          CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
          ZE_AFFINITY_MASK OMP_NUM_THREADS)
    if(NOT "$ENV{${_name}}" STREQUAL "")
      list(APPEND _smoke_env "${_name}=$ENV{${_name}}")
    endif()
  endforeach()
  set(_smoke_command "${_env_program}" -i ${_smoke_env}
      "${TARGET_BIN}" "${LAUNCH_DIR}/examples/ex1.yml")
else()
  set(_smoke_command "${TARGET_BIN}" "${LAUNCH_DIR}/examples/ex1.yml")
endif()

execute_process(
  COMMAND ${_smoke_command}
  WORKING_DIRECTORY "${LAUNCH_DIR}"
  RESULT_VARIABLE _ret
)
if(NOT _ret EQUAL 0)
  message(FATAL_ERROR "Smoke test failed with exit code ${_ret}")
endif()
