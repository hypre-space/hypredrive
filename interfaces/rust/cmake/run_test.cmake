# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: MIT

# Use the project's MPI/GPU launcher selection and apply sanitizer runtimes
# after the launcher, only to the Rust test process on each rank.
include("${TEST_LAUNCHER}")
hypredrv_prepare_test_launcher(_launcher_command _launcher_postflags)
# `cmake -E env` accepts a command without a separator on every supported CMake.
# Skip it entirely when there is no environment to apply.
if(TEST_RUNTIME_ENV)
    set(_env_command "${CMAKE_COMMAND}" -E env ${TEST_RUNTIME_ENV})
else()
    set(_env_command "")
endif()
execute_process(
    COMMAND ${_launcher_command} ${_env_command} "${TARGET_BIN}" ${_launcher_postflags}
    RESULT_VARIABLE _test_status)
if(NOT _test_status EQUAL 0)
    message(FATAL_ERROR "Rust smoke test failed with status ${_test_status}")
endif()
