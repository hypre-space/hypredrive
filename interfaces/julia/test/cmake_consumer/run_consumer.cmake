# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BUILD_DIR OR NOT DEFINED HYPREDRV_CMAKE_DIR)
    message(FATAL_ERROR "SOURCE_DIR, BUILD_DIR, and HYPREDRV_CMAKE_DIR are required")
endif()

file(REMOVE_RECURSE "${BUILD_DIR}")

set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${SOURCE_DIR}"
    -B "${BUILD_DIR}"
    "-DHYPREDRV_DIR=${HYPREDRV_CMAKE_DIR}"
)

if(DEFINED CONSUMER_CMAKE_PREFIX_PATH AND NOT "${CONSUMER_CMAKE_PREFIX_PATH}" STREQUAL "")
    list(APPEND _configure_command "-DCMAKE_PREFIX_PATH=${CONSUMER_CMAKE_PREFIX_PATH}")
endif()

execute_process(
    COMMAND ${_configure_command}
    RESULT_VARIABLE _configure_result
)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR "Julia CMake consumer configure failed")
endif()

set(_build_command "${CMAKE_COMMAND}" --build "${BUILD_DIR}")
if(DEFINED CONSUMER_CONFIG AND NOT "${CONSUMER_CONFIG}" STREQUAL "")
    list(APPEND _build_command --config "${CONSUMER_CONFIG}")
endif()

execute_process(
    COMMAND ${_build_command}
    RESULT_VARIABLE _build_result
)
if(NOT _build_result EQUAL 0)
    message(FATAL_ERROR "Julia CMake consumer build failed")
endif()
