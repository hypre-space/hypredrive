# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Download a dataset archive, verify its SHA-256 checksum, and extract it.
# Usage: cmake -DURL=<url> -DTARBALL=<file> -DDEST=<dir> -DSHA256=<hash> -P <this file>

foreach(_var URL TARBALL DEST SHA256)
    if(NOT ${_var})
        message(FATAL_ERROR "HYPREDRV_FetchDataset.cmake: ${_var} is required")
    endif()
endforeach()

# Retry transient gateway/server errors (for example Zenodo 429/5xx) with
# exponential backoff: 2, 5, 15, 30 and 60 seconds between six attempts.
set(_backoff_seconds 2 5 15 30 60)
foreach(_attempt RANGE 1 6)
    file(DOWNLOAD "${URL}" "${TARBALL}"
        TLS_VERIFY ON
        INACTIVITY_TIMEOUT 60
        STATUS _status)
    list(GET _status 0 _code)
    if(_code EQUAL 0)
        break()
    endif()
    message(STATUS "Download attempt ${_attempt} of ${URL} failed: ${_status}")
    if(_attempt LESS 6)
        math(EXPR _idx "${_attempt} - 1")
        list(GET _backoff_seconds ${_idx} _delay)
        execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${_delay})
    endif()
endforeach()
if(NOT _code EQUAL 0)
    file(REMOVE "${TARBALL}")
    message(FATAL_ERROR "Failed to download ${URL}: ${_status}")
endif()

file(SHA256 "${TARBALL}" _actual)
if(NOT "${_actual}" STREQUAL "${SHA256}")
    file(REMOVE "${TARBALL}")
    message(FATAL_ERROR "Checksum mismatch for ${TARBALL}: ${_actual} != ${SHA256}")
endif()

file(ARCHIVE_EXTRACT INPUT "${TARBALL}" DESTINATION "${DEST}")
