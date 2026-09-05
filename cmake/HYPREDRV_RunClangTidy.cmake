# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Keep the command as a CMake list so paths and arguments retain their boundaries.
execute_process(
    COMMAND ${CLANG_TIDY_COMMAND}
    RESULT_VARIABLE _status
    OUTPUT_FILE "${CLANG_TIDY_OUTPUT}"
    ERROR_FILE "${CLANG_TIDY_OUTPUT}"
)
file(STRINGS "${CLANG_TIDY_OUTPUT}" _diagnostics REGEX "(^|: )(error|warning):")
if(NOT _status STREQUAL "0" OR _diagnostics)
    file(READ "${CLANG_TIDY_OUTPUT}" _output LIMIT 16384)
    message(FATAL_ERROR
        "clang-tidy failed (exit status: ${_status}). See ${CLANG_TIDY_OUTPUT}.\n${_output}")
endif()
message(STATUS "clang-tidy: No warnings found.")
