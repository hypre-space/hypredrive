# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Formatting target
find_program(CLANG_FORMAT "clang-format")
if(NOT CLANG_FORMAT)
    message(STATUS "clang-format not found, formatting targets will not be available")
else()
    add_custom_target(format
        COMMAND find include src examples/src -type f -name "*.c" -exec ${CLANG_FORMAT} -i {} +
        COMMAND find include src examples/src -type f -name "*.h" -exec ${CLANG_FORMAT} -i {} +
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on include/, src/, and examples/src/..."
    )
endif()
