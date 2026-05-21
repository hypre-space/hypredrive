# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Formatting target. Prefer clang-format-18 project-wide when available so C,
# C++, and public headers are formatted with the same clang-format version in CI
# and local validation. Fall back to clang-format for older developer machines.
find_program(CLANG_FORMAT
    NAMES clang-format-18 clang-format)
set(_hypredrv_format_targets)

if(CLANG_FORMAT)
    add_custom_target(format-c
        COMMAND ${CMAKE_COMMAND} -E env find include src examples/src -type f -name "*.c" -exec ${CLANG_FORMAT} -i {} +
        COMMAND ${CMAKE_COMMAND} -E env find include src examples/src -type f -name "*.h" -exec ${CLANG_FORMAT} -i {} +
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on include/, src/, and examples/src/..."
        VERBATIM
    )
    list(APPEND _hypredrv_format_targets format-c)
else()
    message(STATUS "clang-format not found, C formatting target will not be available")
endif()

if(HYPREDRV_ENABLE_CPP)
    if(CLANG_FORMAT)
        add_custom_target(format-cpp
            COMMAND ${CMAKE_COMMAND} -E env find interfaces/cpp -type f
                    \( -name "*.cpp" -o -name "*.hpp" \)
                    -exec ${CLANG_FORMAT} -i {} +
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-format on interfaces/cpp/..."
            VERBATIM
        )
        list(APPEND _hypredrv_format_targets format-cpp)
    endif()
endif()

if(HYPREDRV_ENABLE_FORTRAN)
    find_program(FPRETTIFY "fprettify"
        HINTS
            "$ENV{VIRTUAL_ENV}/bin"
            "${CMAKE_SOURCE_DIR}/.venv/bin"
            "$ENV{HOME}/.local/bin")
    if(FPRETTIFY)
        add_custom_target(format-fortran
            COMMAND ${CMAKE_COMMAND} -E env find interfaces/fortran -type f -name "*.f90" -exec ${FPRETTIFY} -i 3 {} +
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running fprettify on interfaces/fortran/..."
            VERBATIM
        )
        list(APPEND _hypredrv_format_targets format-fortran)
    else()
        message(STATUS "fprettify not found, Fortran formatting target will not be available")
    endif()
endif()

if(_hypredrv_format_targets)
    add_custom_target(format DEPENDS ${_hypredrv_format_targets})
endif()
