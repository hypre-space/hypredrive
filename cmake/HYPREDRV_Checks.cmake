# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckFunctionExists)
include(CheckTypeSize)
include(CheckCSourceCompiles)
include(CheckSymbolExists)

# Hypre 2.20 predates HYPRE_RELEASE_NUMBER. Prefer the version already
# discovered for fetched/autotools builds, otherwise inspect the installed
# config header instead of silently treating an unknown version as supported.
set(_hypredrv_hypre_release "${HYPREDRV_HYPRE_RELEASE_NUMBER}")
if(NOT _hypredrv_hypre_release)
    hypredrv_collect_hypre_include_dirs(_hypredrv_version_includes)
    foreach(_include IN LISTS _hypredrv_version_includes)
        if(EXISTS "${_include}/HYPRE_config.h")
            file(READ "${_include}/HYPRE_config.h" _config)
            if(_config MATCHES "#define[ \t]+HYPRE_RELEASE_NUMBER[ \t]+([0-9]+)")
                set(_hypredrv_hypre_release "${CMAKE_MATCH_1}")
            elseif(_config MATCHES "#define[ \t]+HYPRE_RELEASE_VERSION[ \t]+\"([0-9]+)\\.([0-9]+)\\.([0-9]+)")
                math(EXPR _hypredrv_hypre_release
                    "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
            endif()
        endif()
    endforeach()
endif()
if(NOT _hypredrv_hypre_release OR _hypredrv_hypre_release LESS 22000)
    message(FATAL_ERROR "hypredrive requires hypre >= 2.20.0 (detected: ${_hypredrv_hypre_release})")
endif()

# Check for libraries
if(TARGET HYPRE::HYPRE)
    set(HAVE_LIBHYPRE 1)
endif()
check_library_exists(m sqrt "" HAVE_LIBM)
if(HAVE_LIBM)
    set(CMAKE_REQUIRED_LIBRARIES m)
endif()

# Check for headers
check_include_file(sys/stat.h HAVE_SYS_STAT_H)

# Check for functions
check_function_exists(malloc HAVE_MALLOC)

# Check for types and compiler features
check_type_size(_Bool HAVE__BOOL)

# Check for symbols
check_symbol_exists(sqrt "math.h" HAVE_SQRT)
