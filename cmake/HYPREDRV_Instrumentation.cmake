# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Configure compiler/linker instrumentation early so it also applies to
# dependencies built via add_subdirectory()/FetchContent (not just hypredrive
# targets created later in the configure step).

set(_hypredrv_instrumentation_compile_flags "")
set(_hypredrv_instrumentation_link_flags "")
set(_hypredrv_sanitizer_link_flags "")
set(_hypredrv_sanitizer_enabled OFF)
set(_hypredrv_coverage_runtime_library "")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    if(HYPREDRV_ENABLE_COVERAGE)
        list(APPEND _hypredrv_instrumentation_compile_flags -O0 -g --coverage)
        list(APPEND _hypredrv_instrumentation_link_flags --coverage)

        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            # Static consumers of libHYPREDRV.a need the gcov runtime too.
            set(_hypredrv_coverage_runtime_library gcov)
        endif()
    endif()

    if(HYPREDRV_ENABLE_ANALYSIS AND NOT HYPREDRV_ENABLE_COVERAGE)
        # HIP compilation has both host and AMDGPU actions.  UBSan is
        # supported by the host compiler, but not by the AMDGPU target.  Keep
        # ASan on HIP device code (GPU ASan requires an xnack+ architecture),
        # and route UBSan only to the host action.
        if(HYPREDRV_ENABLE_HIP OR HYPRE_ENABLE_HIP OR CMAKE_HIP_COMPILER)
            list(APPEND _hypredrv_instrumentation_compile_flags
                -fno-omit-frame-pointer
                "$<$<COMPILE_LANGUAGE:HIP>:-fsanitize=address>"
                "$<$<COMPILE_LANGUAGE:HIP>:-Xarch_host>"
                "$<$<COMPILE_LANGUAGE:HIP>:-fsanitize=undefined>"
                "$<$<NOT:$<COMPILE_LANGUAGE:HIP>>:-fsanitize=address,undefined>"
            )
            list(APPEND _hypredrv_sanitizer_link_flags
                "$<$<LINK_LANGUAGE:HIP>:-fsanitize=address>"
                "$<$<LINK_LANGUAGE:HIP>:-Xarch_host>"
                "$<$<LINK_LANGUAGE:HIP>:-fsanitize=undefined>"
                "$<$<NOT:$<LINK_LANGUAGE:HIP>>:-fsanitize=address,undefined>"
            )
        else()
            list(APPEND _hypredrv_instrumentation_compile_flags
                -fno-omit-frame-pointer
                -fsanitize=address,undefined
            )
            list(APPEND _hypredrv_sanitizer_link_flags
                -fsanitize=address,undefined
            )
        endif()
        list(APPEND _hypredrv_instrumentation_link_flags
            ${_hypredrv_sanitizer_link_flags}
        )
        set(_hypredrv_sanitizer_enabled ON)
    endif()
endif()

list(REMOVE_DUPLICATES _hypredrv_instrumentation_compile_flags)
list(REMOVE_DUPLICATES _hypredrv_instrumentation_link_flags)
list(REMOVE_DUPLICATES _hypredrv_sanitizer_link_flags)

set(HYPREDRV_INSTRUMENTATION_COMPILE_FLAGS
    "${_hypredrv_instrumentation_compile_flags}"
    CACHE INTERNAL "Compile flags for hypredrive instrumentation" FORCE)
set(HYPREDRV_INSTRUMENTATION_LINK_FLAGS
    "${_hypredrv_instrumentation_link_flags}"
    CACHE INTERNAL "Link flags for hypredrive instrumentation" FORCE)
set(HYPREDRV_SANITIZER_LINK_FLAGS
    "${_hypredrv_sanitizer_link_flags}"
    CACHE INTERNAL "Sanitizer link flags for executables" FORCE)
set(HYPREDRV_SANITIZER_ENABLED
    "${_hypredrv_sanitizer_enabled}"
    CACHE INTERNAL "Whether sanitizers are enabled" FORCE)
set(HYPREDRV_COVERAGE_RUNTIME_LIBRARY
    "${_hypredrv_coverage_runtime_library}"
    CACHE INTERNAL "Runtime library required for coverage builds" FORCE)

set_property(GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED ${_hypredrv_sanitizer_enabled})
set_property(GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS ${_hypredrv_sanitizer_link_flags})

if(_hypredrv_instrumentation_compile_flags)
    add_compile_options(${_hypredrv_instrumentation_compile_flags})
endif()

if(_hypredrv_instrumentation_link_flags)
    add_link_options(${_hypredrv_instrumentation_link_flags})
endif()
