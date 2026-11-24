# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Include BEFORE project(), then call hypredrv_check_toolchain() AFTER it.
#
# Apple's linker rejects static archives written by GNU ar (e.g. Homebrew's
# binutils ahead of /usr/bin on PATH) with "64-bit mach-o not 8-byte aligned".
# CMAKE_AR only takes effect if set before project(), so default it here.
# Builds with a toolchain file or explicit CMAKE_SYSTEM_NAME are left alone
# (the target is unknown before project() and may not be Apple).
if(CMAKE_HOST_APPLE AND NOT CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_SYSTEM_NAME)
    find_program(_hypredrv_apple_ar ar PATHS /usr/bin NO_DEFAULT_PATH)
    find_program(_hypredrv_apple_ranlib ranlib PATHS /usr/bin NO_DEFAULT_PATH)
    if(NOT CMAKE_AR AND _hypredrv_apple_ar)
        set(CMAKE_AR "${_hypredrv_apple_ar}" CACHE FILEPATH "Archiver")
    endif()
    if(NOT CMAKE_RANLIB AND _hypredrv_apple_ranlib)
        set(CMAKE_RANLIB "${_hypredrv_apple_ranlib}" CACHE FILEPATH "Ranlib")
    endif()
    unset(_hypredrv_apple_ar CACHE)
    unset(_hypredrv_apple_ranlib CACHE)
endif()

# Fail at configure time if GNU ar was chosen explicitly or cached.
function(hypredrv_check_toolchain)
    if(NOT APPLE OR NOT CMAKE_AR)
        return()
    endif()
    execute_process(
        COMMAND "${CMAKE_AR}" --version
        OUTPUT_VARIABLE _hypredrv_ar_version
        ERROR_QUIET
        RESULT_VARIABLE _hypredrv_ar_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_hypredrv_ar_result EQUAL 0 AND _hypredrv_ar_version MATCHES "GNU")
        message(FATAL_ERROR
            "CMAKE_AR is GNU ar (${CMAKE_AR}), which writes static archives that "
            "Apple's linker rejects (\"64-bit mach-o not 8-byte aligned\").\n"
            "Use Apple's archiver instead:\n"
            "  cmake ... -DCMAKE_AR=/usr/bin/ar -DCMAKE_RANLIB=/usr/bin/ranlib\n"
            "If this build directory was configured earlier, remove its CMakeCache.txt "
            "(the old value is cached).")
    endif()
endfunction()
