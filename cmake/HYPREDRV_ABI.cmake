# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Independent of the package version: increment when an exported calling
# convention changes or a public symbol is removed. Keep configure.ac aligned.
set(HYPREDRV_ABI_VERSION 1)

function(hypredrv_set_library_abi target)
    set_target_properties(${target} PROPERTIES
        VERSION ${HYPREDRV_ABI_VERSION}.0.0
        SOVERSION ${HYPREDRV_ABI_VERSION}
    )
    # Libtool's Darwin compatibility/current versions start at current + 1.
    math(EXPR _macho_version "${HYPREDRV_ABI_VERSION} + 1")
    set_target_properties(${target} PROPERTIES
        MACHO_COMPATIBILITY_VERSION ${_macho_version}.0.0
        MACHO_CURRENT_VERSION ${_macho_version}.0.0
    )
    if(WIN32)
        get_target_property(_name ${target} OUTPUT_NAME)
        if(NOT _name)
            set(_name ${target})
        endif()
        # Windows has no SONAME; encode the ABI in the DLL name instead.
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_NAME "${_name}-${HYPREDRV_ABI_VERSION}")
    endif()
endfunction()
