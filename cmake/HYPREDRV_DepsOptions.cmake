# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Propagate HYPREDRV dependency options
############################################################

# Keep accelerator selection at the HYPREDRV level and propagate it to every
# bundled dependency that supports the selected backend. HYPRE forwards these
# options to its own nested dependencies (for example, Umpire).
set(_hypredrv_enabled_accelerators "")
foreach(_hypredrv_accelerator IN ITEMS CUDA HIP SYCL)
    if(HYPREDRV_ENABLE_${_hypredrv_accelerator})
        list(APPEND _hypredrv_enabled_accelerators
             "${_hypredrv_accelerator}")
        set(HYPRE_ENABLE_${_hypredrv_accelerator} ON CACHE BOOL
            "Enabled by HYPREDRV_ENABLE_${_hypredrv_accelerator}" FORCE)
        message(STATUS
            "HYPREDRV_ENABLE_${_hypredrv_accelerator}=ON: enabling "
            "HYPRE_ENABLE_${_hypredrv_accelerator}")
    endif()
endforeach()
list(LENGTH _hypredrv_enabled_accelerators
     _hypredrv_enabled_accelerator_count)
if(_hypredrv_enabled_accelerator_count GREATER 1)
    list(JOIN _hypredrv_enabled_accelerators ", "
         _hypredrv_enabled_accelerators_text)
    message(FATAL_ERROR
        "HYPREDRV accelerator backends are mutually exclusive; enabled: "
        "${_hypredrv_enabled_accelerators_text}")
endif()

# Enable HYPRE's SuperLU_DIST integration whenever HYPREDRV owns the
# SuperLU_DIST build.
if(HYPREDRV_BUILD_DSUPERLU)
    set(HYPRE_ENABLE_DSUPERLU ON CACHE BOOL
        "Use TPL SuperLU_DIST" FORCE)
endif()

############################################################
