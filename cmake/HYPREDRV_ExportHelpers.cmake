# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Shared helpers for deciding whether build-tree exports are safe. This file is
# included by the top-level CMakeLists.txt before interface subdirectories are
# added, and by cmake/HYPREDRV_Install.cmake when generating package exports.
include_guard(GLOBAL)

function(hypredrv_export_dependency_is_imported out_var)
    set(_all_imported TRUE)
    foreach(_target IN LISTS ARGN)
        if(TARGET ${_target})
            get_target_property(_imported ${_target} IMPORTED)
            if(NOT _imported)
                set(_all_imported FALSE)
            endif()
        endif()
    endforeach()
    set(${out_var} ${_all_imported} PARENT_SCOPE)
endfunction()

function(hypredrv_compute_build_tree_export_available out_var)
    set(_available TRUE)

    if(TARGET HYPRE::HYPRE)
        get_target_property(_hypredrv_hypre_imported HYPRE::HYPRE IMPORTED)
        if(NOT _hypredrv_hypre_imported)
            set(_available FALSE)
        endif()
    endif()

    hypredrv_export_dependency_is_imported(
        _hypredrv_optional_dependencies_imported
        caliper::caliper caliper superlu_dist)
    if(NOT _hypredrv_optional_dependencies_imported)
        set(_available FALSE)
    endif()

    set(${out_var} ${_available} PARENT_SCOPE)
endfunction()
