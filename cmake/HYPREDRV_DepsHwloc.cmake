# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Find and configure hwloc
############################################################
if(HYPREDRV_ENABLE_HWLOC)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND AND PKG_CONFIG_HWLOC_FOUND) # Check if pkg-config found hwloc
        set(HWLOC_INCLUDE_DIR ${PKG_CONFIG_HWLOC_INCLUDE_DIRS})
        set(HWLOC_LIBRARY ${PKG_CONFIG_HWLOC_LIBRARIES})
        set(HWLOC_FOUND TRUE)
    else()
        find_path(HWLOC_INCLUDE_DIR hwloc.h
            PATHS /usr/include /usr/local/include
        )
        find_library(HWLOC_LIBRARY hwloc
            PATHS /usr/lib /usr/local/lib
        )
        if(HWLOC_INCLUDE_DIR AND HWLOC_LIBRARY)
            set(HWLOC_FOUND TRUE)
        else()
            set(HWLOC_FOUND FALSE)
        endif()
    endif()

    if(HWLOC_FOUND)
        set(HAVE_HWLOC 1 CACHE INTERNAL "Have hwloc")
        message(STATUS "Found hwloc:")
        message(STATUS "  include directories: ${HWLOC_INCLUDE_DIR}")
        message(STATUS "  libraries: ${HWLOC_LIBRARY}")
    else()
        message(WARNING "HYPREDRV_ENABLE_HWLOC is ON but hwloc was not found. Disabling hwloc support.")
        message(STATUS "hwloc not found, using basic system information.")
        set(HWLOC_FOUND FALSE)
    endif()
else()
    message(STATUS "hwloc support disabled (HYPREDRV_ENABLE_HWLOC=OFF)")
    set(HWLOC_FOUND FALSE)
endif()

############################################################
