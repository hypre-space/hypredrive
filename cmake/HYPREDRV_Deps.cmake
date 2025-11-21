# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

############################################################
# Find and configure HYPRE
############################################################
find_package(HYPRE REQUIRED CONFIG)
if(NOT HYPRE_FOUND)
    message(FATAL_ERROR "HYPRE library not found. Please specify -DHYPRE_ROOT=<path> to the root of the HYPRE installation.")
endif()

get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)

# Try to get library location - handle both Release and Debug configurations
get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION)
if(NOT HYPRE_LIBRARY_FILE)
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
endif()
if(NOT HYPRE_LIBRARY_FILE)
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_DEBUG)
endif()
if(NOT HYPRE_LIBRARY_FILE)
    set(HYPRE_LIBRARY_FILE "not found (using target)")
endif()

message(STATUS "Found HYPRE:")
message(STATUS "  include directories: ${HYPRE_INCLUDE_DIRS}")
message(STATUS "  libraries: ${HYPRE_LIBRARY_FILE}")

############################################################
# Find and configure hwloc
############################################################
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
    message(STATUS "hwloc not found, using basic system information.")
endif()
