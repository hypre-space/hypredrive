# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

############################################################
# Find and configure HYPRE
############################################################

# Option to specify HYPRE version/branch/tag for FetchContent
set(HYPRE_VERSION "master" CACHE STRING "HYPRE version/branch/tag to fetch (e.g., master, v2.32.0)")

# First, try to find HYPRE via find_package
find_package(HYPRE CONFIG QUIET)

if(NOT HYPRE_FOUND)
    # HYPRE not found via find_package - automatically fetch and build
    message(STATUS "HYPRE not found. Fetching and building HYPRE from source (version: ${HYPRE_VERSION})...")

    include(FetchContent)

    # Check for MPI (required by HYPRE)
    find_package(MPI REQUIRED COMPONENTS C)

    FetchContent_Declare(
        hypre
        GIT_REPOSITORY https://github.com/hypre-space/hypre.git
        GIT_TAG        ${HYPRE_VERSION}
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        SOURCE_SUBDIR  src
    )

    # Enable verbose output for FetchContent to show progress
    set(FETCHCONTENT_QUIET OFF)

    # Generic CMake argument inheritance
    # Forward all relevant cache variables to HYPRE build, including TPLs (MAGMA, CUDA, etc.)
    message(STATUS "Inheriting CMake arguments to HYPRE build...")

    # Get all cache variables
    get_cmake_property(_cache_vars CACHE_VARIABLES)

    # Variables to exclude from inheritance (project-specific)
    set(_exclude_patterns
        "^HYPREDRV_"               # Our project variables
        "^CMAKE_TOOLCHAIN_FILE$"   # Toolchain files shouldn't be inherited
        "^CMAKE_GENERATOR$"        # Generator is already set
        "^CMAKE_SOURCE_DIR$"       # Source directories
        "^CMAKE_BINARY_DIR$"       # Binary directories
        "^CMAKE_PROJECT_NAME$"     # Project name
        "^CMAKE_CURRENT_LIST_DIR$" # Current list directory
        "^CMAKE_CURRENT_SOURCE_DIR$"
        "^CMAKE_CURRENT_BINARY_DIR$"
    )

    # Patterns for variables to include
    set(_include_patterns
        "^HYPRE_ENABLE"        # HYPRE enable options (HYPRE_ENABLE_*, etc.)
        "^.*_ROOT$"            # TPL root variables (MAGMA_ROOT, CUDA_ROOT, etc.)
        "^.*_DIR$"             # TPL directory variables
        "^CMAKE_"              # CMake configuration variables
        "^BUILD_SHARED_LIBS$"  # Library type
    )

    # Function to check if a variable matches any pattern
    function(_matches_pattern var_name patterns result)
        set(${result} FALSE PARENT_SCOPE)
        foreach(pattern IN LISTS patterns)
            if(var_name MATCHES "${pattern}")
                set(${result} TRUE PARENT_SCOPE)
                break()
            endif()
        endforeach()
    endfunction()

    # Forward relevant cache variables
    set(_forwarded_count 0)
    foreach(_var IN LISTS _cache_vars)
        # Skip if matches exclude patterns
        _matches_pattern("${_var}" "${_exclude_patterns}" _should_exclude)
        if(_should_exclude)
            continue()
        endif()

        # Check if matches include patterns
        _matches_pattern("${_var}" "${_include_patterns}" _should_include)
        if(_should_include)
            # Get variable type and value
            get_property(_var_type CACHE "${_var}" PROPERTY TYPE)
            get_property(_var_value CACHE "${_var}" PROPERTY VALUE)

            # Forward the variable
            set("${_var}" "${_var_value}" CACHE "${_var_type}" "" FORCE)
            math(EXPR _forwarded_count "${_forwarded_count} + 1")
        endif()
    endforeach()

    # Ensure MPI compiler is used if CMAKE_C_COMPILER not explicitly set in cache
    get_property(_c_compiler_set CACHE CMAKE_C_COMPILER PROPERTY VALUE SET)
    if(NOT _c_compiler_set AND MPI_C_COMPILER)
        set(CMAKE_C_COMPILER ${MPI_C_COMPILER} CACHE FILEPATH "C compiler" FORCE)
    endif()

    # Configure HYPRE-specific build options (override any user settings)
    set(HYPRE_BUILD_TESTS OFF CACHE BOOL "Build HYPRE tests" FORCE)
    set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "Build HYPRE examples" FORCE)

    # Configure HYPRE to output libraries and headers in the same directories as main project
    # This ensures all libraries are in the same lib/ folder and headers in the same include/ folder
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH "Single output directory for all libraries" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH "Single output directory for all static libraries" FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE PATH "Single output directory for all executables" FORCE)

    if(_forwarded_count GREATER 0)
        message(STATUS "Forwarded ${_forwarded_count} CMake variables to HYPRE build")
    endif()

    # Fetch and configure HYPRE with progress output
    FetchContent_GetProperties(hypre)
    if(NOT hypre_POPULATED)
        message(STATUS "Fetching HYPRE from GitHub (branch/tag: ${HYPRE_VERSION})...")
        message(STATUS "  Repository: https://github.com/hypre-space/hypre.git")

        # Use FetchContent_Populate to have more control and show progress
        # This will show git clone output when FETCHCONTENT_QUIET is OFF
        FetchContent_Populate(hypre)

        message(STATUS "HYPRE source fetched successfully")
        message(STATUS "  Source directory: ${hypre_SOURCE_DIR}")
    else()
        message(STATUS "HYPRE source already available at: ${hypre_SOURCE_DIR}")
    endif()

    # Add HYPRE subdirectory (SOURCE_SUBDIR is handled by pointing to src subdirectory)
    message(STATUS "Configuring HYPRE build...")
    message(STATUS "  Libraries will be built to: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    add_subdirectory(${hypre_SOURCE_DIR}/src ${hypre_BINARY_DIR})

    message(STATUS "HYPRE configured and ready")
endif()

# Get HYPRE properties
if(TARGET HYPRE::HYPRE)
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
else()
    message(FATAL_ERROR "HYPRE::HYPRE target not available")
endif()

############################################################
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
