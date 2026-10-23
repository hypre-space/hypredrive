# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Sync Caliper options between HYPREDRV and HYPRE
############################################################

# Bidirectional sync: HYPREDRV_ENABLE_CALIPER <-> HYPRE_ENABLE_CALIPER
# Priority: HYPREDRV_ENABLE_CALIPER takes precedence if both are defined
# Note: We need to handle both cache variables and command-line -D flags
# First, ensure both variables exist as cache variables with defaults
if(NOT DEFINED HYPREDRV_ENABLE_CALIPER)
    set(HYPREDRV_ENABLE_CALIPER OFF CACHE BOOL "Enable Caliper instrumentation support")
endif()
if(NOT DEFINED HYPRE_ENABLE_CALIPER)
    set(HYPRE_ENABLE_CALIPER OFF CACHE BOOL "Use TPL Caliper")
endif()

# Now do the sync - check actual values
if(HYPREDRV_ENABLE_CALIPER)
    # HYPREDRV_ENABLE_CALIPER is ON - sync to HYPRE_ENABLE_CALIPER
    set(HYPRE_ENABLE_CALIPER ON CACHE BOOL "Use TPL Caliper" FORCE)
elseif(HYPRE_ENABLE_CALIPER)
    # HYPRE_ENABLE_CALIPER is ON but HYPREDRV_ENABLE_CALIPER is OFF - sync to HYPREDRV_ENABLE_CALIPER
    # This handles the case where user passes -DHYPRE_ENABLE_CALIPER=ON when building HYPRE automatically
    set(HYPREDRV_ENABLE_CALIPER ON CACHE BOOL "Enable Caliper instrumentation support" FORCE)
    message(STATUS "HYPRE_ENABLE_CALIPER=ON detected - enabling HYPREDRV_ENABLE_CALIPER")
endif()
# Note: For find_package case, additional detection from HYPRE target properties
# is done after HYPRE is found (see below)

############################################################
# Find and configure Caliper (must be before HYPRE)
############################################################

# Option to specify Caliper version/branch/tag for FetchContent
set(CALIPER_VERSION "master" CACHE STRING "Caliper version/branch/tag to fetch (e.g., master, v2.12.0)")

if(HYPREDRV_ENABLE_CALIPER)
    # First, try to find Caliper via find_package
    find_package(caliper CONFIG QUIET)

    if(NOT caliper_FOUND)
        # Caliper not found via find_package - automatically fetch and build
        # Check if Ninja generator is being used - Caliper auto-build requires Makefiles
        if(CMAKE_GENERATOR STREQUAL "Ninja" OR CMAKE_GENERATOR MATCHES "^Ninja")
            message(FATAL_ERROR
                "Caliper auto-fetch/build is not supported with Ninja generator.\n"
                "Please either:\n"
                "  1. Use the default Makefile generator (remove -G Ninja), or\n"
                "  2. Build Caliper separately and point to it via find_package or CALIPER_DIR"
            )
        endif()

        message(STATUS "Caliper not found. Fetching and building Caliper from source (version: ${CALIPER_VERSION})...")

        include(FetchContent)

        FetchContent_Declare(
            caliper
            GIT_REPOSITORY https://github.com/LLNL/Caliper.git
            GIT_TAG        ${CALIPER_VERSION}
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            GIT_SUBMODULES ""  # Skip submodules (scripts/radiuss-spack-configs, scripts/uberenv)
            #PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/caliper.patch
        )

        # Enable verbose output for FetchContent to show progress
        set(FETCHCONTENT_QUIET OFF)

        # Disable features we don't need to speed up build
        # Note: Caliper uses WITH_* prefix (not CALIPER_WITH_*)
        set(WITH_MPI ON CACHE BOOL "Build Caliper with MPI support" FORCE)
        set(WITH_KOKKOS OFF CACHE BOOL "Build Caliper with Kokkos support" FORCE)
        set(WITH_GOTCHA OFF CACHE BOOL "Build Caliper with GOTCHA support" FORCE)
        set(WITH_PAPI OFF CACHE BOOL "Build Caliper with PAPI support" FORCE)
        set(WITH_SAMPLER OFF CACHE BOOL "Build Caliper with sampler support" FORCE)
        set(WITH_NVTX OFF CACHE BOOL "Build Caliper with NVTX support" FORCE)
        set(WITH_CUPTI OFF CACHE BOOL "Build Caliper with CUPTI support" FORCE)
        set(WITH_LIBUNWIND OFF CACHE BOOL "Build Caliper with libunwind support" FORCE)
        set(WITH_TOOLS OFF CACHE BOOL "Build Caliper tools" FORCE)

        # Configure Caliper to output libraries and headers in the same directories as main project
        _hypredrv_set_common_output_directories()

        # Explicitly propagate BUILD_SHARED_LIBS to Caliper build
        # FetchContent subdirectories need cache variables to inherit values
        # Get the current value from cache (it's defined as an option in main CMakeLists.txt)
        get_property(BUILD_SHARED_LIBS_VALUE CACHE BUILD_SHARED_LIBS PROPERTY VALUE)
        if(BUILD_SHARED_LIBS_VALUE)
            set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
        else()
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
        endif()
        message(STATUS "  Propagating BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} to Caliper build")

        # Fetch and configure Caliper using FetchContent_MakeAvailable
        message(STATUS "Fetching Caliper from GitHub (branch/tag: ${CALIPER_VERSION})...")
        message(STATUS "  Repository: https://github.com/LLNL/Caliper.git")

        FetchContent_MakeAvailable(caliper)

        # Ensure Caliper target exists and is built before HYPRE
        if(TARGET caliper)
            # Make sure Caliper builds early
            set_target_properties(caliper PROPERTIES EXCLUDE_FROM_ALL FALSE)
        endif()

        # Construct the library path based on BUILD_SHARED_LIBS
        # This must match what Caliper will actually build
        if(BUILD_SHARED_LIBS)
            set(CALIPER_LIBRARY_FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}caliper${CMAKE_SHARED_LIBRARY_SUFFIX}")
        else()
            set(CALIPER_LIBRARY_FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}caliper${CMAKE_STATIC_LIBRARY_SUFFIX}")
        endif()

        set(TPL_CALIPER_LIBRARIES ${CALIPER_LIBRARY_FILE} CACHE FILEPATH "Caliper library for HYPRE" FORCE)

        # Set TPL_CALIPER_INCLUDE_DIRS to a dummy value to satisfy HYPRE's setup_tpl check
        # We'll override it manually after HYPRE is configured to use BUILD_INTERFACE
        # This prevents CMake errors about _deps paths in INTERFACE_INCLUDE_DIRECTORIES
        set(TPL_CALIPER_INCLUDE_DIRS "${caliper_SOURCE_DIR}/include" CACHE PATH "Caliper include directories for HYPRE" FORCE)

        # Mark that Caliper is being built as part of this project
        set(HYPRE_BUILD_CALIPER ON CACHE INTERNAL "Caliper is being built as part of this project" FORCE)
        set(CALIPER_INCLUDE_DIR_FOR_HYPRE "${caliper_SOURCE_DIR}/include" CACHE INTERNAL "Caliper include directory for manual addition to HYPRE" FORCE)

        message(STATUS "  Setting TPL_CALIPER_LIBRARIES: ${TPL_CALIPER_LIBRARIES}")
        message(STATUS "  Note: TPL_CALIPER_INCLUDE_DIRS not set to avoid CMake export issues")
        message(STATUS "  Caliper headers will be added to HYPRE manually with BUILD_INTERFACE")

        # Mark that we're building Caliper via FetchContent
        set(CALIPER_FOUND TRUE)
        message(STATUS "Caliper configured and ready (built via FetchContent)")
    endif()

    # Check if Caliper target is available (either from find_package or FetchContent)
    if(NOT CALIPER_FOUND)
        if(TARGET caliper::caliper)
            set(CALIPER_FOUND TRUE)
            message(STATUS "Found Caliper (via find_package):")
            get_target_property(CALIPER_INCLUDE_DIRS caliper::caliper INTERFACE_INCLUDE_DIRECTORIES)
            if(CALIPER_INCLUDE_DIRS)
                # Extract the first include directory that's not a generator expression
                list(GET CALIPER_INCLUDE_DIRS 0 FIRST_INCLUDE_DIR)
                if(NOT TPL_CALIPER_INCLUDE_DIRS)
                    set(TPL_CALIPER_INCLUDE_DIRS ${FIRST_INCLUDE_DIR} CACHE PATH "Caliper include directories for HYPRE" FORCE)
                endif()
            endif()
            message(STATUS "  include directories: ${TPL_CALIPER_INCLUDE_DIRS}")
        elseif(TARGET caliper)
            # Caliper was built via FetchContent - target is just 'caliper', not 'caliper::caliper'
            set(CALIPER_FOUND TRUE)
            message(STATUS "Found Caliper (built via FetchContent):")
            message(STATUS "  include directories: ${TPL_CALIPER_INCLUDE_DIRS}")
        else()
            # Try alternative find methods as fallback
            find_path(CALIPER_INCLUDE_DIR caliper/cali.h
                PATHS /usr/include /usr/local/include
                      ENV CALIPER_DIR
                      ENV CALIPER_ROOT
            )
            find_library(CALIPER_LIBRARY caliper
                PATHS /usr/lib /usr/local/lib
                      ENV CALIPER_DIR
                      ENV CALIPER_ROOT
            )
            if(CALIPER_INCLUDE_DIR AND CALIPER_LIBRARY)
                set(CALIPER_FOUND TRUE)
                message(STATUS "Found Caliper (via find_path/find_library):")
                message(STATUS "  include directories: ${CALIPER_INCLUDE_DIR}")
                message(STATUS "  libraries: ${CALIPER_LIBRARY}")
            else()
                set(CALIPER_FOUND FALSE)
                message(WARNING "HYPREDRV_ENABLE_CALIPER is ON but Caliper was not found. Disabling Caliper support.")
            endif()
        endif()
    endif()

    if(CALIPER_FOUND)
        _hypredrv_set_using_caliper(TRUE)
    else()
        _hypredrv_set_using_caliper(FALSE)
    endif()
else()
    message(STATUS "Caliper support disabled (HYPREDRV_ENABLE_CALIPER=OFF)")
    set(CALIPER_FOUND FALSE)
    _hypredrv_set_using_caliper(FALSE)
endif()

############################################################
