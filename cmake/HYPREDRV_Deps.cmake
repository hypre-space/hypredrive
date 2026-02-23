# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

############################################################
# Internal helpers (single-file structure, lower duplication)
############################################################

function(_hypredrv_set_common_output_directories)
    # Keep all generated artifacts in predictable top-level output paths.
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH
        "Single output directory for all libraries" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH
        "Single output directory for all static libraries" FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE PATH
        "Single output directory for all executables" FORCE)
endfunction()

function(_hypredrv_link_mpi_interface target_name)
    if(TARGET MPI::MPI_C)
        target_link_libraries(${target_name} INTERFACE MPI::MPI_C)
    else()
        if(MPI_C_LIBRARIES)
            target_link_libraries(${target_name} INTERFACE ${MPI_C_LIBRARIES})
        endif()
        if(MPI_C_INCLUDE_DIRS)
            target_include_directories(${target_name} INTERFACE ${MPI_C_INCLUDE_DIRS})
        endif()
    endif()
endfunction()

function(_hypredrv_set_using_caliper enabled)
    if(enabled)
        set(HYPREDRV_USING_CALIPER 1 PARENT_SCOPE)
        set(HYPREDRV_USING_CALIPER 1 CACHE INTERNAL "Using Caliper")
    else()
        set(HYPREDRV_USING_CALIPER 0 PARENT_SCOPE)
        set(HYPREDRV_USING_CALIPER 0 CACHE INTERNAL "Not using Caliper")
    endif()
endfunction()

function(_hypredrv_matches_pattern var_name pattern_list_var result_var)
    set(_matched FALSE)
    foreach(pattern IN LISTS ${pattern_list_var})
        if(var_name MATCHES "${pattern}")
            set(_matched TRUE)
            break()
        endif()
    endforeach()
    set(${result_var} ${_matched} PARENT_SCOPE)
endfunction()

function(_hypredrv_forward_hypre_cache_vars forwarded_count_var)
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
        "^TPL_"                # TPL variables (TPL_CALIPER_*, etc.)
        "^.*_ROOT$"            # TPL root variables (MAGMA_ROOT, CUDA_ROOT, etc.)
        "^.*_DIR$"             # TPL directory variables
        "^CMAKE_"              # CMake configuration variables
        "^BUILD_SHARED_LIBS$"  # Library type
    )

    get_cmake_property(_cache_vars CACHE_VARIABLES)
    set(_forwarded_count 0)
    foreach(_var IN LISTS _cache_vars)
        _hypredrv_matches_pattern("${_var}" _exclude_patterns _should_exclude)
        if(_should_exclude)
            continue()
        endif()

        _hypredrv_matches_pattern("${_var}" _include_patterns _should_include)
        if(NOT _should_include)
            continue()
        endif()

        get_property(_var_type CACHE "${_var}" PROPERTY TYPE)
        get_property(_var_value CACHE "${_var}" PROPERTY VALUE)
        set("${_var}" "${_var_value}" CACHE "${_var_type}" "" FORCE)
        math(EXPR _forwarded_count "${_forwarded_count} + 1")
    endforeach()

    set(${forwarded_count_var} ${_forwarded_count} PARENT_SCOPE)
endfunction()

############################################################
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
# Find and configure HYPRE
############################################################

# Option to specify HYPRE version/branch/tag for FetchContent
set(HYPRE_VERSION "master" CACHE STRING "HYPRE version/branch/tag to fetch (e.g., master, v2.32.0)")

# Allow users to point to an autotools build by providing include/lib paths.
# This bypasses find_package(CONFIG) which only works with CMake installs.
if(NOT TARGET HYPRE::HYPRE AND DEFINED HYPRE_INCLUDE_DIRS AND
   (DEFINED HYPRE_LIBRARIES OR DEFINED HYPRE_LIBRARY))
    if(DEFINED HYPRE_LIBRARIES)
        set(_hypre_libs "${HYPRE_LIBRARIES}")
    else()
        set(_hypre_libs "${HYPRE_LIBRARY}")
    endif()

    message(STATUS "Using user-provided HYPRE include/lib paths (autotools or custom install)")
    add_library(HYPRE::HYPRE INTERFACE IMPORTED)
    set_target_properties(HYPRE::HYPRE PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${_hypre_libs}"
    )
    set(HYPREDRV_HYPRE_USER_PROVIDED TRUE)
    set(HYPRE_FOUND TRUE)
endif()

# First, try to find HYPRE via find_package
if(NOT HYPRE_FOUND)
    find_package(HYPRE CONFIG QUIET)
endif()

if(NOT HYPRE_FOUND)
    # HYPRE not found via find_package - automatically fetch and build
    message(STATUS "HYPRE not found. Fetching and building HYPRE from source (version: ${HYPRE_VERSION})...")

    include(FetchContent)

    # Check for MPI (required by HYPRE)
    find_package(MPI REQUIRED COMPONENTS C)

    # Decide whether to use autotools (hypre < 3.0.0) or CMake build
    set(_hypre_use_autotools FALSE)
    if(HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(_hypre_major "${CMAKE_MATCH_1}")
        if(_hypre_major LESS 3)
            set(_hypre_use_autotools TRUE)
        endif()
    endif()

    set(_hypre_git_shallow TRUE)
    if(NOT HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        # HYPRE's CMake computes HYPRE_DEVELOP_NUMBER from git tag distance. A shallow
        # clone on moving refs (master/main/develop or arbitrary branches/commits) may
        # not have enough history/tags for `git describe`, which drops DEVELOP_NUMBER
        # from generated HYPRE_config.h and breaks hypredrive feature/version probes.
        set(_hypre_git_shallow FALSE)
    endif()

    FetchContent_Declare(
        hypre
        GIT_REPOSITORY https://github.com/hypre-space/hypre.git
        GIT_TAG        ${HYPRE_VERSION}
        GIT_SHALLOW    ${_hypre_git_shallow}
        GIT_PROGRESS   TRUE
        # SOURCE_SUBDIR removed - we'll add it manually after patching
    )

    # Enable verbose output for FetchContent to show progress
    set(FETCHCONTENT_QUIET OFF)

    if(_hypre_use_autotools)
        message(STATUS "HYPRE version < 3.0.0 detected - building with autotools.")

        if(HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            math(EXPR _hypre_release_number
                 "${CMAKE_MATCH_1}*10000 + ${CMAKE_MATCH_2}*100 + ${CMAKE_MATCH_3}")
            if(NOT HYPREDRV_HYPRE_RELEASE_NUMBER)
                set(HYPREDRV_HYPRE_RELEASE_NUMBER "${_hypre_release_number}")
                add_compile_definitions(HYPREDRV_HYPRE_RELEASE_NUMBER=${HYPREDRV_HYPRE_RELEASE_NUMBER})
            endif()
            if(NOT HYPREDRV_HYPRE_DEVELOP_NUMBER)
                set(HYPREDRV_HYPRE_DEVELOP_NUMBER 0)
                add_compile_definitions(HYPREDRV_HYPRE_DEVELOP_NUMBER=0)
            endif()
            if(NOT HYPREDRV_HYPRE_RELEASE_VERSION)
                set(HYPREDRV_HYPRE_RELEASE_VERSION
                    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
                add_compile_definitions(
                    HYPREDRV_HYPRE_RELEASE_VERSION=\"${HYPREDRV_HYPRE_RELEASE_VERSION}\")
            endif()
        endif()

        FetchContent_GetProperties(hypre)
        if(NOT hypre_POPULATED)
            FetchContent_Populate(hypre)
        endif()

        include(ExternalProject)

        string(REPLACE "/" "_" _hypre_version_tag "${HYPRE_VERSION}")
        set(_hypre_autotools_prefix "${CMAKE_BINARY_DIR}/hypre-autotools-${_hypre_version_tag}")
        set(_hypre_autotools_src "${hypre_SOURCE_DIR}/src")
        # Match typical CMake build-type flags for autotools builds.
        set(_hypre_autotools_cflags "${CMAKE_C_FLAGS}")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O0 -g")
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O2 -g")
        else()
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O3 -DNDEBUG")
        endif()

        set(_hypre_autotools_configure_extra "")
        if(HYPRE_ENABLE_MIXEDINT)
            if(DEFINED _hypre_release_number AND _hypre_release_number GREATER_EQUAL 22000)
                list(APPEND _hypre_autotools_configure_extra --enable-mixedint)
            else()
                list(APPEND _hypre_autotools_configure_extra --disable-mixedint)
                message(WARNING "HYPRE_ENABLE_MIXEDINT=ON but ${HYPRE_VERSION} does not support mixedint; forcing --disable-mixedint.")
            endif()
        endif()
        if(HYPREDRV_ENABLE_CALIPER)
            if(TPL_CALIPER_INCLUDE_DIRS AND TPL_CALIPER_LIBRARIES)
                list(APPEND _hypre_autotools_configure_extra
                    --with-caliper
                    --with-caliper-include=${TPL_CALIPER_INCLUDE_DIRS}
                    --with-caliper-lib=${TPL_CALIPER_LIBRARIES}
                )
            elseif(CALIPER_INCLUDE_DIR AND CALIPER_LIBRARY)
                list(APPEND _hypre_autotools_configure_extra
                    --with-caliper
                    --with-caliper-include=${CALIPER_INCLUDE_DIR}
                    --with-caliper-lib=${CALIPER_LIBRARY}
                )
            else()
                message(WARNING "HYPREDRV_ENABLE_CALIPER=ON but Caliper include/lib not found for hypre autotools build.")
            endif()
        endif()

        if(HYPRE_ENABLE_CUDA OR CMAKE_CUDA_ARCHITECTURES OR CUDA_ROOT OR CUDA_TOOLKIT_ROOT_DIR OR CUDAToolkit_ROOT)
            list(APPEND _hypre_autotools_configure_extra --with-cuda)
            if(CUDA_ROOT)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDA_ROOT})
            elseif(CUDA_TOOLKIT_ROOT_DIR)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDA_TOOLKIT_ROOT_DIR})
            elseif(CUDAToolkit_ROOT)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDAToolkit_ROOT})
            endif()
            if(CMAKE_CUDA_ARCHITECTURES)
                string(REPLACE ";" "," _hypre_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
                list(APPEND _hypre_autotools_configure_extra --with-gpu-arch=${_hypre_cuda_arch})
            endif()
        endif()

        if(HYPRE_ENABLE_HIP OR CMAKE_HIP_ARCHITECTURES OR HIP_ROOT OR ROCM_ROOT OR ROCM_PATH)
            list(APPEND _hypre_autotools_configure_extra --with-hip)
            list(APPEND _hypre_autotools_configure_extra --with-MPI-include=${MPI_C_INCLUDE_DIRS})
            if(HIP_ROOT)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${HIP_ROOT})
            elseif(ROCM_ROOT)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${ROCM_ROOT})
            elseif(ROCM_PATH)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${ROCM_PATH})
            endif()
            if(CMAKE_HIP_ARCHITECTURES)
                string(REPLACE ";" "," _hypre_hip_arch "${CMAKE_HIP_ARCHITECTURES}")
                list(APPEND _hypre_autotools_configure_extra --with-gpu-arch=${_hypre_hip_arch})
            endif()
        endif()

        ExternalProject_Add(hypre_autotools
            SOURCE_DIR ${_hypre_autotools_src}
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
                CC=${MPI_C_COMPILER}
                CFLAGS=${_hypre_autotools_cflags}
                ${_hypre_autotools_src}/configure --prefix=${_hypre_autotools_prefix}
                ${_hypre_autotools_configure_extra}
            BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${_hypre_autotools_src} ${CMAKE_MAKE_PROGRAM}
            INSTALL_COMMAND ${CMAKE_COMMAND} -E chdir ${_hypre_autotools_src} ${CMAKE_MAKE_PROGRAM} install
            BUILD_IN_SOURCE 1
        )

        # Ensure the install prefix directories exist at configure time for imported targets.
        file(MAKE_DIRECTORY "${_hypre_autotools_prefix}/include")
        file(MAKE_DIRECTORY "${_hypre_autotools_prefix}/lib")

        set(HYPRE_INCLUDE_DIRS "${_hypre_autotools_prefix}/include")
        set(HYPRE_LIBRARIES "${_hypre_autotools_prefix}/lib/libHYPRE.a")

        add_library(HYPRE::HYPRE INTERFACE IMPORTED)
        set_target_properties(HYPRE::HYPRE PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${HYPRE_LIBRARIES}"
        )
        _hypredrv_link_mpi_interface(HYPRE::HYPRE)
        add_dependencies(HYPRE::HYPRE hypre_autotools)

        set(HYPREDRV_HYPRE_USER_PROVIDED TRUE)
        set(HYPREDRV_HYPRE_AUTOTOOLS TRUE)
        set(HYPRE_FOUND TRUE)

        unset(_hypre_version_tag)
        unset(_hypre_autotools_prefix)
        unset(_hypre_autotools_src)
        unset(_hypre_autotools_cflags)
        unset(_hypre_autotools_configure_extra)
        unset(_hypre_cuda_arch)
        unset(_hypre_hip_arch)
    else()
    # Generic CMake argument inheritance
    # Forward all relevant cache variables to HYPRE build, including TPLs (MAGMA, CUDA, etc.)
    message(STATUS "Inheriting CMake arguments to HYPRE build...")

    # Forward relevant cache variables for hypre sub-builds.
    _hypredrv_forward_hypre_cache_vars(_forwarded_count)

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
    _hypredrv_set_common_output_directories()

    if(_forwarded_count GREATER 0)
        message(STATUS "Forwarded ${_forwarded_count} CMake variables to HYPRE build")
    endif()

    # Fetch and configure HYPRE
    # We need to patch HYPRE's CMakeLists.txt before configuration, so we use
    # FetchContent_MakeAvailable to populate, then patch, then manually add subdirectory
    message(STATUS "Fetching HYPRE from GitHub (branch/tag: ${HYPRE_VERSION})...")
    message(STATUS "  Repository: https://github.com/hypre-space/hypre.git")

    # Use FetchContent_MakeAvailable to populate the source
    # Since SOURCE_SUBDIR was removed from FetchContent_Declare, it won't automatically
    # add the subdirectory, allowing us to patch first
    FetchContent_MakeAvailable(hypre)

    message(STATUS "HYPRE source fetched successfully")
    message(STATUS "  Source directory: ${hypre_SOURCE_DIR}")

    # Patch HYPRE's CMakeLists.txt to skip export when Caliper is auto-built
    # This must be done before add_subdirectory is called
    if(HYPRE_BUILD_CALIPER AND EXISTS "${hypre_SOURCE_DIR}/src/CMakeLists.txt")
        file(READ "${hypre_SOURCE_DIR}/src/CMakeLists.txt" HYPRE_CMAKE_CONTENT)
        if(HYPRE_CMAKE_CONTENT MATCHES "Export from build tree" AND NOT HYPRE_CMAKE_CONTENT MATCHES "HYPRE_BUILD_CALIPER")
            string(REGEX REPLACE
                "if\\(NOT \\(HYPRE_BUILD_UMPIRE AND TARGET umpire\\)\\)"
                "if(NOT (HYPRE_BUILD_UMPIRE AND TARGET umpire) AND NOT (HYPRE_BUILD_CALIPER AND TARGET caliper))"
                HYPRE_CMAKE_CONTENT "${HYPRE_CMAKE_CONTENT}")
            string(REGEX REPLACE
                "Skipping build-tree export of HYPRETargets due to auto-built Umpire dependency"
                "Skipping build-tree export of HYPRETargets due to auto-built Umpire or Caliper dependency"
                HYPRE_CMAKE_CONTENT "${HYPRE_CMAKE_CONTENT}")
            file(WRITE "${hypre_SOURCE_DIR}/src/CMakeLists.txt" "${HYPRE_CMAKE_CONTENT}")
            message(STATUS "  HYPRE CMakeLists.txt patched to skip export when Caliper is auto-built")
        endif()
    endif()

    # Add HYPRE subdirectory manually (pointing to src subdirectory)
    # This is done after MakeAvailable and patching to allow patching before configuration
    if(NOT TARGET HYPRE::HYPRE)
        message(STATUS "Configuring HYPRE build...")
        message(STATUS "  Libraries will be built to: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        add_subdirectory(${hypre_SOURCE_DIR}/src ${hypre_BINARY_DIR})
    endif()

    # Remove Caliper include directory from HYPRE's INTERFACE_INCLUDE_DIRECTORIES and re-add with BUILD_INTERFACE
    # This avoids CMake errors about _deps paths in INTERFACE_INCLUDE_DIRECTORIES
    if(HYPRE_BUILD_CALIPER AND TARGET HYPRE AND CALIPER_INCLUDE_DIR_FOR_HYPRE)
        # Get current include directories
        get_target_property(HYPRE_INCLUDE_DIRS HYPRE INTERFACE_INCLUDE_DIRECTORIES)
        if(HYPRE_INCLUDE_DIRS)
            # Remove the _deps path if present
            list(REMOVE_ITEM HYPRE_INCLUDE_DIRS "${CALIPER_INCLUDE_DIR_FOR_HYPRE}")
            # Set the cleaned list
            set_target_properties(HYPRE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
            # Re-add with BUILD_INTERFACE
            target_include_directories(HYPRE PUBLIC $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIR_FOR_HYPRE}>)
            message(STATUS "  Fixed Caliper include directory in HYPRE target using BUILD_INTERFACE")
        endif()

        # Ensure HYPRE depends on Caliper target so it builds after Caliper
        # This is critical because HYPRE links against the Caliper library file
        if(TARGET caliper)
            add_dependencies(HYPRE caliper)
            message(STATUS "  Added dependency: HYPRE -> caliper")
        endif()
    endif()

    message(STATUS "HYPRE configured and ready")
    endif()
endif()

# Get HYPRE properties
if(TARGET HYPRE::HYPRE)
    get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)

    # Try to get library location - handle both Release and Debug configurations.
    # For autotools/custom installs we only have INTERFACE_LINK_LIBRARIES, so
    # fall back to the user-provided variables for a clearer message.
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION)
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELWITHDEBINFO)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_DEBUG)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        if(DEFINED HYPRE_LIBRARIES)
            set(HYPRE_LIBRARY_FILE "${HYPRE_LIBRARIES}")
        elseif(DEFINED HYPRE_LIBRARY)
            set(HYPRE_LIBRARY_FILE "${HYPRE_LIBRARY}")
        else()
            set(HYPRE_LIBRARY_FILE "not found (using target)")
        endif()
    endif()

    # Check if HYPRE was built with Caliper support (for find_package case)
    # This is needed because HYPRE_ENABLE_CALIPER might not be set when HYPRE is found via find_package
    # Only check if HYPREDRV_ENABLE_CALIPER is not already explicitly set
    if(NOT DEFINED HYPREDRV_ENABLE_CALIPER)
        # Check if HYPRE links to Caliper
        get_target_property(HYPRE_LINK_LIBS HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
        set(HYPRE_HAS_CALIPER FALSE)

        if(HYPRE_LINK_LIBS)
            # Check if caliper is in the link libraries (could be caliper::caliper, caliper, or libcaliper)
            foreach(lib ${HYPRE_LINK_LIBS})
                if(lib MATCHES "caliper" OR lib STREQUAL "caliper::caliper" OR lib STREQUAL "caliper")
                    set(HYPRE_HAS_CALIPER TRUE)
                    break()
                endif()
            endforeach()
        endif()

        # Also check compile definitions for Caliper-related defines
        if(NOT HYPRE_HAS_CALIPER)
            get_target_property(HYPRE_COMPILE_DEFS HYPRE::HYPRE INTERFACE_COMPILE_DEFINITIONS)
            if(HYPRE_COMPILE_DEFS)
                foreach(def ${HYPRE_COMPILE_DEFS})
                    if(def MATCHES "HYPRE_USING_CALIPER" OR def MATCHES "CALIPER")
                        set(HYPRE_HAS_CALIPER TRUE)
                        break()
                    endif()
                endforeach()
            endif()
        endif()

        # If we detected Caliper support, enable HYPREDRV_ENABLE_CALIPER
        if(HYPRE_HAS_CALIPER)
            set(HYPREDRV_ENABLE_CALIPER ON CACHE BOOL "Enable Caliper instrumentation support" FORCE)
            message(STATUS "  Detected HYPRE was built with Caliper support - enabling HYPREDRV_ENABLE_CALIPER")
        endif()
    endif()

    message(STATUS "Found HYPRE:")
    message(STATUS "  include directories: ${HYPRE_INCLUDE_DIRS}")
    message(STATUS "  libraries: ${HYPRE_LIBRARY_FILE}")

    # Build a runtime library search path for CTest so executables can launch
    # when HYPRE's transitive shared-library dependencies (e.g., Caliper) are
    # not resolved through executable RUNPATH.
    set(_hypredrv_test_runtime_lib_dirs "")

    function(_hypredrv_collect_lib_dirs_from_items out_var)
        set(_dirs "")
        foreach(_item IN LISTS ARGN)
            if(NOT _item)
                continue()
            endif()

            if(TARGET "${_item}")
                get_target_property(_item_lib "${_item}" IMPORTED_LOCATION)
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_RELEASE)
                endif()
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_RELWITHDEBINFO)
                endif()
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_DEBUG)
                endif()
                if(_item_lib AND EXISTS "${_item_lib}")
                    get_filename_component(_item_dir "${_item_lib}" DIRECTORY)
                    list(APPEND _dirs "${_item_dir}")
                endif()
                continue()
            endif()

            # Ignore generator expressions and plain library names.
            if(_item MATCHES "^\\$<" OR _item MATCHES "^[A-Za-z0-9_:+.-]+$")
                continue()
            endif()

            # Extract -L<dir> entries from interface link options/libraries.
            if(_item MATCHES "^-L(.+)$")
                set(_candidate_dir "${CMAKE_MATCH_1}")
                if(IS_DIRECTORY "${_candidate_dir}")
                    list(APPEND _dirs "${_candidate_dir}")
                endif()
                continue()
            endif()

            if(IS_ABSOLUTE "${_item}" AND EXISTS "${_item}")
                get_filename_component(_item_dir "${_item}" DIRECTORY)
                list(APPEND _dirs "${_item_dir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _dirs)
        set(${out_var} "${_dirs}" PARENT_SCOPE)
    endfunction()

    list(APPEND _hypredrv_test_runtime_lib_dirs "${CMAKE_BINARY_DIR}/lib")
    _hypredrv_collect_lib_dirs_from_items(_hypredrv_hypre_lib_dirs ${HYPRE_LIBRARY_FILE})
    list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_hypre_lib_dirs})

    get_target_property(_hypredrv_hypre_link_libs HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
    if(_hypredrv_hypre_link_libs)
        _hypredrv_collect_lib_dirs_from_items(_hypredrv_hypre_link_dirs ${_hypredrv_hypre_link_libs})
        list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_hypre_link_dirs})
    endif()

    # HYPREConfig.cmake exports dependency roots (and sometimes lib dirs). When
    # pkg-config synthesized interface targets hide full library paths (e.g.,
    # "caliper"), use these hints to recover <prefix>/lib.
    if(DEFINED HYPRE_DEPENDENCY_DIRS AND HYPRE_DEPENDENCY_DIRS)
        foreach(_dep_dir IN LISTS HYPRE_DEPENDENCY_DIRS)
            if(NOT IS_DIRECTORY "${_dep_dir}")
                continue()
            endif()
            if(_dep_dir MATCHES "(/|^)lib64?$")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}")
            endif()
            if(IS_DIRECTORY "${_dep_dir}/lib")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}/lib")
            endif()
            if(IS_DIRECTORY "${_dep_dir}/lib64")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}/lib64")
            endif()
        endforeach()
    endif()

    if(DEFINED MPI_C_LIBRARIES AND MPI_C_LIBRARIES)
        _hypredrv_collect_lib_dirs_from_items(_hypredrv_mpi_link_dirs ${MPI_C_LIBRARIES})
        list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_mpi_link_dirs})
    endif()

    list(REMOVE_DUPLICATES _hypredrv_test_runtime_lib_dirs)

    if(UNIX AND NOT APPLE AND _hypredrv_test_runtime_lib_dirs)
        string(JOIN ":" _hypredrv_ld_path ${_hypredrv_test_runtime_lib_dirs})
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "LD_LIBRARY_PATH=${_hypredrv_ld_path}:$ENV{LD_LIBRARY_PATH}"
            CACHE INTERNAL "CTest runtime library path environment assignment")
        message(STATUS "  test runtime env: ${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    elseif(APPLE AND _hypredrv_test_runtime_lib_dirs)
        string(JOIN ":" _hypredrv_dyld_path ${_hypredrv_test_runtime_lib_dirs})
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "DYLD_LIBRARY_PATH=${_hypredrv_dyld_path}:$ENV{DYLD_LIBRARY_PATH}"
            CACHE INTERNAL "CTest runtime library path environment assignment")
        message(STATUS "  test runtime env: ${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    else()
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT "" CACHE INTERNAL
            "CTest runtime library path environment assignment")
    endif()

    # For autotools/custom HYPRE, read HYPRE_config.h to derive version info and MPI mode.
    if(HYPREDRV_HYPRE_USER_PROVIDED AND HYPRE_INCLUDE_DIRS AND NOT HYPREDRV_HYPRE_AUTOTOOLS)
        set(_hypre_config_found FALSE)
        foreach(_inc_dir IN LISTS HYPRE_INCLUDE_DIRS)
            if(EXISTS "${_inc_dir}/HYPRE_config.h")
                set(_hypre_config_found TRUE)
                file(READ "${_inc_dir}/HYPRE_config.h" _hypre_cfg)

                # Extract release number if available (2.21+), otherwise compute from release version.
                set(_hypre_rel_num "")
                set(_hypre_rel_ver "")
                set(_hypre_dev_num "")

                string(REGEX MATCH "#define[ \t]+HYPRE_RELEASE_NUMBER[ \t]+([0-9]+)" _hypre_rel_num_match "${_hypre_cfg}")
                if(_hypre_rel_num_match)
                    set(_hypre_rel_num "${CMAKE_MATCH_1}")
                endif()

                string(REGEX REPLACE ".*#define[ \t]+HYPRE_RELEASE_VERSION[ \t]+\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\1;\\2;\\3" _hypre_rel_parts "${_hypre_cfg}")
                if(_hypre_rel_parts MATCHES ";")
                    list(GET _hypre_rel_parts 0 _hypre_rel_major)
                    list(GET _hypre_rel_parts 1 _hypre_rel_minor)
                    list(GET _hypre_rel_parts 2 _hypre_rel_patch)
                    set(_hypre_rel_ver "${_hypre_rel_major}.${_hypre_rel_minor}.${_hypre_rel_patch}")
                    if(NOT _hypre_rel_num)
                        math(EXPR _hypre_rel_num
                             "${_hypre_rel_major}*10000 + ${_hypre_rel_minor}*100 + ${_hypre_rel_patch}")
                    endif()
                endif()

                string(REGEX MATCH "#define[ \t]+HYPRE_DEVELOP_NUMBER[ \t]+([0-9]+)" _hypre_dev_num_match "${_hypre_cfg}")
                if(_hypre_dev_num_match)
                    set(_hypre_dev_num "${CMAKE_MATCH_1}")
                endif()

                if(_hypre_rel_num)
                    set(HYPREDRV_HYPRE_RELEASE_NUMBER "${_hypre_rel_num}")
                endif()
                if(_hypre_rel_ver)
                    set(HYPREDRV_HYPRE_RELEASE_VERSION "${_hypre_rel_ver}")
                endif()
                if(_hypre_dev_num)
                    set(HYPREDRV_HYPRE_DEVELOP_NUMBER "${_hypre_dev_num}")
                endif()

                if(HYPREDRV_HYPRE_RELEASE_NUMBER)
                    add_compile_definitions(HYPREDRV_HYPRE_RELEASE_NUMBER=${HYPREDRV_HYPRE_RELEASE_NUMBER})
                endif()
                if(HYPREDRV_HYPRE_DEVELOP_NUMBER)
                    add_compile_definitions(HYPREDRV_HYPRE_DEVELOP_NUMBER=${HYPREDRV_HYPRE_DEVELOP_NUMBER})
                endif()
                if(HYPREDRV_HYPRE_RELEASE_VERSION)
                    add_compile_definitions(HYPREDRV_HYPRE_RELEASE_VERSION=\"${HYPREDRV_HYPRE_RELEASE_VERSION}\")
                endif()

                # If HYPRE was built with MPI (HYPRE_SEQUENTIAL not defined), make sure MPI is available.
                string(REGEX MATCH "#define[ \t]+HYPRE_SEQUENTIAL" _hypre_seq "${_hypre_cfg}")
                if(NOT _hypre_seq)
                    message(STATUS "  HYPRE built with MPI support; checking MPI...")
                    find_package(MPI REQUIRED COMPONENTS C)
                    _hypredrv_link_mpi_interface(HYPRE::HYPRE)
                else()
                    message(STATUS "  HYPRE built in sequential mode; MPI not required")
                endif()
                break()
            endif()
        endforeach()
        if(NOT _hypre_config_found)
            message(WARNING "HYPRE_config.h not found in HYPRE_INCLUDE_DIRS; cannot determine MPI requirement or version")
        endif()
    endif()
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

############################################################
# Find and configure optional compression backends
############################################################
set(HYPREDRV_COMPRESSION_INCLUDE_DIRS "")
set(HYPREDRV_COMPRESSION_LIBRARIES "")
set(HYPREDRV_USING_ZLIB 0 CACHE INTERNAL "Compression backend zlib")
set(HYPREDRV_USING_ZSTD 0 CACHE INTERNAL "Compression backend zstd")
set(HYPREDRV_USING_LZ4 0 CACHE INTERNAL "Compression backend lz4")
set(HYPREDRV_USING_BLOSC 0 CACHE INTERNAL "Compression backend blosc")

if(HYPREDRV_ENABLE_COMPRESSION)
    message(STATUS "Compression support enabled: probing optional backends")

    # zlib via CMake package
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ZLIB::ZLIB)
        set(HYPREDRV_USING_ZLIB 1 CACHE INTERNAL "Compression backend zlib" FORCE)
        message(STATUS "  zlib: found")
    else()
        message(STATUS "  zlib: not found")
    endif()

    # zstd/lz4/blosc via pkg-config or fallback path/library lookup
    find_package(PkgConfig QUIET)

    # zstd
    set(_zstd_found FALSE)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_ZSTD QUIET libzstd)
        if(PC_ZSTD_FOUND)
            set(_zstd_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${PC_ZSTD_INCLUDE_DIRS})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${PC_ZSTD_LINK_LIBRARIES})
        endif()
    endif()
    if(NOT _zstd_found)
        find_path(HYPREDRV_ZSTD_INCLUDE_DIR zstd.h)
        find_library(HYPREDRV_ZSTD_LIBRARY zstd)
        if(HYPREDRV_ZSTD_INCLUDE_DIR AND HYPREDRV_ZSTD_LIBRARY)
            set(_zstd_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${HYPREDRV_ZSTD_INCLUDE_DIR})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${HYPREDRV_ZSTD_LIBRARY})
        endif()
    endif()
    if(_zstd_found)
        set(HYPREDRV_USING_ZSTD 1 CACHE INTERNAL "Compression backend zstd" FORCE)
        message(STATUS "  zstd: found")
    else()
        message(STATUS "  zstd: not found")
    endif()

    # lz4
    set(_lz4_found FALSE)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_LZ4 QUIET liblz4)
        if(PC_LZ4_FOUND)
            set(_lz4_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${PC_LZ4_INCLUDE_DIRS})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${PC_LZ4_LINK_LIBRARIES})
        endif()
    endif()
    if(NOT _lz4_found)
        find_path(HYPREDRV_LZ4_INCLUDE_DIR lz4.h)
        find_library(HYPREDRV_LZ4_LIBRARY lz4)
        if(HYPREDRV_LZ4_INCLUDE_DIR AND HYPREDRV_LZ4_LIBRARY)
            set(_lz4_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${HYPREDRV_LZ4_INCLUDE_DIR})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${HYPREDRV_LZ4_LIBRARY})
        endif()
    endif()
    if(_lz4_found)
        set(HYPREDRV_USING_LZ4 1 CACHE INTERNAL "Compression backend lz4" FORCE)
        message(STATUS "  lz4: found")
    else()
        message(STATUS "  lz4: not found")
    endif()

    # blosc
    set(_blosc_found FALSE)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_BLOSC QUIET blosc)
        if(PC_BLOSC_FOUND)
            set(_blosc_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${PC_BLOSC_INCLUDE_DIRS})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${PC_BLOSC_LINK_LIBRARIES})
        endif()
    endif()
    if(NOT _blosc_found)
        find_path(HYPREDRV_BLOSC_INCLUDE_DIR blosc.h)
        find_library(HYPREDRV_BLOSC_LIBRARY NAMES blosc blosc2)
        if(HYPREDRV_BLOSC_INCLUDE_DIR AND HYPREDRV_BLOSC_LIBRARY)
            set(_blosc_found TRUE)
            list(APPEND HYPREDRV_COMPRESSION_INCLUDE_DIRS ${HYPREDRV_BLOSC_INCLUDE_DIR})
            list(APPEND HYPREDRV_COMPRESSION_LIBRARIES ${HYPREDRV_BLOSC_LIBRARY})
        endif()
    endif()
    if(_blosc_found)
        set(HYPREDRV_USING_BLOSC 1 CACHE INTERNAL "Compression backend blosc" FORCE)
        message(STATUS "  blosc: found")
    else()
        message(STATUS "  blosc: not found")
    endif()

    if(HYPREDRV_COMPRESSION_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES HYPREDRV_COMPRESSION_INCLUDE_DIRS)
    endif()
    if(HYPREDRV_COMPRESSION_LIBRARIES)
        list(REMOVE_DUPLICATES HYPREDRV_COMPRESSION_LIBRARIES)
    endif()
else()
    message(STATUS "Compression support disabled (HYPREDRV_ENABLE_COMPRESSION=OFF)")
endif()
