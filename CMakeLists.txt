# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# CMake minimum version requirement
cmake_minimum_required(VERSION 3.23)

# Project name and version
project(hypredrive VERSION 0.2.0 LANGUAGES C)
set(PROJECT_URL "https://github.com/hypre-space/hypredrive")
set(PROJECT_BUGREPORT "https://github.com/hypre-space/hypredrive/issues")

# Include GNUInstallDirs
include(GNUInstallDirs)

# Specify the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Disallow in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

# Option to build shared libraries
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# HYPREDRV build options
option(HYPREDRV_ENABLE_EIGSPEC      "Enable full eigenspectrum computation" OFF)
option(HYPREDRV_ENABLE_EXPERIMENTAL "Enable experimental features" OFF)
option(HYPREDRV_ENABLE_TESTING      "Enable testing support and check target" OFF)
option(HYPREDRV_ENABLE_ALL_TESTS    "Force-enable tests that are otherwise disabled in some configurations" OFF)
option(HYPREDRV_ENABLE_COVERAGE     "Enable code coverage instrumentation" OFF)
option(HYPREDRV_ENABLE_ANALYSIS     "Enable static code analysis (clang-tidy) and sanitizers (ASan/UBSan)" OFF)
option(HYPREDRV_ENABLE_DATA         "Enable 'data' target to fetch example datasets" OFF)
option(HYPREDRV_ENABLE_DOCS         "Enable documentation generation" OFF)
option(HYPREDRV_ENABLE_EXAMPLES     "Enable the HYPREDRV examples" OFF)
option(HYPREDRV_ENABLE_HWLOC        "Enable hwloc support for detailed system topology information" OFF)
option(HYPREDRV_ENABLE_CALIPER      "Enable Caliper instrumentation support" OFF)
option(HYPREDRV_ENABLE_COMPRESSION  "Enable optional lossless compression backends" OFF)

# Optionally set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Testing implies data
if(HYPREDRV_ENABLE_TESTING)
    set(HYPREDRV_ENABLE_DATA ON)
endif()

# Coverage implies testing, examples, and data
if(HYPREDRV_ENABLE_COVERAGE)
    set(HYPREDRV_ENABLE_DATA ON)
    set(HYPREDRV_ENABLE_TESTING ON)
    set(HYPREDRV_ENABLE_EXAMPLES ON)
endif()

# Configure instrumentation before bringing in fetched dependencies so options
# like sanitizers/coverage also apply to HYPRE when we build it in-tree.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Instrumentation.cmake)

# Configure unified output directories for all libraries and executables
# This ensures all libraries (including dependencies) are in the same folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Some external HYPRE package configs pull in C++/SYCL-side imported targets
# during find_package(HYPRE). Make CXX available up front when a compiler
# exists, without forcing pure-C builds to fail on toolchains that lack C++.
enable_language(CXX OPTIONAL)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find and configure dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Deps.cmake)

# Check for headers, functions, and libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Checks.cmake)

# Download and extract datasets
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Data.cmake)

# Capture git version info for hypredrive
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Git.cmake)
hypredrv_setup_git_version_info(${CMAKE_CURRENT_SOURCE_DIR})
if(HYPREDRV_GIT_FOUND)
    message(STATUS "HYPREDRV develop string: ${HYPREDRV_DEVELOP_STRING}")
    message(STATUS "HYPREDRV branch: ${HYPREDRV_BRANCH_NAME}")
    if(HYPREDRV_GIT_SHA)
        message(STATUS "HYPREDRV git SHA: ${HYPREDRV_GIT_SHA}")
    endif()
else()
    message(STATUS "HYPREDRV git SHA: not available (using release ${PROJECT_VERSION})")
endif()

# Generate the config header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRV_config.h"
)

# Include the generated config header
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Explicit library source list (keep in sync with Makefile.am)
set(SOURCE_FILES
    src/error.c
    src/eigspec.c
    src/info.c
    src/stats.c
    src/utils.c
    src/containers.c
    src/yaml.c
    src/field.c
    src/vector.c
    src/matrix.c
    src/linsys.c
    src/linsys_print.c
    src/cheby.c
    src/amg.c
    src/fsai.c
    src/ilu.c
    src/mgr.c
    src/precon.c
    src/precon_reuse.c
    src/presets.c
    src/krylov.c
    src/pcg.c
    src/gmres.c
    src/fgmres.c
    src/bicgstab.c
    src/solver.c
    src/args.c
    src/comp.c
    src/lsseq.c
    src/scaling.c
    src/runtime.c
    src/logging.c
    src/HYPREDRV.c
)

# Define the library
add_library(HYPREDRV ${SOURCE_FILES})
add_library(HYPREDRV::HYPREDRV ALIAS HYPREDRV)

# Set library properties
set_target_properties(HYPREDRV PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/HYPREDRV.h;${PROJECT_SOURCE_DIR}/include/HYPREDRV_utils.h;${CMAKE_CURRENT_BINARY_DIR}/HYPREDRV_config.h"
)

# Include directories for HYPREDRV
target_include_directories(HYPREDRV
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
)

# Link HYPRE library to HYPREDRV
target_link_libraries(HYPREDRV PUBLIC HYPRE::HYPRE)

if(HYPREDRV_INSTRUMENTATION_LINK_FLAGS)
    target_link_options(HYPREDRV PUBLIC ${HYPREDRV_INSTRUMENTATION_LINK_FLAGS})
endif()

if(HYPREDRV_COVERAGE_RUNTIME_LIBRARY)
    target_link_libraries(HYPREDRV PUBLIC ${HYPREDRV_COVERAGE_RUNTIME_LIBRARY})
endif()

# Link math library as PUBLIC so static consumers that also link static HYPRE
# receive libm after transitive dependencies (required in compatibility builds).
target_link_libraries(HYPREDRV PUBLIC m)

if(UNIX AND NOT APPLE)
    # Guard against regressions where static consumers lose libm and fail link.
    get_target_property(_hypredrv_iface_link_libs HYPREDRV INTERFACE_LINK_LIBRARIES)
    set(_hypredrv_has_libm OFF)
    foreach(_lib IN LISTS _hypredrv_iface_link_libs)
        if(_lib STREQUAL "m" OR _lib STREQUAL "$<LINK_ONLY:m>")
            set(_hypredrv_has_libm ON)
            break()
        endif()
    endforeach()
    if(NOT _hypredrv_has_libm)
        message(FATAL_ERROR "HYPREDRV must propagate libm in INTERFACE_LINK_LIBRARIES on Unix")
    endif()
endif()

if(HWLOC_FOUND)
    target_include_directories(HYPREDRV PRIVATE ${HWLOC_INCLUDE_DIR})
    target_link_libraries(HYPREDRV PRIVATE ${HWLOC_LIBRARY})
endif()

if(HYPREDRV_ENABLE_COMPRESSION)
    if(HYPREDRV_COMPRESSION_INCLUDE_DIRS)
        target_include_directories(HYPREDRV PRIVATE ${HYPREDRV_COMPRESSION_INCLUDE_DIRS})
    endif()
    if(HYPREDRV_COMPRESSION_LIBRARIES)
        target_link_libraries(HYPREDRV PRIVATE ${HYPREDRV_COMPRESSION_LIBRARIES})
    endif()
endif()

if(CALIPER_FOUND)
    if(TARGET caliper::caliper)
        target_link_libraries(HYPREDRV PRIVATE caliper::caliper)
        # Ensure Caliper headers are visible to dependents when HYPRE enables Caliper
        get_target_property(CALIPER_INCLUDE_DIRS caliper::caliper INTERFACE_INCLUDE_DIRECTORIES)
        if(CALIPER_INCLUDE_DIRS)
            target_include_directories(HYPREDRV PUBLIC $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIRS}>)
        endif()
        # Caliper is a C++ library - ensure C++ standard library is available
        # Enable C++ language support for linking C++ libraries
        enable_language(CXX OPTIONAL)
        if(CMAKE_CXX_COMPILER)
        # Link C++ standard library PUBLIC so it propagates to dependents (examples, etc.)
        # For Clang, check if we're using libstdc++ (GNU's) or libc++ (Clang's)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(HYPREDRV PUBLIC stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang can use either libstdc++ or libc++
            # Try to detect which one, but default to stdc++ on Linux
            if(UNIX AND NOT APPLE)
                target_link_libraries(HYPREDRV PUBLIC stdc++)
            else()
                target_link_libraries(HYPREDRV PUBLIC c++)
            endif()
        endif()
        endif()
    elseif(TARGET caliper)
        # Caliper was built via FetchContent - add include directories
        get_property(CALIPER_INCLUDE_DIR_FOR_HYPRE CACHE CALIPER_INCLUDE_DIR_FOR_HYPRE PROPERTY VALUE)
        if(CALIPER_INCLUDE_DIR_FOR_HYPRE)
            target_include_directories(HYPREDRV PUBLIC $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIR_FOR_HYPRE}>)
        endif()
        target_link_libraries(HYPREDRV PRIVATE caliper)
        # Enable C++ and link standard library PUBLIC
        enable_language(CXX OPTIONAL)
        if(CMAKE_CXX_COMPILER)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_link_libraries(HYPREDRV PUBLIC stdc++)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                if(UNIX AND NOT APPLE)
                    target_link_libraries(HYPREDRV PUBLIC stdc++)
                else()
                    target_link_libraries(HYPREDRV PUBLIC c++)
                endif()
            endif()
        endif()
    else()
        target_include_directories(HYPREDRV PUBLIC $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIR}>)
        target_link_libraries(HYPREDRV PRIVATE ${CALIPER_LIBRARY})
        # Enable C++ and link standard library PUBLIC for manual linking case
        enable_language(CXX OPTIONAL)
        if(CMAKE_CXX_COMPILER)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_link_libraries(HYPREDRV PUBLIC stdc++)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                target_link_libraries(HYPREDRV PUBLIC c++)
            endif()
        endif()
    endif()
endif()

# Optional eigenspectrum feature
if(HYPREDRV_ENABLE_EIGSPEC)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(HYPREDRV PUBLIC HYPREDRV_ENABLE_EIGSPEC=1)
    target_link_libraries(HYPREDRV PRIVATE LAPACK::LAPACK)
endif()

if(HYPREDRV_ENABLE_EXPERIMENTAL)
    target_compile_definitions(HYPREDRV PUBLIC HYPREDRV_ENABLE_EXPERIMENTAL=1)
endif()

# Define the main executable (hypredrive-cli)
add_executable(hypredrive-cli src/main.c)
target_link_libraries(hypredrive-cli PRIVATE HYPREDRV::HYPREDRV)

if(HYPREDRV_ENABLE_COMPRESSION)
    add_executable(hypredrive-lsseq utils/lsseq_driver.c)
    target_link_libraries(hypredrive-lsseq PRIVATE HYPREDRV::HYPREDRV)
endif()

# Ensure dataset-driven examples/tests have data available before running hypredrive-cli.
if(HYPREDRV_ENABLE_DATA AND TARGET data)
    add_dependencies(hypredrive-cli data)
endif()

# Add Caliper include directories if Caliper is enabled
if(HYPREDRV_ENABLE_CALIPER AND CALIPER_FOUND)
    if(TARGET caliper)
        # Caliper was built via FetchContent - get include dir from cache
        get_property(CALIPER_INCLUDE_DIR_FOR_HYPRE CACHE CALIPER_INCLUDE_DIR_FOR_HYPRE PROPERTY VALUE)
        if(CALIPER_INCLUDE_DIR_FOR_HYPRE)
            target_include_directories(hypredrive-cli PRIVATE $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIR_FOR_HYPRE}>)
        endif()
    elseif(TARGET caliper::caliper)
        # Caliper was found via find_package
        target_include_directories(hypredrive-cli PRIVATE caliper::caliper)
    elseif(CALIPER_INCLUDE_DIR)
        # Caliper was found via find_path
        target_include_directories(hypredrive-cli PRIVATE ${CALIPER_INCLUDE_DIR})
    endif()
endif()

# If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(hypredrive-cli PRIVATE "-lstdc++")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang can use either libstdc++ or libc++
            # Default to stdc++ on Linux
            if(UNIX AND NOT APPLE)
                target_link_options(hypredrive-cli PRIVATE "-lstdc++")
            else()
                target_link_options(hypredrive-cli PRIVATE "-lc++")
            endif()
        endif()
    endif()
endif()

# Ensure Caliper is built before hypredrive-cli if Caliper is enabled
# This is critical because HYPRE may create file dependencies on libcaliper.a
if(CALIPER_FOUND)
    # Find the actual caliper target (not the imported one)
    if(TARGET caliper)
        add_dependencies(hypredrive-cli caliper)
        # Also add to HYPREDRV to ensure it's available when linking
        add_dependencies(HYPREDRV caliper)
    endif()
    # Ensure HYPRE is built (which transitively ensures Caliper is built)
    if(TARGET HYPRE::HYPRE)
        add_dependencies(hypredrive-cli HYPRE::HYPRE)
    endif()
endif()

# Formatting target
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Format.cmake)

# Platform-specific settings
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Platform.cmake)

# Installation and package configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Install.cmake)

# Coverage support (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Coverage.cmake)

# Static analysis and sanitizers support (optional)
# Must be included before testing so sanitizer flags are available when test executables are created
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Analysis.cmake)

# Testing setup (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)

# Documentation support (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Docs.cmake)

# Add examples
if(HYPREDRV_ENABLE_EXAMPLES)
    add_subdirectory(examples/src/C_laplacian)
    add_subdirectory(examples/src/C_elasticity)
    add_subdirectory(examples/src/C_heatflow)
    add_subdirectory(examples/src/C_lidcavity)
endif()

# Add a check target
set(_hypredrv_smoke_check_args
    -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
    -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
)
if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
    list(APPEND _hypredrv_smoke_check_args
         "-DTEST_RUNTIME_ENV_ASSIGNMENT=${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
endif()
if(HYPREDRV_HYPRE_USES_SYCL)
    list(APPEND _hypredrv_smoke_check_args -DHYPREDRV_SMOKE_TEST_USES_SYCL=ON)
endif()

add_custom_target(check
  COMMAND ${CMAKE_COMMAND}
          ${_hypredrv_smoke_check_args}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_SmokeCheck.cmake
  DEPENDS hypredrive-cli
  COMMENT "Running smoke test (ex1.yml)"
  VERBATIM
)
