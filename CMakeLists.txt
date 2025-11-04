# CMake minimum version requirement
cmake_minimum_required(VERSION 3.23)

# Project name and version
project(hypredrive VERSION 0.1 LANGUAGES C)
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
option(HYPREDRV_ENABLE_EIGSPEC  "Enable full eigenspectrum computation" OFF)
option(HYPREDRV_ENABLE_TESTING  "Enable testing support and check target" ON)
option(HYPREDRV_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(HYPREDRV_ENABLE_ANALYSIS "Enable static analysis (clang-tidy) and sanitizers (ASan/UBSan)" OFF)
option(HYPREDRV_ENABLE_DATA     "Enable 'data' target to fetch example datasets" ON)
option(HYPREDRV_BUILD_EXAMPLES  "Build the HYPREDRV examples" ON)

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

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find and configure HYPRE
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_HYPRE.cmake)

# Check for headers, functions, and libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Checks.cmake)

# Download and extract datasets
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Data.cmake)

# Generate the config header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRV_config.h"
)

# Include the generated config header
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Collect all source files
file(GLOB_RECURSE SOURCE_FILES src/*.c)

# Define the library
add_library(HYPREDRV ${SOURCE_FILES})
add_library(HYPREDRV::HYPREDRV ALIAS HYPREDRV)

# Set library properties
set_target_properties(HYPREDRV PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/HYPREDRV.h;${CMAKE_CURRENT_BINARY_DIR}/HYPREDRV_config.h"
)

# Include directories for HYPREDRV
target_include_directories(HYPREDRV
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link HYPRE library to HYPREDRV
target_link_libraries(HYPREDRV PUBLIC HYPRE::HYPRE)

# Link math library as PRIVATE (only needed for implementation, not public API)
# CMake will automatically deduplicate if HYPRE already links m
target_link_libraries(HYPREDRV PRIVATE m)

# Optional eigenspectrum feature
if(HYPREDRV_ENABLE_EIGSPEC)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(HYPREDRV PUBLIC HYPREDRV_ENABLE_EIGSPEC=1)
    target_link_libraries(HYPREDRV PRIVATE LAPACK::LAPACK)
endif()

# Define the main executable (hypredrive)
add_executable(hypredrive src/main.c)
target_link_libraries(hypredrive PRIVATE HYPREDRV::HYPREDRV)

# Formatting target
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Format.cmake)

# Platform-specific settings
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Platform.cmake)

# Installation and package configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Install.cmake)

# Testing setup (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)

# Coverage support (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Coverage.cmake)

# Static analysis and sanitizers support (optional)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Analysis.cmake)

# Add examples
if(HYPREDRV_BUILD_EXAMPLES)
    add_subdirectory(examples/src/C_laplacian)
endif()

# Custom check target
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS hypredrive data
    COMMENT "Running tests"
    VERBATIM
)
