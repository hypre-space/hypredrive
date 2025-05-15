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

# Optionally set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find HYPRE package
find_package(HYPRE REQUIRED CONFIG)
if(NOT HYPRE_FOUND)
  message(FATAL_ERROR "HYPRE library not found. Please specify -DHYPRE_ROOT=<path> to the root of the HYPRE installation.")
endif()
get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
message(STATUS "Found HYPRE:")
message(STATUS "  include directories: ${HYPRE_INCLUDE_DIRS}")
message(STATUS "  libraries: ${HYPRE_LIBRARY_FILE}")

# Check for headers, functions, and libraries
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRV_Checks.cmake)

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
target_link_libraries(HYPREDRV PUBLIC HYPRE::HYPRE PRIVATE m)

# Define the executable
add_executable(hypredrive src/main.c)
target_link_libraries(hypredrive PRIVATE HYPREDRV::HYPREDRV HYPRE::HYPRE)

# macOS RPATH settings
if(APPLE)
    # Check if HYPRE library is shared
    get_property(HYPRE_LIB_TYPE TARGET HYPRE::HYPRE PROPERTY TYPE)
    if(HYPRE_LIB_TYPE STREQUAL "SHARED_LIBRARY")
        get_filename_component(HYPRE_LIBRARY_DIR "${HYPRE_LIBRARY_FILE}/.." ABSOLUTE)
        set(CMAKE_INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set_target_properties(hypredrive PROPERTIES INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
        set_target_properties(HYPREDRV PROPERTIES INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
    endif()
endif()

# Installation rules
install(TARGETS hypredrive HYPREDRV
        EXPORT HYPREDRVTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Create and install Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRVConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV
)

# Export the targets
install(EXPORT HYPREDRVTargets
        FILE HYPREDRVTargets.cmake
        NAMESPACE HYPREDRV::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV)

# Install the Config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV
)

# Export the targets for use in the build tree
export(EXPORT HYPREDRVTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVTargets.cmake"
       NAMESPACE HYPREDRV::)

# Register package in user's package registry
export(PACKAGE HYPREDRV)

# Add examples
option(HYPREDRV_BUILD_EXAMPLES "Build the HYPREDRV examples" ON)
if(HYPREDRV_BUILD_EXAMPLES)
    add_subdirectory(examples/src/C_laplacian)
endif()

# Find clang-format
find_program(CLANG_FORMAT "clang-format")
if(NOT CLANG_FORMAT)
    message(STATUS "clang-format not found, formatting targets will not be available")
else()
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}
            find . -type f -name "*.c" -not -path "./build/*" -exec ${CLANG_FORMAT} -i {} +
        COMMENT "Running clang-format..."
    )
endif()
