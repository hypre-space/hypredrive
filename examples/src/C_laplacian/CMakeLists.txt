cmake_minimum_required(VERSION 3.12)
project(laplacian C)

# Disallow in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

set(CMAKE_C_STANDARD 99)

# Add the cmake folder to the module path
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Check if we're building as part of the main project
if(NOT TARGET HYPREDRV::HYPREDRV)
    # Standalone build - try to find installed HYPREDRV
    find_package(HYPREDRV REQUIRED)
endif()

# Common build configuration
add_executable(laplacian laplacian.c)
target_link_libraries(laplacian PRIVATE HYPREDRV::HYPREDRV)

# Install the laplacian example binary when examples are built
install(TARGETS laplacian
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
