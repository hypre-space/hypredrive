cmake_minimum_required(VERSION 3.12)
project(heatflow C)

# Disallow in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

set(CMAKE_C_STANDARD 99)

# Check if we're building as part of the main project
if(NOT TARGET HYPREDRV::HYPREDRV)
    # Standalone build - try to find installed HYPREDRV
    find_package(HYPREDRV REQUIRED)
endif()

add_executable(heatflow heatflow.c)
target_link_libraries(heatflow PRIVATE HYPREDRV::HYPREDRV)

install(TARGETS heatflow
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



