cmake_minimum_required(VERSION 3.12)
project(elasticity C)

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

# Common build configuration
add_executable(elasticity elasticity.c)
target_link_libraries(elasticity PRIVATE HYPREDRV::HYPREDRV)

# Add an RPATH to the main build directory for MacOS
if(APPLE)
    target_link_options(laplacian PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}")
endif()

# Add the elasticity driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    add_executable_test(elasticity_test_1proc elasticity 1
        ARGS -n 12 4 4 -ns 1
        RUN_SERIAL
    )
    add_executable_test(elasticity_test_4proc elasticity 4
        ARGS -n 12 4 4 -P 2 2 1 -ns 1
    )
endif()

# Install the elasticity example binary when examples are built
install(TARGETS elasticity
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
