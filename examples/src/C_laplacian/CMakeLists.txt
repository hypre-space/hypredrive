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

# Apply sanitizer link flags if sanitizers are enabled
get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(laplacian PRIVATE ${flag})
    endforeach()
endif()

# Add an RPATH to the lib directory for MacOS (libraries are in lib/)
if(APPLE)
    target_link_options(laplacian PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

# Add the laplacian driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    add_executable_test(laplacian_test_1proc laplacian 1
        ARGS -n 6 6 6 -ns 1
        RUN_SERIAL
    )
    add_executable_test(laplacian_test_4proc laplacian 4
        ARGS -n 6 6 6 -P 2 2 1 -ns 1
    )
endif()

# Install the laplacian example binary when examples are built
install(TARGETS laplacian
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
