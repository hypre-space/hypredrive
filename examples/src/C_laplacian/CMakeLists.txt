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

# Add the laplacian driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    add_test(NAME laplacian_driver_1proc
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:laplacian> -n 6 6 6 -ns 1 ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    set_tests_properties(laplacian_driver_1proc
        PROPERTIES
            RUN_SERIAL TRUE
            FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure"
    )
endif()

# Install the laplacian example binary when examples are built
install(TARGETS laplacian
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
