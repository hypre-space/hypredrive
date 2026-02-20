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

# If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(elasticity PRIVATE stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang can use either libstdc++ or libc++
            # Default to stdc++ on Linux
            if(UNIX AND NOT APPLE)
                target_link_libraries(elasticity PRIVATE stdc++)
            else()
                target_link_libraries(elasticity PRIVATE c++)
            endif()
        endif()
    endif()
endif()

# Add OpenMP support for parallel assembly
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(elasticity PRIVATE OpenMP::OpenMP_C)
    message(STATUS "OpenMP enabled for elasticity driver")
else()
    message(STATUS "OpenMP not found - elasticity driver will build without OpenMP")
endif()

# Apply sanitizer link flags if sanitizers are enabled
get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(elasticity PRIVATE ${flag})
    endforeach()
endif()

# Add an RPATH to the lib directory for MacOS (libraries are in lib/)
if(APPLE)
    target_link_options(elasticity PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

# Add the elasticity driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    add_executable_test(elasticity_test_1proc elasticity 1
        ARGS -n 12 4 4 -ns 1 -v 1
        RUN_SERIAL
    )
    add_executable_test(elasticity_test_4proc elasticity 4
        ARGS -n 12 4 4 -P 2 2 1 -ns 1 -v 1
    )
endif()

# Install the elasticity example binary when examples are built
install(TARGETS elasticity
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
