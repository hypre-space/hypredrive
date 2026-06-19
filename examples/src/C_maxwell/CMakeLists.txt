cmake_minimum_required(VERSION 3.12)
project(maxwell C)

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
add_executable(maxwell maxwell.c)
target_link_libraries(maxwell PRIVATE HYPREDRV::HYPREDRV)
if(COMMAND hypredrv_set_relative_install_rpath)
    hypredrv_set_relative_install_rpath(maxwell)
endif()

# If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(maxwell PRIVATE stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            if(UNIX AND NOT APPLE)
                target_link_libraries(maxwell PRIVATE stdc++)
            else()
                target_link_libraries(maxwell PRIVATE c++)
            endif()
        endif()
    endif()
endif()

# Apply sanitizer link flags if sanitizers are enabled
get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(maxwell PRIVATE ${flag})
    endforeach()
endif()

# Add an RPATH to the lib directory for MacOS (libraries are in lib/)
if(APPLE)
    target_link_options(maxwell PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

# Add the maxwell driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    add_executable_test(maxwell_test_1proc maxwell 1
        ARGS -n 9 9 9 -ns 1 -v 1
        RUN_SERIAL
    )
    add_executable_test(maxwell_test_4proc maxwell 4
        ARGS -n 9 9 9 -P 2 2 1 -ns 1 -v 1
    )
endif()

# Install the maxwell example binary when examples are built
install(TARGETS maxwell
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
