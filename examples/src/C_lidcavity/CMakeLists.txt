cmake_minimum_required(VERSION 3.21)
project(lidcavity C)

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
add_executable(lidcavity lidcavity.c)
target_link_libraries(lidcavity PRIVATE HYPREDRV::HYPREDRV m)

# If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(lidcavity PRIVATE stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang can use either libstdc++ or libc++
            # Default to stdc++ on Linux
            if(UNIX AND NOT APPLE)
                target_link_libraries(lidcavity PRIVATE stdc++)
            else()
                target_link_libraries(lidcavity PRIVATE c++)
            endif()
        endif()
    endif()
endif()

# Apply sanitizer link flags if sanitizers are enabled
get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(lidcavity PRIVATE ${flag})
    endforeach()
endif()

# Add an RPATH to the lib directory for MacOS (libraries are in lib/)
if(APPLE)
    target_link_options(lidcavity PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

# Add the elasticity driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    if(HYPREDRV_HAVE_HYPRE_23000_DEV0)
        add_executable_test(lidcavity_test_1proc lidcavity 1
            ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
            RUN_SERIAL
        )
        add_executable_test(lidcavity_test_4proc lidcavity 4
            ARGS -v 1 -n 12 12 -P 2 2 -ns 1 -dt 0.1 -tf 0.3
        )
    endif()
    if(HYPREDRV_HAVE_HYPRE_30100_DEV5)
        add_executable_test(lidcavity_test_mgr_1proc lidcavity 1
            ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr.yml
            RUN_SERIAL
        )
        add_executable_test(lidcavity_test_mgr_4proc lidcavity 4
            ARGS -v 1 -n 12 12 -P 2 2 -ns 1 -dt 0.1 -tf 0.3 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr.yml
            RUN_SERIAL
        )
    endif()
endif()

# Install the elasticity example binary when examples are built
install(TARGETS lidcavity
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
