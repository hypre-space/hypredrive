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
    set(_lidcavity_print_system_test_dir
        ${CMAKE_CURRENT_BINARY_DIR}/print_system_threshold_test)
    set(_lidcavity_adaptive_reuse_test_dir
        ${CMAKE_CURRENT_BINARY_DIR}/adaptive_reuse_test)
    file(MAKE_DIRECTORY ${_lidcavity_print_system_test_dir})
    file(MAKE_DIRECTORY ${_lidcavity_adaptive_reuse_test_dir})
    if(HYPREDRV_HAVE_HYPRE_23000_DEV0)
        add_executable_test(lidcavity_test_1proc lidcavity 1
            ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
            RUN_SERIAL
        )
        add_executable_test(lidcavity_test_4proc lidcavity 4
            ARGS -v 1 -n 12 12 -P 2 2 -ns 1 -dt 0.1 -tf 0.3
        )
        add_executable_test(lidcavity_stats_path_1proc lidcavity 1
            ARGS -v 1 -n 12 12 -dt 0.1 -tf 0.2
            RUN_SERIAL
            REQUIRE_CONTAINS
                "STATISTICS SUMMARY:"
                "Path"
                "1.1.1"
        )
        if(HYPREDRV_ENABLE_EXPERIMENTAL)
            add_executable_test(lidcavity_test_schwarz_ras_iluk_1proc lidcavity 1
                ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                     -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-schwarz-ras-iluk.yml
                RUN_SERIAL
            )
            if(HYPREDRV_HAVE_HYPRE_USING_DSUPERLU)
                add_executable_test(lidcavity_test_schwarz_ras_spdirect_1proc lidcavity 1
                    ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                         -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-schwarz-ras-spdirect.yml
                    RUN_SERIAL
                )
            endif()
        endif()
    endif()
    if(HYPREDRV_HAVE_HYPRE_30100_DEV5)
        add_executable_test(lidcavity_test_mgr_1proc lidcavity 1
            ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr.yml
            RUN_SERIAL
        )
        add_executable_test(lidcavity_test_print_1proc lidcavity 1
            ARGS -v 0 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-print-system.yml
            WORKING_DIRECTORY ${_lidcavity_print_system_test_dir}
            REQUIRE_PATHS
                print-trigger-dumps/lidcavity-print-system/systems_index.txt
                print-trigger-dumps/lidcavity-print-system/ls_00000/metadata.txt
            RUN_SERIAL
        )
        add_executable_test(lidcavity_test_adaptive_reuse_1proc lidcavity 1
            ARGS -v 0 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-adaptive-reuse.yml
            WORKING_DIRECTORY ${_lidcavity_adaptive_reuse_test_dir}
            REQUIRE_CONTAINS
                "adaptive reuse decision"
                "rebuild=0"
                "rebuild=1"
            RUN_SERIAL
        )
        hypredrv_append_test_environment(lidcavity_test_adaptive_reuse_1proc
            "HYPREDRV_LOG_LEVEL=2")
        if(HYPREDRV_ENABLE_EXPERIMENTAL)
            add_executable_test(lidcavity_test_mgr_schwarz_frelax_iluk_1proc lidcavity 1
                ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                     -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-schwarz-frelax-iluk.yml
                RUN_SERIAL
            )
            add_executable_test(lidcavity_test_mgr_schwarz_grelax_iluk_1proc lidcavity 1
                ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                     -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-schwarz-grelax-iluk.yml
                RUN_SERIAL
            )
            add_executable_test(lidcavity_test_mgr_schwarz_coarsest_iluk_1proc lidcavity 1
                ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                     -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-schwarz-coarsest-iluk.yml
                RUN_SERIAL
            )
            if(HYPREDRV_HAVE_HYPRE_USING_DSUPERLU)
                add_executable_test(lidcavity_test_mgr_schwarz_frelax_spdirect_1proc lidcavity 1
                    ARGS -v 1 -n 12 12 -ns 1 -dt 0.1 -tf 0.3
                         -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr-schwarz-frelax-spdirect.yml
                    RUN_SERIAL
                )
            endif()
        endif()
        if(HYPREDRV_HAVE_HYPRE_DEVELOP_NUMBER)
            add_executable_test(lidcavity_test_mgr_4proc lidcavity 4
              ARGS -v 1 -n 12 12 -P 2 2 -ns 1 -dt 0.1 -tf 0.3 -i ${CMAKE_CURRENT_SOURCE_DIR}/fgmres-mgr.yml
            )
        else()
            message(STATUS
                "Skipping lidcavity_test_mgr_4proc for exact Hypre v3.1.0 "
                "due to an upstream MPI MGR failure"
            )
        endif()
    endif()
endif()

# Install the elasticity example binary when examples are built
install(TARGETS lidcavity
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
