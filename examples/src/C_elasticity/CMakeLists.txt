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
if(COMMAND hypredrv_set_relative_install_rpath)
    hypredrv_set_relative_install_rpath(elasticity)
endif()

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

    # Two-material bar (mixed u-p): the built-in FGMRES+MGR configuration needs
    # a recent hypre, matching the lid-cavity Q2-Q1 MGR gate.
    if(HYPREDRV_HAVE_HYPRE_30100_DEV5)
        add_executable_test(elasticity_two_material_1proc elasticity 1
            ARGS --problem two-material -n 13 5 5 --nu-top 0.4999 -ns 1 -v 1
            RUN_SERIAL
            REQUIRE_CONTAINS
                "Two-Material Bar"
                "STATISTICS SUMMARY"
        )
        add_executable_test(elasticity_two_material_4proc elasticity 4
            ARGS --problem two-material -n 13 5 5 -P 2 2 1 --nu-top 0.4999 -ns 1 -v 1
            REQUIRE_CONTAINS
                "Two-Material Bar"
                "STATISTICS SUMMARY"
        )
        # Exercises the mixed-solution VTK reorder path (-vis on the combined layout)
        add_executable_test(elasticity_two_material_vis_4proc elasticity 4
            ARGS --problem two-material -n 13 5 5 -P 1 2 2 --nu-top 0.4999 -ns 1 -vis 1
            REQUIRE_CONTAINS
                "Two-Material Bar"
        )
        # Preconditioner reuse with an application-supplied coarse Schur matrix
        # (coarse_level_type: user, --coarse-schur). Regression for the borrowed-
        # matrix lifecycle: across -ns setups MGR reuses the coarsest solver while
        # re-aliasing the SAME borrowed Schur each rebuild and never owning/freeing
        # it. All solves must complete (STATISTICS SUMMARY) and the coarse handle
        # must actually be reused (the log line below, emitted at HYPREDRV_LOG_LEVEL=2).
        add_executable_test(elasticity_two_material_coarse_reuse_1proc elasticity 1
            ARGS --problem two-material --discretization mixed -n 13 5 5 --nu-top 0.4999
                 --coarse-schur -ns 3 -v 1
                 -i ${CMAKE_CURRENT_SOURCE_DIR}/mgr2-gmres-reuse.yml
            RUN_SERIAL
            REQUIRE_CONTAINS
                "Two-Material Bar"
                "reusing cached MGR coarsest nested Krylov handle"
                "STATISTICS SUMMARY"
        )
        hypredrv_append_test_environment(elasticity_two_material_coarse_reuse_1proc
            "HYPREDRV_LOG_LEVEL=2")
    endif()

    # Two-material bar with standard CG discretization (SPD; PCG + AMG preset).
    # Uses a moderate top Poisson ratio so the run converges quickly; needs no
    # recent-hypre features (no MGR / saddle point).
    add_executable_test(elasticity_two_material_standard_1proc elasticity 1
        ARGS --problem two-material --discretization standard -n 13 5 5
             --nu-top 0.45 -ns 1 -v 1
        RUN_SERIAL
        REQUIRE_CONTAINS
            "standard CG Q1"
            "STATISTICS SUMMARY"
    )

    # Two-material bar with B-bar (Q1-P0 mean-dilatation) discretization: a
    # displacement-only, locking-free SPD path (PCG + AMG). Moderate top Poisson
    # ratio so the condensed operator stays well conditioned and converges quickly.
    add_executable_test(elasticity_two_material_bbar_1proc elasticity 1
        ARGS --problem two-material --discretization bbar -n 13 5 5
             --nu-top 0.45 -ns 1 -v 1
        RUN_SERIAL
        REQUIRE_CONTAINS
            "B-bar Q1-P0 mean-dilatation"
            "STATISTICS SUMMARY"
    )
    add_executable_test(elasticity_two_material_bbar_4proc elasticity 4
        ARGS --problem two-material --discretization bbar -n 13 5 5 -P 2 2 1
             --nu-top 0.45 -ns 1 -v 1
        REQUIRE_CONTAINS
            "B-bar Q1-P0 mean-dilatation"
            "STATISTICS SUMMARY"
    )
endif()

# Install the elasticity example binary when examples are built
install(TARGETS elasticity
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
