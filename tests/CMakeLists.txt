# Unit tests for hypredrive
# Helper function to set up test executable with proper RPATH
# Similar to hypredrive executable - link directly to both HYPREDRV and HYPRE
function(setup_test_executable target_name source_file)
    add_executable(${target_name} ${source_file})

    # Provide an absolute source dir path to tests (avoids dependence on CTest
    # working directory and prevents relative-path flakiness across tests).
    target_compile_definitions(${target_name} PRIVATE HYPREDRIVE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    # Link to both HYPREDRV and HYPRE, same as hypredrive executable does
    target_link_libraries(${target_name} PRIVATE HYPREDRV::HYPREDRV)

    # If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
    if(HYPREDRV_ENABLE_CALIPER)
        enable_language(CXX OPTIONAL)
        if(CMAKE_CXX_COMPILER)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_link_libraries(${target_name} PRIVATE stdc++)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # Clang can use either libstdc++ or libc++
                # Default to stdc++ on Linux
                if(UNIX AND NOT APPLE)
                    target_link_libraries(${target_name} PRIVATE stdc++)
                else()
                    target_link_libraries(${target_name} PRIVATE c++)
                endif()
            endif()
        endif()
    endif()

    # Apply sanitizer link flags if sanitizers are enabled
    # This ensures the sanitizer runtime libraries are linked
    get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
    get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
    if(_sanitizer_enabled AND _sanitizer_flags)
        foreach(flag IN LISTS _sanitizer_flags)
            target_link_options(${target_name} PRIVATE ${flag})
        endforeach()
    endif()

    # Add an RPATH to the lib directory for MacOS (libraries are in lib/)
    if(APPLE)
        target_link_options(${target_name} PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
    endif()
endfunction()

# Add a unit test with stable working directory and robust failure detection.
# Some MPI abort paths can (incorrectly) return exit code 0 depending on the MPI
# implementation; catch these via output regex as well.
function(add_unit_test test_name target_name)
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(${test_name}
        PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort\\("
            LABELS "unit"
    )

    # When running with sanitizers, suppress known external leaks (MPI runtime, etc.)
    # so unit tests fail only on leaks in hypredrive itself.
    get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
    if(_sanitizer_enabled AND EXISTS "${CMAKE_SOURCE_DIR}/.github/lsan.supp")
        set_tests_properties(${test_name}
            PROPERTIES
                ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/.github/lsan.supp"
        )
    endif()
endfunction()

# Test utilities functions
setup_test_executable(test_utils test_utils.c)
add_unit_test(unit_test_utils test_utils)

# Test container functions
setup_test_executable(test_containers test_containers.c)
add_unit_test(unit_test_containers test_containers)

# Test YAML parsing functions
setup_test_executable(test_yaml test_yaml.c)
add_unit_test(unit_test_yaml test_yaml)

# Test error handling functions
setup_test_executable(test_error test_error.c)
add_unit_test(unit_test_error test_error)

# Field parsing helpers
setup_test_executable(test_field test_field.c)
add_unit_test(unit_test_field test_field)

# Argument parsing and YAML validation tests
setup_test_executable(test_args test_args.c)
add_unit_test(unit_test_args test_args)

# Preconditioner configuration helpers (includes AMG argument tests)
setup_test_executable(test_precon test_precon.c)
add_unit_test(unit_test_precon test_precon)
set_tests_properties(unit_test_precon PROPERTIES RUN_SERIAL TRUE)

# Parser coverage tests
setup_test_executable(test_parser test_parser.c)
add_unit_test(unit_test_parser test_parser)
set_tests_properties(unit_test_parser PROPERTIES RUN_SERIAL TRUE)

# Nested MGR YAML regression
setup_test_executable(test_mgr_nested_yaml test_mgr_nested_yaml.c)
add_unit_test(unit_test_mgr_nested_yaml test_mgr_nested_yaml)
set_tests_properties(unit_test_mgr_nested_yaml PROPERTIES RUN_SERIAL TRUE)

# Solver argument helpers, dispatch, and integration tests (all merged)
setup_test_executable(test_solver test_solver.c)
add_unit_test(unit_test_solver test_solver)
set_tests_properties(unit_test_solver PROPERTIES RUN_SERIAL TRUE)

# HYPREDRV API smoke tests
setup_test_executable(test_hypredrv test_hypredrv.c)
add_unit_test(unit_test_hypredrv test_hypredrv)
set_tests_properties(unit_test_hypredrv PROPERTIES RUN_SERIAL TRUE)

# Matrix IO helpers
setup_test_executable(test_matrix test_matrix.c)
add_unit_test(unit_test_matrix test_matrix)
set_tests_properties(unit_test_matrix PROPERTIES RUN_SERIAL TRUE)

# Vector IO and MPI-aware containers utilities
setup_test_executable(test_vector test_vector.c)
add_unit_test(unit_test_vector test_vector)
set_tests_properties(unit_test_vector PROPERTIES RUN_SERIAL TRUE)

# EigSpec configuration helpers
setup_test_executable(test_eigspec test_eigspec.c)
add_unit_test(unit_test_eigspec test_eigspec)

# Linear system helpers
setup_test_executable(test_linsys test_linsys.c)
add_unit_test(unit_test_linsys test_linsys)
set_tests_properties(unit_test_linsys PROPERTIES RUN_SERIAL TRUE)

# Stats helpers
setup_test_executable(test_stats test_stats.c)
add_unit_test(unit_test_stats test_stats)
set_tests_properties(unit_test_stats PROPERTIES RUN_SERIAL TRUE)
