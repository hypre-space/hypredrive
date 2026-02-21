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

function(register_unit_test_case target_name source_file)
    cmake_parse_arguments(UNIT_CASE "RUN_SERIAL" "" "" ${ARGN})

    setup_test_executable(${target_name} ${source_file})

    string(REGEX REPLACE "^test_" "" _unit_suffix "${target_name}")
    set(_unit_test_name "unit_test_${_unit_suffix}")
    add_unit_test(${_unit_test_name} ${target_name})

    if(UNIT_CASE_RUN_SERIAL)
        set_tests_properties(${_unit_test_name} PROPERTIES RUN_SERIAL TRUE)
    endif()
endfunction()

# Table-driven unit-test registration:
#   target|source|run_serial
set(_hypredrv_unit_test_cases
    "test_utils|test_utils.c|FALSE"
    "test_containers|test_containers.c|FALSE"
    "test_yaml|test_yaml.c|FALSE"
    "test_error|test_error.c|FALSE"
    "test_field|test_field.c|FALSE"
    "test_args|test_args.c|FALSE"
    "test_precon|test_precon.c|TRUE"
    "test_parser|test_parser.c|TRUE"
    "test_mgr_nested_yaml|test_mgr_nested_yaml.c|TRUE"
    "test_solver|test_solver.c|TRUE"
    "test_hypredrv|test_hypredrv.c|TRUE"
    "test_comp|test_comp.c|FALSE"
    "test_matrix|test_matrix.c|TRUE"
    "test_lsseq|test_lsseq.c|TRUE"
    "test_vector|test_vector.c|TRUE"
    "test_eigspec|test_eigspec.c|FALSE"
    "test_linsys|test_linsys.c|TRUE"
    "test_stats|test_stats.c|TRUE"
)

foreach(_case IN LISTS _hypredrv_unit_test_cases)
    string(REPLACE "|" ";" _parts "${_case}")
    list(GET _parts 0 _target)
    list(GET _parts 1 _source)
    list(GET _parts 2 _run_serial)

    if(_run_serial STREQUAL "TRUE")
        register_unit_test_case(${_target} ${_source} RUN_SERIAL)
    else()
        register_unit_test_case(${_target} ${_source})
    endif()
endforeach()
