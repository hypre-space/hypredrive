# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Remember the directory that contains this helper so functions can reference scripts
set(HYPREDRV_TESTING_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Function for adding tests
# Options:
#   NO_QUIET - if set, don't pass -q flag (shows full system info)
function(add_hypredrive_test test_name num_procs config_file)
    cmake_parse_arguments(TEST_OPTS "NO_QUIET" "" "" ${ARGN})

    # Automatically prepend "hypredrive_test_" to the test name
    set(full_test_name "hypredrive_test_${test_name}")

    # Build command arguments
    set(_cmd_args
        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
        -DTARGET_BIN=$<TARGET_FILE:hypredrive>
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPI_NUMPROCS=${num_procs}
        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/${config_file}
    )

    # Pass -q flag by default to skip system info (faster tests)
    if(NOT TEST_OPTS_NO_QUIET)
        list(APPEND _cmd_args "-DTARGET_ARGS=-q")
    endif()

    add_test(NAME ${full_test_name}
        COMMAND ${CMAKE_COMMAND} ${_cmd_args}
                -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(${full_test_name}
        PROPERTIES
        FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
    )
endfunction()

# Function for adding an integration test that verifies CLI overrides (-a/--args)
function(add_hypredrive_cli_test test_name num_procs config_file)
    # Optional:
    #   OVERRIDES        : full list of override tokens to use after -a (replaces defaults)
    #   EXTRA_ARGS       : additional CLI override pairs to append after -a
    #   REQUIRE_CONTAINS : additional substrings that must appear in output
    cmake_parse_arguments(TEST_OPTS "" "" "OVERRIDES;EXTRA_ARGS;REQUIRE_CONTAINS" ${ARGN})

    # Automatically prepend "hypredrive_test_" to the test name
    set(full_test_name "hypredrive_test_${test_name}")

    # CLI overrides via -a
    set(_cli_args
        -q
        -a
    )
    if(TEST_OPTS_OVERRIDES)
        list(APPEND _cli_args ${TEST_OPTS_OVERRIDES})
    else()
        # Default overrides used by the generic CLI integration test (ex1.yml)
        list(APPEND _cli_args
            --solver:pcg:max_iter 5
            --preconditioner:amg:print_level 0
        )
    endif()
    if(TEST_OPTS_EXTRA_ARGS)
        list(APPEND _cli_args ${TEST_OPTS_EXTRA_ARGS})
    endif()

    # Encode args for the run script (uses '|' as separator)
    string(JOIN "|" _target_args ${_cli_args})

    # Substrings that must appear in output (prove -a reached InputArgsParse)
    set(_must_contain "")
    if(NOT TEST_OPTS_OVERRIDES)
        list(APPEND _must_contain
            "solver:"
            "pcg:"
            "max_iter: 5"
            "preconditioner:"
            "amg:"
            "print_level: 0"
        )
    endif()
    if(TEST_OPTS_REQUIRE_CONTAINS)
        list(APPEND _must_contain ${TEST_OPTS_REQUIRE_CONTAINS})
    endif()
    string(JOIN "|" _require_contains ${_must_contain})

    add_test(NAME ${full_test_name}
        COMMAND ${CMAKE_COMMAND}
                -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                -DTARGET_BIN=$<TARGET_FILE:hypredrive>
                -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                -DMPI_NUMPROCS=${num_procs}
                -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/${config_file}
                -DTARGET_ARGS:STRING=${_target_args}
                -DREQUIRE_CONTAINS:STRING=${_require_contains}
                -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(${full_test_name}
        PROPERTIES
        FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
    )
endfunction()

# Function for adding tests for standalone executable drivers
function(add_executable_test test_name target num_procs)
    cmake_parse_arguments(EXEC_TEST
        "RUN_SERIAL"
        "FAIL_REGULAR_EXPRESSION;WORKING_DIRECTORY"
        "ARGS"
        ${ARGN}
    )

    # Default values
    if(NOT DEFINED EXEC_TEST_FAIL_REGULAR_EXPRESSION)
        set(EXEC_TEST_FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure")
    endif()
    if(NOT DEFINED EXEC_TEST_WORKING_DIRECTORY)
        set(EXEC_TEST_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    set(_driver_command
        ${CMAKE_COMMAND}
            -DLAUNCH_DIR=${EXEC_TEST_WORKING_DIRECTORY}
            -DTARGET_BIN=$<TARGET_FILE:${target}>
            -DMPIEXEC=${MPIEXEC_EXECUTABLE}
            -DMPI_NUMPROCS=${num_procs}
            -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
            -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
            -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
    )
    if(EXEC_TEST_ARGS)
        string(JOIN "|" _driver_args ${EXEC_TEST_ARGS})
        list(APPEND _driver_command "-DTARGET_ARGS:STRING=${_driver_args}")
    endif()

    add_test(NAME ${test_name}
        COMMAND ${_driver_command} -P ${HYPREDRV_TESTING_DIR}/HYPREDRV_RunScript.cmake
    )

    set_tests_properties(${test_name}
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "${EXEC_TEST_FAIL_REGULAR_EXPRESSION}"
    )

    if(EXEC_TEST_RUN_SERIAL)
        set_tests_properties(${test_name}
            PROPERTIES
                RUN_SERIAL TRUE
        )
    endif()
endfunction()

# Function for adding tests with output verification
function(add_hypredrive_test_with_output test_name num_procs config_file example_id)
    # Create output file path (capturing via CTest output if needed)
    set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/test_output_${test_name}.txt")
    set(REFERENCE_FILE "${CMAKE_SOURCE_DIR}/examples/refOutput/ex${example_id}.txt")

    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
                -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                -DTARGET_BIN=$<TARGET_FILE:hypredrive>
                -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                -DMPI_NUMPROCS=${num_procs}
                -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/${config_file}
                -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # Optional output comparison if script and reference exist
    find_program(COMPARE_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/compare_output.sh")
    if(COMPARE_SCRIPT AND EXISTS ${REFERENCE_FILE})
        add_test(NAME ${test_name}_output
            COMMAND ${COMPARE_SCRIPT} ${OUTPUT_FILE} ${REFERENCE_FILE}
        )
        set_tests_properties(${test_name}_output
            PROPERTIES
            DEPENDS ${test_name}
            SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
        )
    endif()
endfunction()

# Only register tests when included from the main CMakeLists.txt
# (not when included from subdirectories like examples)
if(HYPREDRV_ENABLE_TESTING AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    # Must be called before add_subdirectory(tests) so that add_test() calls work
    enable_testing()

    # Add tests subfolder (contains unit tests that use add_test())
    add_subdirectory(tests)

    # Regression: CLI override of num_repetitions should produce contiguous stats entries.
    set(_cli_reps5_require_contains
        "num_repetitions: 5"
        "|          0 |"
        "|          1 |"
        "|          2 |"
        "|          3 |"
        "|          4 |"
    )

    # Regression: exercise CLI overrides for ex7 (multiple linear systems + repetitions).
    set(_cli_ex7_reps4_ls4_require_contains
        "num_repetitions: 4"
        "last_suffix: 4"
        "Solving linear system #4"
    )

    # Add tests (ex1_1proc shows full system info, others use -q for faster runs)
    add_hypredrive_test(ex1_1proc  1 ex1.yml NO_QUIET)
    add_hypredrive_cli_test(ex1_cli 1 ex1.yml)
    add_hypredrive_cli_test(ex1_cli_reps5 1 ex1.yml
        EXTRA_ARGS --general:num_repetitions 5
        REQUIRE_CONTAINS ${_cli_reps5_require_contains}
    )
    add_hypredrive_test(ex1a_1proc 1 ex1a.yml)
    add_hypredrive_test(ex1b_1proc 1 ex1b.yml)
    add_hypredrive_test(ex1c_1proc 1 ex1c.yml)
    add_hypredrive_test(ex1d_1proc 1 ex1d.yml)
    add_hypredrive_test(ex1_preset_1proc 1 ex1-preset.yml)
    add_hypredrive_test(ex2_4proc  4 ex2.yml)
    add_hypredrive_test(ex3_1proc  1 ex3.yml)
    add_hypredrive_test(ex4_4proc  4 ex4.yml)
    add_hypredrive_test(ex5_1proc  1 ex5.yml)
    if (HYPREDRV_ENABLE_EIGSPEC)
        add_hypredrive_test(ex6_1proc 1 ex6.yml)
    endif()
    add_hypredrive_test(ex7_1proc   1 ex7.yml)
    add_hypredrive_cli_test(ex7_cli_reps4_ls4 1 ex7.yml
        OVERRIDES --general:num_repetitions 4 --linear_system:last_suffix 4
        REQUIRE_CONTAINS ${_cli_ex7_reps4_ls4_require_contains}
    )
    add_hypredrive_test(ex8_1proc   1 ex8.yml)
    add_hypredrive_test(ex8a_4proc  4 ex8-multi-1.yml)
    add_hypredrive_test(ex8b_4proc  4 ex8-multi-2.yml)

    # Test main.c help/usage/error branches
    # Note: --help exits with 0, so we need to allow that
    add_executable_test(hypredrive_help hypredrive 1 ARGS "--help" FAIL_REGULAR_EXPRESSION "^$")
    add_executable_test(hypredrive_help_short hypredrive 1 ARGS "-h" FAIL_REGULAR_EXPRESSION "^$")
    add_executable_test(hypredrive_no_args hypredrive 1 ARGS "" FAIL_REGULAR_EXPRESSION "^$")
    set_tests_properties(hypredrive_no_args PROPERTIES WILL_FAIL TRUE)

    # Exercise the long-form quiet flag parsing ("--quiet")
    add_executable_test(hypredrive_quiet_longflag hypredrive 1
        ARGS "--quiet" "examples/ex1.yml"
        FAIL_REGULAR_EXPRESSION "^$"
    )

    # Exercise config-file detection when override args are present
    add_executable_test(hypredrive_cli_extra hypredrive 1
        ARGS "examples/ex1.yml" "--args" "--solver:pcg:max_iter" "5"
        FAIL_REGULAR_EXPRESSION "^$"
    )
endif()
