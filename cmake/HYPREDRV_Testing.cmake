# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Remember the directory that contains this helper so functions can reference scripts
set(HYPREDRV_TESTING_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Some regression tests require newer Hypre APIs. We gate those tests at CMake
# configure time by probing Hypre version macros from headers.
include(CheckCSourceCompiles)

function(_hypredrv_collect_plain_include_dirs out_var)
    set(_plain_include_dirs "")
    foreach(_inc_dir IN LISTS ARGN)
        if(NOT _inc_dir)
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
            list(APPEND _plain_include_dirs "${CMAKE_MATCH_1}")
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<INSTALL_INTERFACE:")
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<TARGET_PROPERTY:")
            continue()
        endif()
        list(APPEND _plain_include_dirs "${_inc_dir}")
    endforeach()

    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/hypre-build/HYPRE_config.h")
        list(APPEND _plain_include_dirs "${CMAKE_BINARY_DIR}/_deps/hypre-build")
    endif()
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/HYPRE.h")
        list(APPEND _plain_include_dirs "${CMAKE_BINARY_DIR}/_deps/hypre-src/src")
    endif()

    list(REMOVE_DUPLICATES _plain_include_dirs)
    set(${out_var} "${_plain_include_dirs}" PARENT_SCOPE)
endfunction()

# Append shared runtime environment settings (and optional extras) to a CTest
# test without clobbering existing ENVIRONMENT properties.
function(hypredrv_append_test_environment test_name)
    set(_env_list "")
    get_test_property(${test_name} ENVIRONMENT _existing_env)
    if(_existing_env AND NOT _existing_env STREQUAL "NOTFOUND")
        list(APPEND _env_list ${_existing_env})
    endif()
    if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
        list(APPEND _env_list "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    endif()
    get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
    if(_sanitizer_enabled AND EXISTS "${CMAKE_SOURCE_DIR}/.github/lsan.supp")
        list(APPEND _env_list
            "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/.github/lsan.supp")
    endif()
    if(ARGN)
        list(APPEND _env_list ${ARGN})
    endif()
    if(_env_list)
        list(REMOVE_DUPLICATES _env_list)
        set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${_env_list}")
    endif()
endfunction()

# Function for adding tests
# Options:
#   NO_QUIET - if set, don't pass -q flag (shows full system info)
set(HYPREDRV_FAIL_REGEX_DEFAULT
    "HYPREDRIVE Failure!!!|BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES|Segmentation fault|Abort\\("
)

set(HYPREDRV_GPU_PROBLEM_SIZE_MULTIPLIER 1)
if(HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP)
    set(HYPREDRV_GPU_PROBLEM_SIZE_MULTIPLIER 5)
endif()

set(HYPREDRV_GPU_DISABLED_TESTS
    laplacian_7pt_test_4proc
    laplacian_19pt_test_4proc
    laplacian_27pt_test_4proc
    laplacian_125pt_test_4proc
    elasticity_test_4proc
    heatflow_test_4proc
    lidcavity_test_4proc
    lidcavity_test_mgr_1proc
    lidcavity_test_print_1proc
    lidcavity_test_adaptive_reuse_1proc
    lidcavity_test_mgr_4proc
)

get_property(_hypredrv_gpu_test_policy_reported GLOBAL
    PROPERTY HYPREDRV_GPU_TEST_POLICY_REPORTED
)
if((HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP) AND NOT _hypredrv_gpu_test_policy_reported)
    if(HYPREDRV_ENABLE_ALL_TESTS)
        message(STATUS "GPU test policy: all tests enabled (GPU skip list overridden)")
    else()
        list(JOIN HYPREDRV_GPU_DISABLED_TESTS ", " _hypredrv_gpu_disabled_tests_msg)
        message(STATUS
            "GPU test policy: disabling selected tests by default: "
            "${_hypredrv_gpu_disabled_tests_msg}"
        )
        unset(_hypredrv_gpu_disabled_tests_msg)
    endif()
    set_property(GLOBAL PROPERTY HYPREDRV_GPU_TEST_POLICY_REPORTED TRUE)
endif()
unset(_hypredrv_gpu_test_policy_reported)

function(hypredrv_maybe_disable_gpu_test test_name)
    if(NOT HYPREDRV_ENABLE_ALL_TESTS AND
       (HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP) AND
       test_name IN_LIST HYPREDRV_GPU_DISABLED_TESTS)
        set_tests_properties(${test_name} PROPERTIES DISABLED TRUE)
    endif()
endfunction()

function(hypredrv_scale_problem_size_args out_var)
    set(_scaled_args "")
    set(_scaling_n_dims FALSE)

    foreach(_arg IN LISTS ARGN)
        if(_arg STREQUAL "-n")
            set(_scaling_n_dims TRUE)
            list(APPEND _scaled_args "${_arg}")
        elseif(_scaling_n_dims)
            if(_arg MATCHES "^[0-9]+$")
                math(EXPR _scaled_dim
                    "${_arg} * ${HYPREDRV_GPU_PROBLEM_SIZE_MULTIPLIER}"
                )
                list(APPEND _scaled_args "${_scaled_dim}")
            else()
                set(_scaling_n_dims FALSE)
                list(APPEND _scaled_args "${_arg}")
            endif()
        else()
            list(APPEND _scaled_args "${_arg}")
        endif()
    endforeach()

    set(${out_var} "${_scaled_args}" PARENT_SCOPE)
endfunction()

function(add_hypredrive_test test_name num_procs config_file)
    cmake_parse_arguments(TEST_OPTS "NO_QUIET" "" "" ${ARGN})

    # Automatically prepend "hypredrive_test_" to the test name
    set(full_test_name "hypredrive_test_${test_name}")

    # Build command arguments
    set(_cmd_args
        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
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
        FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
        LABELS "integration;hypredrive"
    )
    hypredrv_maybe_disable_gpu_test(${full_test_name})
    hypredrv_append_test_environment(${full_test_name})
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
                -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
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
        FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
        LABELS "integration;hypredrive"
    )
    hypredrv_maybe_disable_gpu_test(${full_test_name})
    hypredrv_append_test_environment(${full_test_name})
endfunction()

# Function for adding tests for standalone executable drivers
function(add_executable_test test_name target num_procs)
    cmake_parse_arguments(EXEC_TEST
        "RUN_SERIAL"
        "FAIL_REGULAR_EXPRESSION;WORKING_DIRECTORY"
        "ARGS;REQUIRE_CONTAINS;REQUIRE_PATHS"
        ${ARGN}
    )

    # Default values
    if(NOT DEFINED EXEC_TEST_FAIL_REGULAR_EXPRESSION)
        set(EXEC_TEST_FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}")
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
        hypredrv_scale_problem_size_args(_scaled_exec_test_args ${EXEC_TEST_ARGS})
        string(JOIN "|" _driver_args ${_scaled_exec_test_args})
        list(APPEND _driver_command "-DTARGET_ARGS:STRING=${_driver_args}")
    endif()
    if(EXEC_TEST_REQUIRE_CONTAINS)
        string(JOIN "|" _require_contains ${EXEC_TEST_REQUIRE_CONTAINS})
        list(APPEND _driver_command "-DREQUIRE_CONTAINS:STRING=${_require_contains}")
    endif()
    if(EXEC_TEST_REQUIRE_PATHS)
        string(JOIN "|" _require_paths ${EXEC_TEST_REQUIRE_PATHS})
        list(APPEND _driver_command "-DREQUIRE_PATHS:STRING=${_require_paths}")
    endif()

    add_test(NAME ${test_name}
        COMMAND ${_driver_command} -P ${HYPREDRV_TESTING_DIR}/HYPREDRV_RunScript.cmake
    )

    set_tests_properties(${test_name}
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "${EXEC_TEST_FAIL_REGULAR_EXPRESSION}"
    )
    hypredrv_maybe_disable_gpu_test(${test_name})
    hypredrv_append_test_environment(${test_name})

    if(target STREQUAL "hypredrive-cli")
        set_tests_properties(${test_name}
            PROPERTIES
                LABELS "hypredrive"
        )
    endif()

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
                -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                -DMPI_NUMPROCS=${num_procs}
                -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/${config_file}
                -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(${test_name}
        PROPERTIES
        FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
        LABELS "integration;hypredrive"
    )
    hypredrv_maybe_disable_gpu_test(${test_name})
    hypredrv_append_test_environment(${test_name})

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
            LABELS "hypredrive"
        )
    endif()
endfunction()

function(hypredrv_check_hypre_symbol symbol)
    set(_hypredrv_out_var "HYPREDRV_HAVE_${symbol}")
    set(_hypredrv_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
    _hypredrv_collect_plain_include_dirs(_hypredrv_required_includes
        ${HYPRE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
        ${MPI_C_INCLUDE_DIRS}
    )
    set(CMAKE_REQUIRED_INCLUDES
        ${_hypredrv_required_includes}
    )
    check_c_source_compiles("
      #include \"HYPRE_config.h\"
      #ifndef ${symbol}
      #error \"${symbol} not defined\"
      #endif
      int main(void) { return 0; }
    " ${_hypredrv_out_var})
    set(CMAKE_REQUIRED_INCLUDES "${_hypredrv_saved_includes}")
    unset(_hypredrv_saved_includes)
    unset(_hypredrv_required_includes)
    unset(_hypredrv_out_var)
endfunction()

function(hypredrv_check_hypre_version release develop)
    set(_hypredrv_out_var "HYPREDRV_HAVE_HYPRE_${release}_DEV${develop}")
    # Determine hypre version checks for selecting which tests to run.
    #set(CMAKE_MESSAGE_LOG_LEVEL DEBUG) # or TRACE for maximum noise
    # Include Hypre headers (from find_package) and HypreDrive headers (for utils.h)
    set(_hypredrv_saved_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_hypredrv_saved_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
    _hypredrv_collect_plain_include_dirs(_hypredrv_required_includes
        ${HYPRE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
        ${MPI_C_INCLUDE_DIRS}
    )
    set(CMAKE_REQUIRED_INCLUDES
        ${_hypredrv_required_includes}
    )
    if(DEFINED HYPREDRV_HYPRE_RELEASE_NUMBER)
        list(APPEND CMAKE_REQUIRED_DEFINITIONS
            "-DHYPREDRV_HYPRE_RELEASE_NUMBER=${HYPREDRV_HYPRE_RELEASE_NUMBER}"
        )
    endif()
    if(DEFINED HYPREDRV_HYPRE_DEVELOP_NUMBER)
        list(APPEND CMAKE_REQUIRED_DEFINITIONS
            "-DHYPREDRV_HYPRE_DEVELOP_NUMBER=${HYPREDRV_HYPRE_DEVELOP_NUMBER}"
        )
    endif()
    check_c_source_compiles("
      #include \"HYPRE_config.h\"
      #define HYPRE_SEQUENTIAL
      #include \"internal/utils.h\"
      #if !HYPRE_CHECK_MIN_VERSION(${release}, ${develop})
      #error \"need HYPRE >= ${release} + develop >= ${develop}\"
      #endif
      int main(void) { return 0; }
    " ${_hypredrv_out_var})
    set(CMAKE_REQUIRED_INCLUDES "${_hypredrv_saved_includes}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_hypredrv_saved_definitions}")
    unset(_hypredrv_saved_includes)
    unset(_hypredrv_saved_definitions)
    unset(_hypredrv_required_includes)
    unset(_hypredrv_out_var)
endfunction()

# Only register tests when included from the main CMakeLists.txt
# (not when included from subdirectories like examples)
if(HYPREDRV_ENABLE_TESTING AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    # Define hypre version checks for selecting which tests to run.
    hypredrv_check_hypre_version(21900 0)
    hypredrv_check_hypre_version(22500 0)
    hypredrv_check_hypre_version(22900 0)
    hypredrv_check_hypre_version(23000 0)
    hypredrv_check_hypre_version(23300 0)
    hypredrv_check_hypre_version(30000 0)
    hypredrv_check_hypre_version(30100 5)
    hypredrv_check_hypre_version(30100 38)

    # Check for optional hypre features used to gate tests.
    hypredrv_check_hypre_symbol(HYPRE_DEVELOP_NUMBER)
    hypredrv_check_hypre_symbol(HYPRE_USING_DSUPERLU)

    # Must be called before add_subdirectory(tests) so that add_test() calls work
    enable_testing()

    # Add tests subfolder (contains unit tests that use add_test())
    add_subdirectory(tests)

    # Regression: CLI override of num_repetitions should produce contiguous stats entries.
    set(_cli_reps5_require_contains
        "num_repetitions: 5"
        "|      0 |"
        "|      1 |"
        "|      2 |"
        "|      3 |"
        "|      4 |"
    )

    # Regression: exercise CLI overrides for ex7 (multiple linear systems + repetitions).
    set(_cli_ex7_reps4_ls4_require_contains
        "num_repetitions: 4"
        "last_suffix: 4"
        "Solving linear system #4"
    )

    if (HYPREDRV_HAVE_HYPRE_21900_DEV0)
        # Add tests (ex1_1proc shows full system info, others use -q for faster runs)
        add_hypredrive_test(ex1_1proc 1 ex1.yml NO_QUIET)
        set(_hypredrv_base_examples
            "ex1a_1proc|1|ex1a.yml"
            "ex1_preset|1|ex1-preset.yml"
            "ex2_4proc|4|ex2.yml"
        )
        foreach(_case IN LISTS _hypredrv_base_examples)
            string(REPLACE "|" ";" _parts "${_case}")
            list(GET _parts 0 _name)
            list(GET _parts 1 _nprocs)
            list(GET _parts 2 _config)
            add_hypredrive_test(${_name} ${_nprocs} ${_config})
        endforeach()
        unset(_hypredrv_base_examples)

        add_hypredrive_cli_test(ex1_cli 1 ex1.yml)
        add_hypredrive_cli_test(ex1_cli_reps5 1 ex1.yml
            EXTRA_ARGS --general:num_repetitions 5
            REQUIRE_CONTAINS ${_cli_reps5_require_contains}
        )
        if(HYPREDRV_ENABLE_COMPRESSION AND TARGET hypredrive-lsseq AND HYPREDRV_HAVE_HYPRE_30000_DEV0)
            add_test(NAME hypredrive_test_ex7_sequence_pack
                COMMAND ${CMAKE_COMMAND}
                        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                        -DPACKER_BIN=$<TARGET_FILE:hypredrive-lsseq>
                        -DSEQ_OUTPUT=${CMAKE_BINARY_DIR}/poromech2k_lsseq_test.bin
                        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                        -DMPI_NUMPROCS=1
                        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/ex7.yml
                        "-DREQUIRE_CONTAINS:STRING=Solving linear system #2"
                        -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_PackAndRunScript.cmake
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(hypredrive_test_ex7_sequence_pack
                PROPERTIES
                FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                LABELS "integration;hypredrive"
            )
            hypredrv_maybe_disable_gpu_test(hypredrive_test_ex7_sequence_pack)
            hypredrv_append_test_environment(hypredrive_test_ex7_sequence_pack)
        endif()
        if (HYPREDRV_HAVE_HYPRE_23000_DEV0)
            add_hypredrive_test(ex1b_1proc 1 ex1b.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_22500_DEV0)
            add_hypredrive_test(ex1c_1proc 1 ex1c.yml)
            add_hypredrive_test(ex1d_1proc 1 ex1d.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_23300_DEV0)
            add_hypredrive_test(ex3_1proc 1 ex3.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_30000_DEV0)
            add_hypredrive_cli_test(ex7_cli_tagres 1 ex7-tagged-gmres.yml
                OVERRIDES
                    --solver:gmres:print_level 4
                REQUIRE_CONTAINS
                    "Initial L2 norm of r0"
                    "Initial L2 norm of r1"
            )
            add_hypredrive_cli_test(ex7_cli_tagerr_randsol 1 ex7-tagged-gmres.yml
                OVERRIDES
                    --solver:gmres:print_level 8
                    --linear_system:rhs_mode randsol
                REQUIRE_CONTAINS
                    "rhs_mode: randsol"
                    "Initial L2 norm of e0"
                    "Final L2 norm of e0"
            )
            add_hypredrive_cli_test(ex7_cli_tagerr_scale 1 ex7-tagged-gmres.yml
                OVERRIDES
                    --solver:gmres:print_level 8
                    --linear_system:rhs_mode randsol
                    --solver:scaling:enabled true
                    --solver:scaling:type dofmap_mag
                REQUIRE_CONTAINS
                    "rhs_mode: randsol"
                    "type: dofmap_mag"
                    "Initial L2 norm of e0"
                    "Final L2 norm of e0"
            )
            add_hypredrive_cli_test(ex7_cli_stats2 1 ex7.yml
                OVERRIDES
                    --general:statistics 2
                    --linear_system:last_suffix 4
                REQUIRE_CONTAINS
                    "statistics: 2"
                    "STATISTICS SUMMARY:"
                    "|   Min. |"
                    "|   Max. |"
                    "|   Avg. |"
                    "|   Std. |"
                    "|  Total |"
            )
        endif()
        if (HYPREDRV_HAVE_HYPRE_30100_DEV5)
            set(_hypredrv_mgr_examples
                "ex3_nested_1|1|ex3-mgr_Frelax_gmres.yml"
                "ex3_nested_2|1|ex3-mgr_coarse_gmres_amg.yml"
                "ex4_4proc|4|ex4.yml"
            )
            foreach(_case IN LISTS _hypredrv_mgr_examples)
                string(REPLACE "|" ";" _parts "${_case}")
                list(GET _parts 0 _name)
                list(GET _parts 1 _nprocs)
                list(GET _parts 2 _config)
                add_hypredrive_test(${_name} ${_nprocs} ${_config})
            endforeach()
            unset(_hypredrv_mgr_examples)
            # TODO: the following requires a hypre fix
            # add_hypredrive_cli_test(ex4_cli_mgr_print_level_4proc 4 ex4.yml
            #     OVERRIDES
            #         --preconditioner:mgr:print_level 1
            #     REQUIRE_CONTAINS
            #         "MGR SETUP PARAMETERS:"
            # )
            add_hypredrive_cli_test(ex4_cli_mgr_g_ilu 1 ex4.yml
                OVERRIDES
                    --preconditioner:mgr:print_level 1
                    --preconditioner:mgr:level:1:g_relaxation none
                    --preconditioner:mgr:level:0:g_relaxation ilu
                REQUIRE_CONTAINS
                    "g_relaxation: ilu"
                    "[0, 1]     BJ-ILU0         Jacobi"
            )
            if(NOT HYPREDRV_HAVE_HYPRE_DEVELOP_NUMBER)
                set(_hypredrv_ex4_cli_mgr_g_amg_expect "Unknown         Jacobi")
            else()
                set(_hypredrv_ex4_cli_mgr_g_amg_expect "User AMG         Jacobi")
            endif()
            add_hypredrive_cli_test(ex4_cli_mgr_g_amg 1 ex4.yml
                OVERRIDES
                    --preconditioner:mgr:print_level 1
                    --preconditioner:mgr:level:1:g_relaxation none
                    --preconditioner:mgr:level:0:g_relaxation amg
                REQUIRE_CONTAINS
                    "g_relaxation: amg"
                    "${_hypredrv_ex4_cli_mgr_g_amg_expect}"
            )
            unset(_hypredrv_ex4_cli_mgr_g_amg_expect)
            add_hypredrive_cli_test(ex4_cli_mgr_f_ilu 1 ex4.yml
                OVERRIDES
                    --preconditioner:mgr:print_level 1
                    --preconditioner:mgr:level:1:g_relaxation none
                    --preconditioner:mgr:level:0:f_relaxation ilu
                REQUIRE_CONTAINS
                    "f_relaxation: ilu"
                    "[0, 1]     Unknown        BJ-ILU0"
            )
            add_hypredrive_cli_test(ex4_cli_mgr_f_amg 1 ex4.yml
                OVERRIDES
                    --preconditioner:mgr:print_level 1
                    --preconditioner:mgr:level:1:g_relaxation none
                    --preconditioner:mgr:level:0:f_relaxation amg
                REQUIRE_CONTAINS
                    "f_relaxation: amg"
                    "User AMG"
                    "Strength Threshold = 0.250000"
                    "Coarsening type = "
            )
        endif()
        if (HYPREDRV_HAVE_HYPRE_23300_DEV0)
            add_hypredrive_test(ex5_1proc 1 ex5.yml)
        endif()
        if (HYPREDRV_ENABLE_EIGSPEC)
            add_hypredrive_test(ex6_1proc 1 ex6.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_23300_DEV0)
            add_hypredrive_test(ex7_1proc 1 ex7.yml)
            add_hypredrive_cli_test(ex7_cli_reps4_ls4 1 ex7.yml
                OVERRIDES --general:num_repetitions 4 --linear_system:last_suffix 4
                REQUIRE_CONTAINS ${_cli_ex7_reps4_ls4_require_contains}
            )
            if(HYPREDRV_ENABLE_EXPERIMENTAL)
                # MGR cycle type tests (5 linear systems each)
                foreach(_cycle_case IN ITEMS
                    "v;ex7-mgr-cycle-v.yml"
                    "v01;ex7-mgr-cycle-v01.yml"
                    "v11;ex7-mgr-cycle-v11.yml"
                    "w;ex7-mgr-cycle-w.yml"
                    "w11;ex7-mgr-cycle-w11.yml"
                )
                    string(REPLACE ";" "|" _parts "${_cycle_case}")
                    string(REPLACE "|" ";" _parts "${_cycle_case}")
                    list(GET _parts 0 _suffix)
                    list(GET _parts 1 _cfg)
                    set(_tname "hypredrive_test_ex7_mgr_cycle_${_suffix}_1proc")
                    add_test(NAME ${_tname}
                        COMMAND ${CMAKE_COMMAND}
                                -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                                -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                                -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                                -DMPI_NUMPROCS=1
                                -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                                -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                                -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                                -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/${_cfg}
                                "-DTARGET_ARGS:STRING=-q"
                                "-DREQUIRE_CONTAINS:STRING=Solving linear system #4"
                                -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    )
                    set_tests_properties(${_tname}
                        PROPERTIES
                        FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                        LABELS "integration;hypredrive"
                    )
                    hypredrv_maybe_disable_gpu_test(${_tname})
                    hypredrv_append_test_environment(${_tname})
                endforeach()
                unset(_cycle_case)
                unset(_suffix)
                unset(_cfg)
                unset(_tname)
            endif()
        endif()
        if(HYPREDRV_HAVE_HYPRE_USING_DSUPERLU)
            add_hypredrive_test(ex7_mgr_frelax_spdirect_1proc 1
                ex7-mgr-frelax-spdirect.yml)
            add_hypredrive_test(ex7_mgr_grelax_spdirect_1proc 1
                ex7-mgr-grelax-spdirect.yml)
            add_hypredrive_test(ex7_mgr_coarsest_spdirect_1proc 1
                ex7-mgr-coarsest-spdirect.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_30100_DEV5)
            add_hypredrive_test(ex7_nested_mgr_1proc 1 ex7-nested-mgr.yml)
            add_hypredrive_test(ex7_nested_krylov_mgr_1proc 1
                ex7-nested-krylov-mgr.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_30000_DEV0)
            add_hypredrive_cli_test(ex7_cli_dofmap_scaling 1 ex7.yml
                OVERRIDES --solver:scaling:enabled true --solver:scaling:type dofmap_mag
                REQUIRE_CONTAINS "scaling:" "enabled:" "type: dofmap_mag"
            )
            add_hypredrive_test(ex7_custom_scaling 1 ex7-custom-scaling.yml)
        endif()
        if (HYPREDRV_HAVE_HYPRE_23000_DEV0)
            set(_hypredrv_ex8_examples
                "ex8_1proc|1|ex8.yml"
                "ex8a_4proc|4|ex8-multi-1.yml"
                "ex8b_4proc|4|ex8-multi-2.yml"
            )
            foreach(_case IN LISTS _hypredrv_ex8_examples)
                string(REPLACE "|" ";" _parts "${_case}")
                list(GET _parts 0 _name)
                list(GET _parts 1 _nprocs)
                list(GET _parts 2 _config)
                add_hypredrive_test(${_name} ${_nprocs} ${_config})
            endforeach()
            unset(_hypredrv_ex8_examples)
        endif()
        if (HYPREDRV_HAVE_HYPRE_30100_DEV38)
            add_test(NAME hypredrive_test_ex7_mgr_frelax_reuse_1proc
                COMMAND ${CMAKE_COMMAND}
                        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                        -DMPI_NUMPROCS=1
                        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/ex7-mgr-frelax-reuse.yml
                        "-DTARGET_ARGS:STRING=-q"
                        "-DREQUIRE_CONTAINS:STRING=preserving cached MGR handles across destroy|cached MGR handles before create|Solving linear system #24"
                        -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(hypredrive_test_ex7_mgr_frelax_reuse_1proc
                PROPERTIES
                FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                LABELS "integration;hypredrive"
            )
            hypredrv_maybe_disable_gpu_test(hypredrive_test_ex7_mgr_frelax_reuse_1proc)
            hypredrv_append_test_environment(hypredrive_test_ex7_mgr_frelax_reuse_1proc
                "HYPREDRV_LOG_LEVEL=2")

            add_test(NAME hypredrive_test_ex7_mgr_grelax_reuse_1proc
                COMMAND ${CMAKE_COMMAND}
                        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                        -DMPI_NUMPROCS=1
                        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/ex7-mgr-grelax-reuse.yml
                        "-DTARGET_ARGS:STRING=-q"
                        "-DREQUIRE_CONTAINS:STRING=preserving cached MGR handles across destroy|cached MGR handles before create|grelax=1|Solving linear system #24"
                        -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(hypredrive_test_ex7_mgr_grelax_reuse_1proc
                PROPERTIES
                FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                LABELS "integration;hypredrive"
            )
            hypredrv_maybe_disable_gpu_test(hypredrive_test_ex7_mgr_grelax_reuse_1proc)
            hypredrv_append_test_environment(hypredrive_test_ex7_mgr_grelax_reuse_1proc
                "HYPREDRV_LOG_LEVEL=2")

            add_test(NAME hypredrive_test_ex7_mgr_coarse_reuse_1proc
                COMMAND ${CMAKE_COMMAND}
                        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                        -DMPI_NUMPROCS=1
                        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/ex7-mgr-coarse-reuse.yml
                        "-DTARGET_ARGS:STRING=-q"
                        "-DREQUIRE_CONTAINS:STRING=preserving cached MGR handles across destroy|cached MGR handles before create|coarse=1|Solving linear system #24"
                        -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(hypredrive_test_ex7_mgr_coarse_reuse_1proc
                PROPERTIES
                FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                LABELS "integration;hypredrive"
            )
            hypredrv_maybe_disable_gpu_test(hypredrive_test_ex7_mgr_coarse_reuse_1proc)
            hypredrv_append_test_environment(hypredrive_test_ex7_mgr_coarse_reuse_1proc
                "HYPREDRV_LOG_LEVEL=2")

            add_test(NAME hypredrive_test_ex7_mgr_frelax_ilu_reuse_1proc
                COMMAND ${CMAKE_COMMAND}
                        -DLAUNCH_DIR=${CMAKE_SOURCE_DIR}
                        -DTARGET_BIN=$<TARGET_FILE:hypredrive-cli>
                        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                        -DMPI_NUMPROCS=1
                        -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                        -DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}
                        -DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}
                        -DCONFIG_FILE=${CMAKE_SOURCE_DIR}/examples/ex7-mgr-frelax-ilu-reuse.yml
                        "-DTARGET_ARGS:STRING=-q"
                        "-DREQUIRE_CONTAINS:STRING=preserving cached MGR handles across destroy|cached MGR handles before create|frelax=1|Solving linear system #24"
                        -P ${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_RunScript.cmake
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            )
            set_tests_properties(hypredrive_test_ex7_mgr_frelax_ilu_reuse_1proc
                PROPERTIES
                FAIL_REGULAR_EXPRESSION "${HYPREDRV_FAIL_REGEX_DEFAULT}"
                SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
                LABELS "integration;hypredrive"
            )
            hypredrv_maybe_disable_gpu_test(hypredrive_test_ex7_mgr_frelax_ilu_reuse_1proc)
            hypredrv_append_test_environment(hypredrive_test_ex7_mgr_frelax_ilu_reuse_1proc
                "HYPREDRV_LOG_LEVEL=2")
        endif()

        # Test main.c help/usage/error branches
        # Note: --help exits with 0, so we need to allow that
        add_executable_test(hypredrive_help hypredrive-cli 1 ARGS "--help" FAIL_REGULAR_EXPRESSION "^$")
        add_executable_test(hypredrive_help_short hypredrive-cli 1 ARGS "-h" FAIL_REGULAR_EXPRESSION "^$")
        add_executable_test(hypredrive_no_args hypredrive-cli 1 ARGS "" FAIL_REGULAR_EXPRESSION "^$")
        set_tests_properties(hypredrive_no_args PROPERTIES WILL_FAIL TRUE)

        # Exercise the long-form quiet flag parsing ("--quiet")
        add_executable_test(hypredrive_quiet_longflag hypredrive-cli 1
            ARGS "--quiet" "examples/ex1.yml"
            FAIL_REGULAR_EXPRESSION "^$"
        )

        # Exercise config-file detection when override args are present
        add_executable_test(hypredrive_cli_extra hypredrive-cli 1
            ARGS "examples/ex1.yml" "--args" "--solver:pcg:max_iter" "5"
            FAIL_REGULAR_EXPRESSION "^$"
        )
        add_executable_test(hypredrive_cli_extra_nodash hypredrive-cli 1
            ARGS "examples/ex1.yml" "--args" "solver:pcg:max_iter" "5"
            REQUIRE_CONTAINS "max_iter: 5"
        )
    else()
        message(STATUS "Skipping hypredrive integration tests (requires hypre >= 2.19.0).")
    endif()
endif()
