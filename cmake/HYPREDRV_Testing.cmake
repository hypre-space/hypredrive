# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Function for adding tests
function(add_hypredrive_test test_name num_procs config_file)
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

    set_tests_properties(${test_name}
        PROPERTIES
        FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure"
        SKIP_REGULAR_EXPRESSION "\\[test\\] Skipping example:"
    )
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

if(HYPREDRV_ENABLE_TESTING)
    # Must be called before add_subdirectory(tests) so that add_test() calls work
    enable_testing()

    # Add tests subfolder (contains unit tests that use add_test())
    add_subdirectory(tests)

    # Add tests
    add_hypredrive_test(test_ex1_1proc  1 ex1.yml)
    add_hypredrive_test(test_ex1a_1proc 1 ex1a.yml)
    add_hypredrive_test(test_ex1b_1proc 1 ex1b.yml)
    add_hypredrive_test(test_ex1c_1proc 1 ex1c.yml)
    add_hypredrive_test(test_ex1d_1proc 1 ex1d.yml)
    add_hypredrive_test(test_ex2_4proc  4 ex2.yml)
    add_hypredrive_test(test_ex3_1proc  1 ex3.yml)
    add_hypredrive_test(test_ex4_4proc  4 ex4.yml)
    add_hypredrive_test(test_ex5_1proc  1 ex5.yml)
    if (HYPREDRV_ENABLE_EIGSPEC)
        add_hypredrive_test(test_ex6_1proc 1 ex6.yml)
    endif()
    add_hypredrive_test(test_ex7_1proc  1 ex7.yml)
endif()
