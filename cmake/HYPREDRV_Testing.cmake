# Function for adding tests
function(add_hypredrive_test test_name num_procs config_file)
    add_test(NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:hypredrive> ${CMAKE_SOURCE_DIR}/examples/${config_file} ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(${test_name}
        PROPERTIES
        FAIL_REGULAR_EXPRESSION "HYPREDRIVE Failure!!!|Abort|Error|failure"
    )
endfunction()

# Function for adding tests with output verification
function(add_hypredrive_test_with_output test_name num_procs config_file example_id)
    # Create output file path
    set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/test_output_${test_name}.txt")
    set(REFERENCE_FILE "${CMAKE_SOURCE_DIR}/examples/refOutput/ex${example_id}.txt")

    # Run test and capture output
    add_test(NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_procs} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:hypredrive> ${CMAKE_SOURCE_DIR}/examples/${config_file} ${MPIEXEC_POSTFLAGS}
        DEPENDS data
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # Add output comparison test
    find_program(COMPARE_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/compare_output.sh")
    if(COMPARE_SCRIPT AND EXISTS ${REFERENCE_FILE})
        add_test(NAME ${test_name}_output
            COMMAND ${COMPARE_SCRIPT} ${OUTPUT_FILE} ${REFERENCE_FILE}
        )
        set_tests_properties(${test_name}_output
            PROPERTIES
            DEPENDS ${test_name}
        )
    endif()
endfunction()

if(HYPREDRV_ENABLE_TESTING)
    # Must be called before add_subdirectory(tests) so that add_test() calls work
    enable_testing()

    # Add tests subfolder (contains unit tests that use add_test())
    add_subdirectory(tests)

    # Add tests
    add_hypredrive_test(test_ex1_1proc 1 ex1.yml)
    add_hypredrive_test(test_ex2_4proc 4 ex2.yml)
    add_hypredrive_test(test_ex3_1proc 1 ex3.yml)
    add_hypredrive_test(test_ex4_4proc 4 ex4.yml)
    add_hypredrive_test(test_ex5_1proc 1 ex5.yml)
    if (HYPREDRV_ENABLE_EIGSPEC)
        add_hypredrive_test(test_ex6_1proc 1 ex6.yml)
    endif()
    add_hypredrive_test(test_ex7_1proc 1 ex7.yml)
endif()