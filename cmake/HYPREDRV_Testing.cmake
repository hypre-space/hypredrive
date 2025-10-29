# Testing setup (equivalent to autotools make check)
if(HYPREDRV_ENABLE_TESTING)
    enable_testing()

    # Find MPI for tests
    find_package(MPI REQUIRED)

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

    # Add tests
    add_hypredrive_test(test_ex1_1proc 1 ex1.yml)
    add_hypredrive_test(test_ex2_4proc 4 ex2.yml)

    # Custom check target
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS hypredrive data
        COMMENT "Running tests"
        VERBATIM
    )
endif()
