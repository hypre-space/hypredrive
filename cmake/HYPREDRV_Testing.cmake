# Testing setup (equivalent to autotools make check)
if(HYPREDRV_ENABLE_TESTING)
    enable_testing()
    
    # Find MPI for tests
    find_package(MPI REQUIRED)
    
    # Add tests (equivalent to autotools make check)
    add_test(NAME test_ex1_1proc
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:hypredrive> ${CMAKE_SOURCE_DIR}/examples/ex1.yml ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    add_test(NAME test_ex2_4proc
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:hypredrive> ${CMAKE_SOURCE_DIR}/examples/ex2.yml ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Custom check target (equivalent to autotools make check)
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS hypredrive
        COMMENT "Running tests (equivalent to autotools make check)"
        VERBATIM
    )
endif()
