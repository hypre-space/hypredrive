if(NOT DEFINED TEST_EXE)
    message(FATAL_ERROR "TEST_EXE is required")
endif()

execute_process(
    COMMAND "${TEST_EXE}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

set(_combined "${_stdout}\n${_stderr}")

if(_result EQUAL 0)
    message(FATAL_ERROR "test_check_failure unexpectedly exited successfully")
endif()

if(NOT _combined MATCHES "HYPREDRV call failed")
    message(FATAL_ERROR
        "test_check_failure failed for the wrong reason; output was:\n${_combined}")
endif()

message(STATUS "HYPREDRV call failed")
