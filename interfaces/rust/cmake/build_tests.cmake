# Cargo chooses a hash in the executable name. Read its machine-readable output
# once, then give CTest a stable path that every MPI rank executes directly.
execute_process(
    COMMAND "${CARGO_EXECUTABLE}" test --locked --manifest-path "${MANIFEST}"
        --no-run --test smoke --message-format=json-render-diagnostics
    OUTPUT_VARIABLE _cargo_output RESULT_VARIABLE _cargo_status)
if(NOT _cargo_status EQUAL 0)
    message(FATAL_ERROR "Cargo failed to build the Rust integration executable")
endif()

string(REPLACE ";" "\\;" _cargo_output "${_cargo_output}")
string(REPLACE "\n" ";" _cargo_lines "${_cargo_output}")
foreach(_cargo_line IN LISTS _cargo_lines)
    string(REGEX REPLACE "\r$" "" _cargo_line "${_cargo_line}")
    string(JSON _reason ERROR_VARIABLE _json_error GET "${_cargo_line}" reason)
    if(_json_error OR NOT _reason STREQUAL "compiler-artifact")
        continue()
    endif()
    string(JSON _target_name ERROR_VARIABLE _name_error GET "${_cargo_line}" target name)
    string(JSON _executable ERROR_VARIABLE _exec_error GET "${_cargo_line}" executable)
    if(NOT _name_error AND NOT _exec_error AND _target_name STREQUAL "smoke" AND _executable)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_executable}" "${TEST_EXECUTABLE}"
            COMMAND_ERROR_IS_FATAL ANY)
        return()
    endif()
endforeach()
message(FATAL_ERROR "Cargo did not report the smoke test executable")
