# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

execute_process(
    COMMAND bash ${SOURCE_DIR}/scripts/list_public_apis.sh
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE _actual
    RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "scripts/list_public_apis.sh failed")
endif()
string(REGEX REPLACE "\nTotal: [0-9]+ public APIs\n?$" "" _actual "${_actual}")
file(READ "${EXPECTED}" _expected)
string(STRIP "${_actual}" _actual)
string(STRIP "${_expected}" _expected)
if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR "C++ API coverage list is stale. Update interfaces/cpp/tests/api_coverage.txt and hypredrive.hpp.\nActual:\n${_actual}\nExpected:\n${_expected}")
endif()

if(NOT DEFINED HEADER)
    message(FATAL_ERROR "HEADER must point to hypredrive.hpp")
endif()
file(READ "${HEADER}" _header)
string(REPLACE "\n" ";" _expected_list "${_expected}")
set(_missing)
foreach(_api IN LISTS _expected_list)
    if(_api STREQUAL "")
        continue()
    endif()
    # Require an actual C API call. Documentation-only mentions such as
    # @see HYPREDRV_Foo do not satisfy wrapper coverage.
    string(REGEX MATCH "${_api}[ \t\r\n]*\\(" _found "${_header}")
    if(NOT _found)
        list(APPEND _missing "${_api}")
    endif()
endforeach()
if(_missing)
    message(FATAL_ERROR "hypredrive.hpp does not call these public C APIs: ${_missing}")
endif()
