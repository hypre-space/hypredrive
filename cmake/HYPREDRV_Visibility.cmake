# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

function(hypredrv_collect_public_api_names out_var header)
    file(READ "${header}" _public_header)
    string(REGEX MATCHALL
        "HYPREDRV_[A-Za-z0-9_]+[ \t\r\n]*\\("
        _api_matches "${_public_header}")

    set(_api_names "")
    foreach(_api_match IN LISTS _api_matches)
        string(REGEX REPLACE "[ \t\r\n]*\\($" "" _api_name "${_api_match}")
        if(NOT _api_name MATCHES
           "^HYPREDRV_(SAFE_CALL|SAFE_CALL_COMM|SUCCESS|operation)$")
            list(APPEND _api_names "${_api_name}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _api_names)
    list(SORT _api_names)

    if(NOT _api_names)
        message(FATAL_ERROR "Could not derive HYPREDRV public exports from ${header}")
    endif()
    set(${out_var} "${_api_names}" PARENT_SCOPE)
endfunction()

# The same module runs as a CMake script for the dynamic-export CTests. Only
# enter this branch when this file is the script entry point; another script
# may include the module without intending to run an export check.
if(DEFINED CMAKE_SCRIPT_MODE_FILE AND
   "${CMAKE_SCRIPT_MODE_FILE}" STREQUAL "${CMAKE_CURRENT_LIST_FILE}")
    cmake_minimum_required(VERSION 3.23)

    foreach(_required IN ITEMS LIBRARY MODE NM)
        if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
            message(FATAL_ERROR "${_required} is required")
        endif()
    endforeach()
    set(_source_dir "${CMAKE_CURRENT_LIST_DIR}/..")

    if(APPLE)
        set(_nm_args -gU "${LIBRARY}")
    else()
        set(_nm_args -D --defined-only "${LIBRARY}")
    endif()
    execute_process(
        COMMAND "${NM}" ${_nm_args}
        RESULT_VARIABLE _nm_result
        OUTPUT_VARIABLE _nm_output
        ERROR_VARIABLE _nm_error)
    if(NOT _nm_result EQUAL 0)
        message(FATAL_ERROR "Could not inspect ${LIBRARY}: ${_nm_error}")
    endif()

    string(REPLACE "\r\n" "\n" _nm_output "${_nm_output}")
    string(REPLACE "\n" ";" _nm_lines "${_nm_output}")
    set(_actual_exports "")
    foreach(_line IN LISTS _nm_lines)
        string(STRIP "${_line}" _line)
        if(_line MATCHES "([^ \t]+)$")
            set(_symbol "${CMAKE_MATCH_1}")
            string(REGEX REPLACE "@.*$" "" _symbol "${_symbol}")
            if(APPLE)
                string(REGEX REPLACE "^_" "" _symbol "${_symbol}")
            endif()
            list(APPEND _actual_exports "${_symbol}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _actual_exports)
    list(SORT _actual_exports)

    if(MODE STREQUAL "CORE")
        hypredrv_collect_public_api_names(
            _expected_exports "${_source_dir}/include/HYPREDRV.h")
        # Declared in HYPREDRV_utils.h, which the collector above does not scan,
        # so this exported helper must be listed explicitly.
        list(APPEND _expected_exports HYPREDRV_SafeCallHandleError)
    elseif(MODE STREQUAL "PYTHON")
        set(_expected_exports PyInit__core)
    elseif(MODE STREQUAL "JULIA")
        file(READ "${_source_dir}/interfaces/julia/src/HypreDrive.jl" _source)
        string(REGEX MATCHALL
            ":HYPREDRV_Julia[A-Za-z0-9_]+"
            _matches "${_source}")
        set(_expected_exports "")
        foreach(_match IN LISTS _matches)
            string(SUBSTRING "${_match}" 1 -1 _symbol)
            list(APPEND _expected_exports "${_symbol}")
        endforeach()
    elseif(MODE STREQUAL "FORTRAN")
        set(_forbidden_exports "")
        foreach(_symbol IN LISTS _actual_exports)
            if((_symbol MATCHES "^HYPREDRV_" AND
                NOT _symbol MATCHES "^HYPREDRV_Fortran") OR
               _symbol MATCHES "^(HYPRE_|hypre_|hypredrv_)")
                list(APPEND _forbidden_exports "${_symbol}")
            endif()
        endforeach()
        if(_forbidden_exports)
            list(JOIN _forbidden_exports "\n  " _forbidden_text)
            message(FATAL_ERROR
                "Fortran bridge leaked dependency symbols:\n  ${_forbidden_text}")
        endif()
        return()
    else()
        message(FATAL_ERROR "Unknown export-check mode: ${MODE}")
    endif()

    list(REMOVE_DUPLICATES _expected_exports)
    list(SORT _expected_exports)
    if(NOT "${_actual_exports}" STREQUAL "${_expected_exports}")
        set(_missing "")
        foreach(_symbol IN LISTS _expected_exports)
            if(NOT _symbol IN_LIST _actual_exports)
                list(APPEND _missing "${_symbol}")
            endif()
        endforeach()
        set(_unexpected "")
        foreach(_symbol IN LISTS _actual_exports)
            if(NOT _symbol IN_LIST _expected_exports)
                list(APPEND _unexpected "${_symbol}")
            endif()
        endforeach()
        list(JOIN _missing "\n  " _missing_text)
        list(JOIN _unexpected "\n  " _unexpected_text)
        message(FATAL_ERROR
            "Dynamic export mismatch for ${LIBRARY}\n"
            "Missing:\n  ${_missing_text}\n"
            "Unexpected:\n  ${_unexpected_text}")
    endif()
    return()
endif()

# Shared libraries may link static HYPREDRV/HYPRE archives. Their archive
# symbols are implementation details and must not become part of the resulting
# dynamic-loader interface.
include(CheckLinkerFlag)
if(UNIX AND NOT APPLE)
    check_linker_flag(C "LINKER:--exclude-libs,ALL"
        HYPREDRV_LINKER_SUPPORTS_EXCLUDE_LIBS_ALL)
endif()

function(_hypredrv_generate_darwin_core_export_list out_var)
    set(_public_header_path
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include/HYPREDRV.h")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_public_header_path}")
    hypredrv_collect_public_api_names(_api_names "${_public_header_path}")

    set(_export_list "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRV_darwin.exports")
    file(WRITE "${_export_list}" "")
    foreach(_api_name IN LISTS _api_names)
        file(APPEND "${_export_list}" "_${_api_name}\n")
    endforeach()
    # This helper is part of the installed HYPREDRV_utils.h API.
    file(APPEND "${_export_list}" "_HYPREDRV_SafeCallHandleError\n")

    set(${out_var} "${_export_list}" PARENT_SCOPE)
endfunction()

function(hypredrv_hide_static_archive_symbols target)
    cmake_parse_arguments(PARSE_ARGV 1 _visibility "" "" "APPLE_EXPORTS")

    if(NOT TARGET ${target})
        message(FATAL_ERROR "Unknown target passed to hypredrv_hide_static_archive_symbols: ${target}")
    endif()

    get_target_property(_target_type ${target} TYPE)
    if(NOT _target_type MATCHES "^(MODULE|SHARED)_LIBRARY$")
        return()
    endif()

    if(APPLE AND "${target}" STREQUAL "HYPREDRV")
        # Default visibility on HYPREDRV's public declarations is sufficient for
        # its own objects, but symbols pulled from a static HYPRE archive retain
        # their original visibility. Restrict the final dylib to the supported
        # public API as the Darwin counterpart to --exclude-libs,ALL.
        _hypredrv_generate_darwin_core_export_list(_hypredrv_export_list)
        target_link_options(${target} PRIVATE
            "LINKER:-exported_symbols_list,${_hypredrv_export_list}")
    elseif(APPLE AND _visibility_APPLE_EXPORTS)
        set(_hypredrv_export_list
            "${CMAKE_CURRENT_BINARY_DIR}/${target}_darwin.exports")
        list(JOIN _visibility_APPLE_EXPORTS "\n" _hypredrv_export_content)
        file(GENERATE OUTPUT "${_hypredrv_export_list}"
            CONTENT "${_hypredrv_export_content}\n")
        target_link_options(${target} PRIVATE
            "LINKER:-exported_symbols_list,${_hypredrv_export_list}")
    elseif(HYPREDRV_LINKER_SUPPORTS_EXCLUDE_LIBS_ALL)
        target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endfunction()

function(hypredrv_add_dynamic_export_test target mode)
    if(NOT HYPREDRV_ENABLE_TESTING OR NOT UNIX OR NOT TARGET ${target})
        return()
    endif()

    get_target_property(_target_type ${target} TYPE)
    if(NOT _target_type MATCHES "^(MODULE|SHARED)_LIBRARY$")
        return()
    endif()

    add_test(NAME ${target}_dynamic_exports
        COMMAND ${CMAKE_COMMAND}
            "-DLIBRARY=$<TARGET_FILE:${target}>"
            "-DMODE=${mode}"
            "-DNM=${CMAKE_NM}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    set_tests_properties(${target}_dynamic_exports PROPERTIES
        LABELS "interface;symbols")
endfunction()
