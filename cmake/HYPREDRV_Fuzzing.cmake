# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

set(HYPREDRV_FUZZ_ENGINE "replay" CACHE STRING
    "Fuzz harness engine: replay, libfuzzer, or afl")
set_property(CACHE HYPREDRV_FUZZ_ENGINE PROPERTY STRINGS replay libfuzzer afl)

option(HYPREDRV_FUZZ_MSAN "Enable MemorySanitizer for fuzzing builds" OFF)

if(HYPREDRV_ENABLE_FUZZING)
    if(HYPREDRV_ENABLE_COVERAGE)
        message(FATAL_ERROR
            "HYPREDRV_ENABLE_FUZZING and HYPREDRV_ENABLE_COVERAGE are mutually exclusive")
    endif()

    if(NOT HYPREDRV_ENABLE_ANALYSIS)
        message(STATUS "HYPREDRV_ENABLE_FUZZING=ON forces HYPREDRV_ENABLE_ANALYSIS=ON")
    endif()
    set(HYPREDRV_ENABLE_ANALYSIS ON CACHE BOOL
        "Enable static code analysis (clang-tidy) and sanitizers (ASan/UBSan)" FORCE)

    if(NOT HYPREDRV_ENABLE_TESTING)
        message(STATUS "HYPREDRV_ENABLE_FUZZING=ON forces HYPREDRV_ENABLE_TESTING=ON")
    endif()
    set(HYPREDRV_ENABLE_TESTING ON CACHE BOOL
        "Enable testing support and check target" FORCE)

    if(BUILD_SHARED_LIBS)
        message(STATUS "HYPREDRV_ENABLE_FUZZING=ON forces BUILD_SHARED_LIBS=OFF")
    endif()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

    if(HYPREDRV_FUZZ_MSAN)
        if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "HYPREDRV_FUZZ_MSAN requires Clang")
        endif()
        add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
        add_link_options(-fsanitize=memory)
    endif()

    if(HYPREDRV_FUZZ_ENGINE STREQUAL "libfuzzer")
        if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "HYPREDRV_FUZZ_ENGINE=libfuzzer requires Clang")
        endif()
    elseif(HYPREDRV_FUZZ_ENGINE STREQUAL "afl")
        get_filename_component(_hypredrv_fuzz_cc "${CMAKE_C_COMPILER}" NAME)
        if(NOT _hypredrv_fuzz_cc MATCHES "^afl-clang-(fast|lto)(\\+\\+)?$")
            message(FATAL_ERROR
                "HYPREDRV_FUZZ_ENGINE=afl requires afl-clang-fast or afl-clang-lto")
        endif()
        unset(_hypredrv_fuzz_cc)
    elseif(NOT HYPREDRV_FUZZ_ENGINE STREQUAL "replay")
        message(FATAL_ERROR
            "Unsupported HYPREDRV_FUZZ_ENGINE='${HYPREDRV_FUZZ_ENGINE}' "
            "(expected replay, libfuzzer, or afl)")
    endif()
endif()

function(_hypredrv_fuzz_mode_number mode out_var)
    if(mode STREQUAL "parse")
        set(_mode_num 1)
    elseif(mode STREQUAL "solve")
        set(_mode_num 2)
    elseif(mode STREQUAL "lsseq")
        set(_mode_num 3)
    elseif(mode STREQUAL "matrix")
        set(_mode_num 4)
    elseif(mode STREQUAL "vector")
        set(_mode_num 5)
    else()
        message(FATAL_ERROR "Unknown fuzz mode '${mode}'")
    endif()
    set(${out_var} "${_mode_num}" PARENT_SCOPE)
endfunction()

function(_hypredrv_fuzz_register_replay_tests target_name mode)
    if(NOT HYPREDRV_ENABLE_TESTING)
        return()
    endif()
    if(mode STREQUAL "solve" AND NOT HYPREDRV_HAVE_HYPRE_21900_DEV0)
        message(STATUS
            "Skipping solve fuzz replay tests: HYPRE >= 2.19.0 development APIs are unavailable")
        return()
    endif()

    set(_corpus_inputs ${ARGN})
    set(_labels "fuzz-replay;unit")
    if(mode STREQUAL "solve")
        set(_labels "fuzz-replay;hypredrive")
    endif()

    foreach(_corpus_input IN LISTS _corpus_inputs)
        if(IS_DIRECTORY "${_corpus_input}")
            file(GLOB _inputs CONFIGURE_DEPENDS "${_corpus_input}/*")
        elseif(EXISTS "${_corpus_input}")
            set(_inputs "${_corpus_input}")
        else()
            set(_inputs)
        endif()
        foreach(_input IN LISTS _inputs)
            if(IS_DIRECTORY "${_input}")
                continue()
            endif()
            get_filename_component(_name "${_input}" NAME)
            if(_name STREQUAL ".gitkeep")
                continue()
            endif()
            string(MAKE_C_IDENTIFIER "${mode}_${_name}" _test_suffix)
            set(_test_name "fuzz_replay_${_test_suffix}")
            add_test(NAME ${_test_name}
                COMMAND $<TARGET_FILE:${target_name}> "${_input}"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
            set_tests_properties(${_test_name}
                PROPERTIES
                    FAIL_REGULAR_EXPRESSION
                        "HYPREDRIVE Failure!!!|BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES|Segmentation fault|Abort\\(|AddressSanitizer:|UndefinedBehaviorSanitizer:|LeakSanitizer:|runtime error:"
                    SKIP_RETURN_CODE 77
                    LABELS "${_labels}")
            if(COMMAND hypredrv_append_test_environment)
                hypredrv_append_test_environment(${_test_name})
            endif()
        endforeach()
    endforeach()
endfunction()

function(hypredrv_add_fuzz_target target_name mode)
    set(_one_value SOURCE ENGINE)
    set(_multi_value DICT CORPUS REGRESSIONS)
    cmake_parse_arguments(FUZZ "" "${_one_value}" "${_multi_value}" ${ARGN})

    if(NOT FUZZ_SOURCE)
        message(FATAL_ERROR "hypredrv_add_fuzz_target(${target_name}) requires SOURCE")
    endif()
    if(NOT FUZZ_ENGINE)
        set(FUZZ_ENGINE "${HYPREDRV_FUZZ_ENGINE}")
    endif()

    _hypredrv_fuzz_mode_number("${mode}" _mode_num)

    add_executable(${target_name} "${FUZZ_SOURCE}")
    target_link_libraries(${target_name} PRIVATE HYPREDRV::HYPREDRV)
    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tests)
    target_compile_definitions(${target_name}
        PRIVATE
            FUZZ_MODE=${_mode_num}
            HYPREDRV_FUZZ_MODE_NAME="${mode}"
            HYPREDRIVE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    if(HYPREDRV_ENABLE_COMPRESSION)
        target_compile_definitions(${target_name} PRIVATE HYPREDRV_FUZZ_HAS_LSSEQ=1)
    endif()

    if(FUZZ_ENGINE STREQUAL "libfuzzer")
        target_compile_definitions(${target_name} PRIVATE HYPREDRV_FUZZ_ENGINE_LIBFUZZER=1)
        target_compile_options(${target_name} PRIVATE -fsanitize=fuzzer-no-link)
        target_link_options(${target_name} PRIVATE -fsanitize=fuzzer)
    elseif(FUZZ_ENGINE STREQUAL "afl")
        target_compile_definitions(${target_name} PRIVATE HYPREDRV_FUZZ_ENGINE_AFL=1)
    elseif(FUZZ_ENGINE STREQUAL "replay")
        target_compile_definitions(${target_name} PRIVATE HYPREDRV_FUZZ_ENGINE_REPLAY=1)
    else()
        message(FATAL_ERROR "Unsupported fuzz engine '${FUZZ_ENGINE}'")
    endif()

    get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
    get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
    if(_sanitizer_enabled AND _sanitizer_flags)
        target_link_options(${target_name} PRIVATE ${_sanitizer_flags})
    endif()

    if(TARGET data)
        add_dependencies(${target_name} data)
    endif()

    if(FUZZ_ENGINE STREQUAL "replay")
        _hypredrv_fuzz_register_replay_tests(${target_name} "${mode}"
            ${FUZZ_CORPUS} ${FUZZ_REGRESSIONS})
    endif()
endfunction()
