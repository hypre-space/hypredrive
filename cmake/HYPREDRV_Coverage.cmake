# Coverage support for hypredrive
if(HYPREDRV_ENABLE_COVERAGE)
    message(STATUS "HYPREDRV: code coverage instrumentation is ENABLED")

    if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
        message(WARNING "HYPREDRV_ENABLE_COVERAGE is ON, but CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'. Consider using Debug for accurate coverage.")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # Apply coverage flags to existing primary targets
        foreach(tgt IN ITEMS HYPREDRV hypredrive)
            if(TARGET ${tgt})
                target_compile_options(${tgt} PRIVATE -O0 -g --coverage)
                target_link_options(${tgt} PRIVATE --coverage)
            endif()
        endforeach()

        # Also apply coverage flags to test executables (they're created before this module is included)
        get_property(_test_targets DIRECTORY tests PROPERTY BUILDSYSTEM_TARGETS)
        if(_test_targets)
            foreach(test_tgt ${_test_targets})
                if(TARGET ${test_tgt})
                    get_target_property(_tgt_type ${test_tgt} TYPE)
                    if(_tgt_type STREQUAL "EXECUTABLE")
                        target_compile_options(${test_tgt} PRIVATE -O0 -g --coverage)
                        target_link_options(${test_tgt} PRIVATE --coverage)
                        message(STATUS "Applied coverage flags to test target ${test_tgt}")
                    endif()
                endif()
            endforeach()
        endif()

        # Also apply directory-wide defaults so future targets (e.g., examples) get flags
        add_compile_options(-O0 -g --coverage)
        add_link_options(--coverage)
    else()
        message(WARNING "Coverage is only supported with GCC/Clang. Current compiler: ${CMAKE_C_COMPILER_ID}")
    endif()

    # Optional: gcovr-based coverage target
    find_program(GCOVR_EXECUTABLE NAMES gcovr)
    set(_gcov_candidates gcov)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION)
        list(APPEND _gcov_candidates gcov-${CMAKE_C_COMPILER_VERSION_MAJOR})
    endif()
    find_program(GCOV_EXECUTABLE NAMES ${_gcov_candidates})
    if(GCOV_EXECUTABLE)
        message(STATUS "HYPREDRV: using gcov executable ${GCOV_EXECUTABLE}")
    else()
        message(WARNING "gcov executable not found; gcovr will fall back to default 'gcov'")
    endif()
    if(GCOVR_EXECUTABLE)
        set(_gcovr_args
            --root ${CMAKE_SOURCE_DIR}
            --object-directory ${CMAKE_BINARY_DIR}
            --exclude ".*(/|^)cmake/.*"
            --exclude ".*(/|^)examples/.*"
            --exclude ".*(/|^)install/.*"
            --exclude ".*(/|^)docs/.*"
            --exclude ".*\\.h$"
            --xml coverage.xml --xml-pretty
            --html-details coverage.html
            --print-summary
        )
        if(GCOV_EXECUTABLE)
            list(APPEND _gcovr_args --gcov-executable ${GCOV_EXECUTABLE})
        endif()

        add_custom_target(coverage
            COMMAND ${GCOVR_EXECUTABLE} ${_gcovr_args}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report with gcovr"
            VERBATIM
        )

        # Ensure tests run before capturing coverage if test target exists
        if(TARGET check)
            add_dependencies(coverage check)
        endif()
    else()
        message(STATUS "gcovr not found; 'coverage' target will not be available. Install with: pip install gcovr")
    endif()
endif()


