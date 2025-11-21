# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

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

        # Apply directory-wide defaults so future targets (e.g., examples, tests) get flags
        add_compile_options(-O0 -g --coverage)
        add_link_options(--coverage)
    else()
        message(WARNING "Coverage is only supported with GCC/Clang. Current compiler: ${CMAKE_C_COMPILER_ID}")
    endif()

    # Optional: gcovr-based coverage target
    find_program(GCOVR_EXECUTABLE NAMES gcovr)

    # Determine which coverage tool to use based on compiler
    set(_gcov_candidates)
    set(_use_llvm_cov FALSE)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        # For Clang, prefer llvm-cov over gcov
        # Try versioned llvm-cov first (e.g., llvm-cov-22), then generic
        set(_llvm_cov_names)

        # Try to get version from compiler
        if(CMAKE_C_COMPILER_VERSION_MAJOR)
            list(APPEND _llvm_cov_names llvm-cov-${CMAKE_C_COMPILER_VERSION_MAJOR})
        endif()
        # Also try common version numbers (14-22) as fallback
        foreach(ver RANGE 22 14 -1)
            list(APPEND _llvm_cov_names llvm-cov-${ver})
        endforeach()
        list(APPEND _llvm_cov_names llvm-cov)

        # Build search paths - try versioned paths first, then generic
        set(_llvm_search_paths)
        if(CMAKE_C_COMPILER_VERSION_MAJOR)
            list(APPEND _llvm_search_paths /usr/lib/llvm-${CMAKE_C_COMPILER_VERSION_MAJOR}/bin)
        endif()
        # Also search common versioned paths
        foreach(ver RANGE 22 14 -1)
            list(APPEND _llvm_search_paths /usr/lib/llvm-${ver}/bin)
        endforeach()
        list(APPEND _llvm_search_paths /usr/lib/llvm/bin /usr/local/llvm/bin)

        # Search in versioned paths first
        find_program(LLVM_COV_EXECUTABLE
            NAMES ${_llvm_cov_names}
            PATHS ${_llvm_search_paths}
            NO_DEFAULT_PATH
        )
        # Also search in default PATH (for systems where llvm-cov is in PATH)
        if(NOT LLVM_COV_EXECUTABLE)
            find_program(LLVM_COV_EXECUTABLE NAMES ${_llvm_cov_names})
        endif()
        if(LLVM_COV_EXECUTABLE)
            # Use llvm-cov with gcovr - gcovr expects "llvm-cov gcov" as the command
            set(GCOV_EXECUTABLE "${LLVM_COV_EXECUTABLE} gcov")
            set(_use_llvm_cov TRUE)
            message(STATUS "HYPREDRV: using llvm-cov executable ${LLVM_COV_EXECUTABLE} for coverage")
        else()
            # Fallback to gcov if llvm-cov not found
            list(APPEND _gcov_candidates gcov)
            message(WARNING "llvm-cov not found; falling back to gcov (may not work with Clang)")
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        # For GCC, use gcov
        list(APPEND _gcov_candidates gcov)
        if(CMAKE_C_COMPILER_VERSION)
            list(APPEND _gcov_candidates gcov-${CMAKE_C_COMPILER_VERSION_MAJOR})
        endif()
    else()
        list(APPEND _gcov_candidates gcov)
    endif()

    # Find gcov if not already set (for GCC or fallback)
    if(NOT GCOV_EXECUTABLE)
        find_program(GCOV_EXECUTABLE NAMES ${_gcov_candidates})
    endif()

    if(GCOV_EXECUTABLE)
        message(STATUS "HYPREDRV: using coverage executable ${GCOV_EXECUTABLE}")
    else()
        message(WARNING "Coverage executable not found; gcovr will fall back to default 'gcov'")
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

        # Create a target to run tests for coverage data generation
        # This runs ctest to execute all tests and generate .gcda files
        if(HYPREDRV_ENABLE_TESTING)
            # Collect dependencies: example executables if examples are enabled
            set(_coverage_test_deps)
            if(HYPREDRV_ENABLE_EXAMPLES)
                if(TARGET laplacian)
                    list(APPEND _coverage_test_deps laplacian)
                endif()
                if(TARGET elasticity)
                    list(APPEND _coverage_test_deps elasticity)
                endif()
            endif()
            
            add_custom_target(run_tests_for_coverage
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Running tests to generate coverage data"
                VERBATIM
            )
            
            # Ensure example executables are built before running tests
            if(_coverage_test_deps)
                add_dependencies(run_tests_for_coverage ${_coverage_test_deps})
            endif()
        endif()

        add_custom_target(coverage
            COMMAND ${GCOVR_EXECUTABLE} ${_gcovr_args}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report with gcovr"
            VERBATIM
        )

        # Ensure tests run before capturing coverage
        if(TARGET run_tests_for_coverage)
            add_dependencies(coverage run_tests_for_coverage)
        elseif(TARGET check)
            # Fallback to check target if testing is not enabled
            add_dependencies(coverage check)
        endif()
    else()
        message(STATUS "gcovr not found; 'coverage' target will not be available. Install with: pip install gcovr")
    endif()
endif()
