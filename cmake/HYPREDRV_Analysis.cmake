# Static code analysis and sanitizer support for hypredrive
if(HYPREDRV_ENABLE_ANALYSIS)
    message(STATUS "Static code analysis and sanitizers are enabled")

    if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
        message(WARNING "HYPREDRV_ENABLE_ANALYSIS is ON, but CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'. Consider using Debug for better analysis.")
    endif()

    # ============================================================================
    # Sanitizers: AddressSanitizer, UndefinedBehaviorSanitizer, MemorySanitizer
    # ============================================================================
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # Sanitizer flags - ASan and UBSan are most commonly used
        # Note: -O1 is recommended for sanitizers, but we respect CMAKE_BUILD_TYPE's optimization
        # if it's set. We only add -fno-omit-frame-pointer for better stack traces.
        set(_sanitizer_flags "-fno-omit-frame-pointer")
        set(_sanitizer_link_flags "")

        # AddressSanitizer (detects memory errors, use-after-free, buffer overflows)
        list(APPEND _sanitizer_flags "-fsanitize=address")
        list(APPEND _sanitizer_link_flags "-fsanitize=address")

        # UndefinedBehaviorSanitizer (detects undefined behavior)
        list(APPEND _sanitizer_flags "-fsanitize=undefined")
        list(APPEND _sanitizer_link_flags "-fsanitize=undefined")

        # MemorySanitizer (detects uninitialized memory reads) - Clang only, experimental
        # Note: MSan is very slow and may not work with all libraries (e.g., MPI)
        # Uncomment if needed:
        # if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        #     list(APPEND _sanitizer_flags "-fsanitize=memory")
        #     list(APPEND _sanitizer_link_flags "-fsanitize=memory")
        # endif()

        # Apply sanitizer flags to existing primary targets
        foreach(tgt IN ITEMS HYPREDRV hypredrive)
            if(TARGET ${tgt})
                target_compile_options(${tgt} PRIVATE ${_sanitizer_flags})
                target_link_options(${tgt} PRIVATE ${_sanitizer_link_flags})
                message(STATUS "Applied sanitizers to target ${tgt}")
            endif()
        endforeach()

        # Also apply sanitizer flags to test executables (they're created before this module is included)
        get_property(_test_targets DIRECTORY tests PROPERTY BUILDSYSTEM_TARGETS)
        if(_test_targets)
            foreach(test_tgt ${_test_targets})
                if(TARGET ${test_tgt})
                    get_target_property(_tgt_type ${test_tgt} TYPE)
                    if(_tgt_type STREQUAL "EXECUTABLE")
                        target_compile_options(${test_tgt} PRIVATE ${_sanitizer_flags})
                        target_link_options(${test_tgt} PRIVATE ${_sanitizer_link_flags})
                        message(STATUS "Applied sanitizers to test target ${test_tgt}")
                    endif()
                endif()
            endforeach()
        endif()

        # Also apply directory-wide defaults so future targets (e.g., examples) get flags
        add_compile_options(${_sanitizer_flags})
        add_link_options(${_sanitizer_link_flags})

        message(STATUS "AddressSanitizer and UndefinedBehaviorSanitizer enabled")
        message(STATUS "Set ASAN_OPTIONS and UBSAN_OPTIONS environment variables to control behavior")
    else()
        message(WARNING "Sanitizers are only supported with GCC/Clang. Current compiler: ${CMAKE_C_COMPILER_ID}")
    endif()

    # ============================================================================
    # clang-tidy static analysis
    # ============================================================================
    find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy clang-tidy-18 clang-tidy-17 clang-tidy-16 clang-tidy-15)
    if(CLANG_TIDY_EXECUTABLE)
        message(STATUS "Found clang-tidy at ${CLANG_TIDY_EXECUTABLE}")

        # Create a .clang-tidy configuration file if it doesn't exist
        set(_clang_tidy_config "${CMAKE_SOURCE_DIR}/.clang-tidy")
        if(NOT EXISTS ${_clang_tidy_config})
            message(STATUS "Creating default .clang-tidy configuration")
            file(WRITE ${_clang_tidy_config}
                "# clang-tidy configuration for hypredrive
Checks: >
    -*,
    bugprone-*,
    cert-*,
    clang-analyzer-*,
    concurrency-*,
    cppcoreguidelines-*,
    modernize-*,
    performance-*,
    portability-*,
    readability-*,
    security-*,
    -cppcoreguidelines-pro-type-vararg,
    -cppcoreguidelines-owning-memory,
    -readability-magic-numbers,
    -readability-named-parameter,
    -readability-function-cognitive-complexity,
    -bugprone-easily-swappable-parameters,
    -readability-identifier-naming,
    -readability-isolate-declaration,
    -readability-uppercase-literal-suffix,
    -modernize-use-auto,
    -cppcoreguidelines-pro-bounds-array-to-pointer-decay,
    -readability-redundant-parentheses,
    -readability-redundant-casting,
    -bugprone-macro-parentheses,
    -cppcoreguidelines-macro-usage,
    -bugprone-reserved-identifier,
    -readability-else-after-return,
    -readability-redundant-control-flow

WarningsAsErrors: ''

HeaderFilterRegex: '^(${CMAKE_SOURCE_DIR}/src|${CMAKE_SOURCE_DIR}/include)/'

")
        else()
            message(STATUS "Using existing .clang-tidy configuration")
        endif()

        # Collect source files for analysis
        file(GLOB_RECURSE _src_files
            "${CMAKE_SOURCE_DIR}/src/*.c"
            "${CMAKE_SOURCE_DIR}/include/*.h"
        )

        # Exclude build directories and other non-source files
        list(FILTER _src_files EXCLUDE REGEX ".*/build/.*")
        list(FILTER _src_files EXCLUDE REGEX ".*/install/.*")
        list(FILTER _src_files EXCLUDE REGEX ".*/docs/.*")
        
        # Exclude gen_macros.h as it contains complex macros that clang-tidy modifies incorrectly
        list(FILTER _src_files EXCLUDE REGEX ".*/gen_macros\\.h$")
        list(FILTER _src_files EXCLUDE REGEX ".*/info\\.c$")

        # Prepare compile commands for clang-tidy
        # CMake generates compile_commands.json if CMAKE_EXPORT_COMPILE_COMMANDS is ON
        # Enable it automatically for static analysis (can be overridden by user)
        if(CMAKE_EXPORT_COMPILE_COMMANDS)
            message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS is already enabled")
        else()
            set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for clang-tidy" FORCE)
            message(STATUS "Enabling CMAKE_EXPORT_COMPILE_COMMANDS for clang-tidy")
        endif()

        # Output file for clang-tidy
        set(CLANG_TIDY_OUTPUT ${CMAKE_BINARY_DIR}/clang-tidy-output.txt)

        # Create a target to run clang-tidy
        set(_clang_tidy_args
            -p=${CMAKE_BINARY_DIR}
            --quiet
            ${_src_files}
        )

        add_custom_target(clang-tidy
            COMMAND ${CLANG_TIDY_EXECUTABLE} ${_clang_tidy_args} > ${CLANG_TIDY_OUTPUT} 2>&1 || true
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-tidy static analysis"
            VERBATIM
        )
        
        message(STATUS "Output saved to ${CLANG_TIDY_OUTPUT}")

        # Create a target to run clang-tidy with fixes
        # Use --fix-errors to continue even if there are errors
        add_custom_target(clang-tidy-fix
            COMMAND ${CLANG_TIDY_EXECUTABLE} ${_clang_tidy_args} --fix --fix-errors
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-tidy static analysis with automatic fixes"
            VERBATIM
        )

        message(STATUS "Clang-tidy targets 'clang-tidy' and 'clang-tidy-fix' are available")
        message(STATUS "Run 'cmake --build . --target clang-tidy' to run static analysis")
    else()
        message(WARNING "clang-tidy not found. Install it to enable static analysis. (e.g., apt-get install clang-tidy)")
        message(WARNING "The 'clang-tidy' and 'clang-tidy-fix' targets will not be available")
    endif()

    # ============================================================================
    # Optional: Additional static analysis tools
    # ============================================================================
    # cppcheck support (optional)
    find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)
    if(CPPCHECK_EXECUTABLE)
        message(STATUS "Found cppcheck at ${CPPCHECK_EXECUTABLE}")

        # XML output file
        set(CPPCHECK_XML_OUTPUT ${CMAKE_BINARY_DIR}/cppcheck-report.xml)
        set(CPPCHECK_HTML_DIR ${CMAKE_BINARY_DIR}/cppcheck-html)
        set(CPPCHECK_CHECKERS_REPORT ${CMAKE_BINARY_DIR}/cppcheck-checkers.txt)

        # Get HYPRE include directories
        get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
        if(HYPRE_INCLUDE_DIRS)
            # Convert list to individual -I flags
            set(CPPCHECK_HYPRE_INCLUDES "")
            foreach(INCDIR ${HYPRE_INCLUDE_DIRS})
                list(APPEND CPPCHECK_HYPRE_INCLUDES -I${INCDIR})
            endforeach()
        else()
            set(CPPCHECK_HYPRE_INCLUDES "")
        endif()

        # Determine parallelism for cppcheck
        if(CMAKE_BUILD_PARALLEL_LEVEL)
            set(CPPCHECK_JOBS_FLAG "-j${CMAKE_BUILD_PARALLEL_LEVEL}")
        else()
            include(ProcessorCount)
            ProcessorCount(N)
            if(N EQUAL 0)
                set(CPPCHECK_JOBS_FLAG "-j1")
            else()
                set(CPPCHECK_JOBS_FLAG "-j${N}")
            endif()
        endif()

        # Create suppressions file for HYPRE_MIXED_PRECISION headers (conditional includes)
        # These headers are only included when HYPRE_MIXED_PRECISION is defined
        # Also suppress all warnings from specific HYPRE header files
        # Format: [error id]:[filename]:[line] (use * for wildcards)
        set(CPPCHECK_SUPPRESSIONS_FILE ${CMAKE_BINARY_DIR}/cppcheck-suppressions.txt)
        file(WRITE ${CPPCHECK_SUPPRESSIONS_FILE}
            "missingInclude:*HYPRE*.h\n"
            "missingInclude:*_hypre*.h\n"
            "*:*_hypre_IJ_mv.h\n"
            "*:*_hypre_utilities.h\n"
        )

        # Simple cppcheck target - analyze only src/ directory
        add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c99
                --suppress=unusedFunction
                --suppress=missingIncludeSystem
                --suppressions-list=${CPPCHECK_SUPPRESSIONS_FILE}
                --inconclusive
                --force
                --check-level=exhaustive
                --xml
                --xml-version=2
                --checkers-report=${CPPCHECK_CHECKERS_REPORT}
                -I${CMAKE_SOURCE_DIR}/include
                -I${CMAKE_BINARY_DIR}
                ${CPPCHECK_JOBS_FLAG}
                ${CPPCHECK_HYPRE_INCLUDES}
                ${CMAKE_SOURCE_DIR}/src
                2> ${CPPCHECK_XML_OUTPUT}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running cppcheck static analysis on src/ directory"
            VERBATIM
        )

        # Check for cppcheck-htmlreport and generate HTML automatically
        find_program(CPPCHECK_HTMLREPORT_EXECUTABLE NAMES cppcheck-htmlreport)
        if(CPPCHECK_HTMLREPORT_EXECUTABLE)
            add_custom_command(TARGET cppcheck POST_BUILD
                COMMAND ${CPPCHECK_HTMLREPORT_EXECUTABLE}
                    --file=${CPPCHECK_XML_OUTPUT}
                    --report-dir=${CPPCHECK_HTML_DIR}
                COMMENT "Generating cppcheck HTML report"
                VERBATIM
            )
            message(STATUS "HTML report will be generated automatically at ${CPPCHECK_HTML_DIR}/index.html")
        else()
            message(STATUS "cppcheck-htmlreport not found. Install it to generate HTML reports (e.g., pip install cppcheck-htmlreport)")
        endif()

        message(STATUS "cppcheck target 'cppcheck' is available")
    endif()

    # ============================================================================
    # Environment variable hints
    # ============================================================================
    message(STATUS "")
    message(STATUS "Code analysis configuration:")
    message(STATUS "  Sanitizers: AddressSanitizer and UndefinedBehaviorSanitizer")
    message(STATUS "  Recommended environment variables:")
    message(STATUS "    ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:print_stacktrace=1")
    message(STATUS "    UBSAN_OPTIONS=print_stacktrace=1:abort_on_error=1")
    message(STATUS "")
endif()
