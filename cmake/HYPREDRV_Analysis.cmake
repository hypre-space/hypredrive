# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Static code analysis and sanitizer support for hypredrive
if(HYPREDRV_ENABLE_ANALYSIS)
    message(STATUS "Static code analysis and sanitizers are enabled")

    # Make sure `ctest` can run analysis builds without requiring manual
    # LD_LIBRARY_PATH exports (common pain point when HYPRE pulls in DSUPERLU).
    #
    # Important: do NOT force BUILD_WITH_INSTALL_RPATH here; that breaks running
    # build-tree tests because it can drop the build-tree rpath to libHYPREDRV.
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # Always allow build-tree executables to find our build-tree shared libs.
    set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
    list(APPEND CMAKE_BUILD_RPATH "$ORIGIN/lib")

    # Heuristic: derive potential external lib dirs from HYPRE's include dirs.
    # This typically finds prefixes like <superlu_prefix>/include -> <superlu_prefix>/lib.
    if(TARGET HYPRE::HYPRE)
        get_target_property(_hypre_inc_dirs HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
        if(_hypre_inc_dirs)
            foreach(_inc IN LISTS _hypre_inc_dirs)
                # Skip generator expressions
                if(_inc MATCHES "^\\$<")
                    continue()
                endif()
                get_filename_component(_prefix "${_inc}" DIRECTORY)
                set(_libdir "${_prefix}/lib")
                if(EXISTS "${_libdir}")
                    list(APPEND CMAKE_BUILD_RPATH "${_libdir}")
                endif()
            endforeach()
        endif()
    endif()
    list(REMOVE_DUPLICATES CMAKE_BUILD_RPATH)

    if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
        message(WARNING "HYPREDRV_ENABLE_ANALYSIS is ON, but CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'. Consider using Debug for better analysis.")
    endif()

    # ============================================================================
    # Sanitizers: AddressSanitizer, UndefinedBehaviorSanitizer, MemorySanitizer
    # ============================================================================
    # Note: Sanitizers are disabled when coverage is enabled to avoid conflicts
    # Coverage and sanitizers both use compiler instrumentation and can conflict
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT HYPREDRV_ENABLE_COVERAGE)
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
                #message(STATUS "Applied sanitizers to target ${tgt}")
            endif()
        endforeach()

        # Note: We don't apply sanitizer flags specifically to test executables here because:
        # 1. The directory-wide options added below will apply to all targets including tests
        # 2. Accessing the tests directory requires it to have been processed by add_subdirectory(),
        #    which only happens if HYPREDRV_ENABLE_TESTING=ON, and we can't safely check if it's
        #    been processed without potentially causing an error if it hasn't been.
        # If you need test-specific sanitizer configuration, it should be done in tests/CMakeLists.txt

        # Also apply directory-wide defaults so future targets (e.g., examples) get flags
        add_compile_options(${_sanitizer_flags})
        add_link_options(${_sanitizer_link_flags})

        # Store sanitizer link flags in cache variables and global property so subdirectories can access them
        set(HYPREDRV_SANITIZER_LINK_FLAGS ${_sanitizer_link_flags} CACHE INTERNAL "Sanitizer link flags for executables")
        set(HYPREDRV_SANITIZER_ENABLED ON CACHE INTERNAL "Whether sanitizers are enabled")
        set_property(GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS ${_sanitizer_link_flags})
        set_property(GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED ON)

        message(STATUS "AddressSanitizer and UndefinedBehaviorSanitizer enabled")
        message(STATUS "Set ASAN_OPTIONS and UBSAN_OPTIONS environment variables to control behavior")
        message(STATUS "Recommended ASAN_OPTIONS for better stack traces:")
        message(STATUS "  ASAN_OPTIONS=symbolize=1:print_stacktrace=1:abort_on_error=1")
        message(STATUS "  Note: 'symbolize=1' requires 'llvm-symbolizer' or 'addr2line' to be available")
    elseif(HYPREDRV_ENABLE_COVERAGE)
        message(STATUS "Sanitizers disabled because coverage is enabled (they conflict)")
    else()
        message(WARNING "Sanitizers are only supported with GCC/Clang. Current compiler: ${CMAKE_C_COMPILER_ID}")
    endif()

    # ============================================================================
    # clang-tidy static analysis
    # ============================================================================
    # Build list of clang-tidy executable names to search
    set(_clang_tidy_names)
    # Try versioned clang-tidy first (e.g., clang-tidy-22), starting from current compiler version
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_VERSION_MAJOR)
        list(APPEND _clang_tidy_names clang-tidy-${CMAKE_C_COMPILER_VERSION_MAJOR})
    endif()
    # Also try common version numbers (15-22) as fallback
    foreach(ver RANGE 22 15 -1)
        list(APPEND _clang_tidy_names clang-tidy-${ver})
    endforeach()
    # Generic name as last resort
    list(APPEND _clang_tidy_names clang-tidy)
    
    find_program(CLANG_TIDY_EXECUTABLE NAMES ${_clang_tidy_names})
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
    -readability-redundant-control-flow,
    -readability-identifier-length,
    -cppcoreguidelines-avoid-magic-numbers,
    -modernize-avoid-c-style-cast,
    -readability-non-const-parameter,
    -bugprone-narrowing-conversions,
    -readability-braces-around-statements,
    -cert-err33-c,
    -concurrency-mt-unsafe,
    -bugprone-assignment-in-if-condition,
    -bugprone-suspicious-realloc-usage,
    -bugprone-command-processor,
    -cert-env33-c,
    -cppcoreguidelines-avoid-non-const-global-variables,
    -bugprone-unchecked-string-to-number-conversion,
    -cert-err34-c,
    -clang-analyzer-optin.taint.TaintedAlloc,
    -clang-analyzer-security.ArrayBound,
    -clang-analyzer-unix.Malloc,
    -clang-analyzer-unix.Stream,
    -bugprone-multi-level-implicit-pointer-conversion,
    -clang-analyzer-core.NullDereference,
    -clang-analyzer-optin.portability.UnixAPI,
    -clang-analyzer-security.insecureAPI.strcpy,
    -cppcoreguidelines-init-variables,
    -bugprone-branch-clone,
    -readability-avoid-unconditional-preprocessor-if,
    -readability-use-concise-preprocessor-directives,
    -performance-no-int-to-ptr

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
        list(JOIN _src_files " " _src_files_str)
        set(_clang_tidy_cmd "${CLANG_TIDY_EXECUTABLE} -p=${CMAKE_BINARY_DIR} --quiet ${_src_files_str}")

        # Create a script to check for warnings
        set(_clang_tidy_check_script ${CMAKE_BINARY_DIR}/check-clang-tidy.sh)
        file(WRITE ${_clang_tidy_check_script}
            "#!/bin/bash\n"
            "set -e\n"
            "${_clang_tidy_cmd} > ${CLANG_TIDY_OUTPUT} 2>&1 || true\n"
            "WARNINGS=$(grep -E 'warning:' ${CLANG_TIDY_OUTPUT} | grep -v 'clang-diagnostic-error' | wc -l)\n"
            "if [ \"$WARNINGS\" -gt 0 ]; then\n"
            "    echo \"ERROR: clang-tidy found $WARNINGS warning(s). See ${CLANG_TIDY_OUTPUT} for details.\"\n"
            "    grep -E 'warning:' ${CLANG_TIDY_OUTPUT} | grep -v 'clang-diagnostic-error' | head -20\n"
            "    exit 1\n"
            "fi\n"
            "echo \"clang-tidy: No warnings found.\"\n"
        )

        add_custom_target(clang-tidy
            COMMAND chmod +x ${_clang_tidy_check_script}
            COMMAND bash ${_clang_tidy_check_script}
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
        message(STATUS "Use 'cmake --build . --target clang-tidy' to run static analysis")
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
        # Build a robust include path set for cppcheck.
        set(CPPCHECK_HYPRE_INCLUDES "")
        set(_cppcheck_hypre_include_candidates
            "${CMAKE_INSTALL_PREFIX}/include"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/utilities"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/struct_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/struct_ls"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/sstruct_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/sstruct_ls"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/multivector"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/krylov"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/parcsr_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/parcsr_ls"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/seq_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/seq_block_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/parcsr_block_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/IJ_mv"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/matrix_matrix"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/distributed_matrix"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/distributed_ls"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/blas"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src/lapack"
            "${CMAKE_BINARY_DIR}/_deps/hypre-build"
        )

        # Include dirs from imported target (if available).
        if(TARGET HYPRE::HYPRE)
            get_target_property(_hypre_target_includes HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
            if(_hypre_target_includes)
                list(APPEND _cppcheck_hypre_include_candidates ${_hypre_target_includes})
            endif()
        endif()

        # Include dirs from find_package variables.
        if(HYPRE_INCLUDE_DIRS)
            list(APPEND _cppcheck_hypre_include_candidates ${HYPRE_INCLUDE_DIRS})
        endif()
        if(MPI_C_INCLUDE_DIRS)
            list(APPEND _cppcheck_hypre_include_candidates ${MPI_C_INCLUDE_DIRS})
        endif()
        if(OpenMP_C_INCLUDE_DIRS)
            list(APPEND _cppcheck_hypre_include_candidates ${OpenMP_C_INCLUDE_DIRS})
        endif()

        foreach(INCDIR IN LISTS _cppcheck_hypre_include_candidates)
            if(INCDIR AND NOT INCDIR MATCHES "^\\$<")
                list(APPEND CPPCHECK_HYPRE_INCLUDES "-I${INCDIR}")
            endif()
        endforeach()

        list(REMOVE_DUPLICATES CPPCHECK_HYPRE_INCLUDES)

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

        # Create suppressions file for known false positives in external headers.
        # Format: [error id]:[filename]
        set(CPPCHECK_SUPPRESSIONS_FILE ${CMAKE_BINARY_DIR}/cppcheck-suppressions.txt)
        file(WRITE ${CPPCHECK_SUPPRESSIONS_FILE}
            "unreachableCode:src/error.c\n"
            "constParameterPointer:*_hypre_utilities.h\n"
            "constVariablePointer:*_hypre_utilities.h\n"
            "missingInclude:*_hypre_utilities.h\n"
        )

        # Simple cppcheck target - analyze only src/ directory
        # Runs cppcheck, generates HTML report, then fails if errors were found
        find_program(CPPCHECK_HTMLREPORT_EXECUTABLE NAMES cppcheck-htmlreport)

        # Fast defaults keep cppcheck practical for local iteration.
        # Turn this OFF when you need a deep, slower scan.
        option(HYPREDRV_CPPCHECK_DEEP "Run exhaustive cppcheck analysis" OFF)

        set(_cppcheck_depth_flags "")
        if(HYPREDRV_CPPCHECK_DEEP)
            list(APPEND _cppcheck_depth_flags
                --inconclusive
                --force
                --check-level=exhaustive
            )
            message(STATUS "cppcheck depth: deep (exhaustive)")
        else()
            list(APPEND _cppcheck_depth_flags
                --check-level=normal
            )
            message(STATUS "cppcheck depth: fast (normal)")
        endif()

        # Build the command list - always generate XML first
        set(_cppcheck_commands
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c99
                --suppress=unusedFunction
                --suppress=missingIncludeSystem
                --suppress=checkersReport
                --suppress=toomanyconfigs
                --suppress=unmatchedSuppression
                --suppressions-list=${CPPCHECK_SUPPRESSIONS_FILE}
                --inline-suppr
                ${_cppcheck_depth_flags}
                -UPETSC_AVAILABLE
                -UISIS_AVAILABLE
                -UHYPRE_USING_OPENMP
                -UHYPRE_USING_UMPIRE
                --xml
                --xml-version=2
                -I${CMAKE_SOURCE_DIR}/include
                -I${CMAKE_BINARY_DIR}
                ${CPPCHECK_JOBS_FLAG}
                ${CPPCHECK_HYPRE_INCLUDES}
                -i${CMAKE_INSTALL_PREFIX}/include
                -i${CMAKE_BINARY_DIR}/_deps/hypre-src/src
                -i${CMAKE_SOURCE_DIR}/src/info.c
                ${CMAKE_SOURCE_DIR}/src
                2> ${CPPCHECK_XML_OUTPUT}
        )

        # Add HTML report generation if available
        if(CPPCHECK_HTMLREPORT_EXECUTABLE)
            list(APPEND _cppcheck_commands
                COMMAND ${CPPCHECK_HTMLREPORT_EXECUTABLE}
                    --file=${CPPCHECK_XML_OUTPUT}
                    --report-dir=${CPPCHECK_HTML_DIR}
            )
            message(STATUS "HTML report will be generated automatically at ${CPPCHECK_HTML_DIR}/index.html")
        else()
            message(STATUS "cppcheck-htmlreport not found. Install it to generate HTML reports (e.g., pip install cppcheck-htmlreport)")
        endif()

        # Add error check - fail if XML contains <error> elements with severity="error" or "warning"
        # (excluding informational messages like checkersReport which have severity="information")
        # grep returns 0 if pattern found (errors exist), 1 if not found (no errors)
        # We invert with '!' so the command fails when errors are found
        list(APPEND _cppcheck_commands
            COMMAND ${CMAKE_COMMAND} -E echo "Checking for cppcheck errors..."
            COMMAND sh -c "! grep -E '<error[^>]*severity=\"(error|warning)\"' ${CPPCHECK_XML_OUTPUT}"
        )

        add_custom_target(cppcheck
            ${_cppcheck_commands}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running cppcheck static analysis on src/ directory"
            VERBATIM
        )

        message(STATUS "cppcheck target 'cppcheck' is available")
    endif()

    # ============================================================================
    # Environment variable hints
    # ============================================================================
    message(STATUS "")
    message(STATUS "Code analysis configuration:")
    message(STATUS "  Sanitizers: AddressSanitizer and UndefinedBehaviorSanitizer")
    message(STATUS "  Recommended environment variables for better stack traces:")
    message(STATUS "    ASAN_OPTIONS=symbolize=1:print_stacktrace=1:abort_on_error=1:detect_leaks=1")
    message(STATUS "    UBSAN_OPTIONS=symbolize=1:print_stacktrace=1:abort_on_error=1")
    message(STATUS "  Note: 'symbolize=1' requires 'llvm-symbolizer' or 'addr2line' to be available")
    message(STATUS "")
endif()
