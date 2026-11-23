# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.23)

# Cargo configures this directory independently to discover an installed library.
# In a hypredrive build, the live target also works when exports are unavailable.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(HYPREDRV_RustMetadata LANGUAGES C)
    # Cargo passes the crate's major.minor version, its single source of truth.
    if(NOT HYPREDRV_RUST_REQUIRED_VERSION)
        message(FATAL_ERROR "Set HYPREDRV_RUST_REQUIRED_VERSION (Cargo's build script does)")
    endif()
    find_package(HYPREDRV ${HYPREDRV_RUST_REQUIRED_VERSION} CONFIG REQUIRED)
    set(_hypredrv_rust_standalone TRUE)
endif()

get_target_property(_hypredrv_rust_library_type HYPREDRV::HYPREDRV TYPE)
if(NOT _hypredrv_rust_library_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
        "The Rust interface requires shared libHYPREDRV. Configure hypredrive "
        "with BUILD_SHARED_LIBS=ON and select its shared CMake package.")
endif()
if(NOT TARGET MPI::MPI_C)
    find_package(MPI COMPONENTS C REQUIRED)
endif()
if(NOT MPI_C_COMPILER)
    message(FATAL_ERROR "The Rust interface requires an MPI C compiler (MPI_C_COMPILER)")
endif()

if(_hypredrv_rust_standalone)
    if(NOT HYPREDRV_HYPRE_INT_BYTES OR NOT HYPREDRV_HYPRE_BIGINT_BYTES OR
       NOT HYPREDRV_HYPRE_REAL_BYTES OR NOT HYPREDRV_HYPRE_COMPLEX_BYTES)
        message(FATAL_ERROR
            "The Rust interface requires HYPREDRV producer ABI metadata. "
            "Rebuild and reinstall hypredrive with its current CMake package.")
    endif()
    if(NOT HYPREDRV_HYPRE_REAL_BYTES EQUAL 8 OR NOT HYPREDRV_HYPRE_COMPLEX_BYTES EQUAL 8)
        message(FATAL_ERROR
            "The Rust interface requires real, double-precision HYPRE in the "
            "producer libHYPREDRV build.")
    endif()
    include("${HYPREDRV_DIR}/HYPREDRV_HypreABI.cmake")
    if(NOT HYPREDRV_MPI_COMM_BYTES OR NOT HYPREDRV_MPI_COMM_KIND OR NOT HYPREDRV_MPI_FAMILY)
        message(FATAL_ERROR
            "The Rust interface requires HYPREDRV producer MPI ABI metadata. "
            "Rebuild and reinstall hypredrive with its current CMake package.")
    endif()
    hypredrv_get_mpi_abi(_hypredrv_rust_mpi_comm_bytes
        _hypredrv_rust_mpi_comm_kind _hypredrv_rust_mpi_family)
    if(NOT _hypredrv_rust_mpi_comm_bytes STREQUAL HYPREDRV_MPI_COMM_BYTES OR
       NOT _hypredrv_rust_mpi_comm_kind STREQUAL HYPREDRV_MPI_COMM_KIND OR
       NOT _hypredrv_rust_mpi_family STREQUAL HYPREDRV_MPI_FAMILY)
        message(FATAL_ERROR
            "MPI ABI mismatch. Select the MPI installation used to build libHYPREDRV.")
    endif()
    hypredrv_get_hypre_abi_sizes(_hypredrv_rust_int_bytes _hypredrv_rust_bigint_bytes
        _hypredrv_rust_real_bytes _hypredrv_rust_complex_bytes)
    if(NOT _hypredrv_rust_real_bytes EQUAL 8 OR NOT _hypredrv_rust_complex_bytes EQUAL 8)
        message(FATAL_ERROR "The Rust interface requires real, double-precision HYPRE")
    endif()
    if(NOT _hypredrv_rust_int_bytes STREQUAL HYPREDRV_HYPRE_INT_BYTES OR
       NOT _hypredrv_rust_bigint_bytes STREQUAL HYPREDRV_HYPRE_BIGINT_BYTES)
        message(FATAL_ERROR
            "HYPRE integer ABI mismatch. libHYPREDRV was built with "
            "HYPRE_Int=${HYPREDRV_HYPRE_INT_BYTES} and "
            "HYPRE_BigInt=${HYPREDRV_HYPRE_BIGINT_BYTES} bytes, but the selected "
            "headers use HYPRE_Int=${_hypredrv_rust_int_bytes} and "
            "HYPRE_BigInt=${_hypredrv_rust_bigint_bytes} bytes. "
            "Select the HYPRE installation used to build libHYPREDRV.")
    endif()
endif()

# Diagnose unsupported scalars before compiling the core library, which itself
# assumes real arithmetic in several paths. Bindgen repeats this check against
# the exact resolved headers when Cargo runs.
if(COMMAND hypredrv_collect_hypre_include_dirs)
    hypredrv_collect_hypre_include_dirs(_hypredrv_rust_hypre_includes)
else()
    get_target_property(_hypredrv_rust_hypre_includes
        HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
endif()
set(_hypredrv_rust_found_config FALSE)
foreach(_hypredrv_rust_include IN LISTS _hypredrv_rust_hypre_includes)
    if(EXISTS "${_hypredrv_rust_include}/HYPRE_config.h")
        set(_hypredrv_rust_found_config TRUE)
        file(READ "${_hypredrv_rust_include}/HYPRE_config.h" _hypredrv_rust_hypre_config)
        if(_hypredrv_rust_hypre_config MATCHES
           "#[ \t]*define[ \t]+HYPRE_(COMPLEX|SINGLE|LONG_DOUBLE)([ \t\r\n]|$)")
            message(FATAL_ERROR "The Rust interface requires real, double-precision HYPRE")
        endif()
        break()
    endif()
endforeach()
if(NOT _hypredrv_rust_found_config AND
   (HYPRE_ENABLE_COMPLEX OR HYPRE_ENABLE_SINGLE OR HYPRE_ENABLE_LONG_DOUBLE))
    message(FATAL_ERROR "The Rust interface requires real, double-precision HYPRE")
endif()

# Evaluate usage requirements through the dependency graph, including HYPRE and
# MPI. This includes generated configuration headers and compile definitions.
add_library(_hypredrv_rust_metadata INTERFACE)
target_link_libraries(_hypredrv_rust_metadata INTERFACE
    HYPREDRV::HYPREDRV MPI::MPI_C)
# FindMPI omits include flags when CMAKE_C_COMPILER is already an MPI wrapper.
# Libclang needs that wrapper's MPI include directory explicitly.
foreach(_hypredrv_rust_implicit_include IN LISTS CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    if(EXISTS "${_hypredrv_rust_implicit_include}/mpi.h")
        target_include_directories(_hypredrv_rust_metadata INTERFACE
            "${_hypredrv_rust_implicit_include}")
    endif()
endforeach()
set(_hypredrv_rust_metadata_file "${CMAKE_CURRENT_BINARY_DIR}/native-$<CONFIG>.txt")
set(_hypredrv_rust_package_configs "")
foreach(_hypredrv_rust_config_dir IN ITEMS "${HYPREDRV_DIR}" "${HYPRE_DIR}")
    if(_hypredrv_rust_config_dir AND EXISTS "${_hypredrv_rust_config_dir}")
        file(GLOB _hypredrv_rust_dir_configs "${_hypredrv_rust_config_dir}/*.cmake")
        list(APPEND _hypredrv_rust_package_configs ${_hypredrv_rust_dir_configs})
    endif()
endforeach()
file(GENERATE OUTPUT "${_hypredrv_rust_metadata_file}"
    CONTENT "library=$<TARGET_FILE:HYPREDRV::HYPREDRV>
link_dir=$<TARGET_LINKER_FILE_DIR:HYPREDRV::HYPREDRV>
mpi_compiler=${MPI_C_COMPILER}
include=$<JOIN:$<TARGET_PROPERTY:_hypredrv_rust_metadata,INTERFACE_INCLUDE_DIRECTORIES>,\ninclude=>
define=$<JOIN:$<TARGET_PROPERTY:_hypredrv_rust_metadata,INTERFACE_COMPILE_DEFINITIONS>,\ndefine=>
config=$<JOIN:${_hypredrv_rust_package_configs},\nconfig=>
"
    TARGET _hypredrv_rust_metadata)

if(_hypredrv_rust_standalone)
    return()
endif()

find_program(Cargo_EXECUTABLE NAMES cargo REQUIRED)
# Rust executables have no CMake-generated RPATH. Keep the existing test helper's
# paths and add MPI libraries hidden in a compiler wrapper's implicit link path.
set(_hypredrv_rust_mpi_runtime_dirs "")
foreach(_hypredrv_rust_link_dir IN LISTS CMAKE_C_IMPLICIT_LINK_DIRECTORIES)
    if(EXISTS "${_hypredrv_rust_link_dir}/libmpi.so" OR
       EXISTS "${_hypredrv_rust_link_dir}/libmpi.dylib")
        list(APPEND _hypredrv_rust_mpi_runtime_dirs "${_hypredrv_rust_link_dir}")
    endif()
endforeach()
foreach(_hypredrv_rust_mpi_library IN LISTS MPI_C_LIBRARIES)
    if(IS_ABSOLUTE "${_hypredrv_rust_mpi_library}")
        get_filename_component(_hypredrv_rust_link_dir
            "${_hypredrv_rust_mpi_library}" DIRECTORY)
        list(APPEND _hypredrv_rust_mpi_runtime_dirs "${_hypredrv_rust_link_dir}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _hypredrv_rust_mpi_runtime_dirs)
if(UNIX)
    if(APPLE)
        set(_hypredrv_rust_loader_var DYLD_LIBRARY_PATH)
    else()
        set(_hypredrv_rust_loader_var LD_LIBRARY_PATH)
    endif()
    string(JOIN ":" _hypredrv_rust_mpi_runtime_path
        "$<TARGET_FILE_DIR:HYPREDRV::HYPREDRV>" ${_hypredrv_rust_mpi_runtime_dirs})
    if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
        string(REGEX REPLACE "^[^=]*=" "" _hypredrv_rust_existing_runtime
            "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    else()
        set(_hypredrv_rust_existing_runtime "$ENV{${_hypredrv_rust_loader_var}}")
    endif()
    string(REGEX REPLACE ":+$" "" _hypredrv_rust_existing_runtime
        "${_hypredrv_rust_existing_runtime}")
    if(_hypredrv_rust_existing_runtime)
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "${_hypredrv_rust_loader_var}=${_hypredrv_rust_mpi_runtime_path}:${_hypredrv_rust_existing_runtime}")
    else()
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "${_hypredrv_rust_loader_var}=${_hypredrv_rust_mpi_runtime_path}")
    endif()
endif()
# Cargo output and generated bindings stay out of the source tree. The committed
# lockfile is staged with the sources and enforced with --locked, so dependency
# resolution cannot drift between builds.
set(_hypredrv_rust_workspace "${CMAKE_CURRENT_BINARY_DIR}/workspace")
set(_hypredrv_rust_sources Cargo.toml Cargo.lock build.rs wrapper.h CMakeLists.txt)
file(GLOB_RECURSE _hypredrv_rust_workspace_sources CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "src/*.rs" "examples/*.rs" "tests/*.rs" "cmake/*.cmake")
list(APPEND _hypredrv_rust_sources ${_hypredrv_rust_workspace_sources})
# Cargo auto-discovers sources, so files deleted or renamed in the source tree
# must also disappear from the staged copy.
file(GLOB_RECURSE _hypredrv_rust_staged_sources
    RELATIVE "${_hypredrv_rust_workspace}"
    "${_hypredrv_rust_workspace}/src/*.rs" "${_hypredrv_rust_workspace}/examples/*.rs"
    "${_hypredrv_rust_workspace}/tests/*.rs" "${_hypredrv_rust_workspace}/cmake/*.cmake")
foreach(_hypredrv_rust_staged IN LISTS _hypredrv_rust_staged_sources)
    if(NOT _hypredrv_rust_staged IN_LIST _hypredrv_rust_sources)
        file(REMOVE "${_hypredrv_rust_workspace}/${_hypredrv_rust_staged}")
    endif()
endforeach()
foreach(_hypredrv_rust_source IN LISTS _hypredrv_rust_sources)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${_hypredrv_rust_source}"
        "${_hypredrv_rust_workspace}/${_hypredrv_rust_source}" COPYONLY)
endforeach()
set(_hypredrv_rust_configuration_dir "${CMAKE_CURRENT_BINARY_DIR}")
if(CMAKE_CONFIGURATION_TYPES)
    string(APPEND _hypredrv_rust_configuration_dir "/$<CONFIG>")
endif()
set(_hypredrv_rust_target_dir "${_hypredrv_rust_configuration_dir}/cargo-target")
set(_hypredrv_rust_env
    "HYPREDRV_RUST_METADATA=${_hypredrv_rust_metadata_file}"
    "CARGO_TARGET_DIR=${_hypredrv_rust_target_dir}"
    "MPICC=${MPI_C_COMPILER}")
if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
    list(APPEND _hypredrv_rust_env "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
endif()
set(_hypredrv_rust_executable_env "")
if(HYPREDRV_SANITIZER_ENABLED AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(_hypredrv_rust_sanitizer_names
            "libclang_rt.asan-${CMAKE_SYSTEM_PROCESSOR}.so")
    else()
        set(_hypredrv_rust_sanitizer_names libasan.so libubsan.so)
    endif()
    set(_hypredrv_rust_sanitizer_runtimes "")
    foreach(_hypredrv_rust_sanitizer IN LISTS _hypredrv_rust_sanitizer_names)
        execute_process(
            COMMAND "${CMAKE_C_COMPILER}" "-print-file-name=${_hypredrv_rust_sanitizer}"
            OUTPUT_VARIABLE _hypredrv_rust_sanitizer_runtime
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
        if(NOT IS_ABSOLUTE "${_hypredrv_rust_sanitizer_runtime}" OR
           NOT EXISTS "${_hypredrv_rust_sanitizer_runtime}")
            message(FATAL_ERROR "Cannot find ${_hypredrv_rust_sanitizer} for Rust sanitizer tests")
        endif()
        list(APPEND _hypredrv_rust_sanitizer_runtimes
            "${_hypredrv_rust_sanitizer_runtime}")
    endforeach()
    list(JOIN _hypredrv_rust_sanitizer_runtimes ":" _hypredrv_rust_sanitizer_preload)
    if(DEFINED ENV{LD_PRELOAD} AND NOT "$ENV{LD_PRELOAD}" STREQUAL "")
        string(APPEND _hypredrv_rust_sanitizer_preload ":$ENV{LD_PRELOAD}")
    endif()
    # Apply the matching native runtimes only to the Rust executable, after
    # the MPI launcher, so Cargo, rustc, and the launcher stay uninstrumented.
    list(APPEND _hypredrv_rust_executable_env
        "LD_PRELOAD=${_hypredrv_rust_sanitizer_preload}")
    # Clang's shared library can retain unresolved sanitizer symbols. Link the
    # same runtime into Rust executables as well, then preload it for ordering.
    # Cargo's encoded form preserves each argument and any caller-supplied flags.
    string(ASCII 31 _hypredrv_rust_flag_separator)
    if(DEFINED ENV{CARGO_ENCODED_RUSTFLAGS})
        set(_hypredrv_rust_flags "$ENV{CARGO_ENCODED_RUSTFLAGS}")
    else()
        separate_arguments(_hypredrv_rust_flag_list UNIX_COMMAND "$ENV{RUSTFLAGS}")
        list(JOIN _hypredrv_rust_flag_list "${_hypredrv_rust_flag_separator}"
            _hypredrv_rust_flags)
    endif()
    foreach(_hypredrv_rust_sanitizer_runtime IN LISTS _hypredrv_rust_sanitizer_runtimes)
        if(NOT _hypredrv_rust_flags STREQUAL "")
            string(APPEND _hypredrv_rust_flags "${_hypredrv_rust_flag_separator}")
        endif()
        string(APPEND _hypredrv_rust_flags
            "-C${_hypredrv_rust_flag_separator}link-arg=${_hypredrv_rust_sanitizer_runtime}")
    endforeach()
    list(APPEND _hypredrv_rust_env "CARGO_ENCODED_RUSTFLAGS=${_hypredrv_rust_flags}")
endif()
set(_hypredrv_rust_test_executable
    "${_hypredrv_rust_configuration_dir}/smoke${CMAKE_EXECUTABLE_SUFFIX}")

add_custom_target(rust-build
    COMMAND ${CMAKE_COMMAND} -E env
        ${_hypredrv_rust_env}
        ${Cargo_EXECUTABLE} build --locked
        --manifest-path ${_hypredrv_rust_workspace}/Cargo.toml --examples
    COMMAND ${CMAKE_COMMAND} -E env
        ${_hypredrv_rust_env}
        ${CMAKE_COMMAND}
        -DCARGO_EXECUTABLE=${Cargo_EXECUTABLE}
        -DMANIFEST=${_hypredrv_rust_workspace}/Cargo.toml
        -DTEST_EXECUTABLE=${_hypredrv_rust_test_executable}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_tests.cmake
    DEPENDS HYPREDRV
    COMMENT "Building the Rust interface, examples, and MPI test executable"
    USES_TERMINAL VERBATIM)

if(HYPREDRV_ENABLE_TESTING)
    add_test(NAME rust_build
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
            --target rust-build --config $<CONFIG>)
    set_tests_properties(rust_build PROPERTIES
        LABELS "rust;interface;build" FIXTURES_SETUP rust_built)

    set(_hypredrv_rust_launcher_args
        "-DTARGET_BIN=${_hypredrv_rust_test_executable}"
        "-DMPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        "-DTEST_RUNTIME_ENV=${_hypredrv_rust_executable_env}"
        "-DTEST_LAUNCHER=${PROJECT_SOURCE_DIR}/cmake/HYPREDRV_TestLauncher.cmake")
    set(_hypredrv_rust_launcher_script
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_test.cmake")
    add_test(NAME rust_smoke_serial
        COMMAND ${CMAKE_COMMAND} ${_hypredrv_rust_launcher_args}
            -DMPI_NUMPROCS=1 -P "${_hypredrv_rust_launcher_script}")
    set_tests_properties(rust_smoke_serial PROPERTIES
        LABELS "rust;interface" FIXTURES_REQUIRED rust_built TIMEOUT 120)
    if(COMMAND hypredrv_set_gpu_test_resources)
        hypredrv_set_gpu_test_resources(rust_smoke_serial 1)
    endif()
    hypredrv_append_test_environment(rust_smoke_serial)

    if(MPIEXEC_EXECUTABLE)
        foreach(_hypredrv_rust_ranks IN ITEMS 2 4)
            set(_hypredrv_rust_test "rust_smoke_mpi_${_hypredrv_rust_ranks}proc")
            add_test(NAME ${_hypredrv_rust_test}
                COMMAND ${CMAKE_COMMAND} ${_hypredrv_rust_launcher_args}
                    -DMPI_NUMPROCS=${_hypredrv_rust_ranks}
                    -P "${_hypredrv_rust_launcher_script}")
            set_tests_properties(${_hypredrv_rust_test} PROPERTIES
                LABELS "rust;interface;mpi" FIXTURES_REQUIRED rust_built
                PROCESSORS ${_hypredrv_rust_ranks} RUN_SERIAL TRUE TIMEOUT 120)
            if(COMMAND hypredrv_set_gpu_test_resources)
                hypredrv_set_gpu_test_resources(${_hypredrv_rust_test} ${_hypredrv_rust_ranks})
            endif()
            hypredrv_append_test_environment(${_hypredrv_rust_test})
        endforeach()
    endif()

    add_custom_target(rust-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L rust
            -E "^rust_build$" --fixture-exclude-setup "^rust_built$" -C $<CONFIG>
        DEPENDS rust-build WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running hypredrive Rust tests" USES_TERMINAL VERBATIM)
else()
    add_custom_target(rust-test
        COMMAND ${CMAKE_COMMAND} -E echo
            "Configure with HYPREDRV_ENABLE_TESTING=ON to run Rust tests"
        COMMAND ${CMAKE_COMMAND} -E false VERBATIM)
endif()
