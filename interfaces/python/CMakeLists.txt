# ----------------------------------------------------------------------------
# Python bindings for hypredrive (driven by scikit-build-core).
#
# This CMake project is invoked by `pip install ./interfaces/python`. It expects the
# hypredrive C library (HYPREDRV) to be discoverable. Two modes are supported:
#
#   1. Installed:   set CMAKE_PREFIX_PATH to the hypredrive install root, e.g.
#                   pip install ./interfaces/python \
#                     --config-settings=cmake.define.CMAKE_PREFIX_PATH=$INSTALL
#   2. In-tree dev: set HYPREDRV_DIR to a build-tree CMake config directory:
#                   pip install -e ./interfaces/python \
#                     --config-settings=cmake.define.HYPREDRV_DIR=$BUILD
#
# We rely on hypredrive exporting a CMake package config (HYPREDRVConfig.cmake)
# from its top-level CMakeLists.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23)
project(hypredrive_python LANGUAGES C)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_hypredrive_python_package_root "${CMAKE_CURRENT_BINARY_DIR}")
    set(_hypredrive_python_install_package_dir "hypredrive")
else()
    set(_hypredrive_python_package_root "${CMAKE_CURRENT_BINARY_DIR}/python")
    set(_hypredrive_python_install_package_dir "python")
endif()
set(_hypredrive_python_package_dir "${_hypredrive_python_package_root}/hypredrive")

# ---- Find HYPREDRV ---------------------------------------------------------
if(NOT TARGET HYPREDRV::HYPREDRV)
    find_package(HYPREDRV CONFIG REQUIRED)
endif()

# ---- Find Python + numpy + Cython ------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(MPI COMPONENTS C QUIET)

# Discover NumPy headers via the interpreter (avoid relying on FindNumPy).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE _numpy_include
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _numpy_rc
)
if(NOT _numpy_rc EQUAL 0)
    message(FATAL_ERROR "Failed to locate NumPy include directory; is numpy installed?")
endif()

# Cython is supplied via PEP 517 build requirements. Invoke it as a module from
# the selected Python interpreter so isolated/venv builds do not accidentally use
# an older system `cython` executable from PATH.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import Cython"
    RESULT_VARIABLE _cython_rc
)
if(NOT _cython_rc EQUAL 0)
    message(FATAL_ERROR "Failed to import Cython from ${Python_EXECUTABLE}; install cython>=3.0 to build the Python interface")
endif()
set(_cython_exe "${Python_EXECUTABLE};-m;cython")

# ---- Cython -> C compilation ----------------------------------------------
set(_pyx_in  "${CMAKE_CURRENT_SOURCE_DIR}/src/_native.pyx")
set(_pyx_out "${CMAKE_CURRENT_BINARY_DIR}/_native.c")
set(_native_abi_h "${CMAKE_CURRENT_SOURCE_DIR}/src/_native_abi.h")

add_custom_command(
    OUTPUT  "${_pyx_out}"
    COMMAND ${_cython_exe} -3
            --module-name hypredrive._native
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -o "${_pyx_out}"
            "${_pyx_in}"
    DEPENDS "${_pyx_in}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/_native.pxd"
            "${_native_abi_h}"
    COMMENT "Cythonizing _native.pyx -> _native.c"
    VERBATIM
)

python_add_library(_native MODULE WITH_SOABI "${_pyx_out}")
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_hypredrive_python_package_dir}"
)

target_include_directories(_native PRIVATE
    "${_numpy_include}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(_native PRIVATE
    HYPREDRV::HYPREDRV
)

target_compile_definitions(_native PRIVATE
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
)

set(_python_package_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/driver.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/errors.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/options.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/result.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/session.py"
)

set(_python_package_build_dir "${_hypredrive_python_package_dir}")
set(_python_package_outputs)
foreach(_python_package_source IN LISTS _python_package_sources)
    get_filename_component(_python_package_name "${_python_package_source}" NAME)
    set(_python_package_output "${_python_package_build_dir}/${_python_package_name}")
    add_custom_command(
        OUTPUT "${_python_package_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_python_package_build_dir}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_python_package_source}"
                "${_python_package_output}"
        DEPENDS "${_python_package_source}"
        VERBATIM
    )
    list(APPEND _python_package_outputs "${_python_package_output}")
endforeach()

add_custom_target(hypredrive_python_sources
    DEPENDS ${_python_package_outputs}
)

# Install the extension into the package directory inside the wheel.
install(FILES ${_python_package_sources}
        DESTINATION "${_hypredrive_python_install_package_dir}"
)
install(TARGETS _native
        LIBRARY DESTINATION "${_hypredrive_python_install_package_dir}"
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mpi4py"
    RESULT_VARIABLE _mpi4py_rc
)

set(_python_test_env
    "PYTHONPATH=${_hypredrive_python_package_root}"
)

if(HYPREDRV_SANITIZER_ENABLED AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -print-file-name=libasan.so
        OUTPUT_VARIABLE _asan_runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -print-file-name=libubsan.so
        OUTPUT_VARIABLE _ubsan_runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(_python_sanitizer_preload "")
    if(IS_ABSOLUTE "${_asan_runtime}" AND EXISTS "${_asan_runtime}")
        list(APPEND _python_sanitizer_preload "${_asan_runtime}")
    endif()
    if(IS_ABSOLUTE "${_ubsan_runtime}" AND EXISTS "${_ubsan_runtime}")
        list(APPEND _python_sanitizer_preload "${_ubsan_runtime}")
    endif()
    if(_python_sanitizer_preload)
        list(JOIN _python_sanitizer_preload ":" _python_sanitizer_preload_value)
        list(APPEND _python_test_env
            "LD_PRELOAD=${_python_sanitizer_preload_value}"
            "ASAN_OPTIONS=detect_leaks=0:symbolize=1:print_stacktrace=1:abort_on_error=1"
            "UBSAN_OPTIONS=print_stacktrace=1:abort_on_error=1"
        )
    else()
        message(WARNING "Sanitizers are enabled, but no ASan/UBSan runtime could be resolved for Python tests")
    endif()
endif()

set(_python_test_commands
    COMMAND "${CMAKE_COMMAND}" -E env
            ${_python_test_env}
            "${Python_EXECUTABLE}" -m pytest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_options.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_solve_serial.py"
)

if(MPIEXEC_EXECUTABLE AND _mpi4py_rc EQUAL 0)
    list(APPEND _python_test_commands
        COMMAND "${CMAKE_COMMAND}" -E env
                ${_python_test_env}
                "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
                ${MPIEXEC_PREFLAGS}
                "${Python_EXECUTABLE}" -m pytest
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_solve_mpi.py"
                ${MPIEXEC_POSTFLAGS}
    )
else()
    message(STATUS "Python MPI tests are not attached to python-test because MPIEXEC or mpi4py was not found")
endif()

add_custom_target(python-test
    ${_python_test_commands}
    DEPENDS _native hypredrive_python_sources
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running hypredrive Python tests"
    COMMAND_EXPAND_LISTS
    VERBATIM
)
