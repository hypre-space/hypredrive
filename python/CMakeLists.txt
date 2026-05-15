# ----------------------------------------------------------------------------
# Python bindings for hypredrive (driven by scikit-build-core).
#
# This CMake project is invoked by `pip install ./python`. It expects the
# hypredrive C library (HYPREDRV) to be discoverable. Two modes are supported:
#
#   1. Installed:   set CMAKE_PREFIX_PATH to the hypredrive install root, e.g.
#                   pip install ./python \
#                     --config-settings=cmake.define.CMAKE_PREFIX_PATH=$INSTALL
#   2. In-tree dev: set HYPREDRV_DIR to a build-tree CMake config directory:
#                   pip install -e ./python \
#                     --config-settings=cmake.define.HYPREDRV_DIR=$BUILD
#
# We rely on hypredrive exporting a CMake package config (HYPREDRVConfig.cmake)
# from its top-level CMakeLists.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23)
project(hypredrive_python LANGUAGES C)

# ---- Find HYPREDRV ---------------------------------------------------------
find_package(HYPREDRV CONFIG REQUIRED)

# ---- Find Python + numpy + Cython ------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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

# Cython is supplied via PEP 517 build requirements; locate the executable.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import shutil, sys; p = shutil.which('cython'); print(p) if p else sys.exit(1)"
    OUTPUT_VARIABLE _cython_exe
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cython_rc
)
if(NOT _cython_rc EQUAL 0)
    # Fall back to invoking cython (the translator CLI) as a python module.
    set(_cython_exe "${Python_EXECUTABLE};-m;cython")
endif()

# ---- Cython -> C compilation ----------------------------------------------
set(_pyx_in  "${CMAKE_CURRENT_SOURCE_DIR}/src/hypredrive/_native.pyx")
set(_pyx_out "${CMAKE_CURRENT_BINARY_DIR}/_native.c")
set(_native_abi_h "${CMAKE_CURRENT_SOURCE_DIR}/src/hypredrive/_native_abi.h")

add_custom_command(
    OUTPUT  "${_pyx_out}"
    COMMAND ${_cython_exe} -3 -o "${_pyx_out}" "${_pyx_in}"
    DEPENDS "${_pyx_in}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/hypredrive/_native.pxd"
            "${_native_abi_h}"
    COMMENT "Cythonizing _native.pyx -> _native.c"
    VERBATIM
)

python_add_library(_native MODULE WITH_SOABI "${_pyx_out}")

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

# Install the extension into the package directory inside the wheel.
install(TARGETS _native
        LIBRARY DESTINATION hypredrive
)
