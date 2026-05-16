# ----------------------------------------------------------------------------
# Python bindings for hypredrive (driven by scikit-build-core).
#
# This CMake project is invoked by `pip install ./interfaces/python`. It expects the
# hypredrive C library (HYPREDRV) to be discoverable. Three modes are supported:
#
#   1. Installed:   set CMAKE_PREFIX_PATH to the hypredrive install root, e.g.
#                   pip install ./interfaces/python \
#                     --config-settings=cmake.define.CMAKE_PREFIX_PATH=$INSTALL
#   2. In-tree dev: set HYPREDRV_DIR to a build-tree CMake config directory:
#                   pip install -e ./interfaces/python \
#                     --config-settings=cmake.define.HYPREDRV_DIR=$BUILD
#   3. Local/wheel: HYPREDRV_PYTHON_BUNDLE_CORE=AUTO builds and bundles
#                   HYPREDRV/HYPRE automatically when this package is built
#                   from a full hypredrive source checkout. Set it to ON to
#                   require bundling, or OFF to require find_package.
#
# We rely on hypredrive exporting a CMake package config (HYPREDRVConfig.cmake)
# from its top-level CMakeLists.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23)
project(hypredrive_python LANGUAGES C)

set(HYPREDRV_PYTHON_BUNDLE_CORE "AUTO" CACHE STRING
    "Build and bundle in-tree HYPREDRV/HYPRE for Python wheels (AUTO, ON, or OFF)")
set_property(CACHE HYPREDRV_PYTHON_BUNDLE_CORE PROPERTY STRINGS AUTO ON OFF)
set(HYPREDRV_PYTHON_DIST_NAME "" CACHE STRING
    "Python distribution name recorded in package build metadata")
set(HYPREDRV_PYTHON_MPI_FLAVOR "" CACHE STRING
    "MPI flavor recorded in package build metadata")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_hypredrive_python_standalone_project ON)
    set(_hypredrive_python_package_root "${CMAKE_CURRENT_BINARY_DIR}")
    set(_hypredrive_python_install_package_dir "hypredrive")
else()
    set(_hypredrive_python_standalone_project OFF)
    set(_hypredrive_python_package_root "${CMAKE_CURRENT_BINARY_DIR}/python")
    set(_hypredrive_python_install_package_dir "python")
endif()
set(_hypredrive_python_package_dir "${_hypredrive_python_package_root}/hypredrive")

get_filename_component(_hypredrive_python_repo_root
                       "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                       ABSOLUTE)
set(_hypredrive_python_has_repo_root OFF)
if(EXISTS "${_hypredrive_python_repo_root}/CMakeLists.txt"
   AND EXISTS "${_hypredrive_python_repo_root}/include/HYPREDRV.h")
    set(_hypredrive_python_has_repo_root ON)
endif()

string(TOUPPER "${HYPREDRV_PYTHON_BUNDLE_CORE}" _hypredrive_python_bundle_mode)
if(NOT _hypredrive_python_bundle_mode MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "HYPREDRV_PYTHON_BUNDLE_CORE must be AUTO, ON, or OFF; got "
        "'${HYPREDRV_PYTHON_BUNDLE_CORE}'")
endif()
set(_hypredrive_python_bundle_core OFF)
if(_hypredrive_python_bundle_mode STREQUAL "ON")
    set(_hypredrive_python_bundle_core ON)
elseif(_hypredrive_python_bundle_mode STREQUAL "AUTO"
       AND _hypredrive_python_standalone_project
       AND _hypredrive_python_has_repo_root)
    set(_hypredrive_python_bundle_core ON)
endif()

if(NOT HYPREDRV_PYTHON_MPI_FLAVOR)
    if(DEFINED ENV{HYPREDRV_WHEEL_MPI_FLAVOR} AND NOT "$ENV{HYPREDRV_WHEEL_MPI_FLAVOR}" STREQUAL "")
        set(HYPREDRV_PYTHON_MPI_FLAVOR "$ENV{HYPREDRV_WHEEL_MPI_FLAVOR}")
    else()
        set(HYPREDRV_PYTHON_MPI_FLAVOR "source")
    endif()
endif()
if(NOT HYPREDRV_PYTHON_DIST_NAME)
    if(HYPREDRV_PYTHON_MPI_FLAVOR STREQUAL "mpich")
        set(HYPREDRV_PYTHON_DIST_NAME "hypredrive-mpich")
    elseif(HYPREDRV_PYTHON_MPI_FLAVOR STREQUAL "openmpi")
        set(HYPREDRV_PYTHON_DIST_NAME "hypredrive-openmpi")
    else()
        set(HYPREDRV_PYTHON_DIST_NAME "hypredrive")
    endif()
endif()
if(_hypredrive_python_bundle_core)
    set(HYPREDRV_PYTHON_BUNDLED_CORE_PY "True")
else()
    set(HYPREDRV_PYTHON_BUNDLED_CORE_PY "False")
endif()

# ---- Find HYPREDRV ---------------------------------------------------------
if(_hypredrive_python_bundle_core AND NOT TARGET HYPREDRV::HYPREDRV)
    if(NOT _hypredrive_python_has_repo_root)
        message(FATAL_ERROR
            "HYPREDRV_PYTHON_BUNDLE_CORE=${HYPREDRV_PYTHON_BUNDLE_CORE} requires building from the "
            "hypredrive source checkout, but ${_hypredrive_python_repo_root} "
            "does not look like the repository root")
    endif()

    if(NOT DEFINED BUILD_SHARED_LIBS)
        set(BUILD_SHARED_LIBS ON CACHE BOOL
            "Build shared libraries for bundled Python wheels")
    endif()
    set(HYPREDRV_ENABLE_PYTHON OFF CACHE BOOL
        "Disable nested Python interface while bundling core" FORCE)
    set(HYPREDRV_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_ANALYSIS OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_DATA OFF CACHE BOOL "" FORCE)
    set(HYPREDRV_ENABLE_COMPRESSION OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_COMPLEX OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_SINGLE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_LONG_DOUBLE OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_HIP OFF CACHE BOOL "" FORCE)
    set(HYPRE_ENABLE_SYCL OFF CACHE BOOL "" FORCE)

    add_subdirectory("${_hypredrive_python_repo_root}"
                     "${CMAKE_CURRENT_BINARY_DIR}/_hypredrive_core"
                     EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET HYPREDRV::HYPREDRV)
    find_package(HYPREDRV CONFIG REQUIRED)
endif()

# ---- Find Python + numpy + Cython ------------------------------------------
if(DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED Python_EXECUTABLE)
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python" CACHE FILEPATH
        "Python interpreter from the active virtual environment")
endif()

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
set(_pyx_in  "${CMAKE_CURRENT_SOURCE_DIR}/src/_core.pyx")
set(_pyx_out "${CMAKE_CURRENT_BINARY_DIR}/_core.c")
set(_bridge_c "${CMAKE_CURRENT_SOURCE_DIR}/src/HYPREDRV_python.c")
set(_bridge_h "${CMAKE_CURRENT_SOURCE_DIR}/src/HYPREDRV_python.h")

add_custom_command(
    OUTPUT  "${_pyx_out}"
    COMMAND ${_cython_exe} -3
            --module-name hypredrive._core
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -o "${_pyx_out}"
            "${_pyx_in}"
    DEPENDS "${_pyx_in}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/_core.pxd"
            "${_bridge_h}"
    COMMENT "Cythonizing _core.pyx -> _core.c"
    VERBATIM
)

python_add_library(_core MODULE WITH_SOABI "${_pyx_out}" "${_bridge_c}")
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_hypredrive_python_package_dir}"
)

if(_hypredrive_python_bundle_core)
    if(APPLE)
        set(_hypredrive_python_relative_rpath "@loader_path/.libs")
        set(_hypredrive_python_lib_relative_rpath "@loader_path")
    else()
        set(_hypredrive_python_relative_rpath "$ORIGIN/.libs")
        set(_hypredrive_python_lib_relative_rpath "$ORIGIN")
    endif()
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "${_hypredrive_python_relative_rpath}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
    if(TARGET HYPREDRV)
        set_target_properties(HYPREDRV PROPERTIES
            PUBLIC_HEADER ""
            INSTALL_RPATH "${_hypredrive_python_lib_relative_rpath}"
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )
    endif()
    if(TARGET HYPRE)
        set_target_properties(HYPRE PROPERTIES
            INSTALL_RPATH "${_hypredrive_python_lib_relative_rpath}"
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )
    endif()

endif()

target_include_directories(_core PRIVATE
    "${_numpy_include}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(_core PRIVATE
    HYPREDRV::HYPREDRV
)

target_compile_definitions(_core PRIVATE
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

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/_build_info.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
    @ONLY
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
set(_python_build_info_output "${_python_package_build_dir}/_build_info.py")
add_custom_command(
    OUTPUT "${_python_build_info_output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_python_package_build_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
            "${_python_build_info_output}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
    VERBATIM
)
list(APPEND _python_package_outputs "${_python_build_info_output}")

add_custom_target(hypredrive_python_sources
    DEPENDS ${_python_package_outputs}
)

# Install the extension into the package directory inside the wheel.
install(FILES ${_python_package_sources}
        DESTINATION "${_hypredrive_python_install_package_dir}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py"
        DESTINATION "${_hypredrive_python_install_package_dir}"
)
install(TARGETS _core
        LIBRARY DESTINATION "${_hypredrive_python_install_package_dir}"
)

if(_hypredrive_python_bundle_core)
    if(NOT TARGET HYPREDRV)
        message(FATAL_ERROR
            "HYPREDRV_PYTHON_BUNDLE_CORE=${HYPREDRV_PYTHON_BUNDLE_CORE} requires a concrete HYPREDRV target")
    endif()
    install(TARGETS HYPREDRV
            LIBRARY DESTINATION "${_hypredrive_python_install_package_dir}/.libs"
            RUNTIME DESTINATION "${_hypredrive_python_install_package_dir}/.libs")
    if(TARGET HYPRE)
        install(TARGETS HYPRE
                LIBRARY DESTINATION "${_hypredrive_python_install_package_dir}/.libs"
                RUNTIME DESTINATION "${_hypredrive_python_install_package_dir}/.libs")
    else()
        message(FATAL_ERROR
            "HYPREDRV_PYTHON_BUNDLE_CORE=${HYPREDRV_PYTHON_BUNDLE_CORE} requires auto-fetched/shared HYPRE")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_program(_hypredrive_python_patchelf patchelf)
        if(_hypredrive_python_patchelf)
            install(CODE "
                set(_hypredrive_pkgdir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_hypredrive_python_install_package_dir}\")
                file(GLOB _hypredrive_exts \"\${_hypredrive_pkgdir}/_core*.so\")
                foreach(_hypredrive_file IN LISTS _hypredrive_exts)
                    if(EXISTS \"\${_hypredrive_file}\" AND NOT IS_SYMLINK \"\${_hypredrive_file}\")
                        execute_process(
                            COMMAND \"${_hypredrive_python_patchelf}\" --set-rpath \"\\\$ORIGIN/.libs\" \"\${_hypredrive_file}\"
                            RESULT_VARIABLE _hypredrive_patchelf_rc
                        )
                        if(NOT _hypredrive_patchelf_rc EQUAL 0)
                            message(FATAL_ERROR \"patchelf failed for \${_hypredrive_file}\")
                        endif()
                    endif()
                endforeach()
                file(GLOB _hypredrive_libs \"\${_hypredrive_pkgdir}/.libs/*.so*\")
                foreach(_hypredrive_file IN LISTS _hypredrive_libs)
                    if(EXISTS \"\${_hypredrive_file}\" AND NOT IS_SYMLINK \"\${_hypredrive_file}\")
                        execute_process(
                            COMMAND \"${_hypredrive_python_patchelf}\" --set-rpath \"\\\$ORIGIN\" \"\${_hypredrive_file}\"
                            RESULT_VARIABLE _hypredrive_patchelf_rc
                        )
                        if(NOT _hypredrive_patchelf_rc EQUAL 0)
                            message(FATAL_ERROR \"patchelf failed for \${_hypredrive_file}\")
                        endif()
                    endif()
                endforeach()
            ")
        else()
            message(WARNING
                "patchelf was not found; bundled Linux wheels may retain MPI "
                "wrapper RPATH entries")
        endif()
    endif()
endif()

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
            -q
            --tb=short
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
                -q
                --tb=short
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_solve_mpi.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_laplacian_example_mpi.py"
                ${MPIEXEC_POSTFLAGS}
    )
else()
    message(STATUS "Python MPI tests are not attached to python-test because MPIEXEC or mpi4py was not found")
endif()

add_custom_target(python-test
    ${_python_test_commands}
    DEPENDS _core hypredrive_python_sources
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running hypredrive Python tests"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

add_custom_target(python-benchmark
    COMMAND "${CMAKE_COMMAND}" -E env
            ${_python_test_env}
            "${Python_EXECUTABLE}"
            -c
            "import scipy, pyamg"
    COMMAND "${CMAKE_COMMAND}" -E env
            ${_python_test_env}
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/compare_laplacian.py"
            --dim 3
            --n 100
            --repeat 1
    DEPENDS _core hypredrive_python_sources
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running hypredrive Python serial benchmark smoke case"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

add_custom_target(python-benchmarks
    DEPENDS python-benchmark
    COMMENT "Running hypredrive Python benchmark smoke cases"
)

if(MPIEXEC_EXECUTABLE AND _mpi4py_rc EQUAL 0)
    add_custom_target(python-benchmark-mpi
        COMMAND "${CMAKE_COMMAND}" -E env
                ${_python_test_env}
                "${Python_EXECUTABLE}"
                -c
                "import mpi4py, scipy"
        COMMAND "${CMAKE_COMMAND}" -E env
                ${_python_test_env}
                "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
                ${MPIEXEC_PREFLAGS}
                "${Python_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/scaling_laplacian.py"
                --n 8 8 8
                --P 2 1 1
                --repeat 1
                ${MPIEXEC_POSTFLAGS}
        DEPENDS _core hypredrive_python_sources
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running hypredrive Python MPI benchmark smoke case"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
else()
    add_custom_target(python-benchmark-mpi
        COMMAND "${CMAKE_COMMAND}" -E echo
                "python-benchmark-mpi requires MPIEXEC and mpi4py"
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "Python MPI benchmark prerequisites are missing"
        VERBATIM
    )
endif()
