# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

include(CheckCSourceCompiles)

find_package(Matlab QUIET COMPONENTS MAIN_PROGRAM MX_LIBRARY MEX_COMPILER)
find_program(OCTAVE_MKOCTFILE_EXECUTABLE NAMES mkoctfile)
find_program(OCTAVE_EXECUTABLE NAMES octave)

if(NOT Matlab_FOUND AND NOT OCTAVE_MKOCTFILE_EXECUTABLE)
    message(FATAL_ERROR
        "HYPREDRV_ENABLE_MATLAB requires either MATLAB with mex or GNU Octave "
        "with mkoctfile. Set Matlab_ROOT_DIR for MATLAB or ensure mkoctfile is "
        "on PATH.")
endif()
if(NOT Matlab_FOUND AND OCTAVE_MKOCTFILE_EXECUTABLE)
    message(WARNING
        "HYPREDRV_ENABLE_MATLAB is ON but MATLAB was not found; building only "
        "the GNU Octave MEX interface.")
endif()

set(_hypredrv_matlab_mpi_include_dirs ${MPI_C_INCLUDE_DIRS})
if(NOT _hypredrv_matlab_mpi_include_dirs AND MPI_C_COMPILER)
    execute_process(
        COMMAND ${MPI_C_COMPILER} --showme:compile
        OUTPUT_VARIABLE _hypredrv_matlab_mpi_show_compile
        RESULT_VARIABLE _hypredrv_matlab_mpi_show_compile_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _hypredrv_matlab_mpi_show_compile_result EQUAL 0)
        execute_process(
            COMMAND ${MPI_C_COMPILER} -show
            OUTPUT_VARIABLE _hypredrv_matlab_mpi_show_compile
            RESULT_VARIABLE _hypredrv_matlab_mpi_show_compile_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(_hypredrv_matlab_mpi_show_compile_result EQUAL 0)
        separate_arguments(_hypredrv_matlab_mpi_show_compile_args
                           UNIX_COMMAND "${_hypredrv_matlab_mpi_show_compile}")
        foreach(_hypredrv_matlab_mpi_compile_arg IN LISTS
                _hypredrv_matlab_mpi_show_compile_args)
            if(_hypredrv_matlab_mpi_compile_arg MATCHES "^-I(.+)$")
                list(APPEND _hypredrv_matlab_mpi_include_dirs
                     "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()
endif()
if(_hypredrv_matlab_mpi_include_dirs)
    list(REMOVE_DUPLICATES _hypredrv_matlab_mpi_include_dirs)
endif()

set(_hypredrv_matlab_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(_hypredrv_matlab_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
hypredrv_collect_hypre_include_dirs(_hypredrv_matlab_plain_hypre_includes
                                    INCLUDE_SOURCE_SUBDIRS)
set(CMAKE_REQUIRED_INCLUDES
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}"
    ${_hypredrv_matlab_plain_hypre_includes}
    ${_hypredrv_matlab_mpi_include_dirs})
check_c_source_compiles("
#include \"HYPREDRV.h\"
typedef char hypredrv_matlab_requires_double_hypre_real[
   (sizeof(HYPRE_Real) == sizeof(double)) ? 1 : -1];
typedef char hypredrv_matlab_rejects_complex_hypre[
   (sizeof(HYPRE_Complex) == sizeof(HYPRE_Real)) ? 1 : -1];
int main(void) { return 0; }
" HYPREDRV_MATLAB_HYPRE_REAL_IS_DOUBLE)
set(CMAKE_REQUIRED_INCLUDES "${_hypredrv_matlab_saved_required_includes}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_hypredrv_matlab_saved_try_compile_target_type}")
if(NOT HYPREDRV_MATLAB_HYPRE_REAL_IS_DOUBLE)
    message(FATAL_ERROR
        "HYPREDRV_ENABLE_MATLAB requires a real-valued HYPRE build where "
        "HYPRE_Real uses the C double ABI. Disable HYPRE_ENABLE_SINGLE, "
        "HYPRE_ENABLE_LONG_DOUBLE, and HYPRE_ENABLE_COMPLEX or disable "
        "HYPREDRV_ENABLE_MATLAB.")
endif()

set(_hypredrv_matlab_mfiles
    src/hypredrive_solve.m
    src/hypredrive_is_text.m
    src/hypredrive_options.m
    src/hypredrive_setup.m)

set(_hypredrv_matlab_mpi_library_dirs)
foreach(_mpi_library IN LISTS MPI_C_LIBRARIES)
    if(IS_ABSOLUTE "${_mpi_library}")
        get_filename_component(_mpi_library_dir "${_mpi_library}" DIRECTORY)
        list(APPEND _hypredrv_matlab_mpi_library_dirs "${_mpi_library_dir}")
    endif()
endforeach()
if(_hypredrv_matlab_mpi_library_dirs)
    list(REMOVE_DUPLICATES _hypredrv_matlab_mpi_library_dirs)
endif()

if(Matlab_FOUND)
    matlab_add_mex(
        NAME hypredrive_matlab_mex
        OUTPUT_NAME hypredrive_mex
        SRC src/hypredrive_mex.c
        LINK_TO HYPREDRV::HYPREDRV
    )

    target_include_directories(hypredrive_matlab_mex PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}"
    )
    # Only relocate the target; do not override SUFFIX because FindMatlab owns
    # the platform MEX extension (for example .mexmaca64 on Apple silicon).
    set_target_properties(hypredrive_matlab_mex PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/matlab"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/matlab")
    hypredrv_set_relative_install_rpath(hypredrive_matlab_mex)
    if(UNIX)
        if(APPLE)
            set(_hypredrv_matlab_origin "@loader_path")
        else()
            set(_hypredrv_matlab_origin "$ORIGIN")
        endif()
        set_property(TARGET hypredrive_matlab_mex APPEND PROPERTY
            INSTALL_RPATH "${_hypredrv_matlab_origin}/..")
        foreach(_mpi_library_dir IN LISTS _hypredrv_matlab_mpi_library_dirs)
            set_property(TARGET hypredrive_matlab_mex APPEND PROPERTY
                BUILD_RPATH "${_mpi_library_dir}")
            set_property(TARGET hypredrive_matlab_mex APPEND PROPERTY
                INSTALL_RPATH "${_mpi_library_dir}")
        endforeach()
    endif()

    add_custom_target(hypredrive-matlab
        DEPENDS hypredrive_matlab_mex
        COMMENT "Building hypredrive MATLAB interface")

    install(TARGETS hypredrive_matlab_mex
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/matlab"
            RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}/matlab")
    install(FILES ${_hypredrv_matlab_mfiles}
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/matlab")
endif()

if(OCTAVE_MKOCTFILE_EXECUTABLE)
    execute_process(
        COMMAND ${OCTAVE_MKOCTFILE_EXECUTABLE} -p MEXEXT
        OUTPUT_VARIABLE _hypredrv_octave_mexext
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _hypredrv_octave_mexext)
        set(_hypredrv_octave_mexext mex)
    endif()
    set(_hypredrv_octave_output
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/octave/hypredrive_mex.${_hypredrv_octave_mexext}")
    set(_hypredrv_octave_rpath_flags)
    if(UNIX)
        if(APPLE)
            list(APPEND _hypredrv_octave_rpath_flags
                 "-Wl,-rpath,@loader_path/.."
                 "-Wl,-rpath,$<TARGET_FILE_DIR:HYPREDRV>"
            )
        else()
            list(APPEND _hypredrv_octave_rpath_flags
                 "-Wl,-rpath,\\$ORIGIN/.."
                 "-Wl,-rpath,$<TARGET_FILE_DIR:HYPREDRV>")
        endif()
    endif()
    foreach(_mpi_library_dir IN LISTS _hypredrv_matlab_mpi_library_dirs)
        list(APPEND _hypredrv_octave_rpath_flags
             "-Wl,-rpath,${_mpi_library_dir}")
    endforeach()

    set(_hypredrv_octave_include_flags
        "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
        "-I${CMAKE_SOURCE_DIR}/include"
        "-I${CMAKE_BINARY_DIR}")
    foreach(_hypredrv_hypre_include IN LISTS _hypredrv_matlab_plain_hypre_includes)
        list(APPEND _hypredrv_octave_include_flags "-I${_hypredrv_hypre_include}")
    endforeach()
    foreach(_mpi_include IN LISTS _hypredrv_matlab_mpi_include_dirs)
        list(APPEND _hypredrv_octave_include_flags "-I${_mpi_include}")
    endforeach()

    # mkoctfile rejects bare absolute shared-library paths. Use target-derived
    # -L/-l flags instead of hard-coded library names or -Wl,<absolute-path>.
    set(_hypredrv_octave_link_inputs
        "-L$<TARGET_LINKER_FILE_DIR:HYPREDRV>"
        "-l$<TARGET_LINKER_FILE_BASE_NAME:HYPREDRV>")
    if(TARGET HYPRE::HYPRE)
        get_target_property(_hypredrv_octave_hypre_type HYPRE::HYPRE TYPE)
        if(NOT _hypredrv_octave_hypre_type STREQUAL "INTERFACE_LIBRARY")
            set(_hypredrv_octave_hypre_location "")
            foreach(_hypredrv_octave_hypre_config IN ITEMS RELEASE RELWITHDEBINFO DEBUG MINSIZEREL "")
                if(_hypredrv_octave_hypre_config STREQUAL "")
                    get_target_property(_hypredrv_octave_hypre_candidate
                                        HYPRE::HYPRE IMPORTED_LOCATION)
                else()
                    get_target_property(_hypredrv_octave_hypre_candidate
                                        HYPRE::HYPRE
                                        IMPORTED_LOCATION_${_hypredrv_octave_hypre_config})
                endif()
                if(_hypredrv_octave_hypre_candidate)
                    set(_hypredrv_octave_hypre_location
                        "${_hypredrv_octave_hypre_candidate}")
                    break()
                endif()
            endforeach()
            if(_hypredrv_octave_hypre_location)
                get_filename_component(_hypredrv_octave_hypre_library_dir
                                       "${_hypredrv_octave_hypre_location}"
                                       DIRECTORY)
                get_filename_component(_hypredrv_octave_hypre_library_name
                                       "${_hypredrv_octave_hypre_location}"
                                       NAME)
                if(_hypredrv_octave_hypre_library_name MATCHES "^lib(.+)\\.(a|so|dylib)(\\..*)?$")
                    list(APPEND _hypredrv_octave_link_inputs
                         "-L${_hypredrv_octave_hypre_library_dir}"
                         "-l${CMAKE_MATCH_1}")
                else()
                    list(APPEND _hypredrv_octave_link_inputs
                         "${_hypredrv_octave_hypre_location}")
                endif()
            endif()
        endif()
    endif()
    foreach(_mpi_library IN LISTS MPI_C_LIBRARIES)
        if(_mpi_library MATCHES "^-")
            list(APPEND _hypredrv_octave_link_inputs "${_mpi_library}")
        elseif(IS_ABSOLUTE "${_mpi_library}")
            get_filename_component(_mpi_library_dir "${_mpi_library}" DIRECTORY)
            get_filename_component(_mpi_library_name "${_mpi_library}" NAME)
            if(_mpi_library_name MATCHES "^lib(.+)\\.(a|so|dylib)(\\..*)?$")
                list(APPEND _hypredrv_octave_link_inputs
                     "-L${_mpi_library_dir}"
                     "-l${CMAKE_MATCH_1}")
            else()
                list(APPEND _hypredrv_octave_link_inputs "${_mpi_library}")
            endif()
        else()
            list(APPEND _hypredrv_octave_link_inputs "${_mpi_library}")
        endif()
    endforeach()

    add_custom_command(
        OUTPUT "${_hypredrv_octave_output}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/octave"
        COMMAND ${OCTAVE_MKOCTFILE_EXECUTABLE} --mex
                -o "${_hypredrv_octave_output}"
                ${_hypredrv_octave_include_flags}
                ${_hypredrv_octave_rpath_flags}
                "${CMAKE_CURRENT_SOURCE_DIR}/src/hypredrive_mex.c"
                ${_hypredrv_octave_link_inputs}
                -lm
        DEPENDS HYPREDRV "${CMAKE_CURRENT_SOURCE_DIR}/src/hypredrive_mex.c"
        COMMENT "Building hypredrive Octave MEX interface"
        VERBATIM
        COMMAND_EXPAND_LISTS)
    add_custom_target(hypredrive-octave
        DEPENDS "${_hypredrv_octave_output}")

    install(FILES "${_hypredrv_octave_output}" ${_hypredrv_matlab_mfiles}
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/octave")
endif()

install(FILES examples/laplacian.m examples/build_laplacian.m
              examples/elasticity.m examples/build_elasticity.m
        DESTINATION "${CMAKE_INSTALL_DATADIR}/matlab/examples")

if(HYPREDRV_ENABLE_TESTING)
    set(_hypredrv_matlab_test_deps)
    set(_hypredrv_matlab_example_cases
        "laplacian|1d|laplacian(64)"
        "laplacian|2d|laplacian(32, 32)"
        "laplacian|3d|laplacian(16, 16, 16)"
        "elasticity|1d|elasticity(64)"
        "elasticity|2d|elasticity(32, 32)"
        "elasticity|3d|elasticity(16, 16, 16)")
    if(Matlab_FOUND)
        add_test(NAME matlab_laplacian_serial
                 COMMAND ${Matlab_MAIN_PROGRAM} -batch
                         "addpath('$<TARGET_FILE_DIR:hypredrive_matlab_mex>'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/src'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/examples'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/tests'); test_hypredrive_mex")
        set_tests_properties(matlab_laplacian_serial PROPERTIES
            LABELS "matlab;interface"
            RUN_SERIAL TRUE)
        if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
            set_tests_properties(matlab_laplacian_serial PROPERTIES
                ENVIRONMENT "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
        endif()
        foreach(_case IN LISTS _hypredrv_matlab_example_cases)
            string(REPLACE "|" ";" _case_parts "${_case}")
            list(GET _case_parts 0 _case_name)
            list(GET _case_parts 1 _case_tag)
            list(GET _case_parts 2 _case_call)
            add_test(NAME matlab_${_case_name}_example_${_case_tag}
                     COMMAND ${Matlab_MAIN_PROGRAM} -batch
                             "addpath('$<TARGET_FILE_DIR:hypredrive_matlab_mex>'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/src'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/examples'); ${_case_call}")
            set_tests_properties(matlab_${_case_name}_example_${_case_tag} PROPERTIES
                LABELS "matlab;interface;example"
                RUN_SERIAL TRUE)
            if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
                set_tests_properties(matlab_${_case_name}_example_${_case_tag} PROPERTIES
                    ENVIRONMENT "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
            endif()
        endforeach()
        list(APPEND _hypredrv_matlab_test_deps hypredrive_matlab_mex)
    endif()
    if(OCTAVE_EXECUTABLE AND OCTAVE_MKOCTFILE_EXECUTABLE)
        add_test(NAME octave_laplacian_serial
                 COMMAND ${OCTAVE_EXECUTABLE} --quiet --eval
                         "addpath('${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/octave'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/src'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/examples'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/tests'); test_hypredrive_mex")
        set_tests_properties(octave_laplacian_serial PROPERTIES
            LABELS "octave;matlab;interface"
            RUN_SERIAL TRUE)
        if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
            set_tests_properties(octave_laplacian_serial PROPERTIES
                ENVIRONMENT "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
        endif()
        foreach(_case IN LISTS _hypredrv_matlab_example_cases)
            string(REPLACE "|" ";" _case_parts "${_case}")
            list(GET _case_parts 0 _case_name)
            list(GET _case_parts 1 _case_tag)
            list(GET _case_parts 2 _case_call)
            add_test(NAME octave_${_case_name}_example_${_case_tag}
                     COMMAND ${OCTAVE_EXECUTABLE} --quiet --eval
                             "addpath('${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/octave'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/src'); addpath('${CMAKE_CURRENT_SOURCE_DIR}/examples'); ${_case_call}")
            set_tests_properties(octave_${_case_name}_example_${_case_tag} PROPERTIES
                LABELS "octave;matlab;interface;example"
                RUN_SERIAL TRUE)
            if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
                set_tests_properties(octave_${_case_name}_example_${_case_tag} PROPERTIES
                    ENVIRONMENT "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
            endif()
        endforeach()
        list(APPEND _hypredrv_matlab_test_deps hypredrive-octave)
    endif()

    add_custom_target(matlab-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "matlab|octave"
        DEPENDS ${_hypredrv_matlab_test_deps}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running hypredrive MATLAB/Octave tests"
        VERBATIM)
endif()
