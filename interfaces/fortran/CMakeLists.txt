# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

include(CheckCSourceCompiles)

set(_hypredrv_fortran_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(_hypredrv_fortran_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
hypredrv_collect_hypre_include_dirs(_hypredrv_fortran_hypre_includes
                                    INCLUDE_SOURCE_SUBDIRS)
set(CMAKE_REQUIRED_INCLUDES
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}"
    ${_hypredrv_fortran_hypre_includes}
    ${MPI_C_INCLUDE_DIRS})
check_c_source_compiles("
#include \"HYPREDRV.h\"
typedef char hypredrv_fortran_requires_double_hypre_real[
   (sizeof(HYPRE_Real) == sizeof(double)) ? 1 : -1];
int main(void) { return 0; }
" HYPREDRV_FORTRAN_HYPRE_REAL_IS_DOUBLE)
set(CMAKE_REQUIRED_INCLUDES "${_hypredrv_fortran_saved_required_includes}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_hypredrv_fortran_saved_try_compile_target_type}")
if(NOT HYPREDRV_FORTRAN_HYPRE_REAL_IS_DOUBLE)
    message(FATAL_ERROR
        "HYPREDRV_ENABLE_FORTRAN requires HYPRE_Real to have the C double ABI. "
        "Disable HYPRE_ENABLE_SINGLE/HYPRE_ENABLE_LONG_DOUBLE or disable "
        "HYPREDRV_ENABLE_FORTRAN.")
endif()

add_library(HYPREDRV_Fortran
    src/hypredrive.f90
    src/HYPREDRV_fortran.c
)
add_library(HYPREDRV::Fortran ALIAS HYPREDRV_Fortran)
hypredrv_hide_static_archive_symbols(HYPREDRV_Fortran)
hypredrv_add_dynamic_export_test(HYPREDRV_Fortran FORTRAN)

set_target_properties(HYPREDRV_Fortran PROPERTIES
    OUTPUT_NAME HYPREDRV_Fortran
    EXPORT_NAME Fortran
    Fortran_STANDARD 2003
    Fortran_STANDARD_REQUIRED ON
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(HYPREDRV_Fortran PUBLIC HYPREDRV::HYPREDRV MPI::MPI_C MPI::MPI_Fortran)
hypredrv_set_relative_install_rpath(HYPREDRV_Fortran)
target_include_directories(HYPREDRV_Fortran
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(CheckFortranCompilerFlag)

set(_hypredrv_fortran_long_line_flags)
set(_hypredrv_fortran_long_line_check_flags)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(_hypredrv_fortran_long_line_flags "-ffree-line-length-none")
    set(_hypredrv_fortran_long_line_check_flags "-ffree-line-length-none")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(_hypredrv_fortran_long_line_flags "-extend-source" "132")
    set(_hypredrv_fortran_long_line_check_flags "-extend-source" "132")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    set(_hypredrv_fortran_long_line_flags "-Mfree" "-Mextend")
    set(_hypredrv_fortran_long_line_check_flags "-Mfree" "-Mextend")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
    set(_hypredrv_fortran_long_line_flags "-N" "255")
    set(_hypredrv_fortran_long_line_check_flags "-N" "255")
endif()

if(_hypredrv_fortran_long_line_check_flags)
    list(JOIN _hypredrv_fortran_long_line_check_flags " " _hypredrv_fortran_long_line_check_flags_text)
    string(MAKE_C_IDENTIFIER "${_hypredrv_fortran_long_line_check_flags_text}" _hypredrv_fortran_flag_id)
    check_fortran_compiler_flag("${_hypredrv_fortran_long_line_check_flags}"
        HYPREDRV_HAVE_Fortran_${_hypredrv_fortran_flag_id})
    if(NOT HYPREDRV_HAVE_Fortran_${_hypredrv_fortran_flag_id})
        message(STATUS
            "Fortran compiler ${CMAKE_Fortran_COMPILER_ID} does not accept "
            "${_hypredrv_fortran_long_line_check_flags_text}; building without a line-length override")
        set(_hypredrv_fortran_long_line_flags)
    endif()
elseif(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Flang|LLVMFlang")
    message(STATUS
        "No Fortran line-length override is configured for compiler "
        "${CMAKE_Fortran_COMPILER_ID}; source lines are kept within standard limits")
endif()

if(_hypredrv_fortran_long_line_flags)
    target_compile_options(HYPREDRV_Fortran PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:${_hypredrv_fortran_long_line_flags}>
    )
endif()

install(TARGETS HYPREDRV_Fortran
        EXPORT HYPREDRVTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.mod")

function(hypredrv_add_fortran_executable target source)
    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES
        Fortran_STANDARD 2003
        Fortran_STANDARD_REQUIRED ON)
    target_link_libraries(${target} PRIVATE HYPREDRV::Fortran MPI::MPI_Fortran)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/mod")
    if(_hypredrv_fortran_long_line_flags)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:${_hypredrv_fortran_long_line_flags}>
        )
    endif()
    hypredrv_set_relative_install_rpath(${target})
endfunction()

# Build and install the Laplacian driver whenever the Fortran interface is
# enabled. It is both the user-facing example and the reproducible smoke driver
# used by the Fortran tests.
hypredrv_add_fortran_executable(laplacian-fortran examples/laplacian/laplacian.f90)
install(TARGETS laplacian-fortran
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(HYPREDRV_ENABLE_EXAMPLES)
    hypredrv_add_fortran_executable(hypredrive-fortran-yaml examples/yaml_solve_mpi.f90)
    add_custom_target(fortran-examples
        DEPENDS hypredrive-fortran-yaml laplacian-fortran
        COMMENT "Building hypredrive Fortran examples")
endif()

if(HYPREDRV_ENABLE_TESTING)
    set(_hypredrv_fortran_failure_regex
        "HYPREDRIVE Failure!!!|BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES|Segmentation fault|Abort\\(")
    function(hypredrv_enable_fortran_failure_detection test_name)
        set_property(TEST ${test_name} PROPERTY
            FAIL_REGULAR_EXPRESSION "${_hypredrv_fortran_failure_regex}")
    endfunction()

    hypredrv_add_fortran_executable(test_fortran_lifecycle tests/test_lifecycle.f90)
    hypredrv_add_fortran_executable(test_fortran_csr_mpi tests/test_csr_mpi.f90)
    hypredrv_add_fortran_executable(test_fortran_check_failure tests/test_check_failure.f90)
    target_sources(test_fortran_csr_mpi PRIVATE tests/test_helpers.c)

    add_test(NAME fortran_lifecycle COMMAND test_fortran_lifecycle)
    set_tests_properties(fortran_lifecycle PROPERTIES LABELS "unit;fortran")
    hypredrv_enable_fortran_failure_detection(fortran_lifecycle)

    add_test(NAME fortran_check_failure
             COMMAND ${CMAKE_COMMAND}
                     -DTEST_EXE=$<TARGET_FILE:test_fortran_check_failure>
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_check_failure.cmake")
    set_tests_properties(fortran_check_failure PROPERTIES LABELS "unit;fortran")

    if(MPIEXEC_EXECUTABLE)
        add_test(NAME fortran_csr_mpi_2proc
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${MPIEXEC_PREFLAGS}
                         $<TARGET_FILE:test_fortran_csr_mpi>
                         ${MPIEXEC_POSTFLAGS})
        set_tests_properties(fortran_csr_mpi_2proc PROPERTIES
            LABELS "unit;fortran"
            RUN_SERIAL TRUE)
        hypredrv_enable_fortran_failure_detection(fortran_csr_mpi_2proc)
    endif()

    set(_hypredrv_fortran_test_target_deps
        test_fortran_lifecycle
        test_fortran_csr_mpi
        test_fortran_check_failure)
    if(TARGET laplacian-fortran)
        set(_fortran_laplacian_dir "${CMAKE_CURRENT_SOURCE_DIR}/examples/laplacian")

        add_test(NAME fortran_laplacian_serial
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/default.nml")
        set_tests_properties(fortran_laplacian_serial PROPERTIES
            LABELS "fortran;example"
            RUN_SERIAL TRUE)
        hypredrv_enable_fortran_failure_detection(fortran_laplacian_serial)

        add_test(NAME fortran_laplacian_anisotropic
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/anisotropic.nml")
        set_tests_properties(fortran_laplacian_anisotropic PROPERTIES
            LABELS "fortran;example"
            RUN_SERIAL TRUE)
        hypredrv_enable_fortran_failure_detection(fortran_laplacian_anisotropic)

        add_test(NAME fortran_laplacian_external_yaml
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/external_yaml.nml")
        set_tests_properties(fortran_laplacian_external_yaml PROPERTIES
            LABELS "fortran;example"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RUN_SERIAL TRUE)
        hypredrv_enable_fortran_failure_detection(fortran_laplacian_external_yaml)

        if(MPIEXEC_EXECUTABLE)
            add_test(NAME fortran_laplacian_mpi_x_2proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/x_2proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_x_2proc PROPERTIES
                LABELS "fortran;example"
                RUN_SERIAL TRUE)
            hypredrv_enable_fortran_failure_detection(fortran_laplacian_mpi_x_2proc)

            add_test(NAME fortran_laplacian_mpi_z_2proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/z_2proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_z_2proc PROPERTIES
                LABELS "fortran;example"
                RUN_SERIAL TRUE)
            hypredrv_enable_fortran_failure_detection(fortran_laplacian_mpi_z_2proc)

            add_test(NAME fortran_laplacian_mpi_xy_4proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/xy_4proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_xy_4proc PROPERTIES
                LABELS "fortran;example"
                RUN_SERIAL TRUE)
            hypredrv_enable_fortran_failure_detection(fortran_laplacian_mpi_xy_4proc)
        endif()
        list(APPEND _hypredrv_fortran_test_target_deps laplacian-fortran)
    endif()

    add_custom_target(fortran-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L fortran
        DEPENDS ${_hypredrv_fortran_test_target_deps}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Running hypredrive Fortran tests")
endif()
