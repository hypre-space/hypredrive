# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

add_library(HYPREDRV_Fortran
    src/hypredrive.f90
    src/HYPREDRV_fortran.c
)
add_library(HYPREDRV::Fortran ALIAS HYPREDRV_Fortran)

set_target_properties(HYPREDRV_Fortran PROPERTIES
    OUTPUT_NAME HYPREDRV_Fortran
    EXPORT_NAME Fortran
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
    check_fortran_compiler_flag("${_hypredrv_fortran_long_line_check_flags_text}"
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
    target_link_libraries(${target} PRIVATE HYPREDRV::Fortran MPI::MPI_Fortran)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/mod")
    if(_hypredrv_fortran_long_line_flags)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:${_hypredrv_fortran_long_line_flags}>
        )
    endif()
    if(APPLE)
        target_link_options(${target} PRIVATE "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
    endif()
    hypredrv_set_relative_install_rpath(${target})
endfunction()

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
    enable_testing()
    hypredrv_add_fortran_executable(test_fortran_lifecycle tests/test_lifecycle.f90)
    hypredrv_add_fortran_executable(test_fortran_csr_mpi tests/test_csr_mpi.f90)
    hypredrv_add_fortran_executable(test_fortran_check_failure tests/test_check_failure.f90)
    target_sources(test_fortran_csr_mpi PRIVATE tests/test_helpers.c)

    add_test(NAME fortran_lifecycle COMMAND test_fortran_lifecycle)
    set_tests_properties(fortran_lifecycle PROPERTIES LABELS "unit;fortran")

    add_test(NAME fortran_check_failure
             COMMAND ${CMAKE_COMMAND}
                     -DTEST_EXE=$<TARGET_FILE:test_fortran_check_failure>
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_check_failure.cmake")
    set_tests_properties(fortran_check_failure PROPERTIES
        LABELS "unit;fortran"
        PASS_REGULAR_EXPRESSION "HYPREDRV call failed")

    if(MPIEXEC_EXECUTABLE)
        add_test(NAME fortran_csr_mpi_2proc
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${MPIEXEC_PREFLAGS}
                         $<TARGET_FILE:test_fortran_csr_mpi>
                         ${MPIEXEC_POSTFLAGS})
        set_tests_properties(fortran_csr_mpi_2proc PROPERTIES LABELS "unit;fortran" RUN_SERIAL TRUE)
    endif()

    set(_hypredrv_fortran_test_deps
        test_fortran_lifecycle
        test_fortran_csr_mpi
        test_fortran_check_failure)
    if(TARGET laplacian-fortran)
        set(_fortran_laplacian_dir "${CMAKE_CURRENT_SOURCE_DIR}/examples/laplacian")

        add_test(NAME fortran_laplacian_serial
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/default.nml")
        set_tests_properties(fortran_laplacian_serial PROPERTIES LABELS "fortran;example" RUN_SERIAL TRUE)

        add_test(NAME fortran_laplacian_anisotropic
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/anisotropic.nml")
        set_tests_properties(fortran_laplacian_anisotropic PROPERTIES LABELS "fortran;example" RUN_SERIAL TRUE)

        add_test(NAME fortran_laplacian_external_yaml
                 COMMAND $<TARGET_FILE:laplacian-fortran>
                         "${_fortran_laplacian_dir}/external_yaml.nml")
        set_tests_properties(fortran_laplacian_external_yaml PROPERTIES
            LABELS "fortran;example"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RUN_SERIAL TRUE)

        if(MPIEXEC_EXECUTABLE)
            add_test(NAME fortran_laplacian_mpi_x_2proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/x_2proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_x_2proc PROPERTIES LABELS "fortran;example" RUN_SERIAL TRUE)

            add_test(NAME fortran_laplacian_mpi_z_2proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/z_2proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_z_2proc PROPERTIES LABELS "fortran;example" RUN_SERIAL TRUE)

            add_test(NAME fortran_laplacian_mpi_xy_4proc
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                             ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:laplacian-fortran>
                             "${_fortran_laplacian_dir}/xy_4proc.nml"
                             ${MPIEXEC_POSTFLAGS})
            set_tests_properties(fortran_laplacian_mpi_xy_4proc PROPERTIES LABELS "fortran;example" RUN_SERIAL TRUE)
        endif()
        list(APPEND _hypredrv_fortran_test_deps laplacian-fortran)
    endif()

    add_custom_target(fortran-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L fortran
        DEPENDS ${_hypredrv_fortran_test_deps}
        COMMENT "Running hypredrive Fortran tests")
endif()
