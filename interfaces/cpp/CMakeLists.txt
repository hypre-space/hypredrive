# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)

add_library(HYPREDRV_CXX INTERFACE)
add_library(HYPREDRV::CXX ALIAS HYPREDRV_CXX)
set_target_properties(HYPREDRV_CXX PROPERTIES EXPORT_NAME CXX)
target_compile_features(HYPREDRV_CXX INTERFACE cxx_std_17)
target_link_libraries(HYPREDRV_CXX INTERFACE HYPREDRV::HYPREDRV)
target_include_directories(HYPREDRV_CXX
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS HYPREDRV_CXX
        EXPORT HYPREDRVTargets)
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

if(HYPREDRV_ENABLE_EXAMPLES)
    add_executable(laplacian-cpp examples/laplacian.cpp)
    target_link_libraries(laplacian-cpp PRIVATE HYPREDRV::CXX)
    # Match the project example convention: example binaries are installed when
    # HYPREDRV_ENABLE_EXAMPLES is enabled.
    install(TARGETS laplacian-cpp
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(HYPREDRV_ENABLE_TESTING)
    add_executable(test_cpp_interface tests/test_cpp_interface.cpp)
    target_link_libraries(test_cpp_interface PRIVATE HYPREDRV::CXX)
    add_test(NAME cpp_interface_serial COMMAND test_cpp_interface)
    set_tests_properties(cpp_interface_serial PROPERTIES LABELS "cpp;interface;unit")
    if(COMMAND hypredrv_append_test_environment)
        hypredrv_append_test_environment(cpp_interface_serial)
    endif()

    if(MPIEXEC_EXECUTABLE)
        add_test(NAME cpp_interface_mpi_2proc
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                         ${MPIEXEC_PREFLAGS}
                         $<TARGET_FILE:test_cpp_interface>
                         ${MPIEXEC_POSTFLAGS})
        set_tests_properties(cpp_interface_mpi_2proc PROPERTIES
            LABELS "cpp;interface;unit"
            RUN_SERIAL TRUE)
        if(COMMAND hypredrv_append_test_environment)
            hypredrv_append_test_environment(cpp_interface_mpi_2proc)
        endif()
    endif()

    add_test(NAME cpp_api_coverage
             COMMAND ${CMAKE_COMMAND}
                     -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
                     -DHEADER=${CMAKE_CURRENT_SOURCE_DIR}/include/hypredrive.hpp
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_api_coverage.cmake)
    set_tests_properties(cpp_api_coverage PROPERTIES LABELS "cpp;interface")

    set(_hypredrv_cpp_test_deps test_cpp_interface)
    if(TARGET laplacian-cpp)
        list(APPEND _hypredrv_cpp_test_deps laplacian-cpp)
    endif()

    add_custom_target(cpp-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "cpp"
        DEPENDS ${_hypredrv_cpp_test_deps}
        COMMENT "Running hypredrive C++ interface tests")
endif()
