# Copyright (c) 2024, Lawrence Livermore National Security, LLC.
# See the top-level LICENSE and NOTICE files for details.

include(GNUInstallDirs)

find_program(Julia_EXECUTABLE NAMES julia)
if(Julia_EXECUTABLE)
    message(STATUS "Julia executable: ${Julia_EXECUTABLE}")
else()
    message(STATUS "Julia executable not found; julia-test target will report this when run")
endif()

if(NOT TARGET MPI::MPI_C)
    find_package(MPI COMPONENTS C REQUIRED)
endif()
if(MPI_C_FOUND AND MPIEXEC_EXECUTABLE)
    message(STATUS "Julia MPI tests enabled: ${MPIEXEC_EXECUTABLE}")
else()
    message(STATUS "Julia MPI tests disabled: mpiexec not found")
endif()

if(UNIX AND DEFINED HYPRE_LIBRARY AND HYPRE_LIBRARY MATCHES "\\.a$")
    message(WARNING
        "The Julia interface builds libHYPREDRV_Julia as a shared library. "
        "The external static HYPRE archive selected by HYPRE_LIBRARY must be "
        "compiled with -fPIC, otherwise the bridge link will fail.")
endif()

add_library(HYPREDRV_Julia SHARED
    src/HYPREDRV_julia.c
)
add_library(HYPREDRV::Julia ALIAS HYPREDRV_Julia)
hypredrv_hide_static_archive_symbols(HYPREDRV_Julia APPLE_EXPORTS
    "_HYPREDRV_Julia*")
hypredrv_add_dynamic_export_test(HYPREDRV_Julia JULIA)

target_link_libraries(HYPREDRV_Julia
    PUBLIC
        HYPREDRV::HYPREDRV
)

target_include_directories(HYPREDRV_Julia
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/include/internal
)

set(_hypredrv_julia_build_rpath "$<TARGET_FILE_DIR:HYPREDRV>")
if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
    string(REGEX REPLACE "^[^=]*=" "" _hypredrv_julia_runtime_dirs
           "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    if(WIN32)
        string(REPLACE ";" "\\;" _hypredrv_julia_runtime_dirs
               "${_hypredrv_julia_runtime_dirs}")
    else()
        string(REPLACE ":" ";" _hypredrv_julia_runtime_dirs
               "${_hypredrv_julia_runtime_dirs}")
    endif()
    foreach(_hypredrv_julia_runtime_dir IN LISTS _hypredrv_julia_runtime_dirs)
        if(_hypredrv_julia_runtime_dir)
            list(APPEND _hypredrv_julia_build_rpath
                 "${_hypredrv_julia_runtime_dir}")
        endif()
    endforeach()
endif()
set_target_properties(HYPREDRV_Julia PROPERTIES
    OUTPUT_NAME HYPREDRV_Julia
    EXPORT_NAME Julia
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "${_hypredrv_julia_build_rpath}"
)

hypredrv_set_relative_install_rpath(HYPREDRV_Julia)
foreach(_hypredrv_julia_runtime_dir IN LISTS _hypredrv_julia_runtime_dirs)
    if(IS_ABSOLUTE "${_hypredrv_julia_runtime_dir}")
        set_property(TARGET HYPREDRV_Julia APPEND PROPERTY INSTALL_RPATH
            "${_hypredrv_julia_runtime_dir}")
    endif()
endforeach()

install(TARGETS HYPREDRV_Julia
    EXPORT HYPREDRVTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    Artifacts.toml
    Project.toml
    README.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/julia/HypreDrive
)
install(DIRECTORY src test examples
    DESTINATION ${CMAKE_INSTALL_DATADIR}/julia/HypreDrive
    FILES_MATCHING PATTERN "*.jl"
)
install(DIRECTORY test/cmake_consumer
    DESTINATION ${CMAKE_INSTALL_DATADIR}/julia/HypreDrive/test
)

if(HYPREDRV_ENABLE_TESTING AND Julia_EXECUTABLE)
    set(_hypredrv_julia_test_project
        "${CMAKE_CURRENT_BINARY_DIR}/test-project")
    set(_hypredrv_julia_depot
        "${CMAKE_CURRENT_BINARY_DIR}/julia-depot")
    set(_hypredrv_julia_test_env
        "HYPREDRV_LIBRARY=$<TARGET_FILE:HYPREDRV_Julia>"
        "JULIA_DEPOT_PATH=${_hypredrv_julia_depot}"
        "JULIA_PKG_PRECOMPILE_AUTO=0"
    )
    if(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT)
        list(APPEND _hypredrv_julia_test_env "${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    endif()

    add_test(NAME julia_build_bridge
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                --target HYPREDRV_Julia --config $<CONFIG>
    )
    set_tests_properties(julia_build_bridge PROPERTIES
        LABELS "julia;interface;build"
        FIXTURES_SETUP julia_built
    )

    add_test(NAME julia_instantiate_project
        COMMAND ${Julia_EXECUTABLE} --startup-file=no
                -e "import Pkg; Pkg.activate(ARGS[1]); Pkg.develop(path=ARGS[2]); Pkg.instantiate()"
                ${_hypredrv_julia_test_project}
                ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(julia_instantiate_project PROPERTIES
        LABELS "julia;interface;deps"
        ENVIRONMENT "JULIA_DEPOT_PATH=${_hypredrv_julia_depot};JULIA_PKG_PRECOMPILE_AUTO=0"
        FIXTURES_SETUP julia_project_instantiated
    )

    add_test(NAME julia_laplacian_serial
        COMMAND ${Julia_EXECUTABLE} --startup-file=no
                --project=${_hypredrv_julia_test_project}
                ${CMAKE_CURRENT_SOURCE_DIR}/test/runtests.jl
    )
    set_tests_properties(julia_laplacian_serial PROPERTIES
        LABELS "julia;interface"
        ENVIRONMENT "${_hypredrv_julia_test_env}"
        FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
    )

    add_test(NAME julia_laplacian1d_args
        COMMAND ${Julia_EXECUTABLE} --startup-file=no
                --project=${_hypredrv_julia_test_project}
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/laplacian1d.jl
                64 -a --solver:pcg:relative_tol 1.0e-2
    )
    set_tests_properties(julia_laplacian1d_args PROPERTIES
        LABELS "julia;interface"
        ENVIRONMENT "${_hypredrv_julia_test_env}"
        FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
    )

    if(MPI_C_FOUND AND MPIEXEC_EXECUTABLE)
        add_test(NAME julia_laplacian_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    ${Julia_EXECUTABLE} --startup-file=no
                    --project=${_hypredrv_julia_test_project}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/mpi.jl
        )
        set_tests_properties(julia_laplacian_mpi PROPERTIES
            LABELS "julia;interface;mpi"
            ENVIRONMENT "${_hypredrv_julia_test_env}"
            FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
        )
    endif()

    if(HYPREDRV_ENABLE_EXAMPLES)
        set(_hypredrv_julia_darcy_perm
            "${CMAKE_CURRENT_BINARY_DIR}/darcy_heterogeneous_perm.dat")
        file(WRITE "${_hypredrv_julia_darcy_perm}"
            "1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10\n"
            "2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10\n"
            "3.00 3.10 3.20 3.30 3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10\n")

        set(_hypredrv_julia_darcy_compare_common
            ${Julia_EXECUTABLE} --startup-file=no
            --project=${_hypredrv_julia_test_project}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/darcy_compare.jl
            --julia-exe ${Julia_EXECUTABLE}
            --project ${_hypredrv_julia_test_project}
            --example ${CMAKE_CURRENT_SOURCE_DIR}/examples/darcy.jl
            --c-darcy $<TARGET_FILE:darcy>
        )

        add_test(NAME julia_darcy_help
            COMMAND ${_hypredrv_julia_darcy_compare_common}
                    --mode help
        )
        set_tests_properties(julia_darcy_help PROPERTIES
            LABELS "julia;interface"
            ENVIRONMENT "${_hypredrv_julia_test_env}"
            FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
        )

        add_test(NAME julia_darcy_serial_compare
            COMMAND ${_hypredrv_julia_darcy_compare_common}
                    --mode serial
        )
        set_tests_properties(julia_darcy_serial_compare PROPERTIES
            LABELS "julia;interface"
            ENVIRONMENT "${_hypredrv_julia_test_env}"
            FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
            RUN_SERIAL TRUE
        )

        add_test(NAME julia_darcy_heterogeneous_compare
            COMMAND ${_hypredrv_julia_darcy_compare_common}
                    --mode heterogeneous
                    --perm-file ${_hypredrv_julia_darcy_perm}
        )
        set_tests_properties(julia_darcy_heterogeneous_compare PROPERTIES
            LABELS "julia;interface"
            ENVIRONMENT "${_hypredrv_julia_test_env}"
            FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
            RUN_SERIAL TRUE
        )

        if(MPI_C_FOUND AND MPIEXEC_EXECUTABLE)
            add_test(NAME julia_darcy_mpi_compare
                COMMAND ${_hypredrv_julia_darcy_compare_common}
                        --mode mpi
                        --mpiexec ${MPIEXEC_EXECUTABLE}
                        --np-flag ${MPIEXEC_NUMPROC_FLAG}
            )
            set_tests_properties(julia_darcy_mpi_compare PROPERTIES
                LABELS "julia;interface;mpi"
                ENVIRONMENT "${_hypredrv_julia_test_env}"
                FIXTURES_REQUIRED "julia_built;julia_project_instantiated"
            )
        endif()
    endif()

    if(HYPREDRV_BUILD_TREE_EXPORT_AVAILABLE)
        add_test(NAME julia_cmake_consumer
            COMMAND ${CMAKE_COMMAND}
                -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/cmake_consumer
                -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake_consumer
                -DHYPREDRV_CMAKE_DIR=${CMAKE_BINARY_DIR}
                "-DCONSUMER_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                -DCONSUMER_CONFIG=$<CONFIG>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test/cmake_consumer/run_consumer.cmake
        )
        set_tests_properties(julia_cmake_consumer PROPERTIES
            LABELS "julia;interface;cmake"
            FIXTURES_REQUIRED julia_built
        )
    else()
        add_test(NAME julia_cmake_consumer
            COMMAND ${CMAKE_COMMAND} -E echo
                    "SKIP: HYPREDRV build-tree export was not generated")
        set_tests_properties(julia_cmake_consumer PROPERTIES
            LABELS "julia;interface;cmake"
            DISABLED TRUE
        )
    endif()

    set(_hypredrv_julia_install_consumer_prefix
        "${CMAKE_CURRENT_BINARY_DIR}/install-consumer-prefix")
    # Install-tree consumption needs the exported core package, installed Julia
    # bridge, and installed test fixture, so this target intentionally installs
    # the whole project into an isolated prefix before configuring the consumer.
    add_custom_target(julia-install-consumer-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_hypredrv_julia_install_consumer_prefix}
        COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}
                --prefix ${_hypredrv_julia_install_consumer_prefix}
                --config $<CONFIG>
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${_hypredrv_julia_install_consumer_prefix}/${CMAKE_INSTALL_DATADIR}/julia/HypreDrive/test/cmake_consumer
            -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake_consumer_install
            -DHYPREDRV_CMAKE_DIR=${_hypredrv_julia_install_consumer_prefix}/${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV
            "-DCONSUMER_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
            -DCONSUMER_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test/cmake_consumer/run_consumer.cmake
        DEPENDS HYPREDRV_Julia HYPREDRV hypredrive-cli
        COMMENT "Installing hypredrive and building a Julia bridge CMake consumer"
        VERBATIM
    )

    add_test(NAME julia_install_cmake_consumer
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                --target julia-install-consumer-test --config $<CONFIG>
    )
    set_tests_properties(julia_install_cmake_consumer PROPERTIES
        LABELS "julia;interface;cmake;install"
        FIXTURES_REQUIRED julia_built
        RESOURCE_LOCK julia_install_consumer
    )

    add_custom_target(julia-test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L julia
        DEPENDS HYPREDRV_Julia
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running hypredrive Julia tests"
        VERBATIM
    )
else()
    if(NOT Julia_EXECUTABLE)
        set(_hypredrv_julia_test_message "Julia executable not found; cannot run Julia tests")
    else()
        set(_hypredrv_julia_test_message "Configure with HYPREDRV_ENABLE_TESTING=ON to run Julia tests")
    endif()

    add_custom_target(julia-test
        COMMAND ${CMAKE_COMMAND} -E echo "${_hypredrv_julia_test_message}"
        COMMAND ${CMAKE_COMMAND} -E false
        COMMENT "Julia tests unavailable"
        VERBATIM
    )
endif()
