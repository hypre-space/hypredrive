# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Relative install RPATH helper. Installed executables live in bindir and
# installed libraries live in libdir, so examples and tools should find
# HYPREDRV/HYPRE from the same install prefix without LD_LIBRARY_PATH.
function(hypredrv_set_relative_install_rpath target_name)
    if(NOT TARGET ${target_name} OR NOT UNIX)
        return()
    endif()

    if(APPLE)
        set(_hypredrv_origin "@loader_path")
    else()
        set(_hypredrv_origin "$ORIGIN")
    endif()

    get_property(_hypredrv_target_type TARGET ${target_name} PROPERTY TYPE)
    if(_hypredrv_target_type STREQUAL "EXECUTABLE")
        file(RELATIVE_PATH _hypredrv_lib_from_bin
             "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_LIBDIR}")
        set(_hypredrv_install_rpath "${_hypredrv_origin}/${_hypredrv_lib_from_bin}")
    else()
        set(_hypredrv_install_rpath "${_hypredrv_origin}")
    endif()

    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "${_hypredrv_install_rpath}")
endfunction()

hypredrv_set_relative_install_rpath(HYPREDRV)
hypredrv_set_relative_install_rpath(hypredrive-cli)
if(TARGET hypredrive-lsseq)
    hypredrv_set_relative_install_rpath(hypredrive-lsseq)
endif()

function(hypredrv_link_example_omp target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "Unknown example target: ${target_name}")
    endif()

    option(HYPREDRV_ENABLE_EXAMPLE_OMP
        "Enable OpenMP parallel assembly in example drivers" OFF)

    if(NOT HYPREDRV_ENABLE_EXAMPLE_OMP)
        message(STATUS "OpenMP disabled for ${target_name} driver")
        return()
    endif()

    find_package(OpenMP)
    if(NOT OpenMP_C_FOUND)
        message(STATUS
            "OpenMP not found - ${target_name} driver will build without OpenMP")
        return()
    endif()

    set(_hypredrv_omp_uses_threaded_cray_libsci FALSE)
    foreach(_hypredrv_omp_lib_name IN LISTS OpenMP_C_LIB_NAMES)
        if(_hypredrv_omp_lib_name MATCHES "^sci_cray_.*_mp$")
            set(_hypredrv_omp_uses_threaded_cray_libsci TRUE)
        endif()
    endforeach()

    if(TARGET OpenMP::OpenMP_C)
        get_target_property(_hypredrv_omp_link_libraries
            OpenMP::OpenMP_C INTERFACE_LINK_LIBRARIES)
        foreach(_hypredrv_omp_link_library IN LISTS _hypredrv_omp_link_libraries)
            if(_hypredrv_omp_link_library MATCHES "libsci_cray.*_mp\\.so")
                set(_hypredrv_omp_uses_threaded_cray_libsci TRUE)
            endif()
        endforeach()
    endif()

    if(_hypredrv_omp_uses_threaded_cray_libsci AND
       OpenMP_C_FLAGS AND OpenMP_craymp_LIBRARY)
        set(_hypredrv_omp_target "HYPREDRV_${target_name}_CrayOpenMP_C")
        if(NOT TARGET ${_hypredrv_omp_target})
            add_library(${_hypredrv_omp_target} INTERFACE)
            separate_arguments(_hypredrv_omp_c_flags
                NATIVE_COMMAND "${OpenMP_C_FLAGS}")
            target_compile_options(${_hypredrv_omp_target}
                INTERFACE ${_hypredrv_omp_c_flags})
            if(OpenMP_C_INCLUDE_DIRS)
                target_include_directories(${_hypredrv_omp_target}
                    INTERFACE ${OpenMP_C_INCLUDE_DIRS})
            endif()
            target_link_libraries(${_hypredrv_omp_target}
                INTERFACE "${OpenMP_craymp_LIBRARY}")
        endif()

        target_link_libraries(${target_name} PRIVATE ${_hypredrv_omp_target})
        message(STATUS
            "OpenMP enabled for ${target_name} driver "
            "(Cray runtime without threaded LibSci)")
        return()
    endif()

    if(TARGET OpenMP::OpenMP_C)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_C)
        message(STATUS "OpenMP enabled for ${target_name} driver")
    else()
        message(STATUS
            "OpenMP target not available - ${target_name} driver will build without OpenMP")
    endif()
endfunction()

# macOS RPATH settings
if(APPLE)
    # Set library install name to use @rpath
    set_target_properties(HYPREDRV PROPERTIES
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
    
    # Add lib directory to rpath for executables (since libraries are in lib/)
    set(LIB_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    target_link_options(hypredrive-cli PRIVATE "-Wl,-rpath,${LIB_OUTPUT_DIR}")
    
    # Check if HYPRE library is shared
    get_property(HYPRE_LIB_TYPE TARGET HYPRE::HYPRE PROPERTY TYPE)
    if(HYPRE_LIB_TYPE STREQUAL "SHARED_LIBRARY")
        # Use generator expression to get the actual library file at build time
        get_target_property(HYPRE_LIBRARY_FILE_NEW HYPRE::HYPRE IMPORTED_LOCATION)
        if(NOT HYPRE_LIBRARY_FILE_NEW)
            get_target_property(HYPRE_LIBRARY_FILE_NEW HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
        endif()
        if(NOT HYPRE_LIBRARY_FILE_NEW)
            get_target_property(HYPRE_LIBRARY_FILE_NEW HYPRE::HYPRE IMPORTED_LOCATION_DEBUG)
        endif()
        if(HYPRE_LIBRARY_FILE_NEW)
            get_filename_component(HYPRE_LIBRARY_DIR "${HYPRE_LIBRARY_FILE_NEW}" DIRECTORY)
            set(CMAKE_INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
            set_target_properties(hypredrive-cli PROPERTIES INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
            set_target_properties(HYPREDRV PROPERTIES INSTALL_RPATH "${HYPRE_LIBRARY_DIR}")
        endif()
    endif()
endif()
