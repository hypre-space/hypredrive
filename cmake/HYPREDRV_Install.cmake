# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Installation rules
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_ExportHelpers.cmake)

function(_hypredrv_install_hypre_public_headers_from_target target)
    get_target_property(_hypre_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _hypre_include_dirs)
        return()
    endif()

    set(_hypre_installed_header_dirs "")
    foreach(_hypre_include_dir IN LISTS _hypre_include_dirs)
        if("${_hypre_include_dir}" MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
            set(_hypre_include_dir "${CMAKE_MATCH_1}")
        elseif("${_hypre_include_dir}" MATCHES "^\\$<")
            continue()
        endif()

        if(EXISTS "${_hypre_include_dir}/HYPRE.h")
            list(FIND _hypre_installed_header_dirs "${_hypre_include_dir}" _hypre_header_dir_index)
            if(_hypre_header_dir_index EQUAL -1)
                install(DIRECTORY "${_hypre_include_dir}/"
                        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                        FILES_MATCHING
                        PATTERN "*.h"
                        PATTERN "*.hpp")
                list(APPEND _hypre_installed_header_dirs "${_hypre_include_dir}")
            endif()
        endif()
    endforeach()
endfunction()

function(_hypredrv_target_links_mpi out_var target)
    set(_links_mpi FALSE)
    if(TARGET ${target})
        foreach(_link_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_link_items ${target} ${_link_property})
            if(_link_items)
                foreach(_link_item IN LISTS _link_items)
                    if("${_link_item}" MATCHES "(^|[:>])MPI::MPI_C($|[>,])")
                        set(_links_mpi TRUE)
                    endif()
                endforeach()
            endif()
        endforeach()
    endif()
    set(${out_var} ${_links_mpi} PARENT_SCOPE)
endfunction()

# Optional lsseq pack/unpack driver (HYPREDRV_ENABLE_COMPRESSION)
set(_hypredrive_executables hypredrive-cli)
if(TARGET hypredrive-lsseq)
    list(APPEND _hypredrive_executables hypredrive-lsseq)
endif()

install(TARGETS ${_hypredrive_executables} HYPREDRV
        EXPORT HYPREDRVTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(TARGET HYPRE::HYPRE)
    get_target_property(_hypre_imported_for_install HYPRE::HYPRE IMPORTED)
    if(NOT _hypre_imported_for_install)
        get_target_property(_hypre_concrete_target HYPRE::HYPRE ALIASED_TARGET)
        if(NOT _hypre_concrete_target AND TARGET HYPRE)
            set(_hypre_concrete_target HYPRE)
        endif()
        if(_hypre_concrete_target)
            # FetchContent HYPRE may not install an export/config target into
            # the HYPREDRV prefix. HYPREDRVConfig.cmake therefore keeps captured
            # include/link metadata as the intentional fallback for consumers.
            install(TARGETS ${_hypre_concrete_target}
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
            _hypredrv_install_hypre_public_headers_from_target(${_hypre_concrete_target})
        endif()
    endif()
endif()

# Create and install Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

set(HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS "")
set(HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES "")
set(HYPREDRV_CONFIG_CALIPER_DIR "")
set(HYPREDRV_CONFIG_NEEDS_ZLIB OFF)
set(HYPREDRV_CONFIG_NEEDS_MPI OFF)
set(HYPREDRV_CONFIG_ENABLE_FORTRAN ${HYPREDRV_ENABLE_FORTRAN})

function(_hypredrv_make_package_relative_list out_var)
    set(_result "")
    foreach(_item IN LISTS ARGN)
        if("${_item}" MATCHES "^\\$<LINK_ONLY:(.*)>$")
            set(_inner "${CMAKE_MATCH_1}")
            _hypredrv_make_package_relative_list(_inner_result ${_inner})
            list(APPEND _result "$<LINK_ONLY:${_inner_result}>")
        elseif("${_item}" MATCHES "^\\$<.*>$")
            list(APPEND _result "${_item}")
        elseif(IS_ABSOLUTE "${_item}")
            file(RELATIVE_PATH _rel "${CMAKE_INSTALL_PREFIX}" "${_item}")
            if(NOT _rel MATCHES "^\\.\\.")
                list(APPEND _result "\${PACKAGE_PREFIX_DIR}/${_rel}")
            else()
                list(APPEND _result "${_item}")
            endif()
        else()
            list(APPEND _result "${_item}")
        endif()
    endforeach()
    set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

function(_hypredrv_get_imported_location out_var target)
    set(_location "")
    get_target_property(_imported ${target} IMPORTED)
    if(NOT _imported)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()
    get_target_property(_target_type ${target} TYPE)
    if(_target_type STREQUAL "INTERFACE_LIBRARY")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()
    foreach(_config IN ITEMS RELEASE RELWITHDEBINFO DEBUG MINSIZEREL "")
        if(_config STREQUAL "")
            get_target_property(_candidate ${target} IMPORTED_LOCATION)
        else()
            get_target_property(_candidate ${target} IMPORTED_LOCATION_${_config})
        endif()
        if(_candidate)
            set(_location "${_candidate}")
            break()
        endif()
    endforeach()
    set(${out_var} "${_location}" PARENT_SCOPE)
endfunction()

set(_hypredrv_config_hypre_imported FALSE)
if(TARGET HYPRE::HYPRE)
    get_target_property(_hypredrv_config_hypre_imported HYPRE::HYPRE IMPORTED)
endif()
if(TARGET HYPRE::HYPRE AND _hypredrv_config_hypre_imported)
    get_target_property(HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS
                        HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES
                        HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
    _hypredrv_get_imported_location(_hypredrv_config_hypre_location HYPRE::HYPRE)
    if(_hypredrv_config_hypre_location)
        if(HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES)
            list(PREPEND HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES
                 "${_hypredrv_config_hypre_location}")
        else()
            set(HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES
                "${_hypredrv_config_hypre_location}")
        endif()
    endif()
    if(NOT HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS)
        set(HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS "")
    endif()
    if(NOT HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES)
        set(HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES "")
    endif()
    _hypredrv_make_package_relative_list(
        HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS
        ${HYPREDRV_CONFIG_HYPRE_INCLUDE_DIRS})
    _hypredrv_make_package_relative_list(
        HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES
        ${HYPREDRV_CONFIG_HYPRE_LINK_LIBRARIES})
endif()
if(TARGET ZLIB::ZLIB)
    set(HYPREDRV_CONFIG_NEEDS_ZLIB ON)
endif()
if(TARGET caliper AND caliper_DIR)
    _hypredrv_make_package_relative_list(
        HYPREDRV_CONFIG_CALIPER_DIR "${caliper_DIR}")
endif()
_hypredrv_target_links_mpi(_hypredrv_hypredrv_links_mpi HYPREDRV)
_hypredrv_target_links_mpi(_hypredrv_hypre_links_mpi HYPRE::HYPRE)
if(_hypredrv_hypredrv_links_mpi OR _hypredrv_hypre_links_mpi)
    set(HYPREDRV_CONFIG_NEEDS_MPI ON)
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HYPREDRVConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV
)

# Export the targets
install(EXPORT HYPREDRVTargets
        FILE HYPREDRVTargets.cmake
        NAMESPACE HYPREDRV::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV)

# Install the Config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HYPREDRV
)

# Export the targets for use in the build tree only when non-imported
# dependencies will not make CMake reject the build-tree export.
if(HYPREDRV_BUILD_TREE_EXPORT_AVAILABLE)
    export(EXPORT HYPREDRVTargets
           FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVTargets.cmake"
           NAMESPACE HYPREDRV::)
else()
    message(STATUS "Skipping build-tree export (HYPRE/Caliper/SuperLU_DIST built via FetchContent)")
endif()

# Register package in user's package registry
export(PACKAGE HYPREDRV)
