# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Installation rules
# Check if HYPRE was built via FetchContent (not imported) and needs to be exported
set(_hypre_targets_to_install)
if(TARGET HYPRE::HYPRE)
    get_target_property(_hypre_imported HYPRE::HYPRE IMPORTED)
    if(NOT _hypre_imported)
        # HYPRE was built via FetchContent, we need to find the actual target name
        # HYPRE::HYPRE is likely an alias, find the actual target
        get_target_property(_hypre_aliased HYPRE::HYPRE ALIASED_TARGET)
        if(_hypre_aliased)
            # It's an alias, use the aliased target (without namespace)
            list(APPEND _hypre_targets_to_install ${_hypre_aliased})
        elseif(TARGET HYPRE)
            # Try the non-namespaced target name
            list(APPEND _hypre_targets_to_install HYPRE)
        else()
            # Fallback to the namespaced version
            list(APPEND _hypre_targets_to_install HYPRE::HYPRE)
        endif()
        message(STATUS "HYPRE built via FetchContent will be included in export set")
    endif()
endif()

install(TARGETS hypredrive HYPREDRV ${_hypre_targets_to_install}
        EXPORT HYPREDRVTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Create and install Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

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

# Export the targets for use in the build tree
# Only export if all non-imported dependencies are handled in an export set.
# If HYPRE or Caliper were built via FetchContent, skip build-tree export to avoid errors.
set(_caliper_imported TRUE)
if(TARGET caliper)
    get_target_property(_caliper_imported caliper IMPORTED)
elseif(TARGET caliper::caliper)
    get_target_property(_caliper_imported caliper::caliper IMPORTED)
endif()

if(TARGET HYPRE::HYPRE)
    get_target_property(_hypre_imported HYPRE::HYPRE IMPORTED)
    if(_hypre_imported AND _caliper_imported)
        # HYPRE and Caliper are imported, safe to export
        export(EXPORT HYPREDRVTargets
               FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVTargets.cmake"
               NAMESPACE HYPREDRV::)
    else()
        # HYPRE and/or Caliper were built via FetchContent, skip build-tree export to avoid errors
        message(STATUS "Skipping build-tree export (HYPRE/Caliper built via FetchContent)")
    endif()
else()
    if(_caliper_imported)
        # No HYPRE target, export normally
        export(EXPORT HYPREDRVTargets
               FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVTargets.cmake"
               NAMESPACE HYPREDRV::)
    else()
        message(STATUS "Skipping build-tree export (Caliper built via FetchContent)")
    endif()
endif()

# Register package in user's package registry
export(PACKAGE HYPREDRV)
