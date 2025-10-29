# Installation rules
install(TARGETS hypredrive HYPREDRV
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
export(EXPORT HYPREDRVTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/HYPREDRVTargets.cmake"
       NAMESPACE HYPREDRV::)

# Register package in user's package registry
export(PACKAGE HYPREDRV)
