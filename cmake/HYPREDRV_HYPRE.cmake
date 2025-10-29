# Find and configure HYPRE
find_package(HYPRE REQUIRED CONFIG)
if(NOT HYPRE_FOUND)
    message(FATAL_ERROR "HYPRE library not found. Please specify -DHYPRE_ROOT=<path> to the root of the HYPRE installation.")
endif()

get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)

# Try to get library location - handle both Release and Debug configurations
get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION)
if(NOT HYPRE_LIBRARY_FILE)
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
endif()
if(NOT HYPRE_LIBRARY_FILE)
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_DEBUG)
endif()
if(NOT HYPRE_LIBRARY_FILE)
    set(HYPRE_LIBRARY_FILE "not found (using target)")
endif()

message(STATUS "Found HYPRE:")
message(STATUS "  include directories: ${HYPRE_INCLUDE_DIRS}")
message(STATUS "  libraries: ${HYPRE_LIBRARY_FILE}")
