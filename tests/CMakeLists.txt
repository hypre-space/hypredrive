# Unit tests for hypredrive
# Helper function to set up test executable with proper RPATH
function(setup_test_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE HYPREDRV)
    # Set RPATH for macOS to find the library at build time
    # Use @loader_path/.. since tests are in build/tests/ and library is in build/
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "@loader_path/.."
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${CMAKE_BINARY_DIR}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
        )
    endif()
endfunction()

# Test utilities functions
setup_test_executable(test_utils test_utils.c)
add_test(NAME unit_test_utils COMMAND test_utils)

# Test container functions
setup_test_executable(test_containers test_containers.c)
add_test(NAME unit_test_containers COMMAND test_containers)

# Test YAML parsing functions
setup_test_executable(test_yaml test_yaml.c)
add_test(NAME unit_test_yaml COMMAND test_yaml)

# Test error handling functions
setup_test_executable(test_error test_error.c)
add_test(NAME unit_test_error COMMAND test_error)

