# Unit tests for hypredrive

if(HYPREDRV_ENABLE_TESTING)
    # Test utilities functions
    add_executable(test_utils test_utils.c)
    target_link_libraries(test_utils PRIVATE HYPREDRV)
    add_test(NAME unit_test_utils COMMAND test_utils)
    
    # Test container functions
    add_executable(test_containers test_containers.c)
    target_link_libraries(test_containers PRIVATE HYPREDRV)
    add_test(NAME unit_test_containers COMMAND test_containers)
    
    # Test YAML parsing functions
    add_executable(test_yaml test_yaml.c)
    target_link_libraries(test_yaml PRIVATE HYPREDRV)
    add_test(NAME unit_test_yaml COMMAND test_yaml)
    
    # Test error handling functions
    add_executable(test_error test_error.c)
    target_link_libraries(test_error PRIVATE HYPREDRV)
    add_test(NAME unit_test_error COMMAND test_error)
endif()

