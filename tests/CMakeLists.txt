# Unit tests for hypredrive
# Helper function to set up test executable with proper RPATH
# Similar to hypredrive executable - link directly to both HYPREDRV and HYPRE
function(setup_test_executable target_name source_file)
    add_executable(${target_name} ${source_file})

    # Link to both HYPREDRV and HYPRE, same as hypredrive executable does
    target_link_libraries(${target_name} PRIVATE HYPREDRV::HYPREDRV)

    # If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
    if(HYPREDRV_ENABLE_CALIPER)
        enable_language(CXX OPTIONAL)
        if(CMAKE_CXX_COMPILER)
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_link_libraries(${target_name} PRIVATE stdc++)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # Clang can use either libstdc++ or libc++
                # Default to stdc++ on Linux
                if(UNIX AND NOT APPLE)
                    target_link_libraries(${target_name} PRIVATE stdc++)
                else()
                    target_link_libraries(${target_name} PRIVATE c++)
                endif()
            endif()
        endif()
    endif()

    # Apply sanitizer link flags if sanitizers are enabled
    # This ensures the sanitizer runtime libraries are linked
    get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
    get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
    if(_sanitizer_enabled AND _sanitizer_flags)
        foreach(flag IN LISTS _sanitizer_flags)
            target_link_options(${target_name} PRIVATE ${flag})
        endforeach()
    endif()

    # Add an RPATH to the lib directory for MacOS (libraries are in lib/)
    if(APPLE)
        target_link_options(${target_name} PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
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

# Field parsing helpers
setup_test_executable(test_field test_field.c)
add_test(NAME unit_test_field COMMAND test_field)

# Argument parsing and YAML validation tests
setup_test_executable(test_args test_args.c)
add_test(NAME unit_test_args COMMAND test_args)

# Preconditioner configuration helpers
setup_test_executable(test_precon test_precon.c)
add_test(NAME unit_test_precon COMMAND test_precon)
set_tests_properties(unit_test_precon PROPERTIES RUN_SERIAL TRUE)

# Solver argument helpers
setup_test_executable(test_solver test_solver.c)
add_test(NAME unit_test_solver COMMAND test_solver)
set_tests_properties(unit_test_solver PROPERTIES RUN_SERIAL TRUE)

# HYPREDRV API smoke tests
setup_test_executable(test_hypredrv test_hypredrv.c)
add_test(NAME unit_test_hypredrv COMMAND test_hypredrv)
set_tests_properties(unit_test_hypredrv PROPERTIES RUN_SERIAL TRUE)

# Matrix IO helpers
setup_test_executable(test_matrix test_matrix.c)
add_test(NAME unit_test_matrix COMMAND test_matrix)
set_tests_properties(unit_test_matrix PROPERTIES RUN_SERIAL TRUE)

# Vector IO and MPI-aware containers utilities
setup_test_executable(test_vector test_vector.c)
add_test(NAME unit_test_vector COMMAND test_vector)
set_tests_properties(unit_test_vector PROPERTIES RUN_SERIAL TRUE)

# EigSpec configuration helpers
setup_test_executable(test_eigspec test_eigspec.c)
add_test(NAME unit_test_eigspec COMMAND test_eigspec)
