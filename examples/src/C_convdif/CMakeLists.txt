cmake_minimum_required(VERSION 3.12)
project(convdif C)

# Disallow in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

set(CMAKE_C_STANDARD 99)

# Check if we're building as part of the main project
if(NOT TARGET HYPREDRV::HYPREDRV)
    # Standalone build - try to find installed HYPREDRV
    find_package(HYPREDRV REQUIRED)
endif()

# Common build configuration
add_executable(convdif convdif.c)
target_link_libraries(convdif PRIVATE HYPREDRV::HYPREDRV m)
if(COMMAND hypredrv_set_relative_install_rpath)
    hypredrv_set_relative_install_rpath(convdif)
endif()

# If Caliper is enabled, we need to link C++ standard library (Caliper is C++)
if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(convdif PRIVATE stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang can use either libstdc++ or libc++
            # Default to stdc++ on Linux
            if(UNIX AND NOT APPLE)
                target_link_libraries(convdif PRIVATE stdc++)
            else()
                target_link_libraries(convdif PRIVATE c++)
            endif()
        endif()
    endif()
endif()

# Apply sanitizer link flags if sanitizers are enabled
get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(convdif PRIVATE ${flag})
    endforeach()
endif()

# Add an RPATH to the lib directory for MacOS (libraries are in lib/)
if(APPLE)
    target_link_options(convdif PRIVATE  "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

# Add the convdif driver to the test suite when testing is enabled
if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)
    # Single-rank test: default solver presets (GMRES + BoomerAMG)
    add_executable_test(convdif_test_1proc convdif 1
        ARGS -n 32 8 8 -nt 5 -v 1
        RUN_SERIAL
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    # Single-rank tests: one per preconditioner variant showcased by this example
    add_executable_test(convdif_test_air_1proc convdif 1
        ARGS -n 32 8 8 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-air.yml
        RUN_SERIAL
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    add_executable_test(convdif_test_ilu_1proc convdif 1
        ARGS -n 32 8 8 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-ilu.yml
        RUN_SERIAL
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    add_executable_test(convdif_test_amg_1proc convdif 1
        ARGS -n 32 8 8 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-amg.yml
        RUN_SERIAL
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    # Swirling flow: exercises the transverse convective fluxes
    add_executable_test(convdif_test_swirl_1proc convdif 1
        ARGS -n 32 8 8 -w 2 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-air.yml
        RUN_SERIAL
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    # Multi-rank test: 2x2x1 partitioning
    add_executable_test(convdif_test_4proc convdif 4
        ARGS -n 32 8 8 -P 2 2 1 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-air.yml
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
    # Multi-rank swirling flow: transverse fluxes crossing rank boundaries
    add_executable_test(convdif_test_swirl_4proc convdif 4
        ARGS -n 32 8 8 -P 2 2 1 -w 2 -nt 5 -v 1 -i ${CMAKE_CURRENT_SOURCE_DIR}/gmres-air.yml
        REQUIRE_CONTAINS "STATISTICS SUMMARY"
    )
endif()

# Install the convdif example binary when examples are built
install(TARGETS convdif
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
