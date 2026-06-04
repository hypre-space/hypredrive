cmake_minimum_required(VERSION 3.12)
project(darcy C)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR
           "In-source builds are not allowed. Please create a separate build directory and run CMake from there.")
endif()

set(CMAKE_C_STANDARD 99)

if(NOT TARGET HYPREDRV::HYPREDRV)
    find_package(HYPREDRV REQUIRED)
endif()

add_executable(darcy darcy.c)
target_link_libraries(darcy PRIVATE HYPREDRV::HYPREDRV m)
if(COMMAND hypredrv_set_relative_install_rpath)
    hypredrv_set_relative_install_rpath(darcy)
endif()

if(HYPREDRV_ENABLE_CALIPER)
    enable_language(CXX OPTIONAL)
    if(CMAKE_CXX_COMPILER)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(darcy PRIVATE stdc++)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            if(UNIX AND NOT APPLE)
                target_link_libraries(darcy PRIVATE stdc++)
            else()
                target_link_libraries(darcy PRIVATE c++)
            endif()
        endif()
    endif()
endif()

get_property(_sanitizer_enabled GLOBAL PROPERTY HYPREDRV_SANITIZER_ENABLED)
get_property(_sanitizer_flags GLOBAL PROPERTY HYPREDRV_SANITIZER_LINK_FLAGS)
if(_sanitizer_enabled AND _sanitizer_flags)
    foreach(flag IN LISTS _sanitizer_flags)
        target_link_options(darcy PRIVATE ${flag})
    endforeach()
endif()

if(APPLE)
    target_link_options(darcy PRIVATE "-Wl,-rpath,${CMAKE_BINARY_DIR}/lib")
endif()

if(HYPREDRV_ENABLE_TESTING)
    include(${CMAKE_SOURCE_DIR}/cmake/HYPREDRV_Testing.cmake)

    set(_darcy_heterogeneous_perm "${CMAKE_CURRENT_BINARY_DIR}/darcy_heterogeneous_perm.dat")
    file(WRITE "${_darcy_heterogeneous_perm}"
        "1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10\n"
        "2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10\n"
        "3.00 3.10 3.20 3.30 3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10\n")

    add_executable_test(darcy_rt0_1proc darcy 1
        ARGS -n 4 3 1 -g x -v 1
        RUN_SERIAL
        REQUIRE_CONTAINS
            "Darcy Mixed Problem Setup"
            "MPI grid partition"
            "Row partition"
            "relative pressure L2 error"
    )
    add_executable_test(darcy_rt0_2proc darcy 2
        ARGS -n 4 3 1 -P 1 2 1 -g x -v 1
        REQUIRE_CONTAINS
            "MPI grid partition:   1 x 2 x 1 ranks"
            "relative pressure L2 error"
    )
    add_executable_test(darcy_rt0_heterogeneous_1proc darcy 1
        ARGS -n 4 3 1 --K-file "${_darcy_heterogeneous_perm}" -g x -v 0
        RUN_SERIAL
        REQUIRE_CONTAINS
            "heterogeneous pressure solve completed"
    )
    add_executable_test(darcy_hypredrive_args darcy 1
        ARGS -n 4 3 1 -g x -v 0 -a
             --general:statistics on
             --solver:gmres:print_level 1
        RUN_SERIAL
        REQUIRE_CONTAINS
            "STATISTICS SUMMARY"
    )
    add_executable_test(darcy_help darcy 1
        ARGS --help
        RUN_SERIAL
        REQUIRE_CONTAINS
            "Usage:"
            "Active mesh examples:"
            "--procs"
            "--args"
            "--K-file-grid"
            "Verbosity bitset"
    )
endif()
