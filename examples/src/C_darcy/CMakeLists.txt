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

    add_executable_test(darcy_rt0_1proc darcy 1
        ARGS -n 4 3 1 -g x -v 1
        RUN_SERIAL
        REQUIRE_CONTAINS
            "Darcy Mixed Problem Setup"
            "relative pressure L2 error"
    )
    add_executable_test(darcy_rt0_2proc darcy 2
        ARGS -n 4 3 1 -g x -v 1
        REQUIRE_CONTAINS
            "relative pressure L2 error"
    )
endif()
