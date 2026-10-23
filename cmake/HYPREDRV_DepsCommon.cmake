# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


############################################################
# Dependency discovery helpers
############################################################

function(_hypredrv_set_common_output_directories)
    # Keep all generated artifacts in predictable top-level output paths.
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH
        "Single output directory for all libraries" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH
        "Single output directory for all static libraries" FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE PATH
        "Single output directory for all executables" FORCE)
endfunction()

function(_hypredrv_set_cache_bool_default var_name value doc)
    get_property(_hypredrv_cache_var_exists CACHE "${var_name}" PROPERTY TYPE SET)
    if(NOT _hypredrv_cache_var_exists)
        set("${var_name}" "${value}" CACHE BOOL "${doc}")
    endif()
endfunction()

function(_hypredrv_seed_msmpi_library)
    if(NOT WIN32 OR NOT DEFINED ENV{MSMPI_LIB64} OR MPI_msmpi_LIBRARY)
        return()
    endif()

    file(TO_CMAKE_PATH "$ENV{MSMPI_LIB64}" _hypredrv_msmpi_lib_dir)
    find_file(_hypredrv_msmpi_library
        NAMES msmpi64.lib msmpi.lib
        HINTS "${_hypredrv_msmpi_lib_dir}"
        NO_DEFAULT_PATH)
    if(_hypredrv_msmpi_library)
        set(MPI_msmpi_LIBRARY "${_hypredrv_msmpi_library}" CACHE FILEPATH
            "Location of the Microsoft MPI library" FORCE)
        if(NOT DEFINED MPI_GUESS_LIBRARY_NAME)
            set(MPI_GUESS_LIBRARY_NAME MSMPI CACHE STRING
                "MPI implementation to prefer when locating MPI" FORCE)
        endif()
    endif()
endfunction()

include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsPatches.cmake)

function(_hypredrv_link_mpi_interface target_name)
    if(TARGET MPI::MPI_C)
        target_link_libraries(${target_name} INTERFACE MPI::MPI_C)
    else()
        if(MPI_C_LIBRARIES)
            target_link_libraries(${target_name} INTERFACE ${MPI_C_LIBRARIES})
        endif()
        if(MPI_C_INCLUDE_DIRS)
            target_include_directories(${target_name} INTERFACE ${MPI_C_INCLUDE_DIRS})
        endif()
    endif()
endfunction()

function(_hypredrv_collect_plain_include_dirs_for_deps out_var)
    set(_plain_include_dirs "")
    foreach(_inc_dir IN LISTS ARGN)
        if(NOT _inc_dir)
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
            list(APPEND _plain_include_dirs "${CMAKE_MATCH_1}")
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<INSTALL_INTERFACE:")
            continue()
        endif()
        if(_inc_dir MATCHES "^\\$<TARGET_PROPERTY:")
            continue()
        endif()
        list(APPEND _plain_include_dirs "${_inc_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _plain_include_dirs)
    set(${out_var} "${_plain_include_dirs}" PARENT_SCOPE)
endfunction()

function(hypredrv_collect_hypre_include_dirs out_var)
    set(options INCLUDE_SOURCE_SUBDIRS)
    cmake_parse_arguments(PARSE_ARGV 1 _hypredrv_hypre_includes
                          "${options}" "" "")

    set(_hypredrv_hypre_collected ${HYPRE_INCLUDE_DIRS})
    if(NOT _hypredrv_hypre_collected AND TARGET HYPRE::HYPRE)
        get_target_property(_hypredrv_hypre_collected
                            HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    if(_hypredrv_hypre_collected)
        list(FILTER _hypredrv_hypre_collected EXCLUDE REGEX "^\\$<")
    endif()

    foreach(_hypredrv_hypre_dir IN ITEMS
            "${HYPRE_SOURCE_DIR}"
            "${HYPRE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/_deps/hypre-src/src"
            "${CMAKE_BINARY_DIR}/_deps/hypre-build")
        if(_hypredrv_hypre_dir AND EXISTS "${_hypredrv_hypre_dir}")
            list(APPEND _hypredrv_hypre_collected "${_hypredrv_hypre_dir}")
        endif()
    endforeach()

    if(_hypredrv_hypre_includes_INCLUDE_SOURCE_SUBDIRS)
        set(_hypredrv_hypre_subdirs
            blas
            distributed_ls
            distributed_matrix
            IJ_mv
            krylov
            lapack
            matrix_matrix
            multivector
            parcsr_block_mv
            parcsr_ls
            parcsr_mv
            seq_block_mv
            seq_mv
            sstruct_ls
            sstruct_mv
            struct_ls
            struct_mv
            utilities)
        foreach(_hypredrv_hypre_root IN ITEMS
                "${HYPRE_SOURCE_DIR}"
                "${CMAKE_BINARY_DIR}/_deps/hypre-src/src")
            foreach(_hypredrv_hypre_subdir IN LISTS _hypredrv_hypre_subdirs)
                if(_hypredrv_hypre_root AND
                   EXISTS "${_hypredrv_hypre_root}/${_hypredrv_hypre_subdir}")
                    list(APPEND _hypredrv_hypre_collected
                         "${_hypredrv_hypre_root}/${_hypredrv_hypre_subdir}")
                endif()
            endforeach()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES _hypredrv_hypre_collected)
    set(${out_var} ${_hypredrv_hypre_collected} PARENT_SCOPE)
endfunction()

function(_hypredrv_detect_hypre_sycl_usage)
    set(HYPREDRV_HYPRE_USES_SYCL FALSE PARENT_SCOPE)

    if(NOT TARGET HYPRE::HYPRE)
        return()
    endif()

    get_target_property(_hypre_compile_defs HYPRE::HYPRE INTERFACE_COMPILE_DEFINITIONS)
    if(_hypre_compile_defs)
        foreach(_def IN LISTS _hypre_compile_defs)
            if(_def MATCHES "(^|[^A-Za-z0-9_])HYPRE_USING_SYCL([^A-Za-z0-9_]|$)")
                set(HYPREDRV_HYPRE_USES_SYCL TRUE PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    get_target_property(_hypre_include_dirs HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
    _hypredrv_collect_plain_include_dirs_for_deps(_hypre_plain_include_dirs ${_hypre_include_dirs})
    foreach(_inc_dir IN LISTS _hypre_plain_include_dirs)
        if(EXISTS "${_inc_dir}/HYPRE_config.h")
            file(READ "${_inc_dir}/HYPRE_config.h" _hypre_config_content)
            if(_hypre_config_content MATCHES "#define[ \t]+HYPRE_USING_SYCL([ \t]+1)?")
                set(HYPREDRV_HYPRE_USES_SYCL TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
endfunction()

function(_hypredrv_detect_hypre_openmp_usage)
    # A superbuild may provide this integration hint explicitly. Otherwise
    # infer it from HYPRE's CMake option, target definitions, or config header.
    if(DEFINED HYPREDRV_HYPRE_USE_OPENMP)
        return()
    endif()

    set(HYPREDRV_HYPRE_USE_OPENMP FALSE PARENT_SCOPE)

    if(HYPRE_ENABLE_OPENMP)
        set(HYPREDRV_HYPRE_USE_OPENMP TRUE PARENT_SCOPE)
        return()
    endif()

    if(NOT TARGET HYPRE::HYPRE)
        return()
    endif()

    get_target_property(_hypre_compile_defs HYPRE::HYPRE INTERFACE_COMPILE_DEFINITIONS)
    if(_hypre_compile_defs)
        foreach(_def IN LISTS _hypre_compile_defs)
            if(_def MATCHES "(^|[^A-Za-z0-9_])HYPRE_USING_OPENMP([^A-Za-z0-9_]|$)")
                set(HYPREDRV_HYPRE_USE_OPENMP TRUE PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    get_target_property(_hypre_include_dirs HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
    _hypredrv_collect_plain_include_dirs_for_deps(_hypre_plain_include_dirs ${_hypre_include_dirs})
    foreach(_inc_dir IN LISTS _hypre_plain_include_dirs)
        if(EXISTS "${_inc_dir}/HYPRE_config.h")
            file(READ "${_inc_dir}/HYPRE_config.h" _hypre_config_content)
            if(_hypre_config_content MATCHES "#define[ \t]+HYPRE_USING_OPENMP([ \t]+1)?")
                set(HYPREDRV_HYPRE_USE_OPENMP TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
endfunction()

function(_hypredrv_hypre_needs_cuda_nvjitlink result_var)
    set(_needs_nvjitlink FALSE)

    if(HYPRE_ENABLE_CUDA AND (HYPRE_ENABLE_CUSPARSE OR HYPRE_ENABLE_CUSOLVER))
        set(_needs_nvjitlink TRUE)
    endif()

    if(TARGET HYPRE::HYPRE)
        get_target_property(_hypre_link_libs HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
        if(_hypre_link_libs)
            foreach(_hypre_link_lib IN LISTS _hypre_link_libs)
                if(_hypre_link_lib MATCHES "(^|[:>/])CUDA::(cusparse|cusolver|nvrtc)($|[>,])" OR
                   _hypre_link_lib MATCHES "libcu(sparse|solver)\\.(so|dylib|dll|a)" OR
                   _hypre_link_lib MATCHES "(^|[/-])cu(sparse|solver)$")
                    set(_needs_nvjitlink TRUE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    set(${result_var} ${_needs_nvjitlink} PARENT_SCOPE)
endfunction()

function(_hypredrv_find_cuda_nvjitlink out_var)
    if(TARGET CUDA::nvJitLink)
        set(${out_var} CUDA::nvJitLink PARENT_SCOPE)
        return()
    endif()

    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::nvJitLink)
        set(${out_var} CUDA::nvJitLink PARENT_SCOPE)
        return()
    endif()

    set(_cuda_roots "")
    foreach(_cuda_root IN ITEMS
            "${CUDAToolkit_ROOT}"
            "${CUDA_TOOLKIT_ROOT_DIR}"
            "${CUDA_ROOT}")
        if(_cuda_root)
            list(APPEND _cuda_roots "${_cuda_root}")
        endif()
    endforeach()

    if(CUDAToolkit_BIN_DIR)
        get_filename_component(_cuda_root_from_bin "${CUDAToolkit_BIN_DIR}" DIRECTORY)
        list(APPEND _cuda_roots "${_cuda_root_from_bin}")
    endif()

    set(_cuda_lib_hints "")
    foreach(_cuda_lib IN ITEMS
            "${CUDA_nvJitLink_LIBRARY}"
            "${CUDA_cudart_LIBRARY}"
            "${CUDA_CUDART}")
        if(_cuda_lib)
            get_filename_component(_cuda_lib_dir "${_cuda_lib}" DIRECTORY)
            list(APPEND _cuda_lib_hints "${_cuda_lib_dir}")
        endif()
    endforeach()

    foreach(_cuda_root IN LISTS _cuda_roots)
        list(APPEND _cuda_lib_hints
             "${_cuda_root}/lib64"
             "${_cuda_root}/lib"
             "${_cuda_root}/targets/x86_64-linux/lib")
    endforeach()
    if(CUDAToolkit_LIBRARY_DIR)
        list(APPEND _cuda_lib_hints "${CUDAToolkit_LIBRARY_DIR}")
    endif()

    list(REMOVE_DUPLICATES _cuda_lib_hints)
    find_library(HYPREDRV_CUDA_NVJITLINK_LIBRARY
                 NAMES nvJitLink libnvJitLink.so
                 HINTS ${_cuda_lib_hints}
                 PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)

    if(HYPREDRV_CUDA_NVJITLINK_LIBRARY)
        set(${out_var} "${HYPREDRV_CUDA_NVJITLINK_LIBRARY}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(_hypredrv_ensure_hypre_cuda_nvjitlink)
    if(NOT TARGET HYPRE::HYPRE)
        return()
    endif()

    set(_hypre_link_target HYPRE::HYPRE)
    get_target_property(_hypre_aliased_target HYPRE::HYPRE ALIASED_TARGET)
    if(_hypre_aliased_target)
        set(_hypre_link_target "${_hypre_aliased_target}")
    endif()

    _hypredrv_hypre_needs_cuda_nvjitlink(_needs_nvjitlink)
    if(NOT _needs_nvjitlink)
        return()
    endif()

    get_target_property(_hypre_link_libs ${_hypre_link_target} INTERFACE_LINK_LIBRARIES)
    if(_hypre_link_libs)
        foreach(_hypre_link_lib IN LISTS _hypre_link_libs)
            if(_hypre_link_lib MATCHES "(^|[:>/])CUDA::nvJitLink($|[>,])" OR
               _hypre_link_lib MATCHES "libnvJitLink\\.(so|dylib|dll|a)")
                return()
            endif()
        endforeach()
    endif()

    _hypredrv_find_cuda_nvjitlink(_nvjitlink_lib)
    if(_nvjitlink_lib)
        set_property(TARGET ${_hypre_link_target} APPEND PROPERTY
                     INTERFACE_LINK_LIBRARIES "${_nvjitlink_lib}")
        message(STATUS "  Added CUDA nvJitLink dependency for HYPRE: ${_nvjitlink_lib}")
    else()
        message(WARNING
            "HYPRE links CUDA libraries that may require nvJitLink, but "
            "libnvJitLink was not found. Runtime may require LD_LIBRARY_PATH "
            "to include the CUDA toolkit lib directory.")
    endif()
endfunction()

function(_hypredrv_set_using_caliper enabled)
    if(enabled)
        set(HYPREDRV_USING_CALIPER 1 PARENT_SCOPE)
        set(HYPREDRV_USING_CALIPER 1 CACHE INTERNAL "Using Caliper")
    else()
        set(HYPREDRV_USING_CALIPER 0 PARENT_SCOPE)
        set(HYPREDRV_USING_CALIPER 0 CACHE INTERNAL "Not using Caliper")
    endif()
endfunction()

function(_hypredrv_collect_dsuperlu_pkg_config_hints out_var)
    set(_hint_dirs "")
    foreach(_var IN ITEMS
            DSUPERLU_DIR
            DSUPERLU_ROOT
            SUPERLU_DIST_DIR
            SUPERLU_DIST_ROOT
            superlu_dist_DIR
            superlu_dist_ROOT
            TPL_DSUPERLU_DIR
            TPL_DSUPERLU_ROOT)
        if(NOT DEFINED ${_var})
            continue()
        endif()

        set(_root "${${_var}}")
        if(NOT EXISTS "${_root}")
            continue()
        endif()

        list(APPEND _hint_dirs
            "${_root}"
            "${_root}/lib/pkgconfig"
            "${_root}/lib64/pkgconfig"
            "${_root}/share/pkgconfig")

        get_filename_component(_root_parent "${_root}" DIRECTORY)
        if(_root_parent AND EXISTS "${_root_parent}")
            list(APPEND _hint_dirs
                "${_root_parent}/lib/pkgconfig"
                "${_root_parent}/lib64/pkgconfig"
                "${_root_parent}/share/pkgconfig")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _hint_dirs)
    set(${out_var} "${_hint_dirs}" PARENT_SCOPE)
endfunction()

function(_hypredrv_append_runtime_dirs_from_dependency_roots target_name)
    if(NOT TARGET ${target_name})
        return()
    endif()

    set(_runtime_dirs "")
    foreach(_dep_dir IN LISTS ARGN)
        if(NOT IS_DIRECTORY "${_dep_dir}")
            continue()
        endif()

        if(_dep_dir MATCHES "(/|^)lib64?$")
            list(APPEND _runtime_dirs "${_dep_dir}")
        endif()
        if(IS_DIRECTORY "${_dep_dir}/lib")
            list(APPEND _runtime_dirs "${_dep_dir}/lib")
        endif()
        if(IS_DIRECTORY "${_dep_dir}/lib64")
            list(APPEND _runtime_dirs "${_dep_dir}/lib64")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _runtime_dirs)
    if(_runtime_dirs)
        set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH ${_runtime_dirs})
        set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH ${_runtime_dirs})
    endif()
endfunction()

function(_hypredrv_matches_pattern var_name pattern_list_var result_var)
    set(_matched FALSE)
    foreach(pattern IN LISTS ${pattern_list_var})
        if(var_name MATCHES "${pattern}")
            set(_matched TRUE)
            break()
        endif()
    endforeach()
    set(${result_var} ${_matched} PARENT_SCOPE)
endfunction()

function(_hypredrv_forward_hypre_cache_vars forwarded_count_var)
    # Variables to exclude from inheritance (project-specific)
    set(_exclude_patterns
        "^HYPREDRV_"               # Our project variables
        "^CMAKE_TOOLCHAIN_FILE$"   # Toolchain files shouldn't be inherited
        "^CMAKE_GENERATOR$"        # Generator is already set
        "^CMAKE_SOURCE_DIR$"       # Source directories
        "^CMAKE_BINARY_DIR$"       # Binary directories
        "^CMAKE_PROJECT_NAME$"     # Project name
        "^CMAKE_CURRENT_LIST_DIR$" # Current list directory
        "^CMAKE_CURRENT_SOURCE_DIR$"
        "^CMAKE_CURRENT_BINARY_DIR$"
    )

    # Patterns for variables to include
    set(_include_patterns
        "^HYPRE_ENABLE"        # HYPRE enable options (HYPRE_ENABLE_*, etc.)
        "^TPL_"                # TPL variables (TPL_CALIPER_*, etc.)
        "^.*_ROOT$"            # TPL root variables (MAGMA_ROOT, CUDA_ROOT, etc.)
        "^.*_DIR$"             # TPL directory variables
        "^CMAKE_"              # CMake configuration variables
        "^BUILD_SHARED_LIBS$"  # Library type
    )

    get_cmake_property(_cache_vars CACHE_VARIABLES)
    set(_forwarded_count 0)
    foreach(_var IN LISTS _cache_vars)
        _hypredrv_matches_pattern("${_var}" _exclude_patterns _should_exclude)
        if(_should_exclude)
            continue()
        endif()

        _hypredrv_matches_pattern("${_var}" _include_patterns _should_include)
        if(NOT _should_include)
            continue()
        endif()

        get_property(_var_type CACHE "${_var}" PROPERTY TYPE)
        get_property(_var_value CACHE "${_var}" PROPERTY VALUE)
        set("${_var}" "${_var_value}" CACHE "${_var_type}" "" FORCE)
        math(EXPR _forwarded_count "${_forwarded_count} + 1")
    endforeach()

    set(${forwarded_count_var} ${_forwarded_count} PARENT_SCOPE)
endfunction()

############################################################
