# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Find and configure SuperLU_DIST (must be before HYPRE)
############################################################

set(SUPERLU_DIST_VERSION "v9.2.1" CACHE STRING
    "SuperLU_DIST version/branch/tag to fetch (e.g., v9.2.1)")

if(HYPRE_ENABLE_DSUPERLU)
    set(SUPERLU_DIST_FOUND FALSE)

    if(HYPRE_BUILD_DSUPERLU)
        message(STATUS "Reconfiguring auto-built SuperLU_DIST dependency")
    elseif(HYPREDRV_BUILD_DSUPERLU)
        message(STATUS
            "HYPREDRV_BUILD_DSUPERLU=ON: SuperLU_DIST will be fetched "
            "and built")
    elseif(TPL_DSUPERLU_INCLUDE_DIRS AND TPL_DSUPERLU_LIBRARIES)
        set(SUPERLU_DIST_FOUND TRUE)
        message(STATUS "Using user-provided SuperLU_DIST include/lib paths for HYPRE")
    elseif(TARGET superlu_dist)
        set(SUPERLU_DIST_FOUND TRUE)
        message(STATUS "Found existing SuperLU_DIST target: superlu_dist")
    else()
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            _hypredrv_collect_dsuperlu_pkg_config_hints(_dsuperlu_pkg_hints)
            set(_dsuperlu_old_pkg_config_path "$ENV{PKG_CONFIG_PATH}")
            if(_dsuperlu_pkg_hints)
                string(JOIN ":" _dsuperlu_pkg_hint_path ${_dsuperlu_pkg_hints})
                if(_dsuperlu_old_pkg_config_path)
                    set(ENV{PKG_CONFIG_PATH}
                        "${_dsuperlu_pkg_hint_path}:$ENV{PKG_CONFIG_PATH}")
                else()
                    set(ENV{PKG_CONFIG_PATH} "${_dsuperlu_pkg_hint_path}")
                endif()
                message(STATUS
                    "Augmented PKG_CONFIG_PATH with hints for SuperLU_DIST: ${_dsuperlu_pkg_hint_path}")
            endif()

            pkg_check_modules(PC_DSUPERLU QUIET IMPORTED_TARGET superlu_dist)
            set(ENV{PKG_CONFIG_PATH} "${_dsuperlu_old_pkg_config_path}")

            if(PC_DSUPERLU_FOUND)
                set(SUPERLU_DIST_FOUND TRUE)
                if(PC_DSUPERLU_INCLUDE_DIRS AND NOT TPL_DSUPERLU_INCLUDE_DIRS)
                    set(TPL_DSUPERLU_INCLUDE_DIRS ${PC_DSUPERLU_INCLUDE_DIRS}
                        CACHE PATH "SuperLU_DIST include directories for HYPRE" FORCE)
                endif()
                if(NOT TPL_DSUPERLU_LIBRARIES)
                    if(PC_DSUPERLU_LINK_LIBRARIES)
                        set(TPL_DSUPERLU_LIBRARIES ${PC_DSUPERLU_LINK_LIBRARIES}
                            CACHE STRING "SuperLU_DIST libraries for HYPRE" FORCE)
                    elseif(PC_DSUPERLU_LDFLAGS)
                        set(TPL_DSUPERLU_LIBRARIES ${PC_DSUPERLU_LDFLAGS}
                            CACHE STRING "SuperLU_DIST libraries for HYPRE" FORCE)
                    endif()
                endif()
                if(PC_DSUPERLU_LIBRARY_DIRS)
                    foreach(_dsuperlu_lib_dir IN LISTS PC_DSUPERLU_LIBRARY_DIRS)
                        list(APPEND HYPRE_DEPENDENCY_DIRS "${_dsuperlu_lib_dir}")
                        get_filename_component(_dsuperlu_prefix "${_dsuperlu_lib_dir}" DIRECTORY)
                        list(APPEND HYPRE_DEPENDENCY_DIRS "${_dsuperlu_prefix}")
                    endforeach()
                    list(REMOVE_DUPLICATES HYPRE_DEPENDENCY_DIRS)
                    set(HYPRE_DEPENDENCY_DIRS "${HYPRE_DEPENDENCY_DIRS}" CACHE INTERNAL "" FORCE)
                endif()
                message(STATUS "Found SuperLU_DIST via pkg-config:")
                message(STATUS "  include directories: ${TPL_DSUPERLU_INCLUDE_DIRS}")
                message(STATUS "  libraries: ${TPL_DSUPERLU_LIBRARIES}")
            endif()
        endif()
    endif()

    if(NOT SUPERLU_DIST_FOUND)
        message(STATUS
            "SuperLU_DIST not found. Fetching and building SuperLU_DIST from source (version: ${SUPERLU_DIST_VERSION})...")

        include(FetchContent)

        _hypredrv_seed_msmpi_library()
        find_package(MPI REQUIRED COMPONENTS C CXX)

        FetchContent_Declare(
            superlu_dist
            GIT_REPOSITORY https://github.com/xiaoyeli/superlu_dist.git
            GIT_TAG        ${SUPERLU_DIST_VERSION}
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )

        set(FETCHCONTENT_QUIET OFF)

        # Default to the small CPU-only build HYPRE's DSUPERLU path needs, while
        # preserving explicit user cache values such as -DTPL_ENABLE_PARMETISLIB=ON.
        _hypredrv_set_cache_bool_default(enable_doc OFF
            "Build SuperLU_DIST documentation")
        _hypredrv_set_cache_bool_default(enable_single OFF
            "Build SuperLU_DIST single precision library")
        _hypredrv_set_cache_bool_default(enable_double ON
            "Build SuperLU_DIST double precision library")
        _hypredrv_set_cache_bool_default(enable_complex16 OFF
            "Build SuperLU_DIST complex16 precision library")
        _hypredrv_set_cache_bool_default(enable_tests OFF
            "Build SuperLU_DIST tests")
        _hypredrv_set_cache_bool_default(enable_examples OFF
            "Build SuperLU_DIST examples")
        _hypredrv_set_cache_bool_default(enable_python OFF
            "Build SuperLU_DIST Python interface")
        _hypredrv_set_cache_bool_default(enable_openmp OFF
            "Build SuperLU_DIST with OpenMP support")
        _hypredrv_set_cache_bool_default(XSDK_ENABLE_Fortran OFF
            "Build SuperLU_DIST Fortran interface")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_PARMETISLIB OFF
            "Enable SuperLU_DIST ParMETIS support")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_COMBBLASLIB OFF
            "Enable SuperLU_DIST CombBLAS support")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_COLAMDLIB OFF
            "Enable SuperLU_DIST COLAMD support")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_LAPACKLIB OFF
            "Enable SuperLU_DIST LAPACK support")
        if(HYPREDRV_ENABLE_CUDA)
            set(TPL_ENABLE_CUDALIB ON CACHE BOOL
                "Enable SuperLU_DIST CUDA support" FORCE)
        else()
            _hypredrv_set_cache_bool_default(TPL_ENABLE_CUDALIB OFF
                "Enable SuperLU_DIST CUDA support")
        endif()
        if(HYPREDRV_ENABLE_HIP)
            set(TPL_ENABLE_HIPLIB ON CACHE BOOL
                "Enable SuperLU_DIST HIP support" FORCE)
        else()
            _hypredrv_set_cache_bool_default(TPL_ENABLE_HIPLIB OFF
                "Enable SuperLU_DIST HIP support")
        endif()
        if(HYPREDRV_ENABLE_SYCL)
            message(STATUS
                "SuperLU_DIST ${SUPERLU_DIST_VERSION} has no SYCL backend; "
                "building its CPU implementation for SYCL-enabled HYPRE")
        endif()
        _hypredrv_set_cache_bool_default(TPL_ENABLE_NVSHMEM OFF
            "Enable SuperLU_DIST NVSHMEM support")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_ROCSHMEM OFF
            "Enable SuperLU_DIST ROCSHMEM support")
        _hypredrv_set_cache_bool_default(TPL_ENABLE_MAGMALIB OFF
            "Enable SuperLU_DIST MAGMA support")
        if(TPL_BLAS_LIBRARIES)
            _hypredrv_set_cache_bool_default(TPL_ENABLE_INTERNAL_BLASLIB OFF
                "Build SuperLU_DIST internal BLAS")
        else()
            _hypredrv_set_cache_bool_default(TPL_ENABLE_INTERNAL_BLASLIB ON
                "Build SuperLU_DIST internal BLAS")
        endif()
        if(NOT enable_double)
            message(WARNING
                "HYPRE_ENABLE_DSUPERLU=ON needs SuperLU_DIST double-precision "
                "support, but enable_double=OFF is set in the CMake cache")
        endif()

        _hypredrv_set_common_output_directories()

        set(_hypredrv_superlu_dist_static_build OFF)
        if(NOT BUILD_SHARED_LIBS)
            set(_hypredrv_superlu_dist_static_build ON)
        endif()
        message(STATUS
            "  Using inherited BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} for SuperLU_DIST build")

        message(STATUS "Fetching SuperLU_DIST from GitHub (branch/tag: ${SUPERLU_DIST_VERSION})...")
        message(STATUS "  Repository: https://github.com/xiaoyeli/superlu_dist.git")

        FetchContent_MakeAvailable(superlu_dist)

        if(TARGET superlu_dist)
            set_target_properties(superlu_dist PROPERTIES EXCLUDE_FROM_ALL FALSE)
            target_include_directories(superlu_dist PUBLIC
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/include>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/cuda>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/hip>
                $<BUILD_INTERFACE:${superlu_dist_BINARY_DIR}/SRC>)
        endif()
        if(TARGET superlu_dist-static)
            target_include_directories(superlu_dist-static PUBLIC
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/include>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/cuda>
                $<BUILD_INTERFACE:${superlu_dist_SOURCE_DIR}/SRC/hip>
                $<BUILD_INTERFACE:${superlu_dist_BINARY_DIR}/SRC>)
        endif()

        if(_hypredrv_superlu_dist_static_build AND TARGET superlu_dist)
            get_target_property(_hypredrv_superlu_dist_target_type
                                superlu_dist TYPE)
            if(NOT _hypredrv_superlu_dist_target_type STREQUAL
                   "STATIC_LIBRARY")
                message(FATAL_ERROR
                    "The bundled HYPRE build is static, but SuperLU_DIST "
                    "target superlu_dist is ${_hypredrv_superlu_dist_target_type}")
            endif()
            message(STATUS
                "  Verified bundled SuperLU_DIST target is static")
        endif()

        # HYPRE needs concrete library paths in TPL_DSUPERLU_LIBRARIES. These paths
        # assume the project's single-config, unified output directories.
        if(NOT _hypredrv_superlu_dist_static_build)
            set(SUPERLU_DIST_LIBRARY_FILE
                "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}superlu_dist${CMAKE_SHARED_LIBRARY_SUFFIX}")
        else()
            set(SUPERLU_DIST_LIBRARY_FILE
                "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}superlu_dist${CMAKE_STATIC_LIBRARY_SUFFIX}")
        endif()

        set(_hypredrv_dsuperlu_libraries ${SUPERLU_DIST_LIBRARY_FILE})
        if(TPL_ENABLE_INTERNAL_BLASLIB)
            if(NOT _hypredrv_superlu_dist_static_build)
                list(APPEND _hypredrv_dsuperlu_libraries
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}blas${CMAKE_SHARED_LIBRARY_SUFFIX}")
            else()
                list(APPEND _hypredrv_dsuperlu_libraries
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_STATIC_LIBRARY_PREFIX}blas${CMAKE_STATIC_LIBRARY_SUFFIX}")
            endif()
        endif()
        if(MPI_CXX_LIBRARIES)
            list(APPEND _hypredrv_dsuperlu_libraries ${MPI_CXX_LIBRARIES})
        endif()
        if(TARGET OpenMP::OpenMP_C)
            list(APPEND _hypredrv_dsuperlu_libraries OpenMP::OpenMP_C)
        endif()

        set(TPL_DSUPERLU_LIBRARIES ${_hypredrv_dsuperlu_libraries}
            CACHE STRING "SuperLU_DIST libraries for HYPRE" FORCE)
        set(TPL_DSUPERLU_INCLUDE_DIRS "${superlu_dist_SOURCE_DIR}/SRC/include"
            CACHE PATH "SuperLU_DIST include directories for HYPRE" FORCE)

        set(HYPRE_BUILD_DSUPERLU ON CACHE INTERNAL
            "SuperLU_DIST is being built as part of this project" FORCE)
        set(DSUPERLU_INCLUDE_DIR_FOR_HYPRE "${superlu_dist_SOURCE_DIR}/SRC/include"
            CACHE INTERNAL "SuperLU_DIST include directory for manual addition to HYPRE" FORCE)
        set(SUPERLU_DIST_FOUND TRUE)

        message(STATUS "  Setting TPL_DSUPERLU_INCLUDE_DIRS: ${TPL_DSUPERLU_INCLUDE_DIRS}")
        message(STATUS "  Setting TPL_DSUPERLU_LIBRARIES: ${TPL_DSUPERLU_LIBRARIES}")
        message(STATUS "SuperLU_DIST configured and ready (built via FetchContent)")
        unset(_hypredrv_dsuperlu_libraries)
        unset(_hypredrv_superlu_dist_static_build)
        unset(_hypredrv_superlu_dist_target_type)
    endif()
endif()

############################################################
