# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Find and configure HYPRE
############################################################

# Option to specify HYPRE version/branch/tag for FetchContent
set(HYPRE_VERSION "v3.2.0" CACHE STRING "HYPRE version/branch/tag to fetch (e.g., master, v3.2.0)")

if(HYPRE_ENABLE_MIXEDINT AND HYPRE_ENABLE_BIGINT)
    message(FATAL_ERROR "HYPRE_ENABLE_MIXEDINT and HYPRE_ENABLE_BIGINT are mutually exclusive")
endif()

# Allow users to point to an autotools build by providing include/lib paths.
# This bypasses find_package(CONFIG) which only works with CMake installs.
if(NOT TARGET HYPRE::HYPRE AND DEFINED HYPRE_INCLUDE_DIRS AND
   (DEFINED HYPRE_LIBRARIES OR DEFINED HYPRE_LIBRARY))
    if(DEFINED HYPRE_LIBRARIES)
        set(_hypre_libs "${HYPRE_LIBRARIES}")
    else()
        set(_hypre_libs "${HYPRE_LIBRARY}")
    endif()

    message(STATUS "Using user-provided HYPRE include/lib paths (autotools or custom install)")
    add_library(HYPRE::HYPRE INTERFACE IMPORTED)
    set_target_properties(HYPRE::HYPRE PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${_hypre_libs}"
    )
    set(HYPREDRV_HYPRE_USER_PROVIDED TRUE)
    set(HYPRE_FOUND TRUE)
endif()

# First, try to find HYPRE via find_package
if(NOT HYPRE_FOUND)
    find_package(HYPRE CONFIG QUIET)
endif()

if(NOT HYPRE_FOUND)
    if(HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        math(EXPR _requested_release
            "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
        if(_requested_release LESS 22000)
            message(FATAL_ERROR "hypredrive requires hypre >= 2.20.0")
        endif()
    endif()
    # HYPRE not found via find_package - automatically fetch and build
    message(STATUS "HYPRE not found. Fetching and building HYPRE from source (version: ${HYPRE_VERSION})...")

    include(FetchContent)

    # Check for MPI (required by HYPRE)
    _hypredrv_seed_msmpi_library()
    find_package(MPI REQUIRED COMPONENTS C)

    # Decide whether to use autotools (hypre < 3.0.0) or CMake build
    set(_hypre_use_autotools FALSE)
    if(HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        set(_hypre_major "${CMAKE_MATCH_1}")
        if(_hypre_major LESS 3)
            set(_hypre_use_autotools TRUE)
        endif()
    endif()

    # Auto-fetched GPU builds use Umpire by default unless the user points to
    # an external Umpire or explicitly disables Umpire. Preserve explicit
    # HYPRE_BUILD_UMPIRE settings.
    if(NOT _hypre_use_autotools AND (HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP) AND
       NOT umpire_ROOT AND NOT umpire_DIR AND
       NOT (DEFINED CACHE{HYPRE_ENABLE_UMPIRE} AND NOT HYPRE_ENABLE_UMPIRE))
        _hypredrv_set_cache_bool_default(HYPRE_BUILD_UMPIRE ON
            "Automatically download and build Umpire for HYPRE")
    endif()

    set(_hypre_git_shallow TRUE)
    if(NOT HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        # HYPRE's CMake computes HYPRE_DEVELOP_NUMBER from git tag distance. A shallow
        # clone on moving refs (master/main/develop or arbitrary branches/commits) may
        # not have enough history/tags for `git describe`, which drops DEVELOP_NUMBER
        # from generated HYPRE_config.h and breaks hypredrive feature/version probes.
        set(_hypre_git_shallow FALSE)
    endif()

    FetchContent_Declare(
        hypre
        GIT_REPOSITORY https://github.com/hypre-space/hypre.git
        GIT_TAG        ${HYPRE_VERSION}
        GIT_SHALLOW    ${_hypre_git_shallow}
        GIT_PROGRESS   TRUE
        # SOURCE_SUBDIR removed - we'll add it manually after patching
    )

    # Enable verbose output for FetchContent to show progress
    set(FETCHCONTENT_QUIET OFF)

    if(_hypre_use_autotools)
        message(STATUS "HYPRE version < 3.0.0 detected - building with autotools.")

        if(HYPRE_VERSION MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
            math(EXPR _hypre_release_number
                 "${CMAKE_MATCH_1}*10000 + ${CMAKE_MATCH_2}*100 + ${CMAKE_MATCH_3}")
            if(NOT HYPREDRV_HYPRE_RELEASE_NUMBER)
                set(HYPREDRV_HYPRE_RELEASE_NUMBER "${_hypre_release_number}")
                add_compile_definitions(HYPREDRV_HYPRE_RELEASE_NUMBER=${HYPREDRV_HYPRE_RELEASE_NUMBER})
            endif()
            if(NOT HYPREDRV_HYPRE_DEVELOP_NUMBER)
                set(HYPREDRV_HYPRE_DEVELOP_NUMBER 0)
                add_compile_definitions(HYPREDRV_HYPRE_DEVELOP_NUMBER=0)
            endif()
            if(NOT HYPREDRV_HYPRE_RELEASE_VERSION)
                set(HYPREDRV_HYPRE_RELEASE_VERSION
                    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
                add_compile_definitions(
                    HYPREDRV_HYPRE_RELEASE_VERSION=\"${HYPREDRV_HYPRE_RELEASE_VERSION}\")
            endif()
        endif()

        FetchContent_GetProperties(hypre)
        if(NOT hypre_POPULATED)
            FetchContent_Populate(hypre)
        endif()
        _hypredrv_patch_hypre_mgr_col_lumped_bcf_leak("${hypre_SOURCE_DIR}")

        include(ExternalProject)

        string(REPLACE "/" "_" _hypre_version_tag "${HYPRE_VERSION}")
        set(_hypre_autotools_prefix "${CMAKE_BINARY_DIR}/hypre-autotools-${_hypre_version_tag}")
        set(_hypre_autotools_src "${hypre_SOURCE_DIR}/src")
        # Match typical CMake build-type flags for autotools builds.
        set(_hypre_autotools_cflags "${CMAKE_C_FLAGS}")
        set(_hypre_autotools_ldflags "${CMAKE_EXE_LINKER_FLAGS}")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O0 -g")
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O2 -g")
        else()
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -O3 -DNDEBUG")
        endif()
        if((HYPREDRV_ENABLE_PYTHON OR HYPREDRV_ENABLE_MATLAB OR HYPREDRV_ENABLE_FORTRAN OR HYPREDRV_ENABLE_JULIA) AND
           NOT _hypre_autotools_cflags MATCHES "(^| )-fPIC($| )")
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} -fPIC")
            message(STATUS "Language interface enabled: building auto-fetched HYPRE autotools static library with -fPIC")
        endif()
        if(HYPREDRV_INSTRUMENTATION_COMPILE_FLAGS)
            string(JOIN " " _hypre_instrumentation_cflags ${HYPREDRV_INSTRUMENTATION_COMPILE_FLAGS})
            set(_hypre_autotools_cflags "${_hypre_autotools_cflags} ${_hypre_instrumentation_cflags}")
        endif()
        if(HYPREDRV_INSTRUMENTATION_LINK_FLAGS)
            string(JOIN " " _hypre_instrumentation_ldflags ${HYPREDRV_INSTRUMENTATION_LINK_FLAGS})
            set(_hypre_autotools_ldflags "${_hypre_autotools_ldflags} ${_hypre_instrumentation_ldflags}")
        endif()

        set(_hypre_autotools_configure_extra "")
        if(HYPRE_ENABLE_BIGINT)
            list(APPEND _hypre_autotools_configure_extra --enable-bigint)
        elseif(HYPRE_ENABLE_MIXEDINT)
            if(DEFINED _hypre_release_number AND _hypre_release_number GREATER_EQUAL 22000)
                list(APPEND _hypre_autotools_configure_extra --enable-mixedint)
            else()
                list(APPEND _hypre_autotools_configure_extra --disable-mixedint)
                message(WARNING "HYPRE_ENABLE_MIXEDINT=ON but ${HYPRE_VERSION} does not support mixedint; forcing --disable-mixedint.")
            endif()
        endif()
        if(HYPREDRV_ENABLE_CALIPER)
            if(TPL_CALIPER_INCLUDE_DIRS AND TPL_CALIPER_LIBRARIES)
                list(APPEND _hypre_autotools_configure_extra
                    --with-caliper
                    --with-caliper-include=${TPL_CALIPER_INCLUDE_DIRS}
                    --with-caliper-lib=${TPL_CALIPER_LIBRARIES}
                )
            elseif(CALIPER_INCLUDE_DIR AND CALIPER_LIBRARY)
                list(APPEND _hypre_autotools_configure_extra
                    --with-caliper
                    --with-caliper-include=${CALIPER_INCLUDE_DIR}
                    --with-caliper-lib=${CALIPER_LIBRARY}
                )
            else()
                message(WARNING "HYPREDRV_ENABLE_CALIPER=ON but Caliper include/lib not found for hypre autotools build.")
            endif()
        endif()

        if(HYPRE_ENABLE_CUDA OR CMAKE_CUDA_ARCHITECTURES OR CUDA_ROOT OR CUDA_TOOLKIT_ROOT_DIR OR CUDAToolkit_ROOT)
            list(APPEND _hypre_autotools_configure_extra --with-cuda)
            if(CUDA_ROOT)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDA_ROOT})
            elseif(CUDA_TOOLKIT_ROOT_DIR)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDA_TOOLKIT_ROOT_DIR})
            elseif(CUDAToolkit_ROOT)
                list(APPEND _hypre_autotools_configure_extra --with-cuda-home=${CUDAToolkit_ROOT})
            endif()
            if(CMAKE_CUDA_ARCHITECTURES)
                string(REPLACE ";" "," _hypre_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
                list(APPEND _hypre_autotools_configure_extra --with-gpu-arch=${_hypre_cuda_arch})
            endif()
        endif()

        if(HYPRE_ENABLE_HIP OR CMAKE_HIP_ARCHITECTURES OR HIP_ROOT OR ROCM_ROOT OR ROCM_PATH)
            list(APPEND _hypre_autotools_configure_extra --with-hip)
            list(APPEND _hypre_autotools_configure_extra --with-MPI-include=${MPI_C_INCLUDE_DIRS})
            if(HIP_ROOT)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${HIP_ROOT})
            elseif(ROCM_ROOT)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${ROCM_ROOT})
            elseif(ROCM_PATH)
                list(APPEND _hypre_autotools_configure_extra ROCM_PATH=${ROCM_PATH})
            endif()
            if(CMAKE_HIP_ARCHITECTURES)
                string(REPLACE ";" "," _hypre_hip_arch "${CMAKE_HIP_ARCHITECTURES}")
                list(APPEND _hypre_autotools_configure_extra --with-gpu-arch=${_hypre_hip_arch})
            endif()
        endif()

        find_program(_hypre_autotools_make_program NAMES gmake make REQUIRED)

        ExternalProject_Add(hypre_autotools
            SOURCE_DIR ${_hypre_autotools_src}
            CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
                CC=${MPI_C_COMPILER}
                CFLAGS=${_hypre_autotools_cflags}
                LDFLAGS=${_hypre_autotools_ldflags}
                ${_hypre_autotools_src}/configure --prefix=${_hypre_autotools_prefix}
                ${_hypre_autotools_configure_extra}
            BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${_hypre_autotools_src} ${_hypre_autotools_make_program}
            BUILD_BYPRODUCTS "${_hypre_autotools_prefix}/lib/libHYPRE.a"
            INSTALL_COMMAND ${CMAKE_COMMAND} -E chdir ${_hypre_autotools_src} ${_hypre_autotools_make_program} install
            BUILD_IN_SOURCE 1
        )

        # Ensure the install prefix directories exist at configure time for imported targets.
        file(MAKE_DIRECTORY "${_hypre_autotools_prefix}/include")
        file(MAKE_DIRECTORY "${_hypre_autotools_prefix}/lib")

        set(HYPRE_INCLUDE_DIRS "${_hypre_autotools_prefix}/include")
        set(HYPRE_LIBRARIES "${_hypre_autotools_prefix}/lib/libHYPRE.a")

        add_library(HYPRE::HYPRE INTERFACE IMPORTED)
        set_target_properties(HYPRE::HYPRE PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${HYPRE_LIBRARIES}"
        )
        _hypredrv_link_mpi_interface(HYPRE::HYPRE)
        add_dependencies(HYPRE::HYPRE hypre_autotools)
        set(HYPREDRV_HYPRE_BUILD_DEPENDENCY hypre_autotools)

        set(HYPREDRV_HYPRE_USER_PROVIDED TRUE)
        set(HYPREDRV_HYPRE_AUTOTOOLS TRUE)
        set(HYPRE_FOUND TRUE)

        unset(_hypre_version_tag)
        unset(_hypre_autotools_prefix)
        unset(_hypre_autotools_src)
        unset(_hypre_autotools_cflags)
        unset(_hypre_autotools_ldflags)
        unset(_hypre_autotools_configure_extra)
        unset(_hypre_autotools_make_program)
        unset(_hypre_instrumentation_cflags)
        unset(_hypre_instrumentation_ldflags)
        unset(_hypre_cuda_arch)
        unset(_hypre_hip_arch)
    else()
    # Generic CMake argument inheritance
    # Forward all relevant cache variables to HYPRE build, including TPLs (MAGMA, CUDA, etc.)
    message(STATUS "Inheriting CMake arguments to HYPRE build...")

    # Forward relevant cache variables for hypre sub-builds.
    _hypredrv_forward_hypre_cache_vars(_forwarded_count)

    # Ensure MPI compiler is used if CMAKE_C_COMPILER not explicitly set in cache
    get_property(_c_compiler_set CACHE CMAKE_C_COMPILER PROPERTY VALUE SET)
    if(NOT _c_compiler_set AND MPI_C_COMPILER)
        set(CMAKE_C_COMPILER ${MPI_C_COMPILER} CACHE FILEPATH "C compiler" FORCE)
    endif()

    # HYPRE's test drivers are not needed by default, but preserve an explicit
    # HYPRE_BUILD_TESTS setting so consumers can add native drivers such as
    # struct to the same FetchContent build tree.
    if(NOT DEFINED HYPRE_BUILD_TESTS)
        set(HYPRE_BUILD_TESTS OFF CACHE BOOL "Build HYPRE tests")
    else()
        message(STATUS
            "Preserving caller-provided HYPRE_BUILD_TESTS=${HYPRE_BUILD_TESTS}")
    endif()
    set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "Build HYPRE examples" FORCE)
    if(HYPREDRV_ENABLE_PYTHON OR HYPREDRV_ENABLE_MATLAB OR HYPREDRV_ENABLE_FORTRAN OR HYPREDRV_ENABLE_JULIA)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL
            "Build position independent code for shared-library consumers" FORCE)
        message(STATUS "Language interface enabled: building auto-fetched HYPRE CMake target with position-independent code")
    endif()

    # Configure HYPRE to output libraries and headers in the same directories as main project
    # This ensures all libraries are in the same lib/ folder and headers in the same include/ folder
    _hypredrv_set_common_output_directories()

    if(_forwarded_count GREATER 0)
        message(STATUS "Forwarded ${_forwarded_count} CMake variables to HYPRE build")
    endif()

    # Fetch and configure HYPRE
    # We need to patch HYPRE's CMakeLists.txt before configuration, so we use
    # FetchContent_MakeAvailable to populate, then patch, then manually add subdirectory
    message(STATUS "Fetching HYPRE from GitHub (branch/tag: ${HYPRE_VERSION})...")
    message(STATUS "  Repository: https://github.com/hypre-space/hypre.git")

    # Use FetchContent_MakeAvailable to populate the source
    # Since SOURCE_SUBDIR was removed from FetchContent_Declare, it won't automatically
    # add the subdirectory, allowing us to patch first
    FetchContent_MakeAvailable(hypre)

    message(STATUS "HYPRE source fetched successfully")
    message(STATUS "  Source directory: ${hypre_SOURCE_DIR}")
    _hypredrv_patch_hypre_before_configuration("${hypre_SOURCE_DIR}")

    # Add HYPRE subdirectory manually (pointing to src subdirectory)
    # This is done after MakeAvailable and patching to allow patching before configuration
    if(NOT TARGET HYPRE::HYPRE)
        message(STATUS "Configuring HYPRE build...")
        message(STATUS "  Libraries will be built to: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

        # HYPRE's automatic Umpire path adds Umpire, CAMP, and fmt as nested
        # subprojects. Keep every library from that dependency stack static
        # unless the top-level build explicitly requests shared libraries.
        #
        # Set both the directory-scope and cache values: BLT-based subprojects
        # consult BUILD_SHARED_LIBS at several nesting levels, and a cache-only
        # update can be shadowed by a normal variable in a parent scope.
        set(_hypredrv_force_static_auto_umpire OFF)
        if(HYPRE_BUILD_UMPIRE AND NOT BUILD_SHARED_LIBS)
            set(_hypredrv_force_static_auto_umpire ON)
            set(BUILD_SHARED_LIBS OFF)
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
            message(STATUS
                "  Automatic Umpire build: forcing Umpire, CAMP, and fmt libraries static")
        elseif(HYPRE_BUILD_UMPIRE)
            message(STATUS
                "  Automatic Umpire build: honoring explicit BUILD_SHARED_LIBS=ON")
        endif()

        set(_hypredrv_saved_install_prefix "${CMAKE_INSTALL_PREFIX}")
        add_subdirectory(${hypre_SOURCE_DIR}/src ${hypre_BINARY_DIR})
        if(WIN32 AND BUILD_SHARED_LIBS AND TARGET HYPRE)
            set_target_properties(HYPRE PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        elseif(APPLE AND BUILD_SHARED_LIBS AND TARGET HYPRE)
            set_target_properties(HYPRE PROPERTIES INSTALL_RPATH "@loader_path")
        endif()
        set(CMAKE_INSTALL_PREFIX "${_hypredrv_saved_install_prefix}" CACHE PATH
            "Install path prefix, prepended onto install directories." FORCE)
        unset(_hypredrv_saved_install_prefix)

        if(_hypredrv_force_static_auto_umpire)
            set(_hypredrv_umpire_library_types)
            foreach(_umpire_target IN ITEMS
                    umpire
                    umpire_resource
                    umpire_strategy
                    umpire_op
                    umpire_event
                    umpire_util
                    umpire_interface
                    camp
                    fmt)
                if(TARGET ${_umpire_target})
                    get_target_property(_umpire_target_type
                        ${_umpire_target} TYPE)
                    list(APPEND _hypredrv_umpire_library_types
                        "${_umpire_target}=${_umpire_target_type}")
                    if(_umpire_target_type STREQUAL "SHARED_LIBRARY" OR
                       _umpire_target_type STREQUAL "MODULE_LIBRARY")
                        message(FATAL_ERROR
                            "Automatic Umpire target ${_umpire_target} was "
                            "created as ${_umpire_target_type} even though "
                            "BUILD_SHARED_LIBS is not ON")
                    endif()
                endif()
            endforeach()
            if(_hypredrv_umpire_library_types)
                list(JOIN _hypredrv_umpire_library_types ", "
                    _hypredrv_umpire_library_types_text)
                message(STATUS
                    "  Automatic Umpire target types: "
                    "${_hypredrv_umpire_library_types_text}")
            endif()
            unset(_hypredrv_umpire_library_types)
            unset(_hypredrv_umpire_library_types_text)
            unset(_umpire_target_type)
        endif()
        unset(_hypredrv_force_static_auto_umpire)

        # Umpire's filesystem probe correctly falls back to POSIX when the
        # compiler lacks <filesystem>, but its bundled nlohmann JSON header
        # still enables std::filesystem whenever C++17 is selected. Tell that
        # header to use its C++14-compatible path in this configuration.
        if(TARGET umpire_tpl_json AND DEFINED UMPIRE_ENABLE_FILESYSTEM AND
           NOT UMPIRE_ENABLE_FILESYSTEM)
            target_compile_definitions(umpire_tpl_json INTERFACE JSON_HAS_CPP_14=1)
            # BLT copies dependency usage requirements when it creates object
            # libraries, so changing umpire_tpl_json afterward does not update
            # the already-created target that compiles Umpire's JSON users.
            if(TARGET umpire_event)
                target_compile_definitions(umpire_event PRIVATE JSON_HAS_CPP_14=1)
            endif()
            message(STATUS
                "  Umpire std::filesystem is unavailable; disabled JSON filesystem conversions")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
            foreach(_umpire_target IN ITEMS
                    umpire
                    umpire_resource
                    umpire_strategy
                    umpire_op
                    umpire_event
                    umpire_util
                    umpire_interface)
                if(TARGET ${_umpire_target})
                    target_compile_options(${_umpire_target} PRIVATE
                        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:--diag_suppress code_is_unreachable>")
                endif()
            endforeach()
            unset(_umpire_target)
            message(STATUS
                "  Suppressed NVHPC code_is_unreachable diagnostics for Umpire")
        endif()
    endif()

    # Remove Caliper include directory from HYPRE's INTERFACE_INCLUDE_DIRECTORIES and re-add with BUILD_INTERFACE
    # This avoids CMake errors about _deps paths in INTERFACE_INCLUDE_DIRECTORIES
    if(HYPRE_BUILD_CALIPER AND TARGET HYPRE AND CALIPER_INCLUDE_DIR_FOR_HYPRE)
        # Get current include directories
        get_target_property(HYPRE_INCLUDE_DIRS HYPRE INTERFACE_INCLUDE_DIRECTORIES)
        if(HYPRE_INCLUDE_DIRS)
            # Remove the _deps path if present
            list(REMOVE_ITEM HYPRE_INCLUDE_DIRS "${CALIPER_INCLUDE_DIR_FOR_HYPRE}")
            # Set the cleaned list
            set_target_properties(HYPRE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
            # Re-add with BUILD_INTERFACE
            target_include_directories(HYPRE PUBLIC $<BUILD_INTERFACE:${CALIPER_INCLUDE_DIR_FOR_HYPRE}>)
            message(STATUS "  Fixed Caliper include directory in HYPRE target using BUILD_INTERFACE")
        endif()

        # Ensure HYPRE depends on Caliper target so it builds after Caliper
        # This is critical because HYPRE links against the Caliper library file
        if(TARGET caliper)
            add_dependencies(HYPRE caliper)
            message(STATUS "  Added dependency: HYPRE -> caliper")
        endif()
    endif()

    if(HYPRE_BUILD_DSUPERLU AND TARGET HYPRE)
        if(DSUPERLU_INCLUDE_DIR_FOR_HYPRE)
            get_target_property(HYPRE_INCLUDE_DIRS HYPRE INTERFACE_INCLUDE_DIRECTORIES)
            if(HYPRE_INCLUDE_DIRS)
                list(REMOVE_ITEM HYPRE_INCLUDE_DIRS "${DSUPERLU_INCLUDE_DIR_FOR_HYPRE}")
                set_target_properties(HYPRE PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
                target_include_directories(HYPRE PUBLIC
                    $<BUILD_INTERFACE:${DSUPERLU_INCLUDE_DIR_FOR_HYPRE}>)
                message(STATUS "  Fixed SuperLU_DIST include directory in HYPRE target using BUILD_INTERFACE")
            endif()
        endif()

        if(TARGET superlu_dist)
            add_dependencies(HYPRE superlu_dist)
            message(STATUS "  Added dependency: HYPRE -> superlu_dist")
        endif()

        if(TPL_ENABLE_INTERNAL_BLASLIB AND TARGET blas)
            add_dependencies(HYPRE blas)
            message(STATUS "  Added dependency: HYPRE -> blas")
        endif()
    endif()

    # HYPRE records dependency roots/libdirs in HYPRE_DEPENDENCY_DIRS, but when
    # a TPL comes from an imported CMake package (for example Caliper via
    # find_package(CONFIG)), upstream HYPRE only exports the root hint and does
    # not automatically append the corresponding libdir to libHYPRE's install
    # RPATH. Recover those libdirs here so installed shared libraries continue
    # to resolve external TPLs from their original locations.
    if(BUILD_SHARED_LIBS AND TARGET HYPRE AND DEFINED HYPRE_DEPENDENCY_DIRS AND HYPRE_DEPENDENCY_DIRS)
        _hypredrv_append_runtime_dirs_from_dependency_roots(HYPRE ${HYPRE_DEPENDENCY_DIRS})
    endif()

    message(STATUS "HYPRE configured and ready")
    endif()
endif()

# Get HYPRE properties
if(TARGET HYPRE::HYPRE)
    _hypredrv_detect_hypre_sycl_usage()
    _hypredrv_detect_hypre_openmp_usage()
    _hypredrv_ensure_hypre_cuda_nvjitlink()
    get_target_property(HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)

    # HYPRE headers are external to this project. Preserve strict diagnostics
    # for our sources without promoting warnings in dependency inlines to errors.
    get_target_property(_hypredrv_hypre_target HYPRE::HYPRE ALIASED_TARGET)
    if(NOT _hypredrv_hypre_target)
        set(_hypredrv_hypre_target HYPRE::HYPRE)
    endif()
    if(HYPRE_INCLUDE_DIRS)
        set_property(TARGET ${_hypredrv_hypre_target} APPEND PROPERTY
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}")
    endif()
    unset(_hypredrv_hypre_target)

    # Try to get library location - handle both Release and Debug configurations.
    # For autotools/custom installs we only have INTERFACE_LINK_LIBRARIES, so
    # fall back to the user-provided variables for a clearer message.
    get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION)
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_RELWITHDEBINFO)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        get_target_property(HYPRE_LIBRARY_FILE HYPRE::HYPRE IMPORTED_LOCATION_DEBUG)
    endif()
    if(NOT HYPRE_LIBRARY_FILE)
        if(DEFINED HYPRE_LIBRARIES)
            set(HYPRE_LIBRARY_FILE "${HYPRE_LIBRARIES}")
        elseif(DEFINED HYPRE_LIBRARY)
            set(HYPRE_LIBRARY_FILE "${HYPRE_LIBRARY}")
        else()
            set(HYPRE_LIBRARY_FILE "not found (using target)")
        endif()
    endif()

    # Check if HYPRE was built with Caliper support (for find_package case)
    # This is needed because HYPRE_ENABLE_CALIPER might not be set when HYPRE is found via find_package
    # Only check if HYPREDRV_ENABLE_CALIPER is not already explicitly set
    if(NOT DEFINED HYPREDRV_ENABLE_CALIPER)
        # Check if HYPRE links to Caliper
        get_target_property(HYPRE_LINK_LIBS HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
        set(HYPRE_HAS_CALIPER FALSE)

        if(HYPRE_LINK_LIBS)
            # Check if caliper is in the link libraries (could be caliper::caliper, caliper, or libcaliper)
            foreach(lib ${HYPRE_LINK_LIBS})
                if(lib MATCHES "caliper" OR lib STREQUAL "caliper::caliper" OR lib STREQUAL "caliper")
                    set(HYPRE_HAS_CALIPER TRUE)
                    break()
                endif()
            endforeach()
        endif()

        # Also check compile definitions for Caliper-related defines
        if(NOT HYPRE_HAS_CALIPER)
            get_target_property(HYPRE_COMPILE_DEFS HYPRE::HYPRE INTERFACE_COMPILE_DEFINITIONS)
            if(HYPRE_COMPILE_DEFS)
                foreach(def ${HYPRE_COMPILE_DEFS})
                    if(def MATCHES "HYPRE_USING_CALIPER" OR def MATCHES "CALIPER")
                        set(HYPRE_HAS_CALIPER TRUE)
                        break()
                    endif()
                endforeach()
            endif()
        endif()

        # If we detected Caliper support, enable HYPREDRV_ENABLE_CALIPER
        if(HYPRE_HAS_CALIPER)
            set(HYPREDRV_ENABLE_CALIPER ON CACHE BOOL "Enable Caliper instrumentation support" FORCE)
            message(STATUS "  Detected HYPRE was built with Caliper support - enabling HYPREDRV_ENABLE_CALIPER")
        endif()
    endif()

    message(STATUS "Found HYPRE:")
    message(STATUS "  include directories: ${HYPRE_INCLUDE_DIRS}")
    message(STATUS "  libraries: ${HYPRE_LIBRARY_FILE}")
    message(STATUS "  uses SYCL: ${HYPREDRV_HYPRE_USES_SYCL}")
    message(STATUS "  uses OpenMP: ${HYPREDRV_HYPRE_USE_OPENMP}")

    # Build a runtime library search path for CTest so executables can launch
    # when HYPRE's transitive shared-library dependencies (e.g., Caliper) are
    # not resolved through executable RUNPATH.
    # Prefer libraries produced by this build over same-named libraries in a
    # dependency/install prefix. This matters for coverage builds in particular:
    # loading a stale installed libHYPREDRV writes incompatible counters into
    # the active build tree and causes gcovr to discard entire source files.
    set(_hypredrv_test_runtime_lib_dirs "${CMAKE_BINARY_DIR}/lib")

    function(_hypredrv_collect_lib_dirs_from_items out_var)
        set(_dirs "")
        foreach(_item IN LISTS ARGN)
            if(NOT _item)
                continue()
            endif()

            if(TARGET "${_item}")
                get_target_property(_item_lib "${_item}" IMPORTED_LOCATION)
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_RELEASE)
                endif()
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_RELWITHDEBINFO)
                endif()
                if(NOT _item_lib)
                    get_target_property(_item_lib "${_item}" IMPORTED_LOCATION_DEBUG)
                endif()
                if(_item_lib AND EXISTS "${_item_lib}")
                    get_filename_component(_item_dir "${_item_lib}" DIRECTORY)
                    list(APPEND _dirs "${_item_dir}")
                endif()
                continue()
            endif()

            # Ignore generator expressions and plain library names.
            if(_item MATCHES "^\\$<" OR _item MATCHES "^[A-Za-z0-9_:+.-]+$")
                continue()
            endif()

            # Extract -L<dir> entries from interface link options/libraries.
            if(_item MATCHES "^-L(.+)$")
                set(_candidate_dir "${CMAKE_MATCH_1}")
                if(IS_DIRECTORY "${_candidate_dir}")
                    list(APPEND _dirs "${_candidate_dir}")
                endif()
                continue()
            endif()

            if(IS_ABSOLUTE "${_item}" AND EXISTS "${_item}")
                get_filename_component(_item_dir "${_item}" DIRECTORY)
                list(APPEND _dirs "${_item_dir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _dirs)
        set(${out_var} "${_dirs}" PARENT_SCOPE)
    endfunction()

    _hypredrv_collect_lib_dirs_from_items(_hypredrv_hypre_lib_dirs ${HYPRE_LIBRARY_FILE})
    list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_hypre_lib_dirs})
    if(IS_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/hypre-build/lib")
        list(APPEND _hypredrv_test_runtime_lib_dirs
            "${CMAKE_BINARY_DIR}/_deps/hypre-build/lib")
    endif()
    get_target_property(_hypredrv_hypre_link_libs HYPRE::HYPRE INTERFACE_LINK_LIBRARIES)
    if(_hypredrv_hypre_link_libs)
        _hypredrv_collect_lib_dirs_from_items(_hypredrv_hypre_link_dirs ${_hypredrv_hypre_link_libs})
        list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_hypre_link_dirs})
    endif()

    # HYPREConfig.cmake exports dependency roots (and sometimes lib dirs). When
    # pkg-config synthesized interface targets hide full library paths (e.g.,
    # "caliper"), use these hints to recover <prefix>/lib.
    if(DEFINED HYPRE_DEPENDENCY_DIRS AND HYPRE_DEPENDENCY_DIRS)
        foreach(_dep_dir IN LISTS HYPRE_DEPENDENCY_DIRS)
            if(NOT IS_DIRECTORY "${_dep_dir}")
                continue()
            endif()
            if(_dep_dir MATCHES "(/|^)lib64?$")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}")
            endif()
            if(IS_DIRECTORY "${_dep_dir}/lib")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}/lib")
            endif()
            if(IS_DIRECTORY "${_dep_dir}/lib64")
                list(APPEND _hypredrv_test_runtime_lib_dirs "${_dep_dir}/lib64")
            endif()
        endforeach()
    endif()

    if(DEFINED MPI_C_LIBRARIES AND MPI_C_LIBRARIES)
        _hypredrv_collect_lib_dirs_from_items(_hypredrv_mpi_link_dirs ${MPI_C_LIBRARIES})
        list(APPEND _hypredrv_test_runtime_lib_dirs ${_hypredrv_mpi_link_dirs})
    endif()

    list(REMOVE_DUPLICATES _hypredrv_test_runtime_lib_dirs)

    if(UNIX AND NOT APPLE AND _hypredrv_test_runtime_lib_dirs)
        string(JOIN ":" _hypredrv_ld_path ${_hypredrv_test_runtime_lib_dirs})
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "LD_LIBRARY_PATH=${_hypredrv_ld_path}:$ENV{LD_LIBRARY_PATH}"
            CACHE INTERNAL "CTest runtime library path environment assignment")
        message(STATUS "  test runtime env: ${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    elseif(APPLE AND _hypredrv_test_runtime_lib_dirs)
        string(JOIN ":" _hypredrv_dyld_path ${_hypredrv_test_runtime_lib_dirs})
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT
            "DYLD_LIBRARY_PATH=${_hypredrv_dyld_path}:$ENV{DYLD_LIBRARY_PATH}"
            CACHE INTERNAL "CTest runtime library path environment assignment")
        message(STATUS "  test runtime env: ${HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT}")
    else()
        set(HYPREDRV_TEST_RUNTIME_ENV_ASSIGNMENT "" CACHE INTERNAL
            "CTest runtime library path environment assignment")
    endif()

    # For autotools/custom HYPRE, read HYPRE_config.h to derive version info and MPI mode.
    if(HYPREDRV_HYPRE_USER_PROVIDED AND HYPRE_INCLUDE_DIRS AND NOT HYPREDRV_HYPRE_AUTOTOOLS)
        set(_hypre_config_found FALSE)
        foreach(_inc_dir IN LISTS HYPRE_INCLUDE_DIRS)
            if(EXISTS "${_inc_dir}/HYPRE_config.h")
                set(_hypre_config_found TRUE)
                file(READ "${_inc_dir}/HYPRE_config.h" _hypre_cfg)

                # Extract release number if available (2.21+), otherwise compute from release version.
                set(_hypre_rel_num "")
                set(_hypre_rel_ver "")
                set(_hypre_dev_num "")

                string(REGEX MATCH "#define[ \t]+HYPRE_RELEASE_NUMBER[ \t]+([0-9]+)" _hypre_rel_num_match "${_hypre_cfg}")
                if(_hypre_rel_num_match)
                    set(_hypre_rel_num "${CMAKE_MATCH_1}")
                endif()

                string(REGEX REPLACE ".*#define[ \t]+HYPRE_RELEASE_VERSION[ \t]+\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\1;\\2;\\3" _hypre_rel_parts "${_hypre_cfg}")
                if(_hypre_rel_parts MATCHES ";")
                    list(GET _hypre_rel_parts 0 _hypre_rel_major)
                    list(GET _hypre_rel_parts 1 _hypre_rel_minor)
                    list(GET _hypre_rel_parts 2 _hypre_rel_patch)
                    set(_hypre_rel_ver "${_hypre_rel_major}.${_hypre_rel_minor}.${_hypre_rel_patch}")
                    if(NOT _hypre_rel_num)
                        math(EXPR _hypre_rel_num
                             "${_hypre_rel_major}*10000 + ${_hypre_rel_minor}*100 + ${_hypre_rel_patch}")
                    endif()
                endif()

                string(REGEX MATCH "#define[ \t]+HYPRE_DEVELOP_NUMBER[ \t]+([0-9]+)" _hypre_dev_num_match "${_hypre_cfg}")
                if(_hypre_dev_num_match)
                    set(_hypre_dev_num "${CMAKE_MATCH_1}")
                endif()

                if(_hypre_rel_num)
                    set(HYPREDRV_HYPRE_RELEASE_NUMBER "${_hypre_rel_num}")
                endif()
                if(_hypre_rel_ver)
                    set(HYPREDRV_HYPRE_RELEASE_VERSION "${_hypre_rel_ver}")
                endif()
                if(_hypre_dev_num)
                    set(HYPREDRV_HYPRE_DEVELOP_NUMBER "${_hypre_dev_num}")
                endif()

                if(HYPREDRV_HYPRE_RELEASE_NUMBER)
                    add_compile_definitions(HYPREDRV_HYPRE_RELEASE_NUMBER=${HYPREDRV_HYPRE_RELEASE_NUMBER})
                endif()
                if(HYPREDRV_HYPRE_DEVELOP_NUMBER)
                    add_compile_definitions(HYPREDRV_HYPRE_DEVELOP_NUMBER=${HYPREDRV_HYPRE_DEVELOP_NUMBER})
                endif()
                if(HYPREDRV_HYPRE_RELEASE_VERSION)
                    add_compile_definitions(HYPREDRV_HYPRE_RELEASE_VERSION=\"${HYPREDRV_HYPRE_RELEASE_VERSION}\")
                endif()

                # If HYPRE was built with MPI (HYPRE_SEQUENTIAL not defined), make sure MPI is available.
                string(REGEX MATCH "#define[ \t]+HYPRE_SEQUENTIAL" _hypre_seq "${_hypre_cfg}")
                if(NOT _hypre_seq)
                    message(STATUS "  HYPRE built with MPI support; checking MPI...")
                    find_package(MPI REQUIRED COMPONENTS C)
                    _hypredrv_link_mpi_interface(HYPRE::HYPRE)
                else()
                    message(STATUS "  HYPRE built in sequential mode; MPI not required")
                endif()
                break()
            endif()
        endforeach()
        if(NOT _hypre_config_found)
            message(WARNING "HYPRE_config.h not found in HYPRE_INCLUDE_DIRS; cannot determine MPI requirement or version")
        endif()
    endif()
else()
    message(FATAL_ERROR "HYPRE::HYPRE target not available")
endif()

############################################################
