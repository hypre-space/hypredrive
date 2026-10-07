# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

############################################################
# Compatibility patches applied to fetched HYPRE sources
############################################################

function(_hypredrv_patch_hypre_mgr_col_lumped_bcf_leak hypre_source_dir)
    set(_hypredrv_hypre_mgr_interp_file
        "${hypre_source_dir}/src/parcsr_ls/par_mgr_interp.c")
    if(NOT EXISTS "${_hypredrv_hypre_mgr_interp_file}")
        return()
    endif()

    # HYPRE fixed this leak upstream in the 3.2.0 development stream.  Keep
    # the compatibility patch for the 3.2.0 release (develop number 0) and
    # older releases, but do not add a second destroy for newer snapshots.
    set(_hypredrv_hypre_release_number 0)
    set(_hypredrv_hypre_develop_number 0)
    set(_hypredrv_hypre_cmake_file "${hypre_source_dir}/src/CMakeLists.txt")
    if(EXISTS "${_hypredrv_hypre_cmake_file}")
        file(READ "${_hypredrv_hypre_cmake_file}" _hypredrv_hypre_cmake_content)
        string(REGEX MATCH
               "set\\(HYPRE_NUMBER[ \\t]+([0-9]+)\\)"
               _hypredrv_hypre_number_match
               "${_hypredrv_hypre_cmake_content}")
        if(_hypredrv_hypre_number_match)
            set(_hypredrv_hypre_release_number "${CMAKE_MATCH_1}")
        endif()
    endif()

    if(_hypredrv_hypre_release_number GREATER 30200)
        message(STATUS
            "  HYPRE MGR block-column-sum leak fix already provided by HYPRE; "
            "compatibility patch not needed")
        return()
    elseif(_hypredrv_hypre_release_number EQUAL 30200)
        execute_process(
            COMMAND git -C "${hypre_source_dir}" describe --match "v*"
                    --abbrev=0 --always
            OUTPUT_VARIABLE _hypredrv_hypre_last_tag
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _hypredrv_hypre_describe_result)
        if(_hypredrv_hypre_describe_result EQUAL 0 AND
           _hypredrv_hypre_last_tag)
            execute_process(
                COMMAND git -C "${hypre_source_dir}" rev-list --count
                        "${_hypredrv_hypre_last_tag}..HEAD"
                OUTPUT_VARIABLE _hypredrv_hypre_develop_number
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _hypredrv_hypre_revlist_result)
            if(_hypredrv_hypre_revlist_result EQUAL 0 AND
               _hypredrv_hypre_develop_number GREATER_EQUAL 1)
                message(STATUS
                    "  HYPRE MGR block-column-sum leak fix already provided by "
                    "HYPRE 3.2.0 development snapshot; compatibility patch not needed")
                return()
            endif()
        endif()
    endif()

    file(READ "${_hypredrv_hypre_mgr_interp_file}"
         _hypredrv_hypre_mgr_interp_content)
    if(_hypredrv_hypre_mgr_interp_content MATCHES
       "HYPREDRV: destroy temporary MGR block column sum")
        return()
    endif()
    if(NOT _hypredrv_hypre_mgr_interp_content MATCHES
       "hypre_DenseBlockMatrixNumNonzeros\\(B_CF\\)")
        return()
    endif()

    set(_hypredrv_hypre_mgr_interp_old [=[
      hypre_ParVectorStridedCopy(b_CF,
                                 block_dim, block_dim,
                                 hypre_DenseBlockMatrixNumNonzeros(B_CF),
                                 hypre_DenseBlockMatrixData(B_CF));
]=])
    set(_hypredrv_hypre_mgr_interp_new [=[
      hypre_ParVectorStridedCopy(b_CF,
                                 block_dim, block_dim,
                                 hypre_DenseBlockMatrixNumNonzeros(B_CF),
                                 hypre_DenseBlockMatrixData(B_CF));
      /* HYPREDRV: destroy temporary MGR block column sum. */
      hypre_DenseBlockMatrixDestroy(B_CF);
]=])
    set(_hypredrv_hypre_mgr_interp_original
        "${_hypredrv_hypre_mgr_interp_content}")
    string(REPLACE "${_hypredrv_hypre_mgr_interp_old}"
                   "${_hypredrv_hypre_mgr_interp_new}"
                   _hypredrv_hypre_mgr_interp_content
                   "${_hypredrv_hypre_mgr_interp_content}")

    if("${_hypredrv_hypre_mgr_interp_content}" STREQUAL
       "${_hypredrv_hypre_mgr_interp_original}")
        message(WARNING
            "Could not patch HYPRE MGR temporary block-column-sum leak; "
            "upstream HYPRE may have changed src/parcsr_ls/par_mgr_interp.c")
    else()
        file(WRITE "${_hypredrv_hypre_mgr_interp_file}"
             "${_hypredrv_hypre_mgr_interp_content}")
        message(STATUS
            "  HYPRE MGR temporary block-column-sum leak patched")
    endif()

    unset(_hypredrv_hypre_mgr_interp_file)
    unset(_hypredrv_hypre_mgr_interp_content)
    unset(_hypredrv_hypre_mgr_interp_original)
    unset(_hypredrv_hypre_mgr_interp_old)
    unset(_hypredrv_hypre_mgr_interp_new)
endfunction()

function(_hypredrv_patch_hypre_umpire_header_linkage hypre_source_dir)
    set(_hypredrv_umpire_header_old [=[
#if defined(HYPRE_USING_UMPIRE)
#include "umpire/config.hpp"
#if UMPIRE_VERSION_MAJOR >= 2022
#include "umpire/interface/c_fortran/umpire.h"
#define hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_quick_pool
#else
#include "umpire/interface/umpire.h"
#define hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_pool
#endif /* UMPIRE_VERSION_MAJOR >= 2022 */
#define HYPRE_UMPIRE_POOL_NAME_MAX_LEN 1024
#endif /* defined(HYPRE_USING_UMPIRE) */
]=])
    set(_hypredrv_umpire_header_new [=[
#if defined(HYPRE_USING_UMPIRE)
#ifdef __cplusplus
/* HYPREDRV: Umpire C++ headers require C++ linkage. */
extern "C++" {
#endif
#include "umpire/config.hpp"
#if UMPIRE_VERSION_MAJOR >= 2022
#include "umpire/interface/c_fortran/umpire.h"
#define hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_quick_pool
#else
#include "umpire/interface/umpire.h"
#define hypre_umpire_resourcemanager_make_allocator_pool umpire_resourcemanager_make_allocator_pool
#endif /* UMPIRE_VERSION_MAJOR >= 2022 */
#ifdef __cplusplus
}
#endif
#define HYPRE_UMPIRE_POOL_NAME_MAX_LEN 1024
#endif /* defined(HYPRE_USING_UMPIRE) */
]=])

    set(_hypredrv_umpire_header_patched FALSE)
    foreach(_hypredrv_umpire_header_file IN ITEMS
            "${hypre_source_dir}/src/utilities/handle.h"
            "${hypre_source_dir}/src/utilities/_hypre_utilities.h")
        if(NOT EXISTS "${_hypredrv_umpire_header_file}")
            continue()
        endif()

        file(READ "${_hypredrv_umpire_header_file}"
             _hypredrv_umpire_header_content)
        if(_hypredrv_umpire_header_content MATCHES
           "HYPREDRV: Umpire C\\+\\+ headers require C\\+\\+ linkage")
            continue()
        endif()

        set(_hypredrv_umpire_header_original
            "${_hypredrv_umpire_header_content}")
        string(REPLACE
            "${_hypredrv_umpire_header_old}"
            "${_hypredrv_umpire_header_new}"
            _hypredrv_umpire_header_content
            "${_hypredrv_umpire_header_content}")
        if("${_hypredrv_umpire_header_content}" STREQUAL
           "${_hypredrv_umpire_header_original}")
            message(WARNING
                "Could not isolate Umpire's C++ headers from HYPRE's C linkage in "
                "${_hypredrv_umpire_header_file}; upstream HYPRE may have changed")
        else()
            file(WRITE "${_hypredrv_umpire_header_file}"
                 "${_hypredrv_umpire_header_content}")
            set(_hypredrv_umpire_header_patched TRUE)
        endif()
    endforeach()

    if(_hypredrv_umpire_header_patched)
        message(STATUS
            "  HYPRE Umpire C++ headers patched to use C++ linkage")
    endif()
endfunction()

function(_hypredrv_patch_hypre_sycl_device_code_split hypre_source_dir)
    if(NOT HYPRE_ENABLE_SYCL OR
       NOT HYPREDRV_SYCL_DEVICE_CODE_SPLIT STREQUAL "per_source")
        return()
    endif()

    set(_hypredrv_sycl_toolkit_file
        "${hypre_source_dir}/src/config/cmake/HYPRE_SetupSYCLToolkit.cmake")
    if(NOT EXISTS "${_hypredrv_sycl_toolkit_file}")
        return()
    endif()

    file(READ "${_hypredrv_sycl_toolkit_file}"
         _hypredrv_sycl_toolkit_content)
    if(_hypredrv_sycl_toolkit_content MATCHES
       "-fsycl-device-code-split=per_kernel")
        set(_hypredrv_sycl_toolkit_original
            "${_hypredrv_sycl_toolkit_content}")
        string(REPLACE
            "-fsycl-device-code-split=per_kernel"
            "-fsycl-device-code-split=${HYPREDRV_SYCL_DEVICE_CODE_SPLIT}"
            _hypredrv_sycl_toolkit_content
            "${_hypredrv_sycl_toolkit_content}")
        if("${_hypredrv_sycl_toolkit_content}" STREQUAL
           "${_hypredrv_sycl_toolkit_original}")
            message(WARNING
                "Could not patch HYPRE SYCL device-code split mode; upstream "
                "HYPRE may have changed HYPRE_SetupSYCLToolkit.cmake")
        else()
            file(WRITE "${_hypredrv_sycl_toolkit_file}"
                 "${_hypredrv_sycl_toolkit_content}")
            message(STATUS
                "  HYPRE SYCL device-code split set to "
                "${HYPREDRV_SYCL_DEVICE_CODE_SPLIT}")
        endif()
    endif()
endfunction()


function(_hypredrv_patch_hypre_before_configuration hypre_source_dir)
    _hypredrv_patch_hypre_mgr_col_lumped_bcf_leak("${hypre_source_dir}")
    if(HYPRE_BUILD_UMPIRE OR HYPRE_ENABLE_UMPIRE)
        _hypredrv_patch_hypre_umpire_header_linkage("${hypre_source_dir}")
    endif()
    _hypredrv_patch_hypre_sycl_device_code_split("${hypre_source_dir}")

    # Patch HYPRE's CMakeLists.txt to skip export when TPLs are auto-built
    # This must be done before add_subdirectory is called
    if((HYPRE_BUILD_CALIPER OR HYPRE_BUILD_DSUPERLU) AND EXISTS "${hypre_source_dir}/src/CMakeLists.txt")
        file(READ "${hypre_source_dir}/src/CMakeLists.txt" HYPRE_CMAKE_CONTENT)
        set(_hypre_export_guard_old
            "if(NOT (HYPRE_BUILD_UMPIRE AND TARGET umpire))")
        set(_hypre_export_guard_new
            "if(NOT (HYPRE_BUILD_UMPIRE AND TARGET umpire) AND NOT (HYPRE_BUILD_CALIPER AND TARGET caliper) AND NOT (HYPRE_BUILD_DSUPERLU AND TARGET superlu_dist))")
        if(HYPRE_CMAKE_CONTENT MATCHES
           "HYPRE_BUILD_DSUPERLU AND TARGET superlu_dist")
            message(STATUS
                "  HYPRE CMakeLists.txt already handles auto-built SuperLU_DIST exports")
        elseif(HYPRE_CMAKE_CONTENT MATCHES "Export from build tree")
            set(_hypre_cmake_original "${HYPRE_CMAKE_CONTENT}")
            string(REPLACE "${_hypre_export_guard_old}" "${_hypre_export_guard_new}"
                HYPRE_CMAKE_CONTENT "${HYPRE_CMAKE_CONTENT}")
            string(REPLACE
                "Skipping build-tree export of HYPRETargets due to auto-built Umpire dependency"
                "Skipping build-tree export of HYPRETargets due to auto-built Umpire, Caliper, or SuperLU_DIST dependency"
                HYPRE_CMAKE_CONTENT "${HYPRE_CMAKE_CONTENT}")
            if("${HYPRE_CMAKE_CONTENT}" STREQUAL "${_hypre_cmake_original}")
                message(WARNING
                    "Could not patch HYPRE build-tree export guard for auto-built "
                    "Caliper/SuperLU_DIST dependencies; upstream HYPRE may have "
                    "changed src/CMakeLists.txt")
            else()
                file(WRITE "${hypre_source_dir}/src/CMakeLists.txt" "${HYPRE_CMAKE_CONTENT}")
                message(STATUS
                    "  HYPRE CMakeLists.txt patched to skip export when auto-built TPLs are used")
            endif()
            unset(_hypre_cmake_original)
        else()
            message(WARNING
                "Could not find HYPRE build-tree export section to patch for auto-built "
                "Caliper/SuperLU_DIST dependencies")
        endif()
        unset(_hypre_export_guard_old)
        unset(_hypre_export_guard_new)
    endif()

    # Workaround: HYPRE's SYCL device enumeration constructs a single
    # sycl::platform from gpu_selector_v, which fails when the selected
    # platform exposes no GPUs. Rewrite both call sites to enumerate GPUs
    # across all platforms via sycl::device::get_devices(). Remove this once
    # the equivalent change lands upstream (see hypre-sycl.patch).
    if(HYPRE_ENABLE_SYCL AND EXISTS "${hypre_source_dir}/src/utilities/general.c")
        file(READ "${hypre_source_dir}/src/utilities/general.c" HYPRE_GENERAL_CONTENT)
        if(NOT HYPRE_GENERAL_CONTENT MATCHES "hypre_GetSYCLGpuDevices")
            set(_set_device_old "         sycl::platform platform(sycl::gpu_selector_v);\n         auto gpu_devices = platform.get_devices(sycl::info::device_type::gpu);\n")
            set(_set_device_new "         auto gpu_devices = hypre_GetSYCLGpuDevices();\n")
            set(_count_old "   (*device_count) = 0;\n   sycl::platform platform(sycl::gpu_selector_v);\n   auto const& gpu_devices = platform.get_devices(sycl::info::device_type::gpu);\n   HYPRE_Int i;\n   for (i = 0; i < gpu_devices.size(); i++)\n   {\n      (*device_count)++;\n   }\n")
            set(_count_new "   auto gpu_devices = hypre_GetSYCLGpuDevices();\n   (*device_count) = (hypre_int) gpu_devices.size();\n")

            if(NOT (HYPRE_GENERAL_CONTENT MATCHES "sycl::platform platform.sycl::gpu_selector_v"))
                message(WARNING
                    "  HYPRE general.c does not contain the expected 'sycl::platform(sycl::gpu_selector_v)' call sites. "
                    "The SYCL enumeration workaround will be skipped; upstream HYPRE may have been updated.")
            else()
                string(REPLACE
                    "#include \"_hypre_utilities.hpp\"\n"
                    "#include \"_hypre_utilities.hpp\"\n\n#if defined(HYPRE_USING_SYCL)\nstatic std::vector<sycl::device>\nhypre_GetSYCLGpuDevices(void)\n{\n   return sycl::device::get_devices(sycl::info::device_type::gpu);\n}\n#endif\n"
                    HYPRE_GENERAL_CONTENT "${HYPRE_GENERAL_CONTENT}")
                string(REPLACE "${_set_device_old}" "${_set_device_new}"
                    HYPRE_GENERAL_CONTENT "${HYPRE_GENERAL_CONTENT}")
                string(REPLACE "${_count_old}" "${_count_new}"
                    HYPRE_GENERAL_CONTENT "${HYPRE_GENERAL_CONTENT}")
                file(WRITE "${hypre_source_dir}/src/utilities/general.c" "${HYPRE_GENERAL_CONTENT}")
                message(STATUS "  HYPRE general.c patched to enumerate SYCL GPU devices via sycl::device::get_devices")
            endif()
        endif()
    endif()

endfunction()
