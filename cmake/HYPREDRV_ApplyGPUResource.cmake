# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Translate CTest's allocated GPU resource groups into the visibility
# environment used by the selected accelerator runtime.  CTest only defines
# CTEST_RESOURCE_GROUP_* for a test when the caller supplied a resource
# specification file, so ordinary test invocations retain their existing
# device visibility.

function(hypredrv_apply_gpu_resource)
  if(NOT DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT})
    return()
  endif()

  set(_group_count "$ENV{CTEST_RESOURCE_GROUP_COUNT}")
  if(NOT _group_count MATCHES "^[0-9]+$" OR _group_count LESS 1)
    message(FATAL_ERROR
      "Invalid CTEST_RESOURCE_GROUP_COUNT='${_group_count}'")
  endif()

  set(_gpu_ids "")
  math(EXPR _last_group "${_group_count} - 1")
  foreach(_group RANGE 0 ${_last_group})
    set(_resource_var "CTEST_RESOURCE_GROUP_${_group}_GPUS")
    if(NOT DEFINED ENV{${_resource_var}})
      message(FATAL_ERROR
        "CTest did not provide ${_resource_var} for an allocated GPU test")
    endif()

    set(_resource_spec "$ENV{${_resource_var}}")
    string(REGEX MATCH "id:([^,;]+)" _resource_match "${_resource_spec}")
    if(NOT _resource_match)
      message(FATAL_ERROR
        "Could not parse GPU resource allocation '${_resource_spec}'")
    endif()
    list(APPEND _gpu_ids "${CMAKE_MATCH_1}")
  endforeach()

  string(JOIN "," _visible_devices ${_gpu_ids})

  set(_visible_devices_env "ROCR_VISIBLE_DEVICES")
  if(DEFINED ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV} AND
     NOT "$ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV}" STREQUAL "")
    set(_visible_devices_env "$ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV}")
  endif()
  if(NOT _visible_devices_env MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "Invalid GPU visibility environment variable '${_visible_devices_env}'")
  endif()

  set(ENV{${_visible_devices_env}} "${_visible_devices}")
  message(STATUS
    "[test] ${_visible_devices_env}=${_visible_devices}"
  )
endfunction()

hypredrv_apply_gpu_resource()
