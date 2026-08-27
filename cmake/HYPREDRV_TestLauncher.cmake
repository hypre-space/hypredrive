# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Shared CTest launcher setup for the ordinary and pack-and-run scripts.
# This module selects the MPI/Flux command and applies CTest's GPU allocation
# to the accelerator visibility environment when the launcher does not own
# device assignment itself.

function(_hypredrv_apply_gpu_resource)
  if(NOT DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT})
    return()
  endif()

  set(_group_count "$ENV{CTEST_RESOURCE_GROUP_COUNT}")
  if(NOT _group_count MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "Invalid CTEST_RESOURCE_GROUP_COUNT='${_group_count}'")
  endif()
  if(_group_count LESS 1)
    return()
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

  # CTest may allocate multiple slots on the same physical GPU to different
  # MPI ranks.  Visibility variables describe devices, not rank slots; keep
  # each physical resource only once so HYPRE sees one logical device rather
  # than a duplicated list such as CUDA_VISIBLE_DEVICES=0,0,0,0.
  list(REMOVE_DUPLICATES _gpu_ids)
  if(DEFINED ENV{HYPREDRV_GPU_TEST_APPLY_VISIBILITY})
    string(TOUPPER "$ENV{HYPREDRV_GPU_TEST_APPLY_VISIBILITY}"
      _apply_gpu_visibility_env)
    if(_apply_gpu_visibility_env MATCHES "^(0|OFF|FALSE|NO)$")
      message(STATUS
        "[test] CTest GPU allocation retained; accelerator visibility override disabled")
      return()
    endif()
  endif()
  string(JOIN "," _visible_devices ${_gpu_ids})
  set(_visible_devices_env "ROCR_VISIBLE_DEVICES")
  if(DEFINED ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV} AND
     NOT "x$ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV}" STREQUAL "x")
    set(_visible_devices_env "$ENV{HYPREDRV_GPU_VISIBLE_DEVICES_ENV}")
  endif()
  if(NOT _visible_devices_env MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "Invalid GPU visibility environment variable '${_visible_devices_env}'")
  endif()

  if(_visible_devices_env MATCHES "^(ROCR|HIP)_VISIBLE_DEVICES$")
    # HIP applications and ROCr do not consistently give precedence to the
    # same variable.  Keep both masks identical so a scheduler-provided mask
    # cannot expose a second device to one MPI rank.
    set(ENV{ROCR_VISIBLE_DEVICES} "${_visible_devices}")
    set(ENV{HIP_VISIBLE_DEVICES} "${_visible_devices}")
    message(STATUS "[test] ROCR_VISIBLE_DEVICES=${_visible_devices}")
    message(STATUS "[test] HIP_VISIBLE_DEVICES=${_visible_devices}")
  else()
    set(ENV{${_visible_devices_env}} "${_visible_devices}")
    message(STATUS "[test] ${_visible_devices_env}=${_visible_devices}")
  endif()
endfunction()

function(hypredrv_prepare_test_launcher out_command out_postflags)
  set(_launcher_command "")
  set(_launcher_postflags "")
  set(_uses_flux FALSE)

  set(_launcher_mode "AUTO")
  if(DEFINED ENV{HYPREDRV_GPU_TEST_LAUNCHER} AND
     NOT "x$ENV{HYPREDRV_GPU_TEST_LAUNCHER}" STREQUAL "x")
    set(_launcher_mode "$ENV{HYPREDRV_GPU_TEST_LAUNCHER}")
  endif()
  string(TOUPPER "${_launcher_mode}" _launcher_mode)
  if(NOT _launcher_mode MATCHES "^(AUTO|MPIEXEC|FLUX)$")
    message(FATAL_ERROR
      "HYPREDRV_GPU_TEST_LAUNCHER must be AUTO, MPIEXEC, or FLUX")
  endif()

  set(_use_mpiexec_for_single_rank FALSE)
  if(DEFINED ENV{HYPREDRV_USE_MPIEXEC_FOR_SINGLE_RANK_TESTS})
    string(TOUPPER "$ENV{HYPREDRV_USE_MPIEXEC_FOR_SINGLE_RANK_TESTS}"
      _use_mpiexec_for_single_rank_env)
    if(_use_mpiexec_for_single_rank_env MATCHES "^(1|ON|TRUE|YES)$")
      set(_use_mpiexec_for_single_rank TRUE)
    endif()
  endif()

  set(_bypass_single_rank_launcher FALSE)
  if(DEFINED MPI_NUMPROCS AND MPI_NUMPROCS STREQUAL "1" AND
     NOT _use_mpiexec_for_single_rank)
    set(_bypass_single_rank_launcher TRUE)
    message(STATUS
      "[test] MPIEXEC single-rank bypass enabled; running target directly")
  endif()

  set(_has_ctest_gpu_resources FALSE)
  if(DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT} AND
     "$ENV{CTEST_RESOURCE_GROUP_COUNT}" MATCHES "^[0-9]+$" AND
     "$ENV{CTEST_RESOURCE_GROUP_COUNT}" GREATER 0)
    set(_has_ctest_gpu_resources TRUE)
  endif()

  if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "" AND
     NOT _bypass_single_rank_launcher)
    get_filename_component(_mpiexec_name "${MPIEXEC}" NAME)
    set(_select_flux FALSE)
    if(_has_ctest_gpu_resources)
      if(_launcher_mode STREQUAL "FLUX")
        set(_select_flux TRUE)
      elseif(_launcher_mode STREQUAL "AUTO" AND
             _mpiexec_name STREQUAL "srun" AND
             DEFINED ENV{FLUX_URI} AND
             NOT "x$ENV{FLUX_URI}" STREQUAL "x")
        set(_select_flux TRUE)
      endif()
    endif()

    if(_select_flux)
      if(DEFINED ENV{HYPREDRV_FLUX_EXECUTABLE} AND
         NOT "x$ENV{HYPREDRV_FLUX_EXECUTABLE}" STREQUAL "x")
        set(_flux_executable "$ENV{HYPREDRV_FLUX_EXECUTABLE}")
      else()
        unset(_flux_executable)
        find_program(_flux_executable NAMES flux NO_CACHE)
      endif()

      if(_flux_executable AND EXISTS "${_flux_executable}")
        set(_launcher_command
          "${_flux_executable}" run
          -N 1
          -n "${MPI_NUMPROCS}"
          --gpus-per-task=1)
        set(_uses_flux TRUE)
        message(STATUS
          "[test] GPU launcher=Flux packed allocation: ${_flux_executable} run "
          "-N 1 -n ${MPI_NUMPROCS} --gpus-per-task=1")
      elseif(_launcher_mode STREQUAL "FLUX")
        message(FATAL_ERROR
          "HYPREDRV_GPU_TEST_LAUNCHER=FLUX, but the flux executable was not found")
      else()
        message(STATUS
          "[test] Flux allocation detected, but flux was not found; using ${MPIEXEC}")
      endif()
    endif()

    if(NOT _launcher_command)
      set(_launcher_command
        "${MPIEXEC}" "${MPI_NUMPROC_FLAG}" "${MPI_NUMPROCS}" ${MPI_PREFLAGS})
      set(_launcher_postflags ${MPI_POSTFLAGS})
    endif()
  endif()

  if(NOT _uses_flux)
    _hypredrv_apply_gpu_resource()
  endif()

  set(${out_command} "${_launcher_command}" PARENT_SCOPE)
  set(${out_postflags} "${_launcher_postflags}" PARENT_SCOPE)
endfunction()
