# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Build the launcher command used by CTest wrapper scripts.
#
# Within a Flux allocation, LLNL's srun compatibility wrapper requests an
# exclusive job step.  Concurrent GPU regression tests would therefore queue
# behind one another even when CTest assigned distinct GPU resources.  For
# resource-managed GPU tests, bypass that wrapper and request one GPU per rank
# directly from Flux.  Other launchers and non-Flux environments are unchanged.

function(hypredrv_prepare_test_launcher out_command out_postflags out_uses_flux)
  set(_launcher_command "")
  set(_launcher_postflags "")
  set(_uses_flux FALSE)

  if(NOT DEFINED MPIEXEC OR MPIEXEC STREQUAL "")
    set(${out_command} "" PARENT_SCOPE)
    set(${out_postflags} "" PARENT_SCOPE)
    set(${out_uses_flux} FALSE PARENT_SCOPE)
    return()
  endif()

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

  set(_has_ctest_gpu_resources FALSE)
  if(DEFINED ENV{CTEST_RESOURCE_GROUP_COUNT} AND
     NOT "x$ENV{CTEST_RESOURCE_GROUP_COUNT}" STREQUAL "x")
    set(_has_ctest_gpu_resources TRUE)
  endif()

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
    set(_flux_executable "")
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

  set(${out_command} "${_launcher_command}" PARENT_SCOPE)
  set(${out_postflags} "${_launcher_postflags}" PARENT_SCOPE)
  set(${out_uses_flux} "${_uses_flux}" PARENT_SCOPE)
endfunction()
