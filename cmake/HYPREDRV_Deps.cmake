# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


# Ordered dependency discovery; Caliper and SuperLU must precede hypre.
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsCommon.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsOptions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsCaliper.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsSuperLU.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsHypre.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsHwloc.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/HYPREDRV_DepsCompression.cmake)
