# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

set(_hypredrv_fuzz_harness "${CMAKE_CURRENT_SOURCE_DIR}/harness.c")
set(_hypredrv_fuzz_solve_example_seeds
    "${CMAKE_SOURCE_DIR}/examples/ex1.yml"
    "${CMAKE_SOURCE_DIR}/examples/ex2.yml"
    "${CMAKE_SOURCE_DIR}/examples/ex7.yml")

hypredrv_add_fuzz_target(hypredrv-fuzz-parse parse
    SOURCE "${_hypredrv_fuzz_harness}"
    DICT
        "${CMAKE_CURRENT_SOURCE_DIR}/dicts/yaml.dict"
        "${CMAKE_CURRENT_SOURCE_DIR}/dicts/cli.dict"
    CORPUS "${CMAKE_SOURCE_DIR}/examples"
    REGRESSIONS "${CMAKE_CURRENT_SOURCE_DIR}/regressions/parse")

hypredrv_add_fuzz_target(hypredrv-fuzz-solve solve
    SOURCE "${_hypredrv_fuzz_harness}"
    DICT
        "${CMAKE_CURRENT_SOURCE_DIR}/dicts/yaml.dict"
        "${CMAKE_CURRENT_SOURCE_DIR}/dicts/cli.dict"
    CORPUS
        ${_hypredrv_fuzz_solve_example_seeds}
        "${CMAKE_CURRENT_SOURCE_DIR}/seeds/solve"
    REGRESSIONS "${CMAKE_CURRENT_SOURCE_DIR}/regressions/solve")

if(HYPREDRV_ENABLE_COMPRESSION)
    hypredrv_add_fuzz_target(hypredrv-fuzz-lsseq lsseq
        SOURCE "${_hypredrv_fuzz_harness}"
        DICT "${CMAKE_CURRENT_SOURCE_DIR}/dicts/ijmatrix.dict"
        CORPUS "${CMAKE_CURRENT_SOURCE_DIR}/seeds/lsseq"
        REGRESSIONS "${CMAKE_CURRENT_SOURCE_DIR}/regressions/lsseq")
else()
    message(STATUS "Skipping hypredrv-fuzz-lsseq: HYPREDRV_ENABLE_COMPRESSION is OFF")
endif()

hypredrv_add_fuzz_target(hypredrv-fuzz-matrix matrix
    SOURCE "${_hypredrv_fuzz_harness}"
    DICT "${CMAKE_CURRENT_SOURCE_DIR}/dicts/ijmatrix.dict"
    CORPUS "${CMAKE_CURRENT_SOURCE_DIR}/seeds/matrix"
    REGRESSIONS "${CMAKE_CURRENT_SOURCE_DIR}/regressions/matrix")

hypredrv_add_fuzz_target(hypredrv-fuzz-vector vector
    SOURCE "${_hypredrv_fuzz_harness}"
    DICT "${CMAKE_CURRENT_SOURCE_DIR}/dicts/ijmatrix.dict"
    CORPUS "${CMAKE_CURRENT_SOURCE_DIR}/seeds/vector"
    REGRESSIONS "${CMAKE_CURRENT_SOURCE_DIR}/regressions/vector")
