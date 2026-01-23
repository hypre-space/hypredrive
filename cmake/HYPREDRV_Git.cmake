# Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

function(hypredrv_setup_git_version_info HYPREDRV_GIT_DIR)
    set(HYPREDRV_GIT_FOUND FALSE PARENT_SCOPE)
    set(HYPREDRV_GIT_SHA "" PARENT_SCOPE)
    set(HYPREDRV_GIT_BRANCH "" PARENT_SCOPE)
    set(HYPREDRV_DEVELOP_STRING "" PARENT_SCOPE)
    set(HYPREDRV_BRANCH_NAME "" PARENT_SCOPE)

    if(NOT EXISTS "${HYPREDRV_GIT_DIR}/.git")
        return()
    endif()

    find_package(Git QUIET)
    if(NOT GIT_FOUND)
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${HYPREDRV_GIT_DIR} describe --match v* --long --abbrev=9 --always
        OUTPUT_VARIABLE hypredrv_develop_string
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE git_describe_result
    )
    if(git_describe_result EQUAL 0 AND NOT hypredrv_develop_string STREQUAL "")
        set(HYPREDRV_GIT_FOUND TRUE PARENT_SCOPE)
        set(HYPREDRV_DEVELOP_STRING "${hypredrv_develop_string}" PARENT_SCOPE)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${HYPREDRV_GIT_DIR} rev-parse --short=9 HEAD
        OUTPUT_VARIABLE hypredrv_git_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE git_sha_result
    )
    if(git_sha_result EQUAL 0 AND NOT hypredrv_git_sha STREQUAL "")
        set(HYPREDRV_GIT_SHA "${hypredrv_git_sha}" PARENT_SCOPE)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${HYPREDRV_GIT_DIR} rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE hypredrv_git_branch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE git_branch_result
    )
    if(git_branch_result EQUAL 0 AND NOT hypredrv_git_branch STREQUAL "")
        set(HYPREDRV_GIT_BRANCH "${hypredrv_git_branch}" PARENT_SCOPE)
        set(HYPREDRV_BRANCH_NAME "${hypredrv_git_branch}" PARENT_SCOPE)
    endif()
endfunction()
