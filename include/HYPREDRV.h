/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPREDV_HEADER
#define HYPREDV_HEADER

#include "HYPRE.h"
#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "HYPRE_parcsr_ls.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @defgroup HYPREDRV
 *
 * Public APIs to call hypre via HYPREDRIVE
 *
 * @{
 **/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * HYPREDRV_t
 *
 * Insert documentation
 **/

struct hypredrv_struct;
typedef struct hypredrv_struct* HYPREDRV_t;

/**
 * HYPREDRV_Create
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *obj_ptr);

/**
 * HYPREDRV_Destroy
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *obj_ptr);

/**
 * HYPREDRV_PrintLibInfo
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_PrintLibInfo(void);

/**
 * HYPREDRV_PrintExitInfo
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_PrintExitInfo(const char *argv0);

/**
 * HYPREDRV_InputArgsParse
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t obj);

/**
 * HYPREDRV_InputArgsGetExecPolicy
 *
 * TODO: Insert documentation
 **/

int
HYPREDRV_InputArgsGetExecPolicy(HYPREDRV_t obj);

/**
 * HYPREDRV_InputArgsGetNumRepetitions
 *
 * TODO: Insert documentation
 **/

int
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t obj);

/**
 * HYPREDRV_InputArgsGetNumRepetitions
 *
 * TODO: Insert documentation
 **/

int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemReadMatrix
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemSetRHS
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemSetInitialGuess
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemResetInitialGuess
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemSetPrecMatrix
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSystemReadDofmap
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t obj);

/**
 * HYPREDRV_PreconCreate
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSolverCreate
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSolverSetup
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSolverApply
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t obj);

/**
 * HYPREDRV_PreconDestroy
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t obj);

/**
 * HYPREDRV_LinearSolverDestroy
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t obj);

/**
 * HYPREDRV_StatsPrint
 *
 * TODO: Insert documentation
 **/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t obj);


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
