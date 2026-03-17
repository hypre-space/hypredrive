/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef ERROR_H
#define ERROR_H

#include <HYPRE_utilities.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO: consider using hypredrv_error_t instead of ErrorCode
typedef enum ErrorCode_enum
{
   ERROR_NONE                     = 0x00000000, // No error
   ERROR_YAML_INVALID_INDENT      = 0x00000001, //  1st bit
   ERROR_YAML_INVALID_BASE_INDENT = 0x00000002, //  2nd bit
   ERROR_YAML_INCONSISTENT_INDENT = 0x00000004, //  3rd bit
   ERROR_YAML_INVALID_DIVISOR     = 0x00000008, //  4th bit
   ERROR_YAML_TREE_NULL           = 0x00000010, //  5th bit
   ERROR_YAML_TREE_INVALID        = 0x00000020, //  6th bit
   ERROR_YAML_MIXED_INDENT        = 0x00000040, //  7th bit
   ERROR_YAML_INVALID_INDENT_JUMP = 0x00000080, //  8th bit
   ERROR_INVALID_KEY              = 0x00000100, //  9th bit
   ERROR_INVALID_VAL              = 0x00000200, // 10th bit
   ERROR_UNEXPECTED_VAL           = 0x00000400, // 11th bit
   ERROR_MAYBE_INVALID_VAL        = 0x00000800, // 12th bit
   ERROR_MISSING_KEY              = 0x00001000, // 13th bit
   ERROR_EXTRA_KEY                = 0x00002000, // 14th bit
   ERROR_MISSING_SOLVER           = 0x00004000, // 15th bit
   ERROR_MISSING_PRECON           = 0x00008000, // 16th bit
   ERROR_MISSING_DOFMAP           = 0x00010000, // 17th bit
   ERROR_INVALID_SOLVER           = 0x00020000, // 18th bit
   ERROR_INVALID_PRECON           = 0x00040000, // 19th bit
   ERROR_FILE_NOT_FOUND           = 0x00080000, // 20th bit
   ERROR_FILE_UNEXPECTED_ENTRY    = 0x00100000, // 21st bit
   ERROR_UNKNOWN_HYPREDRV_OBJ     = 0x00200000, // 22nd bit
   ERROR_HYPREDRV_NOT_INITIALIZED = 0x00400000, // 23rd bit
   ERROR_UNKNOWN_TIMING           = 0x00800000, // 24th bit
   ERROR_HYPRE_INTERNAL           = 0x01000000, // 25th bit
   ERROR_MISSING_LIB              = 0x02000000, // 26th bit
   ERROR_ALLOCATION               = 0x20000000, // 28th bit
   ERROR_OUT_OF_BOUNDS            = 0x40000000, // 29th bit
   ERROR_UNKNOWN                  = 0x80000000, // 30th bit
} ErrorCode;

// TODO: consider using hypredrv_ for function names
void     hypredrv_ErrorCodeSet(ErrorCode);
uint32_t hypredrv_ErrorCodeGet(void);
bool     hypredrv_ErrorCodeActive(void);
void     hypredrv_ErrorCodeDescribe(uint32_t);
void     hypredrv_ErrorCodeReset(uint32_t);
void     hypredrv_ErrorCodeResetAll(void);
void     hypredrv_ErrorStateReset(void);
bool     hypredrv_DistributedErrorCodeActive(MPI_Comm);

/*******************************************************************************
 *******************************************************************************/

void hypredrv_ErrorMsgAdd(const char *, ...);
void hypredrv_ErrorMsgAddCodeWithCount(ErrorCode, const char *);
void hypredrv_ErrorMsgAddMissingKey(const char *);
void hypredrv_ErrorMsgAddExtraKey(const char *);
void hypredrv_ErrorMsgAddUnexpectedVal(const char *);
void hypredrv_ErrorMsgAddInvalidFilename(const char *);
void hypredrv_ErrorMsgPrint(void);
void hypredrv_ErrorMsgClear(void);
void hypredrv_ErrorBacktracePrint(void);

/*******************************************************************************
 *******************************************************************************/

#define HYPREDRV_MALLOC_AND_CHECK(ptr, size)       \
   do                                              \
   {                                               \
      (ptr) = malloc(size);                        \
      if ((ptr) == NULL)                           \
      {                                            \
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);           \
         hypredrv_ErrorMsgAdd("Memory allocation failed!"); \
         return;                                   \
      }                                            \
   } while (0)

#endif /* ERROR_H */
