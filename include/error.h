/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef enum ErrorCode_enum
{
   ERROR_NONE                  = 0x00000000, // No error
   ERROR_YAML_INVALID_INDENT   = 0x00000001, // 1st  bit
   ERROR_YAML_TREE_NULL        = 0x00000002, // 2nd  bit

   ERROR_INVALID_KEY        = 0x00000010, // 1st  bit
   ERROR_INVALID_VAL        = 0x00000020, // 2nd  bit
   ERROR_MAYBE_INVALID_VAL  = 0x00000040, // 3rd  bit
   ERROR_MISSING_KEY        = 0x00000080, // 4th  bit
   ERROR_EXTRA_KEY          = 0x00000100, // 5th  bit
   ERROR_MISSING_SOLVER     = 0x00000110, // 6th  bit
   ERROR_MISSING_PRECON     = 0x00000120, // 7th  bit
   ERROR_INVALID_SOLVER     = 0x00000140, // 8th  bit
   ERROR_INVALID_PRECON     = 0x00000180, // 9th  bit
   ERROR_FILE_NOT_FOUND     = 0x00000200, // 10th bit
   ERROR_UNKNOWN            = 0x40000000  // 31th bit
} ErrorCode;

void ErrorCodeSet(ErrorCode);
uint32_t ErrorCodeGet(void);
void ErrorCodeDescribe(void);
void ErrorCodeReset(uint32_t);
void ErrorCodeResetAll(void);

/*******************************************************************************
 *******************************************************************************/

void ErrorMsgAdd(const char*);
void ErrorMsgAddCodeWithCount(ErrorCode, const char*);
void ErrorMsgAddInvalidKeyValPair(const char*, const char*); // TODO: Remove me
void ErrorMsgAddUnknownKey(const char*); // TODO: Remove me
void ErrorMsgAddUnknownVal(const char*); // TODO: Remove me
void ErrorMsgAddMissingKey(const char*);
void ErrorMsgAddExtraKey(const char*);
void ErrorMsgAddInvalidSolverOption(int); // TODO: Remove me
void ErrorMsgAddInvalidPreconOption(int); // TODO: Remove me
void ErrorMsgAddInvalidString(const char*);
void ErrorMsgAddInvalidFilename(const char*); // TODO: Remove me
void ErrorMsgPrint();
void ErrorMsgClear();

#endif
