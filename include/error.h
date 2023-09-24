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

#define ERROR_CODE_MAX 32

typedef enum ErrorCode_enum
{
   ERROR_NONE               = 0x00000000, // No error
   ERROR_INVALID_KEY        = 0x00000001, // 1st  bit
   ERROR_INVALID_VAL        = 0x00000002, // 2nd  bit
   ERROR_MAYBE_INVALID_VAL  = 0x00000004, // 3rd  bit
   ERROR_FILE_NOT_FOUND     = 0x00000008, // 4th  bit
   ERROR_UNKNOWN            = 0x40000000  // 31th bit
} ErrorCode;

void ErrorCodeSet(ErrorCode);
int ErrorCodeGet(void);
void ErrorCodeDescribe(void);
void ErrorCodeReset(uint32_t);
void ErrorCodeResetAll(void);

/*******************************************************************************
 *******************************************************************************/

void ErrorMsgAdd(const char*);
void ErrorMsgAddCodeWithCount(ErrorCode, const char*);
void ErrorMsgAddInvalidKeyValPair(const char*, const char*);
void ErrorMsgAddUnknownKey(const char*);
void ErrorMsgAddUnknownVal(const char*);
void ErrorMsgAddMissingKey(const char*);
void ErrorMsgAddInvalidSolverOption(int);
void ErrorMsgAddInvalidPreconOption(int);
void ErrorMsgAddInvalidString(const char*);
void ErrorMsgAddInvalidFilename(const char*);
void ErrorMsgPrint();
void ErrorMsgClear();

#endif
