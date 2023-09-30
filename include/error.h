/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef ERROR_H
#define ERROR_H

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum ErrorCode_enum
{
   ERROR_NONE                   = 0x00000000, // No error
   ERROR_YAML_INVALID_INDENT    = 0x00000001, // 1st  bit
   ERROR_YAML_INVALID_DIVISOR   = 0x00000002, // 2nd  bit
   ERROR_YAML_TREE_NULL         = 0x00000004, // 3rd  bit
   ERROR_INVALID_KEY            = 0x00000008, // 4th  bit
   ERROR_INVALID_VAL            = 0x00000010, // 5th  bit
   ERROR_UNEXPECTED_VAL         = 0x00000020, // 6th  bit
   ERROR_MAYBE_INVALID_VAL      = 0x00000040, // 7th  bit
   ERROR_MISSING_KEY            = 0x00000080, // 8th  bit
   ERROR_EXTRA_KEY              = 0x00000100, // 9th  bit
   ERROR_MISSING_SOLVER         = 0x00000200, // 10th bit
   ERROR_MISSING_PRECON         = 0x00000400, // 11th bit
   ERROR_MISSING_DOFMAP         = 0x00000800, // 12th bit
   ERROR_INVALID_SOLVER         = 0x00001000, // 13th bit
   ERROR_INVALID_PRECON         = 0x00002000, // 14th bit
   ERROR_FILE_NOT_FOUND         = 0x00004000, // 15th bit
   ERROR_UNKNOWN                = 0x40000000  // 30th bit
} ErrorCode;

void ErrorCodeSet(ErrorCode);
uint32_t ErrorCodeGet(void);
bool ErrorCodeActive(void);
void ErrorCodeDescribe(void);
void ErrorCodeReset(uint32_t);
void ErrorCodeResetAll(void);

/*******************************************************************************
 *******************************************************************************/

void ErrorMsgAdd(const char*);
void ErrorMsgAddCodeWithCount(ErrorCode, const char*);
void ErrorMsgAddMissingKey(const char*);
void ErrorMsgAddExtraKey(const char*);
void ErrorMsgAddUnexpectedVal(const char*);
void ErrorMsgAddInvalidFilename(const char*);
void ErrorMsgPrint();
void ErrorMsgClear();
void ErrorMsgPrintAndAbort(MPI_Comm);

#endif /* ERROR_H */
