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

void ErrorMsgAdd(const char*);
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
