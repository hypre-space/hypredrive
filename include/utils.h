/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef UTILS_HEADER
#define UTILS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include "error.h"
#include "containers.h"

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

char* StrToLowerCase(char*);
char* StrTrim(char*);
int CheckBinaryDataExists(const char* prefix);
int ComputeNumberOfDigits(int);
void SplitFilename(const char*, char**, char**);
void CombineFilename(const char*, const char*, char**);

/*******************************************************************************
 *******************************************************************************/

#define MAX_DIVISOR_LENGTH 84
#define PRINT_DASHED_LINE(_l) for (size_t i = 0; i < (_l); i++) { printf("-"); } printf("\n");
#define PRINT_EQUAL_LINE(_l)  for (size_t i = 0; i < (_l); i++) { printf("="); } printf("\n");

#define HAVE_COLORS 0 // TODO: add a runtime option for this parameter

#if HAVE_COLORS
#define TEXT_RESET      "\033[0m"
#define TEXT_RED        "\033[31m"
#define TEXT_GREEN      "\033[32m"
#define TEXT_YELLOW     "\033[33m"
#define TEXT_BOLD       "\033[1m"
#define TEXT_REDBOLD    "\033[1;31m"
#define TEXT_GREENBOLD  "\033[1;32m"
#define TEXT_YELLOWBOLD "\033[1;33m"
#else
#define TEXT_RESET      ""
#define TEXT_RED        ""
#define TEXT_GREEN      ""
#define TEXT_YELLOW     ""
#define TEXT_BOLD       ""
#define TEXT_REDBOLD    ""
#define TEXT_GREENBOLD  ""
#define TEXT_YELLOWBOLD ""
#endif

#endif /* UTILS_HEADER */
