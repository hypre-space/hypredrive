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
int CheckBinaryDataExists(const char*);
int CountNumberOfPartitions(const char*);
int ComputeNumberOfDigits(int);
void SplitFilename(const char*, char**, char**);
void CombineFilename(const char*, const char*, char**);
bool HasFileExtension(const char*);

/*******************************************************************************
 *******************************************************************************/

/* Helper macros to check if HYPRE_DEVELOP_NUMBER is defined */
#if defined(HYPRE_DEVELOP_NUMBER)
  #define HYPRE_DEVELOP_NUMBER_EXISTS 1
#else
  #define HYPRE_DEVELOP_NUMBER_EXISTS 0
#endif

/* Check if HYPRE_DEVELOP_NUMBER is greater or equal than a given value */
#define HYPRE_DEVELOP_NUMBER_GE(develop) \
    (HYPRE_DEVELOP_NUMBER >= (develop))

/* Check if HYPRE_RELEASE_NUMBER is greater than a given value */
#define HYPRE_RELEASE_NUMBER_GT(release) \
    (HYPRE_RELEASE_NUMBER > (release))

/* Check for a specific hypre release number and whether
   HYPRE_DEVELOP_NUMBER is defined and greater or equal than a given value */
#define HYPRE_RELEASE_NUMBER_EQ_AND_DEVELOP_NUMBER_GE(release, develop) \
    (HYPRE_RELEASE_NUMBER == (release) &&\
     HYPRE_DEVELOP_NUMBER_EXISTS &&\
     HYPRE_DEVELOP_NUMBER_GE(develop))

/* Check for minimum HYPRE version in order to allow certain features */
#define HYPRE_CHECK_MIN_VERSION(release, develop) \
    (HYPRE_RELEASE_NUMBER_GT(release) ||\
     HYPRE_RELEASE_NUMBER_EQ_AND_DEVELOP_NUMBER_GE(release, develop))

#define GB_TO_BYTES (1 << 30)
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
