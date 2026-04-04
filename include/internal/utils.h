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
#include "HYPRE_utilities.h"
#include "internal/compatibility.h"
#include "internal/error.h"

/*-----------------------------------------------------------------------------
 * HYPRE_SAFE_CALL macro
 *-----------------------------------------------------------------------------*/

// Macro for safely calling HYPRE functions.
// Sets HYPREDRV error codes and returns early from the calling function.
// This macro does NOT abort - it allows error propagation through HYPREDRV's error
// system.
#ifndef HYPRE_SAFE_CALL
#define HYPRE_SAFE_CALL(call)                                                     \
   do                                                                             \
   {                                                                              \
      HYPRE_Int hypre_ierr = (call);                                              \
      /* GCOVR_EXCL_BR_START */ /* SAFE_CALL hypre error path */                  \
      if (hypre_ierr != 0)      /* GCOVR_EXCL_BR_STOP */                          \
      {                                                                           \
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);                             \
         char hypre_err_msg[HYPRE_MAX_MSG_LEN];                                   \
         HYPRE_DescribeError(hypre_ierr, hypre_err_msg);                          \
         hypredrv_ErrorMsgAdd("HYPRE call failed at %s:%d in %s(): %s", __FILE__, \
                              __LINE__, __func__, hypre_err_msg);                 \
         return;                                                                  \
      }                                                                           \
   } while (0)
#endif

/*-----------------------------------------------------------------------------
 * Public prototypes
 *-----------------------------------------------------------------------------*/

char *hypredrv_StrToLowerCase(char *);
char *hypredrv_StrTrim(char *);
void  hypredrv_TrimTrailingWhitespace(char *);
void  hypredrv_NormalizeWhitespace(char *);
int   hypredrv_CheckBinaryDataExists(const char *);
int   hypredrv_CheckASCIIDataExists(const char *);
int   hypredrv_CountNumberOfPartitions(const char *);
int   hypredrv_ComputeNumberOfDigits(int);
void  hypredrv_SplitFilename(const char *, char **, char **);
void  hypredrv_CombineFilename(const char *, const char *, char **);
bool  hypredrv_IsYAMLFilename(const char *);

/*******************************************************************************
 *******************************************************************************/

/* Check if HYPRE_DEVELOP_NUMBER is greater or equal than a given value */
#define HYPRE_DEVELOP_NUMBER_GE(develop) (HYPREDRV_HYPRE_DEVELOP_NUMBER >= (develop))

/* Check if HYPRE_RELEASE_NUMBER is greater than a given value */
#define HYPRE_RELEASE_NUMBER_GT(release) (HYPREDRV_HYPRE_RELEASE_NUMBER > (release))

/* Check for a specific hypre release number and whether
   HYPRE_DEVELOP_NUMBER is defined and greater or equal than a given value */
#define HYPRE_RELEASE_NUMBER_EQ_AND_DEVELOP_NUMBER_GE(release, develop) \
   (HYPREDRV_HYPRE_RELEASE_NUMBER == (release) && HYPRE_DEVELOP_NUMBER_GE(develop))

/* Check for minimum HYPRE version in order to allow certain features.
 *
 * For exact tagged releases, HYPRE_DEVELOP_NUMBER may be omitted from
 * generated headers (for example from shallow clones or installed configs).
 * Treat an exact release match with no develop number as satisfying same-release
 * feature checks. */
#define HYPRE_CHECK_MIN_VERSION(release, develop)                               \
   (HYPRE_RELEASE_NUMBER_GT(release) ||                                         \
    (HYPREDRV_HYPRE_RELEASE_NUMBER == (release) &&                              \
     (HYPREDRV_HYPRE_DEVELOP_NUMBER == 0 || HYPRE_DEVELOP_NUMBER_GE(develop))))

enum
{
   GB_TO_BYTES        = (1 << 30),
   MAX_DIVISOR_LENGTH = 87,
};

static inline void
PrintLine(char ch, size_t len)
{
   for (size_t i = 0; i < len; i++)
   {
      putchar(ch);
   }
   putchar('\n');
}

#define PRINT_DASHED_LINE(_l) PrintLine('-', (_l))
#define PRINT_EQUAL_LINE(_l) PrintLine('=', (_l))

#define HAVE_COLORS 0 // TODO: add a runtime option for this parameter

#if HAVE_COLORS
#define TEXT_RESET "\033[0m"
#define TEXT_RED "\033[31m"
#define TEXT_GREEN "\033[32m"
#define TEXT_YELLOW "\033[33m"
#define TEXT_BOLD "\033[1m"
#define TEXT_REDBOLD "\033[1;31m"
#define TEXT_GREENBOLD "\033[1;32m"
#define TEXT_YELLOWBOLD "\033[1;33m"
#else
#define TEXT_RESET ""
#define TEXT_RED ""
#define TEXT_GREEN ""
#define TEXT_YELLOW ""
#define TEXT_BOLD ""
#define TEXT_REDBOLD ""
#define TEXT_GREENBOLD ""
#define TEXT_YELLOWBOLD ""
#endif

/* Check if two types match */
#ifdef __GNUC__
#define TYPES_MATCH(T1, T2) __builtin_types_compatible_p(T1, T2)
#else
#define TYPES_MATCH(T1, T2) (sizeof(T1) == sizeof(T2)) /* fallback */
#endif

#endif /* UTILS_HEADER */
