/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TEST_HELPERS_HEADER
#define TEST_HELPERS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include "utils.h"

/*-----------------------------------------------------------------------------
 * Hypre init/finalize helpers (older hypre releases do not provide them)
 *-----------------------------------------------------------------------------*/

#if HYPRE_CHECK_MIN_VERSION(22900, 0)
#define TEST_HYPRE_INIT() HYPRE_Initialize()
#define TEST_HYPRE_FINALIZE() HYPRE_Finalize()
#else
#define TEST_HYPRE_INIT() ((void)0)
#define TEST_HYPRE_FINALIZE() ((void)0)
#endif

/*-----------------------------------------------------------------------------
 * Fail-fast assertion macros (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

#define TEST_FAIL(msg) \
   do { \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
      exit(1); \
   } while (0)

#define ASSERT_TRUE(expr) \
   do { \
      if (!(expr)) { \
         fprintf(stderr, "FAIL: %s:%d: %s is false\n", __FILE__, __LINE__, #expr); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_FALSE(expr) \
   do { \
      if (expr) { \
         fprintf(stderr, "FAIL: %s:%d: %s is true\n", __FILE__, __LINE__, #expr); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_EQ(a, b) \
   do { \
      if ((a) != (b)) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%d) != %s (%d)\n", \
                __FILE__, __LINE__, #a, (int)(a), #b, (int)(b)); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_NE(a, b) \
   do { \
      if ((a) == (b)) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%d) == %s (%d)\n", \
                __FILE__, __LINE__, #a, (int)(a), #b, (int)(b)); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_STREQ(a, b) \
   do { \
      if (strcmp((a), (b)) != 0) { \
         fprintf(stderr, "FAIL: %s:%d: \"%s\" != \"%s\"\n", \
                __FILE__, __LINE__, (a), (b)); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_STRNE(a, b) \
   do { \
      if (strcmp((a), (b)) == 0) { \
         fprintf(stderr, "FAIL: %s:%d: \"%s\" == \"%s\"\n", \
                __FILE__, __LINE__, (a), (b)); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_NULL(ptr) \
   do { \
      if ((ptr) != NULL) { \
         fprintf(stderr, "FAIL: %s:%d: %s is not NULL\n", __FILE__, __LINE__, #ptr); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_NOT_NULL(ptr) \
   do { \
      if ((ptr) == NULL) { \
         fprintf(stderr, "FAIL: %s:%d: %s is NULL\n", __FILE__, __LINE__, #ptr); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_EQ_DOUBLE(a, b, tol) \
   do { \
      double diff = ((a) > (b)) ? ((a) - (b)) : ((b) - (a)); \
      if (diff > (tol)) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%.10g) != %s (%.10g) (diff: %.10g > %.10g)\n", \
                __FILE__, __LINE__, #a, (a), #b, (b), diff, tol); \
         exit(1); \
      } \
   } while (0)

/*-----------------------------------------------------------------------------
 * Test runner helper (simple wrapper, CTest handles orchestration)
 *-----------------------------------------------------------------------------*/

#define RUN_TEST(test_func) \
   do { \
      test_func(); \
   } while (0)

#define TEST_REQUIRE_FILE(path) \
   do { \
      if (access((path), F_OK) != 0) { \
         fprintf(stderr, "SKIP: missing data file: %s\n", (path)); \
         return; \
      } \
   } while (0)

/*-----------------------------------------------------------------------------
 * Temporary file helpers
 *-----------------------------------------------------------------------------*/

static char *temp_files[1024];
static int temp_file_count = 0;

static inline void
add_temp_file(const char *filename)
{
   if (temp_file_count < 1024)
   {
      temp_files[temp_file_count++] = strdup(filename);
   }
}

static inline void
cleanup_temp_files(void)
{
   for (int i = 0; i < temp_file_count; i++)
   {
      if (temp_files[i])
      {
         remove(temp_files[i]);
         free(temp_files[i]);
         temp_files[i] = NULL;
      }
   }
   temp_file_count = 0;
}

#define CREATE_TEMP_FILE(pattern) \
   ({ \
      char *tmp_file = strdup(pattern); \
      FILE *fp = fopen(tmp_file, "w"); \
      if (fp) { \
         fclose(fp); \
         add_temp_file(tmp_file); \
      } \
      tmp_file; \
   })

#endif /* TEST_HELPERS_HEADER */
