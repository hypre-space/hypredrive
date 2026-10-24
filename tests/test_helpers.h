/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TEST_HELPERS_HEADER
#define TEST_HELPERS_HEADER

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "HYPRE.h"
#include "internal/utils.h"

/*-----------------------------------------------------------------------------
 * Hypre init/finalize helpers (older hypre releases do not provide them)
 *-----------------------------------------------------------------------------*/

#if HYPRE_CHECK_MIN_VERSION(22900, 0)
static inline void
hypredrv_TestHypreInit(void)
{
   HYPRE_Initialize();
#if defined(HYPRE_USING_GPU)
   HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
   HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif
}

#define TEST_HYPRE_INIT() hypredrv_TestHypreInit()
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

/* Comparison macros evaluate each argument once, so side-effecting arguments
 * such as fclose(fp) are not repeated when a failure is reported. */
#define ASSERT_EQ(a, b) \
   do { \
      __typeof__((a) + (b)) _lhs = (a); \
      __typeof__((a) + (b)) _rhs = (b); \
      if (_lhs != _rhs) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%d) != %s (%d)\n", \
                __FILE__, __LINE__, #a, (int)_lhs, #b, (int)_rhs); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_EQ_SIZE(a, b) \
   do { \
      size_t _lhs = (size_t)(a); \
      size_t _rhs = (size_t)(b); \
      if (_lhs != _rhs) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%zu) != %s (%zu)\n", \
                __FILE__, __LINE__, #a, _lhs, #b, _rhs); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_EQ_U32(a, b) \
   do { \
      uint32_t _lhs = (uint32_t)(a); \
      uint32_t _rhs = (uint32_t)(b); \
      if (_lhs != _rhs) { \
         fprintf(stderr, "FAIL: %s:%d: %s (0x%08" PRIx32 ") != %s (0x%08" PRIx32 ")\n", \
                __FILE__, __LINE__, #a, _lhs, #b, _rhs); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_NE(a, b) \
   do { \
      __typeof__((a) + (b)) _lhs = (a); \
      __typeof__((a) + (b)) _rhs = (b); \
      if (_lhs == _rhs) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%d) == %s (%d)\n", \
                __FILE__, __LINE__, #a, (int)_lhs, #b, (int)_rhs); \
         exit(1); \
      } \
   } while (0)

#define ASSERT_PTR_EQ(a, b) \
   do { \
      const void *_lhs = (const void *)(a); \
      const void *_rhs = (const void *)(b); \
      if (_lhs != _rhs) { \
         fprintf(stderr, "FAIL: %s:%d: %s (%p) != %s (%p)\n", \
                __FILE__, __LINE__, #a, _lhs, #b, _rhs); \
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

#define ASSERT_EQ_DOUBLE(a, b, tol)                                                      \
   do                                                                                    \
   {                                                                                     \
      double _lhs  = (double)(a);                                                        \
      double _rhs  = (double)(b);                                                        \
      double _tol  = (double)(tol);                                                      \
      double _diff = fabs(_lhs - _rhs);                                                  \
      if (!(_diff <= _tol))                                                              \
      {                                                                                  \
         fprintf(stderr,                                                                 \
                 "FAIL: %s:%d: %s (%.10g) != %s (%.10g) (difference: %.10g, tolerance: " \
                 "%.10g)\n",                                                             \
                 __FILE__, __LINE__, #a, _lhs, #b, _rhs, _diff, _tol);                   \
         exit(1);                                                                        \
      }                                                                                  \
   } while (0)

#define ASSERT_GT(a, b) \
   do { \
      if (!((a) > (b))) { \
         fprintf(stderr, "FAIL: %s:%d: expected %s > %s\n", \
                __FILE__, __LINE__, #a, #b); \
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
         exit(77); \
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

/*-----------------------------------------------------------------------------
 * Scratch directory helpers: every test process writes under its own private
 * directory, so parallel ctest runs never share paths and no shell is needed.
 *-----------------------------------------------------------------------------*/

static char test_scratch_dir[PATH_MAX];

/* Remove a file or directory tree without following symlinks; missing paths
 * are ignored. */
static inline void
test_rmtree(const char *path)
{
   struct stat st;
   DIR        *dir = (lstat(path, &st) == 0 && S_ISDIR(st.st_mode)) ? opendir(path) : NULL;
   if (dir)
   {
      struct dirent *entry = NULL;
      while ((entry = readdir(dir)))
      {
         if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
         {
            char child[PATH_MAX];
            snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
            test_rmtree(child);
         }
      }
      closedir(dir);
   }
   (void)remove(path);
}

static inline void
test_scratch_cleanup(void)
{
   test_rmtree(test_scratch_dir);
}

/* Each test process owns a private TMPDIR-aware directory. Paths returned
 * here do not exist until the caller creates them. */
static inline char *
test_scratch_path(char *buf, size_t size, const char *name)
{
   if (!test_scratch_dir[0])
   {
      const char *base = getenv("TMPDIR");
#ifdef _WIN32
      if (!base || !*base) { base = getenv("TEMP"); }
      if (!base || !*base) { base = "."; }
      for (int i = 0; ; i++)
      {
         ASSERT_TRUE(snprintf(test_scratch_dir, sizeof(test_scratch_dir),
                              "%s/hypredrv-test-%ld-%d", base, (long)getpid(), i)
                     < (int)sizeof(test_scratch_dir));
         if (mkdir(test_scratch_dir, 0700) == 0) { break; }
         ASSERT_EQ(errno, EEXIST);
      }
#else
      if (!base || !*base) { base = "/tmp"; }
      ASSERT_TRUE(snprintf(test_scratch_dir, sizeof(test_scratch_dir),
                           "%s/hypredrv-test-XXXXXX", base)
                  < (int)sizeof(test_scratch_dir));
      ASSERT_NOT_NULL(mkdtemp(test_scratch_dir));
#endif
      ASSERT_EQ(atexit(test_scratch_cleanup), 0);
   }
   int written = snprintf(buf, size, "%s/%s", test_scratch_dir, name);
   ASSERT_TRUE(written >= 0 && (size_t)written < size);
   return buf;
}

#endif /* TEST_HELPERS_HEADER */
