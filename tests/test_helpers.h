/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TEST_HELPERS_HEADER
#define TEST_HELPERS_HEADER

#ifndef _WIN32
#include <dirent.h>
#endif
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
#ifndef _WIN32
#include <unistd.h>
#endif

#include "HYPRE.h"
#include "internal/compatibility.h"
#include "internal/utils.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef _WIN32
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <windows.h>
/* Keep hypredrive's error enum visible instead of the Win32 error macro. */
#undef ERROR_FILE_NOT_FOUND
#ifndef F_OK
#define F_OK 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif
#define mkdir(path, mode) _mkdir(path)
#define chdir(path) _chdir(path)
#define getcwd(buf, size) _getcwd(buf, (int)(size))

static inline int
test_setenv(const char *name, const char *value, int overwrite)
{
   if (!overwrite && getenv(name)) return 0;
   return _putenv_s(name, value);
}

static inline int
test_unsetenv(const char *name)
{
   return _putenv_s(name, "");
}

static inline int
test_mkstemp(char *path)
{
   size_t size = strlen(path);
   if (size < 6 || strcmp(path + size - 6, "XXXXXX"))
   {
      errno = EINVAL;
      return -1;
   }
   for (int i = 0; i < 128; i++)
   {
      memcpy(path + size - 6, "XXXXXX", 6);
      int error = _mktemp_s(path, size + 1);
      if (error)
      {
         errno = error;
         return -1;
      }
      int fd =
         _open(path, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY, _S_IREAD | _S_IWRITE);
      if (fd >= 0 || errno != EEXIST) return fd;
   }
   errno = EEXIST;
   return -1;
}

#define setenv(name, value, overwrite) test_setenv(name, value, overwrite)
#define unsetenv(name) test_unsetenv(name)
#define mkstemp(path) test_mkstemp(path)
#endif

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

/* Compare using the usual arithmetic conversions, evaluating each operand once. */
#define ASSERT_EQ(a, b)                                                          \
   do                                                                            \
   {                                                                             \
      if ((a) != (b))                                                            \
      {                                                                          \
         fprintf(stderr, "FAIL: %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
         exit(1);                                                                \
      }                                                                          \
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

#define ASSERT_NE(a, b)                                                          \
   do                                                                            \
   {                                                                             \
      if ((a) == (b))                                                            \
      {                                                                          \
         fprintf(stderr, "FAIL: %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); \
         exit(1);                                                                \
      }                                                                          \
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

static inline char *
create_temp_file(const char *pattern)
{
   char *tmp_file = strdup(pattern);
   FILE *fp       = tmp_file ? fopen(tmp_file, "w") : NULL;
   if (fp)
   {
      fclose(fp);
      add_temp_file(tmp_file);
   }
   return tmp_file;
}

#define CREATE_TEMP_FILE(pattern) create_temp_file(pattern)

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
#ifdef _WIN32
   DWORD attributes = GetFileAttributesA(path);
   if (attributes == INVALID_FILE_ATTRIBUTES) return;
   if (attributes & FILE_ATTRIBUTE_DIRECTORY)
   {
      if (!(attributes & FILE_ATTRIBUTE_REPARSE_POINT))
      {
         char pattern[PATH_MAX];
         snprintf(pattern, sizeof(pattern), "%s/*", path);
         WIN32_FIND_DATAA entry;
         HANDLE           search = FindFirstFileA(pattern, &entry);
         if (search != INVALID_HANDLE_VALUE)
         {
            do
            {
               if (strcmp(entry.cFileName, ".") && strcmp(entry.cFileName, ".."))
               {
                  char child[PATH_MAX];
                  snprintf(child, sizeof(child), "%s/%s", path, entry.cFileName);
                  test_rmtree(child);
               }
            } while (FindNextFileA(search, &entry));
            FindClose(search);
         }
      }
      (void)RemoveDirectoryA(path);
   }
   else
   {
      (void)DeleteFileA(path);
   }
#else
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
#endif
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

#ifdef _WIN32
/* The MS CRT tmpfile() may require access to the drive root. Use the private
 * per-process scratch directory and let its registered cleanup remove files. */
static inline FILE *
test_tmpfile(void)
{
   /* _mktemp_s provides only 26 names per template; vary its prefix too. */
   static unsigned long next_capture = 0;
   char                 name[64];
   char                 path[PATH_MAX];
   snprintf(name, sizeof(name), "capture-%lu-XXXXXX", next_capture++);
   test_scratch_path(path, sizeof(path), name);
   int fd = test_mkstemp(path);
   if (fd < 0) return NULL;
   FILE *fp = fdopen(fd, "w+b");
   if (!fp) close(fd);
   return fp;
}
#define tmpfile() test_tmpfile()
#endif

#endif /* TEST_HELPERS_HEADER */
