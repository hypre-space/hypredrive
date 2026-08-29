/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/utils.h"
#include <ctype.h>
#include <fcntl.h>
#include <string.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#endif
#include "internal/containers.h"

/*-----------------------------------------------------------------------------
 * hypredrv_HypreClearConvergenceErrors / hypredrv_HypreConsumeErrors
 *-----------------------------------------------------------------------------*/

void
hypredrv_HypreClearConvergenceErrors(void)
{
   HYPRE_Int hypre_error = HYPRE_GetError();

   /* Nested / inexact solvers often leave HYPRE_ERROR_CONV sticky. Clear that
    * soft failure so it does not poison later hypre calls, but keep any harder
    * error bits for the caller. */
   if (hypre_error && !(hypre_error & ~HYPRE_ERROR_CONV))
   {
      HYPRE_ClearAllErrors();
   }
}

void
hypredrv_HypreConsumeErrors(void)
{
   HYPRE_Int hypre_error = HYPRE_GetError();
   HYPRE_Int hard_error  = hypre_error & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG);

#if HYPRE_CHECK_MIN_VERSION(22900, 0)
   int local_hard_error  = hard_error != 0;
   int global_hard_error = 0;
   MPI_Allreduce(&local_hard_error, &global_hard_error, 1, MPI_INT, MPI_LOR,
                 MPI_COMM_WORLD);
#endif

   if (!hypre_error)
   {
#if HYPRE_CHECK_MIN_VERSION(23300, 0)
      /* HYPRE routines can raise and then clear an error internally. Discard
       * any diagnostics retained by print mode 1 even when no sticky flag is
       * left for us to consume. */
      HYPRE_ClearErrorMessages();
#endif
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
      if (global_hard_error)
      {
         HYPRE_PrintErrorMessages(MPI_COMM_WORLD);
         fflush(stderr);
      }
#endif
      return;
   }

   if (hard_error)
   {
      char hypre_err_msg[HYPRE_MAX_MSG_LEN];
      fprintf(stderr, "[HYPREDRV] HYPRE hard error flags=0x%x, argument=%d\n",
              (unsigned)hypre_error, (int)HYPRE_GetErrorArg());
      fflush(stderr);
      HYPRE_DescribeError(hypre_error, hypre_err_msg);
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAddUnique("HYPRE reported error 0x%x: %s", (unsigned)hypre_error,
                                 hypre_err_msg);
   }
#if HYPRE_CHECK_MIN_VERSION(22900, 0)
   /* HYPRE_PrintErrorMessages is collective; print local diagnostics only
    * after all ranks have agreed that at least one hard error occurred. */
   if (global_hard_error)
   {
      HYPRE_PrintErrorMessages(MPI_COMM_WORLD);
      fflush(stderr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(23300, 0)
   /* HYPRE_ClearErrorMessages was added in 2.33.0. Clear both soft-warning
    * diagnostics and the hard-error messages printed above. */
   HYPRE_ClearErrorMessages();
#endif

   /* Consume the sticky hypre state after preserving every hard error in
    * HypreDrive's error state. Convergence failures and argument-flag warnings
    * (for example ILUT's usable zero-row factorization) remain soft results. */
   HYPRE_ClearAllErrors();
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrToLowerCase
 *-----------------------------------------------------------------------------*/

char *
hypredrv_StrToLowerCase(char *str)
{
   if (!str)
   {
      return str;
   }
   for (int i = 0; str[i]; i++)
   {
      str[i] = (char)tolower((unsigned char)str[i]);
   }
   return str;
}

/*-----------------------------------------------------------------------------
 * hypredrv_StrTrim
 *-----------------------------------------------------------------------------*/

char *
hypredrv_StrTrim(char *str)
{
   if (!str)
   {
      return NULL;
   }

   for (int i = (int)strlen(str) - 1; i >= 0 && str[i] == ' '; i--)
   {
      str[i] = '\0';
   }

   return str;
}

/*-----------------------------------------------------------------------------
 * hypredrv_TrimTrailingWhitespace
 *-----------------------------------------------------------------------------*/

void
hypredrv_TrimTrailingWhitespace(char *str)
{
   if (!str)
   {
      return;
   }

   size_t len = strlen(str);
   while (len > 0 && isspace((unsigned char)str[len - 1]))
   {
      str[--len] = '\0';
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_NormalizeWhitespace
 *-----------------------------------------------------------------------------*/

void
hypredrv_NormalizeWhitespace(char *str)
{
   if (!str)
   {
      return;
   }

   char *src       = str;
   char *dst       = str;
   int   saw_space = 0;

   while (*src)
   {
      unsigned char c = (unsigned char)*src++;
      if (isspace(c))
      {
         saw_space = 1;
         continue;
      }

      if (saw_space && dst != str)
      {
         *dst++ = ' ';
      }
      *dst++    = (char)c;
      saw_space = 0;
   }
   *dst = '\0';
}

/*-----------------------------------------------------------------------------
 * hypredrv_BinaryPathPrefixIsSafe
 *-----------------------------------------------------------------------------*/

int
hypredrv_BinaryPathPrefixIsSafe(const char *prefix)
{
   size_t len;

   if (!prefix || prefix[0] == '\0')
   {
      return 0;
   }
   len = strlen(prefix);
   if (len == 0 || len >= (size_t)MAX_FILENAME_LENGTH)
   {
      return 0;
   }
   if (strstr(prefix, "..") != NULL)
   {
      return 0;
   }
   return 1;
}

static int
FormatPartitionFilename(char *filename, size_t filename_size, const char *prefix,
                        int partition, int binary)
{
   const char *extension = binary ? ".bin" : "";
   int         written =
      snprintf(filename, filename_size, "%s.%05d%s", prefix, partition, extension);

   return written >= 0 && (size_t)written < filename_size;
}

/*-----------------------------------------------------------------------------
 * hypredrv_FopenCreateRestricted
 *
 * Create or open for write with mode 0600 (subject to umask). binary: use "wb"/"ab".
 *-----------------------------------------------------------------------------*/

FILE *
hypredrv_FopenCreateRestricted(const char *path, int append, int binary)
{
   int         fd;
   int         flags = O_WRONLY | O_CREAT;
   const char *fdmode;

   if (!path || path[0] == '\0')
   {
      return NULL;
   }
   if (append)
   {
      flags |= O_APPEND;
   }
   else
   {
      flags |= O_TRUNC;
   }
   fd = open(path, flags, 0600);
   if (fd < 0)
   {
      return NULL;
   }
   if (append)
   {
      fdmode = binary ? "ab" : "a";
   }
   else
   {
      fdmode = binary ? "wb" : "w";
   }
   {
      FILE *fp = fdopen(fd, fdmode);
      if (!fp)
      {
         (void)close(fd);
         return NULL;
      }
      return fp;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_CheckBinaryDataExists
 *-----------------------------------------------------------------------------*/

int
hypredrv_CheckBinaryDataExists(const char *prefix)
{
   char filename[MAX_FILENAME_LENGTH] = {0};

   int   file_exists = 0;
   FILE *fp          = NULL;

   if (!hypredrv_BinaryPathPrefixIsSafe(prefix))
   {
      return 0;
   }

   /* Check if binary data exist */
   if (!FormatPartitionFilename(filename, sizeof(filename), prefix, 0, 1))
   {
      return 0;
   }
   file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
   if (fp)
   {
      fclose(fp);
   }

   return file_exists;
}

/*-----------------------------------------------------------------------------
 * hypredrv_CheckASCIIDataExists
 *-----------------------------------------------------------------------------*/

int
hypredrv_CheckASCIIDataExists(const char *prefix)
{
   char filename[MAX_FILENAME_LENGTH] = {0};

   int   file_exists = 0;
   FILE *fp          = NULL;

   if (!hypredrv_BinaryPathPrefixIsSafe(prefix))
   {
      return 0;
   }

   /* Check if ASCII data exist */
   if (!FormatPartitionFilename(filename, sizeof(filename), prefix, 0, 0))
   {
      return 0;
   }
   file_exists = ((fp = fopen(filename, "r")) == NULL) ? 0 : 1;
   if (fp)
   {
      fclose(fp);
   }

   return file_exists;
}

/*-----------------------------------------------------------------------------
 * hypredrv_CountNumberOfPartitions
 *-----------------------------------------------------------------------------*/

int
hypredrv_CountNumberOfPartitions(const char *prefix)
{
   char filename[MAX_FILENAME_LENGTH];
   int  num_files = 0;

   if (prefix == NULL)
   {
      return 0;
   }
   if (!hypredrv_BinaryPathPrefixIsSafe(prefix))
   {
      return 0;
   }

   while (1)
   {
      FILE *fp = NULL;
      int   file_exists;

      if (!FormatPartitionFilename(filename, sizeof(filename), prefix, num_files, 1))
      {
         break;
      }
      fp          = fopen(filename, "r");
      file_exists = (fp == NULL) ? 0 : 1;
      if (fp)
      {
         fclose(fp);
      }
      if (!file_exists)
      {
         if (!FormatPartitionFilename(filename, sizeof(filename), prefix, num_files, 0))
         {
            break;
         }
         fp          = fopen(filename, "r");
         file_exists = (fp == NULL) ? 0 : 1;
         if (fp)
         {
            fclose(fp);
         }
      }

      if (!file_exists)
      {
         break;
      }

      num_files++;
   }

   return num_files;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ComputeNumberOfDigits
 *-----------------------------------------------------------------------------*/

int
hypredrv_ComputeNumberOfDigits(int number)
{
   int ndigits = 0;

   while (number)
   {
      number /= 10;
      ndigits++;
   }

   return ndigits;
}

/*-----------------------------------------------------------------------------
 * hypredrv_SplitFilename
 *-----------------------------------------------------------------------------*/

void
hypredrv_SplitFilename(const char *filename, char **dirname_ptr, char **basename_ptr)
{
   const char *last_slash = strrchr(filename, '/');
#ifdef _WIN32
   const char *last_backslash = strrchr(filename, '\\');
   if (last_backslash && (!last_slash || last_backslash > last_slash))
   {
      last_slash = last_backslash;
   }
#endif
   char *dirname  = NULL;
   char *basename = NULL;

   if (last_slash != NULL)
   {
      /* Allocate memory and copy dirname */
      int dirname_length = (int)(last_slash - filename);

      dirname = (char *)malloc((size_t)dirname_length + 1);
      if (dirname)
      {
         snprintf(dirname, (size_t)dirname_length + 1, "%.*s", dirname_length, filename);
      }

      /* Allocate memory and copy basename */
      basename = (char *)malloc(strlen(last_slash + 1) + 1);
      if (basename)
      {
         snprintf(basename, strlen(last_slash + 1) + 1, "%s", last_slash + 1);
      }
   }
   else
   {
      /* No slash found, assume current directory and filename is the basename */
      dirname  = strdup(".");
      basename = strdup(filename);
   }

   /* Set output pointers */
   *dirname_ptr  = dirname;
   *basename_ptr = basename;
}

/*-----------------------------------------------------------------------------
 * hypredrv_CombineFilename
 *-----------------------------------------------------------------------------*/

void
hypredrv_CombineFilename(const char *dirname, const char *basename, char **filename_ptr)
{
   size_t length   = 0;
   char  *filename = NULL;

   /* Compute filename length. +2 for the slash and null terminator */
   length = strlen(dirname) + strlen(basename) + 2;

   /* Allocate space for the filename */
   filename = (char *)malloc(length);

   /* Combine dirname and basename */
   if (filename != NULL)
   {
      snprintf(filename, length, "%s", dirname);

      /* Add a slash only if dirname is not empty and does not already end in a slash */
      if (dirname[0] != '\0' && dirname[strlen(dirname) - 1] != '/')
      {
         strncat(filename, "/", length - strlen(filename) - 1);
      }

      strncat(filename, basename, length - strlen(filename) - 1);
   }

   /* Set output pointer */
   *filename_ptr = filename;
}

/*-----------------------------------------------------------------------------
 * hypredrv_Realpath
 *
 * Resolve an existing path and return a newly allocated canonical spelling.
 * MinGW does not provide the POSIX realpath(path, NULL) extension, so use the
 * corresponding CRT routine there and normalize its directory separators for
 * the path-security checks shared with Unix.
 *-----------------------------------------------------------------------------*/

char *
hypredrv_Realpath(const char *path)
{
   if (!path)
   {
      return NULL;
   }

#ifdef _WIN32
   if (_access(path, 0) != 0)
   {
      return NULL;
   }

   char *resolved = _fullpath(NULL, path, 0);
   if (resolved)
   {
      for (char *p = resolved; *p; p++)
      {
         if (*p == '\\')
         {
            *p = '/';
         }
      }
   }
   return resolved;
#else
   return realpath(path, NULL);
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_Mkdtemp
 *
 * Create a unique temporary directory from a template ending in XXXXXX.
 *-----------------------------------------------------------------------------*/

char *
hypredrv_Mkdtemp(char *path_template)
{
   if (!path_template)
   {
      return NULL;
   }

#ifdef _WIN32
   size_t length = strlen(path_template);
   if (length < 6 || strcmp(path_template + length - 6, "XXXXXX") != 0)
   {
      return NULL;
   }
   if (_mktemp_s(path_template, length + 1) != 0 || _mkdir(path_template) != 0)
   {
      return NULL;
   }
   return path_template;
#else
   return mkdtemp(path_template);
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_IsYAMLFilename
 *
 * Returns true if string is a filename (no spaces) with a YAML extension
 * (.yaml or .yml)
 *-----------------------------------------------------------------------------*/

bool
hypredrv_IsYAMLFilename(const char *str)
{
   if (!str || *str == '\0')
   {
      return false;
   }

   /* Check for spaces - filenames should not contain spaces */
   if (strchr(str, ' ') != NULL)
   {
      return false;
   }

   const char *dot = strrchr(str, '.');
   if (!dot || dot == str)
   {
      return false;
   }

   /* Check for .yaml or .yml extension */
   const char *ext = dot + 1;
   return (strcmp(ext, "yaml") == 0 || strcmp(ext, "yml") == 0) != 0;
}

bool
hypredrv_PathIsUnderRoot(const char *path, const char *root)
{
   if (!path || !root)
   {
      return false;
   }

   size_t root_len = strlen(root);
   if (root_len == 0 ||
#ifdef _WIN32
       _strnicmp(path, root, root_len) != 0
#else
       strncmp(path, root, root_len) != 0
#endif
   )
   {
      return false;
   }

   return (path[root_len] == '\0' || path[root_len] == '/' ||
#ifdef _WIN32
           path[root_len] == '\\'
#else
           0
#endif
           ) != 0;
}
