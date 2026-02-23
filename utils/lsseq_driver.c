/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "comp.h"
#include "containers.h"
#include "error.h"
#include "lsseq.h"
#include "utils.h"

/* Temp buffer for path concatenation (two MAX_FILENAME_LENGTH paths + suffix) to satisfy -Wformat-truncation. */
#define PATH_TMP_SIZE (2 * MAX_FILENAME_LENGTH + 64)

static void
path_copy(char *dest, size_t dest_size, const char *src)
{
   size_t len = strlen(src);
   if (len >= dest_size)
   {
      len = dest_size - 1;
   }
   memcpy(dest, src, len + 1);
   dest[len] = '\0';
}

typedef struct MatrixPartRaw_struct
{
   uint64_t row_index_size;
   uint64_t value_size;
   uint64_t nnz;
   uint64_t row_lower;
   uint64_t row_upper;
   uint64_t nrows;
   void    *rows;
   void    *cols;
   void    *vals;
} MatrixPartRaw;

typedef struct RHSPartRaw_struct
{
   uint64_t value_size;
   uint64_t nrows;
   void    *vals;
} RHSPartRaw;

typedef struct DofPartRaw_struct
{
   uint64_t num_entries;
   int32_t *vals;
} DofPartRaw;

typedef struct PatternBuild_struct
{
   uint64_t         hash;
   LSSeqPatternMeta meta;
} PatternBuild;

typedef struct PackArgs_struct
{
   char      input_dirname[MAX_FILENAME_LENGTH];
   char      dirname[MAX_FILENAME_LENGTH]; /* resolved prefix used by BuildPrefix */
   char      matrix_filename[MAX_FILENAME_LENGTH];
   char      rhs_filename[MAX_FILENAME_LENGTH];
   char      dofmap_filename[MAX_FILENAME_LENGTH];
   char      timesteps_filename[MAX_FILENAME_LENGTH];
   char      output_filename[MAX_FILENAME_LENGTH];
   HYPRE_Int init_suffix;
   HYPRE_Int last_suffix;
   HYPRE_Int digits_suffix;
   comp_alg_t algo;
   int       compression_level; /* -1 = default (e.g. ZSTD 5) */
   int       init_suffix_set;
   int       last_suffix_set;
   int       digits_suffix_set;
   int       matrix_filename_set;
   int       rhs_filename_set;
   int       dofmap_filename_set;
   int       timesteps_filename_set;
   int       algo_set;
} PackArgs;

static void
PackArgsSetDefaults(PackArgs *args)
{
   if (!args)
   {
      return;
   }
   memset(args, 0, sizeof(*args));
   args->init_suffix       = 0;
   args->last_suffix      = -1;
   args->digits_suffix    = 5;
   args->algo             = COMP_ZSTD;
   args->compression_level = -1;
}

static int
StringHasSuffix(const char *text, const char *suffix)
{
   size_t n = 0, m = 0;
   if (!text || !suffix)
   {
      return 0;
   }
   n = strlen(text);
   m = strlen(suffix);
   if (m > n)
   {
      return 0;
   }
   return !strcmp(text + (n - m), suffix);
}

static int
CompressionAlgoFromString(const char *name, comp_alg_t *algo)
{
   if (!name || !algo)
   {
      return 0;
   }
   if (!strcmp(name, "none"))
   {
      *algo = COMP_NONE;
      return 1;
   }
   if (!strcmp(name, "zlib"))
   {
      *algo = COMP_ZLIB;
      return 1;
   }
   if (!strcmp(name, "zstd"))
   {
      *algo = COMP_ZSTD;
      return 1;
   }
   if (!strcmp(name, "lz4"))
   {
      *algo = COMP_LZ4;
      return 1;
   }
   if (!strcmp(name, "lz4hc"))
   {
      *algo = COMP_LZ4HC;
      return 1;
   }
   if (!strcmp(name, "blosc"))
   {
      *algo = COMP_BLOSC;
      return 1;
   }
   return 0;
}

static int
ParseArgs(int argc, char **argv, PackArgs *args)
{
   if (!args)
   {
      return 0;
   }

   PackArgsSetDefaults(args);

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         return 0;
      }
      if (i + 1 >= argc)
      {
         fprintf(stderr, "Missing value for option '%s'\n", argv[i]);
         return 0;
      }

      if (!strcmp(argv[i], "--dirname"))
      {
         snprintf(args->input_dirname, sizeof(args->input_dirname), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--matrix-filename"))
      {
         snprintf(args->matrix_filename, sizeof(args->matrix_filename), "%s", argv[++i]);
         args->matrix_filename_set = 1;
      }
      else if (!strcmp(argv[i], "--rhs-filename"))
      {
         snprintf(args->rhs_filename, sizeof(args->rhs_filename), "%s", argv[++i]);
         args->rhs_filename_set = 1;
      }
      else if (!strcmp(argv[i], "--dofmap-filename"))
      {
         snprintf(args->dofmap_filename, sizeof(args->dofmap_filename), "%s", argv[++i]);
         args->dofmap_filename_set = 1;
      }
      else if (!strcmp(argv[i], "--timesteps"))
      {
         snprintf(args->timesteps_filename, sizeof(args->timesteps_filename), "%s",
                  argv[++i]);
         args->timesteps_filename_set = 1;
      }
      else if (!strcmp(argv[i], "--output"))
      {
         snprintf(args->output_filename, sizeof(args->output_filename), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--init-suffix"))
      {
         args->init_suffix = (HYPRE_Int)strtol(argv[++i], NULL, 10);
         args->init_suffix_set = 1;
      }
      else if (!strcmp(argv[i], "--last-suffix"))
      {
         args->last_suffix = (HYPRE_Int)strtol(argv[++i], NULL, 10);
         args->last_suffix_set = 1;
      }
      else if (!strcmp(argv[i], "--digits-suffix"))
      {
         args->digits_suffix = (HYPRE_Int)strtol(argv[++i], NULL, 10);
         args->digits_suffix_set = 1;
      }
      else if (!strcmp(argv[i], "--algo"))
      {
         if (!CompressionAlgoFromString(argv[++i], &args->algo))
         {
            fprintf(stderr, "Unknown compression algorithm\n");
            return 0;
         }
         args->algo_set = 1;
      }
      else if (!strcmp(argv[i], "--compression-level"))
      {
         args->compression_level = (int)strtol(argv[++i], NULL, 10);
      }
      else
      {
         fprintf(stderr, "Unknown option '%s'\n", argv[i]);
         return 0;
      }
   }

   if (args->input_dirname[0] == '\0' || args->output_filename[0] == '\0')
   {
      return 0;
   }
   if (args->digits_suffix <= 0)
   {
      fprintf(stderr, "Invalid --digits-suffix value '%d'\n", (int)args->digits_suffix);
      return 0;
   }

   return 1;
}

static void
PrintUsage(const char *prog)
{
   fprintf(stderr,
           "Usage:\n"
           "  %s --dirname <dir> --output <out> [options]\n"
           "\n"
           "Required:\n"
           "  --dirname <dir>            Dataset root directory OR prefix to ls_XXXXX dirs.\n"
           "                             Examples:\n"
           "                               <root>/            contains ls_00000/, ls_00001/, ...\n"
           "                               <root>/ls          prefix where ls_00000/ exists\n"
           "  --output <out>             Output sequence basename (extension is added).\n"
           "\n"
           "Auto-detected defaults (from the first available ls_XXXXX directory):\n"
           "  --init-suffix <n0>         Default: 0 (or smallest available if 0 is missing)\n"
           "  --last-suffix <n1>         Default: largest available suffix\n"
           "  --matrix-filename <name>   Default: detected IJ matrix prefix (e.g., IJ.out.A)\n"
           "  --rhs-filename <name>      Default: detected IJ RHS prefix (e.g., IJ.out.b)\n"
           "  --dofmap-filename <name>   Default: detected if dofmap parts exist\n"
           "  --timesteps <file>         Default: timesteps.txt if present\n"
           "\n"
           "Options:\n"
           "  --digits-suffix <n>        Digits in ls_XXXXX directory suffix (default: 5)\n"
           "  --algo <codec>             Compression codec:\n"
           "                               none|zlib|zstd|lz4|lz4hc|blosc (default: zstd)\n"
           "  --compression-level <n>   Level 1-22 for zstd; -1 = default (5)\n"
           "\n"
           "Examples:\n"
           "  %s --dirname hypre-data_lidcavity_Re100_16x16_4x4 --output lidcavity_lsseq\n"
           "  %s --dirname data/poromech2k/np1/ls --last-suffix 24 --output poro_lsseq\n",
           prog, prog, prog);
}

static void
BuildPrefix(char *prefix, size_t prefix_size, const char *dirname, int digits_suffix, int suffix,
            const char *filename)
{
   snprintf(prefix, prefix_size, "%s_%0*d/%s", dirname, digits_suffix, suffix, filename);
}

static void
StripTrailingSlashes(char *path)
{
   if (!path)
   {
      return;
   }
   size_t n = strlen(path);
   while (n > 0 && path[n - 1] == '/')
   {
      path[--n] = '\0';
   }
}

static int
PathIsDirectory(const char *path)
{
   struct stat st;
   if (!path || path[0] == '\0')
   {
      return 0;
   }
   if (stat(path, &st) != 0)
   {
      return 0;
   }
   return S_ISDIR(st.st_mode) ? 1 : 0;
}

static int
PathIsRegularFile(const char *path)
{
   struct stat st;
   if (!path || path[0] == '\0')
   {
      return 0;
   }
   if (stat(path, &st) != 0)
   {
      return 0;
   }
   return S_ISREG(st.st_mode) ? 1 : 0;
}

static void
BuildSystemDir(char *dir, size_t dir_size, const char *dirprefix, int digits_suffix, int suffix)
{
   snprintf(dir, dir_size, "%s_%0*d", dirprefix, digits_suffix, suffix);
}

static int
AllDigitsFixedWidth(const char *text, int ndigits)
{
   if (!text || ndigits <= 0)
   {
      return 0;
   }
   for (int i = 0; i < ndigits; i++)
   {
      char c = text[i];
      if (c < '0' || c > '9')
      {
         return 0;
      }
   }
   return text[ndigits] == '\0';
}

static int
ScanSuffixDirs(const char *parent_dir, const char *base, int digits_suffix, int *min_suffix,
               int *max_suffix, int *count_suffix)
{
   DIR           *dir = NULL;
   struct dirent *ent = NULL;
   char           expected_prefix[512];
   size_t         expected_len = 0;
   int            found = 0;
   int            minv = 0, maxv = 0, countv = 0;

   if (!parent_dir || !base || digits_suffix <= 0)
   {
      return 0;
   }

   snprintf(expected_prefix, sizeof(expected_prefix), "%s_", base);
   expected_len = strlen(expected_prefix);
   dir = opendir(parent_dir);
   if (!dir)
   {
      return 0;
   }

   while ((ent = readdir(dir)) != NULL)
   {
      const char *name = ent->d_name;
      const char *digits = NULL;
      int         suffix = 0;
      char        fullpath[MAX_FILENAME_LENGTH];

      if (strncmp(name, expected_prefix, expected_len) != 0)
      {
         continue;
      }

      digits = name + expected_len;
      if (!AllDigitsFixedWidth(digits, digits_suffix))
      {
         continue;
      }

      snprintf(fullpath, sizeof(fullpath), "%s/%s", parent_dir, name);
      if (!PathIsDirectory(fullpath))
      {
         continue;
      }

      suffix = (int)strtol(digits, NULL, 10);
      if (!found)
      {
         minv  = suffix;
         maxv  = suffix;
         found = 1;
      }
      else
      {
         minv = suffix < minv ? suffix : minv;
         maxv = suffix > maxv ? suffix : maxv;
      }
      countv++;
   }

   closedir(dir);
   if (!found)
   {
      return 0;
   }

   if (min_suffix)
   {
      *min_suffix = minv;
   }
   if (max_suffix)
   {
      *max_suffix = maxv;
   }
   if (count_suffix)
   {
      *count_suffix = countv;
   }
   return 1;
}

typedef struct BaseScan_struct
{
   char base[256];
   int  count;
   int  min_suffix;
   int  max_suffix;
} BaseScan;

static int
ResolveDirPrefixAndSuffixRange(PackArgs *args, char *parent_dir, size_t parent_dir_size, char *base,
                               size_t base_size, int *min_suffix, int *max_suffix)
{
   char *cand_parent = NULL;
   char *cand_base   = NULL;

   if (!args || args->input_dirname[0] == '\0')
   {
      return 0;
   }

   StripTrailingSlashes(args->input_dirname);
   if (args->input_dirname[0] == '\0')
   {
      return 0;
   }

   /* Case 1: --dirname is already the prefix (parent contains <base>_00000 dirs). */
   SplitFilename(args->input_dirname, &cand_parent, &cand_base);
   if (cand_parent && cand_base)
   {
      int tmp_min = 0, tmp_max = 0, tmp_count = 0;
      if (ScanSuffixDirs(cand_parent, cand_base, (int)args->digits_suffix, &tmp_min, &tmp_max,
                         &tmp_count))
      {
         snprintf(args->dirname, sizeof(args->dirname), "%s", args->input_dirname);
         if (parent_dir && parent_dir_size > 0)
         {
            snprintf(parent_dir, parent_dir_size, "%s", cand_parent);
         }
         if (base && base_size > 0)
         {
            snprintf(base, base_size, "%s", cand_base);
         }
         if (min_suffix)
         {
            *min_suffix = tmp_min;
         }
         if (max_suffix)
         {
            *max_suffix = tmp_max;
         }
         free(cand_parent);
         free(cand_base);
         return 1;
      }
   }
   free(cand_parent);
   free(cand_base);
   cand_parent = NULL;
   cand_base   = NULL;

   /* Case 2: --dirname is a dataset root directory containing <base>_00000 dirs. */
   if (PathIsDirectory(args->input_dirname))
   {
      DIR           *dir = opendir(args->input_dirname);
      struct dirent *ent = NULL;
      BaseScan      *bases = NULL;
      size_t         nbases = 0, cap = 0;
      int            chosen = -1;
      int            tmp_min = 0, tmp_max = 0;

      if (!dir)
      {
         fprintf(stderr, "Could not open dataset directory '%s': %s\n", args->input_dirname,
                 strerror(errno));
         return 0;
      }

      while ((ent = readdir(dir)) != NULL)
      {
         const char *name = ent->d_name;
         const char *us   = strrchr(name, '_');
         const char *digits = NULL;
         int         suffix  = 0;
         size_t      base_len = 0;
         char        fullpath[MAX_FILENAME_LENGTH];

         if (!us || us == name)
         {
            continue;
         }
         digits = us + 1;
         if (!AllDigitsFixedWidth(digits, (int)args->digits_suffix))
         {
            continue;
         }

         {
            char path_tmp[PATH_TMP_SIZE];
            snprintf(path_tmp, sizeof(path_tmp), "%s/%s", args->input_dirname, name);
            path_copy(fullpath, sizeof(fullpath), path_tmp);
         }
         if (!PathIsDirectory(fullpath))
         {
            continue;
         }

         base_len = (size_t)(us - name);
         if (base_len == 0 || base_len >= sizeof(bases[0].base))
         {
            continue;
         }

         suffix = (int)strtol(digits, NULL, 10);

         /* Find or insert base entry */
         size_t idx = 0;
         for (; idx < nbases; idx++)
         {
            if (strncmp(bases[idx].base, name, base_len) == 0 && bases[idx].base[base_len] == '\0')
            {
               break;
            }
         }
         if (idx == nbases)
         {
            if (nbases == cap)
            {
               size_t   new_cap = cap == 0 ? 8u : (2u * cap);
               BaseScan *tmp = (BaseScan *)realloc(bases, new_cap * sizeof(*bases));
               if (!tmp)
               {
                  closedir(dir);
                  free(bases);
                  fprintf(stderr, "Allocation failure while scanning dataset directory\n");
                  return 0;
               }
               bases = tmp;
               cap   = new_cap;
            }
            memset(&bases[nbases], 0, sizeof(bases[nbases]));
            memcpy(bases[nbases].base, name, base_len);
            bases[nbases].base[base_len] = '\0';
            bases[nbases].count          = 0;
            bases[nbases].min_suffix     = suffix;
            bases[nbases].max_suffix     = suffix;
            idx = nbases++;
         }

         bases[idx].count++;
         bases[idx].min_suffix = suffix < bases[idx].min_suffix ? suffix : bases[idx].min_suffix;
         bases[idx].max_suffix = suffix > bases[idx].max_suffix ? suffix : bases[idx].max_suffix;
      }

      closedir(dir);

      for (size_t i = 0; i < nbases; i++)
      {
         if (chosen < 0)
         {
            chosen  = (int)i;
            tmp_min = bases[i].min_suffix;
            tmp_max = bases[i].max_suffix;
            continue;
         }
         if (bases[i].count > bases[(size_t)chosen].count ||
             (bases[i].count == bases[(size_t)chosen].count &&
              strcmp(bases[i].base, bases[(size_t)chosen].base) < 0))
         {
            chosen  = (int)i;
            tmp_min = bases[i].min_suffix;
            tmp_max = bases[i].max_suffix;
         }
      }

      if (chosen < 0)
      {
         free(bases);
         fprintf(stderr,
                 "Could not find any '<base>_%0*d' directories under '%s'\n",
                 (int)args->digits_suffix, 0, args->input_dirname);
         return 0;
      }

      {
         char path_tmp[PATH_TMP_SIZE];
         snprintf(path_tmp, sizeof(path_tmp), "%s/%s", args->input_dirname,
                  bases[(size_t)chosen].base);
         path_copy(args->dirname, sizeof(args->dirname), path_tmp);
      }
      StripTrailingSlashes(args->dirname);
      if (parent_dir && parent_dir_size > 0)
      {
         snprintf(parent_dir, parent_dir_size, "%s", args->input_dirname);
      }
      if (base && base_size > 0)
      {
         snprintf(base, base_size, "%s", bases[(size_t)chosen].base);
      }
      if (min_suffix)
      {
         *min_suffix = tmp_min;
      }
      if (max_suffix)
      {
         *max_suffix = tmp_max;
      }
      free(bases);
      return 1;
   }

   fprintf(stderr,
           "Could not resolve sequence directory prefix from --dirname '%s'\n"
           "  Expected either:\n"
           "    - <prefix> where <prefix>_00000/ exists, or\n"
           "    - <root> directory containing ls_00000/ (or similar) directories.\n",
           args->input_dirname);
   return 0;
}

static int
GetFileSizeULL(const char *filename, unsigned long long *size_out)
{
   struct stat st;
   if (!size_out)
   {
      return 0;
   }
   *size_out = 0;
   if (!filename || filename[0] == '\0')
   {
      return 0;
   }
   if (stat(filename, &st) != 0)
   {
      return 0;
   }
   if (!S_ISREG(st.st_mode))
   {
      return 0;
   }
   *size_out = (unsigned long long)st.st_size;
   return 1;
}

static int
ProbeMatrixPartFile(const char *filename)
{
   FILE               *fp = NULL;
   uint64_t            header[11];
   unsigned long long  file_size = 0;
   unsigned long long  expected  = 0;
   unsigned long long  entry_sz  = 0;
   uint64_t            row_index_size = 0, value_size = 0, nnz = 0;
   uint64_t            row_lower = 0, row_upper = 0;

   if (!filename || filename[0] == '\0')
   {
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }
   if (fread(header, sizeof(uint64_t), 11, fp) != 11)
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);

   row_index_size = header[1];
   value_size     = header[2];
   nnz            = header[6];
   row_lower      = header[7];
   row_upper      = header[8];

   if (!((row_index_size == 4u) || (row_index_size == 8u)) ||
       !((value_size == 4u) || (value_size == 8u)))
   {
      return 0;
   }
   if (row_upper < row_lower)
   {
      return 0;
   }

   if (!GetFileSizeULL(filename, &file_size))
   {
      return 0;
   }

   entry_sz = (unsigned long long)(2u * row_index_size + value_size);
   if (nnz > 0 && entry_sz > 0 && nnz > (uint64_t)(ULLONG_MAX / entry_sz))
   {
      return 0;
   }
   expected = (unsigned long long)(11u * sizeof(uint64_t)) +
              (unsigned long long)nnz * entry_sz;
   return (file_size == expected) ? 1 : 0;
}

static int
ProbeRHSPartFile(const char *filename)
{
   FILE               *fp = NULL;
   uint64_t            header[8];
   unsigned long long  file_size = 0;
   unsigned long long  expected  = 0;
   uint64_t            value_size = 0, nrows = 0;

   if (!filename || filename[0] == '\0')
   {
      return 0;
   }

   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }
   if (fread(header, sizeof(uint64_t), 8, fp) != 8)
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);

   value_size = header[1];
   nrows      = header[5];
   if (!((value_size == 4u) || (value_size == 8u)))
   {
      return 0;
   }

   if (!GetFileSizeULL(filename, &file_size))
   {
      return 0;
   }

   if (nrows > 0 && value_size > 0 && nrows > (uint64_t)(ULLONG_MAX / value_size))
   {
      return 0;
   }
   expected = (unsigned long long)(8u * sizeof(uint64_t)) +
              (unsigned long long)nrows * (unsigned long long)value_size;
   return (file_size == expected) ? 1 : 0;
}

static int
ExtractPrefixFromPart0Bin(const char *name, char *prefix, size_t prefix_size)
{
   const char *suffix = ".00000.bin";
   size_t      n = 0, m = 0;
   if (!name || !prefix || prefix_size == 0)
   {
      return 0;
   }
   if (!StringHasSuffix(name, suffix))
   {
      return 0;
   }
   n = strlen(name);
   m = strlen(suffix);
   if (m >= n)
   {
      return 0;
   }
   if ((n - m) >= prefix_size)
   {
      return 0;
   }
   memcpy(prefix, name, n - m);
   prefix[n - m] = '\0';
   return 1;
}

static int
ExtractPrefixFromPart0ASCII(const char *name, char *prefix, size_t prefix_size)
{
   const char *suffix = ".00000";
   size_t      n = 0, m = 0;
   if (!name || !prefix || prefix_size == 0)
   {
      return 0;
   }
   if (!StringHasSuffix(name, suffix) || StringHasSuffix(name, ".00000.bin"))
   {
      return 0;
   }
   n = strlen(name);
   m = strlen(suffix);
   if (m >= n)
   {
      return 0;
   }
   if ((n - m) >= prefix_size)
   {
      return 0;
   }
   memcpy(prefix, name, n - m);
   prefix[n - m] = '\0';
   return 1;
}

static int
StringCompare(const void *a, const void *b)
{
   const char *const *sa = (const char *const *)a;
   const char *const *sb = (const char *const *)b;
   return strcmp(*sa, *sb);
}

/* Forward declarations for helpers used during auto-detection. */
static int  ReadDofPart(const char *prefix, int part_id, DofPartRaw *raw);
static void DofPartRawDestroy(DofPartRaw *part);

static int
AutoDetectFilenames(PackArgs *args, const char *system_dir)
{
   DIR           *dir = NULL;
   struct dirent *ent = NULL;
   char         **bin_prefixes = NULL;
   size_t         nbin = 0, cap_bin = 0;
   char         **ascii_prefixes = NULL;
   size_t         nascii = 0, cap_ascii = 0;
   int            have_matrix = (args && args->matrix_filename[0] != '\0');
   int            have_rhs    = (args && args->rhs_filename[0] != '\0');

   if (!args || !system_dir || system_dir[0] == '\0')
   {
      return 0;
   }

   dir = opendir(system_dir);
   if (!dir)
   {
      fprintf(stderr, "Could not open system directory '%s': %s\n", system_dir, strerror(errno));
      return 0;
   }

   while ((ent = readdir(dir)) != NULL)
   {
      char prefix_buf[256];
      if (ExtractPrefixFromPart0Bin(ent->d_name, prefix_buf, sizeof(prefix_buf)))
      {
         char *s = strdup(prefix_buf);
         if (s)
         {
            if (nbin == cap_bin)
            {
               size_t new_cap = cap_bin == 0 ? 8u : (2u * cap_bin);
               char **tmp = (char **)realloc(bin_prefixes, new_cap * sizeof(*bin_prefixes));
               if (!tmp)
               {
                  free(s);
                  break;
               }
               bin_prefixes = tmp;
               cap_bin      = new_cap;
            }
            bin_prefixes[nbin++] = s;
         }
      }
      else if (ExtractPrefixFromPart0ASCII(ent->d_name, prefix_buf, sizeof(prefix_buf)))
      {
         char *s = strdup(prefix_buf);
         if (s)
         {
            if (nascii == cap_ascii)
            {
               size_t new_cap = cap_ascii == 0 ? 4u : (2u * cap_ascii);
               char **tmp = (char **)realloc(ascii_prefixes, new_cap * sizeof(*ascii_prefixes));
               if (!tmp)
               {
                  free(s);
                  break;
               }
               ascii_prefixes = tmp;
               cap_ascii      = new_cap;
            }
            ascii_prefixes[nascii++] = s;
         }
      }
   }
   closedir(dir);

   if (nbin > 1)
   {
      qsort(bin_prefixes, nbin, sizeof(*bin_prefixes), StringCompare);
   }
   if (nascii > 1)
   {
      qsort(ascii_prefixes, nascii, sizeof(*ascii_prefixes), StringCompare);
   }

   /* Deduplicate identical prefix strings (readdir can include duplicates on some FS). */
   size_t write_idx = 0;
   for (size_t i = 0; i < nbin; i++)
   {
      if (write_idx == 0 || strcmp(bin_prefixes[i], bin_prefixes[write_idx - 1]) != 0)
      {
         bin_prefixes[write_idx++] = bin_prefixes[i];
      }
      else
      {
         free(bin_prefixes[i]);
      }
   }
   nbin = write_idx;

   write_idx = 0;
   for (size_t i = 0; i < nascii; i++)
   {
      if (write_idx == 0 || strcmp(ascii_prefixes[i], ascii_prefixes[write_idx - 1]) != 0)
      {
         ascii_prefixes[write_idx++] = ascii_prefixes[i];
      }
      else
      {
         free(ascii_prefixes[i]);
      }
   }
   nascii = write_idx;

   for (size_t i = 0; i < nbin; i++)
   {
      char part0_filename[MAX_FILENAME_LENGTH];
      snprintf(part0_filename, sizeof(part0_filename), "%s/%s.00000.bin", system_dir,
               bin_prefixes[i]);

      if (!have_matrix && ProbeMatrixPartFile(part0_filename))
      {
         snprintf(args->matrix_filename, sizeof(args->matrix_filename), "%s", bin_prefixes[i]);
         have_matrix = 1;
      }
      else if (!have_rhs && ProbeRHSPartFile(part0_filename))
      {
         snprintf(args->rhs_filename, sizeof(args->rhs_filename), "%s", bin_prefixes[i]);
         have_rhs = 1;
      }
   }

   if (args->dofmap_filename[0] == '\0' && nascii > 0)
   {
      for (size_t i = 0; i < nascii; i++)
      {
         char      dof_prefix[MAX_FILENAME_LENGTH];
         DofPartRaw draw;
         memset(&draw, 0, sizeof(draw));

         snprintf(dof_prefix, sizeof(dof_prefix), "%s/%s", system_dir, ascii_prefixes[i]);
         if (ReadDofPart(dof_prefix, 0, &draw))
         {
            snprintf(args->dofmap_filename, sizeof(args->dofmap_filename), "%s", ascii_prefixes[i]);
            DofPartRawDestroy(&draw);
            break;
         }
         DofPartRawDestroy(&draw);
      }
   }

   for (size_t i = 0; i < nbin; i++)
   {
      free(bin_prefixes[i]);
   }
   for (size_t i = 0; i < nascii; i++)
   {
      free(ascii_prefixes[i]);
   }
   free(bin_prefixes);
   free(ascii_prefixes);

   return (args->matrix_filename[0] != '\0' && args->rhs_filename[0] != '\0');
}

static void
MatrixPartRawDestroy(MatrixPartRaw *part)
{
   if (!part)
   {
      return;
   }
   free(part->rows);
   free(part->cols);
   free(part->vals);
   memset(part, 0, sizeof(*part));
}

static void
RHSPartRawDestroy(RHSPartRaw *part)
{
   if (!part)
   {
      return;
   }
   free(part->vals);
   memset(part, 0, sizeof(*part));
}

static void
DofPartRawDestroy(DofPartRaw *part)
{
   if (!part)
   {
      return;
   }
   free(part->vals);
   memset(part, 0, sizeof(*part));
}

static int
ReadMatrixPart(const char *prefix, int part_id, MatrixPartRaw *raw)
{
   FILE    *fp = NULL;
   uint64_t header[11];
   char     filename[MAX_FILENAME_LENGTH];
   size_t   row_bytes = 0, val_bytes = 0, nnz = 0;

   if (!prefix || !raw)
   {
      return 0;
   }
   memset(raw, 0, sizeof(*raw));

   snprintf(filename, sizeof(filename), "%s.%05d.bin", prefix, part_id);
   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }

   if (fread(header, sizeof(uint64_t), 11, fp) != 11)
   {
      fclose(fp);
      return 0;
   }

   raw->row_index_size = header[1];
   raw->value_size     = header[2];
   raw->nrows          = header[5];
   raw->nnz            = header[6];
   raw->row_lower      = header[7];
   raw->row_upper      = header[8];

   nnz = (size_t)raw->nnz;
   row_bytes = (size_t)raw->row_index_size;
   val_bytes = (size_t)raw->value_size;

   if (nnz > 0)
   {
      raw->rows = malloc(nnz * row_bytes);
      raw->cols = malloc(nnz * row_bytes);
      raw->vals = malloc(nnz * val_bytes);
      if (!raw->rows || !raw->cols || !raw->vals)
      {
         fclose(fp);
         MatrixPartRawDestroy(raw);
         return 0;
      }

      if (fread(raw->rows, row_bytes, nnz, fp) != nnz ||
          fread(raw->cols, row_bytes, nnz, fp) != nnz ||
          fread(raw->vals, val_bytes, nnz, fp) != nnz)
      {
         fclose(fp);
         MatrixPartRawDestroy(raw);
         return 0;
      }
   }

   fclose(fp);
   return 1;
}

static int
ReadRHSPart(const char *prefix, int part_id, RHSPartRaw *raw)
{
   FILE    *fp = NULL;
   uint64_t header[8];
   char     filename[MAX_FILENAME_LENGTH];

   if (!prefix || !raw)
   {
      return 0;
   }
   memset(raw, 0, sizeof(*raw));

   snprintf(filename, sizeof(filename), "%s.%05d.bin", prefix, part_id);
   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }

   if (fread(header, sizeof(uint64_t), 8, fp) != 8)
   {
      fclose(fp);
      return 0;
   }

   raw->value_size = header[1];
   raw->nrows      = header[5];

   if (raw->nrows > 0)
   {
      raw->vals = malloc((size_t)raw->nrows * (size_t)raw->value_size);
      if (!raw->vals)
      {
         fclose(fp);
         return 0;
      }
      if (fread(raw->vals, (size_t)raw->value_size, (size_t)raw->nrows, fp) !=
          (size_t)raw->nrows)
      {
         fclose(fp);
         RHSPartRawDestroy(raw);
         return 0;
      }
   }

   fclose(fp);
   return 1;
}

static int
ReadDofPart(const char *prefix, int part_id, DofPartRaw *raw)
{
   char  filename[MAX_FILENAME_LENGTH];
   FILE *fp = NULL;
   size_t n = 0;

   if (!prefix || !raw)
   {
      return 0;
   }
   memset(raw, 0, sizeof(*raw));

   snprintf(filename, sizeof(filename), "%s.%05d", prefix, part_id);
   fp = fopen(filename, "r");
   if (fp)
   {
      if (fscanf(fp, "%zu", &n) != 1)
      {
         fclose(fp);
         return 0;
      }
      raw->num_entries = (uint64_t)n;
      if (n > 0)
      {
         raw->vals = (int32_t *)calloc(n, sizeof(int32_t));
         if (!raw->vals)
         {
            fclose(fp);
            return 0;
         }
         for (size_t i = 0; i < n; i++)
         {
            int v = 0;
            if (fscanf(fp, "%d", &v) != 1)
            {
               fclose(fp);
               DofPartRawDestroy(raw);
               return 0;
            }
            raw->vals[i] = (int32_t)v;
         }
      }
      fclose(fp);
      return 1;
   }

   snprintf(filename, sizeof(filename), "%s.%05d.bin", prefix, part_id);
   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }
   if (fread(&n, sizeof(size_t), 1, fp) != 1)
   {
      fclose(fp);
      return 0;
   }
   raw->num_entries = (uint64_t)n;
   if (n > 0)
   {
      raw->vals = (int32_t *)calloc(n, sizeof(int32_t));
      if (!raw->vals)
      {
         fclose(fp);
         return 0;
      }
      if (fread(raw->vals, sizeof(int32_t), n, fp) != n)
      {
         fclose(fp);
         DofPartRawDestroy(raw);
         return 0;
      }
   }
   fclose(fp);
   return 1;
}

static uint64_t
FNV1a64(const void *data, size_t nbytes, uint64_t hash)
{
   const unsigned char *bytes = (const unsigned char *)data;
   if (!bytes)
   {
      return hash;
   }
   for (size_t i = 0; i < nbytes; i++)
   {
      hash ^= (uint64_t)bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static uint64_t
PatternHash(const MatrixPartRaw *raw, uint32_t part_id)
{
   uint64_t hash = UINT64_C(1469598103934665603);
   if (!raw)
   {
      return hash;
   }
   hash = FNV1a64(&part_id, sizeof(part_id), hash);
   hash = FNV1a64(&raw->nnz, sizeof(raw->nnz), hash);
   hash = FNV1a64(&raw->row_index_size, sizeof(raw->row_index_size), hash);
   hash = FNV1a64(&raw->row_lower, sizeof(raw->row_lower), hash);
   hash = FNV1a64(&raw->row_upper, sizeof(raw->row_upper), hash);
   hash = FNV1a64(raw->rows, (size_t)raw->nnz * (size_t)raw->row_index_size, hash);
   hash = FNV1a64(raw->cols, (size_t)raw->nnz * (size_t)raw->row_index_size, hash);
   return hash;
}

static int
PackWriteBlob(FILE *blob_fp, comp_alg_t algo, int compression_level, const void *data,
              size_t size, uint64_t *cursor, uint64_t *offset, uint64_t *blob_size)
{
   void  *comp_data = NULL;
   size_t comp_size = 0;

   if (!blob_fp || !cursor || !offset || !blob_size)
   {
      return 0;
   }

   if (size == 0)
   {
      *offset    = 0;
      *blob_size = 0;
      return 1;
   }

   ErrorCodeResetAll();
   ErrorMsgClear();
   hypredrv_compress(algo, size, data, &comp_size, &comp_data, compression_level);
   if (ErrorCodeActive() || !comp_data)
   {
      if (ErrorCodeActive())
      {
         ErrorMsgPrint();
      }
      return 0;
   }
   if (fwrite(comp_data, 1, comp_size, blob_fp) != comp_size)
   {
      free(comp_data);
      return 0;
   }

   *offset    = *cursor;
   *blob_size = (uint64_t)comp_size;
   *cursor += (uint64_t)comp_size;
   free(comp_data);
   return 1;
}

static void
FormatBytesHuman(unsigned long long bytes, char *buf, size_t buf_size)
{
   if (!buf || buf_size == 0)
   {
      return;
   }
   if (bytes >= 1024ULL * 1024ULL * 1024ULL)
   {
      snprintf(buf, buf_size, "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
   }
   else if (bytes >= 1024ULL * 1024ULL)
   {
      snprintf(buf, buf_size, "%.2f MB", (double)bytes / (1024.0 * 1024.0));
   }
   else if (bytes >= 1024ULL)
   {
      snprintf(buf, buf_size, "%.2f KB", (double)bytes / 1024.0);
   }
   else
   {
      snprintf(buf, buf_size, "%llu B", (unsigned long long)bytes);
   }
}

static int
LoadTimesteps(const char *filename, LSSeqTimestepEntry **entries_ptr, uint32_t *count_ptr)
{
   FILE               *fp = NULL;
   LSSeqTimestepEntry *entries = NULL;
   uint32_t            count = 0, cap = 0;
   char                line[512];

   if (!entries_ptr || !count_ptr || !filename || filename[0] == '\0')
   {
      return 0;
   }

   fp = fopen(filename, "r");
   if (!fp)
   {
      return 0;
   }

   if (!fgets(line, sizeof(line), fp))
   {
      fclose(fp);
      return 0;
   }

   while (fgets(line, sizeof(line), fp))
   {
      int timestep = 0;
      int ls_start = 0;
      if (sscanf(line, "%d %d", &timestep, &ls_start) != 2)
      {
         continue;
      }

      if (count == cap)
      {
         uint32_t new_cap = cap == 0 ? 16u : (2u * cap);
         LSSeqTimestepEntry *tmp =
            (LSSeqTimestepEntry *)realloc(entries, (size_t)new_cap * sizeof(*entries));
         if (!tmp)
         {
            fclose(fp);
            free(entries);
            return 0;
         }
         entries = tmp;
         cap     = new_cap;
      }
      entries[count].timestep = timestep;
      entries[count].ls_start = ls_start;
      count++;
   }

   fclose(fp);
   *entries_ptr = entries;
   *count_ptr   = count;
   return 1;
}

static int
HelpRequested(int argc, char **argv)
{
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         return 1;
      }
   }
   return 0;
}

static void
ComputePartRange(int num_parts, int nprocs, int rank, int *start_part, int *num_local_parts)
{
   int base = 0, rem = 0;
   int count = 0, start = 0;

   if (start_part)
   {
      *start_part = 0;
   }
   if (num_local_parts)
   {
      *num_local_parts = 0;
   }
   if (num_parts <= 0 || nprocs <= 0 || rank < 0 || rank >= nprocs)
   {
      return;
   }

   base  = num_parts / nprocs;
   rem   = num_parts % nprocs;
   count = base + ((rank < rem) ? 1 : 0);
   start = (rank < rem) ? (rank * (base + 1)) : (rem * (base + 1) + (rank - rem) * base);

   if (start_part)
   {
      *start_part = start;
   }
   if (num_local_parts)
   {
      *num_local_parts = count;
   }
}

/* Append raw bytes to a growable buffer (for v2 batched part buffers). */
static int
BufAppendRaw(void **buf_ptr, size_t *len_ptr, size_t *cap_ptr, const void *data, size_t data_len)
{
   void  *buf = NULL;
   size_t len = 0, cap = 0;
   size_t need = 0;

   if (!buf_ptr || !len_ptr || !cap_ptr || !data_len)
   {
      return (data_len == 0) ? 1 : 0;
   }
   buf = *buf_ptr;
   len = *len_ptr;
   cap = *cap_ptr;
   need = len + data_len;
   if (need > cap || !buf)
   {
      size_t new_cap = (cap > 0 && cap <= SIZE_MAX / 2) ? (2u * cap) : 4096u;
      while (new_cap < need && new_cap <= SIZE_MAX / 2)
      {
         new_cap *= 2u;
      }
      if (new_cap < need)
      {
         new_cap = need;
      }
      {
         void *tmp = realloc(buf, new_cap);
         if (!tmp)
         {
            return 0;
         }
         buf = tmp;
         cap = new_cap;
      }
   }
   memcpy((char *)buf + len, data, data_len);
   len += data_len;
   *buf_ptr = buf;
   *len_ptr = len;
   *cap_ptr = cap;
   return 1;
}

static int
BufAppendf(char **buf_ptr, size_t *len_ptr, size_t *cap_ptr, const char *fmt, ...)
{
   char   *buf = NULL;
   size_t  len = 0, cap = 0;
   int     needed = 0;
   va_list ap;

   if (!buf_ptr || !len_ptr || !cap_ptr || !fmt)
   {
      return 0;
   }
   buf = *buf_ptr;
   len = *len_ptr;
   cap = *cap_ptr;

   if (!buf)
   {
      cap = 1024u;
      buf = (char *)malloc(cap);
      if (!buf)
      {
         return 0;
      }
      buf[0] = '\0';
      len    = 0;
   }

   while (1)
   {
      if (len >= cap)
      {
         size_t new_cap = cap < (SIZE_MAX / 2u) ? (2u * cap) : 0u;
         char  *tmp     = new_cap ? (char *)realloc(buf, new_cap) : NULL;
         if (!tmp)
         {
            free(buf);
            *buf_ptr = NULL;
            *len_ptr = 0;
            *cap_ptr = 0;
            return 0;
         }
         buf = tmp;
         cap = new_cap;
      }

      va_start(ap, fmt);
      needed = vsnprintf(buf + len, cap - len, fmt, ap);
      va_end(ap);

      if (needed < 0)
      {
         free(buf);
         *buf_ptr = NULL;
         *len_ptr = 0;
         *cap_ptr = 0;
         return 0;
      }
      if ((size_t)needed < (cap - len))
      {
         len += (size_t)needed;
         break;
      }

      /* Grow and retry. */
      size_t target = len + (size_t)needed + 1u;
      size_t new_cap = cap;
      while (new_cap < target)
      {
         new_cap = new_cap < (SIZE_MAX / 2u) ? (2u * new_cap) : target;
         if (new_cap == target)
         {
            break;
         }
      }
      char *tmp = (char *)realloc(buf, new_cap);
      if (!tmp)
      {
         free(buf);
         *buf_ptr = NULL;
         *len_ptr = 0;
         *cap_ptr = 0;
         return 0;
      }
      buf = tmp;
      cap = new_cap;
   }

   *buf_ptr = buf;
   *len_ptr = len;
   *cap_ptr = cap;
   return 1;
}

static void
FormatUTCTimeISO8601(char *dst, size_t dst_size)
{
   time_t    now = 0;
   struct tm tm_utc;

   if (!dst || dst_size == 0)
   {
      return;
   }

   now = time(NULL);
   if (gmtime_r(&now, &tm_utc) == NULL)
   {
      dst[0] = '\0';
      return;
   }
   strftime(dst, dst_size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

static char *
BuildLSSeqInfoPayload(const PackArgs *args, const char *output_filename, int nprocs, int num_systems,
                      int num_parts, unsigned int num_patterns, uint32_t num_timesteps,
                      unsigned long long blob_bytes, unsigned long long raw_blob_bytes,
                      size_t *payload_size_out)
{
   char   *buf = NULL;
   size_t  len = 0, cap = 0;
   char    created[64];

   if (payload_size_out)
   {
      *payload_size_out = 0;
   }

   if (!args || !output_filename)
   {
      return NULL;
   }

   created[0] = '\0';
   FormatUTCTimeISO8601(created, sizeof(created));

   if (!BufAppendf(&buf, &len, &cap, "lsseq_info_version=1\n") ||
       !BufAppendf(&buf, &len, &cap, "created_utc=%s\n", created[0] ? created : "unknown") ||
       !BufAppendf(&buf, &len, &cap, "codec=%s\n", hypredrv_compression_get_name(args->algo)) ||
       !BufAppendf(&buf, &len, &cap, "output=%s\n", output_filename) ||
       !BufAppendf(&buf, &len, &cap, "dirname_input=%s\n", args->input_dirname) ||
       !BufAppendf(&buf, &len, &cap, "dirname_resolved=%s\n", args->dirname) ||
       !BufAppendf(&buf, &len, &cap, "digits_suffix=%d\n", (int)args->digits_suffix) ||
       !BufAppendf(&buf, &len, &cap, "init_suffix=%d\n", (int)args->init_suffix) ||
       !BufAppendf(&buf, &len, &cap, "last_suffix=%d\n", (int)args->last_suffix) ||
       !BufAppendf(&buf, &len, &cap, "matrix_filename=%s\n", args->matrix_filename) ||
       !BufAppendf(&buf, &len, &cap, "rhs_filename=%s\n", args->rhs_filename) ||
       !BufAppendf(&buf, &len, &cap, "dofmap_filename=%s\n",
                   args->dofmap_filename[0] ? args->dofmap_filename : "") ||
       !BufAppendf(&buf, &len, &cap, "timesteps=%s\n",
                   args->timesteps_filename[0] ? args->timesteps_filename : "") ||
       !BufAppendf(&buf, &len, &cap, "mpi_ranks=%d\n", nprocs) ||
       !BufAppendf(&buf, &len, &cap, "num_systems=%d\n", num_systems) ||
       !BufAppendf(&buf, &len, &cap, "num_parts=%d\n", num_parts) ||
       !BufAppendf(&buf, &len, &cap, "num_patterns=%u\n", num_patterns) ||
       !BufAppendf(&buf, &len, &cap, "num_timesteps=%u\n", (unsigned int)num_timesteps) ||
       !BufAppendf(&buf, &len, &cap, "blob_bytes=%llu\n", blob_bytes) ||
       !BufAppendf(&buf, &len, &cap, "raw_blob_bytes=%llu\n", raw_blob_bytes) ||
       !BufAppendf(&buf, &len, &cap, "endian_tag=0x01020304\n") ||
       !BufAppendf(&buf, &len, &cap, "sizeof_LSSeqHeader=%zu\n", sizeof(LSSeqHeader)) ||
       !BufAppendf(&buf, &len, &cap, "sizeof_LSSeqInfoHeader=%zu\n", sizeof(LSSeqInfoHeader)))
   {
      free(buf);
      return NULL;
   }

#ifdef HYPREDRV_RELEASE_VERSION
   BufAppendf(&buf, &len, &cap, "hypredrive_release_version=%s\n", HYPREDRV_RELEASE_VERSION);
#endif
#ifdef HYPREDRV_DEVELOP_STRING
   BufAppendf(&buf, &len, &cap, "hypredrive_develop_string=%s\n", HYPREDRV_DEVELOP_STRING);
#endif
#ifdef HYPREDRV_BRANCH_NAME
   BufAppendf(&buf, &len, &cap, "hypredrive_branch_name=%s\n", HYPREDRV_BRANCH_NAME);
#endif
#ifdef HYPREDRV_GIT_SHA
   BufAppendf(&buf, &len, &cap, "hypredrive_git_sha=%s\n", HYPREDRV_GIT_SHA);
#endif
#ifdef HYPREDRV_HYPRE_RELEASE_NUMBER
   BufAppendf(&buf, &len, &cap, "hypre_release_number=%d\n", (int)HYPREDRV_HYPRE_RELEASE_NUMBER);
#endif
#ifdef HYPREDRV_HYPRE_DEVELOP_NUMBER
   BufAppendf(&buf, &len, &cap, "hypre_develop_number=%d\n", (int)HYPREDRV_HYPRE_DEVELOP_NUMBER);
#endif

   if (payload_size_out)
   {
      *payload_size_out = len;
   }
   return buf;
}

typedef enum ToolMode_enum
{
   TOOL_MODE_PACK = 0,
   TOOL_MODE_UNPACK,
   TOOL_MODE_METADATA
} ToolMode;

typedef struct UnpackArgs_struct
{
   char input_filename[MAX_FILENAME_LENGTH];
   char output_dir[MAX_FILENAME_LENGTH];
   char prefix[64];
   int  digits_suffix;
   int  digits_suffix_set;
} UnpackArgs;

typedef struct MetadataArgs_struct
{
   char input_filename[MAX_FILENAME_LENGTH];
   int  show_manifest;
} MetadataArgs;

typedef struct SeqPackedData_struct
{
   LSSeqHeader          header;
   LSSeqInfoHeader      info_header;
   char                *info_payload;
   size_t               info_payload_size;
   LSSeqPartMeta       *parts;
   LSSeqPatternMeta    *patterns;
   LSSeqSystemPartMeta *sys_parts;
   LSSeqTimestepEntry  *timesteps;
} SeqPackedData;

static void
SeqPackedDataDestroy(SeqPackedData *seq)
{
   if (!seq)
   {
      return;
   }
   free(seq->info_payload);
   free(seq->parts);
   free(seq->patterns);
   free(seq->sys_parts);
   free(seq->timesteps);
   memset(seq, 0, sizeof(*seq));
}

static int
SeqReadAt(FILE *fp, uint64_t offset, void *buffer, size_t nbytes)
{
   if (!fp || !buffer)
   {
      return 0;
   }
   if (fseeko(fp, (off_t)offset, SEEK_SET) != 0)
   {
      return 0;
   }
   if (nbytes > 0 && fread(buffer, 1, nbytes, fp) != nbytes)
   {
      return 0;
   }
   return 1;
}

static int
LoadPackedSequence(const char *filename, SeqPackedData *seq, int verify_blob_hash)
{
   FILE          *fp = NULL;
   uint64_t       payload_off = (uint64_t)sizeof(LSSeqHeader) + (uint64_t)sizeof(LSSeqInfoHeader);
   uint64_t       hash = UINT64_C(1469598103934665603);
   uint64_t       blob_hash = UINT64_C(1469598103934665603);
   uint8_t        io_buf[1 << 20];
   size_t         n_sys_parts = 0;
   struct stat    st;
   unsigned long long expected_blob_bytes = 0;

   if (!filename || !seq)
   {
      return 0;
   }
   memset(seq, 0, sizeof(*seq));

   fp = fopen(filename, "rb");
   if (!fp)
   {
      return 0;
   }

   if (!SeqReadAt(fp, 0, &seq->header, sizeof(seq->header)))
   {
      fclose(fp);
      return 0;
   }
   if (seq->header.magic != LSSEQ_MAGIC || seq->header.version != LSSEQ_VERSION)
   {
      fclose(fp);
      return 0;
   }
   if (!(seq->header.flags & LSSEQ_FLAG_HAS_INFO))
   {
      fclose(fp);
      return 0;
   }
   if (!SeqReadAt(fp, (uint64_t)sizeof(LSSeqHeader), &seq->info_header, sizeof(seq->info_header)))
   {
      fclose(fp);
      return 0;
   }
   if (seq->info_header.magic != LSSEQ_INFO_MAGIC ||
       seq->info_header.version != LSSEQ_INFO_VERSION ||
       seq->info_header.endian_tag != UINT32_C(0x01020304))
   {
      fclose(fp);
      return 0;
   }
   if (seq->info_header.payload_size > (uint64_t)SIZE_MAX - 1u)
   {
      fclose(fp);
      return 0;
   }

   seq->info_payload_size = (size_t)seq->info_header.payload_size;
   seq->info_payload = (char *)malloc(seq->info_payload_size + 1u);
   if (!seq->info_payload)
   {
      fclose(fp);
      return 0;
   }
   if (seq->info_payload_size > 0 &&
       !SeqReadAt(fp, payload_off, seq->info_payload, seq->info_payload_size))
   {
      fclose(fp);
      SeqPackedDataDestroy(seq);
      return 0;
   }
   seq->info_payload[seq->info_payload_size] = '\0';

   hash = FNV1a64(seq->info_payload, seq->info_payload_size, hash);
   if (hash != seq->info_header.payload_hash_fnv1a64)
   {
      fclose(fp);
      SeqPackedDataDestroy(seq);
      return 0;
   }

   seq->parts = (LSSeqPartMeta *)calloc((size_t)seq->header.num_parts, sizeof(LSSeqPartMeta));
   seq->patterns =
      (LSSeqPatternMeta *)calloc((size_t)seq->header.num_patterns, sizeof(LSSeqPatternMeta));
   n_sys_parts = (size_t)seq->header.num_systems * (size_t)seq->header.num_parts;
   seq->sys_parts = (LSSeqSystemPartMeta *)calloc(n_sys_parts, sizeof(LSSeqSystemPartMeta));
   if (!seq->parts || !seq->patterns || !seq->sys_parts)
   {
      fclose(fp);
      SeqPackedDataDestroy(seq);
      return 0;
   }

   if (!SeqReadAt(fp, seq->header.offset_part_meta, seq->parts,
                  (size_t)seq->header.num_parts * sizeof(LSSeqPartMeta)) ||
       !SeqReadAt(fp, seq->header.offset_pattern_meta, seq->patterns,
                  (size_t)seq->header.num_patterns * sizeof(LSSeqPatternMeta)) ||
       !SeqReadAt(fp, seq->header.offset_sys_part_meta, seq->sys_parts,
                  n_sys_parts * sizeof(LSSeqSystemPartMeta)))
   {
      fclose(fp);
      SeqPackedDataDestroy(seq);
      return 0;
   }

   if ((seq->header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq->header.num_timesteps > 0)
   {
      seq->timesteps = (LSSeqTimestepEntry *)calloc((size_t)seq->header.num_timesteps,
                                                    sizeof(LSSeqTimestepEntry));
      if (!seq->timesteps ||
          !SeqReadAt(fp, seq->header.offset_timestep_meta, seq->timesteps,
                     (size_t)seq->header.num_timesteps * sizeof(LSSeqTimestepEntry)))
      {
         fclose(fp);
         SeqPackedDataDestroy(seq);
         return 0;
      }
   }

   if (stat(filename, &st) == 0 && st.st_size >= 0 &&
       (uint64_t)st.st_size >= seq->header.offset_blob_data)
   {
      expected_blob_bytes = (unsigned long long)((uint64_t)st.st_size - seq->header.offset_blob_data);
      if (seq->info_header.blob_bytes != (uint64_t)expected_blob_bytes)
      {
         fclose(fp);
         SeqPackedDataDestroy(seq);
         return 0;
      }
   }

   if (verify_blob_hash)
   {
      uint64_t remaining = seq->info_header.blob_bytes;
      if (fseeko(fp, (off_t)seq->header.offset_blob_data, SEEK_SET) != 0)
      {
         fclose(fp);
         SeqPackedDataDestroy(seq);
         return 0;
      }
      while (remaining > 0)
      {
         size_t chunk = remaining > (uint64_t)sizeof(io_buf) ? sizeof(io_buf) : (size_t)remaining;
         if (fread(io_buf, 1, chunk, fp) != chunk)
         {
            fclose(fp);
            SeqPackedDataDestroy(seq);
            return 0;
         }
         blob_hash = FNV1a64(io_buf, chunk, blob_hash);
         remaining -= (uint64_t)chunk;
      }
      if (blob_hash != seq->info_header.blob_hash_fnv1a64)
      {
         fclose(fp);
         SeqPackedDataDestroy(seq);
         return 0;
      }
   }

   fclose(fp);
   return 1;
}

static int
DecodeBlob(FILE *fp, comp_alg_t codec, uint64_t offset, uint64_t blob_size, size_t expected_size,
           void **decoded_ptr, size_t *decoded_size_ptr)
{
   void  *blob = NULL;
   void  *decoded = NULL;
   size_t decoded_size = 0;

   if (!fp || !decoded_ptr || !decoded_size_ptr)
   {
      return 0;
   }
   *decoded_ptr      = NULL;
   *decoded_size_ptr = 0;

   if (blob_size == 0)
   {
      if (expected_size == 0)
      {
         return 1;
      }
      return 0;
   }
   if (blob_size > (uint64_t)SIZE_MAX)
   {
      return 0;
   }

   blob = malloc((size_t)blob_size);
   if (!blob)
   {
      return 0;
   }
   if (!SeqReadAt(fp, offset, blob, (size_t)blob_size))
   {
      free(blob);
      return 0;
   }

   if (codec == COMP_NONE)
   {
      decoded = malloc((size_t)blob_size);
      if (!decoded)
      {
         free(blob);
         return 0;
      }
      memcpy(decoded, blob, (size_t)blob_size);
      decoded_size = (size_t)blob_size;
   }
   else
   {
      ErrorCodeResetAll();
      ErrorMsgClear();
      hypredrv_decompress(codec, (size_t)blob_size, blob, &decoded_size, &decoded);
      if (ErrorCodeActive() || !decoded)
      {
         free(blob);
         return 0;
      }
   }
   free(blob);

   if (expected_size > 0 && decoded_size != expected_size)
   {
      free(decoded);
      return 0;
   }
   *decoded_ptr      = decoded;
   *decoded_size_ptr = decoded_size;
   return 1;
}

static int
WriteMatrixPartBinary(const char *filename, const LSSeqPartMeta *part, const LSSeqPatternMeta *pattern,
                      const void *rows, const void *cols, const void *vals)
{
   FILE    *fp = NULL;
   uint64_t header[11] = {0};
   size_t   nnz = 0;

   if (!filename || !part || !pattern)
   {
      return 0;
   }

   fp = fopen(filename, "wb");
   if (!fp)
   {
      return 0;
   }
   header[1] = part->row_index_size;
   header[2] = part->value_size;
   header[5] = part->row_upper - part->row_lower + 1;
   header[6] = pattern->nnz;
   header[7] = part->row_lower;
   header[8] = part->row_upper;
   if (fwrite(header, sizeof(uint64_t), 11, fp) != 11)
   {
      fclose(fp);
      return 0;
   }

   nnz = (size_t)pattern->nnz;
   if (nnz > 0 &&
       ((rows && fwrite(rows, (size_t)part->row_index_size, nnz, fp) != nnz) ||
        (cols && fwrite(cols, (size_t)part->row_index_size, nnz, fp) != nnz) ||
        (vals && fwrite(vals, (size_t)part->value_size, nnz, fp) != nnz)))
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);
   return 1;
}

static int
WriteRHSPartBinary(const char *filename, const LSSeqPartMeta *part, const void *vals)
{
   FILE    *fp = NULL;
   uint64_t header[8] = {0};
   size_t   nrows = 0;

   if (!filename || !part)
   {
      return 0;
   }

   fp = fopen(filename, "wb");
   if (!fp)
   {
      return 0;
   }
   header[1] = part->value_size;
   header[5] = part->nrows;
   if (fwrite(header, sizeof(uint64_t), 8, fp) != 8)
   {
      fclose(fp);
      return 0;
   }
   nrows = (size_t)part->nrows;
   if (nrows > 0 && vals && fwrite(vals, (size_t)part->value_size, nrows, fp) != nrows)
   {
      fclose(fp);
      return 0;
   }
   fclose(fp);
   return 1;
}

static int
WriteDofPartASCII(const char *filename, const int32_t *vals, uint64_t nentries)
{
   FILE *fp = NULL;

   if (!filename)
   {
      return 0;
   }
   fp = fopen(filename, "w");
   if (!fp)
   {
      return 0;
   }
   fprintf(fp, "%llu\n", (unsigned long long)nentries);
   for (uint64_t i = 0; i < nentries; i++)
   {
      fprintf(fp, "%d\n", vals ? (int)vals[i] : 0);
   }
   fclose(fp);
   return 1;
}

static int
ManifestFindValue(const char *payload, const char *key, char *value, size_t value_size)
{
   const char *cur = payload;
   size_t      key_len = 0;
   if (!payload || !key || !value || value_size == 0)
   {
      return 0;
   }
   key_len = strlen(key);
   value[0] = '\0';

   while (*cur)
   {
      const char *line_end = strchr(cur, '\n');
      const char *eq       = strchr(cur, '=');
      size_t      line_len = line_end ? (size_t)(line_end - cur) : strlen(cur);
      if (eq && (size_t)(eq - cur) == key_len && strncmp(cur, key, key_len) == 0)
      {
         const char *val_start = eq + 1;
         size_t      val_len   = line_len - (size_t)(val_start - cur);
         if (val_len >= value_size)
         {
            val_len = value_size - 1u;
         }
         memcpy(value, val_start, val_len);
         value[val_len] = '\0';
         return 1;
      }
      if (!line_end)
      {
         break;
      }
      cur = line_end + 1;
   }
   return 0;
}

static int
ManifestFindInt(const char *payload, const char *key, int *out)
{
   char buf[128];
   char *end = NULL;
   long  v = 0;
   if (!out || !ManifestFindValue(payload, key, buf, sizeof(buf)))
   {
      return 0;
   }
   v = strtol(buf, &end, 10);
   if (end == buf || *end != '\0')
   {
      return 0;
   }
   *out = (int)v;
   return 1;
}

static void
UnpackArgsSetDefaults(UnpackArgs *args)
{
   if (!args)
   {
      return;
   }
   memset(args, 0, sizeof(*args));
   snprintf(args->prefix, sizeof(args->prefix), "ls");
   args->digits_suffix = 5;
}

static int
ParseUnpackArgs(int argc, char **argv, UnpackArgs *args)
{
   if (!args)
   {
      return 0;
   }
   UnpackArgsSetDefaults(args);
   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         return 0;
      }
      if (!strcmp(argv[i], "--input"))
      {
         if (i + 1 >= argc)
         {
            return 0;
         }
         snprintf(args->input_filename, sizeof(args->input_filename), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--output-dir"))
      {
         if (i + 1 >= argc)
         {
            return 0;
         }
         snprintf(args->output_dir, sizeof(args->output_dir), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--prefix"))
      {
         if (i + 1 >= argc)
         {
            return 0;
         }
         snprintf(args->prefix, sizeof(args->prefix), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--digits-suffix"))
      {
         if (i + 1 >= argc)
         {
            return 0;
         }
         args->digits_suffix = (int)strtol(argv[++i], NULL, 10);
         args->digits_suffix_set = 1;
      }
      else
      {
         return 0;
      }
   }
   return args->input_filename[0] != '\0' && args->output_dir[0] != '\0' &&
          args->digits_suffix > 0;
}

static int
ParseMetadataArgs(int argc, char **argv, MetadataArgs *args)
{
   if (!args)
   {
      return 0;
   }
   memset(args, 0, sizeof(*args));
   args->show_manifest = 1;

   for (int i = 1; i < argc; i++)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
      {
         return 0;
      }
      if (!strcmp(argv[i], "--input"))
      {
         if (i + 1 >= argc)
         {
            return 0;
         }
         snprintf(args->input_filename, sizeof(args->input_filename), "%s", argv[++i]);
      }
      else if (!strcmp(argv[i], "--no-manifest"))
      {
         args->show_manifest = 0;
      }
      else
      {
         return 0;
      }
   }
   return args->input_filename[0] != '\0';
}

static void
PrintGeneralUsage(const char *prog)
{
   fprintf(stderr,
           "Usage:\n"
           "  %s [pack] --dirname <dir> --output <out> [pack-options]\n"
           "  %s unpack --input <sequence.bin> --output-dir <dir> [unpack-options]\n"
           "  %s metadata --input <sequence.bin> [--no-manifest]\n"
           "\n"
           "Modes:\n"
           "  pack      Pack directory-based sequence into one LSSeq container (default mode).\n"
           "  unpack    Recreate directory-based sequence files from a LSSeq container.\n"
           "  metadata  Print container metadata and manifest summary.\n",
           prog, prog, prog);
}

static int
EnsureDirectoryExists(const char *path)
{
   char tmp[MAX_FILENAME_LENGTH];
   size_t n = 0;

   if (!path || path[0] == '\0')
   {
      return 0;
   }
   snprintf(tmp, sizeof(tmp), "%s", path);
   n = strlen(tmp);
   if (n == 0)
   {
      return 0;
   }
   if (tmp[n - 1] == '/')
   {
      tmp[n - 1] = '\0';
   }

   for (char *p = tmp + 1; *p; p++)
   {
      if (*p == '/')
      {
         *p = '\0';
         if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
         {
            return 0;
         }
         *p = '/';
      }
   }
   if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
   {
      return 0;
   }
   return 1;
}

static int
WriteTimestepsFile(const char *filename, const LSSeqTimestepEntry *timesteps, uint32_t num_timesteps)
{
   FILE *fp = NULL;
   if (!filename || !timesteps || num_timesteps == 0)
   {
      return 0;
   }
   fp = fopen(filename, "w");
   if (!fp)
   {
      return 0;
   }
   fprintf(fp, "%u\n", (unsigned int)num_timesteps);
   for (uint32_t i = 0; i < num_timesteps; i++)
   {
      fprintf(fp, "%d %d\n", timesteps[i].timestep, timesteps[i].ls_start);
   }
   fclose(fp);
   return 1;
}

static int
RunMetadataMode(MPI_Comm comm, int myid, const MetadataArgs *args)
{
   SeqPackedData seq;
   char          codec_name[64];

   (void)comm;
   if (!args)
   {
      return EXIT_FAILURE;
   }
   if (!myid)
   {
      if (!LoadPackedSequence(args->input_filename, &seq, 1))
      {
         fprintf(stderr, "Could not load sequence file '%s'\n", args->input_filename);
         return EXIT_FAILURE;
      }

      snprintf(codec_name, sizeof(codec_name), "%s",
               hypredrv_compression_get_name((comp_alg_t)seq.header.codec));
      printf("[lsseq] metadata summary\n");
      printf("  file: %s\n", args->input_filename);
      printf("  version: %u\n", seq.header.version);
      printf("  flags: 0x%08x\n", seq.header.flags);
      printf("  codec: %s (%u)\n", codec_name, seq.header.codec);
      printf("  systems: %u\n", seq.header.num_systems);
      printf("  parts: %u\n", seq.header.num_parts);
      printf("  unique patterns: %u\n", seq.header.num_patterns);
      printf("  timesteps: %u\n", seq.header.num_timesteps);
      printf("  offsets: part=%llu pattern=%llu sys=%llu timestep=%llu blob=%llu\n",
             (unsigned long long)seq.header.offset_part_meta,
             (unsigned long long)seq.header.offset_pattern_meta,
             (unsigned long long)seq.header.offset_sys_part_meta,
             (unsigned long long)seq.header.offset_timestep_meta,
             (unsigned long long)seq.header.offset_blob_data);
      printf("  info: payload_bytes=%llu blob_bytes=%llu\n",
             (unsigned long long)seq.info_header.payload_size,
             (unsigned long long)seq.info_header.blob_bytes);
      printf("  info hashes: payload=0x%016llx blob=0x%016llx\n",
             (unsigned long long)seq.info_header.payload_hash_fnv1a64,
             (unsigned long long)seq.info_header.blob_hash_fnv1a64);

      if (args->show_manifest && seq.info_payload && seq.info_payload[0] != '\0')
      {
         printf("\n[lsseq] manifest\n%s", seq.info_payload);
         if (seq.info_payload[seq.info_payload_size - 1] != '\n')
         {
            printf("\n");
         }
      }
      fflush(stdout);
      SeqPackedDataDestroy(&seq);
   }
   return EXIT_SUCCESS;
}

static int
RunUnpackMode(MPI_Comm comm, int myid, int nprocs, const UnpackArgs *args)
{
   SeqPackedData seq;
   FILE         *fp = NULL;
   int           init_suffix = 0;
   int           last_suffix = 0;
   int           digits_suffix = 5;
   char          matrix_filename[MAX_FILENAME_LENGTH];
   char          rhs_filename[MAX_FILENAME_LENGTH];
   char          dofmap_filename[MAX_FILENAME_LENGTH];
   char          timesteps_name[MAX_FILENAME_LENGTH];
   int           local_start = 0, local_nparts = 0;

   if (!args)
   {
      return EXIT_FAILURE;
   }
   if (!LoadPackedSequence(args->input_filename, &seq, 1))
   {
      if (!myid)
      {
         fprintf(stderr, "Could not load sequence file '%s'\n", args->input_filename);
      }
      return EXIT_FAILURE;
   }

   if (!EnsureDirectoryExists(args->output_dir))
   {
      if (!myid)
      {
         fprintf(stderr, "Could not create output directory '%s'\n", args->output_dir);
      }
      SeqPackedDataDestroy(&seq);
      return EXIT_FAILURE;
   }

   snprintf(matrix_filename, sizeof(matrix_filename), "IJ.out.A");
   snprintf(rhs_filename, sizeof(rhs_filename), "IJ.out.b");
   dofmap_filename[0] = '\0';
   snprintf(timesteps_name, sizeof(timesteps_name), "timesteps.txt");

   ManifestFindInt(seq.info_payload, "init_suffix", &init_suffix);
   if (!ManifestFindInt(seq.info_payload, "last_suffix", &last_suffix))
   {
      last_suffix = init_suffix + (int)seq.header.num_systems - 1;
   }
   if (last_suffix < init_suffix ||
       (uint32_t)(last_suffix - init_suffix + 1) != seq.header.num_systems)
   {
      last_suffix = init_suffix + (int)seq.header.num_systems - 1;
   }
   if (args->digits_suffix_set)
   {
      digits_suffix = args->digits_suffix;
   }
   else
   {
      ManifestFindInt(seq.info_payload, "digits_suffix", &digits_suffix);
      if (digits_suffix <= 0)
      {
         digits_suffix = 5;
      }
   }
   ManifestFindValue(seq.info_payload, "matrix_filename", matrix_filename, sizeof(matrix_filename));
   ManifestFindValue(seq.info_payload, "rhs_filename", rhs_filename, sizeof(rhs_filename));
   ManifestFindValue(seq.info_payload, "dofmap_filename", dofmap_filename, sizeof(dofmap_filename));
   {
      char tname[MAX_FILENAME_LENGTH];
      if (ManifestFindValue(seq.info_payload, "timesteps", tname, sizeof(tname)) && tname[0] != '\0')
      {
         const char *b = strrchr(tname, '/');
         snprintf(timesteps_name, sizeof(timesteps_name), "%s", b ? (b + 1) : tname);
      }
   }

   if (!myid)
   {
      printf("[lsseq] unpack summary\n");
      printf("  input: %s\n", args->input_filename);
      printf("  output-dir: %s\n", args->output_dir);
      printf("  prefix: %s\n", args->prefix);
      printf("  suffix range: init=%d last=%d (digits=%d)\n", init_suffix, last_suffix, digits_suffix);
      printf("  matrix-filename: %s\n", matrix_filename);
      printf("  rhs-filename: %s\n", rhs_filename);
      printf("  dofmap-filename: %s\n", dofmap_filename[0] ? dofmap_filename : "(none)");
      fflush(stdout);
   }

   /* Root pre-creates all system directories and optional timesteps. */
   if (!myid)
   {
      char dirpath[MAX_FILENAME_LENGTH];
      for (uint32_t s = 0; s < seq.header.num_systems; s++)
      {
         int suffix = init_suffix + (int)s;
         {
            char path_tmp[PATH_TMP_SIZE];
            snprintf(path_tmp, sizeof(path_tmp), "%s/%s_%0*d", args->output_dir, args->prefix,
                     digits_suffix, suffix);
            path_copy(dirpath, sizeof(dirpath), path_tmp);
         }
         if (!EnsureDirectoryExists(dirpath))
         {
            fprintf(stderr, "Could not create system directory '%s'\n", dirpath);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }
      }
      if ((seq.header.flags & LSSEQ_FLAG_HAS_TIMESTEPS) && seq.header.num_timesteps > 0)
      {
         char tfile[MAX_FILENAME_LENGTH];
         {
            char path_tmp[PATH_TMP_SIZE];
            snprintf(path_tmp, sizeof(path_tmp), "%s/%s", args->output_dir, timesteps_name);
            path_copy(tfile, sizeof(tfile), path_tmp);
         }
         if (!WriteTimestepsFile(tfile, seq.timesteps, seq.header.num_timesteps))
         {
            fprintf(stderr, "Could not write timesteps file '%s'\n", tfile);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }
      }
   }
   MPI_Barrier(comm);

   fp = fopen(args->input_filename, "rb");
   if (!fp)
   {
      SeqPackedDataDestroy(&seq);
      return EXIT_FAILURE;
   }

   ComputePartRange((int)seq.header.num_parts, nprocs, myid, &local_start, &local_nparts);
   printf("[lsseq][unpack][rank %d/%d] parts: start=%d count=%d\n", myid, nprocs, local_start,
          local_nparts);
   fflush(stdout);

   for (uint32_t s = 0; s < seq.header.num_systems; s++)
   {
      char system_dir[MAX_FILENAME_LENGTH];
      int  suffix = init_suffix + (int)s;
      {
         char path_tmp[PATH_TMP_SIZE];
         snprintf(path_tmp, sizeof(path_tmp), "%s/%s_%0*d", args->output_dir, args->prefix,
                  digits_suffix, suffix);
         path_copy(system_dir, sizeof(system_dir), path_tmp);
      }

      for (int lp = 0; lp < local_nparts; lp++)
      {
         uint32_t part_id = (uint32_t)(local_start + lp);
         const LSSeqPartMeta *part = &seq.parts[part_id];
         const LSSeqSystemPartMeta *sp =
            &seq.sys_parts[(size_t)s * (size_t)seq.header.num_parts + (size_t)part_id];
         const LSSeqPatternMeta *pat = NULL;
         void *rows = NULL, *cols = NULL, *vals = NULL, *rhs = NULL, *dof = NULL;
         size_t rows_sz = 0, cols_sz = 0, vals_sz = 0, rhs_sz = 0, dof_sz = 0;
         char mfile[MAX_FILENAME_LENGTH], rfile[MAX_FILENAME_LENGTH], dfile[MAX_FILENAME_LENGTH];

         if (sp->pattern_id >= seq.header.num_patterns)
         {
            fclose(fp);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }
         pat = &seq.patterns[sp->pattern_id];
         if (pat->part_id != part_id)
         {
            fclose(fp);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }

         if (!DecodeBlob(fp, (comp_alg_t)seq.header.codec, pat->rows_blob_offset, pat->rows_blob_size,
                         (size_t)pat->nnz * (size_t)part->row_index_size, &rows, &rows_sz) ||
             !DecodeBlob(fp, (comp_alg_t)seq.header.codec, pat->cols_blob_offset, pat->cols_blob_size,
                         (size_t)pat->nnz * (size_t)part->row_index_size, &cols, &cols_sz) ||
             !DecodeBlob(fp, (comp_alg_t)seq.header.codec, sp->values_blob_offset, sp->values_blob_size,
                         (size_t)sp->nnz * (size_t)part->value_size, &vals, &vals_sz) ||
             !DecodeBlob(fp, (comp_alg_t)seq.header.codec, sp->rhs_blob_offset, sp->rhs_blob_size,
                         (size_t)part->nrows * (size_t)part->value_size, &rhs, &rhs_sz))
         {
            free(rows); free(cols); free(vals); free(rhs); free(dof);
            fclose(fp);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }

         {
            char path_tmp[PATH_TMP_SIZE];
            snprintf(path_tmp, sizeof(path_tmp), "%s/%s.%05u.bin", system_dir, matrix_filename,
                    part_id);
            path_copy(mfile, sizeof(mfile), path_tmp);
            snprintf(path_tmp, sizeof(path_tmp), "%s/%s.%05u.bin", system_dir, rhs_filename,
                    part_id);
            path_copy(rfile, sizeof(rfile), path_tmp);
         }
         if (!WriteMatrixPartBinary(mfile, part, pat, rows, cols, vals) ||
             !WriteRHSPartBinary(rfile, part, rhs))
         {
            free(rows); free(cols); free(vals); free(rhs); free(dof);
            fclose(fp);
            SeqPackedDataDestroy(&seq);
            return EXIT_FAILURE;
         }

         if ((seq.header.flags & LSSEQ_FLAG_HAS_DOFMAP) && dofmap_filename[0] != '\0')
         {
            if (sp->dof_num_entries > 0 &&
                !DecodeBlob(fp, (comp_alg_t)seq.header.codec, sp->dof_blob_offset, sp->dof_blob_size,
                            (size_t)sp->dof_num_entries * sizeof(int32_t), &dof, &dof_sz))
            {
               free(rows); free(cols); free(vals); free(rhs); free(dof);
               fclose(fp);
               SeqPackedDataDestroy(&seq);
               return EXIT_FAILURE;
            }
            {
               char path_tmp[PATH_TMP_SIZE];
               snprintf(path_tmp, sizeof(path_tmp), "%s/%s.%05u", system_dir, dofmap_filename,
                       part_id);
               path_copy(dfile, sizeof(dfile), path_tmp);
            }
            if (!WriteDofPartASCII(dfile, (const int32_t *)dof, sp->dof_num_entries))
            {
               free(rows); free(cols); free(vals); free(rhs); free(dof);
               fclose(fp);
               SeqPackedDataDestroy(&seq);
               return EXIT_FAILURE;
            }
         }

         free(rows);
         free(cols);
         free(vals);
         free(rhs);
         free(dof);
      }
   }

   fclose(fp);
   MPI_Barrier(comm);
   if (!myid)
   {
      printf("[lsseq] unpack completed successfully\n");
      fflush(stdout);
   }
   SeqPackedDataDestroy(&seq);
   return EXIT_SUCCESS;
}

int
main(int argc, char **argv)
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int      myid = 0, nprocs = 1;
   PackArgs args;
   UnpackArgs unpack_args;
   MetadataArgs metadata_args;
   ToolMode mode = TOOL_MODE_PACK;
   int      mode_arg_shift = 0;
   int      invalid_mode = 0;
   int      parse_ok = 0;
   int      exit_code = EXIT_FAILURE;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &nprocs);

   if (argc > 1 && argv[1])
   {
      if (!strcmp(argv[1], "pack"))
      {
         mode = TOOL_MODE_PACK;
         mode_arg_shift = 1;
      }
      else if (!strcmp(argv[1], "unpack"))
      {
         mode = TOOL_MODE_UNPACK;
         mode_arg_shift = 1;
      }
      else if (!strcmp(argv[1], "metadata"))
      {
         mode = TOOL_MODE_METADATA;
         mode_arg_shift = 1;
      }
      else if (argv[1][0] != '-')
      {
         invalid_mode = 1;
      }
   }

   if (invalid_mode)
   {
      if (!myid)
      {
         PrintGeneralUsage(argv[0]);
      }
      MPI_Finalize();
      return EXIT_FAILURE;
   }

   if (mode == TOOL_MODE_PACK && argc > 1 &&
       (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")))
   {
      if (!myid)
      {
         PrintGeneralUsage(argv[0]);
         printf("\n");
         PrintUsage(argv[0]);
      }
      MPI_Finalize();
      return EXIT_SUCCESS;
   }

   if (mode == TOOL_MODE_UNPACK)
   {
      UnpackArgsSetDefaults(&unpack_args);
      if (!myid)
      {
         if (!ParseUnpackArgs(argc - mode_arg_shift, argv + mode_arg_shift, &unpack_args))
         {
            PrintGeneralUsage(argv[0]);
            parse_ok = 0;
            exit_code = EXIT_FAILURE;
         }
         else
         {
            parse_ok = 1;
            exit_code = EXIT_SUCCESS;
         }
      }
      MPI_Bcast(&parse_ok, 1, MPI_INT, 0, comm);
      MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
      if (!parse_ok)
      {
         MPI_Finalize();
         return exit_code;
      }
      MPI_Bcast(&unpack_args, (int)sizeof(unpack_args), MPI_BYTE, 0, comm);
      exit_code = RunUnpackMode(comm, myid, nprocs, &unpack_args);
      MPI_Finalize();
      return exit_code;
   }
   else if (mode == TOOL_MODE_METADATA)
   {
      memset(&metadata_args, 0, sizeof(metadata_args));
      if (!myid)
      {
         if (!ParseMetadataArgs(argc - mode_arg_shift, argv + mode_arg_shift, &metadata_args))
         {
            PrintGeneralUsage(argv[0]);
            parse_ok = 0;
            exit_code = EXIT_FAILURE;
         }
         else
         {
            parse_ok = 1;
            exit_code = EXIT_SUCCESS;
         }
      }
      MPI_Bcast(&parse_ok, 1, MPI_INT, 0, comm);
      MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
      if (!parse_ok)
      {
         MPI_Finalize();
         return exit_code;
      }
      MPI_Bcast(&metadata_args, (int)sizeof(metadata_args), MPI_BYTE, 0, comm);
      exit_code = RunMetadataMode(comm, myid, &metadata_args);
      MPI_Finalize();
      return exit_code;
   }

   PackArgsSetDefaults(&args);

   if (!myid)
   {
      if (HelpRequested(argc - mode_arg_shift, argv + mode_arg_shift))
      {
         PrintUsage(argv[0]);
         parse_ok   = 0;
         exit_code  = EXIT_SUCCESS;
      }
      else if (!ParseArgs(argc - mode_arg_shift, argv + mode_arg_shift, &args))
      {
         PrintUsage(argv[0]);
         parse_ok  = 0;
         exit_code = EXIT_FAILURE;
      }
      else
      {
         parse_ok  = 1;
         exit_code = EXIT_SUCCESS;
      }
   }

   MPI_Bcast(&parse_ok, 1, MPI_INT, 0, comm);
   MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
   if (!parse_ok)
   {
      MPI_Finalize();
      return exit_code;
   }

   /* Root resolves defaults and broadcasts final args to all ranks. */
   if (!myid)
   {
      char parent_dir[MAX_FILENAME_LENGTH];
      char base[256];
      int  min_suffix = 0, max_suffix = 0;
      char system_dir[MAX_FILENAME_LENGTH];
      char timesteps_candidate[MAX_FILENAME_LENGTH];

      parent_dir[0] = '\0';
      base[0]       = '\0';

      if (!ResolveDirPrefixAndSuffixRange(&args, parent_dir, sizeof(parent_dir), base,
                                          sizeof(base), &min_suffix, &max_suffix))
      {
         fprintf(stderr, "Failed to resolve sequence directory layout\n");
         MPI_Abort(comm, 1);
      }

      if (!args.init_suffix_set)
      {
         if (args.init_suffix < min_suffix || args.init_suffix > max_suffix)
         {
            args.init_suffix = (HYPRE_Int)min_suffix;
         }
      }
      else
      {
         if (args.init_suffix < min_suffix || args.init_suffix > max_suffix)
         {
            fprintf(stderr, "Requested --init-suffix %d is not available in '%s' (%s_%0*d..%0*d)\n",
                    (int)args.init_suffix, parent_dir, base, (int)args.digits_suffix, min_suffix,
                    (int)args.digits_suffix, max_suffix);
            MPI_Abort(comm, 1);
         }
      }

      if (!args.last_suffix_set)
      {
         args.last_suffix = (HYPRE_Int)max_suffix;
      }
      else
      {
         if (args.last_suffix < min_suffix || args.last_suffix > max_suffix)
         {
            fprintf(stderr, "Requested --last-suffix %d is not available in '%s' (%s_%0*d..%0*d)\n",
                    (int)args.last_suffix, parent_dir, base, (int)args.digits_suffix, min_suffix,
                    (int)args.digits_suffix, max_suffix);
            MPI_Abort(comm, 1);
         }
      }

      if (args.last_suffix < args.init_suffix)
      {
         fprintf(stderr, "Invalid suffix range: init=%d last=%d\n", (int)args.init_suffix,
                 (int)args.last_suffix);
         MPI_Abort(comm, 1);
      }

      BuildSystemDir(system_dir, sizeof(system_dir), args.dirname, (int)args.digits_suffix,
                     (int)args.init_suffix);

      if ((args.matrix_filename[0] == '\0') || (args.rhs_filename[0] == '\0') ||
          (!args.dofmap_filename_set && args.dofmap_filename[0] == '\0'))
      {
         if (!AutoDetectFilenames(&args, system_dir))
         {
            fprintf(stderr,
                    "Failed to auto-detect --matrix-filename/--rhs-filename in '%s'\n"
                    "  Provide them explicitly or ensure part files exist.\n",
                    system_dir);
            MPI_Abort(comm, 1);
         }
      }

      if (!args.timesteps_filename_set && args.timesteps_filename[0] == '\0')
      {
         char path_tmp[PATH_TMP_SIZE];
         snprintf(path_tmp, sizeof(path_tmp), "%s/timesteps.txt",
                  parent_dir[0] != '\0' ? parent_dir : ".");
         path_copy(timesteps_candidate, sizeof(timesteps_candidate), path_tmp);
         if (PathIsRegularFile(timesteps_candidate))
         {
            path_copy(args.timesteps_filename, sizeof(args.timesteps_filename), timesteps_candidate);
         }
         else
         {
            snprintf(path_tmp, sizeof(path_tmp), "%s/timesteps.txt", system_dir);
            path_copy(timesteps_candidate, sizeof(timesteps_candidate), path_tmp);
            if (PathIsRegularFile(timesteps_candidate))
            {
               path_copy(args.timesteps_filename, sizeof(args.timesteps_filename),
                         timesteps_candidate);
            }
         }
      }

      printf("[lsseq] MPI ranks: %d\n", nprocs);
      printf("[lsseq] dirname (input):   %s\n", args.input_dirname);
      printf("[lsseq] dirname (resolved): %s\n", args.dirname);
      printf("[lsseq] suffix range: init=%d last=%d (digits=%d)\n", (int)args.init_suffix,
             (int)args.last_suffix, (int)args.digits_suffix);
      printf("[lsseq] matrix-filename:    %s\n", args.matrix_filename);
      printf("[lsseq] rhs-filename:       %s\n", args.rhs_filename);
      printf("[lsseq] dofmap-filename:    %s\n",
             args.dofmap_filename[0] ? args.dofmap_filename : "(none)");
      printf("[lsseq] timesteps:          %s\n",
             args.timesteps_filename[0] ? args.timesteps_filename : "(none)");
      printf("[lsseq] codec:              %s\n", hypredrv_compression_get_name(args.algo));
      fflush(stdout);
   }

   MPI_Bcast(&args, (int)sizeof(args), MPI_BYTE, 0, comm);

   /* Compute shape and detect number of parts from first system directory. */
   int num_systems = (int)(args.last_suffix - args.init_suffix + 1);
   char matrix_prefix[MAX_FILENAME_LENGTH];
   BuildPrefix(matrix_prefix, sizeof(matrix_prefix), args.dirname, (int)args.digits_suffix,
               (int)args.init_suffix, args.matrix_filename);

   int num_parts = CountNumberOfPartitions(matrix_prefix);
   if (num_parts <= 0)
   {
      if (!myid)
      {
         fprintf(stderr, "Could not detect matrix part files for prefix '%s'\n", matrix_prefix);
      }
      MPI_Abort(comm, 1);
   }

   int local_start = 0, local_nparts = 0;
   ComputePartRange(num_parts, nprocs, myid, &local_start, &local_nparts);
   printf("[lsseq][pack][rank %d/%d] parts: start=%d count=%d (global num_parts=%d)\n", myid,
          nprocs, local_start, local_nparts, num_parts);
   fflush(stdout);

   LSSeqPartMeta       *part_meta_local = NULL;
   LSSeqSystemPartMeta *sys_meta_local  = NULL;
   PatternBuild        *patterns_local  = NULL;
   size_t               num_patterns_local = 0;
   size_t               cap_patterns_local = 0;
   FILE                *blob_fp = NULL;
   uint64_t              blob_cursor = 0;
   unsigned long long    local_raw_blob_ull = 0;

   if (local_nparts > 0)
   {
      part_meta_local = (LSSeqPartMeta *)calloc((size_t)local_nparts, sizeof(*part_meta_local));
      sys_meta_local =
         (LSSeqSystemPartMeta *)calloc((size_t)num_systems * (size_t)local_nparts,
                                       sizeof(*sys_meta_local));
   }
   if ((local_nparts > 0 && (!part_meta_local || !sys_meta_local)))
   {
      fprintf(stderr, "[lsseq][pack][rank %d] Allocation failure for local metadata arrays\n", myid);
      MPI_Abort(comm, 1);
   }

   blob_fp = tmpfile();
   if (!blob_fp)
   {
      fprintf(stderr, "[lsseq][pack][rank %d] Could not create temporary blob file\n", myid);
      MPI_Abort(comm, 1);
   }

   char rhs_prefix[MAX_FILENAME_LENGTH];
   char dof_prefix[MAX_FILENAME_LENGTH];
   int  want_dofmap = (args.dofmap_filename[0] != '\0');

   /* v2 batched: one buffer per (local part, type) for values/rhs/dof */
   typedef struct
   {
      void  *ptr;
      size_t len;
      size_t cap;
   } PartBuf;
   PartBuf  *part_vals = NULL;
   PartBuf  *part_rhs  = NULL;
   PartBuf  *part_dof  = NULL;
   uint64_t *local_part_blob_sizes = NULL; /* 3*local_nparts: vals, rhs, dof per part */
   uint64_t  local_pattern_blob_ull = 0;  /* cursor after pattern blobs only */

   if (local_nparts > 0)
   {
      part_vals = (PartBuf *)calloc((size_t)local_nparts, sizeof(*part_vals));
      part_rhs  = (PartBuf *)calloc((size_t)local_nparts, sizeof(*part_rhs));
      part_dof  = (PartBuf *)calloc((size_t)local_nparts, sizeof(*part_dof));
      local_part_blob_sizes = (uint64_t *)calloc((size_t)local_nparts * 3u, sizeof(uint64_t));
      if (!part_vals || !part_rhs || !part_dof || !local_part_blob_sizes)
      {
         fprintf(stderr, "[lsseq][pack][rank %d] Allocation failure for v2 part buffers\n", myid);
         MPI_Abort(comm, 1);
      }
   }

   for (int s = 0; s < num_systems; s++)
   {
      int suffix = (int)args.init_suffix + s;
      BuildPrefix(matrix_prefix, sizeof(matrix_prefix), args.dirname, (int)args.digits_suffix,
                  suffix, args.matrix_filename);
      BuildPrefix(rhs_prefix, sizeof(rhs_prefix), args.dirname, (int)args.digits_suffix, suffix,
                  args.rhs_filename);
      if (want_dofmap)
      {
         BuildPrefix(dof_prefix, sizeof(dof_prefix), args.dirname, (int)args.digits_suffix, suffix,
                     args.dofmap_filename);
      }

      if (!myid && (s == 0 || (num_systems > 20 && (s % 10 == 0))))
      {
         printf("[lsseq][pack] packing system %d/%d (suffix=%d)\n", s + 1, num_systems, suffix);
         fflush(stdout);
      }

      for (int lp = 0; lp < local_nparts; lp++)
      {
         int global_p = local_start + lp;
         MatrixPartRaw Araw;
         RHSPartRaw    braw;
         DofPartRaw    draw;
         uint64_t      phash = 0;
         int           pattern_id = -1;
         LSSeqSystemPartMeta *sp =
            &sys_meta_local[(size_t)s * (size_t)local_nparts + (size_t)lp];

         if (!ReadMatrixPart(matrix_prefix, global_p, &Araw))
         {
            fprintf(stderr, "[lsseq][pack][rank %d] Could not read matrix part %d for suffix %d\n",
                    myid, global_p, suffix);
            MPI_Abort(comm, 1);
         }
         if (!ReadRHSPart(rhs_prefix, global_p, &braw))
         {
            fprintf(stderr, "[lsseq][pack][rank %d] Could not read RHS part %d for suffix %d\n", myid,
                    global_p, suffix);
            MatrixPartRawDestroy(&Araw);
            MPI_Abort(comm, 1);
         }

         memset(&draw, 0, sizeof(draw));
         if (want_dofmap)
         {
            if (!ReadDofPart(dof_prefix, global_p, &draw))
            {
               fprintf(stderr,
                       "[lsseq][pack][rank %d] Could not read dofmap part %d for suffix %d\n",
                       myid, global_p, suffix);
               MatrixPartRawDestroy(&Araw);
               RHSPartRawDestroy(&braw);
               MPI_Abort(comm, 1);
            }
         }

         if (s == 0)
         {
            part_meta_local[lp].row_lower      = Araw.row_lower;
            part_meta_local[lp].row_upper      = Araw.row_upper;
            part_meta_local[lp].nrows          = braw.nrows;
            part_meta_local[lp].row_index_size = Araw.row_index_size;
            part_meta_local[lp].value_size     = Araw.value_size;
         }
         else
         {
            if (part_meta_local[lp].row_lower != Araw.row_lower ||
                part_meta_local[lp].row_upper != Araw.row_upper ||
                part_meta_local[lp].nrows != braw.nrows ||
                part_meta_local[lp].row_index_size != Araw.row_index_size ||
                part_meta_local[lp].value_size != Araw.value_size)
            {
               fprintf(stderr,
                       "[lsseq][pack][rank %d] Part %d metadata changed across systems (not supported)\n",
                       myid, global_p);
               MatrixPartRawDestroy(&Araw);
               RHSPartRawDestroy(&braw);
               DofPartRawDestroy(&draw);
               MPI_Abort(comm, 1);
            }
         }

         phash = PatternHash(&Araw, (uint32_t)global_p);
         for (size_t k = 0; k < num_patterns_local; k++)
         {
            if (patterns_local[k].hash == phash &&
                patterns_local[k].meta.part_id == (uint32_t)global_p &&
                patterns_local[k].meta.nnz == Araw.nnz)
            {
               pattern_id = (int)k;
               break;
            }
         }

         if (pattern_id < 0)
         {
            PatternBuild entry;
            memset(&entry, 0, sizeof(entry));
            entry.hash         = phash;
            entry.meta.part_id = (uint32_t)global_p;
            entry.meta.nnz     = Araw.nnz;
            if (!PackWriteBlob(blob_fp, args.algo, args.compression_level, Araw.rows,
                               (size_t)Araw.nnz * (size_t)Araw.row_index_size, &blob_cursor,
                               &entry.meta.rows_blob_offset, &entry.meta.rows_blob_size) ||
                !PackWriteBlob(blob_fp, args.algo, args.compression_level, Araw.cols,
                               (size_t)Araw.nnz * (size_t)Araw.row_index_size, &blob_cursor,
                               &entry.meta.cols_blob_offset, &entry.meta.cols_blob_size))
            {
               fprintf(stderr, "[lsseq][pack][rank %d] Failed to write compressed pattern blob\n",
                       myid);
               MatrixPartRawDestroy(&Araw);
               RHSPartRawDestroy(&braw);
               DofPartRawDestroy(&draw);
               MPI_Abort(comm, 1);
            }
            local_raw_blob_ull +=
               (unsigned long long)((size_t)Araw.nnz * (size_t)Araw.row_index_size);
            local_raw_blob_ull +=
               (unsigned long long)((size_t)Araw.nnz * (size_t)Araw.row_index_size);

            if (num_patterns_local == cap_patterns_local)
            {
               size_t       new_cap = cap_patterns_local == 0 ? 16u : (2u * cap_patterns_local);
               PatternBuild *tmp =
                  (PatternBuild *)realloc(patterns_local, new_cap * sizeof(*patterns_local));
               if (!tmp)
               {
                  fprintf(stderr,
                          "[lsseq][pack][rank %d] Allocation failure while extending pattern list\n",
                          myid);
                  MatrixPartRawDestroy(&Araw);
                  RHSPartRawDestroy(&braw);
                  DofPartRawDestroy(&draw);
                  MPI_Abort(comm, 1);
               }
               patterns_local     = tmp;
               cap_patterns_local = new_cap;
            }

            patterns_local[num_patterns_local] = entry;
            pattern_id = (int)num_patterns_local;
            num_patterns_local++;
         }

         sp->pattern_id = (uint32_t)pattern_id;
         sp->nnz        = Araw.nnz;

         /* v2 batched: append to per-part buffers; store decompressed offset/size in sys_meta */
         {
            size_t vsz = (size_t)Araw.nnz * (size_t)Araw.value_size;
            size_t rsz = (size_t)braw.nrows * (size_t)braw.value_size;
            size_t dsz = (size_t)draw.num_entries * sizeof(int32_t);
            sp->values_blob_offset = (uint64_t)part_vals[lp].len;
            sp->values_blob_size   = (uint64_t)vsz;
            sp->rhs_blob_offset   = (uint64_t)part_rhs[lp].len;
            sp->rhs_blob_size     = (uint64_t)rsz;
            sp->dof_num_entries  = draw.num_entries;
            if (dsz > 0)
            {
               sp->dof_blob_offset = (uint64_t)part_dof[lp].len;
               sp->dof_blob_size   = (uint64_t)dsz;
            }
            if (!BufAppendRaw(&part_vals[lp].ptr, &part_vals[lp].len, &part_vals[lp].cap,
                             Araw.vals, vsz) ||
                !BufAppendRaw(&part_rhs[lp].ptr, &part_rhs[lp].len, &part_rhs[lp].cap,
                             braw.vals, rsz))
            {
               MatrixPartRawDestroy(&Araw);
               RHSPartRawDestroy(&braw);
               DofPartRawDestroy(&draw);
               fprintf(stderr, "[lsseq][pack][rank %d] Failed to append to part buffers\n", myid);
               MPI_Abort(comm, 1);
            }
            if (dsz > 0 &&
                !BufAppendRaw(&part_dof[lp].ptr, &part_dof[lp].len, &part_dof[lp].cap,
                             draw.vals, dsz))
            {
               MatrixPartRawDestroy(&Araw);
               RHSPartRawDestroy(&braw);
               DofPartRawDestroy(&draw);
               fprintf(stderr, "[lsseq][pack][rank %d] Failed to append dof to part buffer\n", myid);
               MPI_Abort(comm, 1);
            }
            local_raw_blob_ull += (unsigned long long)vsz + (unsigned long long)rsz;
            if (dsz > 0)
            {
               local_raw_blob_ull += (unsigned long long)dsz;
            }
         }

         MatrixPartRawDestroy(&Araw);
         RHSPartRawDestroy(&braw);
         DofPartRawDestroy(&draw);
      }
   }

   local_pattern_blob_ull = (unsigned long long)blob_cursor;

   /* v2: write batched part blobs (one compressed blob per part per type) */
   for (int lp = 0; lp < local_nparts; lp++)
   {
      uint64_t voff = 0, vsz = 0, roff = 0, rsz = 0, doff = 0, dsz = 0;
      if (part_vals[lp].len > 0 &&
          !PackWriteBlob(blob_fp, args.algo, args.compression_level, part_vals[lp].ptr,
                         part_vals[lp].len, &blob_cursor, &voff, &vsz))
      {
         fprintf(stderr, "[lsseq][pack][rank %d] Failed to write batched values blob for part %d\n",
                 myid, local_start + lp);
         MPI_Abort(comm, 1);
      }
      local_part_blob_sizes[(size_t)lp * 3u + 0u] = vsz;
      if (part_rhs[lp].len > 0 &&
          !PackWriteBlob(blob_fp, args.algo, args.compression_level, part_rhs[lp].ptr,
                         part_rhs[lp].len, &blob_cursor, &roff, &rsz))
      {
         fprintf(stderr, "[lsseq][pack][rank %d] Failed to write batched RHS blob for part %d\n",
                 myid, local_start + lp);
         MPI_Abort(comm, 1);
      }
      local_part_blob_sizes[(size_t)lp * 3u + 1u] = rsz;
      if (part_dof[lp].len > 0 &&
          !PackWriteBlob(blob_fp, args.algo, args.compression_level, part_dof[lp].ptr,
                         part_dof[lp].len, &blob_cursor, &doff, &dsz))
      {
         fprintf(stderr, "[lsseq][pack][rank %d] Failed to write batched dofmap blob for part %d\n",
                 myid, local_start + lp);
         MPI_Abort(comm, 1);
      }
      local_part_blob_sizes[(size_t)lp * 3u + 2u] = dsz;
   }
   for (int lp = 0; lp < local_nparts; lp++)
   {
      free(part_vals[lp].ptr);
      free(part_rhs[lp].ptr);
      free(part_dof[lp].ptr);
   }
   free(part_vals);
   free(part_rhs);
   free(part_dof);

   /* Compute global pattern and blob offsets in rank order. */
   unsigned int local_patterns_u = (unsigned int)num_patterns_local;
   unsigned int pattern_base = 0;
   unsigned int global_patterns_u = 0;
   MPI_Exscan(&local_patterns_u, &pattern_base, 1, MPI_UNSIGNED, MPI_SUM, comm);
   if (!myid)
   {
      pattern_base = 0;
   }
   MPI_Allreduce(&local_patterns_u, &global_patterns_u, 1, MPI_UNSIGNED, MPI_SUM, comm);

   unsigned long long local_blob_ull = (unsigned long long)blob_cursor;
   unsigned long long blob_base_ull  = 0;
   unsigned long long global_blob_ull = 0;
   unsigned long long global_raw_blob_ull = 0;
   MPI_Exscan(&local_blob_ull, &blob_base_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
   if (!myid)
   {
      blob_base_ull = 0;
   }
   MPI_Allreduce(&local_blob_ull, &global_blob_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
   MPI_Allreduce(&local_raw_blob_ull, &global_raw_blob_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                 comm);

   for (size_t i = 0; i < num_patterns_local; i++)
   {
      if (patterns_local[i].meta.rows_blob_size > 0)
      {
         patterns_local[i].meta.rows_blob_offset += (uint64_t)blob_base_ull;
      }
      if (patterns_local[i].meta.cols_blob_size > 0)
      {
         patterns_local[i].meta.cols_blob_offset += (uint64_t)blob_base_ull;
      }
   }
   for (int i = 0; i < num_systems * local_nparts; i++)
   {
      LSSeqSystemPartMeta *meta = &sys_meta_local[i];
      meta->pattern_id += (uint32_t)pattern_base;
      /* v2: values/rhs/dof offsets are decompressed offsets within part blobs; do not add blob_base */
   }

   printf("[lsseq][pack][rank %d] local patterns=%u local blob bytes=%llu\n", myid, local_patterns_u,
          local_blob_ull);
   fflush(stdout);

   /* Gather metadata to root. */
   LSSeqPartMeta       *part_meta_global = NULL;
   LSSeqSystemPartMeta *sys_meta_global  = NULL;
   PatternBuild        *patterns_global  = NULL;
   LSSeqSystemPartMeta *sys_meta_rankbuf = NULL;
   int                 *pat_counts = NULL, *pat_displs = NULL;
   int                 *sys_counts = NULL, *sys_displs = NULL;

   if (!myid)
   {
      part_meta_global = (LSSeqPartMeta *)calloc((size_t)num_parts, sizeof(*part_meta_global));
      sys_meta_global =
         (LSSeqSystemPartMeta *)calloc((size_t)num_systems * (size_t)num_parts,
                                       sizeof(*sys_meta_global));
      if (!part_meta_global || !sys_meta_global)
      {
         fprintf(stderr, "Allocation failure while creating global metadata arrays\n");
         MPI_Abort(comm, 1);
      }

      pat_counts = (int *)calloc((size_t)nprocs, sizeof(int));
      pat_displs = (int *)calloc((size_t)nprocs, sizeof(int));
      sys_counts = (int *)calloc((size_t)nprocs, sizeof(int));
      sys_displs = (int *)calloc((size_t)nprocs, sizeof(int));
      if (!pat_counts || !pat_displs || !sys_counts || !sys_displs)
      {
         fprintf(stderr, "Allocation failure while creating gather arrays\n");
         MPI_Abort(comm, 1);
      }
   }

   /* Part metadata: gather directly into global array at the correct part offsets. */
   if (!myid)
   {
      int *recvcounts = (int *)calloc((size_t)nprocs, sizeof(int));
      int *displs     = (int *)calloc((size_t)nprocs, sizeof(int));
      if (!recvcounts || !displs)
      {
         fprintf(stderr, "Allocation failure while creating part gather arrays\n");
         MPI_Abort(comm, 1);
      }
      for (int r = 0; r < nprocs; r++)
      {
         int start_r = 0, count_r = 0;
         ComputePartRange(num_parts, nprocs, r, &start_r, &count_r);
         recvcounts[r] = (int)((size_t)count_r * sizeof(LSSeqPartMeta));
         displs[r]     = (int)((size_t)start_r * sizeof(LSSeqPartMeta));
      }
      MPI_Gatherv(part_meta_local, (int)((size_t)local_nparts * sizeof(LSSeqPartMeta)), MPI_BYTE,
                  part_meta_global, recvcounts, displs, MPI_BYTE, 0, comm);
      free(recvcounts);
      free(displs);
   }
   else
   {
      MPI_Gatherv(part_meta_local, (int)((size_t)local_nparts * sizeof(LSSeqPartMeta)), MPI_BYTE,
                  NULL, NULL, NULL, MPI_BYTE, 0, comm);
   }

   /* Pattern metadata: gather in rank order using local pattern counts. */
   int local_patterns_i = (int)local_patterns_u;
   MPI_Gather(&local_patterns_i, 1, MPI_INT, pat_counts, 1, MPI_INT, 0, comm);

   if (!myid)
   {
      int total_patterns = 0;
      for (int r = 0; r < nprocs; r++)
      {
         pat_displs[r] = total_patterns;
         total_patterns += pat_counts[r];
      }
      patterns_global = (PatternBuild *)calloc((size_t)total_patterns, sizeof(*patterns_global));
      if (!patterns_global && total_patterns > 0)
      {
         fprintf(stderr, "Allocation failure while creating global pattern array\n");
         MPI_Abort(comm, 1);
      }

      int *recvcounts_bytes = (int *)calloc((size_t)nprocs, sizeof(int));
      int *displs_bytes     = (int *)calloc((size_t)nprocs, sizeof(int));
      if (!recvcounts_bytes || !displs_bytes)
      {
         fprintf(stderr, "Allocation failure while creating pattern gather arrays\n");
         MPI_Abort(comm, 1);
      }
      for (int r = 0; r < nprocs; r++)
      {
         recvcounts_bytes[r] = (int)((size_t)pat_counts[r] * sizeof(PatternBuild));
         displs_bytes[r]     = (int)((size_t)pat_displs[r] * sizeof(PatternBuild));
      }

      MPI_Gatherv(patterns_local, (int)((size_t)local_patterns_i * sizeof(PatternBuild)), MPI_BYTE,
                  patterns_global, recvcounts_bytes, displs_bytes, MPI_BYTE, 0, comm);
      free(recvcounts_bytes);
      free(displs_bytes);
   }
   else
   {
      MPI_Gatherv(patterns_local, (int)((size_t)local_patterns_i * sizeof(PatternBuild)), MPI_BYTE,
                  NULL, NULL, NULL, MPI_BYTE, 0, comm);
   }

   /* System-part metadata: gather into a rank-ordered buffer, then scatter into global layout. */
   int local_sys_elems = num_systems * local_nparts;
   MPI_Gather(&local_sys_elems, 1, MPI_INT, sys_counts, 1, MPI_INT, 0, comm);
   if (!myid)
   {
      int total_sys = 0;
      for (int r = 0; r < nprocs; r++)
      {
         sys_displs[r] = total_sys;
         total_sys += sys_counts[r];
      }

      sys_meta_rankbuf = (LSSeqSystemPartMeta *)calloc((size_t)total_sys, sizeof(*sys_meta_rankbuf));
      if (!sys_meta_rankbuf && total_sys > 0)
      {
         fprintf(stderr, "Allocation failure while creating system metadata gather buffer\n");
         MPI_Abort(comm, 1);
      }

      int *recvcounts_bytes = (int *)calloc((size_t)nprocs, sizeof(int));
      int *displs_bytes     = (int *)calloc((size_t)nprocs, sizeof(int));
      if (!recvcounts_bytes || !displs_bytes)
      {
         fprintf(stderr, "Allocation failure while creating system gather arrays\n");
         MPI_Abort(comm, 1);
      }
      for (int r = 0; r < nprocs; r++)
      {
         recvcounts_bytes[r] = (int)((size_t)sys_counts[r] * sizeof(LSSeqSystemPartMeta));
         displs_bytes[r]     = (int)((size_t)sys_displs[r] * sizeof(LSSeqSystemPartMeta));
      }

      MPI_Gatherv(sys_meta_local, (int)((size_t)local_sys_elems * sizeof(LSSeqSystemPartMeta)),
                  MPI_BYTE, sys_meta_rankbuf, recvcounts_bytes, displs_bytes, MPI_BYTE, 0, comm);
      free(recvcounts_bytes);
      free(displs_bytes);

      for (int r = 0; r < nprocs; r++)
      {
         int start_r = 0, count_r = 0;
         ComputePartRange(num_parts, nprocs, r, &start_r, &count_r);
         if (count_r == 0)
         {
            continue;
         }
         LSSeqSystemPartMeta *src = &sys_meta_rankbuf[(size_t)sys_displs[r]];
         for (int s = 0; s < num_systems; s++)
         {
            memcpy(&sys_meta_global[(size_t)s * (size_t)num_parts + (size_t)start_r],
                   &src[(size_t)s * (size_t)count_r], (size_t)count_r * sizeof(*src));
         }
      }
   }
   else
   {
      MPI_Gatherv(sys_meta_local, (int)((size_t)local_sys_elems * sizeof(LSSeqSystemPartMeta)),
                  MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, comm);
   }

   /* Gather part blob sizes (v2) so root can build part_blob_table. */
   uint64_t *part_blob_table = NULL;
   {
      int      sendcount = 1 + 3 * local_nparts;
      uint64_t *sendbuf_sizes =
         (uint64_t *)malloc((size_t)(sendcount > 0 ? sendcount : 1) * sizeof(uint64_t));
      if (!sendbuf_sizes)
      {
         fprintf(stderr, "[lsseq][pack][rank %d] Allocation failure for size gather buffer\n", myid);
         MPI_Abort(comm, 1);
      }
      sendbuf_sizes[0] = local_pattern_blob_ull;
      if (local_nparts > 0)
      {
         memcpy(sendbuf_sizes + 1, local_part_blob_sizes,
                (size_t)(3 * local_nparts) * sizeof(uint64_t));
      }
      if (!myid)
      {
         int *recvcounts_sz = (int *)calloc((size_t)nprocs, sizeof(int));
         int *displs_sz     = (int *)calloc((size_t)nprocs, sizeof(int));
         if (!recvcounts_sz || !displs_sz)
         {
            free(sendbuf_sizes);
            fprintf(stderr, "Allocation failure for part blob size gather\n");
            MPI_Abort(comm, 1);
         }
         int total_recv = 0;
         for (int r = 0; r < nprocs; r++)
         {
            int start_r = 0, count_r = 0;
            ComputePartRange(num_parts, nprocs, r, &start_r, &count_r);
            recvcounts_sz[r] = 1 + 3 * count_r;
            displs_sz[r]    = total_recv;
            total_recv += recvcounts_sz[r];
         }
         uint64_t *recvbuf_sz =
            (uint64_t *)malloc((size_t)(total_recv > 0 ? total_recv : 1) * sizeof(uint64_t));
         if (!recvbuf_sz)
         {
            free(recvcounts_sz);
            free(displs_sz);
            free(sendbuf_sizes);
            fprintf(stderr, "Allocation failure for part blob size recv buffer\n");
            MPI_Abort(comm, 1);
         }
         MPI_Gatherv(sendbuf_sizes, sendcount, MPI_UNSIGNED_LONG_LONG, recvbuf_sz, recvcounts_sz,
                     displs_sz, MPI_UNSIGNED_LONG_LONG, 0, comm);
         part_blob_table =
            (uint64_t *)calloc((size_t)num_parts * (size_t)LSSEQ_PART_BLOB_ENTRIES,
                               sizeof(uint64_t));
         if (!part_blob_table)
         {
            free(recvbuf_sz);
            free(recvcounts_sz);
            free(displs_sz);
            free(sendbuf_sizes);
            fprintf(stderr, "Allocation failure for part blob table\n");
            MPI_Abort(comm, 1);
         }
         {
            uint64_t cursor = 0;
            for (int r = 0; r < nprocs; r++)
            {
               int start_r = 0, count_r = 0;
               ComputePartRange(num_parts, nprocs, r, &start_r, &count_r);
               cursor += recvbuf_sz[displs_sz[r]]; /* skip this rank's pattern blobs */
               for (int local_p = 0; local_p < count_r; local_p++)
               {
                  int p   = start_r + local_p;
                  int idx = displs_sz[r] + 1 + local_p * 3;
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 0] = cursor;
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 1] = recvbuf_sz[idx];
                  cursor += recvbuf_sz[idx];
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 2] = cursor;
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 3] = recvbuf_sz[idx + 1];
                  cursor += recvbuf_sz[idx + 1];
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 4] = cursor;
                  part_blob_table[(size_t)p * LSSEQ_PART_BLOB_ENTRIES + 5] = recvbuf_sz[idx + 2];
                  cursor += recvbuf_sz[idx + 2];
               }
            }
         }
         free(recvbuf_sz);
         free(recvcounts_sz);
         free(displs_sz);
      }
      else
      {
         MPI_Gatherv(sendbuf_sizes, sendcount, MPI_UNSIGNED_LONG_LONG, NULL, NULL, NULL,
                     MPI_UNSIGNED_LONG_LONG, 0, comm);
      }
      free(sendbuf_sizes);
   }
   free(local_part_blob_sizes);

   /* Root writes the final output file and receives blob payloads from other ranks. */
   if (!myid)
   {
      char output_filename[MAX_FILENAME_LENGTH];
      snprintf(output_filename, sizeof(output_filename), "%s", args.output_filename);
      if (!StringHasSuffix(output_filename, hypredrv_compression_get_extension(args.algo)))
      {
         strncat(output_filename, hypredrv_compression_get_extension(args.algo),
                 sizeof(output_filename) - strlen(output_filename) - 1);
      }

      LSSeqTimestepEntry *timesteps = NULL;
      uint32_t            num_timesteps = 0;
      if (args.timesteps_filename[0] != '\0')
      {
         LoadTimesteps(args.timesteps_filename, &timesteps, &num_timesteps);
      }

      char   *info_payload = NULL;
      size_t  info_payload_size = 0;
      uint64_t info_payload_hash = UINT64_C(1469598103934665603);
      LSSeqInfoHeader info_header;

      LSSeqHeader header;
      uint64_t    off_info, off_part, off_pattern, off_sys, off_time, off_blob;
      FILE       *out = fopen(output_filename, "wb");
      uint8_t     io_buf[1 << 20];

      if (!out)
      {
         fprintf(stderr, "Could not create output file '%s'\n", output_filename);
         MPI_Abort(comm, 1);
      }

      memset(&header, 0, sizeof(header));
      header.magic         = LSSEQ_MAGIC;
      header.version       = LSSEQ_VERSION;
      header.codec         = (uint32_t)args.algo;
      header.num_systems   = (uint32_t)num_systems;
      header.num_parts     = (uint32_t)num_parts;
      header.num_patterns  = (uint32_t)global_patterns_u;
      header.num_timesteps = num_timesteps;
      if (want_dofmap)
      {
         header.flags |= LSSEQ_FLAG_HAS_DOFMAP;
      }
      if (num_timesteps > 0)
      {
         header.flags |= LSSEQ_FLAG_HAS_TIMESTEPS;
      }
      header.flags |= LSSEQ_FLAG_HAS_INFO;

      info_payload = BuildLSSeqInfoPayload(&args, output_filename, nprocs, num_systems, num_parts,
                                           global_patterns_u, num_timesteps, global_blob_ull,
                                           global_raw_blob_ull, &info_payload_size);
      if (!info_payload)
      {
         fprintf(stderr, "Could not build LSSeq info payload\n");
         MPI_Abort(comm, 1);
      }

      info_payload_hash = FNV1a64(info_payload, info_payload_size, info_payload_hash);
      memset(&info_header, 0, sizeof(info_header));
      info_header.magic              = LSSEQ_INFO_MAGIC;
      info_header.version            = LSSEQ_INFO_VERSION;
      info_header.flags              = LSSEQ_INFO_FLAG_PAYLOAD_KV;
      info_header.endian_tag         = UINT32_C(0x01020304);
      info_header.payload_size       = (uint64_t)info_payload_size;
      info_header.payload_hash_fnv1a64 = info_payload_hash;
      info_header.blob_hash_fnv1a64  = 0;
      info_header.blob_bytes         = (uint64_t)global_blob_ull;

      off_info    = sizeof(LSSeqHeader);
      off_part    = off_info + (uint64_t)sizeof(LSSeqInfoHeader) + (uint64_t)info_payload_size;
      off_pattern = off_part + (uint64_t)num_parts * (uint64_t)sizeof(LSSeqPartMeta);
      off_sys     = off_pattern +
                (uint64_t)global_patterns_u * (uint64_t)sizeof(LSSeqPatternMeta);
      {
         uint64_t off_part_blob_table =
            off_sys + (uint64_t)num_systems * (uint64_t)num_parts *
                         (uint64_t)sizeof(LSSeqSystemPartMeta);
         off_time = off_part_blob_table +
                    (uint64_t)num_parts * (uint64_t)LSSEQ_PART_BLOB_ENTRIES * sizeof(uint64_t);
         off_blob = off_time + (uint64_t)num_timesteps * (uint64_t)sizeof(LSSeqTimestepEntry);

         header.offset_part_meta      = off_part;
         header.offset_pattern_meta  = off_pattern;
         header.offset_sys_part_meta  = off_sys;
         header.offset_part_blob_table = off_part_blob_table;
         header.offset_timestep_meta  = off_time;
         header.offset_blob_data      = off_blob;
      }

      if (fwrite(&header, sizeof(header), 1, out) != 1 ||
          fwrite(&info_header, sizeof(info_header), 1, out) != 1 ||
          (info_payload_size > 0 &&
           fwrite(info_payload, 1, info_payload_size, out) != info_payload_size) ||
          fwrite(part_meta_global, sizeof(LSSeqPartMeta), (size_t)num_parts, out) !=
             (size_t)num_parts)
      {
         fprintf(stderr, "Could not write output metadata header/info/parts\n");
         MPI_Abort(comm, 1);
      }

      for (unsigned int i = 0; i < global_patterns_u; i++)
      {
         LSSeqPatternMeta meta = patterns_global[i].meta;
         if (meta.rows_blob_size > 0)
         {
            meta.rows_blob_offset += off_blob;
         }
         if (meta.cols_blob_size > 0)
         {
            meta.cols_blob_offset += off_blob;
         }
         if (fwrite(&meta, sizeof(meta), 1, out) != 1)
         {
            fprintf(stderr, "Could not write output pattern metadata\n");
            MPI_Abort(comm, 1);
         }
      }

      for (int i = 0; i < num_systems * num_parts; i++)
      {
         LSSeqSystemPartMeta meta = sys_meta_global[i];
         /* v2: values/rhs/dof are decompressed offsets; do not add off_blob */
         if (fwrite(&meta, sizeof(meta), 1, out) != 1)
         {
            fprintf(stderr, "Could not write output system-part metadata\n");
            MPI_Abort(comm, 1);
         }
      }

      if (fwrite(part_blob_table,
                 sizeof(uint64_t),
                 (size_t)num_parts * (size_t)LSSEQ_PART_BLOB_ENTRIES,
                 out) != (size_t)num_parts * (size_t)LSSEQ_PART_BLOB_ENTRIES)
      {
         fprintf(stderr, "Could not write part blob table\n");
         MPI_Abort(comm, 1);
      }
      free(part_blob_table);

      if (num_timesteps > 0 &&
          fwrite(timesteps, sizeof(LSSeqTimestepEntry), num_timesteps, out) != num_timesteps)
      {
         fprintf(stderr, "Could not write timestep metadata\n");
         MPI_Abort(comm, 1);
      }

      /* Write blob payloads in rank order (must match blob_base_ull offsets). */
      uint64_t blob_hash = UINT64_C(1469598103934665603);
      rewind(blob_fp);
      size_t nread = 0;
      while ((nread = fread(io_buf, 1, sizeof(io_buf), blob_fp)) > 0)
      {
         blob_hash = FNV1a64(io_buf, nread, blob_hash);
         if (fwrite(io_buf, 1, nread, out) != nread)
         {
            fprintf(stderr, "Could not write blob payload (rank 0)\n");
            MPI_Abort(comm, 1);
         }
      }

      for (int r = 1; r < nprocs; r++)
      {
         unsigned long long blob_bytes = 0;
         MPI_Recv(&blob_bytes, 1, MPI_UNSIGNED_LONG_LONG, r, 1001, comm, MPI_STATUS_IGNORE);
         unsigned long long remaining = blob_bytes;
         while (remaining > 0)
         {
            unsigned long long chunk = remaining > (unsigned long long)sizeof(io_buf)
                                          ? (unsigned long long)sizeof(io_buf)
                                          : remaining;
            MPI_Recv(io_buf, (int)chunk, MPI_BYTE, r, 1002, comm, MPI_STATUS_IGNORE);
            blob_hash = FNV1a64(io_buf, (size_t)chunk, blob_hash);
            if (fwrite(io_buf, 1, (size_t)chunk, out) != (size_t)chunk)
            {
               fprintf(stderr, "Could not write blob payload (rank %d)\n", r);
               MPI_Abort(comm, 1);
            }
            remaining -= chunk;
         }
      }

      /* Patch in the final blob hash after streaming all payloads. */
      info_header.blob_hash_fnv1a64 = blob_hash;
      if (fseeko(out, (off_t)(off_info + (uint64_t)offsetof(LSSeqInfoHeader, blob_hash_fnv1a64)),
                 SEEK_SET) != 0 ||
          fwrite(&info_header.blob_hash_fnv1a64, sizeof(info_header.blob_hash_fnv1a64), 1, out) !=
             1)
      {
         fprintf(stderr, "Could not update LSSeq info header blob hash\n");
         MPI_Abort(comm, 1);
      }

      fclose(out);
      free(timesteps);
      free(info_payload);

      {
         char hbuf[32];
         printf("Wrote sequence file: %s\n", output_filename);
         printf("  systems: %d\n", num_systems);
         printf("  parts: %d\n", num_parts);
         printf("  unique patterns: %u\n", global_patterns_u);
         printf("  codec: %s\n", hypredrv_compression_get_name(args.algo));
         if (args.algo == COMP_ZSTD)
         {
            int level = args.compression_level < 0 ? 5 : args.compression_level;
            printf("  compression level: %d\n", level);
         }
         FormatBytesHuman(global_blob_ull, hbuf, sizeof(hbuf));
         printf("  blob bytes: %llu (%s)\n", global_blob_ull, hbuf);
         FormatBytesHuman(global_raw_blob_ull, hbuf, sizeof(hbuf));
         printf("  raw blob bytes: %llu (%s)\n", global_raw_blob_ull, hbuf);
      }
      if (global_blob_ull > 0)
      {
         double ratio = (double)global_raw_blob_ull / (double)global_blob_ull;
         double savings = 0.0;
         if (global_raw_blob_ull > 0)
         {
            savings = 100.0 * (1.0 - ((double)global_blob_ull / (double)global_raw_blob_ull));
         }
         printf("  compression ratio (raw/packed): %.3fx\n", ratio);
         printf("  space savings: %.2f%%\n", savings);
      }
      fflush(stdout);
   }
   else
   {
      /* Non-root ranks stream their blob payloads to root for final assembly. */
      unsigned long long blob_bytes = local_blob_ull;
      MPI_Send(&blob_bytes, 1, MPI_UNSIGNED_LONG_LONG, 0, 1001, comm);
      rewind(blob_fp);

      uint8_t buf[1 << 20];
      unsigned long long remaining = blob_bytes;
      while (remaining > 0)
      {
         size_t want = remaining > (unsigned long long)sizeof(buf) ? sizeof(buf) : (size_t)remaining;
         size_t got  = fread(buf, 1, want, blob_fp);
         if (got != want)
         {
            fprintf(stderr, "[lsseq][pack][rank %d] Failed to read local blob for send\n", myid);
            MPI_Abort(comm, 1);
         }
         MPI_Send(buf, (int)got, MPI_BYTE, 0, 1002, comm);
         remaining -= (unsigned long long)got;
      }
   }

   fclose(blob_fp);
   free(part_meta_local);
   free(sys_meta_local);
   free(patterns_local);

   if (!myid)
   {
      free(part_meta_global);
      free(sys_meta_global);
      free(patterns_global);
      free(sys_meta_rankbuf);
      free(pat_counts);
      free(pat_displs);
      free(sys_counts);
      free(sys_displs);
   }

   MPI_Finalize();
   return EXIT_SUCCESS;
}
