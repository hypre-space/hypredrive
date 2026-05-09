/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/comp.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "HYPREDRV_config.h"
#include "internal/error.h"

#ifdef HYPREDRV_USING_ZLIB
#include <zlib.h>
#endif

#ifdef HYPREDRV_USING_ZSTD
#include <zstd.h>
#endif

#ifdef HYPREDRV_USING_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

#ifdef HYPREDRV_USING_BLOSC
#include <blosc.h>
#endif

/* Cap output allocation to mitigate malicious compressed payloads (CWE-789). */
#ifndef HYPREDRV_MAX_DECOMPRESSED_BYTES
#define HYPREDRV_MAX_DECOMPRESSED_BYTES ((size_t)16ULL * 1024ULL * 1024ULL * 1024ULL)
#endif

static int
HypredrvStringHasSuffix(const char *str, const char *suffix)
{
   size_t n = 0;
   size_t m = 0;

   /* Defensive: callers always pass non-NULL; only reachable via internal misuse. */
   /* GCOVR_EXCL_BR_START */
   if (!str || !suffix) /* GCOVR_EXCL_BR_STOP */
   {
      return 0;
   }
   n = strlen(str);
   m = strlen(suffix);
   if (m > n)
   {
      return 0;
   }
   return !strcmp(str + (n - m), suffix);
}

const char *
hypredrv_compression_get_name(comp_alg_t algo)
{
   switch (algo)
   {
      case COMP_NONE:
         return "none";
      case COMP_ZLIB:
         return "zlib";
      case COMP_ZSTD:
         return "zstd";
      case COMP_LZ4:
         return "lz4";
      case COMP_LZ4HC:
         return "lz4hc";
      case COMP_BLOSC:
         return "blosc";
      default:
         return "unknown";
   }
}

const char *
hypredrv_compression_get_extension(comp_alg_t algo)
{
   switch (algo)
   {
      case COMP_NONE:
         return ".bin";
      case COMP_ZLIB:
         return ".zlib.bin";
      case COMP_ZSTD:
         return ".zst.bin";
      case COMP_LZ4:
         return ".lz4.bin";
      case COMP_LZ4HC:
         return ".lz4hc.bin";
      case COMP_BLOSC:
         return ".blosc.bin";
      default:
         return ".bin";
   }
}

comp_alg_t
hypredrv_compression_from_filename(const char *filename)
{
   if (!filename || filename[0] == '\0')
   {
      return COMP_NONE;
   }

   if (HypredrvStringHasSuffix(filename, ".lz4hc.bin"))
   {
      return COMP_LZ4HC;
   }
   if (HypredrvStringHasSuffix(filename, ".zlib.bin"))
   {
      return COMP_ZLIB;
   }
   if (HypredrvStringHasSuffix(filename, ".zst.bin"))
   {
      return COMP_ZSTD;
   }
   if (HypredrvStringHasSuffix(filename, ".lz4.bin"))
   {
      return COMP_LZ4;
   }
   if (HypredrvStringHasSuffix(filename, ".blosc.bin"))
   {
      return COMP_BLOSC;
   }
   if (HypredrvStringHasSuffix(filename, ".bin"))
   {
      return COMP_NONE;
   }

   return COMP_NONE;
}

/*-----------------------------------------------------------------------------
 * Compression backends (per-algorithm; keep hypredrv_compress switch small)
 *-----------------------------------------------------------------------------*/

static int
compress_zlib(size_t isize, const void *input, size_t header_size, void **output_ptr,
              size_t *comp_size)
{
#ifdef HYPREDRV_USING_ZLIB
   *comp_size = (size_t)compressBound((uLong)isize);
   /* GCOVR_EXCL_BR_START */
   *output_ptr = malloc(header_size + *comp_size);
   if (*output_ptr == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + *comp_size);
      return 0;
   }
   /* GCOVR_EXCL_BR_STOP */

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

   uLongf zcomp_size = (uLongf)*comp_size;
   int ierr = compress((unsigned char *)(*output_ptr) + header_size, &zcomp_size, input,
                       (uLong)isize);
   /* GCOVR_EXCL_START */
   if (ierr != Z_OK)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ZLIB compression error: %d", ierr);
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *comp_size = (size_t)zcomp_size;
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)output_ptr;
   (void)comp_size;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("ZLIB compression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
compress_zstd(size_t isize, const void *input, size_t header_size, void **output_ptr,
              size_t *comp_size, int compression_level)
{
#ifdef HYPREDRV_USING_ZSTD
   *comp_size = ZSTD_compressBound(isize);
   /* GCOVR_EXCL_BR_START */
   *output_ptr = malloc(header_size + *comp_size);
   if (*output_ptr == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + *comp_size);
      return 0;
   }
   /* GCOVR_EXCL_BR_STOP */

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

   {
      int level = (compression_level < 0) ? 5 : compression_level;
      if (level < 1)
      {
         level = 1;
      }
      if (level > 22)
      {
         level = 22;
      }
      *comp_size = ZSTD_compress((unsigned char *)(*output_ptr) + header_size, *comp_size,
                                 input, isize, level);
   }
   /* GCOVR_EXCL_START */
   if (ZSTD_isError(*comp_size))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ZSTD compression error: %s", ZSTD_getErrorName(*comp_size));
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)output_ptr;
   (void)comp_size;
   (void)compression_level;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("ZSTD compression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
compress_lz4(size_t isize, const void *input, size_t header_size, void **output_ptr,
             size_t *comp_size)
{
#ifdef HYPREDRV_USING_LZ4
   /* GCOVR_EXCL_START */
   if (isize > (size_t)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LZ4 input is too large (%zu bytes)", isize);
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *comp_size  = (size_t)LZ4_compressBound((int)isize);
   *output_ptr = malloc(header_size + *comp_size);
   if (*output_ptr == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + *comp_size);
      return 0;
   }

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

   {
      int lz4_ret = LZ4_compress_default(input, (char *)(*output_ptr) + header_size,
                                         (int)isize, (int)*comp_size);
      /* GCOVR_EXCL_START */
      if (lz4_ret <= 0)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("LZ4 compression failed!");
         free(*output_ptr);
         *output_ptr = NULL;
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      *comp_size = (size_t)lz4_ret;
   }
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)output_ptr;
   (void)comp_size;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("LZ4 compression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
compress_lz4hc(size_t isize, const void *input, size_t header_size, void **output_ptr,
               size_t *comp_size)
{
#ifdef HYPREDRV_USING_LZ4
   /* GCOVR_EXCL_START */
   if (isize > (size_t)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LZ4HC input is too large (%zu bytes)", isize);
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *comp_size  = (size_t)LZ4_compressBound((int)isize);
   *output_ptr = malloc(header_size + *comp_size);
   if (*output_ptr == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + *comp_size);
      return 0;
   }

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

   {
      int lz4hc_ret =
         LZ4_compress_HC((const char *)input, (char *)(*output_ptr) + header_size,
                         (int)isize, (int)*comp_size, LZ4HC_CLEVEL_DEFAULT);
      /* GCOVR_EXCL_START */
      if (lz4hc_ret <= 0)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("LZ4HC compression failed!");
         free(*output_ptr);
         *output_ptr = NULL;
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      *comp_size = (size_t)lz4hc_ret;
   }
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)output_ptr;
   (void)comp_size;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("LZ4 compression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
compress_blosc(size_t isize, const void *input, size_t header_size, void **output_ptr,
               size_t *comp_size)
{
#ifdef HYPREDRV_USING_BLOSC
   blosc_init();
   blosc_set_compressor("blosclz");

   *comp_size  = isize + BLOSC_MAX_OVERHEAD;
   *output_ptr = malloc(header_size + *comp_size);
   if (*output_ptr == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + *comp_size);
      blosc_destroy();
      return 0;
   }

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

   {
      int blosc_ret = blosc_compress(
         9, 1, 1, isize, input, (unsigned char *)(*output_ptr) + header_size, *comp_size);
      blosc_destroy();
      /* GCOVR_EXCL_START */
      if (blosc_ret <= 0)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Blosc compression failed!");
         free(*output_ptr);
         *output_ptr = NULL;
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      *comp_size = (size_t)blosc_ret;
   }
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)output_ptr;
   (void)comp_size;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("BLOSC compression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_compress
 *-----------------------------------------------------------------------------*/

void
hypredrv_compress(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                  void **output_ptr, int compression_level)
{
   const size_t header_size = sizeof(uint64_t);
   size_t       comp_size   = 0;

   /* GCOVR_EXCL_BR_START */
   if (!osize_ptr || !output_ptr || (!input && isize > 0)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments to hypredrv_compress");
      return;
   }

   *osize_ptr  = 0;
   *output_ptr = NULL;

   if (algo == COMP_NONE)
   {
      /* GCOVR_EXCL_BR_START */
      size_t alloc_n = isize > 0 ? isize : 1;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, alloc_n);
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      if (isize > 0) /* GCOVR_EXCL_BR_STOP */
      {
         memcpy(*output_ptr, input, isize);
      }
      *osize_ptr = isize;
      return;
   }

   /* GCOVR_EXCL_BR_START */
   if (isize > (size_t)UINT64_MAX) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Input too large to be encoded (%zu bytes)", isize);
      return;
   }

#if !defined(HYPREDRV_USING_LZ4)
   if (algo == COMP_LZ4 || algo == COMP_LZ4HC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("LZ4 compression not enabled during build time!");
      return;
   }
#endif
#if !defined(HYPREDRV_USING_BLOSC)
   if (algo == COMP_BLOSC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("BLOSC compression not enabled during build time!");
      return;
   }
#endif

   /* GCOVR_EXCL_BR_START */
   switch (algo)
   /* GCOVR_EXCL_BR_STOP */
   {
      case COMP_ZLIB:
         if (!compress_zlib(isize, input, header_size, output_ptr, &comp_size))
         {
            return;
         }
         break;
      case COMP_ZSTD:
         if (!compress_zstd(isize, input, header_size, output_ptr, &comp_size,
                            compression_level))
         {
            return;
         }
         break;
      case COMP_LZ4: /* GCOVR_EXCL_LINE */
         if (!compress_lz4(isize, input, header_size, output_ptr, &comp_size))
         {
            return;
         }
         break;
      case COMP_LZ4HC: /* GCOVR_EXCL_LINE */
         if (!compress_lz4hc(isize, input, header_size, output_ptr, &comp_size))
         {
            return;
         }
         break;
      case COMP_BLOSC: /* GCOVR_EXCL_LINE */
         if (!compress_blosc(isize, input, header_size, output_ptr, &comp_size))
         {
            return;
         }
         break;
      default:
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown or unsupported compression algorithm: %d", algo);
         return;
      }
   }

   *osize_ptr = header_size + comp_size;

   return;
}

/*-----------------------------------------------------------------------------
 * Decompression backends (per-algorithm; keep hypredrv_decompress switch small)
 *-----------------------------------------------------------------------------*/

static int
decompress_zlib(size_t isize, const void *input, size_t header_size, size_t *orig_size,
                void **output_ptr)
{
#ifdef HYPREDRV_USING_ZLIB
   uLongf zorig_size = (uLongf)*orig_size;
   int    ierr =
      uncompress((unsigned char *)(*output_ptr), &zorig_size,
                 (unsigned char *)input + header_size, (uLong)(isize - header_size));
   /* GCOVR_EXCL_START */
   if (ierr != Z_OK)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ZLIB decompression error: %d", ierr);
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *orig_size = (size_t)zorig_size;
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)orig_size;
   (void)output_ptr;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("ZLIB decompression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
decompress_zstd(size_t isize, const void *input, size_t header_size, size_t orig_size,
                void **output_ptr)
{
#ifdef HYPREDRV_USING_ZSTD
   size_t result =
      ZSTD_decompress((unsigned char *)(*output_ptr), orig_size,
                      (unsigned char *)input + header_size, isize - header_size);
   /* GCOVR_EXCL_START */
   if (ZSTD_isError(result))
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ZSTD decompression error: %s", ZSTD_getErrorName(result));
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   (void)result;
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)orig_size;
   (void)output_ptr;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("ZSTD decompression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
decompress_lz4(size_t isize, const void *input, size_t header_size, size_t orig_size,
               void **output_ptr)
{
#ifdef HYPREDRV_USING_LZ4
   /* GCOVR_EXCL_START */
   if ((isize - header_size) > (size_t)INT_MAX || orig_size > (size_t)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LZ4 payload too large for API limits");
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   int result =
      LZ4_decompress_safe((const char *)input + header_size, (char *)(*output_ptr),
                          (int)(isize - header_size), (int)orig_size);
   /* GCOVR_EXCL_START */
   if (result < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("LZ4 decompression failed!");
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   (void)result;
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)orig_size;
   (void)output_ptr;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("LZ4 decompression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

static int
decompress_blosc(size_t isize, const void *input, size_t header_size, size_t orig_size,
                 void **output_ptr)
{
#ifdef HYPREDRV_USING_BLOSC
   (void)isize;
   blosc_init();

   int result = blosc_decompress((const void *)((unsigned char *)input + header_size),
                                 *output_ptr, orig_size);
   blosc_destroy();
   /* GCOVR_EXCL_START */
   if (result <= 0)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Blosc decompression failed!");
      free(*output_ptr);
      *output_ptr = NULL;
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   (void)result;
   return 1;
#else
   /* GCOVR_EXCL_START */
   (void)isize;
   (void)input;
   (void)header_size;
   (void)orig_size;
   (void)output_ptr;
   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("BLOSC decompression not enabled during build time!");
   return 0;
   /* GCOVR_EXCL_STOP */
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_decompress
 *-----------------------------------------------------------------------------*/

void
hypredrv_decompress(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                    void **output_ptr)
{
   const size_t header_size = sizeof(uint64_t);
   size_t       orig_size   = 0;

   /* GCOVR_EXCL_BR_START */
   if (!osize_ptr || !output_ptr || (!input && isize > 0)) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments to hypredrv_decompress");
      return;
   }

   *osize_ptr  = 0;
   *output_ptr = NULL;

   if (algo == COMP_NONE)
   {
      /* GCOVR_EXCL_BR_START */
      if (isize > HYPREDRV_MAX_DECOMPRESSED_BYTES)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Uncompressed payload size exceeds maximum (%zu bytes)",
                              (size_t)HYPREDRV_MAX_DECOMPRESSED_BYTES);
         return;
      }
      size_t alloc_n = isize > 0 ? isize : 1;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, alloc_n);
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      if (isize > 0) /* GCOVR_EXCL_BR_STOP */
      {
         memcpy(*output_ptr, input, isize);
      }
      *osize_ptr = isize;
      return;
   }

   if (isize < header_size)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Compressed buffer too small (%zu bytes)", isize);
      return;
   }

   orig_size = (size_t)(*((const uint64_t *)input));
   if (orig_size > HYPREDRV_MAX_DECOMPRESSED_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Declared decompressed size exceeds maximum (%zu bytes)",
                           (size_t)HYPREDRV_MAX_DECOMPRESSED_BYTES);
      return;
   }

#if !defined(HYPREDRV_USING_LZ4)
   if (algo == COMP_LZ4 || algo == COMP_LZ4HC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("LZ4 decompression not enabled during build time!");
      return;
   }
#endif
#if !defined(HYPREDRV_USING_BLOSC)
   if (algo == COMP_BLOSC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("BLOSC decompression not enabled during build time!");
      return;
   }
#endif

   {
      /* GCOVR_EXCL_BR_START */
      size_t alloc_n = orig_size > 0 ? orig_size : 1;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, alloc_n);
      /* GCOVR_EXCL_BR_STOP */
   }

   /* GCOVR_EXCL_BR_START */
   switch (algo)
   /* GCOVR_EXCL_BR_STOP */
   {
      case COMP_ZLIB:
         if (!decompress_zlib(isize, input, header_size, &orig_size, output_ptr))
         {
            return;
         }
         break;
      case COMP_ZSTD:
         if (!decompress_zstd(isize, input, header_size, orig_size, output_ptr))
         {
            return;
         }
         break;
      case COMP_LZ4:   /* GCOVR_EXCL_LINE */
      case COMP_LZ4HC: /* GCOVR_EXCL_LINE */
         if (!decompress_lz4(isize, input, header_size, orig_size, output_ptr))
         {
            return;
         }
         break;
      case COMP_BLOSC: /* GCOVR_EXCL_LINE */
         if (!decompress_blosc(isize, input, header_size, orig_size, output_ptr))
         {
            return;
         }
         break;
      default:
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown or unsupported decompression algorithm: %d", algo);
         free(*output_ptr);
         *output_ptr = NULL;
         return;
      }
   }

   *osize_ptr = orig_size;

   return;
}
