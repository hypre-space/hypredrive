/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "comp.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"

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

static int
HypredrvStringHasSuffix(const char *str, const char *suffix)
{
   size_t n = 0;
   size_t m = 0;

   if (!str || !suffix)
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
 * hypredrv_compress
 *-----------------------------------------------------------------------------*/

void
hypredrv_compress(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                  void **output_ptr, int compression_level)
{
   (void)compression_level; /* used only for ZSTD and optionally other codecs */
   const size_t header_size = sizeof(uint64_t);
   size_t       comp_size   = 0;

   if (!osize_ptr || !output_ptr || (!input && isize > 0))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments to hypredrv_compress");
      return;
   }

   *osize_ptr  = 0;
   *output_ptr = NULL;

   if (algo == COMP_NONE)
   {
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, isize > 0 ? isize : 1);
      if (isize > 0)
      {
         memcpy(*output_ptr, input, isize);
      }
      *osize_ptr = isize;
      return;
   }

   if (isize > (size_t)UINT64_MAX)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Input too large to be encoded (%zu bytes)", isize);
      return;
   }

   switch (algo)
   {
      case COMP_ZLIB:
      {
#ifdef HYPREDRV_USING_ZLIB
         comp_size = (size_t)compressBound((uLong)isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

         uLongf zcomp_size = (uLongf)comp_size;
         int    ierr = compress((unsigned char *)(*output_ptr) + header_size, &zcomp_size,
                                input, (uLong)isize);
         if (ierr != Z_OK)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZLIB compression error: %d", ierr);
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
         comp_size = (size_t)zcomp_size;
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("ZLIB compression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_ZSTD:
      {
#ifdef HYPREDRV_USING_ZSTD
         comp_size = ZSTD_compressBound(isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

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
            comp_size = ZSTD_compress((unsigned char *)(*output_ptr) + header_size,
                                      comp_size, input, isize, level);
         }
         if (ZSTD_isError(comp_size))
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZSTD compression error: %s", ZSTD_getErrorName(comp_size));
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("ZSTD compression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_LZ4:
      {
#ifdef HYPREDRV_USING_LZ4
         if (isize > (size_t)INT_MAX)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("LZ4 input is too large (%zu bytes)", isize);
            return;
         }
         comp_size = (size_t)LZ4_compressBound((int)isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

         {
            int lz4_ret = LZ4_compress_default(input, (char *)(*output_ptr) + header_size,
                                               (int)isize, (int)comp_size);
            if (lz4_ret <= 0)
            {
               ErrorCodeSet(ERROR_UNKNOWN);
               ErrorMsgAdd("LZ4 compression failed!");
               free(*output_ptr);
               *output_ptr = NULL;
               return;
            }
            comp_size = (size_t)lz4_ret;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("LZ4 compression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_LZ4HC:
      {
#ifdef HYPREDRV_USING_LZ4
         if (isize > (size_t)INT_MAX)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("LZ4HC input is too large (%zu bytes)", isize);
            return;
         }
         comp_size = (size_t)LZ4_compressBound((int)isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

         {
            int lz4hc_ret =
               LZ4_compress_HC((const char *)input, (char *)(*output_ptr) + header_size,
                               (int)isize, (int)comp_size, LZ4HC_CLEVEL_DEFAULT);
            if (lz4hc_ret <= 0)
            {
               ErrorCodeSet(ERROR_UNKNOWN);
               ErrorMsgAdd("LZ4HC compression failed!");
               free(*output_ptr);
               *output_ptr = NULL;
               return;
            }
            comp_size = (size_t)lz4hc_ret;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("LZ4 compression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_BLOSC:
      {
#ifdef HYPREDRV_USING_BLOSC
         blosc_init();
         blosc_set_compressor("blosclz");

         comp_size = isize + BLOSC_MAX_OVERHEAD;
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         *((uint64_t *)(*output_ptr)) = (uint64_t)isize;

         {
            int blosc_ret =
               blosc_compress(9, 1, 1, isize, input,
                              (unsigned char *)(*output_ptr) + header_size, comp_size);
            blosc_destroy();
            if (blosc_ret <= 0)
            {
               ErrorCodeSet(ERROR_UNKNOWN);
               ErrorMsgAdd("Blosc compression failed!");
               free(*output_ptr);
               *output_ptr = NULL;
               return;
            }
            comp_size = (size_t)blosc_ret;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("BLOSC compression not enabled during build time!");
         return;
#endif
         break;
      }
      default:
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("Unknown or unsupported compression algorithm: %d", algo);
         return;
      }
   }

   *osize_ptr = header_size + comp_size;

   return;
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

   if (!osize_ptr || !output_ptr || (!input && isize > 0))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments to hypredrv_decompress");
      return;
   }

   *osize_ptr  = 0;
   *output_ptr = NULL;

   if (algo == COMP_NONE)
   {
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, isize > 0 ? isize : 1);
      if (isize > 0)
      {
         memcpy(*output_ptr, input, isize);
      }
      *osize_ptr = isize;
      return;
   }

   if (isize < header_size)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Compressed buffer too small (%zu bytes)", isize);
      return;
   }

   orig_size = (size_t)(*((const uint64_t *)input));
   HYPREDRV_MALLOC_AND_CHECK(*output_ptr, orig_size > 0 ? orig_size : 1);

   switch (algo)
   {
      case COMP_ZLIB:
      {
#ifdef HYPREDRV_USING_ZLIB
         uLongf zorig_size = (uLongf)orig_size;
         int    ierr       = uncompress((unsigned char *)(*output_ptr), &zorig_size,
                                        (unsigned char *)input + header_size,
                                        (uLong)(isize - header_size));
         if (ierr != Z_OK)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZLIB decompression error: %d", ierr);
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
         orig_size = (size_t)zorig_size;
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("ZLIB decompression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_ZSTD:
      {
#ifdef HYPREDRV_USING_ZSTD
         size_t result =
            ZSTD_decompress((unsigned char *)(*output_ptr), orig_size,
                            (unsigned char *)input + header_size, isize - header_size);
         if (ZSTD_isError(result))
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZSTD decompression error: %s", ZSTD_getErrorName(result));
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("ZSTD decompression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_LZ4:
      case COMP_LZ4HC:
      {
#ifdef HYPREDRV_USING_LZ4
         if ((isize - header_size) > (size_t)INT_MAX || orig_size > (size_t)INT_MAX)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("LZ4 payload too large for API limits");
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
         int result =
            LZ4_decompress_safe((const char *)input + header_size, (char *)(*output_ptr),
                                (int)(isize - header_size), (int)orig_size);
         if (result < 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("LZ4 decompression failed!");
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("LZ4 decompression not enabled during build time!");
         return;
#endif
         break;
      }
      case COMP_BLOSC:
      {
#ifdef HYPREDRV_USING_BLOSC
         blosc_init();

         int result = blosc_decompress(
            (const void *)((unsigned char *)input + header_size), *output_ptr, orig_size);
         blosc_destroy();
         if (result <= 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("Blosc decompression failed!");
            free(*output_ptr);
            *output_ptr = NULL;
            return;
         }
#else
         ErrorCodeSet(ERROR_MISSING_LIB);
         ErrorMsgAdd("BLOSC decompression not enabled during build time!");
         return;
#endif
         break;
      }
      default:
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("Unknown or unsupported decompression algorithm: %d", algo);
         free(*output_ptr);
         *output_ptr = NULL;
         return;
      }
   }

   *osize_ptr = orig_size;

   return;
}
