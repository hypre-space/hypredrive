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
#include "internal/utils.h"

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
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
         return "lz4";
      case COMP_LZ4HC:
         return "lz4hc";
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
         return "blosc";
#endif
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
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
         return ".lz4.bin";
      case COMP_LZ4HC:
         return ".lz4hc.bin";
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
         return ".blosc.bin";
#endif
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

/* Allocates a compressed blob of header_size + bound bytes and records the
 * uncompressed size in its leading uint64_t. Returns 0 on allocation failure. */
static HYPREDRV_MAYBE_UNUSED int
CompressAllocOutput(size_t isize, size_t header_size, size_t bound, void **output_ptr)
{
   *output_ptr = malloc(header_size + bound);
   /* GCOVR_EXCL_BR_START */
   if (*output_ptr == NULL) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Memory allocation failed at %s:%d (%zu bytes)", __FILE__,
                           __LINE__, header_size + bound);
      return 0;
   }

   *((uint64_t *)(*output_ptr)) = (uint64_t)isize;
   return 1;
}

static int
compress_zlib(size_t isize, const void *input, size_t header_size, void **output_ptr,
              size_t *comp_size)
{
#ifdef HYPREDRV_USING_ZLIB
   *comp_size = (size_t)compressBound((uLong)isize);
   if (!CompressAllocOutput(isize, header_size, *comp_size, output_ptr))
   {
      return 0;
   }

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
   if (!CompressAllocOutput(isize, header_size, *comp_size, output_ptr))
   {
      return 0;
   }

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

#ifdef HYPREDRV_USING_LZ4
static int
compress_lz4(size_t isize, const void *input, size_t header_size, void **output_ptr,
             size_t *comp_size)
{
   /* GCOVR_EXCL_START */
   if (isize > (size_t)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LZ4 input is too large (%zu bytes)", isize);
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *comp_size = (size_t)LZ4_compressBound((int)isize);
   if (!CompressAllocOutput(isize, header_size, *comp_size, output_ptr))
   {
      return 0;
   }

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
}

static int
compress_lz4hc(size_t isize, const void *input, size_t header_size, void **output_ptr,
               size_t *comp_size)
{
   /* GCOVR_EXCL_START */
   if (isize > (size_t)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("LZ4HC input is too large (%zu bytes)", isize);
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   *comp_size = (size_t)LZ4_compressBound((int)isize);
   if (!CompressAllocOutput(isize, header_size, *comp_size, output_ptr))
   {
      return 0;
   }

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
}
#endif

#ifdef HYPREDRV_USING_BLOSC
static int
compress_blosc(size_t isize, const void *input, size_t header_size, void **output_ptr,
               size_t *comp_size)
{
   blosc_init();
   blosc_set_compressor("blosclz");

   *comp_size = isize + BLOSC_MAX_OVERHEAD;
   if (!CompressAllocOutput(isize, header_size, *comp_size, output_ptr))
   {
      blosc_destroy();
      return 0;
   }

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
}
#endif

/*-----------------------------------------------------------------------------
 * hypredrv_compress
 *-----------------------------------------------------------------------------*/

/* Algorithms whose backing library was not compiled in are rejected up front, so
 * the dispatch switches below only ever see something they can handle. */
static int
CompressionAlgorithmAvailable(comp_alg_t algo, const char *direction)
{
#if !defined(HYPREDRV_USING_LZ4)
   if (algo == COMP_LZ4 || algo == COMP_LZ4HC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("LZ4 %s not enabled during build time!", direction);
      return 0;
   }
#endif
#if !defined(HYPREDRV_USING_BLOSC)
   if (algo == COMP_BLOSC)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
      hypredrv_ErrorMsgAdd("BLOSC %s not enabled during build time!", direction);
      return 0;
   }
#endif

   (void)algo;
   (void)direction;

   return 1;
}

/* Hands the payload to the compression backend matching `algo`. */
static int
CompressDispatch(comp_alg_t algo, size_t isize, const void *input, size_t header_size,
                 int compression_level, void **output_ptr, size_t *comp_size)
{
   /* GCOVR_EXCL_BR_START */
   switch (algo)
   /* GCOVR_EXCL_BR_STOP */
   {
      case COMP_ZLIB:
         if (!compress_zlib(isize, input, header_size, output_ptr, comp_size))
         {
            return 0;
         }
         break;
      case COMP_ZSTD:
         if (!compress_zstd(isize, input, header_size, output_ptr, comp_size,
                            compression_level))
         {
            return 0;
         }
         break;
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4: /* GCOVR_EXCL_LINE */
         if (!compress_lz4(isize, input, header_size, output_ptr, comp_size))
         {
            return 0;
         }
         break;
      case COMP_LZ4HC: /* GCOVR_EXCL_LINE */
         if (!compress_lz4hc(isize, input, header_size, output_ptr, comp_size))
         {
            return 0;
         }
         break;
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC: /* GCOVR_EXCL_LINE */
         if (!compress_blosc(isize, input, header_size, output_ptr, comp_size))
         {
            return 0;
         }
         break;
#endif
      default:
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown or unsupported compression algorithm: %d", algo);
         return 0;
      }
   }

   return 1;
}

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

   if (!CompressionAlgorithmAvailable(algo, "compression"))
   {
      return;
   }

   if (!CompressDispatch(algo, isize, input, header_size, compression_level, output_ptr,
                         &comp_size))
   {
      return;
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

#ifdef HYPREDRV_USING_LZ4
static int
decompress_lz4(size_t isize, const void *input, size_t header_size, size_t orig_size,
               void **output_ptr)
{
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
}
#endif

#ifdef HYPREDRV_USING_BLOSC
static int
decompress_blosc(size_t isize, const void *input, size_t header_size, size_t orig_size,
                 void **output_ptr)
{
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
}
#endif

/*-----------------------------------------------------------------------------
 * hypredrv_decompress
 *-----------------------------------------------------------------------------*/

/* Hands the payload to the decompression backend matching `algo`. */
static int
DecompressDispatch(comp_alg_t algo, size_t isize, const void *input, size_t header_size,
                   size_t *orig_size, void **output_ptr)
{
   /* GCOVR_EXCL_BR_START */
   switch (algo)
   /* GCOVR_EXCL_BR_STOP */
   {
      case COMP_ZLIB:
         if (!decompress_zlib(isize, input, header_size, orig_size, output_ptr))
         {
            return 0;
         }
         break;
      case COMP_ZSTD:
         if (!decompress_zstd(isize, input, header_size, *orig_size, output_ptr))
         {
            return 0;
         }
         break;
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:   /* GCOVR_EXCL_LINE */
      case COMP_LZ4HC: /* GCOVR_EXCL_LINE */
         if (!decompress_lz4(isize, input, header_size, *orig_size, output_ptr))
         {
            return 0;
         }
         break;
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC: /* GCOVR_EXCL_LINE */
         if (!decompress_blosc(isize, input, header_size, *orig_size, output_ptr))
         {
            return 0;
         }
         break;
#endif
      default:
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown or unsupported decompression algorithm: %d", algo);
         free(*output_ptr);
         *output_ptr = NULL;
         return 0;
      }
   }

   return 1;
}

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

   /* Read the 8-byte size header via memcpy rather than a pointer cast: the input
    * buffer is not guaranteed to be 8-byte aligned, and a misaligned load is
    * undefined behavior on strict-alignment targets. */
   {
      uint64_t hdr = 0;
      memcpy(&hdr, input, sizeof(hdr));
      orig_size = (size_t)hdr;
   }
   if (orig_size > HYPREDRV_MAX_DECOMPRESSED_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Declared decompressed size exceeds maximum (%zu bytes)",
                           (size_t)HYPREDRV_MAX_DECOMPRESSED_BYTES);
      return;
   }

   if (!CompressionAlgorithmAvailable(algo, "decompression"))
   {
      return;
   }

   {
      /* GCOVR_EXCL_BR_START */
      size_t alloc_n = orig_size > 0 ? orig_size : 1;
      /* GCOVR_EXCL_BR_STOP */
      /* GCOVR_EXCL_BR_START */
      HYPREDRV_MALLOC_AND_CHECK(*output_ptr, alloc_n);
      /* GCOVR_EXCL_BR_STOP */
   }

   if (!DecompressDispatch(algo, isize, input, header_size, &orig_size, output_ptr))
   {
      return;
   }

   *osize_ptr = orig_size;

   return;
}

/*-----------------------------------------------------------------------------
 * Resumable slice decoding (zstd/zlib)
 *
 * A slice stream keeps a copy of one compressed blob plus its decoder state and
 * the number of original bytes produced so far. Reading [offset, offset+size)
 * continues from the current position when offset is at or past it (bytes
 * before the slice go through a small scratch window; the slice itself is
 * decoded straight into the output) and restarts the decoder otherwise, so
 * reading a blob's slices in increasing order decodes it only once.
 *-----------------------------------------------------------------------------*/

struct hypredrv_SliceStream_struct
{
   comp_alg_t     algo;
   unsigned char *blob; /* full compressed blob, including the size header */
   size_t         blob_size;
   size_t         orig_size;
   size_t         pos; /* original bytes produced so far */
   unsigned char *window;
   size_t         window_size;
   int            started;
#ifdef HYPREDRV_USING_ZSTD
   ZSTD_DCtx    *zstd;
   ZSTD_inBuffer zin;
#endif
#ifdef HYPREDRV_USING_ZLIB
   z_stream zs;
#endif
};

static void
SliceStreamEnd(hypredrv_SliceStream *st)
{
   if (!st->started)
   {
      return;
   }
#ifdef HYPREDRV_USING_ZSTD
   if (st->algo == COMP_ZSTD)
   {
      ZSTD_freeDCtx(st->zstd);
      st->zstd = NULL;
   }
#endif
#ifdef HYPREDRV_USING_ZLIB
   if (st->algo == COMP_ZLIB)
   {
      inflateEnd(&st->zs);
   }
#endif
   st->started = 0;
   st->pos     = 0;
}

/* (Re)starts decoding from the beginning of the payload. */
static int
SliceStreamStart(hypredrv_SliceStream *st)
{
   const size_t header_size = sizeof(uint64_t);

   (void)header_size; /* unused when no streaming codec is enabled */
   SliceStreamEnd(st);
#ifdef HYPREDRV_USING_ZSTD
   if (st->algo == COMP_ZSTD)
   {
      st->zstd = ZSTD_createDCtx();
      st->zin  = (ZSTD_inBuffer){st->blob + header_size, st->blob_size - header_size, 0};
      st->started = (st->zstd != NULL);
   }
#endif
#ifdef HYPREDRV_USING_ZLIB
   if (st->algo == COMP_ZLIB)
   {
      memset(&st->zs, 0, sizeof(st->zs));
      st->zs.next_in  = st->blob + header_size;
      st->zs.avail_in = (uInt)(st->blob_size - header_size);
      st->started     = (inflateInit(&st->zs) == Z_OK);
   }
#endif
   return st->started;
}

/* Decodes up to `avail` bytes into `target`; returns the count (0 = no progress). */
static size_t
SliceStreamDecode(hypredrv_SliceStream *st, unsigned char *target, size_t avail)
{
#ifdef HYPREDRV_USING_ZSTD
   if (st->algo == COMP_ZSTD)
   {
      ZSTD_outBuffer o   = {target, avail, 0};
      size_t         ret = ZSTD_decompressStream(st->zstd, &o, &st->zin);
      return ZSTD_isError(ret) ? 0 : o.pos; /* GCOVR_EXCL_BR_LINE */
   }
#endif
#ifdef HYPREDRV_USING_ZLIB
   if (st->algo == COMP_ZLIB)
   {
      st->zs.next_out  = target;
      st->zs.avail_out = (uInt)avail; /* slices stay below LSSEQ blob limits */
      int ret          = inflate(&st->zs, Z_NO_FLUSH);
      /* GCOVR_EXCL_BR_START */
      return (ret == Z_OK || ret == Z_STREAM_END || ret == Z_BUF_ERROR)
                ? avail - st->zs.avail_out
                : 0;
      /* GCOVR_EXCL_BR_STOP */
   }
#endif
   (void)st;
   (void)target;
   (void)avail;
   return 0;
}

hypredrv_SliceStream *
hypredrv_SliceStreamCreate(comp_alg_t algo, size_t isize, const void *input)
{
   hypredrv_SliceStream *st        = NULL;
   uint64_t              orig      = 0;
   int                   resumable = 0;

#ifdef HYPREDRV_USING_ZSTD
   resumable |= (algo == COMP_ZSTD);
#endif
#ifdef HYPREDRV_USING_ZLIB
   resumable |= (algo == COMP_ZLIB);
#endif
   if (!resumable || !input || isize < sizeof(uint64_t))
   {
      return NULL;
   }
   memcpy(&orig, input, sizeof(orig));

   st = (hypredrv_SliceStream *)calloc(1, sizeof(*st));
   if (!st) /* GCOVR_EXCL_BR_LINE */
   {
      return NULL; /* GCOVR_EXCL_LINE */
   }
   st->algo        = algo;
   st->blob_size   = isize;
   st->orig_size   = (size_t)orig;
   st->window_size = (size_t)1 << 17;
   st->blob        = (unsigned char *)malloc(isize);
   st->window      = (unsigned char *)malloc(st->window_size);
   if (!st->blob || !st->window) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_SliceStreamDestroy(&st); /* GCOVR_EXCL_LINE */
      return NULL;                      /* GCOVR_EXCL_LINE */
   }
   memcpy(st->blob, input, isize);
   return st;
}

int
hypredrv_SliceStreamRead(hypredrv_SliceStream *st, size_t offset, size_t size,
                         void **output)
{
   unsigned char *out = NULL;

   *output = NULL;
   /* GCOVR_EXCL_BR_START */
   if (offset > st->orig_size || size > st->orig_size - offset) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Requested slice [%zu, +%zu) exceeds payload (%zu bytes)",
                           offset, size, st->orig_size);
      return 0;
   }
   if (size == 0)
   {
      return 1;
   }
   out = (unsigned char *)malloc(size);
   if (!out) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate slice buffer (%zu bytes)", size);
      return 0;
   }

   /* Resume when the slice starts at or after the current position. */
   if ((!st->started || offset < st->pos) && !SliceStreamStart(st))
   {
      free(out);
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("%s decoder initialization failed",
                           hypredrv_compression_get_name(st->algo));
      return 0;
   }

   while (st->pos < offset + size)
   {
      /* Skip [pos, offset) through the window, capped so it never crosses
       * into the slice; then decode the slice directly into `out`. */
      size_t         avail  = 0;
      unsigned char *target = NULL;
      if (st->pos < offset)
      {
         avail =
            (offset - st->pos < st->window_size) ? offset - st->pos : st->window_size;
         target = st->window;
      }
      else
      {
         avail  = offset + size - st->pos;
         target = out + (st->pos - offset);
      }
      size_t n = SliceStreamDecode(st, target, avail);
      if (n == 0)
      {
         break;
      }
      st->pos += n;
   }

   if (st->pos < offset + size)
   {
      free(out);
      SliceStreamEnd(st); /* state is unusable after a decode failure */
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("%s decompression of slice failed",
                           hypredrv_compression_get_name(st->algo));
      return 0;
   }

   *output = out;
   return 1;
}

void
hypredrv_SliceStreamDestroy(hypredrv_SliceStream **st_ptr)
{
   if (!st_ptr || !*st_ptr)
   {
      return;
   }
   SliceStreamEnd(*st_ptr);
   free((*st_ptr)->blob);
   free((*st_ptr)->window);
   free(*st_ptr);
   *st_ptr = NULL;
}

/*-----------------------------------------------------------------------------
 * hypredrv_decompress_slice
 *
 * Decompresses only bytes [offset, offset + size) of a blob produced by
 * hypredrv_compress into a newly allocated *output (NULL when size is 0).
 * zstd and zlib decode through a slice stream that stops at the end of the
 * slice; other codecs fall back to a full decode. Returns 1 on success, 0 with
 * the error state set.
 *-----------------------------------------------------------------------------*/

int
hypredrv_decompress_slice(comp_alg_t algo, size_t isize, const void *input, size_t offset,
                          size_t size, void **output)
{
   hypredrv_SliceStream *st        = NULL;
   uint64_t              orig_size = 0;
   int                   ok        = 0;

   *output = NULL;
   if (algo == COMP_NONE)
   {
      orig_size = (uint64_t)isize;
   }
   else if (isize >= sizeof(uint64_t) && input)
   {
      memcpy(&orig_size, input, sizeof(orig_size));
   }
   /* GCOVR_EXCL_BR_START */
   if ((algo != COMP_NONE && isize < sizeof(uint64_t)) || offset > orig_size ||
       size > orig_size - offset) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Requested slice [%zu, +%zu) exceeds payload (%llu bytes)",
                           offset, size, (unsigned long long)orig_size);
      return 0;
   }
   if (size == 0)
   {
      return 1;
   }

   if (algo == COMP_NONE)
   {
      *output = malloc(size);
      if (!*output) /* GCOVR_EXCL_BR_LINE */
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate slice buffer (%zu bytes)", size);
         return 0;
      }
      memcpy(*output, (const unsigned char *)input + offset, size);
      return 1;
   }

   st = hypredrv_SliceStreamCreate(algo, isize, input);
   if (st)
   {
      ok = hypredrv_SliceStreamRead(st, offset, size, output);
      hypredrv_SliceStreamDestroy(&st);
      return ok;
   }

   /* Codecs without streaming support: full decode, then copy the slice. */
   {
      void  *full      = NULL;
      size_t full_size = 0;
      hypredrv_decompress(algo, isize, input, &full_size, &full);
      ok = full && offset + size <= full_size;
      if (ok)
      {
         *output = malloc(size);
         ok      = (*output != NULL);
      }
      if (ok)
      {
         memcpy(*output, (const unsigned char *)full + offset, size);
      }
      free(full);
   }
   if (!ok && !hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("%s decompression of slice failed",
                           hypredrv_compression_get_name(algo));
   }
   return ok;
}
