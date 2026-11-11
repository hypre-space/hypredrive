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

/* Validate codec availability at the operation boundary, before allocating
 * output or passing sizes through a codec's narrower integer interface. */
static int
CompressionAlgorithmAvailable(comp_alg_t algo, const char *direction)
{
   const char *missing = NULL;
   switch (algo)
   {
      case COMP_NONE:
         return 1;
      case COMP_ZLIB:
#ifdef HYPREDRV_USING_ZLIB
         return 1;
#else
         missing = "ZLIB";
         break;
#endif
      case COMP_ZSTD:
#ifdef HYPREDRV_USING_ZSTD
         return 1;
#else
         missing = "ZSTD";
         break;
#endif
      case COMP_LZ4:
      case COMP_LZ4HC:
#ifdef HYPREDRV_USING_LZ4
         return 1;
#else
         missing = "LZ4";
         break;
#endif
      case COMP_BLOSC:
#ifdef HYPREDRV_USING_BLOSC
         return 1;
#else
         missing = "BLOSC";
         break;
#endif
      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown or unsupported %s algorithm: %d", direction, algo);
         return 0;
   }

   hypredrv_ErrorCodeSet(ERROR_MISSING_LIB);
   hypredrv_ErrorMsgAdd("%s %s not enabled during build time!", missing, direction);
   return 0;
}

/* Clears whichever outputs were supplied, then rejects missing outputs or a
 * missing non-empty input. */
static int
CodecArgumentsValid(size_t isize, const void *input, size_t *osize_ptr, void **output_ptr,
                    const char *caller)
{
   if (osize_ptr)
   {
      *osize_ptr = 0;
   }
   if (output_ptr)
   {
      *output_ptr = NULL;
   }
   if (!osize_ptr || !output_ptr || (!input && isize > 0))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments to %s", caller);
      return 0;
   }
   return 1;
}

/* Copies `size` bytes at `base + offset` into a new buffer of at least one byte. */
static void *
CopyUncompressed(const void *base, size_t offset, size_t size)
{
   unsigned char *copy = hypredrv_Malloc(size ? size : 1, "uncompressed buffer");
   if (copy && size)
   {
      memcpy(copy, (const unsigned char *)base + offset, size);
   }
   return copy;
}

static void
CompressSizeError(comp_alg_t algo, size_t isize)
{
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd("Input size exceeds %s compression limits (%zu bytes)",
                        hypredrv_compression_get_name(algo), isize);
}

/* Stores the worst-case compressed size of `isize` bytes in *bound; returns 0
 * when the codec cannot take that many. */
static int
CompressBound(comp_alg_t algo, size_t isize, size_t *bound)
{
   (void)isize;
   (void)bound;
   switch (algo)
   {
#ifdef HYPREDRV_USING_ZLIB
      case COMP_ZLIB:
         if (isize > ULONG_MAX)
         {
            return 0;
         }
         *bound = (size_t)compressBound((uLong)isize);
         return 1;
#endif
#ifdef HYPREDRV_USING_ZSTD
      case COMP_ZSTD:
         *bound = ZSTD_compressBound(isize);
         return !ZSTD_isError(*bound);
#endif
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
      case COMP_LZ4HC:
         if (isize > LZ4_MAX_INPUT_SIZE)
         {
            return 0;
         }
         *bound = (size_t)LZ4_compressBound((int)isize);
         return 1;
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
         if (isize > BLOSC_MAX_BUFFERSIZE)
         {
            return 0;
         }
         *bound = isize + BLOSC_MAX_OVERHEAD;
         return 1;
#endif
      default:
         return 1; /* Availability was checked by the caller. */
   }
}

/* Each backend compresses `isize` bytes into `dest` (capacity `bound`). On
 * failure it adds its message and returns 0; the caller sets the error code. */

#ifdef HYPREDRV_USING_ZLIB
static int
CompressZlib(const void *input, size_t isize, unsigned char *dest, size_t bound,
             size_t *comp_size)
{
   uLongf size  = (uLongf)bound;
   int    error = compress(dest, &size, input, (uLong)isize);
   if (error != Z_OK)
   {
      hypredrv_ErrorMsgAdd("ZLIB compression error: %d", error);
      return 0;
   }
   *comp_size = (size_t)size;
   return 1;
}
#endif

#ifdef HYPREDRV_USING_ZSTD
static int
CompressZstd(const void *input, size_t isize, unsigned char *dest, size_t bound,
             int compression_level, size_t *comp_size)
{
   int level = compression_level < 0 ? 5 : compression_level;
   if (level < 1) level = 1;
   if (level > 22) level = 22;
   *comp_size = ZSTD_compress(dest, bound, input, isize, level);
   if (ZSTD_isError(*comp_size))
   {
      hypredrv_ErrorMsgAdd("ZSTD compression error: %s", ZSTD_getErrorName(*comp_size));
      return 0;
   }
   return 1;
}
#endif

#ifdef HYPREDRV_USING_LZ4
static int
CompressLz4(comp_alg_t algo, const void *input, size_t isize, unsigned char *dest,
            size_t bound, size_t *comp_size)
{
   int size = algo == COMP_LZ4
                 ? LZ4_compress_default(input, (char *)dest, (int)isize, (int)bound)
                 : LZ4_compress_HC(input, (char *)dest, (int)isize, (int)bound,
                                   LZ4HC_CLEVEL_DEFAULT);
   if (size <= 0)
   {
      hypredrv_ErrorMsgAdd("%s compression failed", hypredrv_compression_get_name(algo));
      return 0;
   }
   *comp_size = (size_t)size;
   return 1;
}
#endif

#ifdef HYPREDRV_USING_BLOSC
static int
CompressBlosc(const void *input, size_t isize, unsigned char *dest, size_t bound,
              size_t *comp_size)
{
   blosc_init();
   blosc_set_compressor("blosclz");
   int size = blosc_compress(9, 1, 1, isize, input, dest, bound);
   blosc_destroy();
   if (size <= 0)
   {
      hypredrv_ErrorMsgAdd("Blosc compression failed");
      return 0;
   }
   *comp_size = (size_t)size;
   return 1;
}
#endif

static int
CompressPayload(comp_alg_t algo, const void *input, size_t isize, unsigned char *dest,
                size_t bound, int compression_level, size_t *comp_size)
{
   (void)input;
   (void)isize;
   (void)dest;
   (void)bound;
   (void)compression_level;
   (void)comp_size;
   switch (algo)
   {
#ifdef HYPREDRV_USING_ZLIB
      case COMP_ZLIB:
         return CompressZlib(input, isize, dest, bound, comp_size);
#endif
#ifdef HYPREDRV_USING_ZSTD
      case COMP_ZSTD:
         return CompressZstd(input, isize, dest, bound, compression_level, comp_size);
#endif
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
      case COMP_LZ4HC:
         return CompressLz4(algo, input, isize, dest, bound, comp_size);
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
         return CompressBlosc(input, isize, dest, bound, comp_size);
#endif
      default:
         return 0;
   }
}

void
hypredrv_compress(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                  void **output_ptr, int compression_level)
{
   const size_t   header_size = sizeof(uint64_t);
   size_t         bound = 0, comp_size = 0;
   unsigned char *output = NULL;

   if (!CodecArgumentsValid(isize, input, osize_ptr, output_ptr, "hypredrv_compress"))
   {
      return;
   }
   /* Some codec APIs still expect a source address for an empty buffer. */
   if (!input)
   {
      input = "";
   }
   if (algo == COMP_NONE)
   {
      output = CopyUncompressed(input, 0, isize);
      if (!output)
      {
         return;
      }
      *output_ptr = output;
      *osize_ptr  = isize;
      return;
   }
   if ((uint64_t)isize != isize)
   {
      CompressSizeError(algo, isize);
      return;
   }
   if (!CompressionAlgorithmAvailable(algo, "compression"))
   {
      return;
   }

   /* Determine a safe capacity before creating the common size-prefixed buffer. */
   if (!CompressBound(algo, isize, &bound) || bound < isize ||
       bound > SIZE_MAX - header_size)
   {
      CompressSizeError(algo, isize);
      return;
   }
   output = hypredrv_Malloc(header_size + bound, "compressed buffer");
   if (!output)
   {
      return;
   }
   uint64_t original_size = (uint64_t)isize;
   memcpy(output, &original_size, header_size);

   if (!CompressPayload(algo, input, isize, output + header_size, bound,
                        compression_level, &comp_size))
   {
      free(output);
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return;
   }

   *output_ptr = output;
   *osize_ptr  = header_size + comp_size;
}

/* Full reads and non-streamed slices share validation and decoding. A slice
 * request is checked before any output allocation or codec invocation. */
typedef struct
{
   size_t offset;
   size_t size;
} DecompressSlice;

static int
SliceExceeds(const DecompressSlice *slice, size_t total)
{
   return slice->offset > total || slice->size > total - slice->offset;
}

static void
SliceRangeError(const DecompressSlice *slice)
{
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd("Requested slice [%zu, +%zu) exceeds decoded payload",
                        slice->offset, slice->size);
}

static void
DecompressSizeError(comp_alg_t algo)
{
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd("Invalid or excessive %s payload size",
                        hypredrv_compression_get_name(algo));
}

/* COMP_NONE: the input is the payload; copy it, or the requested slice of it. */
static int
DecompressRaw(size_t isize, const void *input, size_t *osize_ptr, void **output_ptr,
              const DecompressSlice *slice)
{
   size_t size = isize, offset = 0;
   void  *output = NULL;

   if (slice)
   {
      if (SliceExceeds(slice, isize))
      {
         SliceRangeError(slice);
         return 0;
      }
      size   = slice->size;
      offset = slice->offset;
   }
   if (size > HYPREDRV_MAX_DECOMPRESSED_BYTES)
   {
      DecompressSizeError(COMP_NONE);
      return 0;
   }
   if (slice && !size)
   {
      return 1;
   }
   output = CopyUncompressed(input, offset, size);
   if (!output)
   {
      return 0;
   }
   *output_ptr = output;
   *osize_ptr  = size;
   return 1;
}

/* Reads and bounds the decoded size recorded in the blob's leading uint64_t. */
static int
DecompressReadHeader(comp_alg_t algo, size_t isize, const void *input,
                     size_t *original_size)
{
   const size_t header_size   = sizeof(uint64_t);
   uint64_t     declared_size = 0;

   if (isize < header_size)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Compressed buffer too small (%zu bytes)", isize);
      return 0;
   }

   /* The incoming byte buffer need not be aligned for uint64_t. */
   memcpy(&declared_size, input, header_size);
   if (declared_size > SIZE_MAX || declared_size > HYPREDRV_MAX_DECOMPRESSED_BYTES)
   {
      DecompressSizeError(algo);
      return 0;
   }
   *original_size = (size_t)declared_size;
   return 1;
}

/* Rejects sizes the codec's integer interface cannot represent, and Blosc
 * payloads whose own header disagrees with the declared size. */
static int
DecompressCheckLimits(comp_alg_t algo, const unsigned char *payload, size_t payload_size,
                      size_t original_size)
{
   (void)payload;
   (void)payload_size;
   (void)original_size;
   switch (algo)
   {
#ifdef HYPREDRV_USING_ZLIB
      case COMP_ZLIB:
         if (payload_size > ULONG_MAX || original_size > ULONG_MAX)
         {
            DecompressSizeError(algo);
            return 0;
         }
         return 1;
#endif
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
      case COMP_LZ4HC:
         if (payload_size > INT_MAX || original_size > INT_MAX)
         {
            DecompressSizeError(algo);
            return 0;
         }
         return 1;
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
      {
         size_t blosc_size = 0;
         if (payload_size > INT_MAX || original_size > BLOSC_MAX_BUFFERSIZE ||
             blosc_cbuffer_validate(payload, payload_size, &blosc_size) != 0 ||
             blosc_size != original_size)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid Blosc payload or inconsistent decoded size");
            return 0;
         }
         return 1;
      }
#endif
      default:
         return 1;
   }
}

/* Each backend decodes `payload` into `output` (capacity `original_size`). On
 * failure it adds its message and returns 0; the caller sets the error code. */

#ifdef HYPREDRV_USING_ZLIB
static int
DecompressZlib(const unsigned char *payload, size_t payload_size, void *output,
               size_t original_size, size_t *decoded_size)
{
   uLongf size  = (uLongf)original_size;
   int    error = uncompress(output, &size, payload, (uLong)payload_size);
   if (error != Z_OK)
   {
      hypredrv_ErrorMsgAdd("ZLIB decompression error: %d", error);
      return 0;
   }
   *decoded_size = (size_t)size;
   return 1;
}
#endif

#ifdef HYPREDRV_USING_ZSTD
static int
DecompressZstd(const unsigned char *payload, size_t payload_size, void *output,
               size_t original_size, size_t *decoded_size)
{
   *decoded_size = ZSTD_decompress(output, original_size, payload, payload_size);
   if (ZSTD_isError(*decoded_size))
   {
      hypredrv_ErrorMsgAdd("ZSTD decompression error: %s",
                           ZSTD_getErrorName(*decoded_size));
      return 0;
   }
   return 1;
}
#endif

#ifdef HYPREDRV_USING_LZ4
static int
DecompressLz4(const unsigned char *payload, size_t payload_size, void *output,
              size_t original_size, size_t *decoded_size)
{
   int size = LZ4_decompress_safe((const char *)payload, output, (int)payload_size,
                                  (int)original_size);
   if (size < 0)
   {
      hypredrv_ErrorMsgAdd("LZ4 decompression failed");
      return 0;
   }
   *decoded_size = (size_t)size;
   return 1;
}
#endif

#ifdef HYPREDRV_USING_BLOSC
static int
DecompressBlosc(const unsigned char *payload, void *output, size_t original_size,
                size_t *decoded_size)
{
   blosc_init();
   int size = blosc_decompress(payload, output, original_size);
   blosc_destroy();
   if (size < 0 || (size == 0 && original_size != 0))
   {
      hypredrv_ErrorMsgAdd("Blosc decompression failed");
      return 0;
   }
   *decoded_size = (size_t)size;
   return 1;
}
#endif

static int
DecompressPayload(comp_alg_t algo, const unsigned char *payload, size_t payload_size,
                  void *output, size_t original_size, size_t *decoded_size)
{
   (void)payload;
   (void)payload_size;
   (void)output;
   (void)original_size;
   (void)decoded_size;
   switch (algo)
   {
#ifdef HYPREDRV_USING_ZLIB
      case COMP_ZLIB:
         return DecompressZlib(payload, payload_size, output, original_size,
                               decoded_size);
#endif
#ifdef HYPREDRV_USING_ZSTD
      case COMP_ZSTD:
         return DecompressZstd(payload, payload_size, output, original_size,
                               decoded_size);
#endif
#ifdef HYPREDRV_USING_LZ4
      case COMP_LZ4:
      case COMP_LZ4HC:
         return DecompressLz4(payload, payload_size, output, original_size, decoded_size);
#endif
#ifdef HYPREDRV_USING_BLOSC
      case COMP_BLOSC:
         return DecompressBlosc(payload, output, original_size, decoded_size);
#endif
      default:
         return 0;
   }
}

/* Replaces the fully decoded buffer with a copy of the requested slice; frees
 * `full` either way and returns NULL when the copy cannot be allocated. */
static void *
DecompressExtractSlice(void *full, const DecompressSlice *slice)
{
   void *selected = hypredrv_Malloc(slice->size, "slice buffer");
   if (selected)
   {
      memcpy(selected, (const unsigned char *)full + slice->offset, slice->size);
   }
   free(full);
   return selected;
}

static int
DecompressBuffer(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                 void **output_ptr, const DecompressSlice *slice)
{
   const size_t header_size   = sizeof(uint64_t);
   size_t       original_size = 0, decoded_size = 0;
   void        *output = NULL;

   if (!CodecArgumentsValid(isize, input, osize_ptr, output_ptr, "hypredrv_decompress"))
   {
      return 0;
   }
   if (algo == COMP_NONE)
   {
      return DecompressRaw(isize, input, osize_ptr, output_ptr, slice);
   }
   if (!DecompressReadHeader(algo, isize, input, &original_size))
   {
      return 0;
   }
   if (!CompressionAlgorithmAvailable(algo, "decompression"))
   {
      return 0;
   }
   if (slice && SliceExceeds(slice, original_size))
   {
      SliceRangeError(slice);
      return 0;
   }
   const unsigned char *payload      = (const unsigned char *)input + header_size;
   size_t               payload_size = isize - header_size;

   if (!DecompressCheckLimits(algo, payload, payload_size, original_size))
   {
      return 0;
   }
   if (slice && !slice->size)
   {
      return 1;
   }
   output = hypredrv_Malloc(original_size ? original_size : 1, "decompressed buffer");
   if (!output)
   {
      return 0;
   }
   if (!DecompressPayload(algo, payload, payload_size, output, original_size,
                          &decoded_size))
   {
      free(output);
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return 0;
   }
   if (decoded_size != original_size)
   {
      free(output);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Decoded size mismatch: declared=%zu actual=%zu",
                           original_size, decoded_size);
      return 0;
   }

   if (slice)
   {
      output = DecompressExtractSlice(output, slice);
      if (!output)
      {
         return 0;
      }
      decoded_size = slice->size;
   }
   *output_ptr = output;
   *osize_ptr  = decoded_size;
   return 1;
}

void
hypredrv_decompress(comp_alg_t algo, size_t isize, const void *input, size_t *osize_ptr,
                    void **output_ptr)
{
   DecompressBuffer(algo, isize, input, osize_ptr, output_ptr, NULL);
}

/*-----------------------------------------------------------------------------
 * Resumable slice decoding (zstd/zlib)
 *
 * A slice stream holds the decoder state for one compressed blob and the
 * number of original bytes produced so far. The compressed bytes are pulled on
 * demand from a caller-supplied source (memory or an open file) through a
 * small input buffer, so a long-lived stream never pins a copy of its blob.
 * Reading [offset, offset+size) continues from the current position when
 * offset is at or past it (bytes before the slice go through a scratch window;
 * the slice itself is decoded straight into the output) and restarts the
 * decoder otherwise, so reading a blob's slices in increasing order decodes it
 * only once.
 *-----------------------------------------------------------------------------*/

enum
{
   SLICE_STREAM_WINDOW_SIZE = 1 << 17, /* scratch output for skipped bytes */
   SLICE_STREAM_INPUT_SIZE  = 1 << 18, /* compressed input chunk */
};

struct hypredrv_SliceStream_struct
{
   comp_alg_t     algo;
   size_t         payload; /* compressed bytes after the size header */
   size_t         orig_size;
   size_t         pos;    /* original bytes produced so far */
   size_t         in_fed; /* payload bytes loaded into `input` so far */
   unsigned char *input;
   unsigned char *window;
   int            started;
#ifdef HYPREDRV_USING_ZSTD
   ZSTD_DCtx    *zstd;
   ZSTD_inBuffer zin;
#endif
#ifdef HYPREDRV_USING_ZLIB
   z_stream zs;
#endif
};

static int
SliceSourceReadMemory(void *ctx, uint64_t offset, void *buffer, size_t size)
{
   memcpy(buffer, (const unsigned char *)ctx + offset, size);
   return 1;
}

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

/* Loads the next compressed chunk (the caller checks that input remains);
 * returns its size, or 0 when the source read fails. */
static HYPREDRV_MAYBE_UNUSED size_t
SliceStreamRefill(hypredrv_SliceStream *st, const hypredrv_SliceSource *src)
{
   size_t n = st->payload - st->in_fed;

   n = (n < SLICE_STREAM_INPUT_SIZE) ? n : SLICE_STREAM_INPUT_SIZE;
   if (n == 0 || !src->read_at(src->ctx, sizeof(uint64_t) + st->in_fed, st->input, n))
   {
      return 0;
   }
   st->in_fed += n;
   return n;
}

#ifdef HYPREDRV_USING_ZSTD
static size_t
SliceStreamDecodeZstd(hypredrv_SliceStream *st, const hypredrv_SliceSource *src,
                      unsigned char *target, size_t avail)
{
   for (;;)
   {
      if (st->zin.pos == st->zin.size && st->in_fed < st->payload)
      {
         size_t got = SliceStreamRefill(st, src);
         if (got == 0) /* GCOVR_EXCL_BR_LINE */
         {
            return 0; /* source read failed */
         }
         st->zin = (ZSTD_inBuffer){st->input, got, 0};
      }
      ZSTD_outBuffer o   = {target, avail, 0};
      size_t         ret = ZSTD_decompressStream(st->zstd, &o, &st->zin);
      /* GCOVR_EXCL_BR_START */
      if (ZSTD_isError(ret))
      {
         return 0;
      }
      /* Retry only when an input chunk ran out before any output. */
      if (o.pos > 0 || st->zin.pos < st->zin.size || st->in_fed == st->payload)
      {
         return o.pos;
      }
      /* GCOVR_EXCL_BR_STOP */
   }
}
#endif

#ifdef HYPREDRV_USING_ZLIB
static size_t
SliceStreamDecodeZlib(hypredrv_SliceStream *st, const hypredrv_SliceSource *src,
                      unsigned char *target, size_t avail)
{
   /* zlib counts bytes in uInt: cap each output request (input chunks are
    * already small) so slices larger than 4 GiB still decode. */
   const uInt out_chunk = (avail > UINT_MAX) ? UINT_MAX : (uInt)avail;
   for (;;)
   {
      if (st->zs.avail_in == 0 && st->in_fed < st->payload)
      {
         size_t got = SliceStreamRefill(st, src);
         if (got == 0) /* GCOVR_EXCL_BR_LINE */
         {
            return 0; /* source read failed */
         }
         st->zs.next_in  = st->input;
         st->zs.avail_in = (uInt)got;
      }
      st->zs.next_out  = target;
      st->zs.avail_out = out_chunk;
      int    ret       = inflate(&st->zs, Z_NO_FLUSH);
      size_t produced  = out_chunk - st->zs.avail_out;
      /* GCOVR_EXCL_BR_START */
      if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR)
      {
         return 0;
      }
      if (produced > 0 || ret == Z_STREAM_END || st->zs.avail_in > 0 ||
          st->in_fed == st->payload)
      {
         return produced;
      }
      /* GCOVR_EXCL_BR_STOP */
   }
}
#endif

/* Decodes up to `avail` bytes into `target`; returns the count (0 = no progress). */
static size_t
SliceStreamDecode(hypredrv_SliceStream *st, const hypredrv_SliceSource *src,
                  unsigned char *target, size_t avail)
{
#ifdef HYPREDRV_USING_ZSTD
   if (st->algo == COMP_ZSTD)
   {
      return SliceStreamDecodeZstd(st, src, target, avail);
   }
#endif
#ifdef HYPREDRV_USING_ZLIB
   if (st->algo == COMP_ZLIB)
   {
      return SliceStreamDecodeZlib(st, src, target, avail);
   }
#endif
   (void)st;
   (void)src;
   (void)target;
   (void)avail;
   return 0;
}

hypredrv_SliceStream *
hypredrv_SliceStreamCreate(comp_alg_t algo, size_t isize, const hypredrv_SliceSource *src)
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
   if (!resumable)
   {
      /* Other codecs intentionally use the full-decode path. */
      if (algo == COMP_ZLIB || algo == COMP_ZSTD)
      {
         CompressionAlgorithmAvailable(algo, "slice decompression");
      }
      return NULL;
   }
   if (!src || !src->read_at || isize < sizeof(uint64_t))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid source or compressed size for slice decoder");
      return NULL;
   }
   if (!src->read_at(src->ctx, 0, &orig, sizeof(orig)))
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unable to read slice decoder size header");
      }
      return NULL;
   }
   if (orig > SIZE_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Decoded size exceeds addressable slice offsets");
      return NULL;
   }

   st = (hypredrv_SliceStream *)calloc(1, sizeof(*st));
   if (!st) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_AllocationFailure("Failed to allocate slice decoder");
      return NULL;
   }
   st->algo      = algo;
   st->payload   = isize - sizeof(uint64_t);
   st->orig_size = (size_t)orig;
   st->input     = (unsigned char *)malloc(SLICE_STREAM_INPUT_SIZE);
   st->window    = (unsigned char *)malloc(SLICE_STREAM_WINDOW_SIZE);
   if (!st->input || !st->window) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_SliceStreamDestroy(&st); /* GCOVR_EXCL_LINE */
      hypredrv_AllocationFailure("Failed to allocate slice decoder buffers");
      return NULL; /* GCOVR_EXCL_LINE */
   }
   return st;
}

/* (Re)starts decoding from the beginning of the payload; returns st->started. */
static int
SliceStreamRestart(hypredrv_SliceStream *st)
{
   SliceStreamEnd(st);
   st->in_fed = 0;
#ifdef HYPREDRV_USING_ZSTD
   if (st->algo == COMP_ZSTD)
   {
      st->zstd    = ZSTD_createDCtx();
      st->zin     = (ZSTD_inBuffer){st->input, 0, 0};
      st->started = (st->zstd != NULL);
   }
#endif
#ifdef HYPREDRV_USING_ZLIB
   if (st->algo == COMP_ZLIB)
   {
      memset(&st->zs, 0, sizeof(st->zs));
      st->zs.next_in = st->input;
      st->started    = (inflateInit(&st->zs) == Z_OK);
   }
#endif
   return st->started;
}

/* Advances the stream through offset + size, decoding [offset, offset + size)
 * into `out`; returns 0 when the decoder stops short of the end. */
static int
SliceStreamDecodeRange(hypredrv_SliceStream *st, const hypredrv_SliceSource *src,
                       unsigned char *out, size_t offset, size_t size)
{
   while (st->pos < offset + size)
   {
      /* Skip [pos, offset) through the window, capped so it never crosses
       * into the slice; then decode the slice directly into `out`. */
      size_t         avail  = 0;
      unsigned char *target = NULL;
      if (st->pos < offset)
      {
         avail  = (offset - st->pos < SLICE_STREAM_WINDOW_SIZE) ? offset - st->pos
                                                                : SLICE_STREAM_WINDOW_SIZE;
         target = st->window;
      }
      else
      {
         avail  = offset + size - st->pos;
         target = out + (st->pos - offset);
      }
      size_t n = SliceStreamDecode(st, src, target, avail);
      if (n == 0)
      {
         break;
      }
      st->pos += n;
   }
   return st->pos >= offset + size;
}

int
hypredrv_SliceStreamRead(hypredrv_SliceStream *st, const hypredrv_SliceSource *src,
                         size_t offset, size_t size, void **output)
{
   unsigned char *out = NULL;

   if (output)
   {
      *output = NULL;
   }
   if (!st || !src || !src->read_at || !output)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments to hypredrv_SliceStreamRead");
      return 0;
   }
   /* GCOVR_EXCL_BR_START */
   if (offset > st->orig_size || size > st->orig_size - offset ||
       size > HYPREDRV_MAX_DECOMPRESSED_BYTES) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Requested slice [%zu, +%zu) exceeds decoded bounds (%zu bytes) "
         "or allocation limit",
         offset, size, st->orig_size);
      return 0;
   }
   if (size == 0)
   {
      return 1;
   }
   out = (unsigned char *)hypredrv_Malloc(size, "slice buffer");
   if (!out) /* GCOVR_EXCL_BR_LINE */
   {
      return 0;
   }

   /* Resume when the slice starts at or after the current position. */
   if ((!st->started || offset < st->pos) && !SliceStreamRestart(st))
   {
      free(out);
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("%s decoder initialization failed",
                           hypredrv_compression_get_name(st->algo));
      return 0;
   }

   if (!SliceStreamDecodeRange(st, src, out, offset, size))
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
   free((*st_ptr)->input);
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
   if (algo == COMP_ZLIB || algo == COMP_ZSTD)
   {
      if (output)
      {
         *output = NULL;
      }
      if (!output || (!input && isize > 0))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid arguments to hypredrv_decompress_slice");
         return 0;
      }
      /* The stream validates the size header once, and reads only through the
       * requested slice instead of allocating the complete decoded payload. */
      hypredrv_SliceSource  src    = {SliceSourceReadMemory, (void *)(uintptr_t)input};
      hypredrv_SliceStream *stream = hypredrv_SliceStreamCreate(algo, isize, &src);
      if (!stream)
      {
         return 0;
      }
      int ok = hypredrv_SliceStreamRead(stream, &src, offset, size, output);
      hypredrv_SliceStreamDestroy(&stream);
      return ok;
   }

   const DecompressSlice slice        = {offset, size};
   size_t                decoded_size = 0;
   return DecompressBuffer(algo, isize, input, &decoded_size, output, &slice);
}
