#include <stdlib.h>
#include <string.h>

#include "HYPREDRV_config.h"
#include "internal/comp.h"
#include "internal/error.h"
#include "test_helpers.h"

static const comp_alg_t available_codecs[] = {
   COMP_NONE,
#ifdef HYPREDRV_USING_ZLIB
   COMP_ZLIB,
#endif
#ifdef HYPREDRV_USING_ZSTD
   COMP_ZSTD,
#endif
#ifdef HYPREDRV_USING_LZ4
   COMP_LZ4,   COMP_LZ4HC,
#endif
#ifdef HYPREDRV_USING_BLOSC
   COMP_BLOSC,
#endif
};

static void
run_roundtrip(comp_alg_t algo)
{
   const size_t nbytes = 1024;
   unsigned char *input = (unsigned char *)malloc(nbytes);
   void          *compressed = NULL;
   void          *decompressed = NULL;
   size_t         compressed_size = 0;
   size_t         decompressed_size = 0;

   ASSERT_NOT_NULL(input);
   for (size_t i = 0; i < nbytes; i++)
   {
      input[i] = (unsigned char)(i % 251);
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(algo, nbytes, input, &compressed_size, &compressed, -1);
   if (hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB)
   {
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      free(input);
      free(compressed);
      free(decompressed);
      return;
   }
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(compressed);
   ASSERT_TRUE(compressed_size > 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(algo, compressed_size, compressed, &decompressed_size, &decompressed);
   if (hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB)
   {
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      free(input);
      free(compressed);
      free(decompressed);
      return;
   }
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(decompressed);
   ASSERT_EQ((int)decompressed_size, (int)nbytes);
   ASSERT_TRUE(memcmp(input, decompressed, nbytes) == 0);

   free(input);
   free(compressed);
   free(decompressed);
}

static void
test_comp_roundtrip_none(void)
{
   run_roundtrip(COMP_NONE);
}

static void
test_comp_name_extension_switch_defaults(void)
{
   ASSERT_STREQ(hypredrv_compression_get_name((comp_alg_t)999), "unknown");
   ASSERT_STREQ(hypredrv_compression_get_extension((comp_alg_t)999), ".bin");
}

static void
test_comp_name_extension_all_cases(void)
{
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_NONE), "none");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_NONE), ".bin");
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_ZLIB), "zlib");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_ZLIB), ".zlib.bin");
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_ZSTD), "zstd");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_ZSTD), ".zst.bin");
#ifdef HYPREDRV_USING_LZ4
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_LZ4), "lz4");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_LZ4), ".lz4.bin");
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_LZ4HC), "lz4hc");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_LZ4HC), ".lz4hc.bin");
#else
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_LZ4), "unknown");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_LZ4), ".bin");
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_LZ4HC), "unknown");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_LZ4HC), ".bin");
#endif
#ifdef HYPREDRV_USING_BLOSC
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_BLOSC), "blosc");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_BLOSC), ".blosc.bin");
#else
   ASSERT_STREQ(hypredrv_compression_get_name(COMP_BLOSC), "unknown");
   ASSERT_STREQ(hypredrv_compression_get_extension(COMP_BLOSC), ".bin");
#endif
}

#if !defined(HYPREDRV_USING_LZ4)
static void
test_comp_missing_lz4_codecs(void)
{
   unsigned char buf[8] = {0};
   void         *out    = NULL;
   size_t        sz     = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_LZ4, sizeof(buf), buf, &sz, &out, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB) != 0);
   free(out);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_LZ4HC, sizeof(buf), buf, &sz, &out, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB) != 0);
   free(out);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_LZ4, sizeof(buf), buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB) != 0);
   free(out);
}
#endif

#if !defined(HYPREDRV_USING_BLOSC)
static void
test_comp_missing_blosc_codec(void)
{
   unsigned char buf[8] = {0};
   void         *out    = NULL;
   size_t        sz     = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_BLOSC, sizeof(buf), buf, &sz, &out, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB) != 0);
   free(out);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_BLOSC, sizeof(buf), buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_MISSING_LIB) != 0);
   free(out);
}
#endif

static void
test_comp_unknown_algo_compress_decompress(void)
{
   unsigned char input[4] = {0};
   unsigned char buf[32]  = {0};
   void         *out = NULL;
   size_t        sz  = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress((comp_alg_t)88, 4, input, &sz, &out, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNKNOWN) != 0);
   free(out);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   /* isize >= sizeof(uint64_t); buffer must hold header read */
   hypredrv_decompress((comp_alg_t)88, 16, buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNKNOWN) != 0);
   free(out);
}

static void
test_comp_decompress_buffer_too_small(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
#ifdef HYPREDRV_USING_ZLIB
   char    buf[4] = {0};
   void   *out = NULL;
   size_t  sz  = 0;

   hypredrv_decompress(COMP_ZLIB, 4, buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
#endif
}

#ifdef HYPREDRV_USING_ZLIB
static void
test_comp_zlib_decompress_corrupt_payload(void)
{
   unsigned char buf[64];
   uint64_t      orig = 8;
   void         *out  = NULL;
   size_t        sz   = 0;

   memcpy(buf, &orig, sizeof(orig));
   memset(buf + sizeof(orig), 0xFF, sizeof(buf) - sizeof(orig));

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_ZLIB, sizeof(buf), buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNKNOWN) != 0);
   free(out);
}
#endif

#ifdef HYPREDRV_USING_ZSTD
static void
test_comp_zstd_decompress_corrupt_payload(void)
{
   unsigned char buf[64];
   uint64_t      orig = 8;
   void         *out  = NULL;
   size_t        sz   = 0;

   memcpy(buf, &orig, sizeof(orig));
   memset(buf + sizeof(orig), 0xFF, sizeof(buf) - sizeof(orig));

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_ZSTD, sizeof(buf), buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNKNOWN) != 0);
   free(out);
}

static void
test_comp_zstd_compression_level_clamp(void)
{
   unsigned char input[32];
   void         *compressed = NULL;
   size_t        sz         = 0;

   for (size_t i = 0; i < sizeof(input); i++)
   {
      input[i] = (unsigned char)(i + 1u);
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_ZSTD, sizeof(input), input, &sz, &compressed, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   free(compressed);
   compressed = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_ZSTD, sizeof(input), input, &sz, &compressed, 99);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   free(compressed);
}
#endif

#ifdef HYPREDRV_USING_LZ4
static void
test_comp_lz4_decompress_corrupt_payload(void)
{
   unsigned char buf[64];
   uint64_t      orig = 8;
   void         *out  = NULL;
   size_t        sz   = 0;

   memcpy(buf, &orig, sizeof(orig));
   memset(buf + sizeof(orig), 0xFF, sizeof(buf) - sizeof(orig));

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_LZ4, sizeof(buf), buf, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_UNKNOWN) != 0);
   free(out);
}
#endif

static void
test_comp_invalid_args(void)
{
   size_t  sz  = 0;
   void   *out = NULL;
   char    buf  = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   sz  = 99;
   out = &buf;
   hypredrv_compress(COMP_ZLIB, 10, NULL, &sz, &out, -1);
   ASSERT_EQ_SIZE(sz, 0);
   ASSERT_NULL(out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   sz  = 99;
   out = &buf;
   hypredrv_decompress(COMP_ZSTD, 0, NULL, &sz, &out);
   ASSERT_EQ_SIZE(sz, 0);
   ASSERT_NULL(out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   /* NULL input with nonzero size: third clause of invalid-arg guard */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_ZSTD, 1, NULL, &sz, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_NONE, 1, &buf, NULL, &out, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_NONE, 1, &buf, &sz, NULL, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_NONE, 1, &buf, NULL, &out);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_decompress(COMP_NONE, 1, &buf, &sz, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
}

static void
test_comp_none_zero_payload(void)
{
   void   *out = NULL;
   size_t  sz  = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_compress(COMP_NONE, 0, NULL, &sz, &out, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(out);
   ASSERT_EQ((int)sz, 0);
   free(out);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   out = NULL;
   sz  = 0;
   hypredrv_decompress(COMP_NONE, 0, NULL, &sz, &out);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(out);
   ASSERT_EQ((int)sz, 0);
   free(out);
}

/* Slices of a compressed payload must match the source bytes, including
 * slices that start mid-stream and cross the 128 KiB decode window. */
static void
run_slice_checks(comp_alg_t algo)
{
   const size_t   nbytes = (size_t)1 << 20;
   unsigned char *input  = (unsigned char *)malloc(nbytes);
   void          *blob   = NULL;
   size_t         blob_size = 0;
   const size_t   ranges[][2] = {
      {0, 1000}, {131000, 2000}, {300000, 400000}, {nbytes - 10, 10}, {nbytes, 0},
   };

   ASSERT_NOT_NULL(input);
   for (size_t i = 0; i < nbytes; i++)
   {
      input[i] = (unsigned char)((i * 2654435761u) >> 24);
   }

   hypredrv_ErrorCodeResetAll();
   if (algo == COMP_NONE)
   {
      blob      = input;
      blob_size = nbytes;
   }
   else
   {
      hypredrv_compress(algo, nbytes, input, &blob_size, &blob, -1);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
   }

   for (size_t r = 0; r < sizeof(ranges) / sizeof(ranges[0]); r++)
   {
      void *slice = NULL;
      ASSERT_EQ(hypredrv_decompress_slice(algo, blob_size, blob, ranges[r][0],
                                          ranges[r][1], &slice),
                1);
      if (ranges[r][1] == 0)
      {
         ASSERT_NULL(slice);
         continue;
      }
      ASSERT_NOT_NULL(slice);
      ASSERT_EQ(memcmp(slice, input + ranges[r][0], ranges[r][1]), 0);
      free(slice);
   }

   /* Out-of-range requests fail without producing output. */
   void *slice = NULL;
   ASSERT_EQ(hypredrv_decompress_slice(algo, blob_size, blob, nbytes - 5, 10, &slice), 0);
   ASSERT_NULL(slice);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   if (blob != input)
   {
      free(blob);
   }
   free(input);
}

/* In-memory source whose reads fail past `limit` bytes into the blob. */
typedef struct
{
   const unsigned char *data;
   uint64_t             limit;
} ShortSource;

static int
ShortSourceRead(void *ctx, uint64_t offset, void *buffer, size_t size)
{
   const ShortSource *src = (const ShortSource *)ctx;
   if (offset > src->limit || size > src->limit - offset)
   {
      return 0;
   }
   memcpy(buffer, src->data + offset, size);
   return 1;
}

/* A slice stream must return correct bytes when resuming forward and when
 * asked for an earlier offset (restart). */
static HYPREDRV_MAYBE_UNUSED void
run_slice_stream_checks(comp_alg_t algo)
{
   const size_t   nbytes = (size_t)1 << 20;
   unsigned char *input  = (unsigned char *)malloc(nbytes);
   void          *blob   = NULL;
   size_t         blob_size = 0;
   const size_t   ranges[][2] = {
      {1000, 5000}, {200000, 300000}, {900000, 1000}, {50, 100}, {500000, 10},
   };

   ASSERT_NOT_NULL(input);
   /* Mostly incompressible bytes, so the compressed blob spans several of the
    * stream's input chunks. */
   uint32_t state = 2463534242u;
   for (size_t i = 0; i < nbytes; i++)
   {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      input[i] = (unsigned char)(state >> 24);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_compress(algo, nbytes, input, &blob_size, &blob, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ShortSource           memory = {blob, blob_size};
   hypredrv_SliceSource  src    = {ShortSourceRead, &memory};
   hypredrv_SliceStream *st  = hypredrv_SliceStreamCreate(algo, blob_size, &src);
   ASSERT_NOT_NULL(st);

   for (size_t r = 0; r < sizeof(ranges) / sizeof(ranges[0]); r++)
   {
      void *slice = NULL;
      ASSERT_EQ(hypredrv_SliceStreamRead(st, &src, ranges[r][0], ranges[r][1], &slice),
                1);
      ASSERT_NOT_NULL(slice);
      ASSERT_EQ(memcmp(slice, input + ranges[r][0], ranges[r][1]), 0);
      free(slice);
   }

   void *slice = NULL;
   ASSERT_EQ(hypredrv_SliceStreamRead(st, &src, nbytes, 1, &slice), 0);
   ASSERT_NULL(slice);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_SliceStreamDestroy(&st);
   ASSERT_NULL(st);

   /* A source that fails partway through makes the read fail, not hang. */
   ShortSource          short_ctx = {blob, 1u << 16};
   hypredrv_SliceSource failing   = {ShortSourceRead, &short_ctx};
   st = hypredrv_SliceStreamCreate(algo, blob_size, &failing);
   ASSERT_NOT_NULL(st);
   ASSERT_EQ(hypredrv_SliceStreamRead(st, &failing, nbytes - 1000, 1000, &slice), 0);
   ASSERT_NULL(slice);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_SliceStreamDestroy(&st);

   ASSERT_NULL(hypredrv_SliceStreamCreate(COMP_NONE, nbytes, &src));
   short_ctx.limit = sizeof(uint64_t) - 1;
   ASSERT_NULL(hypredrv_SliceStreamCreate(algo, blob_size, &failing));
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();
   ASSERT_NULL(hypredrv_SliceStreamCreate(algo, blob_size, NULL));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   free(blob);
   free(input);
}

static void
test_comp_slice_stream(void)
{
#ifdef HYPREDRV_USING_ZLIB
   run_slice_stream_checks(COMP_ZLIB);
#endif
#ifdef HYPREDRV_USING_ZSTD
   run_slice_stream_checks(COMP_ZSTD);
#endif
}

static void
test_comp_decompress_slice(void)
{
   for (size_t c = 0; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      run_slice_checks(available_codecs[c]);
   }
}

static void
test_comp_slice_invalid_arguments(void)
{
   unsigned char byte   = 0;
   uint64_t      header = 0;
   void         *output = &byte;
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_decompress_slice(COMP_NONE, 1, NULL, 0, 1, &output), 0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_NULL(output);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_decompress_slice(COMP_NONE, 1, &byte, 0, 1, NULL), 0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_decompress_slice(COMP_NONE, SIZE_MAX, &byte, 1, SIZE_MAX, &output),
             0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_NULL(output);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(
      hypredrv_decompress_slice((comp_alg_t)88, sizeof(header), &header, 0, 0, &output),
      0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_UNKNOWN);
   ASSERT_NULL(output);
   hypredrv_ErrorCodeResetAll();
   /* Bounds errors take precedence over decoding a malformed payload. */
   unsigned char malformed[16] = {0};
   uint64_t      declared      = 37;
   memcpy(malformed, &declared, sizeof(declared));
   for (size_t c = 1; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      hypredrv_ErrorCodeResetAll();
      ASSERT_EQ(hypredrv_decompress_slice(available_codecs[c], sizeof(malformed),
                                          malformed, (size_t)declared, 1, &output),
                0);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(output);
   }
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_SliceStreamRead(NULL, NULL, 0, 0, &output), 0);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_NULL(output);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_decompress_slice(COMP_NONE, 1, &byte, 1, 0, &output), 1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(output);
}

static void
test_comp_unaligned_input(void)
{
   const unsigned char input[] = "size-prefixed input need not be aligned";
   for (size_t c = 0; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      void  *blob = NULL, *output = NULL;
      size_t blob_size = 0, output_size = 0;
      hypredrv_ErrorCodeResetAll();
      hypredrv_compress(available_codecs[c], sizeof(input), input, &blob_size, &blob, -1);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      unsigned char *unaligned = malloc(blob_size + 1);
      ASSERT_NOT_NULL(unaligned);
      memcpy(unaligned + 1, blob, blob_size);
      hypredrv_decompress(available_codecs[c], blob_size, unaligned + 1, &output_size,
                          &output);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ_SIZE(output_size, sizeof(input));
      ASSERT_EQ(memcmp(output, input, sizeof(input)), 0);
      free(output);
      free(unaligned);
      free(blob);
   }
}

/* A forged size must never make a decoder publish uninitialized tail bytes. */
static void
test_comp_declared_size_matches_decoded(void)
{
   unsigned char input[37];
   for (size_t i = 0; i < sizeof(input); i++)
   {
      input[i] = (unsigned char)(i + 1);
   }
   const uint64_t invalid_sizes[] = {sizeof(input) + 1, sizeof(input) - 1, 0};
   for (size_t c = 1; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      void  *blob      = NULL;
      size_t blob_size = 0;
      hypredrv_ErrorCodeResetAll();
      hypredrv_compress(available_codecs[c], sizeof(input), input, &blob_size, &blob, -1);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(blob);
      for (size_t i = 0; i < sizeof(invalid_sizes) / sizeof(invalid_sizes[0]); i++)
      {
         memcpy(blob, &invalid_sizes[i], sizeof(invalid_sizes[i]));
         void  *output      = NULL;
         size_t output_size = 0;
         hypredrv_ErrorCodeResetAll();
         hypredrv_decompress(available_codecs[c], blob_size, blob, &output_size, &output);
         ASSERT_TRUE(hypredrv_ErrorCodeActive());
         ASSERT_NULL(output);
         ASSERT_EQ_SIZE(output_size, 0);
      }
      free(blob);
   }
   hypredrv_ErrorCodeResetAll();
}

static void
test_comp_empty_roundtrips(void)
{
   for (size_t c = 0; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      void  *blob = NULL, *output = NULL;
      size_t blob_size = 0, output_size = 1;
      hypredrv_ErrorCodeResetAll();
      hypredrv_compress(available_codecs[c], 0, NULL, &blob_size, &blob, -1);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(blob);
      hypredrv_decompress(available_codecs[c], blob_size, blob, &output_size, &output);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(output);
      ASSERT_EQ_SIZE(output_size, 0);
      free(output);
      free(blob);
   }
}

static void
test_comp_rejects_unrepresentable_sizes(void)
{
   unsigned char input     = 0;
   uint64_t      oversized = UINT64_MAX;
   for (size_t c = 1; c < sizeof(available_codecs) / sizeof(available_codecs[0]); c++)
   {
      void  *output      = NULL;
      size_t output_size = 0;
      hypredrv_ErrorCodeResetAll();
      hypredrv_compress(available_codecs[c], SIZE_MAX, &input, &output_size, &output, -1);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(output);
      ASSERT_EQ_SIZE(output_size, 0);
      hypredrv_ErrorCodeResetAll();
      hypredrv_decompress(available_codecs[c], sizeof(oversized), &oversized,
                          &output_size, &output);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
      ASSERT_NULL(output);
      ASSERT_EQ_SIZE(output_size, 0);
   }
   hypredrv_ErrorCodeResetAll();
}

#ifdef HYPREDRV_USING_BLOSC
static void
test_comp_blosc_truncated_payload(void)
{
   const unsigned char input[]   = "a small Blosc buffer";
   void               *blob      = NULL;
   size_t              blob_size = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_compress(COMP_BLOSC, sizeof(input), input, &blob_size, &blob, -1);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(blob);
   for (size_t n = sizeof(uint64_t); n < blob_size; n++)
   {
      void  *output      = NULL;
      size_t output_size = 0;
      hypredrv_ErrorCodeResetAll();
      hypredrv_decompress(COMP_BLOSC, n, blob, &output_size, &output);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_NULL(output);
      ASSERT_EQ_SIZE(output_size, 0);
   }
   free(blob);
   hypredrv_ErrorCodeResetAll();
}
#endif

int
main(void)
{
   RUN_TEST(test_comp_declared_size_matches_decoded);
   RUN_TEST(test_comp_slice_invalid_arguments);
   RUN_TEST(test_comp_unaligned_input);
   RUN_TEST(test_comp_empty_roundtrips);
   RUN_TEST(test_comp_rejects_unrepresentable_sizes);
#ifdef HYPREDRV_USING_BLOSC
   RUN_TEST(test_comp_blosc_truncated_payload);
#endif
   RUN_TEST(test_comp_roundtrip_none);
   RUN_TEST(test_comp_name_extension_switch_defaults);
   RUN_TEST(test_comp_name_extension_all_cases);
   RUN_TEST(test_comp_unknown_algo_compress_decompress);
#if !defined(HYPREDRV_USING_LZ4)
   RUN_TEST(test_comp_missing_lz4_codecs);
#endif
#if !defined(HYPREDRV_USING_BLOSC)
   RUN_TEST(test_comp_missing_blosc_codec);
#endif
   RUN_TEST(test_comp_decompress_buffer_too_small);
#ifdef HYPREDRV_USING_ZLIB
   RUN_TEST(test_comp_zlib_decompress_corrupt_payload);
#endif
#ifdef HYPREDRV_USING_ZSTD
   RUN_TEST(test_comp_zstd_decompress_corrupt_payload);
   RUN_TEST(test_comp_zstd_compression_level_clamp);
#endif
   RUN_TEST(test_comp_invalid_args);
   RUN_TEST(test_comp_none_zero_payload);
   RUN_TEST(test_comp_decompress_slice);
   RUN_TEST(test_comp_slice_stream);

#ifdef HYPREDRV_USING_ZLIB
   run_roundtrip(COMP_ZLIB);
#endif
#ifdef HYPREDRV_USING_ZSTD
   run_roundtrip(COMP_ZSTD);
#endif
#ifdef HYPREDRV_USING_LZ4
   RUN_TEST(test_comp_lz4_decompress_corrupt_payload);
   run_roundtrip(COMP_LZ4);
   run_roundtrip(COMP_LZ4HC);
#endif
#ifdef HYPREDRV_USING_BLOSC
   run_roundtrip(COMP_BLOSC);
#endif

   return 0;
}
