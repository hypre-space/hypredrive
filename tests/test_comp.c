#include <stdlib.h>
#include <string.h>

#include "comp.h"
#include "error.h"
#include "test_helpers.h"

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

   ErrorCodeResetAll();
   ErrorMsgClear();
   hypredrv_compress(algo, nbytes, input, &compressed_size, &compressed);
   if (ErrorCodeGet() & ERROR_MISSING_LIB)
   {
      ErrorCodeResetAll();
      ErrorMsgClear();
      free(input);
      free(compressed);
      free(decompressed);
      return;
   }
   ASSERT_FALSE(ErrorCodeActive());
   ASSERT_NOT_NULL(compressed);
   ASSERT_TRUE(compressed_size > 0);

   ErrorCodeResetAll();
   ErrorMsgClear();
   hypredrv_decompress(algo, compressed_size, compressed, &decompressed_size, &decompressed);
   if (ErrorCodeGet() & ERROR_MISSING_LIB)
   {
      ErrorCodeResetAll();
      ErrorMsgClear();
      free(input);
      free(compressed);
      free(decompressed);
      return;
   }
   ASSERT_FALSE(ErrorCodeActive());
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
test_comp_filename_detection(void)
{
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.bin"), (int)COMP_NONE);
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.zlib.bin"), (int)COMP_ZLIB);
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.zst.bin"), (int)COMP_ZSTD);
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.lz4.bin"), (int)COMP_LZ4);
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.lz4hc.bin"), (int)COMP_LZ4HC);
   ASSERT_EQ((int)hypredrv_compression_from_filename("a.blosc.bin"), (int)COMP_BLOSC);
}

int
main(void)
{
   RUN_TEST(test_comp_roundtrip_none);
   RUN_TEST(test_comp_filename_detection);

#ifdef HYPREDRV_USING_ZLIB
   run_roundtrip(COMP_ZLIB);
#endif
#ifdef HYPREDRV_USING_ZSTD
   run_roundtrip(COMP_ZSTD);
#endif
#ifdef HYPREDRV_USING_LZ4
   run_roundtrip(COMP_LZ4);
   run_roundtrip(COMP_LZ4HC);
#endif
#ifdef HYPREDRV_USING_BLOSC
   run_roundtrip(COMP_BLOSC);
#endif

   return 0;
}
