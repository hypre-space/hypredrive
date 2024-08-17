/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "comp.h"

/*-----------------------------------------------------------------------------
 * hypredrv_compress
 *-----------------------------------------------------------------------------*/

void
hypredrv_compress(comp_alg_t  algo,
                  size_t      isize,
                  const void *input,
                  size_t     *osize_ptr,
                  void      **output_ptr)
{
   size_t  header_size = sizeof(size_t);
   size_t  comp_size;

   switch (algo)
   {
      case COMP_ZLIB:
      {
#ifdef HYPREDRV_USING_ZLIB
         comp_size = (size_t) compressBound(isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         int ierr = compress((unsigned char *)(*output_ptr) + header_size, &comp_size, input, isize);
         if (ierr != Z_OK)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZLIB compression error: %d", ierr);
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
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

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = ZSTD_compress((unsigned char *)(*output_ptr) + header_size, comp_size, input, isize, 1);
         if (ZSTD_isError(comp_size))
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZSTD compression error: %s", ZSTD_getErrorName(comp_size));
            free(*output_ptr); *output_ptr = NULL;
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
         comp_size = LZ4_compressBound(isize);
         HYPREDRV_MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = LZ4_compress_default(input, (unsigned char *)(*output_ptr) + header_size, isize, comp_size);
         if (comp_size <= 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("LZ4 compression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
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
         comp_size = LZ4_compressBound(isize);
         MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = LZ4_compress_HC(input, (unsigned char *)(*output_ptr) + header_size, isize, comp_size, LZ4HC_CLEVEL_DEFAULT);
         if (comp_size <= 0)
         {
            fprintf(stderr, "LZ4HC compression failed!\n");
            free(*output_ptr); *output_ptr = NULL;
            return;
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

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = blosc_compress(9, 1, 1, isize, input, (unsigned char *)(*output_ptr) + header_size, comp_size);
         blosc_destroy();
         if (comp_size <= 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("Blosc compression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
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

   /* Update output size */
   *osize_ptr = header_size + comp_size;

   return;
}

/*-----------------------------------------------------------------------------
 * hypredrv_decompress
 *-----------------------------------------------------------------------------*/

void
hypredrv_decompress(comp_alg_t  algo,
                    size_t      isize,
                    const void *input,
                    size_t     *osize_ptr,
                    void      **output_ptr)
{
   size_t  header_size = sizeof(size_t);
   size_t  orig_size = *(size_t *)input;

   HYPREDRV_MALLOC_AND_CHECK(*output_ptr, orig_size);

   switch (algo)
   {
      case COMP_ZLIB:
      {
#ifdef HYPREDRV_USING_ZLIB
         int ierr = uncompress((unsigned char *)(*output_ptr), &orig_size,
                               (unsigned char *)input + header_size, isize - header_size);
         if (ierr != Z_OK)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZLIB decompression error: %d", ierr);
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
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
         size_t result = ZSTD_decompress((unsigned char *)(*output_ptr), orig_size,
                                         (unsigned char *)input + header_size, isize - header_size);
         if (ZSTD_isError(result))
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("ZSTD decompression error: %s", ZSTD_getErrorName(result));
            free(*output_ptr); *output_ptr = NULL;
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
         int result = LZ4_decompress_safe((const char *)input + header_size,
                                          (char *)(*output_ptr), isize - header_size, orig_size);
         if (result < 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("LZ4 decompression failed!");
            free(*output_ptr); *output_ptr = NULL;
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

         int result = blosc_decompress((const void *)((unsigned char *)input + header_size),
                                       *output_ptr, orig_size);
         blosc_destroy();
         if (result <= 0)
         {
            ErrorCodeSet(ERROR_UNKNOWN);
            ErrorMsgAdd("Blosc decompression failed!");
            free(*output_ptr); *output_ptr = NULL;
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
         return;
      }
   }

   /* Update output size */
   *osize_ptr = orig_size;

   return;
}
