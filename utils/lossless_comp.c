/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/*-----------------------------------------------------------------------------
 * Includes
 *-----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <zlib.h>
#include <zstd.h>
#include <lz4.h>
#include <lz4hc.h>
#include <blosc.h>
#include <time.h>

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

#define MALLOC_AND_CHECK(ptr, size)                                                \
   do {                                                                            \
      (ptr) = malloc(size);                                                        \
      if ((ptr) == NULL)                                                           \
      {                                                                            \
         fprintf(stderr, "Failed while trying to allocate %zu bytes at line %d\n", \
                 size, __LINE__);                                                  \
         exit(1);                                                                  \
      }                                                                            \
   } while (0)

/*-----------------------------------------------------------------------------
 * comp_alg_t: enum to identify the compression algorithm
 *-----------------------------------------------------------------------------*/

typedef enum
{
   COMP_NONE,
   COMP_ZLIB,
   COMP_ZSTD,
   COMP_LZ4,
   COMP_LZ4HC,
   COMP_BLOSC
} comp_alg_t;

const char *comp_names[] = {"none", "zlib", "zstd", "lz4", "lz4hc", "blosc"};
const char *comp_exts[]  = {".bin", ".zlib.bin", ".zst.bin", ".lz4.bin", ".lz4.bin", ".blosc.bin"};

/*-----------------------------------------------------------------------------
 * lossless_compress
 *-----------------------------------------------------------------------------*/

void
lossless_compress(comp_alg_t  algo,
                  size_t      isize,
                  const void *input,
                  size_t     *osize_ptr,
                  void      **output_ptr)
{
   size_t  header_size = sizeof(size_t);
   size_t  comp_size;

   switch (algo)
   {
      case COMP_NONE:
      {
         *osize_ptr = isize;
         MALLOC_AND_CHECK(*output_ptr, isize);
         memcpy(*output_ptr, input, isize);
         return;
      }
      case COMP_ZLIB:
      {
         comp_size = (size_t) compressBound(isize);
         MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         int ierr = compress((unsigned char *)(*output_ptr) + header_size, &comp_size, input, isize);
         if (ierr != Z_OK)
         {
            fprintf(stderr, "ZLIB compression error: %d", ierr);
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_ZSTD:
      {
         comp_size = ZSTD_compressBound(isize);
         MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = ZSTD_compress((unsigned char *)(*output_ptr) + header_size, comp_size, input, isize, 1);
         if (ZSTD_isError(comp_size))
         {
            fprintf(stderr, "ZSTD compression error: %s", ZSTD_getErrorName(comp_size));
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_LZ4:
      {
         comp_size = LZ4_compressBound(isize);
         MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = LZ4_compress_default(input, (unsigned char *)(*output_ptr) + header_size, isize, comp_size);
         if (comp_size <= 0)
         {
            fprintf(stderr, "LZ4 compression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_LZ4HC:
      {
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
         break;
      }
      case COMP_BLOSC:
      {
         blosc_init();
         blosc_set_compressor("blosclz");

         comp_size = isize + BLOSC_MAX_OVERHEAD;
         MALLOC_AND_CHECK(*output_ptr, header_size + comp_size);

         /* Store the original size at the beginning of the buffer */
         *(size_t *)(*output_ptr) = isize;

         comp_size = blosc_compress(9, 1, 1, isize, input, (unsigned char *)(*output_ptr) + header_size, comp_size);
         blosc_destroy();
         if (comp_size <= 0)
         {
            fprintf(stderr, "Blosc compression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      default:
      {
         fprintf(stderr, "Unknown or unsupported compression algorithm: %d", algo);
         return;
      }
   }

   /* Update output size */
   *osize_ptr = header_size + comp_size;

   return;
}

/*-----------------------------------------------------------------------------
 * lossless_decompress
 *-----------------------------------------------------------------------------*/

void
lossless_decompress(comp_alg_t  algo,
                    size_t      isize,
                    const void *input,
                    size_t     *osize_ptr,
                    void      **output_ptr)
{
   size_t  header_size = sizeof(size_t);
   size_t  orig_size = *(size_t *)input;

   MALLOC_AND_CHECK(*output_ptr, orig_size);

   switch (algo)
   {
      case COMP_NONE:
      {
         *osize_ptr = isize;
         MALLOC_AND_CHECK(*output_ptr, isize);
         memcpy(*output_ptr, input, isize);
         return;
      }
      case COMP_ZLIB:
      {
         int ierr = uncompress((unsigned char *)(*output_ptr), &orig_size,
                               (unsigned char *)input + header_size, isize - header_size);
         if (ierr != Z_OK)
         {
            fprintf(stderr, "ZLIB decompression error: %d", ierr);
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_ZSTD:
      {
         size_t result = ZSTD_decompress((unsigned char *)(*output_ptr), orig_size,
                                         (unsigned char *)input + header_size, isize - header_size);
         if (ZSTD_isError(result))
         {
            fprintf(stderr, "ZSTD decompression error: %s", ZSTD_getErrorName(result));
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_LZ4:
      case COMP_LZ4HC:
      {
         int result = LZ4_decompress_safe((const char *)input + header_size,
                                          (char *)(*output_ptr), isize - header_size, orig_size);
         if (result < 0)
         {
            fprintf(stderr, "LZ4 decompression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      case COMP_BLOSC:
      {
         blosc_init();

         int result = blosc_decompress((const void *)((unsigned char *)input + header_size),
                                       *output_ptr, orig_size);
         blosc_destroy();
         if (result <= 0)
         {
            fprintf(stderr, "Blosc decompression failed!");
            free(*output_ptr); *output_ptr = NULL;
            return;
         }
         break;
      }
      default:
      {
         fprintf(stderr, "Unknown or unsupported decompression algorithm: %d", algo);
         return;
      }
   }

   /* Update output size */
   *osize_ptr = orig_size;

   return;
}

/*-----------------------------------------------------------------------------
 * get_compressed_filename
 *-----------------------------------------------------------------------------*/

char*
get_compressed_filename(const char *original_filename,
                        comp_alg_t  algorithm)
{
   // Get the length of the original filename and the chosen extension
   size_t filename_len  = strlen(original_filename);
   size_t extension_len = strlen(comp_exts[algorithm]);

   // Allocate memory for the new filename, including the null terminator
   char *output_filename = (char *)malloc(filename_len + extension_len + 1);

   // Copy the original filename into the new buffer
   strcpy(output_filename, original_filename);

   // Append the appropriate extension based on the compression algorithm
   strcat(output_filename, comp_exts[algorithm]);

   return output_filename;
}

/*-----------------------------------------------------------------------------
 * print_usage
 *-----------------------------------------------------------------------------*/

void
print_usage(const char *program_name)
{
   fprintf(stderr, "(De)compresses binary data and writes binary result to disk\n\n");
   fprintf(stderr, "  Usage: %s -i <inputfilename> -o <outputfilename> -m <c/comp/d/decomp> -a <zlib/zstd/lz4/lz4hc/blosc>\n", program_name);
}

/*-----------------------------------------------------------------------------
 * parse_arguments
 *-----------------------------------------------------------------------------*/

bool
parse_arguments(int          argc,
                char        *argv[],
                char       **input_filename,
                char       **output_filename,
                bool        *is_compress,
                comp_alg_t  *algorithm)
{
   if (argc != 9)
   {
      print_usage(argv[0]);
      return false;
   }

   for (int i = 1; i < argc; i += 2)
   {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help") || !strcmp(argv[i], "--help"))
      {
         print_usage(argv[0]);
         return false;
      }
      if (i + 1 >= argc)
      {
         fprintf(stderr, "Error: Missing value for argument %s\n", argv[i]);
         print_usage(argv[0]);
         return false;
      }
      if (!strcmp(argv[i], "-i"))
      {
         *input_filename = argv[i + 1];
      }
      else if (!strcmp(argv[i], "-o"))
      {
         *output_filename = argv[i + 1];
      }
      else if (!strcmp(argv[i], "-m"))
      {
         if (!strcmp(argv[i + 1], "comp") || !strcmp(argv[i + 1], "c"))
         {
            *is_compress = true;
         }
         else if (!strcmp(argv[i + 1], "decomp") || !strcmp(argv[i + 1], "d"))
         {
            *is_compress = false;
         }
         else
         {
            fprintf(stderr, "Invalid mode. Use 'comp' for compression or 'decomp' for decompression.\n");
            return false;
         }
      }
      else if (!strcmp(argv[i], "-a"))
      {
         if (!strcmp(argv[i + 1], "none"))
         {
            *algorithm = COMP_NONE;
         }
         else if (!strcmp(argv[i + 1], "zlib"))
         {
            *algorithm = COMP_ZLIB;
         }
         else if (!strcmp(argv[i + 1], "zstd"))
         {
            *algorithm = COMP_ZSTD;
         }
         else if (!strcmp(argv[i + 1], "lz4"))
         {
            *algorithm = COMP_LZ4;
         }
         else if (!strcmp(argv[i + 1], "lz4hc"))
         {
            *algorithm = COMP_LZ4HC;
         }
         else if (!strcmp(argv[i + 1], "blosc"))
         {
            *algorithm = COMP_BLOSC;
         }
         else
         {
            fprintf(stderr, "Invalid algorithm. Use 'none', 'zlib', 'zstd', 'lz4', 'lz4hc', or 'blosc'.\n");
            return false;
         }
      }
      else
      {
         fprintf(stderr, "Unknown argument: %s\n", argv[i]);
         print_usage(argv[0]);
         return false;
      }
   }

   return true;
}

/*-----------------------------------------------------------------------------
 * main
 *-----------------------------------------------------------------------------*/

int
main(int argc, char *argv[])
{
   char         *input_filename = NULL;
   char         *output_filename = NULL;
   bool          is_compress = false;
   comp_alg_t    algorithm;

   void         *input_data = NULL;
   void         *output_data = NULL;
   size_t        output_size = 0;

   if (!parse_arguments(argc, argv, &input_filename, &output_filename, &is_compress, &algorithm))
   {
       return EXIT_FAILURE;
   }

   char *extended_output_filename = is_compress ?
     get_compressed_filename(output_filename, algorithm) : output_filename;

   // Print compressor used
   printf(" -   Using compressor: %s\n", comp_names[algorithm]);

   // Start timer for file reading
   clock_t start_time = clock();

   // Open the input file
   FILE *input_file = fopen(input_filename, "rb");
   if (input_file == NULL)
   {
       perror("Error opening input file");
       return EXIT_FAILURE;
   }

   // Get the input file size
   fseek(input_file, 0, SEEK_END);
   size_t input_size = ftell(input_file);
   rewind(input_file);

   // Allocate memory for the input data
   MALLOC_AND_CHECK(input_data, input_size);

   // Read the input file
   if (fread(input_data, 1, input_size, input_file) != input_size)
   {
       perror("Error reading input file");
       free(input_data);
       fclose(input_file);
       return EXIT_FAILURE;
   }

   // File reading phase
   clock_t file_read_time = clock();
   printf(" -  File reading time: %.3f seconds\n",
          (double)(file_read_time - start_time) / CLOCKS_PER_SEC);

   if (is_compress)
   {
      // Compress the data
      clock_t compress_time_start = clock();
      lossless_compress(algorithm, input_size, input_data, &output_size, &output_data);
      clock_t compress_time_end = clock();

      free(input_data);

      // Output compression statistics
      printf(" -      Original size: %zu bytes (%.2f GB)\n", input_size, (double) input_size / (double) (1 << 30));
      printf(" -    Compressed size: %zu bytes (%.2f GB)\n", output_size, (double) output_size / (double) (1 << 30));
      if (input_size != 0)
      {
         double compression_ratio = (double)output_size / (double)input_size;
         printf(" -  Compression ratio: %.2f%%\n", (1 - compression_ratio) * 100.0);
      }

      printf(" -   Compression time: %.3f seconds\n",
             (double)(compress_time_end - compress_time_start) / CLOCKS_PER_SEC);
   }
   else
   {
      // Decompress the data
      clock_t decompress_time_start = clock();
      lossless_decompress(algorithm, input_size, input_data, &output_size, &output_data);
      clock_t decompress_time_end = clock();

      free(input_data);

      // Output decompression statistics
      printf(" -      Original size: %zu bytes (%.2f GB)\n", input_size, (double) input_size / (double) (1 << 30));
      printf(" -  Decompressed size: %zu bytes (%.2f GB)\n", output_size, (double) output_size / (double) (1 << 30));
      if (output_size != 0)
      {
          double expansion_ratio = (double)output_size / (double)input_size;
          printf(" -    Expansion ratio: %.2f%%\n", (expansion_ratio - 1) * 100.0);
      }

      printf(" - Decompression time: %.3f seconds\n",
             (double)(decompress_time_end - decompress_time_start) / CLOCKS_PER_SEC);
   }

   fclose(input_file);

   if (output_data == NULL)
   {
      fprintf(stderr, "Error during %s.\n", is_compress ? "compression" : "decompression");
      return EXIT_FAILURE;
   }

   // Start timer for file writing
   clock_t file_write_time_start = clock();

   // Write the output file
   FILE *output_file = fopen(extended_output_filename, "wb");
   if (output_file == NULL)
   {
      perror("Error opening output file");
      free(output_data);
      return EXIT_FAILURE;
   }

   if (fwrite(output_data, 1, output_size, output_file) != output_size)
   {
      perror("Error writing output file");
      free(output_data);
      fclose(output_file);
      return EXIT_FAILURE;
   }
   fclose(output_file);

   // End timer for file writing
   clock_t file_write_time_end = clock();

   free(output_data);

   printf(" -  File writing time: %.3f seconds\n",
          (double)(file_write_time_end - file_write_time_start) / CLOCKS_PER_SEC);

   printf(" -  Output written to: %s\n", extended_output_filename);
   if (is_compress) free(extended_output_filename);

   return EXIT_SUCCESS;
}
