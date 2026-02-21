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
#include <time.h>

/*-----------------------------------------------------------------------------
 * Macros
 *-----------------------------------------------------------------------------*/

#define MALLOC_AND_CHECK(ptr, num_entries, type)                                   \
   do {                                                                            \
      (ptr) = (type*) malloc(num_entries * sizeof(type));                          \
      if ((ptr) == NULL)                                                           \
      {                                                                            \
         fprintf(stderr, "Failed while trying to allocate %zu bytes at line %d\n", \
                 num_entries * sizeof(type), __LINE__);                            \
         exit(1);                                                                  \
      }                                                                            \
   } while (0)

/*-----------------------------------------------------------------------------
 * coomat_t: data structure for sparse matrices in COO format
 *-----------------------------------------------------------------------------*/

typedef struct
{
   int32_t   base;
   int32_t   num_rows;
   int64_t   num_nonzeros;
   int32_t  *rows;
   int32_t  *cols;
   double   *coefs;
} coomat_t;

/*-----------------------------------------------------------------------------
 * read_coomat_file
 *-----------------------------------------------------------------------------*/

coomat_t* read_coomat_file(const char* filename)
{
   coomat_t *mat;
   int32_t   row, col;
   double    coef;
   FILE     *file = fopen(filename, "r");

   if (file == NULL)
   {
      perror("Failed to open file");
      exit(EXIT_FAILURE);
   }

   // Allocate matrix
   MALLOC_AND_CHECK(mat, 1, coomat_t);

   // Read header
   if (fscanf(file, "%d %ld", &mat->num_rows, &mat->num_nonzeros) != 2)
   {
      fprintf(stderr, "Failed to read header from file: %s\n", filename);
      fclose(file);
      free(mat);
      exit(EXIT_FAILURE);
   }

   // Allocate memory for arrays
   MALLOC_AND_CHECK(mat->rows, mat->num_nonzeros, int32_t);
   MALLOC_AND_CHECK(mat->cols, mat->num_nonzeros, int32_t);
   MALLOC_AND_CHECK(mat->coefs, mat->num_nonzeros, double);

   // Read the remaining entries
   mat->base = -1;
   for (int64_t i = 0; i < mat->num_nonzeros; i++)
   {
      if (fscanf(file, "%d %d %lf", &row, &col, &coef) != 3)
      {
         fprintf(stderr, "Failed to read header from file: %s\n", filename);
         fclose(file);
         free(mat);
         exit(EXIT_FAILURE);
      }
      mat->rows[i]  = row;
      mat->cols[i]  = col;
      mat->coefs[i] = coef;

      // Determine base
      if (mat->base == -1 && row == 1)
      {
         mat->base = 1;
      }
      else if (row == 0)
      {
         mat->base = 0;
      }
   }

   // Update to base 0
   if (mat->base)
   {
      for (int64_t i = 0; i < mat->num_nonzeros; i++)
      {
         mat->rows[i] = mat->rows[i] - mat->base;
         mat->cols[i] = mat->cols[i] - mat->base;
      }
      mat->base = 0;
   }

   fclose(file);

   return mat;
}

/*-----------------------------------------------------------------------------
 * free_coomat
 *-----------------------------------------------------------------------------*/

void free_coomat(coomat_t **mat)
{
   if (mat == NULL || *mat == NULL)
   {
      return;
   }

   // Free the internal arrays
   free((*mat)->rows);
   free((*mat)->cols);
   free((*mat)->coefs);

   // Free the structure itself
   free(*mat);

   // Set the pointer to NULL
   *mat = NULL;
}

/* Compare for qsort: index_array entries are 3 int64_t (row, col, orig_index). */
static int
compare_coo_index(const void *a, const void *b)
{
   const int64_t *ia = (const int64_t *)a;
   const int64_t *ib = (const int64_t *)b;
   if (ia[0] != ib[0])
   {
      return (ia[0] < ib[0]) ? -1 : 1;
   }
   return (ia[1] < ib[1]) ? -1 : (ia[1] > ib[1]) ? 1 : 0;
}

/*-----------------------------------------------------------------------------
 * read_mtx_file
 *-----------------------------------------------------------------------------*/

coomat_t* read_mtx_file(const char* filename)
{
   coomat_t *mat;
   int32_t   num_rows, num_cols, row, col;
   int64_t   num_nonzeros, count = 0;
   double    coef;
   FILE     *file = fopen(filename, "r");
   char      line[1024];
   int       symmetric = 0;

   if (file == NULL)
   {
      perror("Failed to open file");
      exit(EXIT_FAILURE);
   }

   // Allocate matrix
   mat = (coomat_t*) malloc(sizeof(coomat_t));
   if (mat == NULL)
   {
      fprintf(stderr, "Failed to allocate memory for coomat_t\n");
      fclose(file);
      exit(EXIT_FAILURE);
   }

   // Read header to check for symmetry and size
   while (fgets(line, sizeof(line), file))
   {
      if (line[0] == '%') // Comment or header line
      {
         if (strstr(line, "symmetric"))
         {
            symmetric = 1;
         }
         continue;
      }
      else
      {
         sscanf(line, "%d %d %ld", &num_rows, &num_cols, &num_nonzeros);
         break;
      }
   }

   mat->num_rows = num_rows;
   mat->num_nonzeros = num_nonzeros;
   mat->base = 1; // Assume 1-based indexing from MTX format

   // Allocate memory for arrays
   mat->rows = (int32_t*) malloc(2 * num_nonzeros * sizeof(int32_t));
   mat->cols = (int32_t*) malloc(2 * num_nonzeros * sizeof(int32_t));
   mat->coefs = (double*) malloc(2 * num_nonzeros * sizeof(double));

   if (mat->rows == NULL || mat->cols == NULL || mat->coefs == NULL)
   {
      fprintf(stderr, "Failed to allocate memory for matrix data\n");
      fclose(file);
      free(mat->rows);
      free(mat->cols);
      free(mat->coefs);
      free(mat);
      exit(EXIT_FAILURE);
   }

   // Read the data entries
   while (fgets(line, sizeof(line), file))
   {
      if (sscanf(line, "%d %d %lf", &row, &col, &coef) == 3)
      {
         mat->rows[count] = row;
         mat->cols[count] = col;
         mat->coefs[count] = coef;
         count++;

         if (symmetric && row != col)
         {
            mat->rows[count] = col;
            mat->cols[count] = row;
            mat->coefs[count] = coef;
            count++;
         }
      }
   }

   fclose(file);

   // Update the actual number of nonzeros if symmetric
   mat->num_nonzeros = count;

   // Sort the matrix entries by row and then by column
   int64_t *index_array = (int64_t*) malloc(3 * mat->num_nonzeros * sizeof(int64_t));
   if (index_array == NULL)
   {
      fprintf(stderr, "Failed to allocate memory for sorting\n");
      free_coomat(&mat);
      exit(EXIT_FAILURE);
   }
   for (int64_t i = 0; i < mat->num_nonzeros; i++)
   {
      index_array[3*i] = mat->rows[i];
      index_array[3*i+1] = mat->cols[i];
      index_array[3*i+2] = i;  // Store original index for coefficient reference
   }
   qsort(index_array, mat->num_nonzeros, sizeof(int64_t) * 3, compare_coo_index);

   int32_t *sorted_rows = (int32_t*) malloc(mat->num_nonzeros * sizeof(int32_t));
   int32_t *sorted_cols = (int32_t*) malloc(mat->num_nonzeros * sizeof(int32_t));
   double  *sorted_coefs = (double*) malloc(mat->num_nonzeros * sizeof(double));

   if (sorted_rows == NULL || sorted_cols == NULL || sorted_coefs == NULL)
   {
      fprintf(stderr, "Failed to allocate memory for sorted data\n");
      free(index_array);
      free_coomat(&mat);
      exit(EXIT_FAILURE);
   }

   for (int64_t i = 0; i < mat->num_nonzeros; i++)
   {
      int64_t orig_index = index_array[3*i+2];
      sorted_rows[i] = mat->rows[orig_index] - mat->base;
      sorted_cols[i] = mat->cols[orig_index] - mat->base;
      sorted_coefs[i] = mat->coefs[orig_index];
   }

   free(mat->rows);
   free(mat->cols);
   free(mat->coefs);
   free(index_array);

   mat->rows = sorted_rows;
   mat->cols = sorted_cols;
   mat->coefs = sorted_coefs;

   return mat;
}

/*-----------------------------------------------------------------------------
 * read_ijbin
 *-----------------------------------------------------------------------------*/

coomat_t *
read_ijbin(const char *filename)
{
   coomat_t *mat;
   int64_t   header[11];
   FILE     *fp = fopen(filename, "rb");

   if (fp == NULL)
   {
      perror("Failed to open file");
      exit(EXIT_FAILURE);
   }

   if (fread(header, sizeof(int64_t), 11, fp) != 11)
   {
      fprintf(stderr, "Failed to read IJ binary header from %s\n", filename);
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   if (header[0] != 1)
   {
      fprintf(stderr, "Unsupported IJ binary header version %lld in %s\n",
              (long long)header[0], filename);
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   if (header[1] != (int64_t)sizeof(int32_t) || header[2] != (int64_t)sizeof(double))
   {
      fprintf(stderr, "IJ binary format uses unsupported index/value sizes in %s\n", filename);
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   MALLOC_AND_CHECK(mat, 1, coomat_t);
   mat->base         = 0;
   mat->num_rows     = (int32_t)header[3];
   mat->num_nonzeros = header[5];
   MALLOC_AND_CHECK(mat->rows, mat->num_nonzeros, int32_t);
   MALLOC_AND_CHECK(mat->cols, mat->num_nonzeros, int32_t);
   MALLOC_AND_CHECK(mat->coefs, mat->num_nonzeros, double);

   if (fread(mat->rows, sizeof(int32_t), (size_t)mat->num_nonzeros, fp) != (size_t)mat->num_nonzeros ||
       fread(mat->cols, sizeof(int32_t), (size_t)mat->num_nonzeros, fp) != (size_t)mat->num_nonzeros ||
       fread(mat->coefs, sizeof(double), (size_t)mat->num_nonzeros, fp) != (size_t)mat->num_nonzeros)
   {
      fprintf(stderr, "Failed to read matrix data from %s\n", filename);
      fclose(fp);
      free_coomat(&mat);
      exit(EXIT_FAILURE);
   }

   fclose(fp);
   return mat;
}

/*-----------------------------------------------------------------------------
 * write_ijbin
 *-----------------------------------------------------------------------------*/

void write_ijbin(coomat_t *mat, const char *filename)
{
   size_t  count = 11;  /* Number of elements in the header */
   int64_t header[11];

   /* Fill header entries */
   header[0]  = (int64_t) 1;                  /* Header version */
   header[1]  = (int64_t) sizeof(int32_t);    /* 32-bit integers */
   header[2]  = (int64_t) sizeof(double);     /* Double precision data */
   header[3]  = (int64_t) mat->num_rows;      /* Number of rows */
   header[4]  = (int64_t) mat->num_rows;      /* Number of columns */
   header[5]  = (int64_t) mat->num_nonzeros;  /* Global number of nonzeros */
   header[6]  = (int64_t) mat->num_nonzeros;  /* Local  number of nonzeros */
   header[7]  = (int64_t) 0;                  /* Local initial row index */
   header[8]  = (int64_t) mat->num_rows - 1;  /* Local last row index */
   header[9]  = (int64_t) 0;                  /* Local initial column index */
   header[10] = (int64_t) mat->num_rows - 1;  /* Local last column index */

   FILE *fp = fopen(filename, "wb");
   if (fp == NULL)
   {
      perror("Failed to open file for writing");
      exit(EXIT_FAILURE);
   }

   /* Write header to file */
   if (fwrite((const void*) header, sizeof(int64_t), count, fp) != count)
   {
      perror("Failed to write header");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   size_t nnz = (size_t)mat->num_nonzeros;

   /* Write row indices to file */
   if (fwrite((const void *)mat->rows, sizeof(int32_t), nnz, fp) != nnz)
   {
      perror("Failed to write row indices");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   /* Write column indices to file */
   if (fwrite((const void *)mat->cols, sizeof(int32_t), nnz, fp) != nnz)
   {
      perror("Failed to write column indices");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   /* Write coefficients to file */
   if (fwrite((const void *)mat->coefs, sizeof(double), nnz, fp) != nnz)
   {
      perror("Failed to write coefficients");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   fclose(fp);
}

/*-----------------------------------------------------------------------------
 * write_mtx: Matrix Market coordinate format (1-based, real general)
 *-----------------------------------------------------------------------------*/

void write_mtx(coomat_t *mat, const char *filename)
{
   FILE *fp = fopen(filename, "w");
   if (fp == NULL)
   {
      perror("Failed to open file for writing");
      exit(EXIT_FAILURE);
   }

   fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
   fprintf(fp, "%% %d x %d matrix, %lld nonzeros (written by mat2ijbin)\n",
           (int)mat->num_rows, (int)mat->num_rows, (long long)mat->num_nonzeros);
   fprintf(fp, "%d %d %lld\n", (int)mat->num_rows, (int)mat->num_rows, (long long)mat->num_nonzeros);

   for (int64_t i = 0; i < mat->num_nonzeros; i++)
   {
      fprintf(fp, "%d %d %.16g\n",
              (int)(mat->rows[i] + 1), (int)(mat->cols[i] + 1), mat->coefs[i]);
   }

   fclose(fp);
}

/*-----------------------------------------------------------------------------
 * write_coo: plain COO text (num_rows num_nonzeros, then row col value, 0-based)
 *-----------------------------------------------------------------------------*/

void write_coo(coomat_t *mat, const char *filename)
{
   FILE *fp = fopen(filename, "w");
   if (fp == NULL)
   {
      perror("Failed to open file for writing");
      exit(EXIT_FAILURE);
   }

   fprintf(fp, "%d %lld\n", (int)mat->num_rows, (long long)mat->num_nonzeros);
   for (int64_t i = 0; i < mat->num_nonzeros; i++)
   {
      fprintf(fp, "%d %d %.16g\n", (int)mat->rows[i], (int)mat->cols[i], mat->coefs[i]);
   }

   fclose(fp);
}

/*-----------------------------------------------------------------------------
 * format from extension
 *-----------------------------------------------------------------------------*/

typedef enum
{
   FMT_UNKNOWN,
   FMT_COO,
   FMT_MTX,
   FMT_IJBIN
} format_t;

static format_t
format_from_ext(const char *filename)
{
   const char *dot = strrchr(filename, '.');
   if (!dot || dot == filename)
   {
      return FMT_UNKNOWN;
   }
   if (strcmp(dot, ".coo") == 0)
   {
      return FMT_COO;
   }
   if (strcmp(dot, ".mtx") == 0)
   {
      return FMT_MTX;
   }
   if (strcmp(dot, ".bin") == 0 || strcmp(dot, ".ijbin") == 0)
   {
      return FMT_IJBIN;
   }
   return FMT_UNKNOWN;
}

static const char *
format_name(format_t fmt)
{
   switch (fmt)
   {
      case FMT_COO:
         return "COO";
      case FMT_MTX:
         return "MTX";
      case FMT_IJBIN:
         return "IJBIN";
      default:
         return "unknown";
   }
}

/*-----------------------------------------------------------------------------
 * usage
 *-----------------------------------------------------------------------------*/

static void
usage(const char *prog)
{
   fprintf(stderr,
           "Usage: %s -i <input> -o <output> [options]\n"
           "\n"
           "Convert sparse matrices between COO, Matrix Market (MTX), and IJ binary formats.\n"
           "Input and output formats are inferred from file extensions.\n"
           "\n"
           "Required:\n"
           "  -i <file>    Input matrix file\n"
           "  -o <file>    Output matrix file\n"
           "\n"
           "Supported formats (by extension):\n"
           "  .coo         Plain COO text: \"nrows nnz\" then \"row col value\" per line (0- or 1-based)\n"
           "  .mtx         Matrix Market coordinate (real general; symmetric expanded on read)\n"
           "  .bin, .ijbin IJ binary: 11 int64 header + int32 rows/cols + double values\n"
           "\n"
           "Options:\n"
           "  -h, --help   Show this help and exit\n"
           "  -q           Quiet: no progress or timing messages\n"
           "\n"
           "Examples:\n"
           "  %s -i matrix.mtx -o matrix.bin     # MTX -> IJ binary\n"
           "  %s -i matrix.bin -o matrix.mtx     # IJ binary -> MTX\n"
           "  %s -i A.coo -o A.mtx               # COO -> Matrix Market\n"
           "  %s -i data.ijbin -o data.coo       # IJ binary -> COO\n",
           prog, prog, prog, prog, prog);
}

/*-----------------------------------------------------------------------------
 * main
 *-----------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
   const char *input_filename  = NULL;
   const char *output_filename = NULL;
   int         quiet           = 0;
   format_t    in_fmt, out_fmt;
   coomat_t   *matrix = NULL;
   clock_t     start, end;
   double      cpu_time;

   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-i") == 0)
      {
         if (i + 1 >= argc)
         {
            fprintf(stderr, "Missing argument for -i\n");
            usage(argv[0]);
            return EXIT_FAILURE;
         }
         input_filename = argv[++i];
      }
      else if (strcmp(argv[i], "-o") == 0)
      {
         if (i + 1 >= argc)
         {
            fprintf(stderr, "Missing argument for -o\n");
            usage(argv[0]);
            return EXIT_FAILURE;
         }
         output_filename = argv[++i];
      }
      else if (strcmp(argv[i], "-q") == 0)
      {
         quiet = 1;
      }
      else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         usage(argv[0]);
         return EXIT_SUCCESS;
      }
      else
      {
         fprintf(stderr, "Unknown option: %s\n", argv[i]);
         usage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   if (input_filename == NULL || output_filename == NULL)
   {
      fprintf(stderr, "Error: both -i and -o are required.\n\n");
      usage(argv[0]);
      return EXIT_FAILURE;
   }

   in_fmt  = format_from_ext(input_filename);
   out_fmt = format_from_ext(output_filename);

   if (in_fmt == FMT_UNKNOWN)
   {
      fprintf(stderr, "Error: input format not recognized (use .coo, .mtx, .bin, or .ijbin): %s\n",
              input_filename);
      return EXIT_FAILURE;
   }
   if (out_fmt == FMT_UNKNOWN)
   {
      fprintf(stderr, "Error: output format not recognized (use .coo, .mtx, .bin, or .ijbin): %s\n",
              output_filename);
      return EXIT_FAILURE;
   }
   if (in_fmt == out_fmt)
   {
      fprintf(stderr, "Error: input and output format are the same (%s). Use different extensions.\n",
              format_name(in_fmt));
      return EXIT_FAILURE;
   }

   /* Read */
   start = clock();
   if (in_fmt == FMT_COO)
   {
      if (!quiet)
      {
         printf("Reading COO %s ...\n", input_filename);
      }
      matrix = read_coomat_file(input_filename);
   }
   else if (in_fmt == FMT_MTX)
   {
      if (!quiet)
      {
         printf("Reading MTX %s ...\n", input_filename);
      }
      matrix = read_mtx_file(input_filename);
   }
   else
   {
      if (!quiet)
      {
         printf("Reading IJ binary %s ...\n", input_filename);
      }
      matrix = read_ijbin(input_filename);
   }

   if (!matrix)
   {
      fprintf(stderr, "Failed to read matrix from %s\n", input_filename);
      return EXIT_FAILURE;
   }

   end   = clock();
   cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
   if (!quiet)
   {
      printf("  %d x %d, %lld nnz (read %.3f s)\n",
             (int)matrix->num_rows, (int)matrix->num_rows, (long long)matrix->num_nonzeros, cpu_time);
   }

   /* Write */
   start = clock();
   if (out_fmt == FMT_IJBIN)
   {
      if (!quiet)
      {
         printf("Writing IJ binary %s ...\n", output_filename);
      }
      write_ijbin(matrix, output_filename);
   }
   else if (out_fmt == FMT_MTX)
   {
      if (!quiet)
      {
         printf("Writing MTX %s ...\n", output_filename);
      }
      write_mtx(matrix, output_filename);
   }
   else
   {
      if (!quiet)
      {
         printf("Writing COO %s ...\n", output_filename);
      }
      write_coo(matrix, output_filename);
   }

   end      = clock();
   cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
   if (!quiet)
   {
      printf("  (write %.3f s)\n", cpu_time);
   }

   free_coomat(&matrix);
   return EXIT_SUCCESS;
}
