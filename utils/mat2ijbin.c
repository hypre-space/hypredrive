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

/*-----------------------------------------------------------------------------
 * compare_coo: Compare function for sorting rows and columns in coomat_t
 *-----------------------------------------------------------------------------*/

int compare_coo(const void *a, const void *b)
{
   const int32_t *row_a = (const int32_t*) a;
   const int32_t *row_b = (const int32_t*) b;

   if (row_a[0] != row_b[0])
   {
      return (row_a[0] - row_b[0]);
   }
   else
   {
      return (row_a[1] - row_b[1]);
   }
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
   qsort(index_array, mat->num_nonzeros, sizeof(int64_t) * 3, compare_coo);

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
 * write_ijbin
 *-----------------------------------------------------------------------------*/

void write_ijbin(coomat_t* mat, const char *filename)
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

   /* Write row indices to file */
   if (fwrite((const void*) mat->rows, sizeof(int32_t), mat->num_nonzeros, fp) != mat->num_nonzeros)
   {
      perror("Failed to write row indices");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   /* Write column indices to file */
   if (fwrite((const void*) mat->cols, sizeof(int32_t), mat->num_nonzeros, fp) != mat->num_nonzeros)
   {
      perror("Failed to write column indices");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   /* Write coefficients to file */
   if (fwrite((const void*) mat->coefs, sizeof(double), mat->num_nonzeros, fp) != mat->num_nonzeros)
   {
      perror("Failed to write coefficients");
      fclose(fp);
      exit(EXIT_FAILURE);
   }

   fclose(fp);
}

/*-----------------------------------------------------------------------------
 * usage
 *-----------------------------------------------------------------------------*/

void usage(const char *prog_name)
{
   fprintf(stderr, "Usage: %s -i <input_file> -o <output_file>\n", prog_name);
   fprintf(stderr, "   -i <filename> : Input file name with the matrix in COO format\n");
   fprintf(stderr, "   -o <filename> : Output file name with the matrix in ijbin format\n");
}

/*-----------------------------------------------------------------------------
 * main
 *-----------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
   const char *input_filename = NULL;
   const char *output_filename = NULL;
   char       *output_filename_with_ext = NULL;
   clock_t     start, end;
   double      cpu_time;

   // Parse command-line arguments
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-i") == 0)
      {
         if (i + 1 < argc)
         {
            input_filename = argv[++i];
         }
         else
         {
            fprintf(stderr, "Missing argument for -i\n");
            usage(argv[0]);
            return EXIT_FAILURE;
         }
      }
      else if (strcmp(argv[i], "-o") == 0)
      {
         if (i + 1 < argc)
         {
            output_filename = argv[++i];
         }
         else
         {
            fprintf(stderr, "Missing argument for -o\n");
            usage(argv[0]);
            return EXIT_FAILURE;
         }
      }
      else
      {
         fprintf(stderr, "Unknown option: %s\n", argv[i]);
         usage(argv[0]);
         return EXIT_FAILURE;
      }
   }

   // Check that both input and output filenames are provided
   if (input_filename == NULL || output_filename == NULL)
   {
      fprintf(stderr, "Both input and output filenames must be specified\n");
      usage(argv[0]);
      return EXIT_FAILURE;
   }

   // Allocate memory for output_filename with ".bin" extension
   output_filename_with_ext = (char*) malloc(strlen(output_filename) + 5); // +5 for ".bin" and null terminator
   if (output_filename_with_ext == NULL)
   {
      fprintf(stderr, "Failed to allocate memory for output filename\n");
      return EXIT_FAILURE;
   }
   strcpy(output_filename_with_ext, output_filename);
   strcat(output_filename_with_ext, ".bin");

   // Read the matrix from the input file
   // Determine which function to use based on input file extension
   coomat_t *matrix = NULL;
   const char *extension = strrchr(input_filename, '.');

   start = clock();
   if (extension && strcmp(extension, ".coo") == 0)
   {
      printf("Reading COO matrix %s...\n", input_filename);
      matrix = read_coomat_file(input_filename);
   }
   else if (extension && strcmp(extension, ".mtx") == 0)
   {
      printf("Reading MTX matrix %s...\n", input_filename);
      matrix = read_mtx_file(input_filename);
   }
   else
   {
      fprintf(stderr, "Unsupported file format: %s\n", input_filename);
      free(output_filename_with_ext);
      return EXIT_FAILURE;
   }
   if (!matrix)
   {
      fprintf(stderr, "Failed to read matrix from file: %s\n", input_filename);
      return EXIT_FAILURE;
   }
   end = clock();
   cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
   printf("Read time: %f seconds\n", cpu_time);

   // Write the matrix to the output file in ijbin format
   start = clock();
   printf("Writing %s...\n", output_filename_with_ext);
   write_ijbin(matrix, output_filename_with_ext);
   end = clock();
   cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
   printf("Write time: %f seconds\n", cpu_time);

   // Free the allocated memory
   free_coomat(&matrix);

   return EXIT_SUCCESS;
}
