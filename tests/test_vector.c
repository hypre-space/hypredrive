#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "containers.h"
#include "error.h"
#include "linsys.h"
#include "test_helpers.h"

static void
create_temp_binary(const char *prefix, uint32_t part_id, uint64_t nrows,
                   const double *values)
{
   char filename[256];
   snprintf(filename, sizeof(filename), "%s.%05u.bin", prefix, part_id);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[8] = {0};
   header[1]          = sizeof(double);
   header[5]          = nrows;

   fwrite(header, sizeof(uint64_t), 8, fp);
   fwrite(values, sizeof(double), nrows, fp);
   fclose(fp);

   add_temp_file(filename);
}

static void
create_temp_binary_float(const char *prefix, uint32_t part_id, uint64_t nrows,
                         const float *values)
{
   char filename[256];
   snprintf(filename, sizeof(filename), "%s.%05u.bin", prefix, part_id);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[8] = {0};
   header[1]          = sizeof(float);
   header[5]          = nrows;

   fwrite(header, sizeof(uint64_t), 8, fp);
   fwrite(values, sizeof(float), nrows, fp);
   fclose(fp);

   add_temp_file(filename);
}

static void
test_IJVectorReadMultipartBinary_success(void)
{
   const char *prefix    = "test_vec_success";
   double      values[3] = {1.0, 2.0, 3.0};

   create_temp_binary(prefix, 0, 3, values);

   HYPRE_IJVector vec = NULL;
   ErrorCodeResetAll();
   IJVectorReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &vec);

   ASSERT_NOT_NULL(vec);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJVectorDestroy(vec);
   cleanup_temp_files();
}

static void
test_IJVectorReadMultipartBinary_missing_file(void)
{
   HYPRE_IJVector vec = NULL;

   ErrorCodeResetAll();
   IJVectorReadMultipartBinary("missing_vector", MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST,
                               &vec);
   ASSERT_NULL(vec);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_IJVectorReadMultipartBinary_bad_header(void)
{
   const char *prefix = "test_vec_bad_header";
   char        filename[256];
   uint64_t    header[4] = {0};

   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   fwrite(header, sizeof(uint64_t), 4, fp); /* fewer than expected */
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJVector vec = NULL;
   ErrorCodeResetAll();
   IJVectorReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &vec);
   ASSERT_NULL(vec);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJVectorReadMultipartBinary_float_coefficients(void)
{
   const char *prefix    = "test_vec_float";
   float       values[4] = {1.5f, 2.5f, 3.5f, 4.5f};

   create_temp_binary_float(prefix, 0, 4, values);

   HYPRE_IJVector vec = NULL;
   ErrorCodeResetAll();
   IJVectorReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &vec);

   ASSERT_NOT_NULL(vec);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJVectorDestroy(vec);
   cleanup_temp_files();
}

static void
test_IntArrayParRead_ascii(void)
{
   const char *prefix = "test_intarray_prefix";
   char        filename[256];

   snprintf(filename, sizeof(filename), "%s.00000", prefix);
   FILE *fp = fopen(filename, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "%zu\n", (size_t)3);
   fprintf(fp, "%d %d %d\n", 7, 8, 9);
   fclose(fp);
   add_temp_file(filename);

   IntArray *array = NULL;
   ErrorCodeResetAll();
   IntArrayParRead(MPI_COMM_SELF, prefix, &array);
   ASSERT_NOT_NULL(array);
   ASSERT_EQ(array->size, 3);
   ASSERT_EQ(array->data[0], 7);
   ASSERT_EQ(array->data[2], 9);
   ASSERT_EQ(array->unique_size, 3);
   ASSERT_EQ(array->g_unique_size, 3);

   IntArrayDestroy(&array);
   cleanup_temp_files();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_IJVectorReadMultipartBinary_success);
   RUN_TEST(test_IJVectorReadMultipartBinary_missing_file);
   RUN_TEST(test_IJVectorReadMultipartBinary_bad_header);
   RUN_TEST(test_IJVectorReadMultipartBinary_float_coefficients);
   RUN_TEST(test_IntArrayParRead_ascii);

   MPI_Finalize();
   return 0;
}
