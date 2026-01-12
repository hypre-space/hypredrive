#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "error.h"
#include "linsys.h"
#include "test_helpers.h"

static void
create_matrix_part_typed(const char *prefix, uint32_t part_id, HYPRE_BigInt row_lower,
                         HYPRE_BigInt row_upper, size_t nnz, const void *rows,
                         size_t row_elem_size, const void *cols, size_t col_elem_size,
                         const void *vals, size_t val_elem_size)
{
   char filename[256];
   snprintf(filename, sizeof(filename), "%s.%05u.bin", prefix, part_id);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[11] = {0};
   header[1]           = (uint64_t)row_elem_size;
   header[2]           = (uint64_t)val_elem_size;
   header[5]           = (uint64_t)(row_upper - row_lower + 1);
   header[6]           = (uint64_t)nnz;
   header[7]           = (uint64_t)row_lower;
   header[8]           = (uint64_t)row_upper;

   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   if (nnz > 0)
   {
      ASSERT_EQ(fwrite(rows, row_elem_size, nnz, fp), nnz);
      ASSERT_EQ(fwrite(cols, col_elem_size, nnz, fp), nnz);
      ASSERT_EQ(fwrite(vals, val_elem_size, nnz, fp), nnz);
   }
   fclose(fp);

   add_temp_file(filename);
}

static void
create_matrix_part(const char *prefix, uint32_t part_id, HYPRE_BigInt row_lower,
                   HYPRE_BigInt row_upper, size_t nnz, const HYPRE_BigInt *rows,
                   const HYPRE_BigInt *cols, const double *vals)
{
   create_matrix_part_typed(prefix, part_id, row_lower, row_upper, nnz, rows,
                            sizeof(HYPRE_BigInt), cols, sizeof(HYPRE_BigInt), vals,
                            sizeof(double));
}

static void
test_IJMatrixReadMultipartBinary_success(void)
{
   const char    *prefix  = "test_matrix_success";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   const double   vals[1] = {5.0};
   HYPRE_IJMatrix mat     = NULL;
   void          *obj     = NULL;

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_missing_file(void)
{
   HYPRE_IJMatrix mat = NULL;

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary("missing_matrix", MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST,
                               &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_IJMatrixReadMultipartBinary_short_header(void)
{
   const char *prefix = "test_matrix_short";
   char        filename[256];

   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[5] = {0};
   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 5, fp), 5u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_short_header_device_path(void)
{
   /* Covers the second-pass header read error path (skips host-side precompute). */
   const char *prefix = "test_matrix_short_dev";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[5] = {0};
   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 5, fp), 5u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint32_truncated_rows_device_path(void)
{
   /* Trigger fread(row indices) != header[6] in the device path. */
   const char *prefix = "test_matrix_u32_trunc_rows_dev";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(uint32_t); /* row/col dtype */
   header[2]           = (uint64_t)sizeof(double);   /* val dtype */
   header[5]           = 1;                          /* nrows */
   header[6]           = 2;                          /* nnz */
   header[7]           = 0;                          /* row_lower */
   header[8]           = 0;                          /* row_upper */
   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 11, fp), 11u);

   uint32_t rows32[1] = {0}; /* intentionally short (need 2) */
   ASSERT_EQ(fwrite(rows32, sizeof(uint32_t), 1, fp), 1u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint32_truncated_cols_device_path(void)
{
   /* Trigger fread(col indices) != header[6] in the device path after rows succeed. */
   const char *prefix = "test_matrix_u32_trunc_cols_dev";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(uint32_t);
   header[2]           = (uint64_t)sizeof(double);
   header[5]           = 1;
   header[6]           = 2;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 11, fp), 11u);

   uint32_t rows32[2] = {0, 0};
   ASSERT_EQ(fwrite(rows32, sizeof(uint32_t), 2, fp), 2u);
   uint32_t cols32[1] = {0}; /* intentionally short (need 2) */
   ASSERT_EQ(fwrite(cols32, sizeof(uint32_t), 1, fp), 1u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_invalid_dtype(void)
{
   const char *prefix = "test_matrix_invalid";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);

   uint64_t header[11] = {0};
   header[1]           = 3; /* Invalid row/col size */
   header[6]           = 1;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint32_indices(void)
{
   const char    *prefix    = "test_matrix_uint32";
   uint32_t       rows32[1] = {0};
   uint32_t       cols32[1] = {0};
   double         vals[1]   = {3.0};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows32, sizeof(uint32_t), cols32,
                            sizeof(uint32_t), vals, sizeof(double));

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_float_coefficients(void)
{
   const char    *prefix  = "test_matrix_float";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   float          vals[1] = {1.5f};
   HYPRE_IJMatrix mat     = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows, sizeof(HYPRE_BigInt), cols,
                            sizeof(HYPRE_BigInt), vals, sizeof(float));

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint32_indices_float_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint32_float";
   uint32_t       rows32[1] = {0};
   uint32_t       cols32[1] = {0};
   float          vals[1]   = {2.5f};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows32, sizeof(uint32_t), cols32,
                            sizeof(uint32_t), vals, sizeof(float));

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint64_indices_double_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint64_double";
   uint64_t       rows64[1] = {0};
   uint64_t       cols64[1] = {0};
   double         vals[1]   = {4.0};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows64, sizeof(uint64_t), cols64,
                            sizeof(uint64_t), vals, sizeof(double));

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_uint64_indices_float_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint64_float";
   uint64_t       rows64[1] = {0};
   uint64_t       cols64[1] = {0};
   float          vals[1]   = {3.5f};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows64, sizeof(uint64_t), cols64,
                            sizeof(uint64_t), vals, sizeof(float));

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_IJMatrixReadMultipartBinary_invalid_value_type(void)
{
   const char    *prefix  = "test_matrix_invalid_val";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   double         vals[1] = {2.0};
   HYPRE_IJMatrix mat     = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows, sizeof(HYPRE_BigInt), cols,
                            sizeof(HYPRE_BigInt), vals, 3 /* invalid */);

   ErrorCodeResetAll();
   IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   HYPRE_Initialize();

   RUN_TEST(test_IJMatrixReadMultipartBinary_success);
   RUN_TEST(test_IJMatrixReadMultipartBinary_missing_file);
   RUN_TEST(test_IJMatrixReadMultipartBinary_short_header);
   RUN_TEST(test_IJMatrixReadMultipartBinary_short_header_device_path);
   RUN_TEST(test_IJMatrixReadMultipartBinary_invalid_dtype);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint32_indices);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint32_truncated_rows_device_path);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint32_truncated_cols_device_path);
   RUN_TEST(test_IJMatrixReadMultipartBinary_float_coefficients);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint32_indices_float_coeffs);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint64_indices_double_coeffs);
   RUN_TEST(test_IJMatrixReadMultipartBinary_uint64_indices_float_coeffs);
   RUN_TEST(test_IJMatrixReadMultipartBinary_invalid_value_type);

   HYPRE_Finalize();
   MPI_Finalize();
   return 0;
}
