#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "internal/error.h"
#include "internal/linsys.h"
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

   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
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
assert_matrix_read_rejects(const char *prefix)
{
   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_negative_row_rejected(void)
{
   const char     *prefix  = "test_matrix_negative_row";
   HYPRE_BigInt    rows[1] = {-1};
   HYPRE_BigInt    cols[1] = {0};
   const double    vals[1] = {1.0};

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_negative_col_rejected(void)
{
   const char     *prefix  = "test_matrix_negative_col";
   HYPRE_BigInt    rows[1] = {0};
   HYPRE_BigInt    cols[1] = {-1};
   const double    vals[1] = {1.0};

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_row_oob_rejected(void)
{
   const char     *prefix  = "test_matrix_row_oob";
   HYPRE_BigInt    rows[1] = {1};
   HYPRE_BigInt    cols[1] = {0};
   const double    vals[1] = {1.0};

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_col_oob_rejected(void)
{
   const char     *prefix  = "test_matrix_col_oob";
   HYPRE_BigInt    rows[1] = {0};
   HYPRE_BigInt    cols[1] = {1};
   const double    vals[1] = {1.0};

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_nonfinite_float_rejected(void)
{
   const char     *prefix  = "test_matrix_nonfinite_float";
   HYPRE_BigInt    rows[1] = {0};
   HYPRE_BigInt    cols[1] = {0};
   const float     vals[1] = {NAN};

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows, sizeof(HYPRE_BigInt), cols,
                            sizeof(HYPRE_BigInt), vals, sizeof(float));
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_nonfinite_double_rejected(void)
{
   const char     *prefix  = "test_matrix_nonfinite_double";
   HYPRE_BigInt    rows[1] = {0};
   HYPRE_BigInt    cols[1] = {0};
   const double    vals[1] = {INFINITY};

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_success(void)
{
   const char    *prefix  = "test_matrix_success";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   const double   vals[1] = {5.0};
   HYPRE_IJMatrix mat     = NULL;
   void          *obj     = NULL;

   create_matrix_part(prefix, 0, 0, 0, 1, rows, cols, vals);

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_missing_file(void)
{
   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary("missing_matrix", MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST,
                               &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_invalid_part_count(void)
{
   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary("unused_matrix_prefix", MPI_COMM_SELF, 0,
                                        HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_invalid_path_prefix(void)
{
   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary("../unsafe_matrix_prefix", MPI_COMM_SELF, 1,
                                        HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_short_header(void)
{
   const char *prefix = "test_matrix_short";
   char        filename[256];

   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[5] = {0};
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 5, fp), 5u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_short_header_device_path(void)
{
   /* Covers the second-pass header read error path (skips host-side precompute). */
   const char *prefix = "test_matrix_short_dev";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[5] = {0};
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 5, fp), 5u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_invalid_row_range(void)
{
   const char *prefix = "test_matrix_bad_range";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 1;
   header[7]           = 8;
   header[8]           = 4; /* invalid range */
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_oversized_nnz_header(void)
{
   const char *prefix = "test_matrix_oversized_nnz";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = UINT64_C(200000001);
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_oversized_part_row_count(void)
{
   const char *prefix = "test_matrix_oversized_rows";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 1;
   /* nrows = row_upper - row_lower + 1 = 200000001 > IJMATRIX_MAX_PART_NROWS */
   header[7]           = 0;
   header[8]           = 200000000;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_truncated_bigint_rows_host(void)
{
   const char *prefix = "test_matrix_trunc_bigint_rows";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 2;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   {
      HYPRE_BigInt r = 0;
      ASSERT_EQ_SIZE(fwrite(&r, sizeof(r), 1, fp), 1u);
   }
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint32_truncated_rows_device_path(void)
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
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);

   uint32_t rows32[1] = {0}; /* intentionally short (need 2) */
   ASSERT_EQ_SIZE(fwrite(rows32, sizeof(uint32_t), 1, fp), 1u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint32_truncated_cols_device_path(void)
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
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);

   uint32_t rows32[2] = {0, 0};
   ASSERT_EQ_SIZE(fwrite(rows32, sizeof(uint32_t), 2, fp), 2u);
   uint32_t cols32[1] = {0}; /* intentionally short (need 2) */
   ASSERT_EQ_SIZE(fwrite(cols32, sizeof(uint32_t), 1, fp), 1u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_invalid_dtype(void)
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
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

/* Entries arrive with rows interleaved and one (row, col) pair repeated; the
 * assembled matrix must hold every value, with the later duplicate winning. */
static void
test_hypredrv_IJMatrixReadMultipartBinary_interleaved_rows(void)
{
   const char    *prefix      = "test_matrix_interleaved";
   HYPRE_BigInt   rows[7]     = {0, 1, 0, 2, 2, 1, 0};
   HYPRE_BigInt   cols[7]     = {0, 1, 1, 2, 0, 0, 0};
   const double   vals[7]     = {9.0, 4.0, -1.0, 6.0, -3.0, -2.0, 5.0};
   const double   want[9]     = {5.0, -1.0, 0.0, -2.0, 4.0, 0.0, -3.0, 0.0, 6.0};
   HYPRE_Int      ncols[3]    = {3, 3, 3};
   HYPRE_BigInt   get_rows[3] = {0, 1, 2};
   HYPRE_BigInt   get_cols[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
   HYPRE_Complex  got[9]      = {0};
   HYPRE_IJMatrix mat         = NULL;

   create_matrix_part(prefix, 0, 0, 2, 7, rows, cols, vals);

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(HYPRE_IJMatrixGetValues(mat, 3, ncols, get_rows, get_cols, got), 0);
   for (int k = 0; k < 9; k++)
   {
      ASSERT_EQ_DOUBLE(got[k], want[k], 0.0);
   }

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint32_indices(void)
{
   const char    *prefix    = "test_matrix_uint32";
   uint32_t       rows32[1] = {0};
   uint32_t       cols32[1] = {0};
   double         vals[1]   = {3.0};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows32, sizeof(uint32_t), cols32,
                            sizeof(uint32_t), vals, sizeof(double));

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_float_coefficients(void)
{
   const char    *prefix  = "test_matrix_float";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   float          vals[1] = {1.5f};
   HYPRE_IJMatrix mat     = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows, sizeof(HYPRE_BigInt), cols,
                            sizeof(HYPRE_BigInt), vals, sizeof(float));

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint32_indices_float_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint32_float";
   uint32_t       rows32[1] = {0};
   uint32_t       cols32[1] = {0};
   float          vals[1]   = {2.5f};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows32, sizeof(uint32_t), cols32,
                            sizeof(uint32_t), vals, sizeof(float));

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint64_indices_double_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint64_double";
   uint64_t       rows64[1] = {0};
   uint64_t       cols64[1] = {0};
   double         vals[1]   = {4.0};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows64, sizeof(uint64_t), cols64,
                            sizeof(uint64_t), vals, sizeof(double));

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

/* A 64-bit index beyond the HYPRE_BigInt range must be rejected, not wrapped
 * into a valid row (2^32 would narrow to row 0 with 32-bit HYPRE_BigInt). */
static void
test_hypredrv_IJMatrixReadMultipartBinary_uint64_index_overflow_rejected(void)
{
   const char  *prefix    = "test_matrix_uint64_overflow";
   uint64_t     rows64[2] = {0, UINT64_C(1) << 32};
   uint64_t     cols64[2] = {0, 1};
   const double vals[2]   = {1.0, 2.0};

   create_matrix_part_typed(prefix, 0, 0, 1, 2, rows64, sizeof(uint64_t), cols64,
                            sizeof(uint64_t), vals, sizeof(double));
   assert_matrix_read_rejects(prefix);
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_uint64_indices_float_coeffs(void)
{
   const char    *prefix    = "test_matrix_uint64_float";
   uint64_t       rows64[1] = {0};
   uint64_t       cols64[1] = {0};
   float          vals[1]   = {3.5f};
   HYPRE_IJMatrix mat       = NULL;
   void          *obj       = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows64, sizeof(uint64_t), cols64,
                            sizeof(uint64_t), vals, sizeof(float));

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_invalid_value_type(void)
{
   const char    *prefix  = "test_matrix_invalid_val";
   HYPRE_BigInt   rows[1] = {0};
   HYPRE_BigInt   cols[1] = {0};
   double         vals[1] = {2.0};
   HYPRE_IJMatrix mat     = NULL;

   create_matrix_part_typed(prefix, 0, 0, 0, 1, rows, sizeof(HYPRE_BigInt), cols,
                            sizeof(HYPRE_BigInt), vals, 3 /* invalid */);

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);

   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_zero_nnz_success(void)
{
   const char    *prefix = "test_matrix_zero_nnz";
   HYPRE_IJMatrix mat    = NULL;
   void          *obj    = NULL;

   create_matrix_part(prefix, 0, 0, 0, 0, NULL, NULL, NULL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_multipart_single_rank(void)
{
   /* g_nparts=2 on one MPI rank: parts 00000 and 00001 */
   const char    *prefix = "test_matrix_multipart";
   HYPRE_BigInt   r0[1]  = {0};
   HYPRE_BigInt   c0[1]  = {0};
   HYPRE_BigInt   r1[1]  = {1};
   HYPRE_BigInt   c1[1]  = {1};
   const double   v[1]   = {1.0};
   HYPRE_IJMatrix mat    = NULL;
   void          *obj    = NULL;

   create_matrix_part(prefix, 0, 0, 0, 1, r0, c0, v);
   create_matrix_part(prefix, 1, 1, 1, 1, r1, c1, v);

   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 2, HYPRE_MEMORY_HOST, &mat);

   ASSERT_NOT_NULL(mat);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJMatrixGetObject(mat, &obj);
   ASSERT_NOT_NULL(obj);

   HYPRE_IJMatrixDestroy(mat);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_truncated_bigint_cols_host(void)
{
   /* Pass-2 host precompute: rows read OK, column fread short */
   const char *prefix = "test_matrix_trunc_bigint_cols";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 2;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   {
      HYPRE_BigInt r0 = 0, r1 = 0;
      ASSERT_EQ_SIZE(fwrite(&r0, sizeof(r0), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&r1, sizeof(r1), 1, fp), 1u);
   }
   {
      HYPRE_BigInt c0 = 0;
      ASSERT_EQ_SIZE(fwrite(&c0, sizeof(c0), 1, fp), 1u); /* need 2 */
   }
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_truncated_float_coefs_host(void)
{
   const char *prefix = "test_matrix_trunc_float_coef";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(float);
   header[6]           = 2;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   {
      HYPRE_BigInt r0 = 0, r1 = 0, c0 = 0, c1 = 0;
      ASSERT_EQ_SIZE(fwrite(&r0, sizeof(r0), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&r1, sizeof(r1), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&c0, sizeof(c0), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&c1, sizeof(c1), 1, fp), 1u);
   }
   {
      float v0 = 1.0f;
      ASSERT_EQ_SIZE(fwrite(&v0, sizeof(v0), 1, fp), 1u); /* need 2 floats */
   }
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_truncated_double_coefs_host(void)
{
   const char *prefix = "test_matrix_trunc_double_coef";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = (uint64_t)sizeof(HYPRE_BigInt);
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 2;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   {
      HYPRE_BigInt r0 = 0, r1 = 0, c0 = 0, c1 = 0;
      ASSERT_EQ_SIZE(fwrite(&r0, sizeof(r0), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&r1, sizeof(r1), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&c0, sizeof(c0), 1, fp), 1u);
      ASSERT_EQ_SIZE(fwrite(&c1, sizeof(c1), 1, fp), 1u);
   }
   {
      double v0 = 1.0;
      ASSERT_EQ_SIZE(fwrite(&v0, sizeof(v0), 1, fp), 1u); /* need 2 doubles */
   }
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

static void
test_hypredrv_IJMatrixReadMultipartBinary_device_invalid_index_dtype_zero_nnz(void)
{
   /* Device path skips host precompute; invalid header[1] hits final-pass else with nnz==0 */
   const char *prefix = "test_matrix_dev_bad_idx";
   char        filename[256];
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);

   FILE *fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   uint64_t header[11] = {0};
   header[1]           = 7; /* not BigInt / u32 / u64 */
   header[2]           = (uint64_t)sizeof(double);
   header[6]           = 0;
   header[7]           = 0;
   header[8]           = 0;
   ASSERT_EQ_SIZE(fwrite(header, sizeof(uint64_t), 11, fp), 11u);
   fclose(fp);
   add_temp_file(filename);

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_DEVICE, &mat);
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   cleanup_temp_files();
}

typedef struct
{
   int changed_field;
   int change_on_values;
   int fail_want;
   int invalid_metadata;
   int index_loads;
   int value_loads;
} MatrixSourceFixture;

static int
load_matrix_source_fixture(void *ctx, uint32_t p, int want,
                           hypredrv_IJMatrixMemPart *part)
{
   MatrixSourceFixture *fixture = (MatrixSourceFixture *)ctx;
   part->nrows = part->nnz = 1;
   part->index_size        = sizeof(HYPRE_BigInt);
   part->value_size        = sizeof(double);
   part->label             = "memory matrix part";
   if (fixture->fail_want == want)
   {
      return 0;
   }
   if (fixture->invalid_metadata == 1)
   {
      part->nnz = UINT64_MAX;
   }
   else if (fixture->invalid_metadata == 2)
   {
      part->nrows = UINT64_MAX;
   }
   if (want && (!fixture->change_on_values || (want & HYPREDRV_PART_VALUES)))
   {
      switch (fixture->changed_field)
      {
         case 1:
            part->nnz++;
            break;
         case 2:
            part->nrows++;
            break;
         case 3:
            part->index_size = part->index_size == 4 ? 8 : 4;
            break;
         case 4:
            part->value_size = sizeof(float);
            break;
      }
   }
   if (want & HYPREDRV_PART_INDICES)
   {
      fixture->index_loads++;
      HYPRE_BigInt row = (HYPRE_BigInt)p;
      HYPRE_BigInt col = (HYPRE_BigInt)p;
      part->rows       = malloc(sizeof(row));
      part->cols       = malloc(sizeof(col));
      ASSERT_NOT_NULL(part->rows);
      ASSERT_NOT_NULL(part->cols);
      memcpy(part->rows, &row, sizeof(row));
      memcpy(part->cols, &col, sizeof(col));
   }
   if (want & HYPREDRV_PART_VALUES)
   {
      fixture->value_loads++;
      double value = 10.0 + p;
      part->vals   = malloc(sizeof(value));
      ASSERT_NOT_NULL(part->vals);
      memcpy(part->vals, &value, sizeof(value));
   }
   return 1;
}

static void
test_matrix_source_rejects_changed_parts(void)
{
   MatrixSourceFixture         fixture = {0};
   hypredrv_IJMatrixPartSource source  = {&fixture, 1, load_matrix_source_fixture};
   HYPRE_IJMatrix              mat     = NULL;
   fixture.fail_want                   = -1;

   /* Metadata must remain consistent in both the indices and cached values passes. */
   for (int values_pass = 0; values_pass < 2; values_pass++)
   {
      fixture.change_on_values = values_pass;
      for (int field = 1; field <= 4; field++)
      {
         fixture.changed_field = field;
         hypredrv_ErrorCodeResetAll();
         hypredrv_IJMatrixBuildFromSource(MPI_COMM_SELF, &source, HYPRE_MEMORY_HOST,
                                          &mat);
         ASSERT_NULL(mat);
         ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
      }
   }
   fixture.changed_field = 0;
   for (int invalid = 1; invalid <= 2; invalid++)
   {
      fixture.invalid_metadata = invalid;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJMatrixBuildFromSource(MPI_COMM_SELF, &source, HYPRE_MEMORY_HOST, &mat);
      ASSERT_NULL(mat);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }
   fixture.invalid_metadata = 0;
   for (int want = 0; want <= HYPREDRV_PART_VALUES; want++)
   {
      fixture.fail_want = want;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJMatrixBuildFromSource(MPI_COMM_SELF, &source, HYPRE_MEMORY_HOST, &mat);
      ASSERT_NULL(mat);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_UNEXPECTED_ENTRY);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
}

static void
test_matrix_source_values_and_communicator(void)
{
   MatrixSourceFixture         fixture = {0};
   hypredrv_IJMatrixPartSource source  = {&fixture, 2, load_matrix_source_fixture};
   HYPRE_IJMatrix              mat     = NULL;
   HYPRE_BigInt                rows[2] = {0, 1}, cols[2] = {0, 1};
   HYPRE_Int                   ncols[2]    = {1, 1};
   HYPRE_Complex               vals[2]     = {0};
   HYPRE_MemoryLocation        locations[] = {
      HYPRE_MEMORY_HOST,
#ifndef HYPRE_USING_GPU
      HYPRE_MEMORY_DEVICE,
#endif
   };
   fixture.fail_want = -1;
   ASSERT_TRUE(hypredrv_IJMatrixComm(NULL) == MPI_COMM_NULL);
   for (size_t i = 0; i < sizeof(locations) / sizeof(locations[0]); i++)
   {
      fixture.index_loads = fixture.value_loads = 0;
      hypredrv_ErrorCodeResetAll();
      hypredrv_IJMatrixBuildFromSource(MPI_COMM_SELF, &source, locations[i], &mat);
      ASSERT_NOT_NULL(mat);
      ASSERT_EQ(fixture.index_loads, 2);
      ASSERT_EQ(fixture.value_loads, 2);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE(hypredrv_IJMatrixComm(mat) == MPI_COMM_SELF);
      ASSERT_EQ(HYPRE_IJMatrixGetValues(mat, 2, ncols, rows, cols, vals), 0);
      ASSERT_EQ_DOUBLE((double)vals[0], 10.0, 1.0e-12);
      ASSERT_EQ_DOUBLE((double)vals[1], 11.0, 1.0e-12);
      HYPRE_IJMatrixDestroy(mat);
      mat = NULL;
   }
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_TRUE(hypredrv_IJMatrixComm(mat) == MPI_COMM_NULL);
   HYPRE_IJMatrixDestroy(mat);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   RUN_TEST(test_matrix_source_rejects_changed_parts);
   RUN_TEST(test_matrix_source_values_and_communicator);

   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_success);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_missing_file);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_invalid_part_count);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_invalid_path_prefix);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_short_header);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_short_header_device_path);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_invalid_row_range);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_oversized_nnz_header);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_oversized_part_row_count);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_truncated_bigint_rows_host);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_invalid_dtype);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_interleaved_rows);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint32_indices);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint32_truncated_rows_device_path);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint32_truncated_cols_device_path);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_float_coefficients);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint32_indices_float_coeffs);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint64_indices_double_coeffs);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint64_index_overflow_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_uint64_indices_float_coeffs);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_invalid_value_type);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_zero_nnz_success);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_multipart_single_rank);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_truncated_bigint_cols_host);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_truncated_float_coefs_host);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_truncated_double_coefs_host);
   RUN_TEST(
      test_hypredrv_IJMatrixReadMultipartBinary_device_invalid_index_dtype_zero_nnz);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_negative_row_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_negative_col_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_row_oob_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_col_oob_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_nonfinite_float_rejected);
   RUN_TEST(test_hypredrv_IJMatrixReadMultipartBinary_nonfinite_double_rejected);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
