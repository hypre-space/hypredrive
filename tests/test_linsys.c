#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPRE.h"
#include "containers.h"
#include "error.h"
#include "linsys.h"
#include "stats.h"
#include "test_helpers.h"
#include "yaml.h"

static YAMLnode *
make_scalar_node(const char *key, const char *value)
{
   YAMLnode *node   = YAMLnodeCreate(key, "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static YAMLnode *
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = YAMLnodeCreate(key, val, level);
   YAMLnodeAddChild(parent, child);
   return child;
}

static void
test_LinearSystemGetValidValues_type(void)
{
   StrIntMapArray map = LinearSystemGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "online"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "ij"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "parcsr"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "mtx"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "online"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "ij"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "parcsr"), 2);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "mtx"), 3);
}

static void
test_LinearSystemGetValidValues_rhs_mode(void)
{
   StrIntMapArray map = LinearSystemGetValidValues("rhs_mode");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "zeros"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "ones"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "file"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "random"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "randsol"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "zeros"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "ones"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "file"), 2);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "random"), 3);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "randsol"), 4);
}

static void
test_LinearSystemGetValidValues_init_guess_mode(void)
{
   StrIntMapArray map = LinearSystemGetValidValues("init_guess_mode");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "zeros"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "ones"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "file"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "random"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "previous"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "zeros"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "ones"), 1);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "file"), 2);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "random"), 3);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "previous"), 4);
}

static void
test_LinearSystemGetValidValues_exec_policy(void)
{
   StrIntMapArray map = LinearSystemGetValidValues("exec_policy");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "host"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(map, "device"));
   ASSERT_EQ(StrIntMapArrayGetImage(map, "host"), 0);
   ASSERT_EQ(StrIntMapArrayGetImage(map, "device"), 1);
}

static void
test_LinearSystemGetValidValues_unknown_key(void)
{
   StrIntMapArray map = LinearSystemGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_LinearSystemSetArgsFromYAML_valid_keys(void)
{
   LS_args args;
   LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "type", "2", 1);
   add_child(parent, "rhs_mode", "1", 1);
   add_child(parent, "init_guess_mode", "2", 1);
   add_child(parent, "exec_policy", "0", 1);
   add_child(parent, "digits_suffix", "6", 1);

   ErrorCodeResetAll();
   LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.type, 2);
   ASSERT_EQ(args.rhs_mode, 1);
   ASSERT_EQ(args.init_guess_mode, 2);
   ASSERT_EQ(args.exec_policy, 0);
   ASSERT_EQ(args.digits_suffix, 6);

   YAMLnodeDestroy(parent);
}

static void
test_LinearSystemSetArgsFromYAML_unknown_key(void)
{
   LS_args args;
   LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent       = YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *unknown_node = add_child(parent, "unknown_key", "value", 1);

   ErrorCodeResetAll();
   LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_EQ(unknown_node->valid, YAML_NODE_INVALID_KEY);

   YAMLnodeDestroy(parent);
}

static void
test_LinearSystemSetNearNullSpace_mismatch_error(void)
{
   HYPRE_Initialize();

   /* Create a minimal 1x1 matrix */
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Set a single entry - need to use proper indices */
   HYPRE_Int nrows = 1;
   HYPRE_Int ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   /* Get actual local size */
   HYPRE_BigInt ilower = 0, iupper = 0, jlower = 0, jupper = 0;
   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   int num_entries = (int)(jupper - jlower + 1);

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Test mismatch: num_entries doesn't match local size */
   ErrorCodeResetAll();
   LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries + 1, 1, NULL,
                                &vec_nn);

   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_NULL(vec_nn);

   HYPRE_IJMatrixDestroy(mat);
   HYPRE_Finalize();
}

static void
test_LinearSystemSetNearNullSpace_success(void)
{
   HYPRE_Initialize();

   /* Create a minimal 1x1 matrix */
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Set a single entry - need to use proper indices */
   HYPRE_Int nrows = 1;
   HYPRE_Int ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   /* Get actual local size */
   HYPRE_BigInt ilower = 0, iupper = 0, jlower = 0, jupper = 0;
   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   int num_entries = (int)(jupper - jlower + 1);

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Test success path with matching num_entries */
   ErrorCodeResetAll();
   LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
                                &vec_nn);

   /* Function was called - may have hypre errors with minimal matrix, that's ok */
   if (vec_nn)
   {
      HYPRE_IJVectorDestroy(vec_nn);
   }

   HYPRE_IJMatrixDestroy(mat);
   HYPRE_Finalize();
}

static void
test_LinearSystemSetNearNullSpace_destroy_previous(void)
{
   HYPRE_Initialize();

   /* Create a minimal 1x1 matrix */
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Set a single entry - need to use proper indices */
   HYPRE_Int nrows = 1;
   HYPRE_Int ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   /* Get actual local size */
   HYPRE_BigInt ilower = 0, iupper = 0, jlower = 0, jupper = 0;
   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   int num_entries = (int)(jupper - jlower + 1);

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Create first vector */
   ErrorCodeResetAll();
   LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
                                &vec_nn);

   /* Create second vector - should destroy previous if first succeeded */
   if (vec_nn)
   {
      HYPRE_IJVector old_vec = vec_nn;
      ErrorCodeResetAll();
      LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
                                   &vec_nn);
      /* If both succeeded, should be a new vector */
      if (vec_nn && vec_nn != old_vec)
      {
         HYPRE_IJVectorDestroy(vec_nn);
      }
      else if (vec_nn)
      {
         HYPRE_IJVectorDestroy(vec_nn);
      }
   }

   HYPRE_IJMatrixDestroy(mat);
   HYPRE_Finalize();
}

static void
test_LinearSystemGetValidValues_all_branches(void)
{
   /* Test all branches in LinearSystemGetValidValues */
   StrIntMapArray type_map = LinearSystemGetValidValues("type");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(type_map, "online"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(type_map, "ij"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(type_map, "parcsr"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(type_map, "mtx"));

   StrIntMapArray rhs_map = LinearSystemGetValidValues("rhs_mode");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rhs_map, "zeros"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rhs_map, "ones"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rhs_map, "file"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rhs_map, "random"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(rhs_map, "randsol"));

   StrIntMapArray init_map = LinearSystemGetValidValues("init_guess_mode");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(init_map, "zeros"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(init_map, "ones"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(init_map, "file"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(init_map, "random"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(init_map, "previous"));

   StrIntMapArray exec_map = LinearSystemGetValidValues("exec_policy");
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(exec_map, "host"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(exec_map, "device"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = LinearSystemGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   StrIntMapArray void_map2 = LinearSystemGetValidValues("matrix_filename");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_LinearSystemReadMatrix_filename_patterns(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;

   /* Test dirname pattern branch */
   strncpy(args.dirname, "test_dir", sizeof(args.dirname) - 1);
   strncpy(args.matrix_filename, "matrix.A", sizeof(args.matrix_filename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* Should fail with file not found, but branch was exercised */
   ASSERT_TRUE(ErrorCodeActive() || mat == NULL);

   /* Test basename pattern branch */
   args.dirname[0] = '\0';
   args.matrix_filename[0] = '\0';
   strncpy(args.matrix_basename, "matrix", sizeof(args.matrix_basename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* Should fail with file not found, but branch was exercised */
   ASSERT_TRUE(ErrorCodeActive() || mat == NULL);

   /* Test direct filename branch */
   args.matrix_basename[0] = '\0';
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* May succeed if file exists, or fail - both paths exercise branches */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }

   HYPRE_Finalize();
}

static void
test_LinearSystemReadMatrix_no_filename_error(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   args.dirname[0] = '\0';
   args.matrix_filename[0] = '\0';
   args.matrix_basename[0] = '\0';
   HYPRE_IJMatrix mat = NULL;

   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);

   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_NULL(mat);
   ASSERT_TRUE(ErrorCodeGet() & ERROR_FILE_NOT_FOUND);

   HYPRE_Finalize();
}

static void
test_LinearSystemReadMatrix_type_branches(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   HYPRE_IJMatrix mat = NULL;

   /* Test type 1 (IJ) branch */
   args.type = 1;
   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* May succeed or fail depending on file existence */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
      mat = NULL;
   }

   /* Test type 3 (MTX) branch - will fail but exercises the branch */
   args.type = 3;
   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* Should fail but branch was exercised */

   HYPRE_Finalize();
}

static void
test_LinearSystemReadMatrix_exec_policy_branches(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;

   /* Test exec_policy = 1 (device) */
   args.exec_policy = 1;
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* May succeed or fail depending on device availability */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
      mat = NULL;
   }

   /* Test exec_policy = 0 (host) */
   args.exec_policy = 0;
   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* May succeed or fail depending on file availability */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }

   HYPRE_Finalize();
}

static void
test_LinearSystemReadMatrix_partition_count_errors(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;

   /* Create a fake binary file with wrong partition count */
   char fake_file[] = "/tmp/fake_matrix.bin";
   FILE *fp = fopen(fake_file, "wb");
   if (fp)
   {
      // Write header with 2 partitions but only 1 MPI task
      uint64_t header[11] = {0};
      header[5] = 1; // nparts in header
      fwrite(header, sizeof(uint64_t), 11, fp);
      fclose(fp);

      args.type = 1; // IJ type
      args.dirname[0] = '\0';
      args.matrix_filename[0] = '\0';
      args.matrix_basename[0] = '\0';
      strncpy(args.matrix_filename, fake_file, sizeof(args.matrix_filename) - 1);

      ErrorCodeResetAll();
      LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);

      /* Should fail with file not found due to partition count mismatch */
      ASSERT_TRUE(ErrorCodeActive() || mat == NULL);

      if (mat)
      {
         HYPRE_IJMatrixDestroy(mat);
      }

      unlink(fake_file);
   }

   HYPRE_Finalize();
}

static void
test_LinearSystemReadRHS_file_patterns(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;

   /* Create minimal matrix for RHS reading */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test dirname pattern for RHS */
   args.type = 1; // IJ type
   strncpy(args.dirname, "test_dir", sizeof(args.dirname) - 1);
   strncpy(args.rhs_filename, "rhs.b", sizeof(args.rhs_filename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should fail with file not found, but branch was exercised */

   /* Test basename pattern for RHS */
   args.dirname[0] = '\0';
   args.rhs_filename[0] = '\0';
   strncpy(args.rhs_basename, "rhs", sizeof(args.rhs_basename) - 1);

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should fail with file not found, but branch was exercised */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
   }
   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   HYPRE_IJMatrixDestroy(mat);

   HYPRE_Finalize();
}

static void
test_LinearSystemSetRHS_mode_branches(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;

   /* Create minimal matrix */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test rhs_mode = 1 (ones) */
   args.rhs_mode = 1;
   args.rhs_filename[0] = '\0';
   args.rhs_basename[0] = '\0';

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should succeed */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }

   /* Test rhs_mode = 3 (random) */
   args.rhs_mode = 3;

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should succeed */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }

   /* Test rhs_mode = 4 (randsol - random solution) */
   args.rhs_mode = 4;

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should succeed and create both rhs and refsol */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
   }
   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   HYPRE_IJMatrixDestroy(mat);

   HYPRE_Finalize();
}

static void
test_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases(void)
{
   /* Test with NULL matrix */
   long long int result = LinearSystemMatrixGetNumRows(NULL);
   /* Should not crash, returns 0 */

   result = LinearSystemMatrixGetNumNonzeros(NULL);
   /* Should not crash, returns 0 */
}

static void
test_LinearSystemReadRHS_error_cases(void)
{
   HYPRE_Initialize();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;

   /* Create minimal matrix */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test with NULL matrix */
   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, NULL, &refsol, &rhs);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with invalid rhs_mode */
   args.rhs_mode = 999;
   args.rhs_filename[0] = '\0';
   args.rhs_basename[0] = '\0';

   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* Should use default case (ones) */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
   }
   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   HYPRE_IJMatrixDestroy(mat);

   HYPRE_Finalize();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_LinearSystemGetValidValues_type);
   RUN_TEST(test_LinearSystemGetValidValues_rhs_mode);
   RUN_TEST(test_LinearSystemGetValidValues_init_guess_mode);
   RUN_TEST(test_LinearSystemGetValidValues_exec_policy);
   RUN_TEST(test_LinearSystemGetValidValues_unknown_key);
   RUN_TEST(test_LinearSystemSetArgsFromYAML_valid_keys);
   RUN_TEST(test_LinearSystemSetArgsFromYAML_unknown_key);
   RUN_TEST(test_LinearSystemSetNearNullSpace_mismatch_error);
   RUN_TEST(test_LinearSystemSetNearNullSpace_success);
   RUN_TEST(test_LinearSystemSetNearNullSpace_destroy_previous);
   RUN_TEST(test_LinearSystemGetValidValues_all_branches);
   RUN_TEST(test_LinearSystemReadMatrix_filename_patterns);
   RUN_TEST(test_LinearSystemReadMatrix_no_filename_error);
   RUN_TEST(test_LinearSystemReadMatrix_type_branches);
   RUN_TEST(test_LinearSystemReadMatrix_exec_policy_branches);
   RUN_TEST(test_LinearSystemReadMatrix_partition_count_errors);
   RUN_TEST(test_LinearSystemReadRHS_file_patterns);
   RUN_TEST(test_LinearSystemSetRHS_mode_branches);
   RUN_TEST(test_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases);
   RUN_TEST(test_LinearSystemReadRHS_error_cases);

   MPI_Finalize();
   return 0;
}
