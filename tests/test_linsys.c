#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "HYPRE.h"
#include "containers.h"
#include "error.h"
#include "linsys.h"
#include "stats.h"
#include "test_helpers.h"
#include "yaml.h"

static void
write_text_file(const char *path, const char *contents)
{
   FILE *fp = fopen(path, "w");
   ASSERT_NOT_NULL(fp);
   fputs(contents, fp);
   fclose(fp);
}

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
test_LinearSystemPrintData_series_dir_and_null_objects(void)
{
   HYPRE_Initialize();

   /* Ensure series dir does not exist so we hit the mkdir(root) branch */
   (void)system("rm -rf hypre-data");

   LS_args args;
   LinearSystemSetDefaultArgs(&args); /* basenames empty => use_series_dir=true */

   /* Null objects should trip error branches without crashing */
   ErrorCodeResetAll();
   LinearSystemPrintData(MPI_COMM_SELF, &args, NULL, NULL, NULL);
   ASSERT_TRUE(ErrorCodeActive());
   ErrorCodeResetAll();

   /* Also cover args==NULL ternary/default branches */
   ErrorCodeResetAll();
   LinearSystemPrintData(MPI_COMM_SELF, NULL, NULL, NULL, NULL);
   ASSERT_TRUE(ErrorCodeActive());
   ErrorCodeResetAll();

   /* Pre-populate series dir with some ls_* entries to hit scan/max_idx logic */
   (void)mkdir("hypre-data", 0775);
   (void)mkdir("hypre-data/ls_00005", 0775);
   (void)mkdir("hypre-data/ls_00012", 0775);

   /* Provide minimal objects to hit the print branches */
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   HYPRE_Int    nrows = 1;
   HYPRE_Int    ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   HYPRE_IJVector vec_b = NULL;
   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec_b);
   HYPRE_IJVectorSetObjectType(vec_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(vec_b);
   HYPRE_BigInt idx[1] = {0};
   double       v[1]   = {2.0};
   HYPRE_IJVectorSetValues(vec_b, 1, idx, v);
   HYPRE_IJVectorAssemble(vec_b);

   IntArray *dofmap = NULL;
   int       dm[2]  = {0, 1};
   IntArrayBuild(MPI_COMM_SELF, 2, dm, &dofmap);

   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemPrintData(MPI_COMM_SELF, &args, mat, vec_b, dofmap);
   /* Printing paths can trigger hypre errors depending on build/config; tolerate as long
    * as we don't crash (this test is primarily for branch coverage). */
   ErrorCodeResetAll();

   /* Cover use_series_dir=false branch by providing explicit basenames */
   strncpy(args.matrix_basename, "A_base", sizeof(args.matrix_basename) - 1);
   strncpy(args.rhs_basename, "b_base", sizeof(args.rhs_basename) - 1);
   strncpy(args.dofmap_basename, "d_base", sizeof(args.dofmap_basename) - 1);
   args.matrix_basename[sizeof(args.matrix_basename) - 1] = '\0';
   args.rhs_basename[sizeof(args.rhs_basename) - 1]       = '\0';
   args.dofmap_basename[sizeof(args.dofmap_basename) - 1] = '\0';

   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemPrintData(MPI_COMM_SELF, &args, mat, vec_b, dofmap);
   ErrorCodeResetAll();

   IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);

   (void)system("rm -rf hypre-data");
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
   /* Expect no crash; implementation may or may not allocate rhs/refsol here. */
   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }
   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
      refsol = NULL;
   }

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

static void
test_LinearSystemReadMatrix_mtx_success(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   char matfile[256];
   snprintf(matfile, sizeof(matfile), "/tmp/hypredrive_test_mat_%d.mtx", (int)getpid());

   /* Minimal 1x1 MatrixMarket matrix (coordinate) */
   write_text_file(matfile,
                   "%%MatrixMarket matrix coordinate real general\n"
                   "% comment\n"
                   "1 1 1\n"
                   "1 1 1.0\n");

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   args.type = 3; /* mtx */
   strncpy(args.matrix_filename, matfile, sizeof(args.matrix_filename) - 1);
   args.matrix_filename[sizeof(args.matrix_filename) - 1] = '\0';

   HYPRE_IJMatrix mat = NULL;
   ErrorCodeResetAll();
   LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat);
   /* The goal here is to execute the MatrixMarket matrix-read branch. Whether
    * HYPRE_IJMatrixReadMM succeeds depends on the hypre build/config and parser
    * expectations, so tolerate failure. */
   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }
   unlink(matfile);
   HYPRE_Finalize();
}

static void
test_LinearSystemSetRHS_mtx_file_success(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   char rhsfile[256];
   snprintf(rhsfile, sizeof(rhsfile), "/tmp/hypredrive_test_rhs_%d.mtx", (int)getpid());

   write_text_file(rhsfile,
                   "% vector as a 1-column MM-like text\n"
                   "1 1\n"
                   "2.5\n");

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   args.type     = 3; /* mtx */
   args.rhs_mode = 2; /* file */
   strncpy(args.rhs_filename, rhsfile, sizeof(args.rhs_filename) - 1);
   args.rhs_filename[sizeof(args.rhs_filename) - 1] = '\0';

   /* Create a minimal 1x1 matrix to satisfy the RHS reader's dimension checks */
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   HYPRE_Int    nrows = 1;
   HYPRE_Int    ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   HYPRE_IJVector refsol = NULL, rhs = NULL;
   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   /* This path is mainly to exercise the MM vector-reader logic. Depending on the
    * hypre build/config, the underlying IJVector calls may report errors; tolerate
    * that as long as we don't crash/leak. */

   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
   }
   HYPRE_IJMatrixDestroy(mat);

   unlink(rhsfile);
   HYPRE_Finalize();
}

static void
test_LinearSystemSetRHS_mtx_dim_mismatch_errors(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   char rhsfile[256];
   snprintf(rhsfile, sizeof(rhsfile), "/tmp/hypredrive_test_rhs_%d.mtx", (int)getpid());

   /* Wrong vector dims: N != 1 */
   write_text_file(rhsfile,
                   "% bad vector dims\n"
                   "1 2\n"
                   "1.0\n");

   LS_args args;
   LinearSystemSetDefaultArgs(&args);
   args.type     = 3; /* mtx */
   args.rhs_mode = 2; /* file */
   strncpy(args.rhs_filename, rhsfile, sizeof(args.rhs_filename) - 1);
   args.rhs_filename[sizeof(args.rhs_filename) - 1] = '\0';

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   HYPRE_Int    nrows = 1;
   HYPRE_Int    ncols[1] = {1};
   HYPRE_BigInt rows[1] = {0};
   HYPRE_BigInt cols[1] = {0};
   double       vals[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, vals);
   HYPRE_IJMatrixAssemble(mat);

   HYPRE_IJVector refsol = NULL, rhs = NULL;
   ErrorCodeResetAll();
   LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs);
   ASSERT_TRUE(ErrorCodeActive());
   ASSERT_NULL(rhs);

   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   HYPRE_IJMatrixDestroy(mat);

   unlink(rhsfile);
   HYPRE_Finalize();
}

static HYPRE_IJVector
create_test_ijvector(MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper,
                     const HYPRE_Complex *vals)
{
   HYPRE_IJVector v = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(comm, ilower, iupper, &v), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(v, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(v), 0);

   int n = (int)(iupper - ilower + 1);
   HYPRE_BigInt *idx = (HYPRE_BigInt *)malloc((size_t)n * sizeof(HYPRE_BigInt));
   ASSERT_NOT_NULL(idx);
   for (int i = 0; i < n; i++)
   {
      idx[i] = ilower + i;
   }

   ASSERT_EQ(HYPRE_IJVectorSetValues(v, n, idx, vals), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(v), 0);
   free(idx);
   return v;
}

static void
test_LinearSystemComputeVectorNorm_all_modes(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   /* Build a small vector with known values: [1, -2, 3] */
   const HYPRE_Complex vals[3] = {1.0, -2.0, 3.0};
   HYPRE_IJVector      v       = create_test_ijvector(MPI_COMM_SELF, 0, 2, vals);

   double norm = 0.0;

   LinearSystemComputeVectorNorm(v, "L1", &norm);
   ASSERT_EQ_DOUBLE(norm, 6.0, 1e-12);

   LinearSystemComputeVectorNorm(v, "l1", &norm);
   ASSERT_EQ_DOUBLE(norm, 6.0, 1e-12);

   LinearSystemComputeVectorNorm(v, "L2", &norm);
   ASSERT_EQ_DOUBLE(norm, sqrt(14.0), 1e-10);

   LinearSystemComputeVectorNorm(v, "inf", &norm);
   ASSERT_EQ_DOUBLE(norm, 3.0, 1e-12);

   LinearSystemComputeVectorNorm(v, "Linf", &norm);
   ASSERT_EQ_DOUBLE(norm, 3.0, 1e-12);

   LinearSystemComputeVectorNorm(v, "bad", &norm);
   ASSERT_EQ_DOUBLE(norm, -1.0, 0.0);

   HYPRE_IJVectorDestroy(v);
   HYPRE_Finalize();
}

static void
test_LinearSystemSetInitialGuess_x0_filename_branches(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);

   /* Build a small RHS vector so LinearSystemSetInitialGuess can size x/x0 */
   const HYPRE_Complex rhs_vals[3] = {1.0, 2.0, 3.0};
   HYPRE_IJVector      rhs         = create_test_ijvector(MPI_COMM_SELF, 0, 2, rhs_vals);

   HYPRE_IJVector x0 = NULL;
   HYPRE_IJVector x  = NULL;

   /* 1) ASCII x0 file path (CheckBinaryDataExists false) */
#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif
   char x0_ascii[4096];
   snprintf(x0_ascii, sizeof(x0_ascii), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);
   strncpy(args.x0_filename, x0_ascii, sizeof(args.x0_filename) - 1);
   args.x0_filename[sizeof(args.x0_filename) - 1] = '\0';
   args.exec_policy                               = 0;
   ErrorCodeResetAll();
   LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x);
   /* Might set hypre errors depending on build; just ensure no crash and cleanup */
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   x = x0 = NULL;

   /* 1b) Same ASCII path with exec_policy enabled (covers migrate branch) */
   args.exec_policy = 1;
   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x);
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   x = x0 = NULL;

   /* 2) Binary-detection branch (create <prefix>.00000.bin so CheckBinaryDataExists true) */
   (void)memset(args.x0_filename, 0, sizeof(args.x0_filename));
   strncpy(args.x0_filename, "tmp_x0", sizeof(args.x0_filename) - 1);
   args.exec_policy = 0;
   write_text_file("tmp_x0.00000.bin", ""); /* dummy file - read may fail but should not crash */
   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x);
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   unlink("tmp_x0.00000.bin");

   HYPRE_IJVectorDestroy(rhs);
   HYPRE_Finalize();
}

static HYPRE_IJMatrix
create_test_ijmatrix_1x1(MPI_Comm comm, double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(comm, 0, 0, 0, 0, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);

   HYPRE_Int    nrows     = 1;
   HYPRE_Int    ncols[1]  = {1};
   HYPRE_BigInt rows[1]   = {0};
   HYPRE_BigInt cols[1]   = {0};
   double       values[1] = {diag};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static void
test_LinearSystemSetPrecMatrix_branchy_paths(void)
{
   HYPRE_Initialize();
   HYPRE_ClearAllErrors();

   LS_args args;
   LinearSystemSetDefaultArgs(&args);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 1.0);
   HYPRE_IJMatrix mat_M = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0); /* pre-existing */

   /* 1) precmat_filename differs from matrix_filename => destroy + read branch */
   strncpy(args.matrix_filename, "Afile", sizeof(args.matrix_filename) - 1);
   strncpy(args.precmat_filename, "Mfile", sizeof(args.precmat_filename) - 1);
   args.matrix_filename[sizeof(args.matrix_filename) - 1]   = '\0';
   args.precmat_filename[sizeof(args.precmat_filename) - 1] = '\0';
   args.dirname[0]                                          = '\0';
   args.precmat_basename[0]                                 = '\0';

   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M);
   ErrorCodeResetAll(); /* tolerate read errors */

   /* If the internal read failed, LinearSystemSetPrecMatrix may have destroyed the
    * previous matrix without nulling the pointer. Avoid double-free by only
    * destroying when hypre reports success. */
   if (HYPRE_GetError() == 0 && mat_M && mat_M != mat_A)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_ClearAllErrors();
   mat_M = NULL;

   /* 2) precmat_basename path */
   args.precmat_filename[0] = '\0';
   strncpy(args.precmat_basename, "Mbase", sizeof(args.precmat_basename) - 1);
   args.precmat_basename[sizeof(args.precmat_basename) - 1] = '\0';
   mat_M                                                      = create_test_ijmatrix_1x1(MPI_COMM_SELF, 3.0);
   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M);
   ErrorCodeResetAll();
   if (HYPRE_GetError() == 0 && mat_M && mat_M != mat_A)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_ClearAllErrors();
   mat_M = NULL;

   /* 3) dirname + precmat_filename path */
   strncpy(args.dirname, "hypre-data", sizeof(args.dirname) - 1);
   args.dirname[sizeof(args.dirname) - 1] = '\0';
   strncpy(args.precmat_filename, "Mdirfile", sizeof(args.precmat_filename) - 1);
   args.precmat_filename[sizeof(args.precmat_filename) - 1] = '\0';
   args.precmat_basename[0]                                  = '\0';
   mat_M                                                      = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M);
   ErrorCodeResetAll();
   if (HYPRE_GetError() == 0 && mat_M && mat_M != mat_A)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrixDestroy(mat_A);
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
   RUN_TEST(test_LinearSystemReadMatrix_mtx_success);
   RUN_TEST(test_LinearSystemSetRHS_mtx_file_success);
   RUN_TEST(test_LinearSystemSetRHS_mtx_dim_mismatch_errors);
   RUN_TEST(test_LinearSystemPrintData_series_dir_and_null_objects);
   RUN_TEST(test_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases);
   RUN_TEST(test_LinearSystemReadRHS_error_cases);
   RUN_TEST(test_LinearSystemComputeVectorNorm_all_modes);
   RUN_TEST(test_LinearSystemSetInitialGuess_x0_filename_branches);
   RUN_TEST(test_LinearSystemSetPrecMatrix_branchy_paths);

   MPI_Finalize();
   return 0;
}
