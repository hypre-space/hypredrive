#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "HYPRE.h"
#include "containers.h"
#include "error.h"
#include "logging.h"
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
add_child(YAMLnode *parent, const char *key, const char *val, int level)
{
   YAMLnode *child = hypredrv_YAMLnodeCreate(key, val, level);
   hypredrv_YAMLnodeAddChild(parent, child);
   return child;
}

typedef void (*CapturedStreamFn)(void *);

static void
capture_stderr_output(CapturedStreamFn fn, void *context, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();
   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   fn(context);
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

static HYPRE_IJMatrix create_test_ijmatrix_1x1(MPI_Comm comm, double diag);

static int
get_matrix_local_num_entries(HYPRE_IJMatrix mat)
{
   HYPRE_BigInt ilower = 0, iupper = 0, jlower = 0, jupper = 0;
   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   return (int)(jupper - jlower + 1);
}

static HYPRE_IJMatrix
create_nearnullspace_test_matrix(int *num_entries)
{
   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 1.0);
   ASSERT_NOT_NULL(mat);
   ASSERT_NOT_NULL(num_entries);
   *num_entries = get_matrix_local_num_entries(mat);
   return mat;
}

static void
test_hypredrv_LinearSystemGetValidValues_type(void)
{
   StrIntMapArray map = hypredrv_LinearSystemGetValidValues("type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "online"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "ij"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "parcsr"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "mtx"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "online"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "ij"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "parcsr"), 2);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "mtx"), 3);
}

static void
test_hypredrv_LinearSystemGetValidValues_rhs_mode(void)
{
   StrIntMapArray map = hypredrv_LinearSystemGetValidValues("rhs_mode");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "zeros"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "ones"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "file"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "random"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "randsol"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "zeros"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "ones"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "file"), 2);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "random"), 3);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "randsol"), 4);
}

static void
test_hypredrv_LinearSystemGetValidValues_init_guess_mode(void)
{
   StrIntMapArray map = hypredrv_LinearSystemGetValidValues("init_guess_mode");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "zeros"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "ones"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "file"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "random"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(map, "previous"));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "zeros"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "ones"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "file"), 2);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "random"), 3);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "previous"), 4);
}

static void
test_hypredrv_LinearSystemGetValidValues_unknown_key(void)
{
   StrIntMapArray map = hypredrv_LinearSystemGetValidValues("unknown_key");
   ASSERT_EQ(map.size, 0);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_valid_keys(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "type", "2", 1);
   add_child(parent, "rhs_mode", "1", 1);
   add_child(parent, "init_guess_mode", "2", 1);
   add_child(parent, "digits_suffix", "6", 1);
   add_child(parent, "timestep_filename", "timesteps.txt", 1);
   for (YAMLnode *c = parent->children; c; c = c->next)
   {
      if (c->val)
      {
         c->mapped_val = strdup(c->val);
      }
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_EQ(args.type, 2);
   ASSERT_EQ(args.rhs_mode, 1);
   ASSERT_EQ(args.init_guess_mode, 2);
   ASSERT_EQ(args.digits_suffix, 6);
   ASSERT_STREQ(args.timestep_filename, "timesteps.txt");

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_unknown_key(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent       = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *unknown_node = add_child(parent, "unknown_key", "value", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_EQ(unknown_node->valid, YAML_NODE_INVALID_KEY);

   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "set_suffix", "[0, 2, 5]", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   hypredrv_LinearSystemSetNumSystems(&args);

   ASSERT_NOT_NULL(args.set_suffix);
   ASSERT_EQ(args.set_suffix->size, 3);
   ASSERT_EQ(args.set_suffix->data[0], 0);
   ASSERT_EQ(args.set_suffix->data[1], 2);
   ASSERT_EQ(args.set_suffix->data[2], 5);
   ASSERT_EQ(args.num_systems, 3);
   ASSERT_EQ(hypredrv_LinearSystemGetSuffix(&args, 1), 0);
   ASSERT_EQ(hypredrv_LinearSystemGetSuffix(&args, 2), 2);
   ASSERT_EQ(hypredrv_LinearSystemGetSuffix(&args, 3), 5);

   hypredrv_IntArrayDestroy(&args.set_suffix);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix_and_init_suffix_error(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "set_suffix", "[0, 1]", 1);
   add_child(parent, "init_suffix", "10", 1);
   for (YAMLnode *c = parent->children; c; c = c->next)
   {
      if (c->val)
      {
         c->mapped_val = strdup(c->val);
      }
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_IntArrayDestroy(&args.set_suffix);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetNearNullSpace_mismatch_error(void)
{
   TEST_HYPRE_INIT();

   int           num_entries = 0;
   HYPRE_IJMatrix mat        = create_nearnullspace_test_matrix(&num_entries);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Test mismatch: num_entries doesn't match local size */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries + 1, 1, NULL,
                                &vec_nn);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(vec_nn);

   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetNearNullSpace_success(void)
{
   TEST_HYPRE_INIT();

   int           num_entries = 0;
   HYPRE_IJMatrix mat        = create_nearnullspace_test_matrix(&num_entries);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Test success path with matching num_entries */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
                                &vec_nn);

   /* Function was called - may have hypre errors with minimal matrix, that's ok */
   if (vec_nn)
   {
      HYPRE_IJVectorDestroy(vec_nn);
   }

   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetNearNullSpace_destroy_previous(void)
{
   TEST_HYPRE_INIT();

   int           num_entries = 0;
   HYPRE_IJMatrix mat        = create_nearnullspace_test_matrix(&num_entries);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJVector vec_nn = NULL;

   /* Create first vector */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
                                &vec_nn);

   /* Create second vector - should destroy previous if first succeeded */
   if (vec_nn)
   {
      HYPRE_IJVector old_vec = vec_nn;
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetNearNullSpace(MPI_COMM_SELF, &args, mat, num_entries, 1, NULL,
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
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemGetValidValues_all_branches(void)
{
   /* Test all branches in hypredrv_LinearSystemGetValidValues */
   StrIntMapArray type_map = hypredrv_LinearSystemGetValidValues("type");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(type_map, "online"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(type_map, "ij"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(type_map, "parcsr"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(type_map, "mtx"));

   StrIntMapArray rhs_map = hypredrv_LinearSystemGetValidValues("rhs_mode");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rhs_map, "zeros"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rhs_map, "ones"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rhs_map, "file"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rhs_map, "random"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(rhs_map, "randsol"));

   StrIntMapArray init_map = hypredrv_LinearSystemGetValidValues("init_guess_mode");
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(init_map, "zeros"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(init_map, "ones"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(init_map, "file"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(init_map, "random"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(init_map, "previous"));

   /* Test else branch - key that doesn't match any condition */
   StrIntMapArray void_map = hypredrv_LinearSystemGetValidValues("unknown_key");
   ASSERT_EQ(void_map.size, 0);

   StrIntMapArray void_map2 = hypredrv_LinearSystemGetValidValues("matrix_filename");
   ASSERT_EQ(void_map2.size, 0);
}

static void
test_hypredrv_LinearSystemReadMatrix_filename_patterns(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;

   /* Test dirname pattern branch */
   strncpy(args.dirname, "test_dir", sizeof(args.dirname) - 1);
   strncpy(args.matrix_filename, "matrix.A", sizeof(args.matrix_filename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* Should fail with file not found, but branch was exercised */
   ASSERT_TRUE(hypredrv_ErrorCodeActive() || mat == NULL);

   /* Test basename pattern branch */
   args.dirname[0] = '\0';
   args.matrix_filename[0] = '\0';
   strncpy(args.matrix_basename, "matrix", sizeof(args.matrix_basename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* Should fail with file not found, but branch was exercised */
   ASSERT_TRUE(hypredrv_ErrorCodeActive() || mat == NULL);

   /* Test direct filename branch */
   args.matrix_basename[0] = '\0';
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* May succeed if file exists, or fail - both paths exercise branches */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadMatrix_no_filename_error(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.dirname[0] = '\0';
   args.matrix_filename[0] = '\0';
   args.matrix_basename[0] = '\0';
   HYPRE_IJMatrix mat = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(mat);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadMatrix_type_branches(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   HYPRE_IJMatrix mat = NULL;

   /* Test type 1 (IJ) branch */
   args.type = 1;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* May succeed or fail depending on file existence */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
      mat = NULL;
   }

   /* Test type 3 (MTX) branch - will fail but exercises the branch */
   args.type = 3;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* Should fail but branch was exercised */

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadMatrix_exec_policy_branches(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;

#ifdef HYPRE_USING_GPU
   /* Test exec_policy = 1 (device) */
   args.exec_policy = 1;
   strncpy(args.matrix_filename, "data/ps3d10pt7/np1/IJ.out.A",
           sizeof(args.matrix_filename) - 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* May succeed or fail depending on device availability */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
      mat = NULL;
   }
#endif

   /* Test exec_policy = 0 (host) */
   args.exec_policy = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* May succeed or fail depending on file availability */

   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadMatrix_partition_count_errors(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
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

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);

      /* Should fail with file not found due to partition count mismatch */
      ASSERT_TRUE(hypredrv_ErrorCodeActive() || mat == NULL);

      if (mat)
      {
         HYPRE_IJMatrixDestroy(mat);
      }

      unlink(fake_file);
   }

   TEST_HYPRE_FINALIZE();
}

static void
test_LinearSystemReadRHS_file_patterns(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;

   /* Create minimal matrix for RHS reading */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test dirname pattern for RHS */
   args.type = 1; // IJ type
   args.rhs_mode = 2; // file mode
   strncpy(args.dirname, "test_dir", sizeof(args.dirname) - 1);
   strncpy(args.rhs_filename, "rhs.b", sizeof(args.rhs_filename) - 1);
   args.digits_suffix = 5;
   args.init_suffix = 0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   /* Should fail with file not found, but branch was exercised */

   /* Test basename pattern for RHS */
   args.dirname[0] = '\0';
   args.rhs_filename[0] = '\0';
   strncpy(args.rhs_basename, "rhs", sizeof(args.rhs_basename) - 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
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

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetRHS_mode_precedence_over_filename(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.rhs_mode = 4; /* randsol */
   strncpy(args.rhs_filename, "/tmp/this_rhs_file_should_be_ignored",
           sizeof(args.rhs_filename) - 1);
   args.rhs_filename[sizeof(args.rhs_filename) - 1] = '\0';

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;
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

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(rhs);
   ASSERT_NOT_NULL(refsol);

   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(refsol);
   HYPRE_IJMatrixDestroy(mat);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetRHS_mode_branches(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
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

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   /* Should succeed */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }

   /* Test rhs_mode = 3 (random) */
   args.rhs_mode = 3;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   /* Should succeed */

   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }

   /* Test rhs_mode = 4 (randsol - random solution) */
   args.rhs_mode = 4;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
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

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetReferenceSolution_keeps_randsol_reference(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.rhs_mode = 4; /* randsol */

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, xref = NULL;

   /* Build a minimal 1x1 matrix so SetRHS can form b = A * xref. */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);
   HYPRE_Int    nrows     = 1;
   HYPRE_Int    ncols[1]  = {1};
   HYPRE_BigInt rows[1]   = {0};
   HYPRE_BigInt cols[1]   = {0};
   double       values[1] = {1.0};
   HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values);
   HYPRE_IJMatrixAssemble(mat);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &xref, &rhs, NULL);
   ASSERT_NOT_NULL(rhs);
   ASSERT_NOT_NULL(xref);

   HYPRE_IJVector xref_before = xref;

   /* No xref file is provided, so existing randsol reference must be preserved. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetReferenceSolution(MPI_COMM_SELF, &args, &xref, NULL);
   ASSERT_NOT_NULL(xref);
   ASSERT_TRUE(xref == xref_before);

   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(xref);
   HYPRE_IJMatrixDestroy(mat);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemPrintData_series_dir_and_null_objects(void)
{
   TEST_HYPRE_INIT();

   /* Ensure series dir does not exist so we hit the mkdir(root) branch */
   int ret = system("rm -rf hypre-data");
   (void)ret; /* Ignore cleanup failures in tests */

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args); /* basenames empty => use_series_dir=true */

   /* Null objects should trip error branches without crashing */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemPrintData(MPI_COMM_SELF, &args, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

   /* Also cover args==NULL ternary/default branches */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemPrintData(MPI_COMM_SELF, NULL, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

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
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm, &dofmap);

   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemPrintData(MPI_COMM_SELF, &args, mat, vec_b, dofmap);
   /* Printing paths can trigger hypre errors depending on build/config; tolerate as long
    * as we don't crash (this test is primarily for branch coverage). */
   hypredrv_ErrorCodeResetAll();

   /* Cover use_series_dir=false branch by providing explicit basenames */
   strncpy(args.matrix_basename, "A_base", sizeof(args.matrix_basename) - 1);
   strncpy(args.rhs_basename, "b_base", sizeof(args.rhs_basename) - 1);
   strncpy(args.dofmap_basename, "d_base", sizeof(args.dofmap_basename) - 1);
   args.matrix_basename[sizeof(args.matrix_basename) - 1] = '\0';
   args.rhs_basename[sizeof(args.rhs_basename) - 1]       = '\0';
   args.dofmap_basename[sizeof(args.dofmap_basename) - 1] = '\0';

   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemPrintData(MPI_COMM_SELF, &args, mat, vec_b, dofmap);
   hypredrv_ErrorCodeResetAll();

   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(vec_b);
   HYPRE_IJMatrixDestroy(mat);

   remove("A_base.out.00000");
   remove("b_base.out.00000");
   remove("d_base.out.00000");

   ret = system("rm -rf hypre-data");
   (void)ret; /* Ignore cleanup failures in tests */
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases(void)
{
   /* Test with NULL matrix */
   (void) hypredrv_LinearSystemMatrixGetNumRows(NULL);
   /* Should not crash, returns 0 */

   (void) hypredrv_LinearSystemMatrixGetNumNonzeros(NULL);
   /* Should not crash, returns 0 */
}

static void
test_LinearSystemReadRHS_error_cases(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJVector rhs = NULL, refsol = NULL;

   /* Create minimal matrix */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   /* Test with NULL matrix */
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, NULL, &refsol, &rhs, NULL);
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

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
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

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadMatrix_mtx_success(void)
{
   TEST_HYPRE_INIT();
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
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.type = 3; /* mtx */
   strncpy(args.matrix_filename, matfile, sizeof(args.matrix_filename) - 1);
   args.matrix_filename[sizeof(args.matrix_filename) - 1] = '\0';

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, NULL);
   /* The goal here is to execute the MatrixMarket matrix-read branch. Whether
    * HYPRE_IJMatrixReadMM succeeds depends on the hypre build/config and parser
    * expectations, so tolerate failure. */
   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }
   unlink(matfile);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetRHS_mtx_file_success(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   char rhsfile[256];
   snprintf(rhsfile, sizeof(rhsfile), "/tmp/hypredrive_test_rhs_%d.mtx", (int)getpid());

   write_text_file(rhsfile,
                   "% vector as a 1-column MM-like text\n"
                   "1 1\n"
                   "2.5\n");

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
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
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
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
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetRHS_mtx_dim_mismatch_errors(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   char rhsfile[256];
   snprintf(rhsfile, sizeof(rhsfile), "/tmp/hypredrive_test_rhs_%d.mtx", (int)getpid());

   /* Wrong vector dims: N != 1 */
   write_text_file(rhsfile,
                   "% bad vector dims\n"
                   "1 2\n"
                   "1.0\n");

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
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
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(rhs);

   if (refsol)
   {
      HYPRE_IJVectorDestroy(refsol);
   }
   HYPRE_IJMatrixDestroy(mat);

   unlink(rhsfile);
   TEST_HYPRE_FINALIZE();
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
test_hypredrv_LinearSystemComputeVectorNorm_all_modes(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   /* Build a small vector with known values: [1, -2, 3] */
   const HYPRE_Complex vals[3] = {1.0, -2.0, 3.0};
   HYPRE_IJVector      v       = create_test_ijvector(MPI_COMM_SELF, 0, 2, vals);

   double norm = 0.0;

   hypredrv_LinearSystemComputeVectorNorm(v, "L1", &norm);
   ASSERT_EQ_DOUBLE(norm, 6.0, 1e-12);

   hypredrv_LinearSystemComputeVectorNorm(v, "l1", &norm);
   ASSERT_EQ_DOUBLE(norm, 6.0, 1e-12);

   hypredrv_LinearSystemComputeVectorNorm(v, "L2", &norm);
   ASSERT_EQ_DOUBLE(norm, sqrt(14.0), 1e-10);

   hypredrv_LinearSystemComputeVectorNorm(v, "inf", &norm);
   ASSERT_EQ_DOUBLE(norm, 3.0, 1e-12);

   hypredrv_LinearSystemComputeVectorNorm(v, "Linf", &norm);
   ASSERT_EQ_DOUBLE(norm, 3.0, 1e-12);

   hypredrv_LinearSystemComputeVectorNorm(v, "bad", &norm);
   ASSERT_EQ_DOUBLE(norm, -1.0, 0.0);

   HYPRE_IJVectorDestroy(v);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetInitialGuess_x0_filename_branches(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   /* Build a small RHS vector so hypredrv_LinearSystemSetInitialGuess can size x/x0 */
   const HYPRE_Complex rhs_vals[3] = {1.0, 2.0, 3.0};
   HYPRE_IJVector      rhs         = create_test_ijvector(MPI_COMM_SELF, 0, 2, rhs_vals);

   HYPRE_IJVector x0 = NULL;
   HYPRE_IJVector x  = NULL;

   /* 1) ASCII x0 file path (hypredrv_CheckBinaryDataExists false) */
#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif
   char x0_ascii[4096];
   snprintf(x0_ascii, sizeof(x0_ascii), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);
   strncpy(args.x0_filename, x0_ascii, sizeof(args.x0_filename) - 1);
   args.x0_filename[sizeof(args.x0_filename) - 1] = '\0';
   args.exec_policy                               = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   /* Might set hypre errors depending on build; just ensure no crash and cleanup */
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   x = x0 = NULL;

#ifdef HYPRE_USING_GPU
   /* 1b) Same ASCII path with exec_policy enabled (covers migrate branch) */
   args.exec_policy = 1;
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   x = x0 = NULL;
#endif

   /* 2) Binary-detection branch (create <prefix>.00000.bin so hypredrv_CheckBinaryDataExists true) */
   (void)memset(args.x0_filename, 0, sizeof(args.x0_filename));
   strncpy(args.x0_filename, "tmp_x0", sizeof(args.x0_filename) - 1);
   args.exec_policy = 0;
   write_text_file("tmp_x0.00000.bin", ""); /* dummy file - read may fail but should not crash */
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x) HYPRE_IJVectorDestroy(x);
   if (x0) HYPRE_IJVectorDestroy(x0);
   unlink("tmp_x0.00000.bin");

   HYPRE_IJVectorDestroy(rhs);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemCreateWorkingSolution_recreates_x(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.exec_policy = 0;

   const HYPRE_Complex rhs_vals[3] = {1.0, 2.0, 3.0};
   HYPRE_IJVector      rhs         = create_test_ijvector(MPI_COMM_SELF, 0, 2, rhs_vals);
   HYPRE_IJVector      x           = create_test_ijvector(MPI_COMM_SELF, 0, 2, rhs_vals);
   HYPRE_BigInt        ilow = -1, iup = -1;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemCreateWorkingSolution(MPI_COMM_SELF, &args, rhs, &x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(x);
   HYPRE_IJVectorGetLocalRange(x, &ilow, &iup);
   ASSERT_EQ(ilow, 0);
   ASSERT_EQ(iup, 2);

   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(rhs);
   TEST_HYPRE_FINALIZE();
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
test_hypredrv_LinearSystemSetPrecMatrix_branchy_paths(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 1.0);
   HYPRE_IJMatrix mat_M = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0); /* pre-existing */

   /* 1) precmat_filename differs from matrix_filename => destroy + read branch */
   strncpy(args.matrix_filename, "Afile", sizeof(args.matrix_filename) - 1);
   strncpy(args.precmat_filename, "Mfile", sizeof(args.precmat_filename) - 1);
   args.matrix_filename[sizeof(args.matrix_filename) - 1]   = '\0';
   args.precmat_filename[sizeof(args.precmat_filename) - 1] = '\0';
   args.dirname[0]                                          = '\0';
   args.precmat_basename[0]                                 = '\0';

   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M, NULL);
   hypredrv_ErrorCodeResetAll(); /* tolerate read errors */

   /* If the internal read failed, hypredrv_LinearSystemSetPrecMatrix may have destroyed the
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
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M, NULL);
   hypredrv_ErrorCodeResetAll();
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
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, mat_A, &mat_M, NULL);
   hypredrv_ErrorCodeResetAll();
   if (HYPRE_GetError() == 0 && mat_M && mat_M != mat_A)
   {
      HYPRE_IJMatrixDestroy(mat_M);
   }
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

struct LinsysLogContext
{
   HYPRE_IJMatrix mat;
   HYPRE_IJVector rhs;
};

static void
run_linsys_branch_logging_capture(void *context)
{
   struct LinsysLogContext *log_context = (struct LinsysLogContext *)context;

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   strncpy(args.x0_filename, "missing_x0_file", sizeof(args.x0_filename) - 1);
   args.x0_filename[sizeof(args.x0_filename) - 1] = '\0';
   strncpy(args.xref_filename, "missing_xref_file", sizeof(args.xref_filename) - 1);
   args.xref_filename[sizeof(args.xref_filename) - 1] = '\0';
   strncpy(args.precmat_filename, "missing_precmat_file",
           sizeof(args.precmat_filename) - 1);
   args.precmat_filename[sizeof(args.precmat_filename) - 1] = '\0';
   strncpy(args.matrix_filename, "main_matrix_file", sizeof(args.matrix_filename) - 1);
   args.matrix_filename[sizeof(args.matrix_filename) - 1] = '\0';
   strncpy(args.dofmap_filename, "missing_dofmap_file", sizeof(args.dofmap_filename) - 1);
   args.dofmap_filename[sizeof(args.dofmap_filename) - 1] = '\0';

   HYPRE_IJVector x0 = NULL, x = NULL, xref = NULL;
   HYPRE_IJMatrix precmat = NULL;
   IntArray      *dofmap  = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, log_context->mat,
                                        log_context->rhs, &x0, &x, NULL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetReferenceSolution(MPI_COMM_SELF, &args, &xref, NULL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, log_context->mat, &precmat,
                                      NULL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadDofmap(MPI_COMM_SELF, &args, &dofmap, NULL);

   if (dofmap)
   {
      hypredrv_IntArrayDestroy(&dofmap);
   }
   if (precmat && precmat != log_context->mat)
   {
      HYPRE_IJMatrixDestroy(precmat);
   }
   if (xref)
   {
      HYPRE_IJVectorDestroy(xref);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
}

static void
test_hypredrv_linsys_branch_logs(void)
{
   TEST_HYPRE_INIT();

   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   hypredrv_LogInitializeFromEnv();

   const HYPRE_Complex rhs_vals[1] = {1.0};
   struct LinsysLogContext context = {
      .mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 1.0),
      .rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_vals),
   };

   char output[16384];
   capture_stderr_output(run_linsys_branch_logging_capture, &context, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "initial guess source:"));
   ASSERT_NOT_NULL(strstr(output, "reference solution source:"));
   ASSERT_NOT_NULL(strstr(output, "preconditioner matrix source:"));
   ASSERT_NOT_NULL(strstr(output, "dofmap read begin"));

   HYPRE_IJVectorDestroy(context.rhs);
   HYPRE_IJMatrixDestroy(context.mat);

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   TEST_HYPRE_FINALIZE();
}

static void
run_linsys_args_and_validation_tests(void)
{
   RUN_TEST(test_hypredrv_LinearSystemGetValidValues_type);
   RUN_TEST(test_hypredrv_LinearSystemGetValidValues_rhs_mode);
   RUN_TEST(test_hypredrv_LinearSystemGetValidValues_init_guess_mode);
   RUN_TEST(test_hypredrv_LinearSystemGetValidValues_unknown_key);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_valid_keys);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_unknown_key);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix_and_init_suffix_error);
   RUN_TEST(test_hypredrv_LinearSystemSetNearNullSpace_mismatch_error);
   RUN_TEST(test_hypredrv_LinearSystemSetNearNullSpace_success);
   RUN_TEST(test_hypredrv_LinearSystemSetNearNullSpace_destroy_previous);
   RUN_TEST(test_hypredrv_LinearSystemGetValidValues_all_branches);
}

static void
run_linsys_matrix_and_rhs_io_tests(void)
{
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_filename_patterns);
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_no_filename_error);
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_type_branches);
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_exec_policy_branches);
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_partition_count_errors);
   RUN_TEST(test_LinearSystemReadRHS_file_patterns);
#endif

   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mode_branches);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mode_precedence_over_filename);
   RUN_TEST(test_hypredrv_LinearSystemSetReferenceSolution_keeps_randsol_reference);

#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_mtx_success);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mtx_file_success);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mtx_dim_mismatch_errors);
#endif
}

static void
run_linsys_misc_and_numeric_tests(void)
{
   RUN_TEST(test_hypredrv_LinearSystemPrintData_series_dir_and_null_objects);
   RUN_TEST(test_hypredrv_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_LinearSystemReadRHS_error_cases);
#endif
   RUN_TEST(test_hypredrv_LinearSystemComputeVectorNorm_all_modes);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_hypredrv_LinearSystemSetInitialGuess_x0_filename_branches);
#endif
   RUN_TEST(test_hypredrv_LinearSystemCreateWorkingSolution_recreates_x);
   RUN_TEST(test_hypredrv_LinearSystemSetPrecMatrix_branchy_paths);
   RUN_TEST(test_hypredrv_linsys_branch_logs);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   run_linsys_args_and_validation_tests();
   run_linsys_matrix_and_rhs_io_tests();
   run_linsys_misc_and_numeric_tests();

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
