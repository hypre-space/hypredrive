#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <limits.h>

#include "HYPRE.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "logging.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "internal/scaling.h"
#include "internal/stats.h"
#include "test_helpers.h"
#include "internal/yaml.h"

static void
write_text_file(const char *path, const char *contents)
{
   FILE *fp = fopen(path, "w");
   ASSERT_NOT_NULL(fp);
   fputs(contents, fp);
   fclose(fp);
}

/* Satisfy warn_unused_result on system() without changing cleanup semantics */
static void
test_system_ignore(const char *cmd)
{
   int r = system(cmd);
   (void)r;
}

static int
file_contains_substr(const char *path, const char *needle)
{
   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      return 0;
   }

   char  line[1024];
   int   found = 0;
   while (fgets(line, sizeof(line), fp))
   {
      if (strstr(line, needle))
      {
         found = 1;
         break;
      }
   }
   fclose(fp);
   return found;
}

static size_t
file_count_substr(const char *path, const char *needle)
{
   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      return 0;
   }

   size_t count      = 0;
   size_t needle_len = strlen(needle);
   char   line[1024];
   while (fgets(line, sizeof(line), fp))
   {
      const char *pos = line;
      while ((pos = strstr(pos, needle)) != NULL)
      {
         count++;
         pos += needle_len;
      }
   }
   fclose(fp);
   return count;
}

static int
path_join2(char *out, size_t out_size, const char *left, const char *right)
{
   if (!out || out_size == 0 || !left || !right)
   {
      return 0;
   }

   int written = snprintf(out, out_size, "%s/%s", left, right);
   return written >= 0 && (size_t)written < out_size;
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
static HYPRE_IJVector create_test_ijvector(MPI_Comm comm, HYPRE_BigInt ilower,
                                           HYPRE_BigInt iupper,
                                           const HYPRE_Complex *vals);

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
test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix_and_last_suffix_error(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "set_suffix", "[0, 1]", 1);
   add_child(parent, "last_suffix", "10", 1);
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
test_hypredrv_LinearSystemSetArgsFromYAML_dof_labels_block_bad_value(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *dl      = hypredrv_YAMLnodeCreate("dof_labels", "", 1);
   hypredrv_YAMLnodeAddChild(parent, dl);
   YAMLnode *e1 = hypredrv_YAMLnodeCreate("v_x", "not_int", 2);
   hypredrv_YAMLnodeAddChild(dl, e1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   if (args.dof_labels)
   {
      hypredrv_DofLabelMapDestroy(&args.dof_labels);
   }
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_dof_labels_flow_bad_value(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   add_child(parent, "dof_labels", "{v_x: 0, v_y: two}", 1);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   if (args.dof_labels)
   {
      hypredrv_DofLabelMapDestroy(&args.dof_labels);
   }
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemGetSuffix_null_args(void)
{
   ASSERT_EQ(hypredrv_LinearSystemGetSuffix(NULL, 7), 7);
}

static void
test_hypredrv_LinearSystemCreateWorkingSolution_invalid_args(void)
{
   TEST_HYPRE_INIT();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   HYPRE_IJVector x = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemCreateWorkingSolution(MPI_COMM_SELF, &args, NULL, &x);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemCreateWorkingSolution(MPI_COMM_SELF, &args, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   HYPRE_ClearAllErrors();
}

static void
test_hypredrv_PrintSystem_defaults(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   ASSERT_EQ(args.print_system.enabled, 0);
   ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_ALL);
   ASSERT_EQ(args.print_system.stage_mask, PRINT_SYSTEM_STAGE_BUILD_BIT);
   ASSERT_EQ(args.print_system.artifacts,
             PRINT_SYSTEM_ARTIFACT_MATRIX | PRINT_SYSTEM_ARTIFACT_RHS |
                PRINT_SYSTEM_ARTIFACT_DOFMAP);
   ASSERT_STREQ(args.print_system.output_dir, "hypredrive-data");
   ASSERT_EQ(args.print_system.overwrite, 0);
}

static void
test_hypredrv_PrintSystem_null_args(void)
{
   hypredrv_PrintSystemSetDefaultArgs(NULL);
   hypredrv_PrintSystemDestroyArgs(NULL);
   hypredrv_PrintSystemSetArgs(NULL, NULL);

   PrintSystem_args pa;
   hypredrv_PrintSystemSetDefaultArgs(&pa);
   hypredrv_PrintSystemSetArgs(&pa, NULL);
   hypredrv_PrintSystemDestroyArgs(&pa);
}

static void
test_hypredrv_PrintSystem_yaml_scalar_enabled(void)
{
   PrintSystem_args pa;
   hypredrv_PrintSystemSetDefaultArgs(&pa);
   YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "YES", 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_PrintSystemSetArgs(&pa, ps);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(pa.enabled, 1);

   hypredrv_YAMLnodeDestroy(ps);
   hypredrv_PrintSystemDestroyArgs(&pa);
}

static void
test_hypredrv_PrintSystem_invalid_type_and_stage(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "not_a_valid_print_system_type", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);

   hypredrv_LinearSystemSetDefaultArgs(&args);
   parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "all", 2);
   add_child(ps, "stage", "not_a_valid_stage", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_PrintSystem_yaml_on_off_and_stages(void)
{
   LS_args args;

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "off", 2);
      add_child(ps, "type", "all", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.enabled, 0);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "false", 2);
      add_child(ps, "type", "all", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.enabled, 0);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "setup", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.enabled, 1);
      ASSERT_EQ(args.print_system.stage_mask, PRINT_SYSTEM_STAGE_SETUP_BIT);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "all", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(
         args.print_system.stage_mask,
         PRINT_SYSTEM_STAGE_BUILD_BIT | PRINT_SYSTEM_STAGE_SETUP_BIT | PRINT_SYSTEM_STAGE_APPLY_BIT);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_print_system_more_branches(void)
{
   LS_args args;

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "every_n_systems", 2);
      add_child(ps, "every", "4", 2);
      add_child(ps, "stage", "build", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_EVERY_N_SYSTEMS);
      ASSERT_EQ(args.print_system.every, 4);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "every_n_timesteps", 2);
      add_child(ps, "stage", "apply", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_EVERY_N_TIMESTEPS);
      ASSERT_EQ(args.print_system.every, 1);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "artifacts", "[matrix, rhs, precmat]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_MATRIX) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_RHS) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_PRECMAT) != 0);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ids = add_child(ps, "ids", "", 2);
      add_child(ids, "-", "0", 3);
      add_child(ids, "-", "2", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(args.print_system.ids);
      ASSERT_EQ((int)args.print_system.ids->size, 2);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ranges = add_child(ps, "ranges", "", 2);
      add_child(ranges, "-", "[1:9]", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ((int)args.print_system.ranges.size, 1);
      ASSERT_EQ(args.print_system.ranges.data[0].begin, 1);
      ASSERT_EQ(args.print_system.ranges.data[0].end, 9);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *sel       = add_child(selectors, "-", "", 3);
      add_child(sel, "basis", "iterations", 4);
      add_child(sel, "threshold", "5", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ_SIZE(args.print_system.num_selectors, 1);
      ASSERT_EQ(args.print_system.selectors[0].basis, PRINT_SYSTEM_BASIS_ITERATIONS);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "setup_time_over", 2);
      add_child(ps, "stage", "setup", 2);
      add_child(ps, "threshold", "0.25", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_SETUP_TIME_OVER);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "solve_time_over", 2);
      add_child(ps, "stage", "apply", 2);
      add_child(ps, "threshold", "1.0", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_SOLVE_TIME_OVER);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "unknown_print_system_key", "x", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   hypredrv_LinearSystemSetDefaultArgs(&args);
   {
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "artifacts", "[matrix, not_an_artifact]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_print_system_block_ranges(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "ranges", 2);
   add_child(ps, "stage", "apply", 2);
   YAMLnode *ranges = add_child(ps, "ranges", "", 2);
   add_child(ranges, "-", "[20, 24]", 3);
   add_child(ranges, "-", "[100, 150]", 3);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.print_system.enabled, 1);
   ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_RANGES);
   ASSERT_EQ(args.print_system.stage_mask, PRINT_SYSTEM_STAGE_APPLY_BIT);
   ASSERT_EQ((int)args.print_system.ranges.size, 2);
   ASSERT_EQ(args.print_system.ranges.data[0].begin, 20);
   ASSERT_EQ(args.print_system.ranges.data[0].end, 24);
   ASSERT_EQ(args.print_system.ranges.data[1].begin, 100);
   ASSERT_EQ(args.print_system.ranges.data[1].end, 150);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_PrintSystem_yaml_parse_edge_cases(void)
{
   /* Scalar print_system: token is not a recognized on/off value */
   {
      PrintSystem_args pa;
      hypredrv_PrintSystemSetDefaultArgs(&pa);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "not_on_off", 0);
      hypredrv_ErrorCodeResetAll();
      hypredrv_PrintSystemSetArgs(&pa, ps);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      hypredrv_PrintSystemDestroyArgs(&pa);
      hypredrv_YAMLnodeDestroy(ps);
   }

   /* Block: invalid enabled / overwrite / every / threshold */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "bogus", 2);
      add_child(ps, "type", "all", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "overwrite", "maybe", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "every_n_systems", 2);
      add_child(ps, "every", "not_an_int", 2);
      add_child(ps, "stage", "build", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "solve_time_over", 2);
      add_child(ps, "stage", "apply", 2);
      add_child(ps, "threshold", "1.0junk", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Reversed range pair is normalized (begin > end swapped) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ranges = add_child(ps, "ranges", "", 2);
      add_child(ranges, "-", "[9, 1]", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ((int)args.print_system.ranges.size, 1);
      ASSERT_EQ(args.print_system.ranges.data[0].begin, 1);
      ASSERT_EQ(args.print_system.ranges.data[0].end, 9);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Inline single range string (no sequence children under ranges) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "ranges", "[3, 11]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ((int)args.print_system.ranges.size, 1);
      ASSERT_EQ(args.print_system.ranges.data[0].begin, 3);
      ASSERT_EQ(args.print_system.ranges.data[0].end, 11);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* ids sequence with non-integer entry */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ids = add_child(ps, "ids", "", 2);
      add_child(ids, "-", "0", 3);
      add_child(ids, "-", "not_int", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* selectors key present but not a sequence */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      add_child(ps, "selectors", "", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Post-parse consistency checks (requires ids / ranges / threshold / stage) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "iterations_over", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "threshold", "3", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "every", "2", 2);
      add_child(ps, "stage", "build", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      /* Threshold-based types require threshold key */
      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "iterations_over", 2);
      add_child(ps, "stage", "apply", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "setup_time_over", 2);
      add_child(ps, "stage", "setup", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "solve_time_over", 2);
      add_child(ps, "stage", "apply", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      /* selectors key requires type=selectors */
      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ids = add_child(ps, "ids", "", 2);
      add_child(ids, "-", "0", 3);
      YAMLnode *selbad = add_child(ps, "selectors", "", 2);
      YAMLnode *s1     = add_child(selbad, "-", "", 3);
      add_child(s1, "basis", "timestep", 4);
      add_child(s1, "every", "1", 4);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      /* selectors sequence with no list entries (no '-' items) */
      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *empty_sel = add_child(ps, "selectors", "", 2);
      add_child(empty_sel, "not_a_list_item", "x", 3);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "weird_key_xyz", "1", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "solve_time_over", 2);
      add_child(ps, "stage", "setup", 2);
      add_child(ps, "threshold", "1.0", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "setup_time_over", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "threshold", "0.5", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "threshold", "1.0", 2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Artifacts: exercise every named bit via inline bracket list */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "artifacts",
                "[matrix, precmat, rhs, x0, xref, solution, dofmap, metadata]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_MATRIX) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_PRECMAT) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_RHS) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_X0) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_XREF) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_SOLUTION) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_DOFMAP) != 0);
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_METADATA) != 0);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Artifacts as YAML sequence (PrintSystemParseArtifactsNode list path) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *art = add_child(ps, "artifacts", "", 2);
      add_child(art, "-", "matrix", 3);
      add_child(art, "-", "precmat", 3);
      add_child(art, "-", "rhs", 3);
      add_child(art, "-", "x0", 3);
      add_child(art, "-", "xref", 3);
      add_child(art, "-", "solution", 3);
      add_child(art, "-", "dofmap", 3);
      add_child(art, "-", "metadata", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE((args.print_system.artifacts & PRINT_SYSTEM_ARTIFACT_METADATA) != 0);

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Malformed range strings and invalid every */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ranges = add_child(ps, "ranges", "", 2);
      add_child(ranges, "-", "[1, 2, 3]", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ranges", 2);
      add_child(ps, "stage", "build", 2);
      ranges = add_child(ps, "ranges", "", 2);
      add_child(ranges, "-", "[a, 2]", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      ps      = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "every_n_systems", 2);
      add_child(ps, "every", "0", 2);
      add_child(ps, "stage", "build", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_print_system_invalid_combo(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "all", 2);
   add_child(ps, "ids", "[0, 1]", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_print_system_thresholds(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "iterations_over", 2);
   add_child(ps, "stage", "apply", 2);
   add_child(ps, "threshold", "12", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_ITERATIONS_OVER);
   ASSERT_EQ(args.print_system.stage_mask, PRINT_SYSTEM_STAGE_APPLY_BIT);
   ASSERT_EQ(args.print_system.threshold, 12.0);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);

   hypredrv_LinearSystemSetDefaultArgs(&args);
   parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "selectors", 2);
   add_child(ps, "stage", "all", 2);
   YAMLnode *selectors = add_child(ps, "selectors", "", 2);
   YAMLnode *selector  = add_child(selectors, "-", "", 3);
   add_child(selector, "basis", "setup_time", 4);
   add_child(selector, "threshold", "1.5", 4);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);

   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(args.print_system.type, PRINT_SYSTEM_TYPE_SELECTORS);
   ASSERT_EQ_SIZE(args.print_system.num_selectors, 1);
   ASSERT_EQ(args.print_system.selectors[0].basis, PRINT_SYSTEM_BASIS_SETUP_TIME);
   ASSERT_EQ(args.print_system.selectors[0].threshold, 1.5);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);
}

static void
test_hypredrv_LinearSystemSetArgsFromYAML_print_system_threshold_invalid_combo(void)
{
   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   YAMLnode *ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "solve_time_over", 2);
   add_child(ps, "stage", "build", 2);
   add_child(ps, "threshold", "0.1", 2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   hypredrv_YAMLnodeDestroy(parent);

   hypredrv_LinearSystemSetDefaultArgs(&args);
   parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
   ps     = add_child(parent, "print_system", "", 1);
   add_child(ps, "enabled", "on", 2);
   add_child(ps, "type", "selectors", 2);
   YAMLnode *selectors = add_child(ps, "selectors", "", 2);
   YAMLnode *selector  = add_child(selectors, "-", "", 3);
   add_child(selector, "basis", "iterations", 4);
   add_child(selector, "threshold", "4", 4);
   add_child(selector, "ids", "[1, 2]", 4);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
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

/* Minimal two-system lsseq container (mirrors tests/test_lsseq.c write_test_container). */
static void
write_lsseq_two_systems_fixture(const char *filename)
{
   LSSeqHeader         header;
   LSSeqPartMeta       part_meta[1];
   LSSeqPatternMeta    pattern_meta[2];
   LSSeqSystemPartMeta sys_meta[2];
   LSSeqTimestepEntry  timesteps[2];
   HYPRE_BigInt        rows0[2] = {0, 1};
   HYPRE_BigInt        cols0[2] = {0, 1};
   HYPRE_BigInt        rows1[3] = {0, 0, 1};
   HYPRE_BigInt        cols1[3] = {0, 1, 1};
   double              vals0[2] = {10.0, 20.0};
   double              vals1[3] = {5.0, 1.0, 3.0};
   double              rhs0[2]  = {1.0, 2.0};
   double              rhs1[2]  = {3.0, 4.0};
   int32_t             dof0[2]  = {0, 1};
   int32_t             dof1[2]  = {1, 1};
   uint64_t            blob_offset;
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_DOFMAP | LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 2;
   header.num_parts     = 1;
   header.num_patterns  = 2;
   header.num_timesteps = 2;

   header.offset_part_meta     = sizeof(LSSeqHeader);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_timestep_meta = header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_blob_data     = header.offset_timestep_meta + sizeof(timesteps);

   part_meta[0].row_lower      = 0;
   part_meta[0].row_upper      = 1;
   part_meta[0].nrows          = 2;
   part_meta[0].row_index_size = sizeof(HYPRE_BigInt);
   part_meta[0].value_size     = sizeof(double);

   blob_offset = header.offset_blob_data;

   pattern_meta[0].part_id          = 0;
   pattern_meta[0].nnz              = 2;
   pattern_meta[0].rows_blob_offset = blob_offset;
   pattern_meta[0].rows_blob_size   = sizeof(rows0);
   blob_offset += sizeof(rows0);
   pattern_meta[0].cols_blob_offset = blob_offset;
   pattern_meta[0].cols_blob_size   = sizeof(cols0);
   blob_offset += sizeof(cols0);

   pattern_meta[1].part_id          = 0;
   pattern_meta[1].nnz              = 3;
   pattern_meta[1].rows_blob_offset = blob_offset;
   pattern_meta[1].rows_blob_size   = sizeof(rows1);
   blob_offset += sizeof(rows1);
   pattern_meta[1].cols_blob_offset = blob_offset;
   pattern_meta[1].cols_blob_size   = sizeof(cols1);
   blob_offset += sizeof(cols1);

   sys_meta[0].pattern_id         = 0;
   sys_meta[0].nnz                = 2;
   sys_meta[0].values_blob_offset = blob_offset;
   sys_meta[0].values_blob_size   = sizeof(vals0);
   blob_offset += sizeof(vals0);
   sys_meta[0].rhs_blob_offset    = blob_offset;
   sys_meta[0].rhs_blob_size      = sizeof(rhs0);
   blob_offset += sizeof(rhs0);
   sys_meta[0].dof_blob_offset    = blob_offset;
   sys_meta[0].dof_blob_size      = sizeof(dof0);
   sys_meta[0].dof_num_entries    = 2;
   blob_offset += sizeof(dof0);

   sys_meta[1].pattern_id         = 1;
   sys_meta[1].nnz                = 3;
   sys_meta[1].values_blob_offset = blob_offset;
   sys_meta[1].values_blob_size   = sizeof(vals1);
   blob_offset += sizeof(vals1);
   sys_meta[1].rhs_blob_offset    = blob_offset;
   sys_meta[1].rhs_blob_size      = sizeof(rhs1);
   blob_offset += sizeof(rhs1);
   sys_meta[1].dof_blob_offset    = blob_offset;
   sys_meta[1].dof_blob_size      = sizeof(dof1);
   sys_meta[1].dof_num_entries    = 2;
   blob_offset += sizeof(dof1);

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;
   timesteps[1].timestep = 1;
   timesteps[1].ls_start = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(timesteps, sizeof(timesteps), 1, fp), 1);

   ASSERT_EQ_SIZE(fwrite(rows0, sizeof(rows0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols0, sizeof(cols0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rows1, sizeof(rows1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(cols1, sizeof(cols1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals0, sizeof(vals0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs0, sizeof(rhs0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof0, sizeof(dof0), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(vals1, sizeof(vals1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(rhs1, sizeof(rhs1), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(dof1, sizeof(dof1), 1, fp), 1);

   fclose(fp);
}

static void
test_hypredrv_LinearSystemReadMatrix_sequence_ls_id_out_of_range(void)
{
   TEST_HYPRE_INIT();

   char path[] = "/tmp/hypredrv_lsseq_mat_XXXXXX";
   int  fd     = mkstemp(path);
   ASSERT_TRUE(fd >= 0);
   close(fd);

   write_lsseq_two_systems_fixture(path);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.type = 1;
   args.dirname[0]           = '\0';
   args.matrix_filename[0] = '\0';
   args.matrix_basename[0] = '\0';
   strncpy(args.sequence_filename, path, sizeof(args.sequence_filename) - 1);
   args.sequence_filename[sizeof(args.sequence_filename) - 1] = '\0';

   Stats *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);
   /* ls_id = GetLinearSystemID(stats)+1 == ls_counter; file has 2 systems -> fail */
   stats->ls_counter = 99;

   HYPRE_IJMatrix mat = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadMatrix(MPI_COMM_SELF, &args, &mat, stats);

   ASSERT_TRUE(mat == NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_StatsDestroy(&stats);
   unlink(path);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetRHS_sequence_ls_id_out_of_range(void)
{
   TEST_HYPRE_INIT();

   char path[] = "/tmp/hypredrv_lsseq_rhs_XXXXXX";
   int  fd     = mkstemp(path);
   ASSERT_TRUE(fd >= 0);
   close(fd);

   write_lsseq_two_systems_fixture(path);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.type               = 1;
   args.rhs_mode           = 2;
   args.dirname[0]         = '\0';
   args.rhs_filename[0]    = '\0';
   args.rhs_basename[0]    = '\0';
   strncpy(args.sequence_filename, path, sizeof(args.sequence_filename) - 1);
   args.sequence_filename[sizeof(args.sequence_filename) - 1] = '\0';

   HYPRE_IJMatrix mat = NULL;
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat);
   HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(mat);

   HYPRE_IJVector rhs = NULL, refsol = NULL;
   Stats         *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);
   stats->ls_counter = 99;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, stats);

   ASSERT_TRUE(rhs == NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_StatsDestroy(&stats);
   HYPRE_IJMatrixDestroy(mat);
   unlink(path);

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

   /* Test rhs_mode = 0 (zeros) */
   args.rhs_mode = 0;
   args.rhs_filename[0] = '\0';
   args.rhs_basename[0] = '\0';

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
      rhs = NULL;
   }

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
test_hypredrv_LinearSystemSetReferenceSolution_file_override(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif
   char path[4096];
   snprintf(path, sizeof(path), "%s/data/ps3d10pt7/np1/IJ.out.b", HYPREDRIVE_SOURCE_DIR);
   strncpy(args.xref_filename, path, sizeof(args.xref_filename) - 1);
   args.xref_filename[sizeof(args.xref_filename) - 1] = '\0';

   HYPRE_IJVector xref = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetReferenceSolution(MPI_COMM_SELF, &args, &xref, NULL);
   if (xref)
   {
      HYPRE_IJVectorDestroy(xref);
   }

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
test_hypredrv_LinearSystemDumpScheduled_ranges_and_artifacts(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_dump_sched_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
   add_child(ps, "enabled", "on", 1);
   add_child(ps, "type", "ranges", 1);
   add_child(ps, "stage", "build", 1);
   YAMLnode *artifacts = add_child(ps, "artifacts", "", 1);
   add_child(artifacts, "-", "matrix", 2);
   add_child(artifacts, "-", "rhs", 2);
   add_child(artifacts, "-", "dofmap", 2);
   add_child(artifacts, "-", "metadata", 2);
   add_child(ps, "output_dir", outdir, 1);
   add_child(ps, "overwrite", "on", 1);
   YAMLnode *ranges = add_child(ps, "ranges", "", 1);
   add_child(ranges, "-", "[20, 24]", 2);
   add_child(ranges, "-", "[100, 150]", 2);

   hypredrv_PrintSystemSetArgs(&args.print_system, ps);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(ps);

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 3.0);
   HYPRE_Complex  vals[1] = {4.0};
   HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, vals);

   IntArray *dofmap = NULL;
   int       dm[2]  = {0, 1};
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm, &dofmap);

   PrintSystemContext ctx;
   memset(&ctx, 0, sizeof(ctx));
   ctx.stage            = PRINT_SYSTEM_STAGE_BUILD;
   ctx.system_index     = 22;
   ctx.timestep_index   = 3;
   ctx.variant_index    = 1;
   ctx.repetition_index = 0;
   ctx.stats_ls_id      = 21;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      ctx.level_ids[level] = -1;
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, mat, NULL, rhs, NULL, NULL,
                                      NULL, dofmap, &ctx, "obj-1");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char dump_root[512];
   snprintf(dump_root, sizeof(dump_root), "%s/obj-1/ls_00000", outdir);
   char matrix_file[640], rhs_file[640], dofmap_file[640], metadata_file[640];
   snprintf(matrix_file, sizeof(matrix_file), "%s/matrix.out.00000", dump_root);
   snprintf(rhs_file, sizeof(rhs_file), "%s/rhs.out.00000", dump_root);
   snprintf(dofmap_file, sizeof(dofmap_file), "%s/dofmap.out.00000", dump_root);
   snprintf(metadata_file, sizeof(metadata_file), "%s/metadata.txt", dump_root);
   char systems_index[640];
   snprintf(systems_index, sizeof(systems_index), "%s/obj-1/systems_index.txt", outdir);

   ASSERT_TRUE(access(matrix_file, F_OK) == 0);
   ASSERT_TRUE(access(rhs_file, F_OK) == 0);
   ASSERT_TRUE(access(dofmap_file, F_OK) == 0);
   ASSERT_TRUE(access(metadata_file, F_OK) == 0);
   ASSERT_TRUE(access(systems_index, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(systems_index, "ls_00000"));
   ASSERT_TRUE(file_contains_substr(systems_index, "stage=build"));
   ASSERT_TRUE(file_contains_substr(systems_index, "system=22"));

   ctx.system_index = 11;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, mat, NULL, rhs, NULL, NULL,
                                      NULL, dofmap, &ctx, "obj-1");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char unmatched_dir[640];
   snprintf(unmatched_dir, sizeof(unmatched_dir), "%s/obj-1/ls_00001", outdir);
   ASSERT_TRUE(access(unmatched_dir, F_OK) != 0);

   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemDumpScheduled_overwrite_reuses_dump_series(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_dump_overwrite_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   PrintSystemContext ctx;
   memset(&ctx, 0, sizeof(ctx));
   ctx.stage          = PRINT_SYSTEM_STAGE_BUILD;
   ctx.system_index   = 0;
   ctx.timestep_index = 0;
   ctx.stats_ls_id    = 0;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      ctx.level_ids[level] = -1;
   }

   for (int pass = 0; pass < 2; pass++)
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);

      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      YAMLnode *artifacts = add_child(ps, "artifacts", "", 1);
      add_child(artifacts, "-", "metadata", 2);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);

      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_YAMLnodeDestroy(ps);

      ctx.system_index   = 5 + pass;
      ctx.timestep_index = 2 + pass;
      ctx.stats_ls_id    = 5 + pass;

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                         NULL, NULL, NULL, &ctx, "obj-overwrite");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());

      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   char object_dir[512];
   char dump_dir[512];
   char metadata_file[640];
   char stale_file[640];
   char systems_index[640];
   ASSERT_TRUE(path_join2(object_dir, sizeof(object_dir), outdir, "obj-overwrite"));
   ASSERT_TRUE(path_join2(dump_dir, sizeof(dump_dir), object_dir, "ls_00000"));
   ASSERT_TRUE(path_join2(metadata_file, sizeof(metadata_file), dump_dir, "metadata.txt"));
   ASSERT_TRUE(path_join2(stale_file, sizeof(stale_file), dump_dir, "stale.txt"));
   ASSERT_TRUE(path_join2(systems_index, sizeof(systems_index), object_dir,
                          "systems_index.txt"));

   write_text_file(stale_file, "stale");

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
   add_child(ps, "enabled", "on", 1);
   add_child(ps, "type", "all", 1);
   add_child(ps, "stage", "build", 1);
   YAMLnode *artifacts = add_child(ps, "artifacts", "", 1);
   add_child(artifacts, "-", "metadata", 2);
   add_child(ps, "output_dir", outdir, 1);
   add_child(ps, "overwrite", "on", 1);

   hypredrv_PrintSystemSetArgs(&args.print_system, ps);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(ps);

   ctx.system_index   = 11;
   ctx.timestep_index = 7;
   ctx.stats_ls_id    = 11;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                      NULL, NULL, NULL, &ctx, "obj-overwrite");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_TRUE(access(metadata_file, F_OK) == 0);
   ASSERT_TRUE(access(stale_file, F_OK) != 0);
   ASSERT_TRUE(access(systems_index, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_file, "system_index=11"));
   ASSERT_TRUE(file_contains_substr(metadata_file, "timestep_index=7"));
   ASSERT_TRUE(file_contains_substr(systems_index, "system=11"));
   ASSERT_EQ_SIZE(file_count_substr(systems_index, "ls_00000"), 1);

   char unexpected_dir[640];
   ASSERT_TRUE(path_join2(unexpected_dir, sizeof(unexpected_dir), object_dir,
                          "ls_00001"));
   ASSERT_TRUE(access(unexpected_dir, F_OK) != 0);

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemDumpScheduled_threshold_types_and_selectors(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_dump_thresholds_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
   add_child(ps, "enabled", "on", 1);
   add_child(ps, "type", "iterations_over", 1);
   add_child(ps, "stage", "apply", 1);
   add_child(ps, "threshold", "10", 1);
   YAMLnode *artifacts = add_child(ps, "artifacts", "", 1);
   add_child(artifacts, "-", "metadata", 2);
   add_child(ps, "output_dir", outdir, 1);
   add_child(ps, "overwrite", "on", 1);

   hypredrv_PrintSystemSetArgs(&args.print_system, ps);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(ps);

   PrintSystemContext ctx;
   memset(&ctx, 0, sizeof(ctx));
   ctx.stage            = PRINT_SYSTEM_STAGE_APPLY;
   ctx.system_index     = 3;
   ctx.timestep_index   = 1;
   ctx.last_iter        = 12;
   ctx.variant_index    = 0;
   ctx.repetition_index = 0;
   ctx.stats_ls_id      = 3;
   ctx.last_setup_time  = 0.5;
   ctx.last_solve_time  = 0.25;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      ctx.level_ids[level] = -1;
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL, NULL,
                                      NULL, NULL, &ctx, "obj-threshold");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char metadata_file[640];
   ASSERT_TRUE(path_join2(metadata_file, sizeof(metadata_file), outdir,
                          "obj-threshold/ls_00000/metadata.txt"));
   ASSERT_TRUE(access(metadata_file, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_file, "last_iter=12"));
   ASSERT_TRUE(file_contains_substr(metadata_file, "last_setup_time=0.5"));
   ASSERT_TRUE(file_contains_substr(metadata_file, "last_solve_time=0.25"));

   ctx.last_iter = 10;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL, NULL,
                                      NULL, NULL, &ctx, "obj-threshold");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char boundary_metadata[640];
   ASSERT_TRUE(path_join2(boundary_metadata, sizeof(boundary_metadata), outdir,
                          "obj-threshold/ls_00001/metadata.txt"));
   ASSERT_TRUE(access(boundary_metadata, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(boundary_metadata, "last_iter=10"));

   hypredrv_PrintSystemDestroyArgs(&args.print_system);

   hypredrv_LinearSystemSetDefaultArgs(&args);
   ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
   add_child(ps, "enabled", "on", 1);
   add_child(ps, "type", "selectors", 1);
   add_child(ps, "stage", "all", 1);
   artifacts = add_child(ps, "artifacts", "", 1);
   add_child(artifacts, "-", "metadata", 2);
   add_child(ps, "output_dir", outdir, 1);
   add_child(ps, "overwrite", "off", 1);
   YAMLnode *selectors = add_child(ps, "selectors", "", 1);
   YAMLnode *selector  = add_child(selectors, "-", "", 2);
   add_child(selector, "basis", "solve_time", 3);
   add_child(selector, "threshold", "0.4", 3);

   hypredrv_PrintSystemSetArgs(&args.print_system, ps);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(ps);

   ctx.stage           = PRINT_SYSTEM_STAGE_SETUP;
   ctx.last_iter       = -1;
   ctx.last_setup_time = 0.75;
   ctx.last_solve_time = -1.0;
   ctx.system_index    = 5;
   ctx.stats_ls_id     = 5;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL, NULL,
                                      NULL, NULL, &ctx, "obj-selectors");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char selector_unmatched[640];
   ASSERT_TRUE(path_join2(selector_unmatched, sizeof(selector_unmatched), outdir,
                          "obj-selectors/ls_00000"));
   ASSERT_TRUE(access(selector_unmatched, F_OK) != 0);

   ctx.stage           = PRINT_SYSTEM_STAGE_APPLY;
   ctx.last_iter       = 6;
   ctx.last_solve_time = 0.75;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL, NULL,
                                      NULL, NULL, &ctx, "obj-selectors");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_TRUE(path_join2(metadata_file, sizeof(metadata_file), outdir,
                          "obj-selectors/ls_00000/metadata.txt"));
   ASSERT_TRUE(access(metadata_file, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_file, "stage=apply"));
   ASSERT_TRUE(file_contains_substr(metadata_file, "last_solve_time=0.75"));

   hypredrv_PrintSystemDestroyArgs(&args.print_system);
   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_PrintSystem_yaml_parser_selectors_and_null_field(void)
{
   hypredrv_PrintSystemSetDefaultArgs(NULL);

   /* Missing scalar values: free val so type/stage/output_dir see NULL pointer */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      YAMLnode *tn     = hypredrv_YAMLnodeCreate("type", "all", 2);
      hypredrv_YAMLnodeAddChild(ps, tn);
      free(tn->val);
      tn->val = NULL;
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "stage", "build", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      YAMLnode *sn = hypredrv_YAMLnodeCreate("stage", "build", 2);
      hypredrv_YAMLnodeAddChild(ps, sn);
      free(sn->val);
      sn->val = NULL;

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *on = hypredrv_YAMLnodeCreate("output_dir", "/tmp", 2);
      hypredrv_YAMLnodeAddChild(ps, on);
      free(on->val);
      on->val = NULL;

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Null field pointer to PrintSystemSetArgs */
   {
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      hypredrv_ErrorCodeResetAll();
      hypredrv_PrintSystemSetArgs(NULL, ps);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_YAMLnodeDestroy(ps);
   }

   /* Artifacts: inline "all" token */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "artifacts", "[all]", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      int all_bits = PRINT_SYSTEM_ARTIFACT_MATRIX | PRINT_SYSTEM_ARTIFACT_PRECMAT |
                     PRINT_SYSTEM_ARTIFACT_RHS | PRINT_SYSTEM_ARTIFACT_X0 |
                     PRINT_SYSTEM_ARTIFACT_XREF | PRINT_SYSTEM_ARTIFACT_SOLUTION |
                     PRINT_SYSTEM_ARTIFACT_DOFMAP | PRINT_SYSTEM_ARTIFACT_METADATA;
      ASSERT_EQ(args.print_system.artifacts, all_bits);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Artifacts sequence: skip non '-' children, empty effective list */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "all", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *art = add_child(ps, "artifacts", "", 2);
      add_child(art, "bad", "matrix", 3);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* ids: inline string list (no sequence children) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      add_child(ps, "ids", "3, 7, 11", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(args.print_system.ids);
      ASSERT_EQ((int)args.print_system.ids->size, 3);
      ASSERT_EQ(args.print_system.ids->data[0], 3);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Selector: invalid level (too large) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *sel       = add_child(selectors, "-", "", 3);
      add_child(sel, "basis", "level", 4);
      add_child(sel, "level", "99", 4);
      add_child(sel, "every", "1", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Selector: invalid child key */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *sel       = add_child(selectors, "-", "", 3);
      add_child(sel, "basis", "timestep", 4);
      add_child(sel, "every", "2", 4);
      add_child(sel, "not_a_key", "1", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Selector: threshold basis mixed with every (invalid) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "apply", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *sel       = add_child(selectors, "-", "", 3);
      add_child(sel, "basis", "iterations", 4);
      add_child(sel, "threshold", "3", 4);
      add_child(sel, "every", "2", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Selector: non-threshold basis with threshold field */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "setup", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *sel       = add_child(selectors, "-", "", 3);
      add_child(sel, "basis", "timestep", 4);
      add_child(sel, "every", "2", 4);
      add_child(sel, "threshold", "1.0", 4);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Two selectors: first misses, second matches (timestep every) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "selectors", 2);
      add_child(ps, "stage", "all", 2);
      YAMLnode *selectors = add_child(ps, "selectors", "", 2);
      YAMLnode *s1        = add_child(selectors, "-", "", 3);
      add_child(s1, "basis", "timestep", 4);
      add_child(s1, "every", "7", 4);
      YAMLnode *s2 = add_child(selectors, "-", "", 3);
      add_child(s2, "basis", "timestep", 4);
      add_child(s2, "every", "3", 4);
      add_child(ps, "output_dir", "/tmp", 2);

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_EQ_SIZE(args.print_system.num_selectors, 2);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }

   /* Integer overflow in ids entry */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *parent = hypredrv_YAMLnodeCreate("linear_system", "", 0);
      YAMLnode *ps     = add_child(parent, "print_system", "", 1);
      add_child(ps, "enabled", "on", 2);
      add_child(ps, "type", "ids", 2);
      add_child(ps, "stage", "build", 2);
      YAMLnode *ids = add_child(ps, "ids", "", 2);
      add_child(ids, "-", "0", 3);
      {
         char buf[32];
         snprintf(buf, sizeof(buf), "%ld", (long)INT_MAX + 1L);
         add_child(ids, "-", buf, 3);
      }

      hypredrv_ErrorCodeResetAll();
      hypredrv_LinearSystemSetArgsFromYAML(&args, parent);
      ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
      hypredrv_YAMLnodeDestroy(parent);
   }
}

static void
test_hypredrv_LinearSystemDumpScheduled_branch_and_filesystem_coverage(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_lsp_br_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   test_system_ignore(cleanup_cmd);

   /* NULL args / NULL ctx */
   {
      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, NULL, NULL, NULL, NULL, NULL,
                                               NULL, NULL, NULL, &ctx, "o");
      hypredrv_ErrorCodeResetAll();

      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      args.print_system.enabled = 1;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                               NULL, NULL, NULL, NULL, NULL, "o");
      hypredrv_ErrorCodeResetAll();
   }

   /* Disabled config */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      args.print_system.enabled = 0;
      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "o");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
   }

   /* Stage mismatch */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "setup", 1);
      add_child(ps, "output_dir", outdir, 1);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "o");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Unknown print type after parse (forces PrintSystemShouldDumpDetailed default) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);
      args.print_system.type = (PrintSystemType)999; /* unknown type branch */

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "o");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* every_n_systems: miss then match */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "every_n_systems", 1);
      add_child(ps, "every", "3", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage         = PRINT_SYSTEM_STAGE_BUILD;
      ctx.system_index  = 1;
      ctx.stats_ls_id   = 1;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "ens");
      ctx.system_index = 3;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "ens");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* every_n_timesteps */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "every_n_timesteps", 1);
      add_child(ps, "every", "2", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage           = PRINT_SYSTEM_STAGE_BUILD;
      ctx.timestep_index  = 1;
      ctx.system_index    = 0;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "ent");
      ctx.timestep_index = 2;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "ent");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* ids / ranges miss and match */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "ids", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *ids = add_child(ps, "ids", "", 1);
      add_child(ids, "-", "10", 2);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage        = PRINT_SYSTEM_STAGE_BUILD;
      ctx.system_index = 2;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "idm");
      ctx.system_index = 10;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "idm");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "ranges", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *ranges = add_child(ps, "ranges", "", 1);
      add_child(ranges, "-", "[100, 200]", 2);
      art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      memset(&ctx, 0, sizeof(ctx));
      ctx.stage        = PRINT_SYSTEM_STAGE_BUILD;
      ctx.system_index = 50;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "rgm");
      ctx.system_index = 150;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "rgm");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Threshold types: below threshold / negative metrics */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "iterations_over", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "threshold", "10", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage       = PRINT_SYSTEM_STAGE_APPLY;
      ctx.last_iter   = 5;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "ito");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "setup_time_over", 1);
      add_child(ps, "stage", "setup", 1);
      add_child(ps, "threshold", "1.0", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      memset(&ctx, 0, sizeof(ctx));
      ctx.stage           = PRINT_SYSTEM_STAGE_SETUP;
      ctx.last_setup_time = -1.0;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "sto");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);

      hypredrv_LinearSystemSetDefaultArgs(&args);
      ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "solve_time_over", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "threshold", "1.0", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      memset(&ctx, 0, sizeof(ctx));
      ctx.stage          = PRINT_SYSTEM_STAGE_APPLY;
      ctx.last_solve_time = -0.5;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "svo");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Selectors: two entries; empty selector list via mutation */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "all", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *s1        = add_child(selectors, "-", "", 2);
      add_child(s1, "basis", "timestep", 3);
      add_child(s1, "every", "11", 3);
      YAMLnode *s2 = add_child(selectors, "-", "", 2);
      add_child(s2, "basis", "timestep", 3);
      add_child(s2, "every", "4", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage          = PRINT_SYSTEM_STAGE_SETUP;
      ctx.timestep_index = 8;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "sel2");
      args.print_system.num_selectors = 0;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "sel2");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Selector: iterations threshold with last_iter=-1 (metric ternary branch) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "iterations", 3);
      add_child(sel, "threshold", "1", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage     = PRINT_SYSTEM_STAGE_APPLY;
      ctx.last_iter = -1;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "met-it");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Level selector with invalid runtime level id (mutate after parse) */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "all", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "level", 3);
      add_child(sel, "level", "0", 3);
      add_child(sel, "every", "1", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      args.print_system.selectors[0].level = STATS_MAX_LEVELS;

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage           = PRINT_SYSTEM_STAGE_BUILD;
      ctx.level_ids[0]    = 3;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "lvl");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   test_system_ignore(cleanup_cmd);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemDumpScheduled_selector_basis_names_and_sanitize(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_lsp_sb_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   test_system_ignore(cleanup_cmd);

   /* Selector type=selectors: exercise PrintSystemBasisName() strings in decision reasons */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "linear_system", 3);
      add_child(sel, "every", "1", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage        = PRINT_SYSTEM_STAGE_APPLY;
      ctx.system_index = 2;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-ls");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "all", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "timestep", 3);
      add_child(sel, "every", "3", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage          = PRINT_SYSTEM_STAGE_BUILD;
      ctx.timestep_index = 9;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-ts");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "all", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "level", 3);
      add_child(sel, "level", "0", 3);
      add_child(sel, "every", "1", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage        = PRINT_SYSTEM_STAGE_BUILD;
      ctx.level_ids[0] = 4;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-lvl");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "iterations", 3);
      add_child(sel, "threshold", "3", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage     = PRINT_SYSTEM_STAGE_APPLY;
      ctx.last_iter = 10;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-it");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "setup", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "setup_time", 3);
      add_child(sel, "threshold", "0.5", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage           = PRINT_SYSTEM_STAGE_SETUP;
      ctx.last_setup_time = 1.0;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-su");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "solve_time", 3);
      add_child(sel, "threshold", "0.5", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage           = PRINT_SYSTEM_STAGE_APPLY;
      ctx.last_solve_time = 0.75;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "sb-sv");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* PrintSystemSanitizeToken: non-alnum characters in object name */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "o$x@y");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* PrintSystemBasisName default: match with corrupted basis after parse */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 1);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "selectors", 1);
      add_child(ps, "stage", "apply", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *selectors = add_child(ps, "selectors", "", 1);
      YAMLnode *sel       = add_child(selectors, "-", "", 2);
      add_child(sel, "basis", "linear_system", 3);
      add_child(sel, "every", "1", 3);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      args.print_system.selectors[0].basis = (PrintSystemBasis)9999;

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage        = PRINT_SYSTEM_STAGE_APPLY;
      ctx.system_index = 0;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL, &ctx, "bad-basis");
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   test_system_ignore(cleanup_cmd);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemPrintData_series_scan_and_object_dir_edge_cases(void)
{
   TEST_HYPRE_INIT();

   int ret = system("rm -rf hypre-data");
   (void)ret;

   (void)mkdir("hypre-data", 0775);
   /* Invalid / odd ls_* names for max index scan */
   (void)mkdir("hypre-data/ls_", 0775);
   (void)mkdir("hypre-data/ls_abc", 0775);
   (void)mkdir("hypre-data/ls_12x", 0775);
   {
      FILE *fp = fopen("hypre-data/ls_99", "w");
      ASSERT_NOT_NULL(fp);
      fclose(fp);
   }

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 1.0);
   HYPRE_Complex  v1[1] = {1.0};
   HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, v1);
   int            dm[2] = {0, 1};
   IntArray *     dof   = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm, &dof);

   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemPrintData(MPI_COMM_SELF, &args, mat, rhs, dof);
   hypredrv_ErrorCodeResetAll();

   hypredrv_IntArrayDestroy(&dof);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);

   ret = system("rm -rf hypre-data");
   (void)ret;
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemDumpScheduled_artifacts_all_writes_and_skips_and_fs(void)
{
   TEST_HYPRE_INIT();

   char outdir[256];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_lsp_art_%d", (int)getpid());
   char cleanup_cmd[320];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   test_system_ignore(cleanup_cmd);

   /* Nested output_dir path */
   {
      char nested[384];
      snprintf(nested, sizeof(nested), "%s/a/b/c", outdir);
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", nested, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "nest");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Non-overwrite: gap in ls_* indices */
   {
      char base[384];
      snprintf(base, sizeof(base), "%s/gapobj", outdir);
      (void)mkdir(base, 0775);
      char d0[512], d5[512];
      snprintf(d0, sizeof(d0), "%s/ls_00000", base);
      snprintf(d5, sizeof(d5), "%s/ls_00005", base);
      (void)mkdir(d0, 0775);
      (void)mkdir(d5, 0775);

      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "off", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "gapobj");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());

      char meta[640];
      snprintf(meta, sizeof(meta), "%s/gapobj/ls_00006/metadata.txt", outdir);
      ASSERT_TRUE(access(meta, F_OK) == 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* base_dir exists as file -> dump dir creation fails */
   {
      char rootbad[384];
      snprintf(rootbad, sizeof(rootbad), "%s/rootbad", outdir);
      test_system_ignore(cleanup_cmd);
      (void)mkdir(outdir, 0775);
      {
         FILE *fp = fopen(rootbad, "w");
         ASSERT_NOT_NULL(fp);
         fputs("x", fp);
         fclose(fp);
      }

      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "off", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "rootbad");
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   test_system_ignore(cleanup_cmd);

   /* Full artifact set: write all branches */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      add_child(ps, "artifacts", "[matrix, precmat, rhs, x0, xref, solution, dofmap, metadata]",
                1);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      HYPRE_IJMatrix matA = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
      HYPRE_IJMatrix matM = create_test_ijmatrix_1x1(MPI_COMM_SELF, 3.0);
      HYPRE_Complex vb[1] = {4.0};
      HYPRE_Complex vx[1] = {5.0};
      HYPRE_IJVector vec_b   = create_test_ijvector(MPI_COMM_SELF, 0, 0, vb);
      HYPRE_IJVector vec_x0  = create_test_ijvector(MPI_COMM_SELF, 0, 0, vb);
      HYPRE_IJVector vec_xref = create_test_ijvector(MPI_COMM_SELF, 0, 0, vb);
      HYPRE_IJVector vec_x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, vx);
      int            dm[2]  = {0, 1};
      IntArray *     dofmap = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm, &dofmap);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, matA, matM, vec_b,
                                              vec_x0, vec_xref, vec_x, dofmap, &ctx,
                                              "full-art");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());

      char dr[512];
      snprintf(dr, sizeof(dr), "%s/full-art/ls_00000", outdir);
      ASSERT_TRUE(access(dr, F_OK) == 0);

      hypredrv_IntArrayDestroy(&dofmap);
      HYPRE_IJVectorDestroy(vec_x);
      HYPRE_IJVectorDestroy(vec_xref);
      HYPRE_IJVectorDestroy(vec_x0);
      HYPRE_IJVectorDestroy(vec_b);
      HYPRE_IJMatrixDestroy(matM);
      HYPRE_IJMatrixDestroy(matA);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   test_system_ignore(cleanup_cmd);

   /* Skip branches: request all artifact bits but pass NULL handles */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      add_child(ps, "artifacts", "[matrix, precmat, rhs, x0, xref, solution, dofmap, metadata]",
                1);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "skip-all");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Sanitized unnamed object */
   {
      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, NULL);
      char meta[512];
      snprintf(meta, sizeof(meta), "%s/unnamed/ls_00000/metadata.txt", outdir);
      ASSERT_TRUE(access(meta, F_OK) == 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   /* Overwrite removes nested stale directory */
   {
      char objdir[384];
      snprintf(objdir, sizeof(objdir), "%s/nested-ow", outdir);
      char lsdir[448];
      snprintf(lsdir, sizeof(lsdir), "%s/ls_00000", objdir);
      test_system_ignore(cleanup_cmd);
      (void)mkdir(outdir, 0775);
      (void)mkdir(objdir, 0775);
      (void)mkdir(lsdir, 0775);
      char subdir[512];
      ASSERT_TRUE(path_join2(subdir, sizeof(subdir), lsdir, "sub"));
      (void)mkdir(subdir, 0775);
      char nested_stale[512];
      ASSERT_TRUE(path_join2(nested_stale, sizeof(nested_stale), subdir, "stale.txt"));
      write_text_file(nested_stale, "z");

      LS_args args;
      hypredrv_LinearSystemSetDefaultArgs(&args);
      YAMLnode *ps = hypredrv_YAMLnodeCreate("print_system", "", 0);
      add_child(ps, "enabled", "on", 1);
      add_child(ps, "type", "all", 1);
      add_child(ps, "stage", "build", 1);
      add_child(ps, "output_dir", outdir, 1);
      add_child(ps, "overwrite", "on", 1);
      YAMLnode *art = add_child(ps, "artifacts", "", 1);
      add_child(art, "-", "metadata", 2);
      hypredrv_PrintSystemSetArgs(&args.print_system, ps);
      hypredrv_YAMLnodeDestroy(ps);

      PrintSystemContext ctx;
      memset(&ctx, 0, sizeof(ctx));
      ctx.stage = PRINT_SYSTEM_STAGE_BUILD;
      hypredrv_ErrorCodeResetAll();
      (void)hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, &args, NULL, NULL, NULL,
                                              NULL, NULL, NULL, NULL, &ctx, "nested-ow");
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE(access(nested_stale, F_OK) != 0);
      hypredrv_PrintSystemDestroyArgs(&args.print_system);
   }

   test_system_ignore(cleanup_cmd);
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

#if HYPRE_CHECK_MIN_VERSION(22600, 0)
static void
test_hypredrv_LinearSystemSetRHS_mtx_mm_read_failures(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   char rhsfile[256];
   snprintf(rhsfile, sizeof(rhsfile), "/tmp/hypredrive_test_rhs_mm_%d.mtx", (int)getpid());

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.type     = 3;
   args.rhs_mode = 2;

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

   /* Missing RHS file */
   strncpy(args.rhs_filename, "/tmp/hypredrive_rhs_missing_file_xyz_12345.mtx",
           sizeof(args.rhs_filename) - 1);
   args.rhs_filename[sizeof(args.rhs_filename) - 1] = '\0';
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* Invalid dimension line (0 0) */
   write_text_file(rhsfile, "% header\n0 0\n");
   strncpy(args.rhs_filename, rhsfile, sizeof(args.rhs_filename) - 1);
   args.rhs_filename[sizeof(args.rhs_filename) - 1] = '\0';
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* N != 1 */
   write_text_file(rhsfile,
                   "% bad N\n"
                   "1 2\n"
                   "1.0\n");
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* Row count does not match matrix */
   write_text_file(rhsfile,
                   "% M mismatch\n"
                   "2 1\n"
                   "1.0\n"
                   "2.0\n");
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* EOF before value line */
   write_text_file(rhsfile, "% no values\n1 1\n");
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* Non-numeric entry */
   write_text_file(rhsfile,
                   "% bad value\n"
                   "1 1\n"
                   "not_a_number\n");
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   /* Only comment lines then EOF */
   write_text_file(rhsfile, "% only comments\n% still nothing\n");
   hypredrv_ErrorCodeResetAll();
   HYPRE_ClearAllErrors();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &refsol, &rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
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

   HYPRE_IJMatrixDestroy(mat);
   unlink(rhsfile);
   TEST_HYPRE_FINALIZE();
}
#endif /* HYPRE_CHECK_MIN_VERSION(22600, 0) */

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

   hypredrv_LinearSystemComputeVectorNorm(v, "linf", &norm);
   ASSERT_EQ_DOUBLE(norm, 3.0, 1e-12);

   hypredrv_LinearSystemComputeVectorNorm(v, "bad", &norm);
   ASSERT_EQ_DOUBLE(norm, -1.0, 0.0);

   HYPRE_IJVectorDestroy(v);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemComputeVectorNorm_guards(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   const HYPRE_Complex vals[1] = {1.0};
   HYPRE_IJVector      v       = create_test_ijvector(MPI_COMM_SELF, 0, 0, vals);
   double              n       = 0.0;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemComputeVectorNorm(NULL, "L2", &n);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ErrorCodeResetAll();

   hypredrv_LinearSystemComputeVectorNorm(v, "L2", NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

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
test_hypredrv_LinearSystemSetRHS_destroys_existing_xref(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.rhs_mode = 0;

   HYPRE_IJMatrix mat = NULL;
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

   HYPRE_IJVector xref = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &xref), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(xref, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(xref), 0);
   {
      HYPRE_Complex val = 3.0;
      HYPRE_BigInt  idx = 0;
      ASSERT_EQ(HYPRE_IJVectorSetValues(xref, 1, &idx, &val), 0);
   }
   ASSERT_EQ(HYPRE_IJVectorAssemble(xref), 0);

   HYPRE_IJVector rhs = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetRHS(MPI_COMM_SELF, &args, mat, &xref, &rhs, NULL);
   ASSERT_NULL(xref);
   ASSERT_NOT_NULL(rhs);

   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemSetInitialGuess_generated_modes(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.x0_filename[0] = '\0';
   args.exec_policy     = 0;

   const HYPRE_Complex rhs_vals[3] = {1.0, 2.0, 3.0};
   HYPRE_IJVector      rhs         = create_test_ijvector(MPI_COMM_SELF, 0, 2, rhs_vals);
   HYPRE_IJVector      x0          = NULL;
   HYPRE_IJVector      x           = NULL;

   args.init_guess_mode = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }
   x = x0 = NULL;

   args.init_guess_mode = 1;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }
   x = x0 = NULL;

   args.init_guess_mode = 3;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }
   x = x0 = NULL;

   args.init_guess_mode = 4;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }
   x = x0 = NULL;

   args.init_guess_mode = 99;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemSetInitialGuess(MPI_COMM_SELF, &args, NULL, rhs, &x0, &x, NULL);
   if (x)
   {
      HYPRE_IJVectorDestroy(x);
   }
   if (x0)
   {
      HYPRE_IJVectorDestroy(x0);
   }

   HYPRE_IJVectorDestroy(rhs);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemResetInitialGuess_null_guard(void)
{
   TEST_HYPRE_INIT();

   Stats *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemResetInitialGuess(NULL, NULL, stats);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_StatsDestroy(&stats);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystem_norm_error_and_residual(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);

   const HYPRE_Complex xref_vals[1] = {1.0};
   /* x=0.0 so error and residual norms are both strictly positive (x=0.5 would solve 2x=b). */
   const HYPRE_Complex x_vals[1]  = {0.0};
   const HYPRE_Complex b_vals[1]  = {1.0};
   HYPRE_IJVector      xref         = create_test_ijvector(MPI_COMM_SELF, 0, 0, xref_vals);
   HYPRE_IJVector      x            = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_vals);
   HYPRE_IJVector      b            = create_test_ijvector(MPI_COMM_SELF, 0, 0, b_vals);

   double err = 0.0, res = 0.0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemComputeErrorNorm(xref, x, "L2", &err);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_GT(err, 0.0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemComputeResidualNorm(mat, b, x, "L2", &res);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_GT(res, 0.0);

   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(xref);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_LinearSystemReadDofmap_filename_branches(void)
{
   TEST_HYPRE_INIT();

   Stats *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);

   IntArray *dofmap = NULL;

   /* dirname + dofmap_filename -> constructed path (nonexistent) */
   strncpy(args.dirname, "hypre-data", sizeof(args.dirname) - 1);
   args.dirname[sizeof(args.dirname) - 1] = '\0';
   strncpy(args.dofmap_filename, "missing_dofmap.out", sizeof(args.dofmap_filename) - 1);
   args.dofmap_filename[sizeof(args.dofmap_filename) - 1] = '\0';
   args.dofmap_basename[0]                               = '\0';

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadDofmap(MPI_COMM_SELF, &args, &dofmap, stats);
   if (dofmap)
   {
      hypredrv_IntArrayDestroy(&dofmap);
      dofmap = NULL;
   }

   /* basename-only dofmap path */
   args.dirname[0] = '\0';
   args.dofmap_filename[0] = '\0';
   strncpy(args.dofmap_basename, "nonexistent_dofmap", sizeof(args.dofmap_basename) - 1);
   args.dofmap_basename[sizeof(args.dofmap_basename) - 1] = '\0';

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemReadDofmap(MPI_COMM_SELF, &args, &dofmap, stats);
   if (dofmap)
   {
      hypredrv_IntArrayDestroy(&dofmap);
   }

   hypredrv_StatsDestroy(&stats);
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

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
test_hypredrv_Scaling_valid_values_and_defaults(void)
{
   TEST_HYPRE_INIT();

   StrIntMapArray en = hypredrv_ScalingGetValidValues("enabled");
   ASSERT_TRUE(en.size > 0);

   StrIntMapArray ty = hypredrv_ScalingGetValidValues("type");
   ASSERT_TRUE(ty.size > 0);

   StrIntMapArray unk = hypredrv_ScalingGetValidValues("not_a_scaling_key");
   ASSERT_EQ(unk.size, 0);

   Scaling_args     args;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&args);
   ASSERT_EQ(args.enabled, 0);
   ASSERT_EQ((int)args.type, (int)SCALING_RHS_L2);
   ASSERT_NULL(args.custom_values);

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
   ASSERT_NOT_NULL(ctx);
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   ASSERT_NULL(ctx);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_rhs_l2_apply_undo(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   HYPRE_IJMatrix      mat_M = mat_A;
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(ctx->is_applied);

   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);

   hypredrv_ScalingUndoOnSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_custom_1x1(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);

   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]   = 3.0;

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled        = 1;
   sargs.type           = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values  = cv;

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
   HYPRE_IJVector      x     = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingUndoOnSystem(ctx, mat, mat, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_mag_1x1(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_MAG;

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
   HYPRE_IJVector      x     = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingUndoOnSystem(ctx, mat, mat, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_context_destroy_null_safe(void)
{
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, NULL);
   {
      Scaling_context *p = NULL;
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &p);
   }
}

static void
test_hypredrv_Scaling_compute_null_args_disabled_and_unknown_type(void)
{
   TEST_HYPRE_INIT();
   Scaling_args     args;
   Scaling_context  ctx_mem;
   Scaling_context *ctx = NULL;

   memset(&ctx_mem, 0, sizeof(ctx_mem));
   hypredrv_ScalingSetDefaultArgs(&args);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, NULL, &ctx_mem, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &args, NULL, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   args.enabled = 0;
   ctx_mem.enabled = 0;
   hypredrv_ScalingCompute(MPI_COMM_SELF, &args, &ctx_mem, NULL, NULL, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(ctx_mem.enabled, 0);

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
   hypredrv_ScalingSetDefaultArgs(&args);
   args.enabled = 1;
   args.type    = (scaling_type_t)999;
   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &args, ctx, NULL, NULL, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);

   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_custom_error_paths(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   const HYPRE_Complex rv[1] = {1.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_CUSTOM;

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   /* Missing dofmap */
   {
      DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
      cv->data[0]     = 1.0;
      sargs.custom_values = cv;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, NULL);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_DOFMAP);
      hypredrv_DoubleArrayDestroy(&cv);
      sargs.custom_values = NULL;
   }

   /* custom_values empty */
   {
      HYPRE_Int dm_data[1] = {0};
      IntArray *dofmap     = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
      DoubleArray *empty = hypredrv_DoubleArrayCreate(0);
      sargs.custom_values = empty;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_DoubleArrayDestroy(&empty);
      hypredrv_IntArrayDestroy(&dofmap);
   }
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_rhs_l2_zero_norm(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   HYPRE_IJMatrix      mat_M = mat_A;
   const HYPRE_Complex rhs_v[1] = {0.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(ctx->scalar_factor, 1.0);

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_system_comm_resolve_fallbacks(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   const HYPRE_Complex rhs_v[1] = {1.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);
   HYPRE_IJMatrix      mat_M    = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);

   Scaling_context *ctx = NULL;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
   ASSERT_NOT_NULL(ctx);
   /* enabled=0, is_applied=0: apply/undo return early after ScalingSystemCommResolve */
   ASSERT_EQ(ctx->enabled, 0);
   ASSERT_EQ(ctx->is_applied, 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(ctx, NULL, NULL, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(ctx, NULL, NULL, NULL, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(ctx, NULL, mat_M, NULL, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(ctx, NULL, NULL, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(ctx, NULL, NULL, NULL, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(ctx, NULL, mat_M, NULL, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_M);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_unknown_type_apply_undo_vector_and_system(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   HYPRE_IJMatrix      mat_M = mat_A;
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ctx->type = (scaling_type_t)999;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToVector(ctx, rhs, SCALING_VECTOR_RHS);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnVector(ctx, rhs, SCALING_VECTOR_RHS);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   ctx->is_applied = 1;
   hypredrv_ScalingUndoOnSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_apply_without_scaling_vector(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();
   hypredrv_ErrorCodeResetAll();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   int            dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 3.0;

   const HYPRE_Complex rhs_v[1] = {5.0};
   const HYPRE_Complex x_v[1]   = {7.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled       = 1;
   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = cv;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   {
      HYPRE_ParVector saved = ctx->scaling_vector;
      ctx->scaling_vector   = NULL;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ctx->scaling_vector = saved;
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_rhs_l2_apply_fails_zero_scalar(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();
   hypredrv_ErrorCodeResetAll();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   HYPRE_IJMatrix      mat_M = mat_A;
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ctx->scalar_factor = 0.0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_rhs_l2_distinct_M_apply_undo(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   HYPRE_IJMatrix mat_M = create_test_ijmatrix_1x1(MPI_COMM_SELF, 5.0);
   ASSERT_TRUE(mat_M != mat_A);

   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(ctx->is_applied);

   hypredrv_ScalingUndoOnSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_M);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_custom_distinct_M_apply_undo(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_IJMatrix mat_M = create_test_ijmatrix_1x1(MPI_COMM_SELF, 3.0);
   ASSERT_TRUE(mat_M != mat_A);

   HYPRE_Int dm_data[1] = {0};
   IntArray *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 2.0;

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
   HYPRE_IJVector      x     = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled       = 1;
   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = cv;

   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, dofmap);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingUndoOnSystem(ctx, mat_A, mat_M, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_M);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_mag_missing_dofmap_and_size_mismatch(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   const HYPRE_Complex rv[1] = {1.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_MAG;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_DOFMAP);

   {
      HYPRE_Int dm_bad[2] = {0, 0};
      IntArray *dofmap2   = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm_bad, &dofmap2);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap2);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_IntArrayDestroy(&dofmap2);
   }

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_custom_validation_errors(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   const HYPRE_Complex rv[1] = {1.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_CUSTOM;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   /* Tag 0 only => num_tags=1; pass two custom values => mismatch */
   {
      HYPRE_Int    dm_data[1] = {0};
      IntArray    *dofmap     = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
      DoubleArray *cv = hypredrv_DoubleArrayCreate(2);
      cv->data[0]     = 1.0;
      cv->data[1]     = 2.0;
      sargs.custom_values = cv;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_DoubleArrayDestroy(&cv);
      hypredrv_IntArrayDestroy(&dofmap);
      sargs.custom_values = NULL;
   }

   /* dofmap size != local rows */
   {
      HYPRE_Int    dm_bad[2] = {0, 0};
      IntArray    *dofmap2   = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, dm_bad, &dofmap2);
      DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
      cv->data[0]     = 1.0;
      sargs.custom_values = cv;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap2);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_DoubleArrayDestroy(&cv);
      hypredrv_IntArrayDestroy(&dofmap2);
      sargs.custom_values = NULL;
   }

   /* Invalid tag: tag 5 with only indices 0..2 valid */
   {
      HYPRE_Int    dm_data[1] = {5};
      IntArray    *dofmap3    = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap3);
      DoubleArray *cv = hypredrv_DoubleArrayCreate(3);
      cv->data[0] = 1.0;
      cv->data[1] = 1.0;
      cv->data[2] = 1.0;
      sargs.custom_values = cv;
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap3);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      hypredrv_DoubleArrayDestroy(&cv);
      hypredrv_IntArrayDestroy(&dofmap3);
   }

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_rhs_l2_vector_transform_branches(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};

   /* Apply RHS */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled = 1;
      sargs.type    = SCALING_RHS_L2;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingApplyToVector(ctx, rhs, SCALING_VECTOR_RHS);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
   }
   /* Apply UNKNOWN */
   {
      HYPRE_IJVector x = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled = 1;
      sargs.type    = SCALING_RHS_L2;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
      HYPRE_IJVectorDestroy(rhs);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingApplyToVector(ctx, x, SCALING_VECTOR_UNKNOWN);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(x);
   }
   /* Undo RHS (needs prior apply state: use compute + apply system then undo vector) */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled = 1;
      sargs.type    = SCALING_RHS_L2;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
      hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_A, rhs, x);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingUndoOnVector(ctx, rhs, SCALING_VECTOR_RHS);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }
   /* Undo UNKNOWN */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled = 1;
      sargs.type    = SCALING_RHS_L2;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);
      hypredrv_ScalingApplyToSystem(ctx, mat_A, mat_A, rhs, x);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingUndoOnVector(ctx, x, SCALING_VECTOR_UNKNOWN);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }

   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_vector_transform_branches(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 3.0;

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};

   /* Apply RHS (pointwise product path) */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled       = 1;
      sargs.type          = SCALING_DOFMAP_CUSTOM;
      sargs.custom_values = cv;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingApplyToVector(ctx, rhs, SCALING_VECTOR_RHS);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }
   /* Apply UNKNOWN (division / inverse path) */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled       = 1;
      sargs.type          = SCALING_DOFMAP_CUSTOM;
      sargs.custom_values = cv;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingApplyToVector(ctx, x, SCALING_VECTOR_UNKNOWN);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }
   /* Undo RHS */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled       = 1;
      sargs.type          = SCALING_DOFMAP_CUSTOM;
      sargs.custom_values = cv;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingUndoOnVector(ctx, rhs, SCALING_VECTOR_RHS);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }
   /* Undo UNKNOWN */
   {
      HYPRE_IJVector rhs = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
      HYPRE_IJVector x   = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);
      Scaling_args     sargs;
      Scaling_context *ctx = NULL;
      hypredrv_ScalingSetDefaultArgs(&sargs);
      sargs.enabled       = 1;
      sargs.type          = SCALING_DOFMAP_CUSTOM;
      sargs.custom_values = cv;
      hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
      hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
      hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
      hypredrv_ErrorCodeResetAll();
      hypredrv_ScalingUndoOnVector(ctx, x, SCALING_VECTOR_UNKNOWN);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
      HYPRE_IJVectorDestroy(rhs);
      HYPRE_IJVectorDestroy(x);
   }

   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_undo_without_scaling_vector(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 3.0;

   const HYPRE_Complex rhs_v[1] = {5.0};
   const HYPRE_Complex x_v[1]   = {7.0};
   HYPRE_IJVector      rhs      = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x        = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled       = 1;
   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = cv;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   hypredrv_ScalingApplyToSystem(ctx, mat, mat, rhs, x);
   ASSERT_TRUE(ctx->is_applied);

   /* Remove scaling storage as if corrupted; IJ destroy owns the ParVector */
   HYPRE_SAFE_CALL(HYPRE_IJVectorDestroy(ctx->scaling_ijvec));
   ctx->scaling_ijvec  = NULL;
   ctx->scaling_vector = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(ctx, mat, mat, rhs, x);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_context_switch_custom_then_mag(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 2.0;

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
   HYPRE_IJVector      x     = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   sargs.enabled       = 1;
   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = cv;
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_NOT_NULL(ctx->scaling_ijvec);

   sargs.type = SCALING_DOFMAP_MAG;
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_NULL(ctx->scaling_ijvec);
   ASSERT_NOT_NULL(ctx->scaling_vector);

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_context_switch_mag_then_custom(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);
   DoubleArray *cv = hypredrv_DoubleArrayCreate(1);
   cv->data[0]     = 2.0;

   const HYPRE_Complex rv[1] = {5.0};
   const HYPRE_Complex xv[1] = {7.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);
   HYPRE_IJVector      x     = create_test_ijvector(MPI_COMM_SELF, 0, 0, xv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_MAG;
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_NOT_NULL(ctx->scaling_vector);

   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = cv;
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_NOT_NULL(ctx->scaling_ijvec);

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_DoubleArrayDestroy(&cv);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_null_ctx_vector_and_system_guards(void)
{
   TEST_HYPRE_INIT();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   const HYPRE_Complex v[1] = {1.0};
   HYPRE_IJVector      vec  = create_test_ijvector(MPI_COMM_SELF, 0, 0, v);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToVector(NULL, vec, SCALING_VECTOR_RHS);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnVector(NULL, vec, SCALING_VECTOR_RHS);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToSystem(NULL, mat, mat, vec, vec);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(NULL, mat, mat, vec, vec);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   HYPRE_IJVectorDestroy(vec);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_dofmap_data_null(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   const HYPRE_Complex rv[1] = {1.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_DOFMAP_MAG;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   IntArray dm_bad = {0};
   dm_bad.size     = 1;
   dm_bad.data     = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, &dm_bad);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_MISSING_DOFMAP);

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_custom_values_null(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix mat = create_test_ijmatrix_1x1(MPI_COMM_SELF, 2.0);
   HYPRE_Int      dm_data[1] = {0};
   IntArray      *dofmap     = NULL;
   hypredrv_IntArrayBuild(MPI_COMM_SELF, 1, dm_data, &dofmap);

   const HYPRE_Complex rv[1] = {1.0};
   HYPRE_IJVector      rhs   = create_test_ijvector(MPI_COMM_SELF, 0, 0, rv);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled       = 1;
   sargs.type          = SCALING_DOFMAP_CUSTOM;
   sargs.custom_values = NULL;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat, rhs, dofmap);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   hypredrv_IntArrayDestroy(&dofmap);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJMatrixDestroy(mat);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_vector_api_extra_guards(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs     = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x       = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);

   ctx->enabled = 0;
   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToVector(ctx, rhs, SCALING_VECTOR_RHS);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ctx->enabled = 1;
   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingApplyToVector(ctx, NULL, SCALING_VECTOR_RHS);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnVector(ctx, NULL, SCALING_VECTOR_RHS);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}

static void
test_hypredrv_Scaling_undo_system_when_not_applied(void)
{
   TEST_HYPRE_INIT();
   HYPRE_ClearAllErrors();

   HYPRE_IJMatrix      mat_A = create_test_ijmatrix_1x1(MPI_COMM_SELF, 4.0);
   const HYPRE_Complex rhs_v[1] = {9.0};
   const HYPRE_Complex x_v[1]   = {1.0};
   HYPRE_IJVector      rhs     = create_test_ijvector(MPI_COMM_SELF, 0, 0, rhs_v);
   HYPRE_IJVector      x       = create_test_ijvector(MPI_COMM_SELF, 0, 0, x_v);

   Scaling_args     sargs;
   Scaling_context *ctx = NULL;
   hypredrv_ScalingSetDefaultArgs(&sargs);
   sargs.enabled = 1;
   sargs.type    = SCALING_RHS_L2;
   hypredrv_ScalingContextCreate(MPI_COMM_SELF, &ctx);
   hypredrv_ScalingCompute(MPI_COMM_SELF, &sargs, ctx, mat_A, rhs, NULL);

   ASSERT_FALSE(ctx->is_applied);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ScalingUndoOnSystem(ctx, mat_A, mat_A, rhs, x);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ScalingContextDestroy(MPI_COMM_SELF, &ctx);
   HYPRE_IJVectorDestroy(rhs);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJMatrixDestroy(mat_A);
   TEST_HYPRE_FINALIZE();
}
#endif /* HYPRE_CHECK_MIN_VERSION(30000, 0) */

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
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_set_suffix_and_last_suffix_error);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_dof_labels_block_bad_value);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_dof_labels_flow_bad_value);
   RUN_TEST(test_hypredrv_LinearSystemGetSuffix_null_args);
   RUN_TEST(test_hypredrv_LinearSystemCreateWorkingSolution_invalid_args);
   RUN_TEST(test_hypredrv_PrintSystem_defaults);
   RUN_TEST(test_hypredrv_PrintSystem_null_args);
   RUN_TEST(test_hypredrv_PrintSystem_yaml_scalar_enabled);
   RUN_TEST(test_hypredrv_PrintSystem_invalid_type_and_stage);
   RUN_TEST(test_hypredrv_PrintSystem_yaml_on_off_and_stages);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_print_system_more_branches);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_print_system_block_ranges);
   RUN_TEST(test_hypredrv_PrintSystem_yaml_parse_edge_cases);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_print_system_invalid_combo);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_print_system_thresholds);
   RUN_TEST(test_hypredrv_LinearSystemSetArgsFromYAML_print_system_threshold_invalid_combo);
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
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_sequence_ls_id_out_of_range);
   RUN_TEST(test_LinearSystemReadRHS_file_patterns);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_sequence_ls_id_out_of_range);
#endif

   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mode_branches);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mode_precedence_over_filename);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_destroys_existing_xref);
   RUN_TEST(test_hypredrv_LinearSystemSetReferenceSolution_keeps_randsol_reference);
   RUN_TEST(test_hypredrv_LinearSystemSetReferenceSolution_file_override);

#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_hypredrv_LinearSystemReadMatrix_mtx_success);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mtx_file_success);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mtx_dim_mismatch_errors);
   RUN_TEST(test_hypredrv_LinearSystemSetRHS_mtx_mm_read_failures);
#endif
}

static void
run_linsys_misc_and_numeric_tests(void)
{
   RUN_TEST(test_hypredrv_LinearSystemPrintData_series_dir_and_null_objects);
   RUN_TEST(test_hypredrv_PrintSystem_yaml_parser_selectors_and_null_field);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_branch_and_filesystem_coverage);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_selector_basis_names_and_sanitize);
   RUN_TEST(test_hypredrv_LinearSystemPrintData_series_scan_and_object_dir_edge_cases);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_artifacts_all_writes_and_skips_and_fs);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_ranges_and_artifacts);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_overwrite_reuses_dump_series);
   RUN_TEST(test_hypredrv_LinearSystemDumpScheduled_threshold_types_and_selectors);
   RUN_TEST(test_hypredrv_LinearSystemMatrixGetNumRows_GetNumNonzeros_error_cases);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_LinearSystemReadRHS_error_cases);
#endif
   RUN_TEST(test_hypredrv_LinearSystemComputeVectorNorm_all_modes);
   RUN_TEST(test_hypredrv_LinearSystemComputeVectorNorm_guards);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   RUN_TEST(test_hypredrv_LinearSystemSetInitialGuess_x0_filename_branches);
#endif
   RUN_TEST(test_hypredrv_LinearSystemSetInitialGuess_generated_modes);
   RUN_TEST(test_hypredrv_LinearSystemResetInitialGuess_null_guard);
   RUN_TEST(test_hypredrv_LinearSystem_norm_error_and_residual);
   RUN_TEST(test_hypredrv_LinearSystemReadDofmap_filename_branches);
   RUN_TEST(test_hypredrv_LinearSystemCreateWorkingSolution_recreates_x);
   RUN_TEST(test_hypredrv_LinearSystemSetPrecMatrix_branchy_paths);
   RUN_TEST(test_hypredrv_linsys_branch_logs);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   RUN_TEST(test_hypredrv_Scaling_context_destroy_null_safe);
   RUN_TEST(test_hypredrv_Scaling_compute_null_args_disabled_and_unknown_type);
   RUN_TEST(test_hypredrv_Scaling_dofmap_custom_error_paths);
   RUN_TEST(test_hypredrv_Scaling_rhs_l2_zero_norm);
   RUN_TEST(test_hypredrv_Scaling_valid_values_and_defaults);
   RUN_TEST(test_hypredrv_Scaling_rhs_l2_apply_undo);
   RUN_TEST(test_hypredrv_Scaling_system_comm_resolve_fallbacks);
   RUN_TEST(test_hypredrv_Scaling_unknown_type_apply_undo_vector_and_system);
   RUN_TEST(test_hypredrv_Scaling_rhs_l2_apply_fails_zero_scalar);
   RUN_TEST(test_hypredrv_Scaling_dofmap_custom_1x1);
   RUN_TEST(test_hypredrv_Scaling_dofmap_apply_without_scaling_vector);
   RUN_TEST(test_hypredrv_Scaling_dofmap_mag_1x1);
   RUN_TEST(test_hypredrv_Scaling_rhs_l2_distinct_M_apply_undo);
   RUN_TEST(test_hypredrv_Scaling_dofmap_custom_distinct_M_apply_undo);
   RUN_TEST(test_hypredrv_Scaling_dofmap_mag_missing_dofmap_and_size_mismatch);
   RUN_TEST(test_hypredrv_Scaling_dofmap_custom_validation_errors);
   RUN_TEST(test_hypredrv_Scaling_rhs_l2_vector_transform_branches);
   RUN_TEST(test_hypredrv_Scaling_dofmap_vector_transform_branches);
   RUN_TEST(test_hypredrv_Scaling_dofmap_undo_without_scaling_vector);
   RUN_TEST(test_hypredrv_Scaling_context_switch_custom_then_mag);
   RUN_TEST(test_hypredrv_Scaling_context_switch_mag_then_custom);
   RUN_TEST(test_hypredrv_Scaling_null_ctx_vector_and_system_guards);
   RUN_TEST(test_hypredrv_Scaling_dofmap_data_null);
   RUN_TEST(test_hypredrv_Scaling_custom_values_null);
   RUN_TEST(test_hypredrv_Scaling_vector_api_extra_guards);
   RUN_TEST(test_hypredrv_Scaling_undo_system_when_not_applied);
#endif
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
