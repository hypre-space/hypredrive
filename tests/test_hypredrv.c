#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPRE_utilities.h"
#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "object.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "internal/stats.h"
#include "logging.h"
#include "test_helpers.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

extern uint32_t HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t obj,
                                                         int        num_local_blocks,
                                                         int        num_dof_types);

static void
reset_state(void)
{
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
   cleanup_temp_files();

   /* Ensure we start each test with clean global error/message and hypre error state.
    * Note: hypre error flags can be sticky across calls; some code paths inspect
    * HYPRE_GetError() and will misdiagnose unrelated operations otherwise. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   HYPREDRV_Finalize();
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
}

static bool
setup_ps3d10pt7_paths(char matrix_path[PATH_MAX], char rhs_path[PATH_MAX])
{
   char matrix_check[PATH_MAX];
   char rhs_check[PATH_MAX];

   snprintf(matrix_path, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.A", HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.b", HYPREDRIVE_SOURCE_DIR);
   snprintf(matrix_check, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.A.00000",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_check, PATH_MAX, "%s/data/ps3d10pt7/np1/IJ.out.b.00000", HYPREDRIVE_SOURCE_DIR);

   if (access(matrix_check, F_OK) != 0 || access(rhs_check, F_OK) != 0)
   {
      fprintf(stderr, "SKIP: missing data files: %s or %s\n", matrix_path, rhs_path);
      return false;
   }

   return true;
}

static bool
setup_poromech2k_dir(char ls_dir[PATH_MAX])
{
   char matrix0[2 * PATH_MAX];
   char rhs0[2 * PATH_MAX];
   char dofmap0[2 * PATH_MAX];
   char matrix1[2 * PATH_MAX];
   char rhs1[2 * PATH_MAX];
   char dofmap1[2 * PATH_MAX];

   snprintf(ls_dir, PATH_MAX, "%s/data/poromech2k/np1/ls", HYPREDRIVE_SOURCE_DIR);
   snprintf(matrix0, sizeof(matrix0), "%s_00000/IJ.out.A.00000.bin", ls_dir);
   snprintf(rhs0, sizeof(rhs0), "%s_00000/IJ.out.b.00000.bin", ls_dir);
   snprintf(dofmap0, sizeof(dofmap0), "%s_00000/dofmap.out.00000", ls_dir);
   snprintf(matrix1, sizeof(matrix1), "%s_00001/IJ.out.A.00000.bin", ls_dir);
   snprintf(rhs1, sizeof(rhs1), "%s_00001/IJ.out.b.00000.bin", ls_dir);
   snprintf(dofmap1, sizeof(dofmap1), "%s_00001/dofmap.out.00000", ls_dir);

   if (access(matrix0, F_OK) != 0 || access(rhs0, F_OK) != 0 || access(dofmap0, F_OK) != 0 ||
       access(matrix1, F_OK) != 0 || access(rhs1, F_OK) != 0 || access(dofmap1, F_OK) != 0)
   {
      fprintf(stderr,
              "SKIP: missing poromech2k timestep files under %s_00000 or %s_00001\n",
              ls_dir, ls_dir);
      return false;
   }

   return true;
}

static HYPREDRV_t
create_initialized_obj(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);
   return obj;
}

static void
parse_yaml_into_obj(HYPREDRV_t obj, char *yaml_config)
{
   char *argv[] = {yaml_config};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
}

static void
parse_yaml_file_into_obj(HYPREDRV_t obj, const char *yaml_config, const char *tmp_name)
{
   char *tmp_yaml = strdup(tmp_name);
   ASSERT_NOT_NULL(tmp_yaml);

   FILE *fp = fopen(tmp_yaml, "w");
   ASSERT_NOT_NULL(fp);
   ASSERT_TRUE(fputs(yaml_config, fp) >= 0);
   fclose(fp);

   char *argv[] = {tmp_yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   remove(tmp_yaml);
   free(tmp_yaml);
}

static void
parse_minimal_library_yaml(HYPREDRV_t obj)
{
   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);
}

static void
parse_library_reuse_yaml(HYPREDRV_t obj, const char *reuse_block)
{
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[1024];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "%s"
            "  amg:\n"
            "    print_level: 0\n",
            reuse_block ? reuse_block : "");
   parse_yaml_into_obj(obj, yaml_config);
}

static void
write_fnv1a64_seeded(const void *data, size_t nbytes, uint64_t *hash)
{
   const unsigned char *bytes = (const unsigned char *)data;
   ASSERT_NOT_NULL(hash);

   for (size_t i = 0; i < nbytes; i++)
   {
      *hash ^= (uint64_t)bytes[i];
      *hash *= UINT64_C(1099511628211);
   }
}

static void
write_test_lsseq_with_timesteps(const char *filename)
{
   const char          payload[] = "source=test\nkind=print_system\n";
   LSSeqHeader         header;
   LSSeqInfoHeader     info;
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
   uint64_t            part_blob_table[LSSEQ_PART_BLOB_ENTRIES] = {0};
   FILE               *fp = fopen(filename, "wb");

   ASSERT_NOT_NULL(fp);

   memset(&header, 0, sizeof(header));
   memset(&info, 0, sizeof(info));
   memset(part_meta, 0, sizeof(part_meta));
   memset(pattern_meta, 0, sizeof(pattern_meta));
   memset(sys_meta, 0, sizeof(sys_meta));
   memset(timesteps, 0, sizeof(timesteps));

   header.magic         = LSSEQ_MAGIC;
   header.version       = LSSEQ_VERSION;
   header.flags         = LSSEQ_FLAG_HAS_INFO | LSSEQ_FLAG_HAS_DOFMAP |
                  LSSEQ_FLAG_HAS_TIMESTEPS;
   header.codec         = (uint32_t)COMP_NONE;
   header.num_systems   = 2;
   header.num_parts     = 1;
   header.num_patterns  = 2;
   header.num_timesteps = 2;

   info.magic               = LSSEQ_INFO_MAGIC;
   info.version             = LSSEQ_INFO_VERSION;
   info.flags               = LSSEQ_INFO_FLAG_PAYLOAD_KV;
   info.endian_tag          = UINT32_C(0x01020304);
   info.payload_size        = (uint64_t)(sizeof(payload) - 1u);
   info.payload_hash_fnv1a64 = UINT64_C(1469598103934665603);
   write_fnv1a64_seeded(payload, sizeof(payload) - 1u, &info.payload_hash_fnv1a64);
   info.blob_hash_fnv1a64 = 0;
   info.blob_bytes        = 0;

   header.offset_part_meta =
      sizeof(LSSeqHeader) + sizeof(LSSeqInfoHeader) + (uint64_t)(sizeof(payload) - 1u);
   header.offset_pattern_meta  = header.offset_part_meta + sizeof(part_meta);
   header.offset_sys_part_meta = header.offset_pattern_meta + sizeof(pattern_meta);
   header.offset_part_blob_table =
      header.offset_sys_part_meta + sizeof(sys_meta);
   header.offset_timestep_meta =
      header.offset_part_blob_table + sizeof(part_blob_table);
   header.offset_blob_data = header.offset_timestep_meta + sizeof(timesteps);

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

   timesteps[0].timestep = 0;
   timesteps[0].ls_start = 0;
   timesteps[1].timestep = 1;
   timesteps[1].ls_start = 1;

   ASSERT_EQ_SIZE(fwrite(&header, sizeof(header), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(&info, sizeof(info), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(payload, sizeof(payload) - 1u, 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_meta, sizeof(part_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(pattern_meta, sizeof(pattern_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(sys_meta, sizeof(sys_meta), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(part_blob_table, sizeof(part_blob_table), 1, fp), 1);
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
attach_library_scalar_system(HYPREDRV_t obj, HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b)
{
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
}

static void
run_library_linear_solve(HYPREDRV_t obj, const char *newton_name)
{
   if (newton_name)
   {
      ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 1, newton_name, -1), ERROR_NONE);
   }

   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   if (newton_name)
   {
      ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 1, newton_name, -1), ERROR_NONE);
   }
}

static size_t
count_substr(const char *haystack, const char *needle)
{
   size_t      count = 0;
   const char *ptr   = haystack;
   size_t      len   = strlen(needle);

   while ((ptr = strstr(ptr, needle)) != NULL)
   {
      count++;
      ptr += len;
   }

   return count;
}

static bool
find_prefixed_entry(const char *dir_path, const char *prefix, char *out_name,
                    size_t out_size)
{
   DIR *dir = opendir(dir_path);
   if (!dir)
   {
      return false;
   }

   bool                found = false;
   struct dirent *entry = NULL;
   while ((entry = readdir(dir)) != NULL)
   {
      if (!strncmp(entry->d_name, prefix, strlen(prefix)))
      {
         snprintf(out_name, out_size, "%s", entry->d_name);
         found = true;
         break;
      }
   }

   closedir(dir);
   return found;
}

static bool
file_contains_substr(const char *path, const char *needle)
{
   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      return false;
   }

   bool found = false;
   char line[1024];
   while (fgets(line, sizeof(line), fp) != NULL)
   {
      if (strstr(line, needle))
      {
         found = true;
         break;
      }
   }

   fclose(fp);
   return found;
}

static bool
path_join2(char *out, size_t out_size, const char *left, const char *right)
{
   if (!out || out_size == 0 || !left || !right)
   {
      return false;
   }

   size_t left_len  = strlen(left);
   size_t right_len = strlen(right);
   bool   add_sep   = (left_len > 0 && left[left_len - 1] != '/');
   size_t total_len = left_len + (size_t)add_sep + right_len + 1;
   if (total_len > out_size)
   {
      return false;
   }

   memcpy(out, left, left_len);
   size_t pos = left_len;
   if (add_sep)
   {
      out[pos++] = '/';
   }
   memcpy(out + pos, right, right_len);
   out[pos + right_len] = '\0';
   return true;
}

static void
capture_eigspec_warning_output(HYPREDRV_t obj, int num_calls, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();

#ifdef __APPLE__
   if (!tmp)
   {
      char path[] = "/tmp/hypredrv_test_eigspec.txt";
      int  fd     = mkstemp(path);
      ASSERT_TRUE(fd != -1);
      tmp = fdopen(fd, "w+");
      unlink(path);
   }
#endif

   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   for (int i = 0; i < num_calls; i++)
   {
      ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
   }
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

typedef void (*CapturedStreamFn)(void *);

static void
capture_stream_output(FILE *stream, CapturedStreamFn fn, void *context, char *buffer,
                      size_t buf_len)
{
   FILE *tmp = tmpfile();

#ifdef __APPLE__
   if (!tmp)
   {
      char path[] = "/tmp/hypredrv_test_XXXXXX";
      int  fd     = mkstemp(path);
      ASSERT_TRUE(fd != -1);
      tmp = fdopen(fd, "w+");
      unlink(path);
   }
#endif

   ASSERT_NOT_NULL(tmp);

   int tmp_fd   = fileno(tmp);
   int saved_fd = dup(fileno(stream));
   ASSERT_TRUE(saved_fd != -1);

   fflush(stream);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stream)) != -1);

   fn(context);
   fflush(stream);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_fd, fileno(stream)) != -1);
   close(saved_fd);
   fclose(tmp);
}

static void
capture_stdout_output(CapturedStreamFn fn, void *context, char *buffer, size_t buf_len)
{
   capture_stream_output(stdout, fn, context, buffer, buf_len);
}

static void
capture_stderr_output(CapturedStreamFn fn, void *context, char *buffer, size_t buf_len)
{
   capture_stream_output(stderr, fn, context, buffer, buf_len);
}

static HYPRE_IJMatrix
create_test_ijmatrix_1x1(double diag)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &mat), 0);
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

static HYPRE_IJVector
create_test_ijvector_1x1(double value)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[1] = {0};
   double       val[1] = {value};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 1, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
}

static HYPRE_IJMatrix
create_test_ijmatrix_2x2(double a00, double a01, double a10, double a11)
{
   HYPRE_IJMatrix mat = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 1, 0, 1, &mat), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize(mat), 0);
   HYPRE_Int    nrows     = 2;
   HYPRE_Int    ncols[2]  = {2, 2};
   HYPRE_BigInt rows[2]   = {0, 1};
   HYPRE_BigInt cols[4]   = {0, 1, 0, 1};
   double       values[4] = {a00, a01, a10, a11};
   ASSERT_EQ(HYPRE_IJMatrixSetValues(mat, nrows, ncols, rows, cols, values), 0);
   ASSERT_EQ(HYPRE_IJMatrixAssemble(mat), 0);
   return mat;
}

static HYPRE_IJVector
create_test_ijvector_2x1(double v0, double v1)
{
   HYPRE_IJVector vec = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 1, &vec), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize(vec), 0);
   HYPRE_BigInt idx[2] = {0, 1};
   double       val[2] = {v0, v1};
   ASSERT_EQ(HYPRE_IJVectorSetValues(vec, 2, idx, val), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vec), 0);
   return vec;
}

#define ASSERT_HAS_FLAG(code, flag) ASSERT_TRUE(((code) & (flag)) != 0)

static void
test_HYPREDRV_all_api_init_guard(void)
{
   reset_state();

   HYPREDRV_t obj = NULL;
   uint32_t   code;

   /* Core lifecycle */
   code = HYPREDRV_Create(MPI_COMM_SELF, &obj);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(obj);

   code = HYPREDRV_Destroy(&obj);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Non-object utilities */
   code = HYPREDRV_PrintLibInfo(MPI_COMM_SELF, 1);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PrintSystemInfo(MPI_COMM_SELF);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PrintExitInfo(MPI_COMM_SELF, "hypredrive");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Args parsing */
   code = HYPREDRV_InputArgsParse(0, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Input getters */
   code = HYPREDRV_InputArgsGetWarmup(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_ObjectSetName(NULL, "guard");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_InputArgsGetNumRepetitions(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_InputArgsGetNumLinearSystems(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Linear system APIs (object methods) */
   code = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemReadMatrix(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetMatrix(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetRHS(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetInitialGuess(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   HYPRE_Complex *ptr = NULL;
   code              = HYPREDRV_LinearSystemGetSolutionValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemGetRHSValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   HYPRE_Matrix mat_out = NULL;
   code = HYPREDRV_LinearSystemGetMatrix(NULL, &mat_out);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   double norm = 0.0;
   code        = HYPREDRV_LinearSystemGetSolutionNorm(NULL, "l2", &norm);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   code = HYPREDRV_LinearSystemSetPrecMatrix(NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetDofmap(NULL, 0, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetInterleavedDofmap(NULL, 0, 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemSetContiguousDofmap(NULL, 0, 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemReadDofmap(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemPrintDofmap(NULL, "dofmap.txt");
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSystemPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Solver / precon APIs */
   code = HYPREDRV_PreconCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconApply(NULL, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_PreconDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_LinearSolverDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);

   /* Stats/annotation APIs */
   code = HYPREDRV_StatsPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateBegin(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateEnd(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateLevelBegin(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_AnnotateLevelEnd(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelGetCount(NULL, 0, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelGetEntry(NULL, 0, 0, NULL, NULL, NULL, NULL, NULL);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
   code = HYPREDRV_StatsLevelPrint(NULL, 0);
   ASSERT_HAS_FLAG(code, ERROR_HYPREDRV_NOT_INITIALIZED);
}

static void
test_HYPREDRV_all_api_obj_guard(void)
{
   reset_state();
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   uint32_t code;

   /* Object methods should trip the OBJ guard when hypredrv == NULL */
   code = HYPREDRV_SetLibraryMode(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   HYPRE_Complex *ptr = NULL;
   code              = HYPREDRV_LinearSystemGetSolutionValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSystemGetRHSValues(NULL, &ptr);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   HYPRE_Matrix mat_null = NULL;
   code = HYPREDRV_LinearSystemGetMatrix(NULL, &mat_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   int typed_iter_null = -1;
   double typed_t_null = -1.0;
   code = HYPREDRV_LinearSolverGetNumIter(NULL, &typed_iter_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverGetSetupTime(NULL, &typed_t_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverGetSolveTime(NULL, &typed_t_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StateVectorApplyCorrection(NULL, 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   double norm = 0.0;
   code        = HYPREDRV_LinearSystemGetSolutionNorm(NULL, "l2", &norm);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_LinearSystemPrintDofmap(NULL, "x");
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_PreconCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverCreate(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_PreconSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverSetup(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_PreconDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_LinearSolverDestroy(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   code = HYPREDRV_StatsPrint(NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateBegin(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateEnd(NULL, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateLevelBegin(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_AnnotateLevelEnd(NULL, 0, "x", 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelGetCount(NULL, 0, &typed_iter_null);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelGetEntry(NULL, 0, 0, &typed_iter_null, NULL, NULL, NULL,
                                        NULL);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);
   code = HYPREDRV_StatsLevelPrint(NULL, 0);
   ASSERT_HAS_FLAG(code, ERROR_UNKNOWN_HYPREDRV_OBJ);

   /* Reset the global error state before finalizing */
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_requires_initialization_guard(void)
{
   reset_state();

   HYPREDRV_t obj = NULL;
   uint32_t   code;

   code = HYPREDRV_Create(MPI_COMM_SELF, &obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(obj);
   hypredrv_ErrorCodeResetAll();

   code = HYPREDRV_SetLibraryMode(obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   hypredrv_ErrorCodeResetAll();
}

static void
test_initialize_and_finalize_idempotent(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE); /* second call should be no-op */
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE); /* already finalized */
}

static void
test_HYPREDRV_Create_null_output_pointer(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_HAS_FLAG(HYPREDRV_Create(MPI_COMM_SELF, NULL), ERROR_UNKNOWN);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_default_object_names_are_sequential(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj1 = NULL;
   HYPREDRV_t obj2 = NULL;
   HYPREDRV_t obj3 = NULL;

   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj2), ERROR_NONE);
   ASSERT_NOT_NULL(obj1);
   ASSERT_NOT_NULL(obj2);

   struct hypredrv_struct *state1 = (struct hypredrv_struct *)obj1;
   struct hypredrv_struct *state2 = (struct hypredrv_struct *)obj2;
   ASSERT_TRUE(state1->runtime_object_id == 1);
   ASSERT_TRUE(state2->runtime_object_id == 2);
   ASSERT_TRUE(state1->stats->object_name[0] == '\0');
   ASSERT_TRUE(state2->stats->object_name[0] == '\0');

   ASSERT_EQ(HYPREDRV_Destroy(&obj1), ERROR_NONE);
   ASSERT_NULL(obj1);

   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj3), ERROR_NONE);
   ASSERT_NOT_NULL(obj3);
   struct hypredrv_struct *state3 = (struct hypredrv_struct *)obj3;
   ASSERT_TRUE(state3->runtime_object_id == 3);
   ASSERT_TRUE(state3->stats->object_name[0] == '\0');

   ASSERT_EQ(HYPREDRV_Destroy(&obj2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj3), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_default_object_name_persists_without_user_name(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->stats->object_name[0] == '\0');

   parse_minimal_library_yaml(obj);
   ASSERT_TRUE(state->stats->object_name[0] == '\0');

   char yaml_named[] =
      "general:\n"
      "  name: user-name\n"
      "  statistics: off\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_named);
   ASSERT_TRUE(strcmp(state->stats->object_name, "user-name") == 0);

   ASSERT_EQ(HYPREDRV_ObjectSetName(obj, NULL), ERROR_NONE);
   ASSERT_TRUE(state->stats->object_name[0] == '\0');

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_Finalize_auto_destroys_live_objects(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj1 = NULL;
   HYPREDRV_t obj2 = NULL;
   HYPREDRV_t obj3 = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj2), ERROR_NONE);
   ASSERT_NOT_NULL(obj1);
   ASSERT_NOT_NULL(obj2);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj3), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj3), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
run_minimal_lifecycle_for_trace_capture(void *context)
{
   (void)context;
   HYPREDRV_t obj = NULL;

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_default_off(void)
{
   reset_state();
   unsetenv("HYPREDRV_LOG_LEVEL");

   char output[8192];
   capture_stderr_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NULL(strstr(output, "[HYPREDRV][L"));
}

static void
test_HYPREDRV_log_level_enabled_emits_trace(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);

   char output[16384];
   capture_stderr_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L1]"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_Create begin"));
   ASSERT_NOT_NULL(strstr(output, "registered object"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_Destroy end"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_numbered_default_name_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj1 = NULL;
   HYPREDRV_t obj2 = NULL;
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_default_object_names_are_numbered(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "1", 1);

   char output[16384];
   capture_stderr_output(run_numbered_default_name_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[obj-1]"));
   ASSERT_NOT_NULL(strstr(output, "[obj-2]"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_level_invalid_value_disables_trace(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "invalid", 1);

   char output[8192];
   capture_stderr_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NULL(strstr(output, "[HYPREDRV][L"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_level_enabled_stays_off_stdout(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);

   char output[8192];
   capture_stdout_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NULL(strstr(output, "[HYPREDRV][L"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_stream_stdout_emits_trace_on_stdout(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   setenv("HYPREDRV_LOG_STREAM", "stdout", 1);

   char output[16384];
   capture_stdout_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L1]"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_Create begin"));

   unsetenv("HYPREDRV_LOG_STREAM");
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_stream_stdout_stays_off_stderr(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   setenv("HYPREDRV_LOG_STREAM", "stdout", 1);

   char output[8192];
   capture_stderr_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NULL(strstr(output, "[HYPREDRV][L"));

   unsetenv("HYPREDRV_LOG_STREAM");
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_stream_invalid_value_falls_back_to_stderr(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   setenv("HYPREDRV_LOG_STREAM", "bogus", 1);

   char output[16384];
   capture_stderr_output(run_minimal_lifecycle_for_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L1]"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_Create begin"));

   unsetenv("HYPREDRV_LOG_STREAM");
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_input_args_object_name_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_input_args_internal_logs_use_object_name(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);

   char output[32768];
   capture_stderr_output(run_input_args_object_name_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L3][obj-1] args parse begin"));
   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L3][obj-1] yaml tree build complete"));
   ASSERT_NOT_NULL(strstr(output, "[HYPREDRV][L2][obj-1] args parse end:"));
   ASSERT_NULL(strstr(output, "[HYPREDRV][L3][unnamed] args parse begin"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_named_library_linear_solve_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_ObjectSetName(obj, "named-handle"), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
run_default_named_library_linear_solve_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   parse_minimal_library_yaml(obj);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_solver_and_linsys_internal_logs_use_object_name(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);

   char output[32768];
   capture_stderr_output(run_named_library_linear_solve_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[named-handle]"));
   ASSERT_NOT_NULL(strstr(output, "solver setup begin"));
   ASSERT_NOT_NULL(strstr(output, "solver setup end"));
   ASSERT_NOT_NULL(strstr(output, "initial guess reset begin"));
   ASSERT_NOT_NULL(strstr(output, "initial guess reset end"));
   ASSERT_NOT_NULL(strstr(output, "solver apply begin"));
   ASSERT_NOT_NULL(strstr(output, "solver apply end"));
   ASSERT_NULL(strstr(output, "unnamed][ls="));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_HYPREDRV_log_level_solver_and_linsys_internal_logs_use_default_object_name(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);

   char output[32768];
   capture_stderr_output(run_default_named_library_linear_solve_trace_capture, NULL, output,
                         sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "[obj-1]"));
   ASSERT_NOT_NULL(strstr(output, "solver setup begin"));
   ASSERT_NOT_NULL(strstr(output, "solver setup end"));
   ASSERT_NOT_NULL(strstr(output, "initial guess reset begin"));
   ASSERT_NOT_NULL(strstr(output, "initial guess reset end"));
   ASSERT_NOT_NULL(strstr(output, "solver apply begin"));
   ASSERT_NOT_NULL(strstr(output, "solver apply end"));
   ASSERT_NULL(strstr(output, "unnamed][ls="));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_boundary_logging_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   parse_minimal_library_yaml(obj);
   ASSERT_TRUE(HYPREDRV_PreconApply(obj, NULL, NULL) & ERROR_INVALID_PRECON);
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_boundary_api_traces(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);

   char output[16384];
   capture_stderr_output(run_boundary_logging_trace_capture, NULL, output, sizeof(output));

   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_PreconApply begin"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_PreconDestroy begin"));
   ASSERT_NOT_NULL(strstr(output, "HYPREDRV_LinearSolverDestroy begin"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_precon_variant_trace_capture(void *context)
{
   (void)context;

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "linear_system:\n"
      "  matrix_filename: data/ps3d10pt7/np1/IJ.out.A\n"
      "  rhs_filename: data/ps3d10pt7/np1/IJ.out.b\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    - print_level: 0\n"
      "    - print_level: 1\n";
   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_precon_variant_decisions(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);

   char output[16384];
   capture_stderr_output(run_precon_variant_trace_capture, NULL, output, sizeof(output));

   ASSERT_NOT_NULL(
      strstr(output, "preconditioner variant selection: current=0 requested=0 changed=0"));
   ASSERT_NOT_NULL(
      strstr(output, "preconditioner variant selection: current=0 requested=1 changed=1"));
   ASSERT_NOT_NULL(strstr(output, "preconditioner variant selected: idx=1"));

   unsetenv("HYPREDRV_LOG_LEVEL");
}

struct BuildTraceContext
{
   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
};

static void
run_linear_system_build_trace_capture(void *context)
{
   struct BuildTraceContext *build_context = (struct BuildTraceContext *)context;

   HYPREDRV_t obj = create_initialized_obj();
   char       yaml_config[2 * PATH_MAX + 256];
   int        yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                           "general:\n"
                           "  statistics: off\n"
                           "linear_system:\n"
                           "  matrix_filename: %s\n"
                           "  rhs_filename: %s\n"
                           "solver:\n"
                           "  pcg:\n"
                           "    max_iter: 5\n"
                           "preconditioner:\n"
                           "  amg:\n"
                           "    print_level: 0\n",
                           build_context->matrix_path, build_context->rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));
   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_avoids_linear_system_ready_duplicate(void)
{
   reset_state();

   struct BuildTraceContext context;
   if (!setup_ps3d10pt7_paths(context.matrix_path, context.rhs_path))
   {
      return;
   }

   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   char output[16384];
   capture_stderr_output(run_linear_system_build_trace_capture, &context, output,
                         sizeof(output));

   ASSERT_NULL(strstr(output, "linear system ready: rows="));
   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
run_eigspec_trace_capture(void *context)
{
   (void)context;
   HYPREDRV_t obj = create_initialized_obj();

#ifdef HYPREDRV_ENABLE_EIGSPEC
   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "linear_system:\n"
      "  eigspec:\n"
      "    enable: off\n";
   parse_yaml_into_obj(obj, yaml_config);
#endif

   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_log_level_eigenspectrum_trace_paths(void)
{
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);

   char output[16384];
   capture_stderr_output(run_eigspec_trace_capture, NULL, output, sizeof(output));

#ifdef HYPREDRV_ENABLE_EIGSPEC
   ASSERT_NOT_NULL(strstr(output, "eigenspectrum computation skipped"));
#else
   ASSERT_NOT_NULL(strstr(output, "eigenspectrum support disabled in build"));
#endif

   unsetenv("HYPREDRV_LOG_LEVEL");
}

static void
test_create_parse_and_destroy(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  matrix_filename: %s\n"
                            "  rhs_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 5\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n",
                            matrix_path, rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);


   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   HYPRE_Complex *sol_data = NULL;
   HYPRE_Complex *rhs_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data), ERROR_NONE);
   ASSERT_NOT_NULL(sol_data);
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_NOT_NULL(rhs_data);
   ASSERT_TRUE(rhs_data != sol_data);

   HYPRE_Matrix mat_retrieved = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetMatrix(obj, &mat_retrieved), ERROR_NONE);
   ASSERT_NOT_NULL(mat_retrieved);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetMatrix(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_Vector vec_sol = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_Vector vec_rhs = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs), ERROR_NONE);
   ASSERT_NOT_NULL(vec_rhs);
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Ensure we have a dofmap to work with */
   if (!state->dofmap || state->dofmap->size == 0)
   {
      const int bootstrap_map[] = {0, 1, 2, 3};
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 4, bootstrap_map), ERROR_NONE);
   }

   ASSERT_NOT_NULL(state->dofmap);
   const size_t dofmap_size = state->dofmap->size;
   ASSERT_TRUE(dofmap_size > 0);

   int *manual_dofmap = malloc(dofmap_size * sizeof(int));
   ASSERT_NOT_NULL(manual_dofmap);
   for (size_t i = 0; i < dofmap_size; i++)
   {
      manual_dofmap[i] = (int)((i * 3) % 11);
   }

   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, (int)dofmap_size, manual_dofmap),
             ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, dofmap_size);
   ASSERT_EQ(state->dofmap->data[0], manual_dofmap[0]);
   if (dofmap_size > 5)
   {
      ASSERT_EQ(state->dofmap->data[5], manual_dofmap[5]);
   }
   free(manual_dofmap);

   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 3, 2), ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, (size_t)(3 * 2));
   ASSERT_EQ(state->dofmap->data[0], 0);
   ASSERT_EQ(state->dofmap->data[1], 1);
   ASSERT_EQ(state->dofmap->data[2], 0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 3, 2), ERROR_NONE);
   ASSERT_NOT_NULL(state->dofmap);
   ASSERT_EQ(state->dofmap->size, (size_t)(3 * 2));
   ASSERT_EQ(state->dofmap->data[0], 0);
   ASSERT_EQ(state->dofmap->data[1], 1);
   ASSERT_EQ(state->dofmap->data[2], 2);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);

   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);

   /* Destroy again to exercise unknown-object branch */
   uint32_t code = HYPREDRV_Destroy(&obj);
   ASSERT_TRUE(code & ERROR_UNKNOWN_HYPREDRV_OBJ);
   hypredrv_ErrorCodeResetAll();

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconCreate_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Test reuse logic: first call should create */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Second call with reuse=1 should reuse (not create again) */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   /* Precon should still exist */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverApply_with_xref(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* xref_filename isn't a supported key in this build; cover the xref branch by
    * directly setting the internal pointer to a valid vector. */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->vec_xref               = state->vec_b;

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

   /* Test apply with xref - should compute error norm */
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_stats_level_apis(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Drive Stats using internal timer keys ("prec"/"solve") */
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_BEGIN, "prec");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_END, "prec");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(state->stats, HYPREDRV_ANNOTATE_END, "solve");

   /* Typed stat getters */
   int    typed_iter   = -1;
   double setup_time   = -1.0;
   double solve_time   = -1.0;
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &typed_iter), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSetupTime(obj, &setup_time), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSolveTime(obj, &solve_time), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetNumIter(obj, NULL) & ERROR_UNKNOWN);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetSetupTime(obj, NULL) & ERROR_UNKNOWN);
   ASSERT_TRUE(HYPREDRV_LinearSolverGetSolveTime(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_state_vectors_and_eigspec_error_paths(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Cover dofmap API entrypoints + PrintDofmap error branches */
   int dofmap_one[1] = {0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, dofmap_one), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 1, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, 2), ERROR_NONE);

   ASSERT_TRUE(HYPREDRV_LinearSystemPrintDofmap(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   /* Temporarily hide the dofmap to exercise the ERROR_MISSING_DOFMAP branch,
    * but restore it so the object can cleanly destroy owned state. */
   IntArray *saved_dofmap = state->dofmap;
   state->dofmap          = NULL;
   char *tmp_dof          = CREATE_TEMP_FILE("tmp_dofmap.txt");
   ASSERT_TRUE(HYPREDRV_LinearSystemPrintDofmap(obj, tmp_dof) & ERROR_MISSING_DOFMAP);
   hypredrv_ErrorCodeResetAll();
   free(tmp_dof);
   state->dofmap = saved_dofmap;

   uint32_t print_code = HYPREDRV_LinearSystemPrint(obj);
   ASSERT_TRUE(print_code == ERROR_NONE ||
               (print_code & (ERROR_MISSING_DOFMAP | ERROR_UNKNOWN)));
   hypredrv_ErrorCodeResetAll();

   /* Create two state vectors with the same range as vec_x */
   HYPRE_BigInt ilower = 0, iupper = 0;
   HYPRE_IJVectorGetLocalRange(state->vec_x, &ilower, &iupper);

   HYPRE_IJVector vecs[2] = {NULL, NULL};
   for (int i = 0; i < 2; i++)
   {
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, ilower, iupper, &vecs[i]), 0);
      ASSERT_NOT_NULL(vecs[i]);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vecs[i], HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize(vecs[i]), 0);
      ASSERT_EQ(HYPRE_IJVectorAssemble(vecs[i]), 0);
   }

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 2, vecs), ERROR_NONE);

   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_NOT_NULL(data);

   ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 0, 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_StateVectorApplyCorrection(obj, -1) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorApplyCorrection(obj, 99) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Force xref path in LinearSolverApply */
   state->vec_xref = state->vec_b;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   /* Hit HYPREDRV_PreconApply entrypoint */
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
             ERROR_NONE);

   /* Eigenspectrum API entrypoint: always succeeds (no-op when built without eigspec) */
   ASSERT_EQ(HYPREDRV_LinearSystemComputeEigenspectrum(obj), ERROR_NONE);

   /* Force error branches in GetValues/Copy by nulling an internal vec_s entry */
   HYPRE_IJVector saved_state0 = state->vec_s[state->states[0]];
   state->vec_s[state->states[0]] = NULL;
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorGetValues(obj, 0, &data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_StateVectorCopy(obj, 0, 1) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_s[state->states[0]] = saved_state0;

   /* vecs[] are now owned by hypredrv (Destroy() will destroy hypredrv->vec_s[i]). */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_InputArgsParse_exec_policy(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "  use_vendor_spgemm: on\n"
            "  use_vendor_spmv: on\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

#if HYPRE_CHECK_MIN_VERSION(22100, 0)
   HYPRE_MemoryLocation   memory_location = HYPRE_MEMORY_HOST;
   HYPRE_ExecutionPolicy  exec_policy     = HYPRE_EXEC_HOST;

   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

#if HYPRE_CHECK_MIN_VERSION(22100, 0)
   ASSERT_EQ(HYPRE_GetMemoryLocation(&memory_location), 0);
   ASSERT_EQ(HYPRE_GetExecutionPolicy(&exec_policy), 0);
   ASSERT_EQ(memory_location, HYPRE_MEMORY_HOST);
   ASSERT_EQ(exec_policy, HYPRE_EXEC_HOST);
#endif

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_InputArgsParse_gpu_standard_amg_forces_host_exec(void)
{
#if !defined(HYPRE_USING_GPU) || !HYPRE_CHECK_MIN_VERSION(22100, 0)
   return;
#else
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_variants[2 * PATH_MAX + 1024];
   int  yaml_variants_len = snprintf(
      yaml_variants, sizeof(yaml_variants),
      "general:\n"
      "  statistics: off\n"
      "linear_system:\n"
      "  matrix_filename: %s\n"
      "  rhs_filename: %s\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  variants:\n"
      "    - amg:\n"
      "        print_level: 0\n"
      "        interpolation:\n"
      "          prolongation_type: MM-ext+i\n"
      "    - amg:\n"
      "        print_level: 0\n"
      "        interpolation:\n"
      "          prolongation_type: standard\n",
      matrix_path, rhs_path);
   ASSERT_TRUE(yaml_variants_len > 0 && (size_t)yaml_variants_len < sizeof(yaml_variants));

   parse_yaml_into_obj(obj, yaml_variants);
   ASSERT_EQ(obj->iargs->general.exec_policy, 1);
   ASSERT_EQ(obj->iargs->ls.exec_policy, 1);

   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);
   ASSERT_EQ(obj->iargs->general.exec_policy, 0);
   ASSERT_EQ(obj->iargs->ls.exec_policy, 0);

   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(obj->iargs->general.exec_policy, 1);
   ASSERT_EQ(obj->iargs->ls.exec_policy, 1);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
}

static void
test_HYPREDRV_LinearSystemComputeEigenspectrum_warns_once_when_disabled(void)
{
#ifdef HYPREDRV_ENABLE_EIGSPEC
   return;
#else
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   char       buffer[1024];

   capture_eigspec_warning_output(obj, 2, buffer, sizeof(buffer));

   /* Warning is process-global and may already have been emitted by prior tests. */
   ASSERT_TRUE(count_substr(buffer, "eigenspectrum support is disabled") <= 1);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
}

static void
test_HYPREDRV_PreconCreate_reuse_logic_variations(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   /* Test reuse=0 (always create) */
   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 0\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* With reuse=0, should always create */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Second call should also create (reuse=0 means no reuse) */
   HYPREDRV_PreconDestroy(obj);
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverCreate_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 2\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* First call should create (ls_id=0, (0+1) % (2+1) = 1, not 0, but first system) */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->solver);

   /* Second call with reuse=2: ls_id=1, (1+1) % 3 = 2, should reuse (not create) */
   HYPREDRV_LinearSolverDestroy(obj);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   /* Solver should be NULL because we're reusing */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Create precon */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* With reuse=1, ls_id=0: (0+1) % 2 = 1, should NOT destroy (ls_id must be > 0) */
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   /* Precon should still exist because ls_id=0 */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverDestroy_reuse_logic(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    frequency: 2\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Create precon and solver */
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->solver);

   /* Test ls_id = 0: should NOT destroy (ls_id must be > 0) */
   if (state->stats) {
      state->stats->ls_counter = 1; /* ls_id = ls_counter - 1 = 0 */
   }
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   /* Solver should still exist */

   /* Test ls_id = 3 with reuse=2: should destroy ((3+1)%(2+1)=0) */
   if (state->stats) {
      state->stats->ls_counter = 4; /* ls_id = ls_counter - 1 = 3 */
   }
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   /* Solver should be destroyed */

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_linear_system_ids(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    linear_system_ids: [0, 2]\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 1, not in [0,2], so object should be kept. */
   state->stats->ls_counter = 1;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 2, in [0,2], so object should be dropped now. */
   state->stats->ls_counter = 2;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconDestroy_reuse_per_timestep(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   char *tmp_ts = CREATE_TEMP_FILE("tmp_timesteps_reuse.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "2\n");
   fprintf(tf, "0 0\n");
   fprintf(tf, "1 3\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 640];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    per_timestep: on\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path, tmp_ts);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 1, not a timestep start, so keep object. */
   state->stats->ls_counter = 1;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* Next solve id = 3, timestep start, so destroy for recompute. */
   state->stats->ls_counter = 3;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_PreconDestroy_reuse_per_timestep_frequency(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   char *tmp_ts = CREATE_TEMP_FILE("tmp_timesteps_reuse_freq.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "4\n");
   fprintf(tf, "0 0\n");
   fprintf(tf, "1 3\n");
   fprintf(tf, "2 6\n");
   fprintf(tf, "3 9\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 700];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    per_timestep: on\n"
            "    frequency: 1\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path, tmp_ts);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 4 is inside timestep 1, so keep reusing the current timestep preconditioner. */
   state->stats->ls_counter = 4;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 5 is still inside timestep 1, so keep reusing. */
   state->stats->ls_counter = 5;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   /* ls id 6 is the first system in timestep 2, so rebuild for the new timestep. */
   state->stats->ls_counter = 6;
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_library_mode_reuse_per_timestep_frequency_with_object_annotations(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_library_reuse_yaml(obj,
                            "  reuse:\n"
                            "    enabled: yes\n"
                            "    per_timestep: on\n"
                            "    frequency: 1\n");

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-1");
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_adaptive_reuse_rebuilds_after_degradation(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[4096];
   snprintf(
      yaml_config, sizeof(yaml_config),
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  reuse:\n"
      "    type: adaptive\n"
      "    guards:\n"
      "      max_iteration_ratio: -1\n"
      "  amg:\n"
      "    print_level: 0\n");
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   run_library_linear_solve(obj, NULL);
   ASSERT_EQ_SIZE(state->precon_reuse_state.count, 1);

   run_library_linear_solve(obj, NULL);
   ASSERT_EQ_SIZE(state->precon_reuse_state.count, 2);

   run_library_linear_solve(obj, NULL);
   ASSERT_EQ_SIZE(state->precon_reuse_state.count, 3);
   ASSERT_TRUE(state->precon_reuse_state.baseline_valid);

   /* Inject a bad observation so the next solve decision sees high iteration count.
    * Using setup_time=solve_time=0 means only the stability (iterations) component
    * contributes; the efficiency component's fmax(0 - baseline_solve, 0) = 0. */
   PreconReuseObservation bad_obs;
   memset(&bad_obs, 0, sizeof(bad_obs));
   bad_obs.iters           = 100;
   bad_obs.solve_succeeded = 1;
   for (int l = 0; l < STATS_MAX_LEVELS; l++) bad_obs.level_ids[l] = -1;
   hypredrv_PreconReuseStateRecordObservation(&state->precon_reuse_state, &bad_obs);

   /* Solve #4: score exceeds threshold but streak(1) < bad_decisions_to_rebuild(2),
    * so no rebuild yet.  A real observation is appended after the solve. */
   run_library_linear_solve(obj, NULL);
   ASSERT_EQ(state->precon_reuse_state.bad_decision_streak, 1);

   /* Inject a second bad observation to push the streak to the rebuild threshold. */
   hypredrv_PreconReuseStateRecordObservation(&state->precon_reuse_state, &bad_obs);

   /* Solve #5: streak reaches 2 >= bad_decisions_to_rebuild(2), rebuild fires.
    * MarkRebuild resets state; one real observation is recorded after the rebuild. */
   run_library_linear_solve(obj, NULL);
   ASSERT_EQ_SIZE(state->precon_reuse_state.count, 1);
   /* After rebuild: baseline is cleared and streak is reset. */
   ASSERT_FALSE(state->precon_reuse_state.baseline_valid);
   ASSERT_EQ(state->precon_reuse_state.bad_decision_streak, 0);
   /* Preconditioner was actually rebuilt (not NULL). */
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconReuseBuildObservation_and_MarkRebuild_library(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  reuse: adaptive\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   run_library_linear_solve(obj, NULL);

   PreconReuseObservation obs;
   memset(&obs, 0, sizeof(obs));
   hypredrv_PreconReuseBuildObservation(obj, NULL, &obs);
   ASSERT_TRUE(obs.iters >= 0);
   ASSERT_TRUE(obs.setup_time >= 0.0);
   ASSERT_TRUE(obs.solve_time >= 0.0);

   {
      int       starts[2] = { 0, 5 };
      IntArray *ts        = NULL;
      hypredrv_IntArrayBuild(MPI_COMM_SELF, 2, starts, &ts);
      ASSERT_NOT_NULL(ts);
      hypredrv_PreconReuseBuildObservation(obj, ts, &obs);
      hypredrv_IntArrayDestroy(&ts);
   }

   ASSERT_NOT_NULL(state->stats);
   state->stats->level_active |= (1u << 0);
   state->stats->level_current_id[0] = 4;

   hypredrv_PreconReuseMarkRebuild(obj, &state->precon_reuse_state);
   ASSERT_EQ(state->precon_reuse_state.last_rebuild_level_ids[0], 3);
   ASSERT_EQ_SIZE(state->precon_reuse_state.count, 0u);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSolverApply_error_cases(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSolverApply(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);

   /* Test with uninitialized solver */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->solver = NULL;
   result = HYPREDRV_LinearSolverApply(obj);
   ASSERT_TRUE(result & ERROR_INVALID_SOLVER);

   /* Clear sticky error state before cleanup assertions */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_Annotate_functions(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test annotate functions */
   HYPREDRV_AnnotateBegin(obj, "test", 1);
   HYPREDRV_AnnotateEnd(obj, "test", 1);

   /* Test with NULL name */
   HYPREDRV_AnnotateBegin(obj, NULL, 1);
   HYPREDRV_AnnotateEnd(obj, NULL, 1);

   /* These wrappers intentionally tolerate unknown annotation keys, but they can set
    * sticky error flags; clear before cleanup assertions. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_object_scoped_annotation_isolation(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj1 = NULL;
   HYPREDRV_t obj2 = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj2), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj2, 0, "obj2-timestep", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj2, 0, "obj2-timestep", -1), ERROR_NONE);

   int count1 = -1;
   int count2 = -1;
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj1, 0, &count1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj2, 0, &count2), ERROR_NONE);
   ASSERT_EQ(count1, 0);
   ASSERT_EQ(count2, 1);

   int    entry_id     = -1;
   int    num_solves   = -1;
   int    linear_iters = -1;
   double setup_time   = -1.0;
   double solve_time   = -1.0;
   ASSERT_EQ(HYPREDRV_StatsLevelGetEntry(obj2, 0, 0, &entry_id, &num_solves,
                                           &linear_iters, &setup_time, &solve_time),
             ERROR_NONE);
   ASSERT_EQ(entry_id, 1);
   ASSERT_EQ(num_solves, 0);
   ASSERT_EQ(linear_iters, 0);
   ASSERT_EQ_DOUBLE(setup_time, 0.0, 1e-12);
   ASSERT_EQ_DOUBLE(solve_time, 0.0, 1e-12);

   /* Annotating obj1 should not affect obj2 */
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj1, 0, "obj1-timestep", 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj1, 0, "obj1-timestep", 0), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj1, 0, &count1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj2, 0, &count2), ERROR_NONE);
   ASSERT_EQ(count1, 1);
   ASSERT_EQ(count2, 1);

   ASSERT_EQ(HYPREDRV_StatsLevelPrint(obj2, 0), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_reuse_per_timestep_with_object_annotations(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_library_reuse_yaml(obj,
                            "  reuse:\n"
                            "    enabled: yes\n"
                            "    per_timestep: on\n"
                            "    frequency: 0\n");

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-1");
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
   ASSERT_NULL(state->precon);

   run_library_linear_solve(obj, "newton-0");
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   int timestep_count = -1;
   ASSERT_EQ(HYPREDRV_StatsLevelGetCount(obj, 0, &timestep_count), ERROR_NONE);
   ASSERT_EQ(timestep_count, 2);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_mgr_recreates_precon_on_new_timestep(void)
{
   reset_state();

   /* MGR with AMG f-relaxation + filter_functions crashes in hypre < 2.21.0 */
#if HYPREDRV_HYPRE_RELEASE_NUMBER < 22100
   printf("SKIP: MGR AMG f-relaxation requires hypre >= 2.21.0\n");
   return;
#endif

   char ls_dir[PATH_MAX];
   if (!setup_poromech2k_dir(ls_dir))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[8192];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  dirname: %s\n"
            "  init_suffix: 0\n"
            "  last_suffix: 24\n"
            "  rhs_filename: IJ.out.b\n"
            "  matrix_filename: IJ.out.A\n"
            "  dofmap_filename: dofmap.out\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  fgmres:\n"
            "    max_iter: 100\n"
            "    krylov_dim: 30\n"
            "    print_level: 0\n"
            "    relative_tol: 1.0e-6\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    enabled: yes\n"
            "    per_timestep: on\n"
            "    frequency: 0\n"
            "  mgr:\n"
            "    max_iter: 1\n"
            "    tolerance: 0.0\n"
            "    print_level: 0\n"
            "    coarse_th: 1e-20\n"
            "    level:\n"
            "      0:\n"
            "        f_dofs: [0, 1, 2]\n"
            "        f_relaxation:\n"
            "          amg:\n"
            "            coarsening:\n"
            "              type: pmis\n"
            "              strong_th: 0.5\n"
            "              num_functions: 3\n"
            "              filter_functions: on\n"
            "        g_relaxation: none\n"
            "        restriction_type: injection\n"
            "        prolongation_type: blk-jacobi\n"
            "        coarse_level_type: non-galerkin\n"
            "      1:\n"
            "        f_dofs: [5]\n"
            "        f_relaxation: jacobi\n"
            "        g_relaxation: none\n"
            "        restriction_type: injection\n"
            "        prolongation_type: jacobi\n"
            "        coarse_level_type: rap\n"
            "      2:\n"
            "        f_dofs: [4]\n"
            "        f_relaxation: single\n"
            "        g_relaxation: ilu\n"
            "        restriction_type: columped\n"
            "        prolongation_type: injection\n"
            "        coarse_level_type: rap\n"
            "    coarsest_level:\n"
            "      amg:\n"
            "        max_iter: 1\n"
            "        tolerance: 0.0\n"
            "        relaxation:\n"
            "          down_type: l1-jacobi\n"
            "          up_type: l1-jacobi\n",
            ls_dir);
   parse_yaml_into_obj(obj, yaml_config);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-0", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   run_library_linear_solve(obj, NULL);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_TRUE(state->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-0", -1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep-1", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* Regression guard: the first solve in a new timestep must mark the reused MGR
    * preconditioner dirty again before setup. */
   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_FALSE(state->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep-1", -1), ERROR_NONE);

   /* LinearSystemBuild created mat_A/vec_b from files, but lib_mode
    * prevents Destroy from freeing them.  Clean up explicitly. */
   HYPRE_IJMatrixDestroy(state->mat_A);
   state->mat_A = NULL;
   HYPRE_IJVectorDestroy(state->vec_b);
   state->vec_b = NULL;

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

struct LinearSetupCaptureContext
{
   HYPREDRV_t obj;
};

static void
run_linear_setup_for_capture(void *context)
{
   struct LinearSetupCaptureContext *setup_context =
      (struct LinearSetupCaptureContext *)context;
   ASSERT_EQ(HYPREDRV_AnnotateBegin(setup_context->obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(setup_context->obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(setup_context->obj, "system", -1), ERROR_NONE);
}

static void
park_mgr_component_reuse_handles(HYPREDRV_t obj)
{
   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
}

struct SetPreconVariantCaptureContext
{
   HYPREDRV_t obj;
   int        variant_idx;
};

static void
set_precon_variant_for_capture(void *context)
{
   struct SetPreconVariantCaptureContext *variant_context =
      (struct SetPreconVariantCaptureContext *)context;
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(variant_context->obj,
                                                variant_context->variant_idx),
             ERROR_NONE);
}

struct SetPreconPresetCaptureContext
{
   HYPREDRV_t  obj;
   const char *preset;
};

static void
set_precon_preset_for_capture(void *context)
{
   struct SetPreconPresetCaptureContext *preset_context =
      (struct SetPreconPresetCaptureContext *)context;
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(preset_context->obj, preset_context->preset),
             ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_mgr_component_reuse_refreshes_selected_handles(void)
{
#if !defined(HYPREDRV_ENABLE_EXPERIMENTAL) || !HYPRE_CHECK_MIN_VERSION(23100, 9)
   return;
#else
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   hypredrv_LogInitializeFromEnv();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  reuse:\n"
      "    linear_system_ids: [1]\n"
      "  mgr:\n"
      "    max_iter: 1\n"
      "    print_level: 0\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation:\n"
      "          amg:\n"
      "            max_iter: 1\n"
      "          reuse:\n"
      "            linear_system_ids: [1]\n"
      "        g_relaxation:\n"
      "          ilu:\n"
      "            max_iter: 1\n"
      "        restriction_type: injection\n"
      "        prolongation_type: injection\n"
      "        coarse_level_type: rap\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        max_iter: 1\n"
      "      reuse:\n"
      "        linear_system_ids: [1]\n";

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, 2), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_2x2(4.0, 1.0, 1.0, 3.0);
   HYPRE_IJVector vec_b = create_test_ijvector_2x1(1.0, 2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_AnnotateBegin(obj, "system", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateEnd(obj, "system", -1), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_TRUE(state->precon_is_setup);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.frelax[0]);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.grelax[0]);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.csolver);

   HYPRE_Solver outer0 = state->precon->main;
   HYPRE_Solver f0     = state->iargs->precon.mgr.frelax[0];
   HYPRE_Solver c0     = state->iargs->precon.mgr.csolver;

   struct LinearSetupCaptureContext setup_context = {obj};
   char                             output[4096];
   capture_stderr_output(run_linear_setup_for_capture, &setup_context, output,
                         sizeof(output));

   ASSERT_NOT_NULL(state->precon);
   ASSERT_TRUE(state->precon_is_setup);
   ASSERT_PTR_EQ(state->precon->main, outer0);
   ASSERT_PTR_EQ(state->iargs->precon.mgr.frelax[0], f0);
   ASSERT_PTR_EQ(state->iargs->precon.mgr.csolver, c0);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.grelax[0]);
   ASSERT_TRUE(strstr(output, "rerun_mgr_setup=1") != NULL);

   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
}

static void
test_HYPREDRV_InputArgsSetPreconVariant_discards_cached_mgr_handles(void)
{
#if !defined(HYPREDRV_ENABLE_EXPERIMENTAL) || !HYPRE_CHECK_MIN_VERSION(23100, 9)
   return;
#else
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   hypredrv_LogInitializeFromEnv();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  - mgr:\n"
      "      max_iter: 1\n"
      "      print_level: 0\n"
      "      level:\n"
      "        0:\n"
      "          f_dofs: [0]\n"
      "          f_relaxation:\n"
      "            amg:\n"
      "              max_iter: 1\n"
      "            reuse:\n"
      "              linear_system_ids: [1]\n"
      "          g_relaxation:\n"
      "            ilu:\n"
      "              max_iter: 1\n"
      "          restriction_type: injection\n"
      "          prolongation_type: injection\n"
      "          coarse_level_type: rap\n"
      "      coarsest_level:\n"
      "        amg:\n"
      "          max_iter: 1\n"
      "        reuse:\n"
      "          linear_system_ids: [1]\n"
      "  - amg:\n"
      "      print_level: 0\n";

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, 2), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_2x2(4.0, 1.0, 1.0, 3.0);
   HYPRE_IJVector vec_b = create_test_ijvector_2x1(1.0, 2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   park_mgr_component_reuse_handles(obj);
   ASSERT_NULL(state->precon);
   ASSERT_EQ(state->iargs->precon_method, PRECON_MGR);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.frelax[0]);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.csolver);

   struct SetPreconVariantCaptureContext variant_context = {obj, 1};
   char                                  output[4096];
   capture_stderr_output(set_precon_variant_for_capture, &variant_context, output,
                         sizeof(output));

   ASSERT_EQ(state->iargs->active_precon_variant, 1);
   ASSERT_EQ(state->iargs->precon_method, PRECON_BOOMERAMG);
   ASSERT_TRUE(strstr(output,
                      "discarding cached MGR handles before switching preconditioner "
                      "variant") != NULL);

   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
}

static void
test_HYPREDRV_InputArgsSetPreconPreset_discards_cached_mgr_handles(void)
{
#if !defined(HYPREDRV_ENABLE_EXPERIMENTAL) || !HYPRE_CHECK_MIN_VERSION(23100, 9)
   return;
#else
   reset_state();
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   hypredrv_LogInitializeFromEnv();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    max_iter: 1\n"
      "    print_level: 0\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation:\n"
      "          amg:\n"
      "            max_iter: 1\n"
      "          reuse:\n"
      "            linear_system_ids: [1]\n"
      "        g_relaxation:\n"
      "          ilu:\n"
      "            max_iter: 1\n"
      "        restriction_type: injection\n"
      "        prolongation_type: injection\n"
      "        coarse_level_type: rap\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        max_iter: 1\n"
      "      reuse:\n"
      "        linear_system_ids: [1]\n";

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 1, 2), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_2x2(4.0, 1.0, 1.0, 3.0);
   HYPRE_IJVector vec_b = create_test_ijvector_2x1(1.0, 2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   park_mgr_component_reuse_handles(obj);
   ASSERT_NULL(state->precon);
   ASSERT_EQ(state->iargs->precon_method, PRECON_MGR);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.frelax[0]);
   ASSERT_NOT_NULL(state->iargs->precon.mgr.csolver);

   struct SetPreconPresetCaptureContext preset_context = {obj, "elasticity-2D"};
   char                                 output[4096];
   capture_stderr_output(set_precon_preset_for_capture, &preset_context, output,
                         sizeof(output));

   ASSERT_EQ(state->iargs->precon_method, PRECON_BOOMERAMG);
   ASSERT_EQ(state->iargs->precon.amg.coarsening.num_functions, 2);
   ASSERT_TRUE(strstr(output,
                      "discarding cached MGR handles before applying preconditioner "
                      "preset") != NULL);

   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   unsetenv("HYPREDRV_LOG_LEVEL");
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
#endif
}

struct DestroyObjectContext
{
   HYPREDRV_t *obj_ptr;
};

static void
destroy_object_for_capture(void *context)
{
   struct DestroyObjectContext *destroy_context = (struct DestroyObjectContext *)context;
   ASSERT_EQ(HYPREDRV_Destroy(destroy_context->obj_ptr), ERROR_NONE);
}

static void
finalize_for_capture(void *context)
{
   (void)context;
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

struct StatsPrintContext
{
   HYPREDRV_t obj;
};

static void
read_text_file(const char *path, char *buffer, size_t buf_len)
{
   ASSERT_NOT_NULL(path);
   ASSERT_NOT_NULL(buffer);
   ASSERT_TRUE(buf_len > 0);

   FILE *fp = fopen(path, "r");
   ASSERT_NOT_NULL(fp);

   size_t read_bytes  = fread(buffer, 1, buf_len - 1, fp);
   buffer[read_bytes] = '\0';
   fclose(fp);
}

static void
stats_print_for_capture(void *context)
{
   struct StatsPrintContext *stats_context = (struct StatsPrintContext *)context;
   ASSERT_EQ(HYPREDRV_StatsPrint(stats_context->obj), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_destroy_prints_named_statistics_summary(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_ObjectSetName(obj, "named-handle"), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct DestroyObjectContext destroy_context = {&obj};
   capture_stdout_output(destroy_object_for_capture, &destroy_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "STATISTICS SUMMARY for named-handle:") != NULL);
   ASSERT_TRUE(strstr(output, "|          0 |") != NULL);
   ASSERT_NULL(obj);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_library_mode_finalize_prints_named_statistics_summary(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_ObjectSetName(obj, "finalize-handle"), ERROR_NONE);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   capture_stdout_output(finalize_for_capture, NULL, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "STATISTICS SUMMARY for finalize-handle:") != NULL);
   ASSERT_TRUE(strstr(output, "|          0 |") != NULL);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
}

static void
test_HYPREDRV_statistics_filename_routes_stats_to_file(void)
{
   reset_state();

   char *stats_file = CREATE_TEMP_FILE("tmp_stats_output_file.txt");
   ASSERT_NOT_NULL(stats_file);

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "  statistics_filename: %s\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            stats_file);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   struct StatsPrintContext stats_context = {obj};
   char                     stdout_output[8192];
   capture_stdout_output(stats_print_for_capture, &stats_context, stdout_output,
                         sizeof(stdout_output));
   ASSERT_NULL(strstr(stdout_output, "STATISTICS SUMMARY"));

   char file_output[8192];
   read_text_file(stats_file, file_output, sizeof(file_output));
   ASSERT_NOT_NULL(strstr(file_output, "STATISTICS SUMMARY"));
   ASSERT_NOT_NULL(strstr(file_output, "|          0 |"));

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(stats_file);
}

static void
test_HYPREDRV_statistics_filename_fallbacks_to_stdout_on_open_failure(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char invalid_path[PATH_MAX];
   snprintf(invalid_path, sizeof(invalid_path), "/tmp/hypredrv_missing_%d/stats.out",
            (int)getpid());

   char yaml_config[PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "  statistics_filename: %s\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            invalid_path);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   struct StatsPrintContext stats_context = {obj};
   char                     stderr_output[8192];
   capture_stderr_output(stats_print_for_capture, &stats_context, stderr_output,
                         sizeof(stderr_output));
   ASSERT_NOT_NULL(strstr(stderr_output, "failed to open general.statistics_filename"));

   char stdout_output[8192];
   capture_stdout_output(stats_print_for_capture, &stats_context, stdout_output,
                         sizeof(stdout_output));
   ASSERT_NOT_NULL(strstr(stdout_output, "STATISTICS SUMMARY"));

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_stats_flat_runs_keep_entry_column(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct StatsPrintContext stats_context = {obj};
   capture_stdout_output(stats_print_for_capture, &stats_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "Entry") != NULL);
   ASSERT_TRUE(strstr(output, "Path") == NULL);
   ASSERT_TRUE(strstr(output, "|          0 |") != NULL);

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_stats_annotated_runs_switch_to_path_column(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: on\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 5\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "timestep", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 1, "nonlinear", -1), ERROR_NONE);
   run_library_linear_solve(obj, NULL);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 1, "nonlinear", -1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "timestep", -1), ERROR_NONE);

   char output[8192];
   struct StatsPrintContext stats_context = {obj};
   capture_stdout_output(stats_print_for_capture, &stats_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "Path") != NULL);
   ASSERT_TRUE(strstr(output, "1.1.1") != NULL);

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_stats_timestep_file_paths_use_preserved_ids(void)
{
   reset_state();

   char *tmp_ts = CREATE_TEMP_FILE("tmp_stats_path_timesteps.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "2\n");
   fprintf(tf, "10 0\n");
   fprintf(tf, "20 1\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    enabled: on\n"
            "    per_timestep: on\n"
            "  amg:\n"
            "    print_level: 0\n",
            tmp_ts);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   run_library_linear_solve(obj, NULL);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct StatsPrintContext stats_context = {obj};
   capture_stdout_output(stats_print_for_capture, &stats_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "Path") != NULL);
   ASSERT_TRUE(strstr(output, "10.1") != NULL);
   ASSERT_TRUE(strstr(output, "20.2") != NULL);

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_stats_timestep_file_paths_without_reuse_use_path_column(void)
{
   reset_state();

   char *tmp_ts = CREATE_TEMP_FILE("tmp_stats_path_timesteps_no_reuse.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "2\n");
   fprintf(tf, "10 0\n");
   fprintf(tf, "20 1\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            tmp_ts);
   parse_yaml_into_obj(obj, yaml_config);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_NOT_NULL(state->precon_reuse_timesteps.ids);
   ASSERT_NOT_NULL(state->precon_reuse_timesteps.starts);
   ASSERT_EQ_SIZE(state->precon_reuse_timesteps.ids->size, 2);
   ASSERT_EQ_SIZE(state->precon_reuse_timesteps.starts->size, 2);
   ASSERT_EQ(state->precon_reuse_timesteps.ids->data[0], 10);
   ASSERT_EQ(state->precon_reuse_timesteps.ids->data[1], 20);
   ASSERT_EQ(state->precon_reuse_timesteps.starts->data[0], 0);
   ASSERT_EQ(state->precon_reuse_timesteps.starts->data[1], 1);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   run_library_linear_solve(obj, NULL);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct StatsPrintContext stats_context = {obj};
   capture_stdout_output(stats_print_for_capture, &stats_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "Path") != NULL);
   ASSERT_TRUE(strstr(output, "10.1") != NULL);
   ASSERT_TRUE(strstr(output, "20.2") != NULL);

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_stats_path_column_truncates_long_paths(void)
{
   reset_state();

   char *tmp_ts = CREATE_TEMP_FILE("tmp_stats_path_truncation.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "1\n");
   fprintf(tf, "123456789 0\n");
   fclose(tf);

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: on\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  timestep_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  reuse:\n"
            "    enabled: on\n"
            "    per_timestep: on\n"
            "  amg:\n"
            "    print_level: 0\n",
            tmp_ts);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char output[8192];
   struct StatsPrintContext stats_context = {obj};
   capture_stdout_output(stats_print_for_capture, &stats_context, output, sizeof(output));

   ASSERT_TRUE(strstr(output, "| ...56789.1 |") != NULL);

   ((struct hypredrv_struct *)obj)->iargs->general.statistics = 0;
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

static void
test_HYPREDRV_print_system_lifecycle_dumps_setup_and_apply(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char outdir[PATH_MAX];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_lifecycle_dump_%d", (int)getpid());
   char cleanup_cmd[PATH_MAX + 32];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   char yaml_config[PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: all\n"
            "    stage: all\n"
            "    artifacts: [matrix, rhs, solution, metadata]\n"
            "    output_dir: %s\n"
            "    overwrite: on\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            outdir);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char object_dir[PATH_MAX];
   ASSERT_TRUE(path_join2(object_dir, sizeof(object_dir), outdir, "obj-1"));

   char dump0_metadata[PATH_MAX];
   char dump1_metadata[PATH_MAX];
   ASSERT_TRUE(path_join2(dump0_metadata, sizeof(dump0_metadata), object_dir,
                          "ls_00000/metadata.txt"));
   ASSERT_TRUE(path_join2(dump1_metadata, sizeof(dump1_metadata), object_dir,
                          "ls_00001/metadata.txt"));
   ASSERT_TRUE(access(dump0_metadata, F_OK) == 0);
   ASSERT_TRUE(access(dump1_metadata, F_OK) == 0);

   char systems_index[PATH_MAX];
   ASSERT_TRUE(path_join2(systems_index, sizeof(systems_index), object_dir,
                          "systems_index.txt"));
   ASSERT_TRUE(access(systems_index, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(systems_index, "ls_00000"));
   ASSERT_TRUE(file_contains_substr(systems_index, "ls_00001"));
   ASSERT_TRUE(file_contains_substr(systems_index, "stage=setup"));
   ASSERT_TRUE(file_contains_substr(systems_index, "stage=apply"));

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;
}

static void
test_HYPREDRV_print_system_apply_stage_ids_use_current_stats_id(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char outdir[PATH_MAX];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_apply_dump_%d", (int)getpid());
   char cleanup_cmd[PATH_MAX + 32];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   char yaml_config[PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: ids\n"
            "    stage: apply\n"
            "    ids: [0]\n"
            "    artifacts: [metadata]\n"
            "    output_dir: %s\n"
            "    overwrite: on\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            outdir);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char metadata_path[PATH_MAX];
   ASSERT_TRUE(path_join2(metadata_path, sizeof(metadata_path), outdir,
                          "obj-1/ls_00000/metadata.txt"));
   ASSERT_TRUE(access(metadata_path, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_path, "stage=apply"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "system_index=0"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "stats_ls_id=0"));

   char unexpected_path[PATH_MAX];
   ASSERT_TRUE(path_join2(unexpected_path, sizeof(unexpected_path), outdir,
                          "obj-1/ls_00001"));
   ASSERT_TRUE(access(unexpected_path, F_OK) != 0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;
}

static void
test_HYPREDRV_print_system_setup_stage_ids_advance_for_library_cycles(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char outdir[PATH_MAX];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_setup_dump_%d", (int)getpid());
   char cleanup_cmd[PATH_MAX + 32];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   char yaml_config[PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: ids\n"
            "    stage: setup\n"
            "    ids: [1]\n"
            "    artifacts: [metadata]\n"
            "    output_dir: %s\n"
            "    overwrite: on\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            outdir);
   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A0 = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b0 = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A0, vec_b0);
   run_library_linear_solve(obj, NULL);

   char metadata_path[PATH_MAX];
   ASSERT_TRUE(path_join2(metadata_path, sizeof(metadata_path), outdir,
                          "obj-1/ls_00000/metadata.txt"));
   ASSERT_TRUE(access(metadata_path, F_OK) != 0);

   HYPRE_IJMatrix mat_A1 = create_test_ijmatrix_1x1(8.0);
   HYPRE_IJVector vec_b1 = create_test_ijvector_1x1(3.0);
   attach_library_scalar_system(obj, mat_A1, vec_b1);
   run_library_linear_solve(obj, NULL);

   ASSERT_TRUE(access(metadata_path, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_path, "stage=setup"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "system_index=1"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "stats_ls_id=0"));

   char unexpected_path[PATH_MAX];
   ASSERT_TRUE(path_join2(unexpected_path, sizeof(unexpected_path), outdir,
                          "obj-1/ls_00001"));
   ASSERT_TRUE(access(unexpected_path, F_OK) != 0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b1), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A1), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b0), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A0), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;
}

static void
test_HYPREDRV_InputArgsParse_loads_lsseq_timesteps_for_print_system(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();

   char seq_path[PATH_MAX];
   snprintf(seq_path, sizeof(seq_path), "/tmp/hypredrive_print_system_%d.lsseq",
            (int)getpid());
   write_test_lsseq_with_timesteps(seq_path);

   char yaml_config[PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  sequence_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: every_n_timesteps\n"
            "    every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            seq_path);
   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_NOT_NULL(obj->precon_reuse_timesteps.ids);
   ASSERT_NOT_NULL(obj->precon_reuse_timesteps.starts);
   ASSERT_EQ_SIZE(obj->precon_reuse_timesteps.ids->size, 2);
   ASSERT_EQ_SIZE(obj->precon_reuse_timesteps.starts->size, 2);
   ASSERT_EQ(obj->precon_reuse_timesteps.ids->data[0], 0);
   ASSERT_EQ(obj->precon_reuse_timesteps.ids->data[1], 1);
   ASSERT_EQ(obj->precon_reuse_timesteps.starts->data[0], 0);
   ASSERT_EQ(obj->precon_reuse_timesteps.starts->data[1], 1);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   remove(seq_path);
}

static void
test_HYPREDRV_print_system_uses_timestep_filename_without_reuse(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char *tmp_ts = CREATE_TEMP_FILE("tmp_print_system_timesteps.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "1\n");
   fprintf(tf, "0 0\n");
   fclose(tf);

   char outdir[PATH_MAX];
   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_timestep_dump_%d", (int)getpid());
   char cleanup_cmd[PATH_MAX + 32];
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);
   int cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;

   char yaml_config[2 * PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  init_guess_mode: zeros\n"
            "  timestep_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: every_n_timesteps\n"
            "    every: 1\n"
            "    stage: apply\n"
            "    artifacts: [metadata]\n"
            "    output_dir: %s\n"
            "    overwrite: on\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            tmp_ts, outdir);
   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_NOT_NULL(obj->precon_reuse_timesteps.ids);
   ASSERT_NOT_NULL(obj->precon_reuse_timesteps.starts);
   ASSERT_EQ_SIZE(obj->precon_reuse_timesteps.ids->size, 1);
   ASSERT_EQ_SIZE(obj->precon_reuse_timesteps.starts->size, 1);
   ASSERT_EQ(obj->precon_reuse_timesteps.ids->data[0], 0);
   ASSERT_EQ(obj->precon_reuse_timesteps.starts->data[0], 0);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);
   run_library_linear_solve(obj, NULL);

   char metadata_path[PATH_MAX];
   ASSERT_TRUE(path_join2(metadata_path, sizeof(metadata_path), outdir,
                          "obj-1/ls_00000/metadata.txt"));
   ASSERT_TRUE(access(metadata_path, F_OK) == 0);
   ASSERT_TRUE(file_contains_substr(metadata_path, "stage=apply"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "system_index=0"));
   ASSERT_TRUE(file_contains_substr(metadata_path, "timestep_index=0"));

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);

   cleanup_rc = system(cleanup_cmd);
   (void)cleanup_rc;
}

static void
test_HYPREDRV_PrintLibInfo_PrintExitInfo(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test print functions */
   HYPREDRV_PrintLibInfo(MPI_COMM_SELF, 1);
   HYPREDRV_PrintExitInfo(MPI_COMM_SELF, "test_prog");

   /* Avoid MPI_COMM_NULL here: some MPI implementations abort on MPI_Comm_rank(NULL). */
   HYPREDRV_PrintExitInfo(MPI_COMM_SELF, NULL);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_public_wrappers_and_getters(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   /* ErrorCodeDescribe(0) is a no-op */
   HYPREDRV_ErrorCodeDescribe(0);

   HYPREDRV_PrintSystemInfo(MPI_COMM_SELF);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetSolverPreset(obj, NULL), ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetSolverPreset(obj, "not_a_valid_solver_name"),
                   ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "pcg"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "gmres"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SolverPresetRegister(
                "test_solver_preset",
                "fgmres:\n"
                "  max_iter: 17\n"
                "  krylov_dim: 23\n"
                "  print_level: 0",
                "test solver preset"),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetSolverPreset(obj, "test_solver_preset"), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver_method, SOLVER_FGMRES);
   ASSERT_EQ(obj->iargs->solver.fgmres.max_iter, 17);
   ASSERT_EQ(obj->iargs->solver.fgmres.krylov_dim, 23);

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
      return;
   }

   char yaml_config[2 * PATH_MAX + 384];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  warmup: yes\n"
            "  num_repetitions: 7\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);
   parse_yaml_into_obj(obj, yaml_config);

   int w = -1, nrep = -1, nls = -1, nvar = -1;
   ASSERT_EQ(HYPREDRV_InputArgsGetWarmup(obj, &w), ERROR_NONE);
   ASSERT_EQ(w, 1);
   ASSERT_EQ(HYPREDRV_InputArgsGetNumRepetitions(obj, &nrep), ERROR_NONE);
   ASSERT_EQ(nrep, 7);
   ASSERT_EQ(HYPREDRV_InputArgsGetNumLinearSystems(obj, &nls), ERROR_NONE);
   ASSERT_EQ(nls, 1);
   ASSERT_EQ(HYPREDRV_InputArgsGetNumPreconVariants(obj, &nvar), ERROR_NONE);
   ASSERT_EQ(nvar, 1);

   ASSERT_TRUE(HYPREDRV_InputArgsGetWarmup(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_InputArgsGetNumRepetitions(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_InputArgsGetNumLinearSystems(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_InputArgsGetNumPreconVariants(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* Wrapper success path (also invoked from LinearSystemBuild) */
   ASSERT_EQ(HYPREDRV_LinearSystemReadMatrix(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSystemSetNearNullSpace_public_wrapper(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   HYPRE_IJMatrix mat_nn = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b_nn = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_nn, vec_b_nn);

   HYPRE_Complex nn_val = 1.0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 1, 1, &nn_val), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b_nn), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_nn), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_InputArgsSetPreconVariant_branches(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
      return;
   }

   char yaml_config[2 * PATH_MAX + 640];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  rhs_filename: %s\n"
            "  matrix_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    - print_level: 0\n"
            "    - print_level: 0\n",
            rhs_path, matrix_path);
   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetPreconVariant(obj, -1), ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetPreconVariant(obj, 99), ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 0), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, 1), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_PreconApply_null_matrix_or_vector_args(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);

   HYPRE_IJMatrix saved_A = state->mat_A;
   state->mat_A = NULL;
   ASSERT_TRUE(HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b,
                                    (HYPRE_Vector)state->vec_x) &
               ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   state->mat_A = saved_A;

   ASSERT_TRUE(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_TRUE(HYPREDRV_PreconApply(obj, NULL, (HYPRE_Vector)state->vec_x) &
               ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
static void
test_HYPREDRV_driver_precon_ilu_lifecycle(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  ilu:\n"
            "    type: bj-iluk\n"
            "    fill_level: 0\n"
            "    max_iter: 1\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}
#endif

#if HYPRE_CHECK_MIN_VERSION(22500, 0)
static void
test_HYPREDRV_driver_precon_fsai_lifecycle(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  fsai:\n"
            "    algo_type: 2\n"
            "    max_steps: 5\n"
            "    max_step_size: 5\n"
            "    kap_tolerance: 1.0e-3\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}
#endif

static void
test_HYPREDRV_LinearSolverDestroy_without_precon(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 2\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_NULL(state->precon);

   ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_InputArgsParse_timestep_schedule_negative_cases(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();

   char *bad_empty = CREATE_TEMP_FILE("tmp_ts_empty.txt");
   FILE  *ef         = fopen(bad_empty, "w");
   ASSERT_NOT_NULL(ef);
   fclose(ef);

   char yaml_ts[PATH_MAX + 512];
   snprintf(yaml_ts, sizeof(yaml_ts),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  timestep_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: selectors\n"
            "    selectors:\n"
            "      - basis: timestep\n"
            "        every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            bad_empty);
   {
      char *argv[] = {yaml_ts};
      ASSERT_HAS_FLAG(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_INVALID_VAL);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   char *bad_header = CREATE_TEMP_FILE("tmp_ts_bad_header.txt");
   FILE  *hf         = fopen(bad_header, "w");
   ASSERT_NOT_NULL(hf);
   fprintf(hf, "0\n");
   fclose(hf);

   snprintf(yaml_ts, sizeof(yaml_ts),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  timestep_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: selectors\n"
            "    selectors:\n"
            "      - basis: timestep\n"
            "        every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            bad_header);
   {
      char *argv[] = {yaml_ts};
      ASSERT_HAS_FLAG(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_INVALID_VAL);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   char *bad_row = CREATE_TEMP_FILE("tmp_ts_bad_row.txt");
   FILE  *rf     = fopen(bad_row, "w");
   ASSERT_NOT_NULL(rf);
   fprintf(rf, "1\n");
   fprintf(rf, "0 -1\n");
   fclose(rf);

   snprintf(yaml_ts, sizeof(yaml_ts),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  timestep_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: selectors\n"
            "    selectors:\n"
            "      - basis: timestep\n"
            "        every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            bad_row);
   {
      char *argv[] = {yaml_ts};
      ASSERT_HAS_FLAG(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_INVALID_VAL);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   snprintf(yaml_ts, sizeof(yaml_ts),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  timestep_filename: /nonexistent/path/does_not_exist_ts.txt\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: selectors\n"
            "    selectors:\n"
            "      - basis: timestep\n"
            "        every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n");
   {
      char *argv[] = {yaml_ts};
      ASSERT_HAS_FLAG(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_FILE_NOT_FOUND);
   }
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);

   free(bad_empty);
   free(bad_header);
   free(bad_row);
}

static void
test_HYPREDRV_print_system_timestep_selector_loads_schedule(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();

   char *tmp_ts = CREATE_TEMP_FILE("tmp_ts_selector_basis.txt");
   ASSERT_NOT_NULL(tmp_ts);
   FILE *tf = fopen(tmp_ts, "w");
   ASSERT_NOT_NULL(tf);
   fprintf(tf, "1\n");
   fprintf(tf, "0 0\n");
   fclose(tf);

   char yaml_config[2 * PATH_MAX + 768];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "  exec_policy: host\n"
            "linear_system:\n"
            "  timestep_filename: %s\n"
            "  print_system:\n"
            "    enabled: on\n"
            "    type: selectors\n"
            "    stage: apply\n"
            "    artifacts: [metadata]\n"
            "    output_dir: /tmp\n"
            "    overwrite: on\n"
            "    selectors:\n"
            "      - basis: timestep\n"
            "        every: 1\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 5\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            tmp_ts);

   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_NOT_NULL(obj->precon_reuse_timesteps.starts);
   ASSERT_EQ_SIZE(obj->precon_reuse_timesteps.starts->size, 1);
   ASSERT_EQ(obj->precon_reuse_timesteps.starts->data[0], 0);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   free(tmp_ts);
}

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
test_HYPREDRV_LinearSolverApply_scaling_deferred_and_xref(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   char yaml_config[] =
      "general:\n"
      "  statistics: off\n"
      "  exec_policy: host\n"
      "linear_system:\n"
      "  init_guess_mode: zeros\n"
      "solver:\n"
      "  scaling:\n"
      "    enabled: 1\n"
      "    type: rhs_l2\n"
      "  pcg:\n"
      "    max_iter: 20\n"
      "    relative_tol: 1.0e-12\n"
      "preconditioner:\n"
      "  amg:\n"
      "    print_level: 0\n";

   parse_yaml_into_obj(obj, yaml_config);

   HYPRE_IJMatrix mat_A = create_test_ijmatrix_1x1(4.0);
   HYPRE_IJVector vec_b = create_test_ijvector_1x1(2.0);
   attach_library_scalar_system(obj, mat_A, vec_b);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

   HYPRE_IJVector xref = create_test_ijvector_1x1(0.5);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)xref),
             ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   /* Second apply without setup: scaling deferred apply path in LinearSolverApply */
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

   ASSERT_EQ(HYPRE_IJVectorDestroy(xref), 0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}
#endif /* HYPRE_CHECK_MIN_VERSION(30000, 0) */

static void
test_HYPREDRV_LinearSystemResetInitialGuess_error_cases(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSystemResetInitialGuess(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   /* Test with NULL x vector */
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   state->vec_x = NULL;
   result = HYPREDRV_LinearSystemResetInitialGuess(obj);
   ASSERT_TRUE(result & ERROR_UNKNOWN);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_LinearSystemBuild_error_cases(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   /* Test with NULL object */
   uint32_t result = HYPREDRV_LinearSystemBuild(NULL);
   ASSERT_TRUE(result & ERROR_UNKNOWN_HYPREDRV_OBJ);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_misc_0hit_branches(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);

   char yaml_config[2 * PATH_MAX + 256];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver:\n"
            "  pcg:\n"
            "    max_iter: 1\n"
            "preconditioner:\n"
            "  amg:\n"
            "    print_level: 0\n",
            matrix_path, rhs_path);

   parse_yaml_into_obj(obj, yaml_config);
   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);
   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   /* Exercise solution/RHS getters and a successful solution norm path */
   HYPRE_Complex *sol_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data), ERROR_NONE);
   ASSERT_NOT_NULL(sol_data);

   HYPRE_Complex *rhs_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_NOT_NULL(rhs_data);
   ASSERT_TRUE(rhs_data != sol_data);

   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionValues(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHSValues(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   HYPRE_IJVector saved_b = state->vec_b;
   state->vec_b           = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = saved_b;

   HYPRE_IJVector saved_x = state->vec_x;
   state->vec_x           = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_x = saved_x;

   /* GetSolution / GetRHS: happy path */
   HYPRE_Vector vec_sol_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol_out);

   HYPRE_Vector vec_rhs_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_rhs_out);

   /* GetSolution / GetRHS: NULL output arg */
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* GetSolution when vec_x is NULL */
   HYPRE_IJVector restore_x = state->vec_x;
   state->vec_x             = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_x = restore_x;

   /* GetRHS when vec_b is NULL */
   HYPRE_IJVector restore_b = state->vec_b;
   state->vec_b             = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetRHS(obj, &vec_rhs_out) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = restore_b;

   /* SetSolution: NULL recreates from vec_b */
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   vec_sol_out = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolution(obj, &vec_sol_out), ERROR_NONE);
   ASSERT_NOT_NULL(vec_sol_out);

   /* SetSolution(NULL) with no vec_b → error */
   HYPRE_IJVector restore_b2 = state->vec_b;
   state->vec_b              = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   state->vec_b = restore_b2;

   double ok_norm = 0.0;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionNorm(obj, "L2", &ok_norm), ERROR_NONE);

   /* Delegate printing to linsys */
   ASSERT_EQ(HYPREDRV_LinearSystemPrint(obj), ERROR_NONE);

   /* Cover dofmap family + ReadDofmap no-file default path */
   int dm[2] = {0, 1};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, dm), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 2, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 2, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);

   char *tmp_dof = CREATE_TEMP_FILE("tmp_api_dofmap.txt");
   ASSERT_EQ(HYPREDRV_LinearSystemPrintDofmap(obj, tmp_dof), ERROR_NONE);
   char tmp_dof_rank[PATH_MAX];
   ASSERT_TRUE(strlen(tmp_dof) + strlen(".00000") + 1 < sizeof(tmp_dof_rank));
   snprintf(tmp_dof_rank, sizeof(tmp_dof_rank), "%s.00000", tmp_dof);
   remove(tmp_dof_rank);
   free(tmp_dof);

   /* Cover GetSolutionNorm error branches */
   double norm = 0.0;
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionNorm(obj, NULL, &norm) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();
   ASSERT_TRUE(HYPREDRV_LinearSystemGetSolutionNorm(obj, "L2", NULL) & ERROR_UNKNOWN);
   hypredrv_ErrorCodeResetAll();

   /* Cover AnnotateLevelBegin/End paths */
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 0, "lvl", 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 0, "lvl", 1), ERROR_NONE);

   /* Typed stat getters */
   int    it = -1;
   double t  = -1.0;
   ASSERT_EQ(HYPREDRV_LinearSolverGetNumIter(obj, &it), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSetupTime(obj, &t), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverGetSolveTime(obj, &t), ERROR_NONE);
   hypredrv_ErrorCodeResetAll();

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelBegin(obj, 1, "late", 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_AnnotateLevelEnd(obj, 1, "late", 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_variants(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 512];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  warmup: false\n"
                            "  num_repetitions: 1\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  type: ij\n"
                            "  rhs_filename: %s\n"
                            "  matrix_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    relative_tol: 1.0e-9\n"
                            "    max_iter: 500\n"
                            "    print_level: 3\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: HMIS\n"
                            "        strong_th: 0.25\n"
                            "      interpolation:\n"
                            "        prolongation_type: \"MM-ext+i\"\n"
                            "      relaxation:\n"
                            "        down_type: 16\n"
                            "        down_sweeps: 1\n"
                            "        up_type: 16\n"
                            "        up_sweeps: 1\n"
                            "    - print_level: 1\n"
                            "      coarsening:\n"
                            "        type: PMIS\n"
                            "        strong_th: 0.5\n"
                            "      interpolation:\n"
                            "        prolongation_type: standard\n"
                            "      relaxation:\n"
                            "        down_type: 8\n"
                            "        down_sweeps: 2\n"
                            "        up_type: 8\n"
                            "        up_sweeps: 2\n",
                            rhs_path, matrix_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);

   /* Check that variants were parsed */
   int num_variants = 0;
   ASSERT_EQ(HYPREDRV_InputArgsGetNumPreconVariants(obj, &num_variants), ERROR_NONE);
   ASSERT_EQ(num_variants, 2);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   /* Test setting and using each variant */
   for (int v = 0; v < num_variants; v++)
   {
      ASSERT_EQ(HYPREDRV_InputArgsSetPreconVariant(obj, v), ERROR_NONE);

      /* Reset initial guess */
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

      /* Create and setup */
      ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

      /* Solve */
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);

      /* Destroy */
      ASSERT_EQ(HYPREDRV_PreconDestroy(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverDestroy(obj), ERROR_NONE);
   }

   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_preset_yaml(void)
{
   reset_state();

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   if (!setup_ps3d10pt7_paths(matrix_path, rhs_path))
   {
      return;
   }

   HYPREDRV_t obj = create_initialized_obj();

   char yaml_config[2 * PATH_MAX + 256];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  matrix_filename: %s\n"
                            "  rhs_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 5\n"
                            "preconditioner:\n"
                            "  preset: poisson\n",
                            matrix_path, rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   parse_yaml_into_obj(obj, yaml_config);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_preconditioner_preset_invalid(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);

   hypredrv_ErrorCodeResetAll();
   ASSERT_HAS_FLAG(HYPREDRV_InputArgsSetPreconPreset(obj, "not-a-preset"),
                   ERROR_INVALID_VAL);
   ASSERT_HAS_FLAG(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);

   /* Clear error state so Destroy() isn't masked by the expected preset failure. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   HYPRE_ClearAllErrors();

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_explicit_nonlib_take_ownership(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 == vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(state->mat_M == mat_M);
   ASSERT_TRUE(state->owns_vec_x0);
   ASSERT_TRUE(state->owns_vec_xref);
   ASSERT_TRUE(state->owns_mat_M);
   ASSERT_NOT_NULL(state->vec_x);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_explicit_library_mode_borrow(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 == vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(state->mat_M == mat_M);
   ASSERT_TRUE(!state->owns_vec_x0);
   ASSERT_TRUE(!state->owns_vec_xref);
   ASSERT_TRUE(!state->owns_mat_M);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_ref), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_x0), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_M), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_null_preserve_default_behavior(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);

   HYPRE_IJMatrix mat_A  = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M  = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b  = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_x0 = create_test_ijvector_1x1(4.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)vec_x0),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->vec_x0 != vec_x0);
   ASSERT_TRUE(state->owns_vec_x0);
   ASSERT_TRUE(state->vec_xref == vec_ref);
   ASSERT_TRUE(!state->owns_vec_xref);
   ASSERT_TRUE(state->mat_M == state->mat_A);
   ASSERT_TRUE(!state->owns_mat_M);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_ref), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_x0), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_M), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_b), 0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(mat_A), 0);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
test_HYPREDRV_linear_system_setters_replace_owned_solution_and_xref(void)
{
   reset_state();

   HYPREDRV_t obj = create_initialized_obj();
   parse_minimal_library_yaml(obj);

   HYPRE_IJMatrix mat_A   = create_test_ijmatrix_1x1(1.0);
   HYPRE_IJMatrix mat_M   = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vec_b   = create_test_ijvector_1x1(3.0);
   HYPRE_IJVector vec_ref = create_test_ijvector_1x1(5.0);
   HYPRE_IJVector vec_x_borrow  = create_test_ijvector_1x1(6.0);
   HYPRE_IJVector vec_ref2      = create_test_ijvector_1x1(7.0);

   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)mat_A), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vec_b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)mat_M), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   ASSERT_TRUE(state->owns_vec_x);
   ASSERT_TRUE(state->owns_vec_xref);

   /* Replace owned working solution with a borrowed vector (destroys prior owned x) */
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)vec_x_borrow),
             ERROR_NONE);
   ASSERT_FALSE(state->owns_vec_x);
   ASSERT_TRUE(state->vec_x == vec_x_borrow);

   /* Replacing reference destroys an owned xref distinct from vec_b */
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)vec_ref2),
             ERROR_NONE);
   ASSERT_TRUE(state->vec_xref == vec_ref2);
   ASSERT_TRUE(state->owns_vec_xref);

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   /* Borrowed working solution; remaining matrix/RHS/xref owned by Destroy */
   ASSERT_EQ(HYPRE_IJVectorDestroy(vec_x_borrow), 0);

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

static void
run_hypredrv_lifecycle_and_guards(void)
{
   RUN_TEST(test_HYPREDRV_all_api_init_guard);
   RUN_TEST(test_HYPREDRV_all_api_obj_guard);
   RUN_TEST(test_requires_initialization_guard);
   RUN_TEST(test_initialize_and_finalize_idempotent);
   RUN_TEST(test_HYPREDRV_Create_null_output_pointer);
   RUN_TEST(test_HYPREDRV_default_object_names_are_sequential);
   RUN_TEST(test_HYPREDRV_default_object_name_persists_without_user_name);
   RUN_TEST(test_HYPREDRV_log_level_default_off);
   RUN_TEST(test_HYPREDRV_log_level_enabled_emits_trace);
   RUN_TEST(test_HYPREDRV_log_level_default_object_names_are_numbered);
   RUN_TEST(test_HYPREDRV_log_level_invalid_value_disables_trace);
   RUN_TEST(test_HYPREDRV_log_level_enabled_stays_off_stdout);
   RUN_TEST(test_HYPREDRV_log_stream_stdout_emits_trace_on_stdout);
   RUN_TEST(test_HYPREDRV_log_stream_stdout_stays_off_stderr);
   RUN_TEST(test_HYPREDRV_log_stream_invalid_value_falls_back_to_stderr);
   RUN_TEST(test_HYPREDRV_log_level_input_args_internal_logs_use_object_name);
   RUN_TEST(test_HYPREDRV_log_level_solver_and_linsys_internal_logs_use_object_name);
   RUN_TEST(
      test_HYPREDRV_log_level_solver_and_linsys_internal_logs_use_default_object_name);
   RUN_TEST(test_HYPREDRV_log_level_boundary_api_traces);
   RUN_TEST(test_HYPREDRV_log_level_precon_variant_decisions);
   RUN_TEST(test_HYPREDRV_log_level_avoids_linear_system_ready_duplicate);
   RUN_TEST(test_HYPREDRV_log_level_eigenspectrum_trace_paths);
   RUN_TEST(test_HYPREDRV_Finalize_auto_destroys_live_objects);
}

static void
run_hypredrv_solver_and_reuse(void)
{
   RUN_TEST(test_create_parse_and_destroy);
   RUN_TEST(test_HYPREDRV_PreconCreate_reuse_logic);
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   RUN_TEST(test_HYPREDRV_driver_precon_ilu_lifecycle);
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   RUN_TEST(test_HYPREDRV_driver_precon_fsai_lifecycle);
#endif
   RUN_TEST(test_HYPREDRV_LinearSolverApply_with_xref);
   RUN_TEST(test_HYPREDRV_stats_level_apis);
   RUN_TEST(test_HYPREDRV_InputArgsParse_exec_policy);
   RUN_TEST(test_HYPREDRV_InputArgsParse_gpu_standard_amg_forces_host_exec);
   RUN_TEST(test_HYPREDRV_LinearSystemComputeEigenspectrum_warns_once_when_disabled);
   RUN_TEST(test_HYPREDRV_state_vectors_and_eigspec_error_paths);
   RUN_TEST(test_HYPREDRV_PreconCreate_reuse_logic_variations);
   RUN_TEST(test_HYPREDRV_LinearSolverCreate_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_LinearSolverDestroy_reuse_logic);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_linear_system_ids);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep);
   RUN_TEST(test_HYPREDRV_PreconDestroy_reuse_per_timestep_frequency);
   RUN_TEST(test_HYPREDRV_library_mode_reuse_per_timestep_with_object_annotations);
   RUN_TEST(
      test_HYPREDRV_library_mode_reuse_per_timestep_frequency_with_object_annotations);
   RUN_TEST(test_HYPREDRV_library_mode_adaptive_reuse_rebuilds_after_degradation);
   RUN_TEST(test_HYPREDRV_PreconReuseBuildObservation_and_MarkRebuild_library);
   RUN_TEST(test_HYPREDRV_library_mode_mgr_recreates_precon_on_new_timestep);
   RUN_TEST(test_HYPREDRV_library_mode_mgr_component_reuse_refreshes_selected_handles);
   RUN_TEST(test_HYPREDRV_InputArgsSetPreconVariant_discards_cached_mgr_handles);
   RUN_TEST(test_HYPREDRV_InputArgsSetPreconPreset_discards_cached_mgr_handles);
   RUN_TEST(test_HYPREDRV_library_mode_destroy_prints_named_statistics_summary);
   RUN_TEST(test_HYPREDRV_library_mode_finalize_prints_named_statistics_summary);
   RUN_TEST(test_HYPREDRV_InputArgsParse_loads_lsseq_timesteps_for_print_system);
   RUN_TEST(test_HYPREDRV_InputArgsParse_timestep_schedule_negative_cases);
   RUN_TEST(test_HYPREDRV_print_system_uses_timestep_filename_without_reuse);
   RUN_TEST(test_HYPREDRV_print_system_timestep_selector_loads_schedule);
   RUN_TEST(test_HYPREDRV_print_system_lifecycle_dumps_setup_and_apply);
   RUN_TEST(test_HYPREDRV_print_system_apply_stage_ids_use_current_stats_id);
   RUN_TEST(test_HYPREDRV_print_system_setup_stage_ids_advance_for_library_cycles);
   RUN_TEST(test_HYPREDRV_statistics_filename_routes_stats_to_file);
   RUN_TEST(test_HYPREDRV_statistics_filename_fallbacks_to_stdout_on_open_failure);
   RUN_TEST(test_HYPREDRV_stats_flat_runs_keep_entry_column);
   RUN_TEST(test_HYPREDRV_stats_annotated_runs_switch_to_path_column);
   RUN_TEST(test_HYPREDRV_stats_timestep_file_paths_use_preserved_ids);
   RUN_TEST(test_HYPREDRV_stats_timestep_file_paths_without_reuse_use_path_column);
   RUN_TEST(test_HYPREDRV_stats_path_column_truncates_long_paths);
   RUN_TEST(test_HYPREDRV_LinearSolverApply_error_cases);
   RUN_TEST(test_HYPREDRV_LinearSolverDestroy_without_precon);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   RUN_TEST(test_HYPREDRV_LinearSolverApply_scaling_deferred_and_xref);
#endif
}

static void
run_hypredrv_misc_and_preconditioners(void)
{
   RUN_TEST(test_HYPREDRV_Annotate_functions);
   RUN_TEST(test_HYPREDRV_object_scoped_annotation_isolation);
   RUN_TEST(test_HYPREDRV_PrintLibInfo_PrintExitInfo);
   RUN_TEST(test_HYPREDRV_public_wrappers_and_getters);
   RUN_TEST(test_HYPREDRV_InputArgsSetPreconVariant_branches);
   RUN_TEST(test_HYPREDRV_PreconApply_null_matrix_or_vector_args);
   RUN_TEST(test_HYPREDRV_LinearSystemSetNearNullSpace_public_wrapper);
   RUN_TEST(test_HYPREDRV_LinearSystemResetInitialGuess_error_cases);
   RUN_TEST(test_HYPREDRV_LinearSystemBuild_error_cases);
   RUN_TEST(test_HYPREDRV_misc_0hit_branches);
   RUN_TEST(test_HYPREDRV_preconditioner_variants);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_yaml);
   RUN_TEST(test_HYPREDRV_preconditioner_preset_invalid);
   RUN_TEST(test_HYPREDRV_linear_system_setters_explicit_nonlib_take_ownership);
   RUN_TEST(test_HYPREDRV_linear_system_setters_explicit_library_mode_borrow);
   RUN_TEST(test_HYPREDRV_linear_system_setters_null_preserve_default_behavior);
   RUN_TEST(test_HYPREDRV_linear_system_setters_replace_owned_solution_and_xref);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   run_hypredrv_lifecycle_and_guards();
   run_hypredrv_solver_and_reuse();
   run_hypredrv_misc_and_preconditioners();

   cleanup_temp_files();
   MPI_Finalize();
   return 0;
}
