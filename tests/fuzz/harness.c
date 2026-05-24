/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "fuzz_common.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "HYPREDRV.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "internal/presets.h"
#include "test_helpers.h"

#ifndef FUZZ_MODE
#define FUZZ_MODE FUZZ_MODE_PARSE
#endif

#define FUZZ_MATRIX_MAX_ROWS 10000u
#define FUZZ_MATRIX_MAX_NNZ 100000u
#define FUZZ_VECTOR_MAX_ROWS 100000u
#define FUZZ_MAX_ARGC 32
#define FUZZ_SOLVE_ARGC_CAP 21
#define FUZZ_SOLVE_BASE_ARGC 17
#define FUZZ_SOLVE_FORCED_ARGC 4
#define FUZZ_MULTIPART_MAX_PARTS 3u
#define FUZZ_MULTIPART_HEADER_SIZE 10u
#define FUZZ_MULTIPART_LEN_SIZE 4u

typedef char hypredrv_fuzz_solve_argc_cap_matches_usage
   [(FUZZ_SOLVE_ARGC_CAP == (FUZZ_SOLVE_BASE_ARGC + FUZZ_SOLVE_FORCED_ARGC)) ? 1 : -1];

static int g_runtime_mode = FUZZ_MODE;

static const uint8_t g_multipart_magic[8] = {'H', 'D', 'F', 'Z', 'M', 'P', '0', '1'};

#if defined(HYPREDRV_FUZZ_ENGINE_AFL)
__AFL_FUZZ_INIT();
#endif

static const char *
fuzz_mode_name(int mode)
{
   switch (mode)
   {
      case FUZZ_MODE_PARSE:
         return "parse";
      case FUZZ_MODE_SOLVE:
         return "solve";
      case FUZZ_MODE_LSSEQ:
         return "lsseq";
      case FUZZ_MODE_MATRIX:
         return "matrix";
      case FUZZ_MODE_VECTOR:
         return "vector";
      default:
         return "unknown";
   }
}

int
fuzz_current_mode(void)
{
   return g_runtime_mode;
}

static void
fuzz_warmup_before_forkserver(void)
{
   static const uint8_t yaml[] = "solver: pcg\npreconditioner: amg\n";

   if (g_runtime_mode == FUZZ_MODE_PARSE || g_runtime_mode == FUZZ_MODE_SOLVE)
   {
      fprintf(stderr, "hypredrive fuzz: warming %s mode before AFL forkserver\n",
              fuzz_mode_name(g_runtime_mode));
      (void)fuzz_one(yaml, sizeof(yaml) - 1u);
   }
}

void
fuzz_reset_error_state(void)
{
   hypredrv_ErrorStateReset();
}

void
fuzz_mpi_init_once(int *argc, char ***argv)
{
   int initialized = 0;
   MPI_Initialized(&initialized);
   if (!initialized)
   {
      int provided = 0;
      MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
   }

   (void)HYPREDRV_Initialize();
}

void
fuzz_runtime_mode_override(void)
{
#if defined(HYPREDRV_FUZZ_ENGINE_REPLAY)
   const char *mode = getenv("HYPREDRV_FUZZ_MODE");
   if (!mode || mode[0] == '\0')
   {
      return;
   }
   if (strcmp(mode, "parse") == 0)
   {
      g_runtime_mode = FUZZ_MODE_PARSE;
   }
   else if (strcmp(mode, "solve") == 0)
   {
      g_runtime_mode = FUZZ_MODE_SOLVE;
   }
   else if (strcmp(mode, "lsseq") == 0)
   {
      g_runtime_mode = FUZZ_MODE_LSSEQ;
   }
   else if (strcmp(mode, "matrix") == 0)
   {
      g_runtime_mode = FUZZ_MODE_MATRIX;
   }
   else if (strcmp(mode, "vector") == 0)
   {
      g_runtime_mode = FUZZ_MODE_VECTOR;
   }
#endif
}

void
fuzz_require_startup_assets(void)
{
   if (g_runtime_mode == FUZZ_MODE_SOLVE)
   {
      TEST_REQUIRE_FILE(HYPREDRIVE_SOURCE_DIR "/data/ps3d10pt7/np1/IJ.out.A.00000");
      TEST_REQUIRE_FILE(HYPREDRIVE_SOURCE_DIR "/data/ps3d10pt7/np1/IJ.out.b.00000");
   }
}

static int
fuzz_tmp_template(char *path, size_t path_size)
{
   const char *tmpdir = getenv("TMPDIR");
   if (!tmpdir || tmpdir[0] == '\0')
   {
      tmpdir = "/tmp";
   }

   int written = snprintf(path, path_size, "%s/hypredrv-fuzz-XXXXXX", tmpdir);
   return written >= 0 && (size_t)written < path_size;
}

int
fuzz_write_temp_file(const uint8_t *data, size_t size, const char *suffix, char *path,
                     size_t path_size)
{
   int fd = -1;

   if (!fuzz_tmp_template(path, path_size))
   {
      return 0;
   }

   fd = mkstemp(path);
   if (fd < 0)
   {
      return 0;
   }

   if (suffix && suffix[0] != '\0')
   {
      char suffixed[1024];
      int  written = snprintf(suffixed, sizeof(suffixed), "%s%s", path, suffix);
      close(fd);
      if (written < 0 || (size_t)written >= sizeof(suffixed))
      {
         remove(path);
         return 0;
      }
      if (rename(path, suffixed) != 0)
      {
         remove(path);
         return 0;
      }
      fd = open(suffixed, O_WRONLY | O_TRUNC, 0600);
      if (fd < 0)
      {
         remove(suffixed);
         return 0;
      }
      if ((size_t)snprintf(path, path_size, "%s", suffixed) >= path_size)
      {
         close(fd);
         remove(suffixed);
         return 0;
      }
   }

   if (size > 0 && write(fd, data, size) != (ssize_t)size)
   {
      close(fd);
      remove(path);
      return 0;
   }

   close(fd);
   return 1;
}

int
fuzz_make_temp_prefix(char *prefix, size_t prefix_size)
{
   char dir[1024];

   if (!fuzz_tmp_template(dir, sizeof(dir)))
   {
      return 0;
   }
   if (!mkdtemp(dir))
   {
      return 0;
   }
   if ((size_t)snprintf(prefix, prefix_size, "%s/input", dir) >= prefix_size)
   {
      rmdir(dir);
      return 0;
   }
   return 1;
}

static void
fuzz_remove_temp_prefix_dir(const char *prefix)
{
   char  dir[1024];
   char *slash = NULL;

   if (!prefix || (size_t)snprintf(dir, sizeof(dir), "%s", prefix) >= sizeof(dir))
   {
      return;
   }

   slash = strrchr(dir, '/');
   if (!slash)
   {
      return;
   }
   *slash = '\0';
   rmdir(dir);
}

static int
fuzz_argv_add_copy(char **argv, int *argc, int argc_cap, char **copies, int *num_copies,
                   const char *value)
{
   char *copy = NULL;

   if (*argc >= argc_cap)
   {
      return 0;
   }

   copy = strdup(value ? value : "");
   if (!copy)
   {
      return 0;
   }

   argv[(*argc)++]         = copy;
   copies[(*num_copies)++] = copy;
   return 1;
}

static void
fuzz_argv_free_copies(char **copies, int num_copies)
{
   for (int i = 0; i < num_copies; i++)
   {
      free(copies[i]);
   }
}

int
fuzz_read_file(const char *path, uint8_t **data_ptr, size_t *size_ptr)
{
   FILE    *fp   = NULL;
   long     size = 0;
   uint8_t *buf  = NULL;

   *data_ptr = NULL;
   *size_ptr = 0;

   fp = fopen(path, "rb");
   if (!fp)
   {
      return 0;
   }
   if (fseek(fp, 0, SEEK_END) != 0)
   {
      fclose(fp);
      return 0;
   }
   size = ftell(fp);
   if (size < 0 || (unsigned long)size > FUZZ_MAX_INPUT)
   {
      fclose(fp);
      return 0;
   }
   rewind(fp);

   buf = (uint8_t *)malloc((size_t)size + 1u);
   if (!buf)
   {
      fclose(fp);
      return 0;
   }
   if (size > 0 && fread(buf, 1, (size_t)size, fp) != (size_t)size)
   {
      free(buf);
      fclose(fp);
      return 0;
   }
   fclose(fp);
   buf[size] = 0;

   *data_ptr = buf;
   *size_ptr = (size_t)size;
   return 1;
}

static int
fuzz_build_nul_argv(char *scratch, size_t size, char **argv, int argv_cap)
{
   char *end = scratch + size;
   int   argc;

   if (argv_cap <= 0)
   {
      return 0;
   }

   argc    = 1;
   argv[0] = scratch;

   for (char *cur = scratch; cur < end && argc < argv_cap;)
   {
      if (*cur != '\0')
      {
         cur++;
         continue;
      }

      cur++;
      while (cur < end && *cur == '\0')
      {
         cur++;
      }
      if (cur < end)
      {
         argv[argc++] = cur;
      }
   }

   return argc;
}

static uint32_t
fuzz_read_u32_le(const uint8_t *data)
{
   return ((uint32_t)data[0]) | ((uint32_t)data[1] << 8u) | ((uint32_t)data[2] << 16u) |
          ((uint32_t)data[3] << 24u);
}

static int
fuzz_matrix_part_within_limits(const uint8_t *data, size_t size)
{
   uint64_t header[11];

   if (size < sizeof(header))
   {
      return 1;
   }

   memcpy(header, data, sizeof(header));
   if (header[8] >= header[7])
   {
      uint64_t row_span = header[8] - header[7];
      if (row_span >= FUZZ_MATRIX_MAX_ROWS || header[6] > FUZZ_MATRIX_MAX_NNZ)
      {
         return 0;
      }
   }

   return 1;
}

static int
fuzz_vector_part_within_limits(const uint8_t *data, size_t size)
{
   uint64_t header[8];

   if (size < sizeof(header))
   {
      return 1;
   }

   memcpy(header, data, sizeof(header));
   return header[5] <= FUZZ_VECTOR_MAX_ROWS;
}

static int
fuzz_decode_multipart_envelope(const uint8_t *data, size_t size, const uint8_t **parts,
                               size_t *part_sizes, uint64_t *nparts,
                               HYPRE_MemoryLocation *memory_location)
{
   uint64_t count        = 0;
   size_t   lengths_base = 0;
   size_t   payload      = 0;

   if (size < FUZZ_MULTIPART_HEADER_SIZE + FUZZ_MULTIPART_LEN_SIZE ||
       memcmp(data, g_multipart_magic, sizeof(g_multipart_magic)) != 0)
   {
      return 0;
   }

   count        = 1u + (uint64_t)(data[8] % FUZZ_MULTIPART_MAX_PARTS);
   lengths_base = FUZZ_MULTIPART_HEADER_SIZE;
   payload      = lengths_base + (size_t)count * FUZZ_MULTIPART_LEN_SIZE;
   if (payload > size)
   {
      return 0;
   }

   for (uint64_t i = 0; i < count; i++)
   {
      size_t part_size = (size_t)fuzz_read_u32_le(data + lengths_base +
                                                  (size_t)i * FUZZ_MULTIPART_LEN_SIZE);
      if (part_size > size - payload)
      {
         return 0;
      }
      parts[i]      = data + payload;
      part_sizes[i] = part_size;
      payload += part_size;
   }

   *nparts = count;
#ifdef HYPRE_USING_GPU
   *memory_location = (data[9] & 1u) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
#else
   (void)data;
   *memory_location = HYPRE_MEMORY_HOST;
#endif
   return 1;
}

static int
fuzz_write_multipart_files(const char *prefix, const uint8_t **parts,
                           const size_t *part_sizes, uint64_t nparts)
{
   char part_path[1100];

   for (uint64_t part = 0; part < nparts; part++)
   {
      int fd = -1;
      if (snprintf(part_path, sizeof(part_path), "%s.%05u.bin", prefix, (unsigned)part) <
             0 ||
          strlen(part_path) >= sizeof(part_path))
      {
         return 0;
      }

      fd = open(part_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (fd < 0)
      {
         return 0;
      }
      if (part_sizes[part] > 0 &&
          write(fd, parts[part], part_sizes[part]) != (ssize_t)part_sizes[part])
      {
         close(fd);
         return 0;
      }
      close(fd);
   }

   return 1;
}

static void
fuzz_remove_multipart_files(const char *prefix, uint64_t nparts)
{
   char part_path[1100];

   for (uint64_t part = 0; part < nparts; part++)
   {
      if (snprintf(part_path, sizeof(part_path), "%s.%05u.bin", prefix, (unsigned)part) >=
             0 &&
          strlen(part_path) < sizeof(part_path))
      {
         remove(part_path);
      }
   }
}

static void
fuzz_parse_yaml_text(const uint8_t *data, size_t size)
{
   char      *scratch = NULL;
   char      *argv[FUZZ_MAX_ARGC];
   int        argc = 0;
   HYPREDRV_t obj  = NULL;

   scratch = (char *)malloc(size + 1u);
   if (!scratch)
   {
      return;
   }
   memcpy(scratch, data, size);
   scratch[size] = '\0';
   argc          = fuzz_build_nul_argv(scratch, size, argv, FUZZ_MAX_ARGC);

   if (HYPREDRV_Create(MPI_COMM_SELF, &obj) == ERROR_NONE && obj)
   {
      HYPREDRV_SetLibraryMode(obj);
      (void)HYPREDRV_InputArgsParse(argc, argv, obj);
      HYPREDRV_Destroy(&obj);
   }

   free(scratch);
}

static void
fuzz_solve_yaml_text_once(const uint8_t *data, size_t size, int force_solver)
{
   char      *scratch = NULL;
   char      *argv[FUZZ_SOLVE_ARGC_CAP];
   char      *argv_copies[FUZZ_SOLVE_ARGC_CAP];
   const int  argv_cap        = (int)(sizeof(argv) / sizeof(argv[0]));
   int        num_argv_copies = 0;
   int        argc            = 0;
   HYPREDRV_t obj             = NULL;

   scratch = (char *)malloc(size + 1u);
   if (!scratch)
   {
      return;
   }
   memcpy(scratch, data, size);
   scratch[size] = '\0';

   argv[argc++] = scratch;
#define FUZZ_ARGV_ADD_COPY(value)                                                   \
   do                                                                               \
   {                                                                                \
      if (!fuzz_argv_add_copy(argv, &argc, argv_cap, argv_copies, &num_argv_copies, \
                              (value)))                                             \
      {                                                                             \
         goto cleanup;                                                              \
      }                                                                             \
   } while (0)

   FUZZ_ARGV_ADD_COPY("--linear_system:matrix_filename");
   FUZZ_ARGV_ADD_COPY(HYPREDRIVE_SOURCE_DIR "/data/ps3d10pt7/np1/IJ.out.A");
   FUZZ_ARGV_ADD_COPY("--linear_system:rhs_filename");
   FUZZ_ARGV_ADD_COPY(HYPREDRIVE_SOURCE_DIR "/data/ps3d10pt7/np1/IJ.out.b");
   FUZZ_ARGV_ADD_COPY("--linear_system:dirname");
   FUZZ_ARGV_ADD_COPY("");
   FUZZ_ARGV_ADD_COPY("--linear_system:init_suffix");
   FUZZ_ARGV_ADD_COPY("0");
   FUZZ_ARGV_ADD_COPY("--linear_system:last_suffix");
   FUZZ_ARGV_ADD_COPY("0");
   FUZZ_ARGV_ADD_COPY("--linear_system:dofmap_filename");
   FUZZ_ARGV_ADD_COPY("");
   FUZZ_ARGV_ADD_COPY("--linear_system:sequence_filename");
   FUZZ_ARGV_ADD_COPY("");
   FUZZ_ARGV_ADD_COPY("--general:num_repetitions");
   FUZZ_ARGV_ADD_COPY("1");
   if (force_solver)
   {
      switch ((data[0] ^ data[size - 1u] ^ (uint8_t)size) % 5u)
      {
         case 0:
            FUZZ_ARGV_ADD_COPY("--solver");
            FUZZ_ARGV_ADD_COPY("gmres");
            FUZZ_ARGV_ADD_COPY("--preconditioner");
            FUZZ_ARGV_ADD_COPY("ilu");
            break;
         case 1:
            FUZZ_ARGV_ADD_COPY("--solver");
            FUZZ_ARGV_ADD_COPY("bicgstab");
            FUZZ_ARGV_ADD_COPY("--preconditioner");
            FUZZ_ARGV_ADD_COPY("amg");
            break;
         case 2:
            FUZZ_ARGV_ADD_COPY("--solver");
            FUZZ_ARGV_ADD_COPY("bicgstab");
            FUZZ_ARGV_ADD_COPY("--preconditioner");
            FUZZ_ARGV_ADD_COPY("ilu");
            break;
         case 3:
            FUZZ_ARGV_ADD_COPY("--solver");
            FUZZ_ARGV_ADD_COPY("fgmres");
            FUZZ_ARGV_ADD_COPY("--preconditioner");
            FUZZ_ARGV_ADD_COPY("amg");
            break;
         default:
            FUZZ_ARGV_ADD_COPY("--solver");
            FUZZ_ARGV_ADD_COPY("pcg");
            FUZZ_ARGV_ADD_COPY("--preconditioner");
            FUZZ_ARGV_ADD_COPY("amg");
            break;
      }
   }
#undef FUZZ_ARGV_ADD_COPY

   if (HYPREDRV_Create(MPI_COMM_SELF, &obj) == ERROR_NONE && obj)
   {
      HYPREDRV_SetLibraryMode(obj);
      if (HYPREDRV_InputArgsParse(argc, argv, obj) == ERROR_NONE &&
          HYPREDRV_LinearSystemBuild(obj) == ERROR_NONE)
      {
         (void)HYPREDRV_InputArgsSetPreconVariant(obj, 0);
         if (HYPREDRV_PreconCreate(obj) == ERROR_NONE &&
             HYPREDRV_LinearSolverCreate(obj) == ERROR_NONE &&
             HYPREDRV_LinearSolverSetup(obj) == ERROR_NONE)
         {
            (void)HYPREDRV_LinearSolverApply(obj);
         }
         (void)HYPREDRV_PreconDestroy(obj);
         (void)HYPREDRV_LinearSolverDestroy(obj);
      }
      HYPREDRV_Destroy(&obj);
   }

cleanup:
   fuzz_argv_free_copies(argv_copies, num_argv_copies);
   free(scratch);
}

static int
fuzz_input_mentions_solver(const uint8_t *data, size_t size)
{
   static const char solver_key[] = "solver";

   for (size_t i = 0; i < size; i++)
   {
      if (i + sizeof(solver_key) - 1u <= size &&
          memcmp(data + i, solver_key, sizeof(solver_key) - 1u) == 0)
      {
         return 1;
      }
   }

   return 0;
}

static void
fuzz_solve_yaml_text(const uint8_t *data, size_t size)
{
   fuzz_solve_yaml_text_once(data, size, 0);
   if (!fuzz_input_mentions_solver(data, size))
   {
      fuzz_reset_error_state();
      fuzz_solve_yaml_text_once(data, size, 1);
   }
}

static void
fuzz_lsseq_read_system(const char *path, int ls_id, int read_dofmap)
{
   HYPRE_IJMatrix mat    = NULL;
   HYPRE_IJVector rhs    = NULL;
   IntArray      *dofmap = NULL;

   fuzz_reset_error_state();
   (void)hypredrv_LSSeqReadMatrix(MPI_COMM_SELF, path, ls_id, HYPRE_MEMORY_HOST, &mat);
   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }

   fuzz_reset_error_state();
   (void)hypredrv_LSSeqReadRHS(MPI_COMM_SELF, path, ls_id, HYPRE_MEMORY_HOST, &rhs);
   if (rhs)
   {
      HYPRE_IJVectorDestroy(rhs);
   }

   if (read_dofmap)
   {
      fuzz_reset_error_state();
      (void)hypredrv_LSSeqReadDofmap(MPI_COMM_SELF, path, ls_id, &dofmap);
      hypredrv_IntArrayDestroy(&dofmap);
   }
}

static void
fuzz_lsseq_file(const uint8_t *data, size_t size)
{
#if defined(HYPREDRV_FUZZ_HAS_LSSEQ)
   char path[1024];
   int  num_systems   = 0;
   int  num_patterns  = 0;
   int  has_dofmap    = 0;
   int  has_timesteps = 0;

   if (!fuzz_write_temp_file(data, size, ".bin", path, sizeof(path)))
   {
      return;
   }

   (void)hypredrv_LSSeqReadSummary(path, NULL, NULL, NULL, NULL);
   fuzz_reset_error_state();
   {
      char  *payload      = NULL;
      size_t payload_size = 0;
      (void)hypredrv_LSSeqReadInfo(path, &payload, &payload_size);
      free(payload);
      fuzz_reset_error_state();
   }
   {
      IntArray *ids    = NULL;
      IntArray *starts = NULL;
      (void)hypredrv_LSSeqReadTimestepsWithIds(path, &ids, &starts);
      hypredrv_IntArrayDestroy(&ids);
      hypredrv_IntArrayDestroy(&starts);
      fuzz_reset_error_state();
   }
   fuzz_lsseq_read_system(path, 0, 1);

   if (hypredrv_LSSeqReadSummary(path, &num_systems, &num_patterns, &has_dofmap,
                                 &has_timesteps))
   {
      char  *payload      = NULL;
      size_t payload_size = 0;
      (void)hypredrv_LSSeqReadInfo(path, &payload, &payload_size);
      free(payload);
      fuzz_reset_error_state();

      if (has_timesteps)
      {
         IntArray *ids    = NULL;
         IntArray *starts = NULL;
         (void)hypredrv_LSSeqReadTimestepsWithIds(path, &ids, &starts);
         hypredrv_IntArrayDestroy(&ids);
         hypredrv_IntArrayDestroy(&starts);
         fuzz_reset_error_state();
         (void)hypredrv_LSSeqReadTimesteps(path, &starts);
         hypredrv_IntArrayDestroy(&starts);
         fuzz_reset_error_state();
      }

      if (num_systems > 0)
      {
         fuzz_lsseq_read_system(path, 0, has_dofmap);
         if (num_systems > 1)
         {
            fuzz_lsseq_read_system(path, num_systems - 1, has_dofmap);
         }
         fuzz_lsseq_read_system(path, -1, has_dofmap);
         fuzz_lsseq_read_system(path, num_systems, has_dofmap);
      }
   }

   remove(path);
#else
   (void)data;
   (void)size;
#endif
}

static void
fuzz_matrix_file(const uint8_t *data, size_t size)
{
   char                 prefix[1024];
   char                 part_path[1100];
   HYPRE_IJMatrix       mat = NULL;
   int                  fd  = -1;
   const uint8_t       *parts[FUZZ_MULTIPART_MAX_PARTS];
   size_t               part_sizes[FUZZ_MULTIPART_MAX_PARTS];
   uint64_t             nparts          = 0;
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;

   if (fuzz_decode_multipart_envelope(data, size, parts, part_sizes, &nparts,
                                      &memory_location))
   {
      for (uint64_t part = 0; part < nparts; part++)
      {
         if (!fuzz_matrix_part_within_limits(parts[part], part_sizes[part]))
         {
            return;
         }
      }
      if (!fuzz_make_temp_prefix(prefix, sizeof(prefix)))
      {
         return;
      }
      if (!fuzz_write_multipart_files(prefix, parts, part_sizes, nparts))
      {
         fuzz_remove_multipart_files(prefix, nparts);
         fuzz_remove_temp_prefix_dir(prefix);
         return;
      }
      hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, nparts, memory_location,
                                           &mat);
      if (mat)
      {
         HYPRE_IJMatrixDestroy(mat);
      }
      fuzz_remove_multipart_files(prefix, nparts);
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }

   if (!fuzz_matrix_part_within_limits(data, size))
   {
      return;
   }

   if (!fuzz_make_temp_prefix(prefix, sizeof(prefix)))
   {
      return;
   }
   if (snprintf(part_path, sizeof(part_path), "%s.00000.bin", prefix) < 0 ||
       strlen(part_path) >= sizeof(part_path))
   {
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }

   fd = open(part_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (fd < 0)
   {
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }
   if (size > 0 && write(fd, data, size) != (ssize_t)size)
   {
      close(fd);
      remove(part_path);
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }
   close(fd);

   hypredrv_IJMatrixReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST,
                                        &mat);
   if (mat)
   {
      HYPRE_IJMatrixDestroy(mat);
   }
   remove(part_path);
   fuzz_remove_temp_prefix_dir(prefix);
}

static void
fuzz_vector_file(const uint8_t *data, size_t size)
{
   char                 prefix[1024];
   char                 part_path[1100];
   HYPRE_IJVector       vec = NULL;
   int                  fd  = -1;
   const uint8_t       *parts[FUZZ_MULTIPART_MAX_PARTS];
   size_t               part_sizes[FUZZ_MULTIPART_MAX_PARTS];
   uint64_t             nparts          = 0;
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;

   if (fuzz_decode_multipart_envelope(data, size, parts, part_sizes, &nparts,
                                      &memory_location))
   {
      for (uint64_t part = 0; part < nparts; part++)
      {
         if (!fuzz_vector_part_within_limits(parts[part], part_sizes[part]))
         {
            return;
         }
      }
      if (!fuzz_make_temp_prefix(prefix, sizeof(prefix)))
      {
         return;
      }
      if (!fuzz_write_multipart_files(prefix, parts, part_sizes, nparts))
      {
         fuzz_remove_multipart_files(prefix, nparts);
         fuzz_remove_temp_prefix_dir(prefix);
         return;
      }
      hypredrv_IJVectorReadMultipartBinary(prefix, MPI_COMM_SELF, nparts, memory_location,
                                           &vec);
      if (vec)
      {
         HYPRE_IJVectorDestroy(vec);
      }
      fuzz_remove_multipart_files(prefix, nparts);
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }

   if (!fuzz_vector_part_within_limits(data, size))
   {
      return;
   }

   if (!fuzz_make_temp_prefix(prefix, sizeof(prefix)))
   {
      return;
   }
   if (snprintf(part_path, sizeof(part_path), "%s.00000.bin", prefix) < 0 ||
       strlen(part_path) >= sizeof(part_path))
   {
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }

   fd = open(part_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (fd < 0)
   {
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }
   if (size > 0 && write(fd, data, size) != (ssize_t)size)
   {
      close(fd);
      remove(part_path);
      fuzz_remove_temp_prefix_dir(prefix);
      return;
   }
   close(fd);

   hypredrv_IJVectorReadMultipartBinary(prefix, MPI_COMM_SELF, 1, HYPRE_MEMORY_HOST,
                                        &vec);
   if (vec)
   {
      HYPRE_IJVectorDestroy(vec);
   }
   remove(part_path);
   fuzz_remove_temp_prefix_dir(prefix);
}

int
fuzz_one(const uint8_t *data, size_t size)
{
   if (!data || size == 0 || size > FUZZ_MAX_INPUT)
   {
      return 0;
   }

   fuzz_reset_error_state();

   switch (g_runtime_mode)
   {
      case FUZZ_MODE_PARSE:
         fuzz_parse_yaml_text(data, size);
         break;
      case FUZZ_MODE_SOLVE:
         fuzz_solve_yaml_text(data, size);
         break;
      case FUZZ_MODE_LSSEQ:
         fuzz_lsseq_file(data, size);
         break;
      case FUZZ_MODE_MATRIX:
         fuzz_matrix_file(data, size);
         break;
      case FUZZ_MODE_VECTOR:
         fuzz_vector_file(data, size);
         break;
      default:
         break;
   }

   hypredrv_PresetFreeUserPresets();
   fuzz_reset_error_state();
   return 0;
}

#if defined(HYPREDRV_FUZZ_ENGINE_LIBFUZZER)
int
LLVMFuzzerInitialize(int *argc, char ***argv)
{
   fuzz_mpi_init_once(argc, argv);
   fuzz_require_startup_assets();
   return 0;
}

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
   return fuzz_one(data, size);
}
#elif defined(HYPREDRV_FUZZ_ENGINE_AFL)
int
main(int argc, char **argv)
{
#ifdef __AFL_HAVE_MANUAL_CONTROL
   unsigned char *buf = NULL;
#else
   static uint8_t buf[FUZZ_MAX_INPUT];
   ssize_t        nread = 0;
#endif

   fuzz_mpi_init_once(&argc, &argv);
   fuzz_require_startup_assets();
   fuzz_warmup_before_forkserver();

   __AFL_INIT();
#ifdef __AFL_HAVE_MANUAL_CONTROL
   buf = __AFL_FUZZ_TESTCASE_BUF;
   while (__AFL_LOOP(1000))
   {
      int len = __AFL_FUZZ_TESTCASE_LEN;
      if (len < 0)
      {
         len = 0;
      }
      (void)fuzz_one(buf, (size_t)len);
   }
#else
   nread = read(STDIN_FILENO, buf, sizeof(buf));
   if (nread < 0)
   {
      nread = 0;
   }
   (void)fuzz_one(buf, (size_t)nread);
#endif

   return 0;
}
#else
int
main(int argc, char **argv)
{
   int status = 0;

   fuzz_mpi_init_once(&argc, &argv);
   fuzz_runtime_mode_override();
   fuzz_require_startup_assets();

   if (argc <= 1)
   {
      fprintf(stderr, "usage: %s <seed> [seed ...]\n", argv[0]);
      return 0;
   }

   for (int i = 1; i < argc; i++)
   {
      uint8_t *data = NULL;
      size_t   size = 0;
      if (!fuzz_read_file(argv[i], &data, &size))
      {
         fprintf(stderr, "Could not read replay input for %s mode: %s\n",
                 fuzz_mode_name(g_runtime_mode), argv[i]);
         status = 1;
         continue;
      }
      (void)fuzz_one(data, size);
      free(data);
   }

   (void)HYPREDRV_Finalize();
   return status;
}
#endif
