/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "test_helpers.h"

static int
file_count_substr(const char *path, const char *needle)
{
   FILE *fp = fopen(path, "r");
   if (!fp)
   {
      return 0;
   }

   int    count      = 0;
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
   int written = 0;

   if (!out || out_size == 0 || !left || !right)
   {
      return 0;
   }

   written = snprintf(out, out_size, "%s/%s", left, right);
   return written >= 0 && (size_t)written < out_size;
}

static void
test_dump_dir_is_shared_across_ranks(void)
{
   int myid = 0;
   int nprocs = 0;
   int root_pid = 0;
   char outdir[PATH_MAX];
   char cleanup_cmd[PATH_MAX + 32];
   LS_args args;
   IntArray *dofmap = NULL;
   PrintSystemContext ctx;

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   if (myid == 0)
   {
      root_pid = getpid();
   }
   MPI_Bcast(&root_pid, 1, MPI_INT, 0, MPI_COMM_WORLD);

   snprintf(outdir, sizeof(outdir), "/tmp/hypredrive_dump_mpi_%d", root_pid);
   snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", outdir);

   if (myid == 0)
   {
      int cleanup_rc = system(cleanup_cmd);
      (void)cleanup_rc;
   }
   MPI_Barrier(MPI_COMM_WORLD);

   hypredrv_LinearSystemSetDefaultArgs(&args);
   args.print_system.enabled    = 1;
   args.print_system.type       = PRINT_SYSTEM_TYPE_ALL;
   args.print_system.stage_mask = PRINT_SYSTEM_STAGE_BUILD_BIT;
   args.print_system.artifacts =
      PRINT_SYSTEM_ARTIFACT_DOFMAP | PRINT_SYSTEM_ARTIFACT_METADATA;
   ASSERT_TRUE(strlen(outdir) < sizeof(args.print_system.output_dir));
   memcpy(args.print_system.output_dir, outdir, strlen(outdir) + 1);
   args.print_system.overwrite = 0;

   int dm[2] = {myid, myid + 10};
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, 2, dm, &dofmap);

   memset(&ctx, 0, sizeof(ctx));
   ctx.stage          = PRINT_SYSTEM_STAGE_BUILD;
   ctx.system_index   = 4;
   ctx.timestep_index = 2;
   ctx.stats_ls_id    = 4;
   for (int level = 0; level < STATS_MAX_LEVELS; level++)
   {
      ctx.level_ids[level] = -1;
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_LinearSystemDumpScheduled(MPI_COMM_WORLD, &args, NULL, NULL, NULL, NULL,
                                      NULL, NULL, dofmap, &ctx, "obj-mpi");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   MPI_Barrier(MPI_COMM_WORLD);

   if (myid == 0)
   {
      char object_dir[PATH_MAX];
      char dump_dir[PATH_MAX];
      char metadata_file[PATH_MAX];
      char systems_index[PATH_MAX];
      char unexpected_dir[PATH_MAX];

      ASSERT_TRUE(path_join2(object_dir, sizeof(object_dir), outdir, "obj-mpi"));
      ASSERT_TRUE(path_join2(dump_dir, sizeof(dump_dir), object_dir, "ls_00000"));
      ASSERT_TRUE(path_join2(metadata_file, sizeof(metadata_file), dump_dir,
                             "metadata.txt"));
      ASSERT_TRUE(path_join2(systems_index, sizeof(systems_index), object_dir,
                             "systems_index.txt"));
      ASSERT_TRUE(path_join2(unexpected_dir, sizeof(unexpected_dir), object_dir,
                             "ls_00001"));

      ASSERT_TRUE(access(metadata_file, F_OK) == 0);
      ASSERT_TRUE(access(systems_index, F_OK) == 0);
      ASSERT_TRUE(access(unexpected_dir, F_OK) != 0);
      ASSERT_EQ(file_count_substr(systems_index, "ls_00000"), 1);

      for (int rank = 0; rank < nprocs; rank++)
      {
         char rank_leaf[32];
         char dofmap_file[PATH_MAX];

         snprintf(rank_leaf, sizeof(rank_leaf), "dofmap.out.%05d", rank);
         ASSERT_TRUE(path_join2(dofmap_file, sizeof(dofmap_file), dump_dir, rank_leaf));
         ASSERT_TRUE(access(dofmap_file, F_OK) == 0);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   hypredrv_IntArrayDestroy(&dofmap);
   hypredrv_PrintSystemDestroyArgs(&args.print_system);

   if (myid == 0)
   {
      int cleanup_rc = system(cleanup_cmd);
      (void)cleanup_rc;
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
   (void)argv;

   MPI_Init(&argc, &argv);
   TEST_HYPRE_INIT();

   test_dump_dir_is_shared_across_ranks();

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
