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

#include "internal/args.h"
#include "internal/error.h"
#include "test_helpers.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

static void
test_InputArgsRead_mpi_broadcast(void)
{
   int nprocs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (nprocs < 2)
   {
      return;
   }

   char path[PATH_MAX];
   snprintf(path, sizeof(path), "%s/test_args_mpi_config.yaml", HYPREDRIVE_SOURCE_DIR);

   if (rank == 0)
   {
      FILE *fp = fopen(path, "w");
      ASSERT_NOT_NULL(fp);
      fprintf(fp,
              "solver:\n"
              "  pcg:\n"
              "    max_iter: 10\n"
              "preconditioner:\n"
              "  amg:\n"
              "    print_level: 0\n");
      fclose(fp);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   int    base_indent = -1;
   char  *text        = NULL;
   char  *argv[]      = {path};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsRead(MPI_COMM_WORLD, path, &base_indent, &text);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(text);
   ASSERT_NOT_NULL(strstr(text, "max_iter: 10"));
   free(text);

   input_args *iargs = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_WORLD, 0, 1, argv, &iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(iargs);
   hypredrv_InputArgsDestroy(&iargs);

   if (rank == 0)
   {
      remove(path);
   }
   MPI_Barrier(MPI_COMM_WORLD);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   RUN_TEST(test_InputArgsRead_mpi_broadcast);
   MPI_Finalize();
   return 0;
}
