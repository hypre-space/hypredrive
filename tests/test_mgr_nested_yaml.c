/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "args.h"
#include "error.h"
#include "mgr.h"
#include "nested_krylov.h"
#include "test_helpers.h"

static void
test_mgr_nested_yaml_parse(void)
{
   input_args *iargs = NULL;
   char        yaml_path[4096];

   snprintf(yaml_path, sizeof(yaml_path),
            "%s/tests/fixtures/ex3-mgr_Frelax_gmres.yml", HYPREDRIVE_SOURCE_DIR);

   char *argv[] = {yaml_path};
   int   argc   = 1;

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, true, argc, argv, &iargs);
   ASSERT_FALSE(ErrorCodeActive());
   ASSERT_NOT_NULL(iargs);
   ASSERT_EQ(iargs->precon_method, PRECON_MGR);
   ASSERT_TRUE(iargs->precon.mgr.level[1].f_relaxation.use_krylov);
   ASSERT_NOT_NULL(iargs->precon.mgr.level[1].f_relaxation.krylov);
   ASSERT_EQ(iargs->precon.mgr.level[1].f_relaxation.krylov->solver_method, SOLVER_GMRES);

   InputArgsDestroy(&iargs);
}

static void
test_mgr_nested_mgr_yaml_parse(void)
{
   input_args *iargs = NULL;
   char        yaml_path[4096];

   snprintf(yaml_path, sizeof(yaml_path), "%s/examples/ex7-nested-mgr.yml",
            HYPREDRIVE_SOURCE_DIR);

   char *argv[] = {yaml_path};
   int   argc   = 1;

   ErrorCodeResetAll();
   InputArgsParse(MPI_COMM_SELF, true, argc, argv, &iargs);
   ASSERT_FALSE(ErrorCodeActive());
   ASSERT_NOT_NULL(iargs);
   ASSERT_EQ(iargs->precon_method, PRECON_MGR);

   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.type, MGR_FRLX_TYPE_NESTED_MGR);
   ASSERT_NOT_NULL(iargs->precon.mgr.level[0].f_relaxation.mgr);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->num_levels, 3);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[0].f_dofs.size, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[0].f_dofs.data[0], 0);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[1].f_dofs.size, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[1].f_dofs.data[0], 1);

   InputArgsDestroy(&iargs);
}

int
main(int argc, char **argv)
{
   (void)argv;
   MPI_Init(&argc, &argv);

   RUN_TEST(test_mgr_nested_yaml_parse);
   RUN_TEST(test_mgr_nested_mgr_yaml_parse);

   MPI_Finalize();
   return 0;
}
