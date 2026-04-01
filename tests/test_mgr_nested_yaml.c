/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "internal/args.h"
#include "internal/error.h"
#include "internal/mgr.h"
#include "internal/krylov.h"
#include "test_helpers.h"

static input_args *
parse_config(const char *yaml_text)
{
   input_args *args    = NULL;
   char       *argv0   = strdup(yaml_text);
   char       *argv[1] = {argv0};

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, false, 1, argv, &args);
   free(argv0);
   return args;
}

static void
test_mgr_nested_yaml_parse(void)
{
   input_args *iargs = NULL;
   char        yaml_path[4096];

   snprintf(yaml_path, sizeof(yaml_path),
            "%s/tests/fixtures/ex3-mgr_Frelax_gmres.yml", HYPREDRIVE_SOURCE_DIR);

   char *argv[] = {yaml_path};
   int   argc   = 1;

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, true, argc, argv, &iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(iargs);
   ASSERT_EQ(iargs->precon_method, PRECON_MGR);
   ASSERT_TRUE(iargs->precon.mgr.level[1].f_relaxation.use_krylov);
   ASSERT_NOT_NULL(iargs->precon.mgr.level[1].f_relaxation.krylov);
   ASSERT_EQ(iargs->precon.mgr.level[1].f_relaxation.krylov->solver_method, SOLVER_GMRES);

   hypredrv_InputArgsDestroy(&iargs);
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

   hypredrv_ErrorCodeResetAll();
   hypredrv_InputArgsParse(MPI_COMM_SELF, true, argc, argv, &iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(iargs);
   ASSERT_EQ(iargs->precon_method, PRECON_MGR);

   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.type, MGR_FRLX_TYPE_NESTED_MGR);
   ASSERT_NOT_NULL(iargs->precon.mgr.level[0].f_relaxation.mgr);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->num_levels, 3);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[0].f_dofs.size, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[0].f_dofs.data[0], 0);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[1].f_dofs.size, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.mgr->level[1].f_dofs.data[0], 1);

   hypredrv_InputArgsDestroy(&iargs);
}

static void
test_mgr_relaxation_block_yaml_parse(void)
{
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation:\n"
      "          amg:\n"
      "            max_iter: 1\n"
      "        g_relaxation:\n"
      "          amg:\n"
      "            max_iter: 2\n"
      "        restriction_type: injection\n"
      "        prolongation_type: jacobi\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *iargs = parse_config(yaml_text);
   ASSERT_NOT_NULL(iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(iargs->precon_method, PRECON_MGR);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.type, 2);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.amg.max_iter, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].g_relaxation.type, 20);
   ASSERT_EQ(iargs->precon.mgr.level[0].g_relaxation.amg.max_iter, 2);

   hypredrv_InputArgsDestroy(&iargs);
}

static void
test_mgr_nested_yaml_grelax_nested_krylov_inline(void)
{
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation: jacobi\n"
      "        g_relaxation:\n"
      "          gmres:\n"
      "            max_iter: 2\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *iargs = parse_config(yaml_text);
   ASSERT_NOT_NULL(iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(iargs->precon.mgr.level[0].g_relaxation.use_krylov);
   ASSERT_EQ(iargs->precon.mgr.level[0].g_relaxation.krylov->solver_method, SOLVER_GMRES);

   hypredrv_InputArgsDestroy(&iargs);
}

static void
test_mgr_nested_yaml_frelax_nested_krylov_mgr_precon_inline(void)
{
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation:\n"
      "          fgmres:\n"
      "            max_iter: 2\n"
      "            relative_tol: 0.0\n"
      "            preconditioner:\n"
      "              mgr:\n"
      "                max_iter: 1\n"
      "                level:\n"
      "                  0:\n"
      "                    f_dofs: [0]\n"
      "                    f_relaxation: jacobi\n"
      "                    g_relaxation: none\n"
      "                coarsest_level:\n"
      "                  amg:\n"
      "                    print_level: 0\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *iargs = parse_config(yaml_text);
   ASSERT_NOT_NULL(iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(iargs->precon.mgr.level[0].f_relaxation.use_krylov);
   ASSERT_NOT_NULL(iargs->precon.mgr.level[0].f_relaxation.krylov);
   ASSERT_TRUE(iargs->precon.mgr.level[0].f_relaxation.krylov->has_precon);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.krylov->precon_method, PRECON_MGR);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.krylov->precon.mgr.max_iter, 1);
   ASSERT_EQ(iargs->precon.mgr.level[0].f_relaxation.krylov->precon.mgr.num_levels, 2);

   hypredrv_InputArgsDestroy(&iargs);
}

static void
test_mgr_nested_yaml_coarsest_nested_krylov_mgr_precon_inline(void)
{
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0]\n"
      "        f_relaxation: jacobi\n"
      "        g_relaxation: none\n"
      "    coarsest_level:\n"
      "      gmres:\n"
      "        max_iter: 2\n"
      "        preconditioner:\n"
      "          mgr:\n"
      "            max_iter: 1\n"
      "            level:\n"
      "              0:\n"
      "                f_dofs: [0]\n"
      "                f_relaxation: jacobi\n"
      "                g_relaxation: none\n"
      "            coarsest_level:\n"
      "              amg:\n"
      "                print_level: 0\n";

   input_args *iargs = parse_config(yaml_text);
   ASSERT_NOT_NULL(iargs);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(iargs->precon.mgr.coarsest_level.use_krylov);
   ASSERT_NOT_NULL(iargs->precon.mgr.coarsest_level.krylov);
   ASSERT_TRUE(iargs->precon.mgr.coarsest_level.krylov->has_precon);
   ASSERT_EQ(iargs->precon.mgr.coarsest_level.krylov->precon_method, PRECON_MGR);
   ASSERT_EQ(iargs->precon.mgr.coarsest_level.krylov->precon.mgr.max_iter, 1);
   ASSERT_EQ(iargs->precon.mgr.coarsest_level.krylov->precon.mgr.num_levels, 2);

   hypredrv_InputArgsDestroy(&iargs);
}

int
main(int argc, char **argv)
{
   (void)argv;
   MPI_Init(&argc, &argv);

   RUN_TEST(test_mgr_nested_yaml_parse);
   RUN_TEST(test_mgr_nested_mgr_yaml_parse);
   RUN_TEST(test_mgr_relaxation_block_yaml_parse);
   RUN_TEST(test_mgr_nested_yaml_grelax_nested_krylov_inline);
   RUN_TEST(test_mgr_nested_yaml_frelax_nested_krylov_mgr_precon_inline);
   RUN_TEST(test_mgr_nested_yaml_coarsest_nested_krylov_mgr_precon_inline);

   MPI_Finalize();
   return 0;
}
