/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
#include "containers.h"
#include "error.h"
#include "mgr.h"
#include "test_helpers.h"

/*-----------------------------------------------------------------------------
 * DofLabelMap lifecycle tests
 *-----------------------------------------------------------------------------*/

static void
test_DofLabelMapCreate(void)
{
   DofLabelMap *map = hypredrv_DofLabelMapCreate();
   ASSERT_NOT_NULL(map);
   ASSERT_EQ((int)map->size, 0);
   hypredrv_DofLabelMapDestroy(&map);
   ASSERT_NULL(map);
}

static void
test_DofLabelMapAdd_and_Lookup(void)
{
   DofLabelMap *map = hypredrv_DofLabelMapCreate();
   hypredrv_DofLabelMapAdd(map, "v_x", 0);
   hypredrv_DofLabelMapAdd(map, "v_y", 1);
   hypredrv_DofLabelMapAdd(map, "P", 2);

   ASSERT_EQ((int)map->size, 3);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(map, "v_x"), 0);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(map, "v_y"), 1);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(map, "P"), 2);

   hypredrv_DofLabelMapDestroy(&map);
}

static void
test_DofLabelMapLookup_unknown_returns_minus1(void)
{
   DofLabelMap *map = hypredrv_DofLabelMapCreate();
   hypredrv_DofLabelMapAdd(map, "v_x", 0);

   ASSERT_EQ(hypredrv_DofLabelMapLookup(map, "unknown"), -1);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(map, ""), -1);

   hypredrv_DofLabelMapDestroy(&map);
}

static void
test_DofLabelMapLookup_null_map(void)
{
   ASSERT_EQ(hypredrv_DofLabelMapLookup(NULL, "v_x"), -1);
}

static void
test_DofLabelMapAdd_grows_beyond_initial_capacity(void)
{
   DofLabelMap *map = hypredrv_DofLabelMapCreate();

   /* Add more than the initial capacity (8) entries */
   char name[32];
   for (int i = 0; i < 20; i++)
   {
      snprintf(name, sizeof(name), "label_%d", i);
      hypredrv_DofLabelMapAdd(map, name, i);
   }
   ASSERT_EQ((int)map->size, 20);
   for (int i = 0; i < 20; i++)
   {
      snprintf(name, sizeof(name), "label_%d", i);
      ASSERT_EQ(hypredrv_DofLabelMapLookup(map, name), i);
   }
   hypredrv_DofLabelMapDestroy(&map);
}

/*-----------------------------------------------------------------------------
 * Integration tests: YAML parsing with dof_labels
 *-----------------------------------------------------------------------------*/

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
test_dof_labels_parsed_from_yaml(void)
{
   /* Keys in dof_labels are lowercased on storage (to match the behaviour of
    * the YAML parser which always lowercases node values but not node keys).
    * "P" in the YAML key position is therefore stored and looked up as "p". */
   const char yaml_text[] =
      "linear_system:\n"
      "  dof_labels:\n"
      "    v_x: 0\n"
      "    v_y: 1\n"
      "    p: 2\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [v_x, v_y]\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   /* Verify dof_labels map was parsed - keys are normalised to lowercase */
   ASSERT_NOT_NULL(args->ls.dof_labels);
   ASSERT_EQ((int)args->ls.dof_labels->size, 3);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "v_x"), 0);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "v_y"), 1);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "p"), 2);

   /* Verify f_dofs were resolved correctly */
   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_dof_labels_mixed_case_key_normalised(void)
{
   /* The YAML parser lowercases values but not keys, so "P" written as a
    * dof_labels key is different from "P" written in f_dofs (which becomes
    * "p").  We normalise dof_labels keys to lowercase at parse time so that
    * both sides match regardless of how the user writes the label. */
   const char yaml_text[] =
      "linear_system:\n"
      "  dof_labels:\n"
      "    V_X: 0\n"   /* uppercase key - stored as "v_x" */
      "    v_y: 1\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [V_X, v_y]\n" /* V_X lowercased to v_x by YAML parser */
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   /* "V_X" key stored as "v_x" */
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "v_x"), 0);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "V_X"), -1);

   /* f_dofs resolved correctly even though key was uppercase in dof_labels */
   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_f_dofs_integer_backward_compat(void)
{
   /* Plain integers in f_dofs should still work without dof_labels */
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [0, 1]\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_NULL(args->ls.dof_labels);
   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_f_dofs_label_without_dof_labels_is_error(void)
{
   /* Using a label in f_dofs without dof_labels defined should be an error */
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [v_x, v_y]\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   /* Should fail: label used but no dof_labels defined */
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   if (args)
   {
      hypredrv_InputArgsDestroy(&args);
   }
}

/*-----------------------------------------------------------------------------
 * Alternative YAML syntax tests
 *-----------------------------------------------------------------------------*/

static void
test_dof_labels_flow_mapping(void)
{
   /* dof_labels: {v_x: 0, v_y: 1, p: 2}  - inline flow-mapping form */
   const char yaml_text[] =
      "linear_system:\n"
      "  dof_labels: {v_x: 0, v_y: 1, p: 2}\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs: [v_x, v_y]\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_NOT_NULL(args->ls.dof_labels);
   ASSERT_EQ((int)args->ls.dof_labels->size, 3);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "v_x"), 0);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "v_y"), 1);
   ASSERT_EQ(hypredrv_DofLabelMapLookup(args->ls.dof_labels, "p"), 2);

   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_f_dofs_block_sequence_labels(void)
{
   /* f_dofs as a block sequence of symbolic labels */
   const char yaml_text[] =
      "linear_system:\n"
      "  dof_labels:\n"
      "    v_x: 0\n"
      "    v_y: 1\n"
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs:\n"
      "          - v_x\n"
      "          - v_y\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

static void
test_f_dofs_block_sequence_integers(void)
{
   /* f_dofs as a block sequence of plain integers (no dof_labels needed) */
   const char yaml_text[] =
      "solver:\n"
      "  pcg:\n"
      "    max_iter: 10\n"
      "preconditioner:\n"
      "  mgr:\n"
      "    level:\n"
      "      0:\n"
      "        f_dofs:\n"
      "          - 0\n"
      "          - 1\n"
      "    coarsest_level:\n"
      "      amg:\n"
      "        print_level: 0\n";

   input_args *args = parse_config(yaml_text);
   ASSERT_NOT_NULL(args);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_EQ((int)args->precon.mgr.level[0].f_dofs.size, 2);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[0], 0);
   ASSERT_EQ(args->precon.mgr.level[0].f_dofs.data[1], 1);

   hypredrv_InputArgsDestroy(&args);
}

/*-----------------------------------------------------------------------------
 * Main test runner
 *-----------------------------------------------------------------------------*/

int
main(void)
{
   MPI_Init(NULL, NULL);
   TEST_HYPRE_INIT();

   /* DofLabelMap lifecycle */
   RUN_TEST(test_DofLabelMapCreate);
   RUN_TEST(test_DofLabelMapAdd_and_Lookup);
   RUN_TEST(test_DofLabelMapLookup_unknown_returns_minus1);
   RUN_TEST(test_DofLabelMapLookup_null_map);
   RUN_TEST(test_DofLabelMapAdd_grows_beyond_initial_capacity);

   /* Integration: YAML parsing - block mapping + flow sequence (primary forms) */
   RUN_TEST(test_dof_labels_parsed_from_yaml);
   RUN_TEST(test_dof_labels_mixed_case_key_normalised);
   RUN_TEST(test_f_dofs_integer_backward_compat);
   RUN_TEST(test_f_dofs_label_without_dof_labels_is_error);

   /* Alternative YAML syntax forms */
   RUN_TEST(test_dof_labels_flow_mapping);
   RUN_TEST(test_f_dofs_block_sequence_labels);
   RUN_TEST(test_f_dofs_block_sequence_integers);

   TEST_HYPRE_FINALIZE();
   MPI_Finalize();
   return 0;
}
