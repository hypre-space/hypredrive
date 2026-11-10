/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HYPREDRV.h"
#include "internal/error.h"
#include "internal/precon.h"
#include "internal/presets.h"
#include "internal/solver.h"
#include "test_helpers.h"

/* Loaded configuration trees outlive registry mutations and are caller-owned. */
static YAMLtree *
load_test_preset(const char *name, hypredrv_PresetKind kind, int expected_method)
{
   YAMLtree *tree   = NULL;
   int       method = -1;
   ASSERT_NOT_NULL(hypredrv_PresetLoadTypeNode(name, kind, &tree, &method));
   ASSERT_EQ(method, expected_method);
   return tree;
}

static void
assert_preset_missing(const char *name, hypredrv_PresetKind kind)
{
   YAMLtree *tree   = NULL;
   int       method = -1;
   hypredrv_ErrorStateReset();
   ASSERT_NULL(hypredrv_PresetLoadTypeNode(name, kind, &tree, &method));
   ASSERT_NULL(tree);
   ASSERT_EQ(method, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
}

/*-----------------------------------------------------------------------------
 * test_PresetRegister_success
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_success(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret = hypredrv_PresetRegisterTyped("my_custom", "amg:\n  max_iter: 1",
                                          "Custom AMG", HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLtree *tree =
      load_test_preset("my_custom", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   ASSERT_STREQ(tree->root->children->children->key, "max_iter");
   ASSERT_STREQ(tree->root->children->children->val, "1");
   hypredrv_YAMLtreeDestroy(&tree);
   char *help = hypredrv_PresetHelpTyped(HYPREDRV_PRESET_PRECON);
   ASSERT_NOT_NULL(help);
   ASSERT_NOT_NULL(strstr(help, "my_custom: Custom AMG"));
   free(help);

   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetRegister_duplicate_builtin
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_duplicate_builtin(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret = hypredrv_PresetRegisterTyped("poisson", "amg", "duplicate of builtin",
                                          HYPREDRV_PRESET_PRECON);
   ASSERT_EQ_U32(ret, ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetRegister_duplicate_user
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_duplicate_user(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret1 = hypredrv_PresetRegisterTyped("my_dup", "amg", "first registration",
                                           HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret1, 0);

   hypredrv_ErrorCodeResetAll();

   int ret2 = hypredrv_PresetRegisterTyped("my_dup", "amg:\n  max_iter: 2", "second",
                                           HYPREDRV_PRESET_PRECON);
   ASSERT_EQ_U32(ret2, ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetRegister_null_args
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_null_args(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   /* NULL name */
   int ret = hypredrv_PresetRegisterTyped(NULL, "amg", "help", HYPREDRV_PRESET_PRECON);
   ASSERT_EQ_U32(ret, ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();

   /* NULL yaml_text */
   ret =
      hypredrv_PresetRegisterTyped("my_null_text", NULL, "help", HYPREDRV_PRESET_PRECON);
   ASSERT_EQ_U32(ret, ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();

   /* NULL help is allowed - treated as empty string */
   ret =
      hypredrv_PresetRegisterTyped("my_null_help", "amg", NULL, HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);
   hypredrv_ErrorCodeResetAll();

   hypredrv_PresetFreeUserPresets();
}

/*-----------------------------------------------------------------------------
 * test_PresetRegister_hyphen_in_name_normalizes
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_hyphen_in_name_normalizes(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   ASSERT_EQ(hypredrv_PresetRegisterTyped("my-hyphen-preset", "amg:\n  max_iter: 1",
                                          "hyphens", HYPREDRV_PRESET_PRECON),
             0);
   YAMLtree *tree =
      load_test_preset("my_hyphen_preset", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);

   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetFind_user_preset_case_insensitive
 *-----------------------------------------------------------------------------*/

static void
test_PresetFind_user_preset_case_insensitive(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret = hypredrv_PresetRegisterTyped("my_case_test", "amg", "Case test preset",
                                          HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);

   YAMLtree *tree = NULL;
   /* Different casing */
   tree = load_test_preset("MY_CASE_TEST", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);
   tree = load_test_preset("My_Case_Test", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);
   tree = load_test_preset("my_case_test", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);

   /* Hyphen treated as underscore */
   tree = load_test_preset("my-case-test", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);
   tree = load_test_preset("MY-CASE-TEST", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);

   /* Unknown name still returns NULL */
   assert_preset_missing("no_such_preset", HYPREDRV_PRESET_PRECON);

   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetHelp_includes_user_preset
 *-----------------------------------------------------------------------------*/

static void
test_PresetHelp_includes_user_preset(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret = hypredrv_PresetRegisterTyped("help_test_preset", "amg", "My help text",
                                          HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);

   char *help = hypredrv_PresetHelpTyped(HYPREDRV_PRESET_PRECON);
   ASSERT_NOT_NULL(help);
   ASSERT_NOT_NULL(strstr(help, "help_test_preset"));
   ASSERT_NOT_NULL(strstr(help, "My help text"));
   /* Built-in presets also present */
   ASSERT_NOT_NULL(strstr(help, "poisson"));

   free(help);
   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetHelp_keeps_long_help_text
 *-----------------------------------------------------------------------------*/

static void
test_PresetHelp_keeps_long_help_text(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   /* Longer than the 1024-byte line buffer the help text was once built in */
   char long_help[1501];
   memset(long_help, 'x', sizeof(long_help) - 1);
   long_help[sizeof(long_help) - 1] = '\0';

   int ret = hypredrv_PresetRegisterTyped("long_help_preset", "amg", long_help,
                                          HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);

   char *help = hypredrv_PresetHelpTyped(HYPREDRV_PRESET_PRECON);
   ASSERT_NOT_NULL(help);
   char expected[1600];
   snprintf(expected, sizeof(expected), "    - long_help_preset: %s\n", long_help);
   ASSERT_NOT_NULL(strstr(help, expected));

   free(help);
   hypredrv_PresetFreeUserPresets();
   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_PresetFreeUserPresets_idempotent
 *-----------------------------------------------------------------------------*/

static void
test_PresetFreeUserPresets_idempotent(void)
{
   hypredrv_ErrorCodeResetAll();

   int ret = hypredrv_PresetRegisterTyped("idempotent_preset", "amg", "to be freed",
                                          HYPREDRV_PRESET_PRECON);
   ASSERT_EQ(ret, 0);

   /* First free */
   hypredrv_PresetFreeUserPresets();

   /* Second free should be a no-op (NULL guard) */
   hypredrv_PresetFreeUserPresets();

   /* After freeing, the preset should no longer be found */
   assert_preset_missing("idempotent_preset", HYPREDRV_PRESET_PRECON);

   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * test_HYPREDRV_PreconPresetRegister_public
 *-----------------------------------------------------------------------------*/

static void
test_HYPREDRV_PreconPresetRegister_public(void)
{
   HYPREDRV_Initialize();

   uint32_t err = HYPREDRV_PreconPresetRegister(
      "public_test_preset", "amg:\n  max_iter: 1", "Public test preset");
   ASSERT_EQ((int)err, 0);

   YAMLtree *tree =
      load_test_preset("public_test_preset", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);

   /* NULL args should return an error */
   err = HYPREDRV_PreconPresetRegister(NULL, "amg", "help");
   ASSERT_NE((int)err, 0);

   /* HYPREDRV_Finalize should clean up user presets */
   HYPREDRV_Finalize();

   /* After finalize the preset must be gone */
   assert_preset_missing("public_test_preset", HYPREDRV_PRESET_PRECON);
}

/*-----------------------------------------------------------------------------
 * test_HYPREDRV_SolverPresetRegister_public
 *-----------------------------------------------------------------------------*/

static void
test_solver_name_conflicts_use_normalized_identity(void)
{
   const char *names[] = {"GMRES", "Pcg", "Fgmres", "BICGSTAB", "0", "01", "+2", "-0"};
   for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
   {
      ASSERT_TRUE(HYPREDRV_SolverPresetRegister(names[i], "gmres", "conflict") &
                  ERROR_INVALID_VAL);
      hypredrv_ErrorStateReset();
      ASSERT_EQ_U32(hypredrv_PresetRegisterTyped(names[i], "gmres", "conflict",
                                                 HYPREDRV_PRESET_SOLVER),
                    ERROR_INVALID_VAL);
   }
}

static void
test_owned_trees_survive_registry_changes(void)
{
   char name[] = "Owned-Definition";
   char text[] = "fgmres:\n  max_iter: 29\n";
   char help[] = "owned help";
   ASSERT_EQ(HYPREDRV_SolverPresetRegister(name, text, help), ERROR_NONE);
   memset(name, 'x', sizeof(name) - 1);
   memset(text, 'x', sizeof(text) - 1);
   memset(help, 'x', sizeof(help) - 1);
   YAMLtree *first =
      load_test_preset("OWNED_DEFINITION", HYPREDRV_PRESET_SOLVER, SOLVER_FGMRES);
   ASSERT_STREQ(first->root->children->children->val, "29");
   first->root->children->children->val[0] = '1';
   /* Force registry growth after taking an owned configuration tree. */
   for (int i = 0; i < 40; i++)
   {
      char key[32];
      snprintf(key, sizeof(key), "growth-%d", i);
      ASSERT_EQ(HYPREDRV_PreconPresetRegister(key, "amg", NULL), ERROR_NONE);
   }
   YAMLtree *second =
      load_test_preset("owned-definition", HYPREDRV_PRESET_SOLVER, SOLVER_FGMRES);
   ASSERT_STREQ(second->root->children->children->val, "29");
   char *listing = hypredrv_PresetHelpTyped(HYPREDRV_PRESET_SOLVER);
   ASSERT_NOT_NULL(listing);
   ASSERT_NOT_NULL(strstr(listing, "owned_definition: owned help"));
   free(listing);
   ASSERT_TRUE(HYPREDRV_PreconPresetRegister("OWNED_definition", "amg", NULL) &
               ERROR_INVALID_VAL);
   hypredrv_PresetFreeUserPresets();
   ASSERT_STREQ(first->root->children->children->val, "19");
   ASSERT_STREQ(second->root->children->children->val, "29");
   hypredrv_YAMLtreeDestroy(&first);
   hypredrv_YAMLtreeDestroy(&second);
   assert_preset_missing("owned-definition", HYPREDRV_PRESET_SOLVER);
}

static void
test_typed_load_rejects_malformed_definitions(void)
{
   const char    *texts[]  = {"amg\nfsai\n", "unknown_type\n", "gmres\n"};
   const uint32_t errors[] = {ERROR_INVALID_VAL, ERROR_INVALID_KEY, ERROR_INVALID_KEY};
   for (size_t i = 0; i < sizeof(texts) / sizeof(texts[0]); i++)
   {
      ASSERT_EQ(HYPREDRV_PreconPresetRegister("bad-shape", texts[i], NULL), ERROR_NONE);
      YAMLtree *tree   = NULL;
      int       method = -1;
      ASSERT_NULL(hypredrv_PresetLoadTypeNode("bad_shape", HYPREDRV_PRESET_PRECON, &tree,
                                              &method));
      ASSERT_NULL(tree);
      ASSERT_EQ(method, -1);
      ASSERT_TRUE(hypredrv_ErrorCodeGet() & errors[i]);
      hypredrv_PresetFreeUserPresets();
   }
   ASSERT_EQ(HYPREDRV_SolverPresetRegister("typed-solver", "gmres", NULL), ERROR_NONE);
   assert_preset_missing("typed_solver", HYPREDRV_PRESET_PRECON);
   hypredrv_ErrorStateReset();
   YAMLtree *tree =
      load_test_preset("typed_solver", HYPREDRV_PRESET_SOLVER, SOLVER_GMRES);
   hypredrv_YAMLtreeDestroy(&tree);
   hypredrv_PresetFreeUserPresets();
}

static void
test_registration_validates_identity_and_preserves_diagnostics(void)
{
   ASSERT_EQ_U32(HYPREDRV_PreconPresetRegister("", "amg", NULL), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_PreconPresetRegister("empty-text", "", NULL),
                 ERROR_INVALID_VAL);
   hypredrv_ErrorStateReset();
   ASSERT_EQ_U32(
      hypredrv_PresetRegisterTyped("invalid-kind", "amg", NULL, (hypredrv_PresetKind)99),
      ERROR_INVALID_VAL);
   ASSERT_NULL(hypredrv_PresetHelpTyped((hypredrv_PresetKind)99));
   hypredrv_ErrorStateReset();
   YAMLtree *tree   = NULL;
   int       method = -1;
   ASSERT_NULL(
      hypredrv_PresetLoadTypeNode("poisson", (hypredrv_PresetKind)99, &tree, &method));
   ASSERT_EQ(method, -1);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ(
      hypredrv_PresetRegisterTyped("private-scope", "amg", NULL, HYPREDRV_PRESET_PRECON),
      ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_NULL(hypredrv_PresetLoadTypeNode("private_scope", HYPREDRV_PRESET_PRECON, &tree,
                                           &method));
   ASSERT_NULL(tree);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(
      hypredrv_PresetRegisterTyped("PRIVATE_scope", "amg", NULL, HYPREDRV_PRESET_PRECON),
      ERROR_INVALID_VAL);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP | ERROR_INVALID_VAL);
   hypredrv_ErrorStateReset();
   tree = load_test_preset("private_scope", HYPREDRV_PRESET_PRECON, PRECON_BOOMERAMG);
   hypredrv_YAMLtreeDestroy(&tree);
   hypredrv_PresetFreeUserPresets();
}

static void
test_HYPREDRV_SolverPresetRegister_public(void)
{
   HYPREDRV_Initialize();

   uint32_t err = HYPREDRV_SolverPresetRegister("public_solver_preset",
                                                "fgmres:\n"
                                                "  max_iter: 17\n"
                                                "  krylov_dim: 23\n"
                                                "  print_level: 0",
                                                "Public solver preset");
   ASSERT_EQ((int)err, 0);

   YAMLtree *tree =
      load_test_preset("public_solver_preset", HYPREDRV_PRESET_SOLVER, SOLVER_FGMRES);
   ASSERT_STREQ(tree->root->children->children->val, "17");
   hypredrv_YAMLtreeDestroy(&tree);

   err = HYPREDRV_SolverPresetRegister("gmres", "gmres:\n  max_iter: 1", "conflict");
   ASSERT_NE((int)err, 0);

   err = HYPREDRV_SolverPresetRegister(NULL, "gmres:\n  max_iter: 1", "help");
   ASSERT_NE((int)err, 0);

   HYPREDRV_Finalize();

   assert_preset_missing("public_solver_preset", HYPREDRV_PRESET_SOLVER);
}

/*-----------------------------------------------------------------------------
 * Main test runner
 *-----------------------------------------------------------------------------*/

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_PresetRegister_success);
   RUN_TEST(test_PresetRegister_duplicate_builtin);
   RUN_TEST(test_PresetRegister_duplicate_user);
   RUN_TEST(test_PresetRegister_null_args);
   RUN_TEST(test_PresetRegister_hyphen_in_name_normalizes);
   RUN_TEST(test_PresetFind_user_preset_case_insensitive);
   RUN_TEST(test_PresetHelp_includes_user_preset);
   RUN_TEST(test_PresetHelp_keeps_long_help_text);
   RUN_TEST(test_PresetFreeUserPresets_idempotent);
   RUN_TEST(test_HYPREDRV_PreconPresetRegister_public);
   RUN_TEST(test_HYPREDRV_SolverPresetRegister_public);
   RUN_TEST(test_solver_name_conflicts_use_normalized_identity);
   RUN_TEST(test_owned_trees_survive_registry_changes);
   RUN_TEST(test_typed_load_rejects_malformed_definitions);
   RUN_TEST(test_registration_validates_identity_and_preserves_diagnostics);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   MPI_Finalize();

   return 0;
}
