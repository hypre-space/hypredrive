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

#include "internal/error.h"
#include "internal/presets.h"
#include "test_helpers.h"
#include "HYPREDRV.h"

/*-----------------------------------------------------------------------------
 * test_PresetRegister_success
 *-----------------------------------------------------------------------------*/

static void
test_PresetRegister_success(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   int ret = hypredrv_PresetRegister("my_custom", "amg:\n  max_iter: 1", "Custom AMG");
   ASSERT_EQ(ret, 0);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   const hypredrv_Preset *p = hypredrv_PresetFind("my_custom");
   ASSERT_NOT_NULL(p);
   ASSERT_STREQ(p->name, "my_custom");
   ASSERT_NOT_NULL(strstr(p->text, "amg"));
   ASSERT_STREQ(p->help, "Custom AMG");

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

   int ret = hypredrv_PresetRegister("poisson", "amg", "duplicate of builtin");
   ASSERT_EQ(ret, -1);
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

   int ret1 = hypredrv_PresetRegister("my_dup", "amg", "first registration");
   ASSERT_EQ(ret1, 0);

   hypredrv_ErrorCodeResetAll();

   int ret2 = hypredrv_PresetRegister("my_dup", "amg:\n  max_iter: 2", "second");
   ASSERT_EQ(ret2, -1);
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
   int ret = hypredrv_PresetRegister(NULL, "amg", "help");
   ASSERT_EQ(ret, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();

   /* NULL yaml_text */
   ret = hypredrv_PresetRegister("my_null_text", NULL, "help");
   ASSERT_EQ(ret, -1);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   hypredrv_ErrorCodeResetAll();

   /* NULL help is allowed - treated as empty string */
   ret = hypredrv_PresetRegister("my_null_help", "amg", NULL);
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

   ASSERT_EQ(hypredrv_PresetRegister("my-hyphen-preset", "amg:\n  max_iter: 1", "hyphens"), 0);
   const hypredrv_Preset *p = hypredrv_PresetFind("my_hyphen_preset");
   ASSERT_NOT_NULL(p);
   ASSERT_STREQ(p->name, "my_hyphen_preset");

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

   int ret = hypredrv_PresetRegister("my_case_test", "amg", "Case test preset");
   ASSERT_EQ(ret, 0);

   /* Different casing */
   ASSERT_NOT_NULL(hypredrv_PresetFind("MY_CASE_TEST"));
   ASSERT_NOT_NULL(hypredrv_PresetFind("My_Case_Test"));
   ASSERT_NOT_NULL(hypredrv_PresetFind("my_case_test"));

   /* Hyphen treated as underscore */
   ASSERT_NOT_NULL(hypredrv_PresetFind("my-case-test"));
   ASSERT_NOT_NULL(hypredrv_PresetFind("MY-CASE-TEST"));

   /* Unknown name still returns NULL */
   ASSERT_NULL(hypredrv_PresetFind("no_such_preset"));

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

   int ret = hypredrv_PresetRegister("help_test_preset", "amg", "My help text");
   ASSERT_EQ(ret, 0);

   char *help = hypredrv_PresetHelp();
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
 * test_PresetFreeUserPresets_idempotent
 *-----------------------------------------------------------------------------*/

static void
test_PresetFreeUserPresets_idempotent(void)
{
   hypredrv_ErrorCodeResetAll();

   int ret = hypredrv_PresetRegister("idempotent_preset", "amg", "to be freed");
   ASSERT_EQ(ret, 0);

   /* First free */
   hypredrv_PresetFreeUserPresets();

   /* Second free should be a no-op (NULL guard) */
   hypredrv_PresetFreeUserPresets();

   /* After freeing, the preset should no longer be found */
   ASSERT_NULL(hypredrv_PresetFind("idempotent_preset"));

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

   /* Verify the preset is findable via the private API */
   const hypredrv_Preset *p = hypredrv_PresetFind("public_test_preset");
   ASSERT_NOT_NULL(p);
   ASSERT_STREQ(p->help, "Public test preset");

   /* NULL args should return an error */
   err = HYPREDRV_PreconPresetRegister(NULL, "amg", "help");
   ASSERT_NE((int)err, 0);

   /* HYPREDRV_Finalize should clean up user presets */
   HYPREDRV_Finalize();

   /* After finalize the preset must be gone */
   ASSERT_NULL(hypredrv_PresetFind("public_test_preset"));
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
   RUN_TEST(test_PresetFreeUserPresets_idempotent);
   RUN_TEST(test_HYPREDRV_PreconPresetRegister_public);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   MPI_Finalize();

   return 0;
}
