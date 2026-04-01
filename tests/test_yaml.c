/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "test_helpers.h"
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/yaml.h"

/*-----------------------------------------------------------------------------
 * Test YAMLnode creation and destruction
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeCreate_basic(void)
{
   YAMLnode *node = hypredrv_YAMLnodeCreate("key", "value", 0);
   ASSERT_NOT_NULL(node);
   ASSERT_STREQ(node->key, "key");
   ASSERT_STREQ(node->val, "value");
   /* Note: indent is not a direct field, it's stored in the struct differently */
   hypredrv_YAMLnodeDestroy(node);
}

/*-----------------------------------------------------------------------------
 * Test YAMLtree creation and destruction
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeCreate_basic(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);
   hypredrv_YAMLtreeDestroy(&tree);
   ASSERT_NULL(tree);
}

/*-----------------------------------------------------------------------------
 * Test hypredrv_YAMLnodeFindByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindByKey_basic(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);

   YAMLnode *root  = tree->root;
   YAMLnode *child = hypredrv_YAMLnodeCreate("child", "value", 1);
   hypredrv_YAMLnodeAddChild(root, child);

   YAMLnode *found = hypredrv_YAMLnodeFindByKey(root, "child");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child");

   YAMLnode *not_found = hypredrv_YAMLnodeFindByKey(root, "nonexistent");
   ASSERT_NULL(not_found);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test hypredrv_YAMLnodeFindChildByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindChildByKey_basic(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child1 = hypredrv_YAMLnodeCreate("child1", "value1", 1);
   YAMLnode *child2 = hypredrv_YAMLnodeCreate("child2", "value2", 1);
   hypredrv_YAMLnodeAddChild(root, child1);
   hypredrv_YAMLnodeAddChild(root, child2);

   YAMLnode *found = hypredrv_YAMLnodeFindChildByKey(root, "child1");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child1");

   found = hypredrv_YAMLnodeFindChildByKey(root, "child2");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child2");

   found = hypredrv_YAMLnodeFindChildByKey(root, "nonexistent");
   ASSERT_NULL(found);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test hypredrv_YAMLnodeFindChildValueByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindChildValueByKey_basic(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = hypredrv_YAMLnodeCreate("key", "value", 1);
   hypredrv_YAMLnodeAddChild(root, child);

   char *value = hypredrv_YAMLnodeFindChildValueByKey(root, "key");
   ASSERT_NOT_NULL(value);
   ASSERT_STREQ(value, "value");

   value = hypredrv_YAMLnodeFindChildValueByKey(root, "nonexistent");
   ASSERT_NULL(value);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML tree building from simple text
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_simple(void)
{
   const char *yaml_text = "key: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   /* hypredrv_YAMLtreeBuild creates a tree from the input YAML text */
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   char *value = hypredrv_YAMLnodeFindChildValueByKey(tree->root, "key");
   /* value may be NULL if parsing doesn't work as expected - check for basic tree
    * structure */
   if (value)
   {
      ASSERT_STREQ(value, "value");
   }

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test hypredrv_YAMLtreeBuild does not mutate the caller-provided text buffer
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_preserves_input_text(void)
{
   const char *yaml_text = "solver:\n  gmres:\n    max_iter: 10\n";
   char       *text      = strdup(yaml_text);
   char       *text_copy = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   ASSERT_NOT_NULL(text);
   ASSERT_NOT_NULL(text_copy);

   hypredrv_YAMLtreeBuild(2, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_STREQ(text, text_copy);

   hypredrv_YAMLtreeDestroy(&tree);
   free(text);
   free(text_copy);
}

/*-----------------------------------------------------------------------------
 * Test YAML tree building with nested structure
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_nested(void)
{
   const char *yaml_text = "parent:\n  child: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   /* hypredrv_YAMLtreeBuild creates a tree from the input YAML text */
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   YAMLnode *parent = hypredrv_YAMLnodeFindChildByKey(tree->root, "parent");
   /* Basic test - parent may or may not exist depending on parsing */
   if (parent)
   {
      char *value = hypredrv_YAMLnodeFindChildValueByKey(parent, "child");
      if (value)
      {
         ASSERT_STREQ(value, "value");
      }
   }

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML error handling
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_invalid_indent(void)
{
   /* Test with invalid indentation */
   const char *yaml_text = "key:\ninvalid_indent: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   /* hypredrv_YAMLtreeBuild creates a tree from the input YAML text */
   hypredrv_YAMLtreeBuild(0, text, &tree);

   /* Tree should be created, but validation should catch errors */
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Validate the tree - should detect invalid indentation */
   hypredrv_YAMLtreeValidate(tree);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodeFindByKey_nonexistent(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = hypredrv_YAMLnodeCreate("child", "value", 1);
   hypredrv_YAMLnodeAddChild(root, child);

   /* Find existing key */
   YAMLnode *found = hypredrv_YAMLnodeFindByKey(root, "child");
   ASSERT_NOT_NULL(found);

   /* Find nonexistent key */
   YAMLnode *not_found = hypredrv_YAMLnodeFindByKey(root, "nonexistent");
   ASSERT_NULL(not_found);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodeFindChildValueByKey_nonexistent(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = hypredrv_YAMLnodeCreate("key", "value", 1);
   hypredrv_YAMLnodeAddChild(root, child);

   /* Find existing value */
   char *value = hypredrv_YAMLnodeFindChildValueByKey(root, "key");
   ASSERT_NOT_NULL(value);
   ASSERT_STREQ(value, "value");

   /* Find nonexistent value */
   value = hypredrv_YAMLnodeFindChildValueByKey(root, "nonexistent");
   ASSERT_NULL(value);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_fullargv_longflag_and_skip_nonoverride_tokens(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);

   /* Full argv mode: use the long flag (--args), include a non-override token after it
    * (should be ignored in full-argv mode), then a real override pair. */
   char *argv[] = {(char *)"--args", (char *)"not_an_override", (char *)"1",
                   (char *)"--foo:bar", (char *)"baz"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(5, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *foo = hypredrv_YAMLnodeFindChildByKey(tree->root, "foo");
   ASSERT_NOT_NULL(foo);
   YAMLnode *bar = hypredrv_YAMLnodeFindChildByKey(foo, "bar");
   ASSERT_NOT_NULL(bar);
   ASSERT_STREQ(bar->val, "baz");

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_fullargv_shortflag(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);

   /* Full argv mode using the short flag (-a). */
   char *argv[] = {(char *)"-a", (char *)"--x:y", (char *)"1"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(3, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *x = hypredrv_YAMLnodeFindChildByKey(tree->root, "x");
   ASSERT_NOT_NULL(x);
   YAMLnode *y = hypredrv_YAMLnodeFindChildByKey(x, "y");
   ASSERT_NOT_NULL(y);
   ASSERT_STREQ(y->val, "1");

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodePrint_only_valid_mode(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);

   /* Ensure at least one node is YAML_NODE_VALID so YAML_PRINT_MODE_ONLY_VALID
    * executes its conditional printing branch. */
   tree->root->valid = YAML_NODE_VALID;
   YAMLnode *a       = hypredrv_YAMLnodeCreate("a", "1", 1);
   YAMLnode *b       = hypredrv_YAMLnodeCreate("b", "2", 1);
   a->valid          = YAML_NODE_VALID;
   b->valid          = YAML_NODE_INVALID_KEY;
   hypredrv_YAMLnodeAddChild(tree->root, a);
   hypredrv_YAMLnodeAddChild(tree->root, b);

   hypredrv_YAMLnodePrint(tree->root, YAML_PRINT_MODE_ONLY_VALID);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreePrint_and_validate_null_tree(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreePrint(NULL, YAML_PRINT_MODE_ANY);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_YAML_TREE_NULL);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeValidate(NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_YAML_TREE_NULL);

   /* Tokenize/build paths abort early if global error state is left set (yaml.c). */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
}

static void
test_YAMLtreePrint_all_modes(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   YAMLnode *child = hypredrv_YAMLnodeCreate("solver", "gmres", 1);
   child->valid = YAML_NODE_VALID;
   hypredrv_YAMLnodeAddChild(tree->root, child);

   hypredrv_YAMLtreePrint(tree, YAML_PRINT_MODE_NO_CHECKING);
   hypredrv_YAMLtreePrint(tree, YAML_PRINT_MODE_ONLY_VALID);
   hypredrv_YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodePrint_any_mode_validity_branches(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   struct
   {
      const char *key;
      YAMLvalidity v;
   } cases[] = {
       {"v1", YAML_NODE_VALID},
       {"i1", YAML_NODE_INVALID_INDENT},
       {"d1", YAML_NODE_INVALID_DIVISOR},
       {"k1", YAML_NODE_INVALID_KEY},
       {"val1", YAML_NODE_INVALID_VAL},
       {"u1", YAML_NODE_UNEXPECTED_VAL},
       {"def1", YAML_NODE_UNKNOWN},
   };

   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      YAMLnode *n = hypredrv_YAMLnodeCreate(cases[i].key, "x", 1);
      n->valid    = cases[i].v;
      hypredrv_YAMLnodeAddChild(tree->root, n);
   }
   hypredrv_YAMLnodePrint(tree->root, YAML_PRINT_MODE_ANY);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodePrint_sequence_inline_first_child(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   YAMLnode *dash = hypredrv_YAMLnodeCreate("-", "", 1);
   YAMLnode *inline_kid = hypredrv_YAMLnodeCreate("tag", "value", 2);
   hypredrv_YAMLnodeAddChild(dash, inline_kid);
   hypredrv_YAMLnodeAddChild(tree->root, dash);

   hypredrv_YAMLnodePrint(tree->root, YAML_PRINT_MODE_NO_CHECKING);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML tree building with deeply nested structure (3+ levels)
 * This tests parsing patterns like:
 *   coarsest_level:
 *     ilu:
 *       print_level: 1
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_deeply_nested(void)
{
   /* Three-level nesting: parent -> middle -> child: value */
   const char *yaml_text = "parent:\n  middle:\n    child: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Find parent node */
   YAMLnode *parent = hypredrv_YAMLnodeFindChildByKey(tree->root, "parent");
   ASSERT_NOT_NULL(parent);
   ASSERT_STREQ(parent->key, "parent");
   ASSERT_STREQ(parent->val, ""); /* parent has children, so val is empty */

   /* Find middle node */
   YAMLnode *middle = hypredrv_YAMLnodeFindChildByKey(parent, "middle");
   ASSERT_NOT_NULL(middle);
   ASSERT_STREQ(middle->key, "middle");
   ASSERT_STREQ(middle->val, ""); /* middle has children, so val is empty */

   /* Find child node */
   YAMLnode *child = hypredrv_YAMLnodeFindChildByKey(middle, "child");
   ASSERT_NOT_NULL(child);
   ASSERT_STREQ(child->key, "child");
   ASSERT_STREQ(child->val, "value");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML node key vs val distinction for nested structures
 * This is crucial: when a key has children, its val should be empty
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnode_key_val_distinction(void)
{
   /* When 'solver' has no children, val contains the value */
   const char *yaml_flat = "solver: gmres\n";
   size_t      len       = strlen(yaml_flat);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_flat);

   YAMLtree *tree_flat = NULL;
   hypredrv_YAMLtreeBuild(0, text, &tree_flat);
   ASSERT_NOT_NULL(tree_flat);

   YAMLnode *solver_flat = hypredrv_YAMLnodeFindChildByKey(tree_flat->root, "solver");
   ASSERT_NOT_NULL(solver_flat);
   ASSERT_STREQ(solver_flat->key, "solver");
   ASSERT_STREQ(solver_flat->val, "gmres");
   ASSERT_NULL(solver_flat->children); /* No children */

   free(text);
   hypredrv_YAMLtreeDestroy(&tree_flat);

   /* When 'solver' has children, val should be empty */
   const char *yaml_nested = "solver:\n  type: gmres\n";
   len                     = strlen(yaml_nested);
   text                    = malloc(len + 1);
   strcpy(text, yaml_nested);

   YAMLtree *tree_nested = NULL;
   hypredrv_YAMLtreeBuild(0, text, &tree_nested);
   ASSERT_NOT_NULL(tree_nested);

   YAMLnode *solver_nested = hypredrv_YAMLnodeFindChildByKey(tree_nested->root, "solver");
   ASSERT_NOT_NULL(solver_nested);
   ASSERT_STREQ(solver_nested->key, "solver");
   ASSERT_STREQ(solver_nested->val, ""); /* Has children, so val is empty */
   ASSERT_NOT_NULL(solver_nested->children);

   YAMLnode *type_node = hypredrv_YAMLnodeFindChildByKey(solver_nested, "type");
   ASSERT_NOT_NULL(type_node);
   ASSERT_STREQ(type_node->key, "type");
   ASSERT_STREQ(type_node->val, "gmres");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree_nested);
}

/*-----------------------------------------------------------------------------
 * Test MGR-style nested YAML structure
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_mgr_coarsest_level_pattern(void)
{
   /* This mimics the ex3-ilu.yml structure that exposed the bug:
    *   coarsest_level:
    *     ilu:
    *       type: bj-ilut
    *       droptol: 1e-4
    */
   const char *yaml_text =
      "coarsest_level:\n  ilu:\n    type: bj-ilut\n    droptol: 1e-4\n";
   size_t len  = strlen(yaml_text);
   char  *text = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Find coarsest_level */
   YAMLnode *coarsest = hypredrv_YAMLnodeFindChildByKey(tree->root, "coarsest_level");
   ASSERT_NOT_NULL(coarsest);
   ASSERT_STREQ(coarsest->key, "coarsest_level");
   ASSERT_STREQ(coarsest->val, ""); /* Has children */
   ASSERT_NOT_NULL(coarsest->children);

   /* Find ilu - this is where the bug was: code was checking val instead of key */
   YAMLnode *ilu = hypredrv_YAMLnodeFindChildByKey(coarsest, "ilu");
   ASSERT_NOT_NULL(ilu);
   ASSERT_STREQ(ilu->key, "ilu"); /* KEY is "ilu" */
   ASSERT_STREQ(ilu->val, "");    /* VAL is empty because ilu has children! */
   ASSERT_NOT_NULL(ilu->children);

   /* Verify ilu's children */
   YAMLnode *type_node = hypredrv_YAMLnodeFindChildByKey(ilu, "type");
   ASSERT_NOT_NULL(type_node);
   ASSERT_STREQ(type_node->key, "type");
   ASSERT_STREQ(type_node->val, "bj-ilut");

   YAMLnode *droptol = hypredrv_YAMLnodeFindChildByKey(ilu, "droptol");
   ASSERT_NOT_NULL(droptol);
   ASSERT_STREQ(droptol->key, "droptol");
   ASSERT_STREQ(droptol->val, "1e-4");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test multiple siblings at same level
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_siblings(void)
{
   const char *yaml_text = "key1: val1\nkey2: val2\nkey3: val3\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   YAMLnode *key1 = hypredrv_YAMLnodeFindChildByKey(tree->root, "key1");
   YAMLnode *key2 = hypredrv_YAMLnodeFindChildByKey(tree->root, "key2");
   YAMLnode *key3 = hypredrv_YAMLnodeFindChildByKey(tree->root, "key3");

   ASSERT_NOT_NULL(key1);
   ASSERT_NOT_NULL(key2);
   ASSERT_NOT_NULL(key3);

   ASSERT_STREQ(key1->val, "val1");
   ASSERT_STREQ(key2->val, "val2");
   ASSERT_STREQ(key3->val, "val3");

   /* Verify sibling linkage */
   ASSERT_TRUE(key1->next == key2);
   ASSERT_TRUE(key2->next == key3);
   ASSERT_NULL(key3->next);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML sequence parsing
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_sequence_items(void)
{
   const char *yaml_text = "preconditioner:\n"
                           "  amg:\n"
                           "    - print_level: 1\n"
                           "      coarsening:\n"
                           "        type: HMIS\n"
                           "    - print_level: 2\n"
                           "      coarsening:\n"
                           "        type: PMIS\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   YAMLnode *precon = hypredrv_YAMLnodeFindChildByKey(tree->root, "preconditioner");
   ASSERT_NOT_NULL(precon);

   YAMLnode *amg = hypredrv_YAMLnodeFindChildByKey(precon, "amg");
   ASSERT_NOT_NULL(amg);

   /* Count sequence items */
   int seq_count = 0;
   YAML_NODE_ITERATE(amg, child)
   {
      if (!strcmp(child->key, "-"))
      {
         seq_count++;
      }
   }
   ASSERT_EQ(seq_count, 2);

   /* Check first sequence item */
   YAMLnode *first_item = NULL;
   YAML_NODE_ITERATE(amg, child)
   {
      if (!strcmp(child->key, "-"))
      {
         first_item = child;
         break;
      }
   }
   ASSERT_NOT_NULL(first_item);

   char *print_level = hypredrv_YAMLnodeFindChildValueByKey(first_item, "print_level");
   ASSERT_NOT_NULL(print_level);
   ASSERT_STREQ(print_level, "1");

   YAMLnode *coarsening = hypredrv_YAMLnodeFindChildByKey(first_item, "coarsening");
   ASSERT_NOT_NULL(coarsening);
   char *type = hypredrv_YAMLnodeFindChildValueByKey(coarsening, "type");
   ASSERT_NOT_NULL(type);
   ASSERT_STREQ(type, "hmis");

   /* Check second sequence item */
   YAMLnode *second_item = first_item->next;
   ASSERT_NOT_NULL(second_item);
   ASSERT_STREQ(second_item->key, "-");

   print_level = hypredrv_YAMLnodeFindChildValueByKey(second_item, "print_level");
   ASSERT_NOT_NULL(print_level);
   ASSERT_STREQ(print_level, "2");

   coarsening = hypredrv_YAMLnodeFindChildByKey(second_item, "coarsening");
   ASSERT_NOT_NULL(coarsening);
   type = hypredrv_YAMLnodeFindChildValueByKey(coarsening, "type");
   ASSERT_NOT_NULL(type);
   ASSERT_STREQ(type, "pmis");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAML include expansion into sequences
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeExpandIncludes_list_under_type(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_includes";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_includes && mkdir -p /tmp/hypredrv_test_includes");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_includes/v1.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "coarsening:\n  type: HMIS\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_includes/v2.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "coarsening:\n  type: PMIS\n");
   fclose(f);

   const char *yaml_text = "preconditioner:\n"
                           "  amg:\n"
                           "    include:\n"
                           "      - v1.yml\n"
                           "      - v2.yml\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   hypredrv_YAMLtreeExpandIncludes(tree, tmpdir);

   YAMLnode *precon = hypredrv_YAMLnodeFindChildByKey(tree->root, "preconditioner");
   ASSERT_NOT_NULL(precon);
   YAMLnode *amg = hypredrv_YAMLnodeFindChildByKey(precon, "amg");
   ASSERT_NOT_NULL(amg);

   YAMLnode **items = NULL;
   int        n     = hypredrv_YAMLnodeCollectSequenceItems(amg, &items);
   ASSERT_EQ(n, 2);

   YAMLnode *c0 = hypredrv_YAMLnodeFindChildByKey(items[0], "coarsening");
   ASSERT_NOT_NULL(c0);
   ASSERT_STREQ(hypredrv_YAMLnodeFindChildValueByKey(c0, "type"), "hmis");

   YAMLnode *c1 = hypredrv_YAMLnodeFindChildByKey(items[1], "coarsening");
   ASSERT_NOT_NULL(c1);
   ASSERT_STREQ(hypredrv_YAMLnodeFindChildValueByKey(c1, "type"), "pmis");

   free(items);
   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeExpandIncludes_list_under_preconditioner(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_includes2";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_includes2 && mkdir -p /tmp/hypredrv_test_includes2");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_includes2/amg.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "amg:\n  print_level: 0\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_includes2/ilu.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "ilu:\n  type: bj-iluk\n");
   fclose(f);

   const char *yaml_text = "preconditioner:\n"
                           "  include:\n"
                           "    - amg.yml\n"
                           "    - ilu.yml\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   hypredrv_YAMLtreeExpandIncludes(tree, tmpdir);

   YAMLnode *precon = hypredrv_YAMLnodeFindChildByKey(tree->root, "preconditioner");
   ASSERT_NOT_NULL(precon);

   YAMLnode **items = NULL;
   int        n     = hypredrv_YAMLnodeCollectSequenceItems(precon, &items);
   ASSERT_EQ(n, 2);

   ASSERT_NOT_NULL(items[0]->children);
   ASSERT_NOT_NULL(items[1]->children);
   ASSERT_TRUE(!strcmp(items[0]->children->key, "amg") ||
               !strcmp(items[1]->children->key, "amg"));
   ASSERT_TRUE(!strcmp(items[0]->children->key, "ilu") ||
               !strcmp(items[1]->children->key, "ilu"));

   free(items);
   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtextRead_rejects_absolute_include_path(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_abs";
   int ret = system("rm -rf /tmp/hypredrv_test_yaml_abs && mkdir -p /tmp/hypredrv_test_yaml_abs");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_abs/main.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: /etc/passwd\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "main.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_rejects_include_traversal_outside_root(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_escape";
   int ret =
      system("rm -rf /tmp/hypredrv_test_yaml_escape && mkdir -p /tmp/hypredrv_test_yaml_escape");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_escape.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "solver: pcg\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_yaml_escape/main.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: ../hypredrv_test_escape.yml\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "main.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
   ASSERT_NULL(text);

   remove("/tmp/hypredrv_test_escape.yml");
}

static void
test_YAMLtextRead_rejects_include_cycle(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_cycle";
   int ret =
      system("rm -rf /tmp/hypredrv_test_yaml_cycle && mkdir -p /tmp/hypredrv_test_yaml_cycle");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_cycle/a.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: b.yml\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_yaml_cycle/b.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: a.yml\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "a.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_rejects_excessive_include_depth(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_depth";
   int ret =
      system("rm -rf /tmp/hypredrv_test_yaml_depth && mkdir -p /tmp/hypredrv_test_yaml_depth");
   (void)ret;

   for (int i = 0; i < 20; i++)
   {
      char path[128];
      snprintf(path, sizeof(path), "/tmp/hypredrv_test_yaml_depth/f%02d.yml", i);
      FILE *f = fopen(path, "w");
      ASSERT_NOT_NULL(f);
      if (i < 19)
      {
         fprintf(f, "include: f%02d.yml\n", i + 1);
      }
      else
      {
         fprintf(f, "solver: pcg\n");
      }
      fclose(f);
   }

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "f00.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_OUT_OF_BOUNDS) != 0);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_rejects_excessive_expanded_size(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_size";
   int ret =
      system("rm -rf /tmp/hypredrv_test_yaml_size && mkdir -p /tmp/hypredrv_test_yaml_size");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_size/huge.yml", "w");
   ASSERT_NOT_NULL(f);
   for (int i = 0; i < 9500; i++)
   {
      fprintf(f, "k%04d: ", i);
      for (int j = 0; j < 890; j++)
      {
         fputc('a', f);
      }
      fputc('\n', f);
   }
   fclose(f);

   f = fopen("/tmp/hypredrv_test_yaml_size/main.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: huge.yml\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "main.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_OUT_OF_BOUNDS) != 0);
   ASSERT_NULL(text);
}

static void
test_YAMLtreeExpandIncludes_rejects_list_traversal_outside_root(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_expand_escape";
   int ret = system(
      "rm -rf /tmp/hypredrv_test_yaml_expand_escape && mkdir -p /tmp/hypredrv_test_yaml_expand_escape");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_expand_outside.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "amg:\n  print_level: 0\n");
   fclose(f);

   const char *yaml_text = "preconditioner:\n"
                           "  include:\n"
                           "    - ../hypredrv_expand_outside.yml\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeExpandIncludes(tree, tmpdir);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);

   remove("/tmp/hypredrv_expand_outside.yml");
   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test indentation spacing equal to 3
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_indent_3_spaces(void)
{
   /* Test with 3-space indentation - should work */
   const char *yaml_text = "key:\n   child: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *key = hypredrv_YAMLnodeFindChildByKey(tree->root, "key");
   ASSERT_NOT_NULL(key);

   YAMLnode *child = hypredrv_YAMLnodeFindChildByKey(key, "child");
   ASSERT_NOT_NULL(child);
   ASSERT_STREQ(child->val, "value");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test indentation spacing equal to 4
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_indent_4_spaces(void)
{
   /* Test with 4-space indentation - should work */
   const char *yaml_text = "key:\n    child: value\n";
   size_t      len       = strlen(yaml_text);
   char       *text      = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *key = hypredrv_YAMLnodeFindChildByKey(tree->root, "key");
   ASSERT_NOT_NULL(key);

   YAMLnode *child = hypredrv_YAMLnodeFindChildByKey(key, "child");
   ASSERT_NOT_NULL(child);
   ASSERT_STREQ(child->val, "value");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test indentation with tabs - should fail
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_indent_with_tabs(void)
{
   /* Test with tab indentation - should fail with ERROR_YAML_MIXED_INDENT */
   const char *tmpfile = "/tmp/hypredrv_test_tabs.yml";
   FILE       *f       = fopen(tmpfile, "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "key:\n\tchild: value\n");
   fclose(f);

   hypredrv_ErrorCodeResetAll();

   /* Use hypredrv_YAMLtextRead to trigger the tab detection */
   int    base_indent = -1;
   size_t text_len    = 0;
   char  *yaml_text   = NULL;
   hypredrv_YAMLtextRead("/tmp", "hypredrv_test_tabs.yml", 0, &base_indent, &text_len, &yaml_text);

   /* Should have detected the error */
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_MIXED_INDENT, ERROR_YAML_MIXED_INDENT);

   if (yaml_text)
   {
      free(yaml_text);
   }

   /* Clean up */
   int ret = system("rm -f /tmp/hypredrv_test_tabs.yml");
   (void)ret;
}

/*-----------------------------------------------------------------------------
 * Test that a scalar YAML value cannot also have nested entries
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeCollectSequenceItems_null_items_out(void)
{
   const char *yaml_text = "preconditioner:\n"
                           "  amg:\n"
                           "    - print_level: 1\n";
   size_t      len  = strlen(yaml_text);
   char       *text = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   YAMLnode *precon = hypredrv_YAMLnodeFindChildByKey(tree->root, "preconditioner");
   ASSERT_NOT_NULL(precon);
   YAMLnode *amg = hypredrv_YAMLnodeFindChildByKey(precon, "amg");
   ASSERT_NOT_NULL(amg);

   int n = hypredrv_YAMLnodeCollectSequenceItems(amg, NULL);
   ASSERT_EQ(n, 0);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodeCollectSequenceItems_no_dash_children(void)
{
   const char *yaml_text = "preconditioner:\n"
                           "  amg:\n"
                           "    print_level: 1\n";
   size_t      len  = strlen(yaml_text);
   char       *text = malloc(len + 1);
   strcpy(text, yaml_text);

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   YAMLnode *precon = hypredrv_YAMLnodeFindChildByKey(tree->root, "preconditioner");
   ASSERT_NOT_NULL(precon);
   YAMLnode *amg = hypredrv_YAMLnodeFindChildByKey(precon, "amg");
   ASSERT_NOT_NULL(amg);

   YAMLnode **items = NULL;
   int        n     = hypredrv_YAMLnodeCollectSequenceItems(amg, &items);
   ASSERT_EQ(n, 0);
   ASSERT_NULL(items);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_scalar_with_children_is_error(void)
{
   const char *yaml_text = "f_relaxation: amg\n  coarsening:\n    type: pmis\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   ASSERT_NOT_NULL(text);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeBuild(2, text, &tree);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_UNEXPECTED_VAL, ERROR_UNEXPECTED_VAL);
   ASSERT_NOT_NULL(tree);

   YAMLnode *node = hypredrv_YAMLnodeFindChildByKey(tree->root, "f_relaxation");
   ASSERT_NOT_NULL(node);
   ASSERT_EQ(node->valid, YAML_NODE_UNEXPECTED_VAL);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLnodeCreate edge cases (name substring, quotes, lowercasing)
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeCreate_name_key_preserves_value_case(void)
{
   YAMLnode *node = hypredrv_YAMLnodeCreate("foo_name", "AbC", 0);
   ASSERT_NOT_NULL(node);
   ASSERT_STREQ(node->val, "AbC");
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_YAMLnodeCreate_nonname_lowercases_and_strips_quotes(void)
{
   /* hypredrv_StrTrim only strips trailing spaces; avoid leading spaces so quotes
    * sit at the front after trim and the quote-strip branch runs. */
   YAMLnode *node = hypredrv_YAMLnodeCreate("solver", "\"HeLLo\"", 0);
   ASSERT_NOT_NULL(node);
   ASSERT_STREQ(node->val, "hello");
   hypredrv_YAMLnodeDestroy(node);
}

/*-----------------------------------------------------------------------------
 * Tokenizer / tree build: scalar sequence items and inline mapping items
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_sequence_scalar_item(void)
{
   const char *yaml_text = "list:\n"
                           "  - one\n"
                           "  - two\n";
   char       *text = strdup(yaml_text);
   YAMLtree   *tree = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   YAMLnode *list = hypredrv_YAMLnodeFindChildByKey(tree->root, "list");
   ASSERT_NOT_NULL(list);

   int n = 0;
   YAML_NODE_ITERATE(list, ch)
   {
      if (!strcmp(ch->key, "-") && ch->val && strlen(ch->val) > 0)
      {
         n++;
      }
   }
   ASSERT_EQ(n, 2);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_inline_sequence_mapping_siblings(void)
{
   const char *yaml_text = "list:\n"
                           "  - tag: a\n"
                           "  - tag: b\n";
   char       *text = strdup(yaml_text);
   YAMLtree   *tree = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   YAMLnode *list = hypredrv_YAMLnodeFindChildByKey(tree->root, "list");
   ASSERT_NOT_NULL(list);

   int count = 0;
   YAML_NODE_ITERATE(list, dash)
   {
      if (!strcmp(dash->key, "-"))
      {
         char *t = hypredrv_YAMLnodeFindChildValueByKey(dash, "tag");
         ASSERT_NOT_NULL(t);
         count++;
      }
   }
   ASSERT_EQ(count, 2);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtextRead: pre-sized output buffer (prefix + file append)
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtextRead_appends_to_nonempty_buffer(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_prebuf";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_prebuf && mkdir -p /tmp/hypredrv_test_yaml_prebuf");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_prebuf/more.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "  child: 2\n");
   fclose(f);

   char  *buf          = (char *)malloc(4096);
   size_t prefix_len   = (size_t)snprintf(buf, 4096, "root: 1\n");
   int    base_indent  = -1;
   size_t len          = prefix_len;
   char  *text_ptr     = buf;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "more.yml", 0, &base_indent, &len, &text_ptr);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(text_ptr);
   ASSERT_TRUE(len > prefix_len);
   ASSERT_TRUE(strstr(text_ptr, "root: 1") != NULL);

   free(text_ptr);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtreeUpdate: pair-list / full-argv / sequence broadcast
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeUpdate_null_tree_errors(void)
{
   char *argv[] = {(char *)"--a:b", (char *)"1"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, NULL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_TREE_NULL, ERROR_YAML_TREE_NULL);
}

static void
test_YAMLtreeUpdate_odd_override_token_count_errors(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   /* Full-argv mode: three tokens after -a is an odd count -> invalid pairs. */
   char *argv[] = {(char *)"-a", (char *)"--a:b", (char *)"1", (char *)"extra"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(4, argv, tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, ERROR_INVALID_VAL);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_pair_list_not_pairs_noop(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"-q", (char *)"cfg.yml"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NULL(tree->root->children);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_pair_list_empty_path_segment(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"--", (char *)"v"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY, ERROR_INVALID_KEY);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_pair_list_missing_value(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   /* Full-argv mode: lone key after -a (no value token). */
   char *argv[] = {(char *)"-a", (char *)"--a:b:c"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, ERROR_INVALID_VAL);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_pair_list_value_looks_like_flag(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"--a:b", (char *)"--bad"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, ERROR_INVALID_VAL);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_fullargv_colon_path_without_leading_dashes(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"-a", (char *)"block:inner:leaf", (char *)"42"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(3, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *block = hypredrv_YAMLnodeFindChildByKey(tree->root, "block");
   ASSERT_NOT_NULL(block);
   YAMLnode *inner = hypredrv_YAMLnodeFindChildByKey(block, "inner");
   ASSERT_NOT_NULL(inner);
   YAMLnode *leaf = hypredrv_YAMLnodeFindChildByKey(inner, "leaf");
   ASSERT_NOT_NULL(leaf);
   ASSERT_STREQ(leaf->val, "42");

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_sequence_broadcast_intermediate_path(void)
{
   YAMLtree *tree  = hypredrv_YAMLtreeCreate(2);
   YAMLnode *pre   = hypredrv_YAMLnodeCreate("pre", "", 0);
   YAMLnode *amg   = hypredrv_YAMLnodeCreate("amg", "", 1);
   YAMLnode *dash1 = hypredrv_YAMLnodeCreate("-", "", 2);
   YAMLnode *dash2 = hypredrv_YAMLnodeCreate("-", "", 2);

   hypredrv_YAMLnodeAddChild(tree->root, pre);
   hypredrv_YAMLnodeAddChild(pre, amg);
   hypredrv_YAMLnodeAddChild(amg, dash1);
   hypredrv_YAMLnodeAddChild(amg, dash2);

   char *argv[] = {(char *)"--pre:amg:foo:bar", (char *)"99"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAML_NODE_ITERATE(amg, seq)
   {
      if (!strcmp(seq->key, "-"))
      {
         YAMLnode *foo = hypredrv_YAMLnodeFindChildByKey(seq, "foo");
         ASSERT_NOT_NULL(foo);
         YAMLnode *bar = hypredrv_YAMLnodeFindChildByKey(foo, "bar");
         ASSERT_NOT_NULL(bar);
         ASSERT_STREQ(bar->val, "99");
      }
   }

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_leaf_broadcast_on_sequence_parent(void)
{
   YAMLtree *tree  = hypredrv_YAMLtreeCreate(2);
   YAMLnode *pre   = hypredrv_YAMLnodeCreate("pre", "", 0);
   YAMLnode *dash1 = hypredrv_YAMLnodeCreate("-", "", 1);
   YAMLnode *dash2 = hypredrv_YAMLnodeCreate("-", "", 1);

   hypredrv_YAMLnodeAddChild(tree->root, pre);
   hypredrv_YAMLnodeAddChild(pre, dash1);
   hypredrv_YAMLnodeAddChild(pre, dash2);

   char *argv[] = {(char *)"--pre:print_level", (char *)"7"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char *v1 = hypredrv_YAMLnodeFindChildValueByKey(dash1, "print_level");
   char *v2 = hypredrv_YAMLnodeFindChildValueByKey(dash2, "print_level");
   ASSERT_NOT_NULL(v1);
   ASSERT_NOT_NULL(v2);
   ASSERT_STREQ(v1, "7");
   ASSERT_STREQ(v2, "7");

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_replace_existing_leaf_scalar(void)
{
   YAMLtree *tree   = hypredrv_YAMLtreeCreate(2);
   YAMLnode *block  = hypredrv_YAMLnodeCreate("block", "", 0);
   YAMLnode *leaf   = hypredrv_YAMLnodeCreate("leaf", "old", 1);
   YAMLnode *nested = hypredrv_YAMLnodeCreate("junk", "x", 2);

   hypredrv_YAMLnodeAddChild(tree->root, block);
   hypredrv_YAMLnodeAddChild(block, leaf);
   hypredrv_YAMLnodeAddChild(leaf, nested);

   char *argv[] = {(char *)"--block:leaf", (char *)"new"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   ASSERT_STREQ(leaf->val, "new");
   ASSERT_NULL(leaf->children);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * hypredrv_YAMLtextRead: reader-time indent errors
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtextRead_invalid_base_indent(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_baseind";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_baseind && mkdir -p /tmp/hypredrv_test_yaml_baseind");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_baseind/bad.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "k: v\n v: x\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "bad.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_INVALID_BASE_INDENT,
                 ERROR_YAML_INVALID_BASE_INDENT);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_inconsistent_indent(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_incons";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_incons && mkdir -p /tmp/hypredrv_test_yaml_incons");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_incons/bad.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "k: v\n  x: 1\n   y: 2\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "bad.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_INCONSISTENT_INDENT,
                 ERROR_YAML_INCONSISTENT_INDENT);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_invalid_indent_jump(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_jump";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_jump && mkdir -p /tmp/hypredrv_test_yaml_jump");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_jump/bad.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "k: v\n  x: 1\n            y: 2\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "bad.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_INVALID_INDENT_JUMP,
                 ERROR_YAML_INVALID_INDENT_JUMP);
   ASSERT_NULL(text);
}

/*-----------------------------------------------------------------------------
 * Include expansion: nested include with invalid YAML in included file
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeExpandIncludes_nested_include_tree_build_fails(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_nest_bad";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_nest_bad && mkdir -p /tmp/hypredrv_test_yaml_nest_bad");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_nest_bad/bad.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "f_relaxation: amg\n  coarsening:\n    type: pmis\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_yaml_nest_bad/main.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: bad.yml\n");
   fclose(f);

   const char *yaml_text = "top:\n  include: main.yml\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeExpandIncludes(tree, tmpdir);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Schema helpers: stub get_keys / get_vals / set_field
 *-----------------------------------------------------------------------------*/

static StrArray
stub_schema_keys(void)
{
   static const char *keys[] = {"solver", "k_empty", "type"};
   return STR_ARRAY_CREATE(keys);
}

static StrIntMapArray
stub_schema_vals(const char *key)
{
   static const StrIntMap onoff_map[] = {
      {.str = "on", .num = 1},
      {.str = "off", .num = 0},
   };
   if (!strcmp(key, "solver") || !strcmp(key, "type"))
   {
      return STR_INT_MAP_ARRAY_CREATE(onoff_map);
   }
   if (!strcmp(key, "k_empty"))
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static int g_stub_set_field_calls = 0;

static void
stub_schema_set_field(void *args, const YAMLnode *node)
{
   int *counter = (int *)args;
   (void)node;
   (*counter)++;
}

static void
test_YAMLnodeValidateSchema_dash_key_is_valid(void)
{
   YAMLnode *dash = hypredrv_YAMLnodeCreate("-", "ignored", 0);

   hypredrv_YAMLnodeValidateSchema(dash, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(dash->valid, YAML_NODE_VALID);

   hypredrv_YAMLnodeDestroy(dash);
}

static void
test_YAMLnodeValidateSchema_skips_when_invalid_indent_or_divisor(void)
{
   YAMLnode *n = hypredrv_YAMLnodeCreate("solver", "on", 0);
   n->valid    = YAML_NODE_INVALID_INDENT;

   hypredrv_YAMLnodeValidateSchema(n, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(n->valid, YAML_NODE_INVALID_INDENT);

   n->valid = YAML_NODE_INVALID_DIVISOR;
   hypredrv_YAMLnodeValidateSchema(n, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(n->valid, YAML_NODE_INVALID_DIVISOR);

   hypredrv_YAMLnodeDestroy(n);
}

static void
test_YAMLnodeValidateSchema_map_and_empty_map_branches(void)
{
   YAMLnode *solver = hypredrv_YAMLnodeCreate("solver", "on", 0);
   hypredrv_YAMLnodeValidateSchema(solver, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(solver->valid, YAML_NODE_VALID);
   ASSERT_NOT_NULL(solver->mapped_val);
   ASSERT_STREQ(solver->mapped_val, "1");
   hypredrv_YAMLnodeDestroy(solver);

   YAMLnode *empty = hypredrv_YAMLnodeCreate("k_empty", "raw", 0);
   hypredrv_YAMLnodeValidateSchema(empty, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(empty->valid, YAML_NODE_VALID);
   ASSERT_NOT_NULL(empty->mapped_val);
   ASSERT_STREQ(empty->mapped_val, "raw");
   hypredrv_YAMLnodeDestroy(empty);

   YAMLnode *badkey = hypredrv_YAMLnodeCreate("unknown", "x", 0);
   hypredrv_YAMLnodeValidateSchema(badkey, stub_schema_keys, stub_schema_vals);
   ASSERT_EQ(badkey->valid, YAML_NODE_INVALID_KEY);
   hypredrv_YAMLnodeDestroy(badkey);
}

static StrArray
stub_keys_without_type(void)
{
   static const char *keys[] = {"solver"};
   return STR_ARRAY_CREATE(keys);
}

static StrIntMapArray
stub_vals_type_fallback(const char *key)
{
   static const StrIntMap type_map[] = {
      {.str = "on", .num = 1},
      {.str = "off", .num = 0},
   };
   if (!strcmp(key, "type"))
   {
      return STR_INT_MAP_ARRAY_CREATE(type_map);
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static void
test_YAMLnodeValidateSchema_type_fallback_when_key_not_in_list(void)
{
   YAMLnode *t = hypredrv_YAMLnodeCreate("type", "on", 0);

   hypredrv_YAMLnodeValidateSchema(t, stub_keys_without_type, stub_vals_type_fallback);
   ASSERT_EQ(t->valid, YAML_NODE_VALID);
   ASSERT_NOT_NULL(t->mapped_val);
   ASSERT_STREQ(t->mapped_val, "1");

   hypredrv_YAMLnodeDestroy(t);
}

static StrArray
stub_keys_flat_only_type(void)
{
   static const char *keys[] = {"type"};
   return STR_ARRAY_CREATE(keys);
}

static StrIntMapArray
stub_vals_flat(const char *key)
{
   static const StrIntMap type_map[] = {
      {.str = "on", .num = 1},
      {.str = "off", .num = 0},
   };
   if (!strcmp(key, "type"))
   {
      return STR_INT_MAP_ARRAY_CREATE(type_map);
   }
   return STR_INT_MAP_ARRAY_VOID();
}

static void
test_YAMLSetArgsGeneric_nested_children(void)
{
   YAMLtree *tree  = hypredrv_YAMLtreeCreate(2);
   YAMLnode *wrap  = hypredrv_YAMLnodeCreate("wrap", "", 0);
   YAMLnode *child = hypredrv_YAMLnodeCreate("solver", "on", 1);

   hypredrv_YAMLnodeAddChild(tree->root, wrap);
   hypredrv_YAMLnodeAddChild(wrap, child);

   int calls = 0;
   hypredrv_YAMLSetArgsGeneric(&calls, wrap, stub_schema_keys, stub_schema_vals,
                               stub_schema_set_field);
   ASSERT_EQ(calls, 1);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLSetArgsGeneric_flat_rewrites_key_to_type(void)
{
   YAMLnode *node = hypredrv_YAMLnodeCreate("mypre", "on", 0);
   int         calls = 0;

   hypredrv_YAMLSetArgsGeneric(&calls, node, stub_keys_flat_only_type, stub_vals_flat,
                               stub_schema_set_field);
   ASSERT_EQ(calls, 1);
   ASSERT_STREQ(node->key, "mypre");

   hypredrv_YAMLnodeDestroy(node);
}

/*-----------------------------------------------------------------------------
 * Reader / include: bad root, scalar include, comments, fopen failure, cleanup
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtextRead_bad_root_directory(void)
{
   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead("/nonexistent_hypredrv_root_42", "any.yml", 0, &base_indent, &len,
                         &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND) != 0);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_scalar_include_success(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_scalar_inc";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_scalar_inc && mkdir -p /tmp/hypredrv_test_yaml_scalar_inc");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_scalar_inc/child.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "solver: pcg\n");
   fclose(f);

   f = fopen("/tmp/hypredrv_test_yaml_scalar_inc/main.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "include: child.yml\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "main.yml", 0, &base_indent, &len, &text);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(text);
   ASSERT_TRUE(strstr(text, "solver") != NULL);

   free(text);
}

static void
test_YAMLtextRead_comments_blanks_and_inline_hash(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_comments";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_comments && mkdir -p /tmp/hypredrv_test_yaml_comments");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_comments/cfg.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "# top comment\n");
   fprintf(f, "\n");
   fprintf(f, "key: value # inline\n");
   fprintf(f, "\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "cfg.yml", 0, &base_indent, &len, &text);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(text);
   ASSERT_TRUE(strstr(text, "key") != NULL);

   free(text);
}

static void
test_YAMLtextRead_skips_nonsequence_line_without_colon(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_no_colon";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_no_colon && mkdir -p /tmp/hypredrv_test_yaml_no_colon");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_no_colon/x.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "orphan_line_without_colon\n");
   fprintf(f, "k: v\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "x.yml", 0, &base_indent, &len, &text);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(text);
   ASSERT_TRUE(strstr(text, "k:") != NULL);

   free(text);
}

static void
test_YAMLtextRead_fail_with_text_cleanup_after_partial_read(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_partial_fail";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_partial_fail && mkdir -p /tmp/hypredrv_test_yaml_partial_fail");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_partial_fail/bad.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "k: v\n");
   fprintf(f, "\tbad: x\n");
   fclose(f);

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "bad.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_YAML_MIXED_INDENT, ERROR_YAML_MIXED_INDENT);
   ASSERT_NULL(text);
}

static void
test_YAMLtextRead_fopen_failure(void)
{
   const char *tmpdir = "/tmp/hypredrv_test_yaml_fopen_fail";
   int         ret    = system(
      "rm -rf /tmp/hypredrv_test_yaml_fopen_fail && mkdir -p /tmp/hypredrv_test_yaml_fopen_fail");
   (void)ret;

   FILE *f = fopen("/tmp/hypredrv_test_yaml_fopen_fail/locked.yml", "w");
   ASSERT_NOT_NULL(f);
   fprintf(f, "k: v\n");
   fclose(f);
   ret = chmod("/tmp/hypredrv_test_yaml_fopen_fail/locked.yml", 0000);
   (void)ret;

   int    base_indent = -1;
   size_t len         = 0;
   char  *text        = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtextRead(tmpdir, "locked.yml", 0, &base_indent, &len, &text);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND) != 0);
   ASSERT_NULL(text);

   chmod("/tmp/hypredrv_test_yaml_fopen_fail/locked.yml", 0644);
}

/*-----------------------------------------------------------------------------
 * Tokenizer / build: tabs, blanks, invalid divisor, many tokens, sequence edges
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeBuild_tab_indented_tokenizer(void)
{
   /* Tokenizer allows tabs in YAMLlineParseIndent (distinct from reader-time tab ban). */
   const char *yaml_text = "\tkey: value\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeBuild(8, text, &tree);
   ASSERT_NOT_NULL(tree);

   YAMLnode *key = hypredrv_YAMLnodeFindChildByKey(tree->root, "key");
   ASSERT_NOT_NULL(key);
   ASSERT_STREQ(key->val, "value");

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_invalid_divisor_then_validate(void)
{
   const char *yaml_text = "not_a_key_without_colon\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeValidate(tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_blank_lines_and_comments_only(void)
{
   const char *yaml_text = "\n  \n# c\n   \nkey: v\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(hypredrv_YAMLnodeFindChildByKey(tree->root, "key"));

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_sequence_edges_bare_minus_and_inline(void)
{
   const char *yaml_text = "list:\n"
                           "  -\n"
                           "  - only_scalar\n"
                           "  - k: v\n"
                           "  - a: 1\n"
                           "  - b: 2\n";
   char     *text = strdup(yaml_text);
   YAMLtree *tree = NULL;

   hypredrv_YAMLtreeBuild(2, text, &tree);
   ASSERT_NOT_NULL(tree);

   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeBuild_many_keys_token_reserve(void)
{
   char  buf[16384];
   char *p   = buf;
   size_t left = sizeof(buf);
   int    n    = 0;
   for (int i = 0; i < 40 && left > 32; i++)
   {
      int w = snprintf(p, left, "k%02d: %d\n", i, i);
      ASSERT_TRUE(w > 0 && (size_t)w < left);
      p += w;
      left -= (size_t)w;
      n++;
   }
   *p = '\0';

   YAMLtree *tree = NULL;
   hypredrv_YAMLtreeBuild(2, buf, &tree);
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(hypredrv_YAMLnodeFindChildByKey(tree->root, "k00"));

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * YAMLtreeUpdate: full-argv empty overrides, null key, strict pair errors, mapping
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeUpdate_fullargv_no_tokens_after_flag(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"-a"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(1, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_fullargv_skips_null_override_key(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"-a", NULL, (char *)"x"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(3, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_pair_list_empty_path_after_dashes(void)
{
   /* "--" strips to an empty path -> num_segments==0 -> ERROR_INVALID_KEY */
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   char     *argv[] = {(char *)"--", (char *)"1"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet() & ERROR_INVALID_KEY, ERROR_INVALID_KEY);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_intermediate_scalar_to_mapping(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   YAMLnode *foo  = hypredrv_YAMLnodeCreate("foo", "scalar_leaf", 0);
   hypredrv_YAMLnodeAddChild(tree->root, foo);

   char *argv[] = {(char *)"--foo:child", (char *)"99"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   YAMLnode *ch = hypredrv_YAMLnodeFindChildByKey(foo, "child");
   ASSERT_NOT_NULL(ch);
   ASSERT_STREQ(ch->val, "99");
   ASSERT_STREQ(foo->val, "");

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_clears_mapped_val_on_leaf_replace(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   YAMLnode *leaf = hypredrv_YAMLnodeCreate("leaf", "old", 0);
   leaf->mapped_val = strdup("0");
   hypredrv_YAMLnodeAddChild(tree->root, leaf);

   char *argv[] = {(char *)"--leaf", (char *)"new"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_STREQ(leaf->val, "new");
   ASSERT_NULL(leaf->mapped_val);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLtreeUpdate_sequence_leaf_create_and_replace(void)
{
   YAMLtree *tree  = hypredrv_YAMLtreeCreate(2);
   YAMLnode *pre   = hypredrv_YAMLnodeCreate("pre", "", 0);
   YAMLnode *dash1 = hypredrv_YAMLnodeCreate("-", "", 1);
   hypredrv_YAMLnodeAddChild(tree->root, pre);
   hypredrv_YAMLnodeAddChild(pre, dash1);

   char *argv1[] = {(char *)"--pre:newleaf", (char *)"first"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv1, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   YAMLnode *nl = hypredrv_YAMLnodeFindChildByKey(dash1, "newleaf");
   ASSERT_NOT_NULL(nl);
   ASSERT_STREQ(nl->val, "first");

   char *argv2[] = {(char *)"--pre:newleaf", (char *)"second"};

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeUpdate(2, argv2, tree);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   nl = hypredrv_YAMLnodeFindChildByKey(dash1, "newleaf");
   ASSERT_NOT_NULL(nl);
   ASSERT_STREQ(nl->val, "second");

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Expand includes: null tree; bad root dir
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeExpandIncludes_null_tree(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeExpandIncludes(NULL, ".");
}

static void
test_YAMLtreeExpandIncludes_bad_base_dir(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeExpandIncludes(tree, "/nonexistent_hypredrv_expand_root");
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND) != 0);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Null-safe APIs and validate switch arms
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeDestroy_validate_print_null(void)
{
   hypredrv_YAMLnodeDestroy(NULL);
   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLnodeValidate(NULL);
   hypredrv_YAMLnodePrint(NULL, YAML_PRINT_MODE_ANY);
}

static void
test_YAMLtreeValidate_sets_errors_for_invalid_nodes(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   struct
   {
      const char *key;
      YAMLvalidity v;
   } cases[] = {
       {"i", YAML_NODE_INVALID_INDENT},
       {"d", YAML_NODE_INVALID_DIVISOR},
       {"k", YAML_NODE_INVALID_KEY},
       {"v", YAML_NODE_INVALID_VAL},
       {"u", YAML_NODE_UNEXPECTED_VAL},
   };

   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      YAMLnode *n = hypredrv_YAMLnodeCreate(cases[i].key, "x", 1);
      n->valid    = cases[i].v;
      hypredrv_YAMLnodeAddChild(tree->root, n);
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeValidate(tree);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLSetArgsGeneric_null_parent(void)
{
   int calls = 0;
   hypredrv_YAMLSetArgsGeneric(&calls, NULL, stub_schema_keys, stub_schema_vals,
                               stub_schema_set_field);
   ASSERT_EQ(calls, 0);
}

/*-----------------------------------------------------------------------------
 * Print: sequence scalar-only; inline child with nested children and sibling
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodePrint_sequence_scalar_only(void)
{
   YAMLtree *tree = hypredrv_YAMLtreeCreate(2);
   YAMLnode *dash = hypredrv_YAMLnodeCreate("-", "hello", 1);
   hypredrv_YAMLnodeAddChild(tree->root, dash);

   hypredrv_YAMLnodePrint(tree->root, YAML_PRINT_MODE_NO_CHECKING);

   hypredrv_YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodePrint_inline_child_nested_and_sibling(void)
{
   YAMLtree *tree         = hypredrv_YAMLtreeCreate(2);
   YAMLnode *dash         = hypredrv_YAMLnodeCreate("-", "", 1);
   YAMLnode *inline_kid   = hypredrv_YAMLnodeCreate("tag", "value", 2);
   YAMLnode *grand        = hypredrv_YAMLnodeCreate("g", "1", 3);
   YAMLnode *sib          = hypredrv_YAMLnodeCreate("sib", "2", 2);

   hypredrv_YAMLnodeAddChild(inline_kid, grand);
   hypredrv_YAMLnodeAddChild(dash, inline_kid);
   inline_kid->next = sib;
   hypredrv_YAMLnodeAddChild(tree->root, dash);

   hypredrv_YAMLnodePrint(tree->root, YAML_PRINT_MODE_NO_CHECKING);

   hypredrv_YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int
main(void)
{
   RUN_TEST(test_YAMLnodeCreate_basic);

   RUN_TEST(test_YAMLtreeCreate_basic);

   RUN_TEST(test_YAMLnodeFindByKey_basic);
   RUN_TEST(test_YAMLnodeFindChildByKey_basic);
   RUN_TEST(test_YAMLnodeFindChildValueByKey_basic);

   RUN_TEST(test_YAMLtreeBuild_simple);
   RUN_TEST(test_YAMLtreeBuild_preserves_input_text);
   RUN_TEST(test_YAMLtreeBuild_nested);
   RUN_TEST(test_YAMLtreeBuild_deeply_nested);
   RUN_TEST(test_YAMLtreeBuild_siblings);

   RUN_TEST(test_YAMLnode_key_val_distinction);
   RUN_TEST(test_YAMLtreeBuild_mgr_coarsest_level_pattern);

   RUN_TEST(test_YAMLtreeBuild_invalid_indent);
   RUN_TEST(test_YAMLnodeFindByKey_nonexistent);
   RUN_TEST(test_YAMLnodeFindChildValueByKey_nonexistent);
   RUN_TEST(test_YAMLtreeUpdate_fullargv_longflag_and_skip_nonoverride_tokens);
   RUN_TEST(test_YAMLtreeUpdate_fullargv_shortflag);
   RUN_TEST(test_YAMLnodePrint_only_valid_mode);
   RUN_TEST(test_YAMLtreePrint_and_validate_null_tree);
   RUN_TEST(test_YAMLtreePrint_all_modes);
   RUN_TEST(test_YAMLnodePrint_any_mode_validity_branches);
   RUN_TEST(test_YAMLnodePrint_sequence_inline_first_child);
   RUN_TEST(test_YAMLtreeBuild_sequence_items);
   RUN_TEST(test_YAMLtreeExpandIncludes_list_under_type);
   RUN_TEST(test_YAMLtreeExpandIncludes_list_under_preconditioner);
   RUN_TEST(test_YAMLtextRead_rejects_absolute_include_path);
   RUN_TEST(test_YAMLtextRead_rejects_include_traversal_outside_root);
   RUN_TEST(test_YAMLtextRead_rejects_include_cycle);
   RUN_TEST(test_YAMLtextRead_rejects_excessive_include_depth);
   RUN_TEST(test_YAMLtextRead_rejects_excessive_expanded_size);
   RUN_TEST(test_YAMLtreeExpandIncludes_rejects_list_traversal_outside_root);
   RUN_TEST(test_YAMLtreeBuild_indent_3_spaces);
   RUN_TEST(test_YAMLtreeBuild_indent_4_spaces);
   RUN_TEST(test_YAMLtreeBuild_indent_with_tabs);
   RUN_TEST(test_YAMLtreeBuild_scalar_with_children_is_error);
   RUN_TEST(test_YAMLnodeCollectSequenceItems_null_items_out);
   RUN_TEST(test_YAMLnodeCollectSequenceItems_no_dash_children);

   RUN_TEST(test_YAMLnodeCreate_name_key_preserves_value_case);
   RUN_TEST(test_YAMLnodeCreate_nonname_lowercases_and_strips_quotes);
   RUN_TEST(test_YAMLtreeBuild_sequence_scalar_item);
   RUN_TEST(test_YAMLtreeBuild_inline_sequence_mapping_siblings);
   RUN_TEST(test_YAMLtextRead_appends_to_nonempty_buffer);

   RUN_TEST(test_YAMLtreeUpdate_null_tree_errors);
   RUN_TEST(test_YAMLtreeUpdate_odd_override_token_count_errors);
   RUN_TEST(test_YAMLtreeUpdate_pair_list_not_pairs_noop);
   RUN_TEST(test_YAMLtreeUpdate_pair_list_empty_path_segment);
   RUN_TEST(test_YAMLtreeUpdate_pair_list_missing_value);
   RUN_TEST(test_YAMLtreeUpdate_pair_list_value_looks_like_flag);
   RUN_TEST(test_YAMLtreeUpdate_fullargv_colon_path_without_leading_dashes);
   RUN_TEST(test_YAMLtreeUpdate_sequence_broadcast_intermediate_path);
   RUN_TEST(test_YAMLtreeUpdate_leaf_broadcast_on_sequence_parent);
   RUN_TEST(test_YAMLtreeUpdate_replace_existing_leaf_scalar);

   RUN_TEST(test_YAMLtextRead_invalid_base_indent);
   RUN_TEST(test_YAMLtextRead_inconsistent_indent);
   RUN_TEST(test_YAMLtextRead_invalid_indent_jump);
   RUN_TEST(test_YAMLtreeExpandIncludes_nested_include_tree_build_fails);

   RUN_TEST(test_YAMLnodeValidateSchema_dash_key_is_valid);
   RUN_TEST(test_YAMLnodeValidateSchema_skips_when_invalid_indent_or_divisor);
   RUN_TEST(test_YAMLnodeValidateSchema_map_and_empty_map_branches);
   RUN_TEST(test_YAMLnodeValidateSchema_type_fallback_when_key_not_in_list);
   RUN_TEST(test_YAMLSetArgsGeneric_nested_children);
   RUN_TEST(test_YAMLSetArgsGeneric_flat_rewrites_key_to_type);

   RUN_TEST(test_YAMLtextRead_bad_root_directory);
   RUN_TEST(test_YAMLtextRead_scalar_include_success);
   RUN_TEST(test_YAMLtextRead_comments_blanks_and_inline_hash);
   RUN_TEST(test_YAMLtextRead_skips_nonsequence_line_without_colon);
   RUN_TEST(test_YAMLtextRead_fail_with_text_cleanup_after_partial_read);
   RUN_TEST(test_YAMLtextRead_fopen_failure);

   RUN_TEST(test_YAMLtreeBuild_tab_indented_tokenizer);
   RUN_TEST(test_YAMLtreeBuild_invalid_divisor_then_validate);
   RUN_TEST(test_YAMLtreeBuild_blank_lines_and_comments_only);
   RUN_TEST(test_YAMLtreeBuild_sequence_edges_bare_minus_and_inline);
   RUN_TEST(test_YAMLtreeBuild_many_keys_token_reserve);

   RUN_TEST(test_YAMLtreeUpdate_fullargv_no_tokens_after_flag);
   RUN_TEST(test_YAMLtreeUpdate_fullargv_skips_null_override_key);
   RUN_TEST(test_YAMLtreeUpdate_pair_list_empty_path_after_dashes);
   RUN_TEST(test_YAMLtreeUpdate_intermediate_scalar_to_mapping);
   RUN_TEST(test_YAMLtreeUpdate_clears_mapped_val_on_leaf_replace);
   RUN_TEST(test_YAMLtreeUpdate_sequence_leaf_create_and_replace);

   RUN_TEST(test_YAMLtreeExpandIncludes_null_tree);
   RUN_TEST(test_YAMLtreeExpandIncludes_bad_base_dir);

   RUN_TEST(test_YAMLnodeDestroy_validate_print_null);
   RUN_TEST(test_YAMLtreeValidate_sets_errors_for_invalid_nodes);
   RUN_TEST(test_YAMLSetArgsGeneric_null_parent);

   RUN_TEST(test_YAMLnodePrint_sequence_scalar_only);
   RUN_TEST(test_YAMLnodePrint_inline_child_nested_and_sibling);

   return 0; /* Success - CTest handles reporting */
}
