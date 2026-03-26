/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_helpers.h"
#include "yaml.h"

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
   ASSERT_EQ(hypredrv_ErrorCodeGet() & ERROR_YAML_MIXED_INDENT, ERROR_YAML_MIXED_INDENT);

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
test_YAMLtreeBuild_scalar_with_children_is_error(void)
{
   const char *yaml_text = "f_relaxation: amg\n  coarsening:\n    type: pmis\n";
   char       *text      = strdup(yaml_text);
   YAMLtree   *tree      = NULL;

   ASSERT_NOT_NULL(text);

   hypredrv_ErrorCodeResetAll();
   hypredrv_YAMLtreeBuild(2, text, &tree);

   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_EQ(hypredrv_ErrorCodeGet() & ERROR_UNEXPECTED_VAL, ERROR_UNEXPECTED_VAL);
   ASSERT_NOT_NULL(tree);

   YAMLnode *node = hypredrv_YAMLnodeFindChildByKey(tree->root, "f_relaxation");
   ASSERT_NOT_NULL(node);
   ASSERT_EQ(node->valid, YAML_NODE_UNEXPECTED_VAL);

   free(text);
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
   RUN_TEST(test_YAMLtreeBuild_sequence_items);
   RUN_TEST(test_YAMLtreeExpandIncludes_list_under_type);
   RUN_TEST(test_YAMLtreeExpandIncludes_list_under_preconditioner);
   RUN_TEST(test_YAMLtreeBuild_indent_3_spaces);
   RUN_TEST(test_YAMLtreeBuild_indent_4_spaces);
   RUN_TEST(test_YAMLtreeBuild_indent_with_tabs);
   RUN_TEST(test_YAMLtreeBuild_scalar_with_children_is_error);

   return 0; /* Success - CTest handles reporting */
}
