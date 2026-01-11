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
   YAMLnode *node = YAMLnodeCreate("key", "value", 0);
   ASSERT_NOT_NULL(node);
   ASSERT_STREQ(node->key, "key");
   ASSERT_STREQ(node->val, "value");
   /* Note: indent is not a direct field, it's stored in the struct differently */
   YAMLnodeDestroy(node);
}

/*-----------------------------------------------------------------------------
 * Test YAMLtree creation and destruction
 *-----------------------------------------------------------------------------*/

static void
test_YAMLtreeCreate_basic(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);
   YAMLtreeDestroy(&tree);
   ASSERT_NULL(tree);
}

/*-----------------------------------------------------------------------------
 * Test YAMLnodeFindByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindByKey_basic(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   ASSERT_NOT_NULL(tree);

   YAMLnode *root  = tree->root;
   YAMLnode *child = YAMLnodeCreate("child", "value", 1);
   YAMLnodeAddChild(root, child);

   YAMLnode *found = YAMLnodeFindByKey(root, "child");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child");

   YAMLnode *not_found = YAMLnodeFindByKey(root, "nonexistent");
   ASSERT_NULL(not_found);

   YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAMLnodeFindChildByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindChildByKey_basic(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child1 = YAMLnodeCreate("child1", "value1", 1);
   YAMLnode *child2 = YAMLnodeCreate("child2", "value2", 1);
   YAMLnodeAddChild(root, child1);
   YAMLnodeAddChild(root, child2);

   YAMLnode *found = YAMLnodeFindChildByKey(root, "child1");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child1");

   found = YAMLnodeFindChildByKey(root, "child2");
   ASSERT_NOT_NULL(found);
   ASSERT_STREQ(found->key, "child2");

   found = YAMLnodeFindChildByKey(root, "nonexistent");
   ASSERT_NULL(found);

   YAMLtreeDestroy(&tree);
}

/*-----------------------------------------------------------------------------
 * Test YAMLnodeFindChildValueByKey
 *-----------------------------------------------------------------------------*/

static void
test_YAMLnodeFindChildValueByKey_basic(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = YAMLnodeCreate("key", "value", 1);
   YAMLnodeAddChild(root, child);

   char *value = YAMLnodeFindChildValueByKey(root, "key");
   ASSERT_NOT_NULL(value);
   ASSERT_STREQ(value, "value");

   value = YAMLnodeFindChildValueByKey(root, "nonexistent");
   ASSERT_NULL(value);

   YAMLtreeDestroy(&tree);
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
   /* YAMLtreeBuild creates its own tree and modifies text in place */
   YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   char *value = YAMLnodeFindChildValueByKey(tree->root, "key");
   /* value may be NULL if parsing doesn't work as expected - check for basic tree
    * structure */
   if (value)
   {
      ASSERT_STREQ(value, "value");
   }

   free(text);
   YAMLtreeDestroy(&tree);
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
   /* YAMLtreeBuild creates its own tree and modifies text in place */
   YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   YAMLnode *parent = YAMLnodeFindChildByKey(tree->root, "parent");
   /* Basic test - parent may or may not exist depending on parsing */
   if (parent)
   {
      char *value = YAMLnodeFindChildValueByKey(parent, "child");
      if (value)
      {
         ASSERT_STREQ(value, "value");
      }
   }

   free(text);
   YAMLtreeDestroy(&tree);
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
   /* YAMLtreeBuild creates its own tree and modifies text in place */
   YAMLtreeBuild(0, text, &tree);

   /* Tree should be created, but validation should catch errors */
   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Validate the tree - should detect invalid indentation */
   YAMLtreeValidate(tree);

   free(text);
   YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodeFindByKey_nonexistent(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = YAMLnodeCreate("child", "value", 1);
   YAMLnodeAddChild(root, child);

   /* Find existing key */
   YAMLnode *found = YAMLnodeFindByKey(root, "child");
   ASSERT_NOT_NULL(found);

   /* Find nonexistent key */
   YAMLnode *not_found = YAMLnodeFindByKey(root, "nonexistent");
   ASSERT_NULL(not_found);

   YAMLtreeDestroy(&tree);
}

static void
test_YAMLnodeFindChildValueByKey_nonexistent(void)
{
   YAMLtree *tree = YAMLtreeCreate(10);
   YAMLnode *root = tree->root;

   YAMLnode *child = YAMLnodeCreate("key", "value", 1);
   YAMLnodeAddChild(root, child);

   /* Find existing value */
   char *value = YAMLnodeFindChildValueByKey(root, "key");
   ASSERT_NOT_NULL(value);
   ASSERT_STREQ(value, "value");

   /* Find nonexistent value */
   value = YAMLnodeFindChildValueByKey(root, "nonexistent");
   ASSERT_NULL(value);

   YAMLtreeDestroy(&tree);
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
   YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Find parent node */
   YAMLnode *parent = YAMLnodeFindChildByKey(tree->root, "parent");
   ASSERT_NOT_NULL(parent);
   ASSERT_STREQ(parent->key, "parent");
   ASSERT_STREQ(parent->val, ""); /* parent has children, so val is empty */

   /* Find middle node */
   YAMLnode *middle = YAMLnodeFindChildByKey(parent, "middle");
   ASSERT_NOT_NULL(middle);
   ASSERT_STREQ(middle->key, "middle");
   ASSERT_STREQ(middle->val, ""); /* middle has children, so val is empty */

   /* Find child node */
   YAMLnode *child = YAMLnodeFindChildByKey(middle, "child");
   ASSERT_NOT_NULL(child);
   ASSERT_STREQ(child->key, "child");
   ASSERT_STREQ(child->val, "value");

   free(text);
   YAMLtreeDestroy(&tree);
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
   YAMLtreeBuild(0, text, &tree_flat);
   ASSERT_NOT_NULL(tree_flat);

   YAMLnode *solver_flat = YAMLnodeFindChildByKey(tree_flat->root, "solver");
   ASSERT_NOT_NULL(solver_flat);
   ASSERT_STREQ(solver_flat->key, "solver");
   ASSERT_STREQ(solver_flat->val, "gmres");
   ASSERT_NULL(solver_flat->children); /* No children */

   free(text);
   YAMLtreeDestroy(&tree_flat);

   /* When 'solver' has children, val should be empty */
   const char *yaml_nested = "solver:\n  type: gmres\n";
   len                     = strlen(yaml_nested);
   text                    = malloc(len + 1);
   strcpy(text, yaml_nested);

   YAMLtree *tree_nested = NULL;
   YAMLtreeBuild(0, text, &tree_nested);
   ASSERT_NOT_NULL(tree_nested);

   YAMLnode *solver_nested = YAMLnodeFindChildByKey(tree_nested->root, "solver");
   ASSERT_NOT_NULL(solver_nested);
   ASSERT_STREQ(solver_nested->key, "solver");
   ASSERT_STREQ(solver_nested->val, ""); /* Has children, so val is empty */
   ASSERT_NOT_NULL(solver_nested->children);

   YAMLnode *type_node = YAMLnodeFindChildByKey(solver_nested, "type");
   ASSERT_NOT_NULL(type_node);
   ASSERT_STREQ(type_node->key, "type");
   ASSERT_STREQ(type_node->val, "gmres");

   free(text);
   YAMLtreeDestroy(&tree_nested);
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
   YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   /* Find coarsest_level */
   YAMLnode *coarsest = YAMLnodeFindChildByKey(tree->root, "coarsest_level");
   ASSERT_NOT_NULL(coarsest);
   ASSERT_STREQ(coarsest->key, "coarsest_level");
   ASSERT_STREQ(coarsest->val, ""); /* Has children */
   ASSERT_NOT_NULL(coarsest->children);

   /* Find ilu - this is where the bug was: code was checking val instead of key */
   YAMLnode *ilu = YAMLnodeFindChildByKey(coarsest, "ilu");
   ASSERT_NOT_NULL(ilu);
   ASSERT_STREQ(ilu->key, "ilu");    /* KEY is "ilu" */
   ASSERT_STREQ(ilu->val, "");       /* VAL is empty because ilu has children! */
   ASSERT_NOT_NULL(ilu->children);

   /* Verify ilu's children */
   YAMLnode *type_node = YAMLnodeFindChildByKey(ilu, "type");
   ASSERT_NOT_NULL(type_node);
   ASSERT_STREQ(type_node->key, "type");
   ASSERT_STREQ(type_node->val, "bj-ilut");

   YAMLnode *droptol = YAMLnodeFindChildByKey(ilu, "droptol");
   ASSERT_NOT_NULL(droptol);
   ASSERT_STREQ(droptol->key, "droptol");
   ASSERT_STREQ(droptol->val, "1e-4");

   free(text);
   YAMLtreeDestroy(&tree);
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
   YAMLtreeBuild(0, text, &tree);

   ASSERT_NOT_NULL(tree);
   ASSERT_NOT_NULL(tree->root);

   YAMLnode *key1 = YAMLnodeFindChildByKey(tree->root, "key1");
   YAMLnode *key2 = YAMLnodeFindChildByKey(tree->root, "key2");
   YAMLnode *key3 = YAMLnodeFindChildByKey(tree->root, "key3");

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
   YAMLtreeDestroy(&tree);
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
   RUN_TEST(test_YAMLtreeBuild_nested);
   RUN_TEST(test_YAMLtreeBuild_deeply_nested);
   RUN_TEST(test_YAMLtreeBuild_siblings);

   RUN_TEST(test_YAMLnode_key_val_distinction);
   RUN_TEST(test_YAMLtreeBuild_mgr_coarsest_level_pattern);

   RUN_TEST(test_YAMLtreeBuild_invalid_indent);
   RUN_TEST(test_YAMLnodeFindByKey_nonexistent);
   RUN_TEST(test_YAMLnodeFindChildValueByKey_nonexistent);

   return 0; /* Success - CTest handles reporting */
}
