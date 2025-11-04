#include <stdlib.h>
#include <string.h>

#include "field.h"
#include "yaml.h"
#include "containers.h"
#include "test_helpers.h"

static YAMLnode *
make_node(const char *value)
{
   YAMLnode *node = YAMLnodeCreate("dummy", "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static void
test_FieldTypeIntSet(void)
{
   int       target = 0;
   YAMLnode *node   = make_node("42");
   FieldTypeIntSet(&target, node);
   ASSERT_EQ(target, 42);
   YAMLnodeDestroy(node);
}

static void
test_FieldTypeDoubleSet(void)
{
   double    target = 0.0;
   YAMLnode *node   = make_node("3.1415");
   FieldTypeDoubleSet(&target, node);
   ASSERT_EQ_DOUBLE(target, 3.1415, 1e-12);
   YAMLnodeDestroy(node);
}

static void
test_FieldTypeCharSet(void)
{
   char      target = 0;
   YAMLnode *node   = make_node("Z");
   FieldTypeCharSet(&target, node);
   ASSERT_EQ(target, 'Z');
   YAMLnodeDestroy(node);
}

static void
test_FieldTypeStringSet(void)
{
   char      buffer[MAX_FILENAME_LENGTH];
   YAMLnode *node = make_node("output.txt");
   FieldTypeStringSet(buffer, node);
   ASSERT_STREQ(buffer, "output.txt");
   YAMLnodeDestroy(node);
}

static void
test_FieldTypeIntArraySet(void)
{
   IntArray *array = NULL;
   YAMLnode *node  = make_node("1, 2, 3, 4");
   FieldTypeIntArraySet(&array, node);
   ASSERT_NOT_NULL(array);
   ASSERT_EQ(array->size, 4);
   ASSERT_EQ(array->data[0], 1);
   ASSERT_EQ(array->data[3], 4);
   IntArrayDestroy(&array);
   YAMLnodeDestroy(node);
}

static void
test_FieldTypeStackIntArraySet(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   YAMLnode     *node = make_node("10, 20, 30");
   FieldTypeStackIntArraySet(&arr, node);
   ASSERT_EQ(arr.size, 3);
   ASSERT_EQ(arr.data[0], 10);
   ASSERT_EQ(arr.data[2], 30);
   YAMLnodeDestroy(node);
}

int
main(void)
{
   RUN_TEST(test_FieldTypeIntSet);
   RUN_TEST(test_FieldTypeDoubleSet);
   RUN_TEST(test_FieldTypeCharSet);
   RUN_TEST(test_FieldTypeStringSet);
   RUN_TEST(test_FieldTypeIntArraySet);
   RUN_TEST(test_FieldTypeStackIntArraySet);
   return 0;
}

