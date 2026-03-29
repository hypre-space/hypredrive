#include <stdlib.h>
#include <string.h>

#include "internal/containers.h"
#include "internal/field.h"
#include "test_helpers.h"
#include "internal/yaml.h"

static YAMLnode *
make_node(const char *value)
{
   YAMLnode *node   = hypredrv_YAMLnodeCreate("dummy", "", 0);
   node->mapped_val = strdup(value);
   return node;
}

static void
test_FieldTypeIntSet(void)
{
   int       target = 0;
   YAMLnode *node   = make_node("42");
   hypredrv_FieldTypeIntSet(&target, node);
   ASSERT_EQ(target, 42);
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeDoubleSet(void)
{
   double    target = 0.0;
   YAMLnode *node   = make_node("3.1415");
   hypredrv_FieldTypeDoubleSet(&target, node);
   ASSERT_EQ_DOUBLE(target, 3.1415, 1e-12);
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeCharSet(void)
{
   char      target = 0;
   YAMLnode *node   = make_node("Z");
   hypredrv_FieldTypeCharSet(&target, node);
   ASSERT_EQ(target, 'Z');
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeStringSet(void)
{
   char      buffer[MAX_FILENAME_LENGTH];
   YAMLnode *node = make_node("output.txt");
   hypredrv_ErrorCodeResetAll();
   hypredrv_FieldTypeStringSet(buffer, node);
   ASSERT_STREQ(buffer, "output.txt");
    ASSERT_FALSE(hypredrv_ErrorCodeActive());
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeStringSet_overlong_rejected(void)
{
   char      buffer[MAX_FILENAME_LENGTH];
   char     *big  = (char *)malloc((size_t)MAX_FILENAME_LENGTH + 32);
   YAMLnode *node = NULL;
   ASSERT_NOT_NULL(big);
   memset(big, 'x', (size_t)MAX_FILENAME_LENGTH + 31);
   big[MAX_FILENAME_LENGTH + 31] = '\0';

   node = make_node(big);
   hypredrv_ErrorCodeResetAll();
   hypredrv_FieldTypeStringSet(buffer, node);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
   ASSERT_STREQ(buffer, "");

   hypredrv_YAMLnodeDestroy(node);
   free(big);
}

static void
test_FieldTypeIntArraySet(void)
{
   IntArray *array = NULL;
   YAMLnode *node  = make_node("1, 2, 3, 4");
   hypredrv_FieldTypeIntArraySet(&array, node);
   ASSERT_NOT_NULL(array);
   ASSERT_EQ(array->size, 4);
   ASSERT_EQ(array->data[0], 1);
   ASSERT_EQ(array->data[3], 4);
   hypredrv_IntArrayDestroy(&array);
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeStackIntArraySet(void)
{
   StackIntArray arr  = STACK_INTARRAY_CREATE();
   YAMLnode     *node = make_node("10, 20, 30");
   hypredrv_FieldTypeStackIntArraySet(&arr, node);
   ASSERT_EQ(arr.size, 3);
   ASSERT_EQ(arr.data[0], 10);
   ASSERT_EQ(arr.data[2], 30);
   hypredrv_YAMLnodeDestroy(node);
}

int
main(void)
{
   RUN_TEST(test_FieldTypeIntSet);
   RUN_TEST(test_FieldTypeDoubleSet);
   RUN_TEST(test_FieldTypeCharSet);
   RUN_TEST(test_FieldTypeStringSet);
   RUN_TEST(test_FieldTypeStringSet_overlong_rejected);
   RUN_TEST(test_FieldTypeIntArraySet);
   RUN_TEST(test_FieldTypeStackIntArraySet);
   return 0;
}
