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
test_FieldTypeDoubleSet_rejects_nonfinite_values(void)
{
   const char *values[] = {"1e309", "1e-400", "nan", "inf", "-inf"};

   for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
   {
      double    target = 17.0;
      YAMLnode *node   = make_node(values[i]);

      hypredrv_ErrorCodeResetAll();
      hypredrv_FieldTypeDoubleSet(&target, node);
      ASSERT_TRUE(hypredrv_ErrorCodeActive());
      ASSERT_TRUE((hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL) != 0);
      ASSERT_EQ_DOUBLE(target, 17.0, 0.0);
      hypredrv_YAMLnodeDestroy(node);
   }
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
test_FieldTypeStringSet_empty_when_unmapped(void)
{
   char      buffer[MAX_FILENAME_LENGTH];
   YAMLnode *node = hypredrv_YAMLnodeCreate("key", "ignored", 0);
   free(node->mapped_val);
   node->mapped_val = NULL;

   hypredrv_ErrorCodeResetAll();
   hypredrv_FieldTypeStringSet(buffer, node);
   ASSERT_STREQ(buffer, "");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeNoopSet(void)
{
   int       x    = 42;
   YAMLnode *node = make_node("x");
   hypredrv_FieldTypeNoopSet(&x, node);
   ASSERT_EQ(x, 42);
   hypredrv_YAMLnodeDestroy(node);
}

static void
test_FieldTypeDoubleArraySet(void)
{
   DoubleArray *arr = NULL;
   YAMLnode    *node = make_node("0.5, 1.0, 2.5");

   hypredrv_FieldTypeDoubleArraySet(&arr, node);
   ASSERT_NOT_NULL(arr);
   ASSERT_EQ(arr->size, 3);
   ASSERT_EQ_DOUBLE(arr->data[0], 0.5, 1e-12);
   ASSERT_EQ_DOUBLE(arr->data[2], 2.5, 1e-12);

   hypredrv_DoubleArrayDestroy(&arr);
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

static void
test_FieldTypeIntSet_rejects_invalid_values(void)
{
   const char *values[] = {"", "12x", "1.25", "2147483648", "-2147483649"};
   for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
   {
      int       value = 17;
      YAMLnode *node  = make_node(values[i]);
      hypredrv_ErrorStateReset();
      hypredrv_FieldTypeIntSet(&value, node);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      ASSERT_EQ(value, 17);
      hypredrv_YAMLnodeDestroy(node);
   }
   hypredrv_ErrorStateReset();
}

static void
test_FieldTypeArraySet_preserves_field_on_error(void)
{
   IntArray    *integers = hypredrv_IntArrayCreate(1);
   DoubleArray *doubles  = hypredrv_DoubleArrayCreate(1);
   ASSERT_NOT_NULL(integers);
   ASSERT_NOT_NULL(doubles);
   integers->data[0]         = 17;
   doubles->data[0]          = 17.0;
   IntArray    *old_integers = integers;
   DoubleArray *old_doubles  = doubles;
   YAMLnode    *bad          = make_node("1, invalid");
   hypredrv_ErrorStateReset();
   hypredrv_FieldTypeIntArraySet(&integers, bad);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(integers, old_integers);
   ASSERT_EQ_SIZE(integers->size, 1);
   ASSERT_EQ(integers->data[0], 17);
   hypredrv_ErrorStateReset();
   hypredrv_FieldTypeDoubleArraySet(&doubles, bad);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(doubles, old_doubles);
   ASSERT_EQ_SIZE(doubles->size, 1);
   ASSERT_EQ_DOUBLE(doubles->data[0], 17.0, 0.0);
   hypredrv_YAMLnodeDestroy(bad);

   YAMLnode *valid = make_node("2, 3");
   hypredrv_ErrorStateReset();
   hypredrv_FieldTypeIntArraySet(&integers, valid);
   hypredrv_FieldTypeDoubleArraySet(&doubles, valid);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_SIZE(integers->size, 2);
   ASSERT_EQ(integers->data[0], 2);
   ASSERT_EQ(integers->data[1], 3);
   ASSERT_EQ_SIZE(doubles->size, 2);
   ASSERT_EQ_DOUBLE(doubles->data[0], 2.0, 0.0);
   ASSERT_EQ_DOUBLE(doubles->data[1], 3.0, 0.0);
   hypredrv_IntArrayDestroy(&integers);
   hypredrv_DoubleArrayDestroy(&doubles);
   hypredrv_YAMLnodeDestroy(valid);
}

int
main(void)
{
   RUN_TEST(test_FieldTypeIntSet);
   RUN_TEST(test_FieldTypeIntSet_rejects_invalid_values);
   RUN_TEST(test_FieldTypeArraySet_preserves_field_on_error);
   RUN_TEST(test_FieldTypeDoubleSet);
   RUN_TEST(test_FieldTypeDoubleSet_rejects_nonfinite_values);
   RUN_TEST(test_FieldTypeCharSet);
   RUN_TEST(test_FieldTypeStringSet);
   RUN_TEST(test_FieldTypeStringSet_empty_when_unmapped);
   RUN_TEST(test_FieldTypeStringSet_overlong_rejected);
   RUN_TEST(test_FieldTypeIntArraySet);
   RUN_TEST(test_FieldTypeStackIntArraySet);
   RUN_TEST(test_FieldTypeDoubleArraySet);
   RUN_TEST(test_FieldTypeNoopSet);
   return 0;
}
