/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "test_helpers.h"
#include "containers.h"

/*-----------------------------------------------------------------------------
 * Test StackIntArray
 *-----------------------------------------------------------------------------*/

static void
test_StackIntArray_create(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   ASSERT_EQ(arr.size, 0);
}

static void
test_StackIntArray_basic_ops(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   
   /* Test initial state */
   ASSERT_EQ(arr.size, 0);

   /* Manually add some data for testing */
   arr.data[0] = 42;
   arr.data[1] = 100;
   arr.size = 2;

   ASSERT_EQ(arr.data[0], 42);
   ASSERT_EQ(arr.data[1], 100);
   ASSERT_EQ(arr.size, 2);
}

/*-----------------------------------------------------------------------------
 * Test StrArray
 *-----------------------------------------------------------------------------*/

static void
test_StrArray_basic(void)
{
   const char *strs[] = {"one", "two", "three"};
   StrArray arr = STR_ARRAY_CREATE(strs);
   ASSERT_EQ(arr.size, 3);
   ASSERT_STREQ(arr.data[0], "one");
   ASSERT_STREQ(arr.data[1], "two");
   ASSERT_STREQ(arr.data[2], "three");
}

static void
test_StrArrayEntryExists(void)
{
   const char *strs[] = {"one", "two", "three"};
   StrArray arr = STR_ARRAY_CREATE(strs);
   
   ASSERT_TRUE(StrArrayEntryExists(arr, "one"));
   ASSERT_TRUE(StrArrayEntryExists(arr, "two"));
   ASSERT_TRUE(StrArrayEntryExists(arr, "three"));
   ASSERT_FALSE(StrArrayEntryExists(arr, "four"));
   ASSERT_FALSE(StrArrayEntryExists(arr, ""));
}

/*-----------------------------------------------------------------------------
 * Test StrIntMap
 *-----------------------------------------------------------------------------*/

static void
test_StrIntMapArray_basic(void)
{
   /* Test OnOffMapArray */
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "on"));
   ASSERT_TRUE(StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "off"));
   ASSERT_FALSE(StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "invalid"));
}

static void
test_StrIntMapArrayGetImage(void)
{
   int img;
   
   img = StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "on");
   ASSERT_TRUE(img >= 0);
   
   img = StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "off");
   ASSERT_TRUE(img >= 0);
   
   img = StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "invalid");
   ASSERT_EQ(img, INT_MIN);  /* Implementation returns INT_MIN for invalid keys */
}

/*-----------------------------------------------------------------------------
 * Test StrToIntArray and StrToStackIntArray
 *-----------------------------------------------------------------------------*/

static void
test_StrToStackIntArray_basic(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   StrToStackIntArray("1,2,3", &arr);
   ASSERT_EQ(arr.size, 3);
   ASSERT_EQ(arr.data[0], 1);
   ASSERT_EQ(arr.data[1], 2);
   ASSERT_EQ(arr.data[2], 3);
}

static void
test_StrToStackIntArray_single(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   StrToStackIntArray("42", &arr);
   ASSERT_EQ(arr.size, 1);
   ASSERT_EQ(arr.data[0], 42);
}

static void
test_StrToStackIntArray_empty(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   StrToStackIntArray("", &arr);
   ASSERT_EQ(arr.size, 0);
}

/*-----------------------------------------------------------------------------
 * Test IntArray (basic - without MPI)
 *-----------------------------------------------------------------------------*/

static void
test_IntArray_create_destroy(void)
{
   IntArray *arr = IntArrayCreate(10);
   ASSERT_NOT_NULL(arr);
   ASSERT_EQ(arr->size, 10);
   
   IntArrayDestroy(&arr);
   ASSERT_NULL(arr);
}

static void
test_IntArray_zero_size(void)
{
   IntArray *arr = IntArrayCreate(0);
   ASSERT_NOT_NULL(arr);
   ASSERT_EQ(arr->size, 0);
   IntArrayDestroy(&arr);
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int main(void)
{
   RUN_TEST(test_StackIntArray_create);
   RUN_TEST(test_StackIntArray_basic_ops);

   RUN_TEST(test_StrArray_basic);
   RUN_TEST(test_StrArrayEntryExists);

   RUN_TEST(test_StrIntMapArray_basic);
   RUN_TEST(test_StrIntMapArrayGetImage);

   RUN_TEST(test_StrToStackIntArray_basic);
   RUN_TEST(test_StrToStackIntArray_single);
   RUN_TEST(test_StrToStackIntArray_empty);

   RUN_TEST(test_IntArray_create_destroy);
   RUN_TEST(test_IntArray_zero_size);

   return 0;  /* Success - CTest handles reporting */
}

