/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "internal/containers.h"
#include "internal/error.h"
#include "test_helpers.h"

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
   arr.size    = 2;

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
   StrArray    arr    = STR_ARRAY_CREATE(strs);
   ASSERT_EQ(arr.size, 3);
   ASSERT_STREQ(arr.data[0], "one");
   ASSERT_STREQ(arr.data[1], "two");
   ASSERT_STREQ(arr.data[2], "three");
}

static void
test_StrArrayEntryExists(void)
{
   const char *strs[] = {"one", "two", "three"};
   StrArray    arr    = STR_ARRAY_CREATE(strs);

   ASSERT_TRUE(hypredrv_StrArrayEntryExists(arr, "one"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(arr, "two"));
   ASSERT_TRUE(hypredrv_StrArrayEntryExists(arr, "three"));
   ASSERT_FALSE(hypredrv_StrArrayEntryExists(arr, "four"));
   ASSERT_FALSE(hypredrv_StrArrayEntryExists(arr, ""));
}

/*-----------------------------------------------------------------------------
 * Test StrIntMap
 *-----------------------------------------------------------------------------*/

static void
test_StrIntMapArray_basic(void)
{
   /* Test OnOffMapArray */
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "on"));
   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "off"));
   ASSERT_FALSE(
      hypredrv_StrIntMapArrayDomainEntryExists(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "invalid"));
}

static void
test_StrIntMapArrayGetImage(void)
{
   int img;

   img = hypredrv_StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "on");
   ASSERT_TRUE(img >= 0);

   img = hypredrv_StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "off");
   ASSERT_TRUE(img >= 0);

   img = hypredrv_StrIntMapArrayGetImage(STR_INT_MAP_ARRAY_CREATE_ON_OFF(), "invalid");
   ASSERT_EQ(img, INT_MIN); /* Implementation returns INT_MIN for invalid keys */
}

static void
test_StrIntMapArray_empty_string_key(void)
{
   static const StrIntMap map[] = {{"", -1}, {"none", -1}, {"jacobi", 7}};
   StrIntMapArray         arr   = STR_INT_MAP_ARRAY_CREATE(map);

   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(arr, ""));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(arr, ""), -1);
}

/*-----------------------------------------------------------------------------
 * Test hypredrv_StrToIntArray and hypredrv_StrToStackIntArray
 *-----------------------------------------------------------------------------*/

static void
test_StrToStackIntArray_basic(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   hypredrv_StrToStackIntArray("1,2,3", &arr);
   ASSERT_EQ(arr.size, 3);
   ASSERT_EQ(arr.data[0], 1);
   ASSERT_EQ(arr.data[1], 2);
   ASSERT_EQ(arr.data[2], 3);
}

static void
test_StrToStackIntArray_single(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   hypredrv_StrToStackIntArray("42", &arr);
   ASSERT_EQ(arr.size, 1);
   ASSERT_EQ(arr.data[0], 42);
}

static void
test_StrToStackIntArray_empty(void)
{
   StackIntArray arr = STACK_INTARRAY_CREATE();
   hypredrv_StrToStackIntArray("", &arr);
   ASSERT_EQ(arr.size, 0);
}

/*-----------------------------------------------------------------------------
 * Test IntArray (basic - without MPI)
 *-----------------------------------------------------------------------------*/

static void
test_IntArray_create_destroy(void)
{
   IntArray *arr = hypredrv_IntArrayCreate(10);
   ASSERT_NOT_NULL(arr);
   ASSERT_EQ(arr->size, 10);

   hypredrv_IntArrayDestroy(&arr);
   ASSERT_NULL(arr);
}

static void
test_IntArray_zero_size(void)
{
   IntArray *arr = hypredrv_IntArrayCreate(0);
   ASSERT_NOT_NULL(arr);
   ASSERT_EQ(arr->size, 0);
   hypredrv_IntArrayDestroy(&arr);
}

static void
test_IntArray_WriteAsciiByRank_fopen_failure(void)
{
   IntArray *arr = hypredrv_IntArrayCreate(1);
   ASSERT_NOT_NULL(arr);
   arr->data[0] = 42;

   hypredrv_ErrorCodeResetAll();
   hypredrv_IntArrayWriteAsciiByRank(MPI_COMM_SELF, arr,
                                     "/nonexistent_dir_hypredrive_zzzz/prefix");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_FILE_NOT_FOUND, 0);

   hypredrv_IntArrayDestroy(&arr);
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_StackIntArray_create);
   RUN_TEST(test_StackIntArray_basic_ops);

   RUN_TEST(test_StrArray_basic);
   RUN_TEST(test_StrArrayEntryExists);

   RUN_TEST(test_StrIntMapArray_basic);
   RUN_TEST(test_StrIntMapArrayGetImage);
   RUN_TEST(test_StrIntMapArray_empty_string_key);

   RUN_TEST(test_StrToStackIntArray_basic);
   RUN_TEST(test_StrToStackIntArray_single);
   RUN_TEST(test_StrToStackIntArray_empty);

   RUN_TEST(test_IntArray_create_destroy);
   RUN_TEST(test_IntArray_zero_size);
   RUN_TEST(test_IntArray_WriteAsciiByRank_fopen_failure);

   MPI_Finalize();
   return 0; /* Success - CTest handles reporting */
}
