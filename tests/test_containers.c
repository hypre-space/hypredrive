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

/*-----------------------------------------------------------------------------
 * Test StrArray
 *-----------------------------------------------------------------------------*/

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
   StrIntMapArray map = STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "on"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "off"), 0);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "1"), 1);
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(map, "invalid"), INT_MIN);
}

static void
test_StrIntMapArray_empty_string_key(void)
{
   static const StrIntMap map[] = {{"", -1}, {"none", -1}, {"jacobi", 7}};
   StrIntMapArray         arr   = STR_INT_MAP_ARRAY_CREATE(map);

   ASSERT_TRUE(hypredrv_StrIntMapArrayDomainEntryExists(arr, ""));
   ASSERT_EQ(hypredrv_StrIntMapArrayGetImage(arr, ""), -1);
}

static void
test_StrIntMapArrayDomainToString_groups_aliases(void)
{
   static const StrIntMap map[] = {{"", -1}, {"none", -1}, {"single", 7}, {"jacobi", 7}};
   char *str = hypredrv_StrIntMapArrayDomainToString(STR_INT_MAP_ARRAY_CREATE(map));

   ASSERT_NOT_NULL(str);
   ASSERT_STREQ(str, "none (-1), single/jacobi (7)");
   free(str);
}

static void
test_StrIntMapArrayDomainToString_no_printable_entries(void)
{
   static const StrIntMap map[] = {{"", -1}};

   ASSERT_NULL(hypredrv_StrIntMapArrayDomainToString(STR_INT_MAP_ARRAY_CREATE(map)));
   ASSERT_NULL(hypredrv_StrIntMapArrayDomainToString(STR_INT_MAP_ARRAY_VOID()));
}

static void
test_StrArrayToString_basic(void)
{
   const char *strs[] = {"one", "two", "three"};
   char       *str    = hypredrv_StrArrayToString(STR_ARRAY_CREATE(strs));

   ASSERT_NOT_NULL(str);
   ASSERT_STREQ(str, "one, two, three");
   free(str);
}

static void
test_StrArrayToString_empty(void)
{
   ASSERT_NULL(hypredrv_StrArrayToString(STR_ARRAY_VOID()));
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

static void
test_numeric_arrays(void)
{
   IntArray    *integers = NULL;
   DoubleArray *doubles  = NULL;
   char         bounds[128];
   snprintf(bounds, sizeof(bounds), "[%d, 0, %d]", INT_MIN, INT_MAX);
   hypredrv_ErrorStateReset();
   hypredrv_StrToIntArray(bounds, &integers);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(integers);
   ASSERT_EQ_SIZE(integers->size, 3);
   ASSERT_EQ(integers->data[0], INT_MIN);
   ASSERT_EQ(integers->data[1], 0);
   ASSERT_EQ(integers->data[2], INT_MAX);
   hypredrv_IntArrayDestroy(&integers);

   hypredrv_StrToDoubleArray("[1.25,\t-2e3,\n0]", &doubles);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(doubles);
   ASSERT_EQ_SIZE(doubles->size, 3);
   ASSERT_EQ_DOUBLE(doubles->data[0], 1.25, 0.0);
   ASSERT_EQ_DOUBLE(doubles->data[1], -2000.0, 0.0);
   ASSERT_EQ_DOUBLE(doubles->data[2], 0.0, 0.0);
   hypredrv_DoubleArrayDestroy(&doubles);

   hypredrv_StrToIntArray("[]", &integers);
   hypredrv_StrToDoubleArray("", &doubles);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(integers);
   ASSERT_NOT_NULL(doubles);
   ASSERT_EQ_SIZE(integers->size, 0);
   ASSERT_EQ_SIZE(doubles->size, 0);
   hypredrv_IntArrayDestroy(&integers);
   hypredrv_DoubleArrayDestroy(&doubles);
}

static void
test_numeric_arrays_reject_invalid_entries(void)
{
   const char *bad_ints[]    = {"1, nope",     "2x",
                                "1.5",         "2147483648",
                                "-2147483649", "99999999999999999999999999999"};
   const char *bad_doubles[] = {"1, nope", "2x", "nan", "inf", "-inf", "1e309", "1e-400"};
   for (size_t i = 0; i < sizeof(bad_ints) / sizeof(bad_ints[0]); i++)
   {
      IntArray     *array = NULL;
      StackIntArray stack = {.data = {42}, .size = 1};
      hypredrv_ErrorStateReset();
      hypredrv_StrToIntArray(bad_ints[i], &array);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      ASSERT_NULL(array);
      hypredrv_ErrorStateReset();
      hypredrv_StrToStackIntArray(bad_ints[i], &stack);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      ASSERT_EQ_SIZE(stack.size, 1);
      ASSERT_EQ(stack.data[0], 42);
   }
   for (size_t i = 0; i < sizeof(bad_doubles) / sizeof(bad_doubles[0]); i++)
   {
      DoubleArray *array = NULL;
      hypredrv_ErrorStateReset();
      hypredrv_StrToDoubleArray(bad_doubles[i], &array);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
      ASSERT_NULL(array);
   }
   hypredrv_ErrorStateReset();
}

static void
test_StackIntArray_capacity(void)
{
   char text[2 * MAX_STACK_ARRAY_LENGTH + 3];
   for (size_t i = 0; i < MAX_STACK_ARRAY_LENGTH + 1; i++)
   {
      text[2 * i]     = '1';
      text[2 * i + 1] = ',';
   }
   text[2 * MAX_STACK_ARRAY_LENGTH] = '\0';
   StackIntArray array              = STACK_INTARRAY_CREATE();
   hypredrv_ErrorStateReset();
   hypredrv_StrToStackIntArray(text, &array);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_EQ_SIZE(array.size, MAX_STACK_ARRAY_LENGTH);
   ASSERT_EQ(array.data[MAX_STACK_ARRAY_LENGTH - 1], 1);

   text[2 * MAX_STACK_ARRAY_LENGTH]     = '2';
   text[2 * MAX_STACK_ARRAY_LENGTH + 1] = '\0';
   hypredrv_StrToStackIntArray(text, &array);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   ASSERT_EQ_SIZE(array.size, MAX_STACK_ARRAY_LENGTH);
   ASSERT_EQ(array.data[MAX_STACK_ARRAY_LENGTH - 1], 1);
   hypredrv_ErrorStateReset();
}

static void
test_IntArray_distributed_labels(void)
{
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   const int  root_data[]  = {INT_MAX, 7, INT_MIN, 7};
   const int  other_data[] = {9, 7, INT_MAX};
   const int  expected[]   = {INT_MIN, 7, 9, INT_MAX};
   const int *data         = rank == 0 ? root_data : other_data;
   int        size         = rank == 0 ? 4 : 3;
   IntArray  *array        = NULL;
   hypredrv_ErrorStateReset();
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, size, data, &array);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(array);
   ASSERT_EQ_SIZE(array->size, size);
   ASSERT_EQ_SIZE(array->unique_size, 3);
   for (int i = 0; i < size; i++) ASSERT_EQ(array->data[i], data[i]);
   ASSERT_EQ_SIZE(array->g_unique_size, nprocs == 1 ? 3 : 4);
   ASSERT_EQ(array->g_unique_data[0], INT_MIN);
   ASSERT_EQ(array->g_unique_data[1], 7);
   ASSERT_EQ(array->g_unique_data[array->g_unique_size - 1], INT_MAX);
   if (nprocs > 1)
   {
      for (size_t i = 0; i < 4; i++) ASSERT_EQ(array->g_unique_data[i], expected[i]);
   }
   hypredrv_IntArrayDestroy(&array);

   /* Empty ranks must participate, including when every rank is empty. */
   for (int all_empty = 0; all_empty <= 1; all_empty++)
   {
      size = rank == 0 || all_empty ? 0 : 3;
      hypredrv_IntArrayBuild(MPI_COMM_WORLD, size, size ? other_data : NULL, &array);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      ASSERT_NOT_NULL(array);
      ASSERT_EQ_SIZE(array->unique_size, size);
      ASSERT_EQ_SIZE(array->g_unique_size, nprocs == 1 || all_empty ? 0 : 3);
      if (array->g_unique_size)
      {
         ASSERT_EQ(array->g_unique_data[0], 7);
         ASSERT_EQ(array->g_unique_data[1], 9);
         ASSERT_EQ(array->g_unique_data[2], INT_MAX);
      }
      hypredrv_IntArrayDestroy(&array);
   }
}

static void
test_IntArray_patterns_and_invalid_sizes(void)
{
   IntArray *array         = NULL;
   const int interleaved[] = {0, 1, 2, 0, 1, 2};
   const int contiguous[]  = {0, 0, 1, 1, 2, 2};
   hypredrv_ErrorStateReset();
   hypredrv_IntArrayBuildInterleaved(MPI_COMM_WORLD, 2, 3, &array);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(array);
   ASSERT_EQ_SIZE(array->size, 6);
   for (int i = 0; i < 6; i++) ASSERT_EQ(array->data[i], interleaved[i]);
   hypredrv_IntArrayDestroy(&array);
   hypredrv_IntArrayBuildContiguous(MPI_COMM_WORLD, 2, 3, &array);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(array);
   ASSERT_EQ_SIZE(array->size, 6);
   for (int i = 0; i < 6; i++) ASSERT_EQ(array->data[i], contiguous[i]);
   hypredrv_IntArrayDestroy(&array);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   int bad_rank = rank == nprocs - 1;
   hypredrv_IntArrayBuild(MPI_COMM_WORLD, bad_rank ? 1 : 0, NULL, &array);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   ASSERT_NULL(array);
   hypredrv_ErrorStateReset();
   hypredrv_IntArrayBuildContiguous(MPI_COMM_WORLD, bad_rank ? -1 : 0, 2, &array);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_VAL);
   ASSERT_NULL(array);
   hypredrv_ErrorStateReset();
   hypredrv_IntArrayBuildInterleaved(MPI_COMM_WORLD, bad_rank ? INT_MAX : 0, 2, &array);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_OUT_OF_BOUNDS);
   ASSERT_NULL(array);
   hypredrv_ErrorStateReset();
   ASSERT_NULL(hypredrv_IntArrayCreate(SIZE_MAX));
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_ALLOCATION);
   hypredrv_ErrorStateReset();
   ASSERT_NULL(hypredrv_DoubleArrayCreate(SIZE_MAX));
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_ALLOCATION);
   hypredrv_ErrorStateReset();
}

static void
test_IntArray_partition_reads(void)
{
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   char directory[128] = "/tmp/hypredrv-containers-XXXXXX";
   if (rank == 0) ASSERT_NOT_NULL(hypredrv_Mkdtemp(directory));
   MPI_Bcast(directory, sizeof(directory), MPI_CHAR, 0, MPI_COMM_WORLD);
   char prefix[160], filename[192];
   snprintf(prefix, sizeof(prefix), "%s/dofmap", directory);
   for (int binary = 0; binary <= 1; binary++)
   {
      snprintf(filename, sizeof(filename), "%s.%05d%s", prefix, rank,
               binary ? ".bin" : "");
      for (int invalid = 0; invalid <= 2; invalid++)
      {
         FILE *fp = fopen(filename, binary ? "wb" : "w");
         ASSERT_NOT_NULL(fp);
         size_t count   = invalid == 2 && rank == nprocs - 1 ? SIZE_MAX : 3;
         size_t written = invalid && rank == nprocs - 1 ? 2 : 3;
         int    data[]  = {rank, rank, rank};
         if (binary)
         {
            ASSERT_EQ_SIZE(fwrite(&count, sizeof(count), 1, fp), 1);
            ASSERT_EQ_SIZE(fwrite(data, sizeof(int), written, fp), written);
         }
         else
         {
            fprintf(fp, "%zu\n", count);
            for (size_t i = 0; i < written; i++) fprintf(fp, "%d\n", data[i]);
         }
         ASSERT_EQ(fclose(fp), 0);
         MPI_Barrier(MPI_COMM_WORLD);
         IntArray *array = NULL;
         hypredrv_ErrorStateReset();
         hypredrv_IntArrayParRead(MPI_COMM_WORLD, prefix, &array);
         if (invalid)
         {
            ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_FILE_UNEXPECTED_ENTRY);
            ASSERT_NULL(array);
         }
         else
         {
            ASSERT_FALSE(hypredrv_ErrorCodeActive());
            ASSERT_NOT_NULL(array);
            ASSERT_EQ_SIZE(array->size, 3);
            for (size_t i = 0; i < 3; i++) ASSERT_EQ(array->data[i], rank);
            ASSERT_EQ_SIZE(array->unique_size, 1);
            ASSERT_EQ(array->unique_data[0], rank);
            ASSERT_EQ_SIZE(array->g_unique_size, nprocs);
            for (int i = 0; i < nprocs; i++) ASSERT_EQ(array->g_unique_data[i], i);
            hypredrv_IntArrayDestroy(&array);
         }
         MPI_Barrier(MPI_COMM_WORLD);
      }
      ASSERT_EQ(unlink(filename), 0);
      MPI_Barrier(MPI_COMM_WORLD);
   }
   if (rank == 0) ASSERT_EQ(rmdir(directory), 0);
   hypredrv_ErrorStateReset();
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_StackIntArray_create);

   RUN_TEST(test_StrArrayEntryExists);

   RUN_TEST(test_StrIntMapArray_basic);
   RUN_TEST(test_StrIntMapArrayGetImage);
   RUN_TEST(test_StrIntMapArray_empty_string_key);
   RUN_TEST(test_StrIntMapArrayDomainToString_groups_aliases);
   RUN_TEST(test_StrIntMapArrayDomainToString_no_printable_entries);
   RUN_TEST(test_StrArrayToString_basic);
   RUN_TEST(test_StrArrayToString_empty);

   RUN_TEST(test_StrToStackIntArray_basic);
   RUN_TEST(test_StrToStackIntArray_single);
   RUN_TEST(test_StrToStackIntArray_empty);

   RUN_TEST(test_numeric_arrays);
   RUN_TEST(test_numeric_arrays_reject_invalid_entries);
   RUN_TEST(test_StackIntArray_capacity);
   RUN_TEST(test_IntArray_distributed_labels);
   RUN_TEST(test_IntArray_partition_reads);
   RUN_TEST(test_IntArray_patterns_and_invalid_sizes);
   RUN_TEST(test_IntArray_create_destroy);
   RUN_TEST(test_IntArray_zero_size);
   RUN_TEST(test_IntArray_WriteAsciiByRank_fopen_failure);

   MPI_Finalize();
   return 0; /* Success - CTest handles reporting */
}
