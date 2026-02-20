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
#include "utils.h"

/*-----------------------------------------------------------------------------
 * Test StrToLowerCase
 *-----------------------------------------------------------------------------*/

static void
test_StrToLowerCase_basic(void)
{
   char *str1 = strdup("HELLO WORLD");
   ASSERT_STREQ(StrToLowerCase(str1), "hello world");
   free(str1);

   char *str2 = strdup("MiXeD CaSe");
   ASSERT_STREQ(StrToLowerCase(str2), "mixed case");
   free(str2);

   char *str3 = strdup("lowercase");
   ASSERT_STREQ(StrToLowerCase(str3), "lowercase");
   free(str3);
}

static void
test_StrToLowerCase_empty(void)
{
   char *str = strdup("");
   ASSERT_STREQ(StrToLowerCase(str), "");
   free(str);
}

static void
test_StrToLowerCase_special_chars(void)
{
   char *str = strdup("HELLO123_WORLD!@#");
   ASSERT_STREQ(StrToLowerCase(str), "hello123_world!@#");
   free(str);
}

/*-----------------------------------------------------------------------------
 * Test StrTrim
 *-----------------------------------------------------------------------------*/

static void
test_StrTrim_trailing(void)
{
   char *str = strdup("  test  ");
   StrTrim(str);
   ASSERT_STREQ(str, "  test");
   free(str);
}

static void
test_StrTrim_leading(void)
{
   char *str = strdup("  test");
   StrTrim(str);
   ASSERT_STREQ(str, "  test"); /* Only trims trailing */
   free(str);
}

static void
test_StrTrim_no_spaces(void)
{
   char *str = strdup("no spaces");
   StrTrim(str);
   ASSERT_STREQ(str, "no spaces");
   free(str);
}

static void
test_StrTrim_empty(void)
{
   char *str = strdup("");
   StrTrim(str);
   ASSERT_STREQ(str, "");
   free(str);
}

static void
test_StrTrim_only_spaces(void)
{
   char *str = strdup("     ");
   StrTrim(str);
   ASSERT_STREQ(str, "");
   free(str);
}

static void
test_StrTrim_null(void)
{
   ASSERT_NULL(StrTrim(NULL));
}

/*-----------------------------------------------------------------------------
 * Test ComputeNumberOfDigits
 *-----------------------------------------------------------------------------*/

static void
test_ComputeNumberOfDigits_basic(void)
{
   ASSERT_EQ(ComputeNumberOfDigits(0), 0); /* Implementation returns 0 for 0 */
   ASSERT_EQ(ComputeNumberOfDigits(1), 1);
   ASSERT_EQ(ComputeNumberOfDigits(9), 1);
   ASSERT_EQ(ComputeNumberOfDigits(10), 2);
   ASSERT_EQ(ComputeNumberOfDigits(99), 2);
   ASSERT_EQ(ComputeNumberOfDigits(100), 3);
   ASSERT_EQ(ComputeNumberOfDigits(999), 3);
   ASSERT_EQ(ComputeNumberOfDigits(9999), 4);
}

static void
test_ComputeNumberOfDigits_large(void)
{
   ASSERT_EQ(ComputeNumberOfDigits(1000000), 7);
   ASSERT_EQ(ComputeNumberOfDigits(9999999), 7);
}

/*-----------------------------------------------------------------------------
 * Test SplitFilename
 *-----------------------------------------------------------------------------*/

static void
test_SplitFilename_full_path(void)
{
   char *dirname, *basename;
   SplitFilename("/path/to/file.txt", &dirname, &basename);
   ASSERT_STREQ(dirname, "/path/to");
   ASSERT_STREQ(basename, "file.txt");
   free(dirname);
   free(basename);
}

static void
test_SplitFilename_no_dir(void)
{
   char *dirname, *basename;
   SplitFilename("file.txt", &dirname, &basename);
   ASSERT_STREQ(dirname, ".");
   ASSERT_STREQ(basename, "file.txt");
   free(dirname);
   free(basename);
}

static void
test_SplitFilename_root_dir(void)
{
   char *dirname, *basename;
   SplitFilename("/file.txt", &dirname, &basename);
   ASSERT_STREQ(dirname, "");
   ASSERT_STREQ(basename, "file.txt");
   free(dirname);
   free(basename);
}

/*-----------------------------------------------------------------------------
 * Test CombineFilename
 *-----------------------------------------------------------------------------*/

static void
test_CombineFilename_basic(void)
{
   char *filename;
   CombineFilename("/path/to", "file.txt", &filename);
   ASSERT_STREQ(filename, "/path/to/file.txt");
   free(filename);
}

static void
test_CombineFilename_no_slash_needed(void)
{
   char *filename;
   CombineFilename("/path/to/", "file.txt", &filename);
   ASSERT_STREQ(filename, "/path/to/file.txt");
   free(filename);
}

static void
test_CombineFilename_empty_dir(void)
{
   char *filename;
   CombineFilename("", "file.txt", &filename);
   /* Implementation doesn't add "/" when dirname is empty */
   ASSERT_STREQ(filename, "file.txt");
   free(filename);
}

/*-----------------------------------------------------------------------------
 * Test IsYAMLFilename
 *-----------------------------------------------------------------------------*/

static void
test_IsYAMLFilename_valid(void)
{
   ASSERT_TRUE(IsYAMLFilename("file.yml"));         /* .yml extension */
   ASSERT_TRUE(IsYAMLFilename("file.yaml"));        /* .yaml extension */
   ASSERT_TRUE(IsYAMLFilename("path/to/file.yml"));  /* .yml with path */
   ASSERT_TRUE(IsYAMLFilename("path/to/file.yaml")); /* .yaml with path */
   ASSERT_FALSE(IsYAMLFilename("file.txt"));         /* wrong extension */
   ASSERT_FALSE(IsYAMLFilename("file.c"));          /* wrong extension */
}

static void
test_IsYAMLFilename_no_extension(void)
{
   ASSERT_FALSE(IsYAMLFilename("file"));
   ASSERT_FALSE(IsYAMLFilename("path/to/file"));
   ASSERT_FALSE(IsYAMLFilename(""));
}

static void
test_IsYAMLFilename_leading_dot(void)
{
   ASSERT_FALSE(IsYAMLFilename(".hidden"));
   ASSERT_FALSE(IsYAMLFilename(".gitignore"));
}

static void
test_IsYAMLFilename_with_spaces(void)
{
   ASSERT_FALSE(IsYAMLFilename("file name.yml"));    /* space in filename */
   ASSERT_FALSE(IsYAMLFilename("file name.yaml"));   /* space in filename */
   ASSERT_FALSE(IsYAMLFilename("path/to/file name.yml")); /* space in filename */
}

/*-----------------------------------------------------------------------------
 * Test CheckBinaryDataExists and CheckASCIIDataExists
 *-----------------------------------------------------------------------------*/

static void
test_CheckDataExists_binary(void)
{
   /* Create a temporary binary file */
   char prefix[] = "test_data";
   char expected_file[256];
   snprintf(expected_file, sizeof(expected_file), "%s.00000.bin", prefix);

   FILE *fp = fopen(expected_file, "wb");
   if (fp)
   {
      fprintf(fp, "binary data");
      fclose(fp);
      add_temp_file(expected_file);
   }

   ASSERT_EQ(CheckBinaryDataExists(prefix), 1);

   cleanup_temp_files();
}

static void
test_CheckDataExists_ascii(void)
{
   /* Create a temporary ASCII file */
   char prefix[] = "test_data";
   char expected_file[256];
   snprintf(expected_file, sizeof(expected_file), "%s.00000", prefix);

   FILE *fp = fopen(expected_file, "w");
   if (fp)
   {
      fprintf(fp, "ascii data");
      fclose(fp);
      add_temp_file(expected_file);
   }

   ASSERT_EQ(CheckASCIIDataExists(prefix), 1);

   cleanup_temp_files();
}

/*-----------------------------------------------------------------------------
 * Test CountNumberOfPartitions
 *-----------------------------------------------------------------------------*/

static void
test_CountNumberOfPartitions_binary(void)
{
   char prefix[] = "test_part";
   char filename[256];

   /* Create mock partition files */
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
   FILE *fp = fopen(filename, "w");
   if (fp) fclose(fp);
   add_temp_file(filename);

   snprintf(filename, sizeof(filename), "%s.00001.bin", prefix);
   fp = fopen(filename, "w");
   if (fp) fclose(fp);
   add_temp_file(filename);

   ASSERT_EQ(CountNumberOfPartitions(prefix), 2);

   cleanup_temp_files();
}

static void
test_CountNumberOfPartitions_ascii(void)
{
   char prefix[] = "test_part";
   char filename[256];

   /* Create mock partition files (ASCII) */
   snprintf(filename, sizeof(filename), "%s.00000", prefix);
   FILE *fp = fopen(filename, "w");
   if (fp) fclose(fp);
   add_temp_file(filename);

   snprintf(filename, sizeof(filename), "%s.00001", prefix);
   fp = fopen(filename, "w");
   if (fp) fclose(fp);
   add_temp_file(filename);

   snprintf(filename, sizeof(filename), "%s.00002", prefix);
   fp = fopen(filename, "w");
   if (fp) fclose(fp);
   add_temp_file(filename);

   ASSERT_EQ(CountNumberOfPartitions(prefix), 3);

   cleanup_temp_files();
}

static void
test_CountNumberOfPartitions_empty(void)
{
   char prefix[] = "nonexistent";
   ASSERT_EQ(CountNumberOfPartitions(prefix), 0);
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int
main(void)
{
   RUN_TEST(test_StrToLowerCase_basic);
   RUN_TEST(test_StrToLowerCase_empty);
   RUN_TEST(test_StrToLowerCase_special_chars);

   RUN_TEST(test_StrTrim_trailing);
   RUN_TEST(test_StrTrim_leading);
   RUN_TEST(test_StrTrim_no_spaces);
   RUN_TEST(test_StrTrim_empty);
   RUN_TEST(test_StrTrim_only_spaces);
   RUN_TEST(test_StrTrim_null);

   RUN_TEST(test_ComputeNumberOfDigits_basic);
   RUN_TEST(test_ComputeNumberOfDigits_large);

   RUN_TEST(test_SplitFilename_full_path);
   RUN_TEST(test_SplitFilename_no_dir);
   RUN_TEST(test_SplitFilename_root_dir);

   RUN_TEST(test_CombineFilename_basic);
   RUN_TEST(test_CombineFilename_no_slash_needed);
   RUN_TEST(test_CombineFilename_empty_dir);

   RUN_TEST(test_IsYAMLFilename_valid);
   RUN_TEST(test_IsYAMLFilename_no_extension);
   RUN_TEST(test_IsYAMLFilename_leading_dot);
   RUN_TEST(test_IsYAMLFilename_with_spaces);

   RUN_TEST(test_CheckDataExists_binary);
   RUN_TEST(test_CheckDataExists_ascii);

   RUN_TEST(test_CountNumberOfPartitions_binary);
   RUN_TEST(test_CountNumberOfPartitions_ascii);
   RUN_TEST(test_CountNumberOfPartitions_empty);

   cleanup_temp_files();

   return 0; /* Success - CTest handles reporting */
}
