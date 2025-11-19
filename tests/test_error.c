/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "error.h"
#include "test_helpers.h"

/*-----------------------------------------------------------------------------
 * Test ErrorCode operations
 *-----------------------------------------------------------------------------*/

static void
test_ErrorCodeSet_get(void)
{
   /* Reset first to ensure clean state */
   ErrorCodeResetAll();
   ASSERT_EQ(ErrorCodeGet(), ERROR_NONE);
   ASSERT_FALSE(ErrorCodeActive());

   ErrorCodeSet(ERROR_INVALID_KEY);
   ASSERT_EQ(ErrorCodeGet(), ERROR_INVALID_KEY);
   ASSERT_TRUE(ErrorCodeActive());

   ErrorCodeResetAll();
   ASSERT_EQ(ErrorCodeGet(), ERROR_NONE);
   ASSERT_FALSE(ErrorCodeActive());
}

static void
test_ErrorCodeReset(void)
{
   ErrorCodeSet(ERROR_INVALID_KEY);
   ASSERT_TRUE(ErrorCodeActive());

   ErrorCodeReset(ERROR_INVALID_KEY);
   ASSERT_FALSE(ErrorCodeActive());

   ErrorCodeResetAll();
   ASSERT_FALSE(ErrorCodeActive());
}

static void
test_ErrorCodeMultiple(void)
{
   ErrorCodeSet(ERROR_INVALID_KEY);
   ErrorCodeSet(ERROR_INVALID_VAL);

   uint32_t codes = ErrorCodeGet();
   ASSERT_TRUE(codes & ERROR_INVALID_KEY);
   ASSERT_TRUE(codes & ERROR_INVALID_VAL);
   ASSERT_TRUE(ErrorCodeActive());

   ErrorCodeResetAll();
   ASSERT_FALSE(ErrorCodeActive());
}

/*-----------------------------------------------------------------------------
 * Test ErrorMsg operations
 *-----------------------------------------------------------------------------*/

static void
test_ErrorMsgAdd(void)
{
   ErrorMsgClear();
   ErrorMsgAdd("Test message %d", 42);
   ErrorMsgAdd("Another message");

   /* Messages should be stored - we can't easily verify without printing */
   /* But we can verify no crash */
}

static void
test_ErrorMsgClear(void)
{
   ErrorMsgAdd("Test message");
   ErrorMsgClear();

   /* Clear should not crash */
}

static void
test_ErrorMsgAddMissingKey(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();
   ErrorMsgAddMissingKey("missing_key");

   /* ErrorMsgAddMissingKey only adds a message, doesn't set error code */
   /* The error code must be set separately via ErrorCodeSet */
   /* This test just verifies the function doesn't crash */
   ErrorCodeSet(ERROR_MISSING_KEY);
   uint32_t code = ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_MISSING_KEY);

   ErrorCodeResetAll();
}

static void
test_ErrorMsgAddExtraKey(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();
   ErrorMsgAddExtraKey("extra_key");

   /* ErrorMsgAddExtraKey only adds a message, doesn't set error code */
   /* The error code must be set separately via ErrorCodeSet */
   ErrorCodeSet(ERROR_EXTRA_KEY);
   uint32_t code = ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_EXTRA_KEY);

   ErrorCodeResetAll();
}

static void
test_ErrorMsgAddUnexpectedVal(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();
   ErrorMsgAddUnexpectedVal("unexpected_value");

   /* ErrorMsgAddUnexpectedVal only adds a message, doesn't set error code */
   /* The error code must be set separately via ErrorCodeSet */
   ErrorCodeSet(ERROR_UNEXPECTED_VAL);
   uint32_t code = ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_UNEXPECTED_VAL);

   ErrorCodeResetAll();
}

static void
test_ErrorMsgAddInvalidFilename(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();
   ErrorMsgAddInvalidFilename("invalid_file.txt");

   /* ErrorMsgAddInvalidFilename only adds a message, doesn't set error code */
   /* The error code must be set separately via ErrorCodeSet */
   ErrorCodeSet(ERROR_FILE_NOT_FOUND);
   uint32_t code = ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_FILE_NOT_FOUND);

   ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * Helpers to capture stderr from ErrorMsgPrint
 *-----------------------------------------------------------------------------*/

static void
capture_error_output(void (*print_fn)(void), char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();

   // macOS fallback
#ifdef __APPLE__
   if (!tmp)
   {
      char path[] = "/tmp/hypredrv_test_error.txt";
      int fd = mkstemp(path);
      ASSERT_TRUE(fd != -1);
      tmp = fdopen(fd, "w+");
      unlink(path);
   }
#endif

   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   print_fn();
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   fflush(tmp);
   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

static void
test_ErrorCodeDescribe_prints_counts(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();

   ErrorCodeSet(ERROR_INVALID_KEY);
   ErrorCodeSet(ERROR_INVALID_KEY);

   ErrorCodeDescribe(ErrorCodeGet());

   char buffer[512];
   capture_error_output(ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NOT_NULL(strstr(buffer, "Found 2 invalid keys"));

   ErrorMsgClear();
   ErrorCodeResetAll();
}

static void
test_DistributedErrorCodeActive(void)
{
   ErrorCodeResetAll();
   ASSERT_FALSE(DistributedErrorCodeActive(MPI_COMM_SELF));

   ErrorCodeSet(ERROR_INVALID_VAL);
   ASSERT_TRUE(DistributedErrorCodeActive(MPI_COMM_SELF));

   ErrorCodeResetAll();
}

static void
test_ErrorMsgPrint_with_no_messages(void)
{
   ErrorCodeResetAll();
   ErrorMsgClear();

   /* ErrorMsgPrint with no messages should not crash */
   char buffer[512];
   capture_error_output(ErrorMsgPrint, buffer, sizeof(buffer));
   /* Should print header but no error details */

   ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_ErrorCodeSet_get);
   RUN_TEST(test_ErrorCodeReset);
   RUN_TEST(test_ErrorCodeMultiple);

   RUN_TEST(test_ErrorMsgAdd);
   RUN_TEST(test_ErrorMsgClear);
   RUN_TEST(test_ErrorMsgAddMissingKey);
   RUN_TEST(test_ErrorMsgAddExtraKey);
   RUN_TEST(test_ErrorMsgAddUnexpectedVal);
   RUN_TEST(test_ErrorMsgAddInvalidFilename);
   RUN_TEST(test_ErrorCodeDescribe_prints_counts);
   RUN_TEST(test_DistributedErrorCodeActive);
   RUN_TEST(test_ErrorMsgPrint_with_no_messages);

   ErrorCodeResetAll();
   ErrorMsgClear();

   MPI_Finalize();

   return 0; /* Success - CTest handles reporting */
}
