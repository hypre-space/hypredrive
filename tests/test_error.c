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
 * Test backtrace symbolization
 *-----------------------------------------------------------------------------*/

/* Helper to call ErrorBacktracePrint directly */
static void
call_ErrorBacktracePrint(void)
{
   ErrorBacktracePrint();
}

static void
test_ErrorBacktracePrint_has_filenames_and_lines(void)
{
#ifdef __linux__
   ErrorCodeResetAll();
   ErrorMsgClear();

   /* Capture output from ErrorBacktracePrint directly */
   char buffer[4096];
   capture_error_output(call_ErrorBacktracePrint, buffer, sizeof(buffer));

   /* Verify backtrace section exists */
   const char *backtrace_start = strstr(buffer, "Backtrace:");
   ASSERT_NOT_NULL(backtrace_start);

   /* Verify backtrace contains file paths and line numbers (not just raw addresses) */
   /* Look for patterns like "at /path/to/file.c:123" or "at file.c:123" */
   /* This ensures addr2line is working and producing meaningful output */
   
   /* Check that we have at least one line with file:line format */
   /* Format should be: "#N function_name at /path/to/file.c:line" */
   int found_file_line = 0;
   
   /* Look for ".c:" pattern which indicates file:line format */
   const char *dot_c = strstr(backtrace_start, ".c:");
   if (dot_c)
   {
      /* Verify there's a digit after the colon (line number) */
      const char *after_colon = dot_c + 3; /* Skip ".c:" */
      if (*after_colon >= '0' && *after_colon <= '9')
      {
         found_file_line = 1;
      }
   }
   
   /* Alternative: look for " at " followed by file path with colon and digits */
   if (!found_file_line)
   {
      const char *at_pos = strstr(backtrace_start, " at ");
      if (at_pos)
      {
         /* Look for colon followed by digits (line number) */
         const char *colon = strchr(at_pos, ':');
         if (colon)
         {
            /* Check if there are digits after the colon */
            const char *after_colon = colon + 1;
            if (*after_colon >= '0' && *after_colon <= '9')
            {
               found_file_line = 1;
            }
         }
      }
   }

   /* Regression check: fail if we see raw addresses instead of file:line */
   const char *raw_addr_pattern = strstr(backtrace_start, "(+0x");
   if (raw_addr_pattern && !found_file_line)
   {
      /* If we see raw addresses like "(+0x...)" and no file:line, that's a regression */
      TEST_FAIL("Backtrace shows raw addresses instead of file:line. "
                "This indicates addr2line is not working correctly.");
   }
   
   /* Require file:line format for the test to pass */
   if (!found_file_line)
   {
      TEST_FAIL("Backtrace does not contain file:line information. "
                "Expected format: 'function_name at /path/to/file.c:line'");
   }

   ErrorCodeResetAll();
   ErrorMsgClear();
#else
   /* Backtrace is only supported on Linux */
   fprintf(stderr, "SKIP: Backtrace test only runs on Linux\n");
#endif
}

static void
test_ErrorBacktracePrint_respects_no_backtrace_env(void)
{
#ifdef __linux__
   ErrorCodeResetAll();
   ErrorMsgClear();

   /* Force early return */
   setenv("HYPREDRV_NO_BACKTRACE", "1", 1);

   char buffer[2048];
   capture_error_output(call_ErrorBacktracePrint, buffer, sizeof(buffer));

   /* Should not print the backtrace header at all */
   ASSERT_NULL(strstr(buffer, "Backtrace:"));

   unsetenv("HYPREDRV_NO_BACKTRACE");
#else
   fprintf(stderr, "SKIP: Backtrace env test only runs on Linux\n");
#endif
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
   RUN_TEST(test_ErrorBacktracePrint_has_filenames_and_lines);
   RUN_TEST(test_ErrorBacktracePrint_respects_no_backtrace_env);

   ErrorCodeResetAll();
   ErrorMsgClear();

   MPI_Finalize();

   return 0; /* Success - CTest handles reporting */
}
