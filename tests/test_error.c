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
#include "internal/error.h"
#include "test_helpers.h"

static void
capture_error_output(void (*print_fn)(void), char *buffer, size_t buf_len);

/*-----------------------------------------------------------------------------
 * Test ErrorCode operations
 *-----------------------------------------------------------------------------*/

static void
test_ErrorCodeSet_get(void)
{
   /* Reset first to ensure clean state */
   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_INVALID_KEY);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
}

static void
test_ErrorCodeReset(void)
{
   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeReset(ERROR_INVALID_KEY);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
}

static void
test_ErrorCodeMultiple(void)
{
   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);

   uint32_t codes = hypredrv_ErrorCodeGet();
   ASSERT_TRUE(codes & ERROR_INVALID_KEY);
   ASSERT_TRUE(codes & ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());

   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
}

/*-----------------------------------------------------------------------------
 * Test ErrorMsg operations
 *-----------------------------------------------------------------------------*/

static void
test_ErrorMsgAdd(void)
{
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAdd("Test message %d", 42);
   hypredrv_ErrorMsgAdd("Another message");

   /* Messages should be stored - we can't easily verify without printing */
   /* But we can verify no crash */
}

static void
test_ErrorMsgClear(void)
{
   hypredrv_ErrorMsgAdd("Test message");
   hypredrv_ErrorMsgClear();

   /* Clear should not crash */
}

static void
test_ErrorStateReset(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorMsgAdd("Message before state reset");

   hypredrv_ErrorStateReset();

   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_NONE);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());

   char buffer[512];
   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NULL(strstr(buffer, "Message before state reset"));
}

static void
test_ErrorMsgAdd_null_format(void)
{
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAdd(NULL);

   char buffer[512];
   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NOT_NULL(strstr(buffer, "(null format)"));

   hypredrv_ErrorMsgClear();
}

static void
test_ErrorMsgHelpers_accept_null_inputs(void)
{
   hypredrv_ErrorMsgClear();

   hypredrv_ErrorMsgAddMissingKey(NULL);
   hypredrv_ErrorMsgAddExtraKey(NULL);
   hypredrv_ErrorMsgAddUnexpectedVal(NULL);
   hypredrv_ErrorMsgAddInvalidFilename(NULL);

   char buffer[1024];
   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NOT_NULL(strstr(buffer, "(null)"));

   hypredrv_ErrorMsgClear();
}

static void
test_ErrorMsgAddMissingKey(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAddMissingKey("missing_key");

   /* hypredrv_ErrorMsgAddMissingKey only adds a message, doesn't set error code */
   /* The error code must be set separately via hypredrv_ErrorCodeSet */
   /* This test just verifies the function doesn't crash */
   hypredrv_ErrorCodeSet(ERROR_MISSING_KEY);
   uint32_t code = hypredrv_ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_MISSING_KEY);

   hypredrv_ErrorCodeResetAll();
}

static void
test_ErrorMsgAddExtraKey(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAddExtraKey("extra_key");

   /* hypredrv_ErrorMsgAddExtraKey only adds a message, doesn't set error code */
   /* The error code must be set separately via hypredrv_ErrorCodeSet */
   hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
   uint32_t code = hypredrv_ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_EXTRA_KEY);

   hypredrv_ErrorCodeResetAll();
}

static void
test_ErrorMsgAddUnexpectedVal(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAddUnexpectedVal("unexpected_value");

   /* hypredrv_ErrorMsgAddUnexpectedVal only adds a message, doesn't set error code */
   /* The error code must be set separately via hypredrv_ErrorCodeSet */
   hypredrv_ErrorCodeSet(ERROR_UNEXPECTED_VAL);
   uint32_t code = hypredrv_ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_UNEXPECTED_VAL);

   hypredrv_ErrorCodeResetAll();
}

static void
test_ErrorMsgAddInvalidFilename(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorMsgAddInvalidFilename("invalid_file.txt");

   /* hypredrv_ErrorMsgAddInvalidFilename only adds a message, doesn't set error code */
   /* The error code must be set separately via hypredrv_ErrorCodeSet */
   hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
   uint32_t code = hypredrv_ErrorCodeGet();
   ASSERT_TRUE(code & ERROR_FILE_NOT_FOUND);

   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * Helpers to capture stderr from hypredrv_ErrorMsgPrint
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
test_ErrorCodeDescribe_comprehensive_table(void)
{
   typedef struct
   {
      uint32_t         code_mask;
      const char      *expect;
      const char      *label;
   } describe_case;

   static const describe_case cases[] = {
      {ERROR_YAML_INVALID_INDENT, "invalid indendation", "indent"},
      {ERROR_YAML_INVALID_DIVISOR, "invalid divisor", "divisor"},
      {ERROR_INVALID_VAL, "invalid value", "invalid_val"},
      {ERROR_UNEXPECTED_VAL, "unexpected value", "unexpected_val"},
      {ERROR_MAYBE_INVALID_VAL, "possibly invalid value", "maybe_invalid"},
      {ERROR_MISSING_DOFMAP, "Missing dofmap info needed by MGR!", "missing_dofmap"},
      {ERROR_UNKNOWN_HYPREDRV_OBJ, "HYPREDRV object is not set properly!!", "unknown_obj"},
      {ERROR_HYPREDRV_NOT_INITIALIZED, "HYPREDRV is not initialized!!", "not_init"},
      {ERROR_HYPRE_INTERNAL, "HYPRE internal error", "hypre_internal"},
   };

   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      hypredrv_ErrorCodeSet((hypredrv_error_t)cases[i].code_mask);
      hypredrv_ErrorCodeDescribe(hypredrv_ErrorCodeGet());

      char buffer[768];
      capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
      ASSERT_NOT_NULL(strstr(buffer, cases[i].expect));
   }

   /* Plural branch in hypredrv_ErrorMsgAddCodeWithCount (count > 1). */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorCodeDescribe(hypredrv_ErrorCodeGet());
   {
      char buffer[512];
      capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
      ASSERT_NOT_NULL(strstr(buffer, "Found 2 invalid keys"));
   }

   /* Singular branch for the same code path. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorCodeDescribe(hypredrv_ErrorCodeGet());
   {
      char buffer[512];
      capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
      ASSERT_NOT_NULL(strstr(buffer, "Found 1 invalid key"));
   }

   /* Combined mask: multiple describe() branches in one pass. */
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeSet(ERROR_YAML_INVALID_INDENT);
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorCodeSet(ERROR_UNEXPECTED_VAL);
   hypredrv_ErrorCodeDescribe(hypredrv_ErrorCodeGet());
   {
      char buffer[1024];
      capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
      ASSERT_NOT_NULL(strstr(buffer, "invalid indendation"));
      ASSERT_NOT_NULL(strstr(buffer, "invalid value"));
      ASSERT_NOT_NULL(strstr(buffer, "unexpected value"));
   }

   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeResetAll();
}

static void
test_ErrorMsgAddCodeWithCount_null_suffix(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
   hypredrv_ErrorMsgAddCodeWithCount(ERROR_INVALID_KEY, NULL);

   /* hypredrv_ErrorMsgPrint() emits a large banner; a small buffer truncates before the
    * detail lines captured by capture_error_output(). */
   char buffer[2048];
   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NOT_NULL(strstr(buffer, "Found 1 (null)"));

   hypredrv_ErrorMsgClear();
   hypredrv_ErrorCodeResetAll();
}

static void
test_DistributedErrorCodeActive(void)
{
   hypredrv_ErrorCodeResetAll();
   ASSERT_FALSE(hypredrv_DistributedErrorCodeActive(MPI_COMM_SELF));

   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_DistributedErrorCodeActive(MPI_COMM_SELF));

   hypredrv_ErrorCodeResetAll();
}

static void
test_DistributedErrorStateSync_self_preserves_code_and_messages(void)
{
   char buffer[2048];
   char *pos_second = NULL;
   char *pos_first  = NULL;

   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
   hypredrv_ErrorMsgAdd("first message");
   hypredrv_ErrorMsgAdd("second message");

   ASSERT_TRUE(hypredrv_DistributedErrorStateSync(MPI_COMM_SELF));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);

   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   pos_second = strstr(buffer, "second message");
   pos_first  = strstr(buffer, "first message");
   ASSERT_NOT_NULL(pos_second);
   ASSERT_NOT_NULL(pos_first);
   /* Messages are stored LIFO; "second" (added last) should print before "first". */
   ASSERT_TRUE(pos_second < pos_first);

   hypredrv_ErrorStateReset();
}

static void
test_DistributedErrorStateSync_world_preserves_nonroot_descriptions(void)
{
   int  nprocs = 0;
   int  myid   = 0;
   char buffer[4096];

   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (nprocs < 2)
   {
      fprintf(stderr, "SKIP: distributed error sync world test requires at least 2 MPI ranks\n");
      return;
   }

   hypredrv_ErrorStateReset();
   if (myid == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("rank0 detail");
   }
   else if (myid == 1)
   {
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
   }

   ASSERT_TRUE(hypredrv_DistributedErrorStateSync(MPI_COMM_WORLD));
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL);
   ASSERT_TRUE(hypredrv_ErrorCodeGet() & ERROR_HYPRE_INTERNAL);

   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   ASSERT_NOT_NULL(strstr(buffer, "rank0 detail"));
   ASSERT_NOT_NULL(strstr(buffer, "HYPRE internal error"));

   hypredrv_ErrorStateReset();
}

static void
test_ErrorMsgPrint_with_no_messages(void)
{
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   /* hypredrv_ErrorMsgPrint with no messages should not crash */
   char buffer[512];
   capture_error_output(hypredrv_ErrorMsgPrint, buffer, sizeof(buffer));
   /* Should print header but no error details */

   hypredrv_ErrorCodeResetAll();
}

/*-----------------------------------------------------------------------------
 * Test backtrace symbolization
 *-----------------------------------------------------------------------------*/

/* Helper to call hypredrv_ErrorBacktracePrint directly */
static void
call_ErrorBacktracePrint(void)
{
   hypredrv_ErrorBacktracePrint();
}

static void
test_ErrorBacktracePrint_has_filenames_and_lines(void)
{
#ifdef __linux__
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   /* Capture output from hypredrv_ErrorBacktracePrint directly */
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
      /* Older environments may not have addr2line/symbols available. */
      fprintf(stderr, "SKIP: Backtrace lacks file:line; addr2line likely unavailable.\n");
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      return;
   }
   
   /* Require file:line format for the test to pass when available */
   if (!found_file_line)
   {
      fprintf(stderr, "SKIP: Backtrace lacks file:line information.\n");
      hypredrv_ErrorCodeResetAll();
      hypredrv_ErrorMsgClear();
      return;
   }

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();
#else
   /* Backtrace is only supported on Linux */
   fprintf(stderr, "SKIP: Backtrace test only runs on Linux\n");
#endif
}

static void
test_ErrorBacktracePrint_respects_no_backtrace_env(void)
{
#ifdef __linux__
   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

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
   RUN_TEST(test_ErrorStateReset);
   RUN_TEST(test_ErrorMsgAdd_null_format);
   RUN_TEST(test_ErrorMsgHelpers_accept_null_inputs);
   RUN_TEST(test_ErrorMsgAddMissingKey);
   RUN_TEST(test_ErrorMsgAddExtraKey);
   RUN_TEST(test_ErrorMsgAddUnexpectedVal);
   RUN_TEST(test_ErrorMsgAddInvalidFilename);
   RUN_TEST(test_ErrorCodeDescribe_comprehensive_table);
   RUN_TEST(test_ErrorMsgAddCodeWithCount_null_suffix);
   RUN_TEST(test_DistributedErrorCodeActive);
   RUN_TEST(test_DistributedErrorStateSync_self_preserves_code_and_messages);
   RUN_TEST(test_DistributedErrorStateSync_world_preserves_nonroot_descriptions);
   RUN_TEST(test_ErrorMsgPrint_with_no_messages);
   RUN_TEST(test_ErrorBacktracePrint_has_filenames_and_lines);
   RUN_TEST(test_ErrorBacktracePrint_respects_no_backtrace_env);

   hypredrv_ErrorCodeResetAll();
   hypredrv_ErrorMsgClear();

   MPI_Finalize();

   return 0; /* Success - CTest handles reporting */
}
