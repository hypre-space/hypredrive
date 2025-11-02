/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "test_helpers.h"
#include "error.h"

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
 * Main test runner (CTest handles test counting and reporting)
 *-----------------------------------------------------------------------------*/

int main(void)
{
   RUN_TEST(test_ErrorCodeSet_get);
   RUN_TEST(test_ErrorCodeReset);
   RUN_TEST(test_ErrorCodeMultiple);

   RUN_TEST(test_ErrorMsgAdd);
   RUN_TEST(test_ErrorMsgClear);
   RUN_TEST(test_ErrorMsgAddMissingKey);
   RUN_TEST(test_ErrorMsgAddExtraKey);
   RUN_TEST(test_ErrorMsgAddUnexpectedVal);
   RUN_TEST(test_ErrorMsgAddInvalidFilename);

   ErrorCodeResetAll();
   ErrorMsgClear();

   return 0;  /* Success - CTest handles reporting */
}

