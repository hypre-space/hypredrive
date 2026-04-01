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

#include "HYPREDRV.h"
#include "internal/stats.h"
#include "logging.h"
#include "object.h"
#include "test_helpers.h"

typedef void (*capture_fn)(void *);

static void
capture_stderr_output(capture_fn fn, void *context, char *buffer, size_t buf_len)
{
   FILE *tmp = tmpfile();
   ASSERT_NOT_NULL(tmp);

   int tmp_fd    = fileno(tmp);
   int saved_err = dup(fileno(stderr));
   ASSERT_TRUE(saved_err != -1);

   fflush(stderr);
   ASSERT_TRUE(dup2(tmp_fd, fileno(stderr)) != -1);

   fn(context);
   fflush(stderr);

   fseek(tmp, 0, SEEK_SET);
   size_t read_bytes  = fread(buffer, 1, buf_len - 1, tmp);
   buffer[read_bytes] = '\0';

   ASSERT_TRUE(dup2(saved_err, fileno(stderr)) != -1);
   close(saved_err);
   fclose(tmp);
}

static void
test_LogRankFromComm_before_MPI_Init(void)
{
   /* Reachable path: MPI not initialized yet. */
   ASSERT_EQ(hypredrv_LogRankFromComm(MPI_COMM_WORLD), -1);
}

static void
test_LogRankFromComm_comm_null(void)
{
   ASSERT_EQ(hypredrv_LogRankFromComm(MPI_COMM_NULL), -1);
}

static void
test_LogRankFromComm_cache_hit(void)
{
   MPI_Comm comm = MPI_COMM_SELF;
   int      r0   = hypredrv_LogRankFromComm(comm);
   int      r1   = hypredrv_LogRankFromComm(comm);
   ASSERT_EQ(r0, r1);
}

static void
test_LogLevelGet_and_LogEnabled(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   setenv("HYPREDRV_LOG_LEVEL", "2", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), 2);
   ASSERT_FALSE(hypredrv_LogEnabled(0));
   ASSERT_FALSE(hypredrv_LogEnabled(HYPREDRV_LOG_LEVEL_OFF));
   ASSERT_FALSE(hypredrv_LogEnabled(3));
   ASSERT_TRUE(hypredrv_LogEnabled(1));
   ASSERT_TRUE(hypredrv_LogEnabled(2));
   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogInitializeFromEnv_level_edge_cases(void)
{
   hypredrv_LogReset();

   unsetenv("HYPREDRV_LOG_LEVEL");
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_OFF);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_OFF);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "not_a_number", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_OFF);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3x", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_OFF);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "  2  ", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), 2);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "-5", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_OFF);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "99", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), HYPREDRV_LOG_LEVEL_MAX);

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogToStdoutParse_prefix_and_suffix_mismatch(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "1", 1);
   /* Shorter than "stdout": hits *expected != '\\0' after common prefix */
   setenv("HYPREDRV_LOG_STREAM", "std", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), 1);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_STREAM", "stdoutx", 1);
   hypredrv_LogInitializeFromEnv();

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogObjectf_disabled_early_return(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "0", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();
   hypredrv_LogObjectf(3, NULL, "off");
   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogInitializeFromEnv_stream_stdout(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "1", 1);
   setenv("HYPREDRV_LOG_STREAM", "  StdOut  ", 1);
   hypredrv_LogInitializeFromEnv();
   ASSERT_EQ(hypredrv_LogLevelGet(), 1);

   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
run_log_commf_ls_branch(void *ctx)
{
   (void)ctx;
   hypredrv_LogCommf(3, MPI_COMM_SELF, "commobj", 2, "ls branch");
}

static void
test_LogCommf_ls_id_positive(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();

   char buf[1024];
   capture_stderr_output(run_log_commf_ls_branch, NULL, buf, sizeof(buf));

   ASSERT_NOT_NULL(strstr(buf, "[ls=2]"));
   ASSERT_NOT_NULL(strstr(buf, "ls branch"));

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogCommf_Logf_early_return_when_disabled(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "0", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();

   hypredrv_LogCommf(2, MPI_COMM_SELF, "n", 0, "should not print");
   hypredrv_Logf(2, 0, "n", 0, "should not print");

   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

struct objectf_ctx
{
   int mode;
};

static void
run_log_objectf_cases(void *ctx)
{
   struct objectf_ctx *c = (struct objectf_ctx *)ctx;
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();

   if (c->mode == 0)
   {
      hypredrv_LogObjectf(3, NULL, "null handle");
   }
   else if (c->mode == 1)
   {
      hypredrv_t stub;
      memset(&stub, 0, sizeof(stub));
      stub.mypid              = 0;
      stub.stats              = NULL;
      stub.runtime_object_id  = 0;
      hypredrv_LogObjectf(3, &stub, "stats NULL");
   }
   else
   {
      hypredrv_t stub;
      memset(&stub, 0, sizeof(stub));
      stub.mypid             = 0;
      stub.stats             = NULL;
      stub.runtime_object_id = 7;
      hypredrv_LogObjectf(3, &stub, "obj id fallback");
   }
   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogObjectf_branches(void)
{
   char               buf[2048];
   struct objectf_ctx ctx;

   ctx.mode = 0;
   capture_stderr_output(run_log_objectf_cases, &ctx, buf, sizeof(buf));
   /* NULL handle: mypid stays -1; LogVf suppresses output on rank != 0 */
   ASSERT_STREQ(buf, "");

   ctx.mode = 1;
   capture_stderr_output(run_log_objectf_cases, &ctx, buf, sizeof(buf));
   ASSERT_NOT_NULL(strstr(buf, "stats NULL"));

   ctx.mode = 2;
   capture_stderr_output(run_log_objectf_cases, &ctx, buf, sizeof(buf));
   ASSERT_NOT_NULL(strstr(buf, "obj-7"));
}

struct textblock_ctx
{
   const char *header;
   const char *text;
   int         mypid;
};

static void
run_log_textblock(void *ctx)
{
   struct textblock_ctx *t = (struct textblock_ctx *)ctx;
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();
   hypredrv_LogTextBlock(3, t->mypid, "obj", 0, t->header, t->text);
   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

static void
test_LogTextBlock_variants(void)
{
   char               buf[4096];
   struct textblock_ctx t;

   t.header = "Header line";
   t.text   = "line1\nline2\n";
   t.mypid  = 0;
   capture_stderr_output(run_log_textblock, &t, buf, sizeof(buf));
   ASSERT_NOT_NULL(strstr(buf, "Header line"));
   ASSERT_NOT_NULL(strstr(buf, "line1"));

   t.header = NULL;
   t.text   = "only\nno final newline";
   t.mypid  = 0;
   capture_stderr_output(run_log_textblock, &t, buf, sizeof(buf));
   ASSERT_NOT_NULL(strstr(buf, "no final newline"));

   t.header = "H";
   t.text   = "x";
   t.mypid  = 1;
   capture_stderr_output(run_log_textblock, &t, buf, sizeof(buf));
   /* Rank != 0: no output to rank-0 capture */
   ASSERT_STREQ(buf, "");
}

static void
test_LogTextBlock_null_text_guard(void)
{
   hypredrv_LogReset();
   setenv("HYPREDRV_LOG_LEVEL", "3", 1);
   setenv("HYPREDRV_LOG_STREAM", "stderr", 1);
   hypredrv_LogInitializeFromEnv();
   hypredrv_LogTextBlock(3, 0, "o", 0, "h", NULL);
   hypredrv_LogReset();
   unsetenv("HYPREDRV_LOG_LEVEL");
   unsetenv("HYPREDRV_LOG_STREAM");
}

int
main(int argc, char **argv)
{
   /* Must run before MPI_Init for !MPI_Initialized() branch in LogRankFromComm */
   RUN_TEST(test_LogRankFromComm_before_MPI_Init);

   MPI_Init(&argc, &argv);

   RUN_TEST(test_LogRankFromComm_comm_null);
   RUN_TEST(test_LogRankFromComm_cache_hit);
   RUN_TEST(test_LogLevelGet_and_LogEnabled);
   RUN_TEST(test_LogInitializeFromEnv_level_edge_cases);
   RUN_TEST(test_LogToStdoutParse_prefix_and_suffix_mismatch);
   RUN_TEST(test_LogObjectf_disabled_early_return);
   RUN_TEST(test_LogInitializeFromEnv_stream_stdout);
   RUN_TEST(test_LogCommf_ls_id_positive);
   RUN_TEST(test_LogCommf_Logf_early_return_when_disabled);
   RUN_TEST(test_LogObjectf_branches);
   RUN_TEST(test_LogTextBlock_variants);
   RUN_TEST(test_LogTextBlock_null_text_guard);

   MPI_Finalize();
   return 0;
}
