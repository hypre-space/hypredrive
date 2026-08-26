#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/help.h"
#include "test_helpers.h"

static char *
capture_help_with_status(const char *topic, int *status)
{
   char  *buf  = NULL;
   size_t size = 0;
   FILE  *fp   = open_memstream(&buf, &size);
   ASSERT_NOT_NULL(fp);

   int rc = hypredrv_HelpPrint(fp, "hypredrive-cli", topic);
   fclose(fp);
   ASSERT_NOT_NULL(buf);
   if (status)
   {
      *status = rc;
   }
   return buf;
}

static char *
capture_help(const char *topic)
{
   int   status = -1;
   char *buf    = capture_help_with_status(topic, &status);
   ASSERT_EQ(status, 0);
   return buf;
}

static void
assert_contains(const char *text, const char *needle)
{
   if (!strstr(text, needle))
   {
      fprintf(stderr, "FAIL: expected output to contain \"%s\"\nOutput:\n%s\n",
              needle, text);
      exit(1);
   }
}

static void
test_HelpPrint_root_sections(void)
{
   char *out = capture_help(NULL);
   assert_contains(out, "general");
   assert_contains(out, "linear_system");
   assert_contains(out, "solver");
   assert_contains(out, "preconditioner");
   free(out);
}

static void
test_HelpPrint_general_device_lazy_init(void)
{
   char *out = capture_help("general");
   assert_contains(out, "device_lazy_init");
   assert_contains(out, "Defer GPU device initialization");
   free(out);
}

static void
test_HelpPrint_solver_types(void)
{
   char *out = capture_help("solver");
   assert_contains(out, "pcg");
   assert_contains(out, "gmres");
   assert_contains(out, "fgmres");
   assert_contains(out, "bicgstab");
   free(out);
}

static void
test_HelpPrint_solver_key(void)
{
   char *out = capture_help("solver:gmres:max_iter");
   assert_contains(out, "Help for solver:gmres:max_iter");
   assert_contains(out, "max_iter  <value>  Maximum number of iterations");
   free(out);
}

static void
test_HelpPrint_solver_boolean_aliases_grouped(void)
{
   char *out = capture_help("solver:gmres");
   assert_contains(out,
                   "skip_real_res_check  <one of>  Skip GMRES real residual verification");
   assert_contains(out, "on/yes/true (1)");
   assert_contains(out, "off/no/false (0)");
   free(out);
}

static void
test_HelpPrint_linear_system_values(void)
{
   char *out = capture_help("linear_system");
   assert_contains(out, "init_guess_mode");
   assert_contains(out, "previous");
   assert_contains(out, "rhs_mode");
   assert_contains(out, "randsol");
   free(out);
}

static void
test_HelpPrint_mgr_nested_relaxation(void)
{
   char *out = capture_help("preconditioner:mgr:level:f_relaxation");
   assert_contains(out, "num_sweeps");
   assert_contains(out, "jacobi");
   assert_contains(out, "none (-1)");
   assert_contains(out, "Nested topics:");
   assert_contains(out, "amg");
   free(out);
}

static void
test_HelpPrint_mgr_numeric_level_alias(void)
{
   char *level_out = capture_help("preconditioner:mgr:level:0");
   assert_contains(level_out, "matched_q");
   assert_contains(level_out, "off (0)");
   assert_contains(level_out, "on/polynomial (1)");
   assert_contains(level_out, "afsai (2)");
   assert_contains(level_out, "Q=S*G^T*G*S");
   assert_contains(level_out, "matched_f_backsolve");
   assert_contains(level_out, "Selected-level matched F or Schur-GMRES(1) action");
   free(level_out);

   char *mgr_out = capture_help("preconditioner:mgr");
   assert_contains(mgr_out, "matched_q_sweeps");
   assert_contains(mgr_out, "ignored by afsai");
   assert_contains(mgr_out, "interp_sweeps");
   assert_contains(mgr_out, "interp_weight");
   assert_contains(mgr_out, "injection_upcycle");
   assert_contains(mgr_out, "pmax");
   assert_contains(mgr_out, "G-row contract");
   free(mgr_out);

   char *out = capture_help("preconditioner:mgr:level:0:f_relaxation");
   assert_contains(out, "symmetric_diagonal_scaling");
   free(out);

   out = capture_help("preconditioner:mgr:level:0:f_relaxation:amg");
   assert_contains(out, "Help for preconditioner:mgr:level:0:f_relaxation:amg");
   assert_contains(out, "AMG preconditioner");
   assert_contains(out, "coarsening");
   free(out);
}

static void
test_HelpPrint_reuse_nested_schema(void)
{
   char *out = capture_help("preconditioner:reuse:adaptive:components");
   assert_contains(out, "metric");
   assert_contains(out, "history");
   assert_contains(out, "<index>");
   free(out);
}

static void
assert_not_contains(const char *text, const char *needle)
{
   if (strstr(text, needle))
   {
      fprintf(stderr, "FAIL: expected output NOT to contain \"%s\"\nOutput:\n%s\n",
              needle, text);
      exit(1);
   }
}

static void
test_HelpPrint_reuse_mean_kind_values(void)
{
   char *out = capture_help("preconditioner:reuse:adaptive:components:0:mean");
   /* The mean section selects an algorithm via "kind", not "type", and lists the
    * mean-kind enum (not the reuse policy enum). */
   assert_contains(out, "kind");
   assert_contains(out, "arithmetic");
   assert_contains(out, "rms");
   assert_contains(out, "power");
   assert_not_contains(out, "static/always");
   free(out);
}

static void
test_HelpPrint_print_system_schema(void)
{
   char *out = capture_help("linear_system:print_system");
   assert_contains(out, "enabled");
   assert_contains(out, "every_n_systems");
   assert_contains(out, "setup_time_over");
   assert_contains(out, "build (1)");
   assert_contains(out, "apply (4)");
   assert_contains(out, "matrix");
   assert_contains(out, "selectors");
   free(out);

   out = capture_help("linear_system:print_system:selectors:0");
   assert_contains(out, "basis");
   assert_contains(out, "linear_system");
   assert_contains(out, "iterations");
   assert_contains(out, "threshold");
   free(out);
}

static void
test_HelpPrint_reuse_guard_transform_and_history_schema(void)
{
   char *out = capture_help("preconditioner:reuse");
   assert_contains(out, "frequency");
   assert_contains(out, "static/always");
   assert_contains(out, "guards");
   assert_contains(out, "adaptive");
   free(out);

   out = capture_help("preconditioner:reuse:guards");
   assert_contains(out, "min_reuse_solves");
   assert_contains(out, "rebuild_on_solver_failure");
   free(out);

   out = capture_help("preconditioner:reuse:adaptive");
   assert_contains(out, "rebuild_threshold");
   assert_contains(out, "components");
   free(out);

   out = capture_help("preconditioner:reuse:adaptive:components:0:transform");
   assert_contains(out, "ratio_to_baseline");
   assert_contains(out, "window_mean");
   assert_contains(out, "amortization_window");
   free(out);

   out = capture_help("preconditioner:reuse:adaptive:components:0:history");
   assert_contains(out, "linear_solves");
   assert_contains(out, "completed_level");
   assert_contains(out, "none (0)");
   assert_contains(out, "mean (1)");
   assert_contains(out, "sum (2)");
   free(out);
}

static void
test_HelpPrint_unknown_and_dot_separated_topics(void)
{
   char *out = capture_help("solver.gmres.max_iter");
   assert_contains(out, "Help for solver:gmres:max_iter");
   free(out);

   int status = 0;
   out        = capture_help_with_status("preconditioner:does_not_exist", &status);
   ASSERT_EQ(status, 1);
   assert_contains(out, "Unknown help topic");
   assert_contains(out, "Nested topics:");
   free(out);
}

static void
test_HelpRequested_joins_topic_args(void)
{
   char  topic[128];
   char *argv[] = {"hypredrive-cli", "--help", "preconditioner", "mgr", "level"};

   ASSERT_EQ(hypredrv_HelpRequested(5, argv, topic, sizeof(topic)), 1);
   ASSERT_STREQ(topic, "preconditioner:mgr:level");
}

int
main(void)
{
   RUN_TEST(test_HelpPrint_root_sections);
   RUN_TEST(test_HelpPrint_general_device_lazy_init);
   RUN_TEST(test_HelpPrint_solver_types);
   RUN_TEST(test_HelpPrint_solver_key);
   RUN_TEST(test_HelpPrint_solver_boolean_aliases_grouped);
   RUN_TEST(test_HelpPrint_linear_system_values);
   RUN_TEST(test_HelpPrint_mgr_nested_relaxation);
   RUN_TEST(test_HelpPrint_mgr_numeric_level_alias);
   RUN_TEST(test_HelpPrint_reuse_nested_schema);
   RUN_TEST(test_HelpPrint_reuse_mean_kind_values);
   RUN_TEST(test_HelpPrint_print_system_schema);
   RUN_TEST(test_HelpPrint_reuse_guard_transform_and_history_schema);
   RUN_TEST(test_HelpPrint_unknown_and_dot_separated_topics);
   RUN_TEST(test_HelpRequested_joins_topic_args);

   return 0;
}
