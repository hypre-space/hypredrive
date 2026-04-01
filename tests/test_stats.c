#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "HYPREDRV.h"
#include "internal/error.h"
#include "internal/stats.h"
#include "test_helpers.h"

static void
test_Stats_basic_lifecycle_and_timers(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsTimerSetMilliseconds(s);
   hypredrv_StatsTimerSetSeconds(s);

   hypredrv_StatsSetNumReps(s, 2);
   hypredrv_StatsSetNumLinearSystems(s, 1);

   hypredrv_StatsDestroy(&s);
   ASSERT_NULL(s);
}

static void
test_Stats_annotations_and_levels(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Basic annotate begin/end pairs */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "rhs");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "rhs");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "dofmap");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "dofmap");

   /* Simulate a preconditioner+solve phase with iter/norm */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "prec");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "prec");
   hypredrv_StatsIterSet(s, 7);
   hypredrv_StatsRelativeResNormSet(s, 1.0e-6);

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   /* Hypre init/finalize timer pair (EnsureCapacity + scalar timers) */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "initialize");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "initialize");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "finalize");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "finalize");

   /* Level annotations (nested) */
   hypredrv_StatsAnnotateLevelBegin(s, 0, "L0");
   hypredrv_StatsAnnotateLevelBegin(s, 1, "L1");
   hypredrv_StatsAnnotateLevelEnd(s, 1, "L1");
   hypredrv_StatsAnnotateLevelEnd(s, 0, "L0");

   /* Mismatched end (exercise defensive branches) */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelEnd(s, 2, "bad");
   hypredrv_ErrorCodeResetAll();

   /* Query last stats */
   (void)hypredrv_StatsGetLinearSystemID(s);
   (void)hypredrv_StatsGetLastIter(s);
   (void)hypredrv_StatsGetLastSetupTime(s);
   (void)hypredrv_StatsGetLastSolveTime(s);

   /* Level APIs */
   int c0 = hypredrv_StatsLevelGetCount(s, 0);
   ASSERT_TRUE(c0 >= 0);
   LevelEntry entry;
   (void)hypredrv_StatsLevelGetEntry(s, 0, 0, &entry);
   (void)hypredrv_StatsLevelGetEntry(s, 0, -1, &entry); /* invalid index */
   hypredrv_StatsLevelPrint(s, 0);

   /* Print (smoke) */
   hypredrv_StatsPrint(s, 0);

   hypredrv_StatsDestroy(&s);
}

static void
test_HYPREDRV_StatsLevelGetEntry_wrapper_branches(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Build one synthetic level entry spanning 2 solves */
   s->level_count[0]                  = 1;
   s->level_entries[0][0].id          = 7;
   s->level_entries[0][0].solve_start = 0;
   s->level_entries[0][0].solve_end   = 2;

   s->iters[0] = 3;
   s->iters[1] = 5;
   s->prec[0]  = 1.0;
   s->prec[1]  = 2.0;
   s->solve[0] = 4.0;
   s->solve[1] = 6.0;

   int    entry_id = 0, num_solves = 0, linear_iters = 0;
   double setup_time = 0.0, solve_time = 0.0;

   int ret = hypredrv_StatsLevelGetEntrySummary(s, 0, 0, &entry_id, &num_solves,
                                                &linear_iters, &setup_time, &solve_time);
   ASSERT_EQ(ret, 0);
   ASSERT_EQ(entry_id, 7);
   ASSERT_EQ(num_solves, 2);
   ASSERT_EQ(linear_iters, 8);
   ASSERT_EQ_DOUBLE(setup_time, 3.0, 1e-12);
   ASSERT_EQ_DOUBLE(solve_time, 10.0, 1e-12);

   /* Optional output pointer branches */
   ret = hypredrv_StatsLevelGetEntrySummary(s, 0, 0, NULL, NULL, NULL, NULL, NULL);
   ASSERT_EQ(ret, 0);

   /* ret != 0 branch */
   ret = hypredrv_StatsLevelGetEntrySummary(s, 0, -1, &entry_id, &num_solves,
                                            &linear_iters, &setup_time, &solve_time);
   ASSERT_NE(ret, 0);

   hypredrv_StatsDestroy(&s);
}

/*--------------------------------------------------------------------------
 * hypredrv_StatsAnnotateV with a real va_list (vsnprintf path)
 *--------------------------------------------------------------------------*/

static void
call_stats_annotate_v(Stats *stats, HYPREDRV_AnnotateAction action, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   hypredrv_StatsAnnotateV(stats, action, fmt, ap);
   va_end(ap);
}

static void
call_stats_annotate_v_null_stats(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   hypredrv_StatsAnnotateV(NULL, HYPREDRV_ANNOTATE_BEGIN, fmt, ap);
   va_end(ap);
}

static void
test_Stats_annotate_v_variadic_path(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_BEGIN, "%s", "matrix");
   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_END, "%s", "matrix");

   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_BEGIN, "%s", "reset_x0");
   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_END, "%s", "reset_x0");

   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_BEGIN, "%s", "solve");
   call_stats_annotate_v(s, HYPREDRV_ANNOTATE_END, "%s", "solve");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_annotation_system_alias_and_run_ignore(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* "system" mirrors "matrix" */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "system");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "system");

   /* Run* annotations are ignored (begin + end) */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "RunStep");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "RunStep");
   ASSERT_EQ(hypredrv_ErrorCodeGet() & ERROR_UNKNOWN_TIMING, 0);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_annotation_unknown_keys(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "not_a_timer");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_UNKNOWN_TIMING, 0);

   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "not_a_timer_either");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_UNKNOWN_TIMING, 0);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_rhs_dofmap_prec_counter_negative_branches(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* counter still -1: rhs/dofmap/prec branches set counter to 0 */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "rhs");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "rhs");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "dofmap");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "dofmap");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "prec");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "prec");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_reset_x0_and_solve_ls_counter_branches(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");

   /* First reset_x0: reps was 0 -> counter not incremented via reps>0 branch */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   /* Second repetition: reps>0 -> counter++ on reset_x0 begin */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_initial_res_norm_and_setters(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "prec");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "prec");
   hypredrv_StatsIterSet(s, 11);
   hypredrv_StatsInitialResNormSet(s, 3.0e-2);
   hypredrv_StatsRelativeResNormSet(s, 4.0e-4);
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_level_begin_end_error_branches(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Invalid level on begin */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelBegin(s, -1, "bad");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelBegin(s, STATS_MAX_LEVELS, "bad");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);

   /* Duplicate begin on same level */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelBegin(s, 0, "L0");
   hypredrv_StatsAnnotateLevelBegin(s, 0, "L0dup");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_StatsAnnotateLevelEnd(s, 0, "L0");

   /* End without begin (no-op) */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelEnd(s, 2, "ghost");
   ASSERT_EQ(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);

   /* Mismatched name */
   hypredrv_ErrorCodeResetAll();
   hypredrv_StatsAnnotateLevelBegin(s, 1, "L1");
   hypredrv_StatsAnnotateLevelEnd(s, 1, "wrong");
   ASSERT_NE(hypredrv_ErrorCodeGet() & ERROR_INVALID_VAL, 0);
   hypredrv_StatsAnnotateLevelEnd(s, 1, "L1");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_level_get_count_checked(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   int count = 0;

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_StatsLevelGetCountChecked(s, 0, NULL, "utest"), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_StatsLevelGetCountChecked(NULL, 0, &count, "utest"), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   ASSERT_NE(hypredrv_StatsLevelGetCountChecked(s, -1, &count, "utest"), ERROR_NONE);

   hypredrv_ErrorCodeResetAll();
   ASSERT_EQ(hypredrv_StatsLevelGetCountChecked(s, 0, &count, "utest"), ERROR_NONE);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_annotate_with_id_negative_id(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsAnnotateWithId(s, HYPREDRV_ANNOTATE_BEGIN, "matrix", -1);
   hypredrv_StatsAnnotateWithId(s, HYPREDRV_ANNOTATE_END, "matrix", -1);

   hypredrv_StatsAnnotateLevelWithId(s, HYPREDRV_ANNOTATE_BEGIN, 0, "L", -1);
   hypredrv_StatsAnnotateLevelWithId(s, HYPREDRV_ANNOTATE_END, 0, "L", -1);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_pending_timestep_and_level_path(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsSetPendingTimestepContext(s, 42);
   hypredrv_StatsAnnotateLevelBegin(s, 0, "L0");
   hypredrv_StatsAnnotateLevelBegin(s, 1, "L1");

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   hypredrv_StatsAnnotateLevelEnd(s, 1, "L1");
   hypredrv_StatsAnnotateLevelEnd(s, 0, "L0");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_pending_timestep_empty_path_early_return(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* No timestep / levels: first solve clears path -> no contextual column */
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_ensure_capacity_growth(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Initial REALLOC_EXPAND_FACTOR is 16: grow main timing arrays */
   for (int i = 0; i < 16; i++)
   {
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
   }

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_ensure_level_capacity_growth(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* After 64 stored entries at level 0, the 65th end triggers realloc path */
   for (int i = 0; i < 65; i++)
   {
      hypredrv_StatsAnnotateLevelBegin(s, 0, "L0");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
      hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");
      hypredrv_StatsAnnotateLevelEnd(s, 0, "L0");
   }

   ASSERT_TRUE(hypredrv_StatsLevelGetCount(s, 0) >= 64);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_print_impl_and_stream_branches(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Null stats / null stream / print_level 0 */
   hypredrv_StatsPrintToStream(NULL, 1, stdout);
   hypredrv_StatsPrintToStream(s, 1, NULL);
   hypredrv_StatsPrintToStream(s, 0, stdout);

   /* Three logical entries (0..2): middle entry has no solve -> aggregate loop skips i=1 */
   s->counter        = 2;
   s->matrix_counter = 0;
   s->ls_counter     = 1;
   s->iters[0]       = 2;
   s->iters[1]       = 0;
   s->iters[2]       = 4;
   s->prec[0]        = 0.1;
   s->prec[1]        = 0.0;
   s->prec[2]        = 0.2;
   s->solve[0]       = 0.3;
   s->solve[1]       = 0.0;
   s->solve[2]       = 0.4;
   s->r0norms[0]     = 1.0;
   s->r0norms[1]     = 0.0;
   s->r0norms[2]     = 2.0;
   s->rrnorms[0]     = 0.1;
   s->rrnorms[1]     = 0.0;
   s->rrnorms[2]     = 0.2;
   /* Entry 0: smaller LS build time than entry 2 (exercises aggregate min/max branches) */
   s->dofmap[0] = 0.01;
   s->rhs[0]    = 0.02;
   s->matrix[0] = 0.03;
   /* Row 1: no solve (skipped in table and aggregate inner loop) */
   s->dofmap[1] = s->rhs[1] = s->matrix[1] = 0.0;
   s->dofmap[2] = 0.5;
   s->rhs[2]    = 0.5;
   s->matrix[2] = 0.5;

   hypredrv_StatsPrintToStream(s, 1, stdout);
   hypredrv_StatsPrintToStream(s, 2, stdout);

   /* Object name on vs off */
   hypredrv_StatsSetObjectName(s, "objA");
   hypredrv_StatsPrint(s, 2);
   hypredrv_StatsSetObjectName(s, "");
   hypredrv_StatsPrint(s, 2);

   /* Contextual Path column: long path tail truncation (> STATS_ENTRY_WIDTH) */
   memset(s->entry_paths, 0, (size_t)s->capacity * STATS_PATH_LABEL_LENGTH);
   {
      char *slot0 = s->entry_paths;
      memset(slot0, 'a', (size_t)(STATS_PATH_LABEL_LENGTH - 1));
      slot0[STATS_PATH_LABEL_LENGTH - 1] = '\0';
   }
   hypredrv_StatsPrint(s, 1);

   hypredrv_StatsDestroy(&s);
}

static void
test_Stats_null_pointer_guards(void)
{
   hypredrv_ErrorCodeResetAll();

   hypredrv_StatsDestroy(NULL);

   Stats *dead = NULL;
   hypredrv_StatsDestroy(&dead);

   hypredrv_StatsAnnotate(NULL, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   call_stats_annotate_v_null_stats("%s", "matrix");

   hypredrv_StatsAnnotateLevelBegin(NULL, 0, "L");
   hypredrv_StatsAnnotateLevelEnd(NULL, 0, "L");

   hypredrv_StatsTimerSetMilliseconds(NULL);
   hypredrv_StatsTimerSetSeconds(NULL);
   hypredrv_StatsIterSet(NULL, 1);
   hypredrv_StatsInitialResNormSet(NULL, 1.0);
   hypredrv_StatsRelativeResNormSet(NULL, 1.0);
   hypredrv_StatsSetNumReps(NULL, 1);
   hypredrv_StatsSetNumLinearSystems(NULL, 1);
   hypredrv_StatsSetObjectName(NULL, "x");
   hypredrv_StatsSetPendingTimestepContext(NULL, 0);

   ASSERT_EQ(hypredrv_StatsGetLinearSystemID(NULL), -1);
   ASSERT_EQ(hypredrv_StatsGetLastIter(NULL), -1);
   ASSERT_EQ_DOUBLE(hypredrv_StatsGetLastSetupTime(NULL), 0.0, 0.0);
   ASSERT_EQ_DOUBLE(hypredrv_StatsGetLastSolveTime(NULL), 0.0, 0.0);

   ASSERT_EQ(hypredrv_StatsLevelGetCount(NULL, 0), 0);

   LevelEntry le;
   ASSERT_NE(hypredrv_StatsLevelGetEntry(NULL, 0, 0, &le), 0);
   {
      Stats *sx = hypredrv_StatsCreate();
      ASSERT_NOT_NULL(sx);
      ASSERT_NE(hypredrv_StatsLevelGetEntry(sx, 0, 0, NULL), 0);
      hypredrv_StatsDestroy(&sx);
   }

   int    eid = 0, ns = 0, li = 0;
   double st = 0.0, sv = 0.0;
   ASSERT_NE(hypredrv_StatsLevelGetEntrySummary(NULL, 0, 0, &eid, &ns, &li, &st, &sv), 0);

   hypredrv_StatsLevelPrint(NULL, 0);

   hypredrv_StatsPrint(NULL, 1);
}

static void
test_Stats_print_to_tempfile(void)
{
   Stats *s = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(s);

   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "reset_x0");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   FILE *tmp = tmpfile();
   ASSERT_NOT_NULL(tmp);
   hypredrv_StatsPrintToStream(s, 1, tmp);
   fclose(tmp);

   hypredrv_StatsDestroy(&s);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_Stats_basic_lifecycle_and_timers);
   RUN_TEST(test_Stats_annotations_and_levels);
   RUN_TEST(test_HYPREDRV_StatsLevelGetEntry_wrapper_branches);
   RUN_TEST(test_Stats_annotate_v_variadic_path);
   RUN_TEST(test_Stats_annotation_system_alias_and_run_ignore);
   RUN_TEST(test_Stats_annotation_unknown_keys);
   RUN_TEST(test_Stats_rhs_dofmap_prec_counter_negative_branches);
   RUN_TEST(test_Stats_reset_x0_and_solve_ls_counter_branches);
   RUN_TEST(test_Stats_initial_res_norm_and_setters);
   RUN_TEST(test_Stats_level_begin_end_error_branches);
   RUN_TEST(test_Stats_level_get_count_checked);
   RUN_TEST(test_Stats_annotate_with_id_negative_id);
   RUN_TEST(test_Stats_pending_timestep_and_level_path);
   RUN_TEST(test_Stats_pending_timestep_empty_path_early_return);
   RUN_TEST(test_Stats_ensure_capacity_growth);
   RUN_TEST(test_Stats_ensure_level_capacity_growth);
   RUN_TEST(test_Stats_print_impl_and_stream_branches);
   RUN_TEST(test_Stats_null_pointer_guards);
   RUN_TEST(test_Stats_print_to_tempfile);

   MPI_Finalize();
   return 0;
}
