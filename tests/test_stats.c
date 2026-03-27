#include <mpi.h>
#include <stdio.h>

#include "HYPREDRV.h"
#include "args.h"
#include "error.h"
#include "linsys.h"
#include "stats.h"
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
   s->level_count[0]              = 1;
   s->level_entries[0][0].id      = 7;
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

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_Stats_basic_lifecycle_and_timers);
   RUN_TEST(test_Stats_annotations_and_levels);
   RUN_TEST(test_HYPREDRV_StatsLevelGetEntry_wrapper_branches);

   MPI_Finalize();
   return 0;
}
