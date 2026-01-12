#include <mpi.h>
#include <stdio.h>

#include "HYPREDRV.h"
#include "error.h"
#include "stats.h"
#include "test_helpers.h"

static void
test_Stats_basic_lifecycle_and_timers(void)
{
   Stats *s = StatsCreate();
   ASSERT_NOT_NULL(s);
   StatsSetContext(s);
   ASSERT_TRUE(StatsGetContext() == s);

   StatsTimerSetMilliseconds();
   StatsTimerSetSeconds();

   StatsSetNumReps(2);
   StatsSetNumLinearSystems(1);

   StatsDestroy(&s);
   ASSERT_NULL(s);
   StatsSetContext(NULL);
   ASSERT_NULL(StatsGetContext());
}

static void
test_Stats_annotations_and_levels(void)
{
   Stats *s = StatsCreate();
   ASSERT_NOT_NULL(s);
   StatsSetContext(s);

   /* Basic annotate begin/end pairs */
   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "matrix");
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "matrix");

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "rhs");
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "rhs");

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "dofmap");
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "dofmap");

   /* Simulate a preconditioner+solve phase with iter/norm */
   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "prec");
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "prec");
   StatsIterSet(7);
   StatsRelativeResNormSet(1.0e-6);

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "solve");
   StatsAnnotate(HYPREDRV_ANNOTATE_END, "solve");

   /* Level annotations (nested) */
   StatsAnnotateLevelBegin(0, "L0");
   StatsAnnotateLevelBegin(1, "L1");
   StatsAnnotateLevelEnd(1, "L1");
   StatsAnnotateLevelEnd(0, "L0");

   /* Mismatched end (exercise defensive branches) */
   ErrorCodeResetAll();
   StatsAnnotateLevelEnd(2, "bad");
   ErrorCodeResetAll();

   /* Query last stats */
   (void)StatsGetLinearSystemID();
   (void)StatsGetLastIter();
   (void)StatsGetLastSetupTime();
   (void)StatsGetLastSolveTime();

   /* Level APIs */
   int c0 = StatsLevelGetCount(0);
   ASSERT_TRUE(c0 >= 0);
   LevelEntry entry;
   (void)StatsLevelGetEntry(0, 0, &entry);
   (void)StatsLevelGetEntry(0, -1, &entry); /* invalid index */
   StatsLevelPrint(0);

   /* Print (smoke) */
   StatsPrint(0);

   StatsDestroy(&s);
   StatsSetContext(NULL);
}

static void
test_HYPREDRV_StatsLevelGetEntry_wrapper_branches(void)
{
   /* This test targets the wrapper logic in src/HYPREDRV.c (aggregate computation),
    * without requiring a full HYPREDRV object. */
   Stats *s = StatsCreate();
   ASSERT_NOT_NULL(s);
   StatsSetContext(s);

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

   int ret = HYPREDRV_StatsLevelGetEntry(0, 0, &entry_id, &num_solves, &linear_iters,
                                        &setup_time, &solve_time);
   ASSERT_EQ(ret, 0);
   ASSERT_EQ(entry_id, 7);
   ASSERT_EQ(num_solves, 2);
   ASSERT_EQ(linear_iters, 8);
   ASSERT_EQ_DOUBLE(setup_time, 3.0, 1e-12);
   ASSERT_EQ_DOUBLE(solve_time, 10.0, 1e-12);

   /* Optional output pointer branches */
   ret = HYPREDRV_StatsLevelGetEntry(0, 0, NULL, NULL, NULL, NULL, NULL);
   ASSERT_EQ(ret, 0);

   /* ret != 0 branch */
   ret = HYPREDRV_StatsLevelGetEntry(0, -1, &entry_id, &num_solves, &linear_iters,
                                     &setup_time, &solve_time);
   ASSERT_NE(ret, 0);

   /* When Stats context is NULL, StatsLevelGetEntry returns -1 (defensive). */
   StatsSetContext(NULL);
   ret = HYPREDRV_StatsLevelGetEntry(0, 0, &entry_id, &num_solves, &linear_iters,
                                     &setup_time, &solve_time);
   ASSERT_NE(ret, 0);

   StatsDestroy(&s);
   StatsSetContext(NULL);
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

