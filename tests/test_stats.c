#include <mpi.h>
#include <stdio.h>

#include "HYPREDRV.h"
#include "args.h"
#include "error.h"
#include "linsys.h"
#include "stats.h"
#include "test_helpers.h"

struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   int      nstates;
   int     *states;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix mat_A;
   HYPRE_IJMatrix mat_M;
   HYPRE_IJVector vec_b;
   HYPRE_IJVector vec_x;
   HYPRE_IJVector vec_x0;
   HYPRE_IJVector vec_xref;
   HYPRE_IJVector vec_nn;
   HYPRE_IJVector *vec_s;

   HYPRE_Precon precon;
   HYPRE_Solver solver;

   IntArray *precon_reuse_timestep_starts;
   void *scaling_ctx;

   Stats *stats;
};

static void
test_Stats_basic_lifecycle_and_timers(void)
{
   Stats *s = StatsCreate();
   ASSERT_NOT_NULL(s);

   StatsTimerSetMilliseconds(s);
   StatsTimerSetSeconds(s);

   StatsSetNumReps(s, 2);
   StatsSetNumLinearSystems(s, 1);

   StatsDestroy(&s);
   ASSERT_NULL(s);
}

static void
test_Stats_annotations_and_levels(void)
{
   Stats *s = StatsCreate();
   ASSERT_NOT_NULL(s);

   /* Basic annotate begin/end pairs */
   StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "matrix");

   StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "rhs");
   StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "rhs");

   StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "dofmap");
   StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "dofmap");

   /* Simulate a preconditioner+solve phase with iter/norm */
   StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "prec");
   StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "prec");
   StatsIterSet(s, 7);
   StatsRelativeResNormSet(s, 1.0e-6);

   StatsAnnotate(s, HYPREDRV_ANNOTATE_BEGIN, "solve");
   StatsAnnotate(s, HYPREDRV_ANNOTATE_END, "solve");

   /* Level annotations (nested) */
   StatsAnnotateLevelBegin(s, 0, "L0");
   StatsAnnotateLevelBegin(s, 1, "L1");
   StatsAnnotateLevelEnd(s, 1, "L1");
   StatsAnnotateLevelEnd(s, 0, "L0");

   /* Mismatched end (exercise defensive branches) */
   ErrorCodeResetAll();
   StatsAnnotateLevelEnd(s, 2, "bad");
   ErrorCodeResetAll();

   /* Query last stats */
   (void)StatsGetLinearSystemID(s);
   (void)StatsGetLastIter(s);
   (void)StatsGetLastSetupTime(s);
   (void)StatsGetLastSolveTime(s);

   /* Level APIs */
   int c0 = StatsLevelGetCount(s, 0);
   ASSERT_TRUE(c0 >= 0);
   LevelEntry entry;
   (void)StatsLevelGetEntry(s, 0, 0, &entry);
   (void)StatsLevelGetEntry(s, 0, -1, &entry); /* invalid index */
   StatsLevelPrint(s, 0);

   /* Print (smoke) */
   StatsPrint(s, 0);

   StatsDestroy(&s);
}

static void
test_HYPREDRV_StatsLevelGetEntry_wrapper_branches(void)
{
   /* This test targets the wrapper logic in src/HYPREDRV.c (aggregate computation),
    * without requiring a full HYPREDRV object. */
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;
   Stats                  *s     = state->stats;
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

   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
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
