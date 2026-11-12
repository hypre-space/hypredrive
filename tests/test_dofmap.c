/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "object.h"
#include "test_helpers.h"
#include "test_solution_helpers.h"

static HYPREDRV_t
create_object(MPI_Comm comm, const char *precon, int device)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(comm, &obj), ERROR_NONE);
   char yaml[512];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\n  exec_policy: %s\n"
            "solver: fgmres\npreconditioner:\n%s\n",
            device ? "device" : "host", precon);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   /* Legacy MGR uses the global selector. In hypre 2.20, Jacobi type 7
    * dereferences a missing diagonal buffer; use forward GS for this fixture. */
   if (obj->iargs->precon_method == PRECON_MGR) obj->iargs->precon.mgr.relax_type = 3;
#endif
   int rank;
   MPI_Comm_rank(comm, &rank);
   HYPRE_BigInt first     = 2 * rank;
   HYPRE_BigInt offsets[] = {0, 1, 2}, columns[] = {first, first + 1};
   HYPRE_Real   values[] = {2.0, 2.0}, rhs[] = {4.0, 4.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, first, first + 1, offsets,
                                                   columns, values),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, first, first + 1, rhs),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 1, 2), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   return obj;
}

static void
test_changed_labels_invalidate_consumers(void)
{
   const char *methods[]   = {"  mgr:\n    level:\n      0:\n        f_dofs: [0]\n"
                                "    coarsest_level: amg",
                              "  amg:\n    coarsening:\n      num_functions: 2",
                              "  amg: {}"};
   const int   locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++)
      {
         HYPREDRV_t   obj    = create_object(MPI_COMM_SELF, methods[m], locations[p]);
         HYPRE_Precon precon = obj->precon;
         hypredrv_Solver *solver = obj->solver;
         ASSERT_TRUE(obj->precon_is_setup);
         ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, obj->dofmap->data), ERROR_NONE);
         ASSERT_PTR_EQ(obj->precon, precon);
         ASSERT_PTR_EQ(obj->solver, solver);
         int changed[] = {1, 0};
         ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, changed), ERROR_NONE);
         if (m < 2)
         {
            ASSERT_NULL(obj->precon);
            ASSERT_NULL(obj->solver);
            ASSERT_FALSE(obj->precon_is_setup);
            ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
            ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
         }
         else
         {
            ASSERT_PTR_EQ(obj->precon, precon);
            ASSERT_PTR_EQ(obj->solver, solver);
         }
         ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
         HYPRE_Complex *solution = NULL;
         ASSERT_EQ(test_copy_solution_values(obj, &solution), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)solution[0], 2.0, 1.e-10);
         ASSERT_EQ_DOUBLE((double)solution[1], 2.0, 1.e-10);
         ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      }
   }
}

static void
test_rejected_inputs_keep_map_and_consumers(void)
{
   HYPREDRV_t obj = create_object(
      MPI_COMM_SELF,
      "  mgr:\n    level:\n      0:\n        f_dofs: [0]\n    coarsest_level: amg", 0);
   IntArray    *map        = obj->dofmap;
   HYPRE_Precon precon     = obj->precon;
   hypredrv_Solver *solver     = obj->solver;
   uint64_t     generation = obj->iargs->precon.mgr.rbm_input_generation;
   ASSERT_TRUE(HYPREDRV_LinearSystemSetDofmap(obj, 1, NULL) & ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetDofmap(obj, -1, map->data) & ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, INT_MAX, 2) &
               ERROR_OUT_OF_BOUNDS);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetContiguousDofmap(obj, -1, 2) & ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 2, -1) & ERROR_INVALID_VAL);
   hypredrv_ErrorStateReset();
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   hypredrv_ErrorMsgAdd("Earlier map diagnostic");
   ASSERT_EQ_U32(hypredrv_LinearSystemSetDofmap(obj, DOFMAP_FROM_ARRAY, 2, 0, map->data),
                 ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_PTR_EQ(obj->dofmap, map);
   ASSERT_PTR_EQ(obj->iargs->precon.mgr.dofmap, map);
   ASSERT_EQ(obj->iargs->precon.mgr.rbm_input_generation, generation);
   ASSERT_PTR_EQ(obj->precon, precon);
   ASSERT_PTR_EQ(obj->solver, solver);
   ASSERT_TRUE(obj->precon_is_setup);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_file_replacement_and_recovery(void)
{
   HYPREDRV_t obj = create_object(
      MPI_COMM_SELF,
      "  mgr:\n    level:\n      0:\n        f_dofs: [0]\n    coarsest_level: amg", 0);
   IntArray    *map    = obj->dofmap;
   HYPRE_Precon precon = obj->precon;
   hypredrv_Solver *solver = obj->solver;
   char         prefix[MAX_FILENAME_LENGTH], filename[MAX_FILENAME_LENGTH + 16];
   test_scratch_path(prefix, sizeof(prefix), "dofmap");
   snprintf(obj->iargs->ls.dofmap_filename, sizeof(obj->iargs->ls.dofmap_filename), "%s",
            prefix);
   snprintf(filename, sizeof(filename), "%s.00000", prefix);
   FILE *fp = fopen(filename, "w");
   ASSERT_NOT_NULL(fp);
   ASSERT_GT(fprintf(fp, "2\n1\n"), 0);
   ASSERT_EQ(fclose(fp), 0);
   ASSERT_TRUE(HYPREDRV_LinearSystemReadDofmap(obj) & ERROR_FILE_UNEXPECTED_ENTRY);
   ASSERT_PTR_EQ(obj->dofmap, map);
   ASSERT_PTR_EQ(obj->precon, precon);
   ASSERT_PTR_EQ(obj->solver, solver);
   fp = fopen(filename, "w");
   ASSERT_NOT_NULL(fp);
   ASSERT_GT(fprintf(fp, "2\n1\n0\n"), 0);
   ASSERT_EQ(fclose(fp), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);
   ASSERT_EQ_SIZE(obj->dofmap->size, 2);
   ASSERT_EQ(obj->dofmap->data[0], 1);
   ASSERT_EQ(obj->dofmap->data[1], 0);
   ASSERT_PTR_EQ(obj->iargs->precon.mgr.dofmap, obj->dofmap);
   ASSERT_NULL(obj->precon);
   ASSERT_NULL(obj->solver);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   precon = obj->precon;
   solver = obj->solver;
   /* Identical labels read from binary input also retain native consumers. */
   test_scratch_path(prefix, sizeof(prefix), "dofmap-binary");
   snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
   fp = fopen(filename, "wb");
   ASSERT_NOT_NULL(fp);
   size_t count    = 2;
   int    labels[] = {1, 0};
   ASSERT_EQ_SIZE(fwrite(&count, sizeof(count), 1, fp), 1);
   ASSERT_EQ_SIZE(fwrite(labels, sizeof(int), 2, fp), 2);
   ASSERT_EQ(fclose(fp), 0);
   snprintf(obj->iargs->ls.dofmap_filename, sizeof(obj->iargs->ls.dofmap_filename), "%s",
            prefix);
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);
   ASSERT_PTR_EQ(obj->precon, precon);
   ASSERT_PTR_EQ(obj->solver, solver);
   obj->iargs->ls.dofmap_filename[0] = '\0';
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);
   ASSERT_EQ_SIZE(obj->dofmap->size, 0);
   ASSERT_NULL(obj->precon);
   ASSERT_NULL(obj->solver);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_distributed_construction_and_failures(void)
{
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size == 1) return;
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   int labels[] = {2, 4, 2, 4};
   int count    = rank ? 4 : 0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, count, rank ? labels : NULL),
             ERROR_NONE);
   ASSERT_EQ_SIZE(obj->dofmap->size, (size_t)count);
   ASSERT_EQ_SIZE(obj->dofmap->g_unique_size, 2);
   ASSERT_EQ(obj->dofmap->g_unique_data[0], 2);
   ASSERT_EQ(obj->dofmap->g_unique_data[1], 4);
   ASSERT_TRUE(HYPREDRV_LinearSystemReadDofmap(obj) & ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, count, obj->dofmap->data), ERROR_NONE);
   IntArray *saved = obj->dofmap;
   uint32_t  code  = rank ? HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 2, 2)
                          : HYPREDRV_LinearSystemSetContiguousDofmap(obj, 0, 2);
   ASSERT_TRUE(code & ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetDofmap(obj, rank ? -1 : 0, labels) &
               ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, rank ? INT_MAX : 0, 2) &
               ERROR_OUT_OF_BOUNDS);
   ASSERT_PTR_EQ(obj->dofmap, saved);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, rank ? 2 : 0, 2), ERROR_NONE);
   if (rank) ASSERT_EQ(obj->dofmap->data[1], 1);
   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, rank ? 2 : 0, 2), ERROR_NONE);
   if (rank) ASSERT_EQ(obj->dofmap->data[1], 0);
   char *argv[] = {"general:\n  statistics: off\n  exec_policy: host\n"
                   "solver: gmres\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   input_args *saved_inputs = obj->iargs;
   if (!rank) obj->iargs = NULL;
   ASSERT_TRUE(HYPREDRV_LinearSystemReadDofmap(obj) & ERROR_INVALID_VAL);
   obj->iargs                       = saved_inputs;
   char prefix[MAX_FILENAME_LENGTH] = {0};
   if (!rank) test_scratch_path(prefix, sizeof(prefix), "world-dofmap");
   MPI_Bcast(prefix, sizeof(prefix), MPI_CHAR, 0, MPI_COMM_WORLD);
   ASSERT_EQ(HYPREDRV_LinearSystemPrintDofmap(obj, prefix), ERROR_NONE);
   snprintf(obj->iargs->ls.dofmap_filename, sizeof(obj->iargs->ls.dofmap_filename), "%s",
            prefix);
   saved = obj->dofmap;
   if (!rank) obj->iargs->ls.dofmap_filename[0] = '\0';
   ASSERT_TRUE(HYPREDRV_LinearSystemReadDofmap(obj) & ERROR_INVALID_VAL);
   snprintf(obj->iargs->ls.dofmap_filename, sizeof(obj->iargs->ls.dofmap_filename), "%s",
            prefix);
   if (!rank)
   {
      memset(obj->iargs->ls.dirname, 'x', sizeof(obj->iargs->ls.dirname) - 1);
      obj->iargs->ls.dirname[sizeof(obj->iargs->ls.dirname) - 1] = '\0';
   }
   ASSERT_TRUE(HYPREDRV_LinearSystemReadDofmap(obj) & ERROR_INVALID_VAL);
   obj->iargs->ls.dirname[0] = '\0';
   ASSERT_PTR_EQ(obj->dofmap, saved);
   hypredrv_ErrorStateReset();
   if (rank) hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_LinearSystemSetDofmap(obj, DOFMAP_FROM_INPUT, 0, 0, NULL),
                 ERROR_MISSING_DOFMAP);
   ASSERT_PTR_EQ(obj->dofmap, saved);
   ASSERT_EQ(HYPREDRV_LinearSystemReadDofmap(obj), ERROR_NONE);
   ASSERT_EQ_SIZE(obj->dofmap->size, (size_t)count);
   ASSERT_EQ_SIZE(obj->dofmap->g_unique_size, 2);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_distributed_label_changes_invalidate_every_consumer(void)
{
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size == 1) return;
   const char *methods[] = {
      "  mgr:\n    level:\n      0:\n        f_dofs: [0]\n    coarsest_level: amg",
      "  amg:\n    coarsening:\n      num_functions: 2"};
   for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++)
   {
      HYPREDRV_t   obj    = create_object(MPI_COMM_WORLD, methods[m], 0);
      HYPRE_Precon precon = obj->precon;
      hypredrv_Solver *solver = obj->solver;
      IntArray    *map    = obj->dofmap;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetContiguousDofmap(obj, rank ? -1 : 1, 2) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->dofmap, map);
      ASSERT_PTR_EQ(obj->precon, precon);
      ASSERT_PTR_EQ(obj->solver, solver);
      int labels[] = {1, 0};
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, rank ? map->data : labels),
                ERROR_NONE);
      ASSERT_NULL(obj->precon);
      ASSERT_NULL(obj->solver);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
      HYPRE_Complex *data = NULL;
      ASSERT_EQ(test_copy_solution_values(obj, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 2.0, 1.e-10);
      ASSERT_EQ_DOUBLE((double)data[1], 2.0, 1.e-10);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_changed_labels_invalidate_consumers);
   RUN_TEST(test_rejected_inputs_keep_map_and_consumers);
   RUN_TEST(test_file_replacement_and_recovery);
   RUN_TEST(test_distributed_construction_and_failures);
   RUN_TEST(test_distributed_label_changes_invalidate_every_consumer);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
