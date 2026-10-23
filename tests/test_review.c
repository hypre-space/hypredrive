/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#include <mpi.h>
#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/error.h"
#include "internal/krylov.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "object.h"
#include "test_helpers.h"
#include "test_ij_helpers.h"

static input_args *
parse(const char *yaml)
{
   input_args *args   = NULL;
   char       *argv[] = {(char *)yaml};
   hypredrv_ErrorStateReset();
   hypredrv_InputArgsParse(MPI_COMM_SELF, true, 1, argv, &args);
   return args;
}

static void
test_aggressive_integer_and_preset_retry(void)
{
   input_args *args = parse("preconditioner:\n  amg:\n    aggressive:\n      "
                            "P12_max_elements: 4\n      P12_trunc_factor: 0.25\n");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon.amg.aggressive.P12_max_elements, 4);
   ASSERT_EQ_DOUBLE(args->precon.amg.aggressive.P12_trunc_factor, 0.25, 1e-15);
   hypredrv_InputArgsDestroy(&args);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NE(HYPREDRV_InputArgsSetPreconPreset(obj, "does-not-exist"), ERROR_NONE);
   ASSERT_NULL(obj->iargs->precon_methods);
   ASSERT_NULL(obj->iargs->precon_variants);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NE(HYPREDRV_InputArgsSetPreconPreset(obj, "does-not-exist"), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_invalid_preset_keeps_live_preconditioner(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char yaml[] = "general:\n  statistics: off\nlinear_system:\n  init_guess_mode: zeros\n"
                 "solver:\n  pcg:\n    max_iter: 5\n"
                 "preconditioner:\n  amg:\n    print_level: 0\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_BigInt rows[] = {0, 1}, cols[] = {0};
   HYPRE_Real   values[] = {2.0}, rhs[] = {4.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, rows, cols, values),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);

   /* A typo is rejected before the set-up preconditioner is discarded. */
   ASSERT_NE(HYPREDRV_InputArgsSetPreconPreset(obj, "poison"), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_NOT_NULL(obj->precon);
   ASSERT_TRUE(obj->precon_is_setup);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_setters_before_parse(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NE(HYPREDRV_LinearSystemSetRHS(obj, NULL), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_NE(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_NE(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
   hypredrv_ErrorStateReset();
   HYPRE_BigInt rows[] = {0, 1}, cols[] = {0};
   HYPRE_Real   values[] = {2.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, rows, cols, values),
             ERROR_NONE);
   HYPRE_IJVector rhs   = create_test_ijvector_1x1(1.0);
   HYPRE_IJVector guess = create_test_ijvector_1x1(0.5);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)guess), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_NOT_NULL(obj->vec_x);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_partial_attachments(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector y = create_test_ijvector_1x1(1.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, NULL, (HYPRE_Vector)y, NULL),
             ERROR_NONE);
   ASSERT_TRUE(obj->owns_vec_coord);
   ASSERT_NE(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_InputArgsSetPreconPreset(obj, "poisson"), ERROR_NONE);
   HYPRE_IJMatrix M = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)M), ERROR_NONE);
   ASSERT_NE(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   hypredrv_ErrorStateReset();
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_schedule_allocation_bound(void)
{
   char filename[] = "/tmp/hypredrive-review-schedule-XXXXXX";
   int  fd         = mkstemp(filename);
   ASSERT_TRUE(fd >= 0);
   FILE *fp = fdopen(fd, "w");
   ASSERT_NOT_NULL(fp);
   fprintf(fp, "%u\n", LSSEQ_MAX_TIMESTEPS + 1);
   fclose(fp);
   IntArray *ids = NULL, *starts = NULL;
   ASSERT_NE(hypredrv_LinearSystemLoadTimestepSchedule(filename, &ids, &starts),
             ERROR_NONE);
   ASSERT_NULL(ids);
   ASSERT_NULL(starts);
   unlink(filename);
   hypredrv_ErrorStateReset();
}

static void
test_mgr_repeated_nested_and_dof_bounds(void)
{
   input_args *args = parse(
      "preconditioner:\n  mgr:\n    coarsest_level:\n      gmres:\n        max_iter: 1\n "
      "       preconditioner:\n          mgr:\n            coarsest_level:\n             "
      " gmres:\n                max_iter: 1\n                preconditioner: amg\n      "
      "fgmres:\n        max_iter: 1\n        preconditioner: amg\n");
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(args);
   ASSERT_EQ(args->precon.mgr.coarsest_level.krylov->solver_method, SOLVER_FGMRES);
   hypredrv_InputArgsDestroy(&args);
   args = parse("preconditioner:\n  mgr:\n    level:\n      0:\n        f_dofs: "
                "[999999999999999999999999999999999]\n");
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_InputArgsDestroy(&args);
   char yaml[4096] = "preconditioner:\n  mgr:\n    level:\n      0:\n        f_dofs: [";
   for (int i = 0; i <= MAX_STACK_ARRAY_LENGTH; i++)
   {
      strcat(yaml, i ? ",0" : "0");
   }
   strcat(yaml, "]\n");
   args = parse(yaml);
   ASSERT_TRUE(hypredrv_ErrorCodeActive());
   hypredrv_InputArgsDestroy(&args);
   hypredrv_ErrorStateReset();
}

static void
test_preconditioner_matrix_file_formats_and_read_default(void)
{
   char ascii_path[4096], binary_path[4096];
   snprintf(ascii_path, sizeof(ascii_path), "%s/data/ps3d10pt7/np1/IJ.out.A.00000",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(binary_path, sizeof(binary_path),
            "%s/data/poromech2k/np1/ls_00000/IJ.out.A.00000.bin", HYPREDRIVE_SOURCE_DIR);
   if (access(ascii_path, F_OK) || access(binary_path, F_OK))
   {
      fprintf(stderr, "SKIP: optional matrix datasets are unavailable\n");
      return;
   }
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char yaml[4096];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\nlinear_system:\n  matrix_filename: "
            "%s/data/ps3d10pt7/np1/IJ.out.A\n"
            "solver:\n  pcg:\n    max_iter: 5\n"
            "preconditioner:\n  amg:\n    print_level: 0\n",
            HYPREDRIVE_SOURCE_DIR);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemReadMatrix(obj), ERROR_NONE);
   ASSERT_PTR_EQ(obj->mat_M, obj->mat_A);
   HYPRE_IJMatrix separate = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)separate), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemReadMatrix(obj), ERROR_NONE);
   ASSERT_PTR_EQ(obj->mat_M, obj->mat_A);
   ASSERT_FALSE(obj->owns_mat_M);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);

   LS_args args;
   hypredrv_LinearSystemSetDefaultArgs(&args);
   snprintf(args.precmat_filename, sizeof(args.precmat_filename),
            "%s/data/poromech2k/np1/ls_00000/IJ.out.A", HYPREDRIVE_SOURCE_DIR);
   HYPRE_IJMatrix main = create_test_ijmatrix_1x1(1.0), M = NULL;
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, main, &M, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(M);
   ASSERT_TRUE(hypredrv_LinearSystemMatrixGetNumRows(M) > 1);
   HYPRE_IJMatrixDestroy(M);
   M = NULL;
   /* Types without a reader of their own still read ASCII IJ. */
   args.type = 2;
   snprintf(args.precmat_filename, sizeof(args.precmat_filename),
            "%s/data/ps3d10pt7/np1/IJ.out.A", HYPREDRIVE_SOURCE_DIR);
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, main, &M, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(M);
   ASSERT_TRUE(hypredrv_LinearSystemMatrixGetNumRows(M) > 1);
   HYPRE_IJMatrixDestroy(M);
   M = NULL;
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   /* A generated one-entry matrix keeps this format regression independent
    * of any optional Matrix Market dataset. */
   char filename[] = "/tmp/hypredrive-review-matrix-XXXXXX";
   int  fd         = mkstemp(filename);
   ASSERT_TRUE(fd >= 0);
   FILE *fp = fdopen(fd, "w");
   ASSERT_NOT_NULL(fp);
   fputs("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 3.0\n", fp);
   fclose(fp);
   args.type = 3;
   snprintf(args.precmat_filename, sizeof(args.precmat_filename), "%s", filename);
   hypredrv_LinearSystemSetPrecMatrix(MPI_COMM_SELF, &args, main, &M, NULL);
   ASSERT_FALSE(hypredrv_ErrorCodeActive());
   ASSERT_NOT_NULL(M);
   ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(M), 3.0, 1e-15);
   HYPRE_IJMatrixDestroy(M);
   unlink(filename);
#endif
   HYPRE_IJMatrixDestroy(main);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_aggressive_integer_and_preset_retry);
   RUN_TEST(test_invalid_preset_keeps_live_preconditioner);
   RUN_TEST(test_setters_before_parse);
   RUN_TEST(test_partial_attachments);
   RUN_TEST(test_schedule_allocation_bound);
   RUN_TEST(test_preconditioner_matrix_file_formats_and_read_default);
   RUN_TEST(test_mgr_repeated_nested_and_dof_bounds);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
