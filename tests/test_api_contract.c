/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#include "HYPREDRV.h"
#include "_hypre_parcsr_mv.h"
#include "execution.h"
#include "internal/args.h"
#include "internal/yaml.h"
#include "object.h"
#include "test_ij_helpers.h"
#include "vector_tags.h"

static uint32_t
set_matrix(HYPREDRV_t obj)
{
   HYPRE_BigInt rows[] = {0, 1}, cols[] = {0};
   HYPRE_Real   values[] = {5.0};
   return HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, rows, cols, values);
}

static uint32_t
set_rhs(HYPREDRV_t obj)
{
   HYPRE_Real values[] = {7.0};
   return HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, values);
}

static uint32_t
set_generated_rhs(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetRHS(obj, NULL);
}
static uint32_t
set_guess(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetInitialGuess(obj, NULL);
}
static uint32_t
set_reference(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetReferenceSolution(
      obj, (HYPRE_Vector)create_test_ijvector_1x1(3.0));
}
static uint32_t
set_prec_matrix(HYPREDRV_t obj)
{
   return HYPREDRV_LinearSystemSetPrecMatrix(obj, NULL);
}
static uint32_t
set_nullspace(HYPREDRV_t obj)
{
   HYPRE_Complex value = 1.0;
   return HYPREDRV_LinearSystemSetNullSpace(obj, 1, 1, &value);
}
static uint32_t
set_near_nullspace(HYPREDRV_t obj)
{
   HYPRE_Complex value = 1.0;
   return HYPREDRV_LinearSystemSetNearNullSpace(obj, 1, 1, &value);
}

static void
test_independent_api_error_scopes(void)
{
   const struct
   {
      const char *name;
      uint32_t (*call)(HYPREDRV_t);
   } cases[] = {{"build", HYPREDRV_LinearSystemBuild},
                {"read matrix", HYPREDRV_LinearSystemReadMatrix},
                {"matrix from CSR", set_matrix},
                {"RHS", set_generated_rhs},
                {"RHS from array", set_rhs},
                {"initial guess", set_guess},
                {"reference", set_reference},
                {"preconditioner matrix", set_prec_matrix},
                {"nullspace", set_nullspace},
                {"near-nullspace", set_near_nullspace}};
   char prefix[1024], filename[2048];
   test_scratch_path(prefix, sizeof(prefix), "api-contract-matrix");
   snprintf(filename, sizeof(filename), "%s.00000", prefix);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPRE_IJMatrixPrint(matrix, prefix), 0);
   HYPRE_IJMatrixDestroy(matrix);
   HYPREDRV_t failed = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &failed), ERROR_NONE);
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      char yaml[PATH_MAX + 256];
      snprintf(
         yaml, sizeof(yaml),
         "general:\n  statistics: off\nlinear_system:\n  matrix_filename: %s\n"
         "  rhs_mode: ones\n  init_guess_mode: zeros\nsolver: pcg\npreconditioner: amg\n",
         prefix);
      char *argv[] = {yaml};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_EQ(set_matrix(obj), ERROR_NONE);
      ASSERT_EQ(set_rhs(obj), ERROR_NONE);
      ASSERT_EQ(set_guess(obj), ERROR_NONE);
      ASSERT_NE(HYPREDRV_InputArgsGetWarmup(failed, NULL), ERROR_NONE);
      fprintf(stderr, "API error-scope case: %s\n", cases[i].name);
      ASSERT_EQ(cases[i].call(obj), ERROR_NONE);
      ASSERT_NOT_NULL(obj->mat_A.handle);
      ASSERT_NOT_NULL(obj->vec_b.handle);
      ASSERT_NOT_NULL(obj->vec_x.handle);
      if (cases[i].call == set_matrix)
         ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(obj->mat_A.handle), 5.0, 1e-15);
      if (cases[i].call == set_rhs)
         ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_b.handle), 7.0, 1e-15);
      if (cases[i].call == set_reference) ASSERT_NOT_NULL(obj->vec_xref.handle);
      if (cases[i].call == set_prec_matrix)
         ASSERT_PTR_EQ(obj->mat_M.handle, obj->mat_A.handle);
      if (cases[i].call == set_nullspace) ASSERT_NOT_NULL(obj->vec_ns);
      if (cases[i].call == set_near_nullspace) ASSERT_NOT_NULL(obj->vec_nn);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
   ASSERT_EQ(HYPREDRV_Destroy(&failed), ERROR_NONE);
   unlink(filename);
}

static void
test_setup_creates_default_solution(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char  yaml[] = "solver: pcg\npreconditioner: amg\n";
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector rhs    = create_test_ijvector_1x1(4.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_NULL(obj->vec_x.handle);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSolverSetup(obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj->vec_x.handle);
   ASSERT_EQ(HYPREDRV_LinearSolverApply(obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 2.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   HYPRE_IJMatrixDestroy(matrix);
   HYPRE_IJVectorDestroy(rhs);
}

static void
test_yaml_input_contract(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char text[] =
      "---\r\nsolver:\r\n  pcg:\r\n    max_iter: 50\r\npreconditioner: amg\r\n...\r\n";
   char *argv[] = {text};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 50);
   char path[PATH_MAX];
   test_scratch_path(path, sizeof(path), "crlf-contract.yml");
   FILE *fp = fopen(path, "wb");
   ASSERT_NOT_NULL(fp);
   ASSERT_EQ_SIZE(fwrite(text, 1, strlen(text), fp), strlen(text));
   ASSERT_EQ(fclose(fp), 0);
   argv[0] = path;
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 50);
   unlink(path);
   char bad_indent[] = "solver:\n  pcg:\n      max_iter: 50\n    relative_tol: "
                       "1e-6\npreconditioner: amg\n";
   argv[0]           = bad_indent;
   ASSERT_NE(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   char duplicates[] =
      "solver:\n  pcg:\n    max_iter: 50\n    max_iter: 60\npreconditioner: amg\n";
   argv[0] = duplicates;
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(obj->iargs->solver.pcg.max_iter, 60);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_internal_call_status(void)
{
   HYPRE_BigInt   rowptr[] = {0, 1}, column[] = {0};
   HYPRE_Real     values[] = {2.0};
   HYPRE_IJMatrix matrix   = NULL;
   HYPRE_IJVector rhs      = NULL;
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_ApplyGlobalRuntimeSettings(NULL), ERROR_NONE);
   ASSERT_EQ(hypredrv_ApplyConfiguredDeviceInitialization(NULL), ERROR_NONE);
   ASSERT_EQ(hypredrv_LinearSystemBuildMatrixFromCSR(MPI_COMM_SELF, HYPRE_MEMORY_HOST, 0,
                                                     0, rowptr, column, values, &matrix),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_LinearSystemBuildRHSFromArray(MPI_COMM_SELF, HYPRE_MEMORY_HOST, 0,
                                                    0, values, &rhs),
             ERROR_NONE);
   Stats *stats = hypredrv_StatsCreate();
   ASSERT_NOT_NULL(stats);
   ASSERT_EQ(hypredrv_StatsAnnotateWithId(stats, HYPREDRV_ANNOTATE_BEGIN, "matrix", -1),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_StatsAnnotateWithId(stats, HYPREDRV_ANNOTATE_END, "matrix", -1),
             ERROR_NONE);
   ASSERT_EQ(
      hypredrv_StatsAnnotateLevelWithId(stats, HYPREDRV_ANNOTATE_BEGIN, 0, "level", -1),
      ERROR_NONE);
   ASSERT_EQ(
      hypredrv_StatsAnnotateLevelWithId(stats, HYPREDRV_ANNOTATE_END, 0, "level", -1),
      ERROR_NONE);
   int count = -1;
   ASSERT_EQ(hypredrv_StatsLevelGetCountChecked(stats, 0, &count, "test"), ERROR_NONE);
   ASSERT_EQ(count, 1);
   hypredrv_StatsDestroy(&stats);
   ASSERT_EQ(hypredrv_LinearSystemDumpScheduled(MPI_COMM_SELF, NULL, NULL, NULL, NULL,
                                                NULL, NULL, NULL, NULL, NULL, NULL),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemBuildRHSFromArray(MPI_COMM_SELF, HYPRE_MEMORY_HOST, 0,
                                                    0, NULL, &rhs),
             ERROR_INVALID_VAL);
   ASSERT_EQ(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP | ERROR_INVALID_VAL);
   HYPRE_IJMatrixDestroy(matrix);
   HYPRE_IJVectorDestroy(rhs);
   hypredrv_ErrorStateReset();
}

static void
test_attachment_aliases(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector coordinate = create_test_ijvector_1x1(0.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, NULL, (HYPRE_Vector)coordinate,
                                                 (HYPRE_Vector)coordinate),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   char *argv[] = {"solver: pcg\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(obj->mat_A.handle), 2.0, 1e-12);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)create_test_ijvector_1x1(4.0)),
      ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_x.handle),
      ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_x.handle),
      ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(obj->vec_x.handle), 0.0, 1e-12);
   HYPRE_IJVector reference = create_test_ijvector_1x1(3.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)reference),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)reference),
             ERROR_NONE);
   ASSERT_TRUE(obj->vec_xref.owned);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(reference), 3.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_optional_shared_tags(void)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0) && !defined(HYPRE_USING_GPU)
   const struct
   {
      solver_t solver;
      int      level;
      int      tagged;
   } cases[]      = {{SOLVER_GMRES, 0, 0},   {SOLVER_GMRES, 2, 0},  {SOLVER_GMRES, 3, 1},
                     {SOLVER_FGMRES, 1, 0},  {SOLVER_FGMRES, 2, 1}, {SOLVER_PCG, 3, 0},
                     {SOLVER_BICGSTAB, 3, 0}};
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char *argv[] = {"solver: gmres\npreconditioner: amg\n"};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJVector rhs = create_test_ijvector_1x1(4.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)rhs),
             ERROR_NONE);
   int label = 0;
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   void *par = NULL;
   HYPRE_IJVectorGetObject(rhs, &par);
   hypre_Vector *local_rhs = hypre_ParVectorLocalVector((hypre_ParVector *)par);
   HYPRE_IJVectorGetObject(obj->vec_x.handle, &par);
   hypre_Vector *local_x      = hypre_ParVectorLocalVector((hypre_ParVector *)par);
   HYPRE_Int     original_tag = 0;
   HYPRE_IJVectorSetTags(rhs, 0, 1, &original_tag);
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
   {
      obj->iargs->solver_method            = cases[i].solver;
      obj->iargs->solver.gmres.print_level = cases[i].level;
      if (cases[i].solver == SOLVER_FGMRES)
      {
         obj->iargs->solver.fgmres.print_level = cases[i].level;
      }
      ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
      if (cases[i].tagged)
      {
         ASSERT_NOT_NULL(obj->vector_tags);
         ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), hypre_VectorTags(local_x));
         ASSERT_TRUE(hypre_VectorTags(local_rhs) != &original_tag);
         ASSERT_EQ(hypre_VectorOwnsTags(local_rhs), 0);
         /* Repeated setup must restore and reattach without losing metadata. */
         ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
      }
      else
      {
         ASSERT_NULL(obj->vector_tags);
         ASSERT_NULL(hypre_VectorTags(local_x));
         ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
      }
   }
   obj->iargs->solver_method            = SOLVER_GMRES;
   obj->iargs->solver.gmres.print_level = 3;
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 1, &label), ERROR_NONE);
   ASSERT_NULL(obj->vector_tags);
   ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   obj->iargs->general.exec_policy = 1;
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   ASSERT_NULL(obj->vector_tags);
   obj->iargs->general.exec_policy = 0;
   int mismatched_labels[]         = {0, 1};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, mismatched_labels), ERROR_NONE);
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_INVALID_VAL);
   ASSERT_NULL(obj->vector_tags);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_PTR_EQ(hypre_VectorTags(local_rhs), &original_tag);
   ASSERT_EQ(hypre_VectorOwnsTags(local_rhs), 0);
   HYPRE_IJVectorDestroy(rhs);
#endif
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   RUN_TEST(test_independent_api_error_scopes);
   RUN_TEST(test_setup_creates_default_solution);
   RUN_TEST(test_yaml_input_contract);
   RUN_TEST(test_optional_shared_tags);
   RUN_TEST(test_internal_call_status);
   RUN_TEST(test_attachment_aliases);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
