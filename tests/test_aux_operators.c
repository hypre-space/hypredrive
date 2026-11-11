/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "_hypre_utilities.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "object.h"
#include "test_helpers.h"
#include "test_ij_helpers.h"

static HYPREDRV_t
create_operator_object(MPI_Comm comm, const char *method, int device)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(comm, &obj), ERROR_NONE);
   char yaml[256];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\n  exec_policy: %s\n"
            "solver: gmres\npreconditioner: %s\n",
            device ? "device" : "host",
            !strcmp(method, "mgr") ? "\n  mgr:\n    coarsest_level: amg" : method);
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   /* Include an empty root in the distributed registration tests. */
   int          count = size == 1 || rank != 0;
   HYPRE_BigInt first = size == 1 ? 0 : (rank ? rank - 1 : 0);
   if (!strcmp(method, "mgr"))
   {
      int labels[] = {0};
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, count, labels), ERROR_NONE);
   }
   HYPRE_IJMatrix operators[3] = {NULL};
   for (int i = 0; i < 3; i++)
   {
      ASSERT_EQ(HYPRE_IJMatrixCreate(comm, first, first + count - 1, first,
                                     first + count - 1, &operators[i]),
                0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(operators[i], HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJMatrixInitialize_v2(operators[i], HYPRE_MEMORY_HOST), 0);
      if (count)
      {
         HYPRE_Int  one   = 1;
         HYPRE_Real value = i == 2 ? 2.0 : 0.0;
         ASSERT_EQ(HYPRE_IJMatrixSetValues(operators[i], 1, &one, &first, &first, &value),
                   0);
      }
      ASSERT_EQ(HYPRE_IJMatrixAssemble(operators[i]), 0);
   }
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)operators[2]), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(obj, (HYPRE_Matrix)operators[0]),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteCurl(obj, (HYPRE_Matrix)operators[1]),
             ERROR_NONE);
   HYPRE_IJVector coordinates = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(comm, first, first + count - 1, &coordinates), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(coordinates, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize_v2(coordinates, HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(coordinates), 0);
   void *object = NULL;
   ASSERT_EQ(HYPRE_IJVectorGetObject(coordinates, &object), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)object, 1.0), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)coordinates), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)coordinates,
                                                 (HYPRE_Vector)coordinates,
                                                 (HYPRE_Vector)coordinates),
             ERROR_NONE);
   return obj;
}

static void
test_changed_operators_invalidate_consumers(void)
{
   const char *methods[] = {"ams", "ads", "amg", "mgr"};
   for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++)
   {
      for (int which = 0; which < 4; which++)
      {
         HYPREDRV_t obj = create_operator_object(MPI_COMM_SELF, methods[m], 0);
         ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
         ASSERT_NOT_NULL(obj->precon);
         ASSERT_NOT_NULL(obj->solver);
         HYPRE_Precon precon = obj->precon;
         hypredrv_Solver *solver = obj->solver;
         /* Unchanged registrations keep live consumers. */
         ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(
                      obj, (HYPRE_Matrix)obj->mat_G.handle),
                   ERROR_NONE);
         ASSERT_EQ(
            HYPREDRV_LinearSystemSetDiscreteCurl(obj, (HYPRE_Matrix)obj->mat_C.handle),
            ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(
                      obj, (HYPRE_Vector)obj->vec_coord[0].handle,
                      (HYPRE_Vector)obj->vec_coord[1].handle,
                      (HYPRE_Vector)obj->vec_coord[2].handle),
                   ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemSetCoarseSchur(obj, 0, NULL), ERROR_NONE);
         ASSERT_PTR_EQ(obj->precon, precon);
         ASSERT_PTR_EQ(obj->solver, solver);
         if (which == 0)
            ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(
                         obj, (HYPRE_Matrix)create_test_ijmatrix_1x1(0.0)),
                      ERROR_NONE);
         else if (which == 1)
            ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteCurl(
                         obj, (HYPRE_Matrix)create_test_ijmatrix_1x1(0.0)),
                      ERROR_NONE);
         else if (which == 2)
         {
            HYPRE_IJVector replacement = create_test_ijvector_1x1(2.0);
            ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(
                         obj, (HYPRE_Vector)replacement, (HYPRE_Vector)replacement, NULL),
                      ERROR_NONE);
         }
         else
            ASSERT_EQ(HYPREDRV_LinearSystemSetCoarseSchur(
                         obj, 0, (HYPRE_Matrix)create_test_ijmatrix_1x1(1.0)),
                      ERROR_NONE);
         if ((m == 0 && (which == 0 || which == 2)) || (m == 1 && which < 3) ||
             (m == 3 && which == 3))
         {
            ASSERT_NULL(obj->precon);
            ASSERT_NULL(obj->solver);
            ASSERT_FALSE(obj->precon_is_setup);
         }
         else
         {
            ASSERT_PTR_EQ(obj->precon, precon);
            ASSERT_PTR_EQ(obj->solver, solver);
         }
         ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      }
   }
}

static void
test_invalid_replacements_preserve_registered_objects(void)
{
   HYPREDRV_t obj = create_operator_object(MPI_COMM_SELF, "ads", 0);
   ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
   HYPRE_Precon   precon   = obj->precon;
   hypredrv_Solver *solver   = obj->solver;
   HYPRE_IJMatrix gradient = obj->mat_G.handle, system = obj->mat_A.handle;
   HYPRE_IJVector coordinates = obj->vec_coord[0].handle;
   HYPRE_IJMatrix matrix      = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &matrix), 0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
   for (int initialized = 0; initialized < 2; initialized++)
   {
      if (initialized)
         ASSERT_EQ(HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST), 0);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetDiscreteGradient(obj, (HYPRE_Matrix)matrix) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetDiscreteCurl(obj, (HYPRE_Matrix)matrix) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetCoarseSchur(obj, 0, (HYPRE_Matrix)matrix) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)matrix) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->mat_G.handle, gradient);
      ASSERT_PTR_EQ(obj->mat_A.handle, system);
      ASSERT_PTR_EQ(obj->mat_M.handle, system);
      ASSERT_NULL(obj->mat_coarse_schur[0].handle);
   }
   HYPRE_IJVector vector = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &vector), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
   ASSERT_TRUE(
      HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)vector, NULL, NULL) &
      ERROR_INVALID_VAL);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);
   vector = create_test_ijvector_2x1(2.0, 3.0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)coordinates,
                                                   (HYPRE_Vector)vector, NULL) &
               ERROR_INVALID_VAL);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_IJVector multivector = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &multivector), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(multivector, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorSetNumComponents(multivector, 2), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize_v2(multivector, HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(multivector), 0);
   ASSERT_TRUE(
      HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)multivector, NULL, NULL) &
      ERROR_INVALID_VAL);
   ASSERT_EQ(HYPRE_IJVectorDestroy(multivector), 0);
#endif
   ASSERT_TRUE(HYPREDRV_LinearSystemSetCoarseSchur(obj, -1, NULL) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->precon, precon);
   ASSERT_PTR_EQ(obj->solver, solver);
   ASSERT_PTR_EQ(obj->vec_coord[0].handle, coordinates);
   ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(coordinates), 1.0, 0.0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(matrix), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_coordinates_retain_linear_system_aliases(void)
{
   for (int coordinates_first = 0; coordinates_first < 2; coordinates_first++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      HYPRE_IJVector shared = create_test_ijvector_1x1(4.0);
      if (coordinates_first)
         ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)shared,
                                                       (HYPRE_Vector)shared,
                                                       (HYPRE_Vector)shared),
                   ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)shared),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)shared),
                ERROR_NONE);
      if (!coordinates_first)
         ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)shared,
                                                       (HYPRE_Vector)shared,
                                                       (HYPRE_Vector)shared),
                   ERROR_NONE);
      char *argv[] = {"general:\n  statistics: off\n  exec_policy: host\n"
                      "solver: gmres\npreconditioner: amg\n"};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)create_test_ijmatrix_1x1(2.0)),
         ERROR_NONE);
      HYPRE_Real rhs[] = {8.0};
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 0, rhs), ERROR_NONE);
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)obj->vec_b.handle),
         ERROR_NONE);
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)obj->vec_b.handle),
         ERROR_NONE);
      ASSERT_PTR_EQ(obj->vec_coord[0].handle, shared);
      ASSERT_TRUE(obj->vec_coord[0].owned);
      ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(shared), 4.0, 0.0);
      /* Ownership follows a coordinate permutation, including duplicate handles. */
      HYPRE_IJVector second = create_test_ijvector_1x1(6.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(
                   obj, (HYPRE_Vector)second, (HYPRE_Vector)shared, (HYPRE_Vector)shared),
                ERROR_NONE);
      ASSERT_TRUE(obj->vec_coord[0].owned);
      ASSERT_TRUE(obj->vec_coord[1].owned);
      ASSERT_FALSE(obj->vec_coord[2].owned);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, NULL, NULL, NULL), ERROR_NONE);
      ASSERT_PTR_EQ(obj->vec_b.handle, shared);
      ASSERT_TRUE(obj->vec_b.owned);
      ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(shared), 4.0, 0.0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_borrowed_coordinates_keep_original_ownership(void)
{
   for (int library = 0; library < 2; library++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      if (library) ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
      HYPRE_IJVector shared = create_test_ijvector_1x1(9.0);
      /* Explicit solution storage is borrowed even outside library mode. */
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(obj, (HYPRE_Vector)shared, NULL,
                                                    (HYPRE_Vector)shared),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_FALSE(obj->vec_b.owned);
      ASSERT_FALSE(obj->vec_x.owned);
      ASSERT_FALSE(obj->vec_coord[0].owned);
      ASSERT_FALSE(obj->vec_coord[2].owned);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ_DOUBLE(get_test_ijvector_1x1(shared), 9.0, 0.0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(shared), 0);
   }
}

static void
test_matrix_aliases_survive_replacement(void)
{
   for (int system_first = 0; system_first < 2; system_first++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      HYPRE_IJMatrix shared = create_test_ijmatrix_1x1(3.0);
      if (system_first)
         ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)shared), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(obj, (HYPRE_Matrix)shared),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteCurl(obj, (HYPRE_Matrix)shared),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetCoarseSchur(obj, 0, (HYPRE_Matrix)shared),
                ERROR_NONE);
      if (!system_first)
         ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)shared), ERROR_NONE);
      ASSERT_TRUE(obj->mat_A.owned);
      ASSERT_FALSE(obj->mat_M.owned);
      ASSERT_FALSE(obj->mat_G.owned);
      ASSERT_FALSE(obj->mat_C.owned);
      ASSERT_FALSE(obj->mat_coarse_schur[0].owned);
      char *argv[] = {"general:\n  statistics: off\n  exec_policy: host\n"
                      "solver: gmres\npreconditioner: amg\n"};
      ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
      HYPRE_BigInt offsets[] = {0, 1}, columns[] = {0};
      HYPRE_Real   values[] = {5.0};
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 0, offsets, columns, values),
         ERROR_NONE);
      ASSERT_TRUE(obj->mat_G.owned);
      ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(shared), 3.0, 0.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(obj, NULL), ERROR_NONE);
      ASSERT_TRUE(obj->mat_C.owned);
      ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)shared),
                ERROR_NONE);
      ASSERT_TRUE(obj->mat_M.owned);
      ASSERT_FALSE(obj->mat_C.owned);
      ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteCurl(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetPrecMatrix(obj, (HYPRE_Matrix)obj->mat_A.handle),
                ERROR_NONE);
      ASSERT_TRUE(obj->mat_coarse_schur[0].owned);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetCoarseSchur(obj, 1, (HYPRE_Matrix)shared) &
                  ERROR_INVALID_VAL);
      ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(shared), 3.0, 0.0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   HYPRE_IJMatrix shared = create_test_ijmatrix_1x1(7.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)shared), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(obj, (HYPRE_Matrix)shared),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteCurl(obj, (HYPRE_Matrix)shared), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetCoarseSchur(obj, 0, (HYPRE_Matrix)shared),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ_DOUBLE(get_test_ijmatrix_1x1(shared), 7.0, 0.0);
   ASSERT_EQ(HYPRE_IJMatrixDestroy(shared), 0);
}

static void
test_private_operator_registration_preserves_diagnostics(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = create_test_ijmatrix_1x1(2.0);
   HYPRE_IJVector vector = create_test_ijvector_1x1(3.0);
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   hypredrv_ErrorMsgAdd("Earlier operator diagnostic");
   ASSERT_EQ(hypredrv_PreconSetDiscreteOperator(obj, PRECON_OPERATOR_GRADIENT, matrix),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_PreconSetDiscreteOperator(obj, PRECON_OPERATOR_CURL, matrix),
             ERROR_NONE);
   ASSERT_EQ(hypredrv_PreconSetCoordinates(obj, vector, vector, vector), ERROR_NONE);
   ASSERT_EQ(hypredrv_PreconSetCoarseSchur(obj, 0, matrix), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_distributed_operator_registration(void)
{
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size == 1) return;
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t obj = create_operator_object(MPI_COMM_WORLD, "ads", locations[p]);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      HYPRE_Precon   precon      = obj->precon;
      hypredrv_Solver *solver      = obj->solver;
      HYPRE_IJMatrix gradient    = obj->mat_G.handle;
      HYPRE_IJVector coordinates = obj->vec_coord[0].handle;
      /* Different operations and levels must reject before divergent validation. */
      uint32_t status = rank ? HYPREDRV_LinearSystemSetDiscreteGradient(obj, NULL)
                             : HYPREDRV_LinearSystemSetDiscreteCurl(obj, NULL);
      ASSERT_TRUE(status & ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetCoarseSchur(obj, rank, NULL) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetCoarseSchur(obj, rank ? 0 : -1, NULL) &
                  ERROR_INVALID_VAL);
      if (rank)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Injected operator registration failure");
      ASSERT_TRUE(HYPREDRV_LinearSystemSetDiscreteGradient(obj, (HYPRE_Matrix)gradient) &
                  ERROR_HYPRE_INTERNAL);
      ASSERT_EQ(HYPRE_GetError(), 0);
      /* An uninitialized candidate on one rank leaves every rank unchanged. */
      HYPRE_IJMatrix bad_matrix = NULL;
      HYPRE_IJVector bad_vector = NULL;
      HYPRE_BigInt   first = rank ? rank - 1 : 0, last = rank ? first : -1;
      ASSERT_EQ(
         HYPRE_IJMatrixCreate(MPI_COMM_WORLD, first, last, first, last, &bad_matrix), 0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(bad_matrix, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, first, last, &bad_vector), 0);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(bad_vector, HYPRE_PARCSR), 0);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetDiscreteGradient(
                     obj, (HYPRE_Matrix)(rank ? bad_matrix : gradient)) &
                  ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetCoordinates(
                     obj, (HYPRE_Vector)(rank ? bad_vector : coordinates), NULL, NULL) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->mat_G.handle, gradient);
      ASSERT_PTR_EQ(obj->vec_coord[0].handle, coordinates);
      ASSERT_PTR_EQ(obj->precon, precon);
      ASSERT_PTR_EQ(obj->solver, solver);
      ASSERT_EQ(HYPRE_IJMatrixDestroy(bad_matrix), 0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(bad_vector), 0);
      /* A changed handle only on the nonempty rank invalidates all consumers. */
      ASSERT_EQ(HYPREDRV_LinearSystemSetDiscreteGradient(
                   obj, (HYPRE_Matrix)(rank ? obj->mat_C.handle : gradient)),
                ERROR_NONE);
      ASSERT_NULL(obj->precon);
      ASSERT_NULL(obj->solver);
      ASSERT_EQ(HYPREDRV_LinearSolverCreate(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetCoordinates(
                   obj, (HYPRE_Vector)coordinates, (HYPRE_Vector)coordinates,
                   rank ? NULL : (HYPRE_Vector)coordinates),
                ERROR_NONE);
      ASSERT_NULL(obj->precon);
      ASSERT_NULL(obj->solver);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
#ifdef HYPRE_USING_GPU
   ASSERT_EQ(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST), 0);
#endif
   RUN_TEST(test_changed_operators_invalidate_consumers);
   RUN_TEST(test_invalid_replacements_preserve_registered_objects);
   RUN_TEST(test_coordinates_retain_linear_system_aliases);
   RUN_TEST(test_borrowed_coordinates_keep_original_ownership);
   RUN_TEST(test_matrix_aliases_survive_replacement);
   RUN_TEST(test_private_operator_registration_preserves_diagnostics);
   RUN_TEST(test_distributed_operator_registration);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
