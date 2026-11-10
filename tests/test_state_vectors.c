/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "_hypre_parcsr_mv.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "object.h"
#include "test_helpers.h"

static HYPRE_IJVector
create_state_vector(MPI_Comm comm, HYPRE_BigInt first, HYPRE_BigInt last, double value)
{
   HYPRE_IJVector vector = NULL;
   void          *object = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(comm, first, last, &vector), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vector), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(vector, &object), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)object, value), 0);
   return vector;
}

static void
test_correction_uses_solution_storage(void)
{
   const int locations[][2] = {
      {0, 0},
#ifdef HYPRE_USING_GPU
      {1, 0},
      {0, 1},
      {1, 1},
#endif
   };
   for (size_t i = 0; i < sizeof(locations) / sizeof(locations[0]); i++)
   {
      HYPREDRV_t obj = NULL;
      ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
      HYPRE_IJVector state      = create_state_vector(MPI_COMM_SELF, 0, 0, 10.0);
      HYPRE_IJVector other      = create_state_vector(MPI_COMM_SELF, 0, 0, 20.0);
      HYPRE_IJVector correction = create_state_vector(MPI_COMM_SELF, 0, 0, 2.0);
      HYPRE_IJVector states[]   = {state, other};
#ifdef HYPRE_USING_GPU
      if (locations[i][0])
      {
         ASSERT_EQ(hypredrv_IJVectorMigrate(state, HYPRE_MEMORY_DEVICE), 0);
      }
      if (locations[i][1])
      {
         ASSERT_EQ(hypredrv_IJVectorMigrate(correction, HYPRE_MEMORY_DEVICE), 0);
      }
      ASSERT_EQ(hypredrv_IJVectorMigrate(other, HYPRE_MEMORY_DEVICE), 0);
#endif
      ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 2, states), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)correction),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_NONE);
      void *native_state = NULL, *native_correction = NULL;
      ASSERT_EQ(HYPRE_IJVectorGetObject(state, &native_state), 0);
      ASSERT_EQ(HYPRE_IJVectorGetObject(correction, &native_correction), 0);
      ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native_state),
                hypre_ParVectorMemoryLocation((hypre_ParVector *)native_correction));
      ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 0, 1), ERROR_NONE);
      HYPRE_Complex *data = NULL;
      ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 12.0, 1e-12);
      ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 1, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 12.0, 1e-12);

      /* Exposing the solution can move it back to the host before the next update. */
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 2.0, 1e-12);
      ASSERT_EQ(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 14.0, 1e-12);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ(HYPRE_IJVectorDestroy(state), 0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(other), 0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(correction), 0);
   }
}

static void
test_data_access_preserves_requested_storage(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   HYPRE_IJVector vector = create_state_vector(MPI_COMM_SELF, 0, 0, 7.0);
#ifdef HYPRE_USING_GPU
   ASSERT_EQ(hypredrv_IJVectorMigrate(vector, HYPRE_MEMORY_DEVICE), 0);
#endif
   void *object = NULL;
   ASSERT_EQ(HYPRE_IJVectorGetObject(vector, &object), 0);
   hypre_ParVector     *par      = (hypre_ParVector *)object;
   HYPRE_MemoryLocation location = hypre_ParVectorMemoryLocation(par);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)vector), ERROR_NONE);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &data), ERROR_NONE);
   ASSERT_PTR_EQ(data, hypre_VectorData(hypre_ParVectorLocalVector(par)));
   ASSERT_EQ(hypre_ParVectorMemoryLocation(par), location);
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 1, &vector), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ(hypre_ParVectorMemoryLocation(par), HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)data[0], 7.0, 0.0);
   HYPRE_Complex *rhs_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_PTR_EQ(rhs_data, data);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);
}

static void
test_registration_rotation_and_replacement(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ_U32(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_INVALID_VAL);
   HYPRE_IJVector a        = create_state_vector(MPI_COMM_SELF, 0, 0, 1.0);
   HYPRE_IJVector b        = create_state_vector(MPI_COMM_SELF, 0, 0, 2.0);
   HYPRE_IJVector c        = create_state_vector(MPI_COMM_SELF, 0, 0, 3.0);
   HYPRE_IJVector states[] = {a, b, c};
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 3, states), ERROR_NONE);
   states[0] = c; /* Registration copies the caller's array. */
   for (int rotation = 0; rotation < 7; rotation++)
   {
      for (int index = 0; index < 3; index++)
      {
         ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, index, &data), ERROR_NONE);
         ASSERT_EQ_DOUBLE((double)data[0], (rotation + index) % 3 + 1.0, 0.0);
      }
      ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   }
   /* Re-registering the module's own handle array must copy before replacing it. */
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, obj->nstates, obj->vec_s), ERROR_NONE);
   HYPRE_IJVector invalid[] = {a, NULL, c};
   ASSERT_EQ_U32(HYPREDRV_StateVectorSet(obj, 3, invalid), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorSet(obj, 0, states), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorSet(obj, 3, NULL), ERROR_INVALID_VAL);
   HYPRE_IJVector uninitialized = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &uninitialized), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(uninitialized, HYPRE_PARCSR), 0);
   invalid[1] = uninitialized;
   ASSERT_EQ_U32(HYPREDRV_StateVectorSet(obj, 3, invalid), ERROR_INVALID_VAL);
   ASSERT_EQ(HYPRE_IJVectorDestroy(uninitialized), 0);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 2.0, 0.0);
   ASSERT_EQ_U32(HYPREDRV_StateVectorGetValues(obj, 3, &data), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorGetValues(obj, 0, NULL), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorCopy(obj, -1, 0), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_INVALID_VAL);

   HYPRE_IJVector replacement[] = {c, a};
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 2, replacement), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 3.0, 0.0);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 1.0, 0.0);
   ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 1, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 3.0, 0.0);
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 1, &b), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 0, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   /* All vectors remain caller-owned, including entries dropped by replacement. */
   HYPRE_Complex value = 0.0;
   HYPRE_BigInt  row   = 0;
   ASSERT_EQ(HYPRE_IJVectorGetValues(b, 1, &row, &value), 0);
   ASSERT_EQ_DOUBLE((double)value, 2.0, 0.0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(a), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(b), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(c), 0);
}

static void
test_private_state_operations_preserve_diagnostics(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector a          = create_state_vector(MPI_COMM_SELF, 0, 0, 1.0);
   HYPRE_IJVector b          = create_state_vector(MPI_COMM_SELF, 0, 0, 2.0);
   HYPRE_IJVector correction = create_state_vector(MPI_COMM_SELF, 0, 0, 3.0);
   HYPRE_IJVector states[]   = {a, b};
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)correction), ERROR_NONE);
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ObjectSetStateVectors(obj, 2, states), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ObjectCopyStateVectors(obj, 1, 0), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_ObjectApplyStateCorrection(obj, 0), ERROR_MISSING_DOFMAP);
   HYPRE_Complex *data = hypredrv_IJVectorData(a, HYPRE_MEMORY_HOST);
   ASSERT_NOT_NULL(data);
   ASSERT_EQ_DOUBLE((double)data[0], 5.0, 0.0);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 5.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(a), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(b), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(correction), 0);
}

static void
test_uninitialized_solution_is_rejected(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector state         = create_state_vector(MPI_COMM_SELF, 0, 0, 10.0);
   HYPRE_IJVector uninitialized = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &uninitialized), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(uninitialized, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 1, &state), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)uninitialized),
                 ERROR_INVALID_VAL);
   ASSERT_NULL(obj->vec_x.handle);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ_U32(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_INVALID_VAL);
   ASSERT_NULL(data);
   ASSERT_EQ_U32(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 10.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(state), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(uninitialized), 0);
}

static void
test_incompatible_layouts_are_rejected(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector a        = create_state_vector(MPI_COMM_SELF, 0, 0, 10.0);
   HYPRE_IJVector shifted  = create_state_vector(MPI_COMM_SELF, 1, 1, 20.0);
   HYPRE_IJVector longer   = create_state_vector(MPI_COMM_SELF, 0, 1, 2.0);
   HYPRE_IJVector states[] = {a, shifted, longer};
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 3, states), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_StateVectorCopy(obj, 0, 1), ERROR_INVALID_VAL);
   ASSERT_EQ_U32(HYPREDRV_StateVectorCopy(obj, 0, 2), ERROR_INVALID_VAL);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)longer), ERROR_NONE);
   ASSERT_EQ_U32(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_INVALID_VAL);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 10.0, 0.0);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 1, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 20.0, 0.0);
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 2, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 2.0, 0.0);
   ASSERT_EQ_DOUBLE((double)data[1], 2.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(a), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(shifted), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(longer), 0);
}

static void
test_distributed_states_with_empty_rank(void)
{
   int rank = 0, size = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (size < 2) return;
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_WORLD, &obj), ERROR_NONE);
   HYPRE_BigInt   first = rank ? rank - 1 : 0, last = rank - 1;
   HYPRE_IJVector state      = create_state_vector(MPI_COMM_WORLD, first, last, 10.0);
   HYPRE_IJVector other      = create_state_vector(MPI_COMM_WORLD, first, last, 20.0);
   HYPRE_IJVector correction = create_state_vector(MPI_COMM_WORLD, first, last, 2.0);
   HYPRE_IJVector states[]   = {state, other};
#ifdef HYPRE_USING_GPU
   ASSERT_EQ(hypredrv_IJVectorMigrate(state, HYPRE_MEMORY_DEVICE), 0);
#endif
   ASSERT_EQ(HYPREDRV_StateVectorSet(obj, 2, states), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)correction), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorApplyCorrection(obj, 0), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorCopy(obj, 0, 1), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_StateVectorUpdateAll(obj), ERROR_NONE);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_StateVectorGetValues(obj, 0, &data), ERROR_NONE);
   if (rank) ASSERT_EQ_DOUBLE((double)data[0], 12.0, 1e-12);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(state), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(other), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(correction), 0);
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
   RUN_TEST(test_correction_uses_solution_storage);
   RUN_TEST(test_data_access_preserves_requested_storage);
   RUN_TEST(test_registration_rotation_and_replacement);
   RUN_TEST(test_private_state_operations_preserve_diagnostics);
   RUN_TEST(test_uninitialized_solution_is_rejected);
   RUN_TEST(test_incompatible_layouts_are_rejected);
   RUN_TEST(test_distributed_states_with_empty_rank);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
