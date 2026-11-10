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

static HYPREDRV_t
create_mode_object(MPI_Comm comm, int device, HYPRE_BigInt first, int count)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(comm, &obj), ERROR_NONE);
   char yaml[256];
   snprintf(yaml, sizeof(yaml),
            "general:\n  statistics: off\n  exec_policy: %s\n"
            "solver: pcg\npreconditioner: amg\n",
            device ? "device" : "host");
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJMatrix matrix = NULL;
   ASSERT_EQ(HYPRE_IJMatrixCreate(comm, first, first + count - 1, first,
                                  first + count - 1, &matrix),
             0);
   ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST), 0);
   for (int i = 0; i < count; i++)
   {
      HYPRE_BigInt row = first + i;
      HYPRE_Int    one = 1;
      HYPRE_Real   val = 1.0;
      ASSERT_EQ(HYPRE_IJMatrixSetValues(matrix, 1, &one, &row, &row, &val), 0);
   }
   ASSERT_EQ(HYPRE_IJMatrixAssemble(matrix), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
   obj->iargs->precon_method = PRECON_MGR;
   hypredrv_MGRSetDefaultArgs(&obj->iargs->precon.mgr);
   return obj;
}

static void
test_failed_replacement_preserves_modes(void)
{
   HYPREDRV_t    obj     = create_mode_object(MPI_COMM_SELF, 0, 7, 2);
   HYPRE_Complex modes[] = {1.0, 2.0, 2.0, 4.0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, modes), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 1, modes), ERROR_NONE);
   HYPRE_IJVector exact = obj->vec_ns, near = obj->vec_nn;
   int            generation = obj->iargs->precon.mgr.rbm_input_generation;

   ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, NULL) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_ns, exact);
   ASSERT_EQ(obj->num_ns, 1);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 2, modes) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_ns, exact);
   ASSERT_EQ(obj->num_ns, 1);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(obj, 1, 1, modes) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_nn, near);
   ASSERT_PTR_EQ(obj->iargs->precon.mgr.vec_nn, near);
   ASSERT_EQ(obj->iargs->precon.mgr.rbm_input_generation, generation);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, -1, modes) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_nn, near);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 0, NULL) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_nn, near);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, -1, 1, modes) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_ns, exact);
   HYPRE_Complex zero[] = {0.0, 0.0};
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, zero) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_ns, exact);

   HYPRE_Complex invalid[] = {NAN, 1.0};
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, invalid) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_ns, exact);
   invalid[0] = INFINITY;
   ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 1, invalid) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_nn, near);

   HYPRE_Complex *orthogonal = hypredrv_IJVectorData(exact, HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)orthogonal[0], 1.0 / sqrt(5.0), 1e-12);
   ASSERT_EQ_DOUBLE((double)orthogonal[1], 2.0 / sqrt(5.0), 1e-12);
   /* Replacement may read from the object's currently installed storage. */
   ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, orthogonal), ERROR_NONE);
   HYPRE_Complex *existing = hypredrv_IJVectorData(near, HYPRE_MEMORY_HOST);
   ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 1, existing), ERROR_NONE);
   existing = hypredrv_IJVectorData(obj->vec_nn, HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)existing[0], 1.0, 0.0);
   ASSERT_EQ_DOUBLE((double)existing[1], 2.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_private_modes_preserve_diagnostics(void)
{
   HYPREDRV_t    obj     = create_mode_object(MPI_COMM_SELF, 0, 0, 2);
   HYPRE_Complex modes[] = {1.0, 2.0};
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   hypredrv_ErrorMsgAdd("Earlier private diagnostic");
   ASSERT_EQ(hypredrv_LinearSystemSetNullSpaceModes(obj, true, 2, 1, modes), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemSetNullSpaceModes(obj, false, 2, 1, modes), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemSetNullSpaceModes(obj, true, 0, 0, NULL), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 2, 1, modes), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_storage_copy_and_projection(void)
{
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t    obj        = create_mode_object(MPI_COMM_SELF, locations[p], 0, 2);
      HYPRE_Complex modes[]    = {1.0, 2.0, 3.0, 1.0};
      int           components = 1;
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      components = 2;
#endif
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 2, components, modes), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, components, modes),
                ERROR_NONE);
      void *native = NULL;
      ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_nn, &native), 0);
      ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native),
                locations[p] ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
      ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_ns, &native), 0);
      ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native),
                HYPRE_MEMORY_HOST);
      modes[0]              = 99.0;
      HYPRE_Complex *copied = hypredrv_IJVectorData(obj->vec_nn, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)copied[0], 1.0, 0.0);
      ASSERT_EQ_DOUBLE((double)copied[1], 2.0, 0.0);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      ASSERT_EQ(HYPRE_IJVectorSetComponent(obj->vec_nn, 1), 0);
      HYPRE_Complex actual[2];
      ASSERT_EQ(HYPRE_IJVectorGetValues(obj->vec_nn, 2, NULL, actual), 0);
      ASSERT_EQ_DOUBLE((double)actual[0], 3.0, 0.0);
      ASSERT_EQ_DOUBLE((double)actual[1], 1.0, 0.0);
#endif

      HYPRE_IJVector vector = NULL;
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 1, &vector), 0);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST), 0);
      ASSERT_EQ(HYPRE_IJVectorAssemble(vector), 0);
      ASSERT_EQ(HYPRE_IJVectorGetObject(vector, &native), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)native, 3.0), 0);
#ifdef HYPRE_USING_GPU
      if (locations[p])
         ASSERT_EQ(hypredrv_IJVectorMigrate(vector, HYPRE_MEMORY_DEVICE), 0);
#endif
      hypredrv_LinearSystemProjectOutNullSpace(obj->vec_ns, components, vector);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_Complex *projected = hypredrv_IJVectorData(vector, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)(projected[0] + 2.0 * projected[1]), 0.0, 1e-12);
      if (components == 2)
         ASSERT_EQ_DOUBLE((double)(3.0 * projected[0] + projected[1]), 0.0, 1e-12);
      ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);

      ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 1, NULL), ERROR_NONE);
      copied = hypredrv_IJVectorData(obj->vec_nn, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)copied[0], 0.0, 0.0);
      ASSERT_EQ_DOUBLE((double)copied[1], 0.0, 0.0);
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
      ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, 2, 2, modes), ERROR_NONE);
      copied = hypredrv_IJVectorData(obj->vec_nn, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)copied[0], 3.0, 0.0);
      ASSERT_EQ_DOUBLE((double)copied[1], 1.0, 0.0);
#endif
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 0, 0, NULL), ERROR_NONE);
      ASSERT_NULL(obj->vec_ns);
      ASSERT_EQ(obj->num_ns, 0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_distributed_validation_and_empty_rank(void)
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
      int                  device = locations[p];
      int                  count  = rank ? 2 : 0;
      HYPRE_BigInt         first  = rank ? 2 * (rank - 1) : 0;
      HYPREDRV_t           obj = create_mode_object(MPI_COMM_WORLD, device, first, count);
      HYPRE_Complex        modes[] = {1.0, 2.0, 3.0, 1.0};
      const HYPRE_Complex *values  = count ? modes : NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, values), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetNearNullSpace(obj, count, 1, values), ERROR_NONE);
      HYPRE_IJVector exact = obj->vec_ns, near = obj->vec_nn;

      /* Every rank must reject before entering orthogonalization or native assembly. */
      ASSERT_TRUE(
         HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, rank == 1 ? NULL : values) &
         ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_ns, exact);
      ASSERT_EQ(obj->num_ns, 1);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(
                     obj, rank == 1 ? count + 1 : count, 1, values) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_nn, near);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, count, rank ? 1 : 0, values) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_ns, exact);
      ASSERT_TRUE(
         HYPREDRV_LinearSystemSetNearNullSpace(obj, count, rank ? 1 : 2, values) &
         ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_nn, near);
      ASSERT_TRUE((rank ? HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, values)
                        : HYPREDRV_LinearSystemSetNearNullSpace(obj, count, 1, values)) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_ns, exact);
      ASSERT_PTR_EQ(obj->vec_nn, near);

      if (rank == 1) modes[0] = NAN;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, values) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_ns, exact);
      modes[0] = 1.0;

      HYPRE_IJMatrix saved = obj->mat_A.handle;
      if (!rank) obj->mat_A.handle = NULL;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, values) &
                  ERROR_INVALID_VAL);
      obj->mat_A.handle = saved;
      ASSERT_PTR_EQ(obj->vec_ns, exact);
      if (rank == 1)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "injected mode construction failure");
      ASSERT_TRUE(HYPREDRV_LinearSystemSetNearNullSpace(obj, count, 1, values) &
                  ERROR_HYPRE_INTERNAL);
      ASSERT_PTR_EQ(obj->vec_nn, near);
      ASSERT_PTR_EQ(obj->iargs->precon.mgr.vec_nn, near);
      ASSERT_EQ(HYPRE_GetError(), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, count, 1, values), ERROR_NONE);
      hypredrv_ErrorStateReset();
      if (!rank) hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ASSERT_EQ(hypredrv_LinearSystemSetNullSpaceModes(obj, false, count, 1, values),
                ERROR_NONE);
      ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), rank ? ERROR_NONE : ERROR_MISSING_DOFMAP);
      hypredrv_ErrorStateReset();
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, count, 2, values), ERROR_NONE);
#endif
      HYPRE_IJVector vector = NULL;
      void          *native = NULL;
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, first, first + count - 1, &vector),
                0);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST), 0);
      ASSERT_EQ(HYPRE_IJVectorAssemble(vector), 0);
      ASSERT_EQ(HYPRE_IJVectorGetObject(vector, &native), 0);
      ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)native, 3.0), 0);
#ifdef HYPRE_USING_GPU
      if (device) ASSERT_EQ(hypredrv_IJVectorMigrate(vector, HYPRE_MEMORY_DEVICE), 0);
#endif
      hypredrv_LinearSystemProjectOutNullSpace(obj->vec_ns, obj->num_ns, vector);
      ASSERT_FALSE(hypredrv_ErrorCodeActive());
      HYPRE_Complex *projected = hypredrv_IJVectorData(vector, HYPRE_MEMORY_HOST);
      double local_dot  = count ? (double)(projected[0] + 2.0 * projected[1]) : 0.0;
      double global_dot = 0.0;
      MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      ASSERT_EQ_DOUBLE(global_dot, 0.0, 1e-12);
      ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetNullSpace(obj, 0, 0, NULL), ERROR_NONE);
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
   RUN_TEST(test_failed_replacement_preserves_modes);
   RUN_TEST(test_private_modes_preserve_diagnostics);
   RUN_TEST(test_storage_copy_and_projection);
   RUN_TEST(test_distributed_validation_and_empty_rank);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
