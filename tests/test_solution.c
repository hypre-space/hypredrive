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

/* Write the nonzero-origin rejection fixture directly. hypre 2.21 normalizes
 * ParVector partitions but its IJ printer asserts they are still absolute. */
static void
write_shifted_vector_fixture(const char *prefix, double value)
{
   char path[PATH_MAX];
   snprintf(path, sizeof(path), "%s.00000", prefix);
   FILE *file = fopen(path, "w");
   ASSERT_NOT_NULL(file);
   ASSERT_TRUE(fprintf(file, "1 2\n1 %.17g\n2 %.17g\n", value, value) > 0);
   ASSERT_EQ(fclose(file), 0);
}

static HYPRE_IJVector
create_vector(MPI_Comm comm, HYPRE_BigInt first, int count, double value)
{
   HYPRE_IJVector vector = NULL;
   void          *object = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(comm, first, first + count - 1, &vector), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize_v2(vector, HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(vector), 0);
   ASSERT_EQ(HYPRE_IJVectorGetObject(vector, &object), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)object, value), 0);
   return vector;
}

static HYPREDRV_t
create_object(MPI_Comm comm, int device, HYPRE_BigInt first, int count, int library)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(comm, &obj), ERROR_NONE);
   if (library) ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   char yaml[256];
   snprintf(
      yaml, sizeof(yaml),
      "general:\n  statistics: off\n  exec_policy: %s\n"
      "linear_system:\n  init_guess_mode: previous\nsolver: gmres\npreconditioner: amg\n",
      device ? "device" : "host");
   char *argv[] = {yaml};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   HYPRE_IJVector rhs = create_vector(comm, first, count, 2.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   return obj;
}

static void
test_failed_default_solution_preserves_borrowed_buffer(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector vector = create_vector(MPI_COMM_SELF, 0, 2, 7.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)vector), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, NULL) & ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_x.handle, vector);
   ASSERT_FALSE(obj->vec_x.owned);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(vector), 0);
}

static void
test_failed_guess_preserves_installed_vectors(void)
{
   HYPREDRV_t obj = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   HYPRE_IJVector old_guess = obj->vec_x0.handle, old_solution = obj->vec_x.handle;
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   const int labels[] = {0, 0};
   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 2, labels), ERROR_NONE);
   obj->iargs->solver.gmres.print_level = 3;
   ASSERT_EQ(hypredrv_VectorTagsSet(obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj->vector_tags);
   hypredrv_VectorTags *tags = obj->vector_tags;
#endif
   test_scratch_path(obj->iargs->ls.x0_filename, sizeof(obj->iargs->ls.x0_filename),
                     "unavailable-guess");
   ASSERT_TRUE(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL) & ERROR_FILE_NOT_FOUND);
   ASSERT_PTR_EQ(obj->vec_x0.handle, old_guess);
   ASSERT_PTR_EQ(obj->vec_x.handle, old_solution);
   ASSERT_TRUE(obj->vec_x0.owned);
   ASSERT_TRUE(obj->vec_x.owned);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   ASSERT_PTR_EQ(obj->vector_tags, tags);
#endif
   obj->iargs->ls.x0_filename[0] = '\0';
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_reference_alias_survives_solution_replacement(void)
{
   HYPREDRV_t obj = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   HYPRE_IJVector reference = obj->vec_x.handle;
   void          *native    = NULL;
   ASSERT_EQ(HYPRE_IJVectorGetObject(reference, &native), 0);
   ASSERT_EQ(HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)native, 5.0), 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)reference),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_PTR_EQ(obj->vec_xref.handle, reference);
   HYPRE_Complex *data = hypredrv_IJVectorData(reference, HYPRE_MEMORY_HOST);
   ASSERT_NOT_NULL(data);
   ASSERT_EQ_DOUBLE((double)data[0], 5.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_previous_mode_preserves_borrowed_vectors(void)
{
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t     obj      = create_object(MPI_COMM_SELF, locations[p], 0, 2, 1);
      HYPRE_IJVector rhs      = obj->vec_b.handle;
      HYPRE_IJVector guess    = create_vector(MPI_COMM_SELF, 0, 2, 3.0);
      HYPRE_IJVector solution = create_vector(MPI_COMM_SELF, 0, 2, 7.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)guess),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)solution),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      HYPRE_Complex *data = NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 7.0, 0.0);
      ASSERT_EQ_DOUBLE((double)data[1], 7.0, 0.0);
      data = hypredrv_IJVectorData(guess, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)data[0], 3.0, 0.0);
      data = hypredrv_IJVectorData(solution, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)data[0], 7.0, 0.0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ(HYPRE_IJVectorDestroy(rhs), 0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(guess), 0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(solution), 0);
   }
}

static void
test_alias_ownership_transfers(void)
{
   for (int library = 0; library < 2; library++)
   {
      HYPREDRV_t     obj = create_object(MPI_COMM_SELF, 0, 0, 2, library);
      HYPRE_IJVector rhs = obj->vec_b.handle;
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      HYPRE_IJVector shared = obj->vec_x0.handle;
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)shared),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)shared), ERROR_NONE);
      ASSERT_TRUE(obj->vec_x.owned);
      ASSERT_FALSE(obj->vec_x0.owned);
      ASSERT_FALSE(obj->vec_xref.owned);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)shared),
                ERROR_NONE);
      ASSERT_TRUE(obj->vec_x0.owned);
      ASSERT_FALSE(obj->vec_xref.owned);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_PTR_EQ(obj->vec_xref.handle, shared);
      ASSERT_TRUE(obj->vec_xref.owned);
      HYPRE_Complex *data = hypredrv_IJVectorData(shared, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)data[0], 0.0, 0.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)obj->vec_b.handle),
                ERROR_NONE);
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)obj->vec_b.handle),
         ERROR_NONE);
      ASSERT_EQ(obj->vec_b.owned, !library);
      ASSERT_FALSE(obj->vec_x0.owned);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 2.0, 0.0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      if (library) ASSERT_EQ(HYPRE_IJVectorDestroy(rhs), 0);
   }
}

static void
test_input_validation_preserves_vectors(void)
{
   HYPREDRV_t obj = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
   HYPRE_IJVector old_guess = obj->vec_x0.handle, old_solution = obj->vec_x.handle;
   HYPRE_IJVector uninitialized = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 1, &uninitialized), 0);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(uninitialized, HYPRE_PARCSR), 0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)uninitialized) &
               ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)uninitialized) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_x.handle, old_solution);
   ASSERT_PTR_EQ(obj->vec_x0.handle, old_guess);
   HYPRE_IJVector shifted = create_vector(MPI_COMM_SELF, 1, 2, 6.0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)shifted) &
               ERROR_INVALID_VAL);
   ASSERT_PTR_EQ(obj->vec_x.handle, old_solution);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)shifted), ERROR_NONE);
   ASSERT_TRUE(HYPREDRV_LinearSystemResetInitialGuess(obj) & ERROR_INVALID_VAL);
   HYPRE_Complex *data = hypredrv_IJVectorData(shifted, HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)data[0], 6.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_EQ(HYPRE_IJVectorDestroy(shifted), 0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(uninitialized), 0);
}

static void
test_generated_and_file_guesses(void)
{
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t obj     = create_object(MPI_COMM_SELF, locations[p], 0, 2, 0);
      const int  modes[] = {0, 1, 3, 4, 99};
      for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
      {
         obj->iargs->ls.init_guess_mode = modes[m];
         ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
         void *native = NULL;
         ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_x0.handle, &native), 0);
         ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native),
                   locations[p] ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
         /* Exposing x0 on the host also exercises a copy to a device solution. */
         HYPRE_Complex *data =
            hypredrv_IJVectorData(obj->vec_x0.handle, HYPRE_MEMORY_HOST);
         for (int i = 0; i < 2; i++)
         {
            if (modes[m] == 3)
               ASSERT_TRUE(isfinite((double)data[i]) && fabs((double)data[i]) <= 1.0);
            else ASSERT_EQ_DOUBLE((double)data[i], modes[m] == 1 ? 1.0 : 0.0, 0.0);
         }
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         /* Keep "previous" independent of the random mode above. */
         ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
      }
      HYPRE_IJVector file_vector = create_vector(MPI_COMM_SELF, 0, 2, 4.0);
      char          *prefix      = obj->iargs->ls.x0_filename;
      test_scratch_path(prefix, sizeof(obj->iargs->ls.x0_filename),
                        locations[p] ? "guess-device" : "guess-host");
      ASSERT_EQ(HYPRE_IJVectorPrint(file_vector, prefix), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      HYPRE_Complex *data = NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      ASSERT_EQ_DOUBLE((double)data[0], 4.0, 0.0);
      ASSERT_EQ_DOUBLE((double)data[1], 4.0, 0.0);
      HYPRE_IJVector saved       = obj->vec_x0.handle;
      write_shifted_vector_fixture(prefix, 8.0);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_x0.handle, saved);
      test_scratch_path(prefix, sizeof(obj->iargs->ls.x0_filename),
                        locations[p] ? "bad-device" : "bad-host");
      char filename[PATH_MAX];
      snprintf(filename, sizeof(filename), "%s.00000.bin", prefix);
      FILE *file = fopen(filename, "wb");
      ASSERT_NOT_NULL(file);
      ASSERT_EQ(fclose(file), 0);
      ASSERT_NE(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_PTR_EQ(obj->vec_x0.handle, saved);
      ASSERT_EQ(HYPRE_IJVectorDestroy(file_vector), 0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_supplied_guess_before_parsing(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector rhs   = create_vector(MPI_COMM_SELF, 0, 2, 2.0);
   HYPRE_IJVector guess = create_vector(MPI_COMM_SELF, 0, 2, 3.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)guess), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
   HYPRE_Complex *data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
   ASSERT_EQ_DOUBLE((double)data[0], 3.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_private_solution_operations_preserve_diagnostics(void)
{
   HYPREDRV_t obj = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
   hypredrv_ErrorMsgAdd("Earlier private diagnostic");
   ASSERT_EQ(hypredrv_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(hypredrv_LinearSystemSetInitialGuess(obj, GUESS_REPLACE, NULL), ERROR_NONE);
   ASSERT_EQ_U32(hypredrv_ErrorCodeGet(), ERROR_MISSING_DOFMAP);
   ASSERT_EQ_U32(hypredrv_LinearSystemResetInitialGuess(obj), ERROR_MISSING_DOFMAP);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, NULL), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_rhs_replacement_preserves_retained_aliases(void)
{
   for (int from_array = 0; from_array < 2; from_array++)
   {
      for (int share_solution = 0; share_solution < 2; share_solution++)
      {
         HYPREDRV_t     obj      = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
         HYPRE_IJVector original = obj->vec_b.handle;
         ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)original),
                   ERROR_NONE);
         ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)original),
                   ERROR_NONE);
         if (share_solution)
            ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)original),
                      ERROR_NONE);
         if (from_array)
         {
            HYPRE_BigInt offsets[] = {0, 1, 2}, columns[] = {0, 1};
            HYPRE_Real   diagonal[] = {1.0, 1.0}, rhs[] = {8.0, 8.0};
            ASSERT_EQ(HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 1, offsets, columns,
                                                            diagonal),
                      ERROR_NONE);
            ASSERT_EQ(HYPREDRV_LinearSystemSetRHSFromArray(obj, 0, 1, rhs), ERROR_NONE);
         }
         else
         {
            HYPRE_IJVector replacement = create_vector(MPI_COMM_SELF, 0, 2, 8.0);
            ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)replacement),
                      ERROR_NONE);
         }
         ASSERT_NULL(obj->vec_xref.handle);
         ASSERT_PTR_EQ(obj->vec_x0.handle, original);
         HYPRE_Complex *data = hypredrv_IJVectorData(original, HYPRE_MEMORY_HOST);
         ASSERT_EQ_DOUBLE((double)data[0], 2.0, 0.0);
         ASSERT_EQ(obj->vec_x0.owned, !share_solution);
         if (share_solution) ASSERT_TRUE(obj->vec_x.owned);
         ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
         ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      }
   }
}

static void
test_known_borrowed_handles_keep_their_ownership(void)
{
   HYPREDRV_t     obj      = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   HYPRE_IJVector borrowed = create_vector(MPI_COMM_SELF, 0, 2, 9.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)borrowed), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)borrowed),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)borrowed),
             ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)borrowed), ERROR_NONE);
   ASSERT_FALSE(obj->vec_b.owned);
   ASSERT_FALSE(obj->vec_x0.owned);
   ASSERT_FALSE(obj->vec_xref.owned);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   HYPRE_Complex *data = hypredrv_IJVectorData(borrowed, HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)data[0], 9.0, 0.0);
   ASSERT_EQ(HYPRE_IJVectorDestroy(borrowed), 0);
}

static void
test_rhs_and_reference_validate_input_once(void)
{
   HYPREDRV_t     obj      = create_object(MPI_COMM_SELF, 0, 0, 2, 0);
   HYPRE_IJVector original = obj->vec_b.handle;
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)original),
             ERROR_NONE);
   HYPRE_IJVector invalid = NULL;
   ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 1, &invalid), 0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
   ASSERT_EQ(HYPRE_IJVectorSetObjectType(invalid, HYPRE_PARCSR), 0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   ASSERT_EQ(HYPRE_IJVectorSetNumComponents(invalid, 2), 0);
   ASSERT_EQ(HYPRE_IJVectorInitialize_v2(invalid, HYPRE_MEMORY_HOST), 0);
   ASSERT_EQ(HYPRE_IJVectorAssemble(invalid), 0);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
   ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)invalid) &
               ERROR_INVALID_VAL);
#endif
   ASSERT_PTR_EQ(obj->vec_b.handle, original);
   ASSERT_PTR_EQ(obj->vec_xref.handle, original);
   ASSERT_TRUE(obj->vec_b.owned);
   ASSERT_EQ(HYPRE_IJVectorDestroy(invalid), 0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_configured_rhs_and_reference_storage(void)
{
   const int locations[] = {
      0,
#ifdef HYPRE_USING_GPU
      1,
#endif
   };
   for (size_t p = 0; p < sizeof(locations) / sizeof(locations[0]); p++)
   {
      HYPREDRV_t   obj       = create_object(MPI_COMM_SELF, locations[p], 0, 2, 0);
      HYPRE_BigInt offsets[] = {0, 1, 2}, columns[] = {0, 1};
      HYPRE_Real   diagonal[] = {2.0, 3.0};
      ASSERT_EQ(
         HYPREDRV_LinearSystemSetMatrixFromCSR(obj, 0, 1, offsets, columns, diagonal),
         ERROR_NONE);
      ASSERT_EQ(hypredrv_IJMatrixMigrate(obj->mat_A.handle, HYPRE_MEMORY_HOST), 0);
      const int modes[] = {0, 1, 3, 4, 99};
      for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
      {
         obj->iargs->ls.rhs_mode = modes[m];
         ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, NULL), ERROR_NONE);
         void *native = NULL;
         ASSERT_EQ(HYPRE_IJVectorGetObject(obj->vec_b.handle, &native), 0);
         ASSERT_EQ(hypre_ParVectorMemoryLocation((hypre_ParVector *)native),
                   locations[p] ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST);
         HYPRE_Complex *data =
            hypredrv_IJVectorData(obj->vec_b.handle, HYPRE_MEMORY_HOST);
         HYPRE_Complex *reference =
            modes[m] == 4 ? hypredrv_IJVectorData(obj->vec_xref.handle, HYPRE_MEMORY_HOST)
                          : NULL;
         for (int i = 0; i < 2; i++)
         {
            if (modes[m] == 4)
               ASSERT_EQ_DOUBLE((double)data[i], diagonal[i] * (double)reference[i],
                                1e-12);
            else if (modes[m] == 3)
               ASSERT_TRUE(isfinite((double)data[i]) && fabs((double)data[i]) <= 1.0);
            else ASSERT_EQ_DOUBLE((double)data[i], modes[m] == 0 ? 0.0 : 1.0, 0.0);
         }
         ASSERT_EQ(HYPRE_IJMatrixGetObject(obj->mat_A.handle, &native), 0);
         ASSERT_EQ(hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)native),
                   HYPRE_MEMORY_HOST);
         HYPRE_IJVector saved_reference = obj->vec_xref.handle;
         ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
         ASSERT_PTR_EQ(obj->vec_xref.handle, saved_reference);
      }
      HYPRE_IJVector fixture = create_vector(MPI_COMM_SELF, 0, 2, 8.0);
      char          *prefix  = obj->iargs->ls.rhs_filename;
      test_scratch_path(prefix, sizeof(obj->iargs->ls.rhs_filename),
                        locations[p] ? "rhs-device" : "rhs-host");
      ASSERT_EQ(HYPRE_IJVectorPrint(fixture, prefix), 0);
      obj->iargs->ls.rhs_mode = 2;
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, NULL), ERROR_NONE);
      HYPRE_IJVector saved_rhs   = obj->vec_b.handle;
      write_shifted_vector_fixture(prefix, 3.0);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_b.handle, saved_rhs);
      ASSERT_EQ(HYPRE_IJVectorPrint(fixture, prefix), 0);
      snprintf(obj->iargs->ls.xref_filename, sizeof(obj->iargs->ls.xref_filename), "%s",
               prefix);
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
      HYPRE_Complex *data =
         hypredrv_IJVectorData(obj->vec_xref.handle, HYPRE_MEMORY_HOST);
      ASSERT_EQ_DOUBLE((double)data[0], 8.0, 0.0);
      ASSERT_EQ(HYPRE_IJVectorDestroy(fixture), 0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   }
}

static void
test_configured_vectors_require_arguments(void)
{
   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   HYPRE_IJVector rhs = create_vector(MPI_COMM_SELF, 0, 2, 3.0);
   ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, (HYPRE_Vector)rhs), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)rhs),
             ERROR_NONE);
   HYPRE_IJVector solution                               = obj->vec_x.handle;
   uint32_t (*const setters[])(HYPREDRV_t, HYPRE_Vector) = {
      HYPREDRV_LinearSystemSetRHS, HYPREDRV_LinearSystemSetInitialGuess,
      HYPREDRV_LinearSystemSetReferenceSolution};
   for (size_t i = 0; i < sizeof(setters) / sizeof(setters[0]); i++)
   {
      ASSERT_TRUE(setters[i](obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_b.handle, rhs);
      ASSERT_PTR_EQ(obj->vec_x0.handle, rhs);
      ASSERT_PTR_EQ(obj->vec_xref.handle, rhs);
      ASSERT_PTR_EQ(obj->vec_x.handle, solution);
      ASSERT_TRUE(obj->vec_b.owned);
      ASSERT_FALSE(obj->vec_x0.owned);
      ASSERT_FALSE(obj->vec_xref.owned);
   }
   HYPRE_Complex *data = hypredrv_IJVectorData(rhs, HYPRE_MEMORY_HOST);
   ASSERT_EQ_DOUBLE((double)data[0], 3.0, 0.0);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
}

static void
test_distributed_solution_replacement(void)
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
      int            count = rank ? 2 : 0;
      HYPRE_BigInt   first = rank ? 2 * (rank - 1) : 0;
      HYPREDRV_t     obj   = create_object(MPI_COMM_WORLD, locations[p], first, count, 0);
      HYPRE_IJMatrix matrix = NULL;
      ASSERT_EQ(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, first, first + count - 1, first,
                                     first + count - 1, &matrix),
                0);
      ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJMatrixInitialize_v2(matrix, HYPRE_MEMORY_HOST), 0);
      for (int i = 0; i < count; i++)
      {
         HYPRE_BigInt row   = first + i;
         HYPRE_Int    one   = 1;
         HYPRE_Real   value = 2.0;
         ASSERT_EQ(HYPRE_IJMatrixSetValues(matrix, 1, &one, &row, &row, &value), 0);
      }
      ASSERT_EQ(HYPRE_IJMatrixAssemble(matrix), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetMatrix(obj, (HYPRE_Matrix)matrix), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      HYPRE_IJVector previous_guess    = obj->vec_x0.handle,
                     previous_solution = obj->vec_x.handle;
      HYPRE_IJVector rhs               = obj->vec_b.handle;
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)rhs),
                ERROR_NONE);
      const bool rhs_owned = obj->vec_b.owned, guess_owned = obj->vec_x0.owned,
                 solution_owned = obj->vec_x.owned, reference_owned = obj->vec_xref.owned;
      input_args *saved_args = obj->iargs;
      if (rank == 1) obj->iargs = NULL;
      uint32_t (*const setters[])(HYPREDRV_t, HYPRE_Vector) = {
         HYPREDRV_LinearSystemSetRHS, HYPREDRV_LinearSystemSetInitialGuess,
         HYPREDRV_LinearSystemSetReferenceSolution};
      for (size_t i = 0; i < sizeof(setters) / sizeof(setters[0]); i++)
      {
         ASSERT_TRUE(setters[i](obj, NULL) & ERROR_INVALID_VAL);
         ASSERT_PTR_EQ(obj->vec_b.handle, rhs);
         ASSERT_PTR_EQ(obj->vec_x0.handle, previous_guess);
         ASSERT_PTR_EQ(obj->vec_x.handle, previous_solution);
         ASSERT_PTR_EQ(obj->vec_xref.handle, rhs);
         ASSERT_EQ(obj->vec_b.owned, rhs_owned);
         ASSERT_EQ(obj->vec_x0.owned, guess_owned);
         ASSERT_EQ(obj->vec_x.owned, solution_owned);
         ASSERT_EQ(obj->vec_xref.owned, reference_owned);
      }
      obj->iargs = saved_args;
      if (rank == 1) obj->vec_b.handle = NULL;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, NULL) & ERROR_INVALID_VAL);
      obj->vec_b.handle = rhs;
      ASSERT_PTR_EQ(obj->vec_x0.handle, previous_guess);
      ASSERT_PTR_EQ(obj->vec_x.handle, previous_solution);
      HYPRE_IJVector external = create_vector(MPI_COMM_WORLD, first, count, 7.0);
      ASSERT_TRUE(
         HYPREDRV_LinearSystemSetInitialGuess(obj, rank ? NULL : (HYPRE_Vector)external) &
         ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_x0.handle, previous_guess);
      if (rank == 1)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "injected solution construction failure");
      ASSERT_TRUE(HYPREDRV_LinearSystemSetSolution(obj, NULL) & ERROR_HYPRE_INTERNAL);
      ASSERT_PTR_EQ(obj->vec_x.handle, previous_solution);
      ASSERT_EQ(HYPRE_GetError(), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)external),
                ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      HYPRE_Complex *data = NULL;
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      if (count) ASSERT_EQ_DOUBLE((double)data[0], 7.0, 0.0);
      /* Local attachment must not require participation from the other ranks. */
      if (!rank)
         ASSERT_EQ(HYPREDRV_LinearSystemSetSolution(obj, (HYPRE_Vector)obj->vec_x.handle),
                   ERROR_NONE);
      MPI_Barrier(MPI_COMM_WORLD);
      obj->iargs->ls.rhs_mode = 4;
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, NULL), ERROR_NONE);
      data = hypredrv_IJVectorData(obj->vec_b.handle, HYPRE_MEMORY_HOST);
      HYPRE_Complex *reference_values =
         hypredrv_IJVectorData(obj->vec_xref.handle, HYPRE_MEMORY_HOST);
      for (int i = 0; i < count; i++)
         ASSERT_EQ_DOUBLE((double)data[i], 2.0 * (double)reference_values[i], 1e-12);
      HYPRE_IJVector saved_rhs       = obj->vec_b.handle,
                     saved_reference = obj->vec_xref.handle;
      obj->iargs->ls.rhs_mode        = rank ? 4 : 0;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_b.handle, saved_rhs);
      ASSERT_PTR_EQ(obj->vec_xref.handle, saved_reference);
      char prefix[MAX_FILENAME_LENGTH] = {0};
      if (!rank)
         test_scratch_path(prefix, sizeof(prefix),
                           locations[p] ? "world-rhs-device" : "world-rhs-host");
      MPI_Bcast(prefix, sizeof(prefix), MPI_CHAR, 0, MPI_COMM_WORLD);
      ASSERT_EQ(HYPRE_IJVectorPrint(obj->vec_b.handle, prefix), 0);
      snprintf(obj->iargs->ls.rhs_filename, sizeof(obj->iargs->ls.rhs_filename), "%s",
               prefix);
      obj->iargs->ls.rhs_mode = 2;
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, NULL), ERROR_NONE);
      saved_rhs           = obj->vec_b.handle;
      obj->iargs->ls.type = rank ? 1 : 3;
      ASSERT_TRUE(HYPREDRV_LinearSystemSetRHS(obj, NULL) & ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_b.handle, saved_rhs);
      obj->iargs->ls.type = 1;
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)saved_rhs),
                ERROR_NONE);
      snprintf(obj->iargs->ls.xref_filename, sizeof(obj->iargs->ls.xref_filename), "%s",
               prefix);
      if (!rank) obj->iargs->ls.xref_filename[0] = '\0';
      ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL) &
                  ERROR_INVALID_VAL);
      ASSERT_PTR_EQ(obj->vec_xref.handle, saved_rhs);
      obj->iargs->ls.xref_filename[0] = '\0';
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
      ASSERT_PTR_EQ(obj->vec_xref.handle, saved_rhs);
      snprintf(obj->iargs->ls.xref_filename, sizeof(obj->iargs->ls.xref_filename), "%s",
               prefix);

      char file_prefixes[3][MAX_FILENAME_LENGTH] = {{0}};
      if (!rank)
      {
         test_scratch_path(file_prefixes[0], sizeof(file_prefixes[0]),
                           locations[p] ? "world-binary-device" : "world-binary-host");
         test_scratch_path(file_prefixes[1], sizeof(file_prefixes[1]),
                           locations[p] ? "world-short-device" : "world-short-host");
         test_scratch_path(file_prefixes[2], sizeof(file_prefixes[2]),
                           locations[p] ? "world-missing-device" : "world-missing-host");
      }
      MPI_Bcast(file_prefixes, sizeof(file_prefixes), MPI_CHAR, 0, MPI_COMM_WORLD);
      char file[MAX_FILENAME_LENGTH + 32];
      snprintf(file, sizeof(file), "%s.%05d.bin", file_prefixes[0], rank);
      FILE *fp = fopen(file, "wb");
      ASSERT_NOT_NULL(fp);
      uint64_t header[8]     = {0};
      header[1]              = sizeof(double);
      header[5]              = (uint64_t)count;
      const double values[2] = {5.0, 5.0};
      ASSERT_EQ_SIZE(fwrite(header, sizeof(*header), 8, fp), 8);
      ASSERT_EQ_SIZE(fwrite(values, sizeof(*values), (size_t)count, fp), (size_t)count);
      ASSERT_EQ(fclose(fp), 0);
      if (!rank)
      {
         snprintf(file, sizeof(file), "%s.00000", file_prefixes[1]);
         fp = fopen(file, "w");
         ASSERT_NOT_NULL(fp);
         ASSERT_TRUE(fputs("0 -1\n", fp) >= 0);
         ASSERT_EQ(fclose(fp), 0);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const hypredrv_VectorSlot retained[] = {obj->vec_b, obj->vec_x0, obj->vec_x,
                                              obj->vec_xref};
      for (size_t f = 0; f < sizeof(file_prefixes) / sizeof(file_prefixes[0]); f++)
      {
         const char *selected  = rank == 1 ? file_prefixes[f] : prefix;
         size_t      path_size = strlen(selected) + 1;
         ASSERT_TRUE(path_size <= sizeof(obj->iargs->ls.rhs_filename) &&
                     path_size <= sizeof(obj->iargs->ls.x0_filename) &&
                     path_size <= sizeof(obj->iargs->ls.xref_filename));
         memcpy(obj->iargs->ls.rhs_filename, selected, path_size);
         memcpy(obj->iargs->ls.x0_filename, selected, path_size);
         memcpy(obj->iargs->ls.xref_filename, selected, path_size);
         for (size_t i = 0; i < sizeof(setters) / sizeof(setters[0]); i++)
         {
            ASSERT_TRUE(setters[i](obj, NULL) & ERROR_INVALID_VAL);
            const hypredrv_VectorSlot current[] = {obj->vec_b, obj->vec_x0, obj->vec_x,
                                                   obj->vec_xref};
            for (size_t v = 0; v < sizeof(retained) / sizeof(retained[0]); v++)
            {
               ASSERT_PTR_EQ(current[v].handle, retained[v].handle);
               ASSERT_EQ(current[v].owned, retained[v].owned);
            }
         }
      }
      snprintf(obj->iargs->ls.rhs_filename, sizeof(obj->iargs->ls.rhs_filename), "%s",
               prefix);
      obj->iargs->ls.x0_filename[0] = '\0';
      snprintf(obj->iargs->ls.xref_filename, sizeof(obj->iargs->ls.xref_filename), "%s",
               file_prefixes[0]);
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
      data = hypredrv_IJVectorData(obj->vec_xref.handle, HYPRE_MEMORY_HOST);
      for (int i = 0; i < count; i++) ASSERT_EQ_DOUBLE((double)data[i], 5.0, 0.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, (HYPRE_Vector)saved_rhs),
                ERROR_NONE);
      snprintf(obj->iargs->ls.xref_filename, sizeof(obj->iargs->ls.xref_filename), "%s",
               prefix);
      if (rank == 1)
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "injected reference construction failure");
      ASSERT_TRUE(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL) &
                  ERROR_HYPRE_INTERNAL);
      ASSERT_PTR_EQ(obj->vec_b.handle, saved_rhs);
      ASSERT_PTR_EQ(obj->vec_xref.handle, saved_rhs);
      ASSERT_EQ(HYPRE_GetError(), 0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetReferenceSolution(obj, NULL), ERROR_NONE);
      if (!rank)
         ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)obj->vec_b.handle),
                   ERROR_NONE);
      MPI_Barrier(MPI_COMM_WORLD);
      HYPRE_IJVector shifted = create_vector(MPI_COMM_WORLD, first + 2, count, 2.0);
      ASSERT_EQ(HYPREDRV_LinearSystemSetRHS(obj, (HYPRE_Vector)shifted), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemSetInitialGuess(obj, NULL), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemResetInitialGuess(obj), ERROR_NONE);
      ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &data), ERROR_NONE);
      if (count) ASSERT_EQ_DOUBLE((double)data[0], 0.0, 0.0);
      ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
      ASSERT_EQ(HYPRE_IJVectorDestroy(external), 0);
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
   RUN_TEST(test_failed_default_solution_preserves_borrowed_buffer);
   RUN_TEST(test_failed_guess_preserves_installed_vectors);
   RUN_TEST(test_reference_alias_survives_solution_replacement);
   RUN_TEST(test_previous_mode_preserves_borrowed_vectors);
   RUN_TEST(test_alias_ownership_transfers);
   RUN_TEST(test_input_validation_preserves_vectors);
   RUN_TEST(test_generated_and_file_guesses);
   RUN_TEST(test_supplied_guess_before_parsing);
   RUN_TEST(test_private_solution_operations_preserve_diagnostics);
   RUN_TEST(test_rhs_replacement_preserves_retained_aliases);
   RUN_TEST(test_known_borrowed_handles_keep_their_ownership);
   RUN_TEST(test_rhs_and_reference_validate_input_once);
   RUN_TEST(test_configured_rhs_and_reference_storage);
   RUN_TEST(test_configured_vectors_require_arguments);
   RUN_TEST(test_distributed_solution_replacement);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   MPI_Finalize();
   return 0;
}
