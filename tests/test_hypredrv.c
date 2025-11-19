#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HYPREDRV.h"
#include "args.h"
#include "containers.h"
#include "error.h"
#include "test_helpers.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

extern uint32_t HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t obj,
                                                         int        num_local_blocks,
                                                         int        num_dof_types);

struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix mat_A;
   HYPRE_IJMatrix mat_M;
   HYPRE_IJVector vec_b;
   HYPRE_IJVector vec_x;
   HYPRE_IJVector vec_x0;
   HYPRE_IJVector vec_xref;

   HYPRE_Precon precon;
   HYPRE_Solver solver;
};

static void
reset_state(void)
{
   ErrorCodeResetAll();
   HYPREDRV_Finalize();
   ErrorCodeResetAll();
}

static void
test_requires_initialization_guard(void)
{
   reset_state();

   HYPREDRV_t obj = NULL;
   uint32_t   code;

   code = HYPREDRV_Create(MPI_COMM_SELF, &obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   ASSERT_NULL(obj);
   ErrorCodeResetAll();

   code = HYPREDRV_SetLibraryMode(obj);
   ASSERT_TRUE(code & ERROR_HYPREDRV_NOT_INITIALIZED);
   ErrorCodeResetAll();
}

static void
test_initialize_and_finalize_idempotent(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE); /* second call should be no-op */
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE); /* already finalized */
}

static void
test_create_parse_and_destroy(void)
{
   reset_state();

   ASSERT_EQ(HYPREDRV_Initialize(), ERROR_NONE);

   HYPREDRV_t obj = NULL;
   ASSERT_EQ(HYPREDRV_Create(MPI_COMM_SELF, &obj), ERROR_NONE);
   ASSERT_NOT_NULL(obj);

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   snprintf(matrix_path, sizeof(matrix_path), "%s/data/ps3d10pt7/np1/IJ.out.A",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, sizeof(rhs_path), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);

   char yaml_config[2 * PATH_MAX + 256];
   int  yaml_len = snprintf(yaml_config, sizeof(yaml_config),
                            "general:\n"
                            "  statistics: off\n"
                            "linear_system:\n"
                            "  matrix_filename: %s\n"
                            "  rhs_filename: %s\n"
                            "solver:\n"
                            "  pcg:\n"
                            "    max_iter: 5\n"
                            "preconditioner:\n"
                            "  amg:\n"
                            "    print_level: 0\n",
                            matrix_path, rhs_path);
   ASSERT_TRUE(yaml_len > 0 && (size_t)yaml_len < sizeof(yaml_config));

   char *argv[] = {yaml_config};
   ASSERT_EQ(HYPREDRV_InputArgsParse(1, argv, obj), ERROR_NONE);
   ASSERT_EQ(ErrorCodeGet(), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_SetLibraryMode(obj), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_SetGlobalOptions(obj), ERROR_NONE);

   ASSERT_EQ(HYPREDRV_LinearSystemBuild(obj), ERROR_NONE);

   struct hypredrv_struct *state = (struct hypredrv_struct *)obj;

   HYPRE_Complex *sol_data = NULL;
   HYPRE_Complex *rhs_data = NULL;
   ASSERT_EQ(HYPREDRV_LinearSystemGetSolutionValues(obj, &sol_data), ERROR_NONE);
   ASSERT_NOT_NULL(sol_data);
   ASSERT_EQ(HYPREDRV_LinearSystemGetRHSValues(obj, &rhs_data), ERROR_NONE);
   ASSERT_NOT_NULL(rhs_data);

   IntArray *original_dofmap = state->dofmap;
   if (!original_dofmap || original_dofmap->size == 0)
   {
      const int bootstrap_map[] = {0, 1, 2, 3};
      ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, 4, bootstrap_map), ERROR_NONE);
      original_dofmap = state->dofmap;
   }

   ASSERT_NOT_NULL(original_dofmap);
   const size_t dofmap_size = original_dofmap->size;
   ASSERT_TRUE(dofmap_size > 0);

   int *manual_dofmap = malloc(dofmap_size * sizeof(int));
   ASSERT_NOT_NULL(manual_dofmap);
   for (size_t i = 0; i < dofmap_size; i++)
   {
      manual_dofmap[i] = (int)((i * 3) % 11);
   }

   ASSERT_EQ(HYPREDRV_LinearSystemSetDofmap(obj, (int)dofmap_size, manual_dofmap),
             ERROR_NONE);
   IntArray *custom_map = state->dofmap;
   ASSERT_NOT_NULL(custom_map);
   ASSERT_EQ(custom_map->size, dofmap_size);
   ASSERT_EQ(custom_map->data[0], manual_dofmap[0]);
   if (dofmap_size > 5)
   {
      ASSERT_EQ(custom_map->data[5], manual_dofmap[5]);
   }
   state->dofmap = original_dofmap;
   IntArrayDestroy(&custom_map);
   free(manual_dofmap);

   ASSERT_EQ(HYPREDRV_LinearSystemSetInterleavedDofmap(obj, 3, 2), ERROR_NONE);
   IntArray *interleaved_map = state->dofmap;
   ASSERT_NOT_NULL(interleaved_map);
   ASSERT_EQ(interleaved_map->size, (size_t)(3 * 2));
   ASSERT_EQ(interleaved_map->data[0], 0);
   ASSERT_EQ(interleaved_map->data[1], 1);
   ASSERT_EQ(interleaved_map->data[2], 0);
   state->dofmap = original_dofmap;
   IntArrayDestroy(&interleaved_map);

   ASSERT_EQ(HYPREDRV_LinearSystemSetContiguousDofmap(obj, 3, 2), ERROR_NONE);
   IntArray *contiguous_map = state->dofmap;
   ASSERT_NOT_NULL(contiguous_map);
   ASSERT_EQ(contiguous_map->size, (size_t)(3 * 2));
   ASSERT_EQ(contiguous_map->data[0], 0);
   ASSERT_EQ(contiguous_map->data[1], 1);
   ASSERT_EQ(contiguous_map->data[2], 2);
   state->dofmap = original_dofmap;
   IntArrayDestroy(&contiguous_map);

   ASSERT_EQ(HYPREDRV_PreconCreate(obj), ERROR_NONE);
   ASSERT_NOT_NULL(state->precon);
   ASSERT_EQ(HYPREDRV_PreconSetup(obj), ERROR_NONE);
   ASSERT_EQ(
      HYPREDRV_PreconApply(obj, (HYPRE_Vector)state->vec_b, (HYPRE_Vector)state->vec_x),
      ERROR_NONE);

   ErrorCodeSet(ERROR_INVALID_KEY);
   HYPREDRV_ErrorCodeDescribe(ErrorCodeGet());
   ErrorCodeResetAll();

   ASSERT_EQ(ErrorCodeGet(), ERROR_NONE);
   ASSERT_EQ(HYPREDRV_Destroy(&obj), ERROR_NONE);
   ASSERT_NULL(obj);

   /* Destroy again to exercise unknown-object branch */
   uint32_t code = HYPREDRV_Destroy(&obj);
   ASSERT_TRUE(code & ERROR_UNKNOWN_HYPREDRV_OBJ);
   ErrorCodeResetAll();

   ASSERT_EQ(HYPREDRV_Finalize(), ERROR_NONE);
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_requires_initialization_guard);
   RUN_TEST(test_initialize_and_finalize_idempotent);
   RUN_TEST(test_create_parse_and_destroy);

   MPI_Finalize();
   return 0;
}
