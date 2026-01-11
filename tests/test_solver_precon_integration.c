#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "HYPRE.h"
#include "HYPREDRV.h"
#include "error.h"
#include "test_helpers.h"

#ifndef HYPREDRIVE_SOURCE_DIR
#define HYPREDRIVE_SOURCE_DIR "."
#endif

static void
test_solver_precon_combination(const char *solver_name, const char *precon_name)
{
   HYPREDRV_Initialize();

   HYPREDRV_t obj = NULL;
   HYPREDRV_Create(MPI_COMM_SELF, &obj);

   char matrix_path[PATH_MAX];
   char rhs_path[PATH_MAX];
   snprintf(matrix_path, sizeof(matrix_path), "%s/data/ps3d10pt7/np1/IJ.out.A",
            HYPREDRIVE_SOURCE_DIR);
   snprintf(rhs_path, sizeof(rhs_path), "%s/data/ps3d10pt7/np1/IJ.out.b",
            HYPREDRIVE_SOURCE_DIR);

   char yaml_config[2 * PATH_MAX + 512];
   snprintf(yaml_config, sizeof(yaml_config),
            "general:\n"
            "  statistics: off\n"
            "linear_system:\n"
            "  matrix_filename: %s\n"
            "  rhs_filename: %s\n"
            "solver: %s\n"
            "preconditioner: %s\n",
            matrix_path, rhs_path, solver_name, precon_name);

   char *argv[] = {yaml_config};
   ErrorCodeResetAll();
   HYPREDRV_InputArgsParse(1, argv, obj);

   if (ErrorCodeActive())
   {
      HYPREDRV_Destroy(&obj);
      HYPREDRV_Finalize();
      return; /* Skip invalid combinations */
   }

   HYPREDRV_SetGlobalOptions(obj);
   HYPREDRV_LinearSystemBuild(obj);

   /* Test create/setup/apply/destroy cycle */
   HYPREDRV_PreconCreate(obj);
   HYPREDRV_LinearSolverCreate(obj);
   HYPREDRV_LinearSolverSetup(obj);
   HYPREDRV_LinearSolverApply(obj);
   HYPREDRV_PreconDestroy(obj);
   HYPREDRV_LinearSolverDestroy(obj);

   HYPREDRV_Destroy(&obj);
   HYPREDRV_Finalize();
}

static void
test_all_solver_precon_combinations(void)
{
   const char *solvers[] = {"pcg", "gmres", "fgmres", "bicgstab"};
   const char *precons[] = {"amg", "mgr", "ilu", "fsai"};

   for (size_t i = 0; i < sizeof(solvers) / sizeof(solvers[0]); i++)
   {
      for (size_t j = 0; j < sizeof(precons) / sizeof(precons[0]); j++)
      {
         test_solver_precon_combination(solvers[i], precons[j]);
      }
   }
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_all_solver_precon_combinations);

   MPI_Finalize();
   return 0;
}
