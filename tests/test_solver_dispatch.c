#include <mpi.h>
#include <stdlib.h>

#include "HYPRE.h"
#include "bicgstab.h"
#include "error.h"
#include "fgmres.h"
#include "gmres.h"
#include "pcg.h"
#include "precon.h"
#include "solver.h"
#include "test_helpers.h"

/* Forward declarations */
void PCGSetDefaultArgs(PCG_args *);
void GMRESSetDefaultArgs(GMRES_args *);
void FGMRESSetDefaultArgs(FGMRES_args *);
void BiCGSTABSetDefaultArgs(BiCGSTAB_args *);

static void
test_SolverCreate_all_cases(void)
{
   HYPRE_Initialize();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG */
   PCGSetDefaultArgs(&args.pcg);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES */
   GMRESSetDefaultArgs(&args.gmres);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES */
   FGMRESSetDefaultArgs(&args.fgmres);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB */
   BiCGSTABSetDefaultArgs(&args.bicgstab);
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   ASSERT_FALSE(ErrorCodeActive());
   SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   HYPRE_Finalize();
}

static void
test_SolverCreate_default_case(void)
{
   HYPRE_Initialize();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test default case with invalid enum value */
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, &solver);
   ASSERT_NULL(solver);
   /* Default case should not set error, just return NULL */

   HYPRE_Finalize();
}

static void
test_SolverDestroy_all_cases(void)
{
   HYPRE_Initialize();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Test PCG destroy */
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   /* Test GMRES destroy */
   GMRESSetDefaultArgs(&args.gmres);
   SolverCreate(MPI_COMM_SELF, SOLVER_GMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_GMRES, &solver);
   ASSERT_NULL(solver);

   /* Test FGMRES destroy */
   FGMRESSetDefaultArgs(&args.fgmres);
   SolverCreate(MPI_COMM_SELF, SOLVER_FGMRES, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_FGMRES, &solver);
   ASSERT_NULL(solver);

   /* Test BiCGSTAB destroy */
   BiCGSTABSetDefaultArgs(&args.bicgstab);
   SolverCreate(MPI_COMM_SELF, SOLVER_BICGSTAB, &args, &solver);
   ASSERT_NOT_NULL(solver);
   SolverDestroy(SOLVER_BICGSTAB, &solver);
   ASSERT_NULL(solver);

   HYPRE_Finalize();
}

static void
test_SolverDestroy_default_case(void)
{
   HYPRE_Initialize();

   solver_args args;
   HYPRE_Solver solver = NULL;

   /* Create a solver first */
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);
   ASSERT_NOT_NULL(solver);

   /* Test default case with invalid enum - should return early */
   SolverDestroy((solver_t)999, &solver);
   /* Solver should still exist since default case returns early */
   ASSERT_NOT_NULL(solver);

   /* Clean up properly */
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);

   HYPRE_Finalize();
}

static void
test_SolverDestroy_null_solver(void)
{
   HYPRE_Solver solver = NULL;

   /* Destroy with NULL should not crash */
   SolverDestroy(SOLVER_PCG, &solver);
   ASSERT_NULL(solver);
}

static void
test_SolverSetup_default_case(void)
{
   HYPRE_Initialize();

   HYPRE_IJMatrix M = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Create minimal matrix and vectors */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &M);
   HYPRE_IJMatrixSetObjectType(M, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(M);
   HYPRE_IJMatrixAssemble(M);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);

   /* Create a dummy precon */
   HYPRE_Precon precon = malloc(sizeof(hypre_Precon));
   ASSERT_NOT_NULL(precon);
   precon->main = NULL;

   HYPRE_Solver solver = NULL;
   solver_args  args;
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, (solver_t)999, precon, solver, M, b, x);
   /* Should return early without error */

   SolverDestroy(SOLVER_PCG, &solver);
   free(precon);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(M);
   HYPRE_Finalize();
}

static void
test_SolverApply_default_case(void)
{
   HYPRE_Initialize();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Create minimal matrix and vectors */
   HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, 0, 0, 0, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);
   HYPRE_IJMatrixAssemble(A);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);

   HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, 0, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);

   HYPRE_Solver solver = NULL;
   solver_args  args;
   PCGSetDefaultArgs(&args.pcg);
   SolverCreate(MPI_COMM_SELF, SOLVER_PCG, &args, &solver);

   /* Test default case with invalid solver enum */
   ErrorCodeResetAll();
   SolverApply((solver_t)999, solver, A, b, x);
   /* Should return early without error */

   SolverDestroy(SOLVER_PCG, &solver);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);
   HYPRE_Finalize();
}

static void
test_SolverCreate_default_case_comprehensive(void)
{
   HYPRE_Initialize();

   solver_args args;

   /* Test multiple invalid solver types to exercise default case */
   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)999, &args, NULL);
   ASSERT_TRUE(ErrorCodeActive()); /* Should set error for invalid solver */

   ErrorCodeResetAll();
   SolverCreate(MPI_COMM_SELF, (solver_t)-1, &args, NULL);
   ASSERT_TRUE(ErrorCodeActive()); /* Should set error for invalid solver */

   HYPRE_Finalize();
}

static void
test_SolverApply_error_cases(void)
{
   HYPRE_Initialize();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, NULL, A, b, x);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL matrix */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, NULL, b, x);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL vectors */
   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, NULL, x);
   ASSERT_TRUE(ErrorCodeActive());

   ErrorCodeResetAll();
   SolverApply(SOLVER_PCG, (HYPRE_Solver)1, A, b, NULL);
   ASSERT_TRUE(ErrorCodeActive());

   HYPRE_Finalize();
}

static void
test_SolverSetup_error_cases(void)
{
   HYPRE_Initialize();

   HYPRE_IJMatrix A = NULL;
   HYPRE_IJVector b = NULL, x = NULL;

   /* Test with NULL solver */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, NULL, A, b, x);
   ASSERT_TRUE(ErrorCodeActive());

   /* Test with NULL matrix */
   ErrorCodeResetAll();
   SolverSetup(PRECON_NONE, SOLVER_PCG, NULL, (HYPRE_Solver)1, NULL, b, x);
   ASSERT_TRUE(ErrorCodeActive());

   HYPRE_Finalize();
}

int
main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   RUN_TEST(test_SolverCreate_all_cases);
   RUN_TEST(test_SolverCreate_default_case);
   RUN_TEST(test_SolverDestroy_all_cases);
   RUN_TEST(test_SolverDestroy_default_case);
   RUN_TEST(test_SolverDestroy_null_solver);
   RUN_TEST(test_SolverSetup_default_case);
   RUN_TEST(test_SolverApply_default_case);

   MPI_Finalize();
   return 0;
}
