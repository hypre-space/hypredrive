/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/krylov.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h" // for hypre_ParVectorOwnsPartitioning (older hypre)
#include "internal/error.h"
#include "solver_ops.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 1)
#include "_hypre_utilities.h" // for hypre_Solver
#endif

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
NestedKrylovSetupThunk(void *solver_v, void *A, void *b, void *x)
{
   return hypredrv_NestedKrylovSetup((HYPRE_Solver)solver_v, (HYPRE_Matrix)A,
                                     (HYPRE_Vector)b, (HYPRE_Vector)x);
}

static HYPRE_Int
NestedKrylovSolveThunk(void *solver_v, void *A, void *b, void *x)
{
   return hypredrv_NestedKrylovSolve((HYPRE_Solver)solver_v, (HYPRE_Matrix)A,
                                     (HYPRE_Vector)b, (HYPRE_Vector)x);
}

static HYPRE_Int
NestedKrylovDestroyThunk(void *solver_v)
{
   hypredrv_NestedKrylovDestroy((NestedKrylov_args *)solver_v);
   return 0;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
NestedKrylovParsePrecon(NestedKrylov_args *args, YAMLnode *precon_node)
{
   YAML_NODE_SET_VALID(precon_node);
   if (args->has_precon)
   {
      hypredrv_PreconArgsDestroyOwnedConfig(args->precon_method, &args->precon);
      args->has_precon = 0;
   }

   if (strcmp(precon_node->val, "") != 0)
   {
      if (!hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_PreconGetValidTypeIntMap(),
                                                    precon_node->val))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Unknown nested preconditioner type: '%s'",
                              precon_node->val);
         precon_node->avail_vals = hypredrv_PreconGetValidTypeIntMap();
         YAML_NODE_SET_INVALID_VAL(precon_node);
         return;
      }
      args->precon_method = (precon_t)hypredrv_StrIntMapArrayGetImage(
         hypredrv_PreconGetValidTypeIntMap(), precon_node->val);
      args->has_precon = 1;
      hypredrv_PreconArgsSetDefaultsForName(args->precon_method, precon_node->val,
                                            &args->precon);
      return;
   }

   if (!precon_node->children)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_PRECON);
      hypredrv_ErrorMsgAdd("Nested preconditioner type is missing");
      YAML_NODE_SET_INVALID_KEY(precon_node);
      return;
   }

   if (precon_node->children->next)
   {
      hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
      hypredrv_ErrorMsgAddExtraKey(precon_node->children->next->key);
      YAML_NODE_SET_INVALID_KEY(precon_node->children->next);
      return;
   }

   YAMLnode *type_node = precon_node->children;
   if (!hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_PreconGetValidTypeIntMap(),
                                                 type_node->key))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
      hypredrv_ErrorMsgAdd("Unknown nested preconditioner type: '%s'", type_node->key);
      YAML_NODE_SET_INVALID_KEY(type_node);
      return;
   }

   args->precon_method = (precon_t)hypredrv_StrIntMapArrayGetImage(
      hypredrv_PreconGetValidTypeIntMap(), type_node->key);

   args->has_precon = 1;
   hypredrv_PreconArgsSetDefaultsForMethod(args->precon_method, &args->precon);
   hypredrv_PreconSetArgsFromYAML(&args->precon, precon_node);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static const SolverOps *
NestedKrylovSolverOps(solver_t method)
{
   const SolverOps *ops = hypredrv_SolverOpsLookup(method);
   if (!ops)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
   }
   return ops;
}

static void
NestedKrylovSetPrecond(solver_t solver_method, HYPRE_Solver solver,
                       precon_t precon_method, HYPRE_Precon precon)
{ /* GCOVR_EXCL_BR_LINE */
   HYPRE_PtrToParSolverFcn setup = NULL;
   HYPRE_PtrToParSolverFcn solve = NULL;

   if (!precon ||
       precon_method ==
          PRECON_NONE) /* GCOVR_EXCL_BR_LINE */ /* precon live when has_precon */
   {
      return;
   }

   hypredrv_PreconGetCallbacks(precon_method, &setup, &solve);
   if (!setup || !solve)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Nested Krylov preconditioner method not supported");
      return;
   }

   const SolverOps *ops = NestedKrylovSolverOps(solver_method);
   if (ops)
   {
      ops->set_precond(solver, hypredrv_PreconKrylovSolve, hypredrv_PreconKrylovSetup,
                       (HYPRE_Solver)precon);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
NestedKrylovCreateSetupVector(HYPRE_Matrix A, HYPRE_ParVector *vector_ptr)
{
   MPI_Comm     comm        = MPI_COMM_NULL;
   HYPRE_BigInt global_rows = 0, global_cols = 0;
   HYPRE_BigInt row_start = 0, row_end = -1, col_start = 0, col_end = -1;
   HYPRE_Int    rc = 0;

   if (!A || !vector_ptr)
   {
      return 1;
   }

   rc |= HYPRE_ParCSRMatrixGetComm((HYPRE_ParCSRMatrix)A, &comm);
   rc |= HYPRE_ParCSRMatrixGetDims((HYPRE_ParCSRMatrix)A, &global_rows, &global_cols);
   rc |= HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)A, &row_start, &row_end,
                                         &col_start, &col_end);
   if (!rc)
   {
      /* The vector takes this rank's row range [first, last + 1]. Older hypre
       * keeps (and later frees) the partitioning array it is given; newer
       * releases copy the two values. */
#if defined(hypre_ParVectorOwnsPartitioning)
      HYPRE_BigInt *partitioning = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
#else
      HYPRE_BigInt  local_partitioning[2];
      HYPRE_BigInt *partitioning = local_partitioning;
#endif
      partitioning[0] = row_start;
      partitioning[1] = row_end + 1;
      rc |= HYPRE_ParVectorCreate(comm, global_rows, partitioning, vector_ptr);
      rc |= hypre_ParVectorInitialize_v2(
         (hypre_ParVector *)*vector_ptr,
         hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)A));
   }

   if (rc && *vector_ptr)
   {
      HYPRE_ParVectorDestroy(*vector_ptr);
      *vector_ptr = NULL;
   }

   return rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovSetDefaultArgs(NestedKrylov_args *args)
{
   if (!args)
   {
      return;
   }

   args->setup         = NestedKrylovSetupThunk;
   args->solve         = NestedKrylovSolveThunk;
   args->destroy       = NestedKrylovDestroyThunk;
   args->is_setup      = 0;
   args->is_set        = 0;
   args->solver_method = SOLVER_GMRES;
   args->has_precon    = 0;
   args->precon_method = PRECON_NONE;
   args->base_solver   = NULL;
   args->precon_obj    = NULL;
   args->setup_matrix  = NULL;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovSetArgsFromYAML(NestedKrylov_args *args, YAMLnode *solver_node)
{
   if (!args || !solver_node)
   {
      return;
   }

   if (!hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_SolverGetValidTypeIntMap(),
                                                 solver_node->key))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
      hypredrv_ErrorMsgAdd("Unknown nested solver type: '%s'", solver_node->key);
      YAML_NODE_SET_INVALID_KEY(solver_node);
      return;
   }

   /* Release runtime objects while their original solver method is still set. */
   hypredrv_NestedKrylovDestroy(args);
   args->is_set        = 1;
   args->solver_method = (solver_t)hypredrv_StrIntMapArrayGetImage(
      hypredrv_SolverGetValidTypeIntMap(), solver_node->key);

   YAMLnode *precon_node = hypredrv_YAMLnodeFindChildByKey(solver_node, "preconditioner");
   if (precon_node)
   {
      NestedKrylovParsePrecon(args, precon_node);
   }
   hypredrv_SolverSetMethodArgs(args->solver_method, &args->solver, solver_node,
                                precon_node);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovCreate(MPI_Comm comm, NestedKrylov_args *args, IntArray *dofmap,
                            HYPRE_IJVector vec_nn, HYPRE_Solver *solver_ptr)
{
   HYPRE_Solver base_solver = NULL;

   /* Sanity check */
   if (!args || !solver_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("NestedKrylovCreateWrapped: invalid arguments");
      return;
   }

   /* Return early if input arguments are not set */
   if (!args->is_set)
   {
      *solver_ptr = NULL;
      return;
   }

   if (args->has_precon && args->precon_method == PRECON_MGR &&
       !hypredrv_MGRValidateOuterSolver(&args->precon.mgr, (int)args->solver_method,
                                        "Nested MGR"))
   {
      *solver_ptr = NULL;
      return;
   }

   /* Create preconditioner object if needed */
   if (args->has_precon)
   {
      hypredrv_PreconCreate(comm, args->precon_method, &args->precon, dofmap, vec_nn,
                            &args->precon_obj, NULL, 0, NULL); /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_LINE */ /* PreconCreate failure
                                                                  injection */
      {
         return;
      }
   }

   const SolverOps *ops = NestedKrylovSolverOps(args->solver_method);
   if (!ops)
   {
      return;
   }
   ops->create(comm, &args->solver, &base_solver);

   /* Do not proceed to SetPrecond with a NULL base solver: a failed create would
    * otherwise pass NULL into hypre's SetPrecond. */
   if (hypredrv_ErrorCodeActive() || !base_solver)
   {
      return;
   }

   /* Attach preconditioner to the solver object */
   if (args->has_precon)
   {
      NestedKrylovSetPrecond(args->solver_method, base_solver, args->precon_method,
                             args->precon_obj);                /* GCOVR_EXCL_BR_LINE */
      if (hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_LINE */ /* SetPrecond error path */
      {
         /* Reclaim the base solver we just created so it does not leak (args->base_solver
          * is not set on this path, so destroy it directly). */
         ops->destroy(base_solver);
         return;
      }
   }

   /* Set output pointer */
   *solver_ptr       = base_solver;
   args->base_solver = base_solver;

   return;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_NestedKrylovSetup(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                           HYPRE_Vector x)
{
   NestedKrylov_args *args    = (NestedKrylov_args *)solver;
   HYPRE_ParVector    setup_b = NULL;
   HYPRE_ParVector    setup_x = NULL;
   HYPRE_Int          rc      = 0;

   if (!args || !args->base_solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov setup called with invalid solver");
      return 1;
   }

   const SolverOps *ops = NestedKrylovSolverOps(args->solver_method);
   if (!ops)
   {
      return 1;
   }

   if (!hypredrv_PreconValidateSetup(args->precon_obj, (HYPRE_ParCSRMatrix)A))
   {
      return 1;
   }

   if (!b)
   {
      rc = NestedKrylovCreateSetupVector(A, &setup_b);
      b  = (HYPRE_Vector)setup_b;
   }
   if (!rc && !x)
   {
      rc = NestedKrylovCreateSetupVector(A, &setup_x);
      x  = (HYPRE_Vector)setup_x;
   }

   if (!rc)
   {
      rc = ops->setup(args->base_solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                      (HYPRE_ParVector)x);
   }
   if (setup_b)
   {
      HYPRE_ParVectorDestroy(setup_b);
   }
   if (setup_x)
   {
      HYPRE_ParVectorDestroy(setup_x);
   }
   if (!rc)
   {
      args->is_setup     = 1;
      args->setup_matrix = A;
   }
   return rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_NestedKrylovSolve(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                           HYPRE_Vector x)
{
   NestedKrylov_args *args = (NestedKrylov_args *)solver;
   HYPRE_Int          solve_rc;
   HYPRE_Int          hypre_error;

   if (!args || !args->base_solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov solve called with invalid solver");
      return 1;
   }

   if (!A)
   {
      A = args->setup_matrix;
   }
   if (!A)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov solve called before setup matrix was set");
      return 1;
   }

   const SolverOps *ops = NestedKrylovSolverOps(args->solver_method);
   if (!ops)
   {
      return 1;
   }
   solve_rc    = ops->solve(args->base_solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x);
   hypre_error = HYPRE_GetError();

   /* Nested Krylov objects are only used as inexact MGR smoothers/coarse solves.
    * HYPRE_ERROR_CONV from their internal Krylov iteration should not poison the
    * outer solve; leave harder failures intact. */    /* GCOVR_EXCL_BR_LINE */
   if (hypre_error &&
       !(hypre_error &
         ~HYPRE_ERROR_CONV)) /* GCOVR_EXCL_BR_LINE */ /* HYPRE conv flags hard to
                                                         stabilize in unit tests */
   {
      hypredrv_HypreClearConvergenceErrors();
      return 0;
   }

   return solve_rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovDestroy(NestedKrylov_args *args)
{
   if (!args)
   {
      return;
   }

   if (args->base_solver)
   {
      const SolverOps *ops = NestedKrylovSolverOps(args->solver_method);
      if (ops)
      {
         ops->destroy(args->base_solver);
      }
      args->base_solver = NULL;
   }
   args->is_setup     = 0;
   args->setup_matrix = NULL;

   if (args->precon_obj)
   {
      /* Pass the preconditioner's real setup state. Forcing is_setup=1 here made
       * PreconDestroy take the "was setup" branch even when the nested (e.g. MGR)
       * preconditioner was created but never set up, which skips reclamation of
       * still-hypredrive-owned detached component handles and leaks them. When the
       * precon truly was set up, is_setup is already 1, so behavior is unchanged. */
      hypredrv_PreconDestroy(args->precon_method, &args->precon, &args->precon_obj, NULL,
                             0);
      args->precon_obj = NULL;
   }

   /* Nested Krylov preconditioners do not have an outer HYPREDRV object driving
    * reuse policy. Once the nested preconditioner object is destroyed, clear any
    * method-specific runtime state so subsequent creates start from a clean
    * configuration unless the caller explicitly keeps the nested object alive. */
   hypredrv_PreconArgsDestroyRuntimeState(args->precon_method, &args->precon);
}
