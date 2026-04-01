/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/krylov.h"
#include "internal/bicgstab.h"
#include "internal/error.h"
#include "internal/fgmres.h"
#include "internal/gmres.h"
#include "internal/pcg.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 1)
#include "_hypre_utilities.h" // for hypre_Solver
#endif

#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
static HYPRE_Int
FSAISetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
FSAISolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
              HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define LOCAL_FSAI_SETUP FSAISetupStub
#define LOCAL_FSAI_SOLVE FSAISolveStub
#else
#define LOCAL_FSAI_SETUP HYPRE_FSAISetup
#define LOCAL_FSAI_SOLVE HYPRE_FSAISolve
#endif

#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
static HYPRE_Int
ILUSetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
             HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
ILUSolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
             HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define LOCAL_ILU_SETUP ILUSetupStub
#define LOCAL_ILU_SOLVE ILUSolveStub
#else
#define LOCAL_ILU_SETUP HYPRE_ILUSetup
#define LOCAL_ILU_SOLVE HYPRE_ILUSolve
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
NestedKrylovDetachPrecon(YAMLnode *solver_node, YAMLnode **precon_node_out,
                         YAMLnode **precon_prev_out, YAMLnode **precon_next_out)
{
   YAMLnode *prev  = NULL;
   YAMLnode *child = NULL;

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!solver_node || !precon_node_out || !precon_prev_out || !precon_next_out)
   /* GCOVR_EXCL_BR_STOP */ /* valid pointers from SetArgsFromYAML */
   {
      return;
   }

   *precon_node_out = NULL;
   *precon_prev_out = NULL;
   *precon_next_out = NULL;

   for (child = solver_node->children; child != NULL; child = child->next)
   {
      if (!strcmp(child->key, "preconditioner"))
      {
         *precon_node_out = child;
         *precon_prev_out = prev;
         *precon_next_out = child->next;
         if (prev)
         {
            prev->next = child->next;
         }
         else
         {
            solver_node->children = child->next;
         }
         child->next = NULL;
         return;
      }
      prev = child;
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
NestedKrylovRestorePrecon(YAMLnode *solver_node, YAMLnode *precon_node,
                          YAMLnode *precon_prev, YAMLnode *precon_next)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!solver_node || !precon_node)
   /* GCOVR_EXCL_BR_STOP */ /* !solver_node unreachable after SetArgsFromYAML */
   {
      return;
   }

   if (precon_prev)
   {
      precon_prev->next = precon_node;
      precon_node->next = precon_next;
   }
   else
   {
      precon_node->next     = solver_node->children;
      solver_node->children = precon_node;
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
NestedKrylovParsePrecon(NestedKrylov_args *args, YAMLnode *precon_node)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!args || !precon_node)
   /* GCOVR_EXCL_BR_STOP */ /* only called with live args and precon node */
   {
      return;
   }

   YAML_NODE_SET_VALID(precon_node);

   if (strcmp(precon_node->val, "") != 0)
   {
      if (!hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_PreconGetValidTypeIntMap(),
                                                    precon_node->val))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Unknown nested preconditioner type: '%s'",
                              precon_node->val);
         YAML_NODE_SET_INVALID_VAL(precon_node);
         return;
      }
      args->precon_method = (precon_t)hypredrv_StrIntMapArrayGetImage(
         hypredrv_PreconGetValidTypeIntMap(), precon_node->val);
      args->has_precon = 1;
      hypredrv_PreconArgsSetDefaultsForMethod(args->precon_method, &args->precon);
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

static void
NestedKrylovSetPrecond(solver_t solver_method, HYPRE_Solver solver,
                       precon_t precon_method, HYPRE_Precon precon)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (!precon || precon_method == PRECON_NONE)
   /* GCOVR_EXCL_BR_STOP */ /* precon live when has_precon */
   {
      return;
   }

   HYPRE_PtrToParSolverFcn setup_ptrs[] = {
      HYPRE_BoomerAMGSetup,
      HYPRE_MGRSetup,
      LOCAL_ILU_SETUP,
      LOCAL_FSAI_SETUP,
   };
   HYPRE_PtrToParSolverFcn solve_ptrs[] = {
      HYPRE_BoomerAMGSolve,
      HYPRE_MGRSolve,
      LOCAL_ILU_SOLVE,
      LOCAL_FSAI_SOLVE,
   };

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSetPrecond(solver, solve_ptrs[precon_method],
                                   setup_ptrs[precon_method], precon->main);
         break;
      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                     setup_ptrs[precon_method], precon->main);
         break;
      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSetPrecond(solver, solve_ptrs[precon_method],
                                         setup_ptrs[precon_method], precon->main);
         break;
      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSetPrecond(solver, solve_ptrs[precon_method],
                                        setup_ptrs[precon_method], precon->main);
         break;
      /* GCOVR_EXCL_BR_START */ /* invalid solver_t without memory corruption */
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         break;
         /* GCOVR_EXCL_BR_STOP */
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
NestedKrylovBaseSolverSetup(solver_t solver_method, HYPRE_Solver solver, HYPRE_Matrix A,
                            HYPRE_Vector b, HYPRE_Vector x)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         return HYPRE_ParCSRPCGSetup(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                     (HYPRE_ParVector)x);
      case SOLVER_GMRES:
         return HYPRE_ParCSRGMRESSetup(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                       (HYPRE_ParVector)x);
      case SOLVER_FGMRES:
         return HYPRE_ParCSRFlexGMRESSetup(solver, (HYPRE_ParCSRMatrix)A,
                                           (HYPRE_ParVector)b, (HYPRE_ParVector)x);
      case SOLVER_BICGSTAB:
         return HYPRE_ParCSRBiCGSTABSetup(solver, (HYPRE_ParCSRMatrix)A,
                                          (HYPRE_ParVector)b, (HYPRE_ParVector)x);
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         return 1;
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
NestedKrylovBaseSolverSolve(solver_t solver_method, HYPRE_Solver solver, HYPRE_Matrix A,
                            HYPRE_Vector b, HYPRE_Vector x)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         return HYPRE_ParCSRPCGSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                     (HYPRE_ParVector)x);
      case SOLVER_GMRES:
         return HYPRE_ParCSRGMRESSolve(solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                                       (HYPRE_ParVector)x);
      case SOLVER_FGMRES:
         return HYPRE_ParCSRFlexGMRESSolve(solver, (HYPRE_ParCSRMatrix)A,
                                           (HYPRE_ParVector)b, (HYPRE_ParVector)x);
      case SOLVER_BICGSTAB:
         return HYPRE_ParCSRBiCGSTABSolve(solver, (HYPRE_ParCSRMatrix)A,
                                          (HYPRE_ParVector)b, (HYPRE_ParVector)x);
      /* GCOVR_EXCL_BR_START */ /* invalid solver_t without memory corruption */
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         return 1;
         /* GCOVR_EXCL_BR_STOP */
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
NestedKrylovBaseSolverDestroy(solver_t solver_method, HYPRE_Solver solver)
{
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGDestroy(solver);
         break;
      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESDestroy(solver);
         break;
      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESDestroy(solver);
         break;
      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABDestroy(solver);
         break;
      /* GCOVR_EXCL_BR_START */ /* invalid solver_t without memory corruption */
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         break;
         /* GCOVR_EXCL_BR_STOP */
   }
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
   args->is_set        = 0;
   args->solver_method = SOLVER_GMRES;
   args->has_precon    = 0;
   args->precon_method = PRECON_NONE;
   args->base_solver   = NULL;
   args->precon_obj    = NULL;
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

   args->is_set        = 1;
   args->solver_method = (solver_t)hypredrv_StrIntMapArrayGetImage(
      hypredrv_SolverGetValidTypeIntMap(), solver_node->key);
   hypredrv_SolverArgsSetDefaultsForMethod(args->solver_method, &args->solver);

   YAMLnode *precon_node = NULL;
   YAMLnode *precon_prev = NULL;
   YAMLnode *precon_next = NULL;
   NestedKrylovDetachPrecon(solver_node, &precon_node, &precon_prev, &precon_next);

   if (precon_node)
   {
      NestedKrylovParsePrecon(args, precon_node);
   }

   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (args->solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         hypredrv_PCGSetArgs(&args->solver.pcg, solver_node);
         break;
      case SOLVER_GMRES:
         hypredrv_GMRESSetArgs(&args->solver.gmres, solver_node);
         break;
      case SOLVER_FGMRES:
         hypredrv_FGMRESSetArgs(&args->solver.fgmres, solver_node);
         break;
      case SOLVER_BICGSTAB:
         hypredrv_BiCGSTABSetArgs(&args->solver.bicgstab, solver_node);
         break;
      /* GCOVR_EXCL_BR_START */ /* invalid solver_t without memory corruption */
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         break;
         /* GCOVR_EXCL_BR_STOP */
   }

   NestedKrylovRestorePrecon(solver_node, precon_node, precon_prev, precon_next);
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

   /* Create preconditioner object if needed */
   if (args->has_precon)
   {
      hypredrv_PreconCreate(args->precon_method, &args->precon, dofmap, vec_nn,
                            &args->precon_obj);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (hypredrv_ErrorCodeActive())
      /* GCOVR_EXCL_BR_STOP */ /* PreconCreate failure injection */
      {
         return;
      }
   }

   /* Create solver object always */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   switch (args->solver_method)
   /* GCOVR_EXCL_BR_STOP */ /* default arm excluded below */
   {
      case SOLVER_PCG:
         hypredrv_PCGCreate(comm, &args->solver.pcg, &base_solver);
         break;

      case SOLVER_GMRES:
         hypredrv_GMRESCreate(comm, &args->solver.gmres, &base_solver);
         break;

      case SOLVER_FGMRES:
         hypredrv_FGMRESCreate(comm, &args->solver.fgmres, &base_solver);
         break;

      case SOLVER_BICGSTAB:
         hypredrv_BiCGSTABCreate(comm, &args->solver.bicgstab, &base_solver);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         return;
   }

   /* Attach preconditioner to the solver object */
   if (args->has_precon)
   {
      NestedKrylovSetPrecond(args->solver_method, base_solver, args->precon_method,
                             args->precon_obj);
      /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
      if (hypredrv_ErrorCodeActive()) /* GCOVR_EXCL_BR_STOP */ /* SetPrecond error path */
      {
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
   NestedKrylov_args *args = (NestedKrylov_args *)solver;

   if (!args || !args->base_solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov setup called with invalid solver");
      return 1;
   }

   return NestedKrylovBaseSolverSetup(args->solver_method, args->base_solver, A, b, x);
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

   solve_rc =
      NestedKrylovBaseSolverSolve(args->solver_method, args->base_solver, A, b, x);
   hypre_error = HYPRE_GetError();

   /* Nested Krylov objects are only used as inexact MGR smoothers/coarse solves.
    * HYPRE_ERROR_CONV from their internal Krylov iteration should not poison the
    * outer solve; leave harder failures intact. */
   /* GCOVR_EXCL_BR_START */ /* low-signal branch under CI */
   if (hypre_error && !(hypre_error & ~HYPRE_ERROR_CONV))
   /* GCOVR_EXCL_BR_STOP */ /* HYPRE conv flags hard to stabilize in unit tests */
   {
      HYPRE_ClearAllErrors();
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
      NestedKrylovBaseSolverDestroy(args->solver_method, args->base_solver);
      args->base_solver = NULL;
   }

   if (args->precon_obj)
   {
      hypredrv_PreconDestroy(args->precon_method, &args->precon, &args->precon_obj);
      args->precon_obj = NULL;
   }
}
