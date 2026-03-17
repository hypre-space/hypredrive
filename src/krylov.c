/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "krylov.h"
#include "bicgstab.h"
#include "error.h"
#include "fgmres.h"
#include "gmres.h"
#include "pcg.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 1)
#include "_hypre_utilities.h" // for hypre_Solver
#endif

#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
static HYPRE_Int
HYPREDRV_FSAISetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                       HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
HYPREDRV_FSAISolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                       HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define HYPREDRV_FSAI_SETUP HYPREDRV_FSAISetupStub
#define HYPREDRV_FSAI_SOLVE HYPREDRV_FSAISolveStub
#else
#define HYPREDRV_FSAI_SETUP HYPRE_FSAISetup
#define HYPREDRV_FSAI_SOLVE HYPRE_FSAISolve
#endif

#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
static HYPRE_Int
HYPREDRV_ILUSetupStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

static HYPRE_Int
HYPREDRV_ILUSolveStub(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
}

#define HYPREDRV_ILU_SETUP HYPREDRV_ILUSetupStub
#define HYPREDRV_ILU_SOLVE HYPREDRV_ILUSolveStub
#else
#define HYPREDRV_ILU_SETUP HYPRE_ILUSetup
#define HYPREDRV_ILU_SOLVE HYPRE_ILUSolve
#endif

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
NestedKrylovDetachPrecon(YAMLnode *solver_node, YAMLnode **precon_node_out,
                         YAMLnode **precon_prev_out, YAMLnode **precon_next_out)
{
   YAMLnode *prev  = NULL;
   YAMLnode *child = NULL;

   if (!solver_node || !precon_node_out || !precon_prev_out || !precon_next_out)
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
   if (!solver_node || !precon_node)
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
   if (!args || !precon_node)
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
      if (args->precon_method == PRECON_MGR)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Nested preconditioner 'mgr' is not supported");
         YAML_NODE_SET_INVALID_VAL(precon_node);
         return;
      }
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
   if (args->precon_method == PRECON_MGR)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Nested preconditioner 'mgr' is not supported");
      YAML_NODE_SET_INVALID_VAL(type_node);
      return;
   }

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
   if (!precon || precon_method == PRECON_NONE)
   {
      return;
   }

   HYPRE_PtrToParSolverFcn setup_ptrs[] = {
      HYPRE_BoomerAMGSetup,
      HYPRE_MGRSetup,
      HYPREDRV_ILU_SETUP,
      HYPREDRV_FSAI_SETUP,
   };
   HYPRE_PtrToParSolverFcn solve_ptrs[] = {
      HYPRE_BoomerAMGSolve,
      HYPRE_MGRSolve,
      HYPREDRV_ILU_SOLVE,
      HYPREDRV_FSAI_SOLVE,
   };

   switch (solver_method)
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
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         break;
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

   switch (args->solver_method)
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
      default:
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
         break;
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
      if (hypredrv_ErrorCodeActive())
      {
         return;
      }
   }

   /* Create solver object always */
   switch (args->solver_method)
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
      if (hypredrv_ErrorCodeActive())
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

void
hypredrv_NestedKrylovDestroy(NestedKrylov_args *args)
{
   if (!args)
   {
      return;
   }

   if (args->base_solver)
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_SolverDestroy(args->base_solver);
#else
      switch (args->solver_method)
      {
         case SOLVER_PCG:
            HYPRE_ParCSRPCGDestroy(args->base_solver);
            break;
         case SOLVER_GMRES:
            HYPRE_ParCSRGMRESDestroy(args->base_solver);
            break;
         case SOLVER_FGMRES:
            HYPRE_ParCSRFlexGMRESDestroy(args->base_solver);
            break;
         case SOLVER_BICGSTAB:
            HYPRE_ParCSRBiCGSTABDestroy(args->base_solver);
            break;
         default:
            break;
      }
#endif
      args->base_solver = NULL;
   }

   if (args->precon_obj)
   {
      hypredrv_PreconDestroy(args->precon_method, &args->precon, &args->precon_obj);
      args->precon_obj = NULL;
   }
}