/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "solver.h"

static const FieldOffsetMap solver_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(solver_args, pcg, FIELD_TYPE_STRUCT, PCGSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, gmres, FIELD_TYPE_STRUCT, GMRESSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, fgmres, FIELD_TYPE_STRUCT, FGMRESSetArgs),
   FIELD_OFFSET_MAP_ENTRY(solver_args, bicgstab, FIELD_TYPE_STRUCT, BiCGSTABSetArgs),
};

#define SOLVER_NUM_FIELDS (sizeof(solver_field_offset_map) / sizeof(solver_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * SolverSetFieldByName
 *-----------------------------------------------------------------------------*/

void
SolverSetFieldByName(solver_args *args, const char *name, YAMLnode *node)
{
   for (size_t i = 0; i < SOLVER_NUM_FIELDS; i++)
   {
      /* Which union type are we trying to set? */
      if (!strcmp(solver_field_offset_map[i].name, name))
      {
         solver_field_offset_map[i].setter(
            (void*)((char*) args + solver_field_offset_map[i].offset),
            node);
         return;
      }
   }
}

/*-----------------------------------------------------------------------------
 * SolverGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
SolverGetValidKeys(void)
{
   static const char* keys[SOLVER_NUM_FIELDS];

   for(size_t i = 0; i < SOLVER_NUM_FIELDS; i++)
   {
      keys[i] = solver_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * SolverGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
SolverGetValidValues(const char* key)
{
   /* The "solver" enry does not hold values, so we create a void map */
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * SolverGetValidTypeIntMap
 *-----------------------------------------------------------------------------*/

StrIntMapArray
SolverGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {{"pcg",      (int) SOLVER_PCG},
                             {"gmres",    (int) SOLVER_GMRES},
                             {"fgmres",   (int) SOLVER_FGMRES},
                             {"bicgstab", (int) SOLVER_BICGSTAB}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}

/*-----------------------------------------------------------------------------
 * SolverSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
SolverSetArgsFromYAML(solver_args *args, YAMLnode *parent)
{
   YAMLnode    *child;

   child = parent->children;
   while (child)
   {
      YAML_VALIDATE_NODE(child,
                         SolverGetValidKeys,
                         SolverGetValidValues);

      YAML_SET_ARG_STRUCT(child,
                          args,
                          SolverSetFieldByName);

      child = child->next;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

int
SolverCreate(MPI_Comm comm, solver_t solver_method, solver_args *args, HYPRE_Solver *solver_ptr)
{
   switch (solver_method)
   {
      case SOLVER_PCG:
         PCGCreate(comm, &args->pcg, solver_ptr);
         break;

      case SOLVER_GMRES:
         GMRESCreate(comm, &args->gmres, solver_ptr);
         break;

      case SOLVER_FGMRES:
         FGMRESCreate(comm, &args->fgmres, solver_ptr);
         break;

      case SOLVER_BICGSTAB:
         BiCGSTABCreate(comm, &args->bicgstab, solver_ptr);
         break;

      default:
         *solver_ptr = NULL;
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverSetup
 *-----------------------------------------------------------------------------*/

int
SolverSetup(precon_t precon_method, solver_t solver_method,
            HYPRE_Solver precon, HYPRE_Solver solver,
            HYPRE_IJMatrix M, HYPRE_IJVector b, HYPRE_IJVector x)
{
   void                    *vM, *vb, *vx;
   HYPRE_ParCSRMatrix       par_M;
   HYPRE_ParVector          par_b, par_x;
   HYPRE_PtrToParSolverFcn  setup_ptrs[] = {HYPRE_BoomerAMGSetup,
                                            HYPRE_MGRSetup,
                                            HYPRE_ILUSetup};
   HYPRE_PtrToParSolverFcn  solve_ptrs[] = {HYPRE_BoomerAMGSolve,
                                            HYPRE_MGRSolve,
                                            HYPRE_ILUSolve};

   HYPRE_IJMatrixGetObject(M, &vM); par_M = (HYPRE_ParCSRMatrix) vM;
   HYPRE_IJVectorGetObject(b, &vb); par_b = (HYPRE_ParVector) vb;
   HYPRE_IJVectorGetObject(x, &vx); par_x = (HYPRE_ParVector) vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSetPrecond(solver,
                                   solve_ptrs[precon_method],
                                   setup_ptrs[precon_method],
                                   precon);
         HYPRE_ParCSRPCGSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSetPrecond(solver,
                                     solve_ptrs[precon_method],
                                     setup_ptrs[precon_method],
                                     precon);
         HYPRE_ParCSRGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSetPrecond(solver,
                                         solve_ptrs[precon_method],
                                         setup_ptrs[precon_method],
                                         precon);
         HYPRE_ParCSRFlexGMRESSetup(solver, par_M, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                        solve_ptrs[precon_method],
                                        setup_ptrs[precon_method],
                                        precon);
         HYPRE_ParCSRBiCGSTABSetup(solver, par_M, par_b, par_x);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverApply
 *-----------------------------------------------------------------------------*/

int
SolverApply(solver_t solver_method, HYPRE_Solver solver,
            HYPRE_IJMatrix A, HYPRE_IJVector b, HYPRE_IJVector x)
{
   void               *vA, *vb, *vx;
   HYPRE_ParCSRMatrix  par_A;
   HYPRE_ParVector     par_b, par_x;

   HYPRE_IJMatrixGetObject(A, &vA); par_A = (HYPRE_ParCSRMatrix) vA;
   HYPRE_IJVectorGetObject(b, &vb); par_b = (HYPRE_ParVector) vb;
   HYPRE_IJVectorGetObject(x, &vx); par_x = (HYPRE_ParVector) vx;

   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESSolve(solver, par_A, par_b, par_x);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABSolve(solver, par_A, par_b, par_x);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * SolverDestroy
 *-----------------------------------------------------------------------------*/

int
SolverDestroy(solver_t solver_method, HYPRE_Solver *solver_ptr)
{
   switch (solver_method)
   {
      case SOLVER_PCG:
         HYPRE_ParCSRPCGDestroy(*solver_ptr);
         break;

      case SOLVER_GMRES:
         HYPRE_ParCSRGMRESDestroy(*solver_ptr);
         break;

      case SOLVER_FGMRES:
         HYPRE_ParCSRFlexGMRESDestroy(*solver_ptr);
         break;

      case SOLVER_BICGSTAB:
         HYPRE_ParCSRBiCGSTABDestroy(*solver_ptr);
         break;

      default:
         ErrorMsgAddInvalidSolverOption((int) solver_method);
         return EXIT_FAILURE;
   }

   *solver_ptr = NULL;

   return EXIT_SUCCESS;
}
