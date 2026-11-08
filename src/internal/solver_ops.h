/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef HYPREDRV_SOLVER_OPS_HEADER
#define HYPREDRV_SOLVER_OPS_HEADER

#include "internal/solver.h"

/* The ordinary and nested Krylov paths use the same per-method operations.
 * Field setters accept solver_args: union members share the union's address. */
typedef struct SolverOps_struct
{
   void (*set_defaults)(solver_args *);
   YAMLGetValidKeysFunc get_keys;
   YAMLGetValidValuesFunc get_values;
   YAMLSetFieldByNameFunc set_field;
   void (*create)(MPI_Comm, const solver_args *, HYPRE_Solver *);
   HYPRE_Int (*set_precond)(HYPRE_Solver, HYPRE_PtrToParSolverFcn,
                             HYPRE_PtrToParSolverFcn, HYPRE_Solver);
   HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
   HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
   HYPRE_Int (*destroy)(HYPRE_Solver);
   HYPRE_Int (*get_num_iterations)(HYPRE_Solver, HYPRE_Int *);
   HYPRE_Int (*get_converged)(HYPRE_Solver, HYPRE_Int *);
   HYPRE_Int (*get_final_rel_res_norm)(HYPRE_Solver, HYPRE_Real *);
} SolverOps;

const SolverOps *hypredrv_SolverOpsLookup(solver_t);

#endif /* HYPREDRV_SOLVER_OPS_HEADER */
