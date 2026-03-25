/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef KRYLOV_HEADER
#define KRYLOV_HEADER

#include "HYPRE.h"
#include "precon.h"
#include "solver.h"
#include "yaml.h"

typedef struct NestedKrylov_args_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);

   int         is_set;
   solver_t    solver_method;
   solver_args solver;

   int         has_precon;
   precon_t    precon_method;
   precon_args precon;

   HYPRE_Solver base_solver;
   HYPRE_Precon precon_obj;
} NestedKrylov_args;

void      hypredrv_NestedKrylovSetDefaultArgs(NestedKrylov_args *);
void      hypredrv_NestedKrylovSetArgsFromYAML(NestedKrylov_args *, YAMLnode *);
void      hypredrv_NestedKrylovCreate(MPI_Comm, NestedKrylov_args *, IntArray *,
                                      HYPRE_IJVector, HYPRE_Solver *);
HYPRE_Int hypredrv_NestedKrylovSetup(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector,
                                     HYPRE_Vector);
HYPRE_Int hypredrv_NestedKrylovSolve(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector,
                                     HYPRE_Vector);
void      hypredrv_NestedKrylovDestroy(NestedKrylov_args *);

#endif /* KRYLOV_HEADER */
