/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef NESTED_KRYLOV_HEADER
#define NESTED_KRYLOV_HEADER

#include "HYPRE.h"
#include "precon.h"
#include "solver.h"
#include "yaml.h"

typedef struct NestedKrylov_args_struct
{
   int         is_set;
   solver_t    solver_method;
   solver_args solver;

   int         has_precon;
   precon_t    precon_method;
   precon_args precon;

   HYPRE_Solver base_solver;
   HYPRE_Precon precon_obj;
} NestedKrylov_args;

void NestedKrylovSetDefaultArgs(NestedKrylov_args *);
void NestedKrylovSetArgsFromYAML(NestedKrylov_args *, YAMLnode *);
void NestedKrylovCreate(MPI_Comm, NestedKrylov_args *, IntArray *, HYPRE_IJVector,
                        HYPRE_Solver *);
void NestedKrylovDestroy(NestedKrylov_args *);

#endif /* NESTED_KRYLOV_HEADER */
