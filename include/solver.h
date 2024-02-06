/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef SOLVER_HEADER
#define SOLVER_HEADER

#include "precon.h"
#include "pcg.h"
#include "gmres.h"
#include "fgmres.h"
#include "bicgstab.h"
#include "linsys.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*--------------------------------------------------------------------------
 * Solver types enum
 *--------------------------------------------------------------------------*/

typedef enum solver_type_enum {
   SOLVER_PCG,
   SOLVER_GMRES,
   SOLVER_FGMRES,
   SOLVER_BICGSTAB
} solver_t;

/*--------------------------------------------------------------------------
 * Generic solver arguments struct
 *--------------------------------------------------------------------------*/

typedef union solver_args_union {
   PCG_args        pcg;
   GMRES_args      gmres;
   FGMRES_args     fgmres;
   BiCGSTAB_args   bicgstab;
} solver_args;

typedef solver_args Solver_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray SolverGetValidKeys(void);
StrIntMapArray SolverGetValidValues(const char*);
StrIntMapArray SolverGetValidTypeIntMap(void);

void SolverSetArgsFromYAML(solver_args*, YAMLnode*);
void SolverCreate(MPI_Comm, solver_t, solver_args*, HYPRE_Solver*);
void SolverSetup(precon_t, solver_t, HYPRE_Solver, HYPRE_Solver,
                 HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
void SolverApply(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
void SolverDestroy(solver_t, HYPRE_Solver*);

#endif /* SOLVER_HEADER */
