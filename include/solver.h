/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef SOLVER_HEADER
#define SOLVER_HEADER

#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "bicgstab.h"
#include "fgmres.h"
#include "gmres.h"
#include "linsys.h"
#include "pcg.h"
#include "precon.h"

/*--------------------------------------------------------------------------
 * Solver types enum
 *--------------------------------------------------------------------------*/

typedef enum solver_type_enum
{
   SOLVER_PCG,
   SOLVER_GMRES,
   SOLVER_FGMRES,
   SOLVER_BICGSTAB
} solver_t;

/*--------------------------------------------------------------------------
 * Generic solver arguments struct
 *--------------------------------------------------------------------------*/

typedef union solver_args_union
{
   PCG_args      pcg;
   GMRES_args    gmres;
   FGMRES_args   fgmres;
   BiCGSTAB_args bicgstab;
} solver_args;

typedef solver_args Solver_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray       SolverGetValidKeys(void);
StrIntMapArray SolverGetValidValues(const char *);
StrIntMapArray SolverGetValidTypeIntMap(void);

void SolverSetArgsFromYAML(void *, YAMLnode *);
void SolverArgsSetDefaultsForMethod(solver_t, solver_args *);
void SolverCreate(MPI_Comm, solver_t, solver_args *, HYPRE_Solver *);
void SolverSetupWithReuse(precon_t, solver_t, HYPRE_Precon, HYPRE_Solver, HYPRE_IJMatrix,
                          HYPRE_IJVector, HYPRE_IJVector, Stats *, int);
void SolverSetup(precon_t, solver_t, HYPRE_Precon, HYPRE_Solver, HYPRE_IJMatrix,
                 HYPRE_IJVector, HYPRE_IJVector, Stats *);
void SolverApply(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector,
                 Stats *);
HYPRE_Int SolverSolveOnly(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector);
void      SolverDestroy(solver_t, HYPRE_Solver *);

#endif /* SOLVER_HEADER */
