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
   SOLVER_BICGSTAB,
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

StrArray       hypredrv_SolverGetValidKeys(void);
StrIntMapArray hypredrv_SolverGetValidValues(const char *);
StrIntMapArray hypredrv_SolverGetValidTypeIntMap(void);

void hypredrv_SolverSetArgsFromYAML(void *, YAMLnode *);
void hypredrv_SolverArgsSetDefaultsForMethod(solver_t, solver_args *);
void hypredrv_SolverCreate(MPI_Comm, solver_t, solver_args *, HYPRE_Solver *);
void hypredrv_SolverSetupWithReuse(precon_t, solver_t, HYPRE_Precon, HYPRE_Solver,
                                   HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector,
                                   Stats *, int);
void hypredrv_SolverSetup(precon_t, solver_t, HYPRE_Precon, HYPRE_Solver, HYPRE_IJMatrix,
                          HYPRE_IJVector, HYPRE_IJVector, Stats *);
void hypredrv_SolverApply(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector, Stats *);
HYPRE_Int hypredrv_SolverSolveOnly(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector,
                                   HYPRE_IJVector);
void      hypredrv_SolverDestroy(solver_t, HYPRE_Solver *);

#endif /* SOLVER_HEADER */
