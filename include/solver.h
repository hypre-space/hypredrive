/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef SOLVER_HEADER
#define SOLVER_HEADER

#include "precon.h"
#include "pcg.h"
#include "gmres.h"
#include "fgmres.h"
#include "bicgstab.h"
#include "yaml.h"
#include "field.h"
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

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

StrArray SolverGetValidKeys(void);
StrIntMapArray SolverGetValidValues(const char*);
StrIntMapArray SolverGetValidTypeIntMap(void);

int SolverSetArgsFromYAML(solver_args*, YAMLnode*);
int SolverCreate(MPI_Comm, solver_t, solver_args*, HYPRE_Solver*);
int SolverSetup(precon_t, solver_t, HYPRE_Solver, HYPRE_Solver,
                HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
int SolverApply(solver_t, HYPRE_Solver, HYPRE_IJMatrix, HYPRE_IJVector, HYPRE_IJVector);
int SolverDestroy(solver_t, HYPRE_Solver*);

#endif
