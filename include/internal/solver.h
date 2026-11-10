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
#include "internal/precon.h"

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

typedef struct PCG_args_struct
{
   HYPRE_Int  max_iter;
   HYPRE_Int  two_norm;
   HYPRE_Int  stop_crit;
   HYPRE_Int  rel_change;
   HYPRE_Int  print_level;
   HYPRE_Int  recompute_res;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
   HYPRE_Real residual_tol;
   HYPRE_Real conv_fac_tol;
} PCG_args;

typedef struct GMRES_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  stop_crit;
   HYPRE_Int  skip_real_res_check;
   HYPRE_Int  krylov_dim;
   HYPRE_Int  rel_change;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
   HYPRE_Real conv_fac_tol;
} GMRES_args;

typedef struct FGMRES_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  krylov_dim;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
} FGMRES_args;

typedef struct BiCGSTAB_args_struct
{
   HYPRE_Int  min_iter;
   HYPRE_Int  max_iter;
   HYPRE_Int  stop_crit;
   HYPRE_Int  logging;
   HYPRE_Int  print_level;
   HYPRE_Real relative_tol;
   HYPRE_Real absolute_tol;
   HYPRE_Real conv_fac_tol;
} BiCGSTAB_args;

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

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

/* Method schemas are shared with the help printer. Parsing and creation stay
 * behind the generic solver interface below. */
StrArray       hypredrv_PCGGetValidKeys(void);
StrIntMapArray hypredrv_PCGGetValidValues(const char *);
StrArray       hypredrv_GMRESGetValidKeys(void);
StrIntMapArray hypredrv_GMRESGetValidValues(const char *);
StrArray       hypredrv_FGMRESGetValidKeys(void);
StrIntMapArray hypredrv_FGMRESGetValidValues(const char *);
StrArray       hypredrv_BiCGSTABGetValidKeys(void);
StrIntMapArray hypredrv_BiCGSTABGetValidValues(const char *);

StrArray       hypredrv_SolverGetValidKeys(void);
StrIntMapArray hypredrv_SolverGetValidTypeIntMap(void);

/* Parse a solver section or a typed method body. The optional child belongs to
 * another schema. A scalar section selects quiet defaults; typed bodies retain
 * method defaults. Both operations publish the method and arguments together
 * on success and preserve the caller's configuration on rejection. */
bool hypredrv_SolverSetArgsFromYAML(solver_t *, solver_args *, YAMLnode *,
                                    const YAMLnode *);
bool hypredrv_SolverArgsFromPreset(const char *, solver_t *, solver_args *);
void hypredrv_SolverArgsSetDefaultsForMethod(solver_t, solver_args *);
/* A live runtime binds its native operations at creation. Configuration tags
 * are not needed by setup, execution, queries or destruction. */
typedef struct hypredrv_Solver_struct hypredrv_Solver;
void hypredrv_SolverCreate(MPI_Comm, solver_t, const solver_args *, hypredrv_Solver **);
void hypredrv_SolverSetupWithReuse(hypredrv_Solver *, HYPRE_Precon, HYPRE_IJMatrix,
                                   HYPRE_IJVector, HYPRE_IJVector, HYPRE_IJVector,
                                   Stats *, int);
void hypredrv_SolverGetConverged(hypredrv_Solver *, HYPRE_Int *);
void hypredrv_SolverGetFinalRelativeResidualNorm(hypredrv_Solver *, HYPRE_Real *);
void hypredrv_SolverApply(hypredrv_Solver *, HYPRE_IJMatrix, HYPRE_IJVector,
                          HYPRE_IJVector, HYPRE_IJVector, Stats *);
HYPRE_Int hypredrv_SolverSolveOnly(hypredrv_Solver *, HYPRE_IJMatrix, HYPRE_IJVector,
                                   HYPRE_IJVector, HYPRE_IJVector);
void      hypredrv_SolverDestroy(hypredrv_Solver **);

/* Native MGR component callback prefix. Newer hypre also reads readiness
 * immediately after the three callbacks; older releases ignore that field. */
typedef struct
{
   HYPRE_PtrToSolverFcn setup, solve;
   HYPRE_Int (*destroy)(HYPRE_Solver);
   HYPRE_Int is_setup;
} hypredrv_SolverBase;

/* Nested Krylov configuration and lifecycle for MGR components. */
struct NestedKrylovRuntime_struct;
typedef struct NestedKrylov_args_struct
{
   hypredrv_SolverBase base;
   int                 is_set;
   solver_t            solver_method;
   solver_args         solver;

   int         has_precon;
   precon_t    precon_method;
   precon_args precon;

   struct NestedKrylovRuntime_struct *runtime; /* owned; native state stays private */
} NestedKrylov_args;

void hypredrv_NestedKrylovSetDefaultArgs(NestedKrylov_args *);
/* Check the full solver/preconditioner candidate before replacing runtime or
 * owned configuration. Rejection preserves both; an omitted preconditioner
 * retains the existing configuration. */
void hypredrv_NestedKrylovSetArgsFromYAML(NestedKrylov_args *, YAMLnode *);
/* Ensure one runtime exists. Repeated creation preserves its setup and native
 * handles; parsing accepted configuration retires it before publishing changes. */
bool      hypredrv_NestedKrylovCreate(MPI_Comm, NestedKrylov_args *, IntArray *,
                                      HYPRE_IJVector);
HYPRE_Int hypredrv_NestedKrylovSetup(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector,
                                     HYPRE_Vector);
HYPRE_Int hypredrv_NestedKrylovSolve(HYPRE_Solver, HYPRE_Matrix, HYPRE_Vector,
                                     HYPRE_Vector);
void      hypredrv_NestedKrylovDestroy(NestedKrylov_args *);

#endif /* SOLVER_HEADER */
