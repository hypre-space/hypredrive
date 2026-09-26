/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef MGR_INTERNAL_HEADER
#define MGR_INTERNAL_HEADER

#include <stddef.h>
#include "internal/krylov.h"
#include "internal/mgr.h"
#include "internal/stats.h"

/* Shared helpers used by configuration, wrapper, reuse, and construction code. */
const char  *hypredrv_MGRLogObjectName(const Stats *stats);
HYPRE_Int    hypredrv_MGRLevelInterpTypeCompat(HYPRE_Int interp_type, const Stats *stats,
                                               int next_ls_id, HYPRE_Int level);
void         hypredrv_MGRComponentReuseSetDefaultArgs(MGRComponentReuse_args *reuse);
void         hypredrv_MGRComponentReuseDestroyArgs(MGRComponentReuse_args *reuse);
HYPRE_Int    hypredrv_MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                            HYPRE_ParVector b, HYPRE_ParVector x);
HYPRE_Int    hypredrv_MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                            HYPRE_ParVector b, HYPRE_ParVector x);
void         hypredrv_MGRSetFSolverAtLevel(HYPRE_Solver precon, HYPRE_Solver fsolver,
                                           HYPRE_Int level, HYPRE_Int f_relax_type,
                                           HYPRE_PtrToParSolverFcn fine_grid_solver_solve,
                                           HYPRE_PtrToParSolverFcn fine_grid_solver_setup);
HYPRE_Solver hypredrv_MGRNestedFRelaxWrapperCreate(HYPRE_Solver inner_mgr,
                                                   MGR_args    *nested_args,
                                                   IntArray    *owned_dofmap);
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
HYPRE_Solver hypredrv_MGRFRelaxEquilWrapperCreate(HYPRE_Solver inner);
HYPRE_Int    hypredrv_MGRFRelaxEquilWrapperDestroy(void *wrapper);
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
HYPRE_Solver hypredrv_MGRSchwarzWrapperCreate(const Schwarz_args *args);
HYPRE_Int    hypredrv_MGRSchwarzWrapperDestroy(HYPRE_Solver wrapper);
#endif

int hypredrv_MGRGRelaxUsesUserSmoother(const MGRgrlx_args *args);
int hypredrv_MGRProjectNestedRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                                  const int *selected_dofs, size_t num_selected_dofs);
int hypredrv_MGRProjectNestedRemainingRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                                           int num_eliminated_levels);
HYPRE_Solver hypredrv_MGRFRelaxSolverCreateByType(MGR_args            *args,
                                                  MGRfrlx_args        *f_relaxation,
                                                  const StackIntArray *f_dofs,
                                                  int                  active_lvl);
void hypredrv_MGRFRelaxInstall(HYPRE_Solver precon, const MGRfrlx_args *f_relaxation,
                               HYPRE_Solver frelax, int active_lvl);
HYPRE_Solver hypredrv_MGRGRelaxSolverCreateByType(MGRgrlx_args *g_relaxation);
HYPRE_Solver hypredrv_MGRCoarseSolverCreateByType(MGRcls_args *coarsest_level,
                                                  HYPRE_Int    type);
void         hypredrv_MGRCoarseSolverInstall(HYPRE_Solver mgr_solver, HYPRE_Int type,
                                             HYPRE_Solver coarse_solver);
const char  *hypredrv_MGRCoarseSolverTypeName(const MGRcls_args *args);
int          hypredrv_MGRBuildDofLabelPresenceMask(const IntArray *dofmap,
                                                   size_t         *label_space_size_out,
                                                   size_t         *num_present_labels_out,
                                                   HYPRE_Int     **label_present_out);
IntArray    *hypredrv_MGRBuildProjectedFRelaxDofmap(const IntArray      *parent_dofmap,
                                                    const StackIntArray *parent_f_dofs);

#endif /* MGR_INTERNAL_HEADER */
