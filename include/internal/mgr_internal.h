/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef MGR_INTERNAL_HEADER
#define MGR_INTERNAL_HEADER

#include <stddef.h>
#include "internal/mgr.h"
#include "internal/solver.h"
#include "internal/stats.h"

/* Shared helpers used by configuration, wrapper, reuse, and construction code. */
const char *hypredrv_MGRLogObjectName(const Stats *stats);
HYPRE_Int   hypredrv_MGRLevelInterpTypeCompat(HYPRE_Int interp_type, const Stats *stats,
                                              int next_ls_id, HYPRE_Int level);
void        hypredrv_MGRComponentReuseSetDefaultArgs(MGRComponentReuse_args *reuse);
void        hypredrv_MGRComponentReuseDestroyArgs(MGRComponentReuse_args *reuse);
HYPRE_Int   hypredrv_MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                           HYPRE_ParVector b, HYPRE_ParVector x);
HYPRE_Int   hypredrv_MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                           HYPRE_ParVector b, HYPRE_ParVector x);
void        hypredrv_MGRSetFSolverAtLevel(HYPRE_Solver precon, HYPRE_Solver fsolver,
                                          HYPRE_Int level, HYPRE_Int f_relax_type,
                                          HYPRE_PtrToParSolverFcn fine_grid_solver_solve,
                                          HYPRE_PtrToParSolverFcn fine_grid_solver_setup);
/* Build and own a nested F-block hierarchy, including its projected markers.
 * All ranks of parent->comm admit their projection before creating the child.
 * Destruction also accepts cache slots already reclaimed by native teardown. */
HYPRE_Solver hypredrv_MGRNestedFRelaxCreate(const MGR_args *parent, int level,
                                            const Stats *, int          next_ls_id);
void         hypredrv_MGRNestedFRelaxDestroy(HYPRE_Solver *);
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
/* Bind the MGR/F-block communicator and take ownership of a fresh AMG child,
 * including on failure. Setup owns its operator clone; solve uses that clone. */
HYPRE_Solver hypredrv_MGRFRelaxEquilWrapperCreate(MPI_Comm, HYPRE_Solver inner);
HYPRE_Int    hypredrv_MGRFRelaxEquilWrapperDestroy(HYPRE_Solver wrapper);
#endif

/* Collectively reject marker storage shorter than the local matrix rows. */
int hypredrv_MGRValidatePointMarkerRows(const MGR_args *, HYPRE_ParCSRMatrix);

int hypredrv_MGRGRelaxUsesUserSmoother(const MGRgrlx_args *args);
int hypredrv_MGRProjectNestedRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                                  const int *selected_dofs, size_t num_selected_dofs);
int hypredrv_MGRProjectNestedRemainingRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                                           int num_eliminated_levels);
/* Build and register a managed F-solver using hypre's native handle protocol.
 * All ranks agree on the method, scaling mode and native level before creation. */
int hypredrv_MGRFRelaxAttach(MGR_args *, HYPRE_Solver parent, int active_lvl, int refresh,
                             const Stats *, int next_ls_id);
/* Retire the binding after optional native parent teardown. The binding owns
 * the native lifetime policy; configuration tags are not needed for cleanup. */
void hypredrv_MGRFRelaxDestroy(HYPRE_Solver *, int parent_destroyed);
/* Own the native global smoother and its typed callback prefix. All ranks
 * admit a replacement before releasing their installed runtime. */
int  hypredrv_MGRGRelaxAttach(MGR_args *, HYPRE_Solver parent, int active_level,
                              int refresh, const Stats *, int next_ls_id);
void hypredrv_MGRGRelaxDestroy(HYPRE_Solver *);
/* Collectively attach a cached or fresh native coarse solver on args->comm.
 * A forced refresh builds a candidate before replacing the installed handle;
 * rejection preserves the installed solver on every rank. */
int         hypredrv_MGRCoarseSolverAttach(MGR_args *, HYPRE_Solver parent, int refresh,
                                           const Stats *, int next_ls_id);
const char *hypredrv_MGRCoarseSolverTypeName(const MGRcls_args *args);
int         hypredrv_MGRBuildDofLabelPresenceMask(const IntArray *dofmap,
                                                  size_t         *label_space_size_out,
                                                  size_t         *num_present_labels_out,
                                                  HYPRE_Int     **label_present_out);

#endif /* MGR_INTERNAL_HEADER */
