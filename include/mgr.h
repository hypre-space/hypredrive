/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef MGR_HEADER
#define MGR_HEADER

#include "ilu.h"
#include "amg.h"
#include "utils.h"

#define MAX_MGR_LEVELS 32

/*--------------------------------------------------------------------------
 * Coarsest level solver arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRcls_args_struct {
   HYPRE_Int     type;

   AMG_args      amg;
} MGRcls_args;

/*--------------------------------------------------------------------------
 * F-Relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRfrlx_args_struct {
   HYPRE_Int     type;
   HYPRE_Int     num_sweeps;

   /* TODO: Ideally, these should be inside a union */
   AMG_args      amg;
   ILU_args      ilu;
} MGRfrlx_args;

/*--------------------------------------------------------------------------
 * Global-Relaxation arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRgrlx_args_struct {
   HYPRE_Int     type;
   HYPRE_Int     num_sweeps;

   /* TODO: Ideally, these should be inside a union */
   ILU_args      ilu;
} MGRgrlx_args;

/*--------------------------------------------------------------------------
 * MGR level arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGRlvl_args_struct {
   IntArray      *f_dofs;

   HYPRE_Int      prolongation_type;
   HYPRE_Int      restriction_type;
   HYPRE_Int      coarse_level_type;

   MGRfrlx_args   f_relaxation;
   MGRgrlx_args   g_relaxation;
} MGRlvl_args;

/*--------------------------------------------------------------------------
 * MGR preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGR_args_struct {
   IntArray     *dofmap;

   HYPRE_Int     non_c_to_f;
   HYPRE_Int     pmax;
   HYPRE_Int     max_iter;
   HYPRE_Int     num_levels;
   HYPRE_Int     relax_type;   /* TODO: we shouldn't need this */
   HYPRE_Int     print_level;
   HYPRE_Real    tolerance;
   HYPRE_Real    coarse_th;

   MGRlvl_args   level[MAX_MGR_LEVELS - 1];
   MGRcls_args   coarsest_level;
} MGR_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

void MGRSetArgs(void*, YAMLnode*);
void MGRSetDofmap(MGR_args*, IntArray*);
void MGRCreate(MGR_args*, HYPRE_Solver*);
void MGRDestroyArgs(MGR_args**);

#endif /* MGR_HEADER */
