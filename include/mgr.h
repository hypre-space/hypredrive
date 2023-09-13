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
 * Generic Relaxation methods arguments struct
 *--------------------------------------------------------------------------*/

typedef union relax_args_union {
   AMG_args      amg;
   ILU_args      ilu;
} relax_args;

/*--------------------------------------------------------------------------
 * MGR preconditioner arguments struct
 *--------------------------------------------------------------------------*/

typedef struct MGR_args_struct {
   HYPRE_Int     lvl;
   HYPRE_Int     non_c_to_f;
   HYPRE_Int     pmax;
   HYPRE_Int     max_iter;
   HYPRE_Int     num_levels;
   HYPRE_Int     print_level;
   HYPRE_Real    tol;
   HYPRE_Real    coarse_th;

   HYPRE_Int     num_f_dofs[MAX_MGR_LEVELS];
   HYPRE_Int    *f_dofs[MAX_MGR_LEVELS];

   HYPRE_Int     num_frelax_sweeps[MAX_MGR_LEVELS];
   HYPRE_Int     num_grelax_sweeps[MAX_MGR_LEVELS];
   HYPRE_Int     prolongation_types[MAX_MGR_LEVELS];
   HYPRE_Int     restriction_types[MAX_MGR_LEVELS];
   HYPRE_Int     coarse_grid_types[MAX_MGR_LEVELS];
   HYPRE_Int     frelax_types[MAX_MGR_LEVELS];
   HYPRE_Int     grelax_types[MAX_MGR_LEVELS];

   relax_args    frelax[MAX_MGR_LEVELS];
   relax_args    grelax[MAX_MGR_LEVELS];
} MGR_args;

/*--------------------------------------------------------------------------
 * Public prototypes
 *--------------------------------------------------------------------------*/

int MGRSetDefaultArgs(MGR_args*);
int MGRSetArgsFromYAML(MGR_args*, YAMLnode*);
int MGRCreate(MGR_args*, HYPRE_IntArray*, HYPRE_Solver*);
int MGRDestroyArgs(MGR_args**);

#endif
