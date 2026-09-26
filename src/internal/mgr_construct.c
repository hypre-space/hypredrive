/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include "internal/mgr_internal.h"
/* gcovr: branch-exclusion regions below narrow branch-count noise from YAML
 * helpers and MGR validation/dispatch; single-line exclusions flag allocator
 * and defensive branches that are impractical to fault-inject here. */
#include "_hypre_parcsr_mv.h"
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
#include "_hypre_parcsr_ls.h"
#endif
#include "_hypre_utilities.h" // for hypre_Solver
#include "internal/compatibility.h"
#include "internal/error.h"
#include "internal/gen_macros.h"
#include "internal/krylov.h"
#include "internal/stats.h"
#include "logging.h"

#if HYPRE_CHECK_MIN_VERSION(21900, 0)

typedef struct
{
   HYPRE_Int *label_present;     /* presence mask over the dof label space */
   HYPRE_Int *label_to_dense;    /* sparse-to-dense label remap (optional) */
   HYPRE_Int *inactive_dofs;     /* labels eliminated by earlier levels */
   HYPRE_Int *dofmap_data;       /* point markers for hypre (may alias dofmap) */
   HYPRE_Int *dofmap_data_owned; /* owned copy behind dofmap_data, if any */
   HYPRE_Int  num_dofs;          /* dof label space size */
   HYPRE_Int  num_dofs_hypre;    /* dof-type count passed to hypre */
   HYPRE_Int  num_active_dofs;   /* labels actually present in the dofmap */
   HYPRE_Int  num_levels;        /* compacted level count (active + 1) */
   HYPRE_Int  active_level_map[MAX_MGR_LEVELS - 1];
   HYPRE_Int  num_c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int *c_dofs[MAX_MGR_LEVELS - 1];
} MGRCreatePlan;

static void
MGRCreatePlanDispose(MGRCreatePlan *plan)
{
   for (int lvl = 0; lvl < MAX_MGR_LEVELS - 1; lvl++)
   {
      free(plan->c_dofs[lvl]);
      plan->c_dofs[lvl] = NULL;
   }
   free(plan->dofmap_data_owned);
   free(plan->inactive_dofs);
   free(plan->label_present);
   free(plan->label_to_dense);
   plan->dofmap_data_owned = NULL;
   plan->inactive_dofs     = NULL;
   plan->label_present     = NULL;
   plan->label_to_dense    = NULL;
}

/*-----------------------------------------------------------------------------
 * Build the per-level C-point lists from the configured f_dofs, dropping
 * levels with no active F-points and compacting the remaining ones. Updates
 * args->num_active_levels and args->active_level_map.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
/* Marks labels absent from the dofmap as already eliminated. */
static int
MGRPlanInitInactiveDofs(MGRCreatePlan *plan)
{
   plan->inactive_dofs = (HYPRE_Int *)calloc((size_t)plan->num_dofs, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (plan->num_dofs > 0 && !plan->inactive_dofs)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR inactive dof label mask");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   for (HYPRE_Int i = 0; i < plan->num_dofs; i++)
   {
      if (!plan->label_present[i])
      {
         plan->inactive_dofs[i] = 1;
      }
   }

   return 1;
}

/* Eliminates one level's configured F labels from the still-active set. Returns
 * the number of labels actually eliminated, or -1 when a label is out of range
 * or was already consumed by an earlier level. */
static HYPRE_Int
MGRPlanEliminateLevelFDofs(const MGR_args *args, MGRCreatePlan *plan, HYPRE_Int lvl,
                           int may_ignore_missing_f_dofs, const Stats *stats,
                           int next_ls_id)
{
   HYPRE_Int num_level_f_dofs = 0;

   for (int i = 0; i < (int)args->level[lvl].f_dofs.size; i++)
   {
      HYPRE_Int dof_label = args->level[lvl].f_dofs.data[i];

      if (dof_label < 0 || dof_label >= plan->num_dofs)
      {
         /* Distributed callers may provide a compact active dof label space
          * while configured MGR blocks still reference inactive labels from
          * the original global numbering. Treat those configured labels as
          * absent instead of invalid when global-unique metadata is present. */
         if (dof_label >= 0 && may_ignore_missing_f_dofs && dof_label >= plan->num_dofs)
         {
            continue;
         }
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id,
                            "MGR invalid f_dofs label: level=%d label=%d valid=[0,%d]",
                            (int)lvl, (int)dof_label, (int)plan->num_dofs - 1);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid MGR level %d f_dofs label %d (valid range: [0,%d])", (int)lvl,
            (int)dof_label, (int)plan->num_dofs - 1);
         return -1;
      }
      if (!plan->label_present[dof_label])
      {
         /* Some configured blocks may have zero active dofs in the current
          * system. Ignore those labels instead of rejecting the MGR setup. */
         continue;
      }
      if (plan->inactive_dofs[dof_label])
      {
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id,
                            "MGR duplicate/pruned f_dofs label: level=%d label=%d",
                            (int)lvl, (int)dof_label);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Duplicate/previously eliminated MGR f_dofs label %d at level %d",
            (int)dof_label, (int)lvl);
         return -1;
      }
      plan->inactive_dofs[dof_label] = 1;
      ++num_level_f_dofs;
   }

   return num_level_f_dofs;
}

/* Rebuilds a level's C-point list from the labels that are still active. */
static void
MGRPlanFillLevelCDofs(MGRCreatePlan *plan, HYPRE_Int lvl)
{
   HYPRE_Int j = 0;

   for (HYPRE_Int i = 0; i < plan->num_dofs; i++)
   {
      if (plan->label_present[i] && !plan->inactive_dofs[i])
      {
         plan->c_dofs[lvl][j++] = i;
      }
   }
}

/* Drops levels that collapsed to zero F-points and renumbers the survivors. */
static void
MGRPlanCompactLevels(MGR_args *args, MGRCreatePlan *plan, HYPRE_Int num_levels)
{
   HYPRE_Int active_levels   = 0;
   HYPRE_Int original_levels = num_levels - 1;

   for (HYPRE_Int lvl = 0; lvl < original_levels; lvl++)
   {
      if (!plan->c_dofs[lvl])
      {
         continue;
      }

      plan->active_level_map[active_levels] = lvl;
      if (active_levels != lvl)
      {
         plan->c_dofs[active_levels]     = plan->c_dofs[lvl];
         plan->num_c_dofs[active_levels] = plan->num_c_dofs[lvl];
         plan->c_dofs[lvl]               = NULL;
         plan->num_c_dofs[lvl]           = 0;
      }
      active_levels++;
   }

   plan->num_levels        = active_levels + 1;
   args->num_active_levels = active_levels;
   for (HYPRE_Int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      args->active_level_map[i] = (i < active_levels) ? plan->active_level_map[i] : 0;
   }
}

static int
MGRPlanCoarsening(MGR_args *args, MGRCreatePlan *plan, const Stats *stats, int next_ls_id)
{
   IntArray *dofmap     = args->dofmap;
   HYPRE_Int num_levels = args->num_levels;
   HYPRE_Int num_dofs_last;
   HYPRE_Int lvl;

   args->num_active_levels = 0;
   memset(args->active_level_map, 0, sizeof(args->active_level_map));

   {
      size_t label_space_size = 0;
      size_t present_labels   = 0;
      if (!hypredrv_MGRBuildDofLabelPresenceMask(dofmap, &label_space_size,
                                                 &present_labels, &plan->label_present))
      {
         return 0;
      }
      plan->num_dofs        = (HYPRE_Int)label_space_size;
      plan->num_dofs_hypre  = plan->num_dofs;
      plan->num_active_dofs = (HYPRE_Int)present_labels;
   }
   int may_ignore_missing_f_dofs =
      (dofmap->g_unique_size > 0 && plan->num_active_dofs == plan->num_dofs);

   /* Compute num_c_dofs and c_dofs */
   num_dofs_last = plan->num_active_dofs;
   if (!MGRPlanInitInactiveDofs(plan))
   {
      return 0;
   }

   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      HYPRE_Int num_level_f_dofs = 0;

      plan->c_dofs[lvl] = (HYPRE_Int *)calloc((size_t)plan->num_dofs, sizeof(HYPRE_Int));
      plan->num_c_dofs[lvl] = num_dofs_last;
      if (plan->num_dofs > 0 && !plan->c_dofs[lvl])
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate MGR level %d C-point buffer", (int)lvl);
         return 0;
      }

      num_level_f_dofs = MGRPlanEliminateLevelFDofs(
         args, plan, lvl, may_ignore_missing_f_dofs, stats, next_ls_id);
      if (num_level_f_dofs < 0)
      {
         return 0;
      }

      if (num_level_f_dofs == 0)
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id,
                            "MGR collapsing empty level: level=%d configured=%d",
                            (int)lvl, (int)args->level[lvl].f_dofs.size);
         free(plan->c_dofs[lvl]);
         plan->c_dofs[lvl]     = NULL;
         plan->num_c_dofs[lvl] = 0;
         continue;
      }

      num_dofs_last -= num_level_f_dofs;
      plan->num_c_dofs[lvl] -= num_level_f_dofs;
      MGRPlanFillLevelCDofs(plan, lvl);
   }

   MGRPlanCompactLevels(args, plan, num_levels);

   return 1;
}

/*-----------------------------------------------------------------------------
 * Remap sparse dof labels to a dense space and assemble the point-marker
 * array handed to hypre.
 *-----------------------------------------------------------------------------*/

/* Sparse dof label spaces are remapped onto a dense [0, num_active_dofs) range
 * before the point markers are handed to hypre, which expects contiguous ids. */
static int
MGRPlanRemapSparseLabels(MGRCreatePlan *plan, const Stats *stats, int next_ls_id)
{
   HYPRE_Int lvl, i, j;

   plan->label_to_dense = (HYPRE_Int *)malloc((size_t)plan->num_dofs * sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (!plan->label_to_dense)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR dense label remap");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   for (i = 0; i < plan->num_dofs; i++)
   {
      plan->label_to_dense[i] = -1;
   }
   for (i = 0, j = 0; i < plan->num_dofs; i++)
   {
      if (plan->label_present[i])
      {
         plan->label_to_dense[i] = j++;
      }
   }
   plan->num_dofs_hypre = plan->num_active_dofs;

   for (lvl = 0; lvl < plan->num_levels - 1; lvl++)
   {
      for (i = 0; i < plan->num_c_dofs[lvl]; i++)
      {
         HYPRE_Int raw = plan->c_dofs[lvl][i];
         /* GCOVR_EXCL_START */
         if (raw < 0 || raw >= plan->num_dofs || plan->label_to_dense[raw] < 0)
         {
            HYPREDRV_LOG_COMMF(
               2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
               "MGR invalid C-point label during dense remap: raw=%d "
               "num_dofs=%d mapped=%d",
               (int)raw, (int)plan->num_dofs,
               (raw >= 0 && raw < plan->num_dofs) ? (int)plan->label_to_dense[raw] : -1);
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Invalid MGR C-point label %d during dense remap",
                                 (int)raw);
            return 0;
         }
         /* GCOVR_EXCL_STOP */
         plan->c_dofs[lvl][i] = plan->label_to_dense[raw];
      }
   }

   return 1;
}

static int
MGRPlanPointMarkers(MGR_args *args, MGRCreatePlan *plan, const Stats *stats,
                    int next_ls_id)
{
   IntArray *dofmap = args->dofmap;
   HYPRE_Int i;

   if (plan->num_active_dofs > 0 && plan->num_active_dofs < plan->num_dofs &&
       !MGRPlanRemapSparseLabels(plan, stats, next_ls_id))
   {
      return 0;
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after c-point assembly: code=0x%x num_dofs_hypre=%d",
                      hypredrv_ErrorCodeGet(), (int)plan->num_dofs_hypre);

   /* Set dofmap_data. Always take an owned copy rather than aliasing dofmap->data:
    * hypre reads this point-marker array during MGRSetup (after this function
    * returns), so aliasing the caller's dofmap makes hypre read freed memory if the
    * caller rebuilds/destroys the dofmap between PreconCreate and PreconSetup. The
    * copy is O(local rows), negligible next to MGR setup. */
   {
      plan->dofmap_data =
         (HYPRE_Int *)malloc((dofmap->size ? dofmap->size : 1) * sizeof(HYPRE_Int));
      /* GCOVR_EXCL_START */
      if (!plan->dofmap_data)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate MGR point-marker array");
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      plan->dofmap_data_owned = plan->dofmap_data;
      for (i = 0; i < (int)dofmap->size; i++)
      {
         HYPRE_Int raw = (HYPRE_Int)dofmap->data[i];
         if (plan->label_to_dense)
         {
            /* GCOVR_EXCL_START */
            if (raw < 0 || raw >= plan->num_dofs || plan->label_to_dense[raw] < 0)
            {
               HYPREDRV_LOG_COMMF(
                  2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                  "MGR invalid dof label during dense remap: raw=%d "
                  "num_dofs=%d mapped=%d",
                  (int)raw, (int)plan->num_dofs,
                  (raw >= 0 && raw < plan->num_dofs) ? (int)plan->label_to_dense[raw]
                                                     : -1);
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               hypredrv_ErrorMsgAdd("Invalid dof label %d during MGR dense remap",
                                    (int)raw);
               return 0;
            }
            /* GCOVR_EXCL_STOP */
            plan->dofmap_data[i] = plan->label_to_dense[raw];
         }
         /* GCOVR_EXCL_START */
         else
         {
            plan->dofmap_data[i] = raw;
         }
         /* GCOVR_EXCL_STOP */
      }
   }

   return 1;
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * Transfer scalar MGR options to the hypre solver object.
 *-----------------------------------------------------------------------------*/

static void
MGRApplyBaseSettings(HYPRE_Solver precon, MGR_args *args, MGRCreatePlan *plan,
                     HYPRE_Int any_polynomial_matched_q, const Stats *stats,
                     int next_ls_id)
{
   HYPRE_Int relax_type = args->relax_type;

   HYPRE_MGRSetCpointsByPointMarkerArray(precon, plan->num_dofs_hypre,
                                         plan->num_levels - 1, plan->num_c_dofs,
                                         plan->c_dofs, plan->dofmap_data);
   HYPRE_MGRSetNonCpointsToFpoints(precon, args->non_c_to_f);
   HYPRE_MGRSetPMaxElmts(precon, args->pmax);
#if HYPREDRV_HAS_MGR_DEV_FEATURES
   HYPRE_MGRSetNumInterpSweeps(precon, args->interp_sweeps);
   if (args->interp_sweeps > 0)
   {
      HYPRE_MGRSetInterpRelaxWeight(precon, args->interp_weight);
   }
   HYPRE_MGRSetP2InterpRefinement(precon, args->interp_sweeps > 0);
   HYPRE_MGRSetInjectionUpcycle(precon, args->injection_upcycle);
   if (any_polynomial_matched_q)
   {
      HYPRE_MGRSetMatchedQSweeps(precon, args->matched_q_sweeps);
      HYPRE_MGRSetMatchedQRelaxWeight(precon, args->matched_q_weight);
   }
#else
   (void)any_polynomial_matched_q;
#endif
   HYPRE_MGRSetMaxIter(precon, args->max_iter);
   HYPRE_MGRSetTol(precon, args->tolerance);
   HYPRE_MGRSetPrintLevel(precon, args->print_level);
#if HYPRE_CHECK_MIN_VERSION(30100, 50)
   {
      HYPRE_MGRSetCycleType(precon, args->cycle);
      HYPRE_MGRSetFRelaxCycle(precon, args->cycle_smooth_pos);
      HYPRE_MGRSetGlobalSmoothCycle(precon, args->cycle_smooth_pos);
   }
#else
   if (args->cycle != 1 || args->cycle_smooth_pos != 1)
   {
      HYPREDRV_LOG_COMMF(
         2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
         "MGR cycle setting is ignored because this hypre version does not support "
         "MGR cycle control APIs");
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22000, 0)
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
#endif
   HYPRE_MGRSetRelaxType(precon, relax_type); /* TODO: we shouldn't need this */
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after base hypre setup: code=0x%x",
                      hypredrv_ErrorCodeGet());
}

/*-----------------------------------------------------------------------------
 * Transfer the per-level type/sweep/transfer-operator arrays to hypre.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
static void
MGRApplyLevelSettings(HYPRE_Solver precon, MGR_args *args, const MGRCreatePlan *plan,
                      const Stats *stats, int next_ls_id)
{
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_Int level_frelax_type[MAX_MGR_LEVELS - 1]         = {0};
   HYPRE_Int level_frelax_sweeps[MAX_MGR_LEVELS - 1]       = {0};
   HYPRE_Int level_grelax_type[MAX_MGR_LEVELS - 1]         = {0};
   HYPRE_Int level_grelax_sweeps[MAX_MGR_LEVELS - 1]       = {0};
   HYPRE_Int level_interp_type[MAX_MGR_LEVELS - 1]         = {0};
   HYPRE_Int level_restrict_type[MAX_MGR_LEVELS - 1]       = {0};
   HYPRE_Int level_coarse_type[MAX_MGR_LEVELS - 1]         = {0};
   HYPRE_Int level_matched_q[MAX_MGR_LEVELS - 1]           = {0};
   HYPRE_Int level_matched_f_backsolve[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int any_matched_q                                 = 0;
   HYPRE_Int any_matched_f_backsolve                       = 0;

   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      HYPRE_Int    orig_lvl   = plan->active_level_map[i];
      MGRlvl_args *level_args = &args->level[orig_lvl];
      HYPRE_Int    type       = level_args->f_relaxation.type;

      if (level_args->f_relaxation.use_krylov)
      {
         level_frelax_type[i] = MGR_FRLX_TYPE_CUSTOM_SOLVER_CB;
      }
      else if (type == MGR_FRLX_TYPE_NESTED_MGR)
      {
         level_frelax_type[i] = 7;
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      else if (type == MGR_SOLVER_TYPE_SCHWARZ)
      {
         level_frelax_type[i] = MGR_FRLX_TYPE_CUSTOM_SOLVER_CB;
      }
#endif
      else
      {
         level_frelax_type[i] = type;
      }
      level_frelax_sweeps[i] = level_args->f_relaxation.num_sweeps;
      level_grelax_type[i]   = level_args->g_relaxation.type;
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
      /* HYPRE_MGRSetGlobalSmootherAtLevel below installs these user-owned
       * solvers and determines their effective type. Advertising a concrete
       * built-in type first makes upstream hypre raise HYPRE_ERROR_GENERIC
       * while harmlessly resetting it, which must not turn a valid setup into
       * a HypreDrive failure. */
      if (hypredrv_MGRGRelaxUsesUserSmoother(&level_args->g_relaxation))
      {
         level_grelax_type[i] = MGR_GRLX_TYPE_USER_SMOOTHER;
      }
#endif
      level_grelax_sweeps[i] = level_args->g_relaxation.num_sweeps;
      level_interp_type[i]   = hypredrv_MGRLevelInterpTypeCompat(
         level_args->prolongation_type, stats, next_ls_id, orig_lvl);
      level_restrict_type[i] = level_args->restriction_type;
      level_coarse_type[i]   = level_args->coarse_level_type;
      level_matched_q[i]     = level_args->matched_q;
      any_matched_q |= level_args->matched_q;
      level_matched_f_backsolve[i] = level_args->matched_f_backsolve;
      any_matched_f_backsolve |= level_args->matched_f_backsolve;
   }

   HYPRE_MGRSetLevelFRelaxType(precon, level_frelax_type);
   HYPRE_MGRSetLevelNumRelaxSweeps(precon, level_frelax_sweeps);
   HYPRE_MGRSetLevelSmoothType(precon, level_grelax_type);
   HYPRE_MGRSetLevelSmoothIters(precon, level_grelax_sweeps);
   HYPRE_MGRSetLevelInterpType(precon, level_interp_type);
   HYPRE_MGRSetLevelRestrictType(precon, level_restrict_type);
   HYPRE_MGRSetCoarseGridMethod(precon, level_coarse_type);
#if HYPREDRV_HAS_MGR_DEV_FEATURES
   if (any_matched_q)
   {
      HYPRE_MGRSetLevelMatchedQ(precon, level_matched_q);
   }
   if (any_matched_f_backsolve)
   {
      HYPRE_MGRSetLevelMatchedFBacksolve(precon, level_matched_f_backsolve);
   }
#else
   (void)level_matched_q;
   (void)any_matched_q;
   (void)level_matched_f_backsolve;
   (void)any_matched_f_backsolve;
#endif
#else
   (void)precon;
   (void)args;
   (void)plan;
   (void)stats;
   (void)next_ls_id;
#endif
}

/* Install and validate exactly the matrices selected by coarse_level_type: user. */
static int
MGRInstallUserCoarseMatrices(HYPRE_Solver precon, MGR_args *args,
                             const MGRCreatePlan *plan)
{
#if !HYPRE_CHECK_MIN_VERSION(30100, 77)
   (void)precon;
#endif
   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      HYPRE_Int orig_lvl = plan->active_level_map[i];
      if (args->level[orig_lvl].coarse_level_type != 6)
      {
         continue;
      }
#if !HYPRE_CHECK_MIN_VERSION(30100, 77)
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR coarse_level_type 'user' requires hypre >= 3.1.0 "
                           "(develop 77), which provides "
                           "HYPRE_MGRSetCoarseGridMatrixAtLevel");
      return 0;
#else
      HYPRE_ParCSRMatrix coarse_par = NULL;
      if (!args->coarse_schur || !args->coarse_schur[orig_lvl] ||
          HYPRE_IJMatrixGetObject(args->coarse_schur[orig_lvl], (void **)&coarse_par) ||
          !coarse_par)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR coarse_level_type 'user' requires an assembled application-provided "
            "coarse matrix for level %d",
            (int)orig_lvl);
         return 0;
      }
      HYPRE_MGRSetCoarseGridMatrixAtLevel(precon, i, coarse_par);
#endif
   }
   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the hypredrive-managed F-relaxation handle on one level, reusing
 * a cached solver when available.
 *-----------------------------------------------------------------------------*/

static int
MGRConfigManagedFRelax(MGR_args *args, HYPRE_Solver precon, HYPRE_Int active_lvl,
                       HYPRE_Int orig_lvl, const Stats *stats, int next_ls_id)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];
   HYPRE_Solver frelax     = args->frelax[orig_lvl];

   if (!frelax)
   {
      frelax = hypredrv_MGRFRelaxSolverCreateByType(args, &level_args->f_relaxation,
                                                    &level_args->f_dofs, active_lvl);
      if (hypredrv_ErrorCodeActive() || !frelax)
      {
         return 0;
      }
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR F-relax solver handle at level %d",
                         (int)orig_lvl);
   }

   hypredrv_MGRFRelaxInstall(precon, &level_args->f_relaxation, frelax, active_lvl);
   args->frelax[orig_lvl] = frelax;
   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the F-relaxation solver of each active MGR level.
 *-----------------------------------------------------------------------------*/

/* symmetric_diagonal_scaling only applies to a managed AMG F-solver. */
static int
MGRValidateFRelaxDiagScaling(const MGR_args *args, const MGRlvl_args *level_args)
{
   if (level_args->f_relaxation.symmetric_diagonal_scaling)
   {
#if !HYPRE_CHECK_MIN_VERSION(30100, 0)
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR symmetric_diagonal_scaling requires hypre >= 3.1.0");
      return 0;
#else
      if (level_args->f_relaxation.symmetric_diagonal_scaling != 1 ||
          level_args->f_relaxation.type != 2 || level_args->f_relaxation.use_krylov ||
          level_args->f_relaxation.reuse.present || args->vec_nn)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR symmetric_diagonal_scaling currently requires a managed AMG "
            "F-solver without nested Krylov, component reuse, or projected RBMs");
         return 0;
      }
#endif
   }

   return 1;
}

/* Builds (or reuses) the nested Krylov F-solver and attaches it at level `i`. */
static int
MGRConfigNestedKrylovFRelax(MGR_args *args, HYPRE_Solver precon, MGRlvl_args *level_args,
                            HYPRE_Int i, HYPRE_Int orig_lvl, const Stats *stats,
                            int next_ls_id)
{
   int krylov_was_cached = (level_args->f_relaxation.krylov->base_solver != NULL);
   if (!krylov_was_cached)
   {
      /* Nested-Krylov F-relaxation whose preconditioner is a nodal AMG:
       * project the rigid-body modes onto this level's F-points before the
       * AMG is built, so its GM interpolation resolves the elasticity
       * rotation modes. Without this, PreconCreate would attach full-system
       * modes that overrun the extracted A_FF during interpolation setup. */
      if (!hypredrv_MGRProjectNestedRBMs(level_args->f_relaxation.krylov, args,
                                         level_args->f_dofs.data,
                                         level_args->f_dofs.size))
      {
         return 0;
      }
      hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, level_args->f_relaxation.krylov,
                                  args->dofmap, args->vec_nn,
                                  &level_args->f_relaxation.krylov->base_solver);
      /* GCOVR_EXCL_START */
      if (hypredrv_ErrorCodeActive())
      {
         return 0;
      }
      /* GCOVR_EXCL_STOP */
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR F-relax nested Krylov handle at level %d",
                         (int)orig_lvl);
   }
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   hypredrv_MGRSetFSolverAtLevel(precon, (HYPRE_Solver)level_args->f_relaxation.krylov, i,
                                 level_args->f_relaxation.type,
                                 hypredrv_MGRBaseParSolverSolve,
                                 hypredrv_MGRBaseParSolverSetup);
#else
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("Nested Krylov F-relaxation requires hypre >= 2.31.0");
   return 0;
#endif

   return 1;
}

/* Builds a nested MGR hierarchy over this level's F-points and attaches it. */
static int
MGRConfigNestedMGRFRelax(MGR_args *args, HYPRE_Solver precon, MGRlvl_args *level_args,
                         HYPRE_Int i, HYPRE_Int orig_lvl, const Stats *stats,
                         int next_ls_id)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   if (i != 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Nested MGR F-relaxation is only supported at MGR level 0 by hypre");
      return 0;
   }

   MGR_args      *nested_args    = level_args->f_relaxation.mgr;
   HYPRE_Solver   frelax         = NULL;
   HYPRE_Solver   frelax_wrapper = NULL;
   IntArray      *nested_dofmap  = NULL;
   IntArray      *saved_dofmap   = NULL;
   HYPRE_IJVector saved_vec_nn   = NULL;

   if (!nested_args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
      return 0;
   }

   nested_dofmap =
      hypredrv_MGRBuildProjectedFRelaxDofmap(args->dofmap, &level_args->f_dofs);
   if (hypredrv_ErrorCodeActive() || !nested_dofmap)
   {
      hypredrv_IntArrayDestroy(&nested_dofmap);
      return 0;
   }

   saved_dofmap        = nested_args->dofmap;
   saved_vec_nn        = nested_args->vec_nn;
   nested_args->dofmap = nested_dofmap;
   nested_args->vec_nn = NULL;
   HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "creating nested MGR F-relaxation at level %d "
                      "(coarsest=%s)",
                      (int)orig_lvl,
                      hypredrv_MGRCoarseSolverTypeName(&nested_args->coarsest_level));
   hypredrv_MGRCreate(nested_args, &frelax, stats, next_ls_id);
   nested_args->dofmap = saved_dofmap;
   nested_args->vec_nn = saved_vec_nn;
   /* GCOVR_EXCL_START */
   if (hypredrv_ErrorCodeActive())
   {
      hypredrv_IntArrayDestroy(&nested_dofmap);
      return 0;
   }

   frelax_wrapper =
      hypredrv_MGRNestedFRelaxWrapperCreate(frelax, nested_args, nested_dofmap);
   if (hypredrv_ErrorCodeActive() || !frelax_wrapper)
   {
      HYPRE_MGRDestroy(frelax);
      hypredrv_IntArrayDestroy(&nested_dofmap);
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   nested_dofmap = NULL;
   hypredrv_MGRSetFSolverAtLevel(precon, frelax_wrapper, i, level_args->f_relaxation.type,
                                 NULL, NULL);
   args->frelax[orig_lvl] = frelax_wrapper;
#else
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("Nested MGR F-relaxation requires hypre >= 3.1.0 (develop >= 5)");
   return 0;
#endif

   return 1;
}

/* Attaches the configured F-relaxation solver for one active level. */
/* Types 29 (spdirect), 32, 33 (fsai) and Schwarz all resolve to the same
 * managed F-relaxation call; only the hypre-version/feature gate differs. */
static int
MGRConfigGatedManagedFRelax(MGR_args *args, HYPRE_Solver precon, HYPRE_Int i,
                            HYPRE_Int orig_lvl, const MGRlvl_args *level_args,
                            const Stats *stats, int next_ls_id)
{
   if (level_args->f_relaxation.type == 29)
   {
#if defined(HYPRE_USING_DSUPERLU) && HYPRE_CHECK_MIN_VERSION(23100, 9)
      /* GCOVR_EXCL_START */
      if (!MGRConfigManagedFRelax(args, precon, i, orig_lvl, stats, next_ls_id))
      {
         return 0;
      }
      /* GCOVR_EXCL_STOP */
#elif defined(HYPRE_USING_DSUPERLU)
      /* GCOVR_EXCL_START */
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation 'spdirect' requires hypre >= 2.31.0");
      return 0;
      /* GCOVR_EXCL_STOP */
#else
      /* GCOVR_EXCL_START */
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR F-relaxation 'spdirect' requires hypre built with DSUPERLU");
      return 0;
      /* GCOVR_EXCL_STOP */
#endif
   }
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
   else if (level_args->f_relaxation.type == 32)
   {
      if (!MGRConfigManagedFRelax(args, precon, i, orig_lvl, stats, next_ls_id))
      {
         return 0;
      }
   }
#endif
   else if (level_args->f_relaxation.type == 33)
   {
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
      if (!MGRConfigManagedFRelax(args, precon, i, orig_lvl, stats, next_ls_id))
      {
         return 0;
      }
#else
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation 'fsai' requires hypre >= 2.31.0");
      return 0;
#endif
   }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (level_args->f_relaxation.type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      if (!MGRConfigManagedFRelax(args, precon, i, orig_lvl, stats, next_ls_id))
      {
         return 0;
      }
   }
#endif

   return 1;
}

static int
MGRConfigFRelaxAtLevel(MGR_args *args, HYPRE_Solver precon, HYPRE_Int i,
                       HYPRE_Int orig_lvl, const Stats *stats, int next_ls_id)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];

   if (!MGRValidateFRelaxDiagScaling(args, level_args))
   {
      return 0;
   }

   if (level_args->f_relaxation.use_krylov && level_args->f_relaxation.krylov)
   {
      if (!MGRConfigNestedKrylovFRelax(args, precon, level_args, i, orig_lvl, stats,
                                       next_ls_id))
      {
         return 0;
      }
   }
   else if (level_args->f_relaxation.type == 2)
   {
      if (!MGRConfigManagedFRelax(args, precon, i, orig_lvl, stats, next_ls_id))
      {
         return 0;
      }
   }
   else if (level_args->f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
   {
      if (!MGRConfigNestedMGRFRelax(args, precon, level_args, i, orig_lvl, stats,
                                    next_ls_id))
      {
         return 0;
      }
   }
   else if (!MGRConfigGatedManagedFRelax(args, precon, i, orig_lvl, level_args, stats,
                                         next_ls_id))
   {
      return 0;
   }

   return 1;
}

static int
MGRConfigFRelaxSolvers(MGR_args *args, HYPRE_Solver precon, const MGRCreatePlan *plan,
                       const Stats *stats, int next_ls_id)
{
   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      if (!MGRConfigFRelaxAtLevel(args, precon, i, plan->active_level_map[i], stats,
                                  next_ls_id))
      {
         return 0;
      }
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after F-relax setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the hypredrive-managed global smoother handle on one level,
 * reusing a cached solver when available.
 *-----------------------------------------------------------------------------*/

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
static int
MGRConfigManagedGRelax(MGR_args *args, HYPRE_Solver precon, HYPRE_Int active_lvl,
                       HYPRE_Int orig_lvl, const Stats *stats, int next_ls_id)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];
   HYPRE_Solver grelax     = args->grelax[orig_lvl];

   if (!grelax)
   {
      grelax = hypredrv_MGRGRelaxSolverCreateByType(&level_args->g_relaxation);
      if (hypredrv_ErrorCodeActive() || !grelax)
      {
         return 0;
      }
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR G-relax solver handle at level %d",
                         (int)orig_lvl);
   }

   HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, active_lvl);
   args->grelax[orig_lvl] = grelax;
   return 1;
}
#endif

/*-----------------------------------------------------------------------------
 * Configure the global relaxation solver of each active MGR level.
 *-----------------------------------------------------------------------------*/

static int
MGRConfigGRelaxSolvers(MGR_args *args, HYPRE_Solver precon, const MGRCreatePlan *plan,
                       const Stats *stats, int next_ls_id)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      HYPRE_Int    orig_lvl   = plan->active_level_map[i];
      MGRlvl_args *level_args = &args->level[orig_lvl];

      if (level_args->g_relaxation.use_krylov && level_args->g_relaxation.krylov)
      {
         int krylov_was_cached = (level_args->g_relaxation.krylov->base_solver != NULL);
         if (!krylov_was_cached)
         {
            if (!hypredrv_MGRProjectNestedRemainingRBMs(level_args->g_relaxation.krylov,
                                                        args, i))
            {
               return 0;
            }
            hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, level_args->g_relaxation.krylov,
                                        args->dofmap, args->vec_nn,
                                        &level_args->g_relaxation.krylov->base_solver);
            /* GCOVR_EXCL_START */
            if (hypredrv_ErrorCodeActive())
            {
               return 0;
            }
            /* GCOVR_EXCL_STOP */
         }
         else
         {
            HYPREDRV_LOG_COMMF(
               2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
               "reusing cached MGR G-relax nested Krylov handle at level %d",
               (int)orig_lvl);
         }
         HYPRE_MGRSetGlobalSmootherAtLevel(
            precon, (HYPRE_Solver)level_args->g_relaxation.krylov, i);
      }
      else if (level_args->g_relaxation.type == 20 || level_args->g_relaxation.type == 16)
      {
         if (!MGRConfigManagedGRelax(args, precon, i, orig_lvl, stats, next_ls_id))
         {
            return 0;
         }
      }
      else if (level_args->g_relaxation.type == 29)
      {
#ifdef HYPRE_USING_DSUPERLU
         /* GCOVR_EXCL_START */
         if (!MGRConfigManagedGRelax(args, precon, i, orig_lvl, stats, next_ls_id))
         {
            return 0;
         }
         /* GCOVR_EXCL_STOP */
#else
         /* GCOVR_EXCL_START */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR G-relaxation 'spdirect' requires hypre built with DSUPERLU");
         return 0;
         /* GCOVR_EXCL_STOP */
#endif
      }
      else if (level_args->g_relaxation.type == 33)
      {
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
         if (!MGRConfigManagedGRelax(args, precon, i, orig_lvl, stats, next_ls_id))
         {
            return 0;
         }
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR G-relaxation 'fsai' requires hypre >= 2.25.0");
         return 0;
#endif
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      else if (level_args->g_relaxation.type == MGR_SOLVER_TYPE_SCHWARZ)
      {
         if (!MGRConfigManagedGRelax(args, precon, i, orig_lvl, stats, next_ls_id))
         {
            return 0;
         }
      }
#endif
   }
#else
   (void)args;
   (void)precon;
   (void)plan;
#endif
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after G-relax setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the coarsest-level solver (nested Krylov or a managed handle).
 *-----------------------------------------------------------------------------*/

/* The built-in coarsest solvers (AMG, ILU, direct, ...) as opposed to a
 * user-supplied nested Krylov solver. */
static int
MGRConfigManagedCoarsestSolver(MGR_args *args, HYPRE_Solver precon, const Stats *stats,
                               int next_ls_id)
{
   /* Infer coarsest level solver type if not explicitly set (type == -1).
    * This allows both patterns:
    *   coarsest_level: spdirect        -> type = 29 (explicitly set)
    *   coarsest_level: { ilu: {...} }  -> type inferred from ilu.max_iter > 0
    */
   if (args->coarsest_level.type == -1)
   {
      /* Default to AMG unless the user explicitly selected ILU. */
      args->coarsest_level.type = 0;
   }

   HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR coarsest solver selected: %s",
                      hypredrv_MGRCoarseSolverTypeName(&args->coarsest_level));

   /* Ensure the selected solver has valid max_iter */
   /* GCOVR_EXCL_START */
   if (args->coarsest_level.type == 0 && args->coarsest_level.amg.max_iter < 1)
   {
      args->coarsest_level.amg.max_iter = 1;
   }
   else if (args->coarsest_level.type == 32 && args->coarsest_level.ilu.max_iter < 1)
   {
      args->coarsest_level.ilu.max_iter = 1;
   }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (args->coarsest_level.type == MGR_SOLVER_TYPE_SCHWARZ &&
            args->coarsest_level.schwarz.max_iter < 1)
   {
      args->coarsest_level.schwarz.max_iter = 1;
   }
#endif
   /* GCOVR_EXCL_STOP */

   HYPRE_Int type = args->coarsest_level.type;

#if !defined(HYPRE_USING_DSUPERLU)
   /* GCOVR_EXCL_START */
   if (type == 29)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR coarsest_level 'spdirect' requires hypre built with DSUPERLU");
      return 0;
   }
   /* GCOVR_EXCL_STOP */
#endif
#if !HYPRE_CHECK_MIN_VERSION(22500, 0)
   if (type == 33)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR coarsest_level 'fsai' requires hypre >= 2.25.0");
      return 0;
   }
#endif

   int csolver_was_cached = (args->csolver && args->csolver_type == type);
   if (!csolver_was_cached)
   {
      HYPRE_Solver new_solver =
         hypredrv_MGRCoarseSolverCreateByType(&args->coarsest_level, type);
      if (hypredrv_ErrorCodeActive() || !new_solver)
      {
         return 0;
      }
      hypredrv_MGRCoarseSolverDestroyByType(args->csolver_type, &args->csolver);
      args->csolver = new_solver;
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR coarsest solver handle");
   }
   args->csolver_type = type;
   hypredrv_MGRCoarseSolverInstall(precon, type, args->csolver);

   return 1;
}

static int
MGRConfigCoarsestSolver(MGR_args *args, HYPRE_Solver precon, const Stats *stats,
                        int next_ls_id)
{
   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      int krylov_was_cached = (args->coarsest_level.krylov->base_solver != NULL);
      if (!krylov_was_cached)
      {
         if (!hypredrv_MGRProjectNestedRemainingRBMs(args->coarsest_level.krylov, args,
                                                     args->num_active_levels))
         {
            return 0;
         }
         hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, args->coarsest_level.krylov,
                                     args->dofmap, args->vec_nn,
                                     &args->coarsest_level.krylov->base_solver);
         /* GCOVR_EXCL_START */
         if (hypredrv_ErrorCodeActive())
         {
            return 0;
         }
         /* GCOVR_EXCL_STOP */
      }
      else
      {
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id,
                            "reusing cached MGR coarsest nested Krylov handle");
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(precon, hypredrv_MGRBaseParSolverSolve,
                               hypredrv_MGRBaseParSolverSetup,
                               (HYPRE_Solver)args->coarsest_level.krylov);
#else
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Nested Krylov coarsest solver requires hypre >= 3.1.0");
      return 0;
#endif
   }
   else if (!MGRConfigManagedCoarsestSolver(args, precon, stats, next_ls_id))
   {
      return 0;
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after coarsest solver setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   return 1;
}
/* GCOVR_EXCL_BR_STOP */

/* Aggregated facts about the active hierarchy, gathered by MGRScanActiveLevels
 * and consumed by the validators that follow it. */
typedef struct
{
   HYPRE_Int any_matched_q;
   HYPRE_Int any_polynomial_matched_q;
   HYPRE_Int any_afsai_matched_q;
   HYPRE_Int selected_count;
   HYPRE_Int saw_p2;
   HYPRE_Int selected_active_level;
   HYPRE_Int selected_mode;
} MGRFeatureScan;

/* The validators below reject unsupported MGR option combinations before any
 * hypre object is created. Each returns nonzero when the configuration is
 * acceptable, or zero with the error state already populated. */
/* GCOVR_EXCL_BR_START */

static int
MGRValidateInterpSweeps(const MGR_args *args)
{
   if (args->interp_sweeps < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR interp_sweeps must be nonnegative");
      return 0;
   }

   if (args->interp_sweeps > 0)
   {
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR interp_sweeps requires a hypre build with bounded P2 refinement");
      return 0;
#else
      if (args->pmax <= 0 ||
          !(args->interp_weight >= HYPRE_REAL_MIN && args->interp_weight <= 1.0))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR interp_sweeps requires pmax > 0 and interp_weight in "
                              "[HYPRE_REAL_MIN,1]");
         return 0;
      }
#endif
   }
   return 1;
}

static int
MGRValidateInjectionUpcycle(const MGR_args *args, const MGRCreatePlan *plan)
{
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
   (void)plan;
#endif

   if (args->injection_upcycle)
   {
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR injection_upcycle requires a hypre build advertising that feature");
      return 0;
#else
      if (args->injection_upcycle != 1 || args->cycle != 1 ||
          args->cycle_smooth_pos != 3 || args->interp_sweeps != 0 ||
          plan->num_levels <= 1)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR injection_upcycle requires an active V(1,1) reduction hierarchy "
            "and interp_sweeps: 0");
         return 0;
      }
#endif
   }
   return 1;
}

static int
MGRValidateLevelEnums(const MGR_args *args)
{
   for (HYPRE_Int orig_lvl = 0; orig_lvl < args->num_levels - 1; orig_lvl++)
   {
      const MGRlvl_args *level = &args->level[orig_lvl];
      if (level->matched_q < 0 || level->matched_q > 2)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR level matched_q must be 'off', 'polynomial'/'on', or 'afsai'");
         return 0;
      }
      if (level->matched_f_backsolve < 0 || level->matched_f_backsolve > 2)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR level matched_f_backsolve must be 'off', 'on', or 'gmres1'");
         return 0;
      }
   }
   return 1;
}

#if HYPREDRV_HAS_MGR_DEV_FEATURES
/* Refined interpolation needs a plain P2/R0/Galerkin reduction at every level. */
static int
MGRLevelSupportsInterpRefinement(const MGRlvl_args *level)
{
   return !((level->prolongation_type >= 12 && level->prolongation_type <= 14) ||
            (level->prolongation_type == 2 &&
             (level->restriction_type != 0 || level->coarse_level_type != 0)));
}

/* Injection up-cycling needs P2/R0/Galerkin plus active F relaxation. */
static int
MGRLevelSupportsInjectionUpcycle(const MGRlvl_args *level)
{
   return (level->prolongation_type == 2 && level->restriction_type == 0 &&
           level->coarse_level_type == 0 && level->f_relaxation.type >= 0 &&
           level->f_relaxation.num_sweeps > 0);
}

/* Matched sparse Q needs jacobi P2, injection R0, Galerkin, exactly one built-in
 * Jacobi F sweep, and no global smoother. */
static int
MGRLevelSupportsMatchedQ(const MGRlvl_args *level)
{
   return (level->prolongation_type == 2 && level->restriction_type == 0 &&
           level->coarse_level_type == 0 && level->f_relaxation.type == 7 &&
           level->f_relaxation.num_sweeps == 1 && !level->f_relaxation.use_krylov &&
           level->g_relaxation.type < 0 && !level->g_relaxation.use_krylov);
}
#endif

/* Accumulates one active level's feature flags and rejects option conflicts. */
static int
MGRScanActiveLevel(const MGR_args *args, const MGRlvl_args *level, HYPRE_Int i,
                   MGRFeatureScan *scan)
{
#if HYPREDRV_HAS_MGR_DEV_FEATURES
   scan->saw_p2 |= level->prolongation_type == 2;
#endif
   scan->any_matched_q |= level->matched_q;
   scan->any_polynomial_matched_q |= level->matched_q == 1;
   scan->any_afsai_matched_q |= level->matched_q == 2;

   if (level->matched_f_backsolve)
   {
#if HYPREDRV_HAS_MGR_DEV_FEATURES
      scan->selected_active_level = i;
      scan->selected_mode         = level->matched_f_backsolve;
#endif
      scan->selected_count++;
   }

#if !HYPREDRV_HAS_MGR_DEV_FEATURES
   (void)args;
   (void)i;
#else
   if (args->interp_sweeps > 0 && !MGRLevelSupportsInterpRefinement(level))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR interp_sweeps requires P2/R0/Galerkin at every refined level "
         "and does not support block-Jacobi interpolation");
      return 0;
   }

   if (args->injection_upcycle && !MGRLevelSupportsInjectionUpcycle(level))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR injection_upcycle requires P2/R0/Galerkin and active F "
                           "relaxation at every reduction level");
      return 0;
   }

   if (level->matched_q && !MGRLevelSupportsMatchedQ(level))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR matched_q levels require jacobi P2, injection R0, Galerkin, "
         "one built-in Jacobi F sweep, and no global smoother");
      return 0;
   }
#endif

   return 1;
}

/* Walks the compacted level list, accumulating the feature flags the later
 * validators need while rejecting per-level option conflicts. */
static int
MGRScanActiveLevels(const MGR_args *args, const MGRCreatePlan *plan, MGRFeatureScan *scan)
{
   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      const MGRlvl_args *level = &args->level[plan->active_level_map[i]];

      if (!MGRScanActiveLevel(args, level, i, scan))
      {
         return 0;
      }
   }

#if HYPREDRV_HAS_MGR_DEV_FEATURES
   if (args->interp_sweeps > 0 && !scan->saw_p2)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR interp_sweeps requires at least one active P2 reduction level");
      return 0;
   }
#endif

   return 1;
}

static int
MGRValidateMatchedQ(const MGR_args *args, const MGRCreatePlan *plan,
                    const MGRFeatureScan *scan)
{
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
   (void)args;
   (void)plan;
#endif

   if (scan->any_matched_q)
   {
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR matched_q requires a hypre build advertising matched sparse Q");
      return 0;
#else
      if (args->interp_sweeps != 0 || args->injection_upcycle || args->cycle != 1 ||
          args->cycle_smooth_pos != 1 || args->coarse_th != 0.0 || plan->num_levels <= 1)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR matched_q requires interp_sweeps: 0, coarse_th: 0, "
                              "injection_upcycle: off, and "
                              "an active pre-only V-cycle hierarchy");
         return 0;
      }
      if (scan->any_polynomial_matched_q &&
          (args->pmax <= 0 || args->matched_q_sweeps <= 0 ||
           !(args->matched_q_weight >= HYPRE_REAL_MIN && args->matched_q_weight <= 1.0)))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR polynomial matched_q requires pmax > 0, matched_q_sweeps > 0, "
            "and matched_q_weight in [HYPRE_REAL_MIN,1]");
         return 0;
      }
      if (scan->any_afsai_matched_q)
      {
#if !HYPREDRV_HAS_MGR_DEV_FEATURES || defined(HYPRE_COMPLEX)
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR matched_q: afsai requires a real hypre build advertising matched "
            "aFSAI Q");
         return 0;
#else
         if (args->pmax != 4)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR matched_q: afsai requires pmax: 4 as the FSAI factor row bound");
            return 0;
         }
#endif
      }
#endif
   }

   return 1;
}

#if HYPREDRV_HAS_MGR_DEV_FEATURES
/* gmres1 back-solve needs a managed AMG coarsest solver pinned to one
 * iteration, with no nested Krylov or component reuse of its own. */
static int
MGRBacksolveCoarsestIsManagedAMG(const MGR_args *args)
{
   const HYPRE_Int coarse_type =
      args->coarsest_level.type < 0 ? 0 : args->coarsest_level.type;

   return (coarse_type == 0 && !args->coarsest_level.use_krylov &&
           !args->coarsest_level.reuse.present &&
           args->coarsest_level.amg.max_iter == 1 &&
           args->coarsest_level.amg.tolerance == 0.0);
}

/* The back-solve is only defined for a single selected final reduction level
 * driving a V(1,0) cycle that MGR runs exactly once. */
static int
MGRBacksolveGlobalOptionsOk(const MGR_args *args, const MGRCreatePlan *plan,
                            const MGRFeatureScan *scan)
{
   return (scan->selected_count == 1 &&
           scan->selected_active_level == plan->num_levels - 2 && args->max_iter == 1 &&
           args->tolerance == 0.0 && args->cycle == 1 && args->cycle_smooth_pos == 1 &&
           args->interp_sweeps == 0 && !args->injection_upcycle && !scan->any_matched_q &&
           args->coarse_th == 0.0);
}

/* The selected level must reduce with P2/R0/Galerkin and relax F with a single
 * managed, symmetrically scaled AMG sweep and no global smoother. */
static int
MGRBacksolveLevelOk(const MGR_args *args, const MGRlvl_args *level)
{
   const MGRfrlx_args *frelax = &level->f_relaxation;
   const MGRgrlx_args *grelax = &level->g_relaxation;

   return (level->prolongation_type == 2 && level->restriction_type == 0 &&
           level->coarse_level_type == 0 && frelax->type == 2 &&
           frelax->num_sweeps == 1 && frelax->symmetric_diagonal_scaling == 1 &&
           !frelax->use_krylov && !frelax->reuse.present && !args->vec_nn &&
           frelax->amg.max_iter == 1 && frelax->amg.tolerance == 0.0 &&
           grelax->type < 0 && !grelax->use_krylov);
}
#endif

static int
MGRValidateMatchedFBacksolve(const MGR_args *args, const MGRCreatePlan *plan,
                             const MGRFeatureScan *scan)
{
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
   (void)args;
   (void)plan;
#endif

   /* A selected configured level that collapsed because its F labels are
    * absent is intentionally a no-op for this system. */
   if (scan->selected_count > 0)
   {
#if !HYPREDRV_HAS_MGR_DEV_FEATURES
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR matched_f_backsolve requires a hypre build advertising that feature");
      return 0;
#else
      /* The enclosing #else already establishes the capability; only the
       * coarsest-solver contract remains to be checked here. */
      if (scan->selected_mode == 2 && !MGRBacksolveCoarsestIsManagedAMG(args))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR matched_f_backsolve: gmres1 requires a managed AMG "
                              "coarsest solver with max_iter: 1, tolerance: 0, and no "
                              "nested Krylov or component reuse");
         return 0;
      }

      if (!MGRBacksolveGlobalOptionsOk(args, plan, scan))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR matched_f_backsolve requires exactly one selected final active "
            "reduction level, max_iter: 1, tolerance: 0, a V(1,0) cycle, "
            "interp_sweeps: 0, coarse_th: 0, injection_upcycle: off, and "
            "matched_q: off");
         return 0;
      }

      if (!MGRBacksolveLevelOk(
             args, &args->level[plan->active_level_map[scan->selected_active_level]]))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR matched_f_backsolve level requires P2/R0/Galerkin, one managed "
            "symmetric-diagonally-scaled AMG F sweep with max_iter: 1 and "
            "tolerance: 0, no nested Krylov/component reuse/projected RBMs, and "
            "no global relaxation");
         return 0;
      }
#endif
   }

   return 1;
}

/* Runs the full option-compatibility cascade for a planned MGR hierarchy. */
static int
MGRValidateConfiguration(const MGR_args *args, const MGRCreatePlan *plan,
                         MGRFeatureScan *scan)
{
   return (MGRValidateInterpSweeps(args) && MGRValidateInjectionUpcycle(args, plan) &&
           MGRValidateLevelEnums(args) && MGRScanActiveLevels(args, plan, scan) &&
           MGRValidateMatchedQ(args, plan, scan) &&
           MGRValidateMatchedFBacksolve(args, plan, scan));
}
/* GCOVR_EXCL_BR_STOP */

#endif /* HYPRE_CHECK_MIN_VERSION(21900, 0) */

/*-----------------------------------------------------------------------------
 * hypredrv_MGRCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRCreate(MGR_args *args, HYPRE_Solver *precon_ptr, const Stats *stats,
                   int next_ls_id)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)args;
   (void)stats;
   (void)next_ls_id;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("MGR requires hypre >= 2.19.0");
   *precon_ptr = NULL;
   return;
#else
   HYPRE_Solver   precon = NULL;
   MGRCreatePlan  plan   = {0};
   MGRFeatureScan scan   = {0, 0, 0, 0, 0, -1, 0};

   /* GCOVR_EXCL_BR_START */
   /* Sanity checks */
   if (!args->dofmap)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      return;
   }

   if (args->num_levels < 1 || args->num_levels > MAX_MGR_LEVELS)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR num_levels must be between 1 and %d (got %d)",
                           MAX_MGR_LEVELS, (int)args->num_levels);
      return;
   }

   if (!MGRPlanCoarsening(args, &plan, stats, next_ls_id) ||
       !MGRPlanPointMarkers(args, &plan, stats, next_ls_id))
   {
      goto cleanup;
   }

   if (!MGRValidateConfiguration(args, &plan, &scan))
   {
      goto cleanup;
   }
   /* GCOVR_EXCL_BR_STOP */

   /* Config preconditioner */
   HYPRE_MGRCreate(&precon);
   MGRApplyBaseSettings(precon, args, &plan, scan.any_polynomial_matched_q, stats,
                        next_ls_id);
   /* GCOVR_EXCL_BR_START */
   MGRApplyLevelSettings(precon, args, &plan, stats, next_ls_id);

   if (!MGRInstallUserCoarseMatrices(precon, args, &plan) ||
       !MGRConfigFRelaxSolvers(args, precon, &plan, stats, next_ls_id) ||
       !MGRConfigGRelaxSolvers(args, precon, &plan, stats, next_ls_id) ||
       !MGRConfigCoarsestSolver(args, precon, stats, next_ls_id))
   {
      goto cleanup;
   }
   /* GCOVR_EXCL_BR_STOP */

#if HYPRE_CHECK_MIN_VERSION(23100, 11)
   HYPRE_MGRSetNonGalerkinMaxElmts(precon, args->nonglk_max_elmts);
#endif

   /* Set output pointer */
   *precon_ptr = precon;
   precon      = NULL;
   if (plan.dofmap_data_owned)
   {
      /* hypre uses the point-marker array during MGRSetup, so keep the owned copy
       * alive until PreconDestroyMGRSolver() destroys the MGR object. */
      free(args->point_marker_data);
      args->point_marker_data = plan.dofmap_data_owned;
      plan.dofmap_data_owned  = NULL;
   }

cleanup:
   MGRCreatePlanDispose(&plan);
   if (precon)
   {
      HYPRE_MGRDestroy(precon);
   }
   /* Soft hypre convergence leftovers should not poison later calls. */
   hypredrv_HypreConsumeErrors();
#endif /* !HYPRE_CHECK_MIN_VERSION(21900, 0) */
}
