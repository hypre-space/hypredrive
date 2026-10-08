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

/* GCOVR_EXCL_BR_START */

void
hypredrv_MGRComponentReuseSetDefaultArgs(MGRComponentReuse_args *reuse)
{
   memset(reuse, 0, sizeof(*reuse));
   hypredrv_PreconReuseSetDefaultArgs(&reuse->args);
}

void
hypredrv_MGRComponentReuseDestroyArgs(MGRComponentReuse_args *reuse)
{
   hypredrv_PreconReuseDestroyArgs(&reuse->args);
   reuse->present                    = 0;
   reuse->warned_runtime_unsupported = 0;
   reuse->warned_policy_unsupported  = 0;
   reuse->warned_type_unsupported    = 0;
}

static void
MGRComponentReuseLogWarning(int *warned_flag, const Stats *stats, int next_ls_id,
                            const char *label, const char *detail)
{
   if (*warned_flag)
   {
      return;
   }

   *warned_flag = 1;
   HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "%s %s", label, detail);
}

/* GCOVR_EXCL_BR_STOP */

static void
MGRFillLevelReuseLabel(char *buf, size_t buf_size, int level, const char *name)
{
   snprintf(buf, buf_size, "level[%d].%s.reuse", level, name);
}

static HYPRE_Int
MGRResolveCoarseSolverType(const MGRcls_args *args)
{
   if (!args)
   {
      return -1;
   }

   return (args->type < 0) ? 0 : args->type;
}

const char *
hypredrv_MGRCoarseSolverTypeName(const MGRcls_args *args)
{
   HYPRE_Int type = MGRResolveCoarseSolverType(args);

   if (!args)
   {
      return "unknown";
   }

   if (args->use_krylov && args->krylov)
   {
      return "nested-krylov";
   }

   switch (type)
   {
      case 0:
         return "amg";
      case 29:
         return "spdirect";
      case 32:
         return "ilu";
      case 33:
         return "fsai";
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      case MGR_SOLVER_TYPE_SCHWARZ:
         return "schwarz";
#endif
      default:
         return "unknown";
   }
}

static int
MGRFRelaxUsesManagedHandle(const MGRfrlx_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (args->use_krylov && args->krylov)
   {
      return 1;
   }

   return args->type == 2 || args->type == 29 || args->type == 32 || args->type == 33
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
          || args->type == MGR_SOLVER_TYPE_SCHWARZ
#endif
      ;
}

static int
MGRFRelaxConfiguredReuseSupported(const MGRfrlx_args *args)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   return MGRFRelaxUsesManagedHandle(args);
#else
   (void)args;
   return 0;
#endif
}

static int
MGRGRelaxUsesManagedHandle(const MGRgrlx_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (args->use_krylov && args->krylov)
   {
      return 1;
   }

   return args->type == 20 || args->type == 16 || args->type == 29 || args->type == 33
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
          || args->type == MGR_SOLVER_TYPE_SCHWARZ
#endif
      ;
}

int
hypredrv_MGRGRelaxUsesUserSmoother(const MGRgrlx_args *args)
{
   return args && (args->use_krylov || MGRGRelaxUsesManagedHandle(args));
}

static int
MGRGRelaxConfiguredReuseSupported(const MGRgrlx_args *args)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   return MGRGRelaxUsesManagedHandle(args);
#else
   (void)args;
   return 0;
#endif
}

static int
MGRCoarseUsesManagedHandle(const MGRcls_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (args->use_krylov && args->krylov)
   {
      return 1;
   }

   HYPRE_Int type = MGRResolveCoarseSolverType(args);
   return type == 0 || type == 29 || type == 32
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
          || type == 33
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
          || type == MGR_SOLVER_TYPE_SCHWARZ
#endif
      ;
}

static int
MGRCoarseConfiguredReuseSupported(const MGRcls_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (args->use_krylov && args->krylov)
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      return 1;
#else
      return 0;
#endif
   }

   return MGRCoarseUsesManagedHandle(args);
}

/*-----------------------------------------------------------------------------
 * Uniform iteration over the reuse-capable MGR components: F- and
 * G-relaxation of each active level (in level order), then the coarsest
 * solver. The visiting order matters: hypredrv_MGRComponentReuseSetupMode
 * returns on the first qualifying component, which also defines which
 * warnings are emitted.
 *-----------------------------------------------------------------------------*/

typedef enum
{
   MGR_COMPONENT_FRELAX,
   MGR_COMPONENT_GRELAX,
   MGR_COMPONENT_COARSE,
} MGRComponentKind;

typedef struct
{
   MGRComponentKind kind;
   int              active_lvl; /* -1 for the coarsest component */
   int              orig_lvl;   /* -1 for the coarsest component */
} MGRComponentRef;

enum
{
   MGR_MAX_COMPONENT_REFS = (2 * (MAX_MGR_LEVELS - 1)) + 1,
};

static int
MGRListComponents(const MGR_args *args, MGRComponentRef refs[MGR_MAX_COMPONENT_REFS])
{
   int count = 0;

   for (int active_lvl = 0; active_lvl < args->num_active_levels; active_lvl++)
   {
      int orig_lvl = args->active_level_map[active_lvl];

      refs[count++] = (MGRComponentRef){MGR_COMPONENT_FRELAX, active_lvl, orig_lvl};
      refs[count++] = (MGRComponentRef){MGR_COMPONENT_GRELAX, active_lvl, orig_lvl};
   }
   refs[count++] = (MGRComponentRef){MGR_COMPONENT_COARSE, -1, -1};

   return count;
}

static const char *
MGRComponentName(MGRComponentKind kind)
{
   switch (kind)
   {
      case MGR_COMPONENT_FRELAX:
         return "f_relaxation";
      case MGR_COMPONENT_GRELAX:
         return "g_relaxation";
      default:
         return "coarsest_level";
   }
}

static const char *
MGRComponentNoHandleWarning(MGRComponentKind kind)
{
   switch (kind)
   {
      case MGR_COMPONENT_FRELAX:
         return "is ignored because this F-relaxation type does not expose a reusable "
                "hypredrive-managed handle";
      case MGR_COMPONENT_GRELAX:
         return "is ignored because this global smoother does not expose a reusable "
                "hypredrive-managed handle";
      default:
         return "is ignored because this coarsest solver does not expose a reusable "
                "hypredrive-managed handle";
   }
}

/* Label shown in reuse warnings, e.g. "level[0].f_relaxation.reuse". */
static void
MGRComponentReuseLabel(const MGRComponentRef *ref, char *buf, size_t buf_size)
{
   if (ref->kind == MGR_COMPONENT_COARSE)
   {
      snprintf(buf, buf_size, "coarsest_level.reuse");
   }
   else
   {
      MGRFillLevelReuseLabel(buf, buf_size, ref->orig_lvl, MGRComponentName(ref->kind));
   }
}

static const MGRComponentReuse_args *
MGRComponentReuseArgsConst(const MGR_args *args, const MGRComponentRef *ref)
{
   switch (ref->kind)
   {
      case MGR_COMPONENT_FRELAX:
         return &args->level[ref->orig_lvl].f_relaxation.reuse;
      case MGR_COMPONENT_GRELAX:
         return &args->level[ref->orig_lvl].g_relaxation.reuse;
      default:
         return &args->coarsest_level.reuse;
   }
}

static MGRComponentReuse_args *
MGRComponentReuseArgs(MGR_args *args, const MGRComponentRef *ref)
{
   return (MGRComponentReuse_args *)MGRComponentReuseArgsConst(args, ref);
}

static int
MGRComponentUsesManagedHandle(const MGR_args *args, const MGRComponentRef *ref)
{
   switch (ref->kind)
   {
      case MGR_COMPONENT_FRELAX:
         return MGRFRelaxUsesManagedHandle(&args->level[ref->orig_lvl].f_relaxation);
      case MGR_COMPONENT_GRELAX:
         return MGRGRelaxUsesManagedHandle(&args->level[ref->orig_lvl].g_relaxation);
      default:
         return MGRCoarseUsesManagedHandle(&args->coarsest_level);
   }
}

static int
MGRComponentConfiguredReuseSupported(const MGR_args *args, const MGRComponentRef *ref)
{
   switch (ref->kind)
   {
      case MGR_COMPONENT_FRELAX:
         return MGRFRelaxConfiguredReuseSupported(
            &args->level[ref->orig_lvl].f_relaxation);
      case MGR_COMPONENT_GRELAX:
         return MGRGRelaxConfiguredReuseSupported(
            &args->level[ref->orig_lvl].g_relaxation);
      default:
         return MGRCoarseConfiguredReuseSupported(&args->coarsest_level);
   }
}

static int
MGRHasNestedFRelaxWrapper(const MGR_args *args)
{
   if (!args)
   {
      return 0;
   }

   for (int active_lvl = 0; active_lvl < args->num_active_levels; active_lvl++)
   {
      int orig_lvl = args->active_level_map[active_lvl];
      if (args->level[orig_lvl].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
      {
         return 1;
      }
   }

   return 0;
}

static int
MGRManagedRefreshShapeSupported(const MGR_args *args)
{
   if (!args)
   {
      return 0;
   }

   if (MGRHasNestedFRelaxWrapper(args))
   {
      return 0;
   }

   MGRComponentRef refs[MGR_MAX_COMPONENT_REFS];
   int             num_refs = MGRListComponents(args, refs);

   for (int n = 0; n < num_refs; n++)
   {
      if (MGRComponentUsesManagedHandle(args, &refs[n]) &&
          !MGRComponentConfiguredReuseSupported(args, &refs[n]))
      {
         return 0;
      }
   }

   return 1;
}

static int
MGRComponentReuseShouldKeep(const MGRComponentReuse_args *reuse,
                            const IntArray *timestep_starts, const Stats *stats,
                            int next_ls_id)
{
   if (!reuse || !reuse->present)
   {
      return 0;
   }

   if (reuse->args.policy != PRECON_REUSE_POLICY_STATIC)
   {
      return 0;
   }

   return !hypredrv_PreconReuseShouldRebuildStatic(&reuse->args, timestep_starts, stats,
                                                   next_ls_id);
}

/* Returns 1 only when a component solver can safely reuse its prior setup.
 * Returns 0 for reset requests, NULL solvers, and fresh solvers that must still
 * run their first setup. Callers pass hypre-compatible solver objects:
 * BoomerAMG, ILU, FSAI, direct, Schwarz/NestedKrylov wrappers, or MGR wrappers. */
static int
MGRSetComponentSetupReuse(HYPRE_Solver solver, int set_reuse)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 50)
   if (!solver)
   {
      return 0;
   }

   hypre_Solver *base = (hypre_Solver *)solver;
   if (set_reuse)
   {
      if (!hypre_SolverSetupIsDone(base))
      {
         return 0;
      }
      hypre_SolverSetSetupReuse(base);
      return (hypre_SolverSetupReuseRequested(base) != 0);
   }

   /* Reset is intentional for freshly rebuilt component solvers too: it also
    * normalizes any cached handle that was refreshed against a new matrix. */
   hypre_SolverResetIsSetup(base);
   return 0;
#else
   (void)solver;
   (void)set_reuse;
   return 0;
#endif
}

static HYPRE_Solver
MGRFRelaxSetupSolver(MGR_args *args, int orig_lvl)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];

   if (level_args->f_relaxation.use_krylov && level_args->f_relaxation.krylov &&
       level_args->f_relaxation.krylov->base_solver)
   {
      return (HYPRE_Solver)level_args->f_relaxation.krylov;
   }

   return args->frelax[orig_lvl];
}

static HYPRE_Solver
MGRGRelaxSetupSolver(MGR_args *args, int orig_lvl)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];

   if (level_args->g_relaxation.use_krylov && level_args->g_relaxation.krylov &&
       level_args->g_relaxation.krylov->base_solver)
   {
      return (HYPRE_Solver)level_args->g_relaxation.krylov;
   }

   return args->grelax[orig_lvl];
}

static HYPRE_Solver
MGRCoarseSetupSolver(MGR_args *args)
{
   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov &&
       args->coarsest_level.krylov->base_solver)
   {
      return (HYPRE_Solver)args->coarsest_level.krylov;
   }

   return args->csolver;
}

/* Solver handle whose hypre setup-reuse flag controls the component. */
static HYPRE_Solver
MGRComponentSetupSolver(MGR_args *args, const MGRComponentRef *ref)
{
   switch (ref->kind)
   {
      case MGR_COMPONENT_FRELAX:
         return MGRFRelaxSetupSolver(args, ref->orig_lvl);
      case MGR_COMPONENT_GRELAX:
         return MGRGRelaxSetupSolver(args, ref->orig_lvl);
      default:
         return MGRCoarseSetupSolver(args);
   }
}

static void
MGRDestroyDetachedFSolver(const MGRfrlx_args *f_relaxation, HYPRE_Solver *solver_ptr)
{
   if (!f_relaxation || !solver_ptr || !*solver_ptr)
   {
      return;
   }

   if (f_relaxation->type == 2)
   {
      if (f_relaxation->symmetric_diagonal_scaling)
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
         hypredrv_MGRFRelaxEquilWrapperDestroy(*solver_ptr);
#else
         HYPRE_BoomerAMGDestroy(*solver_ptr);
#endif
      }
      else
      {
         HYPRE_BoomerAMGDestroy(*solver_ptr);
      }
   }
#if defined(HYPRE_USING_DSUPERLU)
   else if (f_relaxation->type == 29)
   {
      HYPRE_MGRDirectSolverDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (f_relaxation->type == 32)
   {
      HYPRE_ILUDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (f_relaxation->type == 33)
   {
      HYPRE_FSAIDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (f_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_MGRSchwarzWrapperDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   else if (f_relaxation->type == MGR_FRLX_TYPE_NESTED_MGR)
   {
      /* Only destroy live wrappers: a prior reclaim (or failed create cleanup)
       * may already have freed the object while leaving a stale cache slot. */
      if (hypredrv_MGRNestedFRelaxWrapperIsLive(*solver_ptr))
      {
         hypredrv_MGRNestedFRelaxWrapperFree(solver_ptr);
      }
   }
#endif

   *solver_ptr = NULL;
}

static void
MGRDestroyDetachedGSolver(const MGRgrlx_args *g_relaxation, HYPRE_Solver *solver_ptr)
{
   if (!g_relaxation || !solver_ptr || !*solver_ptr)
   {
      return;
   }

   if (g_relaxation->type == 20)
   {
      HYPRE_BoomerAMGDestroy(*solver_ptr);
   }
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (g_relaxation->type == 16)
   {
      HYPRE_ILUDestroy(*solver_ptr);
   }
#endif
#ifdef HYPRE_USING_DSUPERLU
   /* GCOVR_EXCL_START */
   else if (g_relaxation->type == 29)
   {
      HYPRE_MGRDirectSolverDestroy(*solver_ptr);
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (g_relaxation->type == 33)
   {
      HYPRE_FSAIDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (g_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_MGRSchwarzWrapperDestroy(*solver_ptr);
   }
#endif

   *solver_ptr = NULL;
}

int
hypredrv_MGRProjectNestedRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                              const int *selected_dofs, size_t num_selected_dofs)
{
   if (!krylov->has_precon || krylov->precon_method != PRECON_BOOMERAMG)
   {
      return 1;
   }

   hypredrv_AMGSetProjectedRBMs(&krylov->precon.amg, mgr_args->vec_nn, mgr_args->dofmap,
                                mgr_args->rbm_input_generation, selected_dofs,
                                num_selected_dofs);
   return !hypredrv_ErrorCodeActive();
}

int
hypredrv_MGRProjectNestedRemainingRBMs(NestedKrylov_args *krylov, MGR_args *mgr_args,
                                       int num_eliminated_levels)
{
   if (!krylov->has_precon || krylov->precon_method != PRECON_BOOMERAMG ||
       num_eliminated_levels <= 0)
   {
      return 1;
   }

   const IntArray *dofmap      = mgr_args->dofmap;
   const int      *source      = dofmap->unique_data ? dofmap->unique_data : dofmap->data;
   size_t          source_size = dofmap->unique_data ? dofmap->unique_size : dofmap->size;
   int *remaining = source_size ? (int *)malloc(source_size * sizeof(*remaining)) : NULL;
   size_t num_remaining = 0;
   if (source_size && !remaining)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
   }
   else
   {
      for (size_t i = 0; i < source_size; i++)
      {
         int eliminated = 0;
         for (int active_lvl = 0; active_lvl < num_eliminated_levels && !eliminated;
              active_lvl++)
         {
            const StackIntArray *f_dofs =
               &mgr_args->level[mgr_args->active_level_map[active_lvl]].f_dofs;
            for (size_t j = 0; j < f_dofs->size; j++)
            {
               eliminated = (source[i] == f_dofs->data[j]);
               if (eliminated)
               {
                  break;
               }
            }
         }
         if (!eliminated)
         {
            remaining[num_remaining++] = source[i];
         }
      }
   }

   int ok = hypredrv_MGRProjectNestedRBMs(krylov, mgr_args, remaining, num_remaining);
   free(remaining);
   return ok;
}

static int
MGRRebuildNestedKrylovSolver(NestedKrylov_args *krylov, MGR_args *mgr_args,
                             const StackIntArray *f_dofs, int num_eliminated_levels)
{
   if (!krylov || !mgr_args)
   {
      return 0;
   }

   AMG_args *amg = (krylov->has_precon && krylov->precon_method == PRECON_BOOMERAMG)
                      ? &krylov->precon.amg
                      : NULL;
   int       keep_projected_rbms = amg && amg->interp_vec_variant == 1 &&
                             amg->rbm_source_generation == mgr_args->rbm_input_generation;
   HYPRE_Int       cached_num_rbms    = 0;
   HYPRE_ParVector cached_rbms[3]     = {NULL, NULL, NULL};
   size_t          cached_labels_size = 0;
   uint64_t        cached_labels_hash = 0;
   if (keep_projected_rbms)
   {
      cached_num_rbms    = amg->num_rbms;
      cached_labels_size = amg->rbm_source_labels_size;
      cached_labels_hash = amg->rbm_source_labels_hash;
      for (int i = 0; i < 3; i++)
      {
         cached_rbms[i] = amg->rbms[i];
         amg->rbms[i]   = NULL;
      }
      amg->num_rbms = 0;
   }
   hypredrv_NestedKrylovDestroy(krylov);
   if (keep_projected_rbms)
   {
      amg->num_rbms               = cached_num_rbms;
      amg->interp_vec_variant     = 1;
      amg->rbm_source_generation  = mgr_args->rbm_input_generation;
      amg->rbm_source_labels_size = cached_labels_size;
      amg->rbm_source_labels_hash = cached_labels_hash;
      for (int i = 0; i < 3; i++)
      {
         amg->rbms[i] = cached_rbms[i];
      }
   }
   if ((f_dofs &&
        !hypredrv_MGRProjectNestedRBMs(krylov, mgr_args, f_dofs->data, f_dofs->size)) ||
       (!f_dofs &&
        !hypredrv_MGRProjectNestedRemainingRBMs(krylov, mgr_args, num_eliminated_levels)))
   {
      return 0;
   }
   hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, krylov, mgr_args->dofmap, mgr_args->vec_nn,
                               &krylov->base_solver);
   return !hypredrv_ErrorCodeActive();
}

/*-----------------------------------------------------------------------------
 * Per-type create/install/destroy helpers for the hypredrive-managed MGR
 * component solver handles (F-relaxation, G-relaxation, coarsest level).
 * Shared between first-time configuration (hypredrv_MGRCreate) and component
 * refresh between setups (hypredrv_MGRRefreshComponentsForSetup). Creators
 * return NULL for types that do not use a managed handle in this build;
 * invalid configurations additionally set the error state.
 *-----------------------------------------------------------------------------*/

HYPRE_Solver
hypredrv_MGRFRelaxSolverCreateByType(MGR_args *args, MGRfrlx_args *f_relaxation,
                                     const StackIntArray *f_dofs, int active_lvl)
{
   HYPRE_Solver solver = NULL;

#if !HYPRE_CHECK_MIN_VERSION(30100, 55)
   (void)active_lvl;
#endif

   if (f_relaxation->type == 2)
   {
      hypredrv_AMGSetProjectedRBMs(&f_relaxation->amg, args->vec_nn, args->dofmap,
                                   args->rbm_input_generation, f_dofs->data,
                                   f_dofs->size);
      if (hypredrv_ErrorCodeActive())
      {
         return NULL;
      }
      hypredrv_AMGCreate(&f_relaxation->amg, &solver);
      if (solver && f_relaxation->symmetric_diagonal_scaling)
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
         solver = hypredrv_MGRFRelaxEquilWrapperCreate(solver);
#else
         HYPRE_BoomerAMGDestroy(solver);
         solver = NULL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR symmetric_diagonal_scaling requires hypre >= 3.1.0");
#endif
      }
   }
#if defined(HYPRE_USING_DSUPERLU)
   /* GCOVR_EXCL_START */
   else if (f_relaxation->type == 29)
   {
      HYPRE_MGRDirectSolverCreate(&solver);
      if (!solver)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'spdirect' unavailable: direct solver creation failed");
      }
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
   else if (f_relaxation->type == 32)
   {
      hypredrv_ILUCreate(&f_relaxation->ilu, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (f_relaxation->type == 33)
   {
      hypredrv_FSAICreate(&f_relaxation->fsai, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (f_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      if (active_lvl != 0)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'schwarz' is only supported at MGR level 0");
         return NULL;
      }
      solver = hypredrv_MGRSchwarzWrapperCreate(&f_relaxation->schwarz);
   }
#endif

   return solver;
}

HYPRE_Solver
hypredrv_MGRGRelaxSolverCreateByType(MGRgrlx_args *g_relaxation)
{
   HYPRE_Solver solver = NULL;

   if (g_relaxation->type == 20)
   {
      hypredrv_AMGCreate(&g_relaxation->amg, &solver);
   }
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (g_relaxation->type == 16)
   {
      hypredrv_ILUCreate(&g_relaxation->ilu, &solver);
   }
#endif
#ifdef HYPRE_USING_DSUPERLU
   /* GCOVR_EXCL_START */
   else if (g_relaxation->type == 29)
   {
      HYPRE_MGRDirectSolverCreate(&solver);
      if (!solver)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR G-relaxation 'spdirect' unavailable: direct solver creation failed");
      }
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (g_relaxation->type == 33)
   {
      hypredrv_FSAICreate(&g_relaxation->fsai, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (g_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      solver = hypredrv_MGRSchwarzWrapperCreate(&g_relaxation->schwarz);
   }
#endif

   return solver;
}

HYPRE_Solver
hypredrv_MGRCoarseSolverCreateByType(MGRcls_args *coarsest_level, HYPRE_Int type)
{
   HYPRE_Solver solver = NULL;

   if (type == 0)
   {
      hypredrv_AMGCreate(&coarsest_level->amg, &solver);
   }
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (type == 32)
   {
      hypredrv_ILUCreate(&coarsest_level->ilu, &solver);
   }
#endif
#if defined(HYPRE_USING_DSUPERLU)
   /* GCOVR_EXCL_START */
   else if (type == 29)
   {
      HYPRE_MGRDirectSolverCreate(&solver);
      if (!solver)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR coarsest_level 'spdirect' unavailable: direct solver creation failed");
      }
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (type == 33)
   {
      hypredrv_FSAICreate(&coarsest_level->fsai, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_SchwarzCreate(&coarsest_level->schwarz, &solver);
   }
#endif

   return solver;
}

void
hypredrv_MGRCoarseSolverInstall(HYPRE_Solver mgr_solver, HYPRE_Int type,
                                HYPRE_Solver coarse_solver)
{
   if (type == 0)
   {
      HYPRE_MGRSetCoarseSolver(mgr_solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                               coarse_solver);
   }
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (type == 32)
   {
      HYPRE_MGRSetCoarseSolver(mgr_solver, HYPRE_ILUSolve, HYPRE_ILUSetup, coarse_solver);
   }
#endif
#if defined(HYPRE_USING_DSUPERLU)
   /* GCOVR_EXCL_START */
   else if (type == 29)
   {
      HYPRE_MGRSetCoarseSolver(mgr_solver, HYPRE_MGRDirectSolverSolve,
                               HYPRE_MGRDirectSolverSetup, coarse_solver);
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (type == 33)
   {
      HYPRE_MGRSetCoarseSolver(mgr_solver, HYPRE_FSAISolve, HYPRE_FSAISetup,
                               coarse_solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      HYPRE_MGRSetCoarseSolver(mgr_solver, HYPRE_SchwarzSolve, HYPRE_SchwarzSetup,
                               coarse_solver);
   }
#endif
}

/* hypre never destroys user-installed coarse solvers (hypre_MGRSetCoarseSolver
 * clears use_default_cgrid_solver), so every type set by hypredrv_MGRCoarseSolverInstall
 * must be reclaimed here. */
void
hypredrv_MGRCoarseSolverDestroyByType(HYPRE_Int type, HYPRE_Solver *solver_ptr)
{
   if (!solver_ptr || !*solver_ptr)
   {
      return;
   }

   if (type == 0)
   {
      HYPRE_BoomerAMGDestroy(*solver_ptr);
   }
#if defined(HYPRE_USING_DSUPERLU)
   else if (type == 29)
   {
      HYPRE_MGRDirectSolverDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   else if (type == 32)
   {
      HYPRE_ILUDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (type == 33)
   {
      HYPRE_FSAIDestroy(*solver_ptr);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      HYPRE_SchwarzDestroy(*solver_ptr);
   }
#endif

   *solver_ptr = NULL;
}

static void
MGRRefreshFRelaxAtLevel(MGR_args *args, HYPRE_Solver mgr_solver, int active_lvl,
                        int orig_lvl)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];

   if (level_args->f_relaxation.use_krylov && level_args->f_relaxation.krylov)
   {
      if (!MGRRebuildNestedKrylovSolver(level_args->f_relaxation.krylov, args,
                                        &level_args->f_dofs, -1))
      {
         return;
      }

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
      hypredrv_MGRSetFSolverAtLevel(
         mgr_solver, (HYPRE_Solver)level_args->f_relaxation.krylov, active_lvl,
         level_args->f_relaxation.type, hypredrv_MGRBaseParSolverSolve,
         hypredrv_MGRBaseParSolverSetup);
      MGRSetComponentSetupReuse((HYPRE_Solver)level_args->f_relaxation.krylov, 0);
#endif
      return;
   }

   HYPRE_Solver old_fsolver = args->frelax[orig_lvl];
   HYPRE_Solver fsolver     = hypredrv_MGRFRelaxSolverCreateByType(
      args, &level_args->f_relaxation, &level_args->f_dofs, active_lvl);

   if (hypredrv_ErrorCodeActive() || !fsolver)
   {
      return;
   }

   hypredrv_MGRFRelaxInstall(mgr_solver, &level_args->f_relaxation, fsolver, active_lvl);
   MGRSetComponentSetupReuse(fsolver, 0);
   MGRDestroyDetachedFSolver(&level_args->f_relaxation, &old_fsolver);
   args->frelax[orig_lvl] = fsolver;
}

static void
MGRRefreshGRelaxAtLevel(MGR_args *args, HYPRE_Solver mgr_solver, int active_lvl,
                        int orig_lvl)
{
   MGRlvl_args *level_args = &args->level[orig_lvl];

   if (level_args->g_relaxation.use_krylov && level_args->g_relaxation.krylov)
   {
      if (!MGRRebuildNestedKrylovSolver(level_args->g_relaxation.krylov, args, NULL,
                                        active_lvl))
      {
         return;
      }

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
      HYPRE_MGRSetGlobalSmootherAtLevel(
         mgr_solver, (HYPRE_Solver)level_args->g_relaxation.krylov, active_lvl);
      MGRSetComponentSetupReuse((HYPRE_Solver)level_args->g_relaxation.krylov, 0);
#endif
      return;
   }

   HYPRE_Solver old_smoother = args->grelax[orig_lvl];
   HYPRE_Solver smoother =
      hypredrv_MGRGRelaxSolverCreateByType(&level_args->g_relaxation);

   if (hypredrv_ErrorCodeActive() || !smoother)
   {
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   HYPRE_MGRSetGlobalSmootherAtLevel(mgr_solver, smoother, active_lvl);
   MGRSetComponentSetupReuse(smoother, 0);
#endif
   MGRDestroyDetachedGSolver(&level_args->g_relaxation, &old_smoother);
   args->grelax[orig_lvl] = smoother;
}

static void
MGRRefreshCoarseSolver(MGR_args *args, HYPRE_Solver mgr_solver)
{
   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      if (!MGRRebuildNestedKrylovSolver(args->coarsest_level.krylov, args, NULL,
                                        args->num_active_levels))
      {
         return;
      }

#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(mgr_solver, hypredrv_MGRBaseParSolverSolve,
                               hypredrv_MGRBaseParSolverSetup,
                               (HYPRE_Solver)args->coarsest_level.krylov);
      MGRSetComponentSetupReuse((HYPRE_Solver)args->coarsest_level.krylov, 0);
#endif
      return;
   }

   HYPRE_Solver old_coarse_solver = args->csolver;
   HYPRE_Int    old_type          = args->csolver_type;
   HYPRE_Int    type              = MGRResolveCoarseSolverType(&args->coarsest_level);
   HYPRE_Solver coarse_solver =
      hypredrv_MGRCoarseSolverCreateByType(&args->coarsest_level, type);

   if (hypredrv_ErrorCodeActive() || !coarse_solver)
   {
      return;
   }

   hypredrv_MGRCoarseSolverInstall(mgr_solver, type, coarse_solver);
   MGRSetComponentSetupReuse(coarse_solver, 0);
   args->csolver_type = type;
   hypredrv_MGRCoarseSolverDestroyByType(old_type, &old_coarse_solver);
   args->csolver = coarse_solver;
}

static void
MGRComponentRefresh(MGR_args *args, HYPRE_Solver precon, const MGRComponentRef *ref)
{
   switch (ref->kind)
   {
      case MGR_COMPONENT_FRELAX:
         MGRRefreshFRelaxAtLevel(args, precon, ref->active_lvl, ref->orig_lvl);
         break;
      case MGR_COMPONENT_GRELAX:
         MGRRefreshGRelaxAtLevel(args, precon, ref->active_lvl, ref->orig_lvl);
         break;
      default:
         MGRRefreshCoarseSolver(args, precon);
         break;
   }
}

int
hypredrv_MGRComponentReuseShouldKeepOuter(const MGR_args *args,
                                          const IntArray *timestep_starts,
                                          const Stats *stats, int next_ls_id)
{
   if (!args || !MGRManagedRefreshShapeSupported(args))
   {
      return 0;
   }
#if !HYPRE_CHECK_MIN_VERSION(30100, 50)
   return 0;
#endif

   MGRComponentRef refs[MGR_MAX_COMPONENT_REFS];
   int             num_refs = MGRListComponents(args, refs);

   for (int n = 0; n < num_refs; n++)
   {
      if (MGRComponentUsesManagedHandle(args, &refs[n]) &&
          MGRComponentReuseShouldKeep(MGRComponentReuseArgsConst(args, &refs[n]),
                                      timestep_starts, stats, next_ls_id))
      {
         return 1;
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Decide how MGR component reuse should be configured for the next solve.
 *
 * This routine inspects all managed reuse handles associated with the MGR
 * hierarchy (fine- and coarse-grid relaxations on each active level, plus the
 * coarsest-level solver) and decides whether any reusable components are
 * currently present. If no reuse handles are present, the function returns 0
 * immediately, indicating that the driver should perform a fresh setup.
 *
 * Otherwise, it prepares the internal "setup mode" state in @a args based on
 * the current solver statistics @a stats and the next linear-solve identifier
 * @a next_ls_id, so that subsequent setup/solve calls can either reuse or
 * rebuild components as appropriate.
 *
 * Return value:
 *    0 : no reusable components are currently present, or @a args is NULL;
 *        the caller should perform a full (non-reuse) setup.
 *    1 : at least one managed reuse handle is present and @a args has been
 *        updated to reflect the selected reuse strategy for the next solve.
 *
 * Side effects:
 *    This routine may modify internal reuse-related fields in @a args,
 *    e.g., per-level or coarsest-level "setup mode" indicators that control
 *    whether existing components are reused or rebuilt on subsequent calls
 *    to MGR setup/solve drivers.
 *--------------------------------------------------------------------------*/

int
hypredrv_MGRComponentReuseSetupMode(MGR_args *args, const Stats *stats, int next_ls_id)
{
   /* Defensive check: if the MGR argument block is not available, there is
    * no meaningful reuse decision to make, so fall back to "no reuse". */
   if (!args)
   {
      return 0;
   }

   MGRComponentRef refs[MGR_MAX_COMPONENT_REFS];
   int             num_refs = MGRListComponents(args, refs);
   char            label[96];

   /* First, detect whether *any* managed reuse handle is currently present
    * on the hierarchy. If none are present, we can immediately signal that
    * a fresh setup is required without inspecting solver statistics. */
   int any_present = 0;
   for (int n = 0; n < num_refs; n++)
   {
      any_present |= MGRComponentReuseArgsConst(args, &refs[n])->present;
   }

   /* If no reuse handles are active anywhere in the hierarchy, instruct the
    * caller to perform a full setup and return without touching reuse state. */
   if (!any_present)
   {
      return 0;
   }

#if !HYPRE_CHECK_MIN_VERSION(30100, 50)
   for (int n = 0; n < num_refs; n++)
   {
      MGRComponentReuse_args *reuse = MGRComponentReuseArgs(args, &refs[n]);

      if (!reuse->present)
      {
         continue;
      }
      MGRComponentReuseLabel(&refs[n], label, sizeof(label));
      MGRComponentReuseLogWarning(&reuse->warned_runtime_unsupported, stats, next_ls_id,
                                  label,
                                  "is ignored because the current hypre version does "
                                  "not support managed MGR component reuse");
   }
   return 0;
#endif

   if (!MGRManagedRefreshShapeSupported(args))
   {
      for (int n = 0; n < num_refs; n++)
      {
         MGRComponentReuse_args *reuse = MGRComponentReuseArgs(args, &refs[n]);

         if (!reuse->present)
         {
            continue;
         }
         MGRComponentReuseLabel(&refs[n], label, sizeof(label));
         MGRComponentReuseLogWarning(&reuse->warned_type_unsupported, stats, next_ls_id,
                                     label,
                                     "is ignored because the current MGR configuration "
                                     "includes solver handles that cannot be safely "
                                     "refreshed yet");
      }
      return 0;
   }

   for (int n = 0; n < num_refs; n++)
   {
      MGRComponentReuse_args *reuse = MGRComponentReuseArgs(args, &refs[n]);

      if (!reuse->present)
      {
         continue;
      }

      MGRComponentReuseLabel(&refs[n], label, sizeof(label));
      if (reuse->args.policy != PRECON_REUSE_POLICY_STATIC)
      {
         MGRComponentReuseLogWarning(&reuse->warned_policy_unsupported, stats, next_ls_id,
                                     label,
                                     "accepts adaptive reuse syntax, but only "
                                     "static/scheduled component reuse is supported "
                                     "today");
      }
      else if (MGRComponentUsesManagedHandle(args, &refs[n]))
      {
         return 1;
      }
      else
      {
         MGRComponentReuseLogWarning(&reuse->warned_type_unsupported, stats, next_ls_id,
                                     label, MGRComponentNoHandleWarning(refs[n].kind));
      }
   }

   return 0;
}

void
hypredrv_MGRRefreshComponentsForSetup(MGR_args *args, HYPRE_Solver precon,
                                      const IntArray *timestep_starts, const Stats *stats,
                                      int next_ls_id)
{
   if (!args || !precon || !MGRManagedRefreshShapeSupported(args))
   {
      return;
   }
#if !HYPRE_CHECK_MIN_VERSION(30100, 50)
   (void)timestep_starts;
   (void)stats;
   (void)next_ls_id;
   return;
#endif

   MGRComponentRef refs[MGR_MAX_COMPONENT_REFS];
   int             num_refs = MGRListComponents(args, refs);

   for (int n = 0; n < num_refs; n++)
   {
      const MGRComponentRef *ref = &refs[n];

      if (!MGRComponentUsesManagedHandle(args, ref))
      {
         continue;
      }

      int reuse_accepted = 0;
      int keep_requested = MGRComponentReuseShouldKeep(
         MGRComponentReuseArgsConst(args, ref), timestep_starts, stats, next_ls_id);
      if (keep_requested)
      {
         reuse_accepted =
            MGRSetComponentSetupReuse(MGRComponentSetupSolver(args, ref), 1);
      }
      if (!reuse_accepted)
      {
         MGRComponentRefresh(args, precon, ref);
         if (hypredrv_ErrorCodeActive())
         {
            return;
         }
      }
      if (ref->kind == MGR_COMPONENT_COARSE)
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id, "MGR coarsest setup reuse: reuse=%d",
                            reuse_accepted);
      }
      else
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats),
                            next_ls_id, "MGR %s setup reuse at level %d: reuse=%d",
                            (ref->kind == MGR_COMPONENT_FRELAX) ? "F-relax" : "G-relax",
                            ref->orig_lvl, reuse_accepted);
      }
   }
}

static int
MGRDestroyCachedSolversExplicitly(void)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
   return 1;
#else
   return 0;
#endif
}

static int
MGRLegacyPostDestroyNeedsGRelaxReclaim(void)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   return 0;
#else
   return 1;
#endif
}

static int
MGRLegacyPostDestroyNeedsFRelaxReclaim(void)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   return 0;
#else
   return 1;
#endif
}

static void
MGRResetCachedSolverKeepFlags(MGR_args *args)
{
   if (!args)
   {
      return;
   }

   args->keep_csolver = 0;
   memset(args->keep_frelax, 0, sizeof(args->keep_frelax));
   memset(args->keep_grelax, 0, sizeof(args->keep_grelax));
}

void
hypredrv_MGRCountCachedSolvers(const MGR_args *args, int *num_frelax, int *num_grelax,
                               int *num_coarse)
{
   int frelax = 0;
   int grelax = 0;
   int coarse = 0;

   if (args)
   {
      int max_levels = hypredrv_MGRNumFineLevels(args);
      for (int i = 0; i < max_levels; i++)
      {
         frelax += (args->frelax[i] != NULL);
         grelax += (args->grelax[i] != NULL);
      }
      coarse = (args->csolver != NULL);
   }

   if (num_frelax)
   {
      *num_frelax = frelax;
   }
   if (num_grelax)
   {
      *num_grelax = grelax;
   }
   if (num_coarse)
   {
      *num_coarse = coarse;
   }
}

void
hypredrv_MGRCountKeepFlags(const MGR_args *args, int *num_frelax, int *num_grelax,
                           int *num_coarse)
{
   int frelax = 0;
   int grelax = 0;
   int coarse = 0;

   if (args)
   {
      int max_levels = hypredrv_MGRNumFineLevels(args);
      for (int i = 0; i < max_levels; i++)
      {
         frelax += (args->keep_frelax[i] != 0);
         grelax += (args->keep_grelax[i] != 0);
      }
      coarse = (args->keep_csolver != 0);
   }

   if (num_frelax)
   {
      *num_frelax = frelax;
   }
   if (num_grelax)
   {
      *num_grelax = grelax;
   }
   if (num_coarse)
   {
      *num_coarse = coarse;
   }
}

static void
MGRLogComponentReuseDecision(const Stats *stats, const char *component_name, int level,
                             const MGRComponentReuse_args *reuse, int next_ls_id,
                             int keep)
{
   if (!reuse || !reuse->present)
   {
      return;
   }

   const char *selector = "frequency";

   if (!reuse->args.enabled)
   {
      selector = "disabled";
   }
   else if (reuse->args.linear_system_ids && reuse->args.linear_system_ids->size == 1 &&
            reuse->args.linear_system_ids->data[0] == 0)
   {
      selector = "always_after_first_setup";
   }
   else if (reuse->args.linear_system_ids && reuse->args.linear_system_ids->size > 0)
   {
      selector = "linear_system_ids";
   }
   else if (reuse->args.per_timestep)
   {
      selector = "per_timestep";
   }

   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "MGR component reuse decision: component=%s level=%d next_ls_id=%d "
                      "selector=%s present=%d enabled=%d keep=%d",
                      component_name, level, next_ls_id, selector, reuse->present,
                      reuse->args.enabled, keep);
}

void
hypredrv_MGRSelectCachedSolversToKeep(MGR_args *args, const IntArray *timestep_starts,
                                      const Stats *stats, int next_ls_id)
{
   if (!args)
   {
      return;
   }

   MGRResetCachedSolverKeepFlags(args);

   if (!MGRManagedRefreshShapeSupported(args))
   {
      return;
   }
#if !HYPRE_CHECK_MIN_VERSION(30100, 50)
   (void)timestep_starts;
   (void)stats;
   (void)next_ls_id;
   return;
#endif

   MGRComponentRef refs[MGR_MAX_COMPONENT_REFS];
   int             num_refs = MGRListComponents(args, refs);

   for (int n = 0; n < num_refs; n++)
   {
      const MGRComponentRef *ref = &refs[n];

      if (!MGRComponentUsesManagedHandle(args, ref))
      {
         continue;
      }

      const MGRComponentReuse_args *reuse = MGRComponentReuseArgsConst(args, ref);
      int keep = MGRComponentReuseShouldKeep(reuse, timestep_starts, stats, next_ls_id);

      MGRLogComponentReuseDecision(stats, MGRComponentName(ref->kind), ref->orig_lvl,
                                   reuse, next_ls_id, keep);
      if (!keep)
      {
         continue;
      }

      switch (ref->kind)
      {
         case MGR_COMPONENT_FRELAX:
            args->keep_frelax[ref->orig_lvl] = 1;
            break;
         case MGR_COMPONENT_GRELAX:
            args->keep_grelax[ref->orig_lvl] = 1;
            break;
         default:
            args->keep_csolver = 1;
            break;
      }
   }
}

/* Teardown policy resolved once per destroy call.
 *
 * Detached user-managed handles are always reclaimed before parent teardown.
 * After parent teardown, experimental builds reclaim dropped detached handles
 * explicitly. Standard builds keep the legacy first-active-level fallback only
 * where older hypre MGR teardowns still leave ownership to us: HYPRE's
 * ownership-aware MGR teardown leaves user-installed solvers to the caller,
 * while legacy teardowns expose no release APIs and require the caller to
 * reclaim detached handles after destroying the parent. */
typedef struct
{
   int destroy_handles;
   int first_active_level;
   int legacy_grelax_reclaim;
   int hypre_destroyed;
} MGRDestroyPolicy;

/* A detached solver may only be reclaimed once its owner has released it. */
static int
MGRDetachedReclaimAllowed(const MGRDestroyPolicy *policy, int level, int legacy_gate)
{
   return (!policy->hypre_destroyed || policy->destroy_handles ||
           (legacy_gate && level == policy->first_active_level));
}

static void
MGRDestroyCachedCoarsestSolver(MGR_args *args, const MGRDestroyPolicy *policy)
{
   int drop_csolver = !policy->destroy_handles || !args->keep_csolver;

   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      if (drop_csolver)
      {
         hypredrv_NestedKrylovDestroy(args->coarsest_level.krylov);
      }
   }
   else if (args->csolver && drop_csolver)
   {
      hypredrv_MGRCoarseSolverDestroyByType(args->csolver_type, &args->csolver);
   }

   if (drop_csolver)
   {
      args->csolver      = NULL;
      args->csolver_type = -1;
   }
}

static void
MGRDestroyCachedFRelax(MGR_args *args, int i, const MGRDestroyPolicy *policy)
{
   int drop_frelax = !policy->destroy_handles || !args->keep_frelax[i];

   if (args->level[i].f_relaxation.use_krylov && args->level[i].f_relaxation.krylov)
   {
      if (drop_frelax)
      {
         hypredrv_NestedKrylovDestroy(args->level[i].f_relaxation.krylov);
      }
   }
   else if (args->frelax[i] && drop_frelax &&
            MGRDetachedReclaimAllowed(policy, i,
                                      MGRLegacyPostDestroyNeedsFRelaxReclaim()))
   {
      MGRDestroyDetachedFSolver(&args->level[i].f_relaxation, &args->frelax[i]);
   }

   if (drop_frelax)
   {
      args->frelax[i] = NULL;
      if (args->level[i].f_relaxation.type == 2)
      {
         hypredrv_AMGDestroyRBMs(&args->level[i].f_relaxation.amg);
      }
   }
}

static void
MGRDestroyCachedGRelax(MGR_args *args, int i, const MGRDestroyPolicy *policy)
{
   int drop_grelax = !policy->destroy_handles || !args->keep_grelax[i];

   if (args->level[i].g_relaxation.use_krylov && args->level[i].g_relaxation.krylov)
   {
      if (drop_grelax)
      {
         hypredrv_NestedKrylovDestroy(args->level[i].g_relaxation.krylov);
      }
   }
   else if (args->grelax[i] && drop_grelax &&
            MGRDetachedReclaimAllowed(policy, i, policy->legacy_grelax_reclaim))
   {
      MGRDestroyDetachedGSolver(&args->level[i].g_relaxation, &args->grelax[i]);
   }

   if (drop_grelax)
   {
      args->grelax[i] = NULL;
   }
}

void
hypredrv_MGRDestroyCachedSolvers(MGR_args *args, int hypre_destroyed)
{
   MGRDestroyPolicy policy;
   int              max_levels = 0;

   if (!args)
   {
      return;
   }

   policy.destroy_handles = MGRDestroyCachedSolversExplicitly();
   policy.first_active_level =
      (args->num_active_levels > 0) ? (int)args->active_level_map[0] : -1;
   policy.legacy_grelax_reclaim = MGRLegacyPostDestroyNeedsGRelaxReclaim();
   policy.hypre_destroyed       = hypre_destroyed;

   MGRDestroyCachedCoarsestSolver(args, &policy);

   max_levels = hypredrv_MGRNumFineLevels(args);
   for (int i = 0; i < max_levels; i++)
   {
      MGRDestroyCachedFRelax(args, i, &policy);
      MGRDestroyCachedGRelax(args, i, &policy);
   }

   MGRResetCachedSolverKeepFlags(args);
}

/*-----------------------------------------------------------------------------
 * Destroy an MGR handle together with the component solvers cached in args,
 * in the order required by the linked hypre. Without setup, cached handles may
 * be owned by hypredrive only and are reclaimed before the parent; after setup
 * (or on hypre builds that always destroy installed level solvers) the parent
 * goes first and only the cached-handle state is cleared afterwards.
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRDestroyWithCachedSolvers(MGR_args *args, HYPRE_Solver solver, int was_setup)
{
   int destroy_parent_first = was_setup;
#if HYPRE_RELEASE_NUMBER_EQ_AND_DEVELOP_NUMBER_GE(30100, 5) && \
   !HYPRE_CHECK_MIN_VERSION(30100, 28)
   /* These development builds destroy installed level solvers even when MGR
    * setup has not run. Reclaiming cached handles first leaves dangling solver
    * pointers in the parent, which then destroys them a second time. */
   destroy_parent_first = 1;
#endif

   if (!destroy_parent_first)
   {
      hypredrv_MGRDestroyCachedSolvers(args, 0);
      if (solver)
      {
         HYPRE_MGRDestroy(solver);
      }
   }
   else
   {
      if (solver)
      {
         HYPRE_MGRDestroy(solver);
      }
      /* Parent MGR is gone; clear any preserved cached-handle state without
       * destroying parent-owned internals a second time. */
      hypredrv_MGRDestroyCachedSolvers(args, 1);
   }
}

void
hypredrv_MGRForgetCachedSolvers(MGR_args *args)
{
   if (!args)
   {
      return;
   }

   int max_levels = hypredrv_MGRNumFineLevels(args);
   for (int i = 0; i < max_levels; i++)
   {
      if (args->level[i].f_relaxation.type == 2)
      {
         hypredrv_AMGDestroyRBMs(&args->level[i].f_relaxation.amg);
      }
   }

   args->csolver      = NULL;
   args->csolver_type = -1;
   memset(args->frelax, 0, sizeof(args->frelax));
   memset(args->grelax, 0, sizeof(args->grelax));
   MGRResetCachedSolverKeepFlags(args);
}

/*-----------------------------------------------------------------------------
 * hypredrv_MGRCreate is split into phases that communicate through an
 * MGRCreatePlan: coarsening layout (C-point lists per level), point-marker
 * assembly, base/level option transfer, and per-component solver setup.
 *-----------------------------------------------------------------------------*/
