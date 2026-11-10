/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/solver.h"

#include <math.h>
#include "_hypre_IJ_mv.h"
/* The BiCGSTAB convergence query is only declared in hypre's internal headers. */
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
#include "_hypre_krylov.h"
#else
#include "krylov.h"
#endif
#include "_hypre_parcsr_mv.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 1)
#include "_hypre_utilities.h" // for hypre_Solver
#endif
#include "internal/gen_macros.h"
#include "internal/linsys.h"
#include "internal/presets.h"
#include "logging.h"

static HYPRE_Int
PreconSetupNoop(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 0;
}

static int
SolverLinearSystemID(const Stats *stats)
{
   return stats ? hypredrv_StatsGetLinearSystemID(stats) : 0;
}

typedef struct SolverOps_struct
{
   void (*set_defaults)(solver_args *);
   YAMLGetValidKeysFunc   get_keys;
   YAMLGetValidValuesFunc get_values;
   const FieldOffsetMap  *fields;
   void (*create)(MPI_Comm, const solver_args *, HYPRE_Solver *);
   HYPRE_Int (*set_precond)(HYPRE_Solver, HYPRE_PtrToParSolverFcn,
                            HYPRE_PtrToParSolverFcn, HYPRE_Solver);
   HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
   HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
   HYPRE_Int (*destroy)(HYPRE_Solver);
   HYPRE_Int (*set_reference)(HYPRE_Solver, HYPRE_ParVector);
   HYPRE_Int (*get_num_iterations)(HYPRE_Solver, HYPRE_Int *);
   HYPRE_Int (*get_converged)(HYPRE_Solver, HYPRE_Int *);
   HYPRE_Int (*get_final_rel_res_norm)(HYPRE_Solver, HYPRE_Real *);
} SolverOps;

struct hypredrv_Solver_struct
{
   const SolverOps *ops;
   HYPRE_Solver     native;
   MPI_Comm         comm;
   solver_t         method;
};

typedef struct NestedKrylovRuntime_struct
{
   const SolverOps *ops;
   MPI_Comm         comm;
   HYPRE_Solver     solver;
   HYPRE_Precon     precon;
   HYPRE_Matrix     setup_matrix;
} NestedKrylovRuntime;

#define PCG_FIELDS(_X, _p)                                   \
   _X(_p, max_iter, hypredrv_FieldTypeNonNegIntSet, 100)     \
   _X(_p, two_norm, hypredrv_FieldTypeIntSet, 1)             \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, rel_change, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, recompute_res, hypredrv_FieldTypeIntSet, 0)        \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, residual_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

#define PCG_NUM_FIELDS (sizeof(PCG_field_offset_map) / sizeof(PCG_field_offset_map[0]))

// clang-format off
DEFINE_FIELD_OFFSET_MAP4(PCG)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_PCGGetValidKeys, PCG_NUM_FIELDS,
                          PCG_field_offset_map)

static void
SolverPCGSetDefaults(solver_args *args)
{
   PCG_args *_args = &args->pcg;
   PCG_FIELDS(ADD_FIELD_DEFAULT_ENTRY4, PCG)
}
// clang-format on

StrIntMapArray
hypredrv_PCGGetValidValues(const char *key)
{
   if (!strcmp(key, "two_norm") || !strcmp(key, "stop_crit") ||
       !strcmp(key, "rel_change"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }

   return STR_INT_MAP_ARRAY_VOID();
}

static void
SolverPCGCreate(MPI_Comm comm, const solver_args *options, HYPRE_Solver *solver_ptr)
{
   const PCG_args *args   = &options->pcg;
   HYPRE_Solver    solver = NULL;

   HYPRE_ParCSRPCGCreate(comm, &solver);
   HYPRE_PCGSetMaxIter(solver, args->max_iter);
   HYPRE_PCGSetTwoNorm(solver, args->two_norm);
   HYPRE_PCGSetStopCrit(solver, args->stop_crit);
   HYPRE_PCGSetRelChange(solver, args->rel_change);
   HYPRE_PCGSetPrintLevel(solver, args->print_level);
   HYPRE_PCGSetRecomputeResidual(solver, args->recompute_res);
   HYPRE_PCGSetTol(solver, args->relative_tol);
   HYPRE_PCGSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_PCGSetResidualTol(solver, args->residual_tol);
   HYPRE_PCGSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}

#define GMRES_FIELDS(_X, _p)                                 \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeNonNegIntSet, 300)     \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, skip_real_res_check, hypredrv_FieldTypeIntSet, 0)  \
   _X(_p, krylov_dim, hypredrv_FieldTypePositiveIntSet, 30)  \
   _X(_p, rel_change, hypredrv_FieldTypeIntSet, 0)           \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

#define GMRES_NUM_FIELDS \
   (sizeof(GMRES_field_offset_map) / sizeof(GMRES_field_offset_map[0]))

// clang-format off
DEFINE_FIELD_OFFSET_MAP4(GMRES)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_GMRESGetValidKeys, GMRES_NUM_FIELDS,
                          GMRES_field_offset_map)

static void
SolverGMRESSetDefaults(solver_args *args)
{
   GMRES_args *_args = &args->gmres;
   GMRES_FIELDS(ADD_FIELD_DEFAULT_ENTRY4, GMRES)
}
// clang-format on

StrIntMapArray
hypredrv_GMRESGetValidValues(const char *key)
{
   if (!strcmp(key, "skip_real_res_check") || !strcmp(key, "rel_change"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

static void
SolverGMRESCreate(MPI_Comm comm, const solver_args *options, HYPRE_Solver *solver_ptr)
{
   const GMRES_args *args   = &options->gmres;
   HYPRE_Solver      solver = NULL;

   HYPRE_ParCSRGMRESCreate(comm, &solver);
   HYPRE_GMRESSetMinIter(solver, args->min_iter);
   HYPRE_GMRESSetMaxIter(solver, args->max_iter);
   HYPRE_GMRESSetStopCrit(solver, args->stop_crit);
   HYPRE_GMRESSetSkipRealResidualCheck(solver, args->skip_real_res_check);
   HYPRE_GMRESSetKDim(solver, args->krylov_dim);
   HYPRE_GMRESSetRelChange(solver, args->rel_change);
   HYPRE_GMRESSetLogging(solver, args->logging);
   HYPRE_GMRESSetPrintLevel(solver, args->print_level);
   HYPRE_GMRESSetTol(solver, args->relative_tol);
   HYPRE_GMRESSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_GMRESSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}

#define FGMRES_FIELDS(_X, _p)                                \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeNonNegIntSet, 300)     \
   _X(_p, krylov_dim, hypredrv_FieldTypePositiveIntSet, 30)  \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)

#define FGMRES_NUM_FIELDS \
   (sizeof(FGMRES_field_offset_map) / sizeof(FGMRES_field_offset_map[0]))

// clang-format off
DEFINE_FIELD_OFFSET_MAP4(FGMRES)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_FGMRESGetValidKeys, FGMRES_NUM_FIELDS,
                          FGMRES_field_offset_map)
hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC(hypredrv_FGMRES)

static void
SolverFGMRESSetDefaults(solver_args *args)
{
   FGMRES_args *_args = &args->fgmres;
   FGMRES_FIELDS(ADD_FIELD_DEFAULT_ENTRY4, FGMRES)
}
// clang-format on

static void
SolverFGMRESCreate(MPI_Comm comm, const solver_args *options, HYPRE_Solver *solver_ptr)
{
   const FGMRES_args *args   = &options->fgmres;
   HYPRE_Solver       solver = NULL;

   HYPRE_ParCSRFlexGMRESCreate(comm, &solver);
   HYPRE_FlexGMRESSetMinIter(solver, args->min_iter);
   HYPRE_FlexGMRESSetMaxIter(solver, args->max_iter);
   HYPRE_FlexGMRESSetKDim(solver, args->krylov_dim);
   HYPRE_FlexGMRESSetLogging(solver, args->logging);
   HYPRE_FlexGMRESSetPrintLevel(solver, args->print_level);
   HYPRE_FlexGMRESSetTol(solver, args->relative_tol);
   HYPRE_FlexGMRESSetAbsoluteTol(solver, args->absolute_tol);

   *solver_ptr = solver;
}

#define BiCGSTAB_FIELDS(_X, _p)                              \
   _X(_p, min_iter, hypredrv_FieldTypeIntSet, 0)             \
   _X(_p, max_iter, hypredrv_FieldTypeNonNegIntSet, 100)     \
   _X(_p, stop_crit, hypredrv_FieldTypeIntSet, 0)            \
   _X(_p, logging, hypredrv_FieldTypeIntSet, 1)              \
   _X(_p, print_level, hypredrv_FieldTypeIntSet, 1)          \
   _X(_p, relative_tol, hypredrv_FieldTypeDoubleSet, 1.0e-6) \
   _X(_p, absolute_tol, hypredrv_FieldTypeDoubleSet, 0.0)    \
   _X(_p, conv_fac_tol, hypredrv_FieldTypeDoubleSet, 0.0)

#define BiCGSTAB_NUM_FIELDS \
   (sizeof(BiCGSTAB_field_offset_map) / sizeof(BiCGSTAB_field_offset_map[0]))

// clang-format off
DEFINE_FIELD_OFFSET_MAP4(BiCGSTAB)
DEFINE_GET_VALID_KEYS_FUNC(hypredrv_BiCGSTABGetValidKeys, BiCGSTAB_NUM_FIELDS,
                          BiCGSTAB_field_offset_map)
hypredrv_DEFINE_VOID_GET_VALID_VALUES_FUNC(hypredrv_BiCGSTAB)

static void
SolverBiCGSTABSetDefaults(solver_args *args)
{
   BiCGSTAB_args *_args = &args->bicgstab;
   BiCGSTAB_FIELDS(ADD_FIELD_DEFAULT_ENTRY4, BiCGSTAB)
}
// clang-format on

static void
SolverBiCGSTABCreate(MPI_Comm comm, const solver_args *options, HYPRE_Solver *solver_ptr)
{
   const BiCGSTAB_args *args   = &options->bicgstab;
   HYPRE_Solver         solver = NULL;

   HYPRE_ParCSRBiCGSTABCreate(comm, &solver);
   HYPRE_BiCGSTABSetMinIter(solver, args->min_iter);
   HYPRE_BiCGSTABSetMaxIter(solver, args->max_iter);
   HYPRE_BiCGSTABSetStopCrit(solver, args->stop_crit);
   HYPRE_BiCGSTABSetLogging(solver, args->logging);
   HYPRE_BiCGSTABSetPrintLevel(solver, args->print_level);
   HYPRE_BiCGSTABSetTol(solver, args->relative_tol);
   HYPRE_BiCGSTABSetAbsoluteTol(solver, args->absolute_tol);
   HYPRE_BiCGSTABSetConvergenceFactorTol(solver, args->conv_fac_tol);

   *solver_ptr = solver;
}

static HYPRE_Int
SolverBiCGSTABGetConverged(HYPRE_Solver solver, HYPRE_Int *converged)
{
   /* hypre does not expose HYPRE_BiCGSTABGetConverged; use the internal API. */
   return hypre_BiCGSTABGetConverged(solver, converged);
}

static const SolverOps solver_ops[] = {
   [SOLVER_PCG] =
      {
         .set_defaults           = SolverPCGSetDefaults,
         .get_keys               = hypredrv_PCGGetValidKeys,
         .get_values             = hypredrv_PCGGetValidValues,
         .fields                 = PCG_field_offset_map,
         .create                 = SolverPCGCreate,
         .set_precond            = HYPRE_ParCSRPCGSetPrecond,
         .setup                  = HYPRE_ParCSRPCGSetup,
         .solve                  = HYPRE_ParCSRPCGSolve,
         .destroy                = HYPRE_ParCSRPCGDestroy,
         .get_num_iterations     = HYPRE_PCGGetNumIterations,
         .get_converged          = HYPRE_PCGGetConverged,
         .get_final_rel_res_norm = HYPRE_PCGGetFinalRelativeResidualNorm,
      },
   [SOLVER_GMRES] =
      {
         .set_defaults = SolverGMRESSetDefaults,
         .get_keys     = hypredrv_GMRESGetValidKeys,
         .get_values   = hypredrv_GMRESGetValidValues,
         .fields       = GMRES_field_offset_map,
         .create       = SolverGMRESCreate,
         .set_precond  = HYPRE_ParCSRGMRESSetPrecond,
         .setup        = HYPRE_ParCSRGMRESSetup,
         .solve        = HYPRE_ParCSRGMRESSolve,
         .destroy      = HYPRE_ParCSRGMRESDestroy,
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
         .set_reference = HYPRE_ParCSRGMRESSetRefSolution,
#endif
         .get_num_iterations     = HYPRE_GMRESGetNumIterations,
         .get_converged          = HYPRE_GMRESGetConverged,
         .get_final_rel_res_norm = HYPRE_GMRESGetFinalRelativeResidualNorm,
      },
   [SOLVER_FGMRES] =
      {
         .set_defaults           = SolverFGMRESSetDefaults,
         .get_keys               = hypredrv_FGMRESGetValidKeys,
         .get_values             = hypredrv_FGMRESGetValidValues,
         .fields                 = FGMRES_field_offset_map,
         .create                 = SolverFGMRESCreate,
         .set_precond            = HYPRE_ParCSRFlexGMRESSetPrecond,
         .setup                  = HYPRE_ParCSRFlexGMRESSetup,
         .solve                  = HYPRE_ParCSRFlexGMRESSolve,
         .destroy                = HYPRE_ParCSRFlexGMRESDestroy,
         .get_num_iterations     = HYPRE_FlexGMRESGetNumIterations,
         .get_converged          = HYPRE_FlexGMRESGetConverged,
         .get_final_rel_res_norm = HYPRE_FlexGMRESGetFinalRelativeResidualNorm,
      },
   [SOLVER_BICGSTAB] =
      {
         .set_defaults           = SolverBiCGSTABSetDefaults,
         .get_keys               = hypredrv_BiCGSTABGetValidKeys,
         .get_values             = hypredrv_BiCGSTABGetValidValues,
         .fields                 = BiCGSTAB_field_offset_map,
         .create                 = SolverBiCGSTABCreate,
         .set_precond            = HYPRE_ParCSRBiCGSTABSetPrecond,
         .setup                  = HYPRE_ParCSRBiCGSTABSetup,
         .solve                  = HYPRE_ParCSRBiCGSTABSolve,
         .destroy                = HYPRE_ParCSRBiCGSTABDestroy,
         .get_num_iterations     = HYPRE_BiCGSTABGetNumIterations,
         .get_converged          = SolverBiCGSTABGetConverged,
         .get_final_rel_res_norm = HYPRE_BiCGSTABGetFinalRelativeResidualNorm,
      },
};

static const SolverOps *
SolverOpsLookup(solver_t method)
{
   if ((unsigned)method >= (unsigned)(sizeof(solver_ops) / sizeof(solver_ops[0])) ||
       !solver_ops[method].setup)
   {
      return NULL;
   }

   return &solver_ops[method];
}

StrIntMapArray
hypredrv_SolverGetValidTypeIntMap(void)
{
   static StrIntMap map[] = {
      {"pcg", (int)SOLVER_PCG},
      {"gmres", (int)SOLVER_GMRES},
      {"fgmres", (int)SOLVER_FGMRES},
      {"bicgstab", (int)SOLVER_BICGSTAB},
   };

   return STR_INT_MAP_ARRAY_CREATE(map);
}

StrArray
hypredrv_SolverGetValidKeys(void)
{
   static const char *keys[sizeof(solver_ops) / sizeof(solver_ops[0])];
   StrIntMapArray     methods = hypredrv_SolverGetValidTypeIntMap();
   for (size_t i = 0; i < methods.size; i++)
   {
      keys[i] = methods.data[i].str;
   }
   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * SolverArgsSetDefaultsForMethod
 *
 * Convenience helper for callers that need per-method defaults without constructing
 * fake YAML nodes (e.g., value-only `solver: gmres`).
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverArgsSetDefaultsForMethod(solver_t method, solver_args *args)
{
   const SolverOps *ops;

   if (!args)
   {
      return;
   }

   ops = SolverOpsLookup(method);
   if (!ops)
   {
      return;
   }

   ops->set_defaults(args);
}

/* Both ingress paths select the method and stage fresh storage before parsing. */
static bool
SolverParseBody(const SolverOps *ops, solver_args *args, YAMLnode *body,
                const YAMLnode *ignored_child)
{
   ops->set_defaults(args);
   YAML_NODE_SET_VALID(body);
   if ((body->children || body->val[0]) &&
       !hypredrv_YAMLSetArgsGeneric(args, body, ops->get_keys(), ops->get_values,
                                    ops->fields, ignored_child))
      return false;
   if (hypredrv_ErrorCodeActive()) return false;
   return true;
}

bool
hypredrv_SolverSetArgsFromYAML(solver_t *method, solver_args *args, YAMLnode *node,
                               const YAMLnode *ignored_child)
{
   if (!method || !args || !node)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid solver configuration arguments");
      return false;
   }
   if (hypredrv_ErrorCodeActive()) return false;

   bool      section = !strcmp(node->key, "solver");
   bool      scalar  = section && node->val[0];
   YAMLnode *body    = node;
   if (section)
   {
      YAML_NODE_SET_VALID(node);
      if (!scalar)
      {
         body = NULL;
         YAML_NODE_ITERATE(node, child)
         {
            if (child == ignored_child) continue;
            if (body)
            {
               hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
               hypredrv_ErrorMsgAddExtraKey(child->key);
               return false;
            }
            body = child;
         }
         if (!body)
         {
            hypredrv_ErrorCodeSet(ERROR_MISSING_SOLVER);
            return false;
         }
      }
   }

   StrIntMapArray types    = hypredrv_SolverGetValidTypeIntMap();
   int            selected = -1;
   if (scalar)
   {
      if (!hypredrv_StrIntMapArrayLookup(types, node->val, &selected))
      {
         node->avail_vals = types;
         YAML_NODE_SET_INVALID_VAL(node);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Unknown solver type: '%s'", node->val);
         return false;
      }
   }
   else
   {
      /* Method keys use their literal schema names; numeric values are allowed
       * only in the scalar section syntax. Bind the selected operations here. */
      for (size_t i = 0; i < types.size; i++)
      {
         if (!strcmp(body->key, types.data[i].str))
         {
            selected = types.data[i].num;
            break;
         }
      }
      if (selected < 0)
      {
         body->avail_keys = hypredrv_SolverGetValidKeys();
         YAML_NODE_SET_INVALID_KEY(body);
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Unknown solver type: '%s'", body->key);
         return false;
      }
   }

   const SolverOps *ops = &solver_ops[selected];
   solver_args      parsed;
   if (scalar)
   {
      ops->set_defaults(&parsed);
      switch ((solver_t)selected)
      {
         case SOLVER_PCG:
            parsed.pcg.print_level = 0;
            break;
         case SOLVER_GMRES:
            parsed.gmres.print_level = 0;
            break;
         case SOLVER_FGMRES:
            parsed.fgmres.print_level = 0;
            break;
         case SOLVER_BICGSTAB:
            parsed.bicgstab.print_level = 0;
            break;
      }
   }
   else if (!SolverParseBody(ops, &parsed, body, ignored_child)) return false;
   *method = (solver_t)selected;
   *args   = parsed;
   return true;
}

bool
hypredrv_SolverArgsFromPreset(const char *name, solver_t *method, solver_args *args)
{
   if (!name || !method || !args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid solver preset arguments");
      return false;
   }
   if (hypredrv_ErrorCodeActive()) return false;

   int         selected;
   solver_args parsed;
   bool        valid = true;
   if (hypredrv_StrIntMapArrayLookup(hypredrv_SolverGetValidTypeIntMap(), name,
                                     &selected))
      solver_ops[selected].set_defaults(&parsed);
   else
   {
      YAMLtree *tree = NULL;
      YAMLnode *body =
         hypredrv_PresetLoadTypeNode(name, HYPREDRV_PRESET_SOLVER, &tree, &selected);
      valid = body && SolverParseBody(&solver_ops[selected], &parsed, body, NULL);
      if (tree) hypredrv_YAMLtreeValidate(tree);
      hypredrv_YAMLtreeDestroy(&tree);
   }
   if (!valid || hypredrv_ErrorCodeActive()) return false;
   *method = (solver_t)selected;
   *args   = parsed;
   return true;
}

/*-----------------------------------------------------------------------------
 * SolverCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_SolverCreate(MPI_Comm comm, solver_t method, const solver_args *args,
                      hypredrv_Solver **output)
{
   if (output) *output = NULL;
   if (!output || !args)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Solver creation requires arguments and an output pointer");
      HYPREDRV_LOGF(2, hypredrv_LogRankFromComm(comm), NULL, 0,
                    "solver create failed: %s is NULL", output ? "args" : "solver_ptr");
      return;
   }
   if (hypredrv_ErrorCodeActive()) return;
   const SolverOps *ops = SolverOpsLookup(method);
   if (!ops)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("Invalid solver method: %d", (int)method);
      HYPREDRV_LOGF(2, hypredrv_LogRankFromComm(comm), NULL, 0,
                    "solver create failed: invalid solver method=%d", (int)method);
      return;
   }
   hypredrv_Solver *solver = hypredrv_Calloc(1, sizeof(*solver), "solver runtime");
   if (!solver) return;
   solver->ops    = ops;
   solver->comm   = comm;
   solver->method = method;
   ops->create(comm, args, &solver->native);
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_ErrorCodeActive() || !solver->native)
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Solver creation returned no native solver");
      }
      if (solver->native) ops->destroy(solver->native);
      free(solver);
      return;
   }
   *output = solver;
}

/* GMRES alone uses this optional capability. Clearing a reference must also
 * clear the native pointer, so a replaced vector cannot leave a stale borrower. */
static void
SolverSetReference(hypredrv_Solver *solver, HYPRE_IJVector reference)
{
   if (!solver->ops->set_reference) return;
   void *object = NULL;
   if (reference) HYPRE_IJVectorGetObject(reference, &object);
   solver->ops->set_reference(solver->native, (HYPRE_ParVector)object);
}

void
hypredrv_SolverSetupWithReuse(hypredrv_Solver *solver, HYPRE_Precon precon,
                              HYPRE_IJMatrix M, HYPRE_IJVector b, HYPRE_IJVector x,
                              HYPRE_IJVector reference, Stats *stats,
                              int skip_precon_setup)
{
   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSetup: solver is NULL");
      HYPREDRV_LOG_COMMF(2, hypredrv_IJSystemComm(M, NULL, b, x), NULL, 0,
                         "solver setup failed: solver is NULL");
      return;
   }
   if (!M || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverSetup: matrix or vector is NULL");
      HYPREDRV_LOGF(2, hypredrv_LogRankFromComm(solver->comm), NULL, 0,
                    "solver setup failed: matrix or vector is NULL");
      return;
   }
   int  ls_id = SolverLinearSystemID(stats);
   int  rank  = hypredrv_LogEnabled(2) ? hypredrv_LogRankFromComm(solver->comm) : -1;
   char name_buffer[32];
   const char *name =
      hypredrv_StatsGetLogObjectName(stats, name_buffer, sizeof(name_buffer));
   HYPREDRV_LOGF(3, rank, name, ls_id,
                 "solver setup begin (solver=%d precon=%d skip_precon_setup=%d)",
                 (int)solver->method, precon ? (int)precon->method : (int)PRECON_NONE,
                 skip_precon_setup);
   void *matrix = NULL, *rhs = NULL, *solution = NULL;
   HYPRE_IJMatrixGetObject(M, &matrix);
   HYPRE_IJVectorGetObject(b, &rhs);
   HYPRE_IJVectorGetObject(x, &solution);
   if (precon)
   {
      precon->stats = skip_precon_setup ? NULL : stats;
      if (skip_precon_setup)
      {
         HYPREDRV_LOGF(3, rank, name, ls_id,
                       "solver setup: detached preconditioner stats (skip_precon_setup)");
      }
      if (!hypredrv_PreconValidateSetup(precon, (HYPRE_ParCSRMatrix)matrix)) return;
      if (precon->method != PRECON_NONE)
         solver->ops->set_precond(solver->native, hypredrv_PreconKrylovSolve,
                                  skip_precon_setup ? PreconSetupNoop
                                                    : hypredrv_PreconKrylovSetup,
                                  (HYPRE_Solver)precon);
   }
   SolverSetReference(solver, reference);
   solver->ops->setup(solver->native, (HYPRE_ParCSRMatrix)matrix, (HYPRE_ParVector)rhs,
                      (HYPRE_ParVector)solution);
   hypredrv_HypreConsumeErrors(solver->comm);
   HYPREDRV_LOGF(3, rank, name, ls_id, "solver setup end");
}

/* Both execution paths have admitted a live runtime and complete IJ system. */
static HYPRE_Int
SolverSolve(hypredrv_Solver *solver, HYPRE_IJMatrix A, HYPRE_IJVector b, HYPRE_IJVector x,
            HYPRE_IJVector reference)
{
   void *matrix = NULL, *rhs = NULL, *solution = NULL;
   HYPRE_IJMatrixGetObject(A, &matrix);
   HYPRE_IJVectorGetObject(b, &rhs);
   HYPRE_IJVectorGetObject(x, &solution);
   SolverSetReference(solver, reference);
   solver->ops->solve(solver->native, (HYPRE_ParCSRMatrix)matrix, (HYPRE_ParVector)rhs,
                      (HYPRE_ParVector)solution);
   HYPRE_Int iters = 0;
   solver->ops->get_num_iterations(solver->native, &iters);
   hypredrv_HypreConsumeErrors(solver->comm);
   return iters;
}

HYPRE_Int
hypredrv_SolverSolveOnly(hypredrv_Solver *solver, HYPRE_IJMatrix A, HYPRE_IJVector b,
                         HYPRE_IJVector x, HYPRE_IJVector reference)
{
   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverSolveOnly: solver is NULL");
      HYPREDRV_LOG_COMMF(2, hypredrv_IJSystemComm(A, NULL, b, x), NULL, 0,
                         "solver solve failed: solver is NULL");
      return -1;
   }
   if (!A || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverSolveOnly: matrix or vector is NULL");
      HYPREDRV_LOGF(2, hypredrv_LogRankFromComm(solver->comm), NULL, 0,
                    "solver solve failed: matrix or vector is NULL");
      return -1;
   }
   return SolverSolve(solver, A, b, x, reference);
}

void
hypredrv_SolverApply(hypredrv_Solver *solver, HYPRE_IJMatrix A, HYPRE_IJVector b,
                     HYPRE_IJVector x, HYPRE_IJVector reference, Stats *stats)
{
   MPI_Comm    log_comm = hypredrv_IJSystemComm(A, NULL, b, x);
   int         ls_id    = SolverLinearSystemID(stats);
   int         log_rank = -1;
   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   /* GCOVR_EXCL_BR_START */
   if (hypredrv_LogEnabled(2)) /* GCOVR_EXCL_BR_STOP */
   {
      log_rank = hypredrv_LogRankFromComm(log_comm);
   }
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver apply begin (solver=%d)",
                 solver ? (int)solver->method : -1);

   if (!solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("SolverApply: solver is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed: solver is NULL");
      return;
   }

   if (!A || !b || !x)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("SolverApply: matrix or vector is NULL");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed: matrix or vector is NULL");
      return;
   }

   HYPRE_Int      iters  = 0;
   HYPRE_Complex  b_norm = NAN, r_norm = NAN, r0_norm = NAN;
   HYPRE_IJVector vec_r = NULL; /* residual work vector shared by both norms */

   /* Compute initial residual norm (absolute L2) before timing the solve */
   hypredrv_LinearSystemComputeResidualNormWork(A, b, x, "L2", &vec_r, &r0_norm);

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "solve");
   hypredrv_StatsInitialResNormSet(stats, r0_norm);

   iters = SolverSolve(solver, A, b, x, reference);

   if (iters < 0)
   {
      if (vec_r)
      {
         HYPRE_IJVectorDestroy(vec_r);
      }
      hypredrv_StatsIterSet(stats, 0);
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");
      HYPREDRV_LOGF(2, log_rank, log_object_name, ls_id,
                    "solver apply failed during solve");
      return;
   }

   hypredrv_StatsIterSet(stats, (int)iters);
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "solve");

   /* Compute the real relative residual norm. Note this is not timed */
   hypredrv_LinearSystemComputeVectorNorm(b, "L2", &b_norm);
   hypredrv_LinearSystemComputeResidualNormWork(A, b, x, "L2", &vec_r, &r_norm);
   if (vec_r)
   {
      HYPRE_IJVectorDestroy(vec_r);
   }
   b_norm = (b_norm > 0.0) ? b_norm : 1.0;

   hypredrv_StatsRelativeResNormSet(stats, r_norm / b_norm);
   HYPREDRV_LOGF(3, log_rank, log_object_name, ls_id, "solver apply end (iters=%d)",
                 (int)iters);
}

void
hypredrv_SolverDestroy(hypredrv_Solver **solver_ptr)
{
   if (!solver_ptr || !*solver_ptr) return;
   hypredrv_Solver *solver = *solver_ptr;
   solver->ops->destroy(solver->native);
   free(solver);
   *solver_ptr = NULL;
}

void
hypredrv_SolverGetConverged(hypredrv_Solver *solver, HYPRE_Int *converged)
{
   if (!solver || !converged)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Convergence query requires a live solver and output pointer");
      return;
   }
   solver->ops->get_converged(solver->native, converged);
}

void
hypredrv_SolverGetFinalRelativeResidualNorm(hypredrv_Solver *solver, HYPRE_Real *norm)
{
   if (!solver || !norm)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("Residual query requires a live solver and output pointer");
      return;
   }
   solver->ops->get_final_rel_res_norm(solver->native, norm);
}

/* Nested Krylov solvers used by MGR smoothers and coarse solves. */

static HYPRE_Int
NestedKrylovDestroyThunk(HYPRE_Solver solver_v)
{
   hypredrv_NestedKrylovDestroy((NestedKrylov_args *)solver_v);
   return 0;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
NestedKrylovCreateSetupVector(HYPRE_Matrix A, HYPRE_ParVector *vector_ptr)
{
   MPI_Comm     comm        = MPI_COMM_NULL;
   HYPRE_BigInt global_rows = 0, global_cols = 0;
   HYPRE_BigInt row_start = 0, row_end = -1, col_start = 0, col_end = -1;
   HYPRE_Int    rc = 0;

   if (!A || !vector_ptr)
   {
      return 1;
   }

   rc |= HYPRE_ParCSRMatrixGetComm((HYPRE_ParCSRMatrix)A, &comm);
   rc |= HYPRE_ParCSRMatrixGetDims((HYPRE_ParCSRMatrix)A, &global_rows, &global_cols);
   rc |= HYPRE_ParCSRMatrixGetLocalRange((HYPRE_ParCSRMatrix)A, &row_start, &row_end,
                                         &col_start, &col_end);
   if (!rc)
   {
      /* The vector takes this rank's row range [first, last + 1]. Older hypre
       * keeps (and later frees) the partitioning array it is given; newer
       * releases copy the two values. */
#if defined(hypre_ParVectorOwnsPartitioning)
      HYPRE_BigInt *partitioning = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
#else
      HYPRE_BigInt  local_partitioning[2];
      HYPRE_BigInt *partitioning = local_partitioning;
#endif
      partitioning[0] = row_start;
      partitioning[1] = row_end + 1;
      rc |= HYPRE_ParVectorCreate(comm, global_rows, partitioning, vector_ptr);
      rc |= hypre_ParVectorInitialize_v2(
         (hypre_ParVector *)*vector_ptr,
         hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)A));
   }

   if (rc && *vector_ptr)
   {
      HYPRE_ParVectorDestroy(*vector_ptr);
      *vector_ptr = NULL;
   }

   return rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovSetDefaultArgs(NestedKrylov_args *args)
{
   if (!args)
   {
      return;
   }

   args->base          = (hypredrv_SolverBase){.setup   = hypredrv_NestedKrylovSetup,
                                               .solve   = hypredrv_NestedKrylovSolve,
                                               .destroy = NestedKrylovDestroyThunk};
   args->is_set        = 0;
   args->solver_method = SOLVER_GMRES;
   args->has_precon    = 0;
   args->precon_method = PRECON_NONE;
   args->runtime       = NULL;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovSetArgsFromYAML(NestedKrylov_args *args, YAMLnode *solver_node)
{
   if (!args || !solver_node)
   {
      return;
   }

   YAMLnode *precon_node = hypredrv_YAMLnodeFindChildByKey(solver_node, "preconditioner");
   solver_t  method;
   solver_args parsed;
   if (!hypredrv_SolverSetArgsFromYAML(&method, &parsed, solver_node, precon_node))
      return;

   precon_t    parsed_method = PRECON_NONE;
   precon_args parsed_precon;
   if (precon_node)
   {
      StrIntMapArray types = hypredrv_PreconGetValidTypeIntMap();
      int            selected;
      YAML_NODE_SET_VALID(precon_node);
      if (precon_node->val[0])
      {
         if (!hypredrv_StrIntMapArrayLookup(types, precon_node->val, &selected))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Unknown nested preconditioner type: '%s'",
                                 precon_node->val);
            precon_node->avail_vals = types;
            YAML_NODE_SET_INVALID_VAL(precon_node);
            goto precon_failure;
         }
         parsed_method = (precon_t)selected;
         hypredrv_PreconArgsSetDefaultsForName(parsed_method, precon_node->val,
                                               &parsed_precon);
      }
      else
      {
         YAMLnode *body = precon_node->children;
         if (!body)
         {
            hypredrv_ErrorCodeSet(ERROR_MISSING_PRECON);
            hypredrv_ErrorMsgAdd("Nested preconditioner type is missing");
            YAML_NODE_SET_INVALID_KEY(precon_node);
            goto precon_failure;
         }
         if (body->next)
         {
            hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
            hypredrv_ErrorMsgAddExtraKey(body->next->key);
            YAML_NODE_SET_INVALID_KEY(body->next);
            goto precon_failure;
         }
         if (!hypredrv_StrIntMapArrayLookup(types, body->key, &selected))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
            hypredrv_ErrorMsgAdd("Unknown nested preconditioner type: '%s'", body->key);
            YAML_NODE_SET_INVALID_KEY(body);
            goto precon_failure;
         }
         parsed_method = (precon_t)selected;
         hypredrv_PreconSetMethodArgs(parsed_method, &parsed_precon, body);
      }
      /* Collect the validity flags assigned by the method parser before any
       * live configuration or native handles can be replaced. */
      hypredrv_YAMLnodeValidate(precon_node);
      if (hypredrv_ErrorCodeActive()) goto precon_failure;
   }

   /* Release runtime objects while their original method and arguments remain
    * installed. An omitted preconditioner preserves its existing configuration. */
   hypredrv_NestedKrylovDestroy(args);
   if (precon_node)
   {
      if (args->has_precon)
         hypredrv_PreconArgsDestroyOwnedConfig(args->precon_method, &args->precon);
      args->has_precon    = 1;
      args->precon_method = parsed_method;
      args->precon        = parsed_precon;
   }
   args->is_set        = 1;
   args->solver_method = method;
   args->solver        = parsed;
   return;

precon_failure:
   hypredrv_PreconArgsDestroyOwnedConfig(parsed_method, &parsed_precon);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

bool
hypredrv_NestedKrylovCreate(MPI_Comm comm, NestedKrylov_args *args, IntArray *dofmap,
                            HYPRE_IJVector vec_nn)
{
   if (!args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov creation requires arguments");
      return false;
   }
   if (hypredrv_ErrorCodeActive()) return false;
   if (args->runtime) return true;
   if (!args->is_set) return false;

   const SolverOps *ops = SolverOpsLookup(args->solver_method);
   if (!ops)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("Nested Krylov solver method not supported");
      return false;
   }
   if (args->has_precon && args->precon_method == PRECON_MGR &&
       !hypredrv_MGRValidateOuterSolver(&args->precon.mgr, (int)args->solver_method,
                                        "Nested MGR"))
      return false;

   NestedKrylovRuntime *runtime =
      hypredrv_Calloc(1, sizeof(*runtime), "nested Krylov runtime");
   if (!runtime) return false;
   runtime->ops  = ops;
   runtime->comm = comm;
   if (args->has_precon)
   {
      hypredrv_PreconCreate(comm, args->precon_method, &args->precon, dofmap, vec_nn,
                            &runtime->precon, NULL, 0, NULL);
      if (hypredrv_ErrorCodeActive()) goto failure;
   }
   ops->create(comm, &args->solver, &runtime->solver);
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_ErrorCodeActive()) goto failure;
   if (!runtime->solver)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
      hypredrv_ErrorMsgAdd("Nested Krylov creation returned no native solver");
      goto failure;
   }
   if (args->has_precon && args->precon_method != PRECON_NONE)
   {
      HYPRE_Int rc =
         ops->set_precond(runtime->solver, hypredrv_PreconKrylovSolve,
                          hypredrv_PreconKrylovSetup, (HYPRE_Solver)runtime->precon);
      hypredrv_HypreConsumeErrors(comm);
      if (rc || hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
         hypredrv_ErrorMsgAdd("Failed to bind nested Krylov preconditioner");
         goto failure;
      }
   }
   args->runtime       = runtime;
   args->base.is_setup = 0;
   return true;

failure:
   if (runtime->solver) ops->destroy(runtime->solver);
   if (runtime->precon) hypredrv_PreconDestroy(&runtime->precon, NULL, 0);
   hypredrv_PreconArgsDestroyRuntimeState(args->precon_method, &args->precon);
   hypredrv_HypreConsumeErrors(comm);
   free(runtime);
   return false;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_NestedKrylovSetup(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                           HYPRE_Vector x)
{
   NestedKrylov_args *args    = (NestedKrylov_args *)solver;
   HYPRE_ParVector    setup_b = NULL, setup_x = NULL;
   HYPRE_Int          rc = 0;
   if (!args || !args->runtime)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov setup requires a live runtime");
      return 1;
   }
   NestedKrylovRuntime *runtime = args->runtime;
   if (runtime->precon)
   {
      if (!hypredrv_PreconValidateSetup(runtime->precon, (HYPRE_ParCSRMatrix)A)) return 1;
   }
   else
   {
      if (!A)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Nested Krylov setup requires a matrix");
      }
      if (hypredrv_DistributedErrorStateSync(runtime->comm)) return 1;
   }
   if (!b)
   {
      rc = NestedKrylovCreateSetupVector(A, &setup_b);
      b  = (HYPRE_Vector)setup_b;
   }
   if (!rc && !x)
   {
      rc = NestedKrylovCreateSetupVector(A, &setup_x);
      x  = (HYPRE_Vector)setup_x;
   }
   if (!rc)
   {
      args->base.is_setup   = 0;
      runtime->setup_matrix = NULL;
      rc = runtime->ops->setup(runtime->solver, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                               (HYPRE_ParVector)x);
      hypredrv_HypreConsumeErrors(runtime->comm);
      if (!rc && hypredrv_ErrorCodeActive()) rc = 1;
   }
   if (setup_b) HYPRE_ParVectorDestroy(setup_b);
   if (setup_x) HYPRE_ParVectorDestroy(setup_x);
   if (!rc)
   {
      args->base.is_setup   = 1;
      runtime->setup_matrix = A;
   }
   return rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

HYPRE_Int
hypredrv_NestedKrylovSolve(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b,
                           HYPRE_Vector x)
{
   NestedKrylov_args *args = (NestedKrylov_args *)solver;
   if (!args || !args->runtime)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov solve called with invalid solver");
      return 1;
   }
   NestedKrylovRuntime *runtime = args->runtime;
   if (!A) A = runtime->setup_matrix;
   if (!runtime->setup_matrix || !A)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested Krylov solve requires successful setup");
      return 1;
   }
   HYPRE_Int solve_rc    = runtime->ops->solve(runtime->solver, (HYPRE_ParCSRMatrix)A,
                                               (HYPRE_ParVector)b, (HYPRE_ParVector)x);
   HYPRE_Int hypre_error = HYPRE_GetError();
   /* Nested iterations are inexact MGR smoothers/coarse solves. Convergence
    * failure must not poison the outer solve; leave harder failures intact. */
   if (hypre_error && !(hypre_error & ~HYPRE_ERROR_CONV))
   {
      hypredrv_HypreClearConvergenceErrors();
      return 0;
   }
   return solve_rc;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_NestedKrylovDestroy(NestedKrylov_args *args)
{
   if (!args) return;
   NestedKrylovRuntime *runtime = args->runtime;
   if (runtime)
   {
      runtime->ops->destroy(runtime->solver);
      if (runtime->precon) hypredrv_PreconDestroy(&runtime->precon, NULL, 0);
      free(runtime);
      args->runtime = NULL;
   }
   args->base.is_setup = 0;
   /* Nested preconditioners have no outer driver managing reuse. Clear cached
    * native state while preserving their owned configuration for another create. */
   hypredrv_PreconArgsDestroyRuntimeState(args->precon_method, &args->precon);
}
