/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/mgr.h"
#include <mpi.h>
#include <stddef.h>
/* gcovr: branch-exclusion regions below narrow branch-count noise from YAML
 * helpers and MGR validation/dispatch; single-line exclusions flag allocator
 * and defensive branches that are impractical to fault-inject here. */
#include "_hypre_utilities.h" // for hypre_Solver
#include "internal/compatibility.h"
#include "internal/error.h"
#include "internal/gen_macros.h"
#include "internal/krylov.h"
#include "internal/stats.h"
#include "logging.h"

typedef HYPRE_Int (*MGRHyprePtrToDestroyFcn)(HYPRE_Solver);

typedef struct MGRFRelaxWrapper_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);
   HYPRE_Int                       is_setup; /* offset 24: mirrors hypre_Solver layout */
   HYPRE_Solver                    inner_mgr;
   IntArray                       *owned_dofmap;
   struct MGRFRelaxWrapper_struct *next_live;
} MGRFRelaxWrapper;

enum
{
   MGR_HYPRE_SOLVER_IS_SETUP_OFFSET = sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(MGRHyprePtrToDestroyFcn),
};

typedef char MGRNestedKrylovLayoutCheck
   [(offsetof(NestedKrylov_args, is_setup) == MGR_HYPRE_SOLVER_IS_SETUP_OFFSET) ? 1 : -1];

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
typedef struct MGRSchwarzWrapper_struct
{
   /* Keep the first fields layout-compatible with hypre_Solver as validated
    * against the hypre 3.1 development stream used by the Schwarz branch. */
   HYPRE_PtrToSolverFcn    setup;
   HYPRE_PtrToSolverFcn    solve;
   MGRHyprePtrToDestroyFcn destroy;
   HYPRE_Int               is_setup;
   HYPRE_Solver            inner;
} MGRSchwarzWrapper;

typedef char MGRSchwarzWrapperLayoutCheck
   [(offsetof(MGRSchwarzWrapper, is_setup) == MGR_HYPRE_SOLVER_IS_SETUP_OFFSET) ? 1 : -1];
#endif

static MGRFRelaxWrapper *g_mgr_frelax_wrapper_live_head = NULL;

static const char *MGRCoarseSolverTypeName(const MGRcls_args *args);

/* GCOVR_EXCL_START */
static void
MGRFRelaxWrapperRegister(MGRFRelaxWrapper *wrapper)
{
   if (!wrapper)
   {
      return;
   }

   wrapper->next_live             = g_mgr_frelax_wrapper_live_head;
   g_mgr_frelax_wrapper_live_head = wrapper;
}

static void
MGRFRelaxWrapperUnregister(MGRFRelaxWrapper *wrapper)
{
   MGRFRelaxWrapper **cursor = &g_mgr_frelax_wrapper_live_head;

   while (*cursor)
   {
      if (*cursor == wrapper)
      {
         *cursor            = wrapper->next_live;
         wrapper->next_live = NULL;
         return;
      }
      cursor = &(*cursor)->next_live;
   }
}

static HYPRE_Int
MGRFRelaxWrapperSetup(void *wrapper_v, void *A, void *b, void *x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   if (HYPRE_MGRSetup((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                      (HYPRE_ParVector)b, (HYPRE_ParVector)x))
   {
      return 1;
   }
   wrapper->is_setup = 1;
   return 0;
}

static HYPRE_Int
MGRFRelaxWrapperSolve(void *wrapper_v, void *A, void *b, void *x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   /* hypre calls the setup callback for level-specific F-solvers during the
    * parent MGR setup. Avoid calling inner MGR setup again from solve because
    * repeated nested re-setup trips a cleanup bug in current hypre. */
   return HYPRE_MGRSolve((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRFRelaxWrapperDestroy(void *wrapper_v)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 0;
   }

   MGRFRelaxWrapperUnregister(wrapper);
   hypredrv_IntArrayDestroy(&wrapper->owned_dofmap);
   free(wrapper);
   return 0;
}

static void
hypredrv_MGRSetFSolverAtLevel(HYPRE_Solver precon, HYPRE_Solver fsolver, HYPRE_Int level,
                              HYPRE_Int               f_relax_type,
                              HYPRE_PtrToParSolverFcn fine_grid_solver_solve,
                              HYPRE_PtrToParSolverFcn fine_grid_solver_setup)
{
   if (!precon || !fsolver)
   {
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   if (level == 0 && fine_grid_solver_solve && fine_grid_solver_setup)
   {
      (void)f_relax_type;
      HYPRE_MGRSetFSolver(precon, fine_grid_solver_solve, fine_grid_solver_setup,
                          fsolver);
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   if (level == 0 && f_relax_type == 32)
   {
      HYPRE_MGRSetFSolver(precon, HYPRE_ILUSolve, HYPRE_ILUSetup, fsolver);
      return;
   }
#endif

   (void)f_relax_type;
   (void)fine_grid_solver_solve;
   (void)fine_grid_solver_setup;
   HYPRE_MGRSetFSolverAtLevel(precon, fsolver, level);
#else
   (void)fine_grid_solver_solve;
   (void)fine_grid_solver_setup;
   (void)level;
   (void)f_relax_type;
#endif
   (void)fsolver;
   (void)precon;
}

static HYPRE_Solver
MGRNestedFRelaxWrapperCreate(HYPRE_Solver inner_mgr, IntArray *owned_dofmap)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate nested MGR F-relaxation wrapper");
      return NULL;
   }

   wrapper->setup        = MGRFRelaxWrapperSetup;
   wrapper->solve        = MGRFRelaxWrapperSolve;
   wrapper->destroy      = MGRFRelaxWrapperDestroy;
   wrapper->inner_mgr    = inner_mgr;
   wrapper->owned_dofmap = owned_dofmap;
   MGRFRelaxWrapperRegister(wrapper);
   return (HYPRE_Solver)wrapper;
}

int
hypredrv_MGRNestedFRelaxWrapperIsLive(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *cursor = g_mgr_frelax_wrapper_live_head;

   while (cursor)
   {
      if ((HYPRE_Solver)cursor == wrapper_solver)
      {
         return 1;
      }
      cursor = cursor->next_live;
   }

   return 0;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxWrapperGetInner(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_solver;
   return wrapper ? wrapper->inner_mgr : NULL;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxWrapperDetachInner(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_solver;
   HYPRE_Solver      inner   = NULL;

   if (!wrapper)
   {
      return NULL;
   }

   inner              = wrapper->inner_mgr;
   wrapper->inner_mgr = NULL;
   return inner;
}

void
hypredrv_MGRNestedFRelaxWrapperFree(HYPRE_Solver *wrapper_ptr)
{
   if (!wrapper_ptr || !*wrapper_ptr)
   {
      return;
   }
   MGRFRelaxWrapperDestroy((void *)(*wrapper_ptr));
   *wrapper_ptr = NULL;
}

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static HYPRE_Int
MGRSchwarzWrapperSetup(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                       HYPRE_Vector x)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner)
   {
      return 1;
   }

   if (HYPRE_SchwarzSetup((HYPRE_Solver)wrapper->inner, (HYPRE_ParCSRMatrix)A,
                          (HYPRE_ParVector)b, (HYPRE_ParVector)x))
   {
      return 1;
   }

   wrapper->is_setup = 1;
   return 0;
}

static HYPRE_Int
MGRSchwarzWrapperSolve(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                       HYPRE_Vector x)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner)
   {
      return 1;
   }

   if (!wrapper->is_setup && MGRSchwarzWrapperSetup(wrapper_v, A, b, x))
   {
      return 1;
   }

   return HYPRE_SchwarzSolve((HYPRE_Solver)wrapper->inner, (HYPRE_ParCSRMatrix)A,
                             (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRSchwarzWrapperDestroy(HYPRE_Solver wrapper_v)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 0;
   }

   if (wrapper->inner)
   {
      HYPRE_SchwarzDestroy(wrapper->inner);
      wrapper->inner = NULL;
   }
   free(wrapper);
   return 0;
}

static HYPRE_Int
MGRSchwarzWrapperParSetup(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSetup(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

static HYPRE_Int
MGRSchwarzWrapperParSolve(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                          HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSolve(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

static HYPRE_Solver
MGRSchwarzWrapperCreate(const Schwarz_args *args)
{
   HYPRE_Solver       inner   = NULL;
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR Schwarz solver wrapper");
      return NULL;
   }

   hypredrv_SchwarzCreate(args, &inner);
   if (hypredrv_ErrorCodeActive() || !inner)
   {
      free(wrapper);
      return NULL;
   }

   wrapper->setup   = MGRSchwarzWrapperSetup;
   wrapper->solve   = MGRSchwarzWrapperSolve;
   wrapper->destroy = MGRSchwarzWrapperDestroy;
   wrapper->inner   = inner;
   return (HYPRE_Solver)wrapper;
}
#endif
/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 * Field definitions using the type-setting wrappers
 *-----------------------------------------------------------------------------*/

/* Module-level DOF label map (set before MGR YAML parsing, may be NULL) */
static const DofLabelMap *g_dof_labels = NULL;

/*-----------------------------------------------------------------------------
 * hypredrv_MGRSetDofLabels
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRSetDofLabels(const DofLabelMap *labels)
{
   g_dof_labels = labels;
}

/*-----------------------------------------------------------------------------
 * MGRlvlFDofsSet
 *
 * Custom setter for f_dofs that resolves symbolic label names through
 * g_dof_labels when the value is not a plain integer array.
 *-----------------------------------------------------------------------------*/

/* Resolve a single token (already lowercased) into an integer DOF index.
 * Returns true on success, false (+ error code set) on failure. */
/* GCOVR_EXCL_BR_START */
static bool
MGRlvlResolveDofToken(const char *tok, StackIntArray *arr)
{
   char *end  = NULL;
   long  ival = strtol(tok, &end, 10);

   if (end != tok && *end == '\0')
   {
      /* Plain integer */
      if (arr->size < MAX_STACK_ARRAY_LENGTH)
      {
         arr->data[arr->size++] = (int)ival;
      }
      return true;
   }

   /* Symbolic label */
   if (!g_dof_labels)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("f_dofs: symbolic label used but no dof_labels defined "
                           "in linear_system");
      return false;
   }

   int val = hypredrv_DofLabelMapLookup(g_dof_labels, tok);
   if (val < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("f_dofs: unknown label '%s'", tok);
      return false;
   }

   if (arr->size < MAX_STACK_ARRAY_LENGTH)
   {
      arr->data[arr->size++] = val;
   }
   return true;
}

static void
MGRlvlFDofsSet(void *field, const YAMLnode *node)
{
   StackIntArray *arr = (StackIntArray *)field;
   arr->size          = 0;

   /* Block sequence form:
    *   f_dofs:
    *     - v_x
    *     - v_y
    * Each "-" child carries the token in its val (already lowercased). */
   if (node->children)
   {
      for (const YAMLnode *item                                     = node->children;
           item != NULL && arr->size < MAX_STACK_ARRAY_LENGTH; item = item->next)
      {
         if (!strcmp(item->key, "-") && !MGRlvlResolveDofToken(item->val, arr))
         {
            return;
         }
      }
      return;
   }

   /* Flow sequence form: [v_x, v_y] or [0, 1].
    * MGRlvlResolveDofToken handles both plain integers and symbolic labels. */
   /* GCOVR_EXCL_START */
   char *buf = strdup(node->mapped_val);
   if (!buf)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate temporary buffer for f_dofs");
      return;
   }
   /* GCOVR_EXCL_STOP */
   const char *tok = strtok(buf, "[], ");
   while (tok && arr->size < MAX_STACK_ARRAY_LENGTH)
   {
      if (!MGRlvlResolveDofToken(tok, arr))
      {
         free(buf);
         return;
      }
      tok = strtok(NULL, "[], ");
   }
   free(buf);
}
/* GCOVR_EXCL_BR_STOP */

/* Generate type-setting wrappers for union fields */
DEFINE_TYPED_SETTER(MGRclsAMGSetArgs, MGRcls_args, amg, 0, hypredrv_AMGSetArgs)
DEFINE_TYPED_SETTER(MGRclsILUSetArgs, MGRcls_args, ilu, 32, hypredrv_ILUSetArgs)
DEFINE_TYPED_SETTER(MGRclsFSAISetArgs, MGRcls_args, fsai, 33, hypredrv_FSAISetArgs)
DEFINE_TYPED_SETTER(MGRfrlxAMGSetArgs, MGRfrlx_args, amg, 2, hypredrv_AMGSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxILUSetArgs, MGRfrlx_args, ilu, 32, hypredrv_ILUSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxFSAISetArgs, MGRfrlx_args, fsai, 33, hypredrv_FSAISetArgs)
DEFINE_TYPED_SETTER(MGRgrlxAMGSetArgs, MGRgrlx_args, amg, 20, hypredrv_AMGSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxILUSetArgs, MGRgrlx_args, ilu, 16, hypredrv_ILUSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxFSAISetArgs, MGRgrlx_args, fsai, 33, hypredrv_FSAISetArgs)
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
DEFINE_TYPED_SETTER(MGRclsSchwarzSetArgs, MGRcls_args, schwarz, MGR_SOLVER_TYPE_SCHWARZ,
                    hypredrv_SchwarzSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxSchwarzSetArgs, MGRfrlx_args, schwarz, MGR_SOLVER_TYPE_SCHWARZ,
                    hypredrv_SchwarzSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxSchwarzSetArgs, MGRgrlx_args, schwarz, MGR_SOLVER_TYPE_SCHWARZ,
                    hypredrv_SchwarzSetArgs)
#endif
static void MGRfrlxMGRSetArgs(void *, const YAMLnode *);
static void MGRCycleSet(void *, const YAMLnode *);
void        hypredrv_MGRSetArgsFromYAML(void *, YAMLnode *);

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
#define MGRcls_SCHWARZ_FIELD(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, schwarz, MGRclsSchwarzSetArgs)
#define MGRfrlx_SCHWARZ_FIELD(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, schwarz, MGRfrlxSchwarzSetArgs)
#define MGRgrlx_SCHWARZ_FIELD(_prefix) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, schwarz, MGRgrlxSchwarzSetArgs)
#else
#define MGRcls_SCHWARZ_FIELD(_prefix)
#define MGRfrlx_SCHWARZ_FIELD(_prefix)
#define MGRgrlx_SCHWARZ_FIELD(_prefix)
#endif

#define MGRcls_FIELDS(_prefix)                                     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRclsAMGSetArgs)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRclsILUSetArgs)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, MGRclsFSAISetArgs)        \
   MGRcls_SCHWARZ_FIELD(_prefix)

#define MGRfrlx_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, MGRfrlxMGRSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRfrlxAMGSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRfrlxILUSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, MGRfrlxFSAISetArgs)             \
   MGRfrlx_SCHWARZ_FIELD(_prefix)

#define MGRgrlx_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRgrlxAMGSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRgrlxILUSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, MGRgrlxFSAISetArgs)             \
   MGRgrlx_SCHWARZ_FIELD(_prefix)

#define MGRlvl_FIELDS(_prefix)                                                  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_dofs, MGRlvlFDofsSet)                      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, hypredrv_FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_level_type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_relaxation, hypredrv_MGRfrlxSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, g_relaxation, hypredrv_MGRgrlxSetArgs)

#define MGR_CYCLE_FIELDS(_prefix) ADD_FIELD_OFFSET_ENTRY(_prefix, cycle, MGRCycleSet)

#define MGR_FIELDS(_prefix)                                                    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, non_c_to_f, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pmax, hypredrv_FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, hypredrv_FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relax_type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, nonglk_max_elmts, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, hypredrv_FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_th, hypredrv_FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsest_level, hypredrv_MGRclsSetArgs)     \
   MGR_CYCLE_FIELDS(_prefix)

#define MGRcls_NUM_FIELDS \
   (sizeof(MGRcls_field_offset_map) / sizeof(MGRcls_field_offset_map[0]))
#define MGRfrlx_NUM_FIELDS \
   (sizeof(MGRfrlx_field_offset_map) / sizeof(MGRfrlx_field_offset_map[0]))
#define MGRgrlx_NUM_FIELDS \
   (sizeof(MGRgrlx_field_offset_map) / sizeof(MGRgrlx_field_offset_map[0]))
#define MGRlvl_NUM_FIELDS \
   (sizeof(MGRlvl_field_offset_map) / sizeof(MGRlvl_field_offset_map[0]))
#define MGR_NUM_FIELDS (sizeof(MGR_field_offset_map) / sizeof(MGR_field_offset_map[0]))

/* Define the prefix list */
#define GENERATE_PREFIXED_LIST_MGR                                  \
   hypredrv_GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRcls)        \
      hypredrv_GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRfrlx)    \
         hypredrv_GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRgrlx) \
            GENERATE_PREFIXED_COMPONENTS(MGRlvl)

/* Generate all boilerplate (field maps, setters, YAML parsing, etc.) */
GENERATE_PREFIXED_LIST_MGR                             // LCOV_EXCL_LINE
hypredrv_GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGR) // LCOV_EXCL_LINE

   static void MGRCycleSet(void *field, const YAMLnode *node)
{
   MGR_args *args =
      (MGR_args *)((char *)field - offsetof(MGR_args, cycle)); /* field is args->cycle */
   const char *value = NULL;

   if (node)
   {
      value = node->mapped_val ? node->mapped_val : node->val;
   }

   if (!args || !value)
   {
      return;
   }

   if (!strcmp(value, "v") || !strcmp(value, "v(1,0)"))
   {
      args->cycle            = 1;
      args->cycle_smooth_pos = 1;
   }
   else if (!strcmp(value, "v(0,1)"))
   {
      args->cycle            = 1;
      args->cycle_smooth_pos = 2;
   }
   else if (!strcmp(value, "v(1,1)"))
   {
      args->cycle            = 1;
      args->cycle_smooth_pos = 3;
   }
   else if (!strcmp(value, "w") || !strcmp(value, "w(1,0)"))
   {
      args->cycle            = 2;
      args->cycle_smooth_pos = 1;
   }
   else if (!strcmp(value, "w(0,1)"))
   {
      args->cycle            = 2;
      args->cycle_smooth_pos = 2;
   }
   else if (!strcmp(value, "w(1,1)"))
   {
      args->cycle            = 2;
      args->cycle_smooth_pos = 3;
   }
   else
   {
      int  cycle = 0;
      char extra = '\0';
      if (sscanf(value, "%d%c", &cycle, &extra) == 1 && (cycle == 1 || cycle == 2))
      {
         args->cycle            = (HYPRE_Int)cycle;
         args->cycle_smooth_pos = 1;
         return;
      }

      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid MGR cycle '%s' (expected 1, 2, v, w, v(1,0), v(0,1), "
                           "v(1,1), w(1,0), w(0,1), or w(1,1))",
                           value);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
static bool
MGRIsNestedKrylovKey(const char *key)
{
   /* GCOVR_EXCL_START */
   if (!key)
   {
      return false;
   }

   char *tmp = hypredrv_StrTrim(strdup(key));
   if (!tmp)
   {
      return false;
   }
   /* GCOVR_EXCL_STOP */
   hypredrv_StrToLowerCase(tmp);
   bool is_valid =
      hypredrv_StrIntMapArrayDomainEntryExists(hypredrv_SolverGetValidTypeIntMap(), tmp);
   free(tmp);
   return is_valid;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static NestedKrylov_args *
MGRGetOrCreateNestedKrylov(NestedKrylov_args **ptr)
{
   /* GCOVR_EXCL_START */
   if (!ptr)
   {
      return NULL;
   }
   /* GCOVR_EXCL_STOP */

   if (!*ptr)
   {
      *ptr = (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
      if (*ptr)
      {
         hypredrv_NestedKrylovSetDefaultArgs(*ptr);
      }
   }

   return *ptr;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static MGR_args *
MGRGetOrCreateNestedMGR(MGR_args **ptr)
{
   /* GCOVR_EXCL_START */
   if (!ptr)
   {
      return NULL;
   }
   /* GCOVR_EXCL_STOP */

   if (!*ptr)
   {
      *ptr = (MGR_args *)malloc(sizeof(MGR_args));
      if (*ptr)
      {
         hypredrv_MGRSetDefaultArgs(*ptr);
      }
   }

   return *ptr;
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * Build a dense label-space mask for labels present in an IntArray dofmap.
 *
 * The dof labels carried in IntArray may be sparse (e.g., nested MGR projected
 * F-blocks preserving parent labels). This helper derives the label-space size
 * as `max_label + 1` and a presence mask over that space.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
static int
MGRBuildDofLabelPresenceMask(const IntArray *dofmap, size_t *label_space_size_out,
                             size_t     *num_present_labels_out,
                             HYPRE_Int **label_present_out)
{
   const int *labels      = NULL;
   size_t     num_labels  = 0;
   HYPRE_Int *label_mask  = NULL;
   int        max_label   = -1;
   size_t     present_cnt = 0;

   /* GCOVR_EXCL_START */
   if (!dofmap || !label_space_size_out || !label_present_out)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Invalid arguments while building MGR dof label presence mask");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   *label_space_size_out = 0;
   *label_present_out    = NULL;
   if (num_present_labels_out)
   {
      *num_present_labels_out = 0;
   }

   if (dofmap->g_unique_data && dofmap->g_unique_size > 0)
   {
      labels     = dofmap->g_unique_data;
      num_labels = dofmap->g_unique_size;
      for (size_t i = 1; i < num_labels; i++)
      {
         if (labels[i] <= labels[i - 1])
         {
            labels = NULL;
            break;
         }
      }
   }
   if (!labels)
   {
      if (dofmap->unique_data && dofmap->unique_size > 0)
      {
         labels     = dofmap->unique_data;
         num_labels = dofmap->unique_size;
      }
      else if (dofmap->data && dofmap->size > 0)
      {
         labels     = dofmap->data;
         num_labels = dofmap->size;
      }
      else
      {
         if (dofmap->g_unique_size > 0)
         {
            /* Fallback for distributed dofmaps that only provide the global
             * unique count without a usable label array. Treat that as a dense
             * [0, ..., g_unique_size-1] label space instead of rejecting MGR. */
            *label_space_size_out = dofmap->g_unique_size;
            label_mask = (HYPRE_Int *)calloc(*label_space_size_out, sizeof(HYPRE_Int));
            /* GCOVR_EXCL_START */
            if (!label_mask)
            {
               hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
               hypredrv_ErrorMsgAdd("Failed to allocate MGR dof label presence mask");
               return 0;
            }
            /* GCOVR_EXCL_STOP */
            for (size_t i = 0; i < *label_space_size_out; i++)
            {
               label_mask[i] = 1;
            }
            *label_present_out = label_mask;
            if (num_present_labels_out)
            {
               *num_present_labels_out = dofmap->g_unique_size;
            }
            return 1;
         }
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                            "MGR invalid dofmap: empty label data and g_unique_size=%zu",
                            dofmap ? dofmap->g_unique_size : 0U);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("MGR requires a non-empty dofmap");
         return 0;
      }
   }

   for (size_t i = 0; i < num_labels; i++)
   {
      if (labels[i] < 0)
      {
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                            "MGR invalid dofmap: negative label %d at index %zu",
                            labels[i], i);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid negative dof label %d in dofmap", labels[i]);
         return 0;
      }
      if (labels[i] > max_label)
      {
         max_label = labels[i];
      }
   }

   *label_space_size_out = (size_t)max_label + 1;
   label_mask            = (HYPRE_Int *)calloc(*label_space_size_out, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (!label_mask)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR dof label presence mask");
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   for (size_t i = 0; i < num_labels; i++)
   {
      int label = labels[i];
      if (!label_mask[label])
      {
         label_mask[label] = 1;
         present_cnt++;
      }
   }

   *label_present_out = label_mask;
   if (num_present_labels_out)
   {
      *num_present_labels_out = present_cnt;
   }
   return 1;
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * Build a projected dofmap for nested MGR F-relaxation.
 *
 * Nested MGR acts on the outer level's F-block. Preserve the parent's original
 * labels when filtering to the selected F-point rows so nested `f_dofs` continue to
 * refer to the same label values as the parent MGR.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
static IntArray *
MGRBuildProjectedFRelaxDofmap(const IntArray      *parent_dofmap,
                              const StackIntArray *parent_f_dofs)
{
   /* GCOVR_EXCL_START */
   if (!parent_dofmap || !parent_f_dofs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires a valid parent dofmap and parent f_dofs");
      return NULL;
   }

   if (parent_f_dofs->size == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires non-empty parent f_dofs");
      return NULL;
   }
   /* GCOVR_EXCL_STOP */

   size_t     parent_label_space = 0;
   size_t     nested_num_labels  = parent_f_dofs->size;
   size_t     nested_size        = 0;
   HYPRE_Int *parent_present     = NULL;
   HYPRE_Int *keep_label         = NULL;
   IntArray  *nested_dofmap      = NULL;
   int        ok                 = 0;

   if (!MGRBuildDofLabelPresenceMask(parent_dofmap, &parent_label_space, NULL,
                                     &parent_present))
   {
      /* GCOVR_EXCL_START */
      goto cleanup;
      /* GCOVR_EXCL_STOP */
   }

   keep_label = (HYPRE_Int *)calloc(parent_label_space, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (!keep_label)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate nested MGR dof label selection mask");
      goto cleanup;
   }
   /* GCOVR_EXCL_STOP */

   for (size_t i = 0; i < nested_num_labels; i++)
   {
      int label = parent_f_dofs->data[i];
      /* GCOVR_EXCL_START */
      if (label < 0 || (size_t)label >= parent_label_space)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid parent MGR f_dofs label %d for nested MGR (valid range: [0,%d])",
            label, (int)parent_label_space - 1);
         goto cleanup;
      }
      if (!parent_present[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Parent MGR f_dofs label %d is not present in parent dofmap", label);
         goto cleanup;
      }
      if (keep_label[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Duplicate parent MGR f_dofs label %d for nested MGR",
                              label);
         goto cleanup;
      }
      /* GCOVR_EXCL_STOP */
      keep_label[label] = 1;
   }

   /* Count filtered entries. Labels in parent_dofmap are bounded by parent_label_space
    * (guaranteed by MGRBuildDofLabelPresenceMask), so no range check is needed here. */
   for (size_t i = 0; i < parent_dofmap->size; i++)
   {
      if (keep_label[parent_dofmap->data[i]])
      {
         nested_size++;
      }
   }

   nested_dofmap = hypredrv_IntArrayCreate(nested_size);
   /* GCOVR_EXCL_START */
   if (!nested_dofmap || (nested_size > 0 && !nested_dofmap->data))
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate nested MGR projected dofmap");
      goto cleanup;
   }
   /* GCOVR_EXCL_STOP */

   for (size_t i = 0, j = 0; i < parent_dofmap->size; i++)
   {
      int label = parent_dofmap->data[i];
      if (keep_label[label])
      {
         nested_dofmap->data[j++] = label;
      }
   }

   /* unique_data and g_unique_data are both the sorted set of kept labels
    * (keep_label[i] == 1 iff label i is an F-dof that appears in nested_dofmap). */
   nested_dofmap->unique_size   = nested_num_labels;
   nested_dofmap->g_unique_size = nested_num_labels;
   if (nested_num_labels > 0)
   {
      nested_dofmap->unique_data   = (int *)malloc(nested_num_labels * sizeof(int));
      nested_dofmap->g_unique_data = (int *)malloc(nested_num_labels * sizeof(int));
      /* GCOVR_EXCL_START */
      if (!nested_dofmap->unique_data || !nested_dofmap->g_unique_data)
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate nested MGR dof label arrays");
         goto cleanup;
      }
      /* GCOVR_EXCL_STOP */
      for (size_t i = 0, j = 0; i < parent_label_space; i++)
      {
         if (keep_label[i])
         {
            nested_dofmap->unique_data[j]   = (int)i;
            nested_dofmap->g_unique_data[j] = (int)i;
            j++;
         }
      }
   }

   ok = 1;

cleanup:
   free(keep_label);
   free(parent_present);
   /* GCOVR_EXCL_START */
   if (!ok)
   {
      hypredrv_IntArrayDestroy(&nested_dofmap);
   }
   /* GCOVR_EXCL_STOP */
   return ok ? nested_dofmap : NULL;
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
/*-----------------------------------------------------------------------------
 * (Re)initialize the union storage of an MGR component when YAML parsing
 * switches its solver type. The AMG and ILU type codes differ per component
 * slot (coarsest/F/G); FSAI and Schwarz use the same code in every slot.
 * Union members alias the same storage, so only the union base is needed.
 *-----------------------------------------------------------------------------*/

static void
MGRUnionApplyTypeDefaults(void *union_base, HYPRE_Int type, HYPRE_Int old_type,
                          HYPRE_Int amg_type, HYPRE_Int ilu_type)
{
   if (!union_base || type == old_type)
   {
      return;
   }

   if (type == amg_type)
   {
      hypredrv_AMGSetDefaultArgs((AMG_args *)union_base);
   }
   else if (type == ilu_type)
   {
      hypredrv_ILUSetDefaultArgs((ILU_args *)union_base);
   }
   else if (type == 33)
   {
      hypredrv_FSAISetDefaultArgs((FSAI_args *)union_base);
   }
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_SchwarzSetDefaultArgs((Schwarz_args *)union_base);
   }
#endif
}

static void
MGRclsApplyTypeDefaults(void *vargs, HYPRE_Int old_type)
{
   MGRcls_args *args = (MGRcls_args *)vargs;

   MGRUnionApplyTypeDefaults(args ? (void *)&args->amg : NULL,
                             args ? args->type : old_type, old_type, 0, 32);
}

static void
MGRfrlxApplyTypeDefaults(void *vargs, HYPRE_Int old_type)
{
   MGRfrlx_args *args = (MGRfrlx_args *)vargs;

   MGRUnionApplyTypeDefaults(args ? (void *)&args->amg : NULL,
                             args ? args->type : old_type, old_type, 2, 32);
}

static void
MGRgrlxApplyTypeDefaults(void *vargs, HYPRE_Int old_type)
{
   MGRgrlx_args *args = (MGRgrlx_args *)vargs;

   MGRUnionApplyTypeDefaults(args ? (void *)&args->amg : NULL,
                             args ? args->type : old_type, old_type, 20, 16);
}

static const char *
MGRLogObjectName(const Stats *stats)
{
   if (!stats)
   {
      return NULL;
   }
   if (stats->object_name[0] != '\0')
   {
      return stats->object_name;
   }
   if (stats->runtime_object_id > 0)
   {
      static char buf[32];

      snprintf(buf, sizeof(buf), "obj-%d", stats->runtime_object_id);
      return buf;
   }
   return NULL;
}

static HYPRE_Int
MGRLevelInterpTypeCompat(HYPRE_Int interp_type, const Stats *stats, int next_ls_id,
                         HYPRE_Int level)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER == 30100 && HYPREDRV_HYPRE_DEVELOP_NUMBER == 0
   if (interp_type == 13 || interp_type == 14)
   {
      const char *interp_name = (interp_type == 13) ? "blk-rowsum" : "blk-absrowsum";

      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                         "MGR level %d prolongation '%s' is unsupported by Hypre v3.1.0; "
                         "falling back to 'blk-jacobi'",
                         (int)level, interp_name);
      return 12;
   }
#else
   (void)stats;
   (void)next_ls_id;
   (void)level;
#endif

   return interp_type;
}

static void
MGRComponentReuseSetDefaultArgs(MGRComponentReuse_args *reuse)
{
   memset(reuse, 0, sizeof(*reuse));
   hypredrv_PreconReuseSetDefaultArgs(&reuse->args);
}

static void
MGRComponentReuseDestroyArgs(MGRComponentReuse_args *reuse)
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
   HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id, "%s %s",
                      label, detail);
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}

/*-----------------------------------------------------------------------------
 * MGRclsSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRclsSetDefaultArgs(MGRcls_args *args)
{
   /* Default coarsest solver: let hypredrv_MGRCreate interpret type < 0 as "default AMG".
    */
   args->type       = -1;
   args->use_krylov = 0;
   args->krylov     = NULL;
   MGRComponentReuseSetDefaultArgs(&args->reuse);

   /* Initialize default AMG args (union storage). If user later selects ILU via YAML,
    * ILUSetArgs/ILUSetDefaultArgs will reinitialize the union storage. */
   hypredrv_AMGSetDefaultArgs(&args->amg);
}

/*-----------------------------------------------------------------------------
 * MGRfrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRfrlxSetDefaultArgs(MGRfrlx_args *args)
{
   args->type       = 7;
   args->num_sweeps = 1;
   args->use_krylov = 0;
   args->krylov     = NULL;
   args->mgr        = NULL;
   MGRComponentReuseSetDefaultArgs(&args->reuse);
   /* Solver-specific args live in a union. We only (re)initialize them if/when a
    * specific solver type is selected during YAML parsing. */
}

/*-----------------------------------------------------------------------------
 * MGRgrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRgrlxSetDefaultArgs(MGRgrlx_args *args)
{
   /* Default to "none" (disabled). If user selects a global smoother type via YAML
    * but omits num_sweeps, we want at least one sweep. */
   args->type       = -1;
   args->num_sweeps = 1;
   args->use_krylov = 0;
   args->krylov     = NULL;
   MGRComponentReuseSetDefaultArgs(&args->reuse);

   /* Initialize default AMG args (union storage). If user later selects ILU via YAML,
    * ILUSetArgs/ILUSetDefaultArgs will reinitialize the union storage. */
   hypredrv_AMGSetDefaultArgs(&args->amg);
}

/*-----------------------------------------------------------------------------
 * MGRlvlSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRlvlSetDefaultArgs(MGRlvl_args *args)
{
   args->f_dofs            = STACK_INTARRAY_CREATE();
   args->prolongation_type = 0;
   args->restriction_type  = 0;
   args->coarse_level_type = 0;

   hypredrv_MGRfrlxSetDefaultArgs(&args->f_relaxation);
   hypredrv_MGRgrlxSetDefaultArgs(&args->g_relaxation);
}

/*-----------------------------------------------------------------------------
 * MGRSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRSetDefaultArgs(MGR_args *args)
{
   args->dofmap           = NULL;
   args->max_iter         = 1;
   args->num_levels       = 0;
   args->print_level      = 0;
   args->non_c_to_f       = 1;
   args->pmax             = 0;
   args->nonglk_max_elmts = 1;
   args->tolerance        = 0.0;
   args->coarse_th        = 0.0;
   args->relax_type       = 7;
   args->cycle            = 1;
   args->cycle_smooth_pos = 1;

   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      hypredrv_MGRlvlSetDefaultArgs(&args->level[i]);
      args->frelax[i]      = NULL;
      args->grelax[i]      = NULL;
      args->keep_frelax[i] = 0;
      args->keep_grelax[i] = 0;
   }
   hypredrv_MGRclsSetDefaultArgs(&args->coarsest_level);
   args->csolver           = NULL;
   args->csolver_type      = -1;
   args->keep_csolver      = 0;
   args->num_active_levels = 0;
   memset(args->active_level_map, 0, sizeof(args->active_level_map));
   args->vec_nn            = NULL;
   args->point_marker_data = NULL;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_START */
static void
MGRfrlxMGRSetArgs(void *field, const YAMLnode *node)
{
   MGRfrlx_args *parent = (MGRfrlx_args *)((char *)field - offsetof(MGRfrlx_args, mgr));
   MGR_args    **nested_ptr = (MGR_args **)field;
   MGR_args     *nested_mgr = MGRGetOrCreateNestedMGR(nested_ptr);

   if (!nested_mgr)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd(
         "Failed to allocate nested MGR arguments for MGR f_relaxation");
      return;
   }

   parent->type = MGR_FRLX_TYPE_NESTED_MGR;
   hypredrv_MGRSetArgsFromYAML(nested_mgr, (YAMLnode *)node);
}
/* GCOVR_EXCL_STOP */

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
MGRComponentReuseSetArgsFromYAML(MGRComponentReuse_args *reuse, YAMLnode *node)
{
   MGRComponentReuseDestroyArgs(reuse);
   MGRComponentReuseSetDefaultArgs(reuse);
   reuse->present = 1;
   hypredrv_PreconReuseSetArgsFromYAML(&reuse->args, node);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
/*-----------------------------------------------------------------------------
 * Shared YAML parsing for the per-component argument structs (coarsest level,
 * F-relaxation, G-relaxation). The three structs follow the same parsing
 * rules but differ in field layout, key/value tables, and how selecting a
 * nested Krylov solver affects the component type.
 *-----------------------------------------------------------------------------*/

typedef struct
{
   StrArray (*get_valid_keys)(void);
   StrIntMapArray (*get_valid_values)(const char *);
   void (*set_field_by_name)(void *, const YAMLnode *);
   void (*apply_type_defaults)(void *, HYPRE_Int);
   void (*krylov_selected)(void *); /* optional type adjustment, may be NULL */
   size_t type_offset;
   size_t reuse_offset;
   size_t use_krylov_offset;
   size_t krylov_offset;
} MGRComponentParseOps;

static void
MGRclsKrylovSelected(void *vargs)
{
   ((MGRcls_args *)vargs)->type = -1;
}

static void
MGRgrlxKrylovSelected(void *vargs)
{
   MGRgrlx_args *args = (MGRgrlx_args *)vargs;

   if (args->type < 0)
   {
      args->type = 0;
   }
}

static void
MGRComponentSetArgsFromYAML(const MGRComponentParseOps *ops, void *vargs,
                            YAMLnode *parent)
{
   if (!parent)
   {
      return;
   }

   HYPRE_Int *type = (HYPRE_Int *)((char *)vargs + ops->type_offset);

   if (!parent->children)
   {
      /* Flat form, e.g. "coarsest_level: amg": parse the scalar as the
       * component type by temporarily renaming the node key to "type". */
      char     *saved_key = parent->key;
      HYPRE_Int old_type  = *type;

      parent->key = strdup("type");
      YAML_NODE_VALIDATE(parent, ops->get_valid_keys, ops->get_valid_values);
      YAML_NODE_SET_FIELD(parent, vargs, ops->set_field_by_name);
      ops->apply_type_defaults(vargs, old_type);
      free(parent->key);
      parent->key = saved_key;
      return;
   }

   for (YAMLnode *child = parent->children; child != NULL; child = child->next)
   {
      if (!strcmp(child->key, "reuse"))
      {
         MGRComponentReuseSetArgsFromYAML(
            (MGRComponentReuse_args *)((char *)vargs + ops->reuse_offset), child);
         if (!hypredrv_ErrorCodeGet())
         {
            YAML_NODE_SET_VALID(child);
         }
         continue;
      }

      if (MGRIsNestedKrylovKey(child->key))
      {
         YAML_NODE_SET_VALID(child);
         *(int *)((char *)vargs + ops->use_krylov_offset) = 1;
         if (ops->krylov_selected)
         {
            ops->krylov_selected(vargs);
         }
         NestedKrylov_args *krylov = MGRGetOrCreateNestedKrylov(
            (NestedKrylov_args **)((char *)vargs + ops->krylov_offset));
         if (krylov)
         {
            hypredrv_NestedKrylovSetArgsFromYAML(krylov, child);
         }
         continue;
      }

      HYPRE_Int old_type = *type;
      YAML_NODE_VALIDATE(child, ops->get_valid_keys, ops->get_valid_values);
      YAML_NODE_SET_FIELD(child, vargs, ops->set_field_by_name);
      if (!strcmp(child->key, "type"))
      {
         ops->apply_type_defaults(vargs, old_type);
      }
   }
}

void
hypredrv_MGRclsSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   static const MGRComponentParseOps cls_parse_ops = {
      .get_valid_keys      = hypredrv_MGRclsGetValidKeys,
      .get_valid_values    = hypredrv_MGRclsGetValidValues,
      .set_field_by_name   = hypredrv_MGRclsSetFieldByName,
      .apply_type_defaults = MGRclsApplyTypeDefaults,
      .krylov_selected     = MGRclsKrylovSelected,
      .type_offset         = offsetof(MGRcls_args, type),
      .reuse_offset        = offsetof(MGRcls_args, reuse),
      .use_krylov_offset   = offsetof(MGRcls_args, use_krylov),
      .krylov_offset       = offsetof(MGRcls_args, krylov),
   };

   MGRComponentSetArgsFromYAML(&cls_parse_ops, vargs, parent);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRfrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   static const MGRComponentParseOps frlx_parse_ops = {
      .get_valid_keys      = hypredrv_MGRfrlxGetValidKeys,
      .get_valid_values    = hypredrv_MGRfrlxGetValidValues,
      .set_field_by_name   = hypredrv_MGRfrlxSetFieldByName,
      .apply_type_defaults = MGRfrlxApplyTypeDefaults,
      .krylov_selected     = NULL,
      .type_offset         = offsetof(MGRfrlx_args, type),
      .reuse_offset        = offsetof(MGRfrlx_args, reuse),
      .use_krylov_offset   = offsetof(MGRfrlx_args, use_krylov),
      .krylov_offset       = offsetof(MGRfrlx_args, krylov),
   };

   MGRComponentSetArgsFromYAML(&frlx_parse_ops, vargs, parent);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRgrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   static const MGRComponentParseOps grlx_parse_ops = {
      .get_valid_keys      = hypredrv_MGRgrlxGetValidKeys,
      .get_valid_values    = hypredrv_MGRgrlxGetValidValues,
      .set_field_by_name   = hypredrv_MGRgrlxSetFieldByName,
      .apply_type_defaults = MGRgrlxApplyTypeDefaults,
      .krylov_selected     = MGRgrlxKrylovSelected,
      .type_offset         = offsetof(MGRgrlx_args, type),
      .reuse_offset        = offsetof(MGRgrlx_args, reuse),
      .use_krylov_offset   = offsetof(MGRgrlx_args, use_krylov),
      .krylov_offset       = offsetof(MGRgrlx_args, krylov),
   };

   MGRComponentSetArgsFromYAML(&grlx_parse_ops, vargs, parent);
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * MGRclsGetValidValues
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
StrIntMapArray
hypredrv_MGRclsGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      static StrIntMap map[] = {
         {"def", -1}, {"amg", 0},   {"spdirect", 29},
         {"ilu", 32}, {"fsai", 33}, {"schwarz", MGR_SOLVER_TYPE_SCHWARZ},
      };
#else
      static StrIntMap map[] = {
         {"def", -1}, {"amg", 0}, {"spdirect", 29}, {"ilu", 32}, {"fsai", 33},
      };
#endif

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRfrlxGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_MGRfrlxGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      static StrIntMap map[] = {
         {"", -1},          {"none", -1},
         {"single", 7},     {"jacobi", 7},
         {"l1-jacobi", 18}, {"v(1,0)", 1},
         {"amg", 2},        {"mgr", MGR_FRLX_TYPE_NESTED_MGR},
         {"chebyshev", 16}, {"ilu", 32},
         {"ge", 9},         {"spdirect", 29},
         {"ge-piv", 99},    {"ge-inv", 199},
         {"fsai", 33},      {"schwarz", MGR_SOLVER_TYPE_SCHWARZ},
      };
#else
      static StrIntMap map[] = {
         {"", -1},          {"none", -1},
         {"single", 7},     {"jacobi", 7},
         {"l1-jacobi", 18}, {"v(1,0)", 1},
         {"amg", 2},        {"mgr", MGR_FRLX_TYPE_NESTED_MGR},
         {"chebyshev", 16}, {"ilu", 32},
         {"ge", 9},         {"spdirect", 29},
         {"ge-piv", 99},    {"ge-inv", 199},
         {"fsai", 33},
      };
#endif

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRgrlxGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_MGRgrlxGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      static StrIntMap map[] = {
         {"", -1},          {"none", -1},
         {"blk-jacobi", 0}, {"blk-gs", 1},
         {"mixed-gs", 2},   {"amg", 20},
         {"h-fgs", 3},      {"h-bgs", 4},
         {"ch-gs", 5},      {"h-ssor", 6},
         {"euclid", 8},     {"2stg-fgs", 11},
         {"2stg-bgs", 12},  {"l1-hfgs", 13},
         {"l1-hbgs", 14},   {"ilu", 16},
         {"spdirect", 29},  {"l1-hsgs", 88},
         {"fsai", 33},      {"schwarz", MGR_SOLVER_TYPE_SCHWARZ},
      };
#else
      static StrIntMap map[] = {
         {"", -1},         {"none", -1},    {"blk-jacobi", 0}, {"blk-gs", 1},
         {"mixed-gs", 2},  {"amg", 20},     {"h-fgs", 3},      {"h-bgs", 4},
         {"ch-gs", 5},     {"h-ssor", 6},   {"euclid", 8},     {"2stg-fgs", 11},
         {"2stg-bgs", 12}, {"l1-hfgs", 13}, {"l1-hbgs", 14},   {"ilu", 16},
         {"spdirect", 29}, {"l1-hsgs", 88}, {"fsai", 33},
      };
#endif

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRlvlGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_MGRlvlGetValidValues(const char *key)
{
   if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {
         {"injection", 0},     {"l1-jacobi", 1},   {"jacobi", 2},
         {"classical-mod", 3}, {"approx-inv", 4},  {"blk-jacobi", 12},
         {"blk-rowlump", 13},  {"blk-rowsum", 13}, {"blk-absrowsum", 14},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "restriction_type"))
   {
      static StrIntMap map[] = {
         {"injection", 0},   {"jacobi", 2},    {"approx-inv", 3},
         {"blk-jacobi", 12}, {"cpr-like", 13}, {"columped", 14},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "coarse_level_type"))
   {
      static StrIntMap map[] = {
         {"rap", 0},           {"galerkin", 0},       {"non-galerkin", 1},
         {"cpr-like-diag", 2}, {"cpr-like-bdiag", 3}, {"approx-inv", 4},
         {"acc", 5},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "f_relaxation"))
   {
      return hypredrv_MGRfrlxGetValidValues("type");
   }
   if (!strcmp(key, "g_relaxation"))
   {
      return hypredrv_MGRgrlxGetValidValues("type");
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * MGRGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_MGRGetValidValues(const char *key)
{
   if (!strcmp(key, "relax_type"))
   {
      static StrIntMap map[] = {
         {"jacobi", 7},     {"h-fgs", 3},      {"h-bgs", 4},   {"ch-gs", 5},
         {"h-ssor", 6},     {"hl1-ssor", 8},   {"l1-fgs", 13}, {"l1-bgs", 14},
         {"chebyshev", 16}, {"l1-jacobi", 18},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * MGRSetArgsFromYAML
 *
 * Parses MGR preconditioner arguments from a YAML tree. Handles the following
 * structure:
 *
 *   mgr:
 *     print_level: 1
 *     level:
 *       0:
 *         f_dofs: [2]
 *         f_relaxation: jacobi          # flat value -> type
 *         g_relaxation:                 # or nested block
 *           ilu:
 *             print_level: 1
 *       1:
 *         f_dofs: [1]
 *         ...
 *     coarsest_level:
 *       ilu:                            # nested solver block
 *         type: bj-ilut
 *         droptol: 1e-4
 *
 * Key insight: When a key like "ilu" or "amg" has children, its `val` field
 * is empty - the actual solver type is determined by the key name, not the val.
 *-----------------------------------------------------------------------------*/

/* GCOVR_EXCL_BR_START */
void
hypredrv_MGRSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   MGR_args *args = (MGR_args *)vargs;
   YAML_NODE_ITERATE(parent, child)
   {
      if (!strcmp(child->key, "level"))
      {
         YAML_NODE_SET_VALID(child);
         YAML_NODE_ITERATE(child, grandchild)
         {
            int lvl = (int)strtol(grandchild->key, NULL, 10);

            if (lvl >= 0 && lvl < MAX_MGR_LEVELS - 1)
            {
               YAML_NODE_ITERATE(grandchild, great_grandchild)
               {
                  if ((!strcmp(great_grandchild->key, "f_relaxation") ||
                       !strcmp(great_grandchild->key, "g_relaxation")) &&
                      great_grandchild->children &&
                      (MGRIsNestedKrylovKey(great_grandchild->children->key) ||
                       (!strcmp(great_grandchild->key, "f_relaxation") &&
                        !strcmp(great_grandchild->children->key, "mgr"))))
                  {
                     YAML_NODE_SET_VALID(great_grandchild);
                     YAML_NODE_SET_FIELD(great_grandchild, &args->level[lvl],
                                         hypredrv_MGRlvlSetFieldByName);
                     continue;
                  }

                  YAML_NODE_VALIDATE(great_grandchild, hypredrv_MGRlvlGetValidKeys,
                                     hypredrv_MGRlvlGetValidValues);

                  YAML_NODE_SET_FIELD(great_grandchild, &args->level[lvl],
                                      hypredrv_MGRlvlSetFieldByName);
               }

               args->num_levels++;
               YAML_NODE_SET_VALID(grandchild);
            }
            else
            {
               YAML_NODE_SET_INVALID_KEY(grandchild);
            }
         }
      }
      else if (!strcmp(child->key, "coarsest_level"))
      {
         args->num_levels++;
         YAML_NODE_SET_VALID(child);
         hypredrv_MGRclsSetArgsFromYAML(&args->coarsest_level, child);
      }
      else
      {
         YAML_NODE_VALIDATE(child, hypredrv_MGRGetValidKeys, hypredrv_MGRGetValidValues);
         YAML_NODE_SET_FIELD(child, args, hypredrv_MGRSetFieldByName);
      }
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_MGRConvertArgInt
 *-----------------------------------------------------------------------------*/

HYPRE_Int *
hypredrv_MGRConvertArgInt(MGR_args *args, const char *name)
{
   static HYPRE_Int buf[MAX_MGR_LEVELS - 1] = {-1};

   /* Sanity check */
   if (args->num_levels == 0 || args->num_levels >= (MAX_MGR_LEVELS - 1))
   {
      return NULL;
   }

   /* hypre MGR SetLevel* setters copy arrays up to their internal
    * max_num_coarse_levels. Always clear the full buffer so unused entries do
    * not retain stale values from previous conversions. */
   for (size_t i = 0; i < (size_t)(MAX_MGR_LEVELS - 1); i++)
   {
      buf[i] = 0;
   }

   if (!strcmp(name, "f_relaxation:type"))
   {
      for (size_t i = 0; i < (size_t)(args->num_levels - 1); i++)
      {
         HYPRE_Int type = args->level[i].f_relaxation.type;
         buf[i]         = (type == MGR_FRLX_TYPE_NESTED_MGR) ? 7 : type;
      }
      return buf;
   }

   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, prolongation_type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, f_relaxation.num_sweeps)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, g_relaxation.type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, g_relaxation.num_sweeps)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, restriction_type)
   HANDLE_MGR_LEVEL_ATTRIBUTE(buf, coarse_level_type)

   /* If we haven't returned yet, return a NULL pointer */
   return NULL;
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_MGRSetDofmap
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRSetDofmap(MGR_args *args, IntArray *dofmap)
{
   args->dofmap = dofmap;
}

void
hypredrv_MGRSetNearNullSpace(MGR_args *args, HYPRE_IJVector vec_nn)
{
   args->vec_nn = vec_nn;
}

/* GCOVR_EXCL_BR_START */
void
hypredrv_MGRDestroyNestedSolverArgs(MGR_args *args)
{
   if (!args)
   {
      return;
   }

   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      MGRComponentReuseDestroyArgs(&args->level[i].f_relaxation.reuse);
      MGRComponentReuseDestroyArgs(&args->level[i].g_relaxation.reuse);

      if (args->level[i].f_relaxation.mgr)
      {
         hypredrv_MGRDestroyNestedSolverArgs(args->level[i].f_relaxation.mgr);
         free(args->level[i].f_relaxation.mgr);
         args->level[i].f_relaxation.mgr = NULL;
      }

      if (args->level[i].f_relaxation.krylov)
      {
         hypredrv_NestedKrylovDestroy(args->level[i].f_relaxation.krylov);
         free(args->level[i].f_relaxation.krylov);
         args->level[i].f_relaxation.krylov     = NULL;
         args->level[i].f_relaxation.use_krylov = 0;
      }

      if (args->level[i].g_relaxation.krylov)
      {
         hypredrv_NestedKrylovDestroy(args->level[i].g_relaxation.krylov);
         free(args->level[i].g_relaxation.krylov);
         args->level[i].g_relaxation.krylov     = NULL;
         args->level[i].g_relaxation.use_krylov = 0;
      }
   }

   if (args->coarsest_level.krylov)
   {
      hypredrv_NestedKrylovDestroy(args->coarsest_level.krylov);
      free(args->coarsest_level.krylov);
      args->coarsest_level.krylov     = NULL;
      args->coarsest_level.use_krylov = 0;
   }

   MGRComponentReuseDestroyArgs(&args->coarsest_level.reuse);
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

static const char *
MGRCoarseSolverTypeName(const MGRcls_args *args)
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
#if HYPRE_CHECK_MIN_VERSION(30100, 38)
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
      HYPRE_BoomerAMGDestroy(*solver_ptr);
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
      MGRSchwarzWrapperDestroy(*solver_ptr);
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
      MGRSchwarzWrapperDestroy(*solver_ptr);
   }
#endif

   *solver_ptr = NULL;
}

static int
MGRRebuildNestedKrylovSolver(NestedKrylov_args *krylov, MGR_args *mgr_args)
{
   if (!krylov || !mgr_args)
   {
      return 0;
   }

   hypredrv_NestedKrylovDestroy(krylov);
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

static HYPRE_Solver
MGRFRelaxSolverCreateByType(MGRfrlx_args *f_relaxation, int active_lvl)
{
   HYPRE_Solver solver = NULL;

#if !HYPRE_CHECK_MIN_VERSION(30100, 55)
   (void)active_lvl;
#endif

   if (f_relaxation->type == 2)
   {
      hypredrv_AMGCreate(&f_relaxation->amg, &solver);
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
      solver = MGRSchwarzWrapperCreate(&f_relaxation->schwarz);
   }
#endif

   return solver;
}

static void
MGRFRelaxInstall(HYPRE_Solver precon, const MGRfrlx_args *f_relaxation,
                 HYPRE_Solver frelax, int active_lvl)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   if (f_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_MGRSetFSolverAtLevel(precon, frelax, active_lvl,
                                    MGR_FRLX_TYPE_CUSTOM_SOLVER_CB,
                                    MGRSchwarzWrapperParSolve, MGRSchwarzWrapperParSetup);
      return;
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   hypredrv_MGRSetFSolverAtLevel(precon, frelax, active_lvl, f_relaxation->type, NULL,
                                 NULL);
#elif HYPRE_CHECK_MIN_VERSION(21900, 0)
   /* Only the level-0 AMG F-solver slot is available on this hypre version. */
   if (f_relaxation->type == 2)
   {
      HYPRE_MGRSetFSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, frelax);
   }
   (void)active_lvl;
#else
   (void)precon;
   (void)f_relaxation;
   (void)frelax;
   (void)active_lvl;
#endif
}

static HYPRE_Solver
MGRGRelaxSolverCreateByType(MGRgrlx_args *g_relaxation)
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
      solver = MGRSchwarzWrapperCreate(&g_relaxation->schwarz);
   }
#endif

   return solver;
}

static HYPRE_Solver
MGRCoarseSolverCreateByType(MGRcls_args *coarsest_level, HYPRE_Int type)
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

static void
MGRCoarseSolverInstall(HYPRE_Solver mgr_solver, HYPRE_Int type,
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
 * clears use_default_cgrid_solver), so every type set by MGRCoarseSolverInstall
 * must be reclaimed here. */
static void
MGRCoarseSolverDestroyByType(HYPRE_Int type, HYPRE_Solver *solver_ptr)
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
      if (!MGRRebuildNestedKrylovSolver(level_args->f_relaxation.krylov, args))
      {
         return;
      }

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
      hypredrv_MGRSetFSolverAtLevel(
         mgr_solver, (HYPRE_Solver)level_args->f_relaxation.krylov, active_lvl,
         level_args->f_relaxation.type, MGRBaseParSolverSolve, MGRBaseParSolverSetup);
      MGRSetComponentSetupReuse((HYPRE_Solver)level_args->f_relaxation.krylov, 0);
#endif
      return;
   }

   HYPRE_Solver old_fsolver = args->frelax[orig_lvl];
   HYPRE_Solver fsolver =
      MGRFRelaxSolverCreateByType(&level_args->f_relaxation, active_lvl);

   if (hypredrv_ErrorCodeActive() || !fsolver)
   {
      return;
   }

   MGRFRelaxInstall(mgr_solver, &level_args->f_relaxation, fsolver, active_lvl);
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
      if (!MGRRebuildNestedKrylovSolver(level_args->g_relaxation.krylov, args))
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
   HYPRE_Solver smoother     = MGRGRelaxSolverCreateByType(&level_args->g_relaxation);

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
      if (!MGRRebuildNestedKrylovSolver(args->coarsest_level.krylov, args))
      {
         return;
      }

#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(mgr_solver, MGRBaseParSolverSolve, MGRBaseParSolverSetup,
                               (HYPRE_Solver)args->coarsest_level.krylov);
      MGRSetComponentSetupReuse((HYPRE_Solver)args->coarsest_level.krylov, 0);
#endif
      return;
   }

   HYPRE_Solver old_coarse_solver = args->csolver;
   HYPRE_Int    old_type          = args->csolver_type;
   HYPRE_Int    type              = MGRResolveCoarseSolverType(&args->coarsest_level);
   HYPRE_Solver coarse_solver = MGRCoarseSolverCreateByType(&args->coarsest_level, type);

   if (hypredrv_ErrorCodeActive() || !coarse_solver)
   {
      return;
   }

   MGRCoarseSolverInstall(mgr_solver, type, coarse_solver);
   MGRSetComponentSetupReuse(coarse_solver, 0);
   args->csolver_type = type;
   MGRCoarseSolverDestroyByType(old_type, &old_coarse_solver);
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
#if !HYPRE_CHECK_MIN_VERSION(30100, 38)
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

#if !HYPRE_CHECK_MIN_VERSION(30100, 38)
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
#if !HYPRE_CHECK_MIN_VERSION(30100, 38)
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
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "MGR coarsest setup reuse: reuse=%d", reuse_accepted);
      }
      else
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "MGR %s setup reuse at level %d: reuse=%d",
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
MGRPostDestroyNeedsUserSolverReclaim(void)
{
#if HYPRE_RELEASE_NUMBER_GT(30100) || \
   HYPRE_RELEASE_NUMBER_EQ_AND_DEVELOP_NUMBER_GE(30100, 5)
   return 1;
#else
   return 0;
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
      int max_levels = (args->num_levels > 0) ? (args->num_levels - 1) : 0;
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
      int max_levels = (args->num_levels > 0) ? (args->num_levels - 1) : 0;
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

   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
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
#if !HYPRE_CHECK_MIN_VERSION(30100, 38)
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

void
hypredrv_MGRDestroyCachedSolvers(MGR_args *args, int hypre_destroyed)
{
   if (!args)
   {
      return;
   }

   int destroy_handles = MGRDestroyCachedSolversExplicitly();
   /* Detached user-managed handles are always reclaimed before parent teardown.
    * After parent teardown, experimental builds reclaim dropped detached
    * handles explicitly. Standard builds keep the legacy first-active-level
    * fallback only where older hypre MGR teardowns still leave ownership to us. */
   int destroy_managed_detached =
      !hypre_destroyed || MGRPostDestroyNeedsUserSolverReclaim();
   int first_active_level =
      (args->num_active_levels > 0) ? (int)args->active_level_map[0] : -1;
   int legacy_grelax_reclaim = MGRLegacyPostDestroyNeedsGRelaxReclaim();
   int drop_csolver          = !destroy_handles || !args->keep_csolver;

   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      if (drop_csolver)
      {
         hypredrv_NestedKrylovDestroy(args->coarsest_level.krylov);
      }
   }
   else if (args->csolver && drop_csolver)
   {
      MGRCoarseSolverDestroyByType(args->csolver_type, &args->csolver);
   }
   if (drop_csolver)
   {
      args->csolver      = NULL;
      args->csolver_type = -1;
   }

   int max_levels = (args->num_levels > 0) ? (args->num_levels - 1) : 0;
   for (int i = 0; i < max_levels; i++)
   {
      int drop_frelax = !destroy_handles || !args->keep_frelax[i];
      int drop_grelax = !destroy_handles || !args->keep_grelax[i];

      if (args->level[i].f_relaxation.use_krylov && args->level[i].f_relaxation.krylov)
      {
         if (drop_frelax)
         {
            hypredrv_NestedKrylovDestroy(args->level[i].f_relaxation.krylov);
         }
      }
      else if (args->frelax[i] && drop_frelax)
      {
         if (destroy_managed_detached || destroy_handles ||
             (hypre_destroyed && i == first_active_level))
         {
            MGRDestroyDetachedFSolver(&args->level[i].f_relaxation, &args->frelax[i]);
         }
      }
      if (drop_frelax)
      {
         args->frelax[i] = NULL;
      }

      if (args->level[i].g_relaxation.use_krylov && args->level[i].g_relaxation.krylov)
      {
         if (drop_grelax)
         {
            hypredrv_NestedKrylovDestroy(args->level[i].g_relaxation.krylov);
         }
      }
      else if (args->grelax[i] && drop_grelax)
      {
         if (destroy_managed_detached || destroy_handles ||
             (hypre_destroyed && legacy_grelax_reclaim && i == first_active_level))
         {
            MGRDestroyDetachedGSolver(&args->level[i].g_relaxation, &args->grelax[i]);
         }
      }
      if (drop_grelax)
      {
         args->grelax[i] = NULL;
      }
   }

   MGRResetCachedSolverKeepFlags(args);
}

void
hypredrv_MGRForgetCachedSolvers(MGR_args *args)
{
   if (!args)
   {
      return;
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
static int
MGRPlanCoarsening(MGR_args *args, MGRCreatePlan *plan, const Stats *stats, int next_ls_id)
{
   IntArray *dofmap     = args->dofmap;
   HYPRE_Int num_levels = args->num_levels;
   HYPRE_Int num_dofs_last;
   HYPRE_Int lvl, i, j;

   args->num_active_levels = 0;
   memset(args->active_level_map, 0, sizeof(args->active_level_map));

   {
      size_t label_space_size = 0;
      size_t present_labels   = 0;
      if (!MGRBuildDofLabelPresenceMask(dofmap, &label_space_size, &present_labels,
                                        &plan->label_present))
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
   num_dofs_last       = plan->num_active_dofs;
   plan->inactive_dofs = (HYPRE_Int *)calloc((size_t)plan->num_dofs, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (plan->num_dofs > 0 && !plan->inactive_dofs)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR inactive dof label mask");
      return 0;
   }
   /* GCOVR_EXCL_STOP */
   for (i = 0; i < plan->num_dofs; i++)
   {
      if (!plan->label_present[i])
      {
         plan->inactive_dofs[i] = 1;
      }
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

      for (i = 0; i < (int)args->level[lvl].f_dofs.size; i++)
      {
         HYPRE_Int dof_label = args->level[lvl].f_dofs.data[i];
         if (dof_label < 0 || dof_label >= plan->num_dofs)
         {
            /* Distributed callers may provide a compact active dof label space
             * while configured MGR blocks still reference inactive labels from
             * the original global numbering. Treat those configured labels as
             * absent instead of invalid when global-unique metadata is present. */
            if (dof_label >= 0 && may_ignore_missing_f_dofs &&
                dof_label >= plan->num_dofs)
            {
               continue;
            }
            HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                               "MGR invalid f_dofs label: level=%d label=%d valid=[0,%d]",
                               (int)lvl, (int)dof_label, (int)plan->num_dofs - 1);
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid MGR level %d f_dofs label %d (valid range: [0,%d])", (int)lvl,
               (int)dof_label, (int)plan->num_dofs - 1);
            return 0;
         }
         if (!plan->label_present[dof_label])
         {
            /* Some configured blocks may have zero active dofs in the current
             * system. Ignore those labels instead of rejecting the MGR setup. */
            continue;
         }
         if (plan->inactive_dofs[dof_label])
         {
            HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                               "MGR duplicate/pruned f_dofs label: level=%d label=%d",
                               (int)lvl, (int)dof_label);
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Duplicate/previously eliminated MGR f_dofs label %d at level %d",
               (int)dof_label, (int)lvl);
            return 0;
         }
         plan->inactive_dofs[dof_label] = 1;
         ++num_level_f_dofs;
         --num_dofs_last;
      }

      if (num_level_f_dofs == 0)
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "MGR collapsing empty level: level=%d configured=%d",
                            (int)lvl, (int)args->level[lvl].f_dofs.size);
         free(plan->c_dofs[lvl]);
         plan->c_dofs[lvl]     = NULL;
         plan->num_c_dofs[lvl] = 0;
         continue;
      }

      plan->num_c_dofs[lvl] -= num_level_f_dofs;

      for (i = 0, j = 0; i < plan->num_dofs; i++)
      {
         if (plan->label_present[i] && !plan->inactive_dofs[i])
         {
            plan->c_dofs[lvl][j++] = i;
         }
      }
   }

   {
      HYPRE_Int active_levels   = 0;
      HYPRE_Int original_levels = num_levels - 1;

      for (lvl = 0; lvl < original_levels; lvl++)
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
      for (i = 0; i < MAX_MGR_LEVELS - 1; i++)
      {
         args->active_level_map[i] = (i < active_levels) ? plan->active_level_map[i] : 0;
      }
   }

   return 1;
}

/*-----------------------------------------------------------------------------
 * Remap sparse dof labels to a dense space and assemble the point-marker
 * array handed to hypre.
 *-----------------------------------------------------------------------------*/

static int
MGRPlanPointMarkers(MGR_args *args, MGRCreatePlan *plan, const Stats *stats,
                    int next_ls_id)
{
   IntArray *dofmap = args->dofmap;
   HYPRE_Int lvl, i, j;

   if (plan->num_active_dofs > 0 && plan->num_active_dofs < plan->num_dofs)
   {
      plan->label_to_dense =
         (HYPRE_Int *)malloc((size_t)plan->num_dofs * sizeof(HYPRE_Int));
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
               HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                                  "MGR invalid C-point label during dense remap: raw=%d "
                                  "num_dofs=%d mapped=%d",
                                  (int)raw, (int)plan->num_dofs,
                                  (raw >= 0 && raw < plan->num_dofs)
                                     ? (int)plan->label_to_dense[raw]
                                     : -1);
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               hypredrv_ErrorMsgAdd("Invalid MGR C-point label %d during dense remap",
                                    (int)raw);
               return 0;
            }
            /* GCOVR_EXCL_STOP */
            plan->c_dofs[lvl][i] = plan->label_to_dense[raw];
         }
      }
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after c-point assembly: code=0x%x num_dofs_hypre=%d",
                      hypredrv_ErrorCodeGet(), (int)plan->num_dofs_hypre);

   /* Set dofmap_data */
   if (!plan->label_to_dense && TYPES_MATCH(HYPRE_Int, int))
   {
      plan->dofmap_data = (HYPRE_Int *)dofmap->data;
   }
   else
   {
      plan->dofmap_data = (HYPRE_Int *)malloc(dofmap->size * sizeof(HYPRE_Int));
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
               HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                                  "MGR invalid dof label during dense remap: raw=%d "
                                  "num_dofs=%d mapped=%d",
                                  (int)raw, (int)plan->num_dofs,
                                  (raw >= 0 && raw < plan->num_dofs)
                                     ? (int)plan->label_to_dense[raw]
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
                     const Stats *stats, int next_ls_id)
{
   HYPRE_MGRSetCpointsByPointMarkerArray(precon, plan->num_dofs_hypre,
                                         plan->num_levels - 1, plan->num_c_dofs,
                                         plan->c_dofs, plan->dofmap_data);
   HYPRE_MGRSetNonCpointsToFpoints(precon, args->non_c_to_f);
   HYPRE_MGRSetPMaxElmts(precon, args->pmax);
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
         2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
         "MGR cycle setting is ignored because this hypre version does not support "
         "MGR cycle control APIs");
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22000, 0)
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
#endif
   HYPRE_MGRSetRelaxType(precon, args->relax_type); /* TODO: we shouldn't need this */
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
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
   HYPRE_Int level_frelax_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int level_frelax_sweeps[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int level_grelax_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int level_grelax_sweeps[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int level_interp_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int level_restrict_type[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int level_coarse_type[MAX_MGR_LEVELS - 1]   = {0};

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
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      if (level_args->g_relaxation.type == MGR_SOLVER_TYPE_SCHWARZ)
      {
         level_grelax_type[i] = MGR_GRLX_TYPE_USER_SMOOTHER;
      }
#endif
      level_grelax_sweeps[i] = level_args->g_relaxation.num_sweeps;
      level_interp_type[i]   = MGRLevelInterpTypeCompat(level_args->prolongation_type,
                                                        stats, next_ls_id, orig_lvl);
      level_restrict_type[i] = level_args->restriction_type;
      level_coarse_type[i]   = level_args->coarse_level_type;
   }

   HYPRE_MGRSetLevelFRelaxType(precon, level_frelax_type);
   HYPRE_MGRSetLevelNumRelaxSweeps(precon, level_frelax_sweeps);
   HYPRE_MGRSetLevelSmoothType(precon, level_grelax_type);
   HYPRE_MGRSetLevelSmoothIters(precon, level_grelax_sweeps);
   HYPRE_MGRSetLevelInterpType(precon, level_interp_type);
   HYPRE_MGRSetLevelRestrictType(precon, level_restrict_type);
   HYPRE_MGRSetCoarseGridMethod(precon, level_coarse_type);
#else
   (void)precon;
   (void)args;
   (void)plan;
   (void)stats;
   (void)next_ls_id;
#endif
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
      frelax = MGRFRelaxSolverCreateByType(&level_args->f_relaxation, active_lvl);
      if (hypredrv_ErrorCodeActive() || !frelax)
      {
         return 0;
      }
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR F-relax solver handle at level %d",
                         (int)orig_lvl);
   }

   MGRFRelaxInstall(precon, &level_args->f_relaxation, frelax, active_lvl);
   args->frelax[orig_lvl] = frelax;
   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the F-relaxation solver of each active MGR level.
 *-----------------------------------------------------------------------------*/

static int
MGRConfigFRelaxSolvers(MGR_args *args, HYPRE_Solver precon, const MGRCreatePlan *plan,
                       const Stats *stats, int next_ls_id)
{
   for (HYPRE_Int i = 0; i < plan->num_levels - 1; i++)
   {
      HYPRE_Int    orig_lvl   = plan->active_level_map[i];
      MGRlvl_args *level_args = &args->level[orig_lvl];

      if (level_args->f_relaxation.use_krylov && level_args->f_relaxation.krylov)
      {
         int krylov_was_cached = (level_args->f_relaxation.krylov->base_solver != NULL);
         if (!krylov_was_cached)
         {
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
            HYPREDRV_LOG_COMMF(
               2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
               "reusing cached MGR F-relax nested Krylov handle at level %d",
               (int)orig_lvl);
         }
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         hypredrv_MGRSetFSolverAtLevel(
            precon, (HYPRE_Solver)level_args->f_relaxation.krylov, i,
            level_args->f_relaxation.type, MGRBaseParSolverSolve, MGRBaseParSolverSetup);
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Nested Krylov F-relaxation requires hypre >= 2.31.0");
         return 0;
#endif
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
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
            return 0;
         }

         nested_dofmap = MGRBuildProjectedFRelaxDofmap(args->dofmap, &level_args->f_dofs);
         if (hypredrv_ErrorCodeActive() || !nested_dofmap)
         {
            hypredrv_IntArrayDestroy(&nested_dofmap);
            return 0;
         }

         saved_dofmap        = nested_args->dofmap;
         saved_vec_nn        = nested_args->vec_nn;
         nested_args->dofmap = nested_dofmap;
         nested_args->vec_nn = NULL;
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "creating nested MGR F-relaxation at level %d "
                            "(coarsest=%s)",
                            (int)orig_lvl,
                            MGRCoarseSolverTypeName(&nested_args->coarsest_level));
         hypredrv_MGRCreate(nested_args, &frelax, stats, next_ls_id);
         nested_args->dofmap = saved_dofmap;
         nested_args->vec_nn = saved_vec_nn;
         /* GCOVR_EXCL_START */
         if (hypredrv_ErrorCodeActive())
         {
            hypredrv_IntArrayDestroy(&nested_dofmap);
            return 0;
         }

         frelax_wrapper = MGRNestedFRelaxWrapperCreate(frelax, nested_dofmap);
         if (hypredrv_ErrorCodeActive() || !frelax_wrapper)
         {
            HYPRE_MGRDestroy(frelax);
            hypredrv_IntArrayDestroy(&nested_dofmap);
            return 0;
         }
         /* GCOVR_EXCL_STOP */
         nested_dofmap = NULL;
         hypredrv_MGRSetFSolverAtLevel(precon, frelax_wrapper, i,
                                       level_args->f_relaxation.type, NULL, NULL);
         args->frelax[orig_lvl] = frelax_wrapper;
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "Nested MGR F-relaxation requires hypre >= 3.1.0 (develop >= 5)");
         return 0;
#endif
      }
      else if (level_args->f_relaxation.type == 29)
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
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
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
      grelax = MGRGRelaxSolverCreateByType(&level_args->g_relaxation);
      if (hypredrv_ErrorCodeActive() || !grelax)
      {
         return 0;
      }
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
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
               2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
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
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after G-relax setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   return 1;
}

/*-----------------------------------------------------------------------------
 * Configure the coarsest-level solver (nested Krylov or a managed handle).
 *-----------------------------------------------------------------------------*/

static int
MGRConfigCoarsestSolver(MGR_args *args, HYPRE_Solver precon, const Stats *stats,
                        int next_ls_id)
{
   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      int krylov_was_cached = (args->coarsest_level.krylov->base_solver != NULL);
      if (!krylov_was_cached)
      {
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
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "reusing cached MGR coarsest nested Krylov handle");
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(precon, MGRBaseParSolverSolve, MGRBaseParSolverSetup,
                               (HYPRE_Solver)args->coarsest_level.krylov);
#else
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Nested Krylov coarsest solver requires hypre >= 3.1.0");
      return 0;
#endif
   }
   else
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

      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                         "MGR coarsest solver selected: %s",
                         MGRCoarseSolverTypeName(&args->coarsest_level));

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
         args->csolver = MGRCoarseSolverCreateByType(&args->coarsest_level, type);
         if (hypredrv_ErrorCodeActive() || !args->csolver)
         {
            return 0;
         }
      }
      else
      {
         HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                            "reusing cached MGR coarsest solver handle");
      }
      args->csolver_type = type;
      MGRCoarseSolverInstall(precon, type, args->csolver);
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, MGRLogObjectName(stats), next_ls_id,
                      "MGR stage after coarsest solver setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   return 1;
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
   HYPRE_Solver  precon = NULL;
   MGRCreatePlan plan   = {0};

   /* GCOVR_EXCL_BR_START */
   /* Sanity checks */
   if (!args->dofmap)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      return;
   }

   if (!MGRPlanCoarsening(args, &plan, stats, next_ls_id) ||
       !MGRPlanPointMarkers(args, &plan, stats, next_ls_id))
   {
      goto cleanup;
   }
   /* GCOVR_EXCL_BR_STOP */

   /* Config preconditioner */
   HYPRE_MGRCreate(&precon);
   MGRApplyBaseSettings(precon, args, &plan, stats, next_ls_id);
   /* GCOVR_EXCL_BR_START */
   MGRApplyLevelSettings(precon, args, &plan, stats, next_ls_id);

   if (!MGRConfigFRelaxSolvers(args, precon, &plan, stats, next_ls_id) ||
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
   /* Silence any hypre errors. TODO: improve error handling */
   HYPRE_ClearAllErrors();
#endif /* !HYPRE_CHECK_MIN_VERSION(21900, 0) */
}
