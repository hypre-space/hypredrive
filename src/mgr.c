/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/mgr.h"
#include <mpi.h>
/* gcovr: branch-exclusion regions below narrow branch-count noise from YAML
 * helpers and MGR validation/dispatch; single-line exclusions flag allocator
 * and defensive branches that are impractical to fault-inject here. */
#include "internal/error.h"
#include "internal/gen_macros.h"
#include "internal/krylov.h"
#include "internal/stats.h"
#include "logging.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 5)
#include "_hypre_utilities.h" // for hypre_Solver
#endif

typedef struct MGRFRelaxWrapper_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);
   HYPRE_Solver                    inner_mgr;
   IntArray                       *owned_dofmap;
   struct MGRFRelaxWrapper_struct *next_live;
} MGRFRelaxWrapper;

static MGRFRelaxWrapper *g_mgr_frelax_wrapper_live_head = NULL;

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
DEFINE_TYPED_SETTER(MGRfrlxAMGSetArgs, MGRfrlx_args, amg, 2, hypredrv_AMGSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxILUSetArgs, MGRfrlx_args, ilu, 32, hypredrv_ILUSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxAMGSetArgs, MGRgrlx_args, amg, 20, hypredrv_AMGSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxILUSetArgs, MGRgrlx_args, ilu, 16, hypredrv_ILUSetArgs)
static void MGRfrlxMGRSetArgs(void *, const YAMLnode *);
void        hypredrv_MGRSetArgsFromYAML(void *, YAMLnode *);

#define MGRcls_FIELDS(_prefix)                                     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRclsAMGSetArgs)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRclsILUSetArgs)

#define MGRfrlx_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, MGRfrlxMGRSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRfrlxAMGSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRfrlxILUSetArgs)

#define MGRgrlx_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRgrlxAMGSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRgrlxILUSetArgs)

#define MGRlvl_FIELDS(_prefix)                                                  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_dofs, MGRlvlFDofsSet)                      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, hypredrv_FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_level_type, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_relaxation, hypredrv_MGRfrlxSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, g_relaxation, hypredrv_MGRgrlxSetArgs)

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
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsest_level, hypredrv_MGRclsSetArgs)

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

   /*-----------------------------------------------------------------------------
    *-----------------------------------------------------------------------------*/

   /* GCOVR_EXCL_BR_START */
   static bool MGRIsNestedKrylovKey(const char *key)
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
      else if (dofmap->g_unique_size > 0)
      {
         /* Compatibility fallback for distributed dofmaps that only provide the
          * global unique count. Prefer explicit local label data when available,
          * because the dense [0, ..., g_unique_size-1] assumption can be wrong. */
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
      else
      {
         if (dofmap->g_unique_size > 0)
         {
            /* Defensive fallback: some callers provide only the global unique
             * count without a usable label array. Treat that as a dense
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
static void
MGRclsApplyTypeDefaults(MGRcls_args *args, HYPRE_Int old_type)
{
   /* GCOVR_EXCL_START */
   if (!args)
   {
      return;
   }
   /* GCOVR_EXCL_STOP */
   /* GCOVR_EXCL_START */
   if (args->type == old_type)
   {
      return;
   }
   /* GCOVR_EXCL_STOP */

   if (args->type == 0)
   {
      hypredrv_AMGSetDefaultArgs(&args->amg);
   }
   else if (args->type == 32)
   {
      hypredrv_ILUSetDefaultArgs(&args->ilu);
   }
}

static void
MGRfrlxApplyTypeDefaults(MGRfrlx_args *args, HYPRE_Int old_type)
{
   if (!args || args->type == old_type)
   {
      return;
   }

   if (args->type == 2)
   {
      hypredrv_AMGSetDefaultArgs(&args->amg);
   }
   else if (args->type == 32)
   {
      hypredrv_ILUSetDefaultArgs(&args->ilu);
   }
}

static void
MGRgrlxApplyTypeDefaults(MGRgrlx_args *args, HYPRE_Int old_type)
{
   if (!args || args->type == old_type)
   {
      return;
   }

   if (args->type == 20)
   {
      hypredrv_AMGSetDefaultArgs(&args->amg);
   }
   else if (args->type == 16)
   {
      hypredrv_ILUSetDefaultArgs(&args->ilu);
   }
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

   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      hypredrv_MGRlvlSetDefaultArgs(&args->level[i]);
      args->frelax[i] = NULL;
      args->grelax[i] = NULL;
   }
   hypredrv_MGRclsSetDefaultArgs(&args->coarsest_level);
   args->csolver           = NULL;
   args->csolver_type      = -1;
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

/* GCOVR_EXCL_BR_START */
void
hypredrv_MGRclsSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   MGRcls_args *args = (MGRcls_args *)vargs;

   if (!parent)
   {
      return;
   }

   if (parent->children)
   {
      for (YAMLnode *child = parent->children; child != NULL; child = child->next)
      {
         if (MGRIsNestedKrylovKey(child->key))
         {
            YAML_NODE_SET_VALID(child);
            args->use_krylov          = 1;
            args->type                = -1;
            NestedKrylov_args *krylov = MGRGetOrCreateNestedKrylov(&args->krylov);
            if (krylov)
            {
               hypredrv_NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         HYPRE_Int old_type = args->type;
         YAML_NODE_VALIDATE(child, hypredrv_MGRclsGetValidKeys,
                            hypredrv_MGRclsGetValidValues);
         YAML_NODE_SET_FIELD(child, args, hypredrv_MGRclsSetFieldByName);
         if (!strcmp(child->key, "type"))
         {
            MGRclsApplyTypeDefaults(args, old_type);
         }
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      HYPRE_Int old_type = args->type;
      YAML_NODE_VALIDATE(parent, hypredrv_MGRclsGetValidKeys,
                         hypredrv_MGRclsGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, hypredrv_MGRclsSetFieldByName);
      MGRclsApplyTypeDefaults(args, old_type);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRfrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   MGRfrlx_args *args = (MGRfrlx_args *)vargs;

   if (!parent)
   {
      return;
   }

   if (parent->children)
   {
      for (YAMLnode *child = parent->children; child != NULL; child = child->next)
      {
         if (MGRIsNestedKrylovKey(child->key))
         {
            YAML_NODE_SET_VALID(child);
            args->use_krylov          = 1;
            NestedKrylov_args *krylov = MGRGetOrCreateNestedKrylov(&args->krylov);
            if (krylov)
            {
               hypredrv_NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         HYPRE_Int old_type = args->type;
         YAML_NODE_VALIDATE(child, hypredrv_MGRfrlxGetValidKeys,
                            hypredrv_MGRfrlxGetValidValues);
         YAML_NODE_SET_FIELD(child, args, hypredrv_MGRfrlxSetFieldByName);
         if (!strcmp(child->key, "type"))
         {
            MGRfrlxApplyTypeDefaults(args, old_type);
         }
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      HYPRE_Int old_type = args->type;
      YAML_NODE_VALIDATE(parent, hypredrv_MGRfrlxGetValidKeys,
                         hypredrv_MGRfrlxGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, hypredrv_MGRfrlxSetFieldByName);
      MGRfrlxApplyTypeDefaults(args, old_type);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRgrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   MGRgrlx_args *args = (MGRgrlx_args *)vargs;

   if (!parent)
   {
      return;
   }

   if (parent->children)
   {
      for (YAMLnode *child = parent->children; child != NULL; child = child->next)
      {
         if (MGRIsNestedKrylovKey(child->key))
         {
            YAML_NODE_SET_VALID(child);
            args->use_krylov = 1;
            if (args->type < 0)
            {
               args->type = 0;
            }
            NestedKrylov_args *krylov = MGRGetOrCreateNestedKrylov(&args->krylov);
            if (krylov)
            {
               hypredrv_NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         HYPRE_Int old_type = args->type;
         YAML_NODE_VALIDATE(child, hypredrv_MGRgrlxGetValidKeys,
                            hypredrv_MGRgrlxGetValidValues);
         YAML_NODE_SET_FIELD(child, args, hypredrv_MGRgrlxSetFieldByName);
         if (!strcmp(child->key, "type"))
         {
            MGRgrlxApplyTypeDefaults(args, old_type);
         }
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      HYPRE_Int old_type = args->type;
      YAML_NODE_VALIDATE(parent, hypredrv_MGRgrlxGetValidKeys,
                         hypredrv_MGRgrlxGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, hypredrv_MGRgrlxSetFieldByName);
      MGRgrlxApplyTypeDefaults(args, old_type);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
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
      static StrIntMap map[] = {{"def", -1}, {"amg", 0}, {"spdirect", 29}, {"ilu", 32}};

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
      static StrIntMap map[] = {
         {"", -1},          {"none", -1},
         {"single", 7},     {"jacobi", 7},
         {"l1-jacobi", 18}, {"v(1,0)", 1},
         {"amg", 2},        {"mgr", MGR_FRLX_TYPE_NESTED_MGR},
         {"chebyshev", 16}, {"ilu", 32},
         {"ge", 9},         {"spdirect", 29},
         {"ge-piv", 99},    {"ge-inv", 199},
      };

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
      static StrIntMap map[] = {
         {"", -1},         {"none", -1},    {"blk-jacobi", 0}, {"blk-gs", 1},
         {"mixed-gs", 2},  {"amg", 20},     {"h-fgs", 3},      {"h-bgs", 4},
         {"ch-gs", 5},     {"h-ssor", 6},   {"euclid", 8},     {"2stg-fgs", 11},
         {"2stg-bgs", 12}, {"l1-hfgs", 13}, {"l1-hbgs", 14},   {"ilu", 16},
         {"l1-hsgs", 88},
      };

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
   if (args->num_levels >= (MAX_MGR_LEVELS - 1))
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
}
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_MGRCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRCreate(MGR_args *args, HYPRE_Solver *precon_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)args;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("MGR requires hypre >= 2.19.0");
   *precon_ptr = NULL;
   return;
#else
   HYPRE_Solver precon                                  = NULL;
   HYPRE_Solver frelax                                  = NULL;
   HYPRE_Solver grelax                                  = NULL;
   HYPRE_Int   *dofmap_data                             = NULL;
   HYPRE_Int   *dofmap_data_owned                       = NULL;
   IntArray    *dofmap                                  = NULL;
   HYPRE_Int   *label_present                           = NULL;
   HYPRE_Int   *label_to_dense                          = NULL;
   HYPRE_Int    num_dofs                                = 0;
   HYPRE_Int    num_dofs_hypre                          = 0;
   HYPRE_Int    num_active_dofs                         = 0;
   HYPRE_Int    num_dofs_last                           = 0;
   HYPRE_Int    num_levels                              = 0;
   HYPRE_Int    active_level_map[MAX_MGR_LEVELS - 1]    = {0};
   HYPRE_Int    level_frelax_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int    level_frelax_sweeps[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int    level_grelax_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int    level_grelax_sweeps[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int    level_interp_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int    level_restrict_type[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int    level_coarse_type[MAX_MGR_LEVELS - 1]   = {0};
   HYPRE_Int    num_c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int   *c_dofs[MAX_MGR_LEVELS - 1] = {0};
   HYPRE_Int   *inactive_dofs              = NULL;
   int          may_ignore_missing_f_dofs  = 0;
   HYPRE_Int    lvl = 0, i = 0;
   HYPRE_Int    j;

   /* GCOVR_EXCL_BR_START */
   /* Sanity checks */
   if (!args->dofmap)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      return;
   }

   /* Initialize variables */
   dofmap     = args->dofmap;
   num_levels = args->num_levels;
   {
      size_t label_space_size = 0;
      size_t present_labels   = 0;
      if (!MGRBuildDofLabelPresenceMask(dofmap, &label_space_size, &present_labels,
                                        &label_present))
      {
         return;
      }
      num_dofs        = (HYPRE_Int)label_space_size;
      num_dofs_hypre  = num_dofs;
      num_active_dofs = (HYPRE_Int)present_labels;
   }
   may_ignore_missing_f_dofs = (dofmap->g_unique_size > 0 && num_active_dofs == num_dofs);

   /* Compute num_c_dofs and c_dofs */
   num_dofs_last = num_active_dofs;
   inactive_dofs = (HYPRE_Int *)calloc((size_t)num_dofs, sizeof(HYPRE_Int));
   /* GCOVR_EXCL_START */
   if (!inactive_dofs)
   {
      free(label_present);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR inactive dof label mask");
      return;
   }
   /* GCOVR_EXCL_STOP */
   for (i = 0; i < num_dofs; i++)
   {
      if (!label_present[i])
      {
         inactive_dofs[i] = 1;
      }
   }
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      HYPRE_Int num_level_f_dofs = 0;

      c_dofs[lvl]     = (HYPRE_Int *)calloc((size_t)num_dofs, sizeof(HYPRE_Int));
      num_c_dofs[lvl] = num_dofs_last;

      for (i = 0; i < (int)args->level[lvl].f_dofs.size; i++)
      {
         HYPRE_Int dof_label = args->level[lvl].f_dofs.data[i];
         if (dof_label < 0 || dof_label >= num_dofs)
         {
            /* Distributed callers may provide a compact active dof label space
             * while configured MGR blocks still reference inactive labels from
             * the original global numbering. Treat those configured labels as
             * absent instead of invalid when global-unique metadata is present. */
            if (dof_label >= 0 && may_ignore_missing_f_dofs && dof_label >= num_dofs)
            {
               continue;
            }
            HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                               "MGR invalid f_dofs label: level=%d label=%d valid=[0,%d]",
                               (int)lvl, (int)dof_label, (int)num_dofs - 1);
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Invalid MGR level %d f_dofs label %d (valid range: [0,%d])", (int)lvl,
               (int)dof_label, (int)num_dofs - 1);
            free(label_present);
            free(inactive_dofs);
            for (HYPRE_Int k = 0; k <= lvl; k++)
            {
               free(c_dofs[k]);
            }
            return;
         }
         if (!label_present[dof_label])
         {
            /* Some configured blocks may have zero active dofs in the current
             * system. Ignore those labels instead of rejecting the MGR setup. */
            continue;
         }
         if (inactive_dofs[dof_label])
         {
            HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                               "MGR duplicate/pruned f_dofs label: level=%d label=%d",
                               (int)lvl, (int)dof_label);
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Duplicate/previously eliminated MGR f_dofs label %d at level %d",
               (int)dof_label, (int)lvl);
            free(label_present);
            free(inactive_dofs);
            for (HYPRE_Int k = 0; k <= lvl; k++)
            {
               free(c_dofs[k]);
            }
            return;
         }
         inactive_dofs[dof_label] = 1;
         ++num_level_f_dofs;
         --num_dofs_last;
      }

      if (num_level_f_dofs == 0)
      {
         HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                            "MGR collapsing empty level: level=%d configured=%d",
                            (int)lvl, (int)args->level[lvl].f_dofs.size);
         free(c_dofs[lvl]);
         c_dofs[lvl]     = NULL;
         num_c_dofs[lvl] = 0;
         continue;
      }

      num_c_dofs[lvl] -= num_level_f_dofs;

      for (i = 0, j = 0; i < num_dofs; i++)
      {
         if (label_present[i] && !inactive_dofs[i])
         {
            c_dofs[lvl][j++] = i;
         }
      }
   }

   {
      HYPRE_Int active_levels   = 0;
      HYPRE_Int original_levels = num_levels - 1;

      for (lvl = 0; lvl < original_levels; lvl++)
      {
         if (!c_dofs[lvl])
         {
            continue;
         }

         active_level_map[active_levels] = lvl;
         if (active_levels != lvl)
         {
            c_dofs[active_levels]     = c_dofs[lvl];
            num_c_dofs[active_levels] = num_c_dofs[lvl];
            c_dofs[lvl]               = NULL;
            num_c_dofs[lvl]           = 0;
         }
         active_levels++;
      }

      num_levels = active_levels + 1;
   }

   if (num_active_dofs > 0 && num_active_dofs < num_dofs)
   {
      label_to_dense = (HYPRE_Int *)malloc((size_t)num_dofs * sizeof(HYPRE_Int));
      /* GCOVR_EXCL_START */
      if (!label_to_dense)
      {
         free(label_present);
         free(inactive_dofs);
         for (HYPRE_Int k = 0; k < num_levels - 1; k++)
         {
            free(c_dofs[k]);
         }
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate MGR dense label remap");
         return;
      }
      /* GCOVR_EXCL_STOP */

      for (i = 0; i < num_dofs; i++)
      {
         label_to_dense[i] = -1;
      }
      for (i = 0, j = 0; i < num_dofs; i++)
      {
         if (label_present[i])
         {
            label_to_dense[i] = j++;
         }
      }
      num_dofs_hypre = num_active_dofs;

      for (lvl = 0; lvl < num_levels - 1; lvl++)
      {
         for (i = 0; i < num_c_dofs[lvl]; i++)
         {
            HYPRE_Int raw = c_dofs[lvl][i];
            /* GCOVR_EXCL_START */
            if (raw < 0 || raw >= num_dofs || label_to_dense[raw] < 0)
            {
               HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                                  "MGR invalid C-point label during dense remap: raw=%d "
                                  "num_dofs=%d mapped=%d",
                                  (int)raw, (int)num_dofs,
                                  (raw >= 0 && raw < num_dofs) ? (int)label_to_dense[raw]
                                                               : -1);
               free(label_to_dense);
               free(label_present);
               free(inactive_dofs);
               for (HYPRE_Int k = 0; k < num_levels - 1; k++)
               {
                  free(c_dofs[k]);
               }
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               hypredrv_ErrorMsgAdd("Invalid MGR C-point label %d during dense remap",
                                    (int)raw);
               return;
            }
            /* GCOVR_EXCL_STOP */
            c_dofs[lvl][i] = label_to_dense[raw];
         }
      }
   }
   free(label_present);
   label_present = NULL;
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                      "MGR stage after c-point assembly: code=0x%x num_dofs_hypre=%d",
                      hypredrv_ErrorCodeGet(), (int)num_dofs_hypre);

   /* Set dofmap_data */
   if (!label_to_dense && TYPES_MATCH(HYPRE_Int, int))
   {
      dofmap_data = (HYPRE_Int *)dofmap->data;
   }
   else
   {
      dofmap_data = (HYPRE_Int *)malloc(dofmap->size * sizeof(HYPRE_Int));
      /* GCOVR_EXCL_START */
      if (!dofmap_data)
      {
         free(inactive_dofs);
         for (lvl = 0; lvl < num_levels - 1; lvl++)
         {
            free(c_dofs[lvl]);
         }
         free(label_to_dense);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate MGR point-marker array");
         return;
      }
      /* GCOVR_EXCL_STOP */
      dofmap_data_owned = dofmap_data;
      for (i = 0; i < (int)dofmap->size; i++)
      {
         HYPRE_Int raw = (HYPRE_Int)dofmap->data[i];
         if (label_to_dense)
         {
            /* GCOVR_EXCL_START */
            if (raw < 0 || raw >= num_dofs || label_to_dense[raw] < 0)
            {
               HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                                  "MGR invalid dof label during dense remap: raw=%d "
                                  "num_dofs=%d mapped=%d",
                                  (int)raw, (int)num_dofs,
                                  (raw >= 0 && raw < num_dofs) ? (int)label_to_dense[raw]
                                                               : -1);
               free(inactive_dofs);
               for (lvl = 0; lvl < num_levels - 1; lvl++)
               {
                  free(c_dofs[lvl]);
               }
               free(label_to_dense);
               free(dofmap_data_owned);
               dofmap_data_owned = NULL;
               hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
               hypredrv_ErrorMsgAdd("Invalid dof label %d during MGR dense remap",
                                    (int)raw);
               return;
            }
            /* GCOVR_EXCL_STOP */
            dofmap_data[i] = label_to_dense[raw];
         }
         /* GCOVR_EXCL_START */
         else
         {
            dofmap_data[i] = raw;
         }
         /* GCOVR_EXCL_STOP */
      }
   }
   free(label_to_dense);
   label_to_dense = NULL;

   /* GCOVR_EXCL_BR_STOP */

   /* Config preconditioner */
   HYPRE_MGRCreate(&precon);
   HYPRE_MGRSetCpointsByPointMarkerArray(precon, num_dofs_hypre, num_levels - 1,
                                         num_c_dofs, c_dofs, dofmap_data);
   HYPRE_MGRSetNonCpointsToFpoints(precon, args->non_c_to_f);
   HYPRE_MGRSetPMaxElmts(precon, args->pmax);
   HYPRE_MGRSetMaxIter(precon, args->max_iter);
   HYPRE_MGRSetTol(precon, args->tolerance);
   HYPRE_MGRSetPrintLevel(precon, args->print_level);
#if HYPRE_CHECK_MIN_VERSION(22000, 0)
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
#endif
   HYPRE_MGRSetRelaxType(precon, args->relax_type); /* TODO: we shouldn't need this */
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                      "MGR stage after base hypre setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   /* Set level parameters */
/* GCOVR_EXCL_BR_START */
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   for (i = 0; i < num_levels - 1; i++)
   {
      MGRlvl_args *level_args = &args->level[active_level_map[i]];
      HYPRE_Int    type       = level_args->f_relaxation.type;

      level_frelax_type[i]   = (type == MGR_FRLX_TYPE_NESTED_MGR) ? 7 : type;
      level_frelax_sweeps[i] = level_args->f_relaxation.num_sweeps;
      level_grelax_type[i]   = level_args->g_relaxation.type;
      level_grelax_sweeps[i] = level_args->g_relaxation.num_sweeps;
      level_interp_type[i]   = level_args->prolongation_type;
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
#endif

   /* Config f-relaxation at each MGR level */
   for (i = 0; i < num_levels - 1; i++)
   {
      HYPRE_Int    orig_lvl   = active_level_map[i];
      MGRlvl_args *level_args = &args->level[orig_lvl];

      if (level_args->f_relaxation.use_krylov && level_args->f_relaxation.krylov)
      {
         hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, level_args->f_relaxation.krylov,
                                     args->dofmap, args->vec_nn,
                                     &level_args->f_relaxation.krylov->base_solver);
         /* GCOVR_EXCL_START */
         if (hypredrv_ErrorCodeActive())
         {
            return;
         }
         /* GCOVR_EXCL_STOP */
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRSetFSolverAtLevel(precon, (HYPRE_Solver)level_args->f_relaxation.krylov,
                                    i);
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Nested Krylov F-relaxation requires hypre >= 2.31.0");
         return;
#endif
      }
      else if (level_args->f_relaxation.type == 2)
      {
         hypredrv_AMGCreate(&level_args->f_relaxation.amg, &frelax);
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
#else
         (void)precon;
         (void)i;
#endif
         args->frelax[orig_lvl] = frelax;
      }
      else if (level_args->f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
         if (i != 0)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "Nested MGR F-relaxation is only supported at MGR level 0 by hypre");
            goto cleanup;
         }

         MGR_args      *nested_args    = level_args->f_relaxation.mgr;
         HYPRE_Solver   frelax_wrapper = NULL;
         IntArray      *nested_dofmap  = NULL;
         IntArray      *saved_dofmap   = NULL;
         HYPRE_IJVector saved_vec_nn   = NULL;

         if (!nested_args)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
            goto cleanup;
         }

         nested_dofmap = MGRBuildProjectedFRelaxDofmap(args->dofmap, &level_args->f_dofs);
         if (hypredrv_ErrorCodeActive() || !nested_dofmap)
         {
            hypredrv_IntArrayDestroy(&nested_dofmap);
            goto cleanup;
         }

         saved_dofmap        = nested_args->dofmap;
         saved_vec_nn        = nested_args->vec_nn;
         nested_args->dofmap = nested_dofmap;
         nested_args->vec_nn = NULL;
         hypredrv_MGRCreate(nested_args, &frelax);
         nested_args->dofmap = saved_dofmap;
         nested_args->vec_nn = saved_vec_nn;
         /* GCOVR_EXCL_START */
         if (hypredrv_ErrorCodeActive())
         {
            hypredrv_IntArrayDestroy(&nested_dofmap);
            return;
         }

         frelax_wrapper = MGRNestedFRelaxWrapperCreate(frelax, nested_dofmap);
         if (hypredrv_ErrorCodeActive() || !frelax_wrapper)
         {
            HYPRE_MGRDestroy(frelax);
            hypredrv_IntArrayDestroy(&nested_dofmap);
            return;
         }
         /* GCOVR_EXCL_STOP */
         nested_dofmap = NULL;
         HYPRE_MGRSetFSolverAtLevel(precon, frelax_wrapper, i);
         args->frelax[orig_lvl] = frelax_wrapper;
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "Nested MGR F-relaxation requires hypre >= 3.1.0 (develop >= 5)");
         return;
#endif
      }
#if defined(HYPRE_USING_DSUPERLU)
      /* GCOVR_EXCL_START */
      else if (level_args->f_relaxation.type == 29)
      {
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRDirectSolverCreate(&frelax);
         if (!frelax)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation 'spdirect' unavailable: direct solver creation failed");
            return;
         }
         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
         args->frelax[orig_lvl] = frelax;
#else
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR F-relaxation 'spdirect' requires hypre >= 2.31.0");
         return;
#endif
      }
      /* GCOVR_EXCL_STOP */
#else
      /* GCOVR_EXCL_START */
      else if (level_args->f_relaxation.type == 29)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'spdirect' requires hypre built with DSUPERLU");
         return;
      }
      /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
      else if (level_args->f_relaxation.type == 32)
      {
         hypredrv_ILUCreate(&level_args->f_relaxation.ilu, &frelax);

         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
         args->frelax[orig_lvl] = frelax;
      }
#endif
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                      "MGR stage after F-relax setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   /* Config global relaxation at each MGR level */
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   for (i = 0; i < num_levels - 1; i++)
   {
      HYPRE_Int    orig_lvl   = active_level_map[i];
      MGRlvl_args *level_args = &args->level[orig_lvl];

      if (level_args->g_relaxation.use_krylov && level_args->g_relaxation.krylov)
      {
         hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, level_args->g_relaxation.krylov,
                                     args->dofmap, args->vec_nn,
                                     &level_args->g_relaxation.krylov->base_solver);
         /* GCOVR_EXCL_START */
         if (hypredrv_ErrorCodeActive())
         {
            return;
         }
         /* GCOVR_EXCL_STOP */
         HYPRE_MGRSetGlobalSmootherAtLevel(
            precon, (HYPRE_Solver)level_args->g_relaxation.krylov, i);
      }
      else if (level_args->g_relaxation.type == 20)
      {
         hypredrv_AMGCreate(&level_args->g_relaxation.amg, &grelax);
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, i);
         args->grelax[orig_lvl] = grelax;
      }
      else if (level_args->g_relaxation.type == 16)
      {
         hypredrv_ILUCreate(&level_args->g_relaxation.ilu, &grelax);
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, i);
         args->grelax[orig_lvl] = grelax;
      }
   }
#endif
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                      "MGR stage after G-relax setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      hypredrv_NestedKrylovCreate(MPI_COMM_WORLD, args->coarsest_level.krylov,
                                  args->dofmap, args->vec_nn,
                                  &args->coarsest_level.krylov->base_solver);
      /* GCOVR_EXCL_START */
      if (hypredrv_ErrorCodeActive())
      {
         return;
      }
      /* GCOVR_EXCL_STOP */
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(precon, MGRBaseParSolverSolve, MGRBaseParSolverSetup,
                               (HYPRE_Solver)args->coarsest_level.krylov);
#else
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Nested Krylov coarsest solver requires hypre >= 3.1.0");
      return;
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
      /* GCOVR_EXCL_STOP */

      /* Config coarsest level solver */
      if (args->coarsest_level.type == 0)
      {
         hypredrv_AMGCreate(&args->coarsest_level.amg, &args->csolver);
         args->csolver_type = 0;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                                  args->csolver);
      }
      else if (args->coarsest_level.type == 32)
      {
         hypredrv_ILUCreate(&args->coarsest_level.ilu, &args->csolver);
         args->csolver_type = 32;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_ILUSolve, HYPRE_ILUSetup, args->csolver);
      }
#ifdef HYPRE_USING_DSUPERLU
      /* GCOVR_EXCL_START */
      else if (args->coarsest_level.type == 29)
      {
         HYPRE_MGRDirectSolverCreate(&args->csolver);
         if (!args->csolver)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR coarsest_level 'spdirect' unavailable: direct solver creation "
               "failed");
            return;
         }
         args->csolver_type = 29;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_MGRDirectSolverSolve,
                                  HYPRE_MGRDirectSolverSetup, args->csolver);
      }
      /* GCOVR_EXCL_STOP */
#else
      /* GCOVR_EXCL_START */
      else if (args->coarsest_level.type == 29)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR coarsest_level 'spdirect' requires hypre built with DSUPERLU");
         return;
      }
      /* GCOVR_EXCL_STOP */
#endif
   }
   HYPREDRV_LOG_COMMF(4, MPI_COMM_WORLD, NULL, 0,
                      "MGR stage after coarsest solver setup: code=0x%x",
                      hypredrv_ErrorCodeGet());

   /* GCOVR_EXCL_BR_STOP */

#if HYPRE_CHECK_MIN_VERSION(23100, 11)
   HYPRE_MGRSetNonGalerkinMaxElmts(precon, args->nonglk_max_elmts);
#endif

   /* Set output pointer */
   *precon_ptr = precon;
   precon      = NULL;
   if (dofmap_data_owned)
   {
      /* hypre uses the point-marker array during MGRSetup, so keep the owned copy
       * alive until PreconDestroyMGRSolver() destroys the MGR object. */
      /* GCOVR_EXCL_START */
      /* GCOVR_EXCL_BR_START */    /* low-signal branch under CI */
      if (args->point_marker_data) /* GCOVR_EXCL_BR_STOP */
      {
         free(args->point_marker_data);
      }
      /* GCOVR_EXCL_STOP */
      args->point_marker_data = dofmap_data_owned;
      dofmap_data_owned       = NULL;
   }
   goto cleanup;

cleanup:
   if (dofmap_data_owned)
   {
      free(dofmap_data_owned);
   }
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      free(c_dofs[lvl]);
   }
   free(inactive_dofs);
   free(label_present);
   free(label_to_dense);
   if (precon)
   {
      HYPRE_MGRDestroy(precon);
   }
   /* Silence any hypre errors. TODO: improve error handling */
   HYPRE_ClearAllErrors();
#endif
}
