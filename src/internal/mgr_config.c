/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <errno.h>
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

#define MGRfrlx_FIELDS(_prefix)                                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)                       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet)                 \
   ADD_FIELD_OFFSET_ENTRY(_prefix, symmetric_diagonal_scaling, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, MGRfrlxMGRSetArgs)                               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRfrlxAMGSetArgs)                               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRfrlxILUSetArgs)                               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, MGRfrlxFSAISetArgs)                             \
   MGRfrlx_SCHWARZ_FIELD(_prefix)

#define MGRgrlx_FIELDS(_prefix)                                          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRgrlxAMGSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRgrlxILUSetArgs)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, fsai, MGRgrlxFSAISetArgs)             \
   MGRgrlx_SCHWARZ_FIELD(_prefix)

#define MGRlvl_FIELDS(_prefix)                                                    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_dofs, MGRlvlFDofsSet)                        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, hypredrv_FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_level_type, hypredrv_FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, matched_q, hypredrv_FieldTypeIntSet)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, matched_f_backsolve, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_relaxation, hypredrv_MGRfrlxSetArgs)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, g_relaxation, hypredrv_MGRgrlxSetArgs)

#define MGR_CYCLE_FIELDS(_prefix) ADD_FIELD_OFFSET_ENTRY(_prefix, cycle, MGRCycleSet)

#define MGR_FIELDS(_prefix)                                                       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, non_c_to_f, hypredrv_FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pmax, hypredrv_FieldTypeIntSet)                \
   ADD_FIELD_OFFSET_ENTRY(_prefix, interp_sweeps, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, injection_upcycle, hypredrv_FieldTypeIntSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, matched_q_sweeps, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, hypredrv_FieldTypePositiveIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, hypredrv_FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relax_type, hypredrv_FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, hypredrv_FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, nonglk_max_elmts, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, hypredrv_FieldTypeDoubleSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_th, hypredrv_FieldTypeDoubleSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, interp_weight, hypredrv_FieldTypeDoubleSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, matched_q_weight, hypredrv_FieldTypeDoubleSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsest_level, hypredrv_MGRclsSetArgs)        \
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
      /* calloc (not malloc): zero-initialize the embedded solver/precon parameter
         unions. They are otherwise read as garbage when MGR treats this object as
         its F-relaxation solver (it mirrors the hypre_Solver layout), e.g. in
         hypre_MGRSetupStats. */
      *ptr = (NestedKrylov_args *)calloc(1, sizeof(NestedKrylov_args));
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
/* Picks the label array to build the presence mask from, preferring the global
 * unique list when it is strictly increasing, then the local unique list, then
 * the raw dofmap. Returns NULL when no usable array exists. */
static const int *
MGRDofmapSelectLabels(const IntArray *dofmap, size_t *num_labels_out)
{
   const int *labels = NULL;

   if (dofmap->g_unique_data && dofmap->g_unique_size > 0)
   {
      labels          = dofmap->g_unique_data;
      *num_labels_out = dofmap->g_unique_size;
      for (size_t i = 1; i < *num_labels_out; i++)
      {
         if (labels[i] <= labels[i - 1])
         {
            labels = NULL;
            break;
         }
      }
      if (labels)
      {
         return labels;
      }
   }

   if (dofmap->unique_data && dofmap->unique_size > 0)
   {
      *num_labels_out = dofmap->unique_size;
      return dofmap->unique_data;
   }
   if (dofmap->data && dofmap->size > 0)
   {
      *num_labels_out = dofmap->size;
      return dofmap->data;
   }

   *num_labels_out = 0;
   return NULL;
}

/* Fallback for distributed dofmaps that only provide the global unique count
 * without a usable label array: treat that as a dense [0, g_unique_size) label
 * space instead of rejecting MGR. */
static int
MGRDofmapDenseFallback(const IntArray *dofmap, size_t *label_space_size_out,
                       size_t *num_present_labels_out, HYPRE_Int **label_present_out)
{
   HYPRE_Int *label_mask = NULL;

   *label_space_size_out = dofmap->g_unique_size;
   label_mask            = (HYPRE_Int *)calloc(*label_space_size_out, sizeof(HYPRE_Int));
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

/* Rejects negative labels and reports the largest one seen. */
static int
MGRDofmapMaxLabel(const int *labels, size_t num_labels, int *max_label_out)
{
   int max_label = -1;

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

   *max_label_out = max_label;

   return 1;
}

int
hypredrv_MGRBuildDofLabelPresenceMask(const IntArray *dofmap,
                                      size_t         *label_space_size_out,
                                      size_t         *num_present_labels_out,
                                      HYPRE_Int     **label_present_out)
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

   labels = MGRDofmapSelectLabels(dofmap, &num_labels);
   if (!labels)
   {
      if (dofmap->g_unique_size > 0)
      {
         return MGRDofmapDenseFallback(dofmap, label_space_size_out,
                                       num_present_labels_out, label_present_out);
      }
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, NULL, 0,
                         "MGR invalid dofmap: empty label data and g_unique_size=%zu",
                         dofmap->g_unique_size);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("MGR requires a non-empty dofmap");
      return 0;
   }

   if (!MGRDofmapMaxLabel(labels, num_labels, &max_label))
   {
      return 0;
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
/* Marks each parent F label that the nested hierarchy keeps, rejecting labels
 * that are out of range, absent from the parent dofmap, or repeated. */
static int
MGRBuildFDofKeepMask(const StackIntArray *parent_f_dofs, const HYPRE_Int *parent_present,
                     size_t parent_label_space, HYPRE_Int *keep_label)
{
   for (size_t i = 0; i < parent_f_dofs->size; i++)
   {
      int label = parent_f_dofs->data[i];

      /* GCOVR_EXCL_START */
      if (label < 0 || (size_t)label >= parent_label_space)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid parent MGR f_dofs label %d for nested MGR (valid range: [0,%d])",
            label, (int)parent_label_space - 1);
         return 0;
      }
      if (!parent_present[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Parent MGR f_dofs label %d is not present in parent dofmap", label);
         return 0;
      }
      if (keep_label[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Duplicate parent MGR f_dofs label %d for nested MGR",
                              label);
         return 0;
      }
      /* GCOVR_EXCL_STOP */
      keep_label[label] = 1;
   }

   return 1;
}

/* unique_data and g_unique_data are both the sorted set of kept labels
 * (keep_label[i] == 1 iff label i is an F-dof that appears in the projection). */
static int
MGRFillProjectedUniqueLabels(IntArray *nested_dofmap, const HYPRE_Int *keep_label,
                             size_t parent_label_space, size_t nested_num_labels)
{
   nested_dofmap->unique_size   = nested_num_labels;
   nested_dofmap->g_unique_size = nested_num_labels;
   if (nested_num_labels == 0)
   {
      return 1;
   }

   nested_dofmap->unique_data   = (int *)malloc(nested_num_labels * sizeof(int));
   nested_dofmap->g_unique_data = (int *)malloc(nested_num_labels * sizeof(int));
   /* GCOVR_EXCL_START */
   if (!nested_dofmap->unique_data || !nested_dofmap->g_unique_data)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate nested MGR dof label arrays");
      return 0;
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

   return 1;
}

IntArray *
hypredrv_MGRBuildProjectedFRelaxDofmap(const IntArray      *parent_dofmap,
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

   if (!hypredrv_MGRBuildDofLabelPresenceMask(parent_dofmap, &parent_label_space, NULL,
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

   if (!MGRBuildFDofKeepMask(parent_f_dofs, parent_present, parent_label_space,
                             keep_label))
   {
      goto cleanup;
   }

   /* Count filtered entries. Labels in parent_dofmap are bounded by parent_label_space
    * (guaranteed by hypredrv_MGRBuildDofLabelPresenceMask), so no range check is needed
    * here. */
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

   if (!MGRFillProjectedUniqueLabels(nested_dofmap, keep_label, parent_label_space,
                                     nested_num_labels))
   {
      goto cleanup;
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

   if (old_type == amg_type)
   {
      hypredrv_AMGDestroyRBMs((AMG_args *)union_base);
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

const char *
hypredrv_MGRLogObjectName(const Stats *stats)
{
   /* Shared buffer: the library keeps global state and is not thread-safe. */
   static char buf[32];
   return hypredrv_StatsGetLogObjectName(stats, buf, sizeof(buf));
}

HYPRE_Int
hypredrv_MGRLevelInterpTypeCompat(HYPRE_Int interp_type, const Stats *stats,
                                  int next_ls_id, HYPRE_Int level)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER == 30100 && HYPREDRV_HYPRE_DEVELOP_NUMBER == 0
   if (interp_type == 13 || interp_type == 14)
   {
      const char *interp_name = (interp_type == 13) ? "blk-rowsum" : "blk-absrowsum";

      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
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
/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

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
   hypredrv_MGRComponentReuseSetDefaultArgs(&args->reuse);

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
   args->type                       = 7;
   args->num_sweeps                 = 1;
   args->symmetric_diagonal_scaling = 0;
   args->use_krylov                 = 0;
   args->krylov                     = NULL;
   args->mgr                        = NULL;
   hypredrv_MGRComponentReuseSetDefaultArgs(&args->reuse);
   /* Initialize the union for embedders that select AMG by assigning type=2
    * directly instead of transitioning through the YAML parser. */
   hypredrv_AMGSetDefaultArgs(&args->amg);
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
   hypredrv_MGRComponentReuseSetDefaultArgs(&args->reuse);

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
   args->f_dofs              = STACK_INTARRAY_CREATE();
   args->prolongation_type   = 0;
   args->restriction_type    = 0;
   args->coarse_level_type   = 0;
   args->matched_q           = 0;
   args->matched_f_backsolve = 0;

   hypredrv_MGRfrlxSetDefaultArgs(&args->f_relaxation);
   hypredrv_MGRgrlxSetDefaultArgs(&args->g_relaxation);
}

/*-----------------------------------------------------------------------------
 * MGRSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRSetDefaultArgs(MGR_args *args)
{
   args->dofmap            = NULL;
   args->max_iter          = 1;
   args->num_levels        = 0;
   args->print_level       = 0;
   args->non_c_to_f        = 1;
   args->pmax              = 0;
   args->interp_sweeps     = 0;
   args->injection_upcycle = 0;
   args->matched_q_sweeps  = 0;
   args->nonglk_max_elmts  = 1;
   args->tolerance         = 0.0;
   args->coarse_th         = 0.0;
   args->interp_weight     = 1.0;
   args->matched_q_weight  = 1.0;
   args->relax_type        = 7;
   args->cycle             = 1;
   args->cycle_smooth_pos  = 1;

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
   args->vec_nn               = NULL;
   args->rbm_input_generation = 0;
   args->coarse_schur         = NULL;
   args->point_marker_data    = NULL;
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
   hypredrv_MGRComponentReuseDestroyArgs(reuse);
   hypredrv_MGRComponentReuseSetDefaultArgs(reuse);
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
   if (!strcmp(key, "symmetric_diagonal_scaling"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else if (!strcmp(key, "type"))
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
   if (!strcmp(key, "matched_q"))
   {
      static StrIntMap map[] = {
         {"off", 0},
         {"on", 1},
         {"polynomial", 1},
         {"afsai", 2},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "matched_f_backsolve"))
   {
      static StrIntMap map[] = {
         {"off", 0},
         {"on", 1},
         {"gmres1", 2},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {
         {"injection", 0},      {"l1-jacobi", 1},    {"jacobi", 2},
         {"classical-mod", 3},  {"approx-inv", 4},
#if HYPRE_CHECK_MIN_VERSION(22400, 0)
         {"mm-ext", 5},         {"mm-ext+i", 6},     {"mm-ext+e", 7},
#endif
         {"blk-jacobi", 12},    {"blk-rowlump", 13}, {"blk-rowsum", 13},
         {"blk-absrowsum", 14},
      };

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "restriction_type"))
   {
#if HYPRE_CHECK_MIN_VERSION(23200, 0)
      static StrIntMap map[] = {
         {"injection", 0}, {"jacobi", 2},    {"approx-inv", 3},
         {"air_1", 4},     {"air_1.5", 5},   {"blk-jacobi", 12},
         {"cpr-like", 13}, {"columped", 14}, {"columped-partial", 15},
      };
#else
      static StrIntMap map[] = {
         {"injection", 0}, {"jacobi", 2},    {"approx-inv", 3},        {"blk-jacobi", 12},
         {"cpr-like", 13}, {"columped", 14}, {"columped-partial", 15},
      };
#endif

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "coarse_level_type"))
   {
      static StrIntMap map[] = {
         {"rap", 0},           {"galerkin", 0},       {"non-galerkin", 1},
         {"cpr-like-diag", 2}, {"cpr-like-bdiag", 3}, {"approx-inv", 4},
         {"acc", 5},           {"user", 6},
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

static int MGRHasMatchedSchurGMRES1AtDepth(const MGR_args *, int);

static int
MGRNestedKrylovHasMatchedSchurGMRES1(const NestedKrylov_args *args, int depth)
{
   return args && args->has_precon && args->precon_method == PRECON_MGR &&
          MGRHasMatchedSchurGMRES1AtDepth(&args->precon.mgr, depth + 1);
}

static int
MGRHasMatchedSchurGMRES1AtDepth(const MGR_args *args, int depth)
{
   if (!args)
   {
      return 0;
   }
   if (depth >= MAX_MGR_LEVELS)
   {
      return 0;
   }

   HYPRE_Int fine_levels = hypredrv_MGRNumFineLevels(args);
   for (HYPRE_Int level = 0; level < fine_levels; level++)
   {
      if (args->level[level].matched_f_backsolve == 2)
      {
         return 1;
      }
      const MGRfrlx_args *frelax = &args->level[level].f_relaxation;
      const MGRgrlx_args *grelax = &args->level[level].g_relaxation;
      if ((frelax->mgr && MGRHasMatchedSchurGMRES1AtDepth(frelax->mgr, depth + 1)) ||
          MGRNestedKrylovHasMatchedSchurGMRES1(frelax->krylov, depth) ||
          MGRNestedKrylovHasMatchedSchurGMRES1(grelax->krylov, depth))
      {
         return 1;
      }
   }
   return MGRNestedKrylovHasMatchedSchurGMRES1(args->coarsest_level.krylov, depth);
}

int
hypredrv_MGRHasMatchedSchurGMRES1(const MGR_args *args)
{
   return MGRHasMatchedSchurGMRES1AtDepth(args, 0);
}

int
hypredrv_MGRValidateOuterSolver(const MGR_args *args, int solver_method,
                                const char *context)
{
   if (!hypredrv_MGRHasMatchedSchurGMRES1(args) || solver_method == SOLVER_FGMRES)
   {
      return 1;
   }

   hypredrv_ErrorCodeSet(ERROR_INVALID_SOLVER);
   hypredrv_ErrorMsgAdd(
      "%s matched_f_backsolve: gmres1 is residual-dependent and requires FGMRES",
      context ? context : "MGR");
   return 0;
}

/*-----------------------------------------------------------------------------
 * MGRGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_MGRGetValidValues(const char *key)
{
   if (!strcmp(key, "injection_upcycle"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   else if (!strcmp(key, "relax_type"))
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
/* The `level:` block is a sequence of per-level mappings; each one is parsed
 * into the matching MGRlvl_args slot. */
/* Marks the per-level component blocks that own nested solver configuration, so
 * the generic field setter does not try to consume them. */
static void
MGRMarkLevelComponentNodes(MGR_args *args, YAMLnode *grandchild, int lvl)
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
}

/* Parses the `level` mapping and returns the number of fine levels defined. */
static HYPRE_Int
MGRSetLevelArgsFromYAML(MGR_args *args, YAMLnode *child)
{
   HYPRE_Int num_fine    = 0;
   uint32_t  seen_levels = 0;
   int       max_lvl     = -1;
   YAML_NODE_SET_VALID(child);
   YAML_NODE_ITERATE(child, grandchild)
   {
      char *lvl_end = NULL;
      errno         = 0;
      long lvl_l    = strtol(grandchild->key, &lvl_end, 10);

      /* Reject non-numeric level keys (e.g. "lvl0"): strtol would otherwise
       * silently map them to 0 and overwrite a real level's configuration. */
      if (grandchild->key[0] == '\0' || !lvl_end || *lvl_end != '\0' || errno == ERANGE ||
          lvl_l < 0 || lvl_l >= MAX_MGR_LEVELS - 1)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("MGR level index '%s' must be an integer from 0 to %d",
                              grandchild->key, MAX_MGR_LEVELS - 2);
         YAML_NODE_SET_INVALID_KEY(grandchild);
         continue;
      }
      int lvl = (int)lvl_l;
      /* A level must be a mapping.  In particular, never accept an
       * unsupported or malformed inline mapping as a scalar and then
       * silently retain the level defaults. */
      if (grandchild->val && grandchild->val[0] != '\0')
      {
         hypredrv_ErrorCodeSet(ERROR_UNEXPECTED_VAL);
         hypredrv_ErrorMsgAdd("MGR level %d must be a mapping (for example, "
                              "\"%d: { f_dofs: [2] }\")",
                              lvl, lvl);
         grandchild->valid = YAML_NODE_UNEXPECTED_VAL;
         continue;
      }
      /* Reject duplicate level indices, which would double-count levels. */
      if (seen_levels & (1u << (unsigned)lvl))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("Duplicate MGR level index %d", lvl);
         YAML_NODE_SET_INVALID_KEY(grandchild);
         continue;
      }

      seen_levels |= (1u << (unsigned)lvl);
      if (lvl > max_lvl)
      {
         max_lvl = lvl;
      }
      MGRMarkLevelComponentNodes(args, grandchild, lvl);

      num_fine++;
      YAML_NODE_SET_VALID(grandchild);
   }

   /* Consumption iterates fine levels densely over [0, num_levels-1), so the
    * configured level indices must be contiguous starting at 0. Reject gaps
    * (e.g. "0:" and "5:") which would otherwise silently process
    * default-initialized levels in place of the intended configuration.
    * Indices are at most MAX_MGR_LEVELS - 2, so the shift below fits. */
   if (max_lvl >= 0)
   {
      uint32_t expected = (1u << (unsigned)(max_lvl + 1)) - 1u;
      if (seen_levels != expected)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd("MGR level indices must be contiguous starting at 0 (found "
                              "non-contiguous set up to level %d)",
                              max_lvl);
      }
   }

   return num_fine;
}

void
hypredrv_MGRSetArgsFromYAML(void *vargs, YAMLnode *parent)
{
   MGR_args *args            = (MGR_args *)vargs;
   HYPRE_Int derived_levels  = 0; /* from `level` and `coarsest_level` entries */
   int       explicit_levels = 0; /* `num_levels` key present */
   YAML_NODE_ITERATE(parent, child)
   {
      if (!strcmp(child->key, "level"))
      {
         derived_levels += MGRSetLevelArgsFromYAML(args, child);
      }
      else if (!strcmp(child->key, "coarsest_level"))
      {
         derived_levels++;
         YAML_NODE_SET_VALID(child);
         hypredrv_MGRclsSetArgsFromYAML(&args->coarsest_level, child);
      }
      else
      {
         explicit_levels |= !strcmp(child->key, "num_levels");
         YAML_NODE_VALIDATE(child, hypredrv_MGRGetValidKeys, hypredrv_MGRGetValidValues);
         YAML_NODE_SET_FIELD(child, args, hypredrv_MGRSetFieldByName);
      }
   }

   /* Level entries define the hierarchy; an explicit num_levels may only
    * restate that count, regardless of where it appears in the mapping. */
   if (derived_levels > 0)
   {
      if (explicit_levels && args->num_levels != derived_levels)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("MGR num_levels (%d) conflicts with the %d levels defined "
                              "by level/coarsest_level; omit num_levels",
                              (int)args->num_levels, (int)derived_levels);
      }
      args->num_levels = derived_levels;
   }
   if (args->num_levels < 0 || args->num_levels > MAX_MGR_LEVELS)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("MGR num_levels must be between 0 and %d (got %d)",
                           MAX_MGR_LEVELS, (int)args->num_levels);
   }
}

/* GCOVR_EXCL_BR_STOP */

/*-----------------------------------------------------------------------------
 * hypredrv_MGRSetDofmap
 *-----------------------------------------------------------------------------*/

void
hypredrv_MGRSetDofmap(MGR_args *args, IntArray *dofmap)
{
   args->dofmap = dofmap;
   args->rbm_input_generation++;
}

void
hypredrv_MGRSetNearNullSpace(MGR_args *args, HYPRE_IJVector vec_nn)
{
   args->vec_nn = vec_nn;
   args->rbm_input_generation++;
}

void
hypredrv_MGRSetCoarseSchur(MGR_args *args, const HYPRE_IJMatrix *coarse_schur)
{
   args->coarse_schur = coarse_schur;
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
      hypredrv_MGRComponentReuseDestroyArgs(&args->level[i].f_relaxation.reuse);
      hypredrv_MGRComponentReuseDestroyArgs(&args->level[i].g_relaxation.reuse);

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

   hypredrv_MGRComponentReuseDestroyArgs(&args->coarsest_level.reuse);
}
/* GCOVR_EXCL_BR_STOP */
