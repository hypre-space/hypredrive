/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "mgr.h"
#include <mpi.h>
#include "gen_macros.h"
#include "nested_krylov.h"
#include "stats.h"
#if !HYPRE_CHECK_MIN_VERSION(30100, 5)
#include "_hypre_utilities.h" // for hypre_Solver
#endif

typedef struct HYPREDRV_MGRFRelaxWrapper_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);
   HYPRE_Solver inner_mgr;
} HYPREDRV_MGRFRelaxWrapper;

static HYPRE_Int
HYPREDRV_MGRFRelaxWrapperSetup(void *wrapper_v, void *A, void *b, void *x)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper = (HYPREDRV_MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   return HYPRE_MGRSetup((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
HYPREDRV_MGRFRelaxWrapperSolve(void *wrapper_v, void *A, void *b, void *x)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper = (HYPREDRV_MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   return HYPRE_MGRSolve((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
HYPREDRV_MGRFRelaxWrapperDestroy(void *wrapper_v)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper = (HYPREDRV_MGRFRelaxWrapper *)wrapper_v;
   free(wrapper);
   return 0;
}

static HYPRE_Solver
MGRNestedFRelaxWrapperCreate(HYPRE_Solver inner_mgr)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper =
      (HYPREDRV_MGRFRelaxWrapper *)calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate nested MGR F-relaxation wrapper");
      return NULL;
   }

   wrapper->setup     = HYPREDRV_MGRFRelaxWrapperSetup;
   wrapper->solve     = HYPREDRV_MGRFRelaxWrapperSolve;
   wrapper->destroy   = HYPREDRV_MGRFRelaxWrapperDestroy;
   wrapper->inner_mgr = inner_mgr;
   return (HYPRE_Solver)wrapper;
}

HYPRE_Solver
MGRNestedFRelaxWrapperGetInner(HYPRE_Solver wrapper_solver)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper = (HYPREDRV_MGRFRelaxWrapper *)wrapper_solver;
   return wrapper ? wrapper->inner_mgr : NULL;
}

HYPRE_Solver
MGRNestedFRelaxWrapperDetachInner(HYPRE_Solver wrapper_solver)
{
   HYPREDRV_MGRFRelaxWrapper *wrapper = (HYPREDRV_MGRFRelaxWrapper *)wrapper_solver;
   HYPRE_Solver               inner   = NULL;

   if (!wrapper)
   {
      return NULL;
   }

   inner              = wrapper->inner_mgr;
   wrapper->inner_mgr = NULL;
   return inner;
}

void
MGRNestedFRelaxWrapperFree(HYPRE_Solver *wrapper_ptr)
{
   if (!wrapper_ptr || !*wrapper_ptr)
   {
      return;
   }
   HYPREDRV_MGRFRelaxWrapperDestroy((void *)(*wrapper_ptr));
   *wrapper_ptr = NULL;
}

/*-----------------------------------------------------------------------------
 * Field definitions using the type-setting wrappers
 *-----------------------------------------------------------------------------*/

/* Generate type-setting wrappers for union fields */
DEFINE_TYPED_SETTER(MGRclsAMGSetArgs, MGRcls_args, amg, 0, AMGSetArgs)
DEFINE_TYPED_SETTER(MGRclsILUSetArgs, MGRcls_args, ilu, 32, ILUSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxAMGSetArgs, MGRfrlx_args, amg, 2, AMGSetArgs)
DEFINE_TYPED_SETTER(MGRfrlxILUSetArgs, MGRfrlx_args, ilu, 32, ILUSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxAMGSetArgs, MGRgrlx_args, amg, 20, AMGSetArgs)
DEFINE_TYPED_SETTER(MGRgrlxILUSetArgs, MGRgrlx_args, ilu, 16, ILUSetArgs)
static void MGRclsTypeSet(void *, const YAMLnode *);
static void MGRfrlxTypeSet(void *, const YAMLnode *);
static void MGRgrlxTypeSet(void *, const YAMLnode *);
static void MGRfrlxMGRSetArgs(void *, const YAMLnode *);
void        MGRSetArgsFromYAML(void *, YAMLnode *);

#define MGRcls_FIELDS(_prefix)                            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, MGRclsTypeSet)   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRclsAMGSetArgs) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRclsILUSetArgs)

#define MGRfrlx_FIELDS(_prefix)                                 \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, MGRfrlxTypeSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, mgr, MGRfrlxMGRSetArgs)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRfrlxAMGSetArgs)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRfrlxILUSetArgs)

#define MGRgrlx_FIELDS(_prefix)                                 \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, MGRgrlxTypeSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_sweeps, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, amg, MGRgrlxAMGSetArgs)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, ilu, MGRgrlxILUSetArgs)

#define MGRlvl_FIELDS(_prefix)                                         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_dofs, FieldTypeStackIntArraySet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, prolongation_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, restriction_type, FieldTypeIntSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_level_type, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, f_relaxation, MGRfrlxSetArgs)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, g_relaxation, MGRgrlxSetArgs)

#define MGR_FIELDS(_prefix)                                           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, non_c_to_f, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pmax, FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, max_iter, FieldTypeIntSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_levels, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, relax_type, FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_level, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, nonglk_max_elmts, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, tolerance, FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarse_th, FieldTypeDoubleSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, coarsest_level, MGRclsSetArgs)

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
#define GENERATE_PREFIXED_LIST_MGR                   \
   GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRcls)  \
   GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRfrlx) \
   GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGRgrlx) \
   GENERATE_PREFIXED_COMPONENTS(MGRlvl)

/* Generate all boilerplate (field maps, setters, YAML parsing, etc.) */
GENERATE_PREFIXED_LIST_MGR                    // LCOV_EXCL_LINE
GENERATE_PREFIXED_COMPONENTS_CUSTOM_YAML(MGR) // LCOV_EXCL_LINE

   /*-----------------------------------------------------------------------------
    *-----------------------------------------------------------------------------*/

   static bool MGRIsNestedKrylovKey(const char *key)
{
   if (!key)
   {
      return false;
   }

   char *tmp = StrTrim(strdup(key));
   if (!tmp)
   {
      return false;
   }
   StrToLowerCase(tmp);
   bool is_valid = StrIntMapArrayDomainEntryExists(SolverGetValidTypeIntMap(), tmp);
   free(tmp);
   return is_valid;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static NestedKrylov_args *
MGRGetOrCreateNestedKrylov(NestedKrylov_args **ptr)
{
   if (!ptr)
   {
      return NULL;
   }

   if (!*ptr)
   {
      *ptr = (NestedKrylov_args *)malloc(sizeof(NestedKrylov_args));
      if (*ptr)
      {
         NestedKrylovSetDefaultArgs(*ptr);
      }
   }

   return *ptr;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static MGR_args *
MGRGetOrCreateNestedMGR(MGR_args **ptr)
{
   if (!ptr)
   {
      return NULL;
   }

   if (!*ptr)
   {
      *ptr = (MGR_args *)malloc(sizeof(MGR_args));
      if (*ptr)
      {
         MGRSetDefaultArgs(*ptr);
      }
   }

   return *ptr;
}

/*-----------------------------------------------------------------------------
 * Build a dense label-space mask for labels present in an IntArray dofmap.
 *
 * The dof labels carried in IntArray may be sparse (e.g., nested MGR projected
 * F-blocks preserving parent labels). This helper derives the label-space size
 * as `max_label + 1` and a presence mask over that space.
 *-----------------------------------------------------------------------------*/

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

   if (!dofmap || !label_space_size_out || !label_present_out)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid arguments while building MGR dof label presence mask");
      return 0;
   }

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
   if (!labels && dofmap->g_unique_size > 0)
   {
      /* Compatibility fallback for distributed dofmaps: some IntArray sources
       * only reliably provide the global count, while label ids are expected
       * to be dense [0, ..., g_unique_size-1]. */
      *label_space_size_out = dofmap->g_unique_size;
      label_mask = (HYPRE_Int *)calloc(*label_space_size_out, sizeof(HYPRE_Int));
      if (!label_mask)
      {
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Failed to allocate MGR dof label presence mask");
         return 0;
      }
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
   else if (dofmap->unique_data && dofmap->unique_size > 0)
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
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("MGR requires a non-empty dofmap");
      return 0;
   }

   for (size_t i = 0; i < num_labels; i++)
   {
      if (labels[i] < 0)
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Invalid negative dof label %d in dofmap", labels[i]);
         return 0;
      }
      if (labels[i] > max_label)
      {
         max_label = labels[i];
      }
   }

   *label_space_size_out = (size_t)max_label + 1;
   label_mask            = (HYPRE_Int *)calloc(*label_space_size_out, sizeof(HYPRE_Int));
   if (!label_mask)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate MGR dof label presence mask");
      return 0;
   }

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

/*-----------------------------------------------------------------------------
 * Build a projected dofmap for nested MGR F-relaxation.
 *
 * Nested MGR acts on the outer level's F-block. Preserve the parent's original
 * labels when filtering to the selected F-point rows so nested `f_dofs` continue to
 * refer to the same label values as the parent MGR.
 *-----------------------------------------------------------------------------*/

static IntArray *
MGRBuildProjectedFRelaxDofmap(const IntArray      *parent_dofmap,
                              const StackIntArray *parent_f_dofs)
{
   if (!parent_dofmap || !parent_f_dofs)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Nested MGR requires a valid parent dofmap and parent f_dofs");
      return NULL;
   }

   if (parent_f_dofs->size == 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Nested MGR requires non-empty parent f_dofs");
      return NULL;
   }

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
      goto cleanup;
   }

   keep_label = (HYPRE_Int *)calloc(parent_label_space, sizeof(HYPRE_Int));
   if (!keep_label)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate nested MGR dof label selection mask");
      goto cleanup;
   }

   for (size_t i = 0; i < nested_num_labels; i++)
   {
      int label = parent_f_dofs->data[i];
      if (label < 0 || (size_t)label >= parent_label_space)
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd(
            "Invalid parent MGR f_dofs label %d for nested MGR (valid range: [0,%d])",
            label, (int)parent_label_space - 1);
         goto cleanup;
      }
      if (!parent_present[label])
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Parent MGR f_dofs label %d is not present in parent dofmap", label);
         goto cleanup;
      }
      if (keep_label[label])
      {
         ErrorCodeSet(ERROR_INVALID_VAL);
         ErrorMsgAdd("Duplicate parent MGR f_dofs label %d for nested MGR", label);
         goto cleanup;
      }
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

   nested_dofmap = IntArrayCreate(nested_size);
   if (!nested_dofmap || (nested_size > 0 && !nested_dofmap->data))
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate nested MGR projected dofmap");
      goto cleanup;
   }

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
      if (!nested_dofmap->unique_data || !nested_dofmap->g_unique_data)
      {
         ErrorCodeSet(ERROR_ALLOCATION);
         ErrorMsgAdd("Failed to allocate nested MGR dof label arrays");
         goto cleanup;
      }
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
   if (!ok)
   {
      IntArrayDestroy(&nested_dofmap);
   }
   return ok ? nested_dofmap : NULL;
}

/*-----------------------------------------------------------------------------
 * Type setters for union-backed solver args.
 *-----------------------------------------------------------------------------*/

#define DEFINE_MGR_TYPE_SETTER(_func, _parent, _amg_type, _ilu_type)        \
   static void _func(void *field, const YAMLnode *node)                     \
   {                                                                        \
      HYPRE_Int old_type = *((HYPRE_Int *)field);                           \
      FieldTypeIntSet(field, node);                                         \
      _parent *args = (_parent *)((char *)field - offsetof(_parent, type)); \
      if (args->type == old_type)                                           \
      {                                                                     \
         return;                                                            \
      }                                                                     \
      if (args->type == (_amg_type))                                        \
      {                                                                     \
         AMGSetDefaultArgs(&args->amg);                                     \
      }                                                                     \
      else if (args->type == (_ilu_type))                                   \
      {                                                                     \
         ILUSetDefaultArgs(&args->ilu);                                     \
      }                                                                     \
   }

DEFINE_MGR_TYPE_SETTER(MGRclsTypeSet, MGRcls_args, 0, 32)
DEFINE_MGR_TYPE_SETTER(MGRfrlxTypeSet, MGRfrlx_args, 2, 32)
DEFINE_MGR_TYPE_SETTER(MGRgrlxTypeSet, MGRgrlx_args, 20, 16)

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   return HYPRE_SolverSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
#else
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
#endif
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static HYPRE_Int
MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
   return HYPRE_SolverSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
#else
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 1;
#endif
}

/*-----------------------------------------------------------------------------
 * MGRclsSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRclsSetDefaultArgs(MGRcls_args *args)
{
   /* Default coarsest solver: let MGRCreate interpret type < 0 as "default AMG". */
   args->type       = -1;
   args->use_krylov = 0;
   args->krylov     = NULL;

   /* Initialize default AMG args (union storage). If user later selects ILU via YAML,
    * ILUSetArgs/ILUSetDefaultArgs will reinitialize the union storage. */
   AMGSetDefaultArgs(&args->amg);
}

/*-----------------------------------------------------------------------------
 * MGRfrlxSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRfrlxSetDefaultArgs(MGRfrlx_args *args)
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
MGRgrlxSetDefaultArgs(MGRgrlx_args *args)
{
   /* Default to "none" (disabled). If user selects a global smoother type via YAML
    * but omits num_sweeps, we want at least one sweep. */
   args->type       = -1;
   args->num_sweeps = 1;
   args->use_krylov = 0;
   args->krylov     = NULL;

   /* Initialize default AMG args (union storage). If user later selects ILU via YAML,
    * ILUSetArgs/ILUSetDefaultArgs will reinitialize the union storage. */
   AMGSetDefaultArgs(&args->amg);
}

/*-----------------------------------------------------------------------------
 * MGRlvlSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRlvlSetDefaultArgs(MGRlvl_args *args)
{
   args->f_dofs            = STACK_INTARRAY_CREATE();
   args->prolongation_type = 0;
   args->restriction_type  = 0;
   args->coarse_level_type = 0;

   MGRfrlxSetDefaultArgs(&args->f_relaxation);
   MGRgrlxSetDefaultArgs(&args->g_relaxation);
}

/*-----------------------------------------------------------------------------
 * MGRSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
MGRSetDefaultArgs(MGR_args *args)
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
      MGRlvlSetDefaultArgs(&args->level[i]);
      args->frelax[i] = NULL;
      args->grelax[i] = NULL;
   }
   MGRclsSetDefaultArgs(&args->coarsest_level);
   args->csolver      = NULL;
   args->csolver_type = -1;
   args->vec_nn       = NULL;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
MGRfrlxMGRSetArgs(void *field, const YAMLnode *node)
{
   MGRfrlx_args *parent = (MGRfrlx_args *)((char *)field - offsetof(MGRfrlx_args, mgr));
   MGR_args    **nested_ptr = (MGR_args **)field;
   MGR_args     *nested_mgr = MGRGetOrCreateNestedMGR(nested_ptr);

   if (!nested_mgr)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate nested MGR arguments for MGR f_relaxation");
      return;
   }

   parent->type = MGR_FRLX_TYPE_NESTED_MGR;
   MGRSetArgsFromYAML(nested_mgr, (YAMLnode *)node);
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
MGRclsSetArgsFromYAML(void *vargs, YAMLnode *parent)
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
               NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         YAML_NODE_VALIDATE(child, MGRclsGetValidKeys, MGRclsGetValidValues);
         YAML_NODE_SET_FIELD(child, args, MGRclsSetFieldByName);
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      YAML_NODE_VALIDATE(parent, MGRclsGetValidKeys, MGRclsGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, MGRclsSetFieldByName);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
MGRfrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
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
               NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         YAML_NODE_VALIDATE(child, MGRfrlxGetValidKeys, MGRfrlxGetValidValues);
         YAML_NODE_SET_FIELD(child, args, MGRfrlxSetFieldByName);
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      YAML_NODE_VALIDATE(parent, MGRfrlxGetValidKeys, MGRfrlxGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, MGRfrlxSetFieldByName);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
MGRgrlxSetArgsFromYAML(void *vargs, YAMLnode *parent)
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
               NestedKrylovSetArgsFromYAML(krylov, child);
            }
            continue;
         }

         YAML_NODE_VALIDATE(child, MGRgrlxGetValidKeys, MGRgrlxGetValidValues);
         YAML_NODE_SET_FIELD(child, args, MGRgrlxSetFieldByName);
      }
   }
   else
   {
      char *temp_key = strdup(parent->key);
      free(parent->key);
      parent->key = (char *)malloc(5 * sizeof(char));
      snprintf(parent->key, 5, "type");

      YAML_NODE_VALIDATE(parent, MGRgrlxGetValidKeys, MGRgrlxGetValidValues);
      YAML_NODE_SET_FIELD(parent, args, MGRgrlxSetFieldByName);

      free(parent->key);
      parent->key = strdup(temp_key);
      free(temp_key);
   }
}

/*-----------------------------------------------------------------------------
 * MGRclsGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
MGRclsGetValidValues(const char *key)
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
MGRfrlxGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"", -1},
                                {"none", -1},
                                {"single", 7},
                                {"jacobi", 7},
                                {"v(1,0)", 1},
                                {"amg", 2},
                                {"mgr", MGR_FRLX_TYPE_NESTED_MGR},
                                {"chebyshev", 16},
                                {"ilu", 32},
                                {"ge", 9},
                                {"spdirect", 29},
                                {"ge-piv", 99},
                                {"ge-inv", 199}};

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
MGRgrlxGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"", -1},         {"none", -1},    {"blk-jacobi", 0},
                                {"blk-gs", 1},    {"mixed-gs", 2}, {"amg", 20},
                                {"h-fgs", 3},     {"h-bgs", 4},    {"ch-gs", 5},
                                {"h-ssor", 6},    {"euclid", 8},   {"2stg-fgs", 11},
                                {"2stg-bgs", 12}, {"l1-hfgs", 13}, {"l1-hbgs", 14},
                                {"ilu", 16},      {"l1-hsgs", 88}};

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
MGRlvlGetValidValues(const char *key)
{
   if (!strcmp(key, "prolongation_type"))
   {
      static StrIntMap map[] = {
         {"injection", 0},     {"l1-jacobi", 1},   {"jacobi", 2},
         {"classical-mod", 3}, {"approx-inv", 4},  {"blk-jacobi", 12},
         {"blk-rowlump", 13},  {"blk-rowsum", 13}, {"blk-absrowsum", 14}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "restriction_type"))
   {
      static StrIntMap map[] = {{"injection", 0},   {"jacobi", 2},    {"approx-inv", 3},
                                {"blk-jacobi", 12}, {"cpr-like", 13}, {"columped", 14}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "coarse_level_type"))
   {
      static StrIntMap map[] = {{"rap", 0},
                                {"galerkin", 0},
                                {"non-galerkin", 1},
                                {"cpr-like-diag", 2},
                                {"cpr-like-bdiag", 3},
                                {"approx-inv", 4},
                                {"acc", 5}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "f_relaxation"))
   {
      return MGRfrlxGetValidValues("type");
   }
   if (!strcmp(key, "g_relaxation"))
   {
      return MGRgrlxGetValidValues("type");
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
MGRGetValidValues(const char *key)
{
   if (!strcmp(key, "relax_type"))
   {
      static StrIntMap map[] = {{"jacobi", 7},    {"h-fgs", 3},   {"h-bgs", 4},
                                {"ch-gs", 5},     {"h-ssor", 6},  {"hl1-ssor", 8},
                                {"l1-fgs", 13},   {"l1-bgs", 14}, {"chebyshev", 16},
                                {"l1-jacobi", 18}};

      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

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

void
MGRSetArgsFromYAML(void *vargs, YAMLnode *parent)
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
                                         MGRlvlSetFieldByName);
                     continue;
                  }

                  YAML_NODE_VALIDATE(great_grandchild, MGRlvlGetValidKeys,
                                     MGRlvlGetValidValues);

                  YAML_NODE_SET_FIELD(great_grandchild, &args->level[lvl],
                                      MGRlvlSetFieldByName);
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
         MGRclsSetArgsFromYAML(&args->coarsest_level, child);
      }
      else
      {
         YAML_NODE_VALIDATE(child, MGRGetValidKeys, MGRGetValidValues);
         YAML_NODE_SET_FIELD(child, args, MGRSetFieldByName);
      }
   }
}

/*-----------------------------------------------------------------------------
 * MGRConvertArgInt
 *-----------------------------------------------------------------------------*/

HYPRE_Int *
MGRConvertArgInt(MGR_args *args, const char *name)
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
         buf[i]         = (type == MGR_FRLX_TYPE_NESTED_MGR) ? 2 : type;
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

/*-----------------------------------------------------------------------------
 * MGRSetDofmap
 *-----------------------------------------------------------------------------*/

void
MGRSetDofmap(MGR_args *args, IntArray *dofmap)
{
   args->dofmap = dofmap;
}

void
MGRSetNearNullSpace(MGR_args *args, HYPRE_IJVector vec_nn)
{
   args->vec_nn = vec_nn;
}

void
MGRDestroyNestedSolverArgs(MGR_args *args)
{
   if (!args)
   {
      return;
   }

   for (int i = 0; i < MAX_MGR_LEVELS - 1; i++)
   {
      if (args->level[i].f_relaxation.mgr)
      {
         MGRDestroyNestedSolverArgs(args->level[i].f_relaxation.mgr);
         free(args->level[i].f_relaxation.mgr);
         args->level[i].f_relaxation.mgr = NULL;
      }

      if (args->level[i].f_relaxation.krylov)
      {
         NestedKrylovDestroy(args->level[i].f_relaxation.krylov);
         free(args->level[i].f_relaxation.krylov);
         args->level[i].f_relaxation.krylov     = NULL;
         args->level[i].f_relaxation.use_krylov = 0;
      }

      if (args->level[i].g_relaxation.krylov)
      {
         NestedKrylovDestroy(args->level[i].g_relaxation.krylov);
         free(args->level[i].g_relaxation.krylov);
         args->level[i].g_relaxation.krylov     = NULL;
         args->level[i].g_relaxation.use_krylov = 0;
      }
   }

   if (args->coarsest_level.krylov)
   {
      NestedKrylovDestroy(args->coarsest_level.krylov);
      free(args->coarsest_level.krylov);
      args->coarsest_level.krylov     = NULL;
      args->coarsest_level.use_krylov = 0;
   }
}

/*-----------------------------------------------------------------------------
 * MGRCreate
 *-----------------------------------------------------------------------------*/

void
MGRCreate(MGR_args *args, HYPRE_Solver *precon_ptr)
{
#if !HYPRE_CHECK_MIN_VERSION(21900, 0)
   (void)args;
   ErrorCodeSet(ERROR_INVALID_PRECON);
   ErrorMsgAdd("MGR requires hypre >= 2.19.0");
   *precon_ptr = NULL;
   return;
#else
   HYPRE_Solver precon          = NULL;
   HYPRE_Solver frelax          = NULL;
   HYPRE_Solver grelax          = NULL;
   HYPRE_Int   *dofmap_data     = NULL;
   IntArray    *dofmap          = NULL;
   HYPRE_Int   *label_present   = NULL;
   HYPRE_Int    num_dofs        = 0;
   HYPRE_Int    num_active_dofs = 0;
   HYPRE_Int    num_dofs_last   = 0;
   HYPRE_Int    num_levels      = 0;
   HYPRE_Int    num_c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int   *c_dofs[MAX_MGR_LEVELS - 1];
   HYPRE_Int   *inactive_dofs = NULL;
   HYPRE_Int    lvl = 0, i = 0;
   HYPRE_Int    j;

   /* Sanity checks */
   if (!args->dofmap)
   {
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
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
      num_active_dofs = (HYPRE_Int)present_labels;
   }

   /* Compute num_c_dofs and c_dofs */
   num_dofs_last = num_active_dofs;
   inactive_dofs = (HYPRE_Int *)calloc((size_t)num_dofs, sizeof(HYPRE_Int));
   if (!inactive_dofs)
   {
      free(label_present);
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate MGR inactive dof label mask");
      return;
   }
   for (i = 0; i < num_dofs; i++)
   {
      if (!label_present[i])
      {
         inactive_dofs[i] = 1;
      }
   }
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      c_dofs[lvl]     = (HYPRE_Int *)malloc((size_t)num_dofs * sizeof(HYPRE_Int));
      num_c_dofs[lvl] = (HYPRE_Int)((size_t)num_dofs_last - args->level[lvl].f_dofs.size);

      for (i = 0; i < (int)args->level[lvl].f_dofs.size; i++)
      {
         HYPRE_Int dof_label = args->level[lvl].f_dofs.data[i];
         if (dof_label < 0 || dof_label >= num_dofs)
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Invalid MGR level %d f_dofs label %d (valid range: [0,%d])",
                        (int)lvl, (int)dof_label, (int)num_dofs - 1);
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
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("MGR level %d f_dofs label %d is not present in dofmap", (int)lvl,
                        (int)dof_label);
            free(label_present);
            free(inactive_dofs);
            for (HYPRE_Int k = 0; k <= lvl; k++)
            {
               free(c_dofs[k]);
            }
            return;
         }
         if (inactive_dofs[dof_label])
         {
            ErrorCodeSet(ERROR_INVALID_VAL);
            ErrorMsgAdd("Duplicate/previously eliminated MGR f_dofs label %d at level %d",
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
         --num_dofs_last;
      }

      for (i = 0, j = 0; i < num_dofs; i++)
      {
         if (label_present[i] && !inactive_dofs[i])
         {
            c_dofs[lvl][j++] = i;
         }
      }
   }
   free(label_present);
   label_present = NULL;

   /* Set dofmap_data */
   if (TYPES_MATCH(HYPRE_Int, int))
   {
      dofmap_data = (HYPRE_Int *)dofmap->data;
   }
   else
   {
      dofmap_data = (HYPRE_Int *)malloc(dofmap->size * sizeof(HYPRE_Int));
      for (i = 0; i < (int)dofmap->size; i++)
      {
         dofmap_data[i] = (HYPRE_Int)dofmap->data[i];
      }
   }

   /* Config preconditioner */
   HYPRE_MGRCreate(&precon);
   HYPRE_MGRSetCpointsByPointMarkerArray(precon, num_dofs, num_levels - 1, num_c_dofs,
                                         c_dofs, dofmap_data);
   HYPRE_MGRSetNonCpointsToFpoints(precon, args->non_c_to_f);
   HYPRE_MGRSetMaxIter(precon, args->max_iter);
   HYPRE_MGRSetTol(precon, args->tolerance);
   HYPRE_MGRSetPrintLevel(precon, args->print_level);
#if HYPRE_CHECK_MIN_VERSION(22000, 0)
   HYPRE_MGRSetTruncateCoarseGridThreshold(precon, args->coarse_th);
#endif
   HYPRE_MGRSetRelaxType(precon, args->relax_type); /* TODO: we shouldn't need this */

   /* Set level parameters */
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_MGRSetLevelFRelaxType(precon, MGRConvertArgInt(args, "f_relaxation:type"));
   HYPRE_MGRSetLevelNumRelaxSweeps(precon,
                                   MGRConvertArgInt(args, "f_relaxation:num_sweeps"));
   HYPRE_MGRSetLevelSmoothType(precon, MGRConvertArgInt(args, "g_relaxation:type"));
   HYPRE_MGRSetLevelSmoothIters(precon,
                                MGRConvertArgInt(args, "g_relaxation:num_sweeps"));
   HYPRE_MGRSetLevelInterpType(precon, MGRConvertArgInt(args, "prolongation_type"));
   HYPRE_MGRSetLevelRestrictType(precon, MGRConvertArgInt(args, "restriction_type"));
   HYPRE_MGRSetCoarseGridMethod(precon, MGRConvertArgInt(args, "coarse_level_type"));
#endif

   /* Config f-relaxation at each MGR level */
   for (i = 0; i < num_levels - 1; i++)
   {
      if (args->level[i].f_relaxation.use_krylov && args->level[i].f_relaxation.krylov)
      {
         HYPRE_Solver wrapper = NULL;
         NestedKrylovCreate(MPI_COMM_WORLD, args->level[i].f_relaxation.krylov,
                            args->dofmap, args->vec_nn, &wrapper);
         if (ErrorCodeActive())
         {
            return;
         }
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRSetFSolverAtLevel(precon, wrapper, i);
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("Nested Krylov F-relaxation requires hypre >= 2.31.0");
         return;
#endif
      }
      else if (args->level[i].f_relaxation.type == 2)
      {
         AMGCreate(&args->level[i].f_relaxation.amg, &frelax);
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
#else
         (void)precon;
         (void)i;
#endif
         args->frelax[i] = frelax;
      }
      else if (args->level[i].f_relaxation.type == MGR_FRLX_TYPE_NESTED_MGR)
      {
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
         if (i != 0)
         {
            ErrorCodeSet(ERROR_INVALID_PRECON);
            ErrorMsgAdd(
               "Nested MGR F-relaxation is only supported at MGR level 0 by hypre");
            return;
         }

         MGR_args      *nested_args    = args->level[i].f_relaxation.mgr;
         HYPRE_Solver   frelax_wrapper = NULL;
         IntArray      *nested_dofmap  = NULL;
         IntArray      *saved_dofmap   = NULL;
         HYPRE_IJVector saved_vec_nn   = NULL;

         if (!nested_args)
         {
            ErrorCodeSet(ERROR_INVALID_PRECON);
            ErrorMsgAdd("MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
            return;
         }

         nested_dofmap =
            MGRBuildProjectedFRelaxDofmap(args->dofmap, &args->level[i].f_dofs);
         if (ErrorCodeActive() || !nested_dofmap)
         {
            IntArrayDestroy(&nested_dofmap);
            return;
         }

         saved_dofmap        = nested_args->dofmap;
         saved_vec_nn        = nested_args->vec_nn;
         nested_args->dofmap = nested_dofmap;
         nested_args->vec_nn = NULL;
         MGRCreate(nested_args, &frelax);
         nested_args->dofmap = saved_dofmap;
         nested_args->vec_nn = saved_vec_nn;
         IntArrayDestroy(&nested_dofmap);
         if (ErrorCodeActive())
         {
            return;
         }

         frelax_wrapper = MGRNestedFRelaxWrapperCreate(frelax);
         if (ErrorCodeActive() || !frelax_wrapper)
         {
            HYPRE_MGRDestroy(frelax);
            return;
         }
         HYPRE_MGRSetFSolverAtLevel(precon, frelax_wrapper, i);
         args->frelax[i] = frelax_wrapper;
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("Nested MGR F-relaxation requires hypre >= 3.1.0 (develop >= 5)");
         return;
#endif
      }
#if defined(HYPRE_USING_DSUPERLU)
      else if (args->level[i].f_relaxation.type == 29)
      {
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         HYPRE_MGRDirectSolverCreate(&frelax);
         if (!frelax)
         {
            ErrorCodeSet(ERROR_INVALID_PRECON);
            ErrorMsgAdd(
               "MGR F-relaxation 'spdirect' unavailable: direct solver creation failed");
            return;
         }
         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
         args->frelax[i] = frelax;
#else
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("MGR F-relaxation 'spdirect' requires hypre >= 2.31.0");
         return;
#endif
      }
#else
      else if (args->level[i].f_relaxation.type == 29)
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("MGR F-relaxation 'spdirect' requires hypre built with DSUPERLU");
         return;
      }
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
      else if (args->level[i].f_relaxation.type == 32)
      {
         ILUCreate(&args->level[i].f_relaxation.ilu, &frelax);

         HYPRE_MGRSetFSolverAtLevel(precon, frelax, i);
         args->frelax[i] = frelax;
      }
#endif
   }

   /* Config global relaxation at each MGR level */
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   for (i = 0; i < num_levels - 1; i++)
   {
      if (args->level[i].g_relaxation.use_krylov && args->level[i].g_relaxation.krylov)
      {
         HYPRE_Solver wrapper = NULL;
         NestedKrylovCreate(MPI_COMM_WORLD, args->level[i].g_relaxation.krylov,
                            args->dofmap, args->vec_nn, &wrapper);
         if (ErrorCodeActive())
         {
            return;
         }
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, wrapper, i);
      }
      else if (args->level[i].g_relaxation.type == 20)
      {
         AMGCreate(&args->level[i].g_relaxation.amg, &grelax);
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, i);
         args->grelax[i] = grelax;
      }
      else if (args->level[i].g_relaxation.type == 16)
      {
         ILUCreate(&args->level[i].g_relaxation.ilu, &grelax);
         HYPRE_MGRSetGlobalSmootherAtLevel(precon, grelax, i);
         args->grelax[i] = grelax;
      }
   }
#endif

   if (args->coarsest_level.use_krylov && args->coarsest_level.krylov)
   {
      HYPRE_Solver wrapper = NULL;
      NestedKrylovCreate(MPI_COMM_WORLD, args->coarsest_level.krylov, args->dofmap,
                         args->vec_nn, &wrapper);
      if (ErrorCodeActive())
      {
         return;
      }
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
      HYPRE_MGRSetCoarseSolver(precon, MGRBaseParSolverSolve, MGRBaseParSolverSetup,
                               wrapper);
#else
      ErrorCodeSet(ERROR_INVALID_PRECON);
      ErrorMsgAdd("Nested Krylov coarsest solver requires hypre >= 3.1.0");
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
      if (args->coarsest_level.type == 0 && args->coarsest_level.amg.max_iter < 1)
      {
         args->coarsest_level.amg.max_iter = 1;
      }
      else if (args->coarsest_level.type == 32 && args->coarsest_level.ilu.max_iter < 1)
      {
         args->coarsest_level.ilu.max_iter = 1;
      }

      /* Config coarsest level solver */
      if (args->coarsest_level.type == 0)
      {
         AMGCreate(&args->coarsest_level.amg, &args->csolver);
         args->csolver_type = 0;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                                  args->csolver);
      }
      else if (args->coarsest_level.type == 32)
      {
         ILUCreate(&args->coarsest_level.ilu, &args->csolver);
         args->csolver_type = 32;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_ILUSolve, HYPRE_ILUSetup, args->csolver);
      }
#ifdef HYPRE_USING_DSUPERLU
      else if (args->coarsest_level.type == 29)
      {
         HYPRE_MGRDirectSolverCreate(&args->csolver);
         if (!args->csolver)
         {
            ErrorCodeSet(ERROR_INVALID_PRECON);
            ErrorMsgAdd(
               "MGR coarsest_level 'spdirect' unavailable: direct solver creation "
               "failed");
            return;
         }
         args->csolver_type = 29;
         HYPRE_MGRSetCoarseSolver(precon, HYPRE_MGRDirectSolverSolve,
                                  HYPRE_MGRDirectSolverSetup, args->csolver);
      }
#else
      else if (args->coarsest_level.type == 29)
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         ErrorMsgAdd("MGR coarsest_level 'spdirect' requires hypre built with DSUPERLU");
         return;
      }
#endif
   }

#if HYPRE_CHECK_MIN_VERSION(23100, 11)
   HYPRE_MGRSetNonGalerkinMaxElmts(precon, args->nonglk_max_elmts);
#endif

   /* Set output pointer */
   *precon_ptr = precon;

   /* Free memory */
   free(inactive_dofs);
   for (lvl = 0; lvl < num_levels - 1; lvl++)
   {
      free(c_dofs[lvl]);
   }
   if ((void *)dofmap_data != (void *)dofmap->data)
   {
      free(dofmap_data);
   }

   /* Silence any hypre errors. TODO: improve error handling */
   HYPRE_ClearAllErrors();
#endif
}
