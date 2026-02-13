/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "args.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "field.h"
#include "gen_macros.h"
#include "presets.h"
#include "scaling.h"
#include "stats.h"
#include "utils.h"
#include "yaml.h"

/*-----------------------------------------------------------------------------
 * General args helpers (schema-driven parsing)
 *-----------------------------------------------------------------------------*/

static void
FieldTypePoolGBToBytesSet(void *field, const YAMLnode *node)
{
   double gb          = strtod(node->mapped_val, NULL);
   *((double *)field) = gb * GB_TO_BYTES;
}

#define General_FIELDS(_prefix)                                               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, warmup, FieldTypeIntSet)                   \
   ADD_FIELD_OFFSET_ENTRY(_prefix, statistics, FieldTypeIntSet)               \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_config_params, FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_millisec, FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, exec_policy, FieldTypeIntSet)              \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_repetitions, FieldTypeIntSet)          \
   ADD_FIELD_OFFSET_ENTRY(_prefix, dev_pool_size, FieldTypePoolGBToBytesSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, uvm_pool_size, FieldTypePoolGBToBytesSet)  \
   ADD_FIELD_OFFSET_ENTRY(_prefix, host_pool_size, FieldTypePoolGBToBytesSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pinned_pool_size, FieldTypePoolGBToBytesSet)

#define General_NUM_FIELDS \
   (sizeof(General_field_offset_map) / sizeof(General_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(General) // LCOV_EXCL_LINE

StrIntMapArray
GeneralGetValidValues(const char *key)
{
   if (!strcmp(key, "warmup") || !strcmp(key, "print_config_params") ||
       !strcmp(key, "use_millisec"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "statistics"))
   {
      static StrIntMap map[] = {{"off", 0},  {"no", 0}, {"false", 0},
                                {"0", 0},    {"on", 1}, {"yes", 1},
                                {"true", 1}, {"1", 1},  {"2", 2}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "exec_policy"))
   {
      static StrIntMap map[] = {{"host", 0}, {"device", 1}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }

   return STR_INT_MAP_ARRAY_VOID();
}

void
GeneralSetDefaultArgs(General_args *args)
{
   args->warmup              = 0;
   args->statistics          = 1;
   args->print_config_params = 1;
   args->use_millisec        = 0;
#ifdef HYPRE_USING_GPU
   args->exec_policy = 1;
#else
   args->exec_policy = 0;
#endif
   args->num_repetitions  = 1;
   args->dev_pool_size    = 8.0 * GB_TO_BYTES;
   args->uvm_pool_size    = 8.0 * GB_TO_BYTES;
   args->host_pool_size   = 8.0 * GB_TO_BYTES;
   args->pinned_pool_size = 0.5 * GB_TO_BYTES;
}

void
InputArgsCreate(bool lib_mode, input_args **iargs_ptr)
{
   input_args *iargs = (input_args *)malloc(sizeof(input_args));

   /* Set general default options */
   GeneralSetDefaultArgs(&iargs->general);
   iargs->general.print_config_params = !lib_mode;

   /* Set default Linear System options */
   LinearSystemSetDefaultArgs(&iargs->ls);
   iargs->ls.exec_policy = iargs->general.exec_policy;

   /* Set default preconditioner and solver */
   iargs->solver_method = SOLVER_PCG;
   iargs->precon_method = PRECON_BOOMERAMG;

   /* Set default scaling */
   ScalingSetDefaultArgs(&iargs->scaling);

   /* Initialize preconditioner variants (default: single variant) */
   iargs->num_precon_variants   = 1;
   iargs->active_precon_variant = 0;
   iargs->precon_methods        = NULL;
   iargs->precon_variants       = NULL;

   *iargs_ptr = iargs;
}

/*-----------------------------------------------------------------------------
 * InputArgsDestroy
 *-----------------------------------------------------------------------------*/

void
InputArgsDestroy(input_args **iargs_ptr)
{
   if (*iargs_ptr)
   {
      input_args *iargs = *iargs_ptr;
      if (iargs->precon_methods && iargs->precon_variants)
      {
         for (int i = 0; i < iargs->num_precon_variants; i++)
         {
            if (iargs->precon_methods[i] == PRECON_MGR)
            {
               MGRDestroyNestedKrylovArgs(&iargs->precon_variants[i].mgr);
            }
         }
      }
      if (iargs->precon_methods)
      {
         free(iargs->precon_methods);
         iargs->precon_methods = NULL;
      }
      if (iargs->precon_variants)
      {
         free(iargs->precon_variants);
         iargs->precon_variants = NULL;
      }
      if (iargs->scaling.custom_values)
      {
         DoubleArrayDestroy(&iargs->scaling.custom_values);
      }
      free(*iargs_ptr);
      *iargs_ptr = NULL;
   }
}

/*-----------------------------------------------------------------------------
 * InputArgsParseGeneral
 *-----------------------------------------------------------------------------*/

void
InputArgsParseGeneral(input_args *iargs, YAMLtree *tree)
{
   YAMLnode *parent = YAMLnodeFindByKey(tree->root, "general");
   if (!parent)
   {
      /* The \"general\" key is optional */
      iargs->ls.exec_policy = iargs->general.exec_policy;
      return;
   }

   YAML_NODE_SET_VALID(parent);
   GeneralSetArgsFromYAML(&iargs->general, parent);

   /* Mirror general exec policy into linear-system args for downstream use */
   iargs->ls.exec_policy = iargs->general.exec_policy;
}

/*-----------------------------------------------------------------------------
 * InputArgsParseLinearSystem
 *-----------------------------------------------------------------------------*/

void
InputArgsParseLinearSystem(input_args *iargs, YAMLtree *tree)
{
   const char key[]  = {"linear_system"};
   YAMLnode  *parent = YAMLnodeFindByKey(tree->root, key);

   if (!parent)
   {
      // TODO: Add "library" mode to skip the following checks
      // ErrorCodeSet(ERROR_MISSING_KEY);
      // ErrorMsgAddMissingKey(key);
      return;
   }
   YAML_NODE_SET_VALID_IF_NO_VAL(parent);

   if (YAML_NODE_GET_VALIDITY(parent) == YAML_NODE_UNEXPECTED_VAL)
   {
      ErrorMsgAddUnexpectedVal(key);
   }

   LinearSystemSetArgsFromYAML(&iargs->ls, parent);
   LinearSystemSetNumSystems(&iargs->ls);
}

/*-----------------------------------------------------------------------------
 * InputArgsParseSolver
 *-----------------------------------------------------------------------------*/

void
InputArgsParseSolver(input_args *iargs, YAMLtree *tree)
{
   YAMLnode *parent       = NULL;
   YAMLnode *scaling_node = NULL;

   parent = YAMLnodeFindByKey(tree->root, "solver");
   if (!parent)
   {
      // TODO: Add "library" mode to skip the following checks
      // ErrorCodeSet(ERROR_MISSING_KEY);
      // ErrorMsgAddMissingKey("solver");
      return;
   }
   YAML_NODE_SET_VALID(parent);

   /* Extract scaling node before parsing solver to avoid schema conflicts */
   if (parent->children)
   {
      YAMLnode *child = parent->children;
      YAMLnode *prev  = NULL;
      while (child)
      {
         if (!strcmp(child->key, "scaling"))
         {
            /* Detach scaling node from parent's children list */
            if (prev)
            {
               prev->next = child->next;
            }
            else
            {
               parent->children = child->next;
            }
            scaling_node       = child;
            scaling_node->next = NULL;
            break;
         }
         prev  = child;
         child = child->next;
      }
   }

   /* Check if the solver type was set with a single (key, val) pair */
   if (!strcmp(parent->val, ""))
   {
      /* Check if a solver type was set */
      if (!parent->children)
      {
         ErrorCodeSet(ERROR_MISSING_SOLVER);
         return;
      }

      /* Check if more than one solver type was set (this is not supported!) */
      if (parent->children->next)
      {
         ErrorCodeSet(ERROR_EXTRA_KEY);
         ErrorMsgAddExtraKey(parent->children->next->key);
         return;
      }

      iargs->solver_method = (solver_t)StrIntMapArrayGetImage(SolverGetValidTypeIntMap(),
                                                              parent->children->key);

      SolverSetArgsFromYAML(&iargs->solver, parent);
   }
   else
   {
      iargs->solver_method =
         (solver_t)StrIntMapArrayGetImage(SolverGetValidTypeIntMap(), parent->val);

      /* Value-only form (e.g., `solver: gmres`): use per-method defaults. */
      SolverArgsSetDefaultsForMethod(iargs->solver_method, &iargs->solver);

      /* Preserve legacy behavior: suppress solver print by default for value-only form.
       */
      switch (iargs->solver_method)
      {
         case SOLVER_PCG:
            iargs->solver.pcg.print_level = 0;
            break;
         case SOLVER_GMRES:
            iargs->solver.gmres.print_level = 0;
            break;
         case SOLVER_FGMRES:
            iargs->solver.fgmres.print_level = 0;
            break;
         case SOLVER_BICGSTAB:
            iargs->solver.bicgstab.print_level = 0;
            break;
         default:
            break;
      }
   }

   /* Parse scaling if present */
   if (scaling_node)
   {
      ScalingSetArgsFromYAML(&iargs->scaling, scaling_node);
      /* Mark scaling node as valid to avoid validation errors */
      YAML_NODE_SET_VALID(scaling_node);
      /* Reattach to parent for tree validation (but after parsing) */
      scaling_node->next = parent->children;
      parent->children   = scaling_node;
   }
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
PreconParseVariantWrapped(precon_args *dst, precon_t method,
                          const YAMLnode *precon_parent, const char *type_key,
                          int type_level, YAMLnode *type_children)
{
   PreconArgsSetDefaultsForMethod(method, dst);

   YAMLnode fake_type = {0};
   /* IMPORTANT: YAMLSetArgsGeneric may free/replace parent->key for flat-value nodes.
    * Never point into another YAML tree's storage here. */
   fake_type.key      = strdup(type_key ? type_key : "");
   fake_type.val      = strdup("");
   fake_type.level    = type_level;
   fake_type.valid    = YAML_NODE_VALID;
   fake_type.children = type_children;
   fake_type.next     = NULL;

   YAMLnode fake_parent = {0};
   fake_parent.key      = "preconditioner";
   fake_parent.val      = "";
   fake_parent.level    = precon_parent ? precon_parent->level : 0;
   fake_parent.valid    = YAML_NODE_VALID;
   fake_parent.children = &fake_type;
   fake_parent.next     = NULL;

   PreconSetArgsFromYAML(dst, &fake_parent);

   /* Clean up mapped_val allocated during validation on fake nodes */
   free(fake_type.mapped_val);
   fake_type.mapped_val = NULL;
   free(fake_type.key);
   fake_type.key = NULL;
   free(fake_type.val);
   fake_type.val = NULL;
   free(fake_parent.mapped_val);
   fake_parent.mapped_val = NULL;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

static void
PreconPresetBuildArgs(const char *preset_name, precon_t *method_out,
                      precon_args *args_out)
{
   if (!method_out || !args_out)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Preset output arguments must be non-NULL");
      return;
   }

   const hypredrv_Preset *preset = hypredrv_PresetFind(preset_name);
   if (!preset)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Unknown preconditioner preset: '%s'", preset_name ? preset_name : "");
      char *help = hypredrv_PresetHelp();
      if (help)
      {
         ErrorMsgAdd("%s", help);
         free(help);
      }
      return;
   }

   char *text = strdup(preset->text);
   if (!text)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate preset YAML text");
      return;
   }

   YAMLtree *preset_tree = NULL;
   YAMLtreeBuild(2, text, &preset_tree);
   free(text);

   if (!preset_tree || !preset_tree->root || !preset_tree->root->children ||
       preset_tree->root->children->next)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Preset '%s' must expand to a single preconditioner type",
                  preset->name);
      YAMLtreeDestroy(&preset_tree);
      return;
   }

   YAMLnode *type_node = preset_tree->root->children;
   if (!StrIntMapArrayDomainEntryExists(PreconGetValidTypeIntMap(), type_node->key))
   {
      ErrorCodeSet(ERROR_INVALID_KEY);
      ErrorMsgAdd("Unknown preconditioner type: '%s'", type_node->key);
      YAMLtreeDestroy(&preset_tree);
      return;
   }

   *method_out =
      (precon_t)StrIntMapArrayGetImage(PreconGetValidTypeIntMap(), type_node->key);
   PreconParseVariantWrapped(args_out, *method_out, preset_tree->root, type_node->key,
                             type_node->level, type_node->children);

   YAMLtreeDestroy(&preset_tree);
   return;
}

typedef enum
{
   PRECON_PARSE_NOT_HANDLED = 0,
   PRECON_PARSE_SUCCESS     = 1,
   PRECON_PARSE_ERROR       = -1
} PreconParseResult;

typedef struct
{
   precon_t    *methods;
   precon_args *variants;
   YAMLnode   **precon_items;
   YAMLnode   **seq_items;
   int          num_variants;
   int          parsed_variants;
} PreconParseContext;

static void
PreconParseContextCleanup(PreconParseContext *ctx)
{
   if (!ctx)
   {
      return;
   }

   if (ctx->methods && ctx->variants)
   {
      int n = ctx->parsed_variants;
      if (n > ctx->num_variants)
      {
         n = ctx->num_variants;
      }

      for (int i = 0; i < n; i++)
      {
         if (ctx->methods[i] == PRECON_MGR)
         {
            MGRDestroyNestedKrylovArgs(&ctx->variants[i].mgr);
         }
      }
   }

   free(ctx->precon_items);
   free(ctx->seq_items);
   free(ctx->methods);
   free(ctx->variants);

   ctx->methods         = NULL;
   ctx->variants        = NULL;
   ctx->precon_items    = NULL;
   ctx->seq_items       = NULL;
   ctx->num_variants    = 0;
   ctx->parsed_variants = 0;
}

static int
PreconParseContextAllocVariants(PreconParseContext *ctx, int num_variants,
                                const char *error_msg)
{
   if (!ctx || num_variants <= 0)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid preconditioner parse allocation request");
      return 0;
   }

   ctx->methods  = (precon_t *)malloc(sizeof(precon_t) * (size_t)num_variants);
   ctx->variants = (precon_args *)malloc(sizeof(precon_args) * (size_t)num_variants);

   if (!ctx->methods || !ctx->variants)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("%s", error_msg);
      return 0;
   }

   ctx->num_variants    = num_variants;
   ctx->parsed_variants = 0;
   for (int i = 0; i < num_variants; i++)
   {
      ctx->methods[i] = PRECON_NONE;
   }

   return 1;
}

static void
InputArgsPreconVariant0Activate(input_args *iargs, PreconParseContext *ctx,
                                int num_variants)
{
   iargs->num_precon_variants   = num_variants;
   iargs->precon_methods        = ctx->methods;
   iargs->precon_variants       = ctx->variants;
   iargs->active_precon_variant = 0;
   iargs->precon_method         = ctx->methods[0];
   iargs->precon                = ctx->variants[0];

   ctx->methods         = NULL;
   ctx->variants        = NULL;
   ctx->num_variants    = 0;
   ctx->parsed_variants = 0;
}

static int
PreconParseMethodResolve(const char *name, int error_code, precon_t *method_out)
{
   if (!StrIntMapArrayDomainEntryExists(PreconGetValidTypeIntMap(), name))
   {
      ErrorCodeSet(error_code);
      ErrorMsgAdd("Unknown preconditioner type: '%s'", name);
      return 0;
   }

   *method_out = (precon_t)StrIntMapArrayGetImage(PreconGetValidTypeIntMap(), name);
   return 1;
}

static int
PreconParsePresetNode(const YAMLnode *preset_node, precon_t *method_out,
                      precon_args *args_out)
{
   if (!strcmp(preset_node->val, ""))
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Preconditioner preset name is missing");
      return 0;
   }

   char *preset_name = strdup(preset_node->val);
   if (!preset_name)
   {
      ErrorCodeSet(ERROR_ALLOCATION);
      ErrorMsgAdd("Failed to allocate preset name");
      return 0;
   }

   PreconPresetBuildArgs(preset_name, method_out, args_out);
   free(preset_name);

   return !ErrorCodeGet();
}

static PreconParseResult
InputArgsParsePreconValueOnly(input_args *iargs, YAMLnode *parent,
                              PreconParseContext *ctx)
{
   if (!strcmp(parent->val, ""))
   {
      return PRECON_PARSE_NOT_HANDLED;
   }

   precon_t method = PRECON_NONE;
   if (!PreconParseMethodResolve(parent->val, ERROR_INVALID_VAL, &method))
   {
      return PRECON_PARSE_ERROR;
   }

   if (!PreconParseContextAllocVariants(ctx, 1, "Failed to allocate preconditioner args"))
   {
      return PRECON_PARSE_ERROR;
   }

   ctx->methods[0] = method;
   PreconArgsSetDefaultsForMethod(method, &ctx->variants[0]);
   ctx->parsed_variants = 1;

   InputArgsPreconVariant0Activate(iargs, ctx, 1);
   YAML_NODE_SET_VALID(parent);
   return PRECON_PARSE_SUCCESS;
}

static PreconParseResult
InputArgsParsePreconRootSequence(input_args *iargs, YAMLnode *parent,
                                 PreconParseContext *ctx)
{
   int num_variants = YAMLnodeCollectSequenceItems(parent, &ctx->precon_items);
   if (num_variants <= 0)
   {
      return PRECON_PARSE_NOT_HANDLED;
   }

   if (!PreconParseContextAllocVariants(ctx, num_variants,
                                        "Failed to allocate preconditioner variants"))
   {
      return PRECON_PARSE_ERROR;
   }

   for (int vi = 0; vi < num_variants; vi++)
   {
      YAMLnode *item = ctx->precon_items[vi];
      if (!item->children || item->children->next)
      {
         ErrorCodeSet(ERROR_INVALID_KEY);
         ErrorMsgAdd("Each preconditioner variant must contain exactly one type key");
         return PRECON_PARSE_ERROR;
      }

      YAMLnode *type   = item->children;
      precon_t  method = PRECON_NONE;

      if (!strcmp(type->key, "preset"))
      {
         if (!PreconParsePresetNode(type, &method, &ctx->variants[vi]))
         {
            return PRECON_PARSE_ERROR;
         }
      }
      else
      {
         if (!PreconParseMethodResolve(type->key, ERROR_INVALID_KEY, &method))
         {
            return PRECON_PARSE_ERROR;
         }

         ctx->methods[vi]     = method;
         ctx->parsed_variants = vi + 1;
         PreconParseVariantWrapped(&ctx->variants[vi], method, parent, type->key,
                                   type->level, type->children);
         if (ErrorCodeGet())
         {
            return PRECON_PARSE_ERROR;
         }
      }

      ctx->methods[vi]     = method;
      ctx->parsed_variants = vi + 1;
      YAML_NODE_SET_VALID(item);
      YAML_NODE_SET_VALID(type);
   }

   InputArgsPreconVariant0Activate(iargs, ctx, num_variants);
   return PRECON_PARSE_SUCCESS;
}

static PreconParseResult
InputArgsParsePreconTypedBlock(input_args *iargs, YAMLnode *parent,
                               PreconParseContext *ctx)
{
   if (!parent->children)
   {
      ErrorCodeSet(ERROR_MISSING_PRECON);
      return PRECON_PARSE_ERROR;
   }
   if (parent->children->next)
   {
      ErrorCodeSet(ERROR_EXTRA_KEY);
      ErrorMsgAddExtraKey(parent->children->next->key);
      return PRECON_PARSE_ERROR;
   }

   YAMLnode *type_node = parent->children;
   if (!strcmp(type_node->key, "preset"))
   {
      precon_t method = PRECON_NONE;
      if (!PreconParseContextAllocVariants(ctx, 1,
                                           "Failed to allocate preconditioner args"))
      {
         return PRECON_PARSE_ERROR;
      }
      if (!PreconParsePresetNode(type_node, &method, &ctx->variants[0]))
      {
         return PRECON_PARSE_ERROR;
      }

      ctx->methods[0]      = method;
      ctx->parsed_variants = 1;

      InputArgsPreconVariant0Activate(iargs, ctx, 1);
      YAML_NODE_SET_VALID(parent);
      YAML_NODE_SET_VALID(type_node);
      return PRECON_PARSE_SUCCESS;
   }

   precon_t method = PRECON_NONE;
   if (!PreconParseMethodResolve(type_node->key, ERROR_INVALID_KEY, &method))
   {
      return PRECON_PARSE_ERROR;
   }

   YAML_NODE_SET_VALID(type_node);
   int num_variants = YAMLnodeCollectSequenceItems(type_node, &ctx->seq_items);
   if (num_variants > 0)
   {
      if (!PreconParseContextAllocVariants(ctx, num_variants,
                                           "Failed to allocate preconditioner variants"))
      {
         return PRECON_PARSE_ERROR;
      }

      for (int vi = 0; vi < num_variants; vi++)
      {
         YAMLnode *seq_item   = ctx->seq_items[vi];
         ctx->methods[vi]     = method;
         ctx->parsed_variants = vi + 1;
         PreconParseVariantWrapped(&ctx->variants[vi], method, parent, type_node->key,
                                   type_node->level, seq_item->children);
         if (ErrorCodeGet())
         {
            return PRECON_PARSE_ERROR;
         }
         YAML_NODE_SET_VALID(seq_item);
      }

      InputArgsPreconVariant0Activate(iargs, ctx, num_variants);
      YAML_NODE_SET_VALID(type_node);
      return PRECON_PARSE_SUCCESS;
   }

   if (!PreconParseContextAllocVariants(ctx, 1, "Failed to allocate preconditioner args"))
   {
      return PRECON_PARSE_ERROR;
   }

   ctx->methods[0] = method;
   PreconArgsSetDefaultsForMethod(method, &ctx->variants[0]);
   ctx->parsed_variants = 1;
   PreconSetArgsFromYAML(&ctx->variants[0], parent);
   if (ErrorCodeGet())
   {
      return PRECON_PARSE_ERROR;
   }

   InputArgsPreconVariant0Activate(iargs, ctx, 1);
   return PRECON_PARSE_SUCCESS;
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

void
InputArgsParsePrecon(input_args *iargs, YAMLtree *tree)
{
   YAMLnode *parent = YAMLnodeFindByKey(tree->root, "preconditioner");
   if (!parent)
   {
      ErrorCodeSet(ERROR_MISSING_KEY);
      ErrorMsgAddMissingKey("preconditioner");
      return;
   }
   YAML_NODE_SET_VALID(parent);
   PreconParseContext ctx = {0};
   PreconParseResult  result;

   result = InputArgsParsePreconValueOnly(iargs, parent, &ctx);
   if (result != PRECON_PARSE_NOT_HANDLED)
   {
      PreconParseContextCleanup(&ctx);
      return;
   }

   result = InputArgsParsePreconRootSequence(iargs, parent, &ctx);
   if (result != PRECON_PARSE_NOT_HANDLED)
   {
      PreconParseContextCleanup(&ctx);
      return;
   }

   InputArgsParsePreconTypedBlock(iargs, parent, &ctx);
   PreconParseContextCleanup(&ctx);
}

/*-----------------------------------------------------------------------------
 * InputArgsApplyPreconPreset
 *-----------------------------------------------------------------------------*/

void
InputArgsApplyPreconPreset(input_args *iargs, const char *preset, int variant_idx)
{
   precon_t    method = PRECON_NONE;
   precon_args args;

   if (!iargs || !preset)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Input args and preset name must be non-NULL");
      return;
   }

   if (variant_idx < 0 || variant_idx >= iargs->num_precon_variants)
   {
      ErrorCodeSet(ERROR_INVALID_VAL);
      ErrorMsgAdd("Invalid preconditioner variant index: %d (valid range: 0-%d)",
                  variant_idx, iargs->num_precon_variants - 1);
      return;
   }

   if (!iargs->precon_methods || !iargs->precon_variants)
   {
      iargs->precon_methods =
         (precon_t *)malloc(sizeof(precon_t) * (size_t)iargs->num_precon_variants);
      iargs->precon_variants =
         (precon_args *)malloc(sizeof(precon_args) * (size_t)iargs->num_precon_variants);
      if (!iargs->precon_methods || !iargs->precon_variants)
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("Failed to allocate preconditioner variants");
         return;
      }
   }

   PreconPresetBuildArgs(preset, &method, &args);

   iargs->precon_methods[variant_idx]  = method;
   iargs->precon_variants[variant_idx] = args;
   iargs->precon_method                = method;
   iargs->precon                       = args;

   return;
}

/*-----------------------------------------------------------------------------
 * InputArgsRead
 *-----------------------------------------------------------------------------*/

void
InputArgsRead(MPI_Comm comm, char *filename, int *base_indent_ptr, char **text_ptr)
{
   size_t text_size   = 0;
   int    base_indent = -1;
   char  *text        = NULL;
   char  *dirname     = NULL;
   char  *basename    = NULL;
   int    myid        = 0;

   MPI_Comm_rank(comm, &myid);

   /* Rank 0: Check if file exists */
   if (!myid)
   {
      if (filename == NULL)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAdd("Configuration filename is NULL");
         return;
      }
      FILE *fp = fopen(filename, "r");
      if (!fp)
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAdd("Configuration file not found: '%s'", filename);
      }
      else
      {
         fclose(fp);
      }
   }

   /* All ranks return if there was an error on rank 0 */
   if (DistributedErrorCodeActive(comm))
   {
      return;
   }

   /* Split filename into dirname and basename */
   SplitFilename(filename, &dirname, &basename);

   /* Rank 0: Expand text from base file */
   if (!myid)
   {
      YAMLtextRead(dirname, basename, 0, &base_indent, &text_size, &text);
   }
   if (DistributedErrorCodeActive(comm))
   {
      /* Free allocated memory before returning on error */
      free(dirname);
      free(basename);
      return;
   }

   /* Broadcast the text size and base indentation */
   MPI_Bcast(&base_indent, 1, MPI_INT, 0, comm);
   MPI_Bcast(&text_size, 1, MPI_UNSIGNED_LONG, 0, comm);

   /* Broadcast the text */
   MPI_Comm_rank(comm, &myid);
   if (myid)
   {
      text = (char *)malloc(text_size + 1);
   } /* +1: for null terminator */
   MPI_Bcast(text, (int)text_size, MPI_CHAR, 0, comm);

   /* Make sure null terminator is in the right place */
   if (text)
   {
      text[text_size] = '\0';
   }

   // printf("text_size: %ld | strlen(text): %ld\n", text_size, strlen(text));

   /* Set output pointers */
   *base_indent_ptr = base_indent;
   *text_ptr        = text;

   /* Free memory */
   free(dirname);
   free(basename);
}

static int
FindConfigIndex(int argc, char **argv)
{
   for (int i = 0; i < argc; i++)
   {
      if (argv[i] && IsYAMLFilename(argv[i]))
      {
         return i;
      }
   }
   return -1;
}

static bool
LoadConfigText(MPI_Comm comm, int argc, char **argv, int config_idx, int *base_indent_ptr,
               char **text_ptr, char **config_dir_ptr)
{
   char *config_base = NULL;

   if (argc > 0 && IsYAMLFilename(argv[0]))
   {
      InputArgsRead(comm, argv[0], base_indent_ptr, text_ptr);
      if (ErrorCodeActive())
      {
         return false;
      }
      SplitFilename(argv[0], config_dir_ptr, &config_base);
      free(config_base);
      return true;
   }

   if (config_idx >= 0)
   {
      InputArgsRead(comm, argv[config_idx], base_indent_ptr, text_ptr);
      if (ErrorCodeActive())
      {
         return false;
      }
      SplitFilename(argv[config_idx], config_dir_ptr, &config_base);
      free(config_base);
      return true;
   }

   /* Direct YAML string input */
   if (argv[0] == NULL)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("YAML string input is NULL");
      return false;
   }

   *text_ptr = strdup(argv[0]); // Make a copy since we'll free it later
   return true;
}

static void
ApplyCLIOverrides(int argc, char **argv, int config_idx, YAMLtree *tree)
{
   if (argc <= 1)
   {
      return;
   }

   if (IsYAMLFilename(argv[0]))
   {
      /* Legacy: overrides are in argv[1..] */
      YAMLtreeUpdate(argc - 1, argv + 1, tree);
   }
   else if (config_idx >= 0)
   {
      /* Driver: allow YAMLtreeUpdate to parse -a/--args inside full argv */
      YAMLtreeUpdate(argc, argv, tree);
   }
   else
   {
      /* YAML string input: overrides are in argv[1..] */
      YAMLtreeUpdate(argc - 1, argv + 1, tree);
   }
}

/*-----------------------------------------------------------------------------
 * InputArgsParse
 *-----------------------------------------------------------------------------*/

void
InputArgsParse(MPI_Comm comm, bool lib_mode, int argc, char **argv, input_args **args_ptr)
{
   input_args *iargs       = NULL;
   char       *text        = NULL;
   YAMLtree   *tree        = NULL;
   int         base_indent = 2;
   int         myid        = 0;
   char       *config_dir  = NULL;

   MPI_Comm_rank(comm, &myid);

   /* Read input arguments from file or string.
    *
    * Supported calling patterns:
    * - Legacy/library mode: argv[0] is YAML filename (and argv[1..] are override pairs)
    * - Driver mode: argv is the full CLI (contains YAML filename somewhere and optionally
    * -a ...)
    * - Unit tests: argv[0] is a YAML string (and argv[1..] are override pairs)
    */
   const int config_idx = FindConfigIndex(argc, argv);
   if (!LoadConfigText(comm, argc, argv, config_idx, &base_indent, &text, &config_dir))
   {
      *args_ptr = NULL;
      return;
   }

   /* Quick way to view/debug the tree */
   // printf("%*s", (int) strlen(text), text);

   /* Build YAML tree */
   YAMLtreeBuild(base_indent, text, &tree);

   /* Expand nested include files (post-build to keep parser simple) */
   YAMLtreeExpandIncludes(tree, config_dir ? config_dir : ".");

   /* Check if any config option has been passed in via CLI.
      If so, overwrite the data stored in the YAMLtree object
      with the new values. */
   ApplyCLIOverrides(argc, argv, config_idx, tree);

   /* Return earlier if YAML tree was not built properly */
   if (!myid && ErrorCodeActive())
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
      ErrorCodeSet(ERROR_YAML_TREE_INVALID);
      free(text);
      YAMLtreeDestroy(&tree);
      return;
   }
   MPI_Barrier(comm);

   /* Free memory */
   free(text);
   free(config_dir);

   /*--------------------------------------------
    * Parse file sections
    *-------------------------------------------*/

   InputArgsCreate(lib_mode, &iargs);
   InputArgsParseGeneral(iargs, tree);
   InputArgsParseLinearSystem(iargs, tree);
   InputArgsParseSolver(iargs, tree);
   InputArgsParsePrecon(iargs, tree);

   /* Validate the YAML tree (Has to occur after input args parsing) */
   YAMLtreeValidate(tree);

   /* Return earlier if YAML tree is invalid */
   if (!myid && ErrorCodeActive())
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
      ErrorCodeSet(ERROR_YAML_TREE_INVALID);
      InputArgsDestroy(&iargs);
      YAMLtreeDestroy(&tree);
      return;
   }

   /* Rank 0: Print tree to stdout */
   if (!myid && iargs->general.print_config_params)
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
   }
   MPI_Barrier(comm);

   /* Free memory */
   YAMLtreeDestroy(&tree);

   /* Set output pointer */
   *args_ptr = iargs;
}
