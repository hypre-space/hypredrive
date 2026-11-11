/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/args.h"
#include <float.h>
#include <limits.h>
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "internal/field.h"
#include "internal/gen_macros.h"
#include "internal/info.h"
#include "internal/lsseq.h"
#include "internal/presets.h"
#include "internal/scaling.h"
#include "internal/stats.h"
#include "internal/utils.h"
#include "internal/yaml.h"
#include "logging.h"
#include "object.h"

/*-----------------------------------------------------------------------------
 * General args helpers (schema-driven parsing)
 *-----------------------------------------------------------------------------*/

static void
FieldTypePoolGBToBytesSet(void *field, YAMLnode *node)
{
   double gb = 0.0;
   if (!hypredrv_ParseDouble(node->mapped_val, &gb) || gb < 0.0 ||
       gb > DBL_MAX / (double)GB_TO_BYTES)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid memory pool size '%s' for key '%s'", node->mapped_val,
                           node->key);
      return;
   }
   *((double *)field) = gb * (double)GB_TO_BYTES;
}

#define General_FIELDS(_prefix)                                                      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, name, hypredrv_FieldTypeStringSet)                \
   ADD_FIELD_OFFSET_ENTRY(_prefix, statistics_filename, hypredrv_FieldTypeStringSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, warmup, hypredrv_FieldTypeIntSet)                 \
   ADD_FIELD_OFFSET_ENTRY(_prefix, statistics, hypredrv_FieldTypeIntSet)             \
   ADD_FIELD_OFFSET_ENTRY(_prefix, print_config_params, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_millisec, hypredrv_FieldTypeIntSet)           \
   ADD_FIELD_OFFSET_ENTRY(_prefix, device_lazy_init, hypredrv_FieldTypeIntSet)       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, exec_policy, hypredrv_FieldTypeIntSet)            \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_vendor_spgemm, hypredrv_FieldTypeIntSet)      \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_vendor_spmv, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, use_vendor_sptrans, hypredrv_FieldTypeIntSet)     \
   ADD_FIELD_OFFSET_ENTRY(_prefix, num_repetitions, hypredrv_FieldTypeIntSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, dev_pool_size, FieldTypePoolGBToBytesSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, uvm_pool_size, FieldTypePoolGBToBytesSet)         \
   ADD_FIELD_OFFSET_ENTRY(_prefix, host_pool_size, FieldTypePoolGBToBytesSet)        \
   ADD_FIELD_OFFSET_ENTRY(_prefix, pinned_pool_size, FieldTypePoolGBToBytesSet)

#define General_NUM_FIELDS \
   (sizeof(General_field_offset_map) / sizeof(General_field_offset_map[0]))
GENERATE_PREFIXED_COMPONENTS(General)
/* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */ /* GCOVR_EXCL_BR_LINE */

StrIntMapArray
hypredrv_GeneralGetValidValues(const char *key)
{
   if (!strcmp(key, "warmup") || !strcmp(key, "print_config_params") ||
       !strcmp(key, "use_millisec") || !strcmp(key, "device_lazy_init") ||
       !strcmp(key, "use_vendor_spgemm") || !strcmp(key, "use_vendor_spmv") ||
       !strcmp(key, "use_vendor_sptrans"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "statistics"))
   {
      static StrIntMap map[] = {
         {"off", 0}, {"no", 0},   {"false", 0}, {"0", 0}, {"on", 1},
         {"yes", 1}, {"true", 1}, {"1", 1},     {"2", 2},
      };
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
hypredrv_GeneralSetDefaultArgs(General_args *args)
{
   args->name[0]                = '\0';
   args->statistics_filename[0] = '\0';
   args->warmup                 = 0;
   args->statistics             = 1;
   args->print_config_params    = 1;
   args->use_millisec           = 0;
   args->device_lazy_init       = 0;
#ifdef HYPRE_USING_GPU
   args->exec_policy       = 1;
   args->use_vendor_spgemm = 1;
   args->use_vendor_spmv   = 1;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   /* Preserve HYPRE's CUDA default and the existing HIP compatibility setting. */
   args->use_vendor_sptrans = 0;
#else
   args->use_vendor_sptrans = 1;
#endif
#else
   args->exec_policy        = 0;
   args->use_vendor_spgemm  = 0;
   args->use_vendor_spmv    = 0;
   args->use_vendor_sptrans = 0;
#endif
   args->num_repetitions  = 1;
   args->dev_pool_size    = 2.0 * (double)GB_TO_BYTES;
   args->uvm_pool_size    = 2.0 * (double)GB_TO_BYTES;
   args->host_pool_size   = 2.0 * (double)GB_TO_BYTES;
   args->pinned_pool_size = 0.1 * (double)GB_TO_BYTES;
}

static input_args *
InputArgsCreate(bool lib_mode)
{
   input_args *iargs = hypredrv_Malloc(sizeof(*iargs), "input arguments");
   if (!iargs)
   {
      return NULL;
   }

   /* Set general default options */
   hypredrv_GeneralSetDefaultArgs(&iargs->general);
   iargs->general.print_config_params = !lib_mode;

   /* Set default Linear System options */
   hypredrv_LinearSystemSetDefaultArgs(&iargs->ls);
   iargs->ls.exec_policy = iargs->general.exec_policy;

   /* Set default preconditioner and solver */
   iargs->solver_method = SOLVER_PCG;
   iargs->precon_method = PRECON_BOOMERAMG;
   hypredrv_SolverArgsSetDefaultsForMethod(iargs->solver_method, &iargs->solver);
   hypredrv_PreconArgsSetDefaultsForMethod(iargs->precon_method, &iargs->precon);
   hypredrv_PreconReuseSetDefaultArgs(&iargs->precon_reuse);

   /* Set default scaling */
   hypredrv_ScalingSetDefaultArgs(&iargs->scaling);

   /* Initialize preconditioner variants (default: single variant) */
   iargs->num_precon_variants   = 1;
   iargs->active_precon_variant = 0;
   iargs->precon_methods        = NULL;
   iargs->precon_variants       = NULL;

   return iargs;
}

/*-----------------------------------------------------------------------------
 * hypredrv_InputArgsDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_InputArgsDestroy(input_args **iargs_ptr)
{
   if (*iargs_ptr)
   {
      input_args *iargs = *iargs_ptr;
      if (iargs->precon_methods && iargs->precon_variants)
      {
         /* The active copy holds current runtime handles; its variant slot may
          * still contain stale handles from the previous activation. */
         int active = iargs->active_precon_variant;
         if (active >= 0 && active < iargs->num_precon_variants)
         {
            iargs->precon_methods[active]  = iargs->precon_method;
            iargs->precon_variants[active] = iargs->precon;
         }
         for (int i = 0; i < iargs->num_precon_variants; i++)
         {
            hypredrv_PreconArgsDestroyRuntimeState(iargs->precon_methods[i],
                                                   &iargs->precon_variants[i]);
            hypredrv_PreconArgsDestroyOwnedConfig(iargs->precon_methods[i],
                                                  &iargs->precon_variants[i]);
         }
      }
      else
      {
         hypredrv_PreconArgsDestroyRuntimeState(iargs->precon_method, &iargs->precon);
         hypredrv_PreconArgsDestroyOwnedConfig(iargs->precon_method, &iargs->precon);
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
      hypredrv_PreconReuseDestroyArgs(&iargs->precon_reuse);
      if (iargs->scaling.custom_values)
      {
         hypredrv_DoubleArrayDestroy(&iargs->scaling.custom_values);
      }
      hypredrv_IntArrayDestroy(&iargs->ls.set_suffix);
      hypredrv_PrintSystemDestroyArgs(&iargs->ls.print_system);
      hypredrv_DofLabelMapDestroy(&iargs->ls.dof_labels);
      free(*iargs_ptr);
      *iargs_ptr = NULL;
   }
}

static void
InputArgsLogExecutionPolicy(hypredrv_t *hypredrv)
{
   HYPREDRV_LOG_OBJECTF(
      1, hypredrv, "HYPRE execution policy: %s",
      hypredrv_ExecutionPolicyName(hypredrv->iargs->general.exec_policy));
}

#ifdef HYPRE_USING_GPU
static int
InputArgsValidateDevicePreconditioner(hypredrv_t *hypredrv, bool use_device,
                                      precon_t method, const precon_args *args)
{
   char reason[160];

   if (!use_device || hypredrv_PreconSupportsDevice(method, args, reason, sizeof(reason)))
   {
      return 1;
   }

   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd(
      "GPU execution requested, but the configured linear solver strategy is "
      "not available on GPUs: %s. Select a GPU-supported strategy or set "
      "general.exec_policy to host.",
      reason);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "rejecting unsupported GPU strategy: %s", reason);
   return 0;
}

#endif

/* Retire the live handle or cached runtime state before replacing its config. */
static void
DropActivePrecon(hypredrv_t *hypredrv, const char *discard_reason)
{
   precon_t method = hypredrv->iargs->precon_method;

   if (hypredrv->precon) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_PreconDestroy(&hypredrv->precon, hypredrv->stats,
                             hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1);
   }
   else
   {
#if defined(HYPREDRV_ENABLE_EXPERIMENTAL)
      if (method == PRECON_MGR)
      {
         int num_frelax = 0, num_grelax = 0, num_coarse = 0;
         hypredrv_MGRCountCachedSolvers(&hypredrv->iargs->precon.mgr, &num_frelax,
                                        &num_grelax, &num_coarse);
         if (num_frelax || num_grelax || num_coarse) /* GCOVR_EXCL_BR_LINE */
         {
            HYPREDRV_LOG_OBJECTF(2, hypredrv, "%s: coarse=%d frelax=%d grelax=%d",
                                 discard_reason, num_coarse, num_frelax, num_grelax);
         }
      }
#else
      (void)discard_reason;
#endif
      hypredrv_PreconArgsDestroyRuntimeState(method, &hypredrv->iargs->precon);
   }
   hypredrv->precon_is_setup = false;
}

/* Collective: true when every rank passed the same `count` (at most 3) values. */
static bool
RanksAgreeOnInts(MPI_Comm comm, const int *values, int count)
{
   int minimum[3], maximum[3];
   MPI_Allreduce(values, minimum, count, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(values, maximum, count, MPI_INT, MPI_MAX, comm);
   for (int i = 0; i < count; i++)
   {
      if (minimum[i] != maximum[i]) return false;
   }
   return true;
}

static YAMLnode *
InputArgsFindUniqueRootSection(const YAMLtree *tree, const char *key)
{
   if (!tree || !tree->root || !key)
   {                                     /* GCOVR_EXCL_BR_LINE */
      return NULL; /* GCOVR_EXCL_LINE */ /* GCOVR_EXCL_BR_LINE */
   }

   YAMLnode *match = NULL;
   int       count = 0;
   YAML_NODE_ITERATE(tree->root, child)
   {
      if (!strcmp(child->key, key))
      {
         if (!match)
         {
            match = child;
         }
         count++;
      }
   }

   if (count > 1)
   {
      hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
      hypredrv_ErrorMsgAdd("Duplicate top-level key '%s' is not allowed", key);
      return NULL;
   }

   return match;
}

/*-----------------------------------------------------------------------------
 * hypredrv_InputArgsParseGeneral
 *-----------------------------------------------------------------------------*/

static void
hypredrv_InputArgsParseGeneral(input_args *iargs, const YAMLtree *tree)
{
   YAMLnode *parent = InputArgsFindUniqueRootSection(tree, "general");
   if (!parent)
   {
      if (hypredrv_ErrorCodeActive())
      {
         return;
      }
      /* The \"general\" key is optional */
      iargs->ls.exec_policy = iargs->general.exec_policy;
      return;
   }

   YAML_NODE_SET_VALID(parent);
   hypredrv_GeneralSetArgsFromYAML(&iargs->general, parent);

   /* Mirror general exec policy into linear-system args for downstream use */
   iargs->ls.exec_policy = iargs->general.exec_policy;
}

/*-----------------------------------------------------------------------------
 * hypredrv_InputArgsParseLinearSystem
 *-----------------------------------------------------------------------------*/

static void
hypredrv_InputArgsParseLinearSystem(input_args *iargs, const YAMLtree *tree)
{
   const char key[]  = {"linear_system"};
   YAMLnode  *parent = InputArgsFindUniqueRootSection(tree, key);

   if (!parent)
   {
      if (hypredrv_ErrorCodeActive())
      {
         return;
      }
      // TODO: Add "library" mode to skip the following checks
      // hypredrv_ErrorCodeSet(ERROR_MISSING_KEY);
      // hypredrv_ErrorMsgAddMissingKey(key);
      return;
   }
   YAML_NODE_SET_VALID_IF_NO_VAL(parent);

   if (YAML_NODE_GET_VALIDITY(parent) == YAML_NODE_UNEXPECTED_VAL)
   {
      /* Unreachable with current parse flow: invalid scalar+children trees set a global
       * error during YAMLtreeBuild, and hypredrv_InputArgsParseWithObjectName() returns
       * before section parsers run. Kept for robustness if ordering changes. */       /* GCOVR_EXCL_BR_LINE */
      hypredrv_ErrorMsgAddUnexpectedVal(key);
      /* GCOVR_EXCL_LINE */ /* GCOVR_EXCL_BR_LINE */
   }

   hypredrv_LinearSystemSetArgsFromYAML(&iargs->ls, parent);
   hypredrv_LinearSystemSetNumSystems(&iargs->ls);
}

static void
PreconPresetBuildArgs(const char *preset_name, precon_t *method_out,
                      precon_args *args_out)
{
   YAMLtree *tree   = NULL;
   int       method = 0;
   YAMLnode *type_node =
      hypredrv_PresetLoadTypeNode(preset_name, HYPREDRV_PRESET_PRECON, &tree, &method);
   if (!type_node)
   {
      return;
   }

   *method_out = (precon_t)method;
   hypredrv_PreconSetMethodArgs(*method_out, args_out, type_node);
   hypredrv_YAMLtreeValidate(tree);
   hypredrv_YAMLtreeDestroy(&tree);
}

static int
PreconParseMethodResolve(const char *name, hypredrv_error_t error_code,
                         precon_t *method_out)
{
   StrIntMapArray types = hypredrv_PreconGetValidTypeIntMap();
   int            method;
   if (!hypredrv_StrIntMapArrayLookup(types, name, &method))
   {
      hypredrv_ErrorCodeSet(error_code);
      char *avail = hypredrv_StrIntMapArrayDomainToString(types);
      if (avail)
      {
         /* Added first so it prints after the message below (LIFO order) */
         hypredrv_ErrorMsgAddUnique("Available values for \"preconditioner\": %s", avail);
         free(avail);
      }
      hypredrv_ErrorMsgAdd("Unknown preconditioner type: '%s'", name);
      return 0;
   }

   *method_out = (precon_t)method;
   return 1;
}

/* Shape of a preconditioner section: a scalar method name (value_only), one
 * typed block or preset (body), or a sequence of variants. With mixed_types,
 * each sequence item names its own type; otherwise all share `method`. */
typedef struct
{
   YAMLnode *body;
   YAMLnode *sequence;
   size_t    num_variants;
   bool      value_only;
   bool      mixed_types;
   precon_t  method;
} PreconLayout;

static size_t
PreconCountSequenceItems(const YAMLnode *node)
{
   size_t count = 0;
   YAML_NODE_ITERATE(node, child)
   {
      if (!strcmp(child->key, "-")) count++;
   }
   return count;
}

/* A mapping section must hold exactly one type key besides reuse_node. */
static YAMLnode *
PreconFindTypedBlock(YAMLnode *parent, const YAMLnode *reuse_node)
{
   YAMLnode *body = NULL;
   YAML_NODE_ITERATE(parent, child)
   {
      if (child == reuse_node) continue;
      if (body)
      {
         hypredrv_ErrorCodeSet(ERROR_EXTRA_KEY);
         hypredrv_ErrorMsgAddExtraKey(child->key);
         return NULL;
      }
      body = child;
   }
   if (!body)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_PRECON);
   }
   return body;
}

static bool
PreconLayoutClassify(PreconLayout *layout, YAMLnode *parent, const YAMLnode *reuse_node)
{
   layout->body         = parent;
   layout->num_variants = 1;
   layout->value_only   = parent->val[0] != '\0';
   if (layout->value_only) return true;

   layout->num_variants = PreconCountSequenceItems(parent);
   if (layout->num_variants)
   {
      layout->sequence    = parent;
      layout->mixed_types = true;
      return true;
   }

   YAMLnode *body = PreconFindTypedBlock(parent, reuse_node);
   layout->body   = body;
   if (!body) return false;
   if (strcmp(body->key, "preset") != 0)
   {
      layout->mixed_types  = (strcmp(body->key, "variants") == 0);
      layout->num_variants = PreconCountSequenceItems(body);
   }
   if (layout->num_variants)
   {
      layout->sequence = body;
   }
   else if (layout->mixed_types)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
      hypredrv_ErrorMsgAdd("preconditioner.variants must be a sequence of variants");
      return false;
   }
   else
   {
      layout->num_variants = 1;
   }
   return true;
}

/* Resolve the method shared by every variant, then mark the section nodes. */
static bool
PreconLayoutResolveMethod(PreconLayout *layout)
{
   YAMLnode *body       = layout->body;
   bool      value_only = layout->value_only;
   if (!layout->mixed_types && (value_only || strcmp(body->key, "preset") != 0))
   {
      if (!PreconParseMethodResolve(value_only ? body->val : body->key,
                                    value_only ? ERROR_INVALID_VAL : ERROR_INVALID_KEY,
                                    &layout->method))
         return false;
      YAML_NODE_SET_VALID(body);
   }
   if (layout->sequence) YAML_NODE_SET_VALID(layout->sequence);
   return true;
}

static bool
PreconVariantsAllocate(size_t num_variants, precon_t **methods, precon_args **variants)
{
   if (num_variants > INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      hypredrv_ErrorMsgAdd("Too many preconditioner variants");
      return false;
   }
   *methods = hypredrv_Calloc(num_variants, sizeof(**methods), "preconditioner methods");
   *variants =
      hypredrv_Calloc(num_variants, sizeof(**variants), "preconditioner variants");
   return (*methods && *variants) != 0;
}

/* Parse one variant (a sequence item, or the single body) into the next slot. */
static bool
PreconParseVariant(PreconLayout *layout, YAMLnode *item, precon_t *methods,
                   precon_args *variants, size_t *parsed_variants)
{
   YAMLnode *config = item;
   if (layout->mixed_types)
   {
      if (!item->children || item->children->next)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_KEY);
         hypredrv_ErrorMsgAdd(
            "Each preconditioner variant must contain exactly one type key");
         return false;
      }
      config = item->children;
   }

   bool preset = (!layout->value_only && !strcmp(config->key, "preset")) != 0;
   if (preset)
   {
      if (!config->val[0])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Preconditioner preset name is missing");
         return false;
      }
   }
   else if (layout->mixed_types &&
            !PreconParseMethodResolve(config->key, ERROR_INVALID_KEY, &layout->method))
   {
      return false;
   }

   /* Include the current entry in cleanup before parsing can allocate any
    * nested MGR configuration, even when a later field rejects the input. */
   precon_t method = layout->method;
   size_t   index  = (*parsed_variants)++;
   methods[index]  = preset ? PRECON_NONE : method;
   if (preset) PreconPresetBuildArgs(config->val, &methods[index], &variants[index]);
   else if (layout->value_only)
      hypredrv_PreconArgsSetDefaultsForName(method, layout->body->val, &variants[index]);
   else hypredrv_PreconSetMethodArgs(method, &variants[index], config);
   if (hypredrv_ErrorCodeActive()) return false;
   if (preset) YAML_NODE_SET_VALID(config);
   if (layout->mixed_types) YAML_NODE_SET_VALID(item);
   return true;
}

static void
PreconVariantsDestroy(precon_t *methods, precon_args *variants, size_t parsed_variants)
{
   if (variants)
   {
      for (size_t i = 0; i < parsed_variants; i++)
         hypredrv_PreconArgsDestroyOwnedConfig(methods[i], &variants[i]);
   }
   free(methods);
   free(variants);
}

/* Scalar, typed and sequence forms share allocation, parsing and publication.
 * YAML nodes stay borrowed; only complete argument arrays leave this operation. */
static void
hypredrv_InputArgsParsePrecon(input_args *iargs, const YAMLtree *tree)
{
   precon_t    *methods         = NULL;
   precon_args *variants        = NULL;
   size_t       parsed_variants = 0;
   PreconLayout layout          = {.method = PRECON_NONE};
   YAMLnode    *parent          = InputArgsFindUniqueRootSection(tree, "preconditioner");

   hypredrv_MGRSetDofLabels(iargs->ls.dof_labels);
   if (!parent)
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_MISSING_KEY);
         hypredrv_ErrorMsgAddMissingKey("preconditioner");
      }
      goto cleanup;
   }
   YAML_NODE_SET_VALID(parent);

   YAMLnode *reuse_node = hypredrv_YAMLnodeFindChildByKey(parent, "reuse");
   if (reuse_node)
   {
      hypredrv_PreconReuseSetArgsFromYAML(&iargs->precon_reuse, reuse_node);
      if (hypredrv_ErrorCodeActive()) goto cleanup;
      YAML_NODE_SET_VALID(reuse_node);
   }

   if (!PreconLayoutClassify(&layout, parent, reuse_node)) goto cleanup;
   if (!PreconLayoutResolveMethod(&layout)) goto cleanup;
   if (!PreconVariantsAllocate(layout.num_variants, &methods, &variants)) goto cleanup;

   YAMLnode *sequence = layout.sequence;
   for (YAMLnode *item = sequence ? sequence->children : layout.body; item;
        item           = sequence ? item->next : NULL)
   {
      if (sequence && strcmp(item->key, "-") != 0) continue;
      if (!PreconParseVariant(&layout, item, methods, variants, &parsed_variants))
         goto cleanup;
   }

   iargs->num_precon_variants   = (int)layout.num_variants;
   iargs->active_precon_variant = 0;
   iargs->precon_methods        = methods;
   iargs->precon_variants       = variants;
   iargs->precon_method         = methods[0];
   iargs->precon                = variants[0];
   methods                      = NULL;
   variants                     = NULL;

cleanup:
   PreconVariantsDestroy(methods, variants, parsed_variants);
   hypredrv_MGRSetDofLabels(NULL);
}

static void
InputArgsValidateArgv(int argc, char **argv, input_args **args_ptr)
{
   if (argc < 1 || !argv || !args_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Configuration parsing requires at least one argument, "
                           "an argument vector, and an output pointer");
      return;
   }
   for (int i = 0; i < argc; i++)
   {
      if (!argv[i])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Configuration argument %d is NULL", i);
         return;
      }
   }
}

/* Inline YAML takes precedence over filename suffixes, including filenames
 * stored in YAML values or CLI overrides. Classify the source only once.
 * Returns the index of the configuration file in argv, or -1 for inline YAML. */
static int
InputArgsFindConfigIndex(int argc, char **argv)
{
   bool inline_yaml = strchr(argv[0], '\n') != NULL;
   for (const char *colon = strchr(argv[0], ':'); !inline_yaml && colon;
        colon             = strchr(colon + 1, ':'))
   {
      inline_yaml = ((colon[1] == '\0' || colon[1] == ' ' || colon[1] == '\t') != 0);
   }
   if (inline_yaml) return -1;

   for (int i = 0; i < argc; i++)
   {
      if (hypredrv_IsYAMLFilename(argv[i])) return i;
   }
   return -1;
}

/* Root rank only: resolve `candidate` under its own directory and read it. */
static void
InputArgsReadConfigFile(const char *candidate, char *config_dir, int *base_indent,
                        size_t *text_size, char **text)
{
   char *dirname = NULL, *basename = NULL, *root_dir = NULL, *resolved = NULL;
   hypredrv_SplitFilename(candidate, &dirname, &basename);
   if (!dirname || !basename)
   {
      hypredrv_AllocationFailure("Failed to split configuration filename");
   }
   else if (!basename[0])
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
      hypredrv_ErrorMsgAdd("Invalid configuration file path");
   }
   else
   {
      root_dir = hypredrv_Realpath(dirname);
      resolved = hypredrv_Realpath(candidate);
      if (!root_dir || !resolved)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Configuration file not found: '%s'", candidate);
      }
      else if (!hypredrv_PathIsUnderRoot(resolved, root_dir) ||
               !hypredrv_BinaryPathPrefixIsSafe(resolved))
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_UNEXPECTED_ENTRY);
         hypredrv_ErrorMsgAdd("Invalid configuration file path");
      }
      else
      {
         free(dirname);
         free(basename);
         hypredrv_SplitFilename(resolved, &dirname, &basename);
         if (!dirname || !basename)
         {
            hypredrv_AllocationFailure("Failed to split resolved configuration filename");
         }
         else
         {
            /* The validated canonical path fits MAX_FILENAME_LENGTH. Use
             * its directory for later mapping/list include expansion too. */
            memcpy(config_dir, dirname, strlen(dirname) + 1);
            *base_indent = -1;
            hypredrv_YAMLtextRead(dirname, basename, 0, base_indent, text_size, text);
         }
      }
   }
   free(resolved);
   free(root_dir);
   free(basename);
   free(dirname);
   if (*text_size > INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_OUT_OF_BOUNDS);
      hypredrv_ErrorMsgAdd("Configuration text exceeds the MPI broadcast count limit");
   }
}

/* Collective. Returns false, on every rank, when any rank failed. */
static bool
InputArgsLoadConfigFile(MPI_Comm comm, int myid, const char *candidate, char *config_dir,
                        int *base_indent, char **text_ptr)
{
   size_t text_size = 0;
   if (!myid)
   {
      InputArgsReadConfigFile(candidate, config_dir, base_indent, &text_size, text_ptr);
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return false;

   /* The root owns file access. Broadcast its text and canonical directory
    * so other ranks need not resolve the configuration in their own cwd. */
   uint64_t text_size_u64 = (uint64_t)text_size;
   MPI_Bcast(base_indent, 1, MPI_INT, 0, comm);
   MPI_Bcast(&text_size_u64, 1, MPI_UINT64_T, 0, comm);
   MPI_Bcast(config_dir, MAX_FILENAME_LENGTH, MPI_CHAR, 0, comm);
   text_size = (size_t)text_size_u64;
   if (!*text_ptr) *text_ptr = hypredrv_Malloc(text_size + 1, "configuration text");
   if (hypredrv_DistributedErrorStateSync(comm)) return false;
   (*text_ptr)[text_size] = '\0';
   MPI_Bcast(*text_ptr, (int)text_size, MPI_CHAR, 0, comm);
   return true;
}

/* Collective. Returns false, on every rank, when any rank failed. */
static bool
InputArgsCopyInlineText(MPI_Comm comm, const char *source, char **text_ptr)
{
   *text_ptr = strdup(source);
   if (!*text_ptr) hypredrv_AllocationFailure("Failed to copy inline YAML configuration");
   return (!hypredrv_DistributedErrorStateSync(comm)) != 0;
}

static void
InputArgsApplyOverrides(int argc, char **argv, int config_idx, YAMLtree *tree)
{
   if (argc <= 1) return;
   if (config_idx > 0) hypredrv_YAMLtreeUpdate(argc, argv, tree);
   else hypredrv_YAMLtreeUpdate(argc - 1, argv + 1, tree);
}

static void
InputArgsParseSolver(input_args *iargs, const YAMLtree *tree)
{
   YAMLnode *solver = InputArgsFindUniqueRootSection(tree, "solver");
   if (!solver) return;

   YAMLnode *scaling = hypredrv_YAMLnodeFindChildByKey(solver, "scaling");
   hypredrv_SolverSetArgsFromYAML(&iargs->solver_method, &iargs->solver, solver, scaling);
   if (scaling)
   {
      hypredrv_ScalingSetArgsFromYAML(&iargs->scaling, scaling);
      YAML_NODE_SET_VALID(scaling);
   }
}

static void
InputArgsReportInvalidTree(int myid, YAMLtree *tree, const char *log_object_name,
                           const char *reason)
{
   hypredrv_ErrorCodeSet(ERROR_YAML_TREE_INVALID);
   hypredrv_ErrorReportSetCollective(true);
   if (!myid) hypredrv_YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
   HYPREDRV_LOGF(2, myid, log_object_name, 0, "%s", reason);
}

/*-----------------------------------------------------------------------------
 * Parse one complete configuration, including source ingress and overrides.
 *-----------------------------------------------------------------------------*/

void
hypredrv_InputArgsParseWithObjectName(MPI_Comm comm, bool lib_mode, int argc, char **argv,
                                      input_args **args_ptr, const char *log_object_name)
{
   input_args *iargs                           = NULL;
   char       *text                            = NULL;
   YAMLtree   *tree                            = NULL;
   char        config_dir[MAX_FILENAME_LENGTH] = {0};
   int         base_indent = 2, myid = 0, config_idx = -1;

   MPI_Comm_rank(comm, &myid);
   HYPREDRV_LOGF(3, myid, log_object_name, 0, "args parse begin (lib_mode=%d argc=%d)",
                 (int)lib_mode, argc);
   if (args_ptr) *args_ptr = NULL;
   InputArgsValidateArgv(argc, argv, args_ptr);
   if (hypredrv_DistributedErrorStateSync(comm)) goto load_failure;

   config_idx = InputArgsFindConfigIndex(argc, argv);

   /* File input enters broadcasts; inline input is rank-local. No rank may
    * choose a different collective path or leave before this agreement. */
   int from_file = config_idx >= 0;
   if (!RanksAgreeOnInts(comm, &from_file, 1))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Configuration source kind must agree across ranks");
   }
   if (hypredrv_DistributedErrorStateSync(comm)) goto load_failure;

   bool loaded =
      (from_file ? (int)InputArgsLoadConfigFile(comm, myid, argv[config_idx], config_dir,
                                                &base_indent, &text)
                 : (int)InputArgsCopyInlineText(comm, argv[0], &text)) != 0;
   if (!loaded) goto load_failure;
   HYPREDRV_LOGF(3, myid, log_object_name, 0,
                 "config text loaded (config_idx=%d base_indent=%d)", config_idx,
                 base_indent);
   HYPREDRV_LOG_TEXTBLOCK(3, myid, log_object_name, 0, "YAML input text:", text);

   hypredrv_YAMLtreeBuild(base_indent, text, &tree);
   HYPREDRV_LOGF(3, myid, log_object_name, 0, "yaml tree build complete");
   hypredrv_YAMLtreeExpandIncludes(tree, config_dir[0] ? config_dir : ".");
   HYPREDRV_LOGF(3, myid, log_object_name, 0, "yaml include expansion complete");
   InputArgsApplyOverrides(argc, argv, config_idx, tree);
   HYPREDRV_LOGF(3, myid, log_object_name, 0, "yaml CLI overrides applied");
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      InputArgsReportInvalidTree(myid, tree, log_object_name,
                                 "args parse failed: invalid YAML tree");
      goto cleanup;
   }
   free(text);
   text = NULL;

   iargs = InputArgsCreate(lib_mode);
   if (iargs)
   {
      hypredrv_InputArgsParseGeneral(iargs, tree);
      hypredrv_InputArgsParseLinearSystem(iargs, tree);
      InputArgsParseSolver(iargs, tree);
      hypredrv_InputArgsParsePrecon(iargs, tree);
   }
   HYPREDRV_LOGF(3, myid, log_object_name, 0,
                 "parsed sections: general/linear_system/solver/preconditioner");
   hypredrv_YAMLtreeValidate(tree);
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      InputArgsReportInvalidTree(myid, tree, log_object_name,
                                 "args parse failed during post-parse validation");
      goto cleanup;
   }
   if (!myid && iargs->general.print_config_params)
      hypredrv_YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
   HYPREDRV_LOGF(2, myid, log_object_name, 0,
                 "args parse end: solver=%d precon=%d stats=%d scaling_enabled=%d "
                 "scaling_type=%d rhs_mode=%d",
                 (int)iargs->solver_method, (int)iargs->precon_method,
                 iargs->general.statistics, iargs->scaling.enabled,
                 (int)iargs->scaling.type, (int)iargs->ls.rhs_mode);
   *args_ptr = iargs;
   iargs     = NULL;
   goto cleanup;

load_failure:
   hypredrv_ErrorReportSetCollective(true);
   HYPREDRV_LOGF(2, myid, log_object_name, 0,
                 "args parse failed while loading config text");
cleanup:
   free(text);
   hypredrv_YAMLtreeDestroy(&tree);
   hypredrv_InputArgsDestroy(&iargs);
}

uint32_t
hypredrv_InputArgsParseObject(hypredrv_t *hypredrv, int argc, char **argv)
{
   char                 log_object_name[32] = {0};
   input_args          *replacement         = NULL;
   PreconReuseTimesteps timesteps           = {0};
   PreconReuseState    *reuse_state         = NULL;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse begin (argc=%d)", argc);

   hypredrv_InputArgsParseWithObjectName(
      hypredrv->comm, hypredrv->lib_mode, argc, argv, &replacement,
      hypredrv_ResolveLogObjectName(hypredrv, log_object_name, sizeof(log_object_name)));
   if (hypredrv_ErrorCodeGet()) goto cleanup;

   /* A complete candidate owns its schedules and parsed method storage. Finish
    * input validation on every rank before changing any live object state. */
   bool needs_timesteps =
      (replacement->general.statistics > 0 ||
       (replacement->precon_reuse.enabled && replacement->precon_reuse.per_timestep) ||
       hypredrv_PrintSystemNeedsTimestepSchedule(&replacement->ls.print_system)) != 0;
   if (needs_timesteps)
   {
      if (replacement->ls.timestep_filename[0])
         hypredrv_LinearSystemLoadTimestepSchedule(replacement->ls.timestep_filename,
                                                   &timesteps.ids, &timesteps.starts);
      else if (replacement->ls.sequence_filename[0])
         hypredrv_LSSeqReadTimestepsWithIds(replacement->ls.sequence_filename,
                                            &timesteps.ids, &timesteps.starts);
   }
#ifdef HYPRE_USING_GPU
   if (!hypredrv_ErrorCodeActive())
      InputArgsValidateDevicePreconditioner(
         hypredrv, (bool)replacement->general.exec_policy, replacement->precon_method,
         &replacement->precon);
#endif
   if (!hypredrv_ErrorCodeActive())
      reuse_state = hypredrv_PreconReuseStateCreate(&replacement->precon_reuse,
                                                    timesteps.starts, hypredrv->stats);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) goto cleanup;

   /* Retire native consumers while their original method tags and options are
    * still installed. Publish the validated replacement and its owned schedule
    * together, then reset state derived from the old configuration. */
   hypredrv_VectorTagsClear(hypredrv);
   hypredrv->precmat_sequence_cache_args = NULL;
   hypredrv_ObjectDestroySolvers(hypredrv);
   hypredrv_PreconReuseStateDestroy(&hypredrv->precon_reuse_state);
   hypredrv_InputArgsDestroy(&hypredrv->iargs);
   hypredrv->iargs = replacement;
   replacement     = NULL;
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.ids);
   hypredrv_IntArrayDestroy(&hypredrv->precon_reuse_timesteps.starts);
   hypredrv->precon_reuse_timesteps = timesteps;
   timesteps                        = (PreconReuseTimesteps){0};
   hypredrv->precon_reuse_state     = reuse_state;
   reuse_state                      = NULL;

   hypredrv_StatsSetNumReps(hypredrv->stats, hypredrv->iargs->general.num_repetitions);
   hypredrv_StatsSetNumLinearSystems(hypredrv->stats, hypredrv->iargs->ls.num_systems);
   if (hypredrv->iargs->general.use_millisec)
      hypredrv_StatsTimerSetMilliseconds(hypredrv->stats);
   else hypredrv_StatsTimerSetSeconds(hypredrv->stats);
   hypredrv_StatsSetObjectName(hypredrv->stats, hypredrv->iargs->general.name);

#if defined(HYPRE_USING_GPU) && HYPRE_CHECK_MIN_VERSION(23100, 0)
   if (hypredrv->iargs->general.exec_policy && !hypredrv->iargs->general.device_lazy_init)
   {
      hypredrv_ObjectApplyRuntimeSettings(hypredrv);
      HYPREDRV_LOG_OBJECTF(1, hypredrv,
                           "eager device initialization requested by "
                           "general.device_lazy_init=off");
      HYPRE_DeviceInitialize();
   }
#endif
   InputArgsLogExecutionPolicy(hypredrv);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse end");

cleanup:
   if (hypredrv_ErrorCodeGet())
      HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsParse failed (code=0x%x)",
                           hypredrv_ErrorCodeGet());
   hypredrv_PreconReuseStateDestroy(&reuse_state);
   hypredrv_InputArgsDestroy(&replacement);
   hypredrv_IntArrayDestroy(&timesteps.ids);
   hypredrv_IntArrayDestroy(&timesteps.starts);
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_InputArgsSetPreconVariant(hypredrv_t *hypredrv, int variant_idx)
{
   input_args *iargs           = hypredrv->iargs;
   int         current_variant = -1, method = -1;
   bool        variant_changed = false;
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsSetPreconVariant begin (idx=%d)",
                        variant_idx);
   if (!iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Preconditioner variant selection requires parsed input arguments");
   }
   else if (variant_idx < 0 || variant_idx >= iargs->num_precon_variants)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid preconditioner variant index: %d (valid range: 0-%d)",
                           variant_idx, iargs->num_precon_variants - 1);
   }
   else
   {
      current_variant = iargs->active_precon_variant;
      variant_changed = variant_idx != current_variant;
      method          = (int)(variant_changed ? iargs->precon_methods[variant_idx]
                                              : iargs->precon_method);

#ifdef HYPRE_USING_GPU
      if (variant_changed && !hypredrv_ErrorCodeActive())
         InputArgsValidateDevicePreconditioner(hypredrv, (bool)iargs->general.exec_policy,
                                               (precon_t)method,
                                               &iargs->precon_variants[variant_idx]);
#endif
   }
   /* All ranks must select the same transition before any rank retires native
    * consumers. An unchanged variant follows the same validation boundary. */
   int selection[] = {variant_idx, current_variant, method};
   if (!RanksAgreeOnInts(hypredrv->comm, selection, 3))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner variant transition must agree across ranks");
   }
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) return hypredrv_ErrorCodeGet();

   HYPREDRV_LOG_OBJECTF(
      2, hypredrv, "preconditioner variant selection: current=%d requested=%d changed=%d",
      current_variant, variant_idx, variant_changed);

   /* Only rebuild solver/preconditioner when switching variants. */
   if (variant_changed)
   {
      hypredrv_VectorTagsClear(hypredrv);
      if (hypredrv->solver)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "switching preconditioner variant: destroying active "
                              "solver");
      }
      hypredrv_SolverDestroy(&hypredrv->solver);

      if (hypredrv->precon)
      {
         HYPREDRV_LOG_OBJECTF(2, hypredrv,
                              "switching preconditioner variant: destroying active "
                              "preconditioner");
      }
      DropActivePrecon(
         hypredrv,
         "discarding cached MGR handles before switching preconditioner variant");
      iargs->precon_variants[current_variant] = iargs->precon;
      iargs->active_precon_variant            = variant_idx;
      iargs->precon_method                    = iargs->precon_methods[variant_idx];
      iargs->precon                           = iargs->precon_variants[variant_idx];
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(2, hypredrv,
                           "preconditioner variant unchanged; reusing setup");
   }

   HYPREDRV_LOG_OBJECTF(2, hypredrv, "preconditioner variant selected: idx=%d method=%d",
                        variant_idx, (int)hypredrv->iargs->precon_method);
   HYPREDRV_LOG_OBJECTF(1, hypredrv, "HYPREDRV_InputArgsSetPreconVariant end");

   return hypredrv_ErrorCodeGet();
}

/* An owned preset candidate: the method and arguments for `kind`, plus variant
 * storage when the current configuration has none yet. */
typedef struct
{
   int          method;
   precon_args  precon;
   solver_args  solver;
   precon_t    *new_methods;
   precon_args *new_variants;
} PresetReplacement;

static void
PreconVariantStorageCreate(int num_variants, precon_t **methods_ptr,
                           precon_args **variants_ptr)
{
   precon_t    *new_methods  = calloc((size_t)num_variants, sizeof(*new_methods));
   precon_args *new_variants = calloc((size_t)num_variants, sizeof(*new_variants));
   if (!new_methods || !new_variants)
      hypredrv_AllocationFailure("Failed to allocate preconditioner variants");
   else
      for (int i = 0; i < num_variants; i++) new_methods[i] = PRECON_NONE;
   *methods_ptr  = new_methods;
   *variants_ptr = new_variants;
}

static void
PresetPreparePrecon(hypredrv_t *hypredrv, input_args *iargs, const char *preset,
                    PresetReplacement *repl)
{
   precon_t     precon_method      = PRECON_NONE;
   precon_args *precon_replacement = &repl->precon;
   hypredrv_MGRSetDofLabels(iargs->ls.dof_labels);
   PreconPresetBuildArgs(preset, &precon_method, precon_replacement);
   hypredrv_MGRSetDofLabels(NULL);
   repl->method = (int)precon_method;
#ifdef HYPRE_USING_GPU
   if (!hypredrv_ErrorCodeActive())
      InputArgsValidateDevicePreconditioner(hypredrv, (bool)iargs->general.exec_policy,
                                            precon_method, precon_replacement);
#else
   (void)hypredrv;
#endif
   if (!iargs->precon_methods && !hypredrv_ErrorCodeActive())
      PreconVariantStorageCreate(iargs->num_precon_variants, &repl->new_methods,
                                 &repl->new_variants);
}

static void
PresetPrepareReplacement(hypredrv_t *hypredrv, input_args *iargs,
                         hypredrv_PresetKind kind, const char *preset,
                         PresetReplacement *repl)
{
   if (kind == HYPREDRV_PRESET_PRECON)
   {
      PresetPreparePrecon(hypredrv, iargs, preset, repl);
   }
   else if (kind == HYPREDRV_PRESET_SOLVER)
   {
      solver_t solver_method = SOLVER_PCG;
      hypredrv_SolverArgsFromPreset(preset, &solver_method, &repl->solver);
      repl->method = (int)solver_method;
   }
   else
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Unknown preset kind: %d", (int)kind);
   }
}

/* All validation and allocation is complete. Release native borrowers
 * before replacing their configuration, then publish the prepared value once. */
static void
PresetPublish(hypredrv_t *hypredrv, input_args *iargs, input_args *created,
              hypredrv_PresetKind kind, PreconReuseState **reuse_state,
              const PresetReplacement *repl)
{
   hypredrv->iargs = iargs;
   if (created)
   {
      hypredrv_PreconReuseStateDestroy(&hypredrv->precon_reuse_state);
      hypredrv->precon_reuse_state = *reuse_state;
      *reuse_state                 = NULL;
   }
   hypredrv_VectorTagsClear(hypredrv);
   hypredrv_SolverDestroy(&hypredrv->solver);
   if (kind == HYPREDRV_PRESET_PRECON)
   {
      DropActivePrecon(
         hypredrv, "discarding cached MGR handles before applying preconditioner preset");
      hypredrv_PreconArgsDestroyOwnedConfig(iargs->precon_method, &iargs->precon);
      if (repl->new_methods)
      {
         iargs->precon_methods  = repl->new_methods;
         iargs->precon_variants = repl->new_variants;
      }
      int active                     = iargs->active_precon_variant;
      iargs->precon_methods[active]  = (precon_t)repl->method;
      iargs->precon_variants[active] = repl->precon;
      iargs->precon_method           = (precon_t)repl->method;
      iargs->precon                  = repl->precon;
   }
   else
   {
      iargs->solver_method = (solver_t)repl->method;
      iargs->solver        = repl->solver;
   }
   if (created) InputArgsLogExecutionPolicy(hypredrv);
}

/* Prepare one owned replacement without publishing defaults or touching live
 * consumers. Both public preset operations share the same collective boundary. */
uint32_t
hypredrv_InputArgsSetPreset(hypredrv_t *hypredrv, hypredrv_PresetKind kind,
                            const char *preset)
{
   input_args       *iargs = hypredrv->iargs, *created = NULL;
   PreconReuseState *reuse_state = NULL;
   PresetReplacement repl        = {0};
   repl.method = kind == HYPREDRV_PRESET_PRECON ? (int)PRECON_NONE : (int)SOLVER_PCG;
   if (!preset)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s preset name cannot be NULL",
                           kind == HYPREDRV_PRESET_PRECON ? "Preconditioner" : "Solver");
   }
   if (!iargs && !hypredrv_ErrorCodeActive())
      iargs = created = InputArgsCreate(hypredrv->lib_mode);
   if (!hypredrv_ErrorCodeActive())
      PresetPrepareReplacement(hypredrv, iargs, kind, preset, &repl);

   int selection[] = {
      (int)kind, repl.method,
      kind == HYPREDRV_PRESET_PRECON && iargs ? iargs->active_precon_variant : -1};
   if (!RanksAgreeOnInts(hypredrv->comm, selection, 3))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Preset kind, selected method and active variant must agree across ranks");
   }
   if (created && !hypredrv_ErrorCodeActive())
      reuse_state =
         hypredrv_PreconReuseStateCreate(&created->precon_reuse, NULL, hypredrv->stats);
   if (hypredrv_DistributedErrorStateSync(hypredrv->comm)) goto failure;

   PresetPublish(hypredrv, iargs, created, kind, &reuse_state, &repl);
   return hypredrv_ErrorCodeGet();

failure:
   if (kind == HYPREDRV_PRESET_PRECON)
      hypredrv_PreconArgsDestroyOwnedConfig((precon_t)repl.method, &repl.precon);
   free(repl.new_methods);
   free(repl.new_variants);
   hypredrv_PreconReuseStateDestroy(&reuse_state);
   hypredrv_InputArgsDestroy(&created);
   return hypredrv_ErrorCodeGet();
}
