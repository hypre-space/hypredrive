/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "args.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "utils.h"
#include "yaml.h"

/*-----------------------------------------------------------------------------
 * InputArgsCreate
 *-----------------------------------------------------------------------------*/

void
InputArgsCreate(bool lib_mode, input_args **iargs_ptr)
{
   input_args *iargs = (input_args *)malloc(sizeof(input_args));

   /* Set general default options */
   iargs->warmup              = 0;
   iargs->num_repetitions     = 1;
   iargs->statistics          = 1;
   iargs->print_config_params = !lib_mode;
   iargs->dev_pool_size       = 8.0 * GB_TO_BYTES;
   iargs->uvm_pool_size       = 8.0 * GB_TO_BYTES;
   iargs->host_pool_size      = 8.0 * GB_TO_BYTES;
   iargs->pinned_pool_size    = 0.5 * GB_TO_BYTES;

   /* Set default Linear System options */
   LinearSystemSetDefaultArgs(&iargs->ls);

   /* Set default preconditioner and solver */
   iargs->solver_method = SOLVER_PCG;
   iargs->precon_method = PRECON_BOOMERAMG;

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
   YAMLnode *parent = NULL;
   YAMLnode *child  = NULL;

   parent = YAMLnodeFindByKey(tree->root, "general");
   if (!parent)
   {
      /* The "general" key is not mandatory,
         so we don't set an error code if it's not found. */
      return;
   }
   YAML_NODE_SET_VALID(parent);

   child = parent->children;
   while (child)
   {
      /* TODO: implement validation of "general" keywords */
      YAML_NODE_SET_VALID(child);

      if (!strcmp(child->key, "warmup") || !strcmp(child->key, "statistics") ||
          !strcmp(child->key, "use_millisec") ||
          !strcmp(child->key, "print_config_params"))
      {
         if (!strcmp(child->val, "off") || !strcmp(child->val, "no") ||
             !strcmp(child->val, "false") || !strcmp(child->val, "0") ||
             !strcmp(child->val, "n"))
         {
            if (!strcmp(child->key, "warmup"))
            {
               iargs->warmup = 0;
            }
            else if (!strcmp(child->key, "statistics"))
            {
               iargs->statistics = 0;
            }
            else if (!strcmp(child->key, "print_config_params"))
            {
               iargs->print_config_params = 0;
            }
            else if (!strcmp(child->key, "use_millisec"))
            {
               StatsTimerSetSeconds();
            }
         }
         else if (!strcmp(child->val, "on") || !strcmp(child->val, "yes") ||
                  !strcmp(child->val, "true") || !strcmp(child->val, "1") ||
                  !strcmp(child->val, "y"))
         {
            if (!strcmp(child->key, "warmup"))
            {
               iargs->warmup = 1;
            }
            else if (!strcmp(child->key, "statistics"))
            {
               iargs->statistics = 1;
            }
            else if (!strcmp(child->key, "print_config_params"))
            {
               iargs->print_config_params = 1;
            }
            else if (!strcmp(child->key, "use_millisec"))
            {
               StatsTimerSetMilliseconds();
            }
         }
         else
         {
            if (!strcmp(child->key, "warmup"))
            {
               iargs->warmup = 0;
            }
            else if (!strcmp(child->key, "statistics"))
            {
               iargs->statistics = 0;
            }
            else if (!strcmp(child->key, "print_config_params"))
            {
               iargs->print_config_params = 0;
            }
            ErrorCodeSet(ERROR_INVALID_VAL);
         }
      }
      else if (!strcmp(child->key, "num_repetitions"))
      {
         iargs->num_repetitions = (int)strtol(child->val, NULL, 10);
      }
      else if (!strcmp(child->key, "dev_pool_size"))
      {
         iargs->dev_pool_size = GB_TO_BYTES * strtod(child->val, NULL);
      }
      else if (!strcmp(child->key, "uvm_pool_size"))
      {
         iargs->uvm_pool_size = GB_TO_BYTES * strtod(child->val, NULL);
      }
      else if (!strcmp(child->key, "host_pool_size"))
      {
         iargs->host_pool_size = GB_TO_BYTES * strtod(child->val, NULL);
      }
      else if (!strcmp(child->key, "pinned_pool_size"))
      {
         iargs->pinned_pool_size = GB_TO_BYTES * strtod(child->val, NULL);
      }
      else
      {
         ErrorCodeSet(ERROR_INVALID_KEY);
      }

      child = child->next;
   }
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
   YAMLnode *parent = NULL;

   parent = YAMLnodeFindByKey(tree->root, "solver");
   if (!parent)
   {
      // TODO: Add "library" mode to skip the following checks
      // ErrorCodeSet(ERROR_MISSING_KEY);
      // ErrorMsgAddMissingKey("solver");
      return;
   }
   YAML_NODE_SET_VALID(parent);

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

      /* Hack for setting default parameters */
      YAMLnode *dummy           = YAMLnodeCreate("solver", "", 0);
      dummy->children           = YAMLnodeCreate(parent->val, "", 1);
      dummy->children->children = YAMLnodeCreate("print_level", "0", 2);

      SolverSetArgsFromYAML(&iargs->solver, dummy);

      /* Free memory */
      YAMLnodeDestroy(dummy);
   }
}

/*-----------------------------------------------------------------------------
 * InputArgsParsePrecon
 *-----------------------------------------------------------------------------*/

void
InputArgsParsePrecon(input_args *iargs, YAMLtree *tree)
{
   YAMLnode *parent = NULL;

   parent = YAMLnodeFindByKey(tree->root, "preconditioner");
   if (!parent)
   {
      ErrorCodeSet(ERROR_MISSING_KEY);
      ErrorMsgAddMissingKey("preconditioner");
      return;
   }
   YAML_NODE_SET_VALID(parent);

   /* Check if the solver type was set with a single (key, val) pair */
   if (!strcmp(parent->val, ""))
   {
      /* Case A: preconditioner is a sequence of variants (possibly different types):
       * preconditioner:
       *   - amg: ...
       *   - ilu: ...
       */
      YAMLnode **precon_items = NULL;
      int        num_items    = YAMLnodeCollectSequenceItems(parent, &precon_items);
      if (num_items > 0)
      {
         iargs->num_precon_variants = num_items;
         iargs->precon_methods = (precon_t *)malloc((size_t)num_items * sizeof(precon_t));
         iargs->precon_variants =
            (precon_args *)malloc((size_t)num_items * sizeof(precon_args));

         for (int vi = 0; vi < num_items; vi++)
         {
            YAMLnode *item = precon_items[vi];
            if (!item->children || item->children->next)
            {
               ErrorCodeSet(ERROR_INVALID_KEY);
               ErrorMsgAdd(
                  "Each preconditioner variant must contain exactly one type key");
               free(precon_items);
               return;
            }

            YAMLnode *type = item->children;
            precon_t  method =
               (precon_t)StrIntMapArrayGetImage(PreconGetValidTypeIntMap(), type->key);
            iargs->precon_methods[vi] = method;

            PreconSetDefaultArgs(&iargs->precon_variants[vi]);

            /* Wrap for existing parser: preconditioner -> <type> -> fields */
            YAMLnode fake_type = {0};
            fake_type.key      = type->key;
            fake_type.val      = "";
            fake_type.level    = type->level;
            fake_type.valid    = YAML_NODE_VALID;
            fake_type.children = type->children;
            fake_type.next     = NULL;

            YAMLnode fake_parent = {0};
            fake_parent.key      = "preconditioner";
            fake_parent.val      = "";
            fake_parent.level    = parent->level;
            fake_parent.valid    = YAML_NODE_VALID;
            fake_parent.children = &fake_type;
            fake_parent.next     = NULL;

            PreconSetArgsFromYAML(&iargs->precon_variants[vi], &fake_parent);

            /* Clean up mapped_val allocated during validation on fake nodes */
            free(fake_type.mapped_val);
            free(fake_parent.mapped_val);
         }

         free(precon_items);

         /* Set active variant to first one */
         iargs->active_precon_variant = 0;
         iargs->precon_method         = iargs->precon_methods[0];
         iargs->precon                = iargs->precon_variants[0];
         return;
      }

      /* Check if a preconditioner type was set */
      if (!parent->children)
      {
         ErrorCodeSet(ERROR_MISSING_PRECON);
         return;
      }

      /* Check if more than one preconditioner type was set (this is not supported!) */
      if (parent->children->next)
      {
         ErrorCodeSet(ERROR_EXTRA_KEY);
         ErrorMsgAddExtraKey(parent->children->next->key);
         return;
      }

      YAMLnode *type_node = parent->children;
      iargs->precon_method =
         (precon_t)StrIntMapArrayGetImage(PreconGetValidTypeIntMap(), type_node->key);

      /* Check if this type node has sequence items (variants) */
      YAML_NODE_SET_VALID(type_node);
      YAMLnode **seq_items    = NULL;
      int        num_variants = YAMLnodeCollectSequenceItems(type_node, &seq_items);

      if (num_variants > 0)
      {
         /* Multiple variants: allocate arrays and parse each */
         iargs->num_precon_variants = num_variants;
         iargs->precon_methods =
            (precon_t *)malloc((size_t)num_variants * sizeof(precon_t));
         iargs->precon_variants =
            (precon_args *)malloc((size_t)num_variants * sizeof(precon_args));

         for (int variant_idx = 0; variant_idx < num_variants; variant_idx++)
         {
            YAMLnode *seq_item = seq_items[variant_idx];

            /* Set method for this variant */
            iargs->precon_methods[variant_idx] = iargs->precon_method;

            /* Initialize defaults */
            PreconSetDefaultArgs(&iargs->precon_variants[variant_idx]);

            /* Minimal wrapper parent -> type -> seq_item children */
            YAMLnode fake_type = {0};
            fake_type.key      = type_node->key;
            fake_type.val      = "";
            fake_type.level    = type_node->level;
            fake_type.valid    = YAML_NODE_VALID;
            fake_type.children = seq_item->children;
            fake_type.next     = NULL;

            YAMLnode fake_parent = {0};
            fake_parent.key      = "preconditioner";
            fake_parent.val      = "";
            fake_parent.level    = parent->level;
            fake_parent.valid    = YAML_NODE_VALID;
            fake_parent.children = &fake_type;
            fake_parent.next     = NULL;

            PreconSetArgsFromYAML(&iargs->precon_variants[variant_idx], &fake_parent);

            /* Clean up mapped_val allocated during validation on fake nodes */
            free(fake_type.mapped_val);
            fake_type.mapped_val = NULL;
            free(fake_parent.mapped_val);
            fake_parent.mapped_val = NULL;

            YAML_NODE_SET_VALID(seq_item);
         }

         free(seq_items);

         /* Set active variant to first one */
         iargs->active_precon_variant = 0;
         iargs->precon                = iargs->precon_variants[0];

         YAML_NODE_SET_VALID(type_node);
      }
      else
      {
         /* Single variant: allocate arrays and populate */
         iargs->num_precon_variants = 1;
         iargs->precon_methods      = (precon_t *)malloc(sizeof(precon_t));
         iargs->precon_variants     = (precon_args *)malloc(sizeof(precon_args));

         iargs->precon_methods[0] = iargs->precon_method;
         PreconSetDefaultArgs(&iargs->precon_variants[0]);
         PreconSetArgsFromYAML(&iargs->precon_variants[0], parent);

         /* Set active variant */
         iargs->active_precon_variant = 0;
         iargs->precon                = iargs->precon_variants[0];
      }
   }
   else
   {
      iargs->precon_method =
         (precon_t)StrIntMapArrayGetImage(PreconGetValidTypeIntMap(), parent->val);

      /* Hack for setting default parameters */
      YAMLnode *dummy           = YAMLnodeCreate("preconditioner", "", 0);
      dummy->children           = YAMLnodeCreate(parent->val, "", 1);
      dummy->children->children = YAMLnodeCreate("print_level", "0", 2);

      /* Single variant: allocate arrays and populate */
      iargs->num_precon_variants = 1;
      iargs->precon_methods      = (precon_t *)malloc(sizeof(precon_t));
      iargs->precon_variants     = (precon_args *)malloc(sizeof(precon_args));

      iargs->precon_methods[0] = iargs->precon_method;
      PreconSetDefaultArgs(&iargs->precon_variants[0]);
      PreconSetArgsFromYAML(&iargs->precon_variants[0], dummy);

      /* Free memory */
      YAMLnodeDestroy(dummy);

      /* Set active variant */
      iargs->active_precon_variant = 0;
      iargs->precon                = iargs->precon_variants[0];

      YAML_NODE_SET_VALID(parent);
   }
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
   char       *config_base = NULL;

   MPI_Comm_rank(comm, &myid);

   /* Read input arguments from file or string.
    *
    * Supported calling patterns:
    * - Legacy/library mode: argv[0] is YAML filename (and argv[1..] are override pairs)
    * - Driver mode: argv is the full CLI (contains YAML filename somewhere and optionally
    * -a ...)
    * - Unit tests: argv[0] is a YAML string (and argv[1..] are override pairs)
    */
   int config_idx = -1;
   for (int i = 0; i < argc; i++)
   {
      if (argv[i] && IsYAMLFilename(argv[i]))
      {
         config_idx = i;
         break;
      }
   }

   if (argc > 0 && IsYAMLFilename(argv[0]))
   {
      /* Treat as file input - will error if file doesn't exist */
      InputArgsRead(comm, argv[0], &base_indent, &text);

      /* Return if there was an error reading the file */
      if (ErrorCodeActive())
      {
         *args_ptr = NULL;
         return;
      }

      SplitFilename(argv[0], &config_dir, &config_base);
   }
   else if (config_idx >= 0)
   {
      /* Driver-style argv: find YAML filename anywhere */
      InputArgsRead(comm, argv[config_idx], &base_indent, &text);
      if (ErrorCodeActive())
      {
         *args_ptr = NULL;
         return;
      }

      SplitFilename(argv[config_idx], &config_dir, &config_base);
   }
   else
   {
      /* Direct YAML string input */
      if (argv[0] == NULL)
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("YAML string input is NULL");
         *args_ptr = NULL;
         return;
      }
      text = strdup(argv[0]); // Make a copy since we'll free it later
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
   if (argc > 1)
   {
      /* Update YAML tree with command line arguments info */
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
   free(config_base);

   /*--------------------------------------------
    * Parse file sections
    *-------------------------------------------*/

   InputArgsCreate(lib_mode, &iargs);
   InputArgsParseGeneral(iargs, tree);
   InputArgsParseLinearSystem(iargs, tree);
   InputArgsParseSolver(iargs, tree);
   InputArgsParsePrecon(iargs, tree);

   /* Set auxiliary data in the Stats structure */
   StatsSetNumReps(iargs->num_repetitions);
   /* Note: num_systems is the base number; variants are handled in the driver loop */
   StatsSetNumLinearSystems(iargs->ls.num_systems);

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
   if (!myid && iargs->print_config_params)
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
   }
   MPI_Barrier(comm);

   /* Free memory */
   YAMLtreeDestroy(&tree);

   /* Set output pointer */
   *args_ptr = iargs;
}
