/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "args.h"
#include "yaml.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*-----------------------------------------------------------------------------
 * InputArgsCreate
 *-----------------------------------------------------------------------------*/

void
InputArgsCreate(bool lib_mode, input_args **iargs_ptr)
{
   input_args *iargs = (input_args*) malloc(sizeof(input_args));

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
   YAMLnode    *parent;
   YAMLnode    *child;

   parent = YAMLnodeFindByKey(tree->root, "general");
   if (!parent)
   {
      /* The "general" key is not mandatory,
         so we don't set an error code if it's not found. */
      return;
   }

   child = parent->children;
   while (child)
   {
      if (!strcmp(child->key, "warmup") ||
          !strcmp(child->key, "statistics") ||
          !strcmp(child->key, "use_millisec") ||
          !strcmp(child->key, "print_config_params"))
      {
         if (!strcmp(child->val, "off") ||
             !strcmp(child->val, "no") ||
             !strcmp(child->val, "false") ||
             !strcmp(child->val, "0")  ||
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
         else if (!strcmp(child->val, "on") ||
                  !strcmp(child->val, "yes") ||
                  !strcmp(child->val, "true") ||
                  !strcmp(child->val, "1")  ||
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
         iargs->num_repetitions = atoi(child->val);
      }
      else if (!strcmp(child->key, "dev_pool_size"))
      {
         iargs->dev_pool_size = GB_TO_BYTES * atof(child->val);
      }
      else if (!strcmp(child->key, "uvm_pool_size"))
      {
         iargs->uvm_pool_size = GB_TO_BYTES * atof(child->val);
      }
      else if (!strcmp(child->key, "host_pool_size"))
      {
         iargs->host_pool_size = GB_TO_BYTES * atof(child->val);
      }
      else if (!strcmp(child->key, "pinned_pool_size"))
      {
         iargs->pinned_pool_size = GB_TO_BYTES * atof(child->val);
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
   const char   key[]  = {"linear_system"};
   YAMLnode    *parent = YAMLnodeFindByKey(tree->root, key);

   if (!parent)
   {
      // TODO: Add "library" mode to skip the following checks
      //ErrorCodeSet(ERROR_MISSING_KEY);
      //ErrorMsgAddMissingKey(key);
      return;
   }
   else
   {
      YAML_NODE_SET_VALID_IF_NO_VAL(parent);
      if (YAML_NODE_GET_VALIDITY(parent) == YAML_NODE_UNEXPECTED_VAL)
      {
         ErrorMsgAddUnexpectedVal(key);
      }
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
   YAMLnode  *parent;

   parent = YAMLnodeFindByKey(tree->root, "solver");
   if (!parent)
   {
      // TODO: Add "library" mode to skip the following checks
      //ErrorCodeSet(ERROR_MISSING_KEY);
      //ErrorMsgAddMissingKey("solver");
      return;
   }
   else
   {
      YAML_NODE_SET_VALID(parent);
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

      iargs->solver_method = (solver_t) StrIntMapArrayGetImage(SolverGetValidTypeIntMap(),
                                                               parent->children->key);

      SolverSetArgsFromYAML(&iargs->solver, parent);
   }
   else
   {
      iargs->solver_method = (solver_t) StrIntMapArrayGetImage(SolverGetValidTypeIntMap(),
                                                               parent->val);

      /* Hack for setting default parameters */
      YAMLnode *dummy = YAMLnodeCreate("solver", "", 0);
      dummy->children = YAMLnodeCreate(parent->val, "", 1);
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
   YAMLnode  *parent;

   parent = YAMLnodeFindByKey(tree->root, "preconditioner");
   if (!parent)
   {
      ErrorCodeSet(ERROR_MISSING_KEY);
      ErrorMsgAddMissingKey("preconditioner");
      return;
   }
   else
   {
      YAML_NODE_SET_VALID(parent);
   }

   /* Check if the solver type was set with a single (key, val) pair */
   if (!strcmp(parent->val, ""))
   {
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

      iargs->precon_method = (precon_t) StrIntMapArrayGetImage(PreconGetValidTypeIntMap(),
                                                               parent->children->key);

      PreconSetArgsFromYAML(&iargs->precon, parent);
   }
   else
   {
      iargs->precon_method = (precon_t) StrIntMapArrayGetImage(PreconGetValidTypeIntMap(),
                                                               parent->val);

      /* Hack for setting default parameters */
      YAMLnode *dummy = YAMLnodeCreate("preconditioner", "", 0);
      dummy->children = YAMLnodeCreate(parent->val, "", 1);
      dummy->children->children = YAMLnodeCreate("print_level", "0", 2);

      PreconSetArgsFromYAML(&iargs->precon, dummy);

      /* Free memory */
      YAMLnodeDestroy(dummy);
   }
}

/*-----------------------------------------------------------------------------
 * InputArgsRead
 *-----------------------------------------------------------------------------*/

void
InputArgsRead(MPI_Comm comm, char *filename, char **text_ptr)
{
   size_t        text_size = 0;
   int           level = 0;
   char         *text = NULL;
   char         *dirname = NULL;
   char         *basename = NULL;
   int           myid;

   MPI_Comm_rank(comm, &myid);

   /* Split filename into dirname and basename */
   SplitFilename(filename, &dirname, &basename);

   /* Rank 0: Expand text from base file */
   if (!myid) YAMLtextRead(dirname, basename, level, &text_size, &text);
   if (DistributedErrorCodeActive(comm))
   {
      return;
   }

   /* Free memory */
   free(dirname);
   free(basename);

   /* Broadcast the text size */
   MPI_Bcast(&text_size, 1, MPI_UNSIGNED_LONG, 0, comm);

   /* Broadcast the text */
   if (myid) text = (char*) malloc(text_size);
   MPI_Bcast(text, text_size, MPI_CHAR, 0, comm);

   /* Set output pointer */
   *text_ptr = text;
}

/*-----------------------------------------------------------------------------
 * InputArgsParse
 *-----------------------------------------------------------------------------*/

void
InputArgsParse(MPI_Comm comm, bool lib_mode, int argc, char **argv, input_args **args_ptr)
{
   input_args   *iargs;
   char         *text;
   YAMLtree     *tree;
   int           myid;

   MPI_Comm_rank(comm, &myid);

   /* Read input arguments from file */
   InputArgsRead(comm, argv[0], &text);

   /* Build YAML tree */
   YAMLtreeBuild(text, &tree);

   /* Return earlier if YAML tree was not built properly */
   if (!myid && ErrorCodeActive())
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
      ErrorMsgPrintAndAbort(comm);
   }
   MPI_Barrier(comm);

   /* Free memory */
   free(text);

   /* TODO: check if any config option has been passed in via CLI.
            If so, overwrite the data stored in the YAMLtree object
            with it. */
   if (argc > 1)
   {
      /* Update YAML tree with command line arguments info */
      YAMLtreeUpdate(argc - 1, argv + 1, tree);
   }

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
   StatsSetNumLinearSystems(iargs->ls.num_systems);

   /* Rank 0: Print tree to stdout */
   if (!myid && iargs->print_config_params)
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);

      if (ErrorCodeActive())
      {
         ErrorMsgPrintAndAbort(comm);
      }
   }
   MPI_Barrier(comm);

   /* Free memory */
   YAMLtreeDestroy(&tree);

   /* Set output pointer */
   *args_ptr = iargs;
}
