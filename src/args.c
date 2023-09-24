/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "args.h"
#include "yaml.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

/*-----------------------------------------------------------------------------
 * InputArgsCreate
 *-----------------------------------------------------------------------------*/

int
InputArgsCreate(input_args **iargs_ptr)
{
   input_args *iargs = (input_args*) malloc(sizeof(input_args));

   /* Set general default options */
   iargs->warmup = 0;
   iargs->num_repetitions = 1;

   /* Set default Linear System options */
   LinearSystemSetDefaultArgs(&iargs->ls);

   /* Set default preconditioner and solver */
   iargs->solver_method = SOLVER_PCG;
   iargs->precon_method = PRECON_BOOMERAMG;

   *iargs_ptr = iargs;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * InputArgsDestroy
 *-----------------------------------------------------------------------------*/

int
InputArgsDestroy(input_args **iargs_ptr)
{
   free(*iargs_ptr);
   *iargs_ptr = NULL;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * InputArgsParseGeneral
 *-----------------------------------------------------------------------------*/

int
InputArgsParseGeneral(input_args *iargs, YAMLtree *tree)
{
   YAMLnode    *parent;
   YAMLnode    *child;

   parent = YAMLnodeFindByKey(tree->root, "general");
   if (!parent)
   {
      return EXIT_SUCCESS;
   }

   child  = parent->children;
   while (child)
   {
      if (!strcmp(child->key, "warmup"))
      {
         if (!strcmp(child->val, "no") ||
             !strcmp(child->val, "0")  ||
             !strcmp(child->val, "n"))
         {
            iargs->warmup = 0;
         }
         else if (!strcmp(child->val, "yes") ||
                  !strcmp(child->val, "1")  ||
                  !strcmp(child->val, "y"))
         {
            iargs->warmup = 0;
         }
         else
         {
            iargs->warmup = 0;
            ErrorMsgAddInvalidKeyValPair(child->key, child->val);
         }
      }
      else if (!strcmp(child->key, "num_repetitions"))
      {
         iargs->num_repetitions = (HYPRE_Int) atoi(child->val);
      }
      else
      {
         ErrorMsgAddUnknownKey(child->key);
      }

      child = child->next;
   }

   ErrorMsgPrint();
   ErrorMsgClear();

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * InputArgsParseLinearSystem
 *-----------------------------------------------------------------------------*/

void
InputArgsParseLinearSystem(input_args *iargs, YAMLtree *tree)
{
   YAMLnode    *parent;

   parent = YAMLnodeFindByKey(tree->root, "linear_system");
   if (!parent)
   {
      ErrorCodeSet(ERROR_MISSING_KEY);
      ErrorMsgAddMissingKey("linear_system");
      return;
   }
   else
   {
      YAML_NODE_SET_VALID(parent);
   }

   LinearSystemSetArgsFromYAML(&iargs->ls, parent);
   ErrorMsgPrint();
   ErrorMsgClear();
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
      ErrorCodeSet(ERROR_MISSING_KEY);
      ErrorMsgAddMissingKey("solver");
      return;
   }
   else
   {
      YAML_NODE_SET_VALID(parent);
   }

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

   iargs->solver_method = StrIntMapArrayGetImage(SolverGetValidTypeIntMap(),
                                                 parent->children->key);
   SolverSetArgsFromYAML(&iargs->solver, parent);
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

   iargs->precon_method = StrIntMapArrayGetImage(PreconGetValidTypeIntMap(),
                                                 parent->children->key);
   PreconSetArgsFromYAML(&iargs->precon, parent);
}

/*-----------------------------------------------------------------------------
 * InputArgsExpand
 *
 * Expands an input file with include directives to a single text pointer.
 *-----------------------------------------------------------------------------*/

int
InputArgsExpand(const char* basefilename, int *text_size, char **text_ptr)
{
   FILE      *fp;
   size_t     file_sizes[3], file_sizes_sum[3];
   char       include_keys[2][24]  = {"solver_include", "preconditioner_include"};
   char      *include_filenames[2] = {NULL, NULL};
   char       line[MAX_LINE_LENGTH];
   char      *key, *val, *sep;
   char      *text;
   int        i;

   fp = fopen(basefilename, "r");
   if (!fp)
   {
      ErrorMsgAddInvalidFilename(basefilename);
      return EXIT_FAILURE;
   }

   /* Determine the base file size */
   fseek(fp, 0, SEEK_END);
   file_sizes[0] = ftell(fp);
   file_sizes_sum[0] = file_sizes[0];
   file_sizes_sum[1] = file_sizes[0];
   file_sizes_sum[2] = file_sizes[0];
   fseek(fp, 0, SEEK_SET);

   /* Do we have an include file for the preconditioner and solver? */
   while (fgets(line, sizeof(line), fp))
   {
      /* Remove trailing newline character */
      line[strcspn(line, "\n")] = '\0';

      /* Ignore empty lines and comments */
      if (line[0] == '\0' || line[0] == '#')
      {
         continue;
      }

      /* Check for divisor character */
      if ((sep = strchr(line, ':')) == NULL)
      {
         continue;
      }

      *sep = '\0';
      key = line;
      val = sep + 1;

      /* Trim leading spaces */
      while (*key == ' ') key++;
      while (*val == ' ') val++;

      for (i = 0; i < 2; i++)
      {
         if (!strcmp(key, include_keys[i]))
         {
            include_filenames[i] = strdup(val);
         }
      }
   }
   fclose(fp);

   /* Find sizes of included files */
   for (i = 0; i < 2; i++)
   {
      if (include_filenames[i])
      {
         fp = fopen(include_filenames[i], "r");
         if (!fp)
         {
            ErrorMsgAddInvalidFilename(include_filenames[i]);
            return EXIT_FAILURE;
         }

         fseek(fp, 0, SEEK_END);
         file_sizes[i + 1] = ftell(fp);
         file_sizes_sum[i + 1] = file_sizes_sum[i] + file_sizes[i + 1];
         fseek(fp, 0, SEEK_SET);
         fclose(fp);
      }
   }

   /* Allocate memory for the expanded text */
   text = (char *) malloc(file_sizes_sum[2] + 1);

   /* Read the expanded text */
   fp = fopen(basefilename, "r");
   if (fread(text, 1, file_sizes[0], fp) != file_sizes[0]) return EXIT_FAILURE;
   fclose(fp);
   for (i = 0; i < 2; i++)
   {
      if (include_filenames[i])
      {
         fp = fopen(include_filenames[i], "r");
         if (fread(text + file_sizes_sum[i], 1, file_sizes[i + 1], fp) != file_sizes[i + 1])
         {
            return EXIT_FAILURE;
         }
         fclose(fp);
      }
   }

   /* Null-terminate the string */
   text[file_sizes_sum[2]] = '\0';

   /* Free memory */
   if (include_filenames[0]) free(include_filenames[0]);
   if (include_filenames[1]) free(include_filenames[1]);

   /* Set output pointer */
   *text_ptr  = text;
   *text_size = file_sizes_sum[2] + 1;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * InputArgsParse
 *-----------------------------------------------------------------------------*/

int
InputArgsParse(MPI_Comm comm, int argc, char **argv, input_args **args_ptr)
{
   input_args   *iargs;
   YAMLtree     *tree;

   int           text_size;
   char         *text = NULL;

   int           myid;

   MPI_Comm_rank(comm, &myid);

   /* Rank 0: Expand text from base file */
   if (!myid)
   {
      InputArgsExpand(argv[1], &text_size, &text);
   }

   /* Broadcast the text size */
   MPI_Bcast(&text_size, 1, MPI_INT, 0, comm);

   /* Broadcast the text */
   if (myid)
   {
      text = (char*) malloc(text_size * sizeof(char));
   }
   MPI_Bcast(text, text_size, MPI_CHAR, 0, comm);

   /* Build YAML tree */
   YAMLtreeBuild(text, &tree);

   /* Return earlier if YAML tree was not built properly */
   if (!myid && ErrorCodeActive())
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);
      ErrorMsgPrintAndAbort(comm);
   }
   MPI_Barrier(comm);

   /* TODO: check if any config option has been passed in via CLI.
            If so, overwrite the data stored in the YAMLtree object
            with it. */
   if (argc > 2)
   {
      /* Update YAML tree with command line arguments info */
      YAMLtreeUpdate(argc, argv, tree);
   }

   /*--------------------------------------------
    * Parse file sections
    *-------------------------------------------*/

   InputArgsCreate(&iargs);
   InputArgsParseGeneral(iargs, tree);
   InputArgsParseLinearSystem(iargs, tree);
   InputArgsParseSolver(iargs, tree);
   InputArgsParsePrecon(iargs, tree);

   /* Rank 0: Print tree to stdout */
   if (!myid)
   {
      YAMLtreePrint(tree, YAML_PRINT_MODE_ANY);

      if (ErrorCodeActive())
      {
         ErrorMsgPrintAndAbort(comm);
      }
   }
   MPI_Barrier(comm);

   *args_ptr = iargs;
   YAMLtreeDestroy(&tree);
   free(text);

   return EXIT_SUCCESS;
}
