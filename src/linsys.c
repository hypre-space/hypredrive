/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "linsys.h"

#if 0
/*-----------------------------------------------------------------------------
 * LinearSystemGetValidTypes
 *-----------------------------------------------------------------------------*/

StrIntMap*
LinearSystemGetValidTypes(void)
{
   static StrIntMap out[] = {{"ij", 1},
                             {"parcsr", 2}};

   return out;
}
#endif

/*-----------------------------------------------------------------------------
 * LinearSystemGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
LinearSystemGetValidKeys(void)
{
   static const char* keys[] = {"matrix_filename",
                                "precmat_filename",
                                "rhs_filename",
                                "x0_filename",
                                "sol_filename",
                                "dofmap_filename",
                                "init_guess_mode",
                                "rhs_mode",
                                "type"};

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * LinearSystemGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
LinearSystemGetValidValues(const char* key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"ij",     1},
                                {"parcsr", 2}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else if (!strcmp(key, "rhs_mode") ||
            !strcmp(key, "init_guess_mode") )
   {
      static StrIntMap map[] = {{"zeros",  0},
                                {"ones",   1},
                                {"file",   2},
                                {"random", 3}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }

   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetDefaultArgs
 *-----------------------------------------------------------------------------*/

int
LinearSystemSetDefaultArgs(LS_args *args)
{
   strcpy(args->matrix_filename, "");
   strcpy(args->precmat_filename, "");
   strcpy(args->rhs_filename, "");
   strcpy(args->x0_filename, "");
   strcpy(args->sol_filename, "");
   strcpy(args->dofmap_filename, "");
   args->init_guess_mode = 0;
   args->rhs_mode = 0;
   args->type = 0;

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

int
LinearSystemSetArgsFromYAML(LS_args *args, YAMLnode* parent)
{
   YAMLnode    *child;

   child = parent->children;
   while (child)
   {
      YAML_VALIDATE_NODE(child,
                         LinearSystemGetValidKeys,
                         LinearSystemGetValidValues);

      if (!strcmp(child->key, "matrix_filename"))
      {
         strcpy(args->matrix_filename, child->val);
      }
      else if (!strcmp(child->key, "precmat_filename"))
      {
         strcpy(args->precmat_filename, child->val);
      }
      else if (!strcmp(child->key, "rhs_filename"))
      {
         strcpy(args->rhs_filename, child->val);
      }
      else if (!strcmp(child->key, "x0_filename"))
      {
         strcpy(args->x0_filename, child->val);
      }
      else if (!strcmp(child->key, "sol_filename"))
      {
         strcpy(args->sol_filename, child->val);
      }
      else if (!strcmp(child->key, "dofmap_filename"))
      {
         strcpy(args->dofmap_filename, child->val);
      }
      else if (!strcmp(child->key, "init_guess_mode"))
      {
         args->init_guess_mode = (HYPRE_Int) atoi(child->val);
      }
      else if (!strcmp(child->key, "rhs_mode"))
      {
         args->rhs_mode = (HYPRE_Int) atoi(child->val);
      }
      else if (!strcmp(child->key, "type"))
      {
         args->type = (HYPRE_Int) atoi(child->val);
      }
      else
      {
         ErrorMsgAddUnknownKey(child->key);
      }

      child = child->next;
   }


   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

int
LinearSystemReadMatrix(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix *matrix_ptr)
{
   /* Read matrix */
   if (args->matrix_filename[0] != '\0')
   {
      if (CheckBinaryDataExists(args->matrix_filename))
      {
         HYPRE_IJMatrixReadBinary(args->matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
      }
      else
      {
         HYPRE_IJMatrixRead(args->matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
      }
   }
   else
   {
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

int
LinearSystemSetRHS(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat, HYPRE_IJVector *rhs_ptr)
{
   HYPRE_BigInt    ilower, iupper;
   HYPRE_BigInt    jlower, jupper;

   /* Read right-hand-side vector */
   if (args->rhs_filename[0] == '\0')
   {
      HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
      HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(*rhs_ptr);

      switch (args->rhs_mode)
      {
         case 1:
         default:
            /* TODO: Vector of ones */
            break;
      }
   }
   else
   {
      if (CheckBinaryDataExists(args->rhs_filename))
      {
         HYPRE_IJVectorReadBinary(args->rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
      }
      else
      {
         HYPRE_IJVectorRead(args->rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
      }
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetInitialGuess
 *-----------------------------------------------------------------------------*/

int
LinearSystemSetInitialGuess(MPI_Comm comm,
                            LS_args *args,
                            HYPRE_IJMatrix mat,
                            HYPRE_IJVector rhs,
                            HYPRE_IJVector *x0_ptr)
{
   HYPRE_BigInt    jlower, jupper;

   if (args->precmat_filename[0] == '\0')
   {
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x0_ptr);
      HYPRE_IJVectorSetObjectType(*x0_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(*x0_ptr);

      switch (args->init_guess_mode)
      {
         case 0:
         default:
            /* Vector of zeros */
            break;

         case 1:
            /* TODO: Vector of ones */
            break;
      }
   }
   else
   {
      if (CheckBinaryDataExists(args->x0_filename))
      {
         HYPRE_IJVectorReadBinary(args->x0_filename, comm, HYPRE_PARCSR, x0_ptr);
      }
      else
      {
         HYPRE_IJVectorRead(args->x0_filename, comm, HYPRE_PARCSR, x0_ptr);
      }
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

int
LinearSystemSetPrecMatrix(MPI_Comm comm,
                          LS_args *args,
                          HYPRE_IJMatrix mat,
                          HYPRE_IJMatrix *precmat_ptr)
{
   if (args->precmat_filename[0] == '\0')
   {
      *precmat_ptr = mat;
   }
   else
   {
      HYPRE_IJMatrixRead(args->precmat_filename, comm, HYPRE_PARCSR, precmat_ptr);
   }

   return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------
 * LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

int
LinearSystemReadDofmap(MPI_Comm comm, LS_args *args, HYPRE_IntArray *dofmap_ptr)
{
   if (args->dofmap_filename[0] == '\0')
   {
      dofmap_ptr->num_entries = 0;
      dofmap_ptr->num_unique_entries = 0;
      dofmap_ptr->data = NULL;
   }
   else
   {
      return IntArrayRead(comm, args->dofmap_filename, dofmap_ptr);
   }

   /* TODO: Print how many dofs types we have (min, max, avg, sum) accross ranks */

   return EXIT_SUCCESS;
}
