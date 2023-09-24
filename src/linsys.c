/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "linsys.h"

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, x0_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sol_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_filename, FIELD_TYPE_STRING),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_guess_mode, FIELD_TYPE_INT),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_mode, FIELD_TYPE_INT),
   FIELD_OFFSET_MAP_ENTRY(LS_args, type, FIELD_TYPE_INT)
};

#define LS_NUM_FIELDS (sizeof(ls_field_offset_map) / sizeof(ls_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * LinearSystemSetFieldByName
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetFieldByName(LS_args *args, const char *name, const char *value)
{
   size_t i;

   for (i = 0; i < LS_NUM_FIELDS; i++)
   {
      if (!strcmp(ls_field_offset_map[i].name, name))
      {
         switch (ls_field_offset_map[i].type)
         {
            case FIELD_TYPE_INT:
               sscanf(value, "%d", (int*)((char*) args + ls_field_offset_map[i].offset));
               break;

            case FIELD_TYPE_DOUBLE:
               sscanf(value, "%lf", (double*)((char*) args + ls_field_offset_map[i].offset));
                    break;

            case FIELD_TYPE_CHAR:
               sscanf(value, "%c", (char*)((char*) args + ls_field_offset_map[i].offset));
               break;

            case FIELD_TYPE_STRING:
               snprintf((char*) args + ls_field_offset_map[i].offset,
                        MAX_FILENAME_LENGTH, "%s", value);
               break;
         }
         return;
      }
   }
}

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

#if 0
/*-----------------------------------------------------------------------------
 * LinearSystemGetDataTypes
 *-----------------------------------------------------------------------------*/

StrIntMapArray
LinearSystemGetDataTypes(void)
{
   static StrIntMap map[] = {{"matrix_filename",  DATA_TYPE_CHAR},
                             {"precmat_filename", DATA_TYPE_CHAR},
                             {"rhs_filename",     DATA_TYPE_CHAR},
                             {"x0_filename",      DATA_TYPE_CHAR},
                             {"sol_filename",     DATA_TYPE_CHAR},
                             {"dofmap_filename",  DATA_TYPE_CHAR},
                             {"init_guess_mode",  DATA_TYPE_INT},
                             {"rhs_mode",         DATA_TYPE_INT},
                             {"type",             DATA_TYPE_INT}};

   return STR_INT_MAP_ARRAY_CREATE(map);
}
#endif

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

      YAML_SET_ARG(args,
                   child,
                   LinearSystemSetFieldByName);

#if 0
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
#endif

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
