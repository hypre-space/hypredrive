/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC, HYPRE and GEOS
 * Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "linsys.h"

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, x0_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sol_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_guess_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, type, FieldTypeIntSet)
};

#define LS_NUM_FIELDS (sizeof(ls_field_offset_map) / sizeof(ls_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * LinearSystemSetFieldByName
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetFieldByName(LS_args *args, YAMLnode *node)
{
   for (size_t i = 0; i < LS_NUM_FIELDS; i++)
   {
      if (!strcmp(ls_field_offset_map[i].name, node->key))
      {
         ls_field_offset_map[i].setter(
            (void*)((char*) args + ls_field_offset_map[i].offset),
            node);
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
   static const char* keys[LS_NUM_FIELDS];

   for (size_t i = 0; i < LS_NUM_FIELDS; i++)
   {
      keys[i] = ls_field_offset_map[i].name;
   }

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
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
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
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetArgsFromYAML(LS_args *args, YAMLnode* parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child,
                         LinearSystemGetValidKeys,
                         LinearSystemGetValidValues);

      YAML_NODE_SET_FIELD(child,
                          args,
                          LinearSystemSetFieldByName);
   }
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
