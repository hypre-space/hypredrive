/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "linsys.h"
#include "HYPRE_parcsr_mv.h" /* TODO: remove after implementing IJVectorClone/Copy */
#include "_hypre_parcsr_mv.h" /* TODO: remove after implementing IJVectorMigrate/IJMatrix */
#include "_hypre_IJ_mv.h" /* TODO: remove after implementing IJVectorClone */

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, x0_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sol_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_guess_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, type, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, exec_policy, FieldTypeIntSet)
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
   else if (!strcmp(key, "exec_policy"))
   {
      static StrIntMap map[] = {{"host",   0},
                                {"device", 1}};
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
   args->exec_policy = 0;
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

void
LinearSystemReadMatrix(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix *matrix_ptr)
{
   StatsTimerStart("matrix");

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

      /* Check if hypre had problems reading the input file */
      if (HYPRE_GetError())
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(args->matrix_filename);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(args->matrix_filename);
   }

   /* Migrate the matrix? TODO: use IJMatrixMigrate */
   if (args->exec_policy)
   {
      HYPRE_ParCSRMatrix   par_A;
      void                *obj;

      HYPRE_IJMatrixGetObject(*matrix_ptr, &obj);
      par_A  = (HYPRE_ParCSRMatrix) obj;

      hypre_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_DEVICE);
   }

   StatsTimerFinish("matrix");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetRHS(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat, HYPRE_IJVector *rhs_ptr)
{
   StatsTimerStart("rhs");

   HYPRE_BigInt    ilower, iupper;
   HYPRE_BigInt    jlower, jupper;

   /* Read right-hand-side vector */
   if (args->rhs_filename[0] == '\0')
   {
      HYPRE_MemoryLocation memloc = (args->exec_policy) ?
                                    HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

      HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
      HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(*rhs_ptr, memloc);

      /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
      void            *obj;
      HYPRE_ParVector  par_b;
      HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
      par_b = (HYPRE_ParVector) obj;

      switch (args->rhs_mode)
      {
         case 1:
         default:
            /* Vector of ones */
            HYPRE_ParVectorSetConstantValues(par_b, 1);
            break;

         case 3:
            /* Vector of random values */
            HYPRE_ParVectorSetRandomValues(par_b, 2023);
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

      /* Check if hypre had problems reading the input file */
      if (HYPRE_GetError())
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(args->rhs_filename);
      }

      /* Migrate the vector? TODO: use IJVectorMigrate */
      if (args->exec_policy)
      {
         HYPRE_ParVector   par_rhs;
         void             *obj;

         HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
         par_rhs = (HYPRE_ParVector) obj;

         hypre_ParVectorMigrate(par_rhs, HYPRE_MEMORY_DEVICE);
      }
   }

   StatsTimerFinish("rhs");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetInitialGuess
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetInitialGuess(MPI_Comm comm,
                            LS_args *args,
                            HYPRE_IJMatrix mat,
                            HYPRE_IJVector rhs,
                            HYPRE_IJVector *x0_ptr,
                            HYPRE_IJVector *x_ptr)
{
   HYPRE_BigInt         jlower, jupper;
   HYPRE_MemoryLocation memloc = (args->exec_policy) ?
                                 HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   if (args->precmat_filename[0] == '\0')
   {
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x0_ptr);
      HYPRE_IJVectorSetObjectType(*x0_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(*x0_ptr, memloc);

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

      /* Migrate the vector? TODO: use IJVectorMigrate */
      if (args->exec_policy)
      {
         HYPRE_ParVector   par_x0;
         void             *obj;

         HYPRE_IJVectorGetObject(*x0_ptr, &obj);
         par_x0 = (HYPRE_ParVector) obj;

         hypre_ParVectorMigrate(par_x0, HYPRE_MEMORY_DEVICE);
      }
   }

   /* TODO: implement HYPRE_IJVectorClone in hypre */
   HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, jlower, jupper, x_ptr);
   HYPRE_IJVectorSetObjectType(*x_ptr, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(*x_ptr, memloc);
}

/*-----------------------------------------------------------------------------
 * LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

void
LinearSystemResetInitialGuess(HYPRE_IJVector x0_ptr,
                              HYPRE_IJVector x_ptr)
{
   HYPRE_ParVector   par_x0, par_x;
   void             *obj_x0, *obj_x;

   /* TODO: implement HYPRE_IJVectorCopy in hypre */
   HYPRE_IJVectorGetObject(x0_ptr, &obj_x0);
   HYPRE_IJVectorGetObject(x_ptr, &obj_x);
   par_x0 = (HYPRE_ParVector) obj_x0;
   par_x  = (HYPRE_ParVector) obj_x;

   HYPRE_ParVectorCopy(par_x0, par_x);
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetPrecMatrix(MPI_Comm comm,
                          LS_args *args,
                          HYPRE_IJMatrix mat,
                          HYPRE_IJMatrix *precmat_ptr)
{
   if (args->precmat_filename[0] == '\0' ||
       !strcmp(args->precmat_filename, args->matrix_filename))
   {
      *precmat_ptr = mat;
   }
   else
   {
      HYPRE_IJMatrixRead(args->precmat_filename, comm, HYPRE_PARCSR, precmat_ptr);
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

void
LinearSystemReadDofmap(MPI_Comm comm, LS_args *args, IntArray **dofmap_ptr)
{
   if (args->dofmap_filename[0] == '\0')
   {
      *dofmap_ptr = IntArrayCreate(0);
   }
   else
   {
      StatsTimerStart("dofmap");
      IntArrayParRead(comm, args->dofmap_filename, dofmap_ptr);
      StatsTimerFinish("dofmap");
   }

   /* TODO: Print how many dofs types we have (min, max, avg, sum) accross ranks */
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeRHSNorm(HYPRE_IJVector  vec_b,
                           HYPRE_Complex  *b_norm)
{
   HYPRE_IJVectorInnerProd(vec_b, vec_b, b_norm);
   *b_norm = sqrt(*b_norm);
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeResidualNorm(HYPRE_IJMatrix  mat_A,
                                HYPRE_IJVector  vec_b,
                                HYPRE_IJVector  vec_x,
                                HYPRE_Complex  *res_norm)
{
   HYPRE_ParCSRMatrix   par_A;
   HYPRE_ParVector      par_b;
   HYPRE_ParVector      par_x;
   HYPRE_ParVector      par_r;
   HYPRE_IJVector       vec_r;
   void                *obj_A, *obj_b, *obj_x, *obj_r;

   HYPRE_BigInt         jlower, jupper;

   HYPRE_Complex        one = 1.0;
   HYPRE_Complex        neg_one = -1.0;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   HYPRE_IJVectorGetObject(vec_x, &obj_x);

   par_A = (HYPRE_ParCSRMatrix) obj_A;
   par_b = (HYPRE_ParVector) obj_b;
   par_x = (HYPRE_ParVector) obj_x;

   /* TODO: implement IJVectorClone */
   HYPRE_IJVectorGetLocalRange(vec_b, &jlower, &jupper);
   HYPRE_IJVectorCreate(hypre_IJVectorComm(vec_b), jlower, jupper, &vec_r);
   HYPRE_IJVectorSetObjectType(vec_r, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec_r, hypre_IJVectorMemoryLocation(vec_b));
   HYPRE_IJVectorGetObject(vec_r, &obj_r);
   par_r = (HYPRE_ParVector) obj_r;
   HYPRE_ParVectorCopy(par_b, par_r);

   /* Compute residual */
   HYPRE_ParCSRMatrixMatvec(neg_one, par_A, par_x, one, par_r);

   /* Compute residual norm */
   HYPRE_ParVectorInnerProd(par_r, par_r, res_norm);
   *res_norm = sqrt(*res_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_r);
}
