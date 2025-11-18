/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "linsys.h"
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "HYPRE_parcsr_mv.h" /* TODO: remove after implementing IJVectorClone/Copy */
#include "_hypre_IJ_mv.h"    /* TODO: remove after implementing IJVectorClone */
#include "_hypre_parcsr_mv.h" /* TODO: remove after implementing IJVectorMigrate/IJMatrix */

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, dirname, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, x0_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sol_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, digits_suffix, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_suffix, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, last_suffix, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_guess_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_mode, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, type, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precon_reuse, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, exec_policy, FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, eigspec, EigSpecSetArgs)};

#define LS_NUM_FIELDS (sizeof(ls_field_offset_map) / sizeof(ls_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * LinearSystemSetFieldByName
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetFieldByName(LS_args *args, const YAMLnode *node)
{
   for (size_t i = 0; i < LS_NUM_FIELDS; i++)
   {
      if (!strcmp(ls_field_offset_map[i].name, node->key))
      {
         ls_field_offset_map[i].setter(
            (void *)((char *)args + ls_field_offset_map[i].offset), node);
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
   static const char *keys[LS_NUM_FIELDS];

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
LinearSystemGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"online", 0}, {"ij", 1}, {"parcsr", 2}, {"mtx", 3}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "rhs_mode"))
   {
      static StrIntMap map[] = {
         {"zeros", 0}, {"ones", 1}, {"file", 2}, {"random", 3}, {"randsol", 4}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "init_guess_mode"))
   {
      static StrIntMap map[] = {
         {"zeros", 0}, {"ones", 1}, {"file", 2}, {"random", 3}, {"previous", 4}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "exec_policy"))
   {
      /* TODO: move to general? */
      static StrIntMap map[] = {{"host", 0}, {"device", 1}};
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
   strcpy(args->dirname, "");
   strcpy(args->matrix_filename, "");
   strcpy(args->matrix_basename, "");
   strcpy(args->precmat_filename, "");
   strcpy(args->precmat_basename, "");
   strcpy(args->rhs_filename, "");
   strcpy(args->rhs_basename, "");
   strcpy(args->x0_filename, "");
   strcpy(args->sol_filename, "");
   strcpy(args->dofmap_filename, "");
   strcpy(args->dofmap_basename, "");
   args->digits_suffix   = 5;
   args->init_suffix     = -1;
   args->last_suffix     = -1;
   args->init_guess_mode = 0;
   args->rhs_mode        = 0;
   args->type            = 1;
   args->precon_reuse    = 0;
   args->num_systems     = 1;
#ifdef HYPRE_USING_GPU
   args->exec_policy = 1;
#else
   args->exec_policy = 0;
#endif

   /* Eigenspectrum defaults */
   EigSpecSetDefaultArgs(&args->eigspec);
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetNumSystems
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetNumSystems(LS_args *args)
{
   args->num_systems = args->last_suffix - args->init_suffix + 1;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetArgsFromYAML(LS_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child, LinearSystemGetValidKeys, LinearSystemGetValidValues);

      YAML_NODE_SET_FIELD(child, args, LinearSystemSetFieldByName);
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

void
LinearSystemReadMatrix(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix *matrix_ptr)
{
   StatsTimerStart("matrix");

   char                 matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int                  ls_id                                = StatsGetLinearSystemID();
   int                  file_not_found                       = 0;
   void                *obj                                  = NULL;
   HYPRE_MemoryLocation memory_location =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   /* Destroy matrix if it already exists */
   if (*matrix_ptr)
   {
      HYPRE_IJMatrixDestroy(*matrix_ptr);
   }

   /* Set matrix filename */
   if (args->dirname[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d/%.*s",
               (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
               (int)args->init_suffix + ls_id, (int)strlen(args->matrix_filename),
               args->matrix_filename);
   }
   else if (args->matrix_filename[0] != '\0')
   {
      strcpy(matrix_filename, args->matrix_filename);
   }
   else if (args->matrix_basename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d",
               (int)strlen(args->matrix_basename), args->matrix_basename,
               (int)args->digits_suffix, (int)args->init_suffix + ls_id);
   }
   else
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename("");
      StatsTimerStop("matrix");
      return;
   }

   /* Read matrix */
   if (args->type == 1)
   {
      if (CheckBinaryDataExists(matrix_filename))
      {
         int nprocs = 0, nparts = 0;

         MPI_Comm_size(comm, &nprocs);
         nparts = CountNumberOfPartitions(matrix_filename);
         if (nparts >= nprocs)
         {
            IJMatrixReadMultipartBinary(matrix_filename, comm, (uint64_t)nparts,
                                        memory_location, matrix_ptr);
         }
         else
         {
            ErrorCodeSet(ERROR_FILE_NOT_FOUND);
            ErrorMsgAddInvalidFilename(matrix_filename);
            StatsTimerStop("matrix");
            return;
         }
      }
      else if (CheckASCIIDataExists(matrix_filename))
      {
         HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
      }
      else
      {
         file_not_found = 1;
      }
   }
   else if (args->type == 3)
   {
      HYPRE_IJMatrixReadMM(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
   }

   /* Check if hypre had problems reading the input file */
   if (HYPRE_GetError() || file_not_found)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(matrix_filename);
      StatsTimerStop("matrix");
      return;
   }

   /* Migrate the matrix? TODO: use IJMatrixMigrate */
   if (args->exec_policy)
   {
      HYPRE_IJMatrixGetObject(*matrix_ptr, &obj);
      HYPRE_ParCSRMatrix par_A = (HYPRE_ParCSRMatrix)obj;

      hypre_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_DEVICE);
   }

   StatsTimerStop("matrix");
}

/*-----------------------------------------------------------------------------
 * LinearSystemMatrixGetNumRows
 *-----------------------------------------------------------------------------*/

long long int
LinearSystemMatrixGetNumRows(HYPRE_IJMatrix matrix)
{
   HYPRE_ParCSRMatrix par_A = NULL;
   void              *obj   = NULL;
   HYPRE_BigInt       nrows = 0, ncols = 0;

   HYPRE_IJMatrixGetObject(matrix, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;

   HYPRE_ParCSRMatrixGetDims(par_A, &nrows, &ncols);

   return (long long int)nrows;
}

/*-----------------------------------------------------------------------------
 * LinearSystemMatrixGetNumNonzeros
 *-----------------------------------------------------------------------------*/

long long int
LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix matrix)
{
   HYPRE_ParCSRMatrix par_A = NULL;
   void              *obj   = NULL;

   HYPRE_IJMatrixGetObject(matrix, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;

   hypre_ParCSRMatrixSetDNumNonzeros(par_A);

   return (long long int)par_A->d_num_nonzeros;
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetRHS(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                   HYPRE_IJVector *refsol_ptr, HYPRE_IJVector *rhs_ptr)
{
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_IJVector       refsol = NULL;
   int                  ls_id  = StatsGetLinearSystemID();
   HYPRE_MemoryLocation memory_location =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   StatsTimerStart("rhs");

   /* Destroy vectors */
   if (*refsol_ptr)
   {
      HYPRE_IJVectorDestroy(*refsol_ptr);
   }
   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
   }

   /* Read right-hand-side vector */
   if (args->rhs_filename[0] == '\0' && args->rhs_basename[0] == '\0')
   {
      HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
      HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(*rhs_ptr, memory_location);

      /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
      void           *obj   = NULL;
      HYPRE_ParVector par_b = NULL;
      HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
      par_b = (HYPRE_ParVector)obj;

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
            break;

         case 4:
            /* Solution has random values */
            HYPRE_IJVectorCreate(comm, ilower, iupper, &refsol);
            HYPRE_IJVectorSetObjectType(refsol, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize_v2(refsol, memory_location);

            /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
            HYPRE_ParVector par_x = NULL;
            HYPRE_IJVectorGetObject(refsol, &obj);
            par_x = (HYPRE_ParVector)obj;
            HYPRE_ParVectorSetRandomValues(par_x, 2023);

            /* TODO (hypre): add IJMatrixMatvec interface */
            void              *obj_A = NULL;
            HYPRE_ParCSRMatrix par_A = NULL;
            HYPRE_IJMatrixGetObject(mat, &obj_A);
            par_A = (HYPRE_ParCSRMatrix)obj_A;
            HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b);

            // hypre_ParVectorPrintIJ(par_b, 0, "test");
            break;
      }
   }
   else
   {
      char rhs_filename[MAX_FILENAME_LENGTH] = {0};

      /* Set RHS filename */
      if (args->dirname[0] != '\0')
      {
         snprintf(rhs_filename, sizeof(rhs_filename), "%.*s_%0*d/%.*s",
                  (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
                  (int)args->init_suffix + ls_id, (int)strlen(args->rhs_filename),
                  args->rhs_filename);
      }
      else if (args->rhs_filename[0] != '\0')
      {
         strcpy(rhs_filename, args->rhs_filename);
      }
      else if (args->rhs_basename[0] != '\0')
      {
         snprintf(rhs_filename, sizeof(rhs_filename), "%.*s_%0*d",
                  (int)strlen(args->rhs_basename), args->rhs_basename,
                  (int)args->digits_suffix, (int)args->init_suffix + ls_id);
      }

      /* Read vector from file (Binary or ASCII) */
      if (CheckBinaryDataExists(rhs_filename))
      {
         int nparts = 0, nprocs = 0;

         MPI_Comm_size(comm, &nprocs);
         nparts = CountNumberOfPartitions(rhs_filename);
         if (nparts >= nprocs)
         {
            IJVectorReadMultipartBinary(rhs_filename, comm, nparts, memory_location,
                                        rhs_ptr);
         }
         else
         {
            HYPRE_IJVectorReadBinary(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
         }
      }
      else
      {
         HYPRE_IJVectorRead(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
      }

      /* Check if hypre had problems reading the input file */
      if (HYPRE_GetError())
      {
         ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         ErrorMsgAddInvalidFilename(rhs_filename);
      }

      /* Migrate the vector? TODO: use IJVectorMigrate */
      if (args->exec_policy)
      {
         HYPRE_ParVector par_rhs = NULL;
         void           *obj     = NULL;

         HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
         par_rhs = (HYPRE_ParVector)obj;

         hypre_ParVectorMigrate(par_rhs, HYPRE_MEMORY_DEVICE);
      }
   }

   /* Set reference solution vector */
   *refsol_ptr = refsol;

   StatsTimerStop("rhs");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetInitialGuess
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetInitialGuess(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                            HYPRE_IJVector rhs, HYPRE_IJVector *x0_ptr,
                            HYPRE_IJVector *x_ptr)
{
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memloc =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   /* Create solution vector
      TODO: implement HYPRE_IJVectorClone in hypre */
   if (!*x_ptr)
   {
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x_ptr);
      HYPRE_IJVectorSetObjectType(*x_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(*x_ptr, memloc);
   }

   /* Destroy initial solution vector */
   if (*x0_ptr)
   {
      HYPRE_IJVectorDestroy(*x0_ptr);
   }

   if (args->x0_filename[0] == '\0')
   {
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x0_ptr);
      HYPRE_IJVectorSetObjectType(*x0_ptr, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize_v2(*x0_ptr, memloc);

      /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
      void           *obj    = NULL;
      HYPRE_ParVector par_x0 = NULL, par_x = NULL;

      HYPRE_IJVectorGetObject(*x0_ptr, &obj);
      par_x0 = (HYPRE_ParVector)obj;

      switch (args->init_guess_mode)
      {
         case 0:
         default:
            /* Vector of zeros */
            break;

         case 1:
            /* Vector of ones */
            HYPRE_ParVectorSetConstantValues(par_x0, 1);
            break;

         case 3:
            /* Vector of random values */
            HYPRE_ParVectorSetRandomValues(par_x0, 2023);
            break;

         case 4:
            /* Use solution from previous linear solve */
            HYPRE_IJVectorGetObject(*x_ptr, &obj);
            par_x = (HYPRE_ParVector)obj;

            HYPRE_ParVectorCopy(par_x, par_x0);
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
         HYPRE_ParVector par_x0 = NULL;
         void           *obj    = NULL;

         HYPRE_IJVectorGetObject(*x0_ptr, &obj);
         par_x0 = (HYPRE_ParVector)obj;

         hypre_ParVectorMigrate(par_x0, HYPRE_MEMORY_DEVICE);
      }
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

void
LinearSystemResetInitialGuess(HYPRE_IJVector x0_ptr, HYPRE_IJVector x_ptr)
{
   HYPRE_ParVector par_x0 = NULL, par_x = NULL;
   void           *obj_x0 = NULL, *obj_x = NULL;

   StatsTimerStart("reset_x0");

   /* TODO: implement HYPRE_IJVectorCopy in hypre */
   HYPRE_IJVectorGetObject(x0_ptr, &obj_x0);
   HYPRE_IJVectorGetObject(x_ptr, &obj_x);
   par_x0 = (HYPRE_ParVector)obj_x0;
   par_x  = (HYPRE_ParVector)obj_x;

   HYPRE_ParVectorCopy(par_x0, par_x);

   StatsTimerStop("reset_x0");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetPrecMatrix(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                          HYPRE_IJMatrix *precmat_ptr)
{
   char matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int  ls_id                                = StatsGetLinearSystemID();

   /* Set matrix filename */
   if (args->dirname[0] != '\0' && args->precmat_filename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d/%.*s",
               (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
               (int)args->init_suffix + ls_id, (int)strlen(args->precmat_filename),
               args->precmat_filename);
   }
   else if (args->precmat_filename[0] != '\0')
   {
      strcpy(matrix_filename, args->precmat_filename);
   }
   else if (args->precmat_basename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d",
               (int)strlen(args->precmat_basename), args->precmat_basename,
               (int)args->digits_suffix, (int)args->init_suffix + ls_id);
   }

   if (matrix_filename[0] == '\0' || !strcmp(matrix_filename, args->matrix_filename))
   {
      *precmat_ptr = mat;
   }
   else
   {
      /* Destroy matrix */
      if (*precmat_ptr)
      {
         HYPRE_IJMatrixDestroy(*precmat_ptr);
      }

      HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, precmat_ptr);
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

void
LinearSystemReadDofmap(MPI_Comm comm, LS_args *args, IntArray **dofmap_ptr)
{
   int ls_id = StatsGetLinearSystemID();

   /* Destroy pre-existing dofmap */
   if (*dofmap_ptr)
   {
      IntArrayDestroy(dofmap_ptr);
   }

   if (args->dofmap_filename[0] == '\0' && args->dofmap_basename[0] == '\0')
   {
      *dofmap_ptr = IntArrayCreate(0);
   }
   else
   {
      char dofmap_filename[MAX_FILENAME_LENGTH] = {0};

      /* Set dofmap filename */
      if (args->dirname[0] != '\0')
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%.*s_%0*d/%.*s",
                  (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
                  (int)args->init_suffix + ls_id, (int)strlen(args->dofmap_filename),
                  args->dofmap_filename);
      }
      else if (args->dofmap_filename[0] != '\0')
      {
         strcpy(dofmap_filename, args->dofmap_filename);
      }
      else
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%.*s_%0*d",
                  (int)strlen(args->dofmap_basename), args->dofmap_basename,
                  (int)args->digits_suffix, (int)args->init_suffix + ls_id);
      }

      /* Destroy previous dofmap array */
      IntArrayDestroy(dofmap_ptr);

      StatsTimerStart("dofmap");
      IntArrayParRead(comm, dofmap_filename, dofmap_ptr);
      StatsTimerStop("dofmap");
   }

   /* TODO: Print how many dofs types we have (min, max, avg, sum) accross ranks
    */
}

/*-----------------------------------------------------------------------------
 * LinearSystemGetSolution
 *-----------------------------------------------------------------------------*/

void
LinearSystemGetSolutionValues(HYPRE_IJVector sol, HYPRE_Complex **data_ptr)
{
   HYPRE_ParVector par_sol = NULL;
   hypre_Vector   *seq_sol = NULL;
   void           *obj     = NULL;

   HYPRE_IJVectorGetObject(sol, &obj);
   par_sol = (HYPRE_ParVector)obj;
   seq_sol = hypre_ParVectorLocalVector(par_sol);

   *data_ptr = hypre_VectorData(seq_sol);
}

/*-----------------------------------------------------------------------------
 * LinearSystemGetRHS
 *-----------------------------------------------------------------------------*/

void
LinearSystemGetRHSValues(HYPRE_IJVector rhs, HYPRE_Complex **data_ptr)
{
   HYPRE_ParVector par_rhs = NULL;
   hypre_Vector   *seq_rhs = NULL;
   void           *obj     = NULL;

   HYPRE_IJVectorGetObject(rhs, &obj);
   par_rhs = (HYPRE_ParVector)obj;
   seq_rhs = hypre_ParVectorLocalVector(par_rhs);

   *data_ptr = hypre_VectorData(seq_rhs);
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeVectorNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeVectorNorm(HYPRE_IJVector vec, HYPRE_Complex *norm)
{
   HYPRE_IJVectorInnerProd(vec, vec, norm);
   *norm = sqrt(*norm);
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeErrorNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeErrorNorm(HYPRE_IJVector vec_xref, HYPRE_IJVector vec_x,
                             HYPRE_Complex *e_norm)
{
   HYPRE_ParVector par_xref = NULL;
   HYPRE_ParVector par_x    = NULL;
   HYPRE_ParVector par_e    = NULL;
   HYPRE_IJVector  vec_e    = NULL;
   void           *obj_xref = NULL, *obj_x = NULL, *obj_e = NULL;

   HYPRE_BigInt jlower = 0, jupper = 0;

   HYPRE_Complex one     = 1.0;
   HYPRE_Complex neg_one = -1.0;

   HYPRE_IJVectorGetObject(vec_xref, &obj_xref);
   HYPRE_IJVectorGetObject(vec_x, &obj_x);

   par_xref = (HYPRE_ParVector)obj_xref;
   par_x    = (HYPRE_ParVector)obj_x;

   HYPRE_IJVectorGetLocalRange(vec_x, &jlower, &jupper);
   HYPRE_IJVectorCreate(hypre_IJVectorComm(vec_x), jlower, jupper, &vec_e);
   HYPRE_IJVectorSetObjectType(vec_e, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec_e, hypre_IJVectorMemoryLocation(vec_x));
   HYPRE_IJVectorGetObject(vec_e, &obj_e);
   par_e = (HYPRE_ParVector)obj_e;

   /* Compute error */
   hypre_ParVectorAxpyz(one, par_x, neg_one, par_xref, par_e);

   /* Compute error norm */
   HYPRE_ParVectorInnerProd(par_e, par_e, e_norm);
   *e_norm = sqrt(*e_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_e);
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeResidualNorm(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                                HYPRE_IJVector vec_x, HYPRE_Complex *res_norm)
{
   HYPRE_ParCSRMatrix par_A = NULL;
   HYPRE_ParVector    par_b = NULL;
   HYPRE_ParVector    par_x = NULL;
   HYPRE_ParVector    par_r = NULL;
   HYPRE_IJVector     vec_r = NULL;
   void              *obj_A = NULL, *obj_b = NULL, *obj_x = NULL, *obj_r = NULL;

   HYPRE_BigInt jlower = 0, jupper = 0;

   HYPRE_Complex one     = 1.0;
   HYPRE_Complex neg_one = -1.0;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   HYPRE_IJVectorGetObject(vec_x, &obj_x);

   par_A = (HYPRE_ParCSRMatrix)obj_A;
   par_b = (HYPRE_ParVector)obj_b;
   par_x = (HYPRE_ParVector)obj_x;

   /* TODO: implement IJVectorClone */
   HYPRE_IJVectorGetLocalRange(vec_b, &jlower, &jupper);
   HYPRE_IJVectorCreate(hypre_IJVectorComm(vec_b), jlower, jupper, &vec_r);
   HYPRE_IJVectorSetObjectType(vec_r, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize_v2(vec_r, hypre_IJVectorMemoryLocation(vec_b));
   HYPRE_IJVectorGetObject(vec_r, &obj_r);
   par_r = (HYPRE_ParVector)obj_r;
   HYPRE_ParVectorCopy(par_b, par_r);

   /* Compute residual */
   HYPRE_ParCSRMatrixMatvec(neg_one, par_A, par_x, one, par_r);

   /* Compute residual norm */
   HYPRE_ParVectorInnerProd(par_r, par_r, res_norm);
   *res_norm = sqrt(*res_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_r);
}

/*-----------------------------------------------------------------------------
 * Print matrix/vector/dofmap with series directory logic
 *---------------------------------------------------------------------------*/

void
LinearSystemPrintData(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat_A,
                      HYPRE_IJVector vec_b, const IntArray *dofmap)
{
   const char *A_base =
      (args && args->matrix_basename[0] != '\0') ? args->matrix_basename : "IJ.out.A";
   const char *b_base =
      (args && args->rhs_basename[0] != '\0') ? args->rhs_basename : "IJ.out.b";
   const char *d_base =
      (args && args->dofmap_basename[0] != '\0') ? args->dofmap_basename : "dofmap";

   char A_name[MAX_FILENAME_LENGTH];
   char b_name[MAX_FILENAME_LENGTH];
   char d_name[MAX_FILENAME_LENGTH];

   {
      int max_base = (int)sizeof(A_name) - 1 - 4;
      if (max_base < 0) max_base = 0;
      snprintf(A_name, sizeof(A_name), "%.*s.out", max_base, A_base);
   }
   {
      int max_base = (int)sizeof(b_name) - 1 - 4;
      if (max_base < 0) max_base = 0;
      snprintf(b_name, sizeof(b_name), "%.*s.out", max_base, b_base);
   }
   {
      int max_base = (int)sizeof(d_name) - 1 - 4;
      if (max_base < 0) max_base = 0;
      snprintf(d_name, sizeof(d_name), "%.*s.out", max_base, d_base);
   }

   int use_series_dir = 1;
   if (args)
   {
      const int has_mat = args->matrix_basename[0] != '\0';
      const int has_rhs = args->rhs_basename[0] != '\0';
      const int has_dmf = args->dofmap_basename[0] != '\0';
      use_series_dir    = !(has_mat || has_rhs || has_dmf);
   }

   char A_path[2 * MAX_FILENAME_LENGTH];
   char b_path[2 * MAX_FILENAME_LENGTH];
   char d_path[2 * MAX_FILENAME_LENGTH];

   if (use_series_dir)
   {
      const char *root = "hypre-data";
      struct stat st;
      if (stat(root, &st) != 0)
      {
         (void)mkdir(root, 0775);
      }

      int  max_idx = -1;
      DIR *dir     = opendir(root);
      if (dir)
      {
         struct dirent *ent;
         while ((ent = readdir(dir)) != NULL)
         {
            if (ent->d_name[0] == 'l' && ent->d_name[1] == 's' && ent->d_name[2] == '_')
            {
               int idx = -1;
               if (sscanf(ent->d_name + 3, "%d", &idx) == 1)
               {
                  if (idx > max_idx) max_idx = idx;
               }
            }
         }
         closedir(dir);
      }
      int  next_idx = max_idx + 1;
      char run_dir[256];
      snprintf(run_dir, sizeof(run_dir), "%s/ls_%05d", root, next_idx);
      if (stat(run_dir, &st) != 0)
      {
         (void)mkdir(run_dir, 0775);
      }

      snprintf(A_path, sizeof(A_path), "%s/%s", run_dir, A_name);
      snprintf(b_path, sizeof(b_path), "%s/%s", run_dir, b_name);
      snprintf(d_path, sizeof(d_path), "%s/%s", run_dir, d_name);
   }
   else
   {
      snprintf(A_path, sizeof(A_path), "%s", A_name);
      snprintf(b_path, sizeof(b_path), "%s", b_name);
      snprintf(d_path, sizeof(d_path), "%s", d_name);
   }

   if (mat_A)
   {
      HYPRE_IJMatrixPrint(mat_A, A_path);
   }
   else
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("Matrix not set; skipping matrix print.");
   }

   if (vec_b)
   {
      HYPRE_IJVectorPrint(vec_b, b_path);
   }
   else
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAdd("RHS not set; skipping vector print.");
   }

   if (dofmap && dofmap->data)
   {
      IntArrayWriteAsciiByRank(comm, dofmap, d_path);
   }
}
