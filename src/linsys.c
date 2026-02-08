/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

/* Add internal hypre headers */
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

/* Undefine autotools package macros from hypre */
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "linsys.h"

static void
HYPREDRV_IJVectorInitialize(HYPRE_IJVector vec, HYPRE_MemoryLocation memory_location)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 22000
   HYPRE_IJVectorInitialize_v2(vec, memory_location);
#else
   (void)memory_location;
   HYPRE_IJVectorInitialize(vec);
#endif
}

#define HYPREDRV_HAVE_MEMORY_APIS (HYPREDRV_HYPRE_RELEASE_NUMBER >= 22000)

/* TODO: implement IJVectorClone/Copy and IJVectorMigrate/IJMatrix in hypre*/

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, dirname, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_basename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, xref_filename, FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, xref_basename, FieldTypeStringSet),
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
   args->dirname[0]          = '\0';
   args->matrix_filename[0]  = '\0';
   args->matrix_basename[0]  = '\0';
   args->precmat_filename[0] = '\0';
   args->precmat_basename[0] = '\0';
   args->rhs_filename[0]     = '\0';
   args->rhs_basename[0]     = '\0';
   args->x0_filename[0]      = '\0';
   args->xref_filename[0]    = '\0';
   args->xref_basename[0]    = '\0';
   args->sol_filename[0]     = '\0';
   args->dofmap_filename[0]  = '\0';
   args->dofmap_basename[0]  = '\0';
   args->digits_suffix       = 5;
   args->init_suffix         = -1;
   args->last_suffix         = -1;
   args->init_guess_mode     = 0;
   args->rhs_mode            = 2;
   args->type                = 1;
   args->precon_reuse        = 0;
   args->num_systems         = 1;
#ifdef HYPRE_USING_GPU
   args->exec_policy = 1;
#else
   args->exec_policy = 0;
#endif

   /* Eigenspectrum defaults */
   EigSpecSetDefaultArgs(&args->eigspec);
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetNearNullSpace
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetNearNullSpace(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                             int num_entries, int num_components,
                             const HYPRE_Complex *values, HYPRE_IJVector *vec_nn_ptr)
{
   HYPRE_BigInt ilower = 0, iupper = 0, jlower = 0, jupper = 0;

   /* Destroy previous NN vector if present */
   if (*vec_nn_ptr)
   {
      HYPRE_IJVectorDestroy(*vec_nn_ptr);
      *vec_nn_ptr = NULL;
   }

   /* Get local vector range from the matrix columns */
   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   HYPRE_BigInt loc_expected = jupper - jlower + 1;

   /* Sanity: check if the number of entries matches the expected local size */
   if (loc_expected != num_entries)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Number of entries (%d) does not match the expected local size (%d)",
                  num_entries, loc_expected);
      return;
   }

   /* Create a ParCSR IJVector with host memory (we'll migrate later if needed) */
   HYPRE_IJVectorCreate(comm, jlower, jupper, vec_nn_ptr);
   HYPRE_IJVectorSetObjectType(*vec_nn_ptr, HYPRE_PARCSR);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_IJVectorSetNumComponents(*vec_nn_ptr, num_components);
#endif
   HYPREDRV_IJVectorInitialize(*vec_nn_ptr, HYPRE_MEMORY_HOST);

   HYPRE_BigInt  *indices = NULL;
   HYPRE_Complex *zeros   = NULL;
   if (num_entries > 0)
   {
      indices = (HYPRE_BigInt *)malloc((size_t)num_entries * sizeof(HYPRE_BigInt));
      if (values == NULL)
      {
         zeros = (HYPRE_Complex *)calloc((size_t)num_entries, sizeof(HYPRE_Complex));
      }
      for (int i = 0; i < num_entries; i++)
      {
         indices[i] = jlower + (HYPRE_BigInt)i;
      }
   }

   /* Set values for each component block contiguously */
   for (HYPRE_Int c = 0; c < num_components; c++)
   {
      const HYPRE_Complex *vals_c =
         values ? (values + ((size_t)c * (size_t)num_entries)) : NULL;
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      HYPRE_IJVectorSetComponent(*vec_nn_ptr, c);
#endif
      HYPRE_IJVectorSetValues(*vec_nn_ptr, num_entries, indices, vals_c ? vals_c : zeros);
   }

   HYPRE_IJVectorAssemble(*vec_nn_ptr);

   free(indices);
   free(zeros);

   /* Migrate to device memory if requested */
   if (args && args->exec_policy)
   {
#if HYPRE_CHECK_MIN_VERSION(23300, 0)
      HYPRE_IJVectorMigrate(*vec_nn_ptr, HYPRE_MEMORY_DEVICE);
#endif
   }
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
   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "matrix");

   char                 matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int                  ls_id          = StatsGetLinearSystemID() + 1;
   int                  file_not_found = 0;
   void                *obj            = NULL;
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
      snprintf(matrix_filename, sizeof(matrix_filename), "%s", args->matrix_filename);
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
      StatsAnnotate(HYPREDRV_ANNOTATE_END, "matrix");
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
            StatsAnnotate(HYPREDRV_ANNOTATE_END, "matrix");
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
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      HYPRE_IJMatrixReadMM(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
#else
      HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
#endif
   }

   /* Check if hypre had problems reading the input file */
   if (HYPRE_GetError() || file_not_found)
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(matrix_filename);
      StatsAnnotate(HYPREDRV_ANNOTATE_END, "matrix");
      return;
   }

   /* Migrate the matrix? TODO: use IJMatrixMigrate */
   if (args->exec_policy)
   {
      HYPRE_IJMatrixGetObject(*matrix_ptr, &obj);
      HYPRE_ParCSRMatrix par_A = (HYPRE_ParCSRMatrix)obj;

#if HYPREDRV_HAVE_MEMORY_APIS
      hypre_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_DEVICE);
#endif
   }

   StatsAnnotate(HYPREDRV_ANNOTATE_END, "matrix");
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

   if (!matrix)
   {
      return 0;
   }

   HYPRE_IJMatrixGetObject(matrix, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;

   if (!par_A)
   {
      return 0;
   }

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

   if (!matrix)
   {
      return 0;
   }

   HYPRE_IJMatrixGetObject(matrix, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;

   if (!par_A)
   {
      return 0;
   }

   hypre_ParCSRMatrixSetDNumNonzeros(par_A);

   return (long long int)par_A->d_num_nonzeros;
}

#if defined(HYPRE_BIG_INT)
#define HYPRE_BIG_INT_SSCANF "%lld"
#else
#define HYPRE_BIG_INT_SSCANF "%d"
#endif

void
LinearSystemSetRHS(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                   HYPRE_IJVector *xref_ptr, HYPRE_IJVector *rhs_ptr)
{
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_IJVector       xref  = NULL;
   int                  ls_id = StatsGetLinearSystemID() + 1;
   HYPRE_MemoryLocation memory_location =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "rhs");

   /* Destroy vectors */
   if (*xref_ptr)
   {
      HYPRE_IJVectorDestroy(*xref_ptr);
   }
   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
   }

   /* Read right-hand-side vector.
    * rhs_mode is authoritative: only file mode reads rhs_filename/rhs_basename. */
   if (args->rhs_mode != 2)
   {
      HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
      HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
      HYPREDRV_IJVectorInitialize(*rhs_ptr, memory_location);

      /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
      void           *obj   = NULL;
      HYPRE_ParVector par_b = NULL;
      HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
      par_b = (HYPRE_ParVector)obj;

      switch (args->rhs_mode)
      {
         case 0:
            /* Vector of zeros */
            HYPRE_ParVectorSetConstantValues(par_b, 0);
            break;

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
            HYPRE_IJVectorCreate(comm, ilower, iupper, &xref);
            HYPRE_IJVectorSetObjectType(xref, HYPRE_PARCSR);
            HYPREDRV_IJVectorInitialize(xref, memory_location);

            /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
            HYPRE_ParVector par_x = NULL;
            HYPRE_IJVectorGetObject(xref, &obj);
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
         snprintf(rhs_filename, sizeof(rhs_filename), "%s", args->rhs_filename);
      }
      else if (args->rhs_basename[0] != '\0')
      {
         snprintf(rhs_filename, sizeof(rhs_filename), "%.*s_%0*d",
                  (int)strlen(args->rhs_basename), args->rhs_basename,
                  (int)args->digits_suffix, (int)args->init_suffix + ls_id);
      }

      if (args->type == 1 && CheckBinaryDataExists(rhs_filename))
      {
         int nparts = 0, nprocs = 0;

         MPI_Comm_size(comm, &nprocs);
         nparts = CountNumberOfPartitions(rhs_filename);
         if (nparts >= nprocs)
         {
            IJVectorReadMultipartBinary(rhs_filename, comm, (uint64_t)nparts,
                                        memory_location, rhs_ptr);
         }
         else
         {
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
            HYPRE_IJVectorReadBinary(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
#else
            HYPRE_IJVectorRead(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
#endif
         }
      }
      else if (args->type == 3)
      {
         int                myid      = 0;
         int                num_procs = 0;
         FILE              *file      = NULL;
         char               line[1024];
         HYPRE_BigInt       M = 0;
         HYPRE_BigInt       N;
         HYPRE_Complex     *all_values      = NULL;
         HYPRE_Complex     *local_values    = NULL;
         HYPRE_BigInt       global_num_rows = 0, global_num_cols = 0;
         HYPRE_ParCSRMatrix par_A  = NULL;
         void              *obj    = NULL;
         int               *counts = NULL;
         int               *displs = NULL;

         MPI_Comm_rank(comm, &myid);
         MPI_Comm_size(comm, &num_procs);

         /* Get matrix dimensions to check against vector dimensions */
         HYPRE_IJMatrixGetObject(mat, &obj);
         par_A = (HYPRE_ParCSRMatrix)obj;
         HYPRE_ParCSRMatrixGetDims(par_A, &global_num_rows, &global_num_cols);

         if (myid == 0)
         {
            file = fopen(rhs_filename, "r");
            if (file == NULL)
            {
               ErrorCodeSet(ERROR_FILE_NOT_FOUND);
               ErrorMsgAdd("Cannot open file %s", rhs_filename);
               M = -1; /* Signal error */
            }
            else
            {
               /* Skip comments */
               do
               {
                  if (fgets(line, sizeof(line), file) == NULL)
                  {
                     ErrorCodeSet(ERROR_FILE_NOT_FOUND);
                     ErrorMsgAdd("Unexpected end of file or error reading %s",
                                 rhs_filename);
                     M = -1; /* Signal error */
                     break;
                  }
               } while (line[0] == '%');

               /* Read dimensions with type-safe temps to satisfy both int and long long
                * builds */
               if (M != -1)
               {
#ifdef HYPRE_BIG_INT
                  long long   tmpM     = strtoll(line, NULL, 10);
                  const char *line_ptr = strchr(line, ' ');
                  long long   tmpN =
                     (line_ptr != NULL) ? strtoll(line_ptr + 1, NULL, 10) : 0;
                  int read_ok = (tmpM != 0 && tmpN != 0);
#else
                  int         tmpM     = (int)strtol(line, NULL, 10);
                  const char *line_ptr = strchr(line, ' ');
                  int tmpN = (line_ptr != NULL) ? (int)strtol(line_ptr + 1, NULL, 10) : 0;
                  int read_ok = (tmpM != 0 && tmpN != 0);
#endif

                  if (read_ok)
                  {
                     M = (HYPRE_BigInt)tmpM;
                     N = (HYPRE_BigInt)tmpN;
                  }
                  else
                  {
                     ErrorCodeSet(ERROR_FILE_NOT_FOUND);
                     ErrorMsgAdd("Failed to read vector dimensions from %s",
                                 rhs_filename);
                     M = -1; /* Signal error */
                     N = 0;
                  }

                  if (N != 1)
                  {
                     ErrorCodeSet(ERROR_FILE_NOT_FOUND);
                     ErrorMsgAdd("File %s is not a vector (N=" HYPRE_BIG_INT_SSCANF ")",
                                 rhs_filename, N);
                     M = -1; /* Signal error */
                  }
                  else if (M != global_num_rows)
                  {
                     ErrorCodeSet(ERROR_UNKNOWN);
                     ErrorMsgAdd("RHS vector size " HYPRE_BIG_INT_SSCANF
                                 " does not match matrix size " HYPRE_BIG_INT_SSCANF,
                                 M, global_num_rows);
                     M = -1; /* Signal error */
                  }
                  else
                  {
                     all_values = hypre_TAlloc(HYPRE_Complex, M, HYPRE_MEMORY_HOST);
                     for (HYPRE_BigInt i = 0; i < M; i++)
                     {
                        char *endptr = NULL;
                        if (fgets(line, sizeof(line), file) == NULL)
                        {
                           ErrorCodeSet(ERROR_UNKNOWN);
                           ErrorMsgAdd(
                              "Error reading value for index " HYPRE_BIG_INT_SSCANF
                              " from %s",
                              i, rhs_filename);
                           M = -1;
                           break;
                        }
                        double tmp_val = strtod(line, &endptr);
                        if (endptr == line || (*endptr != '\0' && *endptr != '\n'))
                        {
                           ErrorCodeSet(ERROR_UNKNOWN);
                           ErrorMsgAdd(
                              "Error converting value for index " HYPRE_BIG_INT_SSCANF
                              " from %s",
                              i, rhs_filename);
                           M = -1;
                           break;
                        }
                        all_values[i] = (HYPRE_Complex)tmp_val;
                     }
                  }
               }
               fclose(file);
            }
         }

         /* Broadcast M to all processes to check for errors */
         MPI_Bcast(&M, 1, HYPRE_MPI_BIG_INT, 0, comm);

         if (M == -1)
         {
            /* Error occurred on rank 0, abort */
            if (myid == 0 && all_values)
            {
               hypre_TFree(all_values, HYPRE_MEMORY_HOST);
            }
            StatsAnnotate(HYPREDRV_ANNOTATE_END, "rhs");
            return;
         }

         /* Create the vector object */
         HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
         HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
         HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
         HYPREDRV_IJVectorInitialize(*rhs_ptr, memory_location);

         HYPRE_BigInt local_size    = iupper - ilower + 1;
         int          my_local_size = local_size;

         /* Gather local sizes to calculate displacements for Scatterv */
         if (myid == 0)
         {
            counts = hypre_TAlloc(int, num_procs, HYPRE_MEMORY_HOST);
            displs = hypre_TAlloc(int, num_procs, HYPRE_MEMORY_HOST);
         }
         MPI_Gather(&my_local_size, 1, MPI_INT, counts, 1, MPI_INT, 0, comm);

         if (myid == 0)
         {
            displs[0] = 0;
            for (int i = 1; i < num_procs; i++)
            {
               displs[i] = displs[i - 1] + counts[i - 1];
            }
         }

         local_values = hypre_TAlloc(HYPRE_Complex, local_size, HYPRE_MEMORY_HOST);

         /* Scatter values from root to all processes */
         MPI_Scatterv(all_values, counts, displs, MPI_DOUBLE, local_values, local_size,
                      MPI_DOUBLE, 0, comm);

         /* Set values in the IJVector */
         HYPRE_IJVectorSetValues(*rhs_ptr, local_size, NULL, local_values);
         HYPRE_IJVectorAssemble(*rhs_ptr);

         /* Clean up */
         hypre_TFree(local_values, HYPRE_MEMORY_HOST);
         if (myid == 0)
         {
            hypre_TFree(all_values, HYPRE_MEMORY_HOST);
            hypre_TFree(counts, HYPRE_MEMORY_HOST);
            hypre_TFree(displs, HYPRE_MEMORY_HOST);
         }
      }
      else
      {
         /* Read vector from file (Binary or ASCII) */
         if (CheckBinaryDataExists(rhs_filename))
         {
            int nparts = 0, nprocs = 0;

            MPI_Comm_size(comm, &nprocs);
            nparts = CountNumberOfPartitions(rhs_filename);
            if (nparts >= nprocs)
            {
               IJVectorReadMultipartBinary(rhs_filename, comm, (uint64_t)nparts,
                                           memory_location, rhs_ptr);
            }
            else
            {
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
               HYPRE_IJVectorReadBinary(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
#else
               HYPRE_IJVectorRead(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
#endif
            }
         }
         else
         {
            HYPRE_IJVectorRead(rhs_filename, comm, HYPRE_PARCSR, rhs_ptr);
         }
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

#if HYPREDRV_HAVE_MEMORY_APIS
         hypre_ParVectorMigrate(par_rhs, HYPRE_MEMORY_DEVICE);
#endif
      }
   }

   /* Set reference solution vector */
   *xref_ptr = xref;

   StatsAnnotate(HYPREDRV_ANNOTATE_END, "rhs");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetInitialGuess
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetInitialGuess(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                            HYPRE_IJVector rhs, HYPRE_IJVector *x0_ptr,
                            HYPRE_IJVector *x_ptr)
{
   (void)mat;
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memloc =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   /* Destroy solution vector if it already exists */
   if (*x_ptr)
   {
      HYPRE_IJVectorDestroy(*x_ptr);
      *x_ptr = NULL;
   }

   /* Destroy initial solution vector */
   if (*x0_ptr)
   {
      HYPRE_IJVectorDestroy(*x0_ptr);
      *x0_ptr = NULL;
   }

   /* Destroy initial solution vector */
   if (*x_ptr) HYPRE_IJVectorDestroy(*x_ptr);

   /* Create solution vector
      TODO: implement HYPRE_IJVectorClone in hypre */
   HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, jlower, jupper, x_ptr);
   HYPRE_IJVectorSetObjectType(*x_ptr, HYPRE_PARCSR);
   HYPREDRV_IJVectorInitialize(*x_ptr, memloc);

   if (args->x0_filename[0] == '\0')
   {
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x0_ptr);
      HYPRE_IJVectorSetObjectType(*x0_ptr, HYPRE_PARCSR);
      HYPREDRV_IJVectorInitialize(*x0_ptr, memloc);

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
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
         HYPRE_IJVectorReadBinary(args->x0_filename, comm, HYPRE_PARCSR, x0_ptr);
#else
         HYPRE_IJVectorRead(args->x0_filename, comm, HYPRE_PARCSR, x0_ptr);
#endif
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

#if HYPREDRV_HAVE_MEMORY_APIS
         hypre_ParVectorMigrate(par_x0, HYPRE_MEMORY_DEVICE);
#endif
      }
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetReferenceSolution
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetReferenceSolution(MPI_Comm comm, LS_args *args, HYPRE_IJVector *xref_ptr)
{
   char                 xref_filename[MAX_FILENAME_LENGTH] = {0};
   int                  ls_id                              = StatsGetLinearSystemID() + 1;
   HYPRE_MemoryLocation memory_location =
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

   /* Keep the existing reference solution (e.g., rhs_mode = randsol) unless a file is
    * explicitly requested. */
   if (args->xref_filename[0] == '\0' && args->xref_basename[0] == '\0')
   {
      return;
   }

   if (*xref_ptr)
   {
      HYPRE_IJVectorDestroy(*xref_ptr);
      *xref_ptr = NULL;
   }

   /* Set reference solution filename */
   if (args->dirname[0] != '\0')
   {
      snprintf(xref_filename, sizeof(xref_filename), "%.*s_%0*d/%.*s",
               (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
               (int)args->init_suffix + ls_id, (int)strlen(args->xref_filename),
               args->xref_filename);
   }
   else if (args->xref_filename[0] != '\0')
   {
      strcpy(xref_filename, args->xref_filename);
   }
   else if (args->xref_basename[0] != '\0')
   {
      snprintf(xref_filename, sizeof(xref_filename), "%.*s_%0*d",
               (int)strlen(args->xref_basename), args->xref_basename,
               (int)args->digits_suffix, (int)args->init_suffix + ls_id);
   }

   /* Read vector from file (Binary or ASCII) */
   if (CheckBinaryDataExists(xref_filename))
   {
      int nprocs = 0;
      int nparts = 0;
      MPI_Comm_size(comm, &nprocs);
      nparts = CountNumberOfPartitions(xref_filename);
      if (nparts >= nprocs)
      {
         IJVectorReadMultipartBinary(xref_filename, comm, (uint64_t)nparts,
                                     memory_location, xref_ptr);
      }
      else
      {
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
         HYPRE_IJVectorReadBinary(xref_filename, comm, HYPRE_PARCSR, xref_ptr);
#else
         HYPRE_IJVectorRead(xref_filename, comm, HYPRE_PARCSR, xref_ptr);
#endif
      }
   }
   else
   {
      HYPRE_IJVectorRead(xref_filename, comm, HYPRE_PARCSR, xref_ptr);
   }

   /* Check if hypre had problems reading the input file */
   if (HYPRE_GetError())
   {
      ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      ErrorMsgAddInvalidFilename(xref_filename);
      *xref_ptr = NULL;
   }
   else
   {
      /* Migrate the vector if needed */
      if (args->exec_policy && *xref_ptr)
      {
         HYPRE_ParVector par_xref;
         void           *obj;

         HYPRE_IJVectorGetObject(*xref_ptr, &obj);
         par_xref = (HYPRE_ParVector)obj;

#if HYPREDRV_HAVE_MEMORY_APIS
         hypre_ParVectorMigrate(par_xref, HYPRE_MEMORY_DEVICE);
#endif
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

   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "reset_x0");

   if (!x0_ptr || !x_ptr)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      StatsAnnotate(HYPREDRV_ANNOTATE_END, "reset_x0");
      return;
   }

   /* TODO: implement HYPRE_IJVectorCopy in hypre */
   HYPRE_IJVectorGetObject(x0_ptr, &obj_x0);
   HYPRE_IJVectorGetObject(x_ptr, &obj_x);
   par_x0 = (HYPRE_ParVector)obj_x0;
   par_x  = (HYPRE_ParVector)obj_x;

   HYPRE_ParVectorCopy(par_x0, par_x);

   StatsAnnotate(HYPREDRV_ANNOTATE_END, "reset_x0");
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetVectorTags
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetVectorTags(HYPRE_IJVector vec, IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (!vec || !dofmap || !dofmap->data || dofmap->size == 0)
   {
      return;
   }

   HYPRE_Int num_tags = 1;
   if (dofmap->g_unique_data && dofmap->g_unique_size > 0)
   {
      int max_tag = dofmap->g_unique_data[dofmap->g_unique_size - 1];
      if (max_tag >= 0)
      {
         num_tags = (HYPRE_Int)max_tag + 1;
      }
   }
   else if (dofmap->unique_data && dofmap->unique_size > 0)
   {
      int max_tag = dofmap->unique_data[dofmap->unique_size - 1];
      if (max_tag >= 0)
      {
         num_tags = (HYPRE_Int)max_tag + 1;
      }
   }

   HYPRE_IJVectorSetTags(vec, 0, num_tags, dofmap->data);
#else
   (void)vec;
   (void)dofmap;
#endif
}

/*-----------------------------------------------------------------------------
 * LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

void
LinearSystemSetPrecMatrix(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                          HYPRE_IJMatrix *precmat_ptr)
{
   char matrix_filename[MAX_FILENAME_LENGTH] = {0};

   /* Set matrix filename */
   if (args->dirname[0] != '\0' && args->precmat_filename[0] != '\0')
   {
      int ls_id = StatsGetLinearSystemID() + 1;
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d/%.*s",
               (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
               (int)args->init_suffix + ls_id, (int)strlen(args->precmat_filename),
               args->precmat_filename);
   }
   else if (args->precmat_filename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%s", args->precmat_filename);
   }
   else if (args->precmat_basename[0] != '\0')
   {
      int ls_id = StatsGetLinearSystemID() + 1;
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
   int ls_id = StatsGetLinearSystemID() + 1;

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
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%s", args->dofmap_filename);
      }
      else
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%.*s_%0*d",
                  (int)strlen(args->dofmap_basename), args->dofmap_basename,
                  (int)args->digits_suffix, (int)args->init_suffix + ls_id);
      }

      /* Destroy previous dofmap array */
      IntArrayDestroy(dofmap_ptr);

      StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, "dofmap");
      IntArrayParRead(comm, dofmap_filename, dofmap_ptr);
      StatsAnnotate(HYPREDRV_ANNOTATE_END, "dofmap");
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
 * TODO: leverage internal hypre APIs for device exec
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeVectorNorm(HYPRE_IJVector vec, const char *norm_type, double *norm)
{
   HYPRE_ParVector      par_vec = NULL;
   const hypre_Vector  *seq_vec = NULL;
   void                *obj     = NULL;
   const HYPRE_Complex *data    = NULL;
   HYPRE_Int            size    = 0;

   if (!vec || !norm)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return;
   }

   HYPRE_IJVectorGetObject(vec, &obj);
   if (!obj)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   par_vec = (HYPRE_ParVector)obj;

   seq_vec = hypre_ParVectorLocalVector(par_vec);
   if (!seq_vec)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   data = hypre_VectorData(seq_vec);
   if (!data)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   size = hypre_VectorSize(seq_vec);
   if (size < 0)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   double   local_norm  = 0.0;
   double   global_norm = 0.0;
   MPI_Comm comm        = hypre_ParVectorComm(par_vec);

   if (!strcmp(norm_type, "L1") || !strcmp(norm_type, "l1"))
   {
      /* L1 norm: sum of absolute values */
      for (HYPRE_Int i = 0; i < size; i++)
      {
         local_norm += fabs((double)data[i]);
      }
      MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
      *norm = global_norm;
   }
   else if (!strcmp(norm_type, "L2") || !strcmp(norm_type, "l2"))
   {
      global_norm = (double)hypre_ParVectorInnerProd(par_vec, par_vec);
      *norm       = sqrt(global_norm);
   }
   else if (!strcmp(norm_type, "inf") || !strcmp(norm_type, "Linf") ||
            !strcmp(norm_type, "linf"))
   {
      /* Linf norm: maximum absolute value */
      for (HYPRE_Int i = 0; i < size; i++)
      {
         double val = fabs((double)data[i]);
         if (val > local_norm) local_norm = val;
      }
      MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, comm);
      *norm = global_norm;
   }
   else
   {
      *norm = -1.0; /* Invalid norm type */
   }
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeErrorNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeErrorNorm(HYPRE_IJVector vec_xref, HYPRE_IJVector vec_x,
                             const char *norm_type, double *e_norm)
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
#if HYPREDRV_HAVE_MEMORY_APIS
   HYPREDRV_IJVectorInitialize(vec_e, hypre_IJVectorMemoryLocation(vec_x));
#else
   HYPREDRV_IJVectorInitialize(vec_e, HYPRE_MEMORY_HOST);
#endif
   HYPRE_IJVectorGetObject(vec_e, &obj_e);
   par_e = (HYPRE_ParVector)obj_e;

   /* Compute error */
#if HYPRE_CHECK_MIN_VERSION(22800, 0)
   hypre_ParVectorAxpyz(one, par_x, neg_one, par_xref, par_e);
#else
   hypre_ParVectorCopy(par_x, par_e);
   hypre_ParVectorAxpy(neg_one, par_xref, par_e);
#endif

   /* Compute error norm */
   LinearSystemComputeVectorNorm(vec_e, norm_type, e_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_e);
}

/*-----------------------------------------------------------------------------
 * LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
LinearSystemComputeResidualNorm(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                                HYPRE_IJVector vec_x, const char *norm_type,
                                double *res_norm)
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
#if HYPREDRV_HAVE_MEMORY_APIS
   HYPREDRV_IJVectorInitialize(vec_r, hypre_IJVectorMemoryLocation(vec_b));
#else
   HYPREDRV_IJVectorInitialize(vec_r, HYPRE_MEMORY_HOST);
#endif
   HYPRE_IJVectorGetObject(vec_r, &obj_r);
   par_r = (HYPRE_ParVector)obj_r;
   HYPRE_ParVectorCopy(par_b, par_r);

   /* Compute residual */
   HYPRE_ParCSRMatrixMatvec(neg_one, par_A, par_x, one, par_r);

   /* Compute residual norm */
   LinearSystemComputeVectorNorm(vec_r, norm_type, res_norm);

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
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
      snprintf(A_name, sizeof(A_name), "%.*s.out", max_base, A_base);
   }
   {
      int max_base = (int)sizeof(b_name) - 1 - 4;
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
      snprintf(b_name, sizeof(b_name), "%.*s.out", max_base, b_base);
   }
   {
      int max_base = (int)sizeof(d_name) - 1 - 4;
      if (max_base < 0) max_base = 0; /* LCOV_EXCL_LINE */ /* GCOVR_EXCL_LINE */
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
         const struct dirent *ent = NULL;
         while ((ent = readdir(dir)) != NULL)
         {
            if (ent->d_name[0] == 'l' && ent->d_name[1] == 's' && ent->d_name[2] == '_')
            {
               int idx = (int)strtol(ent->d_name + 3, NULL, 10);
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
