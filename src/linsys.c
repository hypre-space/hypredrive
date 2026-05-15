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

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "logging.h"

static void
IJVectorInitializeCompat(HYPRE_IJVector vec, HYPRE_MemoryLocation memory_location)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 22000
   HYPRE_IJVectorInitialize_v2(vec, memory_location);
#else
   (void)memory_location;
   HYPRE_IJVectorInitialize(vec);
#endif
}

static void
IJMatrixInitializeCompat(HYPRE_IJMatrix mat, HYPRE_MemoryLocation memory_location)
{
#if HYPREDRV_HYPRE_RELEASE_NUMBER >= 21900
   HYPRE_IJMatrixInitialize_v2(mat, memory_location);
#else
   (void)memory_location;
   HYPRE_IJMatrixInitialize(mat);
#endif
}

#define HYPREDRV_HAVE_MEMORY_APIS (HYPREDRV_HYPRE_RELEASE_NUMBER >= 22000)

/* TODO: implement IJVectorClone/Copy and IJVectorMigrate/IJMatrix in hypre*/

static void
LinearSystemSetSuffixSet(void *field, const YAMLnode *node)
{
   IntArray **ptr = (IntArray **)field;
   /* GCOVR_EXCL_BR_START */
   const char *val = node->mapped_val ? node->mapped_val : node->val;
   /* GCOVR_EXCL_BR_STOP */

   hypredrv_IntArrayDestroy(ptr);
   /* GCOVR_EXCL_BR_START */
   if (val && strlen(val) > 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_StrToIntArray(val, ptr);
   }
}

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, dirname, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sequence_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, matrix_basename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_basename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_basename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, xref_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, xref_basename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, x0_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sol_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, timestep_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, dofmap_basename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, digits_suffix, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_suffix, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, last_suffix, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, set_suffix, LinearSystemSetSuffixSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, init_guess_mode, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, rhs_mode, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, type, hypredrv_FieldTypeIntSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, print_system, hypredrv_PrintSystemSetArgs),
   FIELD_OFFSET_MAP_ENTRY(LS_args, eigspec, hypredrv_EigSpecSetArgs),
   /* dof_labels is handled via a special-case branch in SetArgsFromYAML; the
    * entry here only serves the validator so it accepts the key. */
   FIELD_OFFSET_MAP_ENTRY(LS_args, dof_labels, hypredrv_FieldTypeNoopSet),
};

#define LS_NUM_FIELDS (sizeof(ls_field_offset_map) / sizeof(ls_field_offset_map[0]))

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetFieldByName
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetFieldByName(LS_args *args, const YAMLnode *node)
{
   /* GCOVR_EXCL_BR_START */
   for (size_t i = 0; i < LS_NUM_FIELDS; i++) /* GCOVR_EXCL_BR_STOP */
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
 * hypredrv_LinearSystemGetValidKeys
 *-----------------------------------------------------------------------------*/

StrArray
hypredrv_LinearSystemGetValidKeys(void)
{
   static const char *keys[LS_NUM_FIELDS];

   for (size_t i = 0; i < LS_NUM_FIELDS; i++)
   {
      keys[i] = ls_field_offset_map[i].name;
   }

   return STR_ARRAY_CREATE(keys);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_LinearSystemGetValidValues(const char *key)
{
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"online", 0}, {"ij", 1}, {"parcsr", 2}, {"mtx", 3}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "rhs_mode"))
   {
      static StrIntMap map[] = {
         {"zeros", 0}, {"ones", 1}, {"file", 2}, {"random", 3}, {"randsol", 4},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   if (!strcmp(key, "init_guess_mode"))
   {
      static StrIntMap map[] = {
         {"zeros", 0}, {"ones", 1}, {"file", 2}, {"random", 3}, {"previous", 4},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   else
   {
      return STR_INT_MAP_ARRAY_VOID();
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetDefaultArgs(LS_args *args)
{
   args->dirname[0]           = '\0';
   args->sequence_filename[0] = '\0';
   args->matrix_filename[0]   = '\0';
   args->matrix_basename[0]   = '\0';
   args->precmat_filename[0]  = '\0';
   args->precmat_basename[0]  = '\0';
   args->rhs_filename[0]      = '\0';
   args->rhs_basename[0]      = '\0';
   args->x0_filename[0]       = '\0';
   args->xref_filename[0]     = '\0';
   args->xref_basename[0]     = '\0';
   args->timestep_filename[0] = '\0';
   args->sol_filename[0]      = '\0';
   args->dofmap_filename[0]   = '\0';
   args->dofmap_basename[0]   = '\0';
   args->digits_suffix        = 5;
   args->init_suffix          = -1;
   args->last_suffix          = -1;
   args->set_suffix           = NULL;
   args->init_guess_mode      = 0;
   args->rhs_mode             = 2;
   args->type                 = 1;
   args->num_systems          = 1;
#ifdef HYPRE_USING_GPU
   args->exec_policy = 1;
#else
   args->exec_policy = 0;
#endif

   hypredrv_PrintSystemSetDefaultArgs(&args->print_system);

   /* Eigenspectrum defaults */
   hypredrv_EigSpecSetDefaultArgs(&args->eigspec);

   args->dof_labels = NULL;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetNearNullSpace
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetNearNullSpace(MPI_Comm comm, const LS_args *args,
                                      HYPRE_IJMatrix mat, int num_entries,
                                      int num_components, const HYPRE_Complex *values,
                                      HYPRE_IJVector *vec_nn_ptr)
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
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "Number of entries (%d) does not match the expected local size (%d)",
         num_entries, loc_expected);
      return;
   }

   /* Create a ParCSR IJVector with host memory (we'll migrate later if needed) */
   HYPRE_IJVectorCreate(comm, jlower, jupper, vec_nn_ptr);
   HYPRE_IJVectorSetObjectType(*vec_nn_ptr, HYPRE_PARCSR);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_IJVectorSetNumComponents(*vec_nn_ptr, num_components);
#endif
   IJVectorInitializeCompat(*vec_nn_ptr, HYPRE_MEMORY_HOST);

   HYPRE_BigInt  *indices = NULL;
   HYPRE_Complex *zeros   = NULL;
   /* GCOVR_EXCL_BR_START */
   if (num_entries > 0) /* GCOVR_EXCL_BR_STOP */
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
   /* GCOVR_EXCL_START */
   if (args && args->exec_policy)
   {
#if HYPRE_CHECK_MIN_VERSION(23300, 0)
      HYPRE_IJVectorMigrate(*vec_nn_ptr, HYPRE_MEMORY_DEVICE);
#endif
   }
   /* GCOVR_EXCL_STOP */
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetNumSystems
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetNumSystems(LS_args *args)
{
   /* GCOVR_EXCL_BR_START */
   if (args->sequence_filename[0] != '\0') /* GCOVR_EXCL_BR_STOP */
   {
      int num_systems = 0;
      /* GCOVR_EXCL_BR_START */
      if (hypredrv_LSSeqReadSummary(args->sequence_filename, &num_systems, NULL, NULL,
                                    NULL))
      /* GCOVR_EXCL_BR_STOP */
      {
         args->num_systems = (HYPRE_Int)num_systems;
      }
      return;
   }

   /* GCOVR_EXCL_BR_START */
   if (args->set_suffix != NULL && args->set_suffix->size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      args->num_systems = (HYPRE_Int)args->set_suffix->size;
   }
   else
   {
      args->num_systems = args->last_suffix - args->init_suffix + 1;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemGetSuffix
 *-----------------------------------------------------------------------------*/

int
hypredrv_LinearSystemGetSuffix(const LS_args *args, int ls_id)
{
   /* GCOVR_EXCL_BR_START */
   if (!args) /* GCOVR_EXCL_BR_STOP */
   {
      return ls_id;
   }
   /* GCOVR_EXCL_BR_START */
   if (args->set_suffix != NULL && ls_id >= 1 &&
       (size_t)(ls_id - 1) < args->set_suffix->size)
   /* GCOVR_EXCL_BR_STOP */
   {
      return args->set_suffix->data[ls_id - 1];
   }
   return (int)args->init_suffix + ls_id;
}

static HYPRE_MemoryLocation
LinearSystemMemoryLocationGet(const LS_args *args)
{
   /* GCOVR_EXCL_BR_START */
   return (args && args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
   /* GCOVR_EXCL_BR_STOP */
}

static int
LinearSystemStatsIDGet(const Stats *stats)
{
   return stats ? hypredrv_StatsGetLinearSystemID(stats) : 0;
}

static const char *
LinearSystemLogObjectName(const Stats *stats, char *buf, size_t buf_size)
{
   if (stats && stats->object_name[0] != '\0')
   {
      return stats->object_name;
   }
   /* GCOVR_EXCL_BR_START */
   if (stats && stats->runtime_object_id > 0 && buf && buf_size > 0)
   /* GCOVR_EXCL_BR_STOP */
   {
      snprintf(buf, buf_size, "obj-%d", stats->runtime_object_id);
      return buf;
   }
   return NULL;
}

static MPI_Comm
LinearSystemCommFromVector(HYPRE_IJVector vec)
{
   if (!vec)
   {
      return MPI_COMM_NULL;
   }

   return hypre_IJVectorComm((hypre_IJVector *)vec);
}

static int
LinearSystemDataFilenameResolve(const LS_args *args, int ls_id, const char *filename,
                                const char *basename, char *resolved,
                                size_t resolved_size)
{
   /* GCOVR_EXCL_BR_START */
   if (!args || !resolved || resolved_size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return 0; /* GCOVR_EXCL_LINE */
   }

   resolved[0] = '\0';
   if (args->dirname[0] != '\0')
   {
      int suffix = hypredrv_LinearSystemGetSuffix(args, ls_id);
      snprintf(resolved, resolved_size, "%.*s_%0*d/%.*s", (int)strlen(args->dirname),
               args->dirname, (int)args->digits_suffix, suffix, (int)strlen(filename),
               filename);
      return 1;
   }
   if (filename[0] != '\0')
   {
      snprintf(resolved, resolved_size, "%s", filename);
      return 1;
   }
   if (basename[0] != '\0')
   {
      int suffix = hypredrv_LinearSystemGetSuffix(args, ls_id);
      snprintf(resolved, resolved_size, "%.*s_%0*d", (int)strlen(basename), basename,
               (int)args->digits_suffix, suffix);
      return 1;
   }

   return 0;
}

static int
LinearSystemMultipartCanRead(MPI_Comm comm, const char *prefixname)
{
   int nprocs = 0;
   int nparts = 0;

   MPI_Comm_size(comm, &nprocs);
   nparts = hypredrv_CountNumberOfPartitions(prefixname);
   return (nparts >= nprocs) != 0;
}

static void
LinearSystemIJVectorReadFromFile(MPI_Comm comm, const char *filename,
                                 HYPRE_MemoryLocation memory_location,
                                 HYPRE_IJVector      *vector_ptr)
{
   if (hypredrv_CheckBinaryDataExists(filename))
   {
      if (LinearSystemMultipartCanRead(comm, filename))
      {
         int nparts = hypredrv_CountNumberOfPartitions(filename);
         hypredrv_IJVectorReadMultipartBinary(filename, comm, (uint64_t)nparts,
                                              memory_location, vector_ptr);
      }
      else
      {
#if HYPRE_CHECK_MIN_VERSION(23000, 0)
         HYPRE_IJVectorReadBinary(filename, comm, HYPRE_PARCSR, vector_ptr);
#else
         HYPRE_IJVectorRead(filename, comm, HYPRE_PARCSR, vector_ptr);
#endif
      }
   }
   else
   {
      HYPRE_IJVectorRead(filename, comm, HYPRE_PARCSR, vector_ptr);
   }
}

static void
LinearSystemIJMatrixMigrate(const LS_args *args, HYPRE_IJMatrix matrix)
{
   /* GCOVR_EXCL_START */
   if (!args || !args->exec_policy || !matrix)
   {
      return;
   }

   void *obj = NULL;
   HYPRE_IJMatrixGetObject(matrix, &obj);
   HYPRE_ParCSRMatrix par_A = (HYPRE_ParCSRMatrix)obj;

#if HYPREDRV_HAVE_MEMORY_APIS
   hypre_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_DEVICE);
#endif
   /* GCOVR_EXCL_STOP */
}

static void
LinearSystemIJVectorMigrate(const LS_args *args, HYPRE_IJVector vec)
{
   /* GCOVR_EXCL_START */
   if (!args || !args->exec_policy || !vec)
   {
      return;
   }

   void           *obj = NULL;
   HYPRE_ParVector par = NULL;
   HYPRE_IJVectorGetObject(vec, &obj);
   par = (HYPRE_ParVector)obj;

#if HYPREDRV_HAVE_MEMORY_APIS
   hypre_ParVectorMigrate(par, HYPRE_MEMORY_DEVICE);
#endif
   /* GCOVR_EXCL_STOP */
}

static int
LinearSystemIJMatrixReadFromFile(MPI_Comm comm, const LS_args *args,
                                 const char *matrix_filename, HYPRE_IJMatrix *matrix_ptr)
{
   int                  file_not_found  = 0;
   HYPRE_MemoryLocation memory_location = LinearSystemMemoryLocationGet(args);

   if (args->type == 1)
   {
      if (hypredrv_CheckBinaryDataExists(matrix_filename))
      {
         /* GCOVR_EXCL_START */
         /* GCOVR_EXCL_BR_START */
         if (LinearSystemMultipartCanRead(comm, matrix_filename)) /* GCOVR_EXCL_BR_STOP */
         {
            int nparts = hypredrv_CountNumberOfPartitions(matrix_filename);
            hypredrv_IJMatrixReadMultipartBinary(matrix_filename, comm, (uint64_t)nparts,
                                                 memory_location, matrix_ptr);
         }
         else
         {
            hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
            hypredrv_ErrorMsgAddInvalidFilename(matrix_filename);
            return 0;
         }
         /* GCOVR_EXCL_STOP */
      }
      else if (hypredrv_CheckASCIIDataExists(matrix_filename))
      {
         HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
      }
      else
      {
         file_not_found = 1;
      }
   }
   /* GCOVR_EXCL_BR_START */
   else if (args->type == 3) /* GCOVR_EXCL_BR_STOP */
   {
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      HYPRE_IJMatrixReadMM(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
#else
      HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
#endif
   }

   if (HYPRE_GetError() || file_not_found)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(matrix_filename);
      return 0;
   }

   LinearSystemIJMatrixMigrate(args, *matrix_ptr);
   return 1;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetArgsFromYAML(LS_args *args, YAMLnode *parent)
{
   YAML_NODE_ITERATE(parent, child)
   {
      YAML_NODE_VALIDATE(child, hypredrv_LinearSystemGetValidKeys,
                         hypredrv_LinearSystemGetValidValues);

      /* Special handling for dof_labels: parse as label->int map.
       *
       * Two YAML forms are accepted:
       *
       *   Block mapping (one entry per line):
       *     dof_labels:
       *       v_x: 0
       *       v_y: 1
       *       p:   2
       *
       *   Flow mapping (inline):
       *     dof_labels: {v_x: 0, v_y: 1, p: 2}
       *
       * Label keys are normalised to lowercase on storage so they match the
       * lowercased values the YAML parser produces for f_dofs entries. */
      if (!strcmp(child->key, "dof_labels"))
      {
         args->dof_labels = hypredrv_DofLabelMapCreate();

         if (child->children)
         {
            /* Block mapping: each child node is a label:value pair */
            YAML_NODE_ITERATE(child, entry)
            {
               int  val = 0;
               char lower_key[64];
               if (sscanf(entry->val, "%d", &val) != 1)
               {
                  hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
                  hypredrv_ErrorMsgAdd("dof_labels: expected integer value for "
                                       "label '%s', got '%s'",
                                       entry->key, entry->val);
                  break;
               }
               strncpy(lower_key, entry->key, sizeof(lower_key) - 1);
               lower_key[sizeof(lower_key) - 1] = '\0';
               hypredrv_StrToLowerCase(lower_key);
               hypredrv_DofLabelMapAdd(args->dof_labels, lower_key, val);
               YAML_NODE_SET_VALID(entry);
            }
         }
         /* GCOVR_EXCL_BR_START */
         else if (child->val && child->val[0] == '{') /* GCOVR_EXCL_BR_STOP */
         {
            /* Flow mapping: val is already lowercased by the YAML parser,
             * so keys inside the string are also lowercase. */
            char *buf   = strdup(child->val);
            char *inner = buf;
            /* GCOVR_EXCL_BR_START */
            while (*inner == '{' || *inner == ' ') inner++;
            /* GCOVR_EXCL_BR_STOP */
            char *close = strrchr(inner, '}');
            /* GCOVR_EXCL_BR_START */
            if (close) *close = '\0';
            /* GCOVR_EXCL_BR_STOP */
            char *pair = strtok(inner, ",");
            while (pair)
            {
               while (*pair == ' ') pair++;
               char *colon = strchr(pair, ':');
               /* GCOVR_EXCL_BR_START */
               if (colon) /* GCOVR_EXCL_BR_STOP */
               {
                  *colon         = '\0';
                  char *pair_key = pair;
                  char *pair_val = colon + 1;
                  hypredrv_StrTrim(pair_key);
                  while (*pair_val == ' ') pair_val++;
                  int val_int = 0;
                  if (sscanf(pair_val, "%d", &val_int) != 1)
                  {
                     hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
                     hypredrv_ErrorMsgAdd("dof_labels: expected integer value for "
                                          "label '%s', got '%s'",
                                          pair_key, pair_val);
                     break;
                  }
                  hypredrv_DofLabelMapAdd(args->dof_labels, pair_key, val_int);
               }
               pair = strtok(NULL, ",");
            }
            free(buf);
         }

         YAML_NODE_SET_VALID(child);
         continue;
      }

      YAML_NODE_SET_FIELD(child, args, hypredrv_LinearSystemSetFieldByName);
   }

   /* set_suffix and init_suffix/last_suffix are mutually exclusive */
   /* GCOVR_EXCL_BR_START */
   if (args->set_suffix != NULL && args->set_suffix->size > 0 &&
       (args->init_suffix >= 0 || args->last_suffix >= 0))
   /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system: set_suffix cannot be used with init_suffix or last_suffix");
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemReadMatrix(MPI_Comm comm, const LS_args *args,
                                HYPRE_IJMatrix *matrix_ptr, Stats *stats)
{
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "matrix");

   char matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int  ls_id                                = hypredrv_StatsGetLinearSystemID(stats) + 1;
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix read begin");

   /* Destroy matrix if it already exists */
   if (*matrix_ptr)
   {
      HYPRE_IJMatrixDestroy(*matrix_ptr);
   }

   if (args->sequence_filename[0] != '\0')
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "matrix source: sequence file '%s'", args->sequence_filename);
      if (!hypredrv_LSSeqReadMatrix(comm, args->sequence_filename, ls_id,
                                    LinearSystemMemoryLocationGet(args), matrix_ptr))
      {
         hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "matrix");
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "matrix read failed from sequence source");
         return;
      }

      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "matrix");
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix read end");
      return;
   }

   if (!LinearSystemDataFilenameResolve(args, ls_id, args->matrix_filename,
                                        args->matrix_basename, matrix_filename,
                                        sizeof(matrix_filename)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename("");
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "matrix");
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                         "matrix filename resolution failed");
      return;
   }
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix source: '%s'",
                      matrix_filename);

   if (!LinearSystemIJMatrixReadFromFile(comm, args, matrix_filename, matrix_ptr))
   {
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "matrix");
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id, "matrix read failed from '%s'",
                         matrix_filename);
      return;
   }

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "matrix");
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix read end");
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemBuildMatrixFromCSR
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemBuildMatrixFromCSR(MPI_Comm             comm,
                                        HYPRE_MemoryLocation memory_location,
                                        HYPRE_BigInt row_start, HYPRE_BigInt row_end,
                                        const HYPRE_BigInt *indptr,
                                        const HYPRE_BigInt *col_indices,
                                        const HYPRE_Real *data, HYPRE_IJMatrix *mat_ptr)
{
   if (!mat_ptr || !indptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: mat_ptr and indptr must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }
   if (row_end < row_start)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: row_end (%lld) < row_start (%lld)",
                           (long long)row_end, (long long)row_start);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_BigInt nrows_big = row_end - row_start + 1;
   if (nrows_big > (HYPRE_BigInt)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: local row count (%lld) is out of HYPRE_Int range",
                           (long long)nrows_big);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Int nrows = (HYPRE_Int)nrows_big;
   if (indptr[0] < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: indptr[0] (%lld) must be nonnegative",
                           (long long)indptr[0]);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_BigInt nnz_big = indptr[nrows] - indptr[0];
   if (nnz_big < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: indptr[nrows] (%lld) < indptr[0] (%lld)",
                           (long long)indptr[nrows], (long long)indptr[0]);
      return hypredrv_ErrorCodeGet();
   }
   if (nnz_big > (HYPRE_BigInt)INT_MAX)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: local nonzero count (%lld) exceeds HYPRE_Int range",
                           (long long)nnz_big);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Int nnz = (HYPRE_Int)nnz_big;
   if (nnz > 0 && (!col_indices || !data))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "BuildMatrixFromCSR: col_indices/data must be non-NULL when nnz > 0");
      return hypredrv_ErrorCodeGet();
   }

   /* Destroy any pre-existing matrix at *mat_ptr (caller pattern, like ReadMatrix) */
   if (*mat_ptr)
   {
      HYPRE_IJMatrixDestroy(*mat_ptr);
      *mat_ptr = NULL;
   }

   /* IJMatrixCreate requires a square global column range matching the row range
    * across all ranks. We pass the rank-local row range for both row and column
    * lower/upper bounds; HYPRE composes the global column partition from the
    * concatenation. This matches how matrices read from file are built (see
    * src/matrix.c) and gives standard ParCSR layout. */
   HYPRE_IJMatrixCreate(comm, row_start, row_end, row_start, row_end, mat_ptr);
   HYPRE_IJMatrixSetObjectType(*mat_ptr, HYPRE_PARCSR);
   IJMatrixInitializeCompat(*mat_ptr, memory_location);

   if (nrows == 0)
   {
      HYPRE_IJMatrixAssemble(*mat_ptr);
      return hypredrv_ErrorCodeGet();
   }
   if (nnz == 0)
   {
      HYPRE_IJMatrixAssemble(*mat_ptr);
      return hypredrv_ErrorCodeGet();
   }

   /* Build the per-row column counts and the global row index list expected by
    * HYPRE_IJMatrixSetValues. We allocate using HYPRE's allocator so that the
    * temporary fits cleanly in the existing memory-tracking story. */
   /* HYPRE_IJMatrixSetValues consumes row metadata as host-side scratch even when
    * matrix values are initialized for a device memory location. */
   HYPRE_Int    *ncols_per_row = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   HYPRE_BigInt *row_ids       = hypre_TAlloc(HYPRE_BigInt, nrows, HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < nrows; i++)
   {
      HYPRE_BigInt row_nnz = indptr[i + 1] - indptr[i];
      if (row_nnz < 0)
      {
         hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
         hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
         HYPRE_IJMatrixDestroy(*mat_ptr);
         *mat_ptr = NULL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: indptr is not monotonically "
                              "non-decreasing at row %lld",
                              (long long)i);
         return hypredrv_ErrorCodeGet();
      }
      if (row_nnz > (HYPRE_BigInt)INT_MAX)
      {
         hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
         hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
         HYPRE_IJMatrixDestroy(*mat_ptr);
         *mat_ptr = NULL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: row %lld nonzero count (%lld) exceeds HYPRE_Int range",
                              (long long)i, (long long)row_nnz);
         return hypredrv_ErrorCodeGet();
      }
      ncols_per_row[i] = (HYPRE_Int)row_nnz;
      row_ids[i]       = row_start + (HYPRE_BigInt)i;
   }

   for (HYPRE_Int k = 0; k < nnz; k++)
   {
      HYPRE_BigInt col = col_indices[indptr[0] + (HYPRE_BigInt)k];
      if (col < 0)
      {
         hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
         hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
         HYPRE_IJMatrixDestroy(*mat_ptr);
         *mat_ptr = NULL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: col_indices[%lld] (%lld) must be nonnegative",
                              (long long)(indptr[0] + (HYPRE_BigInt)k),
                              (long long)col);
         return hypredrv_ErrorCodeGet();
      }
   }

   /* Single-shot value insertion. HYPRE copies into its own ParCSR storage. */
   HYPRE_IJMatrixSetValues(*mat_ptr, nrows, ncols_per_row, row_ids,
                           col_indices + indptr[0], data + indptr[0]);
   HYPRE_IJMatrixAssemble(*mat_ptr);

   hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
   hypre_TFree(row_ids, HYPRE_MEMORY_HOST);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemBuildRHSFromArray
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemBuildRHSFromArray(MPI_Comm             comm,
                                       HYPRE_MemoryLocation memory_location,
                                       HYPRE_BigInt row_start, HYPRE_BigInt row_end,
                                       const HYPRE_Real *values, HYPRE_IJVector *rhs_ptr)
{
   if (!rhs_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildRHSFromArray: rhs_ptr must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }
   if (row_end < row_start)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildRHSFromArray: row_end (%lld) < row_start (%lld)",
                           (long long)row_end, (long long)row_start);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_BigInt nrows_big = row_end - row_start + 1;
   if ((HYPRE_BigInt)((HYPRE_Int)nrows_big) != nrows_big)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildRHSFromArray: local row count (%lld) exceeds HYPRE_Int range",
                           (long long)nrows_big);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Int nrows = (HYPRE_Int)nrows_big;
   if (nrows > 0 && !values)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildRHSFromArray: values must be non-NULL when nrows > 0");
      return hypredrv_ErrorCodeGet();
   }

   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }

   HYPRE_IJVectorCreate(comm, row_start, row_end, rhs_ptr);
   HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
   IJVectorInitializeCompat(*rhs_ptr, memory_location);

   if (nrows > 0)
   {
      HYPRE_IJVectorSetValues(*rhs_ptr, nrows, NULL, values);
   }
   HYPRE_IJVectorAssemble(*rhs_ptr);

   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemMatrixGetNumRows
 *-----------------------------------------------------------------------------*/

long long int
hypredrv_LinearSystemMatrixGetNumRows(HYPRE_IJMatrix matrix)
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

   /* GCOVR_EXCL_START */
   if (!par_A)
   {
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   HYPRE_ParCSRMatrixGetDims(par_A, &nrows, &ncols);

   return (long long int)nrows;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemMatrixGetNumNonzeros
 *-----------------------------------------------------------------------------*/

long long int
hypredrv_LinearSystemMatrixGetNumNonzeros(HYPRE_IJMatrix matrix)
{
   HYPRE_ParCSRMatrix par_A = NULL;
   void              *obj   = NULL;

   if (!matrix)
   {
      return 0;
   }

   HYPRE_IJMatrixGetObject(matrix, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;

   /* GCOVR_EXCL_START */
   if (!par_A)
   {
      return 0;
   }
   /* GCOVR_EXCL_STOP */

   hypre_ParCSRMatrixSetDNumNonzeros(par_A);

   return (long long int)par_A->d_num_nonzeros;
}

#if defined(HYPRE_BIG_INT)
#define HYPRE_BIG_INT_SSCANF "%lld"
#else
#define HYPRE_BIG_INT_SSCANF "%d"
#endif

static int
LinearSystemRHSMatrixMarketRead(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                                const char *rhs_filename, HYPRE_IJVector *rhs_ptr)
{
   int                  myid      = 0;
   int                  num_procs = 0;
   FILE                *file      = NULL;
   char                 line[1024];
   HYPRE_BigInt         M = 0;
   HYPRE_BigInt         N;
   HYPRE_Complex       *all_values      = NULL;
   HYPRE_Complex       *local_values    = NULL;
   HYPRE_BigInt         global_num_rows = 0, global_num_cols = 0;
   HYPRE_ParCSRMatrix   par_A  = NULL;
   void                *obj    = NULL;
   int                 *counts = NULL;
   int                 *displs = NULL;
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memory_location = LinearSystemMemoryLocationGet(args);

   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   HYPRE_IJMatrixGetObject(mat, &obj);
   par_A = (HYPRE_ParCSRMatrix)obj;
   HYPRE_ParCSRMatrixGetDims(par_A, &global_num_rows, &global_num_cols);

   /* GCOVR_EXCL_BR_START */
   if (myid == 0) /* GCOVR_EXCL_BR_STOP */
   {
      file = fopen(rhs_filename, "r");
      if (file == NULL)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Cannot open file %s", rhs_filename);
         M = -1;
      }
      else
      {
         do
         {
            if (fgets(line, sizeof(line), file) == NULL)
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
               hypredrv_ErrorMsgAdd("Unexpected end of file or error reading %s",
                                    rhs_filename);
               M = -1;
               break;
            }
         } while (line[0] == '%');

         if (M != -1)
         {
#ifdef HYPRE_BIG_INT
            long long   tmpM     = strtoll(line, NULL, 10);
            const char *line_ptr = strchr(line, ' ');
            /* GCOVR_EXCL_BR_START */
            long long tmpN = (line_ptr != NULL) ? strtoll(line_ptr + 1, NULL, 10) : 0;
            /* GCOVR_EXCL_BR_STOP */
            /* GCOVR_EXCL_BR_START */
            int read_ok = (tmpM != 0 && tmpN != 0);
            /* GCOVR_EXCL_BR_STOP */
#else
            int         tmpM     = (int)strtol(line, NULL, 10);
            const char *line_ptr = strchr(line, ' ');
            /* GCOVR_EXCL_BR_START */
            int tmpN = (line_ptr != NULL) ? (int)strtol(line_ptr + 1, NULL, 10) : 0;
            /* GCOVR_EXCL_BR_STOP */
            /* GCOVR_EXCL_BR_START */
            int read_ok = (tmpM != 0 && tmpN != 0);
            /* GCOVR_EXCL_BR_STOP */
#endif

            if (read_ok)
            {
               M = (HYPRE_BigInt)tmpM;
               N = (HYPRE_BigInt)tmpN;
            }
            else
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
               hypredrv_ErrorMsgAdd("Failed to read vector dimensions from %s",
                                    rhs_filename);
               M = -1;
               N = 0;
            }

            if (N != 1)
            {
               hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
               hypredrv_ErrorMsgAdd("File %s is not a vector (N=" HYPRE_BIG_INT_SSCANF
                                    ")",
                                    rhs_filename, N);
               M = -1;
            }
            else if (M != global_num_rows)
            {
               hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
               hypredrv_ErrorMsgAdd("RHS vector size " HYPRE_BIG_INT_SSCANF
                                    " does not match matrix size " HYPRE_BIG_INT_SSCANF,
                                    M, global_num_rows);
               M = -1;
            }
            else
            {
               all_values = hypre_TAlloc(HYPRE_Complex, M, HYPRE_MEMORY_HOST);
               for (HYPRE_BigInt i = 0; i < M; i++)
               {
                  char *endptr = NULL;
                  if (fgets(line, sizeof(line), file) == NULL)
                  {
                     hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
                     hypredrv_ErrorMsgAdd(
                        "Error reading value for index " HYPRE_BIG_INT_SSCANF " from %s",
                        i, rhs_filename);
                     M = -1;
                     break;
                  }
                  double tmp_val = strtod(line, &endptr);
                  /* GCOVR_EXCL_BR_START */
                  if (endptr == line || (*endptr != '\0' && *endptr != '\n'))
                  /* GCOVR_EXCL_BR_STOP */
                  {
                     hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
                     hypredrv_ErrorMsgAdd(
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

   MPI_Bcast(&M, 1, HYPRE_MPI_BIG_INT, 0, comm);
   if (M == -1)
   {
      /* GCOVR_EXCL_BR_START */
      if (myid == 0 && all_values) /* GCOVR_EXCL_BR_STOP */
      {
         hypre_TFree(all_values, HYPRE_MEMORY_HOST);
      }
      return 0;
   }

   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
   HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
   IJVectorInitializeCompat(*rhs_ptr, memory_location);

   HYPRE_BigInt local_size    = iupper - ilower + 1;
   int          my_local_size = local_size;
   /* GCOVR_EXCL_BR_START */
   if (myid == 0) /* GCOVR_EXCL_BR_STOP */
   {
      counts = hypre_TAlloc(int, num_procs, HYPRE_MEMORY_HOST);
      displs = hypre_TAlloc(int, num_procs, HYPRE_MEMORY_HOST);
   }
   MPI_Gather(&my_local_size, 1, MPI_INT, counts, 1, MPI_INT, 0, comm);

   /* GCOVR_EXCL_BR_START */
   if (myid == 0) /* GCOVR_EXCL_BR_STOP */
   {
      displs[0] = 0;
      /* GCOVR_EXCL_START */
      for (int i = 1; i < num_procs; i++)
      {
         displs[i] = displs[i - 1] + counts[i - 1];
      }
      /* GCOVR_EXCL_STOP */
   }

   local_values = hypre_TAlloc(HYPRE_Complex, local_size, HYPRE_MEMORY_HOST);
   MPI_Scatterv(all_values, counts, displs, MPI_DOUBLE, local_values, local_size,
                MPI_DOUBLE, 0, comm);

   HYPRE_IJVectorSetValues(*rhs_ptr, local_size, NULL, local_values);
   HYPRE_IJVectorAssemble(*rhs_ptr);

   hypre_TFree(local_values, HYPRE_MEMORY_HOST);
   /* GCOVR_EXCL_BR_START */
   if (myid == 0) /* GCOVR_EXCL_BR_STOP */
   {
      hypre_TFree(all_values, HYPRE_MEMORY_HOST);
      hypre_TFree(counts, HYPRE_MEMORY_HOST);
      hypre_TFree(displs, HYPRE_MEMORY_HOST);
   }

   return 1;
}

static void
LinearSystemRHSGeneratedSet(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                            HYPRE_IJVector *xref_ptr, HYPRE_IJVector *rhs_ptr)
{
   HYPRE_BigInt         ilower = 0, iupper = 0;
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memory_location = LinearSystemMemoryLocationGet(args);

   HYPRE_IJMatrixGetLocalRange(mat, &ilower, &iupper, &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, ilower, iupper, rhs_ptr);
   HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR);
   IJVectorInitializeCompat(*rhs_ptr, memory_location);

   void           *obj   = NULL;
   HYPRE_ParVector par_b = NULL;
   HYPRE_IJVectorGetObject(*rhs_ptr, &obj);
   par_b = (HYPRE_ParVector)obj;

   switch (args->rhs_mode)
   {
      case 0:
         HYPRE_ParVectorSetConstantValues(par_b, 0);
         break;
      case 1:
      default:
         HYPRE_ParVectorSetConstantValues(par_b, 1);
         break;
      case 3:
         HYPRE_ParVectorSetRandomValues(par_b, 2023);
         break;
      case 4:
      {
         HYPRE_IJVector xref = NULL;
         HYPRE_IJVectorCreate(comm, ilower, iupper, &xref);
         HYPRE_IJVectorSetObjectType(xref, HYPRE_PARCSR);
         IJVectorInitializeCompat(xref, memory_location);

         HYPRE_ParVector par_x = NULL;
         HYPRE_IJVectorGetObject(xref, &obj);
         par_x = (HYPRE_ParVector)obj;
         HYPRE_ParVectorSetRandomValues(par_x, 2023);

         void              *obj_A = NULL;
         HYPRE_ParCSRMatrix par_A = NULL;
#if HYPREDRV_HAVE_MEMORY_APIS
         HYPRE_MemoryLocation mat_memory_location = HYPRE_MEMORY_UNDEFINED;
#endif
         HYPRE_IJMatrixGetObject(mat, &obj_A);
         par_A = (HYPRE_ParCSRMatrix)obj_A;
#if HYPREDRV_HAVE_MEMORY_APIS
         mat_memory_location =
            hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)par_A);
         if (mat_memory_location != memory_location)
         {
            hypre_ParCSRMatrixMigrate((hypre_ParCSRMatrix *)par_A, memory_location);
         }
#endif
         HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b);
         *xref_ptr = xref;
         break;
      }
   }
}

static int
LinearSystemRHSReadFromFile(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                            const char *rhs_filename, HYPRE_IJVector *rhs_ptr)
{
   int ok = 1;
   if (args->type == 3)
   {
      ok = LinearSystemRHSMatrixMarketRead(comm, args, mat, rhs_filename, rhs_ptr);
   }
   else
   {
      LinearSystemIJVectorReadFromFile(comm, rhs_filename,
                                       LinearSystemMemoryLocationGet(args), rhs_ptr);
   }

   if (HYPRE_GetError())
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(rhs_filename);
      return 0;
   }

   if (ok)
   {
      LinearSystemIJVectorMigrate(args, *rhs_ptr);
   }
   return ok;
}

void
hypredrv_LinearSystemSetRHS(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                            HYPRE_IJVector *xref_ptr, HYPRE_IJVector *rhs_ptr,
                            Stats *stats)
{
   int         ls_id = hypredrv_StatsGetLinearSystemID(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "rhs");
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "rhs setup begin (rhs_mode=%d)",
                      (int)args->rhs_mode);

   if (*xref_ptr)
   {
      HYPRE_IJVectorDestroy(*xref_ptr);
      *xref_ptr = NULL;
   }
   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }

   if (args->rhs_mode != 2)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "rhs source: generated mode=%d",
                         (int)args->rhs_mode);
      LinearSystemRHSGeneratedSet(comm, args, mat, xref_ptr, rhs_ptr);
   }
   else
   {
      if (args->sequence_filename[0] != '\0')
      {
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                            "rhs source: sequence file '%s'", args->sequence_filename);
         if (!hypredrv_LSSeqReadRHS(comm, args->sequence_filename, ls_id,
                                    LinearSystemMemoryLocationGet(args), rhs_ptr))
         {
            hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "rhs");
            HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                               "rhs read failed from sequence source");
            return;
         }
      }
      else
      {
         char rhs_filename[MAX_FILENAME_LENGTH] = {0};
         LinearSystemDataFilenameResolve(args, ls_id, args->rhs_filename,
                                         args->rhs_basename, rhs_filename,
                                         sizeof(rhs_filename));
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "rhs source: '%s'",
                            rhs_filename);
         if (!LinearSystemRHSReadFromFile(comm, args, mat, rhs_filename, rhs_ptr))
         {
            hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "rhs");
            HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                               "rhs read failed from '%s'", rhs_filename);
            return;
         }
      }
   }

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "rhs");
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "rhs setup end");
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetInitialGuess
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemCreateWorkingSolution(MPI_Comm comm, const LS_args *args,
                                           HYPRE_IJVector rhs, HYPRE_IJVector *x_ptr)
{
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memloc =
      /* GCOVR_EXCL_BR_START */
      (args && args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
   /* GCOVR_EXCL_BR_STOP */

   /* GCOVR_EXCL_BR_START */
   if (!rhs || !x_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid arguments for LinearSystemCreateWorkingSolution");
      return;
   }

   if (*x_ptr)
   {
      HYPRE_IJVectorDestroy(*x_ptr);
      *x_ptr = NULL;
   }

   HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
   HYPRE_IJVectorCreate(comm, jlower, jupper, x_ptr);
   HYPRE_IJVectorSetObjectType(*x_ptr, HYPRE_PARCSR);
   IJVectorInitializeCompat(*x_ptr, memloc);
}

void
hypredrv_LinearSystemSetInitialGuess(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                                     HYPRE_IJVector rhs, HYPRE_IJVector *x0_ptr,
                                     HYPRE_IJVector *x_ptr, const Stats *stats)
{
   (void)mat;
   int         ls_id = LinearSystemStatsIDGet(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   HYPRE_BigInt         jlower = 0, jupper = 0;
   HYPRE_MemoryLocation memloc =
      /* GCOVR_EXCL_BR_START */
      (args->exec_policy) ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
   /* GCOVR_EXCL_BR_STOP */

   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                      "initial guess setup begin (mode=%d)", (int)args->init_guess_mode);

   /* Destroy initial solution vector */
   if (*x0_ptr)
   {
      HYPRE_IJVectorDestroy(*x0_ptr);
      *x0_ptr = NULL;
   }

   hypredrv_LinearSystemCreateWorkingSolution(comm, args, rhs, x_ptr);
   /* GCOVR_EXCL_START */
   if (hypredrv_ErrorCodeActive())
   {
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                         "initial guess setup failed: could not create working solution");
      return;
   }
   /* GCOVR_EXCL_STOP */

   if (args->x0_filename[0] == '\0')
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "initial guess source: generated mode=%d",
                         (int)args->init_guess_mode);
      HYPRE_IJVectorGetLocalRange(rhs, &jlower, &jupper);
      HYPRE_IJVectorCreate(comm, jlower, jupper, x0_ptr);
      HYPRE_IJVectorSetObjectType(*x0_ptr, HYPRE_PARCSR);
      IJVectorInitializeCompat(*x0_ptr, memloc);

      /* TODO (hypre): add IJVector interfaces to avoid ParVector here */
      void           *obj    = NULL;
      HYPRE_ParVector par_x0 = NULL, par_x = NULL;

      HYPRE_IJVectorGetObject(*x0_ptr, &obj);
      par_x0 = (HYPRE_ParVector)obj;

      switch (args->init_guess_mode)
      {
         case 0:
            /* Vector of zeros */
            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                               "initial guess mode: zeros");
            break;

         case 1:
            /* Vector of ones */
            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                               "initial guess mode: ones");
            HYPRE_ParVectorSetConstantValues(par_x0, 1);
            break;

         case 3:
            /* Vector of random values */
            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                               "initial guess mode: random");
            HYPRE_ParVectorSetRandomValues(par_x0, 2023);
            break;

         case 4:
            /* Use solution from previous linear solve */
            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                               "initial guess mode: previous");
            HYPRE_IJVectorGetObject(*x_ptr, &obj);
            par_x = (HYPRE_ParVector)obj;

            HYPRE_ParVectorCopy(par_x, par_x0);
            break;

         default:
            HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                               "initial guess mode=%d not recognized; using zeros",
                               (int)args->init_guess_mode);
            break;
      }
   }
   else
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "initial guess source: '%s'",
                         args->x0_filename);
      LinearSystemIJVectorReadFromFile(comm, args->x0_filename, memloc, x0_ptr);
      if (HYPRE_GetError())
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(args->x0_filename);
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "initial guess read failed from '%s'", args->x0_filename);
         return;
      }
      LinearSystemIJVectorMigrate(args, *x0_ptr);
   }

   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "initial guess setup end");
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetReferenceSolution
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetReferenceSolution(MPI_Comm comm, const LS_args *args,
                                          HYPRE_IJVector *xref_ptr, const Stats *stats)
{
   char        xref_filename[MAX_FILENAME_LENGTH] = {0};
   int         ls_id = hypredrv_StatsGetLinearSystemID(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "reference solution setup begin");

   /* Keep the existing reference solution (e.g., rhs_mode = randsol) unless a file is
    * explicitly requested. */
   /* GCOVR_EXCL_BR_START */
   if (args->xref_filename[0] == '\0' && args->xref_basename[0] == '\0')
   /* GCOVR_EXCL_BR_STOP */
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "reference solution setup skipped (no file override)");
      return;
   }

   /* GCOVR_EXCL_START */
   if (*xref_ptr)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "reference solution override: replacing existing xref");
      HYPRE_IJVectorDestroy(*xref_ptr);
      *xref_ptr = NULL;
   }
   /* GCOVR_EXCL_STOP */

   /* GCOVR_EXCL_START */
   if (!LinearSystemDataFilenameResolve(args, ls_id, args->xref_filename,
                                        args->xref_basename, xref_filename,
                                        sizeof(xref_filename)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename("");
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                         "reference solution filename resolution failed");
      return;
   }
   /* GCOVR_EXCL_STOP */
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "reference solution source: '%s'",
                      xref_filename);

   LinearSystemIJVectorReadFromFile(comm, xref_filename,
                                    LinearSystemMemoryLocationGet(args), xref_ptr);

   /* Check if hypre had problems reading the input file */
   /* GCOVR_EXCL_START */
   if (HYPRE_GetError())
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(xref_filename);
      *xref_ptr = NULL;
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                         "reference solution read failed from '%s'", xref_filename);
   }
   else
   {
      LinearSystemIJVectorMigrate(args, *xref_ptr);
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "reference solution loaded from '%s'", xref_filename);
   }
   /* GCOVR_EXCL_STOP */

   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "reference solution setup end");
}
/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemResetInitialGuess(HYPRE_IJVector x0_ptr, HYPRE_IJVector x_ptr,
                                       Stats *stats)
{
   HYPRE_ParVector par_x0 = NULL, par_x = NULL;
   void           *obj_x0 = NULL, *obj_x = NULL;
   MPI_Comm        log_comm = LinearSystemCommFromVector(x_ptr ? x_ptr : x0_ptr);
   int             ls_id    = LinearSystemStatsIDGet(stats);
   char            log_name_buf[32];
   const char     *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   HYPREDRV_LOG_COMMF(3, log_comm, log_object_name, ls_id, "initial guess reset begin");

   /* GCOVR_EXCL_BR_START */
   if (!x0_ptr || !x_ptr) /* GCOVR_EXCL_BR_STOP */
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "reset_x0");
      HYPREDRV_LOG_COMMF(2, log_comm, log_object_name, ls_id,
                         "initial guess reset failed: x0 or x is NULL");
      return;
   }

   /* TODO: implement HYPRE_IJVectorCopy in hypre */
   HYPRE_IJVectorGetObject(x0_ptr, &obj_x0);
   HYPRE_IJVectorGetObject(x_ptr, &obj_x);
   par_x0 = (HYPRE_ParVector)obj_x0;
   par_x  = (HYPRE_ParVector)obj_x;

   HYPRE_ParVectorCopy(par_x0, par_x);

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "reset_x0");
   HYPREDRV_LOG_COMMF(3, log_comm, log_object_name, ls_id, "initial guess reset end");
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetVectorTags
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetVectorTags(HYPRE_IJVector vec, IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   /* GCOVR_EXCL_BR_START */
   if (!vec || !dofmap || !dofmap->data || dofmap->size == 0) /* GCOVR_EXCL_BR_STOP */
   {
      return;
   }

   HYPRE_Int num_tags = 1;
   /* GCOVR_EXCL_BR_START */
   if (dofmap->g_unique_data && dofmap->g_unique_size > 0) /* GCOVR_EXCL_BR_STOP */
   {
      int max_tag = dofmap->g_unique_data[dofmap->g_unique_size - 1];
      /* GCOVR_EXCL_BR_START */
      if (max_tag >= 0) /* GCOVR_EXCL_BR_STOP */
      {
         num_tags = (HYPRE_Int)max_tag + 1;
      }
   }
   /* GCOVR_EXCL_START */
   else if (dofmap->unique_data && dofmap->unique_size > 0)
   {
      int max_tag = dofmap->unique_data[dofmap->unique_size - 1];
      if (max_tag >= 0)
      {
         num_tags = (HYPRE_Int)max_tag + 1;
      }
   }
   /* GCOVR_EXCL_STOP */

   /* GCOVR_EXCL_START */
#if defined(HYPRE_USING_GPU) && HYPREDRV_HAVE_MEMORY_APIS
   HYPRE_MemoryLocation vec_memloc = hypre_IJVectorMemoryLocation((hypre_IJVector *)vec);

   /* In CUDA/HIP builds, library-mode vectors may still be host-backed.
    * Only migrate tags when the vector itself is not host-backed; otherwise,
    * keep the existing host aliasing path and avoid mismatched frees. */
   if (hypre_GetActualMemLocation(vec_memloc) !=
       hypre_GetActualMemLocation(HYPRE_MEMORY_HOST))
   {
      HYPRE_Int *tags = hypre_TAlloc(HYPRE_Int, dofmap->size, vec_memloc);
      hypre_TMemcpy(tags, dofmap->data, HYPRE_Int, dofmap->size, vec_memloc,
                    HYPRE_MEMORY_HOST);
      HYPRE_IJVectorSetTags(vec, 2, num_tags, tags);
   }
   else
   {
      HYPRE_IJVectorSetTags(vec, 0, num_tags, dofmap->data);
   }
#else
   HYPRE_IJVectorSetTags(vec, 0, num_tags, dofmap->data);
#endif
   /* GCOVR_EXCL_STOP */
#else
   (void)vec;
   (void)dofmap;
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetPrecMatrix(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                                   HYPRE_IJMatrix *precmat_ptr, const Stats *stats)
{
   char        matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int         ls_id                                = LinearSystemStatsIDGet(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                      "preconditioner matrix setup begin");

   /* Set matrix filename */
   if (args->dirname[0] != '\0' && args->precmat_filename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d/%.*s",
               (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
               hypredrv_LinearSystemGetSuffix(args, ls_id),
               (int)strlen(args->precmat_filename), args->precmat_filename);
   }
   else if (args->precmat_filename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%s", args->precmat_filename);
   }
   else if (args->precmat_basename[0] != '\0')
   {
      snprintf(matrix_filename, sizeof(matrix_filename), "%.*s_%0*d",
               (int)strlen(args->precmat_basename), args->precmat_basename,
               (int)args->digits_suffix, hypredrv_LinearSystemGetSuffix(args, ls_id));
   }

   /* GCOVR_EXCL_BR_START */
   if (matrix_filename[0] == '\0' || !strcmp(matrix_filename, args->matrix_filename))
   /* GCOVR_EXCL_BR_STOP */
   {
      *precmat_ptr = mat;
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: reusing main matrix");
   }
   else
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: '%s'", matrix_filename);
      /* Destroy matrix */
      if (*precmat_ptr)
      {
         HYPRE_IJMatrixDestroy(*precmat_ptr);
      }

      HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, precmat_ptr);
      /* GCOVR_EXCL_BR_START */
      if (HYPRE_GetError()) /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(matrix_filename);
         *precmat_ptr = NULL;
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "preconditioner matrix read failed from '%s'",
                            matrix_filename);
         return;
      }
   }

   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "preconditioner matrix setup end");
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemReadDofmap(MPI_Comm comm, const LS_args *args, IntArray **dofmap_ptr,
                                Stats *stats)
{
   int         ls_id = hypredrv_StatsGetLinearSystemID(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      LinearSystemLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap read begin");
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "dofmap");

   /* Destroy pre-existing dofmap */
   if (*dofmap_ptr)
   {
      hypredrv_IntArrayDestroy(dofmap_ptr);
   }

   if (args->sequence_filename[0] != '\0')
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "dofmap source: sequence file '%s'", args->sequence_filename);
      /* GCOVR_EXCL_START */
      if (!hypredrv_LSSeqReadDofmap(comm, args->sequence_filename, ls_id, dofmap_ptr))
      {
         hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "dofmap");
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "dofmap read failed from sequence source");
         return;
      }
      /* GCOVR_EXCL_STOP */
      hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "dofmap");
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap read end");
      return;
   }

   if (args->dofmap_filename[0] == '\0' && args->dofmap_basename[0] == '\0')
   {
      *dofmap_ptr = hypredrv_IntArrayCreate(0);
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap source: default empty");
   }
   else
   {
      char dofmap_filename[MAX_FILENAME_LENGTH] = {0};

      /* Set dofmap filename */
      if (args->dirname[0] != '\0')
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%.*s_%0*d/%.*s",
                  (int)strlen(args->dirname), args->dirname, (int)args->digits_suffix,
                  hypredrv_LinearSystemGetSuffix(args, ls_id),
                  (int)strlen(args->dofmap_filename), args->dofmap_filename);
      }
      else if (args->dofmap_filename[0] != '\0')
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%s", args->dofmap_filename);
      }
      else
      {
         snprintf(dofmap_filename, sizeof(dofmap_filename), "%.*s_%0*d",
                  (int)strlen(args->dofmap_basename), args->dofmap_basename,
                  (int)args->digits_suffix, hypredrv_LinearSystemGetSuffix(args, ls_id));
      }

      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap source: '%s'",
                         dofmap_filename);

      hypredrv_IntArrayParRead(comm, dofmap_filename, dofmap_ptr);
      if (hypredrv_ErrorCodeActive())
      {
         hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "dofmap");
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "dofmap read failed from '%s'", dofmap_filename);
         return;
      }
   }

   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "dofmap");
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap read end");

   /* TODO: Print how many dofs types we have (min, max, avg, sum) accross ranks
    */
}

/*-----------------------------------------------------------------------------
 * LinearSystemGetSolution
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemGetSolutionValues(HYPRE_IJVector sol, HYPRE_Complex **data_ptr)
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
hypredrv_LinearSystemGetRHSValues(HYPRE_IJVector rhs, HYPRE_Complex **data_ptr)
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
hypredrv_LinearSystemComputeVectorNorm(HYPRE_IJVector vec, const char *norm_type,
                                       double *norm)
{
   HYPRE_ParVector      par_vec = NULL;
   const hypre_Vector  *seq_vec = NULL;
   void                *obj     = NULL;
   const HYPRE_Complex *data    = NULL;
   HYPRE_Int            size    = 0;

   if (!vec || !norm)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      return;
   }

   HYPRE_IJVectorGetObject(vec, &obj);
   /* GCOVR_EXCL_START */
   if (!obj)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }
   /* GCOVR_EXCL_STOP */

   par_vec = (HYPRE_ParVector)obj;

   seq_vec = hypre_ParVectorLocalVector(par_vec);
   /* GCOVR_EXCL_START */
   if (!seq_vec)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   data = hypre_VectorData(seq_vec);
   if (!data)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }

   size = hypre_VectorSize(seq_vec);
   if (size < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *norm = -1.0;
      return;
   }
   /* GCOVR_EXCL_STOP */

   double   local_norm  = 0.0;
   double   global_norm = 0.0;
   MPI_Comm comm        = hypre_ParVectorComm(par_vec);

   /* GCOVR_EXCL_BR_START */
   if (!strcmp(norm_type, "L2") || !strcmp(norm_type, "l2")) /* GCOVR_EXCL_BR_STOP */
   {
      /* hypre_ParVectorInnerProd is GPU-aware - no migration needed */
      global_norm = (double)hypre_ParVectorInnerProd(par_vec, par_vec);
      *norm       = sqrt(global_norm);
   }
   else
   {
#if defined(HYPRE_USING_GPU)
      /* Manual loops require host-accessible data; save memory location to restore later
       */
      HYPRE_MemoryLocation orig_memloc = hypre_VectorMemoryLocation(seq_vec);
      if (orig_memloc != HYPRE_MEMORY_HOST)
      {
         HYPRE_IJVectorMigrate(vec, HYPRE_MEMORY_HOST);
         seq_vec = hypre_ParVectorLocalVector(par_vec);
         data    = hypre_VectorData(seq_vec);
      }
#endif
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
#if defined(HYPRE_USING_GPU)
      if (orig_memloc != HYPRE_MEMORY_HOST)
      {
         HYPRE_IJVectorMigrate(vec, orig_memloc);
      }
#endif
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemComputeErrorNorm
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemComputeErrorNorm(HYPRE_IJVector vec_xref, HYPRE_IJVector vec_x,
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
   IJVectorInitializeCompat(vec_e, hypre_IJVectorMemoryLocation(vec_x));
#else
   IJVectorInitializeCompat(vec_e, HYPRE_MEMORY_HOST);
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
   hypredrv_LinearSystemComputeVectorNorm(vec_e, norm_type, e_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_e);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemComputeResidualNorm(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
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
   IJVectorInitializeCompat(vec_r, hypre_IJVectorMemoryLocation(vec_b));
#else
   IJVectorInitializeCompat(vec_r, HYPRE_MEMORY_HOST);
#endif
   HYPRE_IJVectorGetObject(vec_r, &obj_r);
   par_r = (HYPRE_ParVector)obj_r;
   HYPRE_ParVectorCopy(par_b, par_r);

   /* Compute residual */
   HYPRE_ParCSRMatrixMatvec(neg_one, par_A, par_x, one, par_r);

   /* Compute residual norm */
   hypredrv_LinearSystemComputeVectorNorm(vec_r, norm_type, res_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_r);
}

/*-----------------------------------------------------------------------------
 * Print matrix/vector/dofmap with series directory logic
 *---------------------------------------------------------------------------*/
