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

/* Modes whose norm shrinks below this fraction of their input norm during
   orthogonalization are considered linearly dependent */
#define HYPREDRV_NULLSPACE_DEP_TOL 1.0e-12

#define HYPREDRV_HAVE_MEMORY_APIS (HYPREDRV_HYPRE_RELEASE_NUMBER >= 22000)

/* Limit diagnostics to a sensible number of physics blocks. This prevents a
 * malformed dofmap from turning level-3 logging into an unbounded allocation. */
enum
{
   HYPREDRV_BLOCK_NORM_MAX_LABELS      = 128,
   HYPREDRV_BLOCK_NORM_MAX_LABEL_SLOTS = 1048576
};

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

/*-----------------------------------------------------------------------------
 * DofLabelName
 *-----------------------------------------------------------------------------*/

static const char *
DofLabelName(const DofLabelMap *labels, int value)
{
   if (labels)
   {
      for (size_t i = 0; i < labels->size; i++)
      {
         if (labels->data[i].value == value)
         {
            return labels->data[i].name;
         }
      }
   }
   return NULL;
}

static void
DofLabelFormat(const DofLabelMap *labels, int value, char *buffer, size_t buffer_size)
{
   const char *name = DofLabelName(labels, value);
   if (name)
   {
      snprintf(buffer, buffer_size, "%s(id=%d)", name, value);
   }
   else
   {
      snprintf(buffer, buffer_size, "%d", value);
   }
}

/* Validate the global label metadata before diagnostic lookup allocations.
 * The metadata normally comes from IntArrayBuild, but library callers can
 * provide sparse or malformed arrays directly. */
static int
DofmapDiagnosticLabelsValid(const IntArray *dofmap, int *max_label_ptr)
{
   if (!dofmap || !dofmap->g_unique_data || dofmap->g_unique_size == 0 ||
       dofmap->g_unique_size > HYPREDRV_BLOCK_NORM_MAX_LABELS)
   {
      return 0;
   }

   int max_label = -1;
   for (size_t i = 0; i < dofmap->g_unique_size; i++)
   {
      int label = dofmap->g_unique_data[i];
      if (label < 0 || label >= HYPREDRV_BLOCK_NORM_MAX_LABEL_SLOTS)
      {
         return 0;
      }
      if (label > max_label)
      {
         max_label = label;
      }
      for (size_t j = 0; j < i; j++)
      {
         if (dofmap->g_unique_data[j] == label)
         {
            return 0;
         }
      }
   }

   *max_label_ptr = max_label;
   return 1;
}

static int
DofmapDiagnosticMetadataAgrees(MPI_Comm comm, const IntArray *dofmap, int max_label)
{
   enum
   {
      METADATA_SIZE = HYPREDRV_BLOCK_NORM_MAX_LABELS + 2
   };
   int metadata[METADATA_SIZE];
   int metadata_min[METADATA_SIZE];
   int metadata_max[METADATA_SIZE];

   for (int i = 0; i < METADATA_SIZE; i++)
   {
      metadata[i] = INT_MIN;
   }
   metadata[0] = (int)dofmap->g_unique_size;
   metadata[1] = max_label;
   for (size_t i = 0; i < dofmap->g_unique_size; i++)
   {
      metadata[i + 2] = dofmap->g_unique_data[i];
   }

   MPI_Allreduce(metadata, metadata_min, METADATA_SIZE, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(metadata, metadata_max, METADATA_SIZE, MPI_INT, MPI_MAX, comm);
   for (int i = 0; i < METADATA_SIZE; i++)
   {
      if (metadata_min[i] != metadata_max[i])
      {
         return 0;
      }
   }
   return 1;
}

static void
DofmapDiagnosticMapFill(const IntArray *dofmap, int num_label_slots, int *label_to_pos,
                        int *block_labels)
{
   for (int i = 0; i < num_label_slots; i++)
   {
      label_to_pos[i] = -1;
   }
   for (size_t i = 0; i < dofmap->g_unique_size; i++)
   {
      int label           = dofmap->g_unique_data[i];
      label_to_pos[label] = (int)i;
      if (block_labels)
      {
         block_labels[i] = label;
      }
   }
}

/*-----------------------------------------------------------------------------
 * GetCSRHostView
 *-----------------------------------------------------------------------------*/

static int
GetCSRHostView(hypre_CSRMatrix *matrix, HYPRE_Int **row_ptr, HYPRE_Int **col_ind,
               HYPRE_Complex **values, int *owns_copy)
{
   HYPRE_Int            num_rows        = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int            num_nnz         = hypre_CSRMatrixNumNonzeros(matrix);
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

   *owns_copy = 0;
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST)
   {
      *row_ptr = hypre_CSRMatrixI(matrix);
      *col_ind = hypre_CSRMatrixJ(matrix);
      *values  = hypre_CSRMatrixData(matrix);
      return *row_ptr && (num_nnz == 0 || (*col_ind && *values));
   }

   *row_ptr = hypre_TAlloc(HYPRE_Int, num_rows + 1, HYPRE_MEMORY_HOST);
   *col_ind = hypre_TAlloc(HYPRE_Int, num_nnz, HYPRE_MEMORY_HOST);
   *values  = hypre_TAlloc(HYPRE_Complex, num_nnz, HYPRE_MEMORY_HOST);
   if (!*row_ptr || (num_nnz > 0 && (!*col_ind || !*values)))
   {
      hypre_TFree(*row_ptr, HYPRE_MEMORY_HOST);
      hypre_TFree(*col_ind, HYPRE_MEMORY_HOST);
      hypre_TFree(*values, HYPRE_MEMORY_HOST);
      *row_ptr = NULL;
      *col_ind = NULL;
      *values  = NULL;
      return 0;
   }

   *owns_copy = 1;
   hypre_TMemcpy(*row_ptr, hypre_CSRMatrixI(matrix), HYPRE_Int, num_rows + 1,
                 HYPRE_MEMORY_HOST, memory_location);
   if (num_nnz > 0)
   {
      hypre_TMemcpy(*col_ind, hypre_CSRMatrixJ(matrix), HYPRE_Int, num_nnz,
                    HYPRE_MEMORY_HOST, memory_location);
      hypre_TMemcpy(*values, hypre_CSRMatrixData(matrix), HYPRE_Complex, num_nnz,
                    HYPRE_MEMORY_HOST, memory_location);
   }
   return 1;
}

/*-----------------------------------------------------------------------------
 * AccumulateBlockNorms
 *-----------------------------------------------------------------------------*/

static HYPRE_BigInt
AccumulateBlockNorms(HYPRE_Int num_rows, const HYPRE_Int *row_ptr,
                     const HYPRE_Int *col_ind, const HYPRE_Complex *values,
                     const int *row_labels, const int *col_labels,
                     const int *label_to_pos, int num_label_slots, int num_blocks,
                     double *norm_sq, double *sum, double *abs_sum, long long *block_nnz,
                     long long *positive_nnz, long long *negative_nnz,
                     long long *zero_nnz)
{
   HYPRE_BigInt ignored = 0;
   for (HYPRE_Int row = 0; row < num_rows; row++)
   {
      int row_label = row_labels[row];
      int row_pos =
         (row_label >= 0 && row_label < num_label_slots) ? label_to_pos[row_label] : -1;
      for (HYPRE_Int entry = row_ptr[row]; entry < row_ptr[row + 1]; entry++)
      {
         int col_label = col_labels[col_ind[entry]];
         int col_pos   = (col_label >= 0 && col_label < num_label_slots)
                            ? label_to_pos[col_label]
                            : -1;
         if (row_pos < 0 || col_pos < 0)
         {
            ignored++;
            continue;
         }

         size_t index     = ((size_t)row_pos * (size_t)num_blocks) + (size_t)col_pos;
         double magnitude = (double)hypre_cabs(values[entry]);
         double value     = (double)hypre_creal(values[entry]);
         norm_sq[index] += magnitude * magnitude;
         sum[index] += value;
         abs_sum[index] += magnitude;
         block_nnz[index]++;
         if (value > 0.0)
         {
            positive_nnz[index]++;
         }
         else if (value < 0.0)
         {
            negative_nnz[index]++;
         }
         else
         {
            zero_nnz[index]++;
         }
      }
   }
   return ignored;
}

static const FieldOffsetMap ls_field_offset_map[] = {
   FIELD_OFFSET_MAP_ENTRY(LS_args, dirname, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, sequence_filename, hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_sequence_filename,
                          hypredrv_FieldTypeStringSet),
   FIELD_OFFSET_MAP_ENTRY(LS_args, precmat_sequence_system_id, hypredrv_FieldTypeIntSet),
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
   args->dirname[0]                   = '\0';
   args->sequence_filename[0]         = '\0';
   args->precmat_sequence_filename[0] = '\0';
   args->precmat_sequence_system_id   = -1;
   args->matrix_filename[0]           = '\0';
   args->matrix_basename[0]           = '\0';
   args->precmat_filename[0]          = '\0';
   args->precmat_basename[0]          = '\0';
   args->rhs_filename[0]              = '\0';
   args->rhs_basename[0]              = '\0';
   args->x0_filename[0]               = '\0';
   args->xref_filename[0]             = '\0';
   args->xref_basename[0]             = '\0';
   args->timestep_filename[0]         = '\0';
   args->sol_filename[0]              = '\0';
   args->dofmap_filename[0]           = '\0';
   args->dofmap_basename[0]           = '\0';
   args->digits_suffix                = 5;
   args->init_suffix                  = -1;
   args->last_suffix                  = -1;
   args->set_suffix                   = NULL;
   args->init_guess_mode              = 0;
   args->rhs_mode                     = 2;
   args->type                         = 1;
   args->num_systems                  = 1;
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
   HYPRE_IJVectorInitialize_v2(*vec_nn_ptr, HYPRE_MEMORY_HOST);

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
      if (!indices || (values == NULL && !zeros))
      {
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate near-null-space index/value buffers");
         free(indices);
         free(zeros);
         return;
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
 * hypredrv_LinearSystemSetNullSpace
 *
 * Orthonormalize the input modes (modified Gram-Schmidt) and store them with
 * the same multi-component vector builder used for the near null space modes
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetNullSpace(MPI_Comm comm, HYPRE_IJMatrix mat, int num_entries,
                                  int num_components, const HYPRE_Complex *values,
                                  HYPRE_IJVector *vec_ns_ptr)
{
   size_t         total = (size_t)num_entries * (size_t)num_components;
   HYPRE_Complex *modes = NULL;

   /* values may be NULL on ranks that own no entries; num_entries and
      num_components must be consistent across all ranks of comm */
   if (num_components < 1 || num_entries < 0 || (!values && num_entries > 0))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid null space input: need num_components >= 1, "
                           "num_entries >= 0, and non-NULL values when num_entries > 0");
      return;
   }

   modes = (HYPRE_Complex *)malloc(total * sizeof(HYPRE_Complex));
   /* GCOVR_EXCL_START */
   if (num_entries > 0 && !modes)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate null space modes buffer");
      return;
   }
   /* GCOVR_EXCL_STOP */
   /* Guard on num_entries (not total) so it is provable that values is
    * non-NULL here: NULL values is only accepted alongside num_entries == 0. */
   if (num_entries > 0)
   {
      memcpy(modes, values, total * sizeof(HYPRE_Complex));
   }

   /* Modified Gram-Schmidt on the component blocks (component-major layout) */
   for (int k = 0; k < num_components; k++)
   {
      HYPRE_Complex *zk = modes + ((size_t)k * (size_t)num_entries);
      double         dot_local, dot, norm_orig;

      /* Input norm of the mode, for the relative dependence check below */
      dot_local = 0.0;
      for (int i = 0; i < num_entries; i++)
      {
         dot_local += (double)(zk[i] * zk[i]);
      }
      MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      norm_orig = sqrt(dot);

      for (int j = 0; j < k; j++)
      {
         const HYPRE_Complex *zj = modes + ((size_t)j * (size_t)num_entries);

         dot_local = 0.0;
         for (int i = 0; i < num_entries; i++)
         {
            dot_local += (double)(zk[i] * zj[i]);
         }
         MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
         for (int i = 0; i < num_entries; i++)
         {
            zk[i] -= (HYPRE_Complex)dot * zj[i];
         }
      }

      dot_local = 0.0;
      for (int i = 0; i < num_entries; i++)
      {
         dot_local += (double)(zk[i] * zk[i]);
      }
      MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      dot = sqrt(dot);
      if (dot <= HYPREDRV_NULLSPACE_DEP_TOL * norm_orig)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Null space modes must be linearly independent "
                              "(mode %d has relative norm %e after orthogonalization)",
                              k, (norm_orig > 0.0) ? dot / norm_orig : 0.0);
         free(modes);
         return;
      }
      for (int i = 0; i < num_entries; i++)
      {
         zk[i] /= (HYPRE_Complex)dot;
      }
   }

   /* Store via the same builder used for the near null space modes. The modes
      are kept host-resident (NULL args skips the device migration): their only
      consumer is the host-side projection in
      hypredrv_LinearSystemProjectOutNullSpace(). */
   hypredrv_LinearSystemSetNearNullSpace(comm, NULL, mat, num_entries, num_components,
                                         modes, vec_ns_ptr);

   free(modes);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemProjectOutNullSpace
 *
 * Remove the (orthonormalized) null space components from a vector, fixing
 * the gauge of solutions that are defined up to a null space contribution
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemProjectOutNullSpace(HYPRE_IJVector vec_ns, int num_ns,
                                         HYPRE_IJVector vec)
{
   HYPRE_BigInt   jlower = 0, jupper = 0, xlower = 0, xupper = 0;
   HYPRE_BigInt  *indices = NULL;
   HYPRE_Complex *xbuf = NULL, *zbuf = NULL;
   double        *dots_local = NULL, *dots = NULL;
   MPI_Comm       comm;
   int            num_entries, mismatch_local, mismatch;

   if (!vec_ns || num_ns < 1 || !vec)
   {
      return;
   }

   comm = hypre_IJVectorComm((hypre_IJVector *)vec_ns);
   HYPRE_IJVectorGetLocalRange(vec_ns, &jlower, &jupper);
   HYPRE_IJVectorGetLocalRange(vec, &xlower, &xupper);
   num_entries = (int)(jupper - jlower + 1);

   /* The modes were built for the partitioning of the matrix that was set when
      HYPREDRV_LinearSystemSetNullSpace() was called; refuse to project onto a
      vector with a different distribution. The check is collective so that all
      ranks agree before entering the reductions below. */
   mismatch_local = (xlower != jlower || xupper != jupper);
   MPI_Allreduce(&mismatch_local, &mismatch, 1, MPI_INT, MPI_MAX, comm);
   if (mismatch)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Null space modes are incompatible with the current linear "
                           "system; call HYPREDRV_LinearSystemSetNullSpace() again (or "
                           "clear the modes with num_components = 0) after changing the "
                           "system size or distribution");
      return;
   }

#if defined(HYPRE_USING_GPU)
   /* The dot/axpy loops below need host-accessible data; vec_ns is built
      host-resident, but the solution may live on device */
   void *obj = NULL;
   HYPRE_IJVectorGetObject(vec, &obj);
   HYPRE_MemoryLocation orig_memloc =
      hypre_VectorMemoryLocation(hypre_ParVectorLocalVector((HYPRE_ParVector)obj));
   if (orig_memloc != HYPRE_MEMORY_HOST)
   {
      HYPRE_IJVectorMigrate(vec, HYPRE_MEMORY_HOST);
   }
#endif

   xbuf       = (HYPRE_Complex *)malloc((size_t)num_entries * sizeof(HYPRE_Complex));
   zbuf       = (HYPRE_Complex *)malloc((size_t)num_entries * sizeof(HYPRE_Complex));
   indices    = (HYPRE_BigInt *)malloc((size_t)num_entries * sizeof(HYPRE_BigInt));
   dots_local = (double *)calloc((size_t)num_ns, sizeof(double));
   dots       = (double *)calloc((size_t)num_ns, sizeof(double));

   /* Allocation must be agreed collectively: this routine performs an MPI_Allreduce
    * below, so a per-rank early return on failure would deadlock the others. */
   int alloc_ok = (xbuf && zbuf && indices && dots_local && dots) ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &alloc_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!alloc_ok)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate null-space projection buffers");
      free(xbuf);
      free(zbuf);
      free(indices);
      free(dots_local);
      free(dots);
      return;
   }
   for (int i = 0; i < num_entries; i++)
   {
      indices[i] = jlower + (HYPRE_BigInt)i;
   }

   HYPRE_IJVectorGetValues(vec, num_entries, NULL, xbuf);
   for (int k = 0; k < num_ns; k++)
   {
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      HYPRE_IJVectorSetComponent(vec_ns, k);
#endif
      HYPRE_IJVectorGetValues(vec_ns, num_entries, NULL, zbuf);
      for (int i = 0; i < num_entries; i++)
      {
         dots_local[k] += (double)(xbuf[i] * zbuf[i]);
      }
   }
   MPI_Allreduce(dots_local, dots, num_ns, MPI_DOUBLE, MPI_SUM, comm);
   for (int k = 0; k < num_ns; k++)
   {
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      HYPRE_IJVectorSetComponent(vec_ns, k);
#endif
      HYPRE_IJVectorGetValues(vec_ns, num_entries, NULL, zbuf);
      for (int i = 0; i < num_entries; i++)
      {
         xbuf[i] -= (HYPRE_Complex)dots[k] * zbuf[i];
      }
   }
   HYPRE_IJVectorSetValues(vec, num_entries, indices, xbuf);
   HYPRE_IJVectorAssemble(vec);

#if defined(HYPRE_USING_GPU)
   if (orig_memloc != HYPRE_MEMORY_HOST)
   {
      HYPRE_IJVectorMigrate(vec, orig_memloc);
   }
#endif

   free(xbuf);
   free(zbuf);
   free(indices);
   free(dots_local);
   free(dots);
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

   if (args->precmat_sequence_filename[0] != '\0' &&
       (args->precmat_filename[0] != '\0' || args->precmat_basename[0] != '\0'))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system: precmat_sequence_filename cannot be used with "
                           "precmat_filename or precmat_basename");
   }
   if (args->precmat_sequence_system_id < -1)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "linear_system: precmat_sequence_system_id must be -1 or nonnegative");
   }
   if (args->precmat_sequence_system_id >= 0 &&
       args->precmat_sequence_filename[0] == '\0')
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("linear_system: precmat_sequence_system_id requires "
                           "precmat_sequence_filename");
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
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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
   HYPRE_Int    *ncols_per_row = NULL;
   HYPRE_BigInt *row_ids       = NULL;

#define HYPREDRV_CSR_HYPRE_CALL(call)                                      \
   do                                                                      \
   {                                                                       \
      HYPRE_Int hypre_ierr = (call);                                       \
      if (hypre_ierr != 0)                                                 \
      {                                                                    \
         char hypre_err_msg[HYPRE_MAX_MSG_LEN];                            \
         HYPRE_DescribeError(hypre_ierr, hypre_err_msg);                   \
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);                      \
         hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: HYPRE call failed: %s", \
                              hypre_err_msg);                              \
         goto fail;                                                        \
      }                                                                    \
   } while (0)

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
   if ((HYPRE_BigInt)((HYPRE_Int)nrows_big) != nrows_big)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "BuildMatrixFromCSR: local row count (%lld) is out of HYPRE_Int range",
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
   if ((HYPRE_BigInt)((HYPRE_Int)nnz_big) != nnz_big)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "BuildMatrixFromCSR: local nonzero count (%lld) exceeds HYPRE_Int range",
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
   HYPREDRV_CSR_HYPRE_CALL(
      HYPRE_IJMatrixCreate(comm, row_start, row_end, row_start, row_end, mat_ptr));
   HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixSetObjectType(*mat_ptr, HYPRE_PARCSR));
   HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(*mat_ptr, memory_location));

   if (nrows == 0)
   {
      HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixAssemble(*mat_ptr));
      return hypredrv_ErrorCodeGet();
   }
   if (nnz == 0)
   {
      HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixAssemble(*mat_ptr));
      return hypredrv_ErrorCodeGet();
   }

   /* HYPRE_IJMatrixSetValues requires per-row counts and row ids on every call.
    * These transient O(nrows) scratch arrays stay on the host even when matrix
    * values are initialized for a device memory location. */
   ncols_per_row = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   row_ids       = hypre_TAlloc(HYPRE_BigInt, nrows, HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < nrows; i++)
   {
      HYPRE_BigInt row_nnz = indptr[i + 1] - indptr[i];
      if (row_nnz < 0)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("BuildMatrixFromCSR: indptr is not monotonically "
                              "non-decreasing at row %lld",
                              (long long)i);
         goto fail;
      }
      if ((HYPRE_BigInt)((HYPRE_Int)row_nnz) != row_nnz)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "BuildMatrixFromCSR: row %lld nonzero count (%lld) exceeds HYPRE_Int range",
            (long long)i, (long long)row_nnz);
         goto fail;
      }
      ncols_per_row[i] = (HYPRE_Int)row_nnz;
      row_ids[i]       = row_start + (HYPRE_BigInt)i;
   }

#ifdef HYPREDRV_USING_DEBUG
   for (HYPRE_Int k = 0; k < nnz; k++)
   {
      long long    col_index = (long long)indptr[0] + (long long)k;
      HYPRE_BigInt col       = col_indices[indptr[0] + (HYPRE_BigInt)k];
      if (col < 0)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "BuildMatrixFromCSR: col_indices[%lld] (%lld) must be nonnegative", col_index,
            (long long)col);
         goto fail;
      }
   }
#endif

#if defined(HYPRE_USING_GPU)
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
   {
      /* hypre's device IJ assembly path requires all SetValues inputs to live in
       * device memory. Stage the caller's host CSR (and the transient per-row
       * arrays) on the device once and assemble the matrix on the GPU. For large
       * matrices this is dramatically faster than assembling on the host and then
       * migrating the ParCSR (host assembly of millions of nonzeros dominates). */
      HYPRE_Int     *d_ncols = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt  *d_rows  = hypre_TAlloc(HYPRE_BigInt, nrows, HYPRE_MEMORY_DEVICE);
      HYPRE_BigInt  *d_cols  = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_Complex *d_data  = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(d_ncols, ncols_per_row, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(d_rows, row_ids, HYPRE_BigInt, nrows, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      hypre_TMemcpy(d_cols, col_indices + indptr[0], HYPRE_BigInt, nnz,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(d_data, data + indptr[0], HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);

      HYPRE_Int ierr =
         HYPRE_IJMatrixSetValues(*mat_ptr, nrows, d_ncols, d_rows, d_cols, d_data);
      if (!ierr)
      {
         ierr = HYPRE_IJMatrixAssemble(*mat_ptr);
      }

      hypre_TFree(d_ncols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_rows, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_cols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_data, HYPRE_MEMORY_DEVICE);
      HYPREDRV_CSR_HYPRE_CALL(ierr);
   }
   else
#endif
   {
      HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixSetValues(*mat_ptr, nrows, ncols_per_row,
                                                      row_ids, col_indices + indptr[0],
                                                      data + indptr[0]));
      HYPREDRV_CSR_HYPRE_CALL(HYPRE_IJMatrixAssemble(*mat_ptr));
   }

   hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
   hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
#undef HYPREDRV_CSR_HYPRE_CALL
   return hypredrv_ErrorCodeGet();

fail:
   hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
   hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
   if (*mat_ptr)
   {
      HYPRE_IJMatrixDestroy(*mat_ptr);
      *mat_ptr = NULL;
   }
#undef HYPREDRV_CSR_HYPRE_CALL
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
#define HYPREDRV_RHS_HYPRE_CALL(call)                                     \
   do                                                                     \
   {                                                                      \
      HYPRE_Int hypre_ierr = (call);                                      \
      if (hypre_ierr != 0)                                                \
      {                                                                   \
         char hypre_err_msg[HYPRE_MAX_MSG_LEN];                           \
         HYPRE_DescribeError(hypre_ierr, hypre_err_msg);                  \
         hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);                     \
         hypredrv_ErrorMsgAdd("BuildRHSFromArray: HYPRE call failed: %s", \
                              hypre_err_msg);                             \
         goto fail;                                                       \
      }                                                                   \
   } while (0)

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
      hypredrv_ErrorMsgAdd(
         "BuildRHSFromArray: local row count (%lld) exceeds HYPRE_Int range",
         (long long)nrows_big);
      return hypredrv_ErrorCodeGet();
   }

   HYPRE_Int nrows = (HYPRE_Int)nrows_big;
   if (!values)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("BuildRHSFromArray: values must be non-NULL");
      return hypredrv_ErrorCodeGet();
   }

   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }

   HYPREDRV_RHS_HYPRE_CALL(HYPRE_IJVectorCreate(comm, row_start, row_end, rhs_ptr));
   HYPREDRV_RHS_HYPRE_CALL(HYPRE_IJVectorSetObjectType(*rhs_ptr, HYPRE_PARCSR));
   HYPREDRV_RHS_HYPRE_CALL(HYPRE_IJVectorInitialize_v2(*rhs_ptr, memory_location));

   if (nrows > 0)
   {
      HYPREDRV_RHS_HYPRE_CALL(HYPRE_IJVectorSetValues(*rhs_ptr, nrows, NULL, values));
   }
   HYPREDRV_RHS_HYPRE_CALL(HYPRE_IJVectorAssemble(*rhs_ptr));

#undef HYPREDRV_RHS_HYPRE_CALL
   return hypredrv_ErrorCodeGet();

fail:
   if (*rhs_ptr)
   {
      HYPRE_IJVectorDestroy(*rhs_ptr);
      *rhs_ptr = NULL;
   }
#undef HYPREDRV_RHS_HYPRE_CALL
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
   HYPRE_IJVectorInitialize_v2(*rhs_ptr, memory_location);

   HYPRE_BigInt local_size = iupper - ilower + 1;
   /* Decide the overflow condition collectively: a per-rank early return here would
    * leave peers blocked in the MPI_Gather/MPI_Scatterv collectives below. */
   int local_overflow  = (local_size > (HYPRE_BigInt)INT_MAX) ? 1 : 0;
   int global_overflow = local_overflow;
   MPI_Allreduce(&local_overflow, &global_overflow, 1, MPI_INT, MPI_MAX, comm);
   if (global_overflow)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Local RHS size (%lld) exceeds MPI int range on some rank",
                           (long long)local_size);
      if (myid == 0 && all_values)
      {
         hypre_TFree(all_values, HYPRE_MEMORY_HOST);
      }
      return 0;
   }
   int my_local_size =
      (int)local_size; /* NOLINT(cppcoreguidelines-narrowing-conversions) */
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
   /* Use the MPI datatype that matches HYPRE_Complex; hard-coding MPI_DOUBLE would
    * corrupt data on single-precision or complex hypre builds. */
   MPI_Scatterv(all_values, counts, displs, HYPRE_MPI_COMPLEX, local_values,
                my_local_size, HYPRE_MPI_COMPLEX, 0, comm);

   HYPRE_Int local_size_hypre =
      (HYPRE_Int)local_size; /* NOLINT(cppcoreguidelines-narrowing-conversions) */
   HYPRE_IJVectorSetValues(*rhs_ptr, local_size_hypre, NULL, local_values);
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
   HYPRE_IJVectorInitialize_v2(*rhs_ptr, memory_location);

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
         HYPRE_IJVectorInitialize_v2(xref, memory_location);

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
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));

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
   HYPRE_IJVectorInitialize_v2(*x_ptr, memloc);
}

void
hypredrv_LinearSystemSetInitialGuess(MPI_Comm comm, LS_args *args, HYPRE_IJMatrix mat,
                                     HYPRE_IJVector rhs, HYPRE_IJVector *x0_ptr,
                                     HYPRE_IJVector *x_ptr, const Stats *stats)
{
   (void)mat;
   int         ls_id = hypredrv_StatsGetLinearSystemID(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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

   /* The working solution (*x_ptr) is recreated only at the end of this
    * function: init_guess_mode "previous" reads the previous solve's values
    * from it while building x0. */

   if (args->x0_filename[0] == '\0')
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "initial guess source: generated mode=%d",
                         (int)args->init_guess_mode);
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
         {
            /* Use solution from previous linear solve */
            HYPRE_BigInt xlower = 0, xupper = 0;

            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                               "initial guess mode: previous");
            if (*x_ptr)
            {
               HYPRE_IJVectorGetLocalRange(*x_ptr, &xlower, &xupper);
            }
            if (*x_ptr && xlower == jlower && xupper == jupper)
            {
               HYPRE_IJVectorGetObject(*x_ptr, &obj);
               par_x = (HYPRE_ParVector)obj;

               HYPRE_ParVectorCopy(par_x, par_x0);
            }
            else
            {
               HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                                  "no compatible previous solution; using zeros");
            }
            break;
         }

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

   /* Recreate the working solution now that x0 has captured any previous
    * solve's values. */
   hypredrv_LinearSystemCreateWorkingSolution(comm, args, rhs, x_ptr);
   /* GCOVR_EXCL_START */
   if (hypredrv_ErrorCodeActive())
   {
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                         "initial guess setup failed: could not create working solution");
      return;
   }
   /* GCOVR_EXCL_STOP */

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
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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
   /* Reports the current system rather than the next one; clamp so a NULL stats
    * logs system 0 instead of -1. */
   int             ls_id = hypredrv_StatsGetLinearSystemID(stats);
   char            log_name_buf[32];
   const char     *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));

   ls_id = (ls_id < 0) ? 0 : ls_id;

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

   /* Skip the copy when x0 and x alias the same vector or the same data
    * (e.g. library-mode callers that pass one vector as both initial guess
    * and solution). */
   if (par_x0 != par_x &&
       hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)par_x0)) !=
          hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)par_x)))
   {
      HYPRE_ParVectorCopy(par_x0, par_x);
   }

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
   /* A rank may legitimately own no rows, in which case its local dofmap is empty.
    * Such a rank must still tag its vector: hypre reductions over tagged vectors
    * (e.g. hypre_ParVectorInnerProdTagged) exchange num_tags + 1 values, so leaving
    * one rank untagged makes it enter the collective with a different length than
    * its peers, which deadlocks the solve and corrupts unrelated reductions.
    * num_tags is derived below from the globally reduced label set, which every
    * rank holds, so it is consistent even where the local dofmap is empty. */
   /* GCOVR_EXCL_BR_START */
   if (!vec || !dofmap || !dofmap->data) /* GCOVR_EXCL_BR_STOP */
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

   /* Convert the application's fixed-width int labels to HYPRE_Int. This is
    * required for bigint builds, where HYPRE_Int is 64-bit. Mode 2 transfers
    * the allocation to the vector and also lets repeated tagging replace the
    * vector's previous owned array safely. */
   /* GCOVR_EXCL_START */
   HYPRE_MemoryLocation vec_memloc = HYPRE_MEMORY_HOST;
#if defined(HYPRE_USING_GPU) && HYPREDRV_HAVE_MEMORY_APIS
   vec_memloc = hypre_IJVectorMemoryLocation((hypre_IJVector *)vec);
#endif
   HYPRE_Int *host_tags = hypre_TAlloc(HYPRE_Int, dofmap->size, HYPRE_MEMORY_HOST);
   for (size_t i = 0; i < dofmap->size; i++)
   {
      host_tags[i] = (HYPRE_Int)dofmap->data[i];
   }

   HYPRE_Int *tags = host_tags;
   if (hypre_GetActualMemLocation(vec_memloc) !=
       hypre_GetActualMemLocation(HYPRE_MEMORY_HOST))
   {
      tags = hypre_TAlloc(HYPRE_Int, dofmap->size, vec_memloc);
      hypre_TMemcpy(tags, host_tags, HYPRE_Int, dofmap->size, vec_memloc,
                    HYPRE_MEMORY_HOST);
      hypre_TFree(host_tags, HYPRE_MEMORY_HOST);
   }
   HYPRE_IJVectorSetTags(vec, 2, num_tags, tags);
   /* GCOVR_EXCL_STOP */
#else
   (void)vec;
   (void)dofmap;
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemLogBlockFrobenius
 *
 * Compute ||A_ij||_F for every pair of dofmap labels. Off-process column labels
 * are exchanged with the matrix communication package, so every ParCSR entry is
 * included exactly once. This intentionally runs only at log level 3 or above.
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemLogBlockFrobenius(MPI_Comm comm, HYPRE_IJMatrix matrix,
                                       const IntArray    *dofmap,
                                       const DofLabelMap *dof_labels,
                                       const char *log_object_name, int ls_id)
{
   int diagnostics_enabled = hypredrv_LogEnabled(3) ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &diagnostics_enabled, 1, MPI_INT, MPI_MIN, comm);
   if (!diagnostics_enabled)
   {
      return;
   }

   void                   *object            = NULL;
   hypre_ParCSRMatrix     *par_matrix        = NULL;
   hypre_CSRMatrix        *diag              = NULL;
   hypre_CSRMatrix        *offd              = NULL;
   hypre_ParCSRCommPkg    *comm_pkg          = NULL;
   hypre_ParCSRCommHandle *comm_handle       = NULL;
   HYPRE_Int              *diag_i            = NULL;
   HYPRE_Int              *diag_j            = NULL;
   HYPRE_Complex          *diag_a            = NULL;
   HYPRE_Int              *offd_i            = NULL;
   HYPRE_Int              *offd_j            = NULL;
   HYPRE_Complex          *offd_a            = NULL;
   int                     owns_diag_copy    = 0;
   int                     owns_offd_copy    = 0;
   HYPRE_Int              *send_labels       = NULL;
   HYPRE_Int              *offd_labels       = NULL;
   int                    *offd_block_labels = NULL;
   int                    *block_labels      = NULL;
   int                    *label_to_pos      = NULL;
   double                 *local_stats       = NULL;
   double                 *global_stats      = NULL;
   long long              *local_counts      = NULL;
   long long              *global_counts     = NULL;
   char                   *line              = NULL;

   int local_valid = matrix && dofmap;
   if (local_valid)
   {
      HYPRE_IJMatrixGetObject(matrix, &object);
      local_valid = object != NULL;
   }
   int global_valid = 0;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: matrix or dofmap missing");
      return;
   }

   par_matrix = (hypre_ParCSRMatrix *)object;
   diag       = hypre_ParCSRMatrixDiag(par_matrix);
   offd       = hypre_ParCSRMatrixOffd(par_matrix);

   HYPRE_Int num_rows      = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int num_cols_diag = hypre_CSRMatrixNumCols(diag);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   local_valid             = dofmap->size == (size_t)num_rows &&
                 dofmap->size == (size_t)num_cols_diag &&
                 (num_rows == 0 || dofmap->data != NULL) &&
                 dofmap->g_unique_data != NULL && dofmap->g_unique_size > 0 &&
                 dofmap->g_unique_size <= HYPREDRV_BLOCK_NORM_MAX_LABELS;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(
         3, comm, log_object_name, ls_id,
         "block Frobenius diagnostics skipped: incompatible matrix/dofmap layout");
      return;
   }

   int max_label = -1;
   local_valid   = DofmapDiagnosticLabelsValid(dofmap, &max_label);
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: invalid dofmap labels");
      return;
   }

   int num_blocks      = (int)dofmap->g_unique_size;
   int num_label_slots = max_label + 1;
   if (!DofmapDiagnosticMetadataAgrees(comm, dofmap, max_label))
   {
      HYPREDRV_LOG_COMMF(
         3, comm, log_object_name, ls_id,
         "block Frobenius diagnostics skipped: inconsistent dofmap metadata");
      return;
   }

   size_t num_block_pairs = (size_t)num_blocks * (size_t)num_blocks;
   block_labels           = hypre_TAlloc(int, num_blocks, HYPRE_MEMORY_HOST);
   label_to_pos           = hypre_TAlloc(int, num_label_slots, HYPRE_MEMORY_HOST);
   local_stats            = hypre_CTAlloc(double, 3 * num_block_pairs, HYPRE_MEMORY_HOST);
   global_stats           = hypre_CTAlloc(double, 3 * num_block_pairs, HYPRE_MEMORY_HOST);
   local_counts  = hypre_CTAlloc(long long, (4 * num_block_pairs) + 1, HYPRE_MEMORY_HOST);
   global_counts = hypre_CTAlloc(long long, (4 * num_block_pairs) + 1, HYPRE_MEMORY_HOST);
   local_valid   = block_labels && label_to_pos && local_stats && global_stats &&
                 local_counts && global_counts;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: allocation failed");
      goto cleanup;
   }

   double    *local_norm_sq   = local_stats;
   double    *local_sum       = local_stats + num_block_pairs;
   double    *local_abs_sum   = local_stats + (2 * num_block_pairs);
   double    *global_norm_sq  = global_stats;
   double    *global_sum      = global_stats + num_block_pairs;
   double    *global_abs_sum  = global_stats + (2 * num_block_pairs);
   long long *local_nnz       = local_counts;
   long long *local_positive  = local_counts + num_block_pairs;
   long long *local_negative  = local_counts + (2 * num_block_pairs);
   long long *local_zero      = local_counts + (3 * num_block_pairs);
   long long *global_nnz      = global_counts;
   long long *global_positive = global_counts + num_block_pairs;
   long long *global_negative = global_counts + (2 * num_block_pairs);
   long long *global_zero     = global_counts + (3 * num_block_pairs);

   DofmapDiagnosticMapFill(dofmap, num_label_slots, label_to_pos, block_labels);

   if (!hypre_ParCSRMatrixCommPkg(par_matrix))
   {
      hypre_MatvecCommPkgCreate(par_matrix);
   }
   comm_pkg    = hypre_ParCSRMatrixCommPkg(par_matrix);
   local_valid = comm_pkg || num_cols_offd == 0;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: matrix halo unavailable");
      goto cleanup;
   }

   HYPRE_Int send_size = 0;
   if (comm_pkg)
   {
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_size           = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_labels         = hypre_TAlloc(HYPRE_Int, send_size, HYPRE_MEMORY_HOST);
      offd_labels         = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      offd_block_labels   = hypre_TAlloc(int, num_cols_offd, HYPRE_MEMORY_HOST);
   }
   local_valid = (send_size == 0 || send_labels) &&
                 (num_cols_offd == 0 || (offd_labels && offd_block_labels));
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: halo allocation failed");
      goto cleanup;
   }

   if (comm_pkg)
   {
      for (HYPRE_Int i = 0; i < send_size; i++)
      {
         HYPRE_Int local_col = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
         send_labels[i]      = (HYPRE_Int)dofmap->data[local_col];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_labels, offd_labels);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      for (HYPRE_Int i = 0; i < num_cols_offd; i++)
      {
         offd_block_labels[i] = (int)offd_labels[i];
      }
   }

   local_valid = GetCSRHostView(diag, &diag_i, &diag_j, &diag_a, &owns_diag_copy) &&
                 GetCSRHostView(offd, &offd_i, &offd_j, &offd_a, &owns_offd_copy);
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block Frobenius diagnostics skipped: CSR host copy failed");
      goto cleanup;
   }

   HYPRE_BigInt local_ignored = AccumulateBlockNorms(
      num_rows, diag_i, diag_j, diag_a, dofmap->data, dofmap->data, label_to_pos,
      num_label_slots, num_blocks, local_norm_sq, local_sum, local_abs_sum, local_nnz,
      local_positive, local_negative, local_zero);
   if (num_cols_offd > 0)
   {
      local_ignored += AccumulateBlockNorms(
         num_rows, offd_i, offd_j, offd_a, dofmap->data, offd_block_labels, label_to_pos,
         num_label_slots, num_blocks, local_norm_sq, local_sum, local_abs_sum, local_nnz,
         local_positive, local_negative, local_zero);
   }

   local_counts[4 * num_block_pairs] = (long long)local_ignored;
   MPI_Allreduce(local_stats, global_stats, (int)(3 * num_block_pairs), MPI_DOUBLE,
                 MPI_SUM, comm);
   MPI_Allreduce(local_counts, global_counts, (int)((4 * num_block_pairs) + 1),
                 MPI_LONG_LONG, MPI_SUM, comm);
   long long global_ignored = global_counts[4 * num_block_pairs];

   double matrix_norm_sq = 0.0;
   for (size_t i = 0; i < num_block_pairs; i++)
   {
      matrix_norm_sq += global_norm_sq[i];
      global_norm_sq[i] = sqrt(global_norm_sq[i]);
   }
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                      "matrix block Frobenius norms: blocks=%d matrix_norm=%.6e "
                      "ignored_nnz=%lld",
                      num_blocks, sqrt(matrix_norm_sq), global_ignored);

   size_t line_capacity = 256 + ((size_t)num_blocks * 256);
   line                 = hypre_TAlloc(char, line_capacity, HYPRE_MEMORY_HOST);
   if (!line)
   {
      goto cleanup;
   }
   for (int row = 0; row < num_blocks; row++)
   {
      char row_label[96];
      DofLabelFormat(dof_labels, block_labels[row], row_label, sizeof(row_label));
      size_t offset =
         (size_t)snprintf(line, line_capacity, "block Frobenius row %s:", row_label);
      for (int col = 0; col < num_blocks && offset < line_capacity; col++)
      {
         size_t index = ((size_t)row * (size_t)num_blocks) + (size_t)col;
         char   col_label[96];
         DofLabelFormat(dof_labels, block_labels[col], col_label, sizeof(col_label));
         offset +=
            (size_t)snprintf(line + offset, line_capacity - offset, " %s=%.6e(nnz=%lld)",
                             col_label, global_norm_sq[index], global_nnz[index]);
      }
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "%s", line);

      offset =
         (size_t)snprintf(line, line_capacity, "block signed-sum row %s:", row_label);
      for (int col = 0; col < num_blocks && offset < line_capacity; col++)
      {
         size_t index = ((size_t)row * (size_t)num_blocks) + (size_t)col;
         char   col_label[96];
         DofLabelFormat(dof_labels, block_labels[col], col_label, sizeof(col_label));
         offset += (size_t)snprintf(line + offset, line_capacity - offset,
                                    " %s=sum:%.6e/abs:%.6e(pos=%lld,neg=%lld,zero=%lld)",
                                    col_label, global_sum[index], global_abs_sum[index],
                                    global_positive[index], global_negative[index],
                                    global_zero[index]);
      }
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "%s", line);

      offset          = (size_t)snprintf(line, line_capacity,
                                         "block relative-coupling row %s:", row_label);
      double row_diag = global_norm_sq[((size_t)row * (size_t)num_blocks) + (size_t)row];
      for (int col = 0; col < num_blocks && offset < line_capacity; col++)
      {
         char col_label[96];
         DofLabelFormat(dof_labels, block_labels[col], col_label, sizeof(col_label));
         double col_diag =
            global_norm_sq[((size_t)col * (size_t)num_blocks) + (size_t)col];
         double denominator = sqrt(row_diag * col_diag);
         double relative =
            denominator > 0.0
               ? global_norm_sq[((size_t)row * (size_t)num_blocks) + (size_t)col] /
                    denominator
               : 0.0;
         offset += (size_t)snprintf(line + offset, line_capacity - offset, " %s=%.6e",
                                    col_label, relative);
      }
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "%s", line);
   }

cleanup:
   if (comm_handle)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }
   if (owns_diag_copy)
   {
      hypre_TFree(diag_i, HYPRE_MEMORY_HOST);
      hypre_TFree(diag_j, HYPRE_MEMORY_HOST);
      hypre_TFree(diag_a, HYPRE_MEMORY_HOST);
   }
   if (owns_offd_copy)
   {
      hypre_TFree(offd_i, HYPRE_MEMORY_HOST);
      hypre_TFree(offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(offd_a, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(send_labels, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_labels, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_block_labels, HYPRE_MEMORY_HOST);
   hypre_TFree(block_labels, HYPRE_MEMORY_HOST);
   hypre_TFree(label_to_pos, HYPRE_MEMORY_HOST);
   hypre_TFree(local_stats, HYPRE_MEMORY_HOST);
   hypre_TFree(global_stats, HYPRE_MEMORY_HOST);
   hypre_TFree(local_counts, HYPRE_MEMORY_HOST);
   hypre_TFree(global_counts, HYPRE_MEMORY_HOST);
   hypre_TFree(line, HYPRE_MEMORY_HOST);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemSetPrecMatrix(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                                   HYPRE_IJMatrix *precmat_ptr, const Stats *stats)
{
   char        matrix_filename[MAX_FILENAME_LENGTH] = {0};
   int         ls_id = hypredrv_StatsGetLinearSystemID(stats) + 1;
   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                      "preconditioner matrix setup begin");

   if (args->precmat_sequence_filename[0] != '\0')
   {
      int precmat_ls_id = args->precmat_sequence_system_id >= 0
                             ? (int)args->precmat_sequence_system_id
                             : ls_id;
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: sequence file '%s', system %d",
                         args->precmat_sequence_filename, precmat_ls_id);

      if (mat && args->sequence_filename[0] != '\0' &&
          !strcmp(args->precmat_sequence_filename, args->sequence_filename) &&
          precmat_ls_id == ls_id)
      {
         if (*precmat_ptr && *precmat_ptr != mat)
         {
            HYPRE_IJMatrixDestroy(*precmat_ptr);
         }
         *precmat_ptr = mat;
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                            "preconditioner matrix source: reusing main sequence matrix");
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                            "preconditioner matrix setup end");
         return;
      }

      if (*precmat_ptr && *precmat_ptr != mat)
      {
         HYPRE_IJMatrixDestroy(*precmat_ptr);
      }
      *precmat_ptr = NULL;

      if (!hypredrv_LSSeqReadMatrix(comm, args->precmat_sequence_filename, precmat_ls_id,
                                    LinearSystemMemoryLocationGet(args), precmat_ptr))
      {
         HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id,
                            "preconditioner matrix read failed from sequence file '%s', "
                            "system %d",
                            args->precmat_sequence_filename, precmat_ls_id);
         return;
      }

      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix setup end");
      return;
   }

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
      if (*precmat_ptr && *precmat_ptr != mat)
      {
         HYPRE_IJMatrixDestroy(*precmat_ptr);
      }
      *precmat_ptr = NULL;

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
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
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
   size = hypre_VectorSize(seq_vec);

   /* An empty local vector is not an error: a rank that owns no rows legitimately has
    * size 0, and hypre leaves its data pointer NULL in that case. Every norm computed
    * below finishes with a reduction over the vector's communicator, so such a rank
    * must still reach that reduction and contribute zero. Returning early here would
    * leave the remaining ranks blocked in MPI_Allreduce for the rest of the run. */
   if (size < 0 || (size > 0 && !data))
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
   HYPRE_IJVectorInitialize_v2(vec_e, hypre_IJVectorMemoryLocation(vec_x));
#else
   HYPRE_IJVectorInitialize_v2(vec_e, HYPRE_MEMORY_HOST);
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
 * LinearSystemBuildResidual
 *-----------------------------------------------------------------------------*/

static HYPRE_IJVector
LinearSystemBuildResidual(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                          HYPRE_IJVector vec_x, hypre_ParVector **par_b_ptr,
                          hypre_ParVector **par_r_ptr)
{
   void          *obj_A = NULL, *obj_b = NULL, *obj_x = NULL, *obj_r = NULL;
   HYPRE_BigInt   jlower = 0, jupper = -1;
   HYPRE_IJVector vec_r = NULL;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   HYPRE_IJVectorGetObject(vec_x, &obj_x);
   MPI_Comm residual_comm  = hypre_IJVectorComm(vec_b);
   int      residual_valid = obj_A && obj_b && obj_x;
   MPI_Allreduce(MPI_IN_PLACE, &residual_valid, 1, MPI_INT, MPI_MIN, residual_comm);
   if (!residual_valid)
   {
      return NULL;
   }

   HYPRE_IJVectorGetLocalRange(vec_b, &jlower, &jupper);
   HYPRE_IJVectorCreate(hypre_IJVectorComm(vec_b), jlower, jupper, &vec_r);
   residual_valid = vec_r != NULL;
   MPI_Allreduce(MPI_IN_PLACE, &residual_valid, 1, MPI_INT, MPI_MIN, residual_comm);
   if (!residual_valid)
   {
      if (vec_r)
      {
         HYPRE_IJVectorDestroy(vec_r);
      }
      return NULL;
   }
   HYPRE_IJVectorSetObjectType(vec_r, HYPRE_PARCSR);
#if HYPREDRV_HAVE_MEMORY_APIS
   HYPRE_IJVectorInitialize_v2(vec_r, hypre_IJVectorMemoryLocation(vec_b));
#else
   HYPRE_IJVectorInitialize_v2(vec_r, HYPRE_MEMORY_HOST);
#endif
   HYPRE_IJVectorGetObject(vec_r, &obj_r);
   residual_valid = obj_r != NULL;
   MPI_Allreduce(MPI_IN_PLACE, &residual_valid, 1, MPI_INT, MPI_MIN, residual_comm);
   if (!residual_valid)
   {
      HYPRE_IJVectorDestroy(vec_r);
      return NULL;
   }

   hypre_ParVector *par_b = (hypre_ParVector *)obj_b;
   hypre_ParVector *par_r = (hypre_ParVector *)obj_r;
   HYPRE_ParVectorCopy(par_b, par_r);
   HYPRE_ParCSRMatrixMatvec(-1.0, (hypre_ParCSRMatrix *)obj_A, (hypre_ParVector *)obj_x,
                            1.0, par_r);

   if (par_b_ptr)
   {
      *par_b_ptr = par_b;
   }
   if (par_r_ptr)
   {
      *par_r_ptr = par_r;
   }
   return vec_r;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemComputeResidualNorm
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemComputeResidualNorm(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                                         HYPRE_IJVector vec_x, const char *norm_type,
                                         double *res_norm)
{
   HYPRE_IJVector vec_r = LinearSystemBuildResidual(mat_A, vec_b, vec_x, NULL, NULL);
   if (!vec_r)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *res_norm = -1.0;
      return;
   }

   /* Compute residual norm */
   hypredrv_LinearSystemComputeVectorNorm(vec_r, norm_type, res_norm);

   /* Free memory */
   HYPRE_IJVectorDestroy(vec_r);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemLogBlockResidualNorms
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemLogBlockResidualNorms(MPI_Comm comm, HYPRE_IJMatrix mat_A,
                                           HYPRE_IJVector vec_b, HYPRE_IJVector vec_x,
                                           const IntArray    *dofmap,
                                           const DofLabelMap *dof_labels,
                                           const char *log_object_name, int ls_id)
{
   int diagnostics_enabled = hypredrv_LogEnabled(3) ? 1 : 0;
   MPI_Allreduce(MPI_IN_PLACE, &diagnostics_enabled, 1, MPI_INT, MPI_MIN, comm);
   if (!diagnostics_enabled)
   {
      return;
   }

   int max_label = -1;
   int local_valid =
      mat_A && vec_b && vec_x && DofmapDiagnosticLabelsValid(dofmap, &max_label);
   int global_valid = 0;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "block residual diagnostics skipped: invalid dofmap labels");
      return;
   }

   hypre_ParVector *par_b = NULL, *par_r = NULL;
   HYPRE_IJVector vec_r = LinearSystemBuildResidual(mat_A, vec_b, vec_x, &par_b, &par_r);
   if (!vec_r)
   {
      return;
   }

   hypre_Vector *local_b = hypre_ParVectorLocalVector(par_b);
   hypre_Vector *local_r = hypre_ParVectorLocalVector(par_r);
   HYPRE_Int     size    = local_r ? hypre_VectorSize(local_r) : -1;
   local_valid = local_b && local_r && size >= 0 && dofmap->size == (size_t)size;
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPRE_IJVectorDestroy(vec_r);
      return;
   }

   int num_blocks      = (int)dofmap->g_unique_size;
   int num_label_slots = max_label + 1;
   if (!DofmapDiagnosticMetadataAgrees(comm, dofmap, max_label))
   {
      HYPREDRV_LOG_COMMF(
         3, comm, log_object_name, ls_id,
         "block residual diagnostics skipped: inconsistent dofmap metadata");
      HYPRE_IJVectorDestroy(vec_r);
      return;
   }
   int           *label_to_pos = hypre_TAlloc(int, num_label_slots, HYPRE_MEMORY_HOST);
   double        *local_norms  = hypre_CTAlloc(double, 2 * num_blocks, HYPRE_MEMORY_HOST);
   double        *global_norms = hypre_CTAlloc(double, 2 * num_blocks, HYPRE_MEMORY_HOST);
   HYPRE_Complex *host_r       = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
   HYPRE_Complex *host_b       = hypre_TAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
   local_valid =
      label_to_pos && local_norms && global_norms && (size == 0 || (host_r && host_b));
   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      goto cleanup;
   }

   double *local_r2  = local_norms;
   double *local_b2  = local_norms + num_blocks;
   double *global_r2 = global_norms;
   double *global_b2 = global_norms + num_blocks;
   DofmapDiagnosticMapFill(dofmap, num_label_slots, label_to_pos, NULL);

   hypre_TMemcpy(host_r, hypre_VectorData(local_r), HYPRE_Complex, size,
                 HYPRE_MEMORY_HOST, hypre_VectorMemoryLocation(local_r));
   hypre_TMemcpy(host_b, hypre_VectorData(local_b), HYPRE_Complex, size,
                 HYPRE_MEMORY_HOST, hypre_VectorMemoryLocation(local_b));
   for (HYPRE_Int i = 0; i < size; i++)
   {
      int label = dofmap->data[i];
      if (label < 0 || label >= num_label_slots || label_to_pos[label] < 0)
      {
         continue;
      }
      int    pos = label_to_pos[label];
      double r   = (double)hypre_cabs(host_r[i]);
      double b   = (double)hypre_cabs(host_b[i]);
      local_r2[pos] += r * r;
      local_b2[pos] += b * b;
   }

   MPI_Allreduce(local_norms, global_norms, 2 * num_blocks, MPI_DOUBLE, MPI_SUM, comm);
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "block residual L2 norms begin");
   for (int i = 0; i < num_blocks; i++)
   {
      int    label = dofmap->g_unique_data[i];
      char   label_text[96];
      double rnorm = sqrt(global_r2[i]);
      double bnorm = sqrt(global_b2[i]);
      DofLabelFormat(dof_labels, label, label_text, sizeof(label_text));
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "  %s: ||r_i||_2=%.6e ||b_i||_2=%.6e rel=%.6e", label_text,
                         rnorm, bnorm, bnorm > 0.0 ? rnorm / bnorm : rnorm);
   }
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "block residual L2 norms end");

cleanup:
   hypre_TFree(label_to_pos, HYPRE_MEMORY_HOST);
   hypre_TFree(local_norms, HYPRE_MEMORY_HOST);
   hypre_TFree(global_norms, HYPRE_MEMORY_HOST);
   hypre_TFree(host_r, HYPRE_MEMORY_HOST);
   hypre_TFree(host_b, HYPRE_MEMORY_HOST);
   HYPRE_IJVectorDestroy(vec_r);
}

/*-----------------------------------------------------------------------------
 * Load a timestep schedule (timestep id, first linear-system id) from a file
 *-----------------------------------------------------------------------------*/

/* timestep_ids is optional: pass NULL to skip loading per-timestep IDs (caller
 * then derives IDs from array position).  timestep_starts is required. */
uint32_t
hypredrv_LinearSystemLoadTimestepSchedule(const char *filename, IntArray **timestep_ids,
                                          IntArray **timestep_starts)
{
   if (!timestep_starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid output pointer for timestep schedule");
      return hypredrv_ErrorCodeGet();
   }

   if (timestep_ids && *timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(timestep_ids);
   }

   if (*timestep_starts) /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_IntArrayDestroy(timestep_starts);
   }

   if (!filename || filename[0] == '\0') /* GCOVR_EXCL_BR_LINE */
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Missing timestep schedule filename");
      return hypredrv_ErrorCodeGet();
   }

   FILE *fp = fopen(filename, "r");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open timestep file: '%s'", filename);
      return hypredrv_ErrorCodeGet();
   }

   int total = 0;
   if (fscanf(fp, "%d", &total) != 1 || total <= 0)
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid timestep file header in '%s'", filename);
      return hypredrv_ErrorCodeGet();
   }

   IntArray *ids = NULL;
   if (timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      ids = hypredrv_IntArrayCreate((size_t)total);
      if (!ids) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
         hypredrv_ErrorMsgAdd("Failed to allocate timestep ids array");
         return hypredrv_ErrorCodeGet();
      }
   }

   IntArray *starts = hypredrv_IntArrayCreate((size_t)total);
   if (!starts) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_IntArrayDestroy(&ids);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate timestep starts array");
      return hypredrv_ErrorCodeGet();
   }

   for (int i = 0; i < total; i++)
   {
      int timestep = 0;
      int ls_start = 0;
      if (fscanf(fp, "%d %d", &timestep, &ls_start) != 2 ||
          ls_start < 0) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_IntArrayDestroy(&ids);
         hypredrv_IntArrayDestroy(&starts);
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid timestep entry in '%s' at line %d", filename,
                              i + 2);
         return hypredrv_ErrorCodeGet();
      }

      if (ids) /* GCOVR_EXCL_BR_LINE */
      {
         ids->data[i] = timestep;
      }
      starts->data[i] = ls_start;
   }

   fclose(fp);
   if (timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      *timestep_ids = ids;
   }
   *timestep_starts = starts;
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Check (debug builds only) that CSR row ranges form a contiguous partition
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemValidateCSRPartitionDebug(MPI_Comm comm, HYPRE_BigInt row_start,
                                               HYPRE_BigInt row_end)
{
#ifdef HYPREDRV_USING_DEBUG
   int comm_size = 1;
   if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemSetMatrixFromCSR: failed to query MPI communicator");
      return hypredrv_ErrorCodeGet();
   }

   const long long local_range[2] = {(long long)row_start, (long long)row_end};
   long long      *all_ranges = hypre_TAlloc(long long, 2 * comm_size, HYPRE_MEMORY_HOST);
   if (!all_ranges)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemSetMatrixFromCSR: failed to allocate debug "
         "partition scratch space");
      return hypredrv_ErrorCodeGet();
   }

   int mpi_ierr =
      MPI_Allgather(local_range, 2, MPI_LONG_LONG, all_ranges, 2, MPI_LONG_LONG, comm);
   if (mpi_ierr != MPI_SUCCESS)
   {
      hypre_TFree(all_ranges, HYPRE_MEMORY_HOST);
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemSetMatrixFromCSR: failed to gather debug row "
         "partition");
      return hypredrv_ErrorCodeGet();
   }

   long long expected_start = 0;
   for (int rank = 0; rank < comm_size; rank++)
   {
      size_t    range_idx = (size_t)2 * (size_t)rank;
      long long start     = all_ranges[range_idx];
      long long end       = all_ranges[range_idx + 1];
      if (end < start || start != expected_start)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "HYPREDRV_LinearSystemSetMatrixFromCSR: rank %d row range [%lld, %lld] "
            "does not continue the global partition at %lld",
            rank, start, end, expected_start);
         hypre_TFree(all_ranges, HYPRE_MEMORY_HOST);
         return hypredrv_ErrorCodeGet();
      }
      expected_start = end + 1;
   }

   hypre_TFree(all_ranges, HYPRE_MEMORY_HOST);
#else
   (void)comm;
   (void)row_start;
   (void)row_end;
#endif
   return hypredrv_ErrorCodeGet();
}
