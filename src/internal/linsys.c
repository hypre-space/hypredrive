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

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "internal/containers.h"
#include "internal/error.h"
#include "internal/linsys.h"
#include "internal/lsseq.h"
#include "logging.h"
#include "object.h"

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

static void
LinearSystemSetSuffixSet(void *field, YAMLnode *node)
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

/* Local calls only record failures. Each builder reaches the next collective
 * agreement point before any rank can leave its current construction phase. */
static uint32_t
LinearSystemBuildRecordError(const char *label, HYPRE_Int code)
{
   if (code)
   {
      char message[HYPRE_MAX_MSG_LEN];
      HYPRE_DescribeError(code, message);
      hypredrv_ErrorCodeSet(ERROR_HYPRE_INTERNAL);
      hypredrv_ErrorMsgAdd("%s: HYPRE call failed: %s", label, message);
      return ERROR_HYPRE_INTERNAL;
   }
   return ERROR_NONE;
}

/* Internal calls report their own status, independently of diagnostics left
 * by an earlier caller. Merge only this operation's failure across ranks. */
static uint32_t
LinearSystemBuildStatusSync(MPI_Comm comm, uint32_t status)
{
   uint32_t collective_status = ERROR_NONE;
   MPI_Allreduce(&status, &collective_status, 1, MPI_UINT32_T, MPI_BOR, comm);
   if (collective_status != status)
   {
      hypredrv_ErrorCodeSet((hypredrv_error_t)collective_status);
      hypredrv_ErrorMsgAdd("Linear-system construction failed on another rank");
   }
   return collective_status;
}

#define LINSYS_HYPRE_CALL(label, call)                         \
   do                                                          \
   {                                                           \
      if (status == ERROR_NONE)                                \
      {                                                        \
         status = LinearSystemBuildRecordError(label, (call)); \
      }                                                        \
   } while (0)

/* Exact and near modes share layout validation, storage construction, and
 * publication. Only exact modes are orthonormalized and kept on the host for
 * projection. A rejected replacement leaves the installed modes intact. */
uint32_t
hypredrv_LinearSystemSetNullSpaceModes(hypredrv_t *hypredrv, bool exact, int num_entries,
                                       int num_components, const HYPRE_Complex *values)
{
   MPI_Comm       comm        = hypredrv->comm;
   uint32_t       status      = ERROR_NONE;
   HYPRE_IJVector replacement = NULL;
   HYPRE_BigInt   ilower = 0, iupper = 0, jlower = 0, jupper = 0;
   HYPRE_BigInt  *indices = NULL;
   HYPRE_Complex *modes = NULL, *zeros = NULL;
   size_t         total     = 0;
   int            request[] = {exact, num_components}, minimum[2], maximum[2];
   const char    *label     = exact ? "SetNullSpace" : "SetNearNullSpace";

   /* Component counts control collective loops, including the clear operation.
    * Local entry counts may differ, including ranks with no matrix columns. */
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1] || num_components < 0 ||
       (!exact && num_components == 0))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: mode type and component count must agree across ranks; "
                           "only exact modes may be cleared with zero components",
                           label);
      status = ERROR_INVALID_VAL;
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   if (num_components == 0) goto cleanup;

   if (!hypredrv->mat_A.handle || num_entries < 0 ||
       (exact && num_entries > 0 && !values))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: set the matrix first and provide a nonnegative local "
                           "entry count and values for nonempty exact modes",
                           label);
      status = ERROR_INVALID_VAL;
   }
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   if (exact && num_components > 1)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Multiple null space modes require hypre >= 2.26.0");
      status = ERROR_INVALID_VAL;
   }
#endif
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;

   LINSYS_HYPRE_CALL(label, HYPRE_IJMatrixGetLocalRange(hypredrv->mat_A.handle, &ilower,
                                                        &iupper, &jlower, &jupper));
   if (!status &&
       ((num_entries == 0 && jupper >= jlower) ||
        (num_entries > 0 && (jupper < jlower || (uintmax_t)jupper - (uintmax_t)jlower !=
                                                   (uintmax_t)num_entries - 1))))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: local entry count %d does not match matrix columns "
                           "[%lld, %lld]",
                           label, num_entries, (long long)jlower, (long long)jupper);
      status = ERROR_INVALID_VAL;
   }
   if (num_entries > 0 &&
       (size_t)num_components > SIZE_MAX / sizeof(HYPRE_Complex) / (size_t)num_entries)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: mode buffer size overflows size_t", label);
      status = ERROR_INVALID_VAL;
   }
   if (!status)
   {
      total = (size_t)num_entries * (size_t)num_components;
      for (size_t i = 0; values && i < total; i++)
      {
         if (!isfinite((double)hypre_creal(values[i])) ||
             !isfinite((double)hypre_cimag(values[i])))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("%s: mode values must be finite", label);
            status = ERROR_INVALID_VAL;
            break;
         }
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;

   indices = hypredrv_Calloc((size_t)num_entries, sizeof(*indices), "mode indices");
   if (!indices) status = ERROR_ALLOCATION;
   if (exact)
   {
      modes = hypredrv_Calloc(total, sizeof(*modes), "exact null space modes");
      if (!modes) status = ERROR_ALLOCATION;
   }
   else if (!values)
   {
      zeros = hypredrv_Calloc((size_t)num_entries, sizeof(*zeros), "zero modes");
      if (!zeros) status = ERROR_ALLOCATION;
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   for (int i = 0; i < num_entries; i++) indices[i] = jlower + (HYPRE_BigInt)i;

   if (exact)
   {
      if (total) memcpy(modes, values, total * sizeof(*modes));
      /* Modified Gram-Schmidt on the component-major input blocks. All ranks
       * enter with validated counts and allocated storage, including empty ranks. */
      for (int k = 0; k < num_components; k++)
      {
         HYPRE_Complex *zk        = modes + (size_t)k * (size_t)num_entries;
         double         dot_local = 0.0, dot = 0.0;
         for (int i = 0; i < num_entries; i++) dot_local += (double)(zk[i] * zk[i]);
         MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
         double norm_orig = sqrt(dot);
         for (int j = 0; j < k; j++)
         {
            const HYPRE_Complex *zj = modes + (size_t)j * (size_t)num_entries;
            dot_local               = 0.0;
            for (int i = 0; i < num_entries; i++) dot_local += (double)(zk[i] * zj[i]);
            MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
            for (int i = 0; i < num_entries; i++) zk[i] -= (HYPRE_Complex)dot * zj[i];
         }
         dot_local = 0.0;
         for (int i = 0; i < num_entries; i++) dot_local += (double)(zk[i] * zk[i]);
         MPI_Allreduce(&dot_local, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
         dot = sqrt(dot);
         if (!isfinite(norm_orig) || !isfinite(dot) ||
             dot <= HYPREDRV_NULLSPACE_DEP_TOL * norm_orig)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("Null space modes must be linearly independent "
                                 "(mode %d has relative norm %e after orthogonalization)",
                                 k, norm_orig > 0.0 ? dot / norm_orig : 0.0);
            status = ERROR_INVALID_VAL;
            goto cleanup;
         }
         for (int i = 0; i < num_entries; i++) zk[i] /= (HYPRE_Complex)dot;
      }
      values = modes;
   }

   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorCreate(comm, jlower, jupper, &replacement));
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorSetObjectType(replacement, HYPRE_PARCSR));
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorSetNumComponents(replacement, num_components));
#endif
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorInitialize_v2(replacement, HYPRE_MEMORY_HOST));
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   for (int c = 0; c < num_components; c++)
   {
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
      LINSYS_HYPRE_CALL(label, HYPRE_IJVectorSetComponent(replacement, c));
#else
      /* Preserve legacy near-mode input on old hypre: its AMG/MGR paths do
       * not consume these modes, and IJ stores only the last component. */
#endif
      if (num_entries)
      {
         const HYPRE_Complex *block =
            values ? values + (size_t)c * (size_t)num_entries : zeros;
         LINSYS_HYPRE_CALL(
            label, HYPRE_IJVectorSetValues(replacement, num_entries, indices, block));
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorAssemble(replacement));
   if (!exact && hypredrv->iargs->ls.exec_policy)
   {
      LINSYS_HYPRE_CALL(label,
                        hypredrv_IJVectorMigrate(replacement, HYPRE_MEMORY_DEVICE));
   }

cleanup:
   free(indices);
   free(modes);
   free(zeros);
   status |= hypredrv_HypreConsumeErrors(comm);
   status = LinearSystemBuildStatusSync(comm, status);
   if (!status)
   {
      HYPRE_IJVector *installed = exact ? &hypredrv->vec_ns : &hypredrv->vec_nn;
      HYPRE_IJVector  previous  = *installed;
      *installed                = replacement;
      replacement               = NULL;
      if (exact) hypredrv->num_ns = num_components;
      else if (hypredrv->iargs->precon_method == PRECON_MGR)
         hypredrv_MGRSetNearNullSpace(&hypredrv->iargs->precon.mgr, *installed);
      if (previous) HYPRE_IJVectorDestroy(previous);
   }
   if (replacement) HYPRE_IJVectorDestroy(replacement);
   return status;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemProjectOutNullSpace
 *
 * Remove the (orthonormalized) null space components from a vector, fixing
 * the gauge of solutions that are defined up to a null space contribution
 *-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues/GetValues with host index and value arrays. hypre's
 * device IJ paths read and write device pointers (they copy with device-to-
 * device transfers), so for a device-resident vector the arrays are staged
 * through temporary device buffers.
 *-----------------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU) && HYPREDRV_HAVE_MEMORY_APIS
static int
LinearSystemVectorOnDevice(HYPRE_IJVector vec)
{
   return hypre_GetActualMemLocation(hypre_IJVectorMemoryLocation(vec)) ==
          hypre_MEMORY_DEVICE;
}
#endif

static HYPRE_Int
LinearSystemSetHostValues(HYPRE_IJVector vec, HYPRE_Int n, const HYPRE_BigInt *indices,
                          const HYPRE_Complex *values)
{
#if defined(HYPRE_USING_GPU) && HYPREDRV_HAVE_MEMORY_APIS
   /* GCOVR_EXCL_START */
   if (n > 0 && LinearSystemVectorOnDevice(vec))
   {
      HYPRE_BigInt  *d_idx  = NULL;
      HYPRE_Complex *d_vals = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_DEVICE);
      if (indices)
      {
         d_idx = hypre_TAlloc(HYPRE_BigInt, n, HYPRE_MEMORY_DEVICE);
      }
      if (!d_vals || (indices && !d_idx))
      {
         hypre_TFree(d_idx, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
         return HYPRE_ERROR_MEMORY;
      }
      hypre_TMemcpy(d_vals, values, HYPRE_Complex, n, HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
      if (indices)
      {
         hypre_TMemcpy(d_idx, indices, HYPRE_BigInt, n, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
      HYPRE_Int ierr = HYPRE_IJVectorSetValues(vec, n, d_idx, d_vals);
      hypre_TFree(d_idx, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
      return ierr;
   }
   /* GCOVR_EXCL_STOP */
#endif
   return HYPRE_IJVectorSetValues(vec, n, indices, values);
}

static HYPRE_Int
LinearSystemGetHostValues(HYPRE_IJVector vec, HYPRE_Int n, const HYPRE_BigInt *indices,
                          HYPRE_Complex *values)
{
#if defined(HYPRE_USING_GPU) && HYPREDRV_HAVE_MEMORY_APIS
   /* GCOVR_EXCL_START */
   if (n > 0 && LinearSystemVectorOnDevice(vec))
   {
      HYPRE_BigInt  *d_idx  = NULL;
      HYPRE_Complex *d_vals = hypre_TAlloc(HYPRE_Complex, n, HYPRE_MEMORY_DEVICE);
      if (indices)
      {
         d_idx = hypre_TAlloc(HYPRE_BigInt, n, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(d_idx, indices, HYPRE_BigInt, n, HYPRE_MEMORY_DEVICE,
                       HYPRE_MEMORY_HOST);
      }
      HYPRE_Int ierr = HYPRE_IJVectorGetValues(vec, n, d_idx, d_vals);
      hypre_TMemcpy(values, d_vals, HYPRE_Complex, n, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_idx, HYPRE_MEMORY_DEVICE);
      hypre_TFree(d_vals, HYPRE_MEMORY_DEVICE);
      return ierr;
   }
   /* GCOVR_EXCL_STOP */
#endif
   return HYPRE_IJVectorGetValues(vec, n, (HYPRE_BigInt *)(uintptr_t)indices, values);
}

/* Projects `vec` onto the orthogonal complement of the null-space modes: form
 * the dot product with each mode, reduce it across ranks, then subtract the
 * corresponding component. Buffers are supplied by the caller so the collective
 * allocation agreement stays in one place. */
static void
LinearSystemProjectOntoModes(HYPRE_IJVector vec_ns, int num_ns, HYPRE_IJVector vec,
                             int num_entries, HYPRE_BigInt jlower, MPI_Comm comm,
                             HYPRE_Complex *xbuf, HYPRE_Complex *zbuf,
                             HYPRE_BigInt *indices, double *dots_local, double *dots)
{
   for (int i = 0; i < num_entries; i++)
   {
      indices[i] = jlower + (HYPRE_BigInt)i;
   }

   LinearSystemGetHostValues(vec, num_entries, NULL, xbuf);
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
   LinearSystemSetHostValues(vec, num_entries, indices, xbuf);
   HYPRE_IJVectorAssemble(vec);
}

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
      hypredrv_IJVectorMigrate(vec, HYPRE_MEMORY_HOST);
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
      hypredrv_AllocationFailure("Failed to allocate null-space projection buffers");
      free(xbuf);
      free(zbuf);
      free(indices);
      free(dots_local);
      free(dots);
      return;
   }
   LinearSystemProjectOntoModes(vec_ns, num_ns, vec, num_entries, jlower, comm, xbuf,
                                zbuf, indices, dots_local, dots);

#if defined(HYPRE_USING_GPU)
   if (orig_memloc != HYPRE_MEMORY_HOST)
   {
      hypredrv_IJVectorMigrate(vec, orig_memloc);
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

/*-----------------------------------------------------------------------------
 * Communicators of IJ objects for logging/collectives: MPI_COMM_NULL when the
 * object (or a matrix's ParCSR object) is missing. hypredrv_IJSystemComm
 * returns the first available among A, M, b and x (any may be NULL).
 *-----------------------------------------------------------------------------*/

MPI_Comm
hypredrv_IJMatrixComm(HYPRE_IJMatrix mat)
{
   void *obj = NULL;

   if (!mat)
   {
      return MPI_COMM_NULL;
   }
   HYPRE_IJMatrixGetObject(mat, &obj);
   /* GCOVR_EXCL_BR_START */
   return obj ? hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *)obj) : MPI_COMM_NULL;
   /* GCOVR_EXCL_BR_STOP */
}

MPI_Comm
hypredrv_IJVectorComm(HYPRE_IJVector vec)
{
   return vec ? hypre_IJVectorComm((hypre_IJVector *)vec) : MPI_COMM_NULL;
}

/* hypre before 2.32.0 develop 37 (hypre PR #1257) copied num_rows entries of a
 * CSR matrix's shorter rownnz array when migrating it, reading past its end
 * (seen as a crash in multi-rank runs, whose off-diagonal blocks have one).
 * rownnz is an optional index of the nonzero rows (NULL means all rows), so
 * drop it before migrating on those versions. */
static void
CSRMatrixDropRownnzForMigrate(hypre_CSRMatrix *A)
{
#if HYPRE_CHECK_MIN_VERSION(23200, 37)
   (void)A;
#else
   if (A && hypre_CSRMatrixRownnz(A))
   {
      hypre_TFree(hypre_CSRMatrixRownnz(A), hypre_CSRMatrixMemoryLocation(A));
      hypre_CSRMatrixNumRownnz(A) = hypre_CSRMatrixNumRows(A);
   }
#endif
}

/* Migrates a ParCSR matrix, working around the rownnz bug above */
HYPRE_Int
hypredrv_ParCSRMatrixMigrate(HYPRE_ParCSRMatrix A, HYPRE_MemoryLocation location)
{
   if (!A)
   {
      return 0;
   }
   CSRMatrixDropRownnzForMigrate(hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A));
   CSRMatrixDropRownnzForMigrate(hypre_ParCSRMatrixOffd((hypre_ParCSRMatrix *)A));
   return hypre_ParCSRMatrixMigrate((hypre_ParCSRMatrix *)A, location);
}

/* HYPRE_IJMatrixMigrate/HYPRE_IJVectorMigrate first appear after hypre 2.32.0
 * (develop 24) and only migrate the underlying ParCSR object, so use that. */
HYPRE_Int
hypredrv_IJMatrixMigrate(HYPRE_IJMatrix mat, HYPRE_MemoryLocation location)
{
   void *obj = NULL;
   HYPRE_IJMatrixGetObject(mat, &obj);
   return hypredrv_ParCSRMatrixMigrate((HYPRE_ParCSRMatrix)obj, location);
}

HYPRE_Int
hypredrv_IJVectorMigrate(HYPRE_IJVector vec, HYPRE_MemoryLocation location)
{
#if HYPRE_CHECK_MIN_VERSION(23200, 24)
   return HYPRE_IJVectorMigrate(vec, location);
#else
   void *obj = NULL;
   HYPRE_IJVectorGetObject(vec, &obj);
   return obj ? hypre_ParVectorMigrate((hypre_ParVector *)obj, location) : 0;
#endif
}

MPI_Comm
hypredrv_IJSystemComm(HYPRE_IJMatrix A, HYPRE_IJMatrix M, HYPRE_IJVector b,
                      HYPRE_IJVector x)
{
   MPI_Comm comm = hypredrv_IJMatrixComm(A);

   /* GCOVR_EXCL_BR_START */
   if (comm == MPI_COMM_NULL)
   {
      comm = hypredrv_IJMatrixComm(M);
   }
   if (comm == MPI_COMM_NULL)
   {
      comm = hypredrv_IJVectorComm(b);
   }
   /* GCOVR_EXCL_BR_STOP */
   return (comm != MPI_COMM_NULL) ? comm : hypredrv_IJVectorComm(x);
}

static int
LinearSystemDataFilenameResolve(const LS_args *args, int ls_id, const char *filename,
                                const char *basename, char *resolved,
                                size_t resolved_size)
{
   int len     = -1;
   resolved[0] = '\0';
   if (args->dirname[0] != '\0' && filename[0] != '\0')
   {
      int suffix = hypredrv_LinearSystemGetSuffix(args, ls_id);
      len        = snprintf(resolved, resolved_size, "%s_%0*d/%s", args->dirname,
                            (int)args->digits_suffix, suffix, filename);
   }
   else if (filename[0] != '\0')
   {
      len = snprintf(resolved, resolved_size, "%s", filename);
   }
   else if (basename[0] != '\0')
   {
      int suffix = hypredrv_LinearSystemGetSuffix(args, ls_id);
      len        = snprintf(resolved, resolved_size, "%s_%0*d", basename,
                            (int)args->digits_suffix, suffix);
   }
   else
   {
      return 0;
   }

   if (len < 0 || (size_t)len >= resolved_size)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Data file path exceeds %zu characters: '%s'",
                           resolved_size - 1, resolved);
      resolved[0] = '\0';
      return 0;
   }

   return 1;
}

static uint32_t
LinearSystemIJVectorReadFromFile(MPI_Comm comm, const char *filename,
                                 HYPRE_MemoryLocation memory_location,
                                 HYPRE_IJVector      *vector_ptr)
{
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();
   /* File visibility can differ between ranks. Agree before choosing readers
    * with different collectives, and reuse the partition count for assembly. */
   int binary    = hypredrv_CheckBinaryDataExists(filename);
   int nparts    = hypredrv_CountNumberOfPartitions(filename);
   int request[] = {binary, nparts}, minimum[2], maximum[2], nprocs;
   MPI_Comm_size(comm, &nprocs);
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Vector input format and partition count must agree across ranks");
      return ERROR_INVALID_VAL;
   }
   if (!binary && nparts != 0 && nparts != nprocs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("ASCII input '%s' has %d part(s) but runs on %d MPI rank(s); "
                           "use matching ranks or binary input",
                           filename, nparts, nprocs);
      return ERROR_INVALID_VAL;
   }
   if (binary)
   {
      if (nparts >= nprocs)
      {
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
   uint32_t status = hypredrv_ErrorCodeGet();
   if (!*vector_ptr || HYPRE_GetError())
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(filename);
      status |= ERROR_FILE_NOT_FOUND;
   }
   return LinearSystemBuildStatusSync(comm, status);
}

static int
LinearSystemIJMatrixReadFromFile(MPI_Comm comm, const LS_args *args,
                                 const char *matrix_filename, HYPRE_IJMatrix *matrix_ptr)
{
   /* Decide the reader and inspect its input once, before any native collective.
    * A rank-local missing file must not strand peers inside HYPRE assembly. */
   enum
   {
      ASCII,
      BINARY,
      MARKET
   };
   int format =
      args->type == 1 && hypredrv_CheckBinaryDataExists(matrix_filename) ? BINARY : ASCII;
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   if (args->type == 3) format = MARKET;
#endif
   int nparts = format == MARKET ? 0 : hypredrv_CountNumberOfPartitions(matrix_filename);
   int request[] = {format, nparts, args->type}, minimum[3], maximum[3], nprocs;
   MPI_Comm_size(comm, &nprocs);
   MPI_Allreduce(request, minimum, 3, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 3, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1] || minimum[2] != maximum[2])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Matrix input format and partition count must agree across ranks");
      return 0;
   }
   int readable = 0;
   if (format == MARKET)
   {
      FILE *file = fopen(matrix_filename, "r");
      readable   = file != NULL;
      if (file) fclose(file);
   }
   else if (format == BINARY) readable = nparts >= nprocs;
   else
   {
      int rank;
      MPI_Comm_rank(comm, &rank);
      char part_filename[MAX_FILENAME_LENGTH + 32];
      snprintf(part_filename, sizeof(part_filename), "%s.%05d", matrix_filename, rank);
      FILE *file = fopen(part_filename, "r");
      readable   = file != NULL;
      if (file) fclose(file);
      if (nparts && nparts != nprocs)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "ASCII input '%s' has %d part(s) but runs on %d MPI rank(s); "
            "use matching ranks or binary input",
            matrix_filename, nparts, nprocs);
      }
   }
   if (!readable)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(matrix_filename);
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return 0;

   if (format == BINARY)
      hypredrv_IJMatrixReadMultipartBinary(matrix_filename, comm, (uint64_t)nparts,
                                           LinearSystemMemoryLocationGet(args),
                                           matrix_ptr);
#if HYPRE_CHECK_MIN_VERSION(22600, 0)
   else if (format == MARKET)
      HYPRE_IJMatrixReadMM(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
#endif
   else HYPRE_IJMatrixRead(matrix_filename, comm, HYPRE_PARCSR, matrix_ptr);
   if (HYPRE_GetError() || !*matrix_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename(matrix_filename);
      return 0;
   }
#if HYPREDRV_HAVE_MEMORY_APIS
   if (args->exec_policy) hypredrv_IJMatrixMigrate(*matrix_ptr, HYPRE_MEMORY_DEVICE);
#endif
   return 1;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetArgsFromYAML
 *-----------------------------------------------------------------------------*/

/* set_suffix and init_suffix/last_suffix are mutually exclusive, and the
 * precmat sequence options are only meaningful together. */
static void
LinearSystemValidateSuffixAndSequence(LS_args *args)
{
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

void
hypredrv_LinearSystemSetArgsFromYAML(LS_args *args, YAMLnode *parent)
{
   StrArray keys = hypredrv_LinearSystemGetValidKeys();
   YAML_NODE_ITERATE(parent, child)
   {
      size_t field = hypredrv_YAMLnodeValidateSchema(child, keys,
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
         /* A repeated dof_labels key replaces the earlier map. */
         hypredrv_DofLabelMapDestroy(&args->dof_labels);
         args->dof_labels = hypredrv_DofLabelMapCreate();

         if (child->children)
         {
            /* Block mapping: each child node is a label:value pair */
            YAML_NODE_ITERATE(child, entry)
            {
               int  val = 0;
               char lower_key[64];
               if (!hypredrv_ParseInt(entry->val, &val))
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
            char *saveptr = NULL;
            char *pair    = strtok_r(inner, ",", &saveptr);
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
                  if (!hypredrv_ParseInt(pair_val, &val_int))
                  {
                     hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
                     hypredrv_ErrorMsgAdd("dof_labels: expected integer value for "
                                          "label '%s', got '%s'",
                                          pair_key, pair_val);
                     break;
                  }
                  hypredrv_DofLabelMapAdd(args->dof_labels, pair_key, val_int);
               }
               pair = strtok_r(NULL, ",", &saveptr);
            }
            free(buf);
         }

         YAML_NODE_SET_VALID(child);
         continue;
      }

      if (field != SIZE_MAX)
         hypredrv_FieldOffsetMapSet(&ls_field_offset_map[field], args, child);
   }

   LinearSystemValidateSuffixAndSequence(args);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemSetMatrix(hypredrv_t *hypredrv, HYPRE_IJMatrix matrix)
{
   uint32_t status = matrix ? hypredrv_IJMatrixValidate(matrix, "SetMatrix") : ERROR_NONE;
   if (!status)
   {
      hypredrv_ObjectPrepareExecution(hypredrv, matrix, 1);
      status |= hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
   }
   else hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
   if (status) return status;
   hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_A, matrix,
                                hypredrv_AttachmentIsOwned(hypredrv->lib_mode));
   hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_M, matrix, false);
   hypredrv->precmat_sequence_cache_args = NULL;
   return hypredrv_ErrorCodeGet();
}

static bool
LinearSystemHasCachedPrecMatrix(const hypredrv_t *hypredrv)
{
   return hypredrv->mat_M.handle && hypredrv->mat_M.owned &&
          hypredrv->precmat_sequence_cache_args == hypredrv->iargs;
}

uint32_t
hypredrv_LinearSystemReadMatrix(hypredrv_t *hypredrv)
{
   MPI_Comm comm = hypredrv->comm;
   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Matrix input requires parsed arguments");
   }
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();

   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(hypredrv->stats, log_name_buf, sizeof(log_name_buf));
   const LS_args *args  = &hypredrv->iargs->ls;
   int            ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int request[] = {args->sequence_filename[0] != '\0', ls_id}, minimum[2], maximum[2];
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Matrix source and system index must agree across ranks");
   }
   char matrix_filename[MAX_FILENAME_LENGTH] = {0};
   if (!request[0] && !LinearSystemDataFilenameResolve(
                         args, ls_id, args->matrix_filename, args->matrix_basename,
                         matrix_filename, sizeof(matrix_filename)))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAddInvalidFilename("");
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();

   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "matrix");
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix read begin");
   HYPRE_IJMatrix replacement = NULL;
   if (request[0])
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "matrix source: sequence file '%s'", args->sequence_filename);
      hypredrv_LSSeqReadMatrix(comm, args->sequence_filename, ls_id,
                               LinearSystemMemoryLocationGet(args), &replacement);
   }
   else
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix source: '%s'",
                         matrix_filename);
      LinearSystemIJMatrixReadFromFile(comm, args, matrix_filename, &replacement);
   }
   hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "matrix");
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      if (replacement) HYPRE_IJMatrixDestroy(replacement);
      HYPREDRV_LOG_COMMF(2, comm, log_object_name, ls_id, "matrix read failed from '%s'",
                         request[0] ? args->sequence_filename : matrix_filename);
      return hypredrv_ErrorCodeGet();
   }

   /* A fixed sequence preconditioner survives subsequent main-matrix reads.
    * Ownership and aliases change only after every rank has a replacement. */
   bool keep_mat_M = LinearSystemHasCachedPrecMatrix(hypredrv);
   if (!keep_mat_M) hypredrv->precmat_sequence_cache_args = NULL;
   hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_A, replacement, true);
   if (!keep_mat_M)
      hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_M, replacement, false);
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "matrix read end");
   return hypredrv_ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Array ingress: validate, construct, and install a complete replacement
 *-----------------------------------------------------------------------------*/

/* Derive a positive local row count without overflowing signed HYPRE_BigInt.
 * Both array builders use this boundary before indexing caller-owned buffers. */
static int
LinearSystemArrayRowCount(const char *label, HYPRE_BigInt first, HYPRE_BigInt last,
                          HYPRE_Int *count)
{
   const uintmax_t max_count = (UINTMAX_C(1) << ((sizeof(HYPRE_Int) * CHAR_BIT) - 1)) - 1;
   const uintmax_t span      = (uintmax_t)last - (uintmax_t)first;
   if (last < first || span >= max_count)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "%s: row range [%lld, %lld] has no positive HYPRE_Int-sized count", label,
         (long long)first, (long long)last);
      return 0;
   }
   *count = (HYPRE_Int)(span + 1);
   return 1;
}

uint32_t
hypredrv_LinearSystemSetMatrixFromCSR(hypredrv_t *hypredrv, HYPRE_BigInt row_start,
                                      HYPRE_BigInt row_end, const HYPRE_BigInt *indptr,
                                      const HYPRE_BigInt *col_indices,
                                      const HYPRE_Real   *data)
{
   MPI_Comm       comm          = hypredrv->comm;
   uint32_t       status        = ERROR_NONE;
   HYPRE_IJMatrix matrix        = NULL;
   HYPRE_Int     *ncols_per_row = NULL, *diag_sizes = NULL, *offd_sizes = NULL;
   HYPRE_BigInt  *row_ids = NULL;
   HYPRE_Int      nrows = 0, nnz = 0;
   HYPRE_BigInt   offset = 0;
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
#ifdef HYPRE_USING_GPU
   memory_location =
      LinearSystemMemoryLocationGet(hypredrv->iargs ? &hypredrv->iargs->ls : NULL);
#endif
   const int host = hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_DEVICE;

   if (!LinearSystemArrayRowCount("SetMatrixFromCSR", row_start, row_end, &nrows))
   {
      status = ERROR_INVALID_VAL;
   }
   /* Validate scalar partition metadata before deriving any CSR buffer access.
    * A prefix maximum gives the preceding end for a valid ordered partition;
    * any earlier invalid partition is rejected by the common status reduction. */
   const long long local_end     = (long long)row_end;
   long long       preceding_end = -1, global_end = -1;
   int             mpi_status =
      MPI_Exscan(&local_end, &preceding_end, 1, MPI_LONG_LONG, MPI_MAX, comm);
   mpi_status |= MPI_Allreduce(&local_end, &global_end, 1, MPI_LONG_LONG, MPI_MAX, comm);
   if (mpi_status != MPI_SUCCESS)
   {
      status = ERROR_UNKNOWN;
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "SetMatrixFromCSR: could not inspect the global row partition");
   }
   if (!status)
   {
      const uintmax_t max_rows =
         (UINTMAX_C(1) << (sizeof(HYPRE_BigInt) * CHAR_BIT - 1)) - 1;
      if (global_end < 0 || (uintmax_t)global_end >= max_rows ||
          (hypredrv->mypid && preceding_end == LLONG_MAX))
      {
         status = ERROR_INVALID_VAL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("SetMatrixFromCSR: global row count is out of range");
      }
      else
      {
         long long expected_start = hypredrv->mypid ? preceding_end + 1 : 0;
         if ((long long)row_start != expected_start)
         {
            status = ERROR_INVALID_VAL;
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("SetMatrixFromCSR: rank %d row range [%lld, %lld] "
                                 "does not continue the global partition at %lld",
                                 hypredrv->mypid, (long long)row_start, local_end,
                                 expected_start);
         }
      }
   }
   if (status) goto inputs_done;
   if (!indptr)
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetMatrixFromCSR: indptr must be non-NULL");
      goto inputs_done;
   }
   offset = indptr[0];
   if (offset < 0 || indptr[nrows] < offset)
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetMatrixFromCSR: invalid first/last row offsets");
      goto inputs_done;
   }
   HYPRE_BigInt nnz_big = indptr[nrows] - offset;
   if ((HYPRE_BigInt)((HYPRE_Int)nnz_big) != nnz_big)
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "SetMatrixFromCSR: local nonzero count exceeds HYPRE_Int range");
      goto inputs_done;
   }
   nnz = (HYPRE_Int)nnz_big;
   if (nnz && (!col_indices || !data))
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "SetMatrixFromCSR: col_indices/data must be non-NULL when nnz > 0");
      goto inputs_done;
   }

   ncols_per_row = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   row_ids       = hypre_TAlloc(HYPRE_BigInt, nrows, HYPRE_MEMORY_HOST);
   if (host)
   {
      diag_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
      offd_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   }
   if (!ncols_per_row || !row_ids || (host && (!diag_sizes || !offd_sizes)))
   {
      status =
         hypredrv_AllocationFailure("Failed to allocate CSR construction scratch arrays");
      goto inputs_done;
   }
   /* Validate and describe each row once. Native assembly below only consumes
    * the checked counts and entries; offset slabs do not inspect leading padding. */
   for (HYPRE_Int row = 0; row < nrows; row++)
   {
      if (indptr[row + 1] < indptr[row] || indptr[row + 1] > indptr[nrows])
      {
         status = ERROR_INVALID_VAL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("SetMatrixFromCSR: invalid row offset at row %lld",
                              (long long)row);
         goto inputs_done;
      }
      ncols_per_row[row] = (HYPRE_Int)(indptr[row + 1] - indptr[row]);
      row_ids[row]       = row_start + (HYPRE_BigInt)row;
      for (HYPRE_BigInt entry = indptr[row]; entry < indptr[row + 1]; entry++)
      {
         HYPRE_BigInt column = col_indices[entry];
         if (column < 0 || (long long)column > global_end)
         {
            status = ERROR_INVALID_VAL;
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd("SetMatrixFromCSR: column %lld is outside [0, %lld]",
                                 (long long)column, global_end);
            goto inputs_done;
         }
         if (host)
         {
            if (column >= row_start && column <= row_end) diag_sizes[row]++;
            else offd_sizes[row]++;
         }
      }
   }
inputs_done:
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;

   /* Creation and assembly are collective. Agree on local failures before
    * entering either phase so no rank can strand its peers. */
   LINSYS_HYPRE_CALL(
      "SetMatrixFromCSR",
      HYPRE_IJMatrixCreate(comm, row_start, row_end, row_start, row_end, &matrix));
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL("SetMatrixFromCSR",
                     HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR));
   if (host)
   {
      LINSYS_HYPRE_CALL("SetMatrixFromCSR",
                        HYPRE_IJMatrixSetDiagOffdSizes(matrix, diag_sizes, offd_sizes));
   }
   LINSYS_HYPRE_CALL("SetMatrixFromCSR",
                     HYPRE_IJMatrixInitialize_v2(matrix, memory_location));
   if (nnz && !status)
   {
#ifdef HYPRE_USING_GPU
      if (!host)
      {
         /* Stage the caller's checked host slab once for device assembly. */
         HYPRE_Int     *d_ncols = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);
         HYPRE_BigInt  *d_rows  = hypre_TAlloc(HYPRE_BigInt, nrows, HYPRE_MEMORY_DEVICE);
         HYPRE_BigInt  *d_cols  = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
         HYPRE_Complex *d_data  = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
         if (!d_ncols || !d_rows || !d_cols || !d_data)
         {
            status = hypredrv_AllocationFailure(
               "Failed to allocate device CSR staging buffers");
         }
         else
         {
            hypre_TMemcpy(d_ncols, ncols_per_row, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            hypre_TMemcpy(d_rows, row_ids, HYPRE_BigInt, nrows, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            hypre_TMemcpy(d_cols, col_indices + offset, HYPRE_BigInt, nnz,
                          HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(d_data, data + offset, HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE,
                          HYPRE_MEMORY_HOST);
            LINSYS_HYPRE_CALL(
               "SetMatrixFromCSR",
               HYPRE_IJMatrixSetValues(matrix, nrows, d_ncols, d_rows, d_cols, d_data));
         }
         hypre_TFree(d_ncols, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_rows, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_cols, HYPRE_MEMORY_DEVICE);
         hypre_TFree(d_data, HYPRE_MEMORY_DEVICE);
      }
      else
#endif
      {
         LINSYS_HYPRE_CALL("SetMatrixFromCSR",
                           HYPRE_IJMatrixSetValues(matrix, nrows, ncols_per_row, row_ids,
                                                   col_indices + offset, data + offset));
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL("SetMatrixFromCSR", HYPRE_IJMatrixAssemble(matrix));

cleanup:
   hypre_TFree(ncols_per_row, HYPRE_MEMORY_HOST);
   hypre_TFree(row_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(diag_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(offd_sizes, HYPRE_MEMORY_HOST);
   status |= hypredrv_HypreConsumeErrors(comm);
   status = LinearSystemBuildStatusSync(comm, status);
   if (!status)
   {
      hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_A, matrix, true);
      hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_M, matrix, false);
      hypredrv->precmat_sequence_cache_args = NULL;
      matrix                                = NULL;
   }
   if (matrix) HYPRE_IJMatrixDestroy(matrix);
   return status;
}

/* Build a complete RHS collectively before publishing the output handle. */
uint32_t
hypredrv_LinearSystemSetRHSFromArray(hypredrv_t *hypredrv, HYPRE_BigInt row_start,
                                     HYPRE_BigInt row_end, const HYPRE_Real *values)
{
   MPI_Comm       comm   = hypredrv->comm;
   uint32_t       status = ERROR_NONE;
   HYPRE_IJVector rhs    = NULL;
   HYPRE_Int      nrows  = 0;
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
#ifdef HYPRE_USING_GPU
   memory_location =
      LinearSystemMemoryLocationGet(hypredrv->iargs ? &hypredrv->iargs->ls : NULL);
#endif
   if (!values)
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetRHSFromArray: values must be non-NULL");
   }
   if (!LinearSystemArrayRowCount("SetRHSFromArray", row_start, row_end, &nrows))
   {
      status = ERROR_INVALID_VAL;
   }
   if (!hypredrv->mat_A.handle)
   {
      status = ERROR_INVALID_VAL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "HYPREDRV_LinearSystemSetRHSFromArray: matrix must be set before RHS");
   }
   else
   {
      HYPRE_BigInt ilower = 0, iupper = -1, jlower = 0, jupper = -1;
      HYPRE_Int    code = HYPRE_IJMatrixGetLocalRange(hypredrv->mat_A.handle, &ilower,
                                                      &iupper, &jlower, &jupper);
      status |=
         LinearSystemBuildRecordError("SetRHSFromArray: query matrix local range", code);
      if (!code && (row_start != ilower || row_end != iupper))
      {
         status |= ERROR_INVALID_VAL;
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "HYPREDRV_LinearSystemSetRHSFromArray: RHS row range [%lld, %lld] "
            "does not match matrix row range [%lld, %lld]",
            (long long)row_start, (long long)row_end, (long long)ilower,
            (long long)iupper);
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL("SetRHSFromArray",
                     HYPRE_IJVectorCreate(comm, row_start, row_end, &rhs));
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL("SetRHSFromArray", HYPRE_IJVectorSetObjectType(rhs, HYPRE_PARCSR));
   LINSYS_HYPRE_CALL("SetRHSFromArray",
                     HYPRE_IJVectorInitialize_v2(rhs, memory_location));
   LINSYS_HYPRE_CALL("SetRHSFromArray",
                     LinearSystemSetHostValues(rhs, nrows, NULL, values));
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   LINSYS_HYPRE_CALL("SetRHSFromArray", HYPRE_IJVectorAssemble(rhs));

cleanup:
   status |= hypredrv_HypreConsumeErrors(comm);
   status = LinearSystemBuildStatusSync(comm, status);
   if (!status)
   {
      hypredrv_VectorSlot reference = hypredrv->vec_xref;
      if (hypredrv->vec_b.owned && reference.handle == hypredrv->vec_b.handle)
         reference = (hypredrv_VectorSlot){NULL, false};
      const hypredrv_VectorUpdate updates[] = {
         {&hypredrv->vec_b, rhs, true},
         {&hypredrv->vec_xref, reference.handle, reference.owned}};
      hypredrv_ObjectInstallVectors(hypredrv, 2, updates);
      rhs = NULL;
   }
   if (rhs) HYPRE_IJVectorDestroy(rhs);
   return status;
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

/* Parses the Matrix Market size line of a dense vector. */
static int
MatrixMarketReadVectorDims(const char *line, HYPRE_BigInt *M_out, HYPRE_BigInt *N_out)
{
   char     *rest = NULL;
   long long tmpM = strtoll(line, &rest, 10);
   long long tmpN = strtoll(rest, NULL, 10); /* any whitespace separates M and N */

   *M_out = (HYPRE_BigInt)tmpM;
   *N_out = (HYPRE_BigInt)tmpN;

   /* GCOVR_EXCL_BR_START */
   return (tmpM != 0 && tmpN != 0);
   /* GCOVR_EXCL_BR_STOP */
}

/* Rank-0 half of the Matrix Market RHS read: pulls the whole vector into
 * `all_values`. Returns its length, or -1 with the error state set. A partially
 * filled buffer is still handed back so the caller can free it. */
static HYPRE_BigInt
MatrixMarketReadVectorOnRoot(const char *rhs_filename, HYPRE_BigInt global_num_rows,
                             HYPRE_Complex **all_values_out)
{
   FILE          *file = NULL;
   char           line[1024];
   HYPRE_BigInt   M          = 0;
   HYPRE_BigInt   N          = 0;
   HYPRE_Complex *all_values = NULL;

   file = fopen(rhs_filename, "r");
   if (file == NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Cannot open file %s", rhs_filename);
      return -1;
   }

   /* Skip the banner and any comment lines. */
   do
   {
      if (fgets(line, sizeof(line), file) == NULL)
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Unexpected end of file or error reading %s", rhs_filename);
         fclose(file);
         return -1;
      }
   } while (line[0] == '%');

   if (!MatrixMarketReadVectorDims(line, &M, &N))
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Failed to read vector dimensions from %s", rhs_filename);
      fclose(file);
      return -1;
   }
   if (N != 1)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("File %s is not a vector (N=%lld)", rhs_filename,
                           (long long)N);
      fclose(file);
      return -1;
   }
   if (M != global_num_rows)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("RHS vector size %lld does not match matrix size %lld",
                           (long long)M, (long long)global_num_rows);
      fclose(file);
      return -1;
   }

   all_values      = hypre_TAlloc(HYPRE_Complex, M, HYPRE_MEMORY_HOST);
   *all_values_out = all_values;
   for (HYPRE_BigInt i = 0; i < M; i++)
   {
      char  *endptr  = NULL;
      double tmp_val = 0.0;

      if (fgets(line, sizeof(line), file) == NULL)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Error reading value for index %lld from %s", (long long)i,
                              rhs_filename);
         fclose(file);
         return -1;
      }

      tmp_val = strtod(line, &endptr);
      while (*endptr != '\0' && isspace((unsigned char)*endptr)) /* e.g. "\r\n" */
      {
         endptr++;
      }
      /* GCOVR_EXCL_BR_START */
      if (endptr == line || *endptr != '\0') /* GCOVR_EXCL_BR_STOP */
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Error converting value for index %lld from %s",
                              (long long)i, rhs_filename);
         fclose(file);
         return -1;
      }
      all_values[i] = (HYPRE_Complex)tmp_val;
   }

   fclose(file);

   return M;
}

static int
LinearSystemRHSMatrixMarketRead(MPI_Comm comm, const LS_args *args, HYPRE_IJMatrix mat,
                                const char *rhs_filename, HYPRE_IJVector *rhs_ptr)
{
   int                  myid            = 0;
   int                  num_procs       = 0;
   HYPRE_BigInt         M               = 0;
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
      M = MatrixMarketReadVectorOnRoot(rhs_filename, global_num_rows, &all_values);
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
   LinearSystemSetHostValues(*rhs_ptr, local_size_hypre, NULL, local_values);
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

/* A temporary view of an initialized input vector. IJ ranges retain the
 * caller's global origin; native ParVector ranges are relative to that origin. */
typedef struct
{
   hypre_ParVector *par;
   HYPRE_BigInt     first, last;
} LinearSystemVectorView;

/* Registered vectors are already initialized. Read their current layout
 * without revalidating input type or storage on every reset/previous-mode copy. */
static uint32_t
LinearSystemVectorViewGet(HYPRE_IJVector vector, const char *label,
                          LinearSystemVectorView *view)
{
   uint32_t status = ERROR_NONE;
   void    *object = NULL;
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorGetObject(vector, &object));
   LINSYS_HYPRE_CALL(label,
                     HYPRE_IJVectorGetLocalRange(vector, &view->first, &view->last));
   view->par = (hypre_ParVector *)object;
   return status;
}

static uint32_t
LinearSystemReadVector(HYPRE_IJVector vector, const char *label,
                       LinearSystemVectorView *view)
{
   uint32_t  status = ERROR_NONE;
   HYPRE_Int type   = -1;
   if (!vector)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: vector must be set first", label);
      return ERROR_INVALID_VAL;
   }
   LINSYS_HYPRE_CALL(label, HYPRE_IJVectorGetObjectType(vector, &type));
   if (status) return status;
   if (type != HYPRE_PARCSR)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: vector must have ParCSR object type", label);
      return ERROR_INVALID_VAL;
   }
   status = LinearSystemVectorViewGet(vector, label, view);
   if (status) return status;
   hypre_Vector *local = view->par ? hypre_ParVectorLocalVector(view->par) : NULL;
   if (!local || hypre_VectorSize(local) < 0 || hypre_VectorNumVectors(local) < 1 ||
       (hypre_VectorSize(local) > 0 && !hypre_VectorData(local)))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s: vector must be initialized", label);
      return ERROR_INVALID_VAL;
   }
   return ERROR_NONE;
}

static bool
LinearSystemVectorLayoutsMatch(const LinearSystemVectorView *a,
                               const LinearSystemVectorView *b)
{
   return a->first == b->first && a->last == b->last &&
          hypre_ParVectorGlobalSize(a->par) == hypre_ParVectorGlobalSize(b->par) &&
          hypre_ParVectorFirstIndex(a->par) == hypre_ParVectorFirstIndex(b->par) &&
          hypre_ParVectorLastIndex(a->par) == hypre_ParVectorLastIndex(b->par) &&
          hypre_VectorSize(hypre_ParVectorLocalVector(a->par)) ==
             hypre_VectorSize(hypre_ParVectorLocalVector(b->par)) &&
          hypre_ParVectorNumVectors(a->par) == hypre_ParVectorNumVectors(b->par);
}

/* Build fresh vectors after collective input validation. Agreement before
 * assembly keeps a local failure from stranding peers. */
static uint32_t
LinearSystemCreateVector(hypredrv_t *hypredrv, HYPRE_BigInt first, HYPRE_BigInt last,
                         HYPRE_IJVector *vector)
{
   uint32_t             status   = ERROR_NONE;
   HYPRE_MemoryLocation location = hypredrv->iargs && hypredrv->iargs->ls.exec_policy
                                      ? HYPRE_MEMORY_DEVICE
                                      : HYPRE_MEMORY_HOST;
   LINSYS_HYPRE_CALL("Solution allocation",
                     HYPRE_IJVectorCreate(hypredrv->comm, first, last, vector));
   LINSYS_HYPRE_CALL("Solution allocation",
                     HYPRE_IJVectorSetObjectType(*vector, HYPRE_PARCSR));
   LINSYS_HYPRE_CALL("Solution allocation",
                     HYPRE_IJVectorInitialize_v2(*vector, location));
   status = LinearSystemBuildStatusSync(hypredrv->comm, status);
   LINSYS_HYPRE_CALL("Solution allocation", HYPRE_IJVectorAssemble(*vector));
   return LinearSystemBuildStatusSync(hypredrv->comm, status);
}

uint32_t
hypredrv_LinearSystemSetRHS(hypredrv_t *hypredrv, HYPRE_IJVector vector)
{
   MPI_Comm               comm   = hypredrv->comm;
   uint32_t               status = ERROR_NONE;
   HYPRE_IJVector         rhs = NULL, reference = NULL;
   LinearSystemVectorView input = {0};
   if (vector)
   {
      status = LinearSystemReadVector(vector, "SetRHS", &input);
      if (!status && hypre_ParVectorNumVectors(input.par) != 1)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("SetRHS: vector must have one component");
         status = ERROR_INVALID_VAL;
      }
      if (!status)
      {
         hypredrv_ObjectPrepareExecution(hypredrv, vector, 0);
         status = LinearSystemBuildRecordError("SetRHS", HYPRE_GetError());
      }
      status |= hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
      if (!status)
      {
         hypredrv_VectorSlot ref = hypredrv->vec_xref;
         if (hypredrv->vec_b.handle != vector && hypredrv->vec_b.owned &&
             ref.handle == hypredrv->vec_b.handle)
            ref = (hypredrv_VectorSlot){NULL, false};
         const hypredrv_VectorUpdate updates[] = {
            {&hypredrv->vec_b, vector, hypredrv_AttachmentIsOwned(hypredrv->lib_mode)},
            {&hypredrv->vec_xref, ref.handle, ref.owned}};
         hypredrv_ObjectInstallVectors(hypredrv, 2, updates);
      }
      return status;
   }

   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetRHS(NULL): input arguments not parsed");
   }
   if (!hypredrv->mat_A.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetRHS(NULL): matrix must be set first");
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();
   const LS_args       *args  = &hypredrv->iargs->ls;
   HYPRE_BigInt         first = 0, last = -1, column_first = 0, column_last = -1;
   HYPRE_ParCSRMatrix   matrix            = NULL;
   HYPRE_MemoryLocation original_location = HYPRE_MEMORY_UNDEFINED;
   bool                 restore_matrix    = false;
   int                  mode              = args->rhs_mode;
   int source    = mode == 2 ? (args->sequence_filename[0] ? -1 : -2) : mode;
   int request[] = {source, source == -2 && args->type == 3}, minimum[2], maximum[2];
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, comm);
   hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "rhs");
   HYPREDRV_LOG_OBJECTF(3, hypredrv, "rhs setup begin (rhs_mode=%d)", mode);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("RHS source, mode, and file format must agree across ranks");
      status = ERROR_INVALID_VAL;
   }
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   LINSYS_HYPRE_CALL("SetRHS",
                     HYPRE_IJMatrixGetLocalRange(hypredrv->mat_A.handle, &first, &last,
                                                 &column_first, &column_last));
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   if (mode == 2)
   {
      int ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
      int ok    = 0;
      if (args->sequence_filename[0])
      {
         HYPREDRV_LOG_OBJECTF(3, hypredrv, "rhs source: sequence file '%s'",
                              args->sequence_filename);
         ok = hypredrv_LSSeqReadRHS(comm, args->sequence_filename, ls_id,
                                    LinearSystemMemoryLocationGet(args), &rhs);
      }
      else
      {
         char filename[MAX_FILENAME_LENGTH] = {0};
         if (!LinearSystemDataFilenameResolve(args, ls_id, args->rhs_filename,
                                              args->rhs_basename, filename,
                                              sizeof(filename)))
         {
            hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
            hypredrv_ErrorMsgAddInvalidFilename(filename);
            status = ERROR_FILE_NOT_FOUND;
         }
         if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
            goto cleanup;
         HYPREDRV_LOG_OBJECTF(3, hypredrv, "rhs source: '%s'", filename);
         if (args->type == 3)
            ok = LinearSystemRHSMatrixMarketRead(comm, args, hypredrv->mat_A.handle,
                                                 filename, &rhs);
         else
            ok = (status = LinearSystemIJVectorReadFromFile(
                     comm, filename, LinearSystemMemoryLocationGet(args), &rhs)) ==
                 ERROR_NONE;
      }

      status |= hypredrv_ErrorCodeGet();
      if (!ok || !rhs || HYPRE_GetError())
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAdd("Could not read the configured RHS vector");
         status |= ERROR_FILE_NOT_FOUND;
      }
      if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
         goto cleanup;
      status = LinearSystemReadVector(rhs, "RHS file", &input);
      if (!status && (input.first != first || input.last != last ||
                      hypre_ParVectorNumVectors(input.par) != 1))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "RHS file must have one component and match the matrix row range");
         status = ERROR_INVALID_VAL;
      }
      LINSYS_HYPRE_CALL(
         "SetRHS", hypredrv_IJVectorMigrate(rhs, LinearSystemMemoryLocationGet(args)));
   }
   else
   {
      HYPREDRV_LOG_OBJECTF(3, hypredrv, "rhs source: generated mode=%d", mode);
      status       = LinearSystemCreateVector(hypredrv, first, last, &rhs);
      void *object = NULL;
      LINSYS_HYPRE_CALL("SetRHS", HYPRE_IJVectorGetObject(rhs, &object));
      HYPRE_ParVector par_rhs = (HYPRE_ParVector)object;
      if (mode == 4)
      {
         if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
            goto cleanup;
         status =
            LinearSystemCreateVector(hypredrv, column_first, column_last, &reference);
         LINSYS_HYPRE_CALL("SetRHS", HYPRE_IJVectorGetObject(reference, &object));
         HYPRE_ParVector par_reference = (HYPRE_ParVector)object;
         HYPRE_Int       type          = -1;
         LINSYS_HYPRE_CALL("SetRHS",
                           HYPRE_IJMatrixGetObjectType(hypredrv->mat_A.handle, &type));
         LINSYS_HYPRE_CALL("SetRHS",
                           HYPRE_IJMatrixGetObject(hypredrv->mat_A.handle, &object));
         if (!status &&
             (type != HYPRE_PARCSR || !object ||
              !hypre_IJMatrixAssembleFlag((hypre_IJMatrix *)hypredrv->mat_A.handle)))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
            hypredrv_ErrorMsgAdd(
               "Random-solution RHS requires an assembled ParCSR matrix");
            status = ERROR_INVALID_VAL;
         }
         if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
            goto cleanup;
         matrix = (HYPRE_ParCSRMatrix)object;
         original_location =
            hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)matrix);
         HYPRE_MemoryLocation target = LinearSystemMemoryLocationGet(args);
         restore_matrix              = original_location != target;
         if (restore_matrix)
            LINSYS_HYPRE_CALL("SetRHS", hypredrv_ParCSRMatrixMigrate(matrix, target));
         LINSYS_HYPRE_CALL("SetRHS", HYPRE_ParVectorSetRandomValues(par_reference, 2023));
         if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
            goto cleanup;
         LINSYS_HYPRE_CALL(
            "SetRHS", HYPRE_ParCSRMatrixMatvec(1.0, matrix, par_reference, 0.0, par_rhs));
      }
      else if (mode == 3)
         LINSYS_HYPRE_CALL("SetRHS", HYPRE_ParVectorSetRandomValues(par_rhs, 2023));
      else
         LINSYS_HYPRE_CALL(
            "SetRHS", HYPRE_ParVectorSetConstantValues(par_rhs, mode == 0 ? 0.0 : 1.0));
   }

cleanup:
   if (restore_matrix)
      status |= LinearSystemBuildRecordError(
         "SetRHS: restore matrix storage",
         hypredrv_ParCSRMatrixMigrate(matrix, original_location));
   status |= hypredrv_HypreConsumeErrors(comm);
   status = LinearSystemBuildStatusSync(comm, status);
   if (!status)
   {
      const hypredrv_VectorUpdate updates[] = {{&hypredrv->vec_b, rhs, true},
                                               {&hypredrv->vec_xref, reference, true}};
      hypredrv_ObjectInstallVectors(hypredrv, 2, updates);
      rhs = reference = NULL;
   }
   if (rhs) HYPRE_IJVectorDestroy(rhs);
   if (reference) HYPRE_IJVectorDestroy(reference);
   hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "rhs");
   HYPREDRV_LOG_OBJECTF(3, hypredrv, "rhs setup end (status=0x%x)", status);
   return status;
}

uint32_t
hypredrv_LinearSystemSetSolution(hypredrv_t *hypredrv, HYPRE_IJVector vector)
{
   uint32_t               status      = ERROR_NONE;
   LinearSystemVectorView input       = {0};
   HYPRE_IJVector         replacement = NULL;
   if (vector)
   {
      /* Attaching an existing buffer stays local, and is allowed before parsing. */
      status = LinearSystemReadVector(vector, "SetSolution", &input);
      if (!status)
      {
         hypredrv_ObjectPrepareExecution(hypredrv, vector, 0);
         status = LinearSystemBuildRecordError("SetSolution", HYPRE_GetError());
      }
      status |= hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
      if (!status)
      {
         const hypredrv_VectorUpdate update = {&hypredrv->vec_x, vector, false};
         hypredrv_ObjectInstallVectors(hypredrv, 1, &update);
      }
      return status;
   }

   if (!hypredrv->vec_b.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("SetSolution(NULL): RHS must be set first");
      status = ERROR_INVALID_VAL;
   }
   else
      status =
         LinearSystemVectorViewGet(hypredrv->vec_b.handle, "SetSolution: RHS", &input);
   if ((status = LinearSystemBuildStatusSync(hypredrv->comm, status)) == ERROR_NONE)
   {
      hypredrv_ObjectApplyRuntimeSettings(hypredrv);
      status = LinearSystemCreateVector(hypredrv, input.first, input.last, &replacement);
   }
   status |= hypredrv_HypreConsumeErrors(hypredrv->comm);
   status = LinearSystemBuildStatusSync(hypredrv->comm, status);
   if (!status)
   {
      const hypredrv_VectorUpdate update = {&hypredrv->vec_x, replacement, true};
      hypredrv_ObjectInstallVectors(hypredrv, 1, &update);
      replacement = NULL;
   }
   if (replacement) HYPRE_IJVectorDestroy(replacement);
   return status;
}

uint32_t
hypredrv_LinearSystemSetInitialGuess(hypredrv_t *hypredrv, hypredrv_GuessUpdate update,
                                     HYPRE_IJVector vector)
{
   MPI_Comm comm    = hypredrv->comm;
   uint32_t status  = ERROR_NONE;
   bool     install = update == GUESS_REPLACE || !hypredrv->vec_x.handle;
   if (update == GUESS_ENSURE_SOLUTION)
   {
      int local_missing = install, any_missing = 0;
      MPI_Allreduce(&local_missing, &any_missing, 1, MPI_INT, MPI_MAX, comm);
      if (!any_missing)
      {
         status = hypredrv_ErrorCodeGet();
         status |= hypredrv_HypreConsumeErrors(comm);
         return LinearSystemBuildStatusSync(comm, status);
      }
   }
   const LS_args         *args = hypredrv->iargs ? &hypredrv->iargs->ls : NULL;
   LinearSystemVectorView rhs = {0}, guess_view = {0};
   HYPRE_IJVector         guess = vector, solution = NULL;
   int                    source = -3, minimum, maximum;
   if (vector) source = -2;
   else if (args) source = args->x0_filename[0] ? -1 : args->init_guess_mode;
   else
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Initial guess(NULL): input arguments not parsed");
      status = ERROR_INVALID_VAL;
   }
   HYPREDRV_LOG_OBJECTF(3, hypredrv, "initial guess setup begin (source=%d)", source);
   MPI_Allreduce(&source, &minimum, 1, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(&source, &maximum, 1, MPI_INT, MPI_MAX, comm);
   if (minimum != maximum)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Initial guess source/mode must agree across ranks");
      status = ERROR_INVALID_VAL;
   }
   if (!hypredrv->vec_b.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Initial guess: RHS must be set first");
      status |= ERROR_INVALID_VAL;
   }
   else
      status |=
         LinearSystemVectorViewGet(hypredrv->vec_b.handle, "Initial guess: RHS", &rhs);
   if (vector)
   {
      status |= LinearSystemReadVector(vector, "Initial guess", &guess_view);
      if (!status && !LinearSystemVectorLayoutsMatch(&rhs, &guess_view))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Initial guess layout must match the RHS");
         status = ERROR_INVALID_VAL;
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   if (!vector && source == -1)
   {
      HYPRE_MemoryLocation location =
         args->exec_policy ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
      HYPREDRV_LOG_OBJECTF(3, hypredrv, "initial guess source: '%s'", args->x0_filename);
      status =
         LinearSystemIJVectorReadFromFile(comm, args->x0_filename, location, &guess);
      if (status) goto cleanup;
      status = LinearSystemReadVector(guess, "Initial guess file", &guess_view);
      if (!status && !LinearSystemVectorLayoutsMatch(&rhs, &guess_view))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Initial guess file layout must match the RHS");
         status = ERROR_INVALID_VAL;
      }
   }
   else if (!vector)
   {
      status       = LinearSystemCreateVector(hypredrv, rhs.first, rhs.last, &guess);
      void *object = NULL;
      LINSYS_HYPRE_CALL("Initial guess", HYPRE_IJVectorGetObject(guess, &object));
      HYPRE_ParVector par_guess = (HYPRE_ParVector)object;
      switch (source)
      {
         case 1:
            LINSYS_HYPRE_CALL("Initial guess",
                              HYPRE_ParVectorSetConstantValues(par_guess, 1));
            break;
         case 3:
            LINSYS_HYPRE_CALL("Initial guess",
                              HYPRE_ParVectorSetRandomValues(par_guess, 2023));
            break;
         case 4:
         {
            LinearSystemVectorView previous   = {0};
            int                    compatible = 0, all_compatible = 0;
            if (!status && hypredrv->vec_x.handle)
            {
               status     = LinearSystemVectorViewGet(hypredrv->vec_x.handle,
                                                      "Previous solution", &previous);
               compatible = !status && LinearSystemVectorLayoutsMatch(&rhs, &previous);
            }
            if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE)
               goto cleanup;
            MPI_Allreduce(&compatible, &all_compatible, 1, MPI_INT, MPI_MIN, comm);
            if (all_compatible)
               LINSYS_HYPRE_CALL(
                  "Initial guess",
                  HYPRE_ParVectorCopy((HYPRE_ParVector)previous.par, par_guess));
            else
               HYPREDRV_LOG_OBJECTF(2, hypredrv,
                                    "no compatible previous solution; using zeros");
            break;
         }
         default:
            /* Initialization already supplied zeros. */
            break;
      }
   }
   if ((status = LinearSystemBuildStatusSync(comm, status)) != ERROR_NONE) goto cleanup;
   /* Capture the previous solution before allocating its replacement. Both new
    * buffers are complete before ownership or installed metadata can change. */
   status = LinearSystemCreateVector(hypredrv, rhs.first, rhs.last, &solution);
   if (hypredrv->iargs && (vector || source == -1))
   {
      HYPRE_MemoryLocation location =
         args->exec_policy ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
      LINSYS_HYPRE_CALL("Initial guess", hypredrv_IJVectorMigrate(guess, location));
   }

cleanup:
   status |= hypredrv_HypreConsumeErrors(comm);
   status = LinearSystemBuildStatusSync(comm, status);
   if (!status && install)
   {
      const hypredrv_VectorUpdate updates[] = {
         {&hypredrv->vec_x0, guess,
          !vector || hypredrv_AttachmentIsOwned(hypredrv->lib_mode)},
         {&hypredrv->vec_x, solution, true}};
      hypredrv_ObjectInstallVectors(hypredrv, 2, updates);
      guess = solution = NULL;
   }
   if (!vector && guess) HYPRE_IJVectorDestroy(guess);
   if (solution) HYPRE_IJVectorDestroy(solution);
   HYPREDRV_LOG_OBJECTF(3, hypredrv, "initial guess setup end (status=0x%x)", status);
   return status;
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetReferenceSolution
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemSetReferenceSolution(hypredrv_t *hypredrv, HYPRE_IJVector vector)
{
   uint32_t               status      = ERROR_NONE;
   LinearSystemVectorView input       = {0};
   HYPRE_IJVector         replacement = vector;
   const LS_args         *args        = hypredrv->iargs ? &hypredrv->iargs->ls : NULL;
   if (!vector)
   {
      int source = args ? (args->xref_filename[0] || args->xref_basename[0]) : -1;
      int minimum, maximum;
      MPI_Allreduce(&source, &minimum, 1, MPI_INT, MPI_MIN, hypredrv->comm);
      MPI_Allreduce(&source, &maximum, 1, MPI_INT, MPI_MAX, hypredrv->comm);
      if (!args)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Reference solution(NULL): input arguments not parsed");
      }
      if (minimum != maximum)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Reference solution source must agree across ranks");
      }
      if (hypredrv_DistributedErrorStateSync(hypredrv->comm))
         return hypredrv_ErrorCodeGet();
      if (!source) return ERROR_NONE;
      char filename[MAX_FILENAME_LENGTH] = {0};
      int  ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
      hypredrv_ObjectApplyRuntimeSettings(hypredrv);
      if (!LinearSystemDataFilenameResolve(args, ls_id, args->xref_filename,
                                           args->xref_basename, filename,
                                           sizeof(filename)))
      {
         hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
         hypredrv_ErrorMsgAddInvalidFilename(filename);
         status = ERROR_FILE_NOT_FOUND;
      }
      HYPREDRV_LOG_OBJECTF(3, hypredrv, "reference solution source: '%s'", filename);
      status |= LinearSystemBuildRecordError("Reference solution", HYPRE_GetError());
      if ((status = LinearSystemBuildStatusSync(hypredrv->comm, status)) != ERROR_NONE)
         goto cleanup;
      status = LinearSystemIJVectorReadFromFile(
         hypredrv->comm, filename, LinearSystemMemoryLocationGet(args), &replacement);
      if (status) goto cleanup;
   }
   status = LinearSystemReadVector(replacement, "Reference solution", &input);
   if (!status && hypre_ParVectorNumVectors(input.par) != 1)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Reference solution must have one component");
      status = ERROR_INVALID_VAL;
   }
   if (!status)
   {
      hypredrv_ObjectPrepareExecution(hypredrv, replacement, 0);
      status = LinearSystemBuildRecordError("Reference solution", HYPRE_GetError());
   }

cleanup:
   /* Existing handles are attached locally; configured reads are collective. */
   status |= hypredrv_HypreConsumeErrors(vector ? MPI_COMM_SELF : hypredrv->comm);
   if (!vector) status = LinearSystemBuildStatusSync(hypredrv->comm, status);
   if (!status)
   {
      const hypredrv_VectorUpdate update = {
         &hypredrv->vec_xref, replacement,
         !vector || hypredrv_AttachmentIsOwned(hypredrv->lib_mode)};
      hypredrv_ObjectInstallVectors(hypredrv, 1, &update);
      replacement = NULL;
   }
   if (!vector && replacement) HYPRE_IJVectorDestroy(replacement);
   return status;
}

/* Reset is local, like a vector copy. The selected pair can change after
 * guess registration through SetSolution, so verify that pair before writing. */
uint32_t
hypredrv_LinearSystemResetInitialGuess(hypredrv_t *hypredrv)
{
   LinearSystemVectorView guess = {0}, solution = {0};
   uint32_t               status = ERROR_NONE;
   Stats                 *stats  = hypredrv->stats;
   int                    ls_id  = hypredrv_StatsGetLinearSystemID(stats);
   char                   log_name_buf[32];
   const char            *name =
      hypredrv_StatsGetLogObjectName(stats, log_name_buf, sizeof(log_name_buf));
   if (ls_id < 0) ls_id = 0;
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_BEGIN, "reset_x0");
   HYPREDRV_LOG_COMMF(3, hypredrv->comm, name, ls_id, "initial guess reset begin");
   if (!hypredrv->vec_x0.handle || !hypredrv->vec_x.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "ResetInitialGuess: guess and working solution must be set first");
      status = ERROR_INVALID_VAL;
   }
   else
   {
      status =
         LinearSystemVectorViewGet(hypredrv->vec_x0.handle, "ResetInitialGuess", &guess);
      status |= LinearSystemVectorViewGet(hypredrv->vec_x.handle, "ResetInitialGuess",
                                          &solution);
   }
   if (!status && !LinearSystemVectorLayoutsMatch(&guess, &solution))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("ResetInitialGuess: guess and solution layouts must match");
      status = ERROR_INVALID_VAL;
   }
   if (!status && guess.par != solution.par &&
       hypre_VectorData(hypre_ParVectorLocalVector(guess.par)) !=
          hypre_VectorData(hypre_ParVectorLocalVector(solution.par)))
   {
      LINSYS_HYPRE_CALL(
         "ResetInitialGuess",
         HYPRE_ParVectorCopy((HYPRE_ParVector)guess.par, (HYPRE_ParVector)solution.par));
   }
   if (!status)
   {
      hypredrv_ScalingApplyToVector(hypredrv->scaling_ctx, hypredrv->vec_x.handle,
                                    SCALING_VECTOR_INITIAL_GUESS);
   }
   hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
   hypredrv_StatsAnnotate(stats, HYPREDRV_ANNOTATE_END, "reset_x0");
   HYPREDRV_LOG_COMMF(3, hypredrv->comm, name, ls_id, "initial guess reset end");
   return hypredrv_ErrorCodeGet();
}

#undef LINSYS_HYPRE_CALL

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemLogBlockFrobenius
 *
 * Compute ||A_ij||_F for every pair of dofmap labels. Off-process column labels
 * are exchanged with the matrix communication package, so every ParCSR entry is
 * included exactly once. This intentionally runs only at log level 3 or above.
 *-----------------------------------------------------------------------------*/

/* Collective gate for the block diagnostics: every rank has to agree that the
 * computation can proceed, so a locally-unusable layout never leaves peers
 * blocked in the reductions further down. Logs `reason` when any rank objects. */
static int
BlockDiagnosticsAgree(MPI_Comm comm, int local_valid, const char *log_object_name,
                      int ls_id, const char *reason)
{
   int global_valid = 0;

   MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm);
   if (!global_valid)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "%s", reason);
   }

   return global_valid;
}

/* Validates that this matrix/dofmap pair can be diagnosed at all, and captures
 * the ParCSR views the accumulation needs. Returns zero when any rank objects. */
static int
BlockFrobeniusPrepare(MPI_Comm comm, HYPRE_IJMatrix matrix, const IntArray *dofmap,
                      const char *log_object_name, int ls_id,
                      hypre_ParCSRMatrix **par_matrix_out, hypre_CSRMatrix **diag_out,
                      hypre_CSRMatrix **offd_out, HYPRE_Int *num_rows_out,
                      HYPRE_Int *num_cols_offd_out, int *num_blocks_out,
                      int *num_label_slots_out)
{
   void               *object      = NULL;
   hypre_ParCSRMatrix *par_matrix  = NULL;
   hypre_CSRMatrix    *diag        = NULL;
   hypre_CSRMatrix    *offd        = NULL;
   int                 local_valid = matrix && dofmap;
   if (local_valid)
   {
      HYPRE_IJMatrixGetObject(matrix, &object);
      local_valid = object != NULL;
   }
   if (!BlockDiagnosticsAgree(
          comm, local_valid, log_object_name, ls_id,
          "block Frobenius diagnostics skipped: matrix or dofmap missing"))
   {
      return 0;
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
   if (!BlockDiagnosticsAgree(
          comm, local_valid, log_object_name, ls_id,
          "block Frobenius diagnostics skipped: incompatible matrix/dofmap layout"))
   {
      return 0;
   }

   int max_label = -1;
   local_valid   = DofmapDiagnosticLabelsValid(dofmap, &max_label);
   if (!BlockDiagnosticsAgree(
          comm, local_valid, log_object_name, ls_id,
          "block Frobenius diagnostics skipped: invalid dofmap labels"))
   {
      return 0;
   }

   int num_blocks      = (int)dofmap->g_unique_size;
   int num_label_slots = max_label + 1;
   if (!DofmapDiagnosticMetadataAgrees(comm, dofmap, max_label))
   {
      HYPREDRV_LOG_COMMF(
         3, comm, log_object_name, ls_id,
         "block Frobenius diagnostics skipped: inconsistent dofmap metadata");
      return 0;
   }

   *par_matrix_out      = par_matrix;
   *diag_out            = diag;
   *offd_out            = offd;
   *num_rows_out        = num_rows;
   *num_cols_offd_out   = num_cols_offd;
   *num_blocks_out      = num_blocks;
   *num_label_slots_out = num_label_slots;

   return 1;
}

/* Prints one log line per block row: the relative Frobenius norm and nnz count
 * of every block in that row. */
static void
BlockFrobeniusReportRows(MPI_Comm comm, const char *log_object_name, int ls_id,
                         int num_blocks, const int *block_labels,
                         const DofLabelMap *dof_labels, const double *global_stats,
                         const long long *global_counts)
{
   const size_t     num_block_pairs = (size_t)num_blocks * (size_t)num_blocks;
   const double    *global_norm_sq  = global_stats;
   const double    *global_sum      = global_stats + num_block_pairs;
   const double    *global_abs_sum  = global_stats + (2 * num_block_pairs);
   const long long *global_nnz      = global_counts;
   const long long *global_positive = global_counts + num_block_pairs;
   const long long *global_negative = global_counts + (2 * num_block_pairs);
   const long long *global_zero     = global_counts + (3 * num_block_pairs);
   char            *line            = NULL;

   size_t line_capacity = 256 + ((size_t)num_blocks * 256);
   line                 = hypre_TAlloc(char, line_capacity, HYPRE_MEMORY_HOST);
   if (!line)
   {
      return;
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
   hypre_TFree(line, HYPRE_MEMORY_HOST);
}

/* Exchanges the dof labels of the halo columns so off-diagonal entries can be
 * attributed to a block pair. Returns zero when any rank could not build the
 * halo; the caller owns (and frees) the buffers either way. */
static int
BlockFrobeniusExchangeHaloLabels(MPI_Comm comm, hypre_ParCSRMatrix *par_matrix,
                                 const IntArray *dofmap, HYPRE_Int num_cols_offd,
                                 const char *log_object_name, int ls_id,
                                 HYPRE_Int **send_labels_out, HYPRE_Int **offd_labels_out,
                                 int **offd_block_labels_out)
{
   hypre_ParCSRCommPkg    *comm_pkg          = NULL;
   hypre_ParCSRCommHandle *comm_handle       = NULL;
   HYPRE_Int              *send_labels       = NULL;
   HYPRE_Int              *offd_labels       = NULL;
   int                    *offd_block_labels = NULL;
   HYPRE_Int               send_size         = 0;

   if (!hypre_ParCSRMatrixCommPkg(par_matrix))
   {
      hypre_MatvecCommPkgCreate(par_matrix);
   }
   comm_pkg = hypre_ParCSRMatrixCommPkg(par_matrix);
   if (!BlockDiagnosticsAgree(
          comm, comm_pkg || num_cols_offd == 0, log_object_name, ls_id,
          "block Frobenius diagnostics skipped: matrix halo unavailable"))
   {
      return 0;
   }

   if (comm_pkg)
   {
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      send_size         = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_labels       = hypre_TAlloc(HYPRE_Int, send_size, HYPRE_MEMORY_HOST);
      offd_labels       = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_HOST);
      offd_block_labels = hypre_TAlloc(int, num_cols_offd, HYPRE_MEMORY_HOST);
   }
   *send_labels_out       = send_labels;
   *offd_labels_out       = offd_labels;
   *offd_block_labels_out = offd_block_labels;

   if (!BlockDiagnosticsAgree(
          comm,
          (send_size == 0 || send_labels) &&
             (num_cols_offd == 0 || (offd_labels && offd_block_labels)),
          log_object_name, ls_id,
          "block Frobenius diagnostics skipped: halo allocation failed"))
   {
      return 0;
   }

   if (comm_pkg)
   {
      for (HYPRE_Int i = 0; i < send_size; i++)
      {
         HYPRE_Int local_col = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);

         send_labels[i] = (HYPRE_Int)dofmap->data[local_col];
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_labels, offd_labels);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      for (HYPRE_Int i = 0; i < num_cols_offd; i++)
      {
         offd_block_labels[i] = (int)offd_labels[i];
      }
   }

   return 1;
}

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

   hypre_ParCSRMatrix *par_matrix        = NULL;
   hypre_CSRMatrix    *diag              = NULL;
   hypre_CSRMatrix    *offd              = NULL;
   HYPRE_Int          *diag_i            = NULL;
   HYPRE_Int          *diag_j            = NULL;
   HYPRE_Complex      *diag_a            = NULL;
   HYPRE_Int          *offd_i            = NULL;
   HYPRE_Int          *offd_j            = NULL;
   HYPRE_Complex      *offd_a            = NULL;
   int                 owns_diag_copy    = 0;
   int                 owns_offd_copy    = 0;
   HYPRE_Int          *send_labels       = NULL;
   HYPRE_Int          *offd_labels       = NULL;
   int                *offd_block_labels = NULL;
   int                *block_labels      = NULL;
   int                *label_to_pos      = NULL;
   double             *local_stats       = NULL;
   double             *global_stats      = NULL;
   long long          *local_counts      = NULL;
   long long          *global_counts     = NULL;
   HYPRE_Int           num_rows          = 0;
   HYPRE_Int           num_cols_offd     = 0;
   int                 num_blocks        = 0;
   int                 num_label_slots   = 0;
   int                 local_valid       = 0;

   if (!BlockFrobeniusPrepare(comm, matrix, dofmap, log_object_name, ls_id, &par_matrix,
                              &diag, &offd, &num_rows, &num_cols_offd, &num_blocks,
                              &num_label_slots))
   {
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
   if (!BlockDiagnosticsAgree(comm, local_valid, log_object_name, ls_id,
                              "block Frobenius diagnostics skipped: allocation failed"))
   {
      goto cleanup;
   }

   double    *local_norm_sq  = local_stats;
   double    *local_sum      = local_stats + num_block_pairs;
   double    *local_abs_sum  = local_stats + (2 * num_block_pairs);
   double    *global_norm_sq = global_stats;
   long long *local_nnz      = local_counts;
   long long *local_positive = local_counts + num_block_pairs;
   long long *local_negative = local_counts + (2 * num_block_pairs);
   long long *local_zero     = local_counts + (3 * num_block_pairs);

   DofmapDiagnosticMapFill(dofmap, num_label_slots, label_to_pos, block_labels);

   if (!BlockFrobeniusExchangeHaloLabels(comm, par_matrix, dofmap, num_cols_offd,
                                         log_object_name, ls_id, &send_labels,
                                         &offd_labels, &offd_block_labels))
   {
      goto cleanup;
   }

   local_valid = GetCSRHostView(diag, &diag_i, &diag_j, &diag_a, &owns_diag_copy) &&
                 GetCSRHostView(offd, &offd_i, &offd_j, &offd_a, &owns_offd_copy);
   if (!BlockDiagnosticsAgree(
          comm, local_valid, log_object_name, ls_id,
          "block Frobenius diagnostics skipped: CSR host copy failed"))
   {
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

   BlockFrobeniusReportRows(comm, log_object_name, ls_id, num_blocks, block_labels,
                            dof_labels, global_stats, global_counts);

cleanup:
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
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
hypredrv_LinearSystemSetPrecMatrix(hypredrv_t *hypredrv, HYPRE_IJMatrix mat)
{
   MPI_Comm comm = hypredrv->comm;
   if (!hypredrv->iargs)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Preconditioner matrix input requires parsed arguments");
   }
   if (mat)
   {
      /* External registration remains local. Configured reads below are collective. */
      if (hypredrv_ErrorCodeActive()) return hypredrv_ErrorCodeGet();
      uint32_t status = hypredrv_IJMatrixValidate(mat, "SetPrecMatrix");
      if (!status) hypredrv_ObjectPrepareExecution(hypredrv, mat, 1);
      status |= hypredrv_HypreConsumeErrors(MPI_COMM_SELF);
      if (status) return status;
      hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_M, mat,
                                   hypredrv_AttachmentIsOwned(hypredrv->lib_mode));
      hypredrv->precmat_sequence_cache_args = NULL;
      return hypredrv_ErrorCodeGet();
   }
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();

   char        log_name_buf[32];
   const char *log_object_name =
      hypredrv_StatsGetLogObjectName(hypredrv->stats, log_name_buf, sizeof(log_name_buf));
   const LS_args *args          = &hypredrv->iargs->ls;
   int            ls_id         = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
   int            precmat_ls_id = args->precmat_sequence_system_id >= 0
                                     ? (int)args->precmat_sequence_system_id
                                     : ls_id;
   enum
   {
      MAIN,
      FILE_INPUT,
      SEQUENCE
   };
   int  source = MAIN;
   bool fixed_sequence =
      args->precmat_sequence_filename[0] && args->precmat_sequence_system_id >= 0;
   char matrix_filename[MAX_FILENAME_LENGTH] = {0};
   if (args->precmat_sequence_filename[0])
   {
      if (!args->sequence_filename[0] ||
          strcmp(args->precmat_sequence_filename, args->sequence_filename) ||
          precmat_ls_id != ls_id)
         source = SEQUENCE;
   }
   else
   {
      char main_filename[MAX_FILENAME_LENGTH] = {0};
      LinearSystemDataFilenameResolve(args, ls_id, args->precmat_filename,
                                      args->precmat_basename, matrix_filename,
                                      sizeof(matrix_filename));
      LinearSystemDataFilenameResolve(args, ls_id, args->matrix_filename,
                                      args->matrix_basename, main_filename,
                                      sizeof(main_filename));
      if (matrix_filename[0] && strcmp(matrix_filename, main_filename))
         source = FILE_INPUT;
   }
   int request[] = {source, source == SEQUENCE ? precmat_ls_id : 0, fixed_sequence};
   int minimum[3], maximum[3];
   MPI_Allreduce(request, minimum, 3, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 3, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1] || minimum[2] != maximum[2])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Preconditioner matrix source and system index must agree across ranks");
   }
   if (source == MAIN && !hypredrv->mat_A.handle)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd(
         "Default preconditioner matrix requires an installed system matrix");
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();

   int cached = fixed_sequence && LinearSystemHasCachedPrecMatrix(hypredrv);
   MPI_Allreduce(MPI_IN_PLACE, &cached, 1, MPI_INT, MPI_MIN, comm);
   if (cached)
   {
      HYPREDRV_LOG_OBJECTF(
         3, hypredrv,
         "reusing fixed preconditioner matrix from sequence file '%s', system %d",
         args->precmat_sequence_filename, precmat_ls_id);
      return hypredrv_ErrorCodeGet();
   }

   hypredrv_ObjectApplyRuntimeSettings(hypredrv);
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                      "preconditioner matrix setup begin");
   HYPRE_IJMatrix replacement = NULL;
   if (source == MAIN)
   {
      replacement = hypredrv->mat_A.handle;
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: reusing main %smatrix",
                         args->precmat_sequence_filename[0] ? "sequence " : "");
   }
   else if (source == SEQUENCE)
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: sequence file '%s', system %d",
                         args->precmat_sequence_filename, precmat_ls_id);
      hypredrv_LSSeqReadMatrix(comm, args->precmat_sequence_filename, precmat_ls_id,
                               LinearSystemMemoryLocationGet(args), &replacement);
   }
   else
   {
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                         "preconditioner matrix source: '%s'", matrix_filename);
      LinearSystemIJMatrixReadFromFile(comm, args, matrix_filename, &replacement);
   }
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      if (replacement && replacement != hypredrv->mat_A.handle)
         HYPRE_IJMatrixDestroy(replacement);
      HYPREDRV_LOG_COMMF(
         2, comm, log_object_name, ls_id, "preconditioner matrix read failed from '%s'",
         source == SEQUENCE ? args->precmat_sequence_filename : matrix_filename);
      return hypredrv_ErrorCodeGet();
   }
   hypredrv_ObjectReplaceMatrix(hypredrv, &hypredrv->mat_M, replacement,
                                replacement != hypredrv->mat_A.handle);
   hypredrv->precmat_sequence_cache_args =
      fixed_sequence && hypredrv->mat_M.owned ? hypredrv->iargs : NULL;
   HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "preconditioner matrix setup end");
   return hypredrv_ErrorCodeGet();
}

uint32_t
hypredrv_LinearSystemSetDofmap(hypredrv_t *hypredrv, hypredrv_DofmapSource source,
                               int size, int num_dof_types, const int *labels)
{
   MPI_Comm comm = hypredrv->comm;
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();
   const LS_args *args      = hypredrv->iargs ? &hypredrv->iargs->ls : NULL;
   int            file_kind = 0;
   if (source == DOFMAP_FROM_INPUT && args)
   {
      if (args->sequence_filename[0]) file_kind = 2;
      else if (args->dofmap_filename[0] || args->dofmap_basename[0]) file_kind = 1;
   }
   int request[] = {(int)source, file_kind}, minimum[2], maximum[2];
   MPI_Allreduce(request, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(request, maximum, 2, MPI_INT, MPI_MAX, comm);
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1] ||
       source < DOFMAP_FROM_ARRAY || source > DOFMAP_FROM_INPUT ||
       (source == DOFMAP_FROM_INPUT && !args))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Dofmap sources must agree across ranks; configured input "
                           "requires parsed arguments");
   }
   if (hypredrv_DistributedErrorStateSync(comm)) return hypredrv_ErrorCodeGet();

   IntArray *replacement = NULL;
   if (source == DOFMAP_FROM_ARRAY)
   {
      hypredrv_IntArrayBuild(comm, size, labels, &replacement);
   }
   else if (source == DOFMAP_INTERLEAVED || source == DOFMAP_CONTIGUOUS)
   {
      hypredrv_IntArrayBuildPattern(comm, size, num_dof_types,
                                    source == DOFMAP_INTERLEAVED, &replacement);
   }
   else
   {
      int         ls_id = hypredrv_StatsGetLinearSystemID(hypredrv->stats) + 1;
      char        log_name_buf[32];
      const char *log_object_name = hypredrv_StatsGetLogObjectName(
         hypredrv->stats, log_name_buf, sizeof(log_name_buf));
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap read begin");
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_BEGIN, "dofmap");
      if (file_kind == 2)
      {
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                            "dofmap source: sequence file '%s'", args->sequence_filename);
         hypredrv_LSSeqReadDofmap(comm, args->sequence_filename, ls_id, &replacement);
      }
      else if (file_kind == 1)
      {
         char filename[MAX_FILENAME_LENGTH] = {0};
         if (!LinearSystemDataFilenameResolve(args, ls_id, args->dofmap_filename,
                                              args->dofmap_basename, filename,
                                              sizeof(filename)))
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         }
         if (!hypredrv_DistributedErrorStateSync(comm))
         {
            HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap source: '%s'",
                               filename);
            hypredrv_IntArrayParRead(comm, filename, &replacement);
         }
      }
      else
      {
         replacement = hypredrv_IntArrayCreate(0);
         HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id,
                            "dofmap source: default empty");
      }
      hypredrv_StatsAnnotate(hypredrv->stats, HYPREDRV_ANNOTATE_END, "dofmap");
      HYPREDRV_LOG_COMMF(3, comm, log_object_name, ls_id, "dofmap read end");
   }
   if (!replacement && !hypredrv_ErrorCodeActive())
      hypredrv_AllocationFailure("Failed to construct the DOF map");
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      hypredrv_IntArrayDestroy(&replacement);
      return hypredrv_ErrorCodeGet();
   }

   IntArray *previous = hypredrv->dofmap;
   int       changed =
      !previous || previous->size != replacement->size ||
      memcmp(previous->data, replacement->data, replacement->size * sizeof(int)) != 0;
   int any_changed = 0;
   MPI_Allreduce(&changed, &any_changed, 1, MPI_INT, MPI_MAX, comm);
   if (any_changed && hypredrv->precon &&
       (hypredrv->precon->method == PRECON_MGR ||
        (hypredrv->precon->method == PRECON_BOOMERAMG &&
         hypredrv->iargs->precon.amg.coarsening.num_functions > 1)))
   {
      hypredrv_ObjectDestroySolvers(hypredrv);
   }
   hypredrv_VectorTagsClear(hypredrv);
   hypredrv->dofmap = replacement;
   if (hypredrv->iargs && hypredrv->iargs->precon_method == PRECON_MGR)
      hypredrv_MGRSetDofmap(&hypredrv->iargs->precon.mgr, replacement);
   hypredrv_IntArrayDestroy(&previous);
   return ERROR_NONE;
}

/*-----------------------------------------------------------------------------
 * TODO: leverage internal hypre APIs for device exec
 *-----------------------------------------------------------------------------*/

/* L1 and Linf need element-wise host loops, so device-resident data is migrated
 * to the host and restored afterwards. An unrecognised norm type yields -1.0. */
static void
LinearSystemComputeHostNorm(HYPRE_IJVector vec, HYPRE_ParVector par_vec,
                            const hypre_Vector *seq_vec, const HYPRE_Complex *data,
                            HYPRE_Int size, MPI_Comm comm, const char *norm_type,
                            double *norm)
{
   double local_norm  = 0.0;
   double global_norm = 0.0;

#if !defined(HYPRE_USING_GPU)
   (void)vec;
   (void)par_vec;
   (void)seq_vec;
#endif

#if defined(HYPRE_USING_GPU)
   /* Manual loops require host-accessible data; save memory location to restore later
    */
   HYPRE_MemoryLocation orig_memloc = hypre_VectorMemoryLocation(seq_vec);
   if (orig_memloc != HYPRE_MEMORY_HOST)
   {
      hypredrv_IJVectorMigrate(vec, HYPRE_MEMORY_HOST);
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
      hypredrv_IJVectorMigrate(vec, orig_memloc);
   }
#endif
}

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
      LinearSystemComputeHostNorm(vec, par_vec, seq_vec, data, size, comm, norm_type,
                                  norm);
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
   (void)one;
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
   HYPRE_IJVector vec_r = NULL;

   hypredrv_LinearSystemComputeResidualNormWork(mat_A, vec_b, vec_x, norm_type, &vec_r,
                                                res_norm);
   if (vec_r)
   {
      HYPRE_IJVectorDestroy(vec_r);
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemComputeResidualNormWork
 *
 * Same as hypredrv_LinearSystemComputeResidualNorm, but keeps the residual in
 * *work_ptr (created on first use, laid out like vec_b) so repeated norms of
 * the same system skip the vector allocation and its validity reductions.
 * The caller destroys *work_ptr.
 *-----------------------------------------------------------------------------*/

void
hypredrv_LinearSystemComputeResidualNormWork(HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                                             HYPRE_IJVector vec_x, const char *norm_type,
                                             HYPRE_IJVector *work_ptr, double *res_norm)
{
   if (!*work_ptr)
   {
      *work_ptr = LinearSystemBuildResidual(mat_A, vec_b, vec_x, NULL, NULL);
   }
   else
   {
      void *obj_A = NULL, *obj_b = NULL, *obj_x = NULL, *obj_r = NULL;

      HYPRE_IJMatrixGetObject(mat_A, &obj_A);
      HYPRE_IJVectorGetObject(vec_b, &obj_b);
      HYPRE_IJVectorGetObject(vec_x, &obj_x);
      HYPRE_IJVectorGetObject(*work_ptr, &obj_r);
      HYPRE_ParVectorCopy((HYPRE_ParVector)obj_b, (HYPRE_ParVector)obj_r);
      HYPRE_ParCSRMatrixMatvec(-1.0, (HYPRE_ParCSRMatrix)obj_A, (HYPRE_ParVector)obj_x,
                               1.0, (HYPRE_ParVector)obj_r);
   }

   if (!*work_ptr)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      *res_norm = -1.0;
      return;
   }

   hypredrv_LinearSystemComputeVectorNorm(*work_ptr, norm_type, res_norm);
}

/*-----------------------------------------------------------------------------
 * hypredrv_LinearSystemLogBlockResidualNorms
 *-----------------------------------------------------------------------------*/

/* Accumulates the per-block squared norms of the residual and the right-hand
 * side, then logs one line per block with their ratio. */
static void
BlockResidualAccumulateAndReport(MPI_Comm comm, const IntArray *dofmap,
                                 const DofLabelMap *dof_labels, hypre_Vector *local_r,
                                 hypre_Vector *local_b, HYPRE_Int size, int num_blocks,
                                 int num_label_slots, int *label_to_pos,
                                 double *local_norms, double *global_norms,
                                 HYPRE_Complex *host_r, HYPRE_Complex *host_b,
                                 const char *log_object_name, int ls_id)
{
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
}

static int
BlockResidualVectorsValid(const hypre_Vector *local_b, const hypre_Vector *local_r,
                          const IntArray *dofmap)
{
   return local_b && local_r && hypre_VectorSize(local_r) >= 0 &&
          dofmap->size == (size_t)hypre_VectorSize(local_r);
}

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
   if (!BlockDiagnosticsAgree(
          comm, local_valid, log_object_name, ls_id,
          "block residual diagnostics skipped: invalid dofmap labels"))
   {
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
   local_valid           = BlockResidualVectorsValid(local_b, local_r, dofmap);
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

   BlockResidualAccumulateAndReport(
      comm, dofmap, dof_labels, local_r, local_b, size, num_blocks, num_label_slots,
      label_to_pos, local_norms, global_norms, host_r, host_b, log_object_name, ls_id);

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

static int
LinearSystemReadTimestepCount(FILE *fp, int *total)
{
   char count_text[64];
   return fscanf(fp, "%63s", count_text) == 1 && hypredrv_ParseInt(count_text, total) &&
          *total > 0 && *total <= (int)LSSEQ_MAX_TIMESTEPS;
}

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
      return ERROR_INVALID_VAL;
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
      return ERROR_INVALID_VAL;
   }

   FILE *fp = fopen(filename, "r");
   if (!fp)
   {
      hypredrv_ErrorCodeSet(ERROR_FILE_NOT_FOUND);
      hypredrv_ErrorMsgAdd("Could not open timestep file: '%s'", filename);
      return ERROR_FILE_NOT_FOUND;
   }

   int total = 0;
   if (!LinearSystemReadTimestepCount(fp, &total))
   {
      fclose(fp);
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid timestep file header in '%s'", filename);
      return ERROR_INVALID_VAL;
   }

   IntArray *ids = NULL;
   if (timestep_ids) /* GCOVR_EXCL_BR_LINE */
   {
      ids = hypredrv_IntArrayCreate((size_t)total);
      if (!ids) /* GCOVR_EXCL_BR_LINE */
      {
         fclose(fp);
         hypredrv_AllocationFailure("Failed to allocate timestep ids array");
         return ERROR_ALLOCATION;
      }
   }

   IntArray *starts = hypredrv_IntArrayCreate((size_t)total);
   if (!starts) /* GCOVR_EXCL_BR_LINE */
   {
      fclose(fp);
      hypredrv_IntArrayDestroy(&ids);
      hypredrv_AllocationFailure("Failed to allocate timestep starts array");
      return ERROR_ALLOCATION;
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
         return ERROR_INVALID_VAL;
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
   return ERROR_NONE;
}
