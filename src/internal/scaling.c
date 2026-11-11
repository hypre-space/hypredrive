/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/scaling.h"
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"
#include "internal/error.h"
#include "internal/field.h"
#include "internal/gen_macros.h"
#include "internal/linsys.h"
#include "internal/utils.h"
#include "logging.h"

struct Scaling_context_struct
{
   MPI_Comm       comm;
   scaling_type_t type;
   /* Which caller-owned system objects are currently scaled. */
   int             matrices_are_scaled;
   int             rhs_is_scaled;
   int             x_is_scaled;
   HYPRE_Complex   scalar_factor;  /* for rhs_l2 */
   HYPRE_ParVector scaling_vector; /* for dofmap */
   HYPRE_ParVector inverse_scaling_vector;
   HYPRE_IJVector  scaling_ijvec; /* IJ wrapper for scaling_vector */
   /* Borrowed only while a system transformation is active. */
   HYPRE_IJMatrix active_A, active_M;
   HYPRE_IJVector active_b, active_x;
};

/* Applied-ness is derived from what is scaled, so it cannot disagree with it. */
static bool
ScalingAnyApplied(const Scaling_context *ctx)
{
   return (ctx->matrices_are_scaled || ctx->rhs_is_scaled || ctx->x_is_scaled) != 0;
}

/*
 * gcovr: branches in #else arms for HYPRE < v3.0.0, in HYPRE_USING_GPU-only
 * sections, and on a few impractical error paths are excluded so branch
 * coverage reflects the active CPU / Hypre configuration (see markers below).
 */

#define Scaling_FIELDS(_prefix)                                       \
   ADD_FIELD_OFFSET_ENTRY(_prefix, enabled, hypredrv_FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, hypredrv_FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, custom_values, hypredrv_FieldTypeDoubleArraySet)

/* Define num_fields macro */
#define Scaling_NUM_FIELDS \
   (sizeof(Scaling_field_offset_map) / sizeof(Scaling_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(Scaling) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * ScalingGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
hypredrv_ScalingGetValidValues(const char *key)
{
   /* GCOVR_EXCL_BR_START */
   if (!strcmp(key, "enabled"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {
         {"rhs_l2", (int)SCALING_RHS_L2},
         {"dofmap_mag", (int)SCALING_DOFMAP_MAG},
         {"dofmap_custom", (int)SCALING_DOFMAP_CUSTOM},
         {"dofmap_row_custom", (int)SCALING_DOFMAP_ROW_CUSTOM},
         {"dofmap_col_custom", (int)SCALING_DOFMAP_COL_CUSTOM},
         {"dofmap_similarity_custom", (int)SCALING_DOFMAP_SIMILARITY_CUSTOM},
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return STR_INT_MAP_ARRAY_VOID();
   /* GCOVR_EXCL_BR_STOP */
}

/*-----------------------------------------------------------------------------
 * ScalingSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingSetDefaultArgs(Scaling_args *args)
{
   args->enabled       = 0;
   args->type          = SCALING_RHS_L2;
   args->custom_values = NULL;
}

bool
hypredrv_ScalingIsApplied(const Scaling_context *ctx)
{
   return (ctx && ScalingAnyApplied(ctx)) != 0;
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingContextDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingContextDestroy(Scaling_context **ctx_ptr)
{
   /* GCOVR_EXCL_BR_START */
   if (!ctx_ptr || !*ctx_ptr)
   {
      return;
   }

   Scaling_context *ctx  = *ctx_ptr;
   MPI_Comm         comm = ctx->comm;

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   /* Reclaim every owned handle even when hypre has a sticky error flag. */
   if (ctx->inverse_scaling_vector) HYPRE_ParVectorDestroy(ctx->inverse_scaling_vector);
   if (ctx->scaling_ijvec) HYPRE_IJVectorDestroy(ctx->scaling_ijvec);
   else if (ctx->scaling_vector) HYPRE_ParVectorDestroy(ctx->scaling_vector);
#endif
   free(ctx);
   *ctx_ptr = NULL;
   HYPREDRV_LOG_COMMF(3, comm, NULL, 0, "scaling context destroyed");
   /* GCOVR_EXCL_BR_STOP */
}

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
/* Shared dofmap-scaling setup: validates the dofmap against the local rows of
 * mat_A and computes the global tag count (max tag + 1, collective over comm).
 * `label` prefixes the missing-dofmap message. Returns zero with the error
 * state set when the request cannot be honoured. */
static int
ScalingDofmapResolve(MPI_Comm comm, const char *label, HYPRE_IJMatrix mat_A,
                     const IntArray *dofmap, HYPRE_ParCSRMatrix *par_A_out,
                     HYPRE_BigInt *ilower_out, HYPRE_BigInt *iupper_out,
                     HYPRE_Int *num_local_rows_out, HYPRE_Int *num_tags_out)
{
   void        *obj_A     = NULL;
   HYPRE_BigInt row_start = 0, row_end = 0, col_start = 0, col_end = 0;
   int          local_ok = 1, global_ok = 0;

   if (mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   }
   *par_A_out = (HYPRE_ParCSRMatrix)obj_A;

   /* Get local range from ParCSRMatrix directly instead of IJMatrix */
   if (obj_A)
   {
      hypre_ParCSRMatrixGetLocalRange((hypre_ParCSRMatrix *)obj_A, &row_start, &row_end,
                                      &col_start, &col_end);
   }
   *ilower_out         = row_start;
   *iupper_out         = row_end;
   *num_local_rows_out = (HYPRE_Int)(row_end - row_start + 1);

   if (!dofmap || !dofmap->data)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      hypredrv_ErrorMsgAdd("%s scaling requires a dofmap to be set", label);
      local_ok = 0;
   }
   else if (!obj_A)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("%s scaling requires an assembled matrix", label);
      local_ok = 0;
   }
   else if (*num_local_rows_out < 0 || dofmap->size != (size_t)*num_local_rows_out)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("dofmap size (%zu) does not match local matrix rows (%lld)",
                           dofmap->size, (long long)*num_local_rows_out);
      local_ok = 0;
   }
   /* hypre indexes per-tag arrays with these labels and only asserts the range */
   for (size_t i = 0; local_ok && i < dofmap->size; i++)
   {
      if (dofmap->data[i] < 0)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("%s scaling: negative dofmap label %d at local row %zu",
                              label, (int)dofmap->data[i], i);
         local_ok = 0;
      }
   }

   /* The dofmap is rank-local: agree on its validity before the collective
    * below, so a rank that rejects it cannot leave the others blocked. */
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!global_ok)
   {
      if (local_ok)
      {
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("%s scaling: invalid dofmap on another rank", label);
      }
      return 0;
   }

   HYPRE_Int max_tag = -1, global_max_tag = 0;
   for (HYPRE_Int i = 0; i < *num_local_rows_out; i++)
   {
      if (dofmap->data[i] > max_tag)
      {
         max_tag = dofmap->data[i];
      }
   }
   MPI_Allreduce(&max_tag, &global_max_tag, 1, MPI_INT, MPI_MAX, comm);
   *num_tags_out = global_max_tag + 1;

   return 1;
}
#endif

/*-----------------------------------------------------------------------------
 * Scaling preparation (dofmap_mag strategy)
 *-----------------------------------------------------------------------------*/

static HYPREDRV_MAYBE_UNUSED void
ScalingComputeDofmapMag(MPI_Comm comm, const Scaling_args *args, Scaling_context *ctx,
                        HYPRE_IJMatrix mat_A, const IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   /* GCOVR_EXCL_BR_START */
   (void)args;
   HYPRE_MemoryLocation memloc_tags = HYPRE_MEMORY_HOST;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation orig_mat_memloc = HYPRE_MEMORY_HOST;
#endif
   HYPRE_ParCSRMatrix par_A  = NULL;
   HYPRE_BigInt       ilower = 0, iupper = 0;
   HYPRE_Int          num_local_rows = 0;
   HYPRE_Int         *tags           = NULL;
   HYPRE_Int          num_tags       = 0;

   if (!ScalingDofmapResolve(comm, "dofmap", mat_A, dofmap, &par_A, &ilower, &iupper,
                             &num_local_rows, &num_tags))
   {
      return;
   }
#if defined(HYPRE_USING_GPU)
   orig_mat_memloc = hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)par_A);
#endif

   /* Tags array from dofmap (HYPRE_Int copy for hypre) */
   tags = hypredrv_Malloc((size_t)num_local_rows * sizeof(*tags), "scaling tags");
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      free(tags);
      return;
   }
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      tags[i] = (HYPRE_Int)dofmap->data[i];
   }

   /* Work around hypre's device TaggedFnorm kernel, which can read tags out of bounds
    * for dofmap_mag on GPU builds. Compute the tagged scaling on host, then migrate the
    * resulting scaling vector back to the matrix memory location. */
#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(orig_mat_memloc) == HYPRE_EXEC_DEVICE)
   {
      hypredrv_ParCSRMatrixMigrate(par_A, HYPRE_MEMORY_HOST);
   }
#endif

   /* Compute scaling into a fresh ParVector. Free the tags and restore the
    * matrix location before reporting a failure, so neither is left behind. */
   HYPRE_Int scaling_ierr = HYPRE_ParCSRMatrixComputeScalingTagged(
      par_A, 1, memloc_tags, num_tags, tags, &ctx->scaling_vector);
   free(tags);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(orig_mat_memloc) == HYPRE_EXEC_DEVICE)
   {
      if (!scaling_ierr)
      {
         hypre_ParVectorMigrate((hypre_ParVector *)ctx->scaling_vector, orig_mat_memloc);
      }
      hypredrv_ParCSRMatrixMigrate(par_A, orig_mat_memloc);
   }
#endif

   HYPRE_SAFE_CALL(scaling_ierr);
   /* GCOVR_EXCL_BR_STOP */
#else
   /* GCOVR_EXCL_BR_START */
   (void)comm;
   (void)args;
   (void)ctx;
   (void)mat_A;
   (void)dofmap;
   /* Scaling disabled on older hypre versions */
   /* GCOVR_EXCL_BR_STOP */
#endif
}

/*-----------------------------------------------------------------------------
 * Scaling preparation (dofmap_custom strategy)
 *-----------------------------------------------------------------------------*/

/* Validates the dofmap/custom-value pairing and resolves this rank's row range.
 * Returns zero with the error state set when the request cannot be honoured. */
static HYPREDRV_MAYBE_UNUSED int
ScalingDofmapCustomPrepare(MPI_Comm comm, const Scaling_args *args, HYPRE_IJMatrix mat_A,
                           const IntArray *dofmap, HYPRE_ParCSRMatrix *par_A_out,
                           HYPRE_BigInt *ilower_out, HYPRE_BigInt *iupper_out,
                           HYPRE_Int *num_local_rows_out)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   HYPRE_Int num_tags = 0;

   if (!ScalingDofmapResolve(comm, "custom dofmap", mat_A, dofmap, par_A_out, ilower_out,
                             iupper_out, num_local_rows_out, &num_tags))
   {
      return 0;
   }

   if (!args->custom_values || args->custom_values->size == 0 ||
       !args->custom_values->data)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("custom dofmap scaling requires custom_values to be set");
      return 0;
   }

   for (size_t i = 0; i < args->custom_values->size; i++)
   {
      if (!hypredrv_DoubleIsFinite(args->custom_values->data[i]) ||
          args->custom_values->data[i] == 0.0)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "custom dofmap scaling requires finite, nonzero custom_values (entry %zu)",
            i);
         return 0;
      }
   }

   /* Verify that the number of custom values matches the number of unique tags */
   if ((size_t)num_tags != args->custom_values->size)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "dofmap_custom: number of custom_values (%zu) does not match number of "
         "unique dofmap tags (%lld)",
         args->custom_values->size, (long long)num_tags);
      return 0;
   }

   return 1;
#else
   /* GCOVR_EXCL_BR_START */
   (void)comm;
   (void)args;
   (void)mat_A;
   (void)dofmap;
   (void)par_A_out;
   (void)ilower_out;
   (void)iupper_out;
   (void)num_local_rows_out;
   /* Scaling disabled on older hypre versions */
   /* GCOVR_EXCL_BR_STOP */
   return 0;
#endif
}

static HYPREDRV_MAYBE_UNUSED void
ScalingComputeDofmapCustom(MPI_Comm comm, const Scaling_args *args, Scaling_context *ctx,
                           HYPRE_IJMatrix mat_A, const IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   /* GCOVR_EXCL_BR_START */
   HYPRE_ParCSRMatrix par_A  = NULL;
   HYPRE_BigInt       ilower = 0, iupper = 0;
   HYPRE_Int          num_local_rows = 0;

   ScalingDofmapCustomPrepare(comm, args, mat_A, dofmap, &par_A, &ilower, &iupper,
                              &num_local_rows);
   if (hypredrv_DistributedErrorStateSync(comm)) return;

   /* Create IJVector wrapper for scaling vector */
   HYPRE_MemoryLocation memory_location =
      hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)par_A);
   HYPRE_Complex *h_values =
      hypre_TAlloc(HYPRE_Complex, num_local_rows, HYPRE_MEMORY_HOST);
   const HYPRE_Complex *values = h_values;
#ifdef HYPRE_USING_GPU
   HYPRE_Complex *d_values = NULL;
#endif

   if (memory_location == HYPRE_MEMORY_UNDEFINED)
   {
      memory_location = HYPRE_MEMORY_HOST;
   }

   HYPRE_SAFE_CALL(HYPRE_IJVectorCreate(comm, ilower, iupper, &ctx->scaling_ijvec));
   HYPRE_SAFE_CALL(HYPRE_IJVectorSetObjectType(ctx->scaling_ijvec, HYPRE_PARCSR));
   HYPRE_SAFE_CALL(HYPRE_IJVectorInitialize_v2(ctx->scaling_ijvec, memory_location));

#ifdef HYPRE_USING_GPU
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      values = d_values =
         hypre_TAlloc(HYPRE_Complex, num_local_rows, HYPRE_MEMORY_DEVICE);
   }
#endif

   /* Admission checked labels and the global value count before allocation. */
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      HYPRE_Int tag = dofmap->data[i];
      h_values[i]   = (HYPRE_Complex)args->custom_values->data[tag];
   }

#ifdef HYPRE_USING_GPU
   if (values != h_values)
   {
      hypre_TMemcpy(d_values, h_values, HYPRE_Complex, num_local_rows,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
#endif

   HYPRE_SAFE_CALL(
      HYPRE_IJVectorSetValues(ctx->scaling_ijvec, num_local_rows, NULL, values));

   /* Assemble the vector */
   HYPRE_SAFE_CALL(HYPRE_IJVectorAssemble(ctx->scaling_ijvec));
   hypre_TFree(h_values, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_GPU
   if (d_values)
   {
      hypre_TFree(d_values, HYPRE_MEMORY_DEVICE);
   }
#endif

   /* Cache the assembled ParVector owned by the IJVector */
   void *obj_scaling = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling);
   ctx->scaling_vector = (HYPRE_ParVector)obj_scaling;
   /* GCOVR_EXCL_BR_STOP */
#else
   /* GCOVR_EXCL_BR_START */
   (void)comm;
   (void)args;
   (void)ctx;
   (void)mat_A;
   (void)dofmap;
   /* Scaling disabled on older hypre versions */
   /* GCOVR_EXCL_BR_STOP */
#endif
}

/*-----------------------------------------------------------------------------
 * Check the scaling request. Every rank takes part in the agreement check, so
 * this is collective even when the local arguments are invalid.
 *-----------------------------------------------------------------------------*/

static void
ScalingRequestCheck(MPI_Comm comm, const Scaling_args *args, Scaling_context **output)
{
   int selection[] = {args && args->enabled ? 1 : 0,
                      args && args->enabled ? (int)args->type : 0};
   int minimum[2], maximum[2];
   MPI_Allreduce(selection, minimum, 2, MPI_INT, MPI_MIN, comm);
   MPI_Allreduce(selection, maximum, 2, MPI_INT, MPI_MAX, comm);
   if (!args || !output)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "Scaling preparation requires arguments and an output pointer");
   }
   if (minimum[0] != maximum[0] || minimum[1] != maximum[1])
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Scaling enablement and type must agree across ranks");
   }
   if (output && hypredrv_ScalingIsApplied(*output))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Restore applied scaling before preparing another system");
   }
}

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
/* Returns the native RHS vector when RHS scaling is requested, flagging an
 * error when that vector is missing or not initialized. */
static hypre_ParVector *
ScalingRhsVectorGet(const Scaling_args *args, HYPRE_IJVector vec_b)
{
   hypre_ParVector *native_b = NULL;
   if (args && args->enabled && args->type == SCALING_RHS_L2)
   {
      void *object = NULL;
      if (vec_b) HYPRE_IJVectorGetObject(vec_b, &object);
      native_b                  = (hypre_ParVector *)object;
      const hypre_Vector *local = native_b ? hypre_ParVectorLocalVector(native_b) : NULL;
      if (!local || hypre_VectorSize(local) < 0 ||
          (hypre_VectorSize(local) > 0 && !hypre_VectorData(local)))
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("RHS scaling requires an initialized RHS vector");
      }
   }
   return native_b;
}

static void
ScalingComputeRhsL2(Scaling_context *ctx, hypre_ParVector *native_b)
{
   double norm = sqrt((double)hypre_ParVectorInnerProd(native_b, native_b));
   if (!hypredrv_DoubleIsFinite(norm) || norm < 0.0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Scaling requires a finite RHS norm");
   }
   else
   {
      ctx->scalar_factor = norm > 0.0 ? 1.0 / sqrt(norm) : 1.0;
   }
}

/* Fills the candidate runtime for the requested scaling type (collective). */
static void
ScalingComputeFactors(MPI_Comm comm, const Scaling_args *args, Scaling_context *ctx,
                      HYPRE_IJMatrix mat_A, const IntArray *dofmap,
                      hypre_ParVector *native_b)
{
   switch (args->type)
   {
      case SCALING_RHS_L2:
         ScalingComputeRhsL2(ctx, native_b);
         break;
      case SCALING_DOFMAP_MAG:
         ScalingComputeDofmapMag(comm, args, ctx, mat_A, dofmap);
         break;
      case SCALING_DOFMAP_CUSTOM:
      case SCALING_DOFMAP_ROW_CUSTOM:
      case SCALING_DOFMAP_COL_CUSTOM:
      case SCALING_DOFMAP_SIMILARITY_CUSTOM:
         ScalingComputeDofmapCustom(comm, args, ctx, mat_A, dofmap);
         break;
      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("Unknown scaling type: %d", (int)args->type);
         break;
   }
}

static void
ScalingInverseVectorCache(Scaling_context *ctx)
{
   hypre_ParVector *inverse = NULL;
   if (ctx->scaling_vector)
      hypre_ParVectorPointwiseInverse((hypre_ParVector *)ctx->scaling_vector, &inverse);
   ctx->inverse_scaling_vector = (HYPRE_ParVector)inverse;
   if (!inverse) hypredrv_AllocationFailure("Failed to cache the inverse scaling vector");
}
#endif

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingPrepare
 *-----------------------------------------------------------------------------*/

bool
hypredrv_ScalingPrepare(MPI_Comm comm, const Scaling_args *args, Scaling_context **output,
                        HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b,
                        const IntArray *dofmap)
{
   ScalingRequestCheck(comm, args, output);
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   hypre_ParVector *native_b = ScalingRhsVectorGet(args, vec_b);
#endif
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm)) return false;
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (!args->enabled)
   {
      hypredrv_ScalingContextDestroy(output);
      return true;
   }
   Scaling_context *candidate = hypredrv_Calloc(1, sizeof(*candidate), "scaling runtime");
   if (hypredrv_DistributedErrorStateSync(comm))
   {
      free(candidate);
      return false;
   }
   candidate->comm          = comm;
   candidate->type          = args->type;
   candidate->scalar_factor = 1.0;
   ScalingComputeFactors(comm, args, candidate, mat_A, dofmap, native_b);
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm)) goto failure;
   if (args->type != SCALING_RHS_L2)
   {
      ScalingInverseVectorCache(candidate);
   }
   hypredrv_HypreConsumeErrors(comm);
   if (hypredrv_DistributedErrorStateSync(comm)) goto failure;
   hypredrv_ScalingContextDestroy(output);
   *output = candidate;
   HYPREDRV_LOG_COMMF(3, comm, NULL, 0, "scaling context prepared (type=%d)",
                      (int)args->type);
   return true;

failure:
   hypredrv_ScalingContextDestroy(&candidate);
   return false;
#else
   (void)mat_A;
   (void)vec_b;
   (void)dofmap;
   hypredrv_ScalingContextDestroy(output);
   HYPREDRV_LOG_COMMF(2, comm, NULL, 0, "scaling disabled (requires Hypre >= v3.0.0)");
   return true;
#endif
}

/*-----------------------------------------------------------------------------
 * Vector scaling helpers
 *-----------------------------------------------------------------------------*/

static HYPREDRV_MAYBE_UNUSED int
ScalingTransformVectorRHSL2(const Scaling_context *ctx, HYPRE_IJVector vec,
                            scaling_vector_kind_t kind, int apply)
{
   /* GCOVR_EXCL_BR_START */
   MPI_Comm log_comm = hypredrv_IJVectorComm(vec);
   if (!vec)
   {
      return 0;
   }

   HYPRE_Complex s      = ctx->scalar_factor;
   HYPRE_Complex factor = 1.0;
   if (kind == SCALING_VECTOR_RHS)
   {
      factor = apply ? s : (1.0 / s);
   }
   else
   {
      factor = apply ? (1.0 / s) : s;
   }

   void            *obj_vec = NULL;
   hypre_ParVector *par_vec = NULL;
   HYPRE_IJVectorGetObject(vec, &obj_vec);
   if (!obj_vec)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingTransformVectorRHSL2: vector object is NULL");
      HYPREDRV_LOG_COMMF(2, log_comm, NULL, 0,
                         "scaling vector transform failed: IJ vector object is NULL");
      return 0;
   }
   par_vec = (hypre_ParVector *)obj_vec;
   hypre_ParVectorScale(factor, par_vec);
   return 1;
   /* GCOVR_EXCL_BR_STOP */
}

static HYPREDRV_MAYBE_UNUSED int
ScalingTransformVectorDofmap(const Scaling_context *ctx, HYPRE_IJVector vec,
                             scaling_vector_kind_t kind, int apply)
{
   /* GCOVR_EXCL_BR_START */
   MPI_Comm log_comm = hypredrv_IJVectorComm(vec);
   if (!vec)
   {
      return 0;
   }

   void            *obj_vec     = NULL;
   hypre_ParVector *par_vec     = NULL;
   hypre_ParVector *par_scaling = (hypre_ParVector *)ctx->scaling_vector;

   HYPRE_IJVectorGetObject(vec, &obj_vec);
   if (!obj_vec)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingTransformVectorDofmap: vector object is NULL");
      HYPREDRV_LOG_COMMF(2, log_comm, NULL, 0,
                         "scaling vector transform failed: IJ vector object is NULL");
      return 0;
   }
   par_vec = (hypre_ParVector *)obj_vec;

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if ((kind == SCALING_VECTOR_RHS && apply) ||
       (kind == SCALING_VECTOR_UNKNOWN && !apply))
   {
      hypre_ParVectorPointwiseProduct(par_scaling, par_vec, &par_vec);
   }
   else
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 18)
      hypre_ParVectorPointwiseDivision(par_scaling, par_vec, &par_vec);
#else
      /* hypre_ParVectorPointwiseDivision(x, y, z) computes z=y/x on host but z=x/y on
       * device (GPU bug). Use inverse+product to get z=y/x=vec/scaling consistently. */
      hypre_ParVector *inv_scaling = (hypre_ParVector *)ctx->inverse_scaling_vector;
      hypre_ParVectorPointwiseProduct(inv_scaling, par_vec, &par_vec);
#endif
   }
#else
   (void)kind;
   (void)apply;
   (void)par_vec;
   (void)par_scaling;
   hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
   hypredrv_ErrorMsgAdd("ScalingTransformVectorDofmap: requires Hypre >= v3.0.0");
   HYPREDRV_LOG_COMMF(2, log_comm, NULL, 0,
                      "scaling vector transform failed: requires Hypre >= v3.0.0");
   return 0;
#endif
   return 1;
   /* GCOVR_EXCL_BR_STOP */
}

static int
ScalingTransformVector(const Scaling_context *ctx, HYPRE_IJVector vec,
                       scaling_vector_kind_t kind, int apply)
{
   /* GCOVR_EXCL_BR_START */
   if (!ctx || !vec)
   {
      return 0;
   }
   if (kind == SCALING_VECTOR_INITIAL_GUESS)
   {
      if (!ctx->x_is_scaled) return 0;
      kind = SCALING_VECTOR_UNKNOWN;
   }

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (ctx->type)
   {
      case SCALING_RHS_L2:
         return ScalingTransformVectorRHSL2(ctx, vec, kind, apply);

      case SCALING_DOFMAP_MAG:
      case SCALING_DOFMAP_CUSTOM:
         return ScalingTransformVectorDofmap(ctx, vec, kind, apply);

      case SCALING_DOFMAP_ROW_CUSTOM:
         if (kind == SCALING_VECTOR_RHS)
         {
            return ScalingTransformVectorDofmap(ctx, vec, kind, apply);
         }
         return 0;

      case SCALING_DOFMAP_COL_CUSTOM:
         if (kind == SCALING_VECTOR_UNKNOWN)
         {
            return ScalingTransformVectorDofmap(ctx, vec, kind, apply);
         }
         return 0;

      case SCALING_DOFMAP_SIMILARITY_CUSTOM:
         if (kind == SCALING_VECTOR_RHS)
         {
            /* c = S^-1 b */
            return ScalingTransformVectorDofmap(ctx, vec, kind, !apply);
         }
         else
         {
            /* y = S^-1 x on apply; x = S y on undo */
            return ScalingTransformVectorDofmap(ctx, vec, kind, apply);
         }

      default:
         return 0; /* Type was admitted when the runtime was prepared. */
   }
#else
   (void)kind;
   if (apply)
   {
      HYPREDRV_LOG_COMMF(2, hypredrv_IJVectorComm(vec), NULL, 0,
                         "scaling apply-to-vector failed: requires Hypre >= v3.0.0");
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, hypredrv_IJVectorComm(vec), NULL, 0,
                         "scaling undo-on-vector failed: requires Hypre >= v3.0.0");
   }
#endif
   return 0;
   /* GCOVR_EXCL_BR_STOP */
}

void
hypredrv_ScalingApplyToVector(const Scaling_context *ctx, HYPRE_IJVector vec,
                              scaling_vector_kind_t kind)
{
   ScalingTransformVector(ctx, vec, kind, 1);
}

void
hypredrv_ScalingUndoOnVector(const Scaling_context *ctx, HYPRE_IJVector vec,
                             scaling_vector_kind_t kind)
{
   ScalingTransformVector(ctx, vec, kind, 0);
}

/*-----------------------------------------------------------------------------
 * System scaling helpers
 *-----------------------------------------------------------------------------*/

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
ScalingDiagScaleMatrices(HYPRE_ParCSRMatrix par_A, HYPRE_ParCSRMatrix par_M,
                         hypre_ParVector *left, hypre_ParVector *right)
{
   hypre_ParCSRMatrixDiagScale(par_A, left, right);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixDiagScale(par_M, left, right);
   }
}

static void
ScalingDofmapMatrixFactors(const Scaling_context *ctx, int apply,
                           hypre_ParVector **left_ptr, hypre_ParVector **right_ptr)
{
   hypre_ParVector *scaling = (hypre_ParVector *)ctx->scaling_vector;
   hypre_ParVector *inverse = (hypre_ParVector *)ctx->inverse_scaling_vector;

   switch (ctx->type)
   {
      case SCALING_DOFMAP_ROW_CUSTOM:
         *left_ptr  = apply ? scaling : inverse;
         *right_ptr = NULL;
         break;
      case SCALING_DOFMAP_COL_CUSTOM:
         *left_ptr  = NULL;
         *right_ptr = apply ? scaling : inverse;
         break;
      case SCALING_DOFMAP_SIMILARITY_CUSTOM:
         *left_ptr  = apply ? inverse : scaling;
         *right_ptr = apply ? scaling : inverse;
         break;
      default:
         *left_ptr = *right_ptr = apply ? scaling : inverse;
         break;
   }
}
#endif

/* Cleanup must not mistake an error from the failed setup/solve for an error
 * raised by the inverse transforms themselves. Keep queued messages intact,
 * temporarily remove the saved bits, then merge them back after restoration. */
static void
ScalingRestoreErrorCode(uint32_t saved_error)
{
   for (uint32_t i = 0; i < 32; i++)
   {
      uint32_t bit = UINT32_C(1) << i;
      if (saved_error & bit)
      {
         hypredrv_ErrorCodeSet((hypredrv_error_t)bit);
      }
   }
}

/* Applies the configured scaling to the operators and then to the vectors. */
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static int
ScalingApplyToSystem(Scaling_context *ctx, HYPRE_ParCSRMatrix par_A,
                     HYPRE_ParCSRMatrix par_M, HYPRE_IJVector vec_b, HYPRE_IJVector vec_x,
                     int log_rank)
{
   if (ctx->type == SCALING_RHS_L2)
   {
      HYPRE_Complex s2 = ctx->scalar_factor * ctx->scalar_factor;
      hypre_ParCSRMatrixScale(par_A, s2);
      if (par_M != par_A)
      {
         hypre_ParCSRMatrixScale(par_M, s2);
      }
   }
   else
   {
      hypre_ParVector *left = NULL, *right = NULL;
      ScalingDofmapMatrixFactors(ctx, 1, &left, &right);
      ScalingDiagScaleMatrices(par_A, par_M, left, right);
   }
   ctx->matrices_are_scaled = 1;

   if (ScalingTransformVector(ctx, vec_b, SCALING_VECTOR_RHS, 1))
   {
      ctx->rhs_is_scaled = 1;
   }
   if (hypredrv_ErrorCodeGet())
   {
      HYPREDRV_LOGF(2, log_rank, NULL, 0,
                    "scaling apply failed: RHS vector transform error");
      return 0;
   }
   if (ScalingTransformVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN, 1))
   {
      ctx->x_is_scaled = 1;
   }
   if (hypredrv_ErrorCodeGet())
   {
      HYPREDRV_LOGF(2, log_rank, NULL, 0,
                    "scaling apply failed: unknown vector transform error");
      return 0;
   }

   return 1;
}
#endif

/* Reverses the scaling, unwinding the vectors before the operators. */
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static int
ScalingUndoOnSystem(Scaling_context *ctx, HYPRE_ParCSRMatrix par_A,
                    HYPRE_ParCSRMatrix par_M, HYPRE_IJVector vec_b, HYPRE_IJVector vec_x,
                    int log_rank)
{
   if (ctx->x_is_scaled)
   {
      if (ScalingTransformVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN, 0))
      {
         ctx->x_is_scaled = 0;
      }
      if (hypredrv_ErrorCodeGet())
      {
         HYPREDRV_LOGF(2, log_rank, NULL, 0,
                       "scaling undo failed: unknown vector transform error");
         return 0;
      }
   }
   if (ctx->rhs_is_scaled)
   {
      if (ScalingTransformVector(ctx, vec_b, SCALING_VECTOR_RHS, 0))
      {
         ctx->rhs_is_scaled = 0;
      }
      if (hypredrv_ErrorCodeGet())
      {
         HYPREDRV_LOGF(2, log_rank, NULL, 0,
                       "scaling undo failed: RHS vector transform error");
         return 0;
      }
   }

   if (ctx->matrices_are_scaled && ctx->type == SCALING_RHS_L2)
   {
      HYPRE_Complex s2     = ctx->scalar_factor * ctx->scalar_factor;
      HYPRE_Complex inv_s2 = 1.0 / s2;
      hypre_ParCSRMatrixScale(par_A, inv_s2);
      if (par_M != par_A)
      {
         hypre_ParCSRMatrixScale(par_M, inv_s2);
      }
      ctx->matrices_are_scaled = 0;
   }
   else if (ctx->matrices_are_scaled)
   {
      hypre_ParVector *left = NULL, *right = NULL;
      ScalingDofmapMatrixFactors(ctx, 0, &left, &right);
      ScalingDiagScaleMatrices(par_A, par_M, left, right);
      ctx->matrices_are_scaled = 0;
   }

   return 1;
}
#endif

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
static void
ScalingSystemMatrices(HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                      HYPRE_ParCSRMatrix *par_A, HYPRE_ParCSRMatrix *par_M)
{
   void *obj_A = NULL, *obj_M = NULL;
   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   *par_A = (HYPRE_ParCSRMatrix)obj_A;
   *par_M = *par_A;
   if (mat_M && mat_M != mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_M, &obj_M);
      *par_M = (HYPRE_ParCSRMatrix)obj_M;
   }
}
#endif

static void
ScalingTransformSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                       HYPRE_IJVector vec_b, HYPRE_IJVector vec_x, int apply)
{
   /* GCOVR_EXCL_BR_START */
   if (!ctx || (!apply && !ScalingAnyApplied(ctx)))
   {
      return;
   }
   int log_rank = -1;
   if (hypredrv_LogEnabled(2))
   {
      log_rank = hypredrv_LogRankFromComm(ctx->comm);
   }

   if (apply && ScalingAnyApplied(ctx))
   {
      HYPREDRV_LOGF(3, log_rank, NULL, 0, "scaling apply skipped (already applied)");
      return;
   }

   if (apply)
   {
      ctx->active_A = mat_A;
      ctx->active_M = mat_M;
      ctx->active_b = vec_b;
      ctx->active_x = vec_x;
   }
   else
   {
      mat_A = ctx->active_A;
      mat_M = ctx->active_M;
      vec_b = ctx->active_b;
      vec_x = ctx->active_x;
   }

   uint32_t saved_error = ERROR_NONE;
   if (!apply)
   {
      saved_error = hypredrv_ErrorCodeGet();
      hypredrv_ErrorCodeReset(saved_error);
   }

   const char *action = apply ? "apply" : "undo";
   HYPREDRV_LOGF(3, log_rank, NULL, 0, "scaling %s begin (type=%d)", action,
                 (int)ctx->type);

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;

   ScalingSystemMatrices(mat_A, mat_M, &par_A, &par_M);

   if (apply)
   {
      if (!ScalingApplyToSystem(ctx, par_A, par_M, vec_b, vec_x, log_rank))
      {
         goto done;
      }
   }
   else if (!ScalingUndoOnSystem(ctx, par_A, par_M, vec_b, vec_x, log_rank))
   {
      goto done;
   }
#else
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
   HYPREDRV_LOGF(2, log_rank, NULL, 0, "scaling %s failed: requires Hypre >= v3.0.0",
                 action);
#endif

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
done:
#endif
   if (!ScalingAnyApplied(ctx))
   {
      ctx->active_A = ctx->active_M = NULL;
      ctx->active_b = ctx->active_x = NULL;
   }
   if (!apply)
   {
      ScalingRestoreErrorCode(saved_error);
   }
   HYPREDRV_LOGF(3, log_rank, NULL, 0, "scaling %s end (is_applied=%d)", action,
                 (int)ScalingAnyApplied(ctx));
   /* GCOVR_EXCL_BR_STOP */
}

/*-----------------------------------------------------------------------------
 * System scaling public wrappers
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingApplyToSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A,
                              HYPRE_IJMatrix mat_M, HYPRE_IJVector vec_b,
                              HYPRE_IJVector vec_x)
{
   ScalingTransformSystem(ctx, mat_A, mat_M, vec_b, vec_x, 1);
}

void
hypredrv_ScalingUndoOnSystem(Scaling_context *ctx)
{
   ScalingTransformSystem(ctx, NULL, NULL, NULL, NULL, 0);
}
