/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "scaling.h"
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "_hypre_parcsr_mv.h"
#include "error.h"
#include "field.h"
#include "gen_macros.h"
#include "hypredrv_log.h"
#include "linsys.h"
#include "utils.h"

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
      };
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return STR_INT_MAP_ARRAY_VOID();
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

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingContextCreate
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingContextCreate(Scaling_context **ctx_ptr)
{
   int              myid = hypredrv_LogEnabled(3) ? hypredrv_LogRankFromComm(MPI_COMM_WORLD) : -1;
   Scaling_context *ctx  = (Scaling_context *)malloc(sizeof(Scaling_context));
   ctx->enabled          = 0;
   ctx->type             = SCALING_RHS_L2;
   ctx->is_applied       = 0;
   ctx->scalar_factor    = 1.0;
   ctx->scaling_vector   = NULL;
   ctx->scaling_ijvec    = NULL;
   *ctx_ptr              = ctx;
   hypredrv_Logf(3, myid, NULL, 0, "scaling context created");
}

/*-----------------------------------------------------------------------------
 * ScalingContextFreeVector
 *
 * Free the scaling vector held by the context, handling both ownership models:
 *   - dofmap_custom: scaling_vector is owned by scaling_ijvec (IJVector wrapper)
 *   - dofmap_mag:    scaling_vector is directly owned (no IJVector wrapper)
 *-----------------------------------------------------------------------------*/

static void
ScalingContextFreeVector(Scaling_context *ctx)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (ctx->scaling_ijvec)
   {
      HYPRE_SAFE_CALL(HYPRE_IJVectorDestroy(ctx->scaling_ijvec));
      ctx->scaling_ijvec  = NULL;
      ctx->scaling_vector = NULL;
   }
   else if (ctx->scaling_vector)
   {
      HYPRE_SAFE_CALL(HYPRE_ParVectorDestroy(ctx->scaling_vector));
      ctx->scaling_vector = NULL;
   }
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingContextDestroy
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingContextDestroy(Scaling_context **ctx_ptr)
{
   int myid = hypredrv_LogEnabled(3) ? hypredrv_LogRankFromComm(MPI_COMM_WORLD) : -1;
   if (!ctx_ptr || !*ctx_ptr)
   {
      return;
   }

   Scaling_context *ctx = *ctx_ptr;

   ScalingContextFreeVector(ctx);

   free(ctx);
   *ctx_ptr = NULL;
   hypredrv_Logf(3, myid, NULL, 0, "scaling context destroyed");
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingCompute (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingComputeRHSL2(MPI_Comm comm, Scaling_context *ctx, HYPRE_IJVector vec_b)
{
   double b_norm = 0.0;

   hypredrv_LinearSystemComputeVectorNorm(vec_b, "L2", &b_norm);

   if (b_norm > 0.0)
   {
      ctx->scalar_factor = 1.0 / sqrt(b_norm);
   }
   else
   {
      ctx->scalar_factor = 1.0;
   }
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingCompute (dofmap_mag strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingComputeDofmapMag(MPI_Comm comm, Scaling_args *args, Scaling_context *ctx,
                        HYPRE_IJMatrix mat_A, IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   HYPRE_MemoryLocation memloc_tags = HYPRE_MEMORY_HOST;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation orig_mat_memloc = HYPRE_MEMORY_HOST;
#endif
   void              *obj_A  = NULL;
   HYPRE_ParCSRMatrix par_A  = NULL;
   HYPRE_BigInt       ilower = 0, iupper = 0;
   HYPRE_Int          num_local_rows = 0;
   HYPRE_Int         *tags           = NULL;
   HYPRE_Int          num_tags       = 0;
   HYPRE_Int          max_tag        = -1;
   int                myid           = 0;

   MPI_Comm_rank(comm, &myid);

   if (!dofmap || !dofmap->data)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      hypredrv_ErrorMsgAdd("dofmap scaling requires a dofmap to be set");
      return;
   }

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;
#if defined(HYPRE_USING_GPU)
   orig_mat_memloc = hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *)par_A);
#endif

   /* Get local range from ParCSRMatrix directly instead of IJMatrix to avoid potential
    * issues */
   HYPRE_BigInt row_start = 0, row_end = 0, col_start = 0, col_end = 0;
   hypre_ParCSRMatrixGetLocalRange((hypre_ParCSRMatrix *)par_A, &row_start, &row_end,
                                   &col_start, &col_end);
   ilower         = row_start;
   iupper         = row_end;
   num_local_rows = (HYPRE_Int)(iupper - ilower + 1);

   if (dofmap->size != num_local_rows)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("dofmap size (%d) does not match local matrix rows (%d)",
                           dofmap->size, num_local_rows);
      return;
   }

   /* Allocate and fill tags array from dofmap */
   tags = (HYPRE_Int *)malloc((size_t)num_local_rows * sizeof(HYPRE_Int));
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      tags[i] = (HYPRE_Int)dofmap->data[i];
      if (tags[i] > max_tag)
      {
         max_tag = tags[i];
      }
   }

   /* Compute global max tag */
   HYPRE_Int global_max_tag = 0;
   MPI_Allreduce(&max_tag, &global_max_tag, 1, MPI_INT, MPI_MAX, comm);
   num_tags = global_max_tag + 1;

   /* Destroy previous scaling vector if it exists (from previous system) */
   ScalingContextFreeVector(ctx);

   /* Work around hypre's device TaggedFnorm kernel, which can read tags out of bounds
    * for dofmap_mag on GPU builds. Compute the tagged scaling on host, then migrate the
    * resulting scaling vector back to the matrix memory location. */
#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(orig_mat_memloc) == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixMigrate((hypre_ParCSRMatrix *)par_A, HYPRE_MEMORY_HOST);
   }
#endif

   /* Compute scaling into a fresh ParVector */
   HYPRE_SAFE_CALL(HYPRE_ParCSRMatrixComputeScalingTagged(par_A, 1, memloc_tags, num_tags,
                                                          tags, &ctx->scaling_vector));

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(orig_mat_memloc) == HYPRE_EXEC_DEVICE)
   {
      hypre_ParVectorMigrate((hypre_ParVector *)ctx->scaling_vector, orig_mat_memloc);
      hypre_ParCSRMatrixMigrate((hypre_ParCSRMatrix *)par_A, orig_mat_memloc);
   }
#endif

   /* Free memory */
   free(tags);
#else
   (void)comm;
   (void)args;
   (void)ctx;
   (void)mat_A;
   (void)dofmap;
   /* Scaling disabled on older hypre versions */
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingCompute (dofmap_custom strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingComputeDofmapCustom(MPI_Comm comm, Scaling_args *args, Scaling_context *ctx,
                           HYPRE_IJMatrix mat_A, IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A  = NULL;
   HYPRE_ParCSRMatrix par_A  = NULL;
   HYPRE_BigInt       ilower = 0, iupper = 0;
   HYPRE_Int          num_local_rows = 0;
   int                myid           = 0;

   MPI_Comm_rank(comm, &myid);

   if (!dofmap || !dofmap->data)
   {
      hypredrv_ErrorCodeSet(ERROR_MISSING_DOFMAP);
      hypredrv_ErrorMsgAdd("dofmap_custom scaling requires a dofmap to be set");
      return;
   }

   if (!args->custom_values || args->custom_values->size == 0)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("dofmap_custom scaling requires custom_values to be set");
      return;
   }

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

   /* Get local range from ParCSRMatrix directly */
   HYPRE_BigInt row_start = 0, row_end = 0, col_start = 0, col_end = 0;
   hypre_ParCSRMatrixGetLocalRange((hypre_ParCSRMatrix *)par_A, &row_start, &row_end,
                                   &col_start, &col_end);
   ilower         = row_start;
   iupper         = row_end;
   num_local_rows = (HYPRE_Int)(iupper - ilower + 1);

   if (dofmap->size != num_local_rows)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("dofmap size (%d) does not match local matrix rows (%d)",
                           dofmap->size, num_local_rows);
      return;
   }

   /* Find the maximum tag value in the dofmap */
   HYPRE_Int max_tag = -1;
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      if (dofmap->data[i] > max_tag)
      {
         max_tag = dofmap->data[i];
      }
   }

   /* Compute global max tag */
   HYPRE_Int global_max_tag = 0;
   MPI_Allreduce(&max_tag, &global_max_tag, 1, MPI_INT, MPI_MAX, comm);
   HYPRE_Int num_tags = global_max_tag + 1;

   /* Verify that the number of custom values matches the number of unique tags */
   if ((size_t)num_tags != args->custom_values->size)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd(
         "dofmap_custom: number of custom_values (%zu) does not match number of "
         "unique dofmap tags (%d)",
         args->custom_values->size, num_tags);
      return;
   }

   /* Destroy previous scaling vector if it exists (from previous system) */
   ScalingContextFreeVector(ctx);

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

   /* Fill scaling vector: for each row i, use custom_values[dofmap[i]] */
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      HYPRE_Int tag = dofmap->data[i];
      if (tag < 0 || tag >= (HYPRE_Int)args->custom_values->size)
      {
         hypre_TFree(h_values, HYPRE_MEMORY_HOST);
#ifdef HYPRE_USING_GPU
         if (d_values)
         {
            hypre_TFree(d_values, HYPRE_MEMORY_DEVICE);
         }
#endif
         HYPRE_SAFE_CALL(HYPRE_IJVectorDestroy(ctx->scaling_ijvec));
         ctx->scaling_ijvec = NULL;
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd(
            "dofmap_custom: invalid tag %d at local row %d (expected 0-%zu)", tag, i,
            args->custom_values->size - 1);
         return;
      }
      h_values[i] = (HYPRE_Complex)args->custom_values->data[tag];
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
#else
   (void)comm;
   (void)args;
   (void)ctx;
   (void)mat_A;
   (void)dofmap;
   /* Scaling disabled on older hypre versions */
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingCompute
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingCompute(MPI_Comm comm, Scaling_args *args, Scaling_context *ctx,
                        HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b, IntArray *dofmap)
{
   int myid = hypredrv_LogEnabled(2) ? hypredrv_LogRankFromComm(comm) : -1;
   if (!args || !ctx)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("hypredrv_ScalingCompute: args or ctx is NULL");
      hypredrv_Logf(2, myid, NULL, 0, "scaling compute failed: args or context is NULL");
      return;
   }

   if (!args->enabled)
   {
      ctx->enabled = 0;
      hypredrv_Logf(3, myid, NULL, 0, "scaling compute skipped (disabled)");
      return;
   }

   ctx->enabled = 1;
   ctx->type    = args->type;
   hypredrv_Logf(3, myid, NULL, 0, "scaling compute begin (type=%d)", (int)args->type);

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (args->type)
   {
      case SCALING_RHS_L2:
         ScalingComputeRHSL2(comm, ctx, vec_b);
         break;

      case SCALING_DOFMAP_MAG:
         ScalingComputeDofmapMag(comm, args, ctx, mat_A, dofmap);
         break;

      case SCALING_DOFMAP_CUSTOM:
         ScalingComputeDofmapCustom(comm, args, ctx, mat_A, dofmap);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("hypredrv_ScalingCompute: unknown scaling type");
         hypredrv_Logf(2, myid, NULL, 0,
                       "scaling compute failed: unknown scaling type=%d",
                       (int)args->type);
         break;
   }
#else
   (void)mat_A;
   (void)vec_b;
   (void)dofmap;
   /* Scaling disabled on older hypre versions - silently ignore */
   ctx->enabled = 0;
#endif
   hypredrv_Logf(3, myid, NULL, 0, "scaling compute end (enabled=%d)", ctx->enabled);
}

/*-----------------------------------------------------------------------------
 * Vector scaling helpers
 *-----------------------------------------------------------------------------*/

static void
ScalingTransformVectorRHSL2(const Scaling_context *ctx, HYPRE_IJVector vec,
                            scaling_vector_kind_t kind, int apply)
{
   if (!vec)
   {
      return;
   }

   HYPRE_Complex s = ctx->scalar_factor;
   if (s == 0.0)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingTransformVectorRHSL2: invalid scaling factor");
      return;
   }

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
   par_vec = (hypre_ParVector *)obj_vec;
   hypre_ParVectorScale(factor, par_vec);
}

static void
ScalingTransformVectorDofmap(const Scaling_context *ctx, HYPRE_IJVector vec,
                             scaling_vector_kind_t kind, int apply)
{
   if (!vec)
   {
      return;
   }

   if (!ctx->scaling_vector)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingTransformVectorDofmap: scaling vector not computed");
      return;
   }

   void            *obj_vec     = NULL;
   hypre_ParVector *par_vec     = NULL;
   hypre_ParVector *par_scaling = (hypre_ParVector *)ctx->scaling_vector;

   HYPRE_IJVectorGetObject(vec, &obj_vec);
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
      hypre_ParVector *inv_scaling = NULL;
      hypre_ParVectorPointwiseInverse(par_scaling, &inv_scaling);
      hypre_ParVectorPointwiseProduct(inv_scaling, par_vec, &par_vec);
      hypre_ParVectorDestroy(inv_scaling);
#endif
   }
#else
   (void)kind;
   (void)apply;
   hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
   hypredrv_ErrorMsgAdd("ScalingTransformVectorDofmap: requires Hypre >= v3.0.0");
#endif
}

void
hypredrv_ScalingApplyToVector(const Scaling_context *ctx, HYPRE_IJVector vec,
                              scaling_vector_kind_t kind)
{
   if (!ctx || !ctx->enabled || !vec)
   {
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (ctx->type)
   {
      case SCALING_RHS_L2:
         ScalingTransformVectorRHSL2(ctx, vec, kind, 1);
         break;

      case SCALING_DOFMAP_MAG:
      case SCALING_DOFMAP_CUSTOM:
         ScalingTransformVectorDofmap(ctx, vec, kind, 1);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("hypredrv_ScalingApplyToVector: unknown scaling type");
         break;
   }
#else
   (void)kind;
#endif
}

void
hypredrv_ScalingUndoOnVector(const Scaling_context *ctx, HYPRE_IJVector vec,
                             scaling_vector_kind_t kind)
{
   if (!ctx || !ctx->enabled || !vec)
   {
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (ctx->type)
   {
      case SCALING_RHS_L2:
         ScalingTransformVectorRHSL2(ctx, vec, kind, 0);
         break;

      case SCALING_DOFMAP_MAG:
      case SCALING_DOFMAP_CUSTOM:
         ScalingTransformVectorDofmap(ctx, vec, kind, 0);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("hypredrv_ScalingUndoOnVector: unknown scaling type");
         break;
   }
#else
   (void)kind;
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingApplyToSystem (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingApplyRHSL2(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                  HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   HYPRE_Complex      s  = ctx->scalar_factor;
   HYPRE_Complex      s2 = s * s;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

   if (mat_M != mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_M, &obj_M);
      par_M = (HYPRE_ParCSRMatrix)obj_M;
   }
   else
   {
      par_M = par_A;
   }

   /* Scale matrix: A = s^2 * A, M = s^2 * M */
   hypre_ParCSRMatrixScale(par_A, s2);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixScale(par_M, s2);
   }

   hypredrv_ScalingApplyToVector(ctx, vec_b, SCALING_VECTOR_RHS);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }
   hypredrv_ScalingApplyToVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }

   ctx->is_applied = 1;
#else
   (void)ctx;
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
   hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
   hypredrv_ErrorMsgAdd("ScalingApplyRHSL2: requires Hypre >= v3.0.0");
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingApplyToSystem (dofmap strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingApplyDofmap(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                   HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;

   if (!ctx->scaling_vector)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingApplyDofmap: scaling vector not computed");
      return;
   }

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

   if (mat_M != mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_M, &obj_M);
      par_M = (HYPRE_ParCSRMatrix)obj_M;
   }
   else
   {
      par_M = par_A;
   }

   /* Apply diagonal scaling: A = diag(M) * A * diag(M), M = diag(M) * M * diag(M) */
   hypre_ParVector *par_scaling_vec = (hypre_ParVector *)ctx->scaling_vector;
   hypre_ParCSRMatrixDiagScale(par_A, par_scaling_vec, par_scaling_vec);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixDiagScale(par_M, par_scaling_vec, par_scaling_vec);
   }

   hypredrv_ScalingApplyToVector(ctx, vec_b, SCALING_VECTOR_RHS);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }
   hypredrv_ScalingApplyToVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }

   ctx->is_applied = 1;
#else
   (void)ctx;
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingApplyToSystem
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingApplyToSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A,
                              HYPRE_IJMatrix mat_M, HYPRE_IJVector vec_b,
                              HYPRE_IJVector vec_x)
{
   int myid = hypredrv_LogEnabled(2) ? hypredrv_LogRankFromComm(MPI_COMM_WORLD) : -1;
   if (!ctx || !ctx->enabled)
   {
      return;
   }

   if (ctx->is_applied)
   {
      /* Already applied, skip */
      hypredrv_Logf(3, myid, NULL, 0, "scaling apply skipped (already applied)");
      return;
   }

   hypredrv_Logf(3, myid, NULL, 0, "scaling apply begin (type=%d)", (int)ctx->type);

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (ctx->type)
   {
      case SCALING_RHS_L2:
         ScalingApplyRHSL2(ctx, mat_A, mat_M, vec_b, vec_x);
         break;

      case SCALING_DOFMAP_MAG:
      case SCALING_DOFMAP_CUSTOM:
         ScalingApplyDofmap(ctx, mat_A, mat_M, vec_b, vec_x);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("hypredrv_ScalingApplyToSystem: unknown scaling type");
         hypredrv_Logf(2, myid, NULL, 0, "scaling apply failed: unknown scaling type=%d",
                       (int)ctx->type);
         break;
   }
#else
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
   hypredrv_Logf(3, myid, NULL, 0, "scaling apply end (is_applied=%d)", ctx->is_applied);
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingUndoOnSystem (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingUndoRHSL2(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                 HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   HYPRE_Complex      s      = ctx->scalar_factor;
   HYPRE_Complex      s2     = s * s;
   HYPRE_Complex      inv_s2 = 1.0 / s2;

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

   if (mat_M != mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_M, &obj_M);
      par_M = (HYPRE_ParCSRMatrix)obj_M;
   }
   else
   {
      par_M = par_A;
   }

   hypredrv_ScalingUndoOnVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }
   hypredrv_ScalingUndoOnVector(ctx, vec_b, SCALING_VECTOR_RHS);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }

   /* Unscale matrices: A = (1/s^2) * A, M = (1/s^2) * M */
   hypre_ParCSRMatrixScale(par_A, inv_s2);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixScale(par_M, inv_s2);
   }

   ctx->is_applied = 0;
#else
   (void)ctx;
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
   hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
   hypredrv_ErrorMsgAdd("ScalingUndoRHSL2: requires Hypre >= v3.0.0");
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingUndoOnSystem (dofmap strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingUndoDofmap(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                  HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;

   if (!ctx->scaling_vector)
   {
      hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
      hypredrv_ErrorMsgAdd("ScalingUndoDofmap: scaling vector not computed");
      return;
   }

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

   if (mat_M != mat_A)
   {
      HYPRE_IJMatrixGetObject(mat_M, &obj_M);
      par_M = (HYPRE_ParCSRMatrix)obj_M;
   }
   else
   {
      par_M = par_A;
   }

   hypredrv_ScalingUndoOnVector(ctx, vec_x, SCALING_VECTOR_UNKNOWN);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }
   hypredrv_ScalingUndoOnVector(ctx, vec_b, SCALING_VECTOR_RHS);
   if (hypredrv_ErrorCodeGet())
   {
      return;
   }

   /* Unscale matrices: A = diag(1/M) * A * diag(1/M), M = diag(1/M) * M * diag(1/M) */
   hypre_ParVector *par_scaling = (hypre_ParVector *)ctx->scaling_vector;
   hypre_ParVector *inv_scaling = NULL;

   hypre_ParVectorPointwiseInverse(par_scaling, &inv_scaling);
   hypre_ParCSRMatrixDiagScale(par_A, inv_scaling, inv_scaling);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixDiagScale(par_M, inv_scaling, inv_scaling);
   }

   /* Clean up temporary inverse scaling vector */
   hypre_ParVectorDestroy(inv_scaling);

   ctx->is_applied = 0;
#else
   (void)ctx;
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
}

/*-----------------------------------------------------------------------------
 * hypredrv_ScalingUndoOnSystem
 *-----------------------------------------------------------------------------*/

void
hypredrv_ScalingUndoOnSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A,
                             HYPRE_IJMatrix mat_M, HYPRE_IJVector vec_b,
                             HYPRE_IJVector vec_x)
{
   int myid = hypredrv_LogEnabled(2) ? hypredrv_LogRankFromComm(MPI_COMM_WORLD) : -1;
   if (!ctx || !ctx->enabled || !ctx->is_applied)
   {
      return;
   }

   hypredrv_Logf(3, myid, NULL, 0, "scaling undo begin (type=%d)", (int)ctx->type);

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   switch (ctx->type)
   {
      case SCALING_RHS_L2:
         ScalingUndoRHSL2(ctx, mat_A, mat_M, vec_b, vec_x);
         break;

      case SCALING_DOFMAP_MAG:
      case SCALING_DOFMAP_CUSTOM:
         ScalingUndoDofmap(ctx, mat_A, mat_M, vec_b, vec_x);
         break;

      default:
         hypredrv_ErrorCodeSet(ERROR_UNKNOWN);
         hypredrv_ErrorMsgAdd("hypredrv_ScalingUndoOnSystem: unknown scaling type");
         hypredrv_Logf(2, myid, NULL, 0, "scaling undo failed: unknown scaling type=%d",
                       (int)ctx->type);
         break;
   }
#else
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
   hypredrv_Logf(3, myid, NULL, 0, "scaling undo end (is_applied=%d)", ctx->is_applied);
}
