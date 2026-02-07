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
#include "linsys.h"
#include "utils.h"

#define Scaling_FIELDS(_prefix)                              \
   ADD_FIELD_OFFSET_ENTRY(_prefix, enabled, FieldTypeIntSet) \
   ADD_FIELD_OFFSET_ENTRY(_prefix, type, FieldTypeIntSet)    \
   ADD_FIELD_OFFSET_ENTRY(_prefix, custom_values, FieldTypeDoubleArraySet)

/* Define num_fields macro */
#define Scaling_NUM_FIELDS \
   (sizeof(Scaling_field_offset_map) / sizeof(Scaling_field_offset_map[0]))

GENERATE_PREFIXED_COMPONENTS(Scaling) // LCOV_EXCL_LINE

/*-----------------------------------------------------------------------------
 * ScalingGetValidValues
 *-----------------------------------------------------------------------------*/

StrIntMapArray
ScalingGetValidValues(const char *key)
{
   if (!strcmp(key, "enabled"))
   {
      return STR_INT_MAP_ARRAY_CREATE_ON_OFF();
   }
   if (!strcmp(key, "type"))
   {
      static StrIntMap map[] = {{"rhs_l2", (int)SCALING_RHS_L2},
                                {"dofmap_mag", (int)SCALING_DOFMAP_MAG},
                                {"dofmap_custom", (int)SCALING_DOFMAP_CUSTOM}};
      return STR_INT_MAP_ARRAY_CREATE(map);
   }
   return STR_INT_MAP_ARRAY_VOID();
}

/*-----------------------------------------------------------------------------
 * ScalingSetDefaultArgs
 *-----------------------------------------------------------------------------*/

void
ScalingSetDefaultArgs(Scaling_args *args)
{
   args->enabled       = 0;
   args->type          = SCALING_RHS_L2;
   args->custom_values = NULL;
}

/*-----------------------------------------------------------------------------
 * ScalingContextCreate
 *-----------------------------------------------------------------------------*/

void
ScalingContextCreate(Scaling_context **ctx_ptr)
{
   Scaling_context *ctx = (Scaling_context *)malloc(sizeof(Scaling_context));
   ctx->enabled         = 0;
   ctx->type            = SCALING_RHS_L2;
   ctx->is_applied      = 0;
   ctx->scalar_factor   = 1.0;
   ctx->scaling_vector  = NULL;
   ctx->scaling_ijvec   = NULL;
   *ctx_ptr             = ctx;
}

/*-----------------------------------------------------------------------------
 * ScalingContextDestroy
 *-----------------------------------------------------------------------------*/

void
ScalingContextDestroy(Scaling_context **ctx_ptr)
{
   if (!ctx_ptr || !*ctx_ptr)
   {
      return;
   }

   Scaling_context *ctx = *ctx_ptr;

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   if (ctx->scaling_ijvec)
   {
      HYPRE_SAFE_CALL(HYPRE_IJVectorDestroy(ctx->scaling_ijvec));
      ctx->scaling_ijvec = NULL;
   }
   /* scaling_vector is owned by scaling_ijvec, so no need to destroy separately */
   ctx->scaling_vector = NULL;
#endif

   free(ctx);
   *ctx_ptr = NULL;
}

/*-----------------------------------------------------------------------------
 * ScalingCompute (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingComputeRHSL2(MPI_Comm comm, Scaling_context *ctx, HYPRE_IJVector vec_b)
{
   double b_norm = 0.0;

   LinearSystemComputeVectorNorm(vec_b, "L2", &b_norm);

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
 * ScalingCompute (dofmap_mag strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingComputeDofmapMag(MPI_Comm comm, Scaling_args *args, Scaling_context *ctx,
                        HYPRE_IJMatrix mat_A, IntArray *dofmap)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
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
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ErrorMsgAdd("dofmap scaling requires a dofmap to be set");
      return;
   }

   HYPRE_IJMatrixGetObject(mat_A, &obj_A);
   par_A = (HYPRE_ParCSRMatrix)obj_A;

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
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dofmap size (%d) does not match local matrix rows (%d)", dofmap->size,
                  num_local_rows);
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

   /* Destroy previous scaling_ijvec if it exists (from previous system) */
   if (ctx->scaling_ijvec)
   {
      HYPRE_IJVectorDestroy(ctx->scaling_ijvec);
      ctx->scaling_ijvec  = NULL;
      ctx->scaling_vector = NULL;
   }

   /* Create IJVector wrapper for scaling vector */
   HYPRE_SAFE_CALL(HYPRE_IJVectorCreate(comm, ilower, iupper, &ctx->scaling_ijvec));
   HYPRE_SAFE_CALL(HYPRE_IJVectorSetObjectType(ctx->scaling_ijvec, HYPRE_PARCSR));
   HYPRE_SAFE_CALL(HYPRE_IJVectorInitialize(ctx->scaling_ijvec));

   /* Get ParVector from IJVector */
   void *obj_scaling = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling);
   ctx->scaling_vector = (HYPRE_ParVector)obj_scaling;

   /* Compute scaling using Hypre's tagged API */
   /* Use scaling_type = 1 by default */
   HYPRE_ParVector scaling_parvec = NULL;
   HYPRE_SAFE_CALL(HYPRE_ParCSRMatrixComputeScalingTagged(
      par_A, 1, HYPRE_MEMORY_HOST, num_tags, tags, &scaling_parvec));

   /* Copy the computed scaling data into scaling_ijvec */
   void           *obj_scaling_vec = NULL;
   HYPRE_ParVector par_scaling_vec = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling_vec);
   par_scaling_vec = (HYPRE_ParVector)obj_scaling_vec;

   /* Copy data from scaling_parvec into scaling_ijvec's ParVector */
   HYPRE_SAFE_CALL(HYPRE_ParVectorCopy(scaling_parvec, par_scaling_vec));

   /* Destroy the temporary vector returned by HYPRE_ParCSRMatrixComputeScalingTagged */
   HYPRE_SAFE_CALL(HYPRE_ParVectorDestroy(scaling_parvec));

   /* Update scaling_vector to point to the vector wrapped by scaling_ijvec */
   ctx->scaling_vector = par_scaling_vec;

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
 * ScalingCompute (dofmap_custom strategy)
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
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ErrorMsgAdd("dofmap_custom scaling requires a dofmap to be set");
      return;
   }

   if (!args->custom_values || args->custom_values->size == 0)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dofmap_custom scaling requires custom_values to be set");
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
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dofmap size (%d) does not match local matrix rows (%d)", dofmap->size,
                  num_local_rows);
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
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("dofmap_custom: number of custom_values (%zu) does not match number of "
                  "unique dofmap tags (%d)",
                  args->custom_values->size, num_tags);
      return;
   }

   /* Destroy previous scaling_ijvec if it exists (from previous system) */
   if (ctx->scaling_ijvec)
   {
      HYPRE_IJVectorDestroy(ctx->scaling_ijvec);
      ctx->scaling_ijvec  = NULL;
      ctx->scaling_vector = NULL;
   }

   /* Create IJVector wrapper for scaling vector */
   HYPRE_SAFE_CALL(HYPRE_IJVectorCreate(comm, ilower, iupper, &ctx->scaling_ijvec));
   HYPRE_SAFE_CALL(HYPRE_IJVectorSetObjectType(ctx->scaling_ijvec, HYPRE_PARCSR));
   HYPRE_SAFE_CALL(HYPRE_IJVectorInitialize(ctx->scaling_ijvec));

   /* Get ParVector from IJVector */
   void *obj_scaling = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling);
   ctx->scaling_vector = (HYPRE_ParVector)obj_scaling;

   /* Get local data array from ParVector */
   hypre_ParVector *par_scaling_vec = (hypre_ParVector *)ctx->scaling_vector;
   HYPRE_Real *local_data = hypre_VectorData(hypre_ParVectorLocalVector(par_scaling_vec));

   /* Fill scaling vector: for each row i, use custom_values[dofmap[i]] */
   for (HYPRE_Int i = 0; i < num_local_rows; i++)
   {
      HYPRE_Int tag = dofmap->data[i];
      if (tag < 0 || tag >= (HYPRE_Int)args->custom_values->size)
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("dofmap_custom: invalid tag %d at local row %d (expected 0-%zu)",
                     tag, i, args->custom_values->size - 1);
         return;
      }
      local_data[i] = (HYPRE_Real)args->custom_values->data[tag];
   }

   /* Assemble the vector */
   HYPRE_SAFE_CALL(HYPRE_IJVectorAssemble(ctx->scaling_ijvec));
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
 * ScalingCompute
 *-----------------------------------------------------------------------------*/

void
ScalingCompute(MPI_Comm comm, Scaling_args *args, Scaling_context *ctx,
               HYPRE_IJMatrix mat_A, HYPRE_IJVector vec_b, IntArray *dofmap)
{
   if (!args || !ctx)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("ScalingCompute: args or ctx is NULL");
      return;
   }

   if (!args->enabled)
   {
      ctx->enabled = 0;
      return;
   }

   ctx->enabled = 1;
   ctx->type    = args->type;

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
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("ScalingCompute: unknown scaling type");
         break;
   }
#else
   (void)mat_A;
   (void)vec_b;
   (void)dofmap;
   /* Scaling disabled on older hypre versions - silently ignore */
   ctx->enabled = 0;
#endif
}

/*-----------------------------------------------------------------------------
 * ScalingApplyToSystem (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingApplyRHSL2(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                  HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL, *obj_b = NULL, *obj_x = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;
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

   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   par_b = (hypre_ParVector *)obj_b;

   /* Scale matrix: A = s^2 * A, M = s^2 * M */
   hypre_ParCSRMatrixScale(par_A, s2);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixScale(par_M, s2);
   }

   /* Scale RHS: b = s * b */
   hypre_ParVectorScale(s, par_b);

   /* Solution vector starts as zero, no need to scale */

   ctx->is_applied = 1;
#else
   (void)ctx;
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
   ErrorCodeSet(ERROR_UNKNOWN);
   ErrorMsgAdd("ScalingApplyRHSL2: requires Hypre >= v3.0.0");
#endif
}

/*-----------------------------------------------------------------------------
 * ScalingApplyToSystem (dofmap strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingApplyDofmap(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                   HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL, *obj_b = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   hypre_ParVector   *par_b = NULL;

   if (!ctx->scaling_vector)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("ScalingApplyDofmap: scaling vector not computed");
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

   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   par_b = (hypre_ParVector *)obj_b;

   /* Apply diagonal scaling: A = diag(M) * A * diag(M), M = diag(M) * M * diag(M) */
   void            *obj_scaling_vec = NULL;
   hypre_ParVector *par_scaling_vec = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling_vec);
   par_scaling_vec = (hypre_ParVector *)obj_scaling_vec;
   hypre_ParCSRMatrixDiagScale(par_A, par_scaling_vec, par_scaling_vec);
   if (par_M != par_A)
   {
      hypre_ParCSRMatrixDiagScale(par_M, par_scaling_vec, par_scaling_vec);
   }

   /* Scale RHS: b = M .* b (pointwise product) */
   void            *obj_scaling = NULL;
   hypre_ParVector *par_scaling = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling);
   par_scaling = (hypre_ParVector *)obj_scaling;

   hypre_ParVectorPointwiseProduct(par_scaling, par_b, &par_b);

   /* Solution vector starts as zero, no need to scale */

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
 * ScalingApplyToSystem
 *-----------------------------------------------------------------------------*/

void
ScalingApplyToSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                     HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
   if (!ctx || !ctx->enabled)
   {
      return;
   }

   if (ctx->is_applied)
   {
      /* Already applied, skip */
      return;
   }

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
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("ScalingApplyToSystem: unknown scaling type");
         break;
   }
#else
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
}

/*-----------------------------------------------------------------------------
 * ScalingUndoOnSystem (rhs_l2 strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingUndoRHSL2(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                 HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL, *obj_b = NULL, *obj_x = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   hypre_ParVector   *par_b = NULL, *par_x = NULL;
   HYPRE_Complex      s      = ctx->scalar_factor;
   HYPRE_Complex      s2     = s * s;
   HYPRE_Complex      inv_s  = 1.0 / s;
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

   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   par_b = (hypre_ParVector *)obj_b;

   HYPRE_IJVectorGetObject(vec_x, &obj_x);
   par_x = (hypre_ParVector *)obj_x;

   /* Unscale solution: x = s * x (since we solved for y, and x = M*y = s*y) */
   hypre_ParVectorScale(inv_s, par_x);

   /* Unscale RHS: b = (1/s) * b */
   hypre_ParVectorScale(inv_s, par_b);

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
   ErrorCodeSet(ERROR_UNKNOWN);
   ErrorMsgAdd("ScalingUndoRHSL2: requires Hypre >= v3.0.0");
#endif
}

/*-----------------------------------------------------------------------------
 * ScalingUndoOnSystem (dofmap strategy)
 *-----------------------------------------------------------------------------*/

static void
ScalingUndoDofmap(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                  HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   void              *obj_A = NULL, *obj_M = NULL, *obj_b = NULL, *obj_x = NULL;
   HYPRE_ParCSRMatrix par_A = NULL, par_M = NULL;
   HYPRE_ParVector    par_b = NULL, par_x = NULL;

   if (!ctx->scaling_vector)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("ScalingUndoDofmap: scaling vector not computed");
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

   HYPRE_IJVectorGetObject(vec_b, &obj_b);
   par_b = (hypre_ParVector *)obj_b;

   HYPRE_IJVectorGetObject(vec_x, &obj_x);
   par_x = (HYPRE_ParVector)obj_x;

   /* Unscale solution: x = M .* x (pointwise product) */
   void            *obj_scaling = NULL;
   hypre_ParVector *par_scaling = NULL;
   HYPRE_IJVectorGetObject(ctx->scaling_ijvec, &obj_scaling);
   par_scaling = (hypre_ParVector *)obj_scaling;

   hypre_ParVectorPointwiseProduct(par_scaling, par_x, &par_x);

   /* Unscale RHS: b = (1/M) .* b (pointwise division) */
   hypre_ParVectorPointwiseDivision(par_scaling, par_b, &par_b);

   /* Unscale matrices: A = diag(1/M) * A * diag(1/M), M = diag(1/M) * M * diag(1/M) */
   /* Create inverse scaling vector */
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
 * ScalingUndoOnSystem
 *-----------------------------------------------------------------------------*/

void
ScalingUndoOnSystem(Scaling_context *ctx, HYPRE_IJMatrix mat_A, HYPRE_IJMatrix mat_M,
                    HYPRE_IJVector vec_b, HYPRE_IJVector vec_x)
{
   if (!ctx || !ctx->enabled || !ctx->is_applied)
   {
      return;
   }

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
         ErrorCodeSet(ERROR_UNKNOWN);
         ErrorMsgAdd("ScalingUndoOnSystem: unknown scaling type");
         break;
   }
#else
   (void)mat_A;
   (void)mat_M;
   (void)vec_b;
   (void)vec_x;
#endif
}
