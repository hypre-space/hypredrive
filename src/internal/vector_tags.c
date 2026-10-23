/******************************************************************************
 * Copyright (c) 2026 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "vector_tags.h"
#include <stdlib.h>
#include "_hypre_parcsr_mv.h"
#include "internal/error.h"
#include "internal/utils.h"
#include "object.h"

#if HYPRE_CHECK_MIN_VERSION(30000, 0)
typedef struct
{
   hypre_Vector *vector;
   HYPRE_Int    *tags;
   HYPRE_Int     owns_tags;
   HYPRE_Int     num_tags;
} VectorTagAttachment;

struct hypredrv_VectorTags
{
   HYPRE_Int          *data;
   int                 count;
   VectorTagAttachment attachments[4];
};

/* Both tests depend only on collective-consistent parsed solver options. */
static int
SolverPrintsBlockResiduals(const input_args *args)
{
   if (!args || args->general.exec_policy)
   {
      return 0;
   }
   switch (args->solver_method)
   {
      case SOLVER_GMRES:
         return args->solver.gmres.print_level > 2;
      case SOLVER_FGMRES:
         return args->solver.fgmres.print_level > 1;
      default:
         return 0;
   }
}

static int
VectorTagsLayoutValid(HYPREDRV_t h)
{
   HYPRE_IJVector vectors[] = {h->vec_b.handle, h->vec_x.handle, h->vec_x0.handle,
                               h->vec_xref.handle};
   for (size_t i = 0; i < sizeof(vectors) / sizeof(vectors[0]); i++)
   {
      if (!vectors[i])
      {
         continue;
      }
      void *object = NULL;
      HYPRE_IJVectorGetObject(vectors[i], &object);
      if (!object)
      {
         return 0;
      }
      hypre_Vector *local = hypre_ParVectorLocalVector((hypre_ParVector *)object);
      if ((size_t)hypre_VectorSize(local) != h->dofmap->size)
      {
         return 0;
      }
   }
   return 1;
}

static void
VectorTagsAttach(hypredrv_VectorTags *ctx, HYPRE_IJVector ij, HYPRE_Int num_tags)
{
   if (!ij)
   {
      return;
   }
   void *object = NULL;
   HYPRE_IJVectorGetObject(ij, &object);
   hypre_Vector *local = hypre_ParVectorLocalVector((hypre_ParVector *)object);
   for (int i = 0; i < ctx->count; i++)
   {
      if (ctx->attachments[i].vector == local)
      {
         return;
      }
   }
   VectorTagAttachment *attachment = &ctx->attachments[ctx->count++];
   attachment->vector              = local;
   attachment->tags                = hypre_VectorTags(local);
   attachment->owns_tags           = hypre_VectorOwnsTags(local);
   attachment->num_tags            = hypre_VectorNumTags(local);
   HYPRE_IJVectorSetTags(ij, 0, num_tags, ctx->data);
}
#endif

void
hypredrv_VectorTagsClear(HYPREDRV_t h)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   hypredrv_VectorTags *ctx = h->vector_tags;
   if (!ctx)
   {
      return;
   }
   for (int i = 0; i < ctx->count; i++)
   {
      VectorTagAttachment *attachment = &ctx->attachments[i];
      /* hypre's SetTags(NULL) frees even borrowed tags. Restore metadata
       * directly so the shared buffer is freed exactly once, below. */
      hypre_VectorTags(attachment->vector)     = attachment->tags;
      hypre_VectorOwnsTags(attachment->vector) = attachment->owns_tags;
      hypre_VectorNumTags(attachment->vector)  = attachment->num_tags;
   }
   free(ctx->data);
   free(ctx);
   h->vector_tags = NULL;
#else
   (void)h;
#endif
}

uint32_t
hypredrv_VectorTagsSet(HYPREDRV_t h)
{
#if HYPRE_CHECK_MIN_VERSION(30000, 0)
   hypredrv_VectorTagsClear(h);
   if (!SolverPrintsBlockResiduals(h->iargs))
   {
      return ERROR_NONE;
   }

   /* Empty ranks must participate too; disable on every rank if any layout
    * lacks a dofmap. Never make a local-only tagged reduction decision. */
   int valid     = h->dofmap && (h->dofmap->data || !h->dofmap->size);
   int all_valid = 0;
   if (valid && !VectorTagsLayoutValid(h))
   {
      valid = -1;
   }
   MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, h->comm);
   if (all_valid < 0)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Vector tag layout does not match the local dofmap");
      return ERROR_INVALID_VAL;
   }
   if (!all_valid)
   {
      return ERROR_NONE;
   }

   hypredrv_VectorTags *ctx = hypredrv_Calloc(1, sizeof(*ctx), "vector tag context");
   if (ctx)
   {
      ctx->data = hypredrv_Calloc(h->dofmap->size, sizeof(*ctx->data), "vector tags");
   }
   valid = ctx && ctx->data;
   MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, h->comm);
   if (!all_valid)
   {
      if (ctx)
      {
         free(ctx->data);
         free(ctx);
      }
      hypredrv_AllocationFailure("Memory allocation failed");
      return ERROR_ALLOCATION;
   }
   for (size_t i = 0; i < h->dofmap->size; i++)
   {
      ctx->data[i] = (HYPRE_Int)h->dofmap->data[i];
   }
   HYPRE_Int num_tags = 1;
   if (h->dofmap->g_unique_size)
   {
      num_tags = (HYPRE_Int)h->dofmap->g_unique_data[h->dofmap->g_unique_size - 1] + 1;
   }
   VectorTagsAttach(ctx, h->vec_b.handle, num_tags);
   VectorTagsAttach(ctx, h->vec_x.handle, num_tags);
   VectorTagsAttach(ctx, h->vec_x0.handle, num_tags);
   VectorTagsAttach(ctx, h->vec_xref.handle, num_tags);
   h->vector_tags = ctx;
#else
   (void)h;
#endif
   return ERROR_NONE;
}
