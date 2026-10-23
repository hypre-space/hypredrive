#ifndef HYPREDRV_OBJECT_HEADER_
#define HYPREDRV_OBJECT_HEADER_

#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/scaling.h"
#include "internal/slots.h"
#include "internal/stats.h"
#include "vector_tags.h"

/* Fields are ordered by descending alignment to minimize struct padding */
typedef struct hypredrv_struct
{
   /* Pointers, opaque HYPRE handles, and aggregates (8-byte aligned). */
   MPI_Comm            comm;
   int                *states; /* Array of state indices */
   input_args         *iargs;  /* Input arguments (passed via YAML) */
   IntArray           *dofmap; /* Mapping array for degrees-of-freedom */
   hypredrv_MatrixSlot mat_A;  /* System matrix */
   hypredrv_MatrixSlot mat_M;  /* Matrix used to build the preconditioner */
   const input_args   *precmat_sequence_cache_args;
   hypredrv_VectorSlot vec_b;    /* Right-hand side vector */
   hypredrv_VectorSlot vec_x;    /* Solution vector */
   hypredrv_VectorSlot vec_x0;   /* Initial solution vector */
   hypredrv_VectorSlot vec_xref; /* Reference solution vector */
   HYPRE_IJVector      vec_nn;   /* Near-null space modes */
   HYPRE_IJVector  vec_ns;    /* Orthonormalized null space modes projected out of sol. */
   HYPRE_IJVector *vec_s;     /* Array of vector states */
   hypredrv_MatrixSlot mat_G; /* Discrete gradient */
   hypredrv_MatrixSlot mat_C; /* Discrete curl */
   /* Per-reduction-level app-provided coarse operators; MGR borrows the ParCSR. */
   hypredrv_MatrixSlot     mat_coarse_schur[MAX_MGR_LEVELS - 1];
   HYPRE_Precon            precon; /* Preconditioner object */
   HYPRE_Solver            solver; /* Solver object */
   hypredrv_VectorTags    *vector_tags;
   Scaling_context        *scaling_ctx; /* Scaling context */
   Stats                  *stats;       /* Solver statistics structure */
   struct hypredrv_struct *next_live;   /* linked-list hook for the runtime registry */
   PreconReuseTimesteps    precon_reuse_timesteps; /* Preconditioner reuse structure */
   hypredrv_VectorSlot     vec_coord[3];           /* Vertex coordinates */
   PreconReuseState        precon_reuse_state;     /* Preconditioner reuse state */

   /* 32-bit scalars. */
   int mypid;
   int nprocs;
   int nstates;
   int num_ns;
   int runtime_object_id;
   int current_system_index;
   int preferred_exec_policy;

   /* Boolean flags */
   bool lib_mode;
   bool precon_is_setup;
   bool stats_printed;
} hypredrv_t;

#endif /* HYPREDRV_OBJECT_HEADER_ */
