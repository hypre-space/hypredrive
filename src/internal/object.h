#ifndef HYPREDRV_OBJECT_HEADER_
#define HYPREDRV_OBJECT_HEADER_

#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/scaling.h"
#include "internal/stats.h"

/* Fields are ordered by descending alignment to minimize struct padding */
typedef struct hypredrv_struct
{
   /* Pointers, opaque HYPRE handles, and aggregates (8-byte aligned). */
   MPI_Comm         comm;
   int             *states; /* Array of state indices */
   input_args      *iargs; /* Input arguments (passed via YAML) */
   IntArray        *dofmap; /* Mapping array for degrees-of-freedom */
   HYPRE_IJMatrix   mat_A; /* System matrix */
   HYPRE_IJMatrix   mat_M; /* Matrix used to build the preconditioner */
   HYPRE_IJVector   vec_b; /* Right-hand side vector */
   HYPRE_IJVector   vec_x; /* Solution vector */
   HYPRE_IJVector   vec_x0; /* Initial solution vector */
   HYPRE_IJVector   vec_xref; /* Reference solution vector */
   HYPRE_IJVector   vec_nn; /* Near-null space modes */
   HYPRE_IJVector   vec_ns; /* Orthonormalized null space modes projected out of sol. */
   HYPRE_IJVector  *vec_s; /* Array of vector states */
   HYPRE_IJMatrix   mat_G; /* Discrete gradient */
   HYPRE_IJMatrix   mat_C; /* Discrete curl */
   HYPRE_Precon     precon; /* Preconditioner object */
   HYPRE_Solver     solver; /* Solver object */
   Scaling_context *scaling_ctx; /* Scaling context */
   Stats           *stats; /* Solver statistics structure */
   struct hypredrv_struct *next_live; /* linked-list hook for the runtime registry */
   PreconReuseTimesteps    precon_reuse_timesteps; /* Preconditioner reuse structure */
   HYPRE_IJVector          vec_coord[3]; /* Vertex coordinates */
   PreconReuseState        precon_reuse_state; /* Preconditioner reuse state */

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
   bool owns_mat_G;
   bool owns_mat_C;
   bool owns_vec_coord;
   bool owns_mat_A;
   bool owns_mat_M;
   bool owns_vec_b;
   bool owns_vec_x;
   bool owns_vec_x0;
   bool owns_vec_xref;
} hypredrv_t;

#endif /* HYPREDRV_OBJECT_HEADER_ */
