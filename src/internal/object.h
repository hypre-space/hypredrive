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
   MPI_Comm                comm;
   int                    *states;
   input_args             *iargs;
   IntArray               *dofmap;
   HYPRE_IJMatrix          mat_A;
   HYPRE_IJMatrix          mat_M;
   HYPRE_IJVector          vec_b;
   HYPRE_IJVector          vec_x;
   HYPRE_IJVector          vec_x0;
   HYPRE_IJVector          vec_xref;
   HYPRE_IJVector          vec_nn;
   HYPRE_IJVector          vec_ns; /* orthonormalized null space modes projected out of sol. */
   HYPRE_IJVector         *vec_s;
   HYPRE_IJMatrix          mat_G; /* Discrete gradient */
   HYPRE_IJMatrix          mat_C; /* Discrete curl */
   HYPRE_Precon            precon;
   HYPRE_Solver            solver;
   Scaling_context        *scaling_ctx;
   Stats                  *stats;
   struct hypredrv_struct *next_live; /* linked-list hook for the runtime registry */
   PreconReuseTimesteps    precon_reuse_timesteps;
   HYPRE_IJVector          vec_coord[3]; /* Vertex coordinates */
   PreconReuseState        precon_reuse_state;

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
