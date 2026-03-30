#ifndef HYPREDRV_OBJECT_HEADER_
#define HYPREDRV_OBJECT_HEADER_

#include "HYPREDRV.h"
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/scaling.h"
#include "internal/stats.h"

typedef struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   int      nstates;
   int     *states;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix  mat_A;
   HYPRE_IJMatrix  mat_M;
   HYPRE_IJVector  vec_b;
   HYPRE_IJVector  vec_x;
   HYPRE_IJVector  vec_x0;
   HYPRE_IJVector  vec_xref;
   HYPRE_IJVector  vec_nn;
   HYPRE_IJVector *vec_s;
   bool            owns_mat_M;
   bool            owns_vec_x;
   bool            owns_vec_x0;
   bool            owns_vec_xref;

   HYPRE_Precon precon;
   HYPRE_Solver solver;
   bool         precon_is_setup;

   Scaling_context     *scaling_ctx;
   PreconReuseTimesteps precon_reuse_timesteps;
   PreconReuseState     precon_reuse_state;

   Stats *stats;
   bool   stats_printed;
   int    runtime_object_id;
   int    current_system_index;

   /* Linked-list hook used by the internal runtime registry. */
   struct hypredrv_struct *next_live;
} hypredrv_t;

#endif /* HYPREDRV_OBJECT_HEADER_ */
