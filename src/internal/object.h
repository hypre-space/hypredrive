#ifndef HYPREDRV_OBJECT_HEADER_
#define HYPREDRV_OBJECT_HEADER_

#include <mpi.h>
#include <stdint.h>
#include "internal/args.h"
#include "internal/containers.h"
#include "internal/object_fwd.h"
#include "internal/scaling.h"
#include "internal/slots.h"
#include "internal/stats.h"
#include "vector_tags.h"

/* Fields are ordered by descending alignment to minimize struct padding */
struct hypredrv_struct
{
   /* Pointers, opaque HYPRE handles, and aggregates (8-byte aligned). */
   MPI_Comm            comm;
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
   HYPRE_IJVector *vec_s;     /* Borrowed state handles in logical order */
   hypredrv_MatrixSlot mat_G; /* Discrete gradient */
   hypredrv_MatrixSlot mat_C; /* Discrete curl */
   /* Per-reduction-level app-provided coarse operators; MGR borrows the ParCSR. */
   hypredrv_MatrixSlot     mat_coarse_schur[MAX_MGR_LEVELS - 1];
   HYPRE_Precon            precon; /* Preconditioner object */
   hypredrv_Solver        *solver; /* Bound solver runtime */
   hypredrv_VectorTags    *vector_tags;
   Scaling_context        *scaling_ctx; /* Scaling context */
   Stats                  *stats;       /* Solver statistics structure */
   struct hypredrv_struct *next_live;   /* linked-list hook for the runtime registry */
   PreconReuseTimesteps    precon_reuse_timesteps; /* Preconditioner reuse structure */
   hypredrv_VectorSlot     vec_coord[3];           /* Vertex coordinates */
   PreconReuseState       *precon_reuse_state;     /* Bound preconditioner reuse state */

   /* 32-bit scalars. */
   int mypid;
   int nprocs;
   int nstates;
   int num_ns;
   int runtime_object_id;
   int current_system_index;

   /* Boolean flags */
   bool lib_mode;
   bool precon_is_setup;
   bool stats_printed;
};

/* Attachment updates receive validated native handles. Commit a complete
 * vector update together so aliases across coordinates/linear-system slots
 * retain their original ownership. Matrix replacement covers every matrix slot. */
typedef struct
{
   hypredrv_VectorSlot *slot;
   HYPRE_IJVector       handle;
   bool                 owned;
} hypredrv_VectorUpdate;
void hypredrv_ObjectInstallVectors(hypredrv_t *, int, const hypredrv_VectorUpdate *);
void hypredrv_ObjectReplaceMatrix(hypredrv_t *, hypredrv_MatrixSlot *, HYPRE_IJMatrix,
                                  bool);

/* API entry checks preserve the caller's diagnostic scope. The API checks output
 * pointers before creation/destruction; shared helpers receive live objects. */
uint32_t hypredrv_ObjectCheckInit(void);
uint32_t hypredrv_ObjectCheckLive(hypredrv_t *);
void     hypredrv_ObjectInitialize(void);
void     hypredrv_ObjectFinalize(void);
uint32_t hypredrv_ObjectCreate(MPI_Comm, hypredrv_t **);
uint32_t hypredrv_ObjectDestroy(hypredrv_t **);
void     hypredrv_ObjectDestroySolvers(hypredrv_t *);
void     hypredrv_ObjectApplyRuntimeSettings(hypredrv_t *);
void     hypredrv_ObjectPrepareExecution(hypredrv_t *, void *, int);
void     hypredrv_ObjectPrintStats(hypredrv_t *, int);
void     hypredrv_ObjectDumpLinearSystem(hypredrv_t *, int);

/* Registration validates and copies initialized, borrowed state handles. Other
 * state operations receive indices checked at the public boundary; correction
 * additionally receives an installed solution handle. Layout compatibility is
 * checked when the copy/correction pair is selected. No diagnostic scope reset. */
uint32_t hypredrv_ObjectSetStateVectors(hypredrv_t *, int, const HYPRE_IJVector *);
uint32_t hypredrv_ObjectCopyStateVectors(hypredrv_t *, int, int);
uint32_t hypredrv_ObjectApplyStateCorrection(hypredrv_t *, int);

/* Configured construction receives a live object and validates arguments collectively.
 * Each configured replacement is installed only after collective success. Internal calls
 * preserve their caller's diagnostic scope; public entry points reset it once. */
uint32_t hypredrv_ObjectLinearSystemBuild(hypredrv_t *);

/* Solver workflows receive live objects. They collectively validate arguments,
 * native lifecycle state and required handles before use. Setup coordinates
 * missing default solutions while preserving configured peers. Setup/apply
 * restore the configured execution memory space after host access. */
uint32_t hypredrv_ObjectPreconCreate(hypredrv_t *);
uint32_t hypredrv_ObjectPreconSetup(hypredrv_t *);
uint32_t hypredrv_ObjectPreconApply(hypredrv_t *, HYPRE_IJVector, HYPRE_IJVector);
uint32_t hypredrv_ObjectPreconDestroy(hypredrv_t *);
uint32_t hypredrv_ObjectLinearSolverCreate(hypredrv_t *);
uint32_t hypredrv_ObjectLinearSolverSetup(hypredrv_t *);
uint32_t hypredrv_ObjectLinearSolverApply(hypredrv_t *);
uint32_t hypredrv_ObjectLinearSolverDestroy(hypredrv_t *);

#endif /* HYPREDRV_OBJECT_HEADER_ */
