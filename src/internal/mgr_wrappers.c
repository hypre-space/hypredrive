/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "internal/mgr_internal.h"
#include <math.h>
#include <mpi.h>
#include <stddef.h>
/* gcovr: branch-exclusion regions below narrow branch-count noise from YAML
 * helpers and MGR validation/dispatch; single-line exclusions flag allocator
 * and defensive branches that are impractical to fault-inject here. */
#include "_hypre_parcsr_mv.h"
#if HYPRE_CHECK_MIN_VERSION(30100, 5)
#include "_hypre_parcsr_ls.h"
#endif
#include "_hypre_utilities.h" // for hypre_Solver
#include "internal/compatibility.h"
#include "internal/error.h"
#include "internal/gen_macros.h"
#include "internal/krylov.h"
#include "internal/stats.h"
#include "logging.h"

typedef HYPRE_Int (*MGRHyprePtrToDestroyFcn)(HYPRE_Solver);

typedef struct MGRFRelaxWrapper_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);
   HYPRE_Int    is_setup; /* offset 24: mirrors hypre_Solver layout */
   HYPRE_Solver inner_mgr;
   MGR_args    *nested_args; /* borrowed; owned by parent f_relaxation */
   IntArray    *owned_dofmap;
   struct MGRFRelaxWrapper_struct *next_live;
} MGRFRelaxWrapper;

enum
{
   MGR_HYPRE_SOLVER_IS_SETUP_OFFSET = sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(MGRHyprePtrToDestroyFcn),
};

typedef char MGRNestedKrylovLayoutCheck
   [(offsetof(NestedKrylov_args, is_setup) == MGR_HYPRE_SOLVER_IS_SETUP_OFFSET) ? 1 : -1];

#if HYPRE_CHECK_MIN_VERSION(30100, 0)
/* CPU-only congruence wrapper for a managed AMG F-solver.  It solves
 *
 *    (D^{-1/2} A_FF D^{-1/2}) y = D^{-1/2} r,
 *    correction = D^{-1/2} y,
 *
 * without changing MGR's A_FF or the outer system/residual norm.  This is an
 * opt-in diagnostic for F blocks whose large diagonal range otherwise
 * destabilizes AMG interpolation. */
typedef struct MGRFRelaxEquilWrapper_struct
{
   HYPRE_Int (*setup)(void *, void *, void *, void *);
   HYPRE_Int (*solve)(void *, void *, void *, void *);
   HYPRE_Int (*destroy)(void *);
   HYPRE_Int           is_setup;
   HYPRE_Solver        inner;
   MPI_Comm            comm;
   hypre_ParCSRMatrix *scaled_A;
   hypre_ParVector    *scale;
   hypre_ParVector    *inverse_scale;
   hypre_ParVector    *scaled_rhs;
   hypre_ParVector    *scaled_solution;
} MGRFRelaxEquilWrapper;

typedef char MGRFRelaxEquilLayoutCheck[(offsetof(MGRFRelaxEquilWrapper, is_setup) ==
                                        MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                          ? 1
                                          : -1];
#endif

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
typedef struct MGRSchwarzWrapper_struct
{
   /* Keep the first fields layout-compatible with hypre_Solver as validated
    * against the hypre 3.1 development stream used by the Schwarz branch. */
   HYPRE_PtrToSolverFcn    setup;
   HYPRE_PtrToSolverFcn    solve;
   MGRHyprePtrToDestroyFcn destroy;
   HYPRE_Int               is_setup;
   HYPRE_Solver            inner;
} MGRSchwarzWrapper;

typedef char MGRSchwarzWrapperLayoutCheck
   [(offsetof(MGRSchwarzWrapper, is_setup) == MGR_HYPRE_SOLVER_IS_SETUP_OFFSET) ? 1 : -1];
#endif

static MGRFRelaxWrapper *g_mgr_frelax_wrapper_live_head = NULL;

#if HYPRE_CHECK_MIN_VERSION(30100, 0)
static void
MGRFRelaxEquilDestroyScaledData(MGRFRelaxEquilWrapper *wrapper)
{
   if (!wrapper)
   {
      return;
   }

   hypre_ParCSRMatrixDestroy(wrapper->scaled_A);
   hypre_ParVectorDestroy(wrapper->scale);
   hypre_ParVectorDestroy(wrapper->inverse_scale);
   hypre_ParVectorDestroy(wrapper->scaled_rhs);
   hypre_ParVectorDestroy(wrapper->scaled_solution);
   wrapper->scaled_A        = NULL;
   wrapper->scale           = NULL;
   wrapper->inverse_scale   = NULL;
   wrapper->scaled_rhs      = NULL;
   wrapper->scaled_solution = NULL;
}

static hypre_ParVector *
MGRFRelaxEquilCreateVector(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *vector =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), hypre_ParCSRMatrixGlobalNumRows(A),
                            hypre_ParCSRMatrixRowStarts(A));
   if (vector)
   {
      hypre_ParVectorInitialize_v2(vector, HYPRE_MEMORY_HOST);
      hypre_Vector *local = hypre_ParVectorLocalVector(vector);
      if (!local || (hypre_VectorSize(local) > 0 && !hypre_VectorData(local)))
      {
         hypre_ParVectorDestroy(vector);
         vector = NULL;
      }
   }
   return vector;
}

static MPI_Comm
MGRFRelaxEquilComm(const MGRFRelaxEquilWrapper *wrapper)
{
   return wrapper ? wrapper->comm : MPI_COMM_NULL;
}

/* Scaling workspace built for one F-block setup: adopted by the wrapper when
 * setup succeeds, destroyed by the caller otherwise. */
typedef struct
{
   hypre_ParCSRMatrix *scaled_A;
   hypre_ParVector    *scale;
   hypre_ParVector    *inverse_scale;
   hypre_ParVector    *scaled_rhs;
   hypre_ParVector    *scaled_solution;
} MGRFRelaxEquilWork;

/* Collective agreement over the F-block communicator: every rank must agree
 * before the setup continues, so a local failure never strands its peers. */
static int
MGRFRelaxEquilAllRanksOk(MPI_Comm comm, int local_ok)
{
   int global_ok = 1;

   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);

   return global_ok;
}

static void
MGRFRelaxEquilWorkDestroy(MGRFRelaxEquilWork *work)
{
   hypre_ParCSRMatrixDestroy(work->scaled_A);
   hypre_ParVectorDestroy(work->scale);
   hypre_ParVectorDestroy(work->inverse_scale);
   hypre_ParVectorDestroy(work->scaled_rhs);
   hypre_ParVectorDestroy(work->scaled_solution);
}

/* Every work vector must expose an n-sized local buffer before it is written. */
static int
MGRFRelaxEquilWorkVectorsMatch(const MGRFRelaxEquilWork *work, HYPRE_Int n)
{
   hypre_Vector *scale_local    = hypre_ParVectorLocalVector(work->scale);
   hypre_Vector *inverse_local  = hypre_ParVectorLocalVector(work->inverse_scale);
   hypre_Vector *rhs_local      = hypre_ParVectorLocalVector(work->scaled_rhs);
   hypre_Vector *solution_local = hypre_ParVectorLocalVector(work->scaled_solution);

   return (scale_local && inverse_local && rhs_local && solution_local &&
           hypre_VectorSize(scale_local) == n && hypre_VectorSize(inverse_local) == n &&
           hypre_VectorSize(rhs_local) == n && hypre_VectorSize(solution_local) == n &&
           (n == 0 || (hypre_VectorData(scale_local) && hypre_VectorData(inverse_local) &&
                       hypre_VectorData(rhs_local) && hypre_VectorData(solution_local))));
}

/* Allocates the scaled operator clone and the four work vectors. Returns
 * nonzero only when every allocation succeeded with the expected local shape. */
static int
MGRFRelaxEquilWorkCreate(hypre_ParCSRMatrix *A, HYPRE_Int n, MGRFRelaxEquilWork *work)
{
   work->scale           = MGRFRelaxEquilCreateVector(A);
   work->inverse_scale   = MGRFRelaxEquilCreateVector(A);
   work->scaled_rhs      = MGRFRelaxEquilCreateVector(A);
   work->scaled_solution = MGRFRelaxEquilCreateVector(A);
   work->scaled_A        = hypre_ParCSRMatrixClone(A, 1);

   if (!work->scale || !work->inverse_scale || !work->scaled_rhs ||
       !work->scaled_solution || !work->scaled_A)
   {
      return 0;
   }

   return MGRFRelaxEquilWorkVectorsMatch(work, n);
}

/* Fills the scaling and inverse-scaling vectors from A's diagonal. Returns
 * nonzero when the partitions are square-aligned and every diagonal entry is
 * finite and positive; rows that fail get a unit factor so the collective
 * abort path below still operates on well-defined data. */
static int
MGRFRelaxEquilComputeScaling(hypre_ParCSRMatrix *A, HYPRE_Int n, const HYPRE_Int *A_i,
                             const HYPRE_Int *A_j, const HYPRE_Complex *A_data,
                             HYPRE_Complex *scale_data, HYPRE_Complex *inverse_data)
{
   int local_ok = 1;

   if (hypre_ParCSRMatrixGlobalNumRows(A) != hypre_ParCSRMatrixGlobalNumCols(A) ||
       hypre_ParCSRMatrixFirstRowIndex(A) != hypre_ParCSRMatrixFirstColDiag(A) ||
       hypre_ParCSRMatrixLastRowIndex(A) != hypre_ParCSRMatrixLastColDiag(A))
   {
      local_ok = 0;
   }

   for (HYPRE_Int i = 0; i < n; i++)
   {
      HYPRE_Real diagonal = 0.0;
      HYPRE_Int  found    = 0;

      for (HYPRE_Int jj = A_i[i]; jj < A_i[i + 1]; jj++)
      {
         if (A_j[jj] == i)
         {
            diagonal = hypre_creal(A_data[jj]);
            found    = (hypre_cimag(A_data[jj]) == 0.0);
            break;
         }
      }

      if (!found || !hypredrv_DoubleIsFinite(diagonal) || diagonal <= HYPRE_REAL_MIN)
      {
         local_ok        = 0;
         inverse_data[i] = 1.0;
         scale_data[i]   = 1.0;
         continue;
      }

      inverse_data[i] = hypre_sqrt(diagonal);
      scale_data[i]   = 1.0 / inverse_data[i];
   }

   return local_ok;
}

/* Hands the freshly built workspace to the wrapper, releasing what it held. */
static void
MGRFRelaxEquilAdoptWork(MGRFRelaxEquilWrapper *wrapper, const MGRFRelaxEquilWork *work)
{
   MGRFRelaxEquilDestroyScaledData(wrapper);
   wrapper->scaled_A        = work->scaled_A;
   wrapper->scale           = work->scale;
   wrapper->inverse_scale   = work->inverse_scale;
   wrapper->scaled_rhs      = work->scaled_rhs;
   wrapper->scaled_solution = work->scaled_solution;
}

/* Validates the setup arguments and pins the wrapper's communicator to A's.
 * hypre builds A_FF over the MGR solver's ranks, which is the scope every
 * collective below must use. A NULL A leaves that scope unknowable, so fail
 * locally rather than guess a communicator peers may not share. */
static int
MGRFRelaxEquilSetupPreconditionsOk(MGRFRelaxEquilWrapper *wrapper, hypre_ParCSRMatrix *A,
                                   hypre_ParVector *b, hypre_ParVector *x)
{
   if (!wrapper || !A || !wrapper->inner || !b || !x)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block scaling requires a valid operator, "
                        "right-hand side, solution vector, and inner solver");
      return 0;
   }

   wrapper->is_setup = 0;
   wrapper->comm     = hypre_ParCSRMatrixComm(A);

   return 1;
}

static HYPRE_Int
MGRFRelaxEquilWrapperSetup(void *wrapper_v, void *A_v, void *b_v, void *x_v)
{
   MGRFRelaxEquilWrapper *wrapper  = (MGRFRelaxEquilWrapper *)wrapper_v;
   hypre_ParCSRMatrix    *A        = (hypre_ParCSRMatrix *)A_v;
   hypre_ParVector       *b        = (hypre_ParVector *)b_v;
   hypre_ParVector       *x        = (hypre_ParVector *)x_v;
   MGRFRelaxEquilWork     work     = {NULL, NULL, NULL, NULL, NULL};
   HYPRE_Int              ierr     = 0;
   int                    local_ok = 1;

   if (!MGRFRelaxEquilSetupPreconditionsOk(wrapper, A, b, x))
   {
      return HYPRE_GetError();
   }

   MPI_Comm comm = MGRFRelaxEquilComm(wrapper);
   local_ok =
      hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A)) == HYPRE_EXEC_HOST;
   if (!MGRFRelaxEquilAllRanksOk(comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block scaling requires a valid CPU matrix");
      return HYPRE_GetError();
   }

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int        n      = A_diag ? hypre_CSRMatrixNumRows(A_diag) : 0;
   HYPRE_Int       *A_i    = A_diag ? hypre_CSRMatrixI(A_diag) : NULL;
   HYPRE_Int       *A_j    = A_diag ? hypre_CSRMatrixJ(A_diag) : NULL;
   HYPRE_Complex   *A_data = A_diag ? hypre_CSRMatrixData(A_diag) : NULL;

   local_ok = MGRFRelaxEquilWorkCreate(A, n, &work) && A_diag &&
              (n == 0 || (A_i && A_j && A_data));
   if (!MGRFRelaxEquilAllRanksOk(comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_MEMORY,
                        "Failed to allocate symmetric F-block scaling data");
      ierr = HYPRE_GetError();
      goto cleanup;
   }

   local_ok = MGRFRelaxEquilComputeScaling(
      A, n, A_i, A_j, A_data, hypre_VectorData(hypre_ParVectorLocalVector(work.scale)),
      hypre_VectorData(hypre_ParVectorLocalVector(work.inverse_scale)));
   if (!MGRFRelaxEquilAllRanksOk(comm, local_ok))
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "MGR symmetric F-block scaling requires aligned square partitions and "
         "finite positive diagonals");
      ierr = HYPRE_GetError();
      goto cleanup;
   }

   ierr     = hypre_ParCSRMatrixDiagScale(work.scaled_A, work.scale, work.scale);
   local_ok = !ierr;
   if (!MGRFRelaxEquilAllRanksOk(comm, local_ok))
   {
      if (local_ok)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Failed to scale the MGR F-block collectively");
      }
      ierr = ierr ? ierr : HYPRE_GetError();
      goto cleanup;
   }

   {
      hypre_Solver *inner_base = (hypre_Solver *)wrapper->inner;
      ierr = hypre_SolverSetup(inner_base)(wrapper->inner, (HYPRE_Matrix)work.scaled_A,
                                           (HYPRE_Vector)work.scaled_rhs,
                                           (HYPRE_Vector)work.scaled_solution);
   }

   /* On a partial setup failure the child may already reference the new matrix,
    * so the workspace is retained either way to keep child destruction safe.
    * Solve stays disabled because is_setup is only raised on success. */
   MGRFRelaxEquilAdoptWork(wrapper, &work);
   if (ierr)
   {
      return ierr;
   }

   wrapper->is_setup = 1;
   return 0;

cleanup:
   MGRFRelaxEquilWorkDestroy(&work);
   return ierr ? ierr : 1;
}

/* Local data views of the vectors the scaled solve reads and writes. */
typedef struct
{
   HYPRE_Complex *b;
   HYPRE_Complex *x;
   HYPRE_Complex *scale;
   HYPRE_Complex *inverse_scale;
   HYPRE_Complex *scaled_rhs;
   HYPRE_Complex *scaled_solution;
   HYPRE_Int      n;
} MGRFRelaxEquilSolveViews;

/* The wrapper must be fully set up with every scaled-solve operand present. */
static int
MGRFRelaxEquilSolveStateReady(const MGRFRelaxEquilWrapper *wrapper, hypre_ParCSRMatrix *A,
                              hypre_ParVector *b, hypre_ParVector *x)
{
   return (A && wrapper && wrapper->inner && wrapper->is_setup && wrapper->scaled_A &&
           b && x && wrapper->scale && wrapper->inverse_scale && wrapper->scaled_rhs &&
           wrapper->scaled_solution);
}

/* All six local vectors and A's diagonal must agree on the local row count. */
static int
MGRFRelaxEquilLocalSizesMatch(hypre_ParCSRMatrix *A, HYPRE_Int n, hypre_Vector *b_local,
                              hypre_Vector *x_local, hypre_Vector *is_local,
                              hypre_Vector *bs_local, hypre_Vector *ys_local)
{
   return (hypre_VectorSize(b_local) == n && hypre_VectorSize(x_local) == n &&
           hypre_VectorSize(is_local) == n && hypre_VectorSize(bs_local) == n &&
           hypre_VectorSize(ys_local) == n &&
           hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A)) == n);
}

/* A zero-length partition legitimately carries null data pointers. */
static int
MGRFRelaxEquilViewsPopulated(const MGRFRelaxEquilSolveViews *views)
{
   return (views->n == 0 ||
           (views->b && views->x && views->scale && views->inverse_scale &&
            views->scaled_rhs && views->scaled_solution));
}

/* Checks that the wrapper is set up and that every vector involved in the
 * scaled solve shares A's local row count, capturing their data pointers. */
static int
MGRFRelaxEquilSolveViewsOk(MGRFRelaxEquilWrapper *wrapper, hypre_ParCSRMatrix *A,
                           hypre_ParVector *b, hypre_ParVector *x,
                           MGRFRelaxEquilSolveViews *views)
{
   hypre_Vector *b_local = NULL, *x_local = NULL, *s_local = NULL;
   hypre_Vector *is_local = NULL, *bs_local = NULL, *ys_local = NULL;
   HYPRE_Int     n = 0;

   if (!MGRFRelaxEquilSolveStateReady(wrapper, A, b, x))
   {
      return 0;
   }

   b_local  = hypre_ParVectorLocalVector(b);
   x_local  = hypre_ParVectorLocalVector(x);
   s_local  = hypre_ParVectorLocalVector(wrapper->scale);
   is_local = hypre_ParVectorLocalVector(wrapper->inverse_scale);
   bs_local = hypre_ParVectorLocalVector(wrapper->scaled_rhs);
   ys_local = hypre_ParVectorLocalVector(wrapper->scaled_solution);
   if (!b_local || !x_local || !s_local || !is_local || !bs_local || !ys_local)
   {
      return 0;
   }

   n = hypre_VectorSize(s_local);
   if (!MGRFRelaxEquilLocalSizesMatch(A, n, b_local, x_local, is_local, bs_local,
                                      ys_local))
   {
      return 0;
   }

   views->b               = hypre_VectorData(b_local);
   views->x               = hypre_VectorData(x_local);
   views->scale           = hypre_VectorData(s_local);
   views->inverse_scale   = hypre_VectorData(is_local);
   views->scaled_rhs      = hypre_VectorData(bs_local);
   views->scaled_solution = hypre_VectorData(ys_local);
   views->n               = n;

   return MGRFRelaxEquilViewsPopulated(views);
}

static HYPRE_Int
MGRFRelaxEquilWrapperSolve(void *wrapper_v, void *A_v, void *b_v, void *x_v)
{
   MGRFRelaxEquilWrapper   *wrapper = (MGRFRelaxEquilWrapper *)wrapper_v;
   hypre_ParCSRMatrix      *A       = (hypre_ParCSRMatrix *)A_v;
   hypre_ParVector         *b       = (hypre_ParVector *)b_v;
   hypre_ParVector         *x       = (hypre_ParVector *)x_v;
   MGRFRelaxEquilSolveViews v       = {NULL, NULL, NULL, NULL, NULL, NULL, 0};
   HYPRE_Int                ierr    = 0;

   if (MGRFRelaxEquilComm(wrapper) == MPI_COMM_NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block solver has no valid communicator");
      return HYPRE_GetError();
   }

   if (!MGRFRelaxEquilSolveViewsOk(wrapper, A, b, x, &v))
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "MGR symmetric F-block solver state or vector sizes do not match");
      return HYPRE_GetError();
   }

   for (HYPRE_Int i = 0; i < v.n; i++)
   {
      v.scaled_rhs[i]      = v.scale[i] * v.b[i];
      v.scaled_solution[i] = v.inverse_scale[i] * v.x[i];
   }

   {
      hypre_Solver *inner_base = (hypre_Solver *)wrapper->inner;
      ierr                     = hypre_SolverSolve(inner_base)(
         wrapper->inner, (HYPRE_Matrix)wrapper->scaled_A,
         (HYPRE_Vector)wrapper->scaled_rhs, (HYPRE_Vector)wrapper->scaled_solution);
   }
   if (ierr)
   {
      return ierr;
   }

   for (HYPRE_Int i = 0; i < v.n; i++)
   {
      v.x[i] = v.scale[i] * v.scaled_solution[i];
   }

   return 0;
}

HYPRE_Int
hypredrv_MGRFRelaxEquilWrapperDestroy(void *wrapper_v)
{
   MGRFRelaxEquilWrapper *wrapper = (MGRFRelaxEquilWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 0;
   }

   if (wrapper->inner)
   {
      HYPRE_BoomerAMGDestroy(wrapper->inner);
      wrapper->inner = NULL;
   }
   MGRFRelaxEquilDestroyScaledData(wrapper);
   free(wrapper);
   return 0;
}

HYPRE_Solver
hypredrv_MGRFRelaxEquilWrapperCreate(HYPRE_Solver inner)
{
   MGRFRelaxEquilWrapper *wrapper = calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      HYPRE_BoomerAMGDestroy(inner);
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate symmetric F-block scaling wrapper");
      return NULL;
   }

   wrapper->setup   = MGRFRelaxEquilWrapperSetup;
   wrapper->solve   = MGRFRelaxEquilWrapperSolve;
   wrapper->destroy = hypredrv_MGRFRelaxEquilWrapperDestroy;
   wrapper->inner   = inner;
   /* Assigned from the F-block operator in MGRFRelaxEquilWrapperSetup. */
   wrapper->comm = MPI_COMM_NULL;
   return (HYPRE_Solver)wrapper;
}
#endif

/* GCOVR_EXCL_START */
static void
MGRFRelaxWrapperRegister(MGRFRelaxWrapper *wrapper)
{
   if (!wrapper)
   {
      return;
   }

   wrapper->next_live             = g_mgr_frelax_wrapper_live_head;
   g_mgr_frelax_wrapper_live_head = wrapper;
}

static void
MGRFRelaxWrapperUnregister(MGRFRelaxWrapper *wrapper)
{
   MGRFRelaxWrapper **cursor = &g_mgr_frelax_wrapper_live_head;

   while (*cursor)
   {
      if (*cursor == wrapper)
      {
         *cursor            = wrapper->next_live;
         wrapper->next_live = NULL;
         return;
      }
      cursor = &(*cursor)->next_live;
   }
}

static HYPRE_Int
MGRFRelaxWrapperSetup(void *wrapper_v, void *A, void *b, void *x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   if (HYPRE_MGRSetup((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                      (HYPRE_ParVector)b, (HYPRE_ParVector)x))
   {
      return 1;
   }
   wrapper->is_setup = 1;
   return 0;
}

static HYPRE_Int
MGRFRelaxWrapperSolve(void *wrapper_v, void *A, void *b, void *x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner_mgr)
   {
      return 1;
   }
   /* hypre calls the setup callback for level-specific F-solvers during the
    * parent MGR setup. Avoid calling inner MGR setup again from solve because
    * repeated nested re-setup trips a cleanup bug in current hypre. */
   return HYPRE_MGRSolve((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRFRelaxWrapperDestroy(void *wrapper_v)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   HYPRE_Solver      inner;
   int               was_setup;

   if (!wrapper)
   {
      return 0;
   }

   MGRFRelaxWrapperUnregister(wrapper);

   /* hypre marks user-set F-solvers as OWNER_USER and does not destroy them.
    * Own the full nested-MGR teardown here (inner handle + its cached user
    * component solvers), mirroring PreconDestroyMGRSolver ownership rules. */
   inner              = wrapper->inner_mgr;
   was_setup          = wrapper->is_setup;
   wrapper->inner_mgr = NULL;
   if (inner)
   {
      if (wrapper->nested_args && !was_setup)
      {
         hypredrv_MGRDestroyCachedSolvers(wrapper->nested_args, 0);
      }
      HYPRE_MGRDestroy(inner);
      if (wrapper->nested_args && was_setup)
      {
         hypredrv_MGRDestroyCachedSolvers(wrapper->nested_args, 1);
      }
   }

   /* Match PreconDestroyMGRSolver: the owned point-marker buffer outlives
    * HYPRE_MGRDestroy and must be released with the nested args. */
   if (wrapper->nested_args && wrapper->nested_args->point_marker_data)
   {
      free(wrapper->nested_args->point_marker_data);
      wrapper->nested_args->point_marker_data = NULL;
   }

   wrapper->nested_args = NULL;
   hypredrv_IntArrayDestroy(&wrapper->owned_dofmap);
   free(wrapper);
   return 0;
}

void
hypredrv_MGRSetFSolverAtLevel(HYPRE_Solver precon, HYPRE_Solver fsolver, HYPRE_Int level,
                              HYPRE_Int               f_relax_type,
                              HYPRE_PtrToParSolverFcn fine_grid_solver_solve,
                              HYPRE_PtrToParSolverFcn fine_grid_solver_setup)
{
   if (!precon || !fsolver)
   {
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   if (level == 0 && fine_grid_solver_solve && fine_grid_solver_setup)
   {
      (void)f_relax_type;
      HYPRE_MGRSetFSolver(precon, fine_grid_solver_solve, fine_grid_solver_setup,
                          fsolver);
      return;
   }

#if HYPRE_CHECK_MIN_VERSION(21900, 0)
   if (level == 0 && f_relax_type == 32)
   {
      HYPRE_MGRSetFSolver(precon, HYPRE_ILUSolve, HYPRE_ILUSetup, fsolver);
      return;
   }
#endif

   (void)f_relax_type;
   (void)fine_grid_solver_solve;
   (void)fine_grid_solver_setup;
   HYPRE_MGRSetFSolverAtLevel(precon, fsolver, level);
#else
   (void)fine_grid_solver_solve;
   (void)fine_grid_solver_setup;
   (void)level;
   (void)f_relax_type;
#endif
   (void)fsolver;
   (void)precon;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxWrapperCreate(HYPRE_Solver inner_mgr, MGR_args *nested_args,
                             IntArray *owned_dofmap)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate nested MGR F-relaxation wrapper");
      return NULL;
   }

   wrapper->setup        = MGRFRelaxWrapperSetup;
   wrapper->solve        = MGRFRelaxWrapperSolve;
   wrapper->destroy      = MGRFRelaxWrapperDestroy;
   wrapper->inner_mgr    = inner_mgr;
   wrapper->nested_args  = nested_args;
   wrapper->owned_dofmap = owned_dofmap;
   MGRFRelaxWrapperRegister(wrapper);
   return (HYPRE_Solver)wrapper;
}

int
hypredrv_MGRNestedFRelaxWrapperIsLive(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *cursor = g_mgr_frelax_wrapper_live_head;

   while (cursor)
   {
      if ((HYPRE_Solver)cursor == wrapper_solver)
      {
         return 1;
      }
      cursor = cursor->next_live;
   }

   return 0;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxWrapperGetInner(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_solver;
   return wrapper ? wrapper->inner_mgr : NULL;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxWrapperDetachInner(HYPRE_Solver wrapper_solver)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_solver;
   HYPRE_Solver      inner   = NULL;

   if (!wrapper)
   {
      return NULL;
   }

   inner              = wrapper->inner_mgr;
   wrapper->inner_mgr = NULL;
   return inner;
}

void
hypredrv_MGRNestedFRelaxWrapperFree(HYPRE_Solver *wrapper_ptr)
{
   if (!wrapper_ptr || !*wrapper_ptr)
   {
      return;
   }
   MGRFRelaxWrapperDestroy((void *)(*wrapper_ptr));
   *wrapper_ptr = NULL;
}

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static HYPRE_Int
MGRSchwarzWrapperSetup(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                       HYPRE_Vector x)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner)
   {
      return 1;
   }

   if (HYPRE_SchwarzSetup((HYPRE_Solver)wrapper->inner, (HYPRE_ParCSRMatrix)A,
                          (HYPRE_ParVector)b, (HYPRE_ParVector)x))
   {
      return 1;
   }

   wrapper->is_setup = 1;
   return 0;
}

static HYPRE_Int
MGRSchwarzWrapperSolve(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                       HYPRE_Vector x)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper || !wrapper->inner)
   {
      return 1;
   }

   if (!wrapper->is_setup && MGRSchwarzWrapperSetup(wrapper_v, A, b, x))
   {
      return 1;
   }

   return HYPRE_SchwarzSolve((HYPRE_Solver)wrapper->inner, (HYPRE_ParCSRMatrix)A,
                             (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

HYPRE_Int
hypredrv_MGRSchwarzWrapperDestroy(HYPRE_Solver wrapper_v)
{
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 0;
   }

   if (wrapper->inner)
   {
      HYPRE_SchwarzDestroy(wrapper->inner);
      wrapper->inner = NULL;
   }
   free(wrapper);
   return 0;
}

HYPRE_Int
hypredrv_MGRSchwarzWrapperParSetup(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSetup(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

HYPRE_Int
hypredrv_MGRSchwarzWrapperParSolve(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSolve(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

HYPRE_Solver
hypredrv_MGRSchwarzWrapperCreate(const Schwarz_args *args)
{
   HYPRE_Solver       inner   = NULL;
   MGRSchwarzWrapper *wrapper = (MGRSchwarzWrapper *)calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_ErrorCodeSet(ERROR_ALLOCATION);
      hypredrv_ErrorMsgAdd("Failed to allocate MGR Schwarz solver wrapper");
      return NULL;
   }

   hypredrv_SchwarzCreate(args, &inner);
   if (hypredrv_ErrorCodeActive() || !inner)
   {
      free(wrapper);
      return NULL;
   }

   wrapper->setup   = MGRSchwarzWrapperSetup;
   wrapper->solve   = MGRSchwarzWrapperSolve;
   wrapper->destroy = hypredrv_MGRSchwarzWrapperDestroy;
   wrapper->inner   = inner;
   return (HYPRE_Solver)wrapper;
}
#endif
/* GCOVR_EXCL_STOP */


HYPRE_Int
hypredrv_MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}


HYPRE_Int
hypredrv_MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                      HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}
