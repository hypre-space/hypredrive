/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include "internal/mgr_internal.h"
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
#include "internal/solver.h"
#include "internal/stats.h"
#include "logging.h"

/* Use the native base where available and retain the readiness slot used by
 * newer hypre. Tagged 3.1 exposes only the three callbacks in hypre_Solver. */
typedef union
{
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
   hypre_Solver native;
#endif
   hypredrv_SolverBase fields;
} MGRSolverBase;

typedef struct MGRFRelaxWrapper_struct
{
   MGRSolverBase base;
   HYPRE_Solver  inner_mgr;
   MGR_args     *nested_args; /* borrowed; owned by parent f_relaxation */
   struct MGRFRelaxWrapper_struct *next_live;
} MGRFRelaxWrapper;

enum
{
   MGR_HYPRE_SOLVER_IS_SETUP_OFFSET = sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(HYPRE_PtrToSolverFcn) +
                                      sizeof(((hypredrv_SolverBase *)0)->destroy),
};

typedef char MGRNestedKrylovLayoutCheck[(offsetof(NestedKrylov_args, base.is_setup) ==
                                         MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                           ? 1
                                           : -1];

typedef char
   MGRFRelaxWrapperLayoutCheck[(offsetof(MGRFRelaxWrapper, base.fields.is_setup) ==
                                MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                  ? 1
                                  : -1];

#if HYPRE_CHECK_MIN_VERSION(30100, 0)
/* CPU-only congruence wrapper for a managed AMG F-solver.  It solves
 *
 *    (D^{-1/2} A_FF D^{-1/2}) y = D^{-1/2} r,
 *    correction = D^{-1/2} y,
 *
 * without changing MGR's A_FF or the outer system/residual norm.  This is an
 * opt-in diagnostic for F blocks whose large diagonal range otherwise
 * destabilizes AMG interpolation. */
typedef struct MGRFRelaxEquilWork_struct
{
   hypre_ParCSRMatrix               *scaled_A;
   hypre_ParVector                  *scale, *inverse_scale, *scaled_rhs, *scaled_solution;
   HYPRE_Int                         n;
   struct MGRFRelaxEquilWork_struct *next;
} MGRFRelaxEquilWork;

typedef struct MGRFRelaxEquilWrapper_struct
{
   MGRSolverBase        base;
   HYPRE_Solver         inner;
   MPI_Comm             comm;
   HYPRE_PtrToSolverFcn child_setup, child_solve;
   /* A failed child setup can still borrow either the old or new inputs. */
   MGRFRelaxEquilWork *work;
} MGRFRelaxEquilWrapper;

typedef char
   MGRFRelaxEquilLayoutCheck[(offsetof(MGRFRelaxEquilWrapper, base.fields.is_setup) ==
                              MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                ? 1
                                : -1];
#endif

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
typedef struct MGRSchwarzWrapper_struct
{
   MGRSolverBase base;
   HYPRE_Solver  inner;
} MGRSchwarzWrapper;

typedef char
   MGRSchwarzWrapperLayoutCheck[(offsetof(MGRSchwarzWrapper, base.fields.is_setup) ==
                                 MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                   ? 1
                                   : -1];
#endif

static MGRFRelaxWrapper *g_mgr_frelax_wrapper_live_head = NULL;

#if HYPRE_CHECK_MIN_VERSION(30100, 0)
static void
MGRFRelaxEquilWorkDestroy(MGRFRelaxEquilWork *work)
{
   while (work)
   {
      MGRFRelaxEquilWork *next = work->next;
      hypre_ParCSRMatrixDestroy(work->scaled_A);
      hypre_ParVectorDestroy(work->scale);
      hypre_ParVectorDestroy(work->inverse_scale);
      hypre_ParVectorDestroy(work->scaled_rhs);
      hypre_ParVectorDestroy(work->scaled_solution);
      free(work);
      work = next;
   }
}

static hypre_ParVector *
MGRFRelaxEquilCreateVector(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *vector =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), hypre_ParCSRMatrixGlobalNumRows(A),
                            hypre_ParCSRMatrixRowStarts(A));
   if (vector) hypre_ParVectorInitialize_v2(vector, HYPRE_MEMORY_HOST);
   return vector;
}

static int
MGRFRelaxEquilAllRanksOk(MPI_Comm comm, int local_ok)
{
   int global_ok = 0;
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
   return global_ok;
}

/* This callback builds its own vectors. Admit only the operator it reads,
 * before any rank allocates or enters the child's collective setup. */
static int
MGRFRelaxEquilOperatorOk(MPI_Comm comm, hypre_ParCSRMatrix *A)
{
   hypre_CSRMatrix *diag       = A ? hypre_ParCSRMatrixDiag(A) : NULL;
   HYPRE_Int        n          = diag ? hypre_CSRMatrixNumRows(diag) : 0;
   int              comparison = MPI_UNEQUAL;
   if (A && hypre_ParCSRMatrixComm(A) != MPI_COMM_NULL)
      MPI_Comm_compare(comm, hypre_ParCSRMatrixComm(A), &comparison);
   return !HYPRE_GetError() && A && diag && n >= 0 &&
          (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
          hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A)) == HYPRE_EXEC_HOST &&
          hypre_ParCSRMatrixGlobalNumRows(A) == hypre_ParCSRMatrixGlobalNumCols(A) &&
          hypre_ParCSRMatrixFirstRowIndex(A) == hypre_ParCSRMatrixFirstColDiag(A) &&
          hypre_ParCSRMatrixLastRowIndex(A) == hypre_ParCSRMatrixLastColDiag(A) &&
          n ==
             hypre_ParCSRMatrixLastRowIndex(A) - hypre_ParCSRMatrixFirstRowIndex(A) + 1 &&
          (n == 0 || (hypre_CSRMatrixI(diag) && hypre_CSRMatrixJ(diag) &&
                      hypre_CSRMatrixData(diag)));
}

static MGRFRelaxEquilWork *
MGRFRelaxEquilWorkCreate(hypre_ParCSRMatrix *A, HYPRE_Int n)
{
   MGRFRelaxEquilWork *work = calloc(1, sizeof(*work));
   if (work)
   {
      work->n               = n;
      work->scale           = MGRFRelaxEquilCreateVector(A);
      work->inverse_scale   = MGRFRelaxEquilCreateVector(A);
      work->scaled_rhs      = MGRFRelaxEquilCreateVector(A);
      work->scaled_solution = MGRFRelaxEquilCreateVector(A);
      work->scaled_A        = hypre_ParCSRMatrixClone(A, 1);
   }
   return work;
}

static int
MGRFRelaxEquilWorkComplete(const MGRFRelaxEquilWork *work)
{
   return work && work->scale && work->inverse_scale && work->scaled_rhs &&
          work->scaled_solution && work->scaled_A && !HYPRE_GetError();
}

/* Fill scale = D^{-1/2} and inverse_scale = D^{1/2} from the local diagonal
 * block. Returns 0 if any local diagonal is missing, complex, non-finite or
 * not positive. */
static int
MGRFRelaxEquilComputeScaling(MGRFRelaxEquilWork *work, const hypre_CSRMatrix *diag)
{
   const HYPRE_Int     *rows  = hypre_CSRMatrixI(diag);
   const HYPRE_Int     *cols  = hypre_CSRMatrixJ(diag);
   const HYPRE_Complex *data  = hypre_CSRMatrixData(diag);
   HYPRE_Complex       *scale = hypre_VectorData(hypre_ParVectorLocalVector(work->scale));
   HYPRE_Complex       *inverse =
      hypre_VectorData(hypre_ParVectorLocalVector(work->inverse_scale));
   int local_ok = 1;
   for (HYPRE_Int i = 0; i < work->n; i++)
   {
      HYPRE_Real diagonal = 0.0;
      int        found    = 0;
      for (HYPRE_Int j = rows[i]; j < rows[i + 1]; j++)
      {
         if (cols[j] == i)
         {
            diagonal = hypre_creal(data[j]);
            found    = hypre_cimag(data[j]) == 0.0;
            break;
         }
      }
      if (!found || !hypredrv_DoubleIsFinite(diagonal) || diagonal <= HYPRE_REAL_MIN)
      {
         local_ok = 0;
         continue;
      }
      inverse[i] = hypre_sqrt(diagonal);
      scale[i]   = 1.0 / inverse[i];
   }
   return local_ok;
}

static HYPRE_Int
MGRFRelaxEquilWrapperSetup(HYPRE_Solver wrapper_v, HYPRE_Matrix A_v, HYPRE_Vector b_v,
                           HYPRE_Vector x_v)
{
   MGRFRelaxEquilWrapper *wrapper = (MGRFRelaxEquilWrapper *)wrapper_v;
   hypre_ParCSRMatrix    *A       = (hypre_ParCSRMatrix *)A_v;
   (void)b_v;
   (void)x_v;
   if (!wrapper)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MGR symmetric F-block solver is NULL");
      return HYPRE_GetError();
   }
   wrapper->base.fields.is_setup = 0;

   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm,
                                 MGRFRelaxEquilOperatorOk(wrapper->comm, A)))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block scaling requires a valid CPU operator "
                        "with aligned square partitions on its creation communicator");
      return HYPRE_GetError();
   }

   /* Every rank admitted A, so its diagonal block exists here. */
   hypre_CSRMatrix    *diag = hypre_ParCSRMatrixDiag(A);
   MGRFRelaxEquilWork *work = MGRFRelaxEquilWorkCreate(A, hypre_CSRMatrixNumRows(diag));
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, MGRFRelaxEquilWorkComplete(work)))
   {
      hypre_error_w_msg(HYPRE_ERROR_MEMORY,
                        "Failed to allocate symmetric F-block scaling data");
      MGRFRelaxEquilWorkDestroy(work);
      return HYPRE_GetError();
   }

   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, MGRFRelaxEquilComputeScaling(work, diag)))
   {
      hypre_error_w_msg(
         HYPRE_ERROR_GENERIC,
         "MGR symmetric F-block scaling requires finite positive diagonals");
      MGRFRelaxEquilWorkDestroy(work);
      return HYPRE_GetError();
   }
   HYPRE_Int code = hypre_ParCSRMatrixDiagScale(work->scaled_A, work->scale, work->scale);
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, !code))
   {
      hypre_error_w_msg(code ? code : HYPRE_ERROR_GENERIC,
                        "Failed to scale the MGR F-block collectively");
      MGRFRelaxEquilWorkDestroy(work);
      return HYPRE_GetError();
   }

   /* Keep every input the child may borrow if setup fails partway through.
    * Successful setup retires all previous work; destruction retires the child
    * before releasing work from failed attempts. */
   work->next    = wrapper->work;
   wrapper->work = work;
   code          = wrapper->child_setup(wrapper->inner, (HYPRE_Matrix)work->scaled_A,
                                        (HYPRE_Vector)work->scaled_rhs,
                                        (HYPRE_Vector)work->scaled_solution);
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, !code && !HYPRE_GetError()))
   {
      hypre_error_w_msg(code ? code : HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block child setup failed collectively");
      return HYPRE_GetError();
   }
   MGRFRelaxEquilWorkDestroy(work->next);
   work->next                    = NULL;
   wrapper->base.fields.is_setup = 1;
   return 0;
}

/* Checks one solve vector against the layout and memory space of the work
 * vectors built by the last successful setup. */
static int
MGRFRelaxEquilVectorOk(hypre_ParVector *v, const MGRFRelaxEquilWork *work)
{
   hypre_Vector *local = v ? hypre_ParVectorLocalVector(v) : NULL;
   return local && hypre_ParVectorNumVectors(v) == 1 &&
          hypre_VectorSize(local) == work->n &&
          hypre_ParVectorGlobalSize(v) == hypre_ParVectorGlobalSize(work->scale) &&
          hypre_ParVectorFirstIndex(v) == hypre_ParVectorFirstIndex(work->scale) &&
          hypre_GetExecPolicy1(hypre_VectorMemoryLocation(local)) == HYPRE_EXEC_HOST &&
          (work->n == 0 || hypre_VectorData(local));
}

static HYPRE_Int
MGRFRelaxEquilWrapperSolve(HYPRE_Solver wrapper_v, HYPRE_Matrix A_v, HYPRE_Vector b_v,
                           HYPRE_Vector x_v)
{
   MGRFRelaxEquilWrapper *wrapper = (MGRFRelaxEquilWrapper *)wrapper_v;
   hypre_ParVector       *b = (hypre_ParVector *)b_v, *x = (hypre_ParVector *)x_v;
   (void)A_v; /* Setup owns the scaled operator used by the child. */
   if (!wrapper)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MGR symmetric F-block solver is NULL");
      return HYPRE_GetError();
   }
   MGRFRelaxEquilWork *work     = wrapper->work;
   int                 local_ok = !HYPRE_GetError() && wrapper->base.fields.is_setup &&
                  MGRFRelaxEquilVectorOk(b, work) && MGRFRelaxEquilVectorOk(x, work);
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block solve requires successful setup and "
                        "compatible CPU vectors on every rank");
      return HYPRE_GetError();
   }

   HYPRE_Complex *b_data = hypre_VectorData(hypre_ParVectorLocalVector(b));
   HYPRE_Complex *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   HYPRE_Complex *scale  = hypre_VectorData(hypre_ParVectorLocalVector(work->scale));
   HYPRE_Complex *inverse =
      hypre_VectorData(hypre_ParVectorLocalVector(work->inverse_scale));
   HYPRE_Complex *rhs = hypre_VectorData(hypre_ParVectorLocalVector(work->scaled_rhs));
   HYPRE_Complex *solution =
      hypre_VectorData(hypre_ParVectorLocalVector(work->scaled_solution));
   for (HYPRE_Int i = 0; i < work->n; i++)
   {
      rhs[i]      = scale[i] * b_data[i];
      solution[i] = inverse[i] * x_data[i];
   }
   HYPRE_Int code = wrapper->child_solve(wrapper->inner, (HYPRE_Matrix)work->scaled_A,
                                         (HYPRE_Vector)work->scaled_rhs,
                                         (HYPRE_Vector)work->scaled_solution);
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, !code && !HYPRE_GetError()))
   {
      hypre_error_w_msg(code ? code : HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block child solve failed collectively");
      return HYPRE_GetError();
   }
   /* Native work-vector migration can replace its local storage. */
   solution = hypre_VectorData(hypre_ParVectorLocalVector(work->scaled_solution));
   for (HYPRE_Int i = 0; i < work->n; i++) x_data[i] = scale[i] * solution[i];
   return 0;
}

HYPRE_Int
hypredrv_MGRFRelaxEquilWrapperDestroy(HYPRE_Solver wrapper_v)
{
   MGRFRelaxEquilWrapper *wrapper = (MGRFRelaxEquilWrapper *)wrapper_v;
   if (!wrapper) return 0;
   hypredrv_AMGDestroy(wrapper->inner);
   MGRFRelaxEquilWorkDestroy(wrapper->work);
   free(wrapper);
   return 0;
}

HYPRE_Solver
hypredrv_MGRFRelaxEquilWrapperCreate(MPI_Comm comm, HYPRE_Solver inner)
{
   hypre_Solver *child = (hypre_Solver *)inner;
   if (comm == MPI_COMM_NULL || !child || !hypre_SolverSetup(child) ||
       !hypre_SolverSolve(child))
   {
      if (inner) hypredrv_AMGDestroy(inner);
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Symmetric F-block scaling requires a communicator and initialized AMG child");
      return NULL;
   }
   MGRFRelaxEquilWrapper *wrapper = calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_AMGDestroy(inner);
      hypredrv_AllocationFailure("Failed to allocate symmetric F-block scaling wrapper");
      return NULL;
   }
   wrapper->base.fields.setup   = MGRFRelaxEquilWrapperSetup;
   wrapper->base.fields.solve   = MGRFRelaxEquilWrapperSolve;
   wrapper->base.fields.destroy = hypredrv_MGRFRelaxEquilWrapperDestroy;
   wrapper->inner               = inner;
   wrapper->comm                = comm;
   wrapper->child_setup         = hypre_SolverSetup(child);
   wrapper->child_solve         = hypre_SolverSolve(child);
   return (HYPRE_Solver)wrapper;
}
#endif

static HYPRE_Int
MGRFRelaxWrapperSetup(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                      HYPRE_Vector x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 1;
   }
   if (!hypredrv_MGRValidatePointMarkerRows(wrapper->nested_args, (HYPRE_ParCSRMatrix)A))
   {
      return 1;
   }
   if (hypredrv_MGRSetup((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x))
   {
      return 1;
   }
   wrapper->base.fields.is_setup = 1;
   return 0;
}

static HYPRE_Int
MGRFRelaxWrapperSolve(HYPRE_Solver wrapper_v, HYPRE_Matrix A, HYPRE_Vector b,
                      HYPRE_Vector x)
{
   MGRFRelaxWrapper *wrapper = (MGRFRelaxWrapper *)wrapper_v;
   if (!wrapper)
   {
      return 1;
   }
   /* hypre calls the setup callback for level-specific F-solvers during the
    * parent MGR setup. Avoid calling inner MGR setup again from solve because
    * repeated nested re-setup trips a cleanup bug in current hypre. */
   return hypredrv_MGRSolve(wrapper->nested_args, wrapper->inner_mgr,
                            (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                            (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRFRelaxWrapperDestroy(HYPRE_Solver wrapper_v)
{
   /* Native teardown can precede release of the parent's cached handle. Find
    * ownership before dereferencing the candidate, then retire it exactly once. */
   MGRFRelaxWrapper **cursor = &g_mgr_frelax_wrapper_live_head;
   while (*cursor && (HYPRE_Solver)*cursor != wrapper_v) cursor = &(*cursor)->next_live;
   if (!*cursor) return 0;
   MGRFRelaxWrapper *wrapper = *cursor;
   *cursor                   = wrapper->next_live;

   /* User-set F-solvers are ours to release, with the same ownership policy
    * as an outer MGR, including partial-create failures. */
   hypredrv_MGRDestroy(wrapper->nested_args, &wrapper->inner_mgr,
                       wrapper->base.fields.is_setup);

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

   if (level == 0 && f_relax_type == 32)
   {
      HYPRE_MGRSetFSolver(precon, HYPRE_ILUSolve, HYPRE_ILUSetup, fsolver);
      return;
   }

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

/* Rank-local admission of the parent level that hosts the nested MGR. Sets
 * *nested_args as soon as the level is known to be valid. */
static int
MGRNestedFRelaxAdmit(const MGR_args *parent, int level, MGR_args **nested_args)
{
   if (hypredrv_ErrorCodeActive() || HYPRE_GetError())
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Cannot create nested MGR with an outstanding error");
      return 0;
   }
   if (level < 0 || level >= hypredrv_MGRNumFineLevels(parent))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid parent level for nested MGR F-relaxation");
      return 0;
   }
   *nested_args = parent->level[level].f_relaxation.mgr;
   if (!*nested_args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
      return 0;
   }
   const IntArray      *parent_dofmap = parent->dofmap;
   const StackIntArray *f_dofs        = &parent->level[level].f_dofs;
   if (!parent_dofmap || (parent_dofmap->size && !parent_dofmap->data) || !f_dofs->size ||
       f_dofs->size > MAX_STACK_ARRAY_LENGTH)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires a parent dofmap and non-empty f_dofs");
      return 0;
   }
   return 1;
}

/* Mark the parent's f_dofs labels in keep_label and count the parent rows
 * that carry one of them. */
static int
MGRNestedFRelaxSelectLabels(const IntArray *parent_dofmap, const StackIntArray *f_dofs,
                            size_t label_space, const HYPRE_Int *parent_present,
                            HYPRE_Int *keep_label, size_t *nested_size)
{
   for (size_t i = 0; i < f_dofs->size; i++)
   {
      int label = f_dofs->data[i];
      if (label < 0 || (size_t)label >= label_space)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid parent MGR f_dofs label %d for nested MGR (valid range: [0,%d])",
            label, (int)label_space - 1);
         return 0;
      }
      if (!parent_present[label] || keep_label[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Parent MGR f_dofs label %d is absent or repeated", label);
         return 0;
      }
      keep_label[label] = 1;
   }

   /* Preserve the parent's labels and row order so the child's f_dofs refer to
    * the same fields. The admitted label mask bounds these row lookups. */
   for (size_t i = 0; i < parent_dofmap->size; i++)
   {
      int label = parent_dofmap->data[i];
      if (label < 0 || (size_t)label >= label_space || !parent_present[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Invalid parent row label %d for nested MGR", label);
         return 0;
      }
      if (keep_label[label]) (*nested_size)++;
   }
   return 1;
}

/* Build the child's dofmap from the selected parent rows. *dofmap_ptr is set
 * as soon as it is created, so the caller releases it on every failure. */
static int
MGRNestedFRelaxBuildDofmap(const IntArray *parent_dofmap, const StackIntArray *f_dofs,
                           const HYPRE_Int *keep_label, size_t label_space,
                           size_t nested_size, IntArray **dofmap_ptr)
{
   IntArray *dofmap = hypredrv_IntArrayCreate(nested_size);
   *dofmap_ptr      = dofmap;
   if (!dofmap || (nested_size && !dofmap->data))
   {
      hypredrv_AllocationFailure("Failed to allocate nested MGR projected dofmap");
      return 0;
   }
   for (size_t i = 0, j = 0; i < parent_dofmap->size; i++)
   {
      int label = parent_dofmap->data[i];
      if (keep_label[label]) dofmap->data[j++] = label;
   }
   dofmap->unique_size = dofmap->g_unique_size = f_dofs->size;
   dofmap->unique_data                         = malloc(f_dofs->size * sizeof(int));
   dofmap->g_unique_data                       = malloc(f_dofs->size * sizeof(int));
   if (!dofmap->unique_data || !dofmap->g_unique_data)
   {
      hypredrv_AllocationFailure("Failed to allocate nested MGR dof label arrays");
      return 0;
   }
   for (size_t i = 0, j = 0; i < label_space; i++)
   {
      if (keep_label[i])
      {
         dofmap->unique_data[j] = dofmap->g_unique_data[j] = (int)i;
         j++;
      }
   }
   return 1;
}

/* Rank-local projection of the parent level onto the nested MGR: admits the
 * level, allocates the wrapper and builds the projected dofmap. Whatever was
 * allocated is returned through *wrapper and *dofmap, also on failure. */
static int
MGRNestedFRelaxProject(const MGR_args *parent, int level, MGR_args **nested_args,
                       MGRFRelaxWrapper **wrapper, IntArray **dofmap)
{
   HYPRE_Int *parent_present = NULL, *keep_label = NULL;
   size_t     label_space = 0, nested_size = 0;
   int        ok = 0;

   if (!MGRNestedFRelaxAdmit(parent, level, nested_args)) return 0;
   const IntArray      *parent_dofmap = parent->dofmap;
   const StackIntArray *f_dofs        = &parent->level[level].f_dofs;
   if (!hypredrv_MGRBuildDofLabelPresenceMask(parent_dofmap, &label_space, NULL,
                                              &parent_present))
      goto done;
   keep_label = calloc(label_space, sizeof(*keep_label));
   if (!keep_label)
   {
      hypredrv_AllocationFailure(
         "Failed to allocate nested MGR dof label selection mask");
      goto done;
   }
   if (!MGRNestedFRelaxSelectLabels(parent_dofmap, f_dofs, label_space, parent_present,
                                    keep_label, &nested_size))
      goto done;
   *wrapper = calloc(1, sizeof(**wrapper));
   if (!*wrapper)
   {
      hypredrv_AllocationFailure("Failed to allocate nested MGR F-relaxation wrapper");
      goto done;
   }
   (*wrapper)->nested_args = *nested_args;
   ok = MGRNestedFRelaxBuildDofmap(parent_dofmap, f_dofs, keep_label, label_space,
                                   nested_size, dofmap);
done:
   free(parent_present);
   free(keep_label);
   return ok;
}

HYPRE_Solver
hypredrv_MGRNestedFRelaxCreate(const MGR_args *parent, int level, const Stats *stats,
                               int next_ls_id)
{
   MGRFRelaxWrapper *wrapper     = NULL;
   MGR_args         *nested_args = NULL;
   IntArray         *dofmap      = NULL;
   int               local_ok = 0, global_ok = 0;

   if (!parent || parent->comm == MPI_COMM_NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires a parent communicator");
      return NULL;
   }
   local_ok = MGRNestedFRelaxProject(parent, level, &nested_args, &wrapper, &dofmap);
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, parent->comm);
   if (!global_ok)
   {
      if (local_ok)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Nested MGR projection failed on another rank");
      }
      goto fail;
   }

   IntArray      *saved_dofmap = nested_args->dofmap;
   HYPRE_IJVector saved_vec_nn = nested_args->vec_nn;
   nested_args->dofmap         = dofmap;
   nested_args->vec_nn         = NULL;
   HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                      "creating nested MGR F-relaxation at level %d (coarsest=%s)", level,
                      hypredrv_MGRCoarseSolverTypeName(&nested_args->coarsest_level));
   hypredrv_MGRCreate(parent->comm, nested_args, &wrapper->inner_mgr, stats, next_ls_id);
   nested_args->dofmap = saved_dofmap;
   nested_args->vec_nn = saved_vec_nn;
   /* MGRCreate copied its point markers; only that owned buffer must outlive
    * native setup. The projected IntArray is construction scratch. */
   hypredrv_IntArrayDestroy(&dofmap);
   local_ok = !hypredrv_ErrorCodeActive() && wrapper->inner_mgr;
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, parent->comm);
   if (!global_ok)
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Nested MGR creation failed%s",
                              local_ok ? " on another rank" : "");
      }
      goto fail;
   }

   wrapper->base.fields.setup     = MGRFRelaxWrapperSetup;
   wrapper->base.fields.solve     = MGRFRelaxWrapperSolve;
   wrapper->base.fields.destroy   = MGRFRelaxWrapperDestroy;
   wrapper->next_live             = g_mgr_frelax_wrapper_live_head;
   g_mgr_frelax_wrapper_live_head = wrapper;
   return (HYPRE_Solver)wrapper;

fail:
   hypredrv_IntArrayDestroy(&dofmap);
   if (wrapper)
   {
      if (wrapper->inner_mgr) hypredrv_MGRDestroy(nested_args, &wrapper->inner_mgr, 0);
      free(wrapper);
   }
   return NULL;
}

void
hypredrv_MGRNestedFRelaxDestroy(HYPRE_Solver *solver_ptr)
{
   if (!solver_ptr) return;
   MGRFRelaxWrapperDestroy(*solver_ptr);
   *solver_ptr = NULL;
}

/* Convergence and argument flags left by earlier solves do not block
 * replacing a smoother; any other hypre error does. */
static int
MGRHypreErrorOutstanding(void)
{
   return (HYPRE_GetError() & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG)) != 0;
}

#if HYPRE_CHECK_MIN_VERSION(23100, 8)
typedef struct
{
   HYPRE_Int               type;
   HYPRE_PtrToParSolverFcn setup, solve;
   HYPRE_Int (*destroy)(HYPRE_Solver);
} MGRGRelaxOps;

typedef struct MGRGRelaxRuntime_struct
{
   MGRSolverBase                   base;
   const MGRGRelaxOps             *ops;
   HYPRE_Solver                    inner;
   struct MGRGRelaxRuntime_struct *next;
} MGRGRelaxRuntime;

typedef char MGRGRelaxLayoutCheck[(offsetof(MGRGRelaxRuntime, base.fields.is_setup) ==
                                   MGR_HYPRE_SOLVER_IS_SETUP_OFFSET)
                                     ? 1
                                     : -1];

static MGRGRelaxRuntime *g_mgr_grelax_live;

static HYPRE_Int
MGRGRelaxSetup(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   MGRGRelaxRuntime *runtime     = (MGRGRelaxRuntime *)solver;
   runtime->base.fields.is_setup = 0;
   HYPRE_Int status = runtime->ops->setup(runtime->inner, (HYPRE_ParCSRMatrix)A,
                                          (HYPRE_ParVector)b, (HYPRE_ParVector)x);
   if (!status) runtime->base.fields.is_setup = 1;
   return status;
}

static HYPRE_Int
MGRGRelaxSolve(HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   MGRGRelaxRuntime *runtime = (MGRGRelaxRuntime *)solver;
   if (!runtime->base.fields.is_setup)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "MGR global smoother requires setup");
      return HYPRE_GetError();
   }
   return runtime->ops->solve(runtime->inner, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)b,
                              (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRGRelaxRuntimeDestroy(HYPRE_Solver solver)
{
   /* Parent hypre teardown may have already reclaimed this cached handle. */
   MGRGRelaxRuntime **cursor = &g_mgr_grelax_live;
   while (*cursor && (HYPRE_Solver)*cursor != solver) cursor = &(*cursor)->next;
   if (!*cursor) return 0;
   MGRGRelaxRuntime *runtime = *cursor;
   *cursor                   = runtime->next;
   runtime->ops->destroy(runtime->inner);
   free(runtime);
   return 0;
}

static const MGRGRelaxOps g_mgr_grelax_methods[] = {
   {20, HYPRE_BoomerAMGSetup, HYPRE_BoomerAMGSolve, hypredrv_AMGDestroy},
   {16, HYPRE_ILUSetup, HYPRE_ILUSolve, HYPRE_ILUDestroy},
#ifdef HYPRE_USING_DSUPERLU
   {29, HYPRE_MGRDirectSolverSetup, HYPRE_MGRDirectSolverSolve,
    HYPRE_MGRDirectSolverDestroy},
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   {33, HYPRE_FSAISetup, HYPRE_FSAISolve, HYPRE_FSAIDestroy},
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   {MGR_SOLVER_TYPE_SCHWARZ, HYPRE_SchwarzSetup, HYPRE_SchwarzSolve,
    HYPRE_SchwarzDestroy},
#endif
};

/* Resolve the active level's configuration, its method and the runtime
 * cached in its slot. Returns the original level index, or -1. */
static int
MGRGRelaxLookup(MGR_args *args, int active_level, MGRgrlx_args **config,
                const MGRGRelaxOps **ops, MGRGRelaxRuntime **cached)
{
   int orig_level = -1;
   if (active_level >= 0 && active_level < args->num_active_levels &&
       active_level < MAX_MGR_LEVELS - 1)
      orig_level = args->active_level_map[active_level];
   if (orig_level >= 0 && orig_level < hypredrv_MGRNumFineLevels(args))
   {
      *config = &args->level[orig_level].g_relaxation;
      for (size_t i = 0; i < sizeof(g_mgr_grelax_methods) / sizeof(*g_mgr_grelax_methods);
           i++)
         if (g_mgr_grelax_methods[i].type == (*config)->type)
            *ops = &g_mgr_grelax_methods[i];
      for (MGRGRelaxRuntime *r = g_mgr_grelax_live; r; r = r->next)
         if ((HYPRE_Solver)r == args->grelax[orig_level]) *cached = r;
   }
   return orig_level;
}

/* Rank-local admission checks; they only record errors. */
static void
MGRGRelaxValidate(const MGRgrlx_args *config, const MGRGRelaxOps *ops, int active_level)
{
   if (!ops)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      if (config && config->type == 29)
         hypredrv_ErrorMsgAdd(
            "MGR G-relaxation 'spdirect' requires hypre built with DSUPERLU");
      else if (config && config->type == MGR_SOLVER_TYPE_SCHWARZ)
         hypredrv_ErrorMsgAdd(
            "MGR G-relaxation 'schwarz' requires hypre >= 3.1.0 (develop >= 55)");
      else
         hypredrv_ErrorMsgAdd(
            "Unsupported MGR global smoother at active level %d (type %d)", active_level,
            config ? (int)config->type : -1);
   }
   if (MGRHypreErrorOutstanding())
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Cannot replace MGR global smoother with an outstanding hypre error");
   }
}

/* Collective: agree that no rank recorded an error and that every rank admitted
 * the same method. Sets *create when any rank needs a new runtime. */
static int
MGRGRelaxAgreeAdmission(MPI_Comm comm, int needs_new, int type, int *create)
{
   int local_state[4] = {(int)hypredrv_ErrorCodeActive() != 0, needs_new, type, -type};
   int global_state[4];
   MPI_Allreduce(local_state, global_state, 4, MPI_INT, MPI_MAX, comm);
   if (global_state[0] || global_state[2] != -global_state[3])
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR global smoother admission failed on another rank");
      }
      return 0;
   }
   *create = global_state[1];
   return 1;
}

/* Collective: every rank allocates its runtime before any rank creates a
 * child. Returns NULL on every rank if any rank failed. */
static MGRGRelaxRuntime *
MGRGRelaxRuntimeAllocate(MPI_Comm comm)
{
   MGRGRelaxRuntime *candidate = calloc(1, sizeof(*candidate));
   int               local_ok = candidate != NULL, global_ok = 0;
   if (!candidate)
      hypredrv_AllocationFailure("Failed to allocate MGR global smoother runtime");
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!global_ok)
   {
      free(candidate);
      if (local_ok)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR global smoother allocation failed on another rank");
      }
      return NULL;
   }
   return candidate;
}

static void
MGRGRelaxCreateInner(MGRgrlx_args *config, const MGRGRelaxOps *ops, HYPRE_Solver *inner)
{
   switch (ops->type)
   {
      case 20:
         hypredrv_AMGCreate(&config->amg, inner);
         break;
      case 16:
         hypredrv_ILUCreate(&config->ilu, inner);
         break;
#ifdef HYPRE_USING_DSUPERLU
      case 29:
         HYPRE_MGRDirectSolverCreate(inner);
         break;
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      case 33:
         hypredrv_FSAICreate(&config->fsai, inner);
         break;
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      case MGR_SOLVER_TYPE_SCHWARZ:
         hypredrv_SchwarzCreate(&config->schwarz, inner);
         break;
#endif
      default:
         break;
   }
}

/* Collective: create the candidate's child and agree that every rank has one.
 * On failure the candidate is released on every rank. */
static int
MGRGRelaxCandidateCreate(MPI_Comm comm, MGRgrlx_args *config, MGRGRelaxRuntime *candidate)
{
   const MGRGRelaxOps *ops      = candidate->ops;
   int                 local_ok = 0, global_ok = 0;
   MGRGRelaxCreateInner(config, ops, &candidate->inner);
   local_ok =
      candidate->inner && !hypredrv_ErrorCodeActive() && !MGRHypreErrorOutstanding();
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!global_ok)
   {
      if (candidate->inner) ops->destroy(candidate->inner);
      free(candidate);
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR global smoother creation failed%s",
                              local_ok ? " on another rank" : "");
      }
      return 0;
   }
   return 1;
}
#endif

void
hypredrv_MGRGRelaxDestroy(HYPRE_Solver *solver)
{
   if (!solver) return;
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   MGRGRelaxRuntimeDestroy(*solver);
#endif
   *solver = NULL;
}

int
hypredrv_MGRGRelaxAttach(MGR_args *args, HYPRE_Solver parent, int active_level,
                         int refresh, const Stats *stats, int next_ls_id)
{
#if HYPRE_CHECK_MIN_VERSION(23100, 8)
   if (!args || !parent || args->comm == MPI_COMM_NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR global smoother requires a parent hierarchy");
      return 0;
   }
   MGRgrlx_args       *config = NULL;
   const MGRGRelaxOps *ops    = NULL;
   MGRGRelaxRuntime   *cached = NULL;
   int orig_level = MGRGRelaxLookup(args, active_level, &config, &ops, &cached);
   MGRGRelaxValidate(config, ops, active_level);
   int create = 0;
   if (!MGRGRelaxAgreeAdmission(args->comm, refresh || !cached || cached->ops != ops,
                                ops ? (int)ops->type : 0, &create))
      return 0;
   if (!create)
   {
      HYPRE_MGRSetGlobalSmootherAtLevel(parent, (HYPRE_Solver)cached, active_level);
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR G-relax solver handle at level %d",
                         orig_level);
      return 1;
   }

   MGRGRelaxRuntime *candidate = MGRGRelaxRuntimeAllocate(args->comm);
   if (!candidate) return 0;
   candidate->ops = ops;
   if (!MGRGRelaxCandidateCreate(args->comm, config, candidate)) return 0;
   candidate->base.fields = (hypredrv_SolverBase){.setup   = MGRGRelaxSetup,
                                                  .solve   = MGRGRelaxSolve,
                                                  .destroy = MGRGRelaxRuntimeDestroy};
   HYPRE_MGRSetGlobalSmootherAtLevel(parent, (HYPRE_Solver)candidate, active_level);
   hypredrv_MGRGRelaxDestroy(&args->grelax[orig_level]);
   candidate->next          = g_mgr_grelax_live;
   g_mgr_grelax_live        = candidate;
   args->grelax[orig_level] = (HYPRE_Solver)candidate;
   return 1;
#else
   (void)args;
   (void)parent;
   (void)active_level;
   (void)refresh;
   (void)stats;
   (void)next_ls_id;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd(
      "User MGR global smoothers require hypre >= 2.31.0 (develop >= 8)");
   return 0;
#endif
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

   wrapper->base.fields.is_setup = 1;
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

   if (!wrapper->base.fields.is_setup && MGRSchwarzWrapperSetup(wrapper_v, A, b, x))
   {
      return 1;
   }

   return HYPRE_SchwarzSolve((HYPRE_Solver)wrapper->inner, (HYPRE_ParCSRMatrix)A,
                             (HYPRE_ParVector)b, (HYPRE_ParVector)x);
}

static HYPRE_Int
MGRSchwarzWrapperDestroy(HYPRE_Solver wrapper_v)
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

static HYPRE_Int
hypredrv_MGRSchwarzWrapperParSetup(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSetup(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

static HYPRE_Int
hypredrv_MGRSchwarzWrapperParSolve(HYPRE_Solver wrapper, HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x)
{
   return MGRSchwarzWrapperSolve(wrapper, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                 (HYPRE_Vector)x);
}

#endif

/* GCOVR_EXCL_START */
static void
MGRFRelaxInstall(HYPRE_Solver precon, const MGRfrlx_args *f_relaxation,
                 HYPRE_Solver frelax, int active_lvl)
{
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   if (f_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      hypredrv_MGRSetFSolverAtLevel(
         precon, frelax, active_lvl, MGR_FRLX_TYPE_CUSTOM_SOLVER_CB,
         hypredrv_MGRSchwarzWrapperParSolve, hypredrv_MGRSchwarzWrapperParSetup);
      return;
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
   hypredrv_MGRSetFSolverAtLevel(precon, frelax, active_lvl, f_relaxation->type, NULL,
                                 NULL);
#else
   /* Only the level-0 AMG F-solver slot is available on this hypre version. */
   if (f_relaxation->type == 2)
   {
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
      /* Legacy hypre registers callbacks independently of its F-relaxation
       * strategy. Select those callbacks for the one user-supplied F level. */
      HYPRE_Int methods[MAX_MGR_LEVELS - 1] = {2};
      HYPRE_MGRSetLevelFRelaxMethod(precon, methods);
#endif
      HYPRE_MGRSetFSolver(precon, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, frelax);
   }
   (void)active_lvl;
#endif
}
/* GCOVR_EXCL_STOP */

#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
typedef struct MGRFRelaxOperator_struct
{
   HYPRE_ParCSRMatrix               matrix;
   struct MGRFRelaxOperator_struct *next;
} MGRFRelaxOperator;
#endif

typedef HYPRE_Int (*MGRFRelaxDestroyFcn)(HYPRE_Solver);

typedef struct MGRFRelaxBinding_struct
{
   HYPRE_Solver *slot;
   HYPRE_Solver  solver;
   HYPRE_Int (*destroy)(HYPRE_Solver);
   int kind;
   int active_level;
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   HYPRE_Solver       parent;
   HYPRE_Int         *f_markers;
   size_t             marker_size;
   int                native_levels;
   MGRFRelaxOperator *operators;
   MGRFRelaxOperator *parent_operator;
#endif
   struct MGRFRelaxBinding_struct *next;
} MGRFRelaxBinding;

/* F-solver handles must retain their native representation. Bind their
 * destructor alongside the owning cache slot rather than wrapping the handle. */
static MGRFRelaxBinding *g_mgr_frelax_bindings;

static void
MGRFRelaxBindingDestroy(MGRFRelaxBinding *binding, int release_child)
{
   if (!binding) return;
   if (release_child && binding->solver) binding->destroy(binding->solver);
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   while (binding->operators)
   {
      MGRFRelaxOperator *work = binding->operators;
      binding->operators      = work->next;
      HYPRE_ParCSRMatrixDestroy(work->matrix);
      free(work);
   }
   free(binding->f_markers);
#endif
   free(binding);
}

void
hypredrv_MGRFRelaxDestroy(HYPRE_Solver *slot, int parent_destroyed)
{
   if (!slot) return;
   MGRFRelaxBinding **cursor = &g_mgr_frelax_bindings;
   while (*cursor && (*cursor)->slot != slot) cursor = &(*cursor)->next;
   if (*cursor)
   {
      MGRFRelaxBinding *binding = *cursor;
      *cursor                   = binding->next;
      int release_child;
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
      (void)parent_destroyed;
      release_child = 1;
#elif !HYPRE_CHECK_MIN_VERSION(30100, 5)
      release_child = !parent_destroyed || binding->active_level == 0;
#else
      release_child = !parent_destroyed;
#endif
      MGRFRelaxBindingDestroy(binding, release_child);
   }
   else
   {
      /* Nested MGR wrappers track native teardown themselves. A prior parent
       * release can leave an already retired wrapper in this cache slot. */
      hypredrv_MGRNestedFRelaxDestroy(slot);
   }
   *slot = NULL;
}

HYPRE_Int
hypredrv_MGRSetup(HYPRE_Solver parent, HYPRE_ParCSRMatrix A, HYPRE_ParVector b,
                  HYPRE_ParVector x)
{
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   MGRFRelaxBinding *binding = g_mgr_frelax_bindings;
   while (binding && binding->parent != parent) binding = binding->next;
   if (binding)
   {
      /* Legacy MGR reads the supplied AMG's A_array[0] instead of setting up
       * that solver. Keep the extracted operator alive for both borrowers. */
      MPI_Comm           comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *)A);
      HYPRE_Int          rows = hypre_ParCSRMatrixNumRows((hypre_ParCSRMatrix *)A);
      MGRFRelaxOperator *work = calloc(1, sizeof(*work));
      int local_ok            = work && rows >= 0 && binding->marker_size >= (size_t)rows;
      int all_ok              = 0;
      MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      if (!all_ok)
      {
         free(work);
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Cannot prepare the legacy MGR F-block on every rank");
         return HYPRE_ERROR_GENERIC;
      }
      work->next         = binding->operators;
      binding->operators = work;
      HYPRE_Int status   = HYPRE_MGRBuildAff(A, binding->f_markers, 0, &work->matrix);
      local_ok           = !status && work->matrix;
      MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      if (all_ok)
      {
         status   = HYPRE_BoomerAMGSetup(binding->solver, work->matrix, NULL, NULL);
         local_ok = !status;
         MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      }
      if (!all_ok)
      {
         /* A partially set-up child can already borrow the candidate matrix. */
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("Legacy MGR F-solver setup failed");
         return status ? status : HYPRE_ERROR_GENERIC;
      }
      if (binding->parent_operator && binding->native_levels > 1)
      {
         /* On repeated setup legacy MGR destroys A_ff_array[0] when it has
          * multiple fine levels, despite borrowing it on initial setup. */
         MGRFRelaxOperator **cursor = &binding->operators;
         while (*cursor != binding->parent_operator) cursor = &(*cursor)->next;
         MGRFRelaxOperator *retired = *cursor;
         *cursor                    = retired->next;
         free(retired);
      }
      status                   = HYPRE_MGRSetup(parent, A, b, x);
      binding->parent_operator = work;
      local_ok                 = !status;
      MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
      if (!all_ok) return status ? status : HYPRE_ERROR_GENERIC;
      while (work->next)
      {
         MGRFRelaxOperator *retired = work->next;
         work->next                 = retired->next;
         HYPRE_ParCSRMatrixDestroy(retired->matrix);
         free(retired);
      }
      return 0;
   }
#endif
   return HYPRE_MGRSetup(parent, A, b, x);
}

/* Resolve the active level's F-relaxation configuration and the bindings in
 * its cache slot. Returns the original level index, or -1. */
static int
MGRFRelaxLookup(MGR_args *args, int active_lvl, MGRfrlx_args **f_relaxation,
                const StackIntArray **f_dofs, MGRFRelaxBinding **previous,
                MGRFRelaxBinding **cached)
{
   int orig_lvl = -1;
   if (active_lvl >= 0 && active_lvl < args->num_active_levels &&
       active_lvl < MAX_MGR_LEVELS - 1)
      orig_lvl = args->active_level_map[active_lvl];
   if (orig_lvl >= 0 && orig_lvl < hypredrv_MGRNumFineLevels(args))
   {
      *f_relaxation = &args->level[orig_lvl].f_relaxation;
      *f_dofs       = &args->level[orig_lvl].f_dofs;
      for (MGRFRelaxBinding *r = g_mgr_frelax_bindings; r; r = r->next)
         if (r->slot == &args->frelax[orig_lvl])
         {
            *previous = r;
            if (r->solver == args->frelax[orig_lvl]) *cached = r;
         }
   }
   return orig_lvl;
}

/* Select the destroy function of a managed F-solver method. Returns 0 if this
 * hypre build or level cannot host it; that only records an error message. */
static int
MGRFRelaxSelectMethod(const MGRfrlx_args *f_relaxation, int active_lvl,
                      MGRFRelaxDestroyFcn *destroy)
{
   int supported = 0;
   (void)active_lvl;
   switch (f_relaxation->type)
   {
      case 2:
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         supported = 1;
#else
         if (active_lvl == 0) supported = 1;
         else
            hypredrv_ErrorMsgAdd("Managed AMG F-solvers above level 0 require hypre "
                                 ">= 2.31.0 (develop >= 9)");
#endif
         *destroy = hypredrv_AMGDestroy;
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
         if (f_relaxation->symmetric_diagonal_scaling)
            *destroy = hypredrv_MGRFRelaxEquilWrapperDestroy;
#endif
         break;
      case 29:
#if defined(HYPRE_USING_DSUPERLU) && HYPRE_CHECK_MIN_VERSION(23100, 9)
         supported = 1;
         *destroy  = HYPRE_MGRDirectSolverDestroy;
#else
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'spdirect' requires hypre >= 2.31.0 and DSUPERLU");
#endif
         break;
      case 32:
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
         supported = 1;
         *destroy  = HYPRE_ILUDestroy;
#else
         hypredrv_ErrorMsgAdd(
            "Managed MGR F-relaxation 'ilu' requires hypre >= 2.32.0 (develop >= 14)");
#endif
         break;
      case 33:
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
         supported = 1;
         *destroy  = HYPRE_FSAIDestroy;
#else
         hypredrv_ErrorMsgAdd("MGR F-relaxation 'fsai' requires hypre >= 2.31.0");
#endif
         break;
      case MGR_SOLVER_TYPE_SCHWARZ:
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
         if (active_lvl == 0)
         {
            supported = 1;
            *destroy  = MGRSchwarzWrapperDestroy;
         }
         else
         {
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation 'schwarz' is only supported at MGR level 0");
         }
#else
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'schwarz' requires hypre >= 3.1.0 (develop >= 55)");
#endif
         break;
      default:
         break;
   }
   return supported;
}

static void
MGRFRelaxValidateScaling(const MGR_args *args, const MGRfrlx_args *f_relaxation)
{
#if !HYPRE_CHECK_MIN_VERSION(30100, 0)
   (void)args;
   (void)f_relaxation;
   hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
   hypredrv_ErrorMsgAdd("MGR symmetric_diagonal_scaling requires hypre >= 3.1.0");
#else
   if (f_relaxation->symmetric_diagonal_scaling != 1 || f_relaxation->type != 2 ||
       f_relaxation->use_krylov || f_relaxation->reuse.present || args->vec_nn)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "MGR symmetric_diagonal_scaling currently requires a managed AMG F-solver "
         "without nested Krylov, component reuse, or projected RBMs");
   }
#endif
}

/* Rank-local admission checks; they only record errors. */
static void
MGRFRelaxValidate(const MGR_args *args, const MGRfrlx_args *f_relaxation, int supported,
                  int active_lvl, int refresh, const MGRFRelaxBinding *previous)
{
   if (!supported)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Unsupported managed MGR F-solver at active level %d",
                           active_lvl);
   }
   if (f_relaxation && f_relaxation->symmetric_diagonal_scaling)
      MGRFRelaxValidateScaling(args, f_relaxation);
   if (MGRHypreErrorOutstanding())
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Cannot replace MGR F-solver with an outstanding hypre error");
   }
   if (refresh && previous && f_relaxation &&
       (previous->kind / 2 != f_relaxation->type || previous->active_level != active_lvl))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Changing the MGR F-solver method or level requires a new parent hierarchy");
   }
}

/* Collective: agree that no rank recorded an error and that every rank admitted
 * the same method and level. Sets *create when any rank needs a new F-solver. */
static int
MGRFRelaxAgreeAdmission(MPI_Comm comm, int needs_new, int kind, int level, int *create)
{
   int local_state[6] = {
      (int)hypredrv_ErrorCodeActive() != 0, needs_new, kind, -kind, level, -level};
   int global_state[6];
   MPI_Allreduce(local_state, global_state, 6, MPI_INT, MPI_MAX, comm);
   if (global_state[0] || global_state[2] != -global_state[3] ||
       global_state[4] != -global_state[5])
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR F-solver admission failed or registrations differ between ranks");
      }
      return 0;
   }
   *create = global_state[1];
   return 1;
}

#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
/* Legacy MGR builds the F-block from these row markers in hypredrv_MGRSetup.
 * A refresh keeps the markers of the binding it replaces. */
static int
MGRFRelaxBindingInitLegacy(MGRFRelaxBinding *binding, const MGR_args *args,
                           HYPRE_Solver parent, int refresh,
                           const MGRFRelaxBinding *previous, const StackIntArray *f_dofs)
{
   binding->parent        = parent;
   binding->native_levels = args->num_active_levels;
   binding->marker_size =
      refresh && previous ? previous->marker_size : args->dofmap->size;
   binding->f_markers = calloc(binding->marker_size ? binding->marker_size : 1,
                               sizeof(*binding->f_markers));
   if (!binding->f_markers) return 0;
   if (refresh && previous)
      memcpy(binding->f_markers, previous->f_markers,
             binding->marker_size * sizeof(*binding->f_markers));
   else
      for (size_t row = 0; row < binding->marker_size; row++)
      {
         binding->f_markers[row] = 1;
         for (size_t j = 0; j < f_dofs->size; j++)
            if (args->dofmap->data[row] == f_dofs->data[j]) binding->f_markers[row] = -1;
      }
   return 1;
}
#endif

/* Collective: every rank allocates the binding for its new F-solver before
 * any rank creates one. Returns NULL on every rank if any rank failed. */
static MGRFRelaxBinding *
MGRFRelaxBindingAllocate(const MGR_args *args, HYPRE_Solver parent, int refresh,
                         const MGRFRelaxBinding *previous, const StackIntArray *f_dofs)
{
   int               local_ok = 0, global_ok = 0;
   MGRFRelaxBinding *binding = calloc(1, sizeof(*binding));
   local_ok                  = binding != NULL;
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   if (binding)
      local_ok =
         MGRFRelaxBindingInitLegacy(binding, args, parent, refresh, previous, f_dofs);
#else
   (void)parent;
   (void)refresh;
   (void)previous;
   (void)f_dofs;
#endif
   if (!local_ok) hypredrv_AllocationFailure("Failed to bind the MGR F-solver lifetime");
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, args->comm);
   if (!global_ok)
   {
      MGRFRelaxBindingDestroy(binding, 0);
      if (local_ok)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR F-solver lifetime allocation failed on another rank");
      }
      return NULL;
   }
   return binding;
}

static HYPRE_Solver
MGRFRelaxCreateAMG(MGR_args *args, MGRfrlx_args *f_relaxation,
                   const StackIntArray *f_dofs)
{
   HYPRE_Solver solver = NULL;
   hypredrv_AMGSetProjectedRBMs(&f_relaxation->amg, args->vec_nn, args->dofmap,
                                args->rbm_input_generation, f_dofs->data, f_dofs->size);
   if (hypredrv_ErrorCodeActive())
   {
      return NULL;
   }
   hypredrv_AMGCreate(&f_relaxation->amg, &solver);
   if (solver && f_relaxation->symmetric_diagonal_scaling)
   {
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
      solver = hypredrv_MGRFRelaxEquilWrapperCreate(args->comm, solver);
#else
      hypredrv_AMGDestroy(solver);
      solver = NULL;
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR symmetric_diagonal_scaling requires hypre >= 3.1.0");
#endif
   }
   return solver;
}

#if HYPRE_CHECK_MIN_VERSION(30100, 55)
static HYPRE_Solver
MGRFRelaxCreateSchwarz(MGRfrlx_args *f_relaxation)
{
   MGRSchwarzWrapper *wrapper = calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_AllocationFailure("Failed to allocate MGR Schwarz solver wrapper");
      return NULL;
   }
   wrapper->base.fields.setup   = MGRSchwarzWrapperSetup;
   wrapper->base.fields.solve   = MGRSchwarzWrapperSolve;
   wrapper->base.fields.destroy = MGRSchwarzWrapperDestroy;
   hypredrv_SchwarzCreate(&f_relaxation->schwarz, &wrapper->inner);
   if (!wrapper->inner && !hypredrv_ErrorCodeActive())
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Failed to create MGR Schwarz F-solver");
   }
   return (HYPRE_Solver)wrapper;
}
#endif

/* Rank-local creation of the candidate F-solver. Returns NULL if creation
 * failed before a handle existed. */
static HYPRE_Solver
MGRFRelaxCreateSolver(MGR_args *args, MGRfrlx_args *f_relaxation,
                      const StackIntArray *f_dofs)
{
   HYPRE_Solver solver = NULL;
   if (f_relaxation->type == 2)
   {
      solver = MGRFRelaxCreateAMG(args, f_relaxation, f_dofs);
   }
#if defined(HYPRE_USING_DSUPERLU)
   /* GCOVR_EXCL_START */
   else if (f_relaxation->type == 29)
   {
      HYPRE_MGRDirectSolverCreate(&solver);
      if (!solver)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd(
            "MGR F-relaxation 'spdirect' unavailable: direct solver creation failed");
      }
   }
   /* GCOVR_EXCL_STOP */
#endif
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
   else if (f_relaxation->type == 32)
   {
      hypredrv_ILUCreate(&f_relaxation->ilu, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
   else if (f_relaxation->type == 33)
   {
      hypredrv_FSAICreate(&f_relaxation->fsai, &solver);
   }
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
   else if (f_relaxation->type == MGR_SOLVER_TYPE_SCHWARZ)
   {
      solver = MGRFRelaxCreateSchwarz(f_relaxation);
   }
#endif
   return solver;
}

/* Collective: agree that every rank created its candidate. On failure the
 * candidate and its binding are released on every rank. */
static int
MGRFRelaxAgreeCandidate(MPI_Comm comm, HYPRE_Solver solver, MGRFRelaxDestroyFcn destroy,
                        MGRFRelaxBinding *binding)
{
   int local_ok  = solver && !hypredrv_ErrorCodeActive() && !MGRHypreErrorOutstanding();
   int global_ok = 0;
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
   if (!global_ok)
   {
      if (solver) destroy(solver);
      MGRFRelaxBindingDestroy(binding, 0);
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR F-solver creation failed%s",
                              local_ok ? " on another rank" : "");
      }
      return 0;
   }
   return 1;
}

/* Retire the slot's previous binding and register the new one in its place. */
static void
MGRFRelaxBindingRegister(MGRFRelaxBinding *binding, MGRFRelaxBinding *previous,
                         HYPRE_Solver parent, HYPRE_Solver *slot, HYPRE_Solver solver,
                         MGRFRelaxDestroyFcn destroy, int kind)
{
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   if (previous)
   {
      /* The old parent may still borrow an operator after its child has
       * been replaced. Release it only after parent setup or teardown. */
      binding->operators  = previous->operators;
      previous->operators = NULL;
      if (previous->parent == parent)
      {
         binding->native_levels   = previous->native_levels;
         binding->parent_operator = previous->parent_operator;
      }
   }
#else
   (void)previous;
   (void)parent;
#endif
   hypredrv_MGRFRelaxDestroy(slot, 0);
   binding->slot         = slot;
   binding->solver       = solver;
   binding->destroy      = destroy;
   binding->kind         = kind;
   binding->next         = g_mgr_frelax_bindings;
   g_mgr_frelax_bindings = binding;
}

int
hypredrv_MGRFRelaxAttach(MGR_args *args, HYPRE_Solver parent, int active_lvl, int refresh,
                         const Stats *stats, int next_ls_id)
{
   if (!args || !parent || args->comm == MPI_COMM_NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation requires a parent hierarchy");
      return 0;
   }
   MGRfrlx_args        *f_relaxation = NULL;
   const StackIntArray *f_dofs       = NULL;
   int                  supported    = 0;
   MGRFRelaxDestroyFcn  destroy      = NULL;
   MGRFRelaxBinding    *cached       = NULL;
   MGRFRelaxBinding    *previous     = NULL;
   int                  orig_lvl =
      MGRFRelaxLookup(args, active_lvl, &f_relaxation, &f_dofs, &previous, &cached);
   if (f_relaxation)
      supported = MGRFRelaxSelectMethod(f_relaxation, active_lvl, &destroy);
   MGRFRelaxValidate(args, f_relaxation, supported, active_lvl, refresh, previous);
   int kind   = supported && f_relaxation ? (2 * (int)f_relaxation->type) +
                                             !!f_relaxation->symmetric_diagonal_scaling
                                          : 0;
   int level  = f_relaxation ? active_lvl : 0;
   int create = 0;
   if (!MGRFRelaxAgreeAdmission(args->comm, refresh || !cached || cached->kind != kind,
                                kind, level, &create))
      return 0;
   HYPRE_Solver      solver  = args->frelax[orig_lvl];
   MGRFRelaxBinding *binding = NULL;
   if (create)
   {
      binding = MGRFRelaxBindingAllocate(args, parent, refresh, previous, f_dofs);
      if (!binding) return 0;
      solver = MGRFRelaxCreateSolver(args, f_relaxation, f_dofs);
      if (!MGRFRelaxAgreeCandidate(args->comm, solver, destroy, binding)) return 0;
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR F-relax solver handle at level %d",
                         orig_lvl);
   }
   MGRFRelaxInstall(parent, f_relaxation, solver, active_lvl);
   if (binding)
      MGRFRelaxBindingRegister(binding, previous, parent, &args->frelax[orig_lvl], solver,
                               destroy, kind);
   else binding = cached;
   binding->active_level = active_lvl;
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
   if (binding->parent != parent)
   {
      binding->parent          = parent;
      binding->parent_operator = NULL;
      binding->native_levels   = args->num_active_levels;
   }
#endif
   args->frelax[orig_lvl] = solver;
   return 1;
}

HYPRE_Int
hypredrv_MGRBaseParSolverSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}

HYPRE_Int
hypredrv_MGRBaseParSolverSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x)
{
   return hypredrv_NestedKrylovSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b,
                                     (HYPRE_Vector)x);
}
