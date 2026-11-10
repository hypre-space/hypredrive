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

   /* This callback builds its own vectors. Admit only the operator it reads,
    * before any rank allocates or enters the child's collective setup. */
   hypre_CSRMatrix *diag       = A ? hypre_ParCSRMatrixDiag(A) : NULL;
   HYPRE_Int        n          = diag ? hypre_CSRMatrixNumRows(diag) : 0;
   HYPRE_Int       *rows       = diag ? hypre_CSRMatrixI(diag) : NULL;
   HYPRE_Int       *cols       = diag ? hypre_CSRMatrixJ(diag) : NULL;
   HYPRE_Complex   *data       = diag ? hypre_CSRMatrixData(diag) : NULL;
   int              comparison = MPI_UNEQUAL;
   if (A && hypre_ParCSRMatrixComm(A) != MPI_COMM_NULL)
      MPI_Comm_compare(wrapper->comm, hypre_ParCSRMatrixComm(A), &comparison);
   int local_ok =
      !HYPRE_GetError() && A && diag && n >= 0 &&
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
      hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A)) == HYPRE_EXEC_HOST &&
      hypre_ParCSRMatrixGlobalNumRows(A) == hypre_ParCSRMatrixGlobalNumCols(A) &&
      hypre_ParCSRMatrixFirstRowIndex(A) == hypre_ParCSRMatrixFirstColDiag(A) &&
      hypre_ParCSRMatrixLastRowIndex(A) == hypre_ParCSRMatrixLastColDiag(A) &&
      n == hypre_ParCSRMatrixLastRowIndex(A) - hypre_ParCSRMatrixFirstRowIndex(A) + 1 &&
      (n == 0 || (rows && cols && data));
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block scaling requires a valid CPU operator "
                        "with aligned square partitions on its creation communicator");
      return HYPRE_GetError();
   }

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
   local_ok = work && work->scale && work->inverse_scale && work->scaled_rhs &&
              work->scaled_solution && work->scaled_A && !HYPRE_GetError();
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_MEMORY,
                        "Failed to allocate symmetric F-block scaling data");
      MGRFRelaxEquilWorkDestroy(work);
      return HYPRE_GetError();
   }

   HYPRE_Complex *scale = hypre_VectorData(hypre_ParVectorLocalVector(work->scale));
   HYPRE_Complex *inverse =
      hypre_VectorData(hypre_ParVectorLocalVector(work->inverse_scale));
   local_ok = 1;
   for (HYPRE_Int i = 0; i < n; i++)
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
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, local_ok))
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
   MGRFRelaxEquilWork *work    = wrapper->work;
   hypre_Vector       *b_local = b ? hypre_ParVectorLocalVector(b) : NULL;
   hypre_Vector       *x_local = x ? hypre_ParVectorLocalVector(x) : NULL;
   int                 local_ok =
      !HYPRE_GetError() && wrapper->base.fields.is_setup && b_local && x_local &&
      hypre_ParVectorNumVectors(b) == 1 && hypre_ParVectorNumVectors(x) == 1 &&
      hypre_VectorSize(b_local) == work->n && hypre_VectorSize(x_local) == work->n &&
      hypre_ParVectorGlobalSize(b) == hypre_ParVectorGlobalSize(work->scale) &&
      hypre_ParVectorGlobalSize(x) == hypre_ParVectorGlobalSize(work->scale) &&
      hypre_ParVectorFirstIndex(b) == hypre_ParVectorFirstIndex(work->scale) &&
      hypre_ParVectorFirstIndex(x) == hypre_ParVectorFirstIndex(work->scale) &&
      hypre_GetExecPolicy1(hypre_VectorMemoryLocation(b_local)) == HYPRE_EXEC_HOST &&
      hypre_GetExecPolicy1(hypre_VectorMemoryLocation(x_local)) == HYPRE_EXEC_HOST &&
      (work->n == 0 || (hypre_VectorData(b_local) && hypre_VectorData(x_local)));
   if (!MGRFRelaxEquilAllRanksOk(wrapper->comm, local_ok))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "MGR symmetric F-block solve requires successful setup and "
                        "compatible CPU vectors on every rank");
      return HYPRE_GetError();
   }

   HYPRE_Complex *b_data = hypre_VectorData(b_local), *x_data = hypre_VectorData(x_local);
   HYPRE_Complex *scale = hypre_VectorData(hypre_ParVectorLocalVector(work->scale));
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
   HYPRE_BoomerAMGDestroy(wrapper->inner);
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
      if (inner) HYPRE_BoomerAMGDestroy(inner);
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Symmetric F-block scaling requires a communicator and initialized AMG child");
      return NULL;
   }
   MGRFRelaxEquilWrapper *wrapper = calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      HYPRE_BoomerAMGDestroy(inner);
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
   return (HYPRE_Solver)&wrapper->base.native;
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
   return HYPRE_MGRSolve((HYPRE_Solver)wrapper->inner_mgr, (HYPRE_ParCSRMatrix)A,
                         (HYPRE_ParVector)b, (HYPRE_ParVector)x);
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

HYPRE_Solver
hypredrv_MGRNestedFRelaxCreate(const MGR_args *parent, int level, const Stats *stats,
                               int next_ls_id)
{
   MGRFRelaxWrapper    *wrapper        = NULL;
   MGR_args            *nested_args    = NULL;
   IntArray            *dofmap         = NULL;
   const IntArray      *parent_dofmap  = NULL;
   const StackIntArray *f_dofs         = NULL;
   HYPRE_Int           *parent_present = NULL, *keep_label = NULL;
   size_t               label_space = 0, nested_size = 0;
   int                  local_ok = 0, global_ok = 0;

   if (!parent || parent->comm == MPI_COMM_NULL)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires a parent communicator");
      return NULL;
   }
   if (hypredrv_ErrorCodeActive() || HYPRE_GetError())
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Cannot create nested MGR with an outstanding error");
      goto admit;
   }
   if (level < 0 || level >= hypredrv_MGRNumFineLevels(parent))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Invalid parent level for nested MGR F-relaxation");
      goto admit;
   }
   nested_args = parent->level[level].f_relaxation.mgr;
   if (!nested_args)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("MGR F-relaxation type 'mgr' requires a nested 'mgr:' block");
      goto admit;
   }
   parent_dofmap = parent->dofmap;
   f_dofs        = &parent->level[level].f_dofs;
   if (!parent_dofmap || (parent_dofmap->size && !parent_dofmap->data) || !f_dofs->size ||
       f_dofs->size > MAX_STACK_ARRAY_LENGTH)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
      hypredrv_ErrorMsgAdd("Nested MGR requires a parent dofmap and non-empty f_dofs");
      goto admit;
   }
   if (!hypredrv_MGRBuildDofLabelPresenceMask(parent_dofmap, &label_space, NULL,
                                              &parent_present))
      goto admit;
   keep_label = calloc(label_space, sizeof(*keep_label));
   if (!keep_label)
   {
      hypredrv_AllocationFailure(
         "Failed to allocate nested MGR dof label selection mask");
      goto admit;
   }
   for (size_t i = 0; i < f_dofs->size; i++)
   {
      int label = f_dofs->data[i];
      if (label < 0 || (size_t)label >= label_space)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd(
            "Invalid parent MGR f_dofs label %d for nested MGR (valid range: [0,%d])",
            label, (int)label_space - 1);
         goto admit;
      }
      if (!parent_present[label] || keep_label[label])
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_VAL);
         hypredrv_ErrorMsgAdd("Parent MGR f_dofs label %d is absent or repeated", label);
         goto admit;
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
         goto admit;
      }
      if (keep_label[label]) nested_size++;
   }
   wrapper = calloc(1, sizeof(*wrapper));
   if (!wrapper)
   {
      hypredrv_AllocationFailure("Failed to allocate nested MGR F-relaxation wrapper");
      goto admit;
   }
   wrapper->nested_args = nested_args;
   dofmap               = hypredrv_IntArrayCreate(nested_size);
   if (!dofmap || (nested_size && !dofmap->data))
   {
      hypredrv_AllocationFailure("Failed to allocate nested MGR projected dofmap");
      goto admit;
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
      goto admit;
   }
   for (size_t i = 0, j = 0; i < label_space; i++)
   {
      if (keep_label[i])
      {
         dofmap->unique_data[j] = dofmap->g_unique_data[j] = (int)i;
         j++;
      }
   }
   local_ok = 1;
admit:
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, parent->comm);
   free(parent_present);
   free(keep_label);
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
   int                       orig_level = -1;
   MGRgrlx_args             *config     = NULL;
   const MGRGRelaxOps       *ops        = NULL;
   MGRGRelaxRuntime         *cached     = NULL;
   static const MGRGRelaxOps methods[]  = {
      {20, HYPRE_BoomerAMGSetup, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGDestroy},
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
   if (active_level >= 0 && active_level < args->num_active_levels &&
       active_level < MAX_MGR_LEVELS - 1)
      orig_level = args->active_level_map[active_level];
   if (orig_level >= 0 && orig_level < hypredrv_MGRNumFineLevels(args))
   {
      config = &args->level[orig_level].g_relaxation;
      for (size_t i = 0; i < sizeof(methods) / sizeof(*methods); i++)
         if (methods[i].type == config->type) ops = &methods[i];
      for (MGRGRelaxRuntime *r = g_mgr_grelax_live; r; r = r->next)
         if ((HYPRE_Solver)r == args->grelax[orig_level]) cached = r;
   }
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
   if (HYPRE_GetError() & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Cannot replace MGR global smoother with an outstanding hypre error");
   }
   int type           = ops ? (int)ops->type : 0;
   int local_state[4] = {hypredrv_ErrorCodeActive() != 0,
                         refresh || !cached || cached->ops != ops, type, -type};
   int global_state[4];
   MPI_Allreduce(local_state, global_state, 4, MPI_INT, MPI_MAX, args->comm);
   if (global_state[0] || global_state[2] != -global_state[3])
   {
      if (!hypredrv_ErrorCodeActive())
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR global smoother admission failed on another rank");
      }
      return 0;
   }
   if (!global_state[1])
   {
      HYPRE_MGRSetGlobalSmootherAtLevel(parent, (HYPRE_Solver)cached, active_level);
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR G-relax solver handle at level %d",
                         orig_level);
      return 1;
   }

   MGRGRelaxRuntime *candidate = calloc(1, sizeof(*candidate));
   int               local_ok = candidate != NULL, global_ok = 0;
   if (!candidate)
      hypredrv_AllocationFailure("Failed to allocate MGR global smoother runtime");
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, args->comm);
   if (!global_ok)
   {
      free(candidate);
      if (local_ok)
      {
         hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
         hypredrv_ErrorMsgAdd("MGR global smoother allocation failed on another rank");
      }
      return 0;
   }
   candidate->ops = ops;
   switch (ops->type)
   {
      case 20:
         hypredrv_AMGCreate(&config->amg, &candidate->inner);
         break;
      case 16:
         hypredrv_ILUCreate(&config->ilu, &candidate->inner);
         break;
#ifdef HYPRE_USING_DSUPERLU
      case 29:
         HYPRE_MGRDirectSolverCreate(&candidate->inner);
         break;
#endif
#if HYPRE_CHECK_MIN_VERSION(22500, 0)
      case 33:
         hypredrv_FSAICreate(&config->fsai, &candidate->inner);
         break;
#endif
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
      case MGR_SOLVER_TYPE_SCHWARZ:
         hypredrv_SchwarzCreate(&config->schwarz, &candidate->inner);
         break;
#endif
   }
   local_ok = candidate->inner && !hypredrv_ErrorCodeActive() &&
              !(HYPRE_GetError() & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG));
   MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, args->comm);
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
      int release_child         = !parent_destroyed;
#if HYPRE_CHECK_MIN_VERSION(30100, 28)
      release_child = 1;
#elif !HYPRE_CHECK_MIN_VERSION(30100, 5)
      release_child |= binding->active_level == 0;
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
   int                  orig_lvl      = -1;
   MGRfrlx_args        *f_relaxation  = NULL;
   const StackIntArray *f_dofs        = NULL;
   int                  supported     = 0;
   HYPRE_Int (*destroy)(HYPRE_Solver) = NULL;
   MGRFRelaxBinding *cached           = NULL;
   MGRFRelaxBinding *previous         = NULL;
   if (active_lvl >= 0 && active_lvl < args->num_active_levels &&
       active_lvl < MAX_MGR_LEVELS - 1)
      orig_lvl = args->active_level_map[active_lvl];
   if (orig_lvl >= 0 && orig_lvl < hypredrv_MGRNumFineLevels(args))
   {
      f_relaxation = &args->level[orig_lvl].f_relaxation;
      f_dofs       = &args->level[orig_lvl].f_dofs;
      for (MGRFRelaxBinding *r = g_mgr_frelax_bindings; r; r = r->next)
         if (r->slot == &args->frelax[orig_lvl])
         {
            previous = r;
            if (r->solver == args->frelax[orig_lvl]) cached = r;
         }
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
            destroy = HYPRE_BoomerAMGDestroy;
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
            if (f_relaxation->symmetric_diagonal_scaling)
               destroy = hypredrv_MGRFRelaxEquilWrapperDestroy;
#endif
            break;
         case 29:
#if defined(HYPRE_USING_DSUPERLU) && HYPRE_CHECK_MIN_VERSION(23100, 9)
            supported = 1;
            destroy   = HYPRE_MGRDirectSolverDestroy;
#else
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation 'spdirect' requires hypre >= 2.31.0 and DSUPERLU");
#endif
            break;
         case 32:
#if HYPRE_CHECK_MIN_VERSION(23200, 14)
            supported = 1;
            destroy   = HYPRE_ILUDestroy;
#else
            hypredrv_ErrorMsgAdd(
               "Managed MGR F-relaxation 'ilu' requires hypre >= 2.32.0 (develop >= 14)");
#endif
            break;
         case 33:
#if HYPRE_CHECK_MIN_VERSION(23100, 9)
            supported = 1;
            destroy   = HYPRE_FSAIDestroy;
#else
            hypredrv_ErrorMsgAdd("MGR F-relaxation 'fsai' requires hypre >= 2.31.0");
#endif
            break;
         case MGR_SOLVER_TYPE_SCHWARZ:
#if HYPRE_CHECK_MIN_VERSION(30100, 55)
            if (active_lvl == 0)
            {
               supported = 1;
               destroy   = MGRSchwarzWrapperDestroy;
            }
            else
               hypredrv_ErrorMsgAdd(
                  "MGR F-relaxation 'schwarz' is only supported at MGR level 0");
#else
            hypredrv_ErrorMsgAdd(
               "MGR F-relaxation 'schwarz' requires hypre >= 3.1.0 (develop >= 55)");
#endif
            break;
      }
   }
   if (!supported)
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Unsupported managed MGR F-solver at active level %d",
                           active_lvl);
   }
   if (f_relaxation && f_relaxation->symmetric_diagonal_scaling)
   {
#if !HYPRE_CHECK_MIN_VERSION(30100, 0)
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
   if (HYPRE_GetError() & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd("Cannot replace MGR F-solver with an outstanding hypre error");
   }
   int kind =
      supported ? 2 * (int)f_relaxation->type + !!f_relaxation->symmetric_diagonal_scaling
                : 0;
   if (refresh && previous && f_relaxation &&
       (previous->kind / 2 != f_relaxation->type || previous->active_level != active_lvl))
   {
      hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
      hypredrv_ErrorMsgAdd(
         "Changing the MGR F-solver method or level requires a new parent hierarchy");
   }
   int level          = f_relaxation ? active_lvl : 0;
   int local_state[6] = {hypredrv_ErrorCodeActive() != 0,
                         refresh || !cached || cached->kind != kind,
                         kind,
                         -kind,
                         level,
                         -level};
   int global_state[6];
   MPI_Allreduce(local_state, global_state, 6, MPI_INT, MPI_MAX, args->comm);
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
   HYPRE_Solver      solver  = args->frelax[orig_lvl];
   MGRFRelaxBinding *binding = NULL;
   if (global_state[1])
   {
      solver       = NULL;
      int local_ok = 0, global_ok = 0;
      binding  = calloc(1, sizeof(*binding));
      local_ok = binding != NULL;
#if !HYPRE_CHECK_MIN_VERSION(22600, 0)
      if (binding)
      {
         binding->parent        = parent;
         binding->native_levels = args->num_active_levels;
         binding->marker_size =
            refresh && previous ? previous->marker_size : args->dofmap->size;
         binding->f_markers = calloc(binding->marker_size ? binding->marker_size : 1,
                                     sizeof(*binding->f_markers));
         local_ok = binding->f_markers != NULL;
         if (local_ok && refresh && previous)
            memcpy(binding->f_markers, previous->f_markers,
                   binding->marker_size * sizeof(*binding->f_markers));
         else if (local_ok)
            for (size_t row = 0; row < binding->marker_size; row++)
            {
               binding->f_markers[row] = 1;
               for (size_t j = 0; j < f_dofs->size; j++)
                  if (args->dofmap->data[row] == f_dofs->data[j])
                     binding->f_markers[row] = -1;
            }
      }
#endif
      if (!local_ok)
         hypredrv_AllocationFailure("Failed to bind the MGR F-solver lifetime");
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, args->comm);
      if (!global_ok)
      {
         MGRFRelaxBindingDestroy(binding, 0);
         if (local_ok)
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR F-solver lifetime allocation failed on another rank");
         }
         return 0;
      }

      if (f_relaxation->type == 2)
      {
         hypredrv_AMGSetProjectedRBMs(&f_relaxation->amg, args->vec_nn, args->dofmap,
                                      args->rbm_input_generation, f_dofs->data,
                                      f_dofs->size);
         if (hypredrv_ErrorCodeActive())
         {
            goto candidate_done;
         }
         hypredrv_AMGCreate(&f_relaxation->amg, &solver);
         if (solver && f_relaxation->symmetric_diagonal_scaling)
         {
#if HYPRE_CHECK_MIN_VERSION(30100, 0)
            solver = hypredrv_MGRFRelaxEquilWrapperCreate(args->comm, solver);
#else
            HYPRE_BoomerAMGDestroy(solver);
            solver = NULL;
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd(
               "MGR symmetric_diagonal_scaling requires hypre >= 3.1.0");
#endif
         }
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
         MGRSchwarzWrapper *wrapper = calloc(1, sizeof(*wrapper));
         if (!wrapper)
         {
            hypredrv_AllocationFailure("Failed to allocate MGR Schwarz solver wrapper");
            goto candidate_done;
         }
         solver                       = (HYPRE_Solver)wrapper;
         wrapper->base.fields.setup   = MGRSchwarzWrapperSetup;
         wrapper->base.fields.solve   = MGRSchwarzWrapperSolve;
         wrapper->base.fields.destroy = MGRSchwarzWrapperDestroy;
         hypredrv_SchwarzCreate(&f_relaxation->schwarz, &wrapper->inner);
         if (!wrapper->inner && !hypredrv_ErrorCodeActive())
         {
            hypredrv_ErrorCodeSet(ERROR_INVALID_PRECON);
            hypredrv_ErrorMsgAdd("Failed to create MGR Schwarz F-solver");
         }
      }
#endif

   candidate_done:
      local_ok = solver && !hypredrv_ErrorCodeActive() &&
                 !(HYPRE_GetError() & ~(HYPRE_ERROR_CONV | HYPRE_ERROR_ARG));
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, args->comm);
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
   }
   else
   {
      HYPREDRV_LOG_COMMF(2, MPI_COMM_WORLD, hypredrv_MGRLogObjectName(stats), next_ls_id,
                         "reusing cached MGR F-relax solver handle at level %d",
                         orig_lvl);
   }
   MGRFRelaxInstall(parent, f_relaxation, solver, active_lvl);
   if (binding)
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
#endif
      hypredrv_MGRFRelaxDestroy(&args->frelax[orig_lvl], 0);
      binding->slot         = &args->frelax[orig_lvl];
      binding->solver       = solver;
      binding->destroy      = destroy;
      binding->kind         = kind;
      binding->next         = g_mgr_frelax_bindings;
      g_mgr_frelax_bindings = binding;
   }
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
