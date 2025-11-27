/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"

#include <math.h>
#include <stdarg.h>
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "args.h"
#include "containers.h"
#include "info.h"
#include "linsys.h"
#include "stats.h"
#ifdef HYPREDRV_ENABLE_CALIPER
#include <caliper/cali.h>
#endif

// Flag to check if HYPREDRV is initialized
static bool hypredrv_is_initialized = 0;

// Macro to check if HYPREDRV is initialized
#define HYPREDRV_CHECK_INIT()                       \
   if (!hypredrv_is_initialized)                    \
   {                                                \
      ErrorCodeSet(ERROR_HYPREDRV_NOT_INITIALIZED); \
      return ErrorCodeGet();                        \
   }

/*-----------------------------------------------------------------------------
 * hypredrv_t data type
 *-----------------------------------------------------------------------------*/

typedef struct hypredrv_struct
{
   MPI_Comm comm;
   int      mypid;
   int      nprocs;
   bool     lib_mode;

   input_args *iargs;

   IntArray *dofmap;

   HYPRE_IJMatrix mat_A;
   HYPRE_IJMatrix mat_M;
   HYPRE_IJVector vec_b;
   HYPRE_IJVector vec_x;
   HYPRE_IJVector vec_x0;
   HYPRE_IJVector vec_xref;
   HYPRE_IJVector vec_nn;

   HYPRE_Precon precon;
   HYPRE_Solver solver;

   Stats *stats;
} hypredrv_t;

/*-----------------------------------------------------------------------------
 * HYPREDRV_Initialize
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Initialize()
{
   if (!hypredrv_is_initialized)
   {
      /* Initialize hypre */
      HYPRE_Initialize();
      HYPRE_DeviceInitialize();

#if HYPRE_CHECK_MIN_VERSION(23100, 16)
      /* Check for environment variables */
      const char *env_log_level = getenv("HYPRE_LOG_LEVEL");
      HYPRE_Int   log_level     = (env_log_level) ? (HYPRE_Int)atoi(env_log_level) : 0;

      HYPRE_SetLogLevel(log_level);
#endif

      /* Set library state to initialized */
      hypredrv_is_initialized = true;
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Finalize
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Finalize()
{
   if (hypredrv_is_initialized)
   {
      HYPRE_Finalize();
      hypredrv_is_initialized = false;
   }

#ifdef HYPREDRV_ENABLE_CALIPER
   /* Flush Caliper data before MPI_Finalize to avoid mpireport warning */
   cali_flush(0);
#endif

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_ErrorCodeDescribe
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_ErrorCodeDescribe(uint32_t error_code)
{
   if (!error_code)
   {
      return;
   }

   ErrorCodeDescribe(error_code);
   ErrorMsgPrint();
   ErrorMsgClear();
   ErrorBacktracePrint();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Create
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t hypredrv = (HYPREDRV_t)malloc(sizeof(hypredrv_t));

   MPI_Comm_rank(comm, &hypredrv->mypid);
   MPI_Comm_size(comm, &hypredrv->nprocs);

   hypredrv->comm     = comm;
   hypredrv->mat_A    = NULL;
   hypredrv->mat_M    = NULL;
   hypredrv->vec_b    = NULL;
   hypredrv->vec_x    = NULL;
   hypredrv->vec_x0   = NULL;
   hypredrv->vec_xref = NULL;
   hypredrv->vec_nn   = NULL;
   hypredrv->dofmap   = NULL;

   hypredrv->precon = NULL;
   hypredrv->solver = NULL;
   hypredrv->stats  = NULL;

   /* Disable library mode by default */
   hypredrv->lib_mode = false;

   /* Create statistics object and set as active context */
   hypredrv->stats = StatsCreate();
   StatsSetContext(hypredrv->stats);

   /* Set output pointer */
   *hypredrv_ptr = hypredrv;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Destroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *hypredrv_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t hypredrv = *hypredrv_ptr;

   if (hypredrv)
   {
      if (hypredrv->mat_A != hypredrv->mat_M)
      {
         HYPRE_IJMatrixDestroy(hypredrv->mat_M);
      }
      if (!hypredrv->lib_mode)
      {
         HYPRE_IJMatrixDestroy(hypredrv->mat_A);
         HYPRE_IJVectorDestroy(hypredrv->vec_b);
      }

      /* Always destroy these vectors since they are created by HYPREDRV */
      HYPRE_IJVectorDestroy(hypredrv->vec_x);
      HYPRE_IJVectorDestroy(hypredrv->vec_x0);
      HYPRE_IJVectorDestroy(hypredrv->vec_nn);

      IntArrayDestroy(&hypredrv->dofmap);
      InputArgsDestroy(&hypredrv->iargs);

      /* Destroy statistics object */
      StatsDestroy(&hypredrv->stats);

      free(*hypredrv_ptr);
      *hypredrv_ptr = NULL;
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintLibInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintLibInfo(MPI_Comm comm)
{
   HYPREDRV_CHECK_INIT();

   PrintLibInfo(comm);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintSystemInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintSystemInfo(MPI_Comm comm)
{
   HYPREDRV_CHECK_INIT();

   PrintSystemInfo(comm);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintExitInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(MPI_Comm comm, const char *argv0)
{
   HYPREDRV_CHECK_INIT();

   PrintExitInfo(comm, argv0);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsParse
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      InputArgsParse(hypredrv->comm, hypredrv->lib_mode, argc, argv, &hypredrv->iargs);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetLibraryMode
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      hypredrv->lib_mode = true;
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetGlobalOptions
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetGlobalOptions(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   // TODO: remove this API and move functionality to InputArgsParse?
   if (hypredrv)
   {
      if (hypredrv->iargs->ls.exec_policy)
      {
         HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
         HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
         HYPRE_SetSpGemmUseVendor(0); // TODO: Control this via input option
         HYPRE_SetSpMVUseVendor(0);   // TODO: Control this via input option

#ifdef HYPRE_USING_UMPIRE
         /* Setup Umpire pools */
         HYPRE_SetUmpireDevicePoolName("HYPRE_DEVICE");
         HYPRE_SetUmpireUMPoolName("HYPRE_UM");
         HYPRE_SetUmpireHostPoolName("HYPRE_HOST");
         HYPRE_SetUmpirePinnedPoolName("HYPRE_PINNED");

         HYPRE_SetUmpireDevicePoolSize(hypredrv->iargs->dev_pool_size);
         HYPRE_SetUmpireUMPoolSize(hypredrv->iargs->uvm_pool_size);
         HYPRE_SetUmpireHostPoolSize(hypredrv->iargs->host_pool_size);
         HYPRE_SetUmpirePinnedPoolSize(hypredrv->iargs->pinned_pool_size);
#endif
      }
      else
      {
         HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
         HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetWarmup
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->warmup : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumRepetitions
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->num_repetitions : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumLinearSystems
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t hypredrv)
{
   return (hypredrv) ? hypredrv->iargs->ls.num_systems : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemBuild
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(hypredrv, NULL));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(hypredrv));

      long long int num_rows     = LinearSystemMatrixGetNumRows(hypredrv->mat_A);
      long long int num_nonzeros = LinearSystemMatrixGetNumNonzeros(hypredrv->mat_A);
      if (!hypredrv->mypid)
      {
         PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
         printf("Solving linear system #%d ", StatsGetLinearSystemID() + 1);
         printf("with %lld rows and %lld nonzeros...\n", num_rows, num_nonzeros);
      }
      HYPRE_ClearAllErrors();
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemReadMatrix(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->mat_A);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      /* Don't annotate "matrix" here - users annotate with "system" in their code */
      /* This was causing build times and solve times to be recorded in separate entries
       */
      hypredrv->mat_A = (HYPRE_IJMatrix)mat_A;
      hypredrv->mat_M = (HYPRE_IJMatrix)mat_A;
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv && !vec)
   {
      LinearSystemSetRHS(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                         &hypredrv->vec_xref, &hypredrv->vec_b);
   }
   else if (hypredrv && vec)
   {
      hypredrv->vec_b = (HYPRE_IJVector)vec;
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetNearNullSpace
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetNearNullSpace(HYPREDRV_t hypredrv, int num_entries,
                                      int num_components, const HYPRE_Complex *values)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemSetNearNullSpace(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                   num_entries, num_components, values,
                                   &hypredrv->vec_nn);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetInitialGuess
 *
 * TODO: add vector as input parameter
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemSetInitialGuess(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                  hypredrv->vec_b, &hypredrv->vec_x0, &hypredrv->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemResetInitialGuess(hypredrv->vec_x0, hypredrv->vec_x);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemGetSolutionValues(hypredrv->vec_x, sol_data);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetRHSValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemGetRHSValues(hypredrv->vec_x, rhs_data);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemSetPrecMatrix(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                &hypredrv->mat_M);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      IntArrayBuild(hypredrv->comm, size, dofmap, &hypredrv->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetInterleavedDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                          int num_dof_types)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      IntArrayBuildInterleaved(hypredrv->comm, num_local_blocks, num_dof_types,
                               &hypredrv->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetContiguousDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t hypredrv, int num_local_blocks,
                                         int num_dof_types)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      IntArrayBuildContiguous(hypredrv->comm, num_local_blocks, num_dof_types,
                              &hypredrv->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      LinearSystemReadDofmap(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemPrintDofmap
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t hypredrv, const char *filename)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv && filename)
   {
      if (!hypredrv->dofmap || !hypredrv->dofmap->data)
      {
         ErrorCodeSet(ERROR_MISSING_DOFMAP);
         ErrorMsgAdd("DOF map not set.");
      }
      else
      {
         IntArrayWriteAsciiByRank(hypredrv->comm, hypredrv->dofmap, filename);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemPrint
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (!hypredrv)
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return ErrorCodeGet();
   }

   /* Delegate printing to linsys */
   LinearSystemPrintData(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                         hypredrv->vec_b, hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      int ls_id = StatsGetLinearSystemID();
      int reuse = hypredrv->iargs->ls.precon_reuse;

      /* Preconditioner creation logic:
       * - Always create if preconditioner doesn't exist (precon is NULL)
       * - If reuse == 0: always create (no reuse)
       * - Always create on first system (ls_id < 0 or ls_id == 0)
       * - If reuse > 0: create every (reuse + 1) systems
       *   This means: create when (ls_id + 1) % (reuse + 1) == 0
       *   Example: reuse=2 means create on ls_id=0, 3, 6, 9, ...
       */
      bool should_create = false;
      if (hypredrv->precon == NULL)
      {
         should_create = true; /* Must create if it doesn't exist */
      }
      else if (reuse == 0)
      {
         should_create = true; /* No reuse: always create */
      }
      else if (ls_id < 0 || ls_id == 0)
      {
         should_create = true; /* Always create on first system */
      }
      else if ((ls_id + 1) % (reuse + 1) == 0)
      {
         should_create = true; /* Reuse period expired, need new preconditioner */
      }

      if (should_create)
      {
         PreconCreate(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                      hypredrv->dofmap, hypredrv->vec_nn, &hypredrv->precon);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      int ls_id = StatsGetLinearSystemID();
      int reuse = hypredrv->iargs->ls.precon_reuse;

      if (!((ls_id + 1) % (reuse + 1)))
      {
         SolverCreate(hypredrv->comm, hypredrv->iargs->solver_method,
                      &hypredrv->iargs->solver, &hypredrv->solver);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      PreconSetup(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A);
      HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      int ls_id = StatsGetLinearSystemID();
      int reuse = hypredrv->iargs->ls.precon_reuse;

      if (!((ls_id + 1) % (reuse + 1)))
      {
         SolverSetup(hypredrv->iargs->precon_method, hypredrv->iargs->solver_method,
                     hypredrv->precon, hypredrv->solver, hypredrv->mat_M, hypredrv->vec_b,
                     hypredrv->vec_x);
      }
      HYPRE_ClearAllErrors();
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   HYPRE_Complex e_norm = NAN, x_norm = NAN, xref_norm = NAN;

   if (hypredrv)
   {
      SolverApply(hypredrv->iargs->solver_method, hypredrv->solver, hypredrv->mat_A,
                  hypredrv->vec_b, hypredrv->vec_x);
      HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

      if (hypredrv->vec_xref)
      {
         LinearSystemComputeVectorNorm(hypredrv->vec_xref, &xref_norm);
         LinearSystemComputeVectorNorm(hypredrv->vec_x, &x_norm);
         LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, &e_norm);
         if (!hypredrv->mypid)
         {
            printf("L2 norm of error: %e\n", (double)e_norm);
            printf("L2 norm of solution: %e\n", (double)x_norm);
            printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
         }
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconApply(HYPREDRV_t hypredrv, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      PreconApply(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A,
                  (HYPRE_IJVector)vec_b, (HYPRE_IJVector)vec_x);
      HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      int ls_id = StatsGetLinearSystemID();
      int reuse = hypredrv->iargs->ls.precon_reuse;

      /* Preconditioner reuse logic:
       * - If reuse == 0: always destroy (no reuse)
       * - If reuse > 0: destroy every (reuse + 1) linear systems
       *   This means: destroy when (ls_id + 1) % (reuse + 1) == 0, but not on first
       * system Example: reuse=2 means reuse for 2 systems, destroy on 3rd (ls_id=2, 5, 8,
       * ...)
       */
      bool should_destroy = false;
      if (reuse == 0)
      {
         should_destroy = true; /* No reuse: always destroy */
      }
      else if (ls_id > 0 && ((ls_id + 1) % (reuse + 1) == 0))
      {
         should_destroy = true; /* Reuse period expired */
      }

      if (should_destroy)
      {
         PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                       &hypredrv->precon);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      if (!((StatsGetLinearSystemID() + 1) % (hypredrv->iargs->ls.precon_reuse + 1)))
      {
         SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsPrint
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsPrint(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

   if (hypredrv)
   {
      StatsPrint(hypredrv->iargs->statistics);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(const char *name, ...)
{
   HYPREDRV_CHECK_INIT();

   va_list args;
   va_start(args, name);
   StatsAnnotateV(HYPREDRV_ANNOTATE_BEGIN, name, args);
   va_end(args);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(const char *name, ...)
{
   HYPREDRV_CHECK_INIT();

   va_list args;
   va_start(args, name);
   StatsAnnotateV(HYPREDRV_ANNOTATE_END, name, args);
   va_end(args);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(int level, const char *name, ...)
{
   HYPREDRV_CHECK_INIT();

   va_list args;
   va_start(args, name);
   StatsAnnotateLevelBegin(level, name, args);
   va_end(args);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(int level, const char *name, ...)
{
   HYPREDRV_CHECK_INIT();

   va_list args;
   va_start(args, name);
   StatsAnnotateLevelEnd(level, name, args);
   va_end(args);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 *-----------------------------------------------------------------------------*/

#ifdef HYPREDRV_ENABLE_EIGSPEC
static void
hypredrv_PreconApplyWrapper(void *ctx, void *b, void *x)
{
   HYPREDRV_PreconApply((HYPREDRV_t)ctx, (HYPRE_Vector)b, (HYPRE_Vector)x);
}
#endif

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemComputeEigenspectrum
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();

#ifdef HYPREDRV_ENABLE_EIGSPEC
   if (hypredrv)
   {
      /* Exit early if not computing eigenspectrum */
      if (!hypredrv->iargs->ls.eigspec.enable)
      {
         return ErrorCodeGet();
      }

      if (!hypredrv->mypid)
      {
         printf("[EigenSpectrum] | mode=%s | vectors=%s | prefix='%s'\n",
                hypredrv->iargs->ls.eigspec.hermitian ? "Hermitian" : "General",
                hypredrv->iargs->ls.eigspec.vectors ? "on" : "off",
                hypredrv->iargs->ls.eigspec.output_prefix[0]
                   ? hypredrv->iargs->ls.eigspec.output_prefix
                   : "eig");
         fflush(stdout);
      }

      /* pass preconditioner apply callback directly */
      if (hypredrv->iargs->ls.eigspec.preconditioned)
      {
         HYPREDRV_PreconCreate(hypredrv);
         HYPREDRV_PreconSetup(hypredrv);

         return hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec,
                                        (void *)hypredrv->mat_A, (void *)hypredrv,
                                        hypredrv_PreconApplyWrapper);
      }
      else
      {
         return hypredrv_EigSpecCompute(&hypredrv->iargs->ls.eigspec,
                                        (void *)hypredrv->mat_A, NULL, NULL);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }
#else
   (void)hypredrv;
   ErrorCodeSet(ERROR_UNKNOWN);
   ErrorMsgAdd("Eigenspectrum feature disabled at build time. Reconfigure with "
               "-DHYPREDRV_ENABLE_EIGSPEC=ON");
#endif

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_GetLastStat
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_GetLastStat(HYPREDRV_t hypredrv, const char *name, void *value)
{
   HYPREDRV_CHECK_INIT();

   if (!hypredrv)
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return ErrorCodeGet();
   }

   if (!name || !value)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Stat name and value cannot be NULL");
      return ErrorCodeGet();
   }

   if (!strcmp(name, "iter"))
   {
      *(int *)value = StatsGetLastIter();
   }
   else if (!strcmp(name, "setup"))
   {
      *(double *)value = StatsGetLastSetupTime();
   }
   else if (!strcmp(name, "solve"))
   {
      *(double *)value = StatsGetLastSolveTime();
   }
   else
   {
      // Unknown stat name
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Unknown stat name: '%s'", name);
   }

   return ErrorCodeGet();
}
