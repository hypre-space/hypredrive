/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"

#include <math.h>
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "args.h"
#include "info.h"
#include "linsys.h"
#include "stats.h"

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

   HYPRE_Precon precon;
   HYPRE_Solver solver;

   // TODO: associate stats variable with hypredrv object
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
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Create
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *obj_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t obj = (HYPREDRV_t)malloc(sizeof(hypredrv_t));

   MPI_Comm_rank(comm, &obj->mypid);
   MPI_Comm_size(comm, &obj->nprocs);

   obj->comm     = comm;
   obj->mat_A    = NULL;
   obj->mat_M    = NULL;
   obj->vec_b    = NULL;
   obj->vec_x    = NULL;
   obj->vec_x0   = NULL;
   obj->vec_xref = NULL;
   obj->dofmap   = NULL;

   obj->precon = NULL;
   obj->solver = NULL;

   /* Disable library mode by default */
   obj->lib_mode = false;

   /* Create global statistics object */
   StatsCreate();

   /* Set output pointer */
   *obj_ptr = obj;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Destroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *obj_ptr)
{
   HYPREDRV_CHECK_INIT();

   HYPREDRV_t obj = *obj_ptr;

   if (obj)
   {
      if (obj->mat_A != obj->mat_M)
      {
         HYPRE_IJMatrixDestroy(obj->mat_M);
      }
      if (!obj->lib_mode)
      {
         HYPRE_IJMatrixDestroy(obj->mat_A);
         HYPRE_IJVectorDestroy(obj->vec_b);
         HYPRE_IJVectorDestroy(obj->vec_x);
         HYPRE_IJVectorDestroy(obj->vec_x0);
      }

      IntArrayDestroy(&obj->dofmap);
      InputArgsDestroy(&obj->iargs);

      /* Destroy global stats variable */
      StatsDestroy();

      free(*obj_ptr);
      *obj_ptr = NULL;
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
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      InputArgsParse(obj->comm, obj->lib_mode, argc, argv, &obj->iargs);
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
HYPREDRV_SetLibraryMode(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      obj->lib_mode = true;
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
HYPREDRV_SetGlobalOptions(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   // TODO: remove this API and move functionality to InputArgsParse?
   if (obj)
   {
      if (obj->iargs->ls.exec_policy)
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

         HYPRE_SetUmpireDevicePoolSize(obj->iargs->dev_pool_size);
         HYPRE_SetUmpireUMPoolSize(obj->iargs->uvm_pool_size);
         HYPRE_SetUmpireHostPoolSize(obj->iargs->host_pool_size);
         HYPRE_SetUmpirePinnedPoolSize(obj->iargs->pinned_pool_size);
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
HYPREDRV_InputArgsGetWarmup(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->warmup : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumRepetitions
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumRepetitions(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->num_repetitions : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsGetNumLinearSystems
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_InputArgsGetNumLinearSystems(HYPREDRV_t obj)
{
   return (obj) ? obj->iargs->ls.num_systems : -1;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemBuild
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadMatrix(obj));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetRHS(obj, NULL));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetInitialGuess(obj));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemSetPrecMatrix(obj));
      HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemReadDofmap(obj));

      long long int num_rows     = LinearSystemMatrixGetNumRows(obj->mat_A);
      long long int num_nonzeros = LinearSystemMatrixGetNumNonzeros(obj->mat_A);
      if (!obj->mypid)
      {
         PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
         printf("Solving linear system #%d ", StatsGetLinearSystemID());
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
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemReadMatrix(obj->comm, &obj->iargs->ls, &obj->mat_A);
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
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t obj, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      StatsTimerStart("matrix");
      obj->mat_A = (HYPRE_IJMatrix)mat_A;
      obj->mat_M = (HYPRE_IJMatrix)mat_A;
      StatsTimerStop("matrix");
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
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t obj, HYPRE_Vector vec_b)
{
   HYPREDRV_CHECK_INIT();

   if (obj && !vec_b)
   {
      LinearSystemSetRHS(obj->comm, &obj->iargs->ls, obj->mat_A, &obj->vec_xref,
                         &obj->vec_b);
   }
   else if (obj && vec_b)
   {
      obj->vec_b = (HYPRE_IJVector)vec_b;
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
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemSetInitialGuess(obj->comm, &obj->iargs->ls, obj->mat_A, obj->vec_b,
                                  &obj->vec_x0, &obj->vec_x);
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
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemResetInitialGuess(obj->vec_x0, obj->vec_x);
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
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t obj, HYPRE_Complex **sol_data)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemGetSolutionValues(obj->vec_x, sol_data);
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
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t obj, HYPRE_Complex **rhs_data)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemGetRHSValues(obj->vec_x, rhs_data);
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
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemSetPrecMatrix(obj->comm, &obj->iargs->ls, obj->mat_A, &obj->mat_M);
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
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t obj, int size, const int *dofmap)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      IntArrayBuild(obj->comm, size, dofmap, &obj->dofmap);
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
HYPREDRV_LinearSystemSetInterleavedDofmap(HYPREDRV_t obj, int num_local_blocks,
                                          int num_dof_types)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      IntArrayBuildInterleaved(obj->comm, num_local_blocks, num_dof_types, &obj->dofmap);
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
HYPREDRV_LinearSystemSetContiguousDofmap(HYPREDRV_t obj, int num_local_blocks,
                                         int num_dof_types)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      IntArrayBuildContiguous(obj->comm, num_local_blocks, num_dof_types, &obj->dofmap);
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
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      LinearSystemReadDofmap(obj->comm, &obj->iargs->ls, &obj->dofmap);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconCreate(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      if (!(StatsGetLinearSystemID() % (obj->iargs->ls.precon_reuse + 1)))
      {
         PreconCreate(obj->iargs->precon_method, &obj->iargs->precon, obj->dofmap,
                      &obj->precon);
      }
      else
      {
         if (!obj->mypid)
         {
            printf("Reusing preconditioner...\n");
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
 * HYPREDRV_LinearSolverCreate
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverCreate(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      if (!(StatsGetLinearSystemID() % (obj->iargs->ls.precon_reuse + 1)))
      {
         SolverCreate(obj->comm, obj->iargs->solver_method, &obj->iargs->solver,
                      &obj->solver);
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
HYPREDRV_PreconSetup(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      PreconSetup(obj->iargs->precon_method, obj->precon, obj->mat_A);
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
HYPREDRV_LinearSolverSetup(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      int ls_id = StatsGetLinearSystemID();
      int reuse = obj->iargs->ls.precon_reuse;

      if (!(ls_id % (reuse + 1)))
      {
         SolverSetup(obj->iargs->precon_method, obj->iargs->solver_method, obj->precon,
                     obj->solver, obj->mat_M, obj->vec_b, obj->vec_x);
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
HYPREDRV_LinearSolverApply(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   HYPRE_Complex e_norm = NAN, x_norm = NAN, xref_norm = NAN;

   if (obj)
   {
      SolverApply(obj->iargs->solver_method, obj->solver, obj->mat_A, obj->vec_b,
                  obj->vec_x);
      HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

      if (obj->vec_xref)
      {
         LinearSystemComputeVectorNorm(obj->vec_xref, &xref_norm);
         LinearSystemComputeVectorNorm(obj->vec_x, &x_norm);
         LinearSystemComputeErrorNorm(obj->vec_xref, obj->vec_x, &e_norm);
         if (!obj->mypid)
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
HYPREDRV_PreconApply(HYPREDRV_t obj, HYPRE_Vector vec_b, HYPRE_Vector vec_x)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      PreconApply(obj->iargs->precon_method, obj->precon, obj->mat_A,
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
HYPREDRV_PreconDestroy(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      if (!((StatsGetLinearSystemID() + 1) % (obj->iargs->ls.precon_reuse + 1)))
      {
         PreconDestroy(obj->iargs->precon_method, &obj->iargs->precon, &obj->precon);
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
HYPREDRV_LinearSolverDestroy(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      if (!((StatsGetLinearSystemID() + 1) % (obj->iargs->ls.precon_reuse + 1)))
      {
         SolverDestroy(obj->iargs->solver_method, &obj->solver);
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
HYPREDRV_StatsPrint(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

   if (obj)
   {
      StatsPrint(obj->iargs->statistics);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_TimerStart
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_TimerStart(const char *name)
{
   HYPREDRV_CHECK_INIT();

   StatsTimerStart(name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_TimerStop
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_TimerStop(const char *name)
{
   HYPREDRV_CHECK_INIT();

   StatsTimerStop(name);

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
HYPREDRV_LinearSystemComputeEigenspectrum(HYPREDRV_t obj)
{
   HYPREDRV_CHECK_INIT();

#ifdef HYPREDRV_ENABLE_EIGSPEC
   if (obj)
   {
      /* Exit early if not computing eigenspectrum */
      if (!obj->iargs->ls.eigspec.enable)
      {
         return ErrorCodeGet();
      }

      if (!obj->mypid)
      {
         printf("[EigenSpectrum] | mode=%s | vectors=%s | prefix='%s'\n",
                obj->iargs->ls.eigspec.hermitian ? "Hermitian" : "General",
                obj->iargs->ls.eigspec.vectors ? "on" : "off",
                obj->iargs->ls.eigspec.output_prefix[0]
                   ? obj->iargs->ls.eigspec.output_prefix
                   : "eig");
         fflush(stdout);
      }

      /* pass preconditioner apply callback directly */
      if (obj->iargs->ls.eigspec.preconditioned)
      {
         HYPREDRV_PreconCreate(obj);
         HYPREDRV_PreconSetup(obj);

         return hypredrv_EigSpecCompute(&obj->iargs->ls.eigspec, (void *)obj->mat_A,
                                        (void *)obj, hypredrv_PreconApplyWrapper);
      }
      else
      {
         return hypredrv_EigSpecCompute(&obj->iargs->ls.eigspec, (void *)obj->mat_A, NULL,
                                        NULL);
      }
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
   }
#else
   (void)obj;
   ErrorCodeSet(ERROR_UNKNOWN);
   ErrorMsgAdd("Eigenspectrum feature disabled at build time. Reconfigure with "
               "-DHYPREDRV_ENABLE_EIGSPEC=ON");
#endif

   return ErrorCodeGet();
}
