/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"

#include <math.h>
#include <stdio.h>
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "_hypre_parcsr_mv.h" /* For hypre_VectorData, hypre_ParVectorLocalVector */
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

// Macro to check if HYPREDRV object is valid
#define HYPREDRV_CHECK_OBJ()                    \
   if (!hypredrv)                               \
   {                                            \
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ); \
      return ErrorCodeGet();                    \
   }

/*-----------------------------------------------------------------------------
 * hypredrv_t data type
 *-----------------------------------------------------------------------------*/

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
      HYPRE_Int   log_level =
         (env_log_level) ? (HYPRE_Int)strtol(env_log_level, NULL, 10) : 0;

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
   hypredrv->nstates  = 0;
   hypredrv->states   = NULL;
   hypredrv->mat_A    = NULL;
   hypredrv->mat_M    = NULL;
   hypredrv->vec_b    = NULL;
   hypredrv->vec_x    = NULL;
   hypredrv->vec_x0   = NULL;
   hypredrv->vec_xref = NULL;
   hypredrv->vec_nn   = NULL;
   hypredrv->vec_s    = NULL;
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

   if (!hypredrv)
   {
      ErrorCodeSet(ERROR_UNKNOWN_HYPREDRV_OBJ);
      return ErrorCodeGet();
   }

   if (hypredrv->mat_A != hypredrv->mat_M)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_M);
   }
   if (!hypredrv->lib_mode)
   {
      HYPRE_IJMatrixDestroy(hypredrv->mat_A);
      HYPRE_IJVectorDestroy(hypredrv->vec_b);
      for (int i = 0; i < hypredrv->nstates; i++)
      {
         HYPRE_IJVectorDestroy(hypredrv->vec_s[i]);
      }
   }

   /* Always destroy these vectors since they are created by HYPREDRV. */
   if (hypredrv->vec_x)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x);
   }
   if (hypredrv->vec_x0)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_x0);
   }
   if (hypredrv->vec_nn)
   {
      HYPRE_IJVectorDestroy(hypredrv->vec_nn);
   }

   IntArrayDestroy(&hypredrv->dofmap);
   InputArgsDestroy(&hypredrv->iargs);

   /* Destroy statistics object */
   StatsDestroy(&hypredrv->stats);

   if ((*hypredrv_ptr)->states) free((*hypredrv_ptr)->states);
   if ((*hypredrv_ptr)->vec_s) free((void *)(*hypredrv_ptr)->vec_s);
   free(*hypredrv_ptr);
   *hypredrv_ptr = NULL;

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
   HYPREDRV_CHECK_OBJ();

   InputArgsParse(hypredrv->comm, hypredrv->lib_mode, argc, argv, &hypredrv->iargs);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetLibraryMode
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetLibraryMode(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv->lib_mode = true;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_SetGlobalOptions
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_SetGlobalOptions(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   // TODO: remove this API and move functionality to InputArgsParse?
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
 * HYPREDRV_StateVectorSet
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorSet(HYPREDRV_t hypredrv, int nstates, HYPRE_IJVector *vecs)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   hypredrv->nstates = nstates;
   hypredrv->states  = (int *)malloc(sizeof(int) * (size_t)nstates);
   hypredrv->vec_s   = (HYPRE_IJVector *)malloc(sizeof(HYPRE_IJVector) * (size_t)nstates);
   for (int i = 0; i < nstates; i++)
   {
      hypredrv->states[i] = i;
      if (vecs && vecs[i])
      {
         hypredrv->vec_s[i] = vecs[i];
      }
      else
      {
         ErrorCodeSet(ERROR_UNKNOWN);
         return ErrorCodeGet();
      }
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorGetValues(HYPREDRV_t hypredrv, int index, HYPRE_Complex **data_ptr)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int             state   = hypredrv->states[index];
   HYPRE_ParVector par_vec = NULL;
   hypre_Vector   *seq_vec = NULL;
   void           *obj     = NULL;

   if (hypredrv->vec_s[state])
   {
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state], &obj);
      par_vec   = (HYPRE_ParVector)obj;
      seq_vec   = hypre_ParVectorLocalVector(par_vec);
      *data_ptr = hypre_VectorData(seq_vec);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Cycle through state vectors
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorCopy(HYPREDRV_t hypredrv, int index_in, int index_out)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int   state_in  = hypredrv->states[index_in];
   int   state_out = hypredrv->states[index_out];
   void *obj_in    = NULL;
   void *obj_out   = NULL;

   if (hypredrv->vec_s[state_in] && hypredrv->vec_s[state_out])
   {
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state_in], &obj_in);
      HYPRE_IJVectorGetObject(hypredrv->vec_s[state_out], &obj_out);

      HYPRE_ParVectorCopy((HYPRE_ParVector)obj_in, (HYPRE_ParVector)obj_out);
   }
   else
   {
      ErrorCodeSet(ERROR_UNKNOWN);
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * Cycle through state vectors
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorUpdateAll(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   for (int i = 0; i < hypredrv->nstates; i++)
   {
      hypredrv->states[i] = (hypredrv->states[i] + 1) % hypredrv->nstates;
   }

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StateVectorApplyCorrection
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StateVectorApplyCorrection(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   void *obj_s = NULL, *obj_delta = NULL;
   int   current = hypredrv->states[0];

   HYPRE_IJVectorGetObject(hypredrv->vec_x, &obj_delta);
   HYPRE_IJVectorGetObject(hypredrv->vec_s[current], &obj_s);

   /* U = U + Δx */
   HYPRE_ParVectorAxpy((HYPRE_Complex)1.0, (HYPRE_ParVector)obj_delta,
                       (HYPRE_ParVector)obj_s);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemBuild
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemBuild(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

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

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemReadMatrix(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->mat_A);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetMatrix(HYPREDRV_t hypredrv, HYPRE_Matrix mat_A)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   /* Don't annotate "matrix" here - users annotate with "system" in their code */
   /* This was causing build times and solve times to be recorded in separate entries
    */
   hypredrv->mat_A = (HYPRE_IJMatrix)mat_A;
   hypredrv->mat_M = (HYPRE_IJMatrix)mat_A;

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t hypredrv, HYPRE_Vector vec)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!vec)
   {
      LinearSystemSetRHS(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                         &hypredrv->vec_xref, &hypredrv->vec_b);
   }
   else
   {
      hypredrv->vec_b = (HYPRE_IJVector)vec;
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
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetNearNullSpace(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                                num_entries, num_components, values, &hypredrv->vec_nn);

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
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetInitialGuess(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                               hypredrv->vec_b, &hypredrv->vec_x0, &hypredrv->vec_x);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemResetInitialGuess
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemResetInitialGuess(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemResetInitialGuess(hypredrv->vec_x0, hypredrv->vec_x);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionValues(HYPREDRV_t hypredrv, HYPRE_Complex **sol_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemGetSolutionValues(hypredrv->vec_x, sol_data);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetSolutionNorm
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetSolutionNorm(HYPREDRV_t hypredrv, const char *norm_type,
                                     double *norm)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!norm_type || !norm)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      return ErrorCodeGet();
   }

   LinearSystemComputeVectorNorm(hypredrv->vec_x, norm_type, norm);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemGetRHSValues
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemGetRHSValues(HYPREDRV_t hypredrv, HYPRE_Complex **rhs_data)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemGetRHSValues(hypredrv->vec_x, rhs_data);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemSetPrecMatrix(hypredrv->comm, &hypredrv->iargs->ls, hypredrv->mat_A,
                             &hypredrv->mat_M);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemSetDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetDofmap(HYPREDRV_t hypredrv, int size, const int *dofmap)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   IntArrayBuild(hypredrv->comm, size, dofmap, &hypredrv->dofmap);

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
   HYPREDRV_CHECK_OBJ();

   IntArrayDestroy(&hypredrv->dofmap);
   IntArrayBuildInterleaved(hypredrv->comm, num_local_blocks, num_dof_types,
                            &hypredrv->dofmap);

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
   HYPREDRV_CHECK_OBJ();

   IntArrayDestroy(&hypredrv->dofmap);
   IntArrayBuildContiguous(hypredrv->comm, num_local_blocks, num_dof_types,
                           &hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   LinearSystemReadDofmap(hypredrv->comm, &hypredrv->iargs->ls, &hypredrv->dofmap);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSystemPrintDofmap
 *----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemPrintDofmap(HYPREDRV_t hypredrv, const char *filename)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   if (!filename)
   {
      ErrorCodeSet(ERROR_UNKNOWN);
      ErrorMsgAdd("Filename cannot be NULL");
      return ErrorCodeGet();
   }

   if (!hypredrv->dofmap || !hypredrv->dofmap->data)
   {
      ErrorCodeSet(ERROR_MISSING_DOFMAP);
      ErrorMsgAdd("DOF map not set.");
   }
   else
   {
      IntArrayWriteAsciiByRank(hypredrv->comm, hypredrv->dofmap, filename);
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
   HYPREDRV_CHECK_OBJ();

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
   HYPREDRV_CHECK_OBJ();

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
   bool should_create = (hypredrv->precon == NULL) || (reuse == 0) ||
                        (ls_id < 0 || ls_id == 0) || ((ls_id + 1) % (reuse + 1) == 0);

   if (should_create)
   {
      PreconCreate(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                   hypredrv->dofmap, hypredrv->vec_nn, &hypredrv->precon);
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
   HYPREDRV_CHECK_OBJ();

   int ls_id = StatsGetLinearSystemID();
   int reuse = hypredrv->iargs->ls.precon_reuse;

   /* First, create the preconditioner if we need */
   if (!hypredrv->precon)
   {
      if (HYPREDRV_PreconCreate(hypredrv))
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         return ErrorCodeGet();
      }
   }

   /* Create the solver object (if not reusing) */
   if (!((ls_id + 1) % (reuse + 1)))
   {
      SolverCreate(hypredrv->comm, hypredrv->iargs->solver_method,
                   &hypredrv->iargs->solver, &hypredrv->solver);
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
   HYPREDRV_CHECK_OBJ();

   PreconSetup(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverSetup
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverSetup(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int ls_id = StatsGetLinearSystemID();
   int reuse = hypredrv->iargs->ls.precon_reuse;

   if (!((ls_id + 1) % (reuse + 1)))
   {
      SolverSetup(hypredrv->iargs->precon_method, hypredrv->iargs->solver_method,
                  hypredrv->precon, hypredrv->solver, hypredrv->mat_M, hypredrv->vec_b,
                  hypredrv->vec_x);
   }
   HYPRE_ClearAllErrors();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_LinearSolverApply
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSolverApply(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   double e_norm = 0.0, x_norm = 0.0, xref_norm = 0.0;

   SolverApply(hypredrv->iargs->solver_method, hypredrv->solver, hypredrv->mat_A,
               hypredrv->vec_b, hypredrv->vec_x);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   if (hypredrv->vec_xref)
   {
      LinearSystemComputeVectorNorm(hypredrv->vec_xref, "L2", &xref_norm);
      LinearSystemComputeVectorNorm(hypredrv->vec_x, "L2", &x_norm);
      LinearSystemComputeErrorNorm(hypredrv->vec_xref, hypredrv->vec_x, "L2", &e_norm);
      if (!hypredrv->mypid)
      {
         printf("L2 norm of error: %e\n", (double)e_norm);
         printf("L2 norm of solution: %e\n", (double)x_norm);
         printf("L2 norm of ref. solution: %e\n", (double)xref_norm);
      }
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
   HYPREDRV_CHECK_OBJ();

   PreconApply(hypredrv->iargs->precon_method, hypredrv->precon, hypredrv->mat_A,
               (HYPRE_IJVector)vec_b, (HYPRE_IJVector)vec_x);
   HYPRE_ClearAllErrors(); /* TODO: error handling from hypre */

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PreconDestroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PreconDestroy(HYPREDRV_t hypredrv)
{
   HYPREDRV_CHECK_INIT();
   HYPREDRV_CHECK_OBJ();

   int ls_id = StatsGetLinearSystemID();
   int reuse = hypredrv->iargs->ls.precon_reuse;

   /* Preconditioner reuse logic:
    * - If reuse == 0: always destroy (no reuse)
    * - If reuse > 0: destroy every (reuse + 1) linear systems
    *   This means: destroy when (ls_id + 1) % (reuse + 1) == 0, but not on first
    * system Example: reuse=2 means reuse for 2 systems, destroy on 3rd (ls_id=2, 5, 8,
    * ...)
    */
   bool should_destroy = (reuse == 0) || (ls_id > 0 && ((ls_id + 1) % (reuse + 1) == 0));

   if (should_destroy)
   {
      PreconDestroy(hypredrv->iargs->precon_method, &hypredrv->iargs->precon,
                    &hypredrv->precon);
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
   HYPREDRV_CHECK_OBJ();

   int ls_id = StatsGetLinearSystemID();
   int reuse = hypredrv->iargs->ls.precon_reuse;

   /* First, destroy the preconditioner if we need */
   if (hypredrv->precon)
   {
      if (HYPREDRV_PreconDestroy(hypredrv))
      {
         ErrorCodeSet(ERROR_INVALID_PRECON);
         return ErrorCodeGet();
      }
   }

   /* Destroy the solver object (if not reusing) */
   if (!((ls_id + 1) % (reuse + 1)))
   {
      SolverDestroy(hypredrv->iargs->solver_method, &hypredrv->solver);
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
   HYPREDRV_CHECK_OBJ();

   StatsPrint(hypredrv->iargs->statistics);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateBegin(const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   if (id >= 0)
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   }
   else
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s", name);
   }
   StatsAnnotate(HYPREDRV_ANNOTATE_BEGIN, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateEnd(const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   if (id >= 0)
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   }
   else
   {
      snprintf(formatted_name, sizeof(formatted_name), "%s", name);
   }
   StatsAnnotate(HYPREDRV_ANNOTATE_END, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelBegin
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelBegin(int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   StatsAnnotateLevelBegin(level, formatted_name);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_AnnotateLevelEnd
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_AnnotateLevelEnd(int level, const char *name, int id)
{
   HYPREDRV_CHECK_INIT();

   char formatted_name[1024];
   snprintf(formatted_name, sizeof(formatted_name), "%s-%d", name, id);
   StatsAnnotateLevelEnd(level, formatted_name);

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
   HYPREDRV_CHECK_OBJ();

#ifdef HYPREDRV_ENABLE_EIGSPEC
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
   HYPREDRV_CHECK_OBJ();

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

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelGetCount
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_StatsLevelGetCount(int level)
{
   return StatsLevelGetCount(level);
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelGetEntry
 *-----------------------------------------------------------------------------*/

int
HYPREDRV_StatsLevelGetEntry(int level, int index, int *entry_id, int *num_solves,
                            int *linear_iters, double *setup_time, double *solve_time)
{
   LevelEntry entry;
   int        ret = StatsLevelGetEntry(level, index, &entry);

   if (ret == 0)
   {
      if (entry_id) *entry_id = entry.id;

      /* Compute aggregates from solve index range */
      int    n_solves = entry.solve_end - entry.solve_start;
      int    l_iters  = 0;
      double s_time   = 0.0;
      double p_time   = 0.0;

      Stats *stats = StatsGetContext();
      if (stats)
      {
         for (int i = entry.solve_start; i < entry.solve_end; i++)
         {
            l_iters += stats->iters[i];
            p_time += stats->prec[i];
            s_time += stats->solve[i];
         }
      }

      if (num_solves) *num_solves = n_solves;
      if (linear_iters) *linear_iters = l_iters;
      if (setup_time) *setup_time = p_time;
      if (solve_time) *solve_time = s_time;
   }

   return ret;
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_StatsLevelPrint
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_StatsLevelPrint(int level)
{
   StatsLevelPrint(level);
   return ErrorCodeGet();
}
