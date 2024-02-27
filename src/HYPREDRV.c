/******************************************************************************
 * Copyright (c) 2024 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "HYPREDRV.h"
#include "args.h"
#include "linsys.h"
#include "info.h"
#include "stats.h"
#include "HYPRE_utilities.h"
#include "HYPRE_parcsr_ls.h"

static bool is_initialized = 0;

/*-----------------------------------------------------------------------------
 * hypredrv_t data type
 *-----------------------------------------------------------------------------*/

typedef struct hypredrv_struct {
   MPI_Comm         comm;
   int              mypid;
   int              nprocs;

   input_args      *iargs;

   IntArray        *dofmap;
   HYPRE_IJMatrix   mat_A;
   HYPRE_IJMatrix   mat_M;
   HYPRE_IJVector   vec_b;
   HYPRE_IJVector   vec_x;
   HYPRE_IJVector   vec_x0;

   HYPRE_Precon     precon;
   HYPRE_Solver     solver;
} hypredrv_t;

/*-----------------------------------------------------------------------------
 * HYPREDRV_Initialize
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_Initialize()
{
   if (!is_initialized)
   {
      /* Initialize hypre */
      HYPRE_Initialize();
      HYPRE_DeviceInitialize();

      is_initialized = true;
   }
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Finalize
 *-----------------------------------------------------------------------------*/

void
HYPREDRV_Finalize()
{
   if (is_initialized)
   {
      HYPRE_Finalize();
      is_initialized = false;
   }
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Create
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Create(MPI_Comm comm, HYPREDRV_t *obj_ptr)
{
   HYPREDRV_t obj = (HYPREDRV_t) malloc(sizeof(hypredrv_t));

   MPI_Comm_rank(comm, &obj->mypid);
   MPI_Comm_size(comm, &obj->nprocs);

   obj->comm   = comm;
   obj->mat_A  = NULL;
   obj->mat_M  = NULL;
   obj->vec_b  = NULL;
   obj->vec_x  = NULL;
   obj->vec_x0 = NULL;
   obj->dofmap = NULL;

   *obj_ptr    = obj;

   /* Create global statistics object */
   StatsCreate();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_Destroy
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_Destroy(HYPREDRV_t *obj_ptr)
{
   HYPREDRV_t obj = *obj_ptr;

   if (obj)
   {
      if (obj->mat_A != obj->mat_M)
      {
         HYPRE_IJMatrixDestroy(obj->mat_M);
      }
      HYPRE_IJMatrixDestroy(obj->mat_A);
      HYPRE_IJVectorDestroy(obj->vec_b);
      HYPRE_IJVectorDestroy(obj->vec_x);
      HYPRE_IJVectorDestroy(obj->vec_x0);
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
HYPREDRV_PrintLibInfo(void)
{
   PrintLibInfo();

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_PrintExitInfo
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_PrintExitInfo(const char *argv0)
{
   PrintExitInfo(argv0);

   return ErrorCodeGet();
}

/*-----------------------------------------------------------------------------
 * HYPREDRV_InputArgsParse
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_InputArgsParse(int argc, char **argv, HYPREDRV_t obj)
{
   if (obj)
   {
      InputArgsParse(obj->comm, argc, argv, &obj->iargs);
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
   if (obj)
   {
      if (obj->iargs->ls.exec_policy)
      {
         HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
         HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
         HYPRE_SetSpGemmUseVendor(0);
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
   long long int num_rows;
   long long int num_nonzeros;

   if (obj)
   {
      HYPREDRV_LinearSystemReadMatrix(obj);
      HYPREDRV_LinearSystemSetRHS(obj);
      HYPREDRV_LinearSystemSetInitialGuess(obj);
      HYPREDRV_LinearSystemSetPrecMatrix(obj);
      HYPREDRV_LinearSystemReadDofmap(obj);

      num_rows     = LinearSystemMatrixGetNumRows(obj->mat_A);
      num_nonzeros = LinearSystemMatrixGetNumNonzeros(obj->mat_A);
      if (!obj->mypid)
      {
         PRINT_EQUAL_LINE(MAX_DIVISOR_LENGTH)
         printf("Solving linear system #%d ", StatsGetLinearSystemID());
         printf("with %lld rows and %lld nonzeros...\n", num_rows, num_nonzeros);
      }
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
 * HYPREDRV_LinearSystemSetRHS
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetRHS(HYPREDRV_t obj)
{
   if (obj)
   {
      LinearSystemSetRHS(obj->comm, &obj->iargs->ls, obj->mat_A, &obj->vec_b);
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
HYPREDRV_LinearSystemSetInitialGuess(HYPREDRV_t obj)
{
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
 * HYPREDRV_LinearSystemSetPrecMatrix
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemSetPrecMatrix(HYPREDRV_t obj)
{
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
 * HYPREDRV_LinearSystemReadDofmap
 *-----------------------------------------------------------------------------*/

uint32_t
HYPREDRV_LinearSystemReadDofmap(HYPREDRV_t obj)
{
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
   if (obj)
   {
      if (!(StatsGetLinearSystemID() % (obj->iargs->ls.precon_reuse + 1)))
      {
         PreconCreate(obj->iargs->precon_method, &obj->iargs->precon, obj->dofmap, &obj->precon);
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
   if (obj)
   {
      if (!(StatsGetLinearSystemID() % (obj->iargs->ls.precon_reuse + 1)))
      {
         SolverCreate(obj->comm, obj->iargs->solver_method, &obj->iargs->solver, &obj->solver);
      }
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
   int ls_id  = StatsGetLinearSystemID();
   int reuse  = obj->iargs->ls.precon_reuse;
   int num_ls = HYPREDRV_InputArgsGetNumLinearSystems(obj);

   if (obj)
   {
      if (!(ls_id % (reuse + 1)))
      {
         SolverSetup(obj->iargs->precon_method, obj->iargs->solver_method,
                     obj->precon, obj->solver, obj->mat_M, obj->vec_b, obj->vec_x);
      }
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
   if (obj)
   {
      SolverApply(obj->iargs->solver_method, obj->solver, obj->mat_A,
                  obj->vec_b, obj->vec_x);
      HYPRE_ClearAllErrors();
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
